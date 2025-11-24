import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lane_diffusion.lpim import LPIM


class TimeEmbedding(nn.Module):
    """Simple sinusoidal time embedding projected to conv-friendly vector."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t shape: [B], returns [B, embed_dim]."""
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * -(torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) / half_dim)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            pad = self.embed_dim - emb.shape[-1]
            emb = F.pad(emb, (0, pad))
        return self.proj(emb)


class FlowMatchingVectorField(nn.Module):
    """Predicts vector field v(x_t, t, x_0)."""

    def __init__(self, bev_channels: int, hidden_channels: int = 256, time_embed_dim: int = 64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_embed_dim)
        in_channels = bev_channels * 2 + time_embed_dim
        layers = []
        channels = [in_channels, hidden_channels, hidden_channels, bev_channels]
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.SiLU(inplace=True))
        layers.pop()  # remove last activation for output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, C, H, W]
            x0: [B, C, H, W]
            t: [B] in [0, 1]
        Returns:
            v(x_t, t, x0): [B, C, H, W]
        """
        B, _, H, W = x_t.shape
        t_emb = self.time_embed(t).view(B, -1, 1, 1).expand(-1, -1, H, W)
        inp = torch.cat([x_t, x0, t_emb], dim=1)
        return self.net(inp)


class FlowMatchingBEV(nn.Module):
    """
    Flow Matching module that replaces diffusion-style generators.
    Stages:
        stage_i   : train LPIM only
        stage_ii  : train vector field with flow matching loss
        stage_iii : inference-time generation (also used for finetune)
        inference : same as stage_iii but keep frozen
    """

    def __init__(
        self,
        bev_channels: int = 256,
        bev_h: int = 128,
        bev_w: int = 192,
        lpim_config: dict | None = None,
        flow_config: dict | None = None,
    ):
        super().__init__()
        if lpim_config is None:
            lpim_config = {
                "bev_channels": bev_channels,
                "prior_dim": 256,
                "num_encoder_layers": 4,
                "num_heads": 8,
                "max_lanes": 50,
                "max_points_per_lane": 20,
            }
        if flow_config is None:
            flow_config = {}

        self.lpim = LPIM(**lpim_config)
        vector_hidden = flow_config.get("hidden_channels", bev_channels)
        time_embed_dim = flow_config.get("time_embed_dim", 64)
        self.vector_field = FlowMatchingVectorField(
            bev_channels=bev_channels,
            hidden_channels=vector_hidden,
            time_embed_dim=time_embed_dim,
        )

        self.num_inference_steps = flow_config.get("num_inference_steps", 2)
        self.noise_std = flow_config.get("noise_std", 0.0)
        self.current_stage = "stage_i"

    def set_stage(self, stage: str):
        self.current_stage = stage
        if stage == "stage_i":
            self._set_requires_grad(self.lpim, True)
            self._set_requires_grad(self.vector_field, False)
        elif stage == "stage_ii":
            self._set_requires_grad(self.lpim, False)
            self._set_requires_grad(self.vector_field, True)
        elif stage in ["stage_iii", "inference"]:
            self._set_requires_grad(self.lpim, False)
            self._set_requires_grad(self.vector_field, stage == "stage_iii")
        else:
            raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def forward(self, raw_bev: torch.Tensor, gt_centerlines=None, **kwargs):
        """Dispatch to corresponding stage."""
        if self.current_stage == "stage_i":
            return self.forward_stage_i(raw_bev, gt_centerlines)
        if self.current_stage == "stage_ii":
            return self.forward_stage_ii(raw_bev, gt_centerlines)
        if self.current_stage == "stage_iii":
            noise_std = kwargs.get("noise_std", self.noise_std)
            num_steps = kwargs.get("num_steps", self.num_inference_steps)
            return self.sample(raw_bev, noise_std=noise_std, num_steps=num_steps)
        if self.current_stage == "inference":
            noise_std = kwargs.get("noise_std", self.noise_std)
            num_steps = kwargs.get("num_steps", self.num_inference_steps)
            with torch.no_grad():
                return self.sample(raw_bev, noise_std=noise_std, num_steps=num_steps)
        raise ValueError(f"Unknown stage: {self.current_stage}")

    def forward_stage_i(self, raw_bev: torch.Tensor, gt_centerlines):
        if gt_centerlines is None:
            raise ValueError("Stage I requires ground-truth centerlines.")
        return self.lpim(raw_bev, gt_centerlines)

    def forward_stage_ii(self, raw_bev: torch.Tensor, gt_centerlines):
        if gt_centerlines is None:
            raise ValueError("Stage II requires ground-truth centerlines.")
        with torch.no_grad():
            target = self.lpim(raw_bev, gt_centerlines)
        B = raw_bev.shape[0]
        device = raw_bev.device
        t = torch.rand(B, device=device)
        x_t = (1.0 - t)[:, None, None, None] * raw_bev + t[:, None, None, None] * target
        residual = target - raw_bev
        pred = self.vector_field(x_t, raw_bev, t)
        loss = F.mse_loss(pred, residual)
        return loss

    def sample(self, x0: torch.Tensor, num_steps: int | None = None, noise_std: float = 0.0):
        """Integrate vector field using simple Euler steps."""
        if num_steps is None:
            num_steps = self.num_inference_steps
        device = x0.device
        B = x0.shape[0]
        if noise_std > 0:
            x = x0 + torch.randn_like(x0) * noise_std
        else:
            x = x0.clone()

        t_values = torch.linspace(0, 1, steps=num_steps + 1, device=device, dtype=x0.dtype)[1:]
        for t in t_values:
            t_batch = torch.full((B,), t.item(), device=device, dtype=x0.dtype)
            v = self.vector_field(x, x0, t_batch)
            step = 1.0 / max(num_steps, 1)
            x = x + step * v
        return x

    def enhance(self, raw_bev: torch.Tensor, **kwargs):
        """Convenience wrapper for stage_iii/inference to keep API parity."""
        num_steps = kwargs.get("num_steps", self.num_inference_steps)
        noise_std = kwargs.get("noise_std", self.noise_std)
        if self.current_stage == "stage_iii" and self.training:
            return self.sample(raw_bev, num_steps=num_steps, noise_std=noise_std)
        with torch.no_grad():
            return self.sample(raw_bev, num_steps=num_steps, noise_std=noise_std)
