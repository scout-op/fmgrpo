"""
Swin Transformer U-Net for Lane Prior Diffusion Module (LPDM)
Based on Swin Transformer architecture for denoising BEV features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for BEV features"""
    def __init__(self, patch_size=4, in_chans=256, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, N, embed_dim
        x = self.norm(x)
        return x, (H // self.patch_size, W // self.patch_size)


class PatchUnEmbed(nn.Module):
    """Patch to Image for BEV features"""
    def __init__(self, patch_size=4, embed_dim=96, out_chans=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        x = x.contiguous().view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SwinTransformerStage(nn.Module):
    """A Swin Transformer stage"""
    def __init__(self, dim, depth, num_heads, window_size, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for blk in self.blocks:
            x = blk(x, hw_shape)
        if self.downsample is not None:
            x, hw_shape = self.downsample(x, hw_shape)
        return x, hw_shape


class PatchMerging(nn.Module):
    """Patch Merging Layer (Downsampling)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, (H // 2, W // 2)


class PatchExpanding(nn.Module):
    """Patch Expanding Layer (Upsampling)"""
    def __init__(self, dim, dim_scale=2):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, L, C = x.shape
        x = self.expand(x)
        
        x = x.view(B, H, W, C * 2)
        x = x.view(B, H, W, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)

        return x, (H * 2, W * 2)


class SwinTransformerUNet(nn.Module):
    """
    Swin Transformer U-Net for denoising BEV features
    Architecture follows the paper's LPDM design
    """
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        patch_size=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim
        )

        # Time embedding (for diffusion timestep)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Condition embedding (for conditioning on raw BEV features)
        self.cond_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        for i in range(self.num_layers):
            stage = SwinTransformerStage(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=PatchMerging(int(embed_dim * 2 ** i)) if i < self.num_layers - 1 else None,
            )
            self.encoder_stages.append(stage)

        # Bottleneck
        self.bottleneck = SwinTransformerStage(
            dim=int(embed_dim * 2 ** (self.num_layers - 1)),
            depth=depths[-1],
            num_heads=num_heads[-1],
            window_size=window_size,
            downsample=None,
        )

        # Decoder stages
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_layers - 1, 0, -1):
            stage = nn.ModuleDict({
                'upsample': PatchExpanding(int(embed_dim * 2 ** i)),
                'fusion': nn.Linear(int(embed_dim * 2 ** i), int(embed_dim * 2 ** (i - 1))),
                'blocks': SwinTransformerStage(
                    dim=int(embed_dim * 2 ** (i - 1)),
                    depth=depths[i - 1],
                    num_heads=num_heads[i - 1],
                    window_size=window_size,
                    downsample=None,
                )
            })
            self.decoder_stages.append(stage)

        # Output projection
        self.patch_unembed = PatchUnEmbed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_chans=embed_dim
        )
        self.output_proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Build sinusoidal embeddings for timesteps
        """
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, x_cond, t):
        """
        Args:
            x: noisy BEV features [B, C, H, W]
            x_cond: condition BEV features (raw) [B, C, H, W]
            t: timestep [B]
        Returns:
            predicted x0 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.embed_dim)
        t_emb = self.time_embed(t_emb)  # [B, embed_dim]

        # Input projection
        x = self.input_proj(x)
        x_cond = self.cond_proj(x_cond)
        
        # Add condition
        x = x + x_cond

        # Patch embedding
        x, hw_shape = self.patch_embed(x)
        
        # Add time embedding
        x = x + t_emb[:, None, :]

        # Encoder
        skip_connections = []
        for stage in self.encoder_stages:
            skip_connections.append((x, hw_shape))
            x, hw_shape = stage(x, hw_shape)

        # Bottleneck
        x, hw_shape = self.bottleneck(x, hw_shape)

        # Decoder with skip connections
        for i, stage in enumerate(self.decoder_stages):
            # Upsample
            x, hw_shape = stage['upsample'](x, hw_shape)
            
            # Fuse with skip connection
            skip_x, skip_hw = skip_connections[-(i + 2)]
            x = torch.cat([x, skip_x], dim=-1)
            x = stage['fusion'](x)
            
            # Transformer blocks
            x, hw_shape = stage['blocks'](x, hw_shape)

        # Output projection
        x = self.patch_unembed(x, hw_shape)
        x = self.output_proj(x)

        return x
