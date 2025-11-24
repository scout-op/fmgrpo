"""
Lane Prior Diffusion Module (LPDM)
Implements ResShift-inspired diffusion for BEV feature generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .swin_unet import SwinTransformerUNet


class LPDM(nn.Module):
    """
    Lane Prior Diffusion Module
    Models the distribution of prior-injected BEV features using diffusion
    """
    def __init__(
        self,
        bev_channels=256,
        bev_h=128,
        bev_w=192,
        num_steps=15,
        kappa=0.5,
        p=1.0,
        denoiser_config=None,
    ):
        super().__init__()
        self.bev_channels = bev_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_steps = num_steps
        self.kappa = kappa
        self.p = p
        
        # Denoising network (Swin Transformer U-Net)
        if denoiser_config is None:
            denoiser_config = {
                'in_channels': bev_channels,
                'out_channels': bev_channels,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
            }
        self.denoiser = SwinTransformerUNet(**denoiser_config)
        
        # Compute and register noise schedule
        eta_schedule = self._compute_eta_schedule(num_steps, kappa, p)
        self.register_buffer('eta_schedule', eta_schedule)
        
        # Compute gamma schedule
        gamma_schedule = self._compute_gamma_schedule(eta_schedule)
        self.register_buffer('gamma_schedule', gamma_schedule)
        
    def _compute_eta_schedule(self, T, kappa, p):
        """
        Compute the shifting schedule eta_t according to paper
        """
        eta = torch.zeros(T + 1)
        
        # eta_1
        eta[1] = min(0.04 / kappa, np.sqrt(0.001))
        
        # eta_T
        eta[T] = np.sqrt(0.999)
        
        # Compute b_0
        b_0 = np.exp((1 / (2 * (T - 1))) * np.log(eta[T] / eta[1]))
        
        # Compute intermediate values
        for t in range(2, T):
            zeta_t = ((t - 1) / (T - 1)) ** p * (T - 1)
            eta[t] = eta[1] * (b_0 ** zeta_t)
        
        # Square the values (since we computed sqrt(eta))
        eta = eta ** 2
        
        return eta
    
    def _compute_gamma_schedule(self, eta):
        """
        Compute gamma schedule from eta
        gamma_1 = eta_1
        gamma_t = eta_t - eta_{t-1} for t > 1
        """
        gamma = torch.zeros_like(eta)
        gamma[1] = eta[1]
        for t in range(2, len(eta)):
            gamma[t] = eta[t] - eta[t - 1]
        return gamma
    
    def forward_diffusion(self, x0, xc, t):
        """
        Forward diffusion process: q(x_t | x_0, x_c)
        
        Args:
            x0: target features (prior-injected BEV) [B, C, H, W]
            xc: condition features (raw BEV) [B, C, H, W]
            t: timestep [B], values in [1, T]
        
        Returns:
            xt: noisy features [B, C, H, W]
            x_res: residual x_c - x_0 [B, C, H, W]
        """
        B = x0.shape[0]
        device = x0.device
        
        # Compute residual
        x_res = xc - x0
        
        # Get eta_t for each sample
        eta_t = self.eta_schedule[t].to(device).view(B, 1, 1, 1)
        
        # Sample noise
        noise = torch.randn_like(x0) * self.kappa * torch.sqrt(eta_t)
        
        # Compute x_t = x_0 + eta_t * x_res + noise
        xt = x0 + eta_t * x_res + noise
        
        return xt, x_res
    
    def predict_x0(self, xt, xc, t):
        """
        Predict x_0 from x_t using the denoising network
        
        Args:
            xt: noisy features [B, C, H, W]
            xc: condition features [B, C, H, W]
            t: timestep [B]
        
        Returns:
            predicted x0 [B, C, H, W]
        """
        t_normalized = t.float() / self.num_steps  # Normalize to [0, 1]
        x0_pred = self.denoiser(xt, xc, t_normalized)
        return x0_pred
    
    def reverse_step(self, xt, xc, t):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t, x_c)
        
        Args:
            xt: current noisy features [B, C, H, W]
            xc: condition features [B, C, H, W]
            t: current timestep [B], values in [1, T]
        
        Returns:
            x_{t-1}: denoised features [B, C, H, W]
        """
        B = xt.shape[0]
        device = xt.device
        
        # Predict x_0
        x0_pred = self.predict_x0(xt, xc, t)
        
        # Get schedule values
        eta_t = self.eta_schedule[t].to(device).view(B, 1, 1, 1)
        eta_t_minus_1 = self.eta_schedule[t - 1].to(device).view(B, 1, 1, 1)
        gamma_t = self.gamma_schedule[t].to(device).view(B, 1, 1, 1)
        
        # Compute mean: mu = (eta_{t-1} / eta_t) * x_t + (gamma_t / eta_t) * x_0_pred
        mu = (eta_t_minus_1 / eta_t) * xt + (gamma_t / eta_t) * x0_pred
        
        # Add noise (except for t=1)
        if t[0].item() > 1:
            # Compute variance
            sigma = self.kappa * torch.sqrt((eta_t_minus_1 * gamma_t) / eta_t)
            noise = torch.randn_like(xt)
            x_t_minus_1 = mu + sigma * noise
        else:
            x_t_minus_1 = mu
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(self, xc, num_steps=None):
        """
        Sample from the diffusion model (reverse process)
        
        Args:
            xc: condition features (raw BEV) [B, C, H, W]
            num_steps: number of diffusion steps (default: self.num_steps)
        
        Returns:
            x0: generated features [B, C, H, W]
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        B, C, H, W = xc.shape
        device = xc.device
        
        # Initialize from noisy condition: x_T ~ N(x_c, kappa^2 * eta_T * I)
        eta_T = self.eta_schedule[num_steps].to(device)
        noise = torch.randn_like(xc) * self.kappa * torch.sqrt(eta_T)
        xt = xc + noise
        
        # Reverse diffusion
        for t in range(num_steps, 0, -1):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            xt = self.reverse_step(xt, xc, t_batch)
        
        return xt
    
    def compute_loss(self, x0, xc, t=None):
        """
        Compute diffusion loss: ||f_theta(x_t, x_c, t) - x_0||^2
        
        Args:
            x0: target features [B, C, H, W]
            xc: condition features [B, C, H, W]
            t: timestep [B] (if None, randomly sampled)
        
        Returns:
            loss: scalar
        """
        B = x0.shape[0]
        device = x0.device
        
        # Sample timestep if not provided
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (B,), device=device, dtype=torch.long)
        
        # Forward diffusion to get noisy x_t
        xt, x_res = self.forward_diffusion(x0, xc, t)
        
        # Predict x_0
        x0_pred = self.predict_x0(xt, xc, t)
        
        # Compute weighted loss
        # w_t = alpha / (2 * kappa^2 * eta_t * eta_{t-1})
        # For simplicity, we use uniform weighting (alpha=1)
        eta_t = self.eta_schedule[t].to(device).view(B, 1, 1, 1)
        eta_t_minus_1 = self.eta_schedule[t - 1].to(device).view(B, 1, 1, 1)
        
        w_t = 1.0 / (2 * self.kappa ** 2 * eta_t * eta_t_minus_1 + 1e-8)
        
        # MSE loss
        loss = w_t * F.mse_loss(x0_pred, x0, reduction='none')
        loss = loss.mean()
        
        return loss


class LanePriorRefinement(nn.Module):
    """
    Lane Prior Refinement (LPR)
    Fuses the generated BEV feature with the original BEV feature
    """
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
    
    def forward(self, x_generated, x_original):
        """
        Args:
            x_generated: generated BEV from diffusion [B, C, H, W]
            x_original: original BEV from LSS [B, C, H, W]
        Returns:
            refined BEV [B, C, H, W]
        """
        # Concatenate
        x = torch.cat([x_generated, x_original], dim=1)
        
        # Encode
        x = self.encoder(x)
        
        # Decode
        x = self.decoder(x)
        
        # Residual connection
        x = x + x_original
        
        return x
