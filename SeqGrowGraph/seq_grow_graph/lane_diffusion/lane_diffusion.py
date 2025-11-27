"""
Complete LaneDiffusion Framework
Integrates LPIM, LPDM, and LPR for end-to-end training
"""

import torch
import torch.nn as nn
from .lpim import LPIM
from .lpdm import LPDM, LanePriorRefinement


class LaneDiffusion(nn.Module):
    """
    Complete LaneDiffusion framework
    Three-stage training:
        Stage I: Train LPIM
        Stage II: Train LPDM (freeze LPIM)
        Stage III: Train Decoder (freeze LPIM + LPDM)
    """
    def __init__(
        self,
        bev_channels=256,
        bev_h=128,
        bev_w=192,
        lpim_config=None,
        lpdm_config=None,
    ):
        super().__init__()
        
        # Default configs
        if lpim_config is None:
            lpim_config = {
                'bev_channels': bev_channels,
                'prior_dim': 256,
                'num_encoder_layers': 4,
                'num_heads': 8,
                'max_lanes': 50,
                'max_points_per_lane': 20,
            }
        
        if lpdm_config is None:
            lpdm_config = {
                'bev_channels': bev_channels,
                'bev_h': bev_h,
                'bev_w': bev_w,
                'num_steps': 15,
                'kappa': 0.5,
                'p': 1.0,
            }
        
        # Initialize modules
        self.lpim = LPIM(**lpim_config)
        self.lpdm = LPDM(**lpdm_config)
        self.lpr = LanePriorRefinement(in_channels=bev_channels, out_channels=bev_channels)
        
        # Training stage control
        self.current_stage = 'stage_i'  # 'stage_i', 'stage_ii', 'stage_iii', 'inference'
    
    def set_stage(self, stage):
        """
        Set training stage and freeze/unfreeze modules accordingly
        
        Args:
            stage: 'stage_i', 'stage_ii', 'stage_iii', or 'inference'
        """
        self.current_stage = stage
        
        if stage == 'stage_i':
            # Train LPIM only
            for param in self.lpim.parameters():
                param.requires_grad = True
            for param in self.lpdm.parameters():
                param.requires_grad = False
            for param in self.lpr.parameters():
                param.requires_grad = False
                
        elif stage == 'stage_ii':
            # Train LPDM only (freeze LPIM)
            for param in self.lpim.parameters():
                param.requires_grad = False
            for param in self.lpdm.parameters():
                param.requires_grad = True
            for param in self.lpr.parameters():
                param.requires_grad = False
                
        elif stage == 'stage_iii':
            # Freeze both LPIM and LPDM (decoder will be trained in main model)
            for param in self.lpim.parameters():
                param.requires_grad = False
            for param in self.lpdm.parameters():
                param.requires_grad = False
            for param in self.lpr.parameters():
                param.requires_grad = True
                
        elif stage == 'inference':
            # Freeze all
            for param in self.lpim.parameters():
                param.requires_grad = False
            for param in self.lpdm.parameters():
                param.requires_grad = False
            for param in self.lpr.parameters():
                param.requires_grad = False
    
    def forward_stage_i(self, raw_bev, gt_centerlines):
        """
        Stage I: Train LPIM to generate prior-injected BEV
        
        Args:
            raw_bev: [B, C, H, W]
            gt_centerlines: List of GT centerlines
        
        Returns:
            prior_injected_bev: [B, C, H, W]
        """
        return self.lpim(raw_bev, gt_centerlines)
    
    def forward_stage_ii(self, raw_bev, gt_centerlines):
        """
        Stage II: Train LPDM to generate prior-injected BEV from raw BEV
        
        Args:
            raw_bev: [B, C, H, W]
            gt_centerlines: List of GT centerlines
        
        Returns:
            loss: diffusion loss
        """
        # Get target (x0) from frozen LPIM
        with torch.no_grad():
            x0 = self.lpim(raw_bev, gt_centerlines)
        
        # Compute diffusion loss
        loss = self.lpdm.compute_loss(x0, raw_bev)
        
        return loss
    
    def forward_stage_iii(self, raw_bev):
        """
        Stage III / Inference: Generate enhanced BEV using LPDM
        
        Args:
            raw_bev: [B, C, H, W]
        
        Returns:
            enhanced_bev: [B, C, H, W]
        """
        # Sample from LPDM
        with torch.no_grad():
            x_generated = self.lpdm.sample(raw_bev)
        
        # Refine with LPR
        enhanced_bev = self.lpr(x_generated, raw_bev)
        
        return enhanced_bev
    
    def forward(self, raw_bev, gt_centerlines=None):
        """
        Forward pass based on current stage
        
        Args:
            raw_bev: [B, C, H, W]
            gt_centerlines: List of GT centerlines (required for stage I and II)
        
        Returns:
            output depends on stage
        """
        if self.current_stage == 'stage_i':
            return self.forward_stage_i(raw_bev, gt_centerlines)
        
        elif self.current_stage == 'stage_ii':
            return self.forward_stage_ii(raw_bev, gt_centerlines)
        
        elif self.current_stage in ['stage_iii', 'inference']:
            return self.forward_stage_iii(raw_bev)
        
        else:
            raise ValueError(f"Unknown stage: {self.current_stage}")
