"""
Lane Prior Injection Module (LPIM)
Constructs diffusion targets by injecting lane prior knowledge into BEV features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for lane coordinates"""
    def __init__(self, d_model=256, max_len=100):
        super().__init__()
        self.d_model = d_model
        
        # Different frequencies for x and y coordinates
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, coords):
        """
        Args:
            coords: [B, M, N, 2] - lane coordinates (x, y)
        Returns:
            embeddings: [B, M, N, d_model]
        """
        B, M, N, _ = coords.shape
        
        # Normalize coordinates to [0, 1]
        x_norm = (coords[..., 0] + 50) / 100  # Assuming [-50, 50] range
        y_norm = (coords[..., 1] + 50) / 100
        
        # Discretize to max_len bins
        x_idx = (x_norm * (len(self.pe) - 1)).long().clamp(0, len(self.pe) - 1)
        y_idx = (y_norm * (len(self.pe) - 1)).long().clamp(0, len(self.pe) - 1)
        
        # Get embeddings
        x_emb = self.pe[x_idx]  # [B, M, N, d_model]
        y_emb = self.pe[y_idx]  # [B, M, N, d_model]
        
        # Combine x and y embeddings
        embeddings = x_emb + y_emb
        
        return embeddings


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for prior encoding"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, src, src_key_padding_mask=None):
        # Self attention
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class PriorEncoder(nn.Module):
    """
    Encodes GT lane centerlines into prior features
    Using Transformer Encoder with sinusoidal position embeddings
    """
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_points=100,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Sinusoidal position embedding
        self.pos_embed = SinusoidalPositionEmbedding(d_model, max_len=max_points)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Learnable lane queries for different lanes
        self.lane_query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, lane_coords, lane_masks=None):
        """
        Args:
            lane_coords: [B, M, N, 2] - lane coordinates
            lane_masks: [B, M, N] - mask for valid points (optional)
        Returns:
            lane_features: [B, M, d_model]
        """
        B, M, N, _ = lane_coords.shape
        
        # Get position embeddings
        pos_emb = self.pos_embed(lane_coords)  # [B, M, N, d_model]
        
        # Reshape for processing
        x = pos_emb.reshape(B * M, N, self.d_model)
        
        # Add learnable lane query
        lane_q = self.lane_query.expand(B * M, -1, -1)
        x = torch.cat([lane_q, x], dim=1)  # [B*M, N+1, d_model]
        
        # Apply transformer layers
        # Use key_padding_mask instead of attn_mask to avoid shape issues
        if lane_masks is not None:
            # Create key_padding_mask: True for positions to ignore
            # Expand mask to include query token (query is always valid)
            mask = lane_masks.reshape(B * M, N)
            query_mask = torch.ones(B * M, 1, device=mask.device, dtype=mask.dtype)
            key_padding_mask = torch.cat([query_mask, mask], dim=1)  # [B*M, N+1]
            # Invert: key_padding_mask should be True for positions to IGNORE
            key_padding_mask = (key_padding_mask == 0)
        else:
            key_padding_mask = None
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # Extract lane feature (from query token)
        lane_features = x[:, 0, :]  # [B*M, d_model]
        lane_features = lane_features.reshape(B, M, self.d_model)
        
        return lane_features


class CrossAttentionFusion(nn.Module):
    """Cross-attention to inject lane prior into BEV features"""
    def __init__(self, bev_channels=256, prior_dim=256, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bev_channels,
            num_heads=num_heads,
            kdim=prior_dim,
            vdim=prior_dim,
            batch_first=True
        )
        self.norm = nn.LayerNorm(bev_channels)
        
    def forward(self, bev_feat, prior_feat):
        """
        Args:
            bev_feat: [B, C, H, W]
            prior_feat: [B, M, D]
        Returns:
            fused_feat: [B, C, H, W]
        """
        B, C, H, W = bev_feat.shape
        
        # Reshape BEV features for attention
        bev_flat = bev_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Cross attention: BEV queries attend to prior keys/values
        attended, _ = self.cross_attn(
            query=bev_flat,
            key=prior_feat,
            value=prior_feat
        )
        
        # Residual + reshape
        fused = bev_flat + attended
        fused = self.norm(fused)
        fused = fused.permute(0, 2, 1).reshape(B, C, H, W)
        
        return fused


class ModifiedBevEncode(nn.Module):
    """
    Modified BEV Encoder with Cross-Attention injection
    Based on the original BevEncode but with prior injection
    """
    def __init__(self, inC=256, outC=256, prior_dim=256, num_heads=8):
        super().__init__()
        from torchvision.models.resnet import resnet18
        
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        
        # Cross-attention after conv1
        self.cross_attn1 = CrossAttentionFusion(64, prior_dim, num_heads)
        
        self.layer1 = trunk.layer1
        self.cross_attn2 = CrossAttentionFusion(64, prior_dim, num_heads)
        
        self.layer2 = trunk.layer2
        self.cross_attn3 = CrossAttentionFusion(128, prior_dim, num_heads)
        
        self.layer3 = trunk.layer3
        self.cross_attn4 = CrossAttentionFusion(256, prior_dim, num_heads)
        
        # Upsample layers
        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )
    
    def forward(self, x, prior_features):
        """
        Args:
            x: [B, C, H, W] - raw BEV features
            prior_features: [B, M, D] - encoded lane prior features
        Returns:
            out: [B, outC, H, W] - prior-injected BEV features
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cross_attn1(x, prior_features)
        
        x1 = self.layer1(x)
        x1 = self.cross_attn2(x1, prior_features)
        
        x = self.layer2(x1)
        x = self.cross_attn3(x, prior_features)
        
        x = self.layer3(x)
        x = self.cross_attn4(x, prior_features)
        
        x = self.up1(x, x1)
        x = self.up2(x)
        
        return x


class Up(nn.Module):
    """Upsampling module"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class LPIM(nn.Module):
    """
    Lane Prior Injection Module (LPIM)
    Injects GT lane centerlines into BEV features to create diffusion targets
    """
    def __init__(
        self,
        bev_channels=256,
        prior_dim=256,
        num_encoder_layers=4,
        num_heads=8,
        max_lanes=50,
        max_points_per_lane=20,
    ):
        super().__init__()
        self.bev_channels = bev_channels
        self.prior_dim = prior_dim
        self.max_lanes = max_lanes
        self.max_points_per_lane = max_points_per_lane
        
        # Prior Encoder
        self.prior_encoder = PriorEncoder(
            d_model=prior_dim,
            nhead=num_heads,
            num_layers=num_encoder_layers,
            max_points=max_points_per_lane,
        )
        
        # Modified BEV Constructor
        self.bev_constructor = ModifiedBevEncode(
            inC=bev_channels,
            outC=bev_channels,
            prior_dim=prior_dim,
            num_heads=num_heads,
        )
    
    def prepare_gt_centerlines(self, gt_centerlines_list, device):
        """
        Convert list of GT centerlines to batched tensor
        Args:
            gt_centerlines_list: List[List[np.ndarray]] - B x [M_i x [N_i, 2]]
        Returns:
            coords: [B, M, N, 2]
            masks: [B, M, N]
        """
        B = len(gt_centerlines_list)
        coords = torch.zeros(B, self.max_lanes, self.max_points_per_lane, 2, device=device)
        masks = torch.zeros(B, self.max_lanes, self.max_points_per_lane, device=device)
        
        for b, centerlines in enumerate(gt_centerlines_list):
            for m, centerline in enumerate(centerlines[:self.max_lanes]):
                if isinstance(centerline, np.ndarray):
                    centerline = torch.from_numpy(centerline).float()
                
                # Resample to fixed number of points
                num_points = min(len(centerline), self.max_points_per_lane)
                if num_points > 0:
                    # Simple linear interpolation
                    indices = torch.linspace(0, len(centerline) - 1, self.max_points_per_lane)
                    indices = indices.long().clamp(0, len(centerline) - 1)
                    resampled = centerline[indices]
                    
                    coords[b, m, :num_points] = resampled[:num_points].to(device)
                    masks[b, m, :num_points] = 1
        
        return coords, masks
    
    def forward(self, raw_bev_feat, gt_centerlines_list):
        """
        Args:
            raw_bev_feat: [B, C, H, W] - raw BEV features from LSS
            gt_centerlines_list: List of GT centerlines for each sample
        Returns:
            prior_injected_bev: [B, C, H, W] - BEV features with lane prior injected
        """
        device = raw_bev_feat.device
        
        # Prepare GT centerlines
        lane_coords, lane_masks = self.prepare_gt_centerlines(gt_centerlines_list, device)
        
        # Encode prior knowledge
        prior_features = self.prior_encoder(lane_coords, lane_masks)  # [B, M, D]
        
        # Inject into BEV features
        prior_injected_bev = self.bev_constructor(raw_bev_feat, prior_features)
        
        return prior_injected_bev
