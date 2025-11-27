"""
Configuration for SeqGrowGraph with LaneDiffusion
Extends the default SeqGrowGraph config to include Lane Diffusion modules
"""

_base_ = ['./seq_grow_graph_default.py']

# Enable LaneDiffusion
model = dict(
    type='SeqGrowGraph',
    use_lane_diffusion=True,
    
    # LaneDiffusion stage: 'stage_i', 'stage_ii', 'stage_iii', or 'inference'
    lane_diffusion_stage='inference',  # Change this for different training stages
    
    # Lane Diffusion configuration
    lane_diffusion_cfg=dict(
        # LPIM configuration
        lpim_config=dict(
            bev_channels=256,  # Will be overridden by lss_cfg['d_out']
            prior_dim=256,
            num_encoder_layers=4,
            num_heads=8,
            max_lanes=50,
            max_points_per_lane=20,
        ),
        
        # LPDM configuration  
        lpdm_config=dict(
            bev_channels=256,  # Will be overridden
            bev_h=128,  # Will be computed from grid_conf
            bev_w=192,  # Will be computed from grid_conf
            num_steps=15,  # Number of diffusion steps
            kappa=0.5,  # Noise variance parameter
            p=1.0,  # Shifting schedule growth rate
            
            # Swin Transformer U-Net configuration
            denoiser_config=dict(
                in_channels=256,
                out_channels=256,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
            ),
        ),
    ),
)

# Training configuration for different stages
# Uncomment the appropriate section based on your training stage

# ===== Stage I: Train LPIM =====
# model['lane_diffusion_stage'] = 'stage_i'
# # You may want to freeze the decoder in this stage
# # Or train it jointly with LPIM

# ===== Stage II: Train LPDM =====
# model['lane_diffusion_stage'] = 'stage_ii'
# # Load Stage I checkpoint here
# load_from = "work_dirs /seq_grow_graph_lanediff_stage_i/latest.pth"

# ===== Stage III: Train Decoder with enhanced features =====
# model['lane_diffusion_stage'] = 'stage_iii'
# # Load Stage I + II checkpoints here
# load_from = "work_dirs/seq_grow_graph_lanediff_stage_ii/latest.pth"

# ===== Inference =====
# model['lane_diffusion_stage'] = 'inference'
# load_from = "work_dirs/seq_grow_graph_lanediff_stage_iii/latest.pth"

# You may need to adjust batch size depending on GPU memory
train_dataloader = dict(
    batch_size=8,  # Reduced from 18 due to LaneDiffusion memory usage
)

val_dataloader = dict(
    batch_size=4,
)

test_dataloader = dict(
    batch_size=4,
)

# Optimizer - you may want different learning rates for different stages
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.0002, weight_decay=0.01),  # Reduced LR
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Work directory
work_dir = "work_dirs/seq_grow_graph_lanediffusion"
