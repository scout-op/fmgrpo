"""
Configuration for SeqGrowGraph with Flow Matching + GRPO
Extends the default config to enable Flow Matching stages explicitly.
"""

_base_ = ['./seq_grow_graph_default.py']

model = dict(
    type='SeqGrowGraph',
    use_flow_matching=True,
    use_grpo_loss=False,
    flow_matching_stage='inference',  # change per stage
    flow_matching_cfg=dict(
        lpim_config=dict(
            bev_channels=256,
            prior_dim=256,
            num_encoder_layers=4,
            num_heads=8,
            max_lanes=50,
            max_points_per_lane=20,
        ),
        flow_config=dict(
            vector_field_type='swin',
            swin_cfg=dict(
                in_channels=256,
                out_channels=256,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=8,
                patch_size=4,
            ),
            num_inference_steps=2,
            noise_std=0.05,
        ),
    ),
)

# ===== Stage I (Train LPIM) =====
# model['flow_matching_stage'] = 'stage_i'
# model['use_grpo_loss'] = False

# ===== Stage II (Train vector field) =====
# model['flow_matching_stage'] = 'stage_ii'
# model['use_grpo_loss'] = False
# load_from = "work_dirs/seq_grow_graph_flowmatching_stage_i/latest.pth"

# ===== Stage III (Decoder + optional GRPO) =====
# model['flow_matching_stage'] = 'stage_iii'
# model['use_grpo_loss'] = True  # enable GRPO if desired
# load_from = "work_dirs/seq_grow_graph_flowmatching_stage_ii/latest.pth"

# ===== Inference =====
# model['flow_matching_stage'] = 'inference'
# model['use_grpo_loss'] = False
# load_from = "work_dirs/seq_grow_graph_flowmatching_stage_iii/latest.pth"

train_dataloader = dict(
    batch_size=12,
)

val_dataloader = dict(
    batch_size=6,
)

test_dataloader = dict(
    batch_size=6,
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

work_dir = "work_dirs/seq_grow_graph_flowmatching"
