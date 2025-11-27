# LaneDiffusion Integration for SeqGrowGraph

This directory contains the implementation of **LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation** integrated into SeqGrowGraph.

## Overview

LaneDiffusion enhances SeqGrowGraph by using diffusion models to generate higher-quality BEV features. The framework consists of three key components:

1. **LPIM (Lane Prior Injection Module)**: Injects GT lane centerlines into BEV features to create diffusion targets
2. **LPDM (Lane Prior Diffusion Module)**: Models the distribution of prior-injected BEV features using a Swin Transformer U-Net
3. **LPR (Lane Prior Refinement)**: Refines the generated features by fusing them with original BEV features

## Architecture

```
Raw Images → LSS → Raw BEV Features → [LaneDiffusion] → Enhanced BEV → AR Decoder → Lane Graph
                                           ↓
                                 (LPIM → LPDM → LPR)
```

## Three-Stage Training

Following the paper, training is divided into three stages:

### Stage I: Train LPIM
Train the Lane Prior Injection Module to learn how to inject lane priors into BEV features.

```bash
# Modify config
# Set: model['lane_diffusion_stage'] = 'stage_i'
./tools/dist_train.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py 8
```

**What happens**: LPIM learns to transform raw BEV features into prior-injected BEV features using GT centerlines.

### Stage II: Train LPDM
Freeze LPIM and train the diffusion model to generate prior-injected BEV features from raw BEV.

```bash
# Modify config
# Set: model['lane_diffusion_stage'] = 'stage_ii'
# Set: load_from = "path/to/stage_i_checkpoint.pth"
./tools/dist_train.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py 8
```

**What happens**: LPDM learns to generate the same quality features as LPIM but without needing GT at inference time.

### Stage III: Train Decoder
Freeze both LPIM and LPDM, and train/fine-tune the AR decoder on enhanced BEV features.

```bash
# Modify config
# Set: model['lane_diffusion_stage'] = 'stage_iii'
# Set: load_from = "path/to/stage_ii_checkpoint.pth"
./tools/dist_train.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py 8
```

**What happens**: The decoder learns to work with the enhanced, diffusion-generated BEV features.

## Inference

```bash
# Set: model['lane_diffusion_stage'] = 'inference'
# Set: load_from = "path/to/stage_iii_checkpoint.pth"
./tools/dist_test.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py path/to/checkpoint.pth 8
```

## File Structure

```
seq_grow_graph/
├── lane_diffusion/
│   ├── __init__.py
│   ├── swin_unet.py          # Swin Transformer U-Net denoiser
│   ├── lpim.py                # Lane Prior Injection Module
│   ├── lpdm.py                # Lane Prior Diffusion Module
│   └── lane_diffusion.py      # Complete framework
└── seq_grow_graph.py          # Modified main model (integrated)

configs/
└── seq_grow_graph/
    ├── seq_grow_graph_default.py
    └── seq_grow_graph_lanediffusion.py   # LaneDiffusion config
```

## Key Parameters

### LPIM Configuration
- `prior_dim`: Dimension of prior features (default: 256)
- `num_encoder_layers`: Number of transformer encoder layers (default: 4)
- `num_heads`: Number of attention heads (default: 8)
- `max_lanes`: Maximum number of lanes per sample (default: 50)
- `max_points_per_lane`: Points per lane after resampling (default: 20)

### LPDM Configuration
- `num_steps`: Number of diffusion steps (default: 15)
- `kappa`: Noise variance hyperparameter (default: 0.5)
- `p`: Shifting schedule growth rate (default: 1.0)
- Swin U-Net: `embed_dim=96`, `depths=[2,2,6,2]`, `num_heads=[3,6,12,24]`

## Hardware Requirements

- **GPU Memory**: Recommended 32GB+ per GPU (due to Swin Transformer U-Net)
- **Batch Size**: Reduced to 8 (from original 18) to fit in memory
- **Multi-GPU**: Highly recommended for Stage II training

## Expected Performance

Based on the LaneDiffusion paper, you should expect:
- **nuScenes improvements**:
  - GEO F1: +4.2%
  - TOPO F1: +4.6%
  - JTOPO F1: +4.7%
  - APLS: +6.4%

## Troubleshooting

### Out of Memory (OOM)
1. Reduce batch size in config
2. Use gradient checkpointing (enable `with_cp=True` in Swin blocks)
3. Reduce Swin U-Net size (smaller `embed_dim` or fewer `depths`)
4. Use mixed precision training (FP16/BF16)

### Training Instability in Stage II
1. Start with smaller learning rate (e.g., 1e-4)
2. Adjust `kappa` parameter (try 0.3-0.7 range)
3. Ensure Stage I converged properly

### Poor Diffusion Quality
1. Check if LPIM (Stage I) is generating good prior-injected features
2. Increase diffusion steps during inference (e.g., 30-50 steps)
3. Verify noise schedule parameters (`p`, `kappa`)

## Citation

If you use this implementation, please cite both papers:

```bibtex
@article{LaneDiffusion2024,
  title={LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation},
  author={...},
  journal={arXiv preprint arXiv:2511.06272},
  year={2024}
}

@article{SeqGrowGraph2025,
  title={SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions},
  author={Xie, Mengwei and Zeng, Shuang and Chang, Xinyuan and Liu, Xinran and Pan, Zheng and Xu, Mu and Wei, Xing},
  journal={arXiv preprint arXiv:2507.04822},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
