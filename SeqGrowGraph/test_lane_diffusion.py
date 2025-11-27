"""
Test script to verify LaneDiffusion modules can be imported and initialized
Run this to check if the implementation is working before full training
"""

import torch
import sys
sys.path.insert(0, '/home/subobo/ro/1120/SeqGrowGraph')

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from seq_grow_graph.lane_diffusion import (
            SwinTransformerUNet,
            LPIM,
            LPDM,
            LanePriorRefinement,
            LaneDiffusion
        )
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_swin_unet():
    """Test Swin Transformer U-Net"""
    print("\nTesting Swin Transformer U-Net...")
    try:
        from seq_grow_graph.lane_diffusion import SwinTransformerUNet
        
        model = SwinTransformerUNet(
            in_channels=256,
            out_channels=256,
            embed_dim=96,
            depths=[2, 2, 2, 2],  # Smaller for testing
            num_heads=[3, 6, 12, 24],
            window_size=7,
        )
        
        # Test forward pass
        B, C, H, W = 2, 256, 64, 96
        x = torch.randn(B, C, H, W)
        x_cond = torch.randn(B, C, H, W)
        t = torch.randint(0, 15, (B,))
        
        with torch.no_grad():
            out = model(x, x_cond, t)
        
        assert out.shape == (B, C, H, W), f"Output shape mismatch: {out.shape}"
        print(f"✓ Swin U-Net works. Input: {x.shape}, Output: {out.shape}")
        return True
    except Exception as e:
        print(f"✗ Swin U-Net failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lpim():
    """Test LPIM"""
    print("\nTesting LPIM...")
    try:
        from seq_grow_graph.lane_diffusion import LPIM
        import numpy as np
        
        model = LPIM(
            bev_channels=256,
            prior_dim=256,
            num_encoder_layers=2,  # Smaller for testing
            num_heads=8,
            max_lanes=10,
            max_points_per_lane=20,
        )
        
        # Test forward pass
        B, C, H, W = 2, 256, 64, 96
        raw_bev = torch.randn(B, C, H, W)
        
        # Create dummy GT centerlines
        gt_centerlines = []
        for _ in range(B):
            # Each sample has a few lanes
            lanes = [np.random.rand(15, 2) * 50 - 25 for _ in range(3)]  # 3 lanes
            gt_centerlines.append(lanes)
        
        with torch.no_grad():
            out = model(raw_bev, gt_centerlines)
        
        assert out.shape == (B, C, H, W), f"Output shape mismatch: {out.shape}"
        print(f"✓ LPIM works. Input: {raw_bev.shape}, Output: {out.shape}")
        return True
    except Exception as e:
        print(f"✗ LPIM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lpdm():
    """Test LPDM"""
    print("\nTesting LPDM...")
    try:
        from seq_grow_graph.lane_diffusion import LPDM
        
        model = LPDM(
            bev_channels=256,
            bev_h=64,
            bev_w=96,
            num_steps=15,
            kappa=0.5,
            p=1.0,
            denoiser_config=dict(
                in_channels=256,
                out_channels=256,
                embed_dim=48,  # Smaller for testing
                depths=[2, 2],
                num_heads=[3, 6],
                window_size=7,
            ),
        )
        
        # Test forward pass
        B, C, H, W = 2, 256, 64, 96
        x0 = torch.randn(B, C, H, W)  # Target (prior-injected)
        xc = torch.randn(B, C, H, W)  # Condition (raw BEV)
        
        # Test loss computation
        with torch.no_grad():
            loss = model.compute_loss(x0, xc)
        
        print(f"✓ LPDM loss computation works. Loss: {loss.item():.4f}")
        
        # Test sampling
        with torch.no_grad():
            sampled = model.sample(xc, num_steps=5)  # Fewer steps for testing
        
        assert sampled.shape == (B, C, H, W), f"Sampled shape mismatch: {sampled.shape}"
        print(f"✓ LPDM sampling works. Sampled: {sampled.shape}")
        return True
    except Exception as e:
        print(f"✗ LPDM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lane_diffusion():
    """Test complete LaneDiffusion framework"""
    print("\nTesting complete LaneDiffusion framework...")
    try:
        from seq_grow_graph.lane_diffusion import LaneDiffusion
        import numpy as np
        
        model = LaneDiffusion(
            bev_channels=256,
            bev_h=64,
            bev_w=96,
            lpim_config=dict(
                num_encoder_layers=2,
                max_lanes=10,
            ),
            lpdm_config=dict(
                num_steps=10,
                denoiser_config=dict(
                    embed_dim=48,
                    depths=[2, 2],
                    num_heads=[3, 6],
                ),
            ),
        )
        
        B, C, H, W = 2, 256, 64, 96
        raw_bev = torch.randn(B, C, H, W)
        
        # Test Stage I
        print("  Testing Stage I (LPIM training)...")
        model.set_stage('stage_i')
        gt_centerlines = [[np.random.rand(15, 2) * 50 - 25 for _ in range(3)] for _ in range(B)]
        
        with torch.no_grad():
            out = model(raw_bev, gt_centerlines)
        assert out.shape == (B, C, H, W)
        print(f"  ✓ Stage I works. Output: {out.shape}")
        
        # Test Stage III/Inference
        print("  Testing Stage III/Inference...")
        model.set_stage('inference')
        
        with torch.no_grad():
            out = model(raw_bev)
        assert out.shape == (B, C, H, W)
        print(f"  ✓ Stage III/Inference works. Output: {out.shape}")
        
        return True
    except Exception as e:
        print(f"✗ LaneDiffusion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("LaneDiffusion Module Tests")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    if results[-1][1]:  # Only continue if imports work
        results.append(("Swin U-Net", test_swin_unet()))
        results.append(("LPIM", test_lpim()))
        results.append(("LPDM", test_lpdm()))
        results.append(("Complete Framework", test_lane_diffusion()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    all_passed = all(r[1] for r in results)
    print("="*60)
    if all_passed:
        print("All tests passed! ✓")
        print("\nYou can now try training with:")
        print("  ./tools/dist_train.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py 8")
    else:
        print("Some tests failed. Please check the errors above.")
    print("="*60)


if __name__ == "__main__":
    main()
