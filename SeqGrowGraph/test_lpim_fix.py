"""
Quick test for the fixed LPIM attention mask issue
"""
import torch
import sys
import numpy as np

# Add path
sys.path.insert(0, '/data/roadnet_data/lanediff/mmdetection3d/projects/SeqGrowGraph')

def test_lpim_fix():
    print("Testing LPIM fix for attention mask issue...")
    
    from seq_grow_graph.lane_diffusion import LPIM
    
    # Create model
    model = LPIM(
        bev_channels=256,
        prior_dim=256,
        num_encoder_layers=2,
        num_heads=8,
        max_lanes=50,
        max_points_per_lane=20,
    ).cuda()
    
    # Create test data matching actual batch
    B = 8  # batch size
    C = 256
    H = 128
    W = 192
    
    raw_bev = torch.randn(B, C, H, W).cuda()
    
    # Create GT centerlines (simulating real data)
    gt_centerlines = []
    for _ in range(B):
        # Each sample has variable number of lanes
        num_lanes = np.random.randint(5, 15)
        lanes = [np.random.rand(np.random.randint(10, 30), 2) * 100 - 50 
                for _ in range(num_lanes)]
        gt_centerlines.append(lanes)
    
    print(f"Input: raw_bev shape = {raw_bev.shape}")
    print(f"GT centerlines: {B} samples, varying lanes per sample")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(raw_bev, gt_centerlines)
        
        print(f"✓ SUCCESS! Output shape: {output.shape}")
        print(f"✓ Expected shape: {raw_bev.shape}")
        assert output.shape == raw_bev.shape, "Shape mismatch!"
        print("✓ All checks passed!")
        return True
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lpim_fix()
    if success:
        print("\n" + "="*60)
        print("Fix verified! You can now restart training.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Fix failed. Please check the error above.")
        print("="*60)
