#!/usr/bin/env python
"""
Convert a LaneDiffusion Stage-I checkpoint so that the LPIM weights can be
loaded by the FlowMatching module.

Usage:
    python scripts/convert_lpim_checkpoint.py \
        --input path/to/stage1_ckpt.pth \
        --output path/to/flow_matching_stage1.pth
"""
import argparse
import os
import torch


def convert_checkpoint(input_path: str, output_path: str) -> None:
    ckpt = torch.load(input_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'state_dict' key.")

    state_dict = ckpt["state_dict"]
    converted = 0
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if "lane_diffusion.lpim" in k:
            new_key = k.replace("lane_diffusion.lpim", "flow_matching.lpim")
            converted += 1
        new_state_dict[new_key] = v

    ckpt["state_dict"] = new_state_dict
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"Converted {converted} LPIM parameters.")
    print(f"Saved new checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert LPIM checkpoint.")
    parser.add_argument("--input", required=True, help="Path to Stage-I checkpoint.")
    parser.add_argument("--output", required=True, help="Path to save converted checkpoint.")
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
