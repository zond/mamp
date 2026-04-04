#!/usr/bin/env python3
"""
export_weights.py — Export Ha et al. 2024 model weights for WebGPU inference.

Usage:
    python export_weights.py --checkpoint path/to/checkpoint.pth --outdir weights/

This produces one .bin file per parameter tensor, containing raw little-endian f32 values.
The Rust/WASM loader reads these directly into GPU storage buffers.

Naming convention follows the WeightManifest in weights.rs:
    enc_conv1.weight.bin, enc_conv1.bias.bin, etc.
"""

import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch weights to raw f32 binaries")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--outdir", default="weights", help="Output directory")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch required: pip install torch", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt.state_dict()

    total_bytes = 0
    for name, param in state.items():
        data = param.float().numpy()
        # Flatten to C-contiguous f32
        data = np.ascontiguousarray(data, dtype=np.float32)

        out_path = os.path.join(args.outdir, f"{name}.bin")
        data.tofile(out_path)

        total_bytes += data.nbytes
        print(f"  {name}: {list(param.shape)} -> {data.nbytes:,} bytes")

    print(f"\nExported {len(state)} tensors ({total_bytes:,} bytes total) to {args.outdir}/")
    print(f"Serve alongside WASM app and load via fetch() in weights.rs")

if __name__ == "__main__":
    main()
