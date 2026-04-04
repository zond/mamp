#!/usr/bin/env python3
"""
train.py — Train the Ha et al. 2024 efficient motion magnification model.

Uses synthetic motion data: for each image, a small random translation creates
the "moved" frame, and the known amplified translation creates the ground truth.
No external dataset required — generates procedural training images on GPU.

Usage:
    pip install torch numpy
    python train.py --epochs 300 --outdir weights/

The trained weights are exported as raw f32 binary files ready for the WASM app.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Ha et al. 2024 Model ────────────────────────────────────────────────────

class HaMotionMag(nn.Module):
    """Efficient motion magnification (Ha et al. 2024)."""

    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.enc_texture = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.dec_conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.dec_conv2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.dec_conv3 = nn.Conv2d(16, 3, 3, stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        shape = F.relu(self.enc_conv3(x))
        texture = self.enc_texture(x)
        return shape, texture

    def decode(self, x):
        x = F.relu(self.dec_conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv3(x)
        return x

    def forward(self, frame_a, frame_b, alpha):
        shape_a, texture_a = self.encode(frame_a)
        shape_b, _ = self.encode(frame_b)
        manipulated = texture_a + alpha * (shape_b - shape_a)
        return self.decode(manipulated)


# ── Synthetic Data (GPU-accelerated) ────────────────────────────────────────

def generate_images_gpu(n, h, w, device):
    """Generate n procedural images directly on GPU. Returns (n, 3, h, w)."""
    imgs = torch.rand(n, 3, h, w, device=device) * 0.3 + 0.35

    yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(n, 1, h, w)
    xx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).expand(n, 1, h, w)

    # Random gradients
    angles = torch.rand(n, 3, 1, 1, device=device) * 6.283
    strength = torch.rand(n, 3, 1, 1, device=device) * 0.4
    grad = strength * (xx * angles.cos() + yy * angles.sin())
    imgs = imgs + grad

    # Random circles (vectorized): 8 circles per image
    for _ in range(8):
        cx = torch.rand(n, 1, 1, 1, device=device)
        cy = torch.rand(n, 1, 1, 1, device=device)
        r = torch.rand(n, 1, 1, 1, device=device) * 0.25 + 0.03
        color = torch.rand(n, 3, 1, 1, device=device)
        dist = ((xx - cx) ** 2 + (yy - cy) ** 2).expand_as(imgs)
        mask = (dist < r ** 2).float()
        imgs = imgs * (1 - mask) + color * mask * 0.7 + imgs * mask * 0.3

    return imgs.clamp(0, 1)


def translate_batch(img, dx, dy):
    """Sub-pixel translation. img: (B, C, H, W), dx/dy: (B,) tensors."""
    B, C, H, W = img.shape
    # Build affine matrices
    theta = torch.zeros(B, 2, 3, device=img.device, dtype=img.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * dx / W
    theta[:, 1, 2] = -2.0 * dy / H
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HaMotionMag().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Pre-generate a pool of images on GPU
    print(f"Generating {args.pool_size} training images on {device}...")
    image_pool = generate_images_gpu(args.pool_size, args.img_size, args.img_size, device)
    print(f"Image pool ready: {image_pool.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_loss = float("inf")
    bs = args.batch_size
    batches_per_epoch = args.pool_size // bs

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Shuffle pool
        perm = torch.randperm(args.pool_size, device=device)
        shuffled = image_pool[perm]

        for i in range(batches_per_epoch):
            frame_a = shuffled[i * bs:(i + 1) * bs]

            # Random displacements
            dx = (torch.rand(bs, device=device) * 2 - 1) * args.max_displacement
            dy = (torch.rand(bs, device=device) * 2 - 1) * args.max_displacement

            # Random alpha per sample
            alpha = torch.rand(bs, device=device) * (args.alpha_max - args.alpha_min) + args.alpha_min
            alpha_4d = alpha.view(-1, 1, 1, 1)

            # Create motion pair and ground truth
            frame_b = translate_batch(frame_a, dx, dy)
            gt = translate_batch(frame_a, alpha * dx, alpha * dy)

            # Main magnification loss
            pred = model(frame_a, frame_b, alpha_4d)
            loss = F.l1_loss(pred, gt) + 0.1 * F.mse_loss(pred, gt)

            # Reconstruction: alpha=1 should reproduce frame_b
            ones = torch.ones_like(alpha_4d)
            recon = model(frame_a, frame_b, ones)
            loss = loss + 0.5 * F.l1_loss(recon, frame_b)

            # Identity: same frame in → same frame out
            identity = model(frame_a, frame_a, alpha_4d)
            loss = loss + 0.3 * F.l1_loss(identity, frame_a)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(batches_per_epoch, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Regenerate some images periodically for diversity
        if epoch % 50 == 0:
            n_new = args.pool_size // 4
            new_imgs = generate_images_gpu(n_new, args.img_size, args.img_size, device)
            idx = torch.randperm(args.pool_size)[:n_new]
            image_pool[idx] = new_imgs

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pth"))

    torch.save(model.state_dict(), os.path.join(args.outdir, "final.pth"))
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")

    return model


# ── Weight Export ────────────────────────────────────────────────────────────

def export_weights(model, outdir):
    """Export model weights as raw f32 binary files for the WASM app."""
    os.makedirs(outdir, exist_ok=True)
    state = model.state_dict()
    total_bytes = 0
    for name, param in state.items():
        data = param.cpu().float().numpy()
        data = np.ascontiguousarray(data, dtype=np.float32)
        out_path = os.path.join(outdir, f"{name}.bin")
        data.tofile(out_path)
        total_bytes += data.nbytes
        print(f"  {name}: {list(param.shape)} -> {data.nbytes:,} bytes")
    print(f"\nExported {len(state)} tensors ({total_bytes:,} bytes) to {outdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train Ha et al. 2024 motion magnification model"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=512,
                        help="Number of images in GPU memory pool")
    parser.add_argument("--max-displacement", type=float, default=3.0)
    parser.add_argument("--alpha-min", type=float, default=2.0)
    parser.add_argument("--alpha-max", type=float, default=30.0)
    parser.add_argument("--outdir", type=str, default="weights")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model = train(args)
    print("\nExporting weights for WASM...")
    export_weights(model, args.outdir)
    print("\nDone! Weights are in the weights/ directory.")


if __name__ == "__main__":
    main()
