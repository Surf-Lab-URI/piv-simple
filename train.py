#!/usr/bin/env python
"""
Simplified PIV-LiteFlowNet Training & Inference Pipeline

Usage:
    # Training
    python train.py --mode train --data /path/to/piv_data --epochs 100
    
    # Resume training
    python train.py --mode train --data /path/to/piv_data --resume checkpoint.pth
    
    # Inference on directory
    python train.py --mode infer --data /path/to/images --weights model_best.pth --output ./results
    
    # Inference on single pair
    python train.py --mode infer --img1 a.tif --img2 b.tif --weights model_best.pth
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# === Existing modules (use as-is) ===
from src.models import piv_liteflownet, hui_liteflownet
from src.loss import piv_loss, hui_loss
from src.utils_plot import write_flow, motion_to_color

# === Simplified dataset (in src/) ===
from src.piv_dataset_simple import PIVDataset, PIVInference, make_splits, get_transform


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Training configuration - replaces argparse complexity."""
    # Data
    data_root: str = ""
    subsets: List[str] = field(default_factory=list)  # empty = all subdirs
    ext: str = "tif"
    crop_size: Tuple[int, int] = (256, 256)
    
    # Model
    model: str = "piv"  # "piv" or "hui"
    version: int = 1    # 1 = LiteFlowNet, 2 = LiteFlowNet2
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 4e-4
    lr_decay_epoch: int = 50
    lr_decay_factor: float = 0.5
    
    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    seed: int = 42
    
    # System
    num_workers: int = 4
    device: str = "cuda"
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_freq: int = 10  # save every N epochs
    
    # Loss
    loss_scale: int = 5  # mul_scale for piv_loss (5 for piv, 20 for hui)
    loss_norm: str = "L1"
    
    def __post_init__(self):
        if not self.subsets:
            self.subsets = None  # None means all subdirs


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    total_epe = 0.0
    
    for i, (imgs, flows) in enumerate(loader):
        # imgs: [tensor(B,3,H,W), tensor(B,3,H,W)], flows: [tensor(B,2,H,W)]
        img1 = imgs[0].to(device)
        img2 = imgs[1].to(device)
        flow_gt = flows[0].to(device)
        
        optimizer.zero_grad()
        
        # Forward - training mode returns pyramid outputs
        flow_pred = model(img1, img2)
        
        # Loss
        loss_values = criterion(flow_pred, flow_gt)
        loss, epe = loss_values[0], loss_values[1]
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_epe += epe.item()
        
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(loader)}: loss={loss.item():.4f}, EPE={epe.item():.4f}")
    
    n = len(loader)
    return total_loss / n, total_epe / n


def validate(model, loader, criterion, device):
    """Validation pass."""
    model.eval()
    total_loss = 0.0
    total_epe = 0.0
    
    with torch.no_grad():
        for imgs, flows in loader:
            # imgs: [tensor(B,3,H,W), tensor(B,3,H,W)], flows: [tensor(B,2,H,W)]
            img1 = imgs[0].to(device)
            img2 = imgs[1].to(device)
            flow_gt = flows[0].to(device)
            
            # Forward - eval mode returns final flow
            flow_pred = model(img1, img2)
            
            # For validation, compute EPE at full resolution
            # flow_pred is already scaled in eval mode
            epe = torch.norm(flow_pred - flow_gt, p=2, dim=1).mean()
            
            total_epe += epe.item()
    
    return total_epe / len(loader)


def train(cfg: Config, resume: Optional[str] = None):
    """Main training loop."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # === Data ===
    print(f"\nLoading data from {cfg.data_root}")
    train_data, val_data, test_data = make_splits(
        root=cfg.data_root,
        subsets=cfg.subsets,
        ext=cfg.ext,
        crop_size=cfg.crop_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    
    train_loader = DataLoader(
        train_data, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    # === Model ===
    print(f"\nCreating {cfg.model} model (version {cfg.version})")
    if cfg.model == "piv":
        model = piv_liteflownet(params=None, version=cfg.version)
        criterion = piv_loss(mul_scale=cfg.loss_scale, norm=cfg.loss_norm, version=cfg.version)
    else:
        model = hui_liteflownet(params=None, version=cfg.version)
        criterion = hui_loss(mul_scale=20, norm=cfg.loss_norm)
    
    model = model.to(device)
    
    # === Optimizer ===
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.lr_decay_epoch, 
        gamma=cfg.lr_decay_factor
    )
    
    # === Resume ===
    start_epoch = 0
    best_epe = float('inf')
    
    if resume:
        print(f"\nResuming from {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_epe = ckpt.get('best_epe', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, best_epe={best_epe:.4f}")
    
    # === Training loop ===
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    
    print(f"\nStarting training for {cfg.epochs} epochs")
    print("=" * 60)
    
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        
        # Train
        train_loss, train_epe = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_epe = validate(model, val_loader, criterion, device)
        
        # LR step
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        dt = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} train_epe={train_epe:.4f} | "
              f"val_epe={val_epe:.4f} | lr={lr:.2e} | {dt:.1f}s")
        
        # Checkpoint
        is_best = val_epe < best_epe
        best_epe = min(val_epe, best_epe)
        
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_epe': best_epe,
            'config': asdict(cfg),
        }
        
        if (epoch + 1) % cfg.save_freq == 0:
            torch.save(ckpt, save_dir / f"checkpoint_{epoch+1:03d}.pth")
        
        torch.save(ckpt, save_dir / "checkpoint_last.pth")
        
        if is_best:
            torch.save(ckpt, save_dir / "model_best.pth")
            print(f"  *** New best model (EPE={best_epe:.4f}) ***")
    
    print("\nTraining complete!")
    print(f"Best validation EPE: {best_epe:.4f}")
    return model


# ============================================================================
# Inference
# ============================================================================

def load_model(weights_path: str, model_type: str = "piv", version: int = 1, device: str = "cuda"):
    """Load trained model from checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(weights_path, map_location=device)
    
    # Try to get config from checkpoint
    if 'config' in ckpt:
        model_type = ckpt['config'].get('model', model_type)
        version = ckpt['config'].get('version', version)
    
    if model_type == "piv":
        model = piv_liteflownet(params=ckpt['model'], version=version)
    else:
        model = hui_liteflownet(params=ckpt['model'], version=version)
    
    model = model.to(device)
    model.eval()
    
    return model, device


def infer_pair(model, img1_path: str, img2_path: str, device, output_path: Optional[str] = None):
    """
    Run inference on a single image pair.
    Uses adaptive sizing (like original estimate() function) to handle arbitrary image sizes.
    """
    # Load images
    img1 = np.array(Image.open(img1_path).convert('RGB'), dtype=np.float32) / 255.0
    img2 = np.array(Image.open(img2_path).convert('RGB'), dtype=np.float32) / 255.0
    
    input_height, input_width = img1.shape[:2]
    
    # Adaptive sizing - round up to multiple of 32 (original uses 32, not 64)
    adaptive_width = int(math.ceil(input_width / 32.0) * 32.0)
    adaptive_height = int(math.ceil(input_height / 32.0) * 32.0)
    
    # Scale factors for flow correction
    scale_width = float(input_width) / float(adaptive_width)
    scale_height = float(input_height) / float(adaptive_height)
    
    # To tensor
    img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).to(device)
    img2_t = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # Resize to adaptive size
    img1_t = torch.nn.functional.interpolate(
        img1_t, size=(adaptive_height, adaptive_width), mode='bilinear', align_corners=False
    )
    img2_t = torch.nn.functional.interpolate(
        img2_t, size=(adaptive_height, adaptive_width), mode='bilinear', align_corners=False
    )
    
    # Inference
    with torch.no_grad():
        flow = model(img1_t, img2_t)
    
    # Resize flow back to original size
    flow = torch.nn.functional.interpolate(
        flow, size=(input_height, input_width), mode='bilinear', align_corners=False
    )
    
    # Correct flow magnitudes for the scaling
    flow[:, 0, :, :] *= scale_width
    flow[:, 1, :, :] *= scale_height
    
    # Back to numpy
    flow = flow[0].cpu().numpy().transpose(1, 2, 0)
    
    # Save if requested
    if output_path:
        write_flow(flow, output_path)
        
        # Also save visualization
        vis_path = output_path.replace('.flo', '_vis.png')
        vis = motion_to_color(flow)
        Image.fromarray(vis).save(vis_path)
        print(f"Saved: {output_path}, {vis_path}")
    
    return flow


def infer_directory(model, data_root: str, device, output_dir: str, 
                    subsets: List[str] = None, ext: str = "tif"):
    """Run inference on all image pairs in directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = PIVInference(data_root, subsets=subsets, ext=ext)
    
    # batch_size=1 since images may have different sizes
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for img1, img2, name in loader:
        # Get original size
        _, _, input_height, input_width = img1.shape
        
        # Adaptive sizing
        adaptive_width = int(math.ceil(input_width / 32.0) * 32.0)
        adaptive_height = int(math.ceil(input_height / 32.0) * 32.0)
        scale_width = float(input_width) / float(adaptive_width)
        scale_height = float(input_height) / float(adaptive_height)
        
        # Resize and move to device
        img1 = torch.nn.functional.interpolate(
            img1, size=(adaptive_height, adaptive_width), mode='bilinear', align_corners=False
        ).to(device)
        img2 = torch.nn.functional.interpolate(
            img2, size=(adaptive_height, adaptive_width), mode='bilinear', align_corners=False
        ).to(device)
        
        with torch.no_grad():
            flow = model(img1, img2)
        
        # Resize back and correct flow magnitudes
        flow = torch.nn.functional.interpolate(
            flow, size=(input_height, input_width), mode='bilinear', align_corners=False
        )
        flow[:, 0, :, :] *= scale_width
        flow[:, 1, :, :] *= scale_height
        
        flow = flow[0].cpu().numpy().transpose(1, 2, 0)
        
        # Save
        out_path = output_dir / f"{name[0]}_flow.flo"
        write_flow(flow, str(out_path))
        
        # Visualization
        vis = motion_to_color(flow)
        vis_path = output_dir / f"{name[0]}_vis.png"
        Image.fromarray(vis).save(vis_path)
        
        print(f"  {name[0]}")
    
    print(f"\nSaved {len(dataset)} flow files to {output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PIV-LiteFlowNet Training & Inference")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"],
                        help="train or infer")
    
    # Data
    parser.add_argument("--data", type=str, default="", help="Data root directory")
    parser.add_argument("--subsets", nargs="+", default=[], help="Subdirectories to use")
    parser.add_argument("--ext", type=str, default="tif", help="Image extension")
    
    # Model
    parser.add_argument("--model", type=str, default="piv", choices=["piv", "hui"])
    parser.add_argument("--version", type=int, default=1, choices=[1, 2])
    parser.add_argument("--weights", type=str, default="", help="Pretrained weights for inference")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    
    # Inference
    parser.add_argument("--img1", type=str, default="", help="First image (single pair mode)")
    parser.add_argument("--img2", type=str, default="", help="Second image (single pair mode)")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    
    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        cfg = Config(
            data_root=args.data,
            subsets=args.subsets if args.subsets else [],
            ext=args.ext,
            crop_size=tuple(args.crop_size),
            model=args.model,
            version=args.version,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=args.save_dir,
            num_workers=args.workers,
            device=args.device,
        )
        train(cfg, resume=args.resume if args.resume else None)
    
    elif args.mode == "infer":
        if not args.weights:
            parser.error("--weights required for inference")
        
        model, device = load_model(args.weights, args.model, args.version, args.device)
        
        if args.img1 and args.img2:
            # Single pair mode
            out_path = Path(args.output)
            out_path.mkdir(parents=True, exist_ok=True)
            name = Path(args.img1).stem.rsplit('_img1', 1)[0]
            flow = infer_pair(model, args.img1, args.img2, device, 
                              str(out_path / f"{name}_flow.flo"))
        else:
            # Directory mode
            infer_directory(model, args.data, device, args.output, 
                           args.subsets if args.subsets else None, args.ext)


if __name__ == "__main__":
    main()
