"""Experiment 2: 2D Cortical Spatial Topology Preservation.
Compares a 2D Depthwise-Separable ConvNet operating directly on the 215x200 cortical flatmap canvas
against the 1D Monolithic MLP.
Tests whether preserving 2D cortical topography and retinotopic spatial layout reduces
training stimulus memorization and improves generalization on in-distribution and cross-subject test sets.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import Flatmap2DConvNet

DATA_DIR = Path("/data/dojo/nsd_data")
VOX_CACHE = DATA_DIR / "masked_voxels_cache.npz"
MASK_FILE = DATA_DIR / "metadata" / "nsd_flat_mask.npy"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NUM_CLASSES = 24


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(voxels, targets, batch_size=256, shuffle=False):
    # Store as uint8 tensors on CPU to keep memory low and fast
    vox_t = torch.from_numpy(voxels)
    tgt_t = torch.from_numpy(targets)
    return DataLoader(TensorDataset(vox_t, tgt_t), batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running 2D Cortical Spatial ConvNet on {device} (Seed {args.seed})...")

    # Load flatmap mask
    mask = np.load(MASK_FILE)
    coords = np.argwhere(mask)
    mask_y = torch.from_numpy(coords[:, 0]).to(device)
    mask_x = torch.from_numpy(coords[:, 1]).to(device)
    H, W = mask.shape

    # Load masked voxels cache
    print("Loading masked voxels cache...")
    t0 = time.time()
    vox_data = np.load(VOX_CACHE)
    print(f"Loaded cache in {time.time() - t0:.2f}s")

    train_loader = create_loader(vox_data["train_voxels"], vox_data["train_target"], batch_size=args.batch_size, shuffle=True)
    val_loader = create_loader(vox_data["validation_voxels"], vox_data["validation_target"], batch_size=args.batch_size)
    testid_loader = create_loader(vox_data["testid_voxels"], vox_data["testid_target"], batch_size=args.batch_size)
    test_loader = create_loader(vox_data["test_voxels"], vox_data["test_target"], batch_size=args.batch_size)

    # Initialize Flatmap2DConvNet (~113k parameters matching SingleStreamMLP 115k)
    model = Flatmap2DConvNet(num_classes=NUM_CLASSES, base_ch=80, dropout=0.5).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Flatmap2DConvNet parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    def voxels_to_2d_canvas(vox_batch):
        # vox_batch: (B, 18577) uint8 on CPU
        vb = vox_batch.to(device, non_blocking=True).float()
        # Per-sample z-norm
        mean = vb.mean(dim=1, keepdim=True)
        std = torch.clamp(vb.std(dim=1, keepdim=True), min=1e-6)
        vb = (vb - mean) / std

        # Scatter onto 2D canvas (B, 1, H, W)
        B = vb.shape[0]
        canvas = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)
        canvas[:, 0, mask_y, mask_x] = vb
        return canvas

    best_val_acc = 0.0
    best_weights = None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for vox_batch, tgt_batch in train_loader:
            tgt = tgt_batch.to(device, non_blocking=True)
            img = voxels_to_2d_canvas(vox_batch)

            out = model(img)
            loss = criterion(out, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * tgt.size(0)
            correct += (out.argmax(dim=-1) == tgt).sum().item()
            total += tgt.size(0)

        train_acc = 100.0 * correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vox_batch, tgt_batch in val_loader:
                tgt = tgt_batch.to(device, non_blocking=True)
                img = voxels_to_2d_canvas(vox_batch)
                out = model(img)
                val_correct += (out.argmax(dim=-1) == tgt).sum().item()
                val_total += tgt.size(0)

        val_acc = 100.0 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    wall_time = time.time() - t_start
    model.load_state_dict(best_weights)
    model.eval()

    def eval_split(loader):
        preds, targets = [], []
        with torch.no_grad():
            for vox_batch, tgt_batch in loader:
                tgt = tgt_batch.to(device, non_blocking=True)
                img = voxels_to_2d_canvas(vox_batch)
                out = model(img)
                preds.append(out.argmax(dim=-1).cpu())
                targets.append(tgt.cpu())
        p = torch.cat(preds).numpy()
        t = torch.cat(targets).numpy()
        return round(100.0 * accuracy_score(t, p), 2)

    res = {
        "model": "Flatmap2DConvNet",
        "params": num_params,
        "train_acc": eval_split(train_loader),
        "val_acc": eval_split(val_loader),
        "testid_acc": eval_split(testid_loader),
        "test_acc": eval_split(test_loader),
        "wall_time": round(wall_time, 2),
    }

    print("\n=== Final Results (Flatmap2DConvNet) ===")
    print(json.dumps(res, indent=2))

    out_file = RESULTS_DIR / f"flatmap_2d_convnet_s{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
