"""Training script for Hierarchical Visual Cortex Decoding on NSD.
Supports single-stream (full, early, higher) and dual-stream hierarchical fusion.
Loads instantly from pre-extracted PCA feature cache.
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

from models import SingleStreamMLP, HierarchicalDualStreamMLP

DATA_DIR = Path("/data/dojo/nsd_data")
CACHE_FILE = DATA_DIR / "pca_features_cache.npz"
NUM_CLASSES = 24


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, is_hierarchical: bool):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if is_hierarchical:
            x_e, x_h, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = model(x_e, x_h)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, is_hierarchical: bool):
    model.eval()
    all_preds = []
    for batch in loader:
        if is_hierarchical:
            x_e, x_h = batch[0].to(device), batch[1].to(device)
            out = model(x_e, x_h)
        else:
            x = batch[0].to(device)
            out = model(x)
        all_preds.append(out.argmax(dim=-1).cpu())
    return torch.cat(all_preds).numpy()


def main():
    parser = argparse.ArgumentParser(description="Hierarchical NSD Visual Decoding")
    parser.add_argument("--stream", type=str, required=True, choices=["full", "early", "higher", "hierarchical"])
    parser.add_argument("--n_components", type=int, default=96, help="PCA components per stream")
    parser.add_argument("--depth", type=int, default=6, help="Residual blocks depth")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dim for dual-stream fusion")
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/admin/home/dojo/hierarchical_decoding/results")
    args = parser.parse_args()

    set_seed(args.seed)
    start_t = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.stream.upper()}] Starting run on device: {device}, seed: {args.seed}, components: {args.n_components}")

    print(f"Loading precomputed PCA feature cache from {CACHE_FILE}...")
    cache = np.load(CACHE_FILE)
    is_hierarchical = args.stream == "hierarchical"
    splits = {}

    split_names = ["train", "validation", "test", "testid", "shared1000"]
    for split in split_names:
        tgt = torch.tensor(cache[f"{split}_target"], dtype=torch.long)
        if args.stream == "full":
            act = torch.tensor(cache[f"{split}_full"][:, :args.n_components], dtype=torch.float32)
            splits[split] = (act, tgt)
        elif args.stream == "early":
            act = torch.tensor(cache[f"{split}_early"][:, :args.n_components], dtype=torch.float32)
            splits[split] = (act, tgt)
        elif args.stream == "higher":
            act = torch.tensor(cache[f"{split}_higher"][:, :args.n_components], dtype=torch.float32)
            splits[split] = (act, tgt)
        elif args.stream == "hierarchical":
            act_e = torch.tensor(cache[f"{split}_early"][:, :args.n_components], dtype=torch.float32)
            act_h = torch.tensor(cache[f"{split}_higher"][:, :args.n_components], dtype=torch.float32)
            splits[split] = (act_e, act_h, tgt)

    # Dataloaders
    train_ds = TensorDataset(*splits["train"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loaders = {s: DataLoader(TensorDataset(*splits[s]), batch_size=512, shuffle=False) for s in splits}

    # Model instantiation
    if is_hierarchical:
        model = HierarchicalDualStreamMLP(
            dim_early=args.n_components,
            dim_higher=args.n_components,
            latent_dim=args.latent_dim,
            depth=args.depth,
            dropout=args.dropout,
            drop_path=args.drop_path,
        ).to(device)
    else:
        model = SingleStreamMLP(
            dim=args.n_components,
            depth=args.depth,
            dropout=args.dropout,
            drop_path=args.drop_path,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, is_hierarchical)
        scheduler.step()

        val_preds = evaluate(model, eval_loaders["validation"], device, is_hierarchical)
        val_targets = splits["validation"][-1].numpy()
        val_acc = 100.0 * accuracy_score(val_targets, val_preds)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            elapsed = time.monotonic() - start_t
            print(f"Epoch {epoch+1:2d}/{args.epochs} | loss={loss:.4f} | train={train_acc:.1f}% | val={val_acc:.1f}% | {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate best model on all splits
    model.load_state_dict(best_state)
    model.to(device)

    results = {
        "stream": args.stream,
        "n_components": args.n_components,
        "seed": args.seed,
        "params": total_params,
        "wall_t": round(time.monotonic() - start_t, 2),
    }

    for split in split_names:
        preds = evaluate(model, eval_loaders[split], device, is_hierarchical)
        targets = splits[split][-1].numpy()
        acc = round(100.0 * accuracy_score(targets, preds), 2)
        results[f"acc_{split}"] = acc

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"result_{args.stream}_s{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal Result [{args.stream}]: {json.dumps(results)}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
