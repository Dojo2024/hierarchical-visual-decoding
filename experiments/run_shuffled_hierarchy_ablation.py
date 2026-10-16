"""Experiment 1: Biological Hierarchy vs. Random Shuffled Dual-Stream Ablation.
Evaluates whether functional division of labor (Early V1-V4 vs Higher Ventral LOC/FFA/PPA)
provides an inductive bias against stimulus memorization, or if arbitrary 50/50 stream
partitioning with cross-stream gating achieves identical performance.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import SingleStreamMLP, HierarchicalDualStreamMLP

DATA_DIR = Path("/data/dojo/nsd_data")
VOX_CACHE = DATA_DIR / "masked_voxels_cache.npz"
PCA_CACHE = DATA_DIR / "pca_features_cache.npz"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NUM_CLASSES = 24


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, val_loader, testid_loader, test_loader, epochs=30, lr=5e-4, weight_decay=1e-4, device="cuda", is_dual=False):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            if is_dual:
                x1, x2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = model(x1, x2)
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

        train_acc = 100.0 * correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if is_dual:
                    x1, x2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    out = model(x1, x2)
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                    out = model(x)
                val_correct += (out.argmax(dim=-1) == y).sum().item()
                val_total += y.size(0)
        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    wall_time = time.time() - t0
    model.load_state_dict(best_weights)
    model.eval()

    # Final evaluation on all splits
    def eval_split(loader):
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                if is_dual:
                    x1, x2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    out = model(x1, x2)
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                    out = model(x)
                preds.append(out.argmax(dim=-1).cpu())
                targets.append(y.cpu())
        p = torch.cat(preds).numpy()
        t = torch.cat(targets).numpy()
        return 100.0 * accuracy_score(t, p)

    final_train = eval_split(train_loader)
    final_val = eval_split(val_loader)
    final_testid = eval_split(testid_loader)
    final_test = eval_split(test_loader)

    return {
        "train_acc": round(final_train, 2),
        "val_acc": round(final_val, 2),
        "testid_acc": round(final_testid, 2),
        "test_acc": round(final_test, 2),
        "wall_time": round(wall_time, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_components", type=int, default=96)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Biological vs Shuffled Stream Ablation on {device} (Seed {args.seed})...")

    # 1. Load biological PCA features (pre-extracted)
    print("Loading precomputed biological PCA features...")
    bio_data = np.load(PCA_CACHE)
    n = args.n_components

    # 2. Fit Shuffled PCA features
    print("Loading raw masked flatmap voxels to compute random shuffled partition...")
    vox_data = np.load(VOX_CACHE)
    total_voxels = vox_data["train_voxels"].shape[1]
    
    # Generate random 50/50 partition
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(total_voxels)
    split_pt = total_voxels // 2
    idx_shuff_a = perm[:split_pt]
    idx_shuff_b = perm[split_pt:]

    print(f"Fitting PCA ({n} PCs) on Shuffled Stream A ({len(idx_shuff_a)} voxels)...")
    pca_a = PCA(n_components=n, svd_solver="randomized", random_state=args.seed)
    X_train_raw = vox_data["train_voxels"].astype(np.float32)
    # Per-sample z-norm
    m = X_train_raw.mean(axis=1, keepdims=True)
    s = np.clip(X_train_raw.std(axis=1, keepdims=True), 1e-6, None)
    X_train_norm = (X_train_raw - m) / s

    X_train_shuff_a = pca_a.fit_transform(X_train_norm[:, idx_shuff_a])

    print(f"Fitting PCA ({n} PCs) on Shuffled Stream B ({len(idx_shuff_b)} voxels)...")
    pca_b = PCA(n_components=n, svd_solver="randomized", random_state=args.seed)
    X_train_shuff_b = pca_b.fit_transform(X_train_norm[:, idx_shuff_b])

    # Transform val, testid, test
    def transform_split(split_name):
        raw = vox_data[f"{split_name}_voxels"].astype(np.float32)
        m_s = raw.mean(axis=1, keepdims=True)
        s_s = np.clip(raw.std(axis=1, keepdims=True), 1e-6, None)
        norm = (raw - m_s) / s_s
        fa = pca_a.transform(norm[:, idx_shuff_a])
        fb = pca_b.transform(norm[:, idx_shuff_b])
        return fa, fb, vox_data[f"{split_name}_target"]

    X_val_shuff_a, X_val_shuff_b, y_val = transform_split("validation")
    X_testid_shuff_a, X_testid_shuff_b, y_testid = transform_split("testid")
    X_test_shuff_a, X_test_shuff_b, y_test = transform_split("test")
    y_train = vox_data["train_target"]

    # Build data loaders
    # Bio
    train_bio_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(bio_data["train_early"][:, :n]),
            torch.from_numpy(bio_data["train_higher"][:, :n]),
            torch.from_numpy(y_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_bio_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(bio_data["validation_early"][:, :n]),
            torch.from_numpy(bio_data["validation_higher"][:, :n]),
            torch.from_numpy(y_val),
        ),
        batch_size=args.batch_size,
    )
    testid_bio_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(bio_data["testid_early"][:, :n]),
            torch.from_numpy(bio_data["testid_higher"][:, :n]),
            torch.from_numpy(y_testid),
        ),
        batch_size=args.batch_size,
    )
    test_bio_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(bio_data["test_early"][:, :n]),
            torch.from_numpy(bio_data["test_higher"][:, :n]),
            torch.from_numpy(y_test),
        ),
        batch_size=args.batch_size,
    )

    # Shuffled
    train_shuff_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_shuff_a.astype(np.float32)),
            torch.from_numpy(X_train_shuff_b.astype(np.float32)),
            torch.from_numpy(y_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_shuff_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val_shuff_a.astype(np.float32)),
            torch.from_numpy(X_val_shuff_b.astype(np.float32)),
            torch.from_numpy(y_val),
        ),
        batch_size=args.batch_size,
    )
    testid_shuff_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_testid_shuff_a.astype(np.float32)),
            torch.from_numpy(X_testid_shuff_b.astype(np.float32)),
            torch.from_numpy(y_testid),
        ),
        batch_size=args.batch_size,
    )
    test_shuff_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test_shuff_a.astype(np.float32)),
            torch.from_numpy(X_test_shuff_b.astype(np.float32)),
            torch.from_numpy(y_test),
        ),
        batch_size=args.batch_size,
    )

    results = {}

    # Arm 1: Biological Dual-Stream
    print("\n--- Arm 1: Biological Dual-Stream (Early vs Higher) ---")
    set_seed(args.seed)
    model_bio = HierarchicalDualStreamMLP(dim_early=n, dim_higher=n, latent_dim=128, depth=4)
    results["biological_dual_stream"] = train_model(
        model_bio, train_bio_loader, val_bio_loader, testid_bio_loader, test_bio_loader,
        epochs=args.epochs, device=device, is_dual=True
    )
    print(results["biological_dual_stream"])

    # Arm 2: Shuffled Dual-Stream (Random 50/50 partition)
    print("\n--- Arm 2: Shuffled Dual-Stream (Random 50/50 cortical partition) ---")
    set_seed(args.seed)
    model_shuff = HierarchicalDualStreamMLP(dim_early=n, dim_higher=n, latent_dim=128, depth=4)
    results["shuffled_dual_stream"] = train_model(
        model_shuff, train_shuff_loader, val_shuff_loader, testid_shuff_loader, test_shuff_loader,
        epochs=args.epochs, device=device, is_dual=True
    )
    print(results["shuffled_dual_stream"])

    out_file = RESULTS_DIR / f"shuffled_ablation_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()

