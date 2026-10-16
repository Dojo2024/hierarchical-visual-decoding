"""Experiment 3: Continuous Latent Retrieval (Decoding 768-d CLIP Visual Embeddings).
Evaluates continuous regression and zero-shot stimulus retrieval across:
1. Early-Only Regressor (V1-V4)
2. Higher-Only Regressor (Ventral LOC/FFA/PPA)
3. Monolithic Control (Full Visual Cortex)
4. Hierarchical Dual-Stream Gated Regressor (Dynamic Gated Fusion)

Directly addresses Kamitani's 2025 papers on visual latent representation and stimulus memorization
by measuring continuous cosine similarity and Top-1 / Top-5 stimulus retrieval accuracy.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import SingleStreamCLIPRegressor, HierarchicalCLIPRegressor

DATA_DIR = Path("/data/dojo/nsd_data")
PCA_CACHE = DATA_DIR / "pca_features_cache.npz"
CLIP_CACHE = DATA_DIR / "clip_targets_cache.npz"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIM = 768


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred and target are both unit-normalized
    return 1.0 - (pred * target).sum(dim=-1).mean()


@torch.no_grad()
def evaluate_retrieval(model, loader, device, is_hierarchical: bool):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        if is_hierarchical:
            x_e, x_h, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pred = model(x_e, x_h)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)
        all_preds.append(pred)
        all_targets.append(y)

    Y_pred = torch.cat(all_preds, dim=0)    # (N, 768)
    Y_true = torch.cat(all_targets, dim=0)  # (N, 768)

    # 1. Mean Cosine Similarity
    mean_cos = (Y_pred * Y_true).sum(dim=-1).mean().item()

    # 2. Top-1 and Top-5 Retrieval Accuracy
    # Note: evaluate in batches of 1000 if memory constrained, but (5000, 5000) is only 100MB
    sim = torch.matmul(Y_pred, Y_true.T)  # (N, N)
    topk = torch.topk(sim, k=5, dim=-1).indices
    idx = torch.arange(Y_pred.shape[0], device=device).unsqueeze(1)

    top1 = (topk[:, :1] == idx).float().mean().item() * 100.0
    top5 = (topk == idx).any(dim=1).float().mean().item() * 100.0

    return {
        "cosine_similarity": round(mean_cos, 4),
        "top1_retrieval_acc": round(top1, 2),
        "top5_retrieval_acc": round(top5, 2),
    }


def train_regressor(model, train_loader, val_loader, testid_loader, test_loader, epochs=30, lr=1e-3, weight_decay=1e-4, device="cuda", is_hierarchical=False):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_cos = -1.0
    best_weights = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for batch in train_loader:
            if is_hierarchical:
                x_e, x_h, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                pred = model(x_e, x_h)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)

            loss = cosine_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        scheduler.step()

        # Check validation cosine similarity
        model.eval()
        with torch.no_grad():
            val_cos_list = []
            for batch in val_loader:
                if is_hierarchical:
                    x_e, x_h, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    p = model(x_e, x_h)
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                    p = model(x)
                val_cos_list.append((p * y).sum(dim=-1).mean().item())
            val_cos = float(np.mean(val_cos_list))

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    wall_time = time.time() - t0
    model.load_state_dict(best_weights)
    model.eval()

    testid_metrics = evaluate_retrieval(model, testid_loader, device, is_hierarchical)
    test_metrics = evaluate_retrieval(model, test_loader, device, is_hierarchical)

    return {
        "val_cosine": round(best_val_cos, 4),
        "testid_cosine": testid_metrics["cosine_similarity"],
        "testid_top1": testid_metrics["top1_retrieval_acc"],
        "testid_top5": testid_metrics["top5_retrieval_acc"],
        "test_cosine": test_metrics["cosine_similarity"],
        "test_top1": test_metrics["top1_retrieval_acc"],
        "test_top5": test_metrics["top5_retrieval_acc"],
        "wall_time": round(wall_time, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_components", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Continuous CLIP Latent Retrieval on {device} (Seed {args.seed})...")

    print("Loading PCA features and CLIP targets caches...")
    pca = np.load(PCA_CACHE)
    clip = np.load(CLIP_CACHE)
    n = args.n_components

    # Targets (already L2-normalized)
    y_train = torch.from_numpy(clip["train_clip"])
    y_val = torch.from_numpy(clip["validation_clip"])
    y_testid = torch.from_numpy(clip["testid_clip"])
    y_test = torch.from_numpy(clip["test_clip"])

    def make_single_loaders(key):
        train_l = DataLoader(TensorDataset(torch.from_numpy(pca[f"train_{key}"][:, :n]), y_train), batch_size=args.batch_size, shuffle=True)
        val_l = DataLoader(TensorDataset(torch.from_numpy(pca[f"validation_{key}"][:, :n]), y_val), batch_size=args.batch_size)
        testid_l = DataLoader(TensorDataset(torch.from_numpy(pca[f"testid_{key}"][:, :n]), y_testid), batch_size=args.batch_size)
        test_l = DataLoader(TensorDataset(torch.from_numpy(pca[f"test_{key}"][:, :n]), y_test), batch_size=args.batch_size)
        return train_l, val_l, testid_l, test_l

    def make_dual_loaders():
        train_l = DataLoader(TensorDataset(torch.from_numpy(pca["train_early"][:, :n]), torch.from_numpy(pca["train_higher"][:, :n]), y_train), batch_size=args.batch_size, shuffle=True)
        val_l = DataLoader(TensorDataset(torch.from_numpy(pca["validation_early"][:, :n]), torch.from_numpy(pca["validation_higher"][:, :n]), y_val), batch_size=args.batch_size)
        testid_l = DataLoader(TensorDataset(torch.from_numpy(pca["testid_early"][:, :n]), torch.from_numpy(pca["testid_higher"][:, :n]), y_testid), batch_size=args.batch_size)
        test_l = DataLoader(TensorDataset(torch.from_numpy(pca["test_early"][:, :n]), torch.from_numpy(pca["test_higher"][:, :n]), y_test), batch_size=args.batch_size)
        return train_l, val_l, testid_l, test_l

    results = {}

    # Arm 1: Early Visual Only
    print("\n--- Arm 1: Early Visual Stream Only (V1-V4) ---")
    set_seed(args.seed)
    tr_e, v_e, tid_e, t_e = make_single_loaders("early")
    m_early = SingleStreamCLIPRegressor(dim=n, latent_dim=256, depth=4)
    results["early_visual"] = train_regressor(m_early, tr_e, v_e, tid_e, t_e, epochs=args.epochs, lr=args.lr, device=device, is_hierarchical=False)
    print(results["early_visual"])

    # Arm 2: Higher Ventral Stream Only
    print("\n--- Arm 2: Higher Ventral Stream Only (LOC/FFA/PPA) ---")
    set_seed(args.seed)
    tr_h, v_h, tid_h, t_h = make_single_loaders("higher")
    m_higher = SingleStreamCLIPRegressor(dim=n, latent_dim=256, depth=4)
    results["higher_visual"] = train_regressor(m_higher, tr_h, v_h, tid_h, t_h, epochs=args.epochs, lr=args.lr, device=device, is_hierarchical=False)
    print(results["higher_visual"])

    # Arm 3: Monolithic Control (Full Cortex)
    print("\n--- Arm 3: Monolithic Control (Full Visual Cortex) ---")
    set_seed(args.seed)
    tr_f, v_f, tid_f, t_f = make_single_loaders("full")
    m_full = SingleStreamCLIPRegressor(dim=n, latent_dim=256, depth=4)
    results["monolithic_control"] = train_regressor(m_full, tr_f, v_f, tid_f, t_f, epochs=args.epochs, lr=args.lr, device=device, is_hierarchical=False)
    print(results["monolithic_control"])

    # Arm 4: Hierarchical Dual-Stream Gated Regressor
    print("\n--- Arm 4: Hierarchical Dual-Stream Gated Regressor ---")
    set_seed(args.seed)
    tr_d, v_d, tid_d, t_d = make_dual_loaders()
    m_dual = HierarchicalCLIPRegressor(dim_early=n, dim_higher=n, latent_dim=256, depth=4)
    results["hierarchical_dual_stream"] = train_regressor(m_dual, tr_d, v_d, tid_d, t_d, epochs=args.epochs, lr=args.lr, device=device, is_hierarchical=True)
    print(results["hierarchical_dual_stream"])

    out_file = RESULTS_DIR / f"continuous_clip_retrieval_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all continuous retrieval results to {out_file}")


if __name__ == "__main__":
    main()

