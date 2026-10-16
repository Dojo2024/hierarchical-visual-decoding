"""Pre-extracts and caches PCA projections for all splits and streams.
This avoids repeatedly parsing Arrow datasets during training sweeps,
reducing dataloading time from ~90s to < 0.2s per run!
"""

from pathlib import Path
import datasets as hfds
import numpy as np

DATA_DIR = Path("/data/dojo/nsd_data")
METADATA_DIR = DATA_DIR / "metadata"
CACHE_FILE = DATA_DIR / "pca_features_cache.npz"


def main():
    print(f"Loading masks and PCAs from {METADATA_DIR}...")
    roi_data = np.load(METADATA_DIR / "roi_masks.npz")
    full_mask = roi_data["full_mask"]
    early_idx = roi_data["early_idx_in_full"]
    higher_idx = roi_data["higher_idx_in_full"]

    pca_full = dict(np.load(DATA_DIR / "nsd_flat_pca.npz"))
    pca_early = dict(np.load(METADATA_DIR / "pca_early_visual.npz"))
    pca_higher = dict(np.load(METADATA_DIR / "pca_higher_visual.npz"))

    print(f"Loading cached dataset from {DATA_DIR / 'nsd_flat_cococlip'}...")
    dataset_dict = hfds.load_from_disk(str(DATA_DIR / "nsd_flat_cococlip"))

    cache_dict = {}

    for split in dataset_dict:
        print(f"Processing split: {split} ({len(dataset_dict[split])} samples)...")
        act = np.array(dataset_dict[split]["activity"])
        if act.ndim == 4:
            act = act[:, 0]
        act_full = act[:, full_mask]

        # Per-sample z-normalization
        mean = act_full.mean(axis=1, keepdims=True)
        std = np.clip(act_full.std(axis=1, keepdims=True), 1e-6, None)
        act_full = (act_full - mean) / std

        # 1. Full cortex PCA (256 components)
        feat_full = (act_full - pca_full["mean"]) @ pca_full["components"][:256].T / pca_full["scale"][:256]
        cache_dict[f"{split}_full"] = feat_full.astype(np.float32)

        # 2. Early visual PCA (256 components)
        act_early = act_full[:, early_idx]
        feat_early = (act_early - pca_early["mean"]) @ pca_early["components"][:256].T / pca_early["scale"][:256]
        cache_dict[f"{split}_early"] = feat_early.astype(np.float32)

        # 3. Higher visual PCA (256 components)
        act_higher = act_full[:, higher_idx]
        feat_higher = (act_higher - pca_higher["mean"]) @ pca_higher["components"][:256].T / pca_higher["scale"][:256]
        cache_dict[f"{split}_higher"] = feat_higher.astype(np.float32)

        # 4. Target classes
        cache_dict[f"{split}_target"] = np.array(dataset_dict[split]["target"], dtype=np.int64)

    print(f"Saving precomputed PCA feature cache to {CACHE_FILE}...")
    np.savez(CACHE_FILE, **cache_dict)
    print(f"Saved cache ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)! Dataloading will now be instant (<0.2s).")


if __name__ == "__main__":
    main()
