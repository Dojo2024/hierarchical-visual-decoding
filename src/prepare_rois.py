"""Prepares hierarchical ROI masks (Early Visual vs Higher/Ventral Visual) 
and precomputes stream-specific PCA whitening bases on NSD training data.
"""

from pathlib import Path
import datasets as hfds
import numpy as np
from sklearn.decomposition import PCA

DATA_DIR = Path("/data/dojo/nsd_data")
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = DATA_DIR / "metadata"


def main():
    print(f"Loading full flatmap mask from {METADATA_DIR / 'nsd_flat_mask.npy'}...")
    full_mask = np.load(METADATA_DIR / "nsd_flat_mask.npy")
    H, W = full_mask.shape
    coords = np.argwhere(full_mask)
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    total_voxels = int(full_mask.sum())
    print(f"Full flatmap shape: {full_mask.shape}, active voxels: {total_voxels}")

    # Partition along anterior-posterior gradient:
    # Lower Y (posterior) = Early visual retinotopic pole (V1, V2, V3, hV4)
    # Higher Y (anterior/ventral) = Higher visual ventral stream (LOC, FFA, PPA)
    y_median = np.median(y_coords)
    
    early_mask_2d = full_mask.copy()
    early_mask_2d[coords[y_coords > y_median, 0], coords[y_coords > y_median, 1]] = False

    higher_mask_2d = full_mask.copy()
    higher_mask_2d[coords[y_coords <= y_median, 0], coords[y_coords <= y_median, 1]] = False

    early_idx_in_full = np.where(y_coords <= y_median)[0]
    higher_idx_in_full = np.where(y_coords > y_median)[0]

    print(f"  Early visual stream:  {early_mask_2d.sum()} voxels")
    print(f"  Higher visual stream: {higher_mask_2d.sum()} voxels")

    # Save ROI masks
    roi_file = OUTPUT_DIR / "roi_masks.npz"
    np.savez(
        roi_file,
        full_mask=full_mask,
        early_mask=early_mask_2d,
        higher_mask=higher_mask_2d,
        early_idx_in_full=early_idx_in_full,
        higher_idx_in_full=higher_idx_in_full,
    )
    print(f"Saved ROI masks to {roi_file}")

    # Load training activity to fit stream-specific PCA
    print("\nLoading NSD-Flat training split from local cache...")
    ds = hfds.load_from_disk(str(DATA_DIR / "nsd_flat_cococlip"))
    train_act = np.array(ds["train"]["activity"])
    if train_act.ndim == 4:
        train_act = train_act[:, 0]

    X_train_full = train_act[:, full_mask]
    # Per-sample z-normalization
    mean = X_train_full.mean(axis=1, keepdims=True)
    std = np.clip(X_train_full.std(axis=1, keepdims=True), 1e-6, None)
    X_train_full = (X_train_full - mean) / std
    print(f"X_train_full shape: {X_train_full.shape}")

    # Fit Early Visual PCA
    X_train_early = X_train_full[:, early_idx_in_full]
    print(f"Fitting 256-component PCA on Early Visual voxels ({X_train_early.shape})...")
    pca_early = PCA(n_components=256, svd_solver="randomized", random_state=42)
    pca_early.fit(X_train_early)
    np.savez(
        OUTPUT_DIR / "pca_early_visual.npz",
        components=pca_early.components_.astype(np.float32),
        mean=pca_early.mean_.astype(np.float32),
        scale=np.sqrt(pca_early.explained_variance_).astype(np.float32),
    )
    print(f"  Early Visual explained variance (256 PCs): {pca_early.explained_variance_ratio_.sum():.4f}")

    # Fit Higher Visual PCA
    X_train_higher = X_train_full[:, higher_idx_in_full]
    print(f"Fitting 256-component PCA on Higher Visual voxels ({X_train_higher.shape})...")
    pca_higher = PCA(n_components=256, svd_solver="randomized", random_state=42)
    pca_higher.fit(X_train_higher)
    np.savez(
        OUTPUT_DIR / "pca_higher_visual.npz",
        components=pca_higher.components_.astype(np.float32),
        mean=pca_higher.mean_.astype(np.float32),
        scale=np.sqrt(pca_higher.explained_variance_).astype(np.float32),
    )
    print(f"  Higher Visual explained variance (256 PCs): {pca_higher.explained_variance_ratio_.sum():.4f}")
    print("\nROI and PCA preparation completed successfully!")


if __name__ == "__main__":
    main()
