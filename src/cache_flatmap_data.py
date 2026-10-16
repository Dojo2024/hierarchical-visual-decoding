"""Precomputes and caches:
1. masked_voxels_cache.npz: the 18,577 active cortical flatmap voxels across all splits (uint8, ~970 MB).
2. clip_targets_cache.npz: the 768-d normalized CLIP visual embeddings for all trials across all splits (~150 MB).
This enables lightning-fast experiments (<1s loading) for spatial convnets, shuffled ablations, and continuous latent retrieval.
"""

from pathlib import Path
import numpy as np
from datasets import load_from_disk
import time

DATA_DIR = Path("/data/dojo/nsd_data")
METADATA_DIR = DATA_DIR / "metadata"


def main():
    print("Loading flatmap mask...")
    mask = np.load(METADATA_DIR / "nsd_flat_mask.npy")
    print(f"Mask shape: {mask.shape}, active voxels: {mask.sum()}")

    print("Loading CLIP stimulus embeddings...")
    clip_embeds = np.load(DATA_DIR / "nsd_clip_embeds.npy")
    print(f"CLIP embeddings shape: {clip_embeds.shape}")

    print("Loading HuggingFace NSD-Flat dataset...")
    t0 = time.time()
    ds = load_from_disk(str(DATA_DIR / "nsd_flat_cococlip"))
    print(f"Loaded dataset structure in {time.time() - t0:.2f}s")

    splits = ["train", "validation", "test", "testid", "shared1000"]
    voxels_dict = {}
    clip_dict = {}
    targets_dict = {}

    for split in splits:
        t_split = time.time()
        print(f"\nProcessing {split} split ({len(ds[split])} trials)...")
        
        # Extract 2D images and mask
        act = np.array(ds[split]["activity"])  # (N, 215, 200) uint8
        if act.ndim == 4:
            act = act[:, 0]
        
        voxels = act[:, mask]  # (N, 18577) uint8
        voxels_dict[f"{split}_voxels"] = voxels
        
        # Targets and CLIP latents
        targets = np.array(ds[split]["target"], dtype=np.int64)
        targets_dict[f"{split}_target"] = targets
        
        nsd_ids = np.array(ds[split]["nsd_id"], dtype=np.int64)
        clip_targets = clip_embeds[nsd_ids].astype(np.float32)  # (N, 768)
        clip_dict[f"{split}_clip"] = clip_targets
        
        print(f"  {split}: voxels shape {voxels.shape} ({voxels.nbytes / 1e6:.1f} MB), "
              f"clip shape {clip_targets.shape} in {time.time() - t_split:.2f}s")

    # Save masked voxels cache
    vox_cache_path = DATA_DIR / "masked_voxels_cache.npz"
    print(f"\nSaving masked voxels cache to {vox_cache_path}...")
    np.savez(vox_cache_path, **voxels_dict, **targets_dict)
    print(f"Saved voxels cache ({vox_cache_path.stat().st_size / 1e6:.1f} MB) successfully!")

    # Save CLIP targets cache
    clip_cache_path = DATA_DIR / "clip_targets_cache.npz"
    print(f"\nSaving CLIP targets cache to {clip_cache_path}...")
    np.savez(clip_cache_path, **clip_dict, **targets_dict)
    print(f"Saved CLIP targets cache ({clip_cache_path.stat().st_size / 1e6:.1f} MB) successfully!")


if __name__ == "__main__":
    main()

