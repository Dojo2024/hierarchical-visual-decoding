#!/usr/bin/env bash
#SBATCH --job-name=nb-nsd-advanced-s42
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/hierarchical_decoding/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/hierarchical_decoding/slurms/%x_%j.err

set -euo pipefail

cd /admin/home/dojo/hierarchical_decoding
echo "=== Advanced NSD Experiments Started: $(date) on $(hostname) ==="
echo "GPU Device: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"

PYTHON="/admin/home/dojo/nanobrain/.venv/bin/python"

echo ""
echo "=========================================================="
echo ">>> [1/3] Biological vs. Random Shuffled Dual-Stream Ablation"
echo "=========================================================="
$PYTHON experiments/run_shuffled_hierarchy_ablation.py --seed 42 --epochs 30

echo ""
echo "=========================================================="
echo ">>> [2/3] 2D Cortical Spatial Topology (Flatmap2DConvNet)"
echo "=========================================================="
$PYTHON experiments/run_2d_flatmap_spatial_decoding.py --seed 42 --epochs 30

echo ""
echo "=========================================================="
echo ">>> [3/3] Continuous CLIP Visual Latent Retrieval (768-d)"
echo "=========================================================="
$PYTHON experiments/run_continuous_latent_retrieval.py --seed 42 --epochs 30

echo ""
echo "=== All Advanced Experiments Completed Successfully: $(date) ==="

