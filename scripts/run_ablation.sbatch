#!/usr/bin/env bash
#SBATCH --job-name=nb-hier-ablation-s42
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/hierarchical_decoding/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/hierarchical_decoding/slurms/%x_%j.err

set -euo pipefail

cd /admin/home/dojo/hierarchical_decoding
echo "=== Job Started: $(date) on $(hostname) ==="
echo "GPU Device: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"

PYTHON="/admin/home/dojo/nanobrain/.venv/bin/python"

$PYTHON experiments/run_hierarchy_ablation.py --seed 42 --epochs 30

echo "=== Job Finished: $(date) ==="

