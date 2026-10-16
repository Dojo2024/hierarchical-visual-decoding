"""Runner script to evaluate all 4 streams (Full, Early-only, Higher-only, Hierarchical Dual-Stream)
and summarize comparative metrics (Train, Val, TestID in-distribution, Test cross-subject, Params, Wall time).
"""

import subprocess
import json
from pathlib import Path

PYTHON_EXEC = "/admin/home/dojo/nanobrain/.venv/bin/python"
SCRIPT_PATH = "/admin/home/dojo/hierarchical_decoding/src/train.py"
RESULTS_DIR = Path("/admin/home/dojo/hierarchical_decoding/results")


def run_experiment(stream: str, seed: int = 42, epochs: int = 30):
    cmd = [
        PYTHON_EXEC,
        SCRIPT_PATH,
        "--stream", stream,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output_dir", str(RESULTS_DIR),
    ]
    print(f"\n{'='*70}\nRunning Stream: {stream.upper()} (Seed {seed})\n{'='*70}")
    subprocess.run(cmd, check=True)


def summarize_results():
    print(f"\n\n{'='*75}\nHIERARCHICAL VISUAL CORTEX DECODING: BENCHMARK SUMMARY\n{'='*75}")
    headers = ["Stream", "Voxels", "Params", "Train %", "Val %", "TestID (In-Dist) %", "Test (Cross-Subj) %", "Wall Time (s)"]
    print(f"{headers[0]:<14} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<8} | {headers[4]:<8} | {headers[5]:<18} | {headers[6]:<19} | {headers[7]}")
    print("-" * 105)

    voxel_map = {
        "full": "18,577",
        "early": "9,344",
        "higher": "9,233",
        "hierarchical": "18,577 (dual)",
    }

    summary_rows = []
    for stream in ["full", "early", "higher", "hierarchical"]:
        result_file = RESULTS_DIR / f"result_{stream}_s42.json"
        if result_file.exists():
            with open(result_file) as f:
                res = json.load(f)
            row = (
                f"{stream.capitalize():<14} | "
                f"{voxel_map.get(stream, 'N/A'):<8} | "
                f"{res.get('params', 0):<8,d} | "
                f"{res.get('acc_train', 0):<8.2f} | "
                f"{res.get('acc_validation', 0):<8.2f} | "
                f"{res.get('acc_testid', 0):<18.2f} | "
                f"{res.get('acc_test', 0):<19.2f} | "
                f"{res.get('wall_t', 0):.1f}s"
            )
            print(row)
            summary_rows.append(res)
    print("=" * 105)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    for stream in ["full", "early", "higher", "hierarchical"]:
        run_experiment(stream, seed=args.seed, epochs=args.epochs)

    summarize_results()
