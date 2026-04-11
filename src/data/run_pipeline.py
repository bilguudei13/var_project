# =============================================================================
# run_pipeline.py
# Master script — runs the full data pipeline in order
# Usage: python src/data/run_pipeline.py
# =============================================================================

import subprocess
import sys

steps = [
    ("Download raw data",        "src/data/download_data.py"),
    ("Compute returns",          "src/data/compute_returns.py"),
    ("Compute instrument P&L",   "src/data/compute_pnl.py"),
]

for name, script in steps:
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], check=True)

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)