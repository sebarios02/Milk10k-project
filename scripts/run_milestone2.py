"""Run the whole Milestone 2 (classical baseline) in order: 09 -> 12. Stops at the first failure.

Pass --skip-features to reuse the committed outputs/features/*.csv.gz (no images needed).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["09_extract_features.py", "10_model_selection.py", "11_final_evaluation.py", "12_fine_target.py"]

if __name__ == "__main__":
    steps = STEPS[1:] if "--skip-features" in sys.argv else STEPS
    for s in steps:
        print(f"\n=== {s} ===", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s)])
        if r.returncode != 0:
            sys.exit(f"{s} failed with exit code {r.returncode}")
    print("\nMilestone 2 done. Report: reports/milestone2_classical_baseline.md")
