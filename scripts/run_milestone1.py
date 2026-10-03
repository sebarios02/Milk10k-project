"""Run the whole Milestone 1 pipeline in order (01 -> 08). Stops at the first failure."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_integrity_check.py", "02_label_eda.py", "03_label_strategy.py",
         "04_quality_report.py", "05_make_splits.py", "06_preprocessing.py",
         "07_augmentation.py", "08_dataloader_checks.py"]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'=' * 70}\n{step}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / step)], cwd=HERE.parent)
        if r.returncode != 0:
            sys.exit(f"{step} failed (exit code {r.returncode})")
    print("\nMilestone 1 pipeline finished OK.")
