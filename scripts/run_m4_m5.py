"""Run Milestones 4 and 5 in order: 13 -> 14 -> 15. Stops at the first failure.

    python scripts/run_m4_m5.py --smoke    # ~5 min check of every code path (writes *_smoke outputs)
    python scripts/run_m4_m5.py            # the real run (~1.5-2 h on a 4-thread CPU)

Then, once the ISIC test data is downloaded:
    python m5_final/make_submission.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["13_m4_best_model.py", "14_m4_error_fairness.py", "15_m5_train_11class.py"]

if __name__ == "__main__":
    extra = ["--smoke"] if "--smoke" in sys.argv else []
    t0 = time.perf_counter()
    for s in STEPS:
        print(f"\n=== {s} {' '.join(extra)} ===", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s), *extra])
        if r.returncode != 0:
            sys.exit(f"{s} failed with exit code {r.returncode}")
    print(f"\nMilestones 4-5 done in {(time.perf_counter() - t0) / 60:.0f} min. "
          "Next: python m5_final/make_submission.py")
