"""
Single place for project configuration: paths, seed, image size, class names.

Nothing in the project hard-codes an absolute path. The data folder is read from
the environment variable ``MILK10K_DATA_DIR``; if it is not set we fall back to
``<repo_root>/../milk10k_project`` (the layout used on my machine, where the
raw data folder sits next to the repository).

Expected layout of the data folder::

    MILK10K_DATA_DIR/
        metadata.csv
        supplements/training_gt.csv     (or training_gt.csv directly in the folder)
        images/ISIC_xxxxxxx.jpg         (10,480 files)

``MILK10K_IMAGES_DIR`` can optionally override only the image folder.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("MILK10K_DATA_DIR", REPO_ROOT.parent / "milk10k_project"))
IMAGE_DIR = Path(os.environ.get("MILK10K_IMAGES_DIR", DATA_DIR / "images"))
METADATA_CSV = DATA_DIR / "metadata.csv"
GT_CSV = (DATA_DIR / "supplements" / "training_gt.csv"
          if (DATA_DIR / "supplements" / "training_gt.csv").exists()
          else DATA_DIR / "training_gt.csv")

# Generated files (small, committed) live in outputs/, figures in figures/,
# written reports in reports/. Source code never writes next to itself.
OUTPUT_DIR = REPO_ROOT / "outputs"
SPLIT_DIR = OUTPUT_DIR / "splits"
FIGURE_DIR = REPO_ROOT / "figures"
REPORT_DIR = REPO_ROOT / "reports"

LABEL_MAP_JSON = OUTPUT_DIR / "label_map.json"
CLASS_WEIGHTS_JSON = OUTPUT_DIR / "class_weights.json"
NORM_STATS_JSON = OUTPUT_DIR / "norm_stats.json"
LESIONS_CSV = OUTPUT_DIR / "lesions.csv"
IMAGE_SIZES_CSV = OUTPUT_DIR / "image_sizes.csv"
IMAGE_SIZE_SUMMARY_CSV = OUTPUT_DIR / "image_size_summary.csv"

# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #
SEED = 42
IMG_SIZE = 224          # network input (crop) size
RESIZE_SIZE = 256       # shorter side before cropping
VAL_SIZE = 0.15
TEST_SIZE = 0.15
BATCH_SIZE = 32
NUM_WORKERS = int(os.environ.get("MILK10K_NUM_WORKERS", 2))

# --------------------------------------------------------------------------- #
# Label vocabulary
# --------------------------------------------------------------------------- #
# The 11 one-hot columns of training_gt.csv. (The Session 2 code listed only 10
# and silently dropped VASC - fixed here: we always read them from the file.)
FINE_CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
                "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
PRIMARY_CLASSES = ["Benign", "Indeterminate", "Malignant"]
RARE_CLASSES = ["MAL_OTH", "DF", "INF", "VASC", "BEN_OTH"]

# Columns that must never be model inputs (they encode the label or the way
# the label was obtained).
LEAKY_COLUMNS = ["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4",
                 "diagnosis_confirm_type", "concomitant_biopsy", "melanocytic"]


def ensure_output_dirs() -> None:
    """Create the folders for generated files if they do not exist yet."""
    for d in (OUTPUT_DIR, SPLIT_DIR, FIGURE_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def image_path(isic_id: str) -> Path:
    """isic_id -> path of the .jpg file under IMAGE_DIR."""
    return IMAGE_DIR / f"{isic_id}.jpg"


def describe() -> str:
    """One-line summary of the active configuration (printed by scripts)."""
    return (f"DATA_DIR={DATA_DIR} | IMAGE_DIR={IMAGE_DIR} | "
            f"SEED={SEED} | IMG_SIZE={IMG_SIZE}")
