"""make_submission.py: 11-class model -> submission.csv for the official MILK10k leaderboard.

Expected test data (downloaded from challenge.isic-archive.com/data, MILK10k section),
by default in <data folder>/test/ (override with the env var MILK10K_TEST_DIR):

    milk10k_project/test/
    ├── MILK10k_Test_Metadata.csv
    └── MILK10k_Test_Input/<lesion_id>/<isic_id>.jpg     (479 lesions, 958 images)

Header: lesion_id + the 11 classes. (The course brief shows the first column as "lesion",
but the ISIC validator rejects that: "Missing column in CSV: 'image' or 'lesion_id'".)

One row per lesion: the model was trained on the dermoscopic image only, so only the
dermoscopic image of each lesion is used. The post-processing chosen on validation
(outputs/m5/postprocess.json) is applied, then the 4 checks of the course brief run.

    python m5_final/make_submission.py      -> m5_final/submission.csv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from milk10k_pipeline import config, deep  # noqa: E402
from m5_final.predict import predict_batch  # noqa: E402

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
TEST_DIR = Path(os.environ.get("MILK10K_TEST_DIR", config.DATA_DIR / "test"))
OUT = Path(__file__).resolve().parent / "submission.csv"


def find_metadata() -> Path:
    hits = sorted(TEST_DIR.rglob("*Test_Metadata*.csv"))
    if not hits:
        sys.exit(f"MILK10k_Test_Metadata.csv not found under {TEST_DIR}. Download it from "
                 "challenge.isic-archive.com/data and unzip it there (or set MILK10K_TEST_DIR).")
    return hits[0]


def image_file(lesion: str, isic: str) -> Path:
    p = TEST_DIR / "MILK10k_Test_Input" / lesion / f"{isic}.jpg"
    if p.exists():
        return p
    hits = list(TEST_DIR.rglob(f"{isic}.jpg"))
    if not hits:
        raise FileNotFoundError(f"{isic}.jpg (lesion {lesion}) not found under {TEST_DIR}")
    return hits[0]


def main():
    meta = pd.read_csv(find_metadata())
    derm = meta[meta["image_type"] == "dermoscopic"].drop_duplicates("lesion_id").reset_index(drop=True)
    print(f"{len(meta)} test images, {meta.lesion_id.nunique()} lesions, {len(derm)} dermoscopic")
    paths = [image_file(l, i) for l, i in zip(derm["lesion_id"], derm["isic_id"])]

    P, classes = predict_batch(paths, "11class")
    assert list(classes) == CLASSES, f"model classes {classes} != submission order"

    pp = json.loads((config.OUTPUT_DIR / "m5" / "postprocess.json").read_text())
    P = deep.postprocess(P, pp["strategy"], pp.get("thresholds"))
    print("post-processing:", pp["strategy"])

    sub = pd.DataFrame(P, columns=CLASSES)
    sub.insert(0, "lesion_id", derm["lesion_id"].values)   # the ISIC validator requires "lesion_id"

    # the 4 checks of the course brief
    assert list(sub.columns) == ["lesion_id"] + CLASSES
    assert len(sub) == 479 and sub.lesion_id.is_unique, f"{len(sub)} rows"
    assert sub[CLASSES].isna().sum().sum() == 0
    assert ((sub[CLASSES] >= 0) & (sub[CLASSES] <= 1)).all().all()

    sub.to_csv(OUT, index=False)
    print(f"OK -> {OUT}  ({len(sub)} rows x {sub.shape[1]} columns)")
    print("predicted classes (>= 0.5):", (sub[CLASSES] >= 0.5).sum().to_dict())


if __name__ == "__main__":
    main()
