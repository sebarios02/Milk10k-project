"""
Loading the raw CSVs and building the tables every other module uses.

- ``load_metadata()``   per-image metadata (10,480 rows)
- ``load_gt()``         per-lesion one-hot ground truth + a ``dx`` string column
- ``load_images_table()`` metadata merged with ``dx`` (one row per image)
- ``build_lesion_table()`` one row per lesion (5,240 rows), derm + clinical ids
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

DERM = "dermoscopic"
CLINICAL = "clinical: close-up"


def load_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    """Read metadata.csv (one row per image)."""
    return pd.read_csv(path or config.METADATA_CSV)


def load_gt(path: Optional[Path] = None) -> pd.DataFrame:
    """Read training_gt.csv and add ``dx``: the name of the positive one-hot column.

    The class columns are read from the file itself (every column except
    lesion_id) so that no class can be silently dropped.
    """
    gt = pd.read_csv(path or config.GT_CSV)
    class_cols = [c for c in gt.columns if c != "lesion_id"]
    gt["dx"] = gt[class_cols].idxmax(axis=1)
    return gt


def load_images_table() -> pd.DataFrame:
    """Per-image metadata with the 11-class label ``dx`` attached."""
    meta = load_metadata()
    gt = load_gt()
    df = meta.merge(gt[["lesion_id", "dx"]], on="lesion_id", how="left", validate="many_to_one")
    assert df["dx"].notna().all(), "some images have no lesion in training_gt.csv"
    return df


def build_lesion_table(images: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per lesion: lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site.

    Built with pivot (image ids) + groupby().first() (lesion-level fields); no
    Python loop over rows. Lesion-level fields are identical for the two images
    of a lesion (checked in A1.1e), so ``first`` is safe.
    """
    if images is None:
        images = load_images_table()

    ids = (images.pivot(index="lesion_id", columns="image_type", values="isic_id")
                 .rename(columns={DERM: "derm_id", CLINICAL: "clinical_id"}))
    ids.columns.name = None

    fields = (images.groupby("lesion_id")[["diagnosis_1", "dx", "age_approx", "sex",
                                           "anatom_site_general"]]
                    .first()
                    .rename(columns={"age_approx": "age", "anatom_site_general": "site"}))

    lesions = ids[["derm_id", "clinical_id"]].join(fields).reset_index()
    return lesions[["lesion_id", "derm_id", "clinical_id", "diagnosis_1", "dx",
                    "age", "sex", "site"]]


def lesions_to_images(lesions: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Return all image rows whose lesion_id is in ``lesions`` (images follow lesions)."""
    return images[images["lesion_id"].isin(lesions["lesion_id"])].reset_index(drop=True)


def available_mask(isic_ids: pd.Series, image_dir: Optional[Path] = None) -> pd.Series:
    """Boolean mask: does the .jpg for each isic_id exist on disk?"""
    image_dir = Path(image_dir or config.IMAGE_DIR)
    return isic_ids.map(lambda i: (image_dir / f"{i}.jpg").exists())
