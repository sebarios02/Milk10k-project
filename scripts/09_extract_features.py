"""Milestone 2, step 1: extract hand-crafted features for all 10,480 images.

Output
------
outputs/features/image_features.csv.gz   one row per image (isic_id + ~150 features)
outputs/features/lesion_features.csv.gz  one row per lesion: d_* (dermoscopic) and
                                          c_* (clinical) features side by side +
                                          metadata + split + labels
figures/m2_lesion_masks.png               Otsu masks on random train images (sanity check)

Run:  python scripts/09_extract_features.py [--workers 4] [--limit N]
Takes ~0.13 s per image on one core (about 10 min with 2-4 workers).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from milk10k_pipeline import config
from milk10k_pipeline import features as F

FEAT_DIR = config.OUTPUT_DIR / "features"
META_COLS = ["age_approx", "sex", "anatom_site_general"]


def _one(isic_id: str):
    return isic_id, F.extract_from_path(config.image_path(isic_id))


def load_split_images() -> pd.DataFrame:
    parts = [pd.read_csv(config.SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")]
    return pd.concat(parts, ignore_index=True)


def mask_figure(images: pd.DataFrame, n: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(config.SEED)
    sub = images[images["split"] == "train"]
    rows = []
    for view in F.VIEW_PREFIX:
        v = sub[sub["image_type"] == view]
        rows.append(v.iloc[rng.choice(len(v), n // 2, replace=False)])
    sample = pd.concat(rows)
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 6.2))
    for ax, (_, r) in zip(axes.flat, sample.iterrows()):
        rgb = F.load_rgb(config.image_path(r["isic_id"]))
        m, fb = F.lesion_mask(rgb)
        ax.imshow(rgb)
        ax.contour(m, colors="lime", linewidths=1)
        ax.set_title(f"{r['dx']} | {'derm' if r['image_type']=='dermoscopic' else 'clinical'}"
                     + (" | fallback" if fb else ""), fontsize=9)
        ax.axis("off")
    fig.suptitle("Lesion masks used for colour / texture / shape features (random train images, seed 42)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "m2_lesion_masks.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N images")
    args = ap.parse_args()

    config.ensure_output_dirs()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    print(config.describe())
    images = load_split_images()
    ids = images["isic_id"].tolist()[: args.limit]

    t0 = time.time()
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (isic_id, feats) in enumerate(ex.map(_one, ids, chunksize=32), 1):
            rows[isic_id] = feats
            if k % 1000 == 0:
                print(f"  {k}/{len(ids)} images  ({time.time() - t0:.0f}s)")
    img_feats = pd.DataFrame.from_dict(rows, orient="index")
    img_feats.index.name = "isic_id"
    img_feats.to_csv(FEAT_DIR / "image_features.csv.gz", float_format="%.6g")
    print(f"image features: {img_feats.shape} in {time.time() - t0:.0f}s")

    les = F.to_lesion_features(img_feats, images)
    info = (images.drop_duplicates("lesion_id")
                  .set_index("lesion_id")[["split", "diagnosis_1", "dx", "fine", "y_primary",
                                           "y_fine"] + META_COLS])
    les = info.join(les, how="inner")
    les.to_csv(FEAT_DIR / "lesion_features.csv.gz", float_format="%.6g")
    print(f"lesion features: {les.shape}  (splits: {les['split'].value_counts().to_dict()})")
    print(f"mask fallback rate: derm {les['d_shp_mask_fallback'].mean():.1%}, "
          f"clinical {les['c_shp_mask_fallback'].mean():.1%}")
    if args.limit is None:
        mask_figure(images)


if __name__ == "__main__":
    main()
