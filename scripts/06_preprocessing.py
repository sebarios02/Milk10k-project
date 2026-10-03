"""B6 - Preprocessing decisions backed by evidence.

1. Input resolution: uses outputs/image_sizes.csv (from 01_integrity_check.py)
   to report which fraction of images is larger than each candidate size.
2. Normalisation statistics computed on the TRAIN split only, after the same
   resize (shorter side 256) + centre crop (224) as eval_transform
   -> outputs/norm_stats.json
3. View strategy: option (a) independent images + lesion-level evaluation
   (implemented in datasets.MILK10kImageDataset + datasets.aggregate_predictions).
"""
import json

import numpy as np
import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from milk10k_pipeline import config


def resize_center_crop(img: Image.Image, resize: int, crop: int) -> np.ndarray:
    """PIL equivalent of T.Resize(resize) + T.CenterCrop(crop); returns float32 HxWx3 in [0,1]."""
    w, h = img.size
    s = resize / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    return np.asarray(img, dtype=np.float32) / 255.0


def resolution_evidence() -> None:
    if not config.IMAGE_SIZES_CSV.exists():
        print("run 01_integrity_check.py first (image_sizes.csv missing)")
        return
    sizes = pd.read_csv(config.IMAGE_SIZES_CSV).dropna(subset=["width"])
    short = sizes[["width", "height"]].min(axis=1)
    print(f"images with known size: {len(sizes)}")
    for cand in (224, 256, 299, 384, 450):
        print(f"  shorter side >= {cand}: {100 * (short >= cand).mean():6.2f}%")
    print("  most common sizes:\n", sizes.groupby(["width", "height"]).size()
          .sort_values(ascending=False).head(5).to_string())


def train_norm_stats(allow_missing: bool = False) -> dict:
    train = pd.read_csv(config.SPLIT_DIR / "train.csv")
    exists = train["isic_id"].map(lambda i: config.image_path(i).exists())
    if not exists.all():
        msg = f"{(~exists).sum()} train images missing"
        if not allow_missing:
            raise FileNotFoundError(msg + " (full image set required for Milestone 1)")
        print(msg + " -> computing on the available ones (allow_missing=True)")
        train = train[exists]
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for i in train["isic_id"]:
        with Image.open(config.image_path(i)) as im:
            a = resize_center_crop(im.convert("RGB"), config.RESIZE_SIZE, config.IMG_SIZE)
        px = a.reshape(-1, 3).astype(np.float64)
        s += px.sum(0)
        s2 += (px ** 2).sum(0)
        n += len(px)
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    stats = {"mean": mean.round(4).tolist(), "std": std.round(4).tolist(),
             "n_images": int(len(train)), "split": "train",
             "preprocessing": f"Resize({config.RESIZE_SIZE}) + CenterCrop({config.IMG_SIZE}), RGB in [0,1]"}
    config.NORM_STATS_JSON.write_text(json.dumps(stats, indent=2))
    return stats


def main(allow_missing: bool = False) -> None:
    config.ensure_output_dirs()
    print("== 1. resolution evidence ==")
    resolution_evidence()
    print(f"-> chosen: Resize({config.RESIZE_SIZE}) then crop {config.IMG_SIZE}x{config.IMG_SIZE}")
    print("\n== 2. train-only normalisation stats ==")
    print(json.dumps(train_norm_stats(allow_missing), indent=2))
    print("\n== 3. view strategy: (a) independent images, evaluated per lesion (average of the 2 views) ==")


if __name__ == "__main__":
    import sys
    main(allow_missing="--allow-missing" in sys.argv)
