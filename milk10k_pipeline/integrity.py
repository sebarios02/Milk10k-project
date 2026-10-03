"""
Full-dataset integrity check (Milestone 1, B1).

- every metadata row resolves to an existing file
- every file passes PIL ``Image.verify()``
- width / height of every image (read from the header, no full decode)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

from . import config


def check_image(path: Path) -> dict:
    """Return exists / readable / width / height / bytes / error for one file."""
    rec = {"exists": path.exists(), "readable": False, "width": None, "height": None,
           "bytes": None, "error": None}
    if not rec["exists"]:
        rec["error"] = "missing"
        return rec
    rec["bytes"] = path.stat().st_size
    try:
        with Image.open(path) as im:
            rec["width"], rec["height"] = im.size
            im.verify()                         # checks the file structure, no full decode
        rec["readable"] = True
    except Exception as exc:  # noqa: BLE001 - we want to record ANY failure
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def scan_images(images: pd.DataFrame, img_dir: Optional[Path] = None) -> pd.DataFrame:
    """Run ``check_image`` on every row of the image table (isic_id, image_type)."""
    img_dir = Path(img_dir or config.IMAGE_DIR)
    recs = [check_image(img_dir / f"{i}.jpg") for i in images["isic_id"]]
    out = pd.DataFrame(recs)
    out.insert(0, "isic_id", images["isic_id"].to_numpy())
    out.insert(1, "image_type", images["image_type"].to_numpy())
    return out


def size_summary(scan: pd.DataFrame) -> pd.DataFrame:
    """min / median / max width and height (overall and per image_type) + counts."""
    ok = scan[scan["readable"]]
    rows = []
    for name, grp in [("all", ok)] + list(ok.groupby("image_type")):
        rows.append({
            "subset": name, "n_images": len(grp),
            "width_min": grp["width"].min(), "width_median": grp["width"].median(),
            "width_max": grp["width"].max(),
            "height_min": grp["height"].min(), "height_median": grp["height"].median(),
            "height_max": grp["height"].max(),
            "pct_shorter_side_ge_224": round(100 * (grp[["width", "height"]].min(axis=1) >= 224).mean(), 2),
            "pct_shorter_side_ge_256": round(100 * (grp[["width", "height"]].min(axis=1) >= 256).mean(), 2),
        })
    summ = pd.DataFrame(rows)
    summ["n_rows_in_metadata"] = len(scan)
    summ["n_missing"] = int((~scan["exists"]).sum())
    summ["n_unreadable"] = int((scan["exists"] & ~scan["readable"]).sum())
    return summ
