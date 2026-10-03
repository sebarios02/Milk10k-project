from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def sample_images_per_class(
    df: pd.DataFrame,
    image_dir: str,
    label_col: str,
    id_col: str = "isic_id",
    path_suffix: str = ".jpg",
    n_per_class: int = 25,
    seed: int = 42,
) -> Dict[str, List[Path]]:

    image_dir = Path(image_dir)
    rng = np.random.default_rng(seed)

    result = {}
    for cls, group in df.groupby(label_col):
        paths = [image_dir / f"{row[id_col]}{path_suffix}" for _, row in group.iterrows()]
        available = [p for p in paths if p.exists()]
        if len(available) == 0:
            continue
        chosen_idx = rng.choice(len(available), size=min(n_per_class, len(available)), replace=False)
        result[cls] = [available[i] for i in chosen_idx]
    return result


def _to_gray(img_array: np.ndarray) -> np.ndarray:
    return (0.299 * img_array[:, :, 0] +
            0.587 * img_array[:, :, 1] +
            0.114 * img_array[:, :, 2]).astype(np.uint8)


def average_grayscale_histogram(
    class_to_paths: Dict[str, List[Path]],
    bins: int = 256,
    figsize=(10, 6),
):

    fig, ax = plt.subplots(figsize=figsize)
    all_histograms = {}

    for cls, paths in class_to_paths.items():
        hist_sum = np.zeros(bins)
        n = 0
        for p in paths:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB"))
            gray = _to_gray(arr)
            h, _ = np.histogram(gray, bins=bins, range=(0, 255))
            hist_sum += h
            n += 1
        if n > 0:
            avg_hist = hist_sum / n
            all_histograms[cls] = avg_hist
            ax.plot(avg_hist, label=str(cls))

    ax.set_title("Average Grayscale Intensity Histogram per Class")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Average Frequency")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, all_histograms


def average_rgb_histogram(
    class_to_paths: Dict[str, List[Path]],
    bins: int = 256,
    figsize=(15, 5),
):
    """
    For each class, average the per-channel RGB histograms across its
    sampled images. Produces one figure with 3 subplots (R, G, B), each
    showing one line per class.
    """
    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["red", "green", "blue"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        for cls, paths in class_to_paths.items():
            hist_sum = np.zeros(bins)
            n = 0
            for p in paths:
                with Image.open(p) as im:
                    arr = np.array(im.convert("RGB"))
                h, _ = np.histogram(arr[:, :, ch_idx], bins=bins, range=(0, 255))
                hist_sum += h
                n += 1
            if n > 0:
                ax.plot(hist_sum / n, label=str(cls), alpha=0.8)
        ax.set_title(f"{ch_name} Channel")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Average Frequency")

    axes[-1].legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def channel_summary_stats(
    class_to_paths: Dict[str, List[Path]],
) -> pd.DataFrame:
    """
    Compute per-image mean and std for each RGB channel, then aggregate
    (mean of means, mean of stds) per class.

    Returns a tidy DataFrame: one row per class, columns
    [class, R_mean, R_std, G_mean, G_std, B_mean, B_std, n_images].
    """
    rows = []
    for cls, paths in class_to_paths.items():
        per_image_means = {c: [] for c in "RGB"}
        per_image_stds = {c: [] for c in "RGB"}
        for p in paths:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB")).astype(np.float32)
            for i, c in enumerate("RGB"):
                per_image_means[c].append(arr[:, :, i].mean())
                per_image_stds[c].append(arr[:, :, i].std())

        row = {"class": cls, "n_images": len(paths)}
        for c in "RGB":
            row[f"{c}_mean"] = float(np.mean(per_image_means[c])) if per_image_means[c] else np.nan
            row[f"{c}_std"] = float(np.mean(per_image_stds[c])) if per_image_stds[c] else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


def plot_channel_summary_boxplots(
    class_to_paths: Dict[str, List[Path]],
    figsize=(15, 5),
):
    """
    Boxplots comparing each class's per-image channel means, one subplot
    per RGB channel. Unlike channel_summary_stats (which aggregates to a
    single number per class), this shows the full per-image spread so you
    can see whether classes actually separate or just have different
    averages with heavy overlap.
    """
    records = []
    for cls, paths in class_to_paths.items():
        for p in paths:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB")).astype(np.float32)
            for i, c in enumerate("RGB"):
                records.append({"class": cls, "channel": c, "mean_intensity": arr[:, :, i].mean()})

    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, c in zip(axes, "RGB"):
        subset = df[df["channel"] == c]
        subset.boxplot(column="mean_intensity", by="class", ax=ax, rot=90)
        ax.set_title(f"{c} channel mean per image")
        ax.set_xlabel("Class")
        ax.set_ylabel("Mean intensity")
    plt.suptitle("")
    plt.tight_layout()
    return fig, df


# ---------------------------------------------------------------------------
# Session 3: lesion-pixel colour statistics (used by the augmentation audit)
# ---------------------------------------------------------------------------

def lesion_mask(rgb: np.ndarray) -> np.ndarray:
    """Rough lesion segmentation: Otsu threshold on the blurred grayscale image.

    Lesions are darker than the surrounding skin, so pixels BELOW the Otsu
    threshold are taken as lesion. Very dark pixels (V < 20, e.g. the black
    vignette of dermoscopes) are excluded. If the mask is tiny (<1% of the
    image) we fall back to the central 50% crop.
    """
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (blur < thr) & (gray > 20)
    if mask.mean() < 0.01:
        h, w = gray.shape
        mask = np.zeros_like(mask)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return mask


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees, result in [0, 360)."""
    a = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def circular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (degrees), in [0, 180]."""
    d = abs(a - b) % 360
    return float(min(d, 360 - d))


def lesion_hue_value(rgb: np.ndarray, mask: Optional[np.ndarray] = None):
    """(circular mean hue in degrees, mean V in [0,255]) of the lesion pixels of an RGB uint8 image."""
    import cv2
    if mask is None:
        mask = lesion_mask(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)       # OpenCV: H in [0,179] -> x2 = degrees
    hue_deg = hsv[..., 0][mask].astype(np.float64) * 2.0
    v = hsv[..., 2][mask].astype(np.float64)
    return circular_mean_deg(hue_deg), float(v.mean())
