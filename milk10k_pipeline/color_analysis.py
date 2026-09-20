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
