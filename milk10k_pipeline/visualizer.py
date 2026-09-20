from typing import List, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt


def _for_display(arr: np.ndarray) -> np.ndarray:
    """Rescale any array to [0, 1] for imshow, regardless of its original
    range (raw 0-255, min-max 0-1, or z-scored around 0)."""
    arr = np.asarray(arr).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def plot_sample_grid(
    images: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[Sequence] = None,
    n_cols: int = 5,
    figsize_per_cell: float = 2.2,
    title: Optional[str] = None,
):
    n = len(images)
    n_cols = min(n_cols, n) if n > 0 else 1
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_cell, n_rows * figsize_per_cell))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            display_img = np.squeeze(_for_display(images[i]))
            cmap = "gray" if display_img.ndim == 2 else None
            ax.imshow(display_img, cmap=cmap)
            if labels is not None:
                ax.set_title(str(labels[i]), fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_class_balance(
    labels: Sequence,
    title: str = "Class Balance",
    figsize=(8, 5),
):
    """
    Plot a bar chart of how many samples fall into each class.

    Parameters
    ----------
    labels : sequence of class labels (e.g. a metadata column, or the
              concatenated labels seen across a data loader's batches)

    Returns the matplotlib Figure, and prints a one-line balanced/imbalanced
    verdict (ratio of the largest to the smallest class) so this can be
    reused as a quick diagnostic, not just a plot.
    """
    import pandas as pd
    counts = pd.Series(labels).value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()

    ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    verdict = "imbalanced" if ratio > 3 else "roughly balanced"
    print(f"[plot_class_balance] largest/smallest class ratio = {ratio:.1f}x -> looks {verdict}")

    return fig


def plot_batch_summary(
    raw_images: Union[np.ndarray, List[np.ndarray]],
    processed_images: Union[np.ndarray, List[np.ndarray]],
    n_examples: int = 4,
    figsize=(14, 6),
):
    n_examples = min(n_examples, len(raw_images))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_examples)

    # Top: full-width histogram of the processed batch's pixel values
    ax_hist = fig.add_subplot(gs[0, :])
    flat = np.concatenate([np.asarray(im).ravel() for im in processed_images])
    ax_hist.hist(flat, bins=100, color="darkorange")
    ax_hist.set_title(
        f"Processed batch pixel value distribution "
        f"(min={flat.min():.3f}, max={flat.max():.3f}, mean={flat.mean():.3f})"
    )
    ax_hist.set_xlabel("Pixel value")
    ax_hist.set_ylabel("Frequency")

    # Middle row: raw examples. Bottom row: their processed counterparts.
    for i in range(n_examples):
        ax_raw = fig.add_subplot(gs[1, i])
        raw_disp = np.squeeze(_for_display(raw_images[i]))
        ax_raw.imshow(raw_disp, cmap="gray" if raw_disp.ndim == 2 else None)
        ax_raw.set_title(f"raw[{i}]", fontsize=9)
        ax_raw.axis("off")

        ax_proc = fig.add_subplot(gs[2, i])
        proc_disp = np.squeeze(_for_display(processed_images[i]))
        ax_proc.imshow(proc_disp, cmap="gray" if proc_disp.ndim == 2 else None)
        ax_proc.set_title(f"processed[{i}]", fontsize=9)
        ax_proc.axis("off")

    plt.tight_layout()
    return fig
