from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Single-image pipeline
# ---------------------------------------------------------------------------

@dataclass
class ProcessedImage:
    """Container returned by process_image: the array plus a record of what
    was done to it, so downstream code doesn't have to re-derive it."""
    array: np.ndarray
    shape: Tuple[int, ...]
    dtype: str
    value_range: Tuple[float, float]
    color_space: str
    normalization: Optional[str]
    source_path: Optional[str] = None


def process_image(
    image: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = (128, 128),
    color_space: str = "rgb",          # "rgb" or "grayscale"
    normalization: Optional[str] = "minmax",  # "minmax", "zscore", or None
) -> ProcessedImage:
    source_path = None

    # --- Load ---
    if isinstance(image, (str, Path)):
        source_path = str(image)
        with Image.open(image) as im:
            img_array = np.array(im.convert("RGB"))
    else:
        img_array = np.asarray(image)
        if img_array.ndim == 2:  # already grayscale input
            img_array = np.stack([img_array] * 3, axis=-1)

    # --- Resize ---
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    pil_img = pil_img.resize((target_size[1], target_size[0]))  # PIL wants (W, H)
    img_array = np.array(pil_img)

    # --- Color space conversion ---
    if color_space == "grayscale":
        gray = (0.299 * img_array[:, :, 0] +
                0.587 * img_array[:, :, 1] +
                0.114 * img_array[:, :, 2])
        img_array = gray.astype(np.uint8)[:, :, np.newaxis]  # keep a channel dim
    elif color_space == "rgb":
        pass
    else:
        raise ValueError(f"Unknown color_space: {color_space!r} (use 'rgb' or 'grayscale')")

    # --- Normalization ---
    img_array = img_array.astype(np.float32)
    if normalization == "minmax":
        img_array = img_array / 255.0
        norm_label = "minmax [0,1]"
    elif normalization == "zscore":
        mean, std = img_array.mean(), img_array.std()
        std = std if std > 1e-8 else 1.0  # guard against a flat/blank image
        img_array = (img_array - mean) / std
        norm_label = "zscore (mean=0, std=1)"
    elif normalization is None:
        norm_label = None
    else:
        raise ValueError(f"Unknown normalization: {normalization!r} (use 'minmax', 'zscore', or None)")

    return ProcessedImage(
        array=img_array,
        shape=img_array.shape,
        dtype=str(img_array.dtype),
        value_range=(float(img_array.min()), float(img_array.max())),
        color_space=color_space,
        normalization=norm_label,
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    images: np.ndarray                    # shape (N, H, W, C)
    labels: Optional[List] = None         # aligned with `images`, if labels were provided
    paths: List[str] = field(default_factory=list)   # paths that succeeded, aligned with `images`
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (path, error message)


def process_batch(
    items: Union[List[Union[str, Path]], "pandas.DataFrame"],
    image_dir: Optional[Union[str, Path]] = None,
    path_col: str = "isic_id",
    label_col: Optional[str] = None,
    path_suffix: str = ".jpg",
    target_size: Tuple[int, int] = (128, 128),
    color_space: str = "rgb",
    normalization: Optional[str] = "minmax",
) -> BatchResult:
    work_items: List[Tuple[str, Optional[object]]] = []

    is_dataframe = hasattr(items, "iterrows")
    if is_dataframe:
        if image_dir is None:
            raise ValueError("image_dir is required when `items` is a DataFrame")
        image_dir = Path(image_dir)
        for _, row in items.iterrows():
            fname = f"{row[path_col]}{path_suffix}"
            full_path = image_dir / fname
            label = row[label_col] if label_col is not None else None
            work_items.append((str(full_path), label))
    else:
        for p in items:
            work_items.append((str(p), None))

    processed_arrays = []
    processed_labels = []
    processed_paths = []
    skipped: List[Tuple[str, str]] = []

    for path, label in work_items:
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"file not found: {path}")
            result = process_image(
                path,
                target_size=target_size,
                color_space=color_space,
                normalization=normalization,
            )
            processed_arrays.append(result.array)
            processed_labels.append(label)
            processed_paths.append(path)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: one bad file shouldn't kill the batch
            skipped.append((path, f"{type(exc).__name__}: {exc}"))

    if processed_arrays:
        stacked = np.stack(processed_arrays, axis=0)
    else:
        stacked = np.empty((0, target_size[0], target_size[1],
                             1 if color_space == "grayscale" else 3), dtype=np.float32)

    return BatchResult(
        images=stacked,
        labels=processed_labels if label_col is not None or any(l is not None for l in processed_labels) else None,
        paths=processed_paths,
        skipped=skipped,
    )
