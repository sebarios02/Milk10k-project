"""
PyTorch datasets (Milestone 1, B8 + Session 3 A3.6).

- ``MILK10kImageDataset``  one item = one IMAGE -> (tensor, int label, isic_id)
                            This is the upgrade of the Session 2 MILK10kDataLoader.
- ``LesionDataset``        one item = one LESION -> dict with one tensor per view
- ``aggregate_predictions`` image-level probabilities -> one prediction per lesion

Both datasets FAIL LOUDLY when an image file is missing (checked once at
construction, so the error appears before training starts), unless
``allow_missing=True`` is passed explicitly, in which case the rows are
dropped and the number dropped is printed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config

VIEW_TO_COLUMN = {"derm": "derm_id", "clinical": "clinical_id"}


def _check_files(ids: pd.Series, img_dir: Path, allow_missing: bool, who: str) -> pd.Series:
    """Return a boolean 'exists' mask; raise FileNotFoundError if anything is missing and not allowed."""
    exists = ids.map(lambda i: (img_dir / f"{i}.jpg").exists())
    n_missing = int((~exists).sum())
    if n_missing:
        examples = ", ".join(ids[~exists].head(5))
        msg = (f"[{who}] {n_missing} of {len(ids)} image files are missing in {img_dir} "
               f"(e.g. {examples}).")
        if not allow_missing:
            raise FileNotFoundError(msg + " Pass allow_missing=True to drop them explicitly.")
        print(msg + " -> dropped because allow_missing=True")
    return exists


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class MILK10kImageDataset(Dataset):
    """One item per image: ``(image_tensor, label, isic_id)``.

    Parameters
    ----------
    table : DataFrame or path to a split CSV (needs ``isic_id`` and ``label_col``)
    img_dir : folder with the .jpg files (default: config.IMAGE_DIR)
    label_col : integer label column, e.g. ``y_primary`` or ``y_fine``
    transform : torchvision transform applied to the PIL image
    allow_missing : drop rows whose file is missing instead of raising
    """

    def __init__(self, table: Union[pd.DataFrame, str, Path], img_dir: Optional[Path] = None,
                 label_col: str = "y_primary", transform: Optional[Callable] = None,
                 allow_missing: bool = False):
        df = pd.read_csv(table) if isinstance(table, (str, Path)) else table.copy()
        self.img_dir = Path(img_dir or config.IMAGE_DIR)
        exists = _check_files(df["isic_id"], self.img_dir, allow_missing, "MILK10kImageDataset")
        self.df = df[exists].reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform
        self.labels = self.df[label_col].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        isic_id = self.df.at[idx, "isic_id"]
        img = _load_rgb(self.img_dir / f"{isic_id}.jpg")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx]), isic_id


class LesionDataset(Dataset):
    """One item per LESION: ``{"derm": tensor, "clinical": tensor, "label": int, "lesion_id": str}``.

    Parameters
    ----------
    lesion_table : one row per lesion with ``lesion_id``, ``derm_id``, ``clinical_id``
                   and ``label_col``
    img_dir : image folder
    label_col : column holding the label. If it is not integer, classes are
                mapped to integers in sorted order (mapping kept in ``self.classes``).
    transform : applied INDEPENDENTLY to each view (each view gets its own
                random augmentation draw)
    views : which views to return, subset of ("derm", "clinical")
    """

    def __init__(self, lesion_table: pd.DataFrame, img_dir: Optional[Path] = None,
                 label_col: str = "diagnosis_1", transform: Optional[Callable] = None,
                 views: Sequence[str] = ("derm", "clinical"), allow_missing: bool = False):
        unknown = set(views) - set(VIEW_TO_COLUMN)
        if unknown:
            raise ValueError(f"unknown views {unknown}; use {tuple(VIEW_TO_COLUMN)}")
        self.img_dir = Path(img_dir or config.IMAGE_DIR)
        self.views = tuple(views)
        df = lesion_table.reset_index(drop=True)

        ok = pd.Series(True, index=df.index)
        for v in self.views:
            ok &= _check_files(df[VIEW_TO_COLUMN[v]], self.img_dir, allow_missing, f"LesionDataset[{v}]")
        self.df = df[ok].reset_index(drop=True)

        if pd.api.types.is_integer_dtype(self.df[label_col]):
            self.classes = None
            self.labels = self.df[label_col].to_numpy()
        else:
            self.classes = sorted(self.df[label_col].unique())
            to_idx = {c: i for i, c in enumerate(self.classes)}
            self.labels = self.df[label_col].map(to_idx).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        item = {}
        for v in self.views:
            img = _load_rgb(self.img_dir / f"{row[VIEW_TO_COLUMN[v]]}.jpg")
            item[v] = self.transform(img) if self.transform is not None else img
        item["label"] = int(self.labels[idx])
        item["lesion_id"] = row["lesion_id"]
        return item


def aggregate_predictions(image_probs: np.ndarray, image_table: pd.DataFrame,
                          class_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Turn image-level class probabilities into ONE prediction per lesion.

    The probabilities of all images of a lesion (normally its two views) are
    averaged; the predicted class is the argmax of the average.

    Parameters
    ----------
    image_probs : array (n_images, n_classes), row i belongs to image_table row i
    image_table : DataFrame with a ``lesion_id`` column, same length/order
    class_names : optional names for the probability columns

    Returns
    -------
    DataFrame indexed by lesion_id with columns p_<class>..., ``n_images`` and ``pred``
    (integer index of the predicted class).
    """
    probs = np.asarray(image_probs, dtype=float)
    if probs.ndim != 2 or len(probs) != len(image_table):
        raise ValueError("image_probs must be (n_images, n_classes) aligned with image_table")
    names = list(class_names) if class_names is not None else [str(i) for i in range(probs.shape[1])]
    cols = [f"p_{c}" for c in names]
    df = pd.DataFrame(probs, columns=cols)
    df["lesion_id"] = image_table["lesion_id"].to_numpy()
    out = df.groupby("lesion_id")[cols].mean()
    out["n_images"] = df.groupby("lesion_id").size()
    out["pred"] = out[cols].to_numpy().argmax(axis=1)
    return out
