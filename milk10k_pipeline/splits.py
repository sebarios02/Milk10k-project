"""
Leak-free train / val / test splits at LESION level.

Rule: we split lesions, never images. Both images of a lesion (dermoscopic +
clinical) always end up in the same split, so a model can never see one view
of a test lesion during training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _one_fold(df: pd.DataFrame, frac: float, label_col: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out ONE fold of a StratifiedGroupKFold whose size is closest to ``frac``.

    Returns (rest_idx, held_idx) as positional indices into ``df``.
    """
    n_splits = max(2, int(round(1.0 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(df, df[label_col], groups=df["lesion_id"]))
    return rest_idx, held_idx


def make_strat_key(lesions: pd.DataFrame) -> pd.Series:
    """Stratification key: 11-class dx refined by diagnosis_1.

    Only AKIEC is affected (it is split into AKIEC|Indeterminate and
    AKIEC|Malignant), so the small Indeterminate class is balanced too.
    """
    return lesions["dx"] + "|" + lesions["diagnosis_1"]


def split_lesions(lesions: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                  seed: int = 42, label_col: str = "dx") -> Tuple[List[str], List[str], List[str]]:
    """Split a lesion table into train / val / test lists of lesion_id.

    Uses StratifiedGroupKFold twice (groups = lesion_id, stratified on
    ``label_col``): first one fold is held out as test, then one fold of the
    remainder is held out as val. Fold counts are chosen so that each held-out
    fold is as close as possible to the requested fraction of ALL lesions.

    Parameters
    ----------
    lesions : one row per lesion, must contain ``lesion_id`` and ``label_col``
    val_size, test_size : fractions of the full lesion table
    seed : controls the shuffling -> same seed, same split
    label_col : column used for stratification (11-class ``dx`` by default)

    Returns
    -------
    (train_ids, val_ids, test_ids) : three sorted lists of lesion_id
    """
    if lesions["lesion_id"].duplicated().any():
        raise ValueError("split_lesions expects ONE row per lesion")
    df = lesions.reset_index(drop=True)

    rest_idx, test_idx = _one_fold(df, test_size, label_col, seed)
    rest = df.iloc[rest_idx].reset_index(drop=True)
    # val must be val_size of the WHOLE table -> fraction of the remainder
    val_frac_of_rest = val_size / (len(rest) / len(df))
    train_idx, val_idx = _one_fold(rest, val_frac_of_rest, label_col, seed + 1)

    train_ids = sorted(rest.iloc[train_idx]["lesion_id"])
    val_ids = sorted(rest.iloc[val_idx]["lesion_id"])
    test_ids = sorted(df.iloc[test_idx]["lesion_id"])
    return train_ids, val_ids, test_ids


def check_split(train_ids, val_ids, test_ids, n_total: int,
                val_size: float, test_size: float, tol_pp: float = 1.0) -> Dict[str, float]:
    """Assert no overlap and sizes within +-tol_pp percentage points. Returns the sizes (%)."""
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    assert not (tr & va), "train/val overlap"
    assert not (tr & te), "train/test overlap"
    assert not (va & te), "val/test overlap"
    assert len(tr) + len(va) + len(te) == n_total, "lesions lost or duplicated"
    sizes = {"train": 100 * len(tr) / n_total, "val": 100 * len(va) / n_total,
             "test": 100 * len(te) / n_total}
    assert abs(sizes["val"] - 100 * val_size) <= tol_pp, f"val size {sizes['val']:.2f}%"
    assert abs(sizes["test"] - 100 * test_size) <= tol_pp, f"test size {sizes['test']:.2f}%"
    return sizes


def assign_split_column(images: pd.DataFrame, train_ids, val_ids, test_ids) -> pd.DataFrame:
    """Add a ``split`` column to an image (or lesion) table, based on lesion_id."""
    lookup = {**{i: "train" for i in train_ids}, **{i: "val" for i in val_ids},
              **{i: "test" for i in test_ids}}
    out = images.copy()
    out["split"] = out["lesion_id"].map(lookup)
    if out["split"].isna().any():
        raise ValueError(f"{out['split'].isna().sum()} rows have a lesion in no split")
    return out


def proportions_by_split(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Class proportions (%) per split plus the global proportions."""
    tab = pd.crosstab(df[label_col], df["split"], normalize="columns") * 100
    tab["global"] = df[label_col].value_counts(normalize=True) * 100
    cols = [c for c in ["train", "val", "test", "global"] if c in tab.columns]
    return tab[cols].round(2)


def leaked_lesions(train_lesions, other_lesions) -> int:
    """Number of lesion_ids present in both collections."""
    return len(set(train_lesions) & set(other_lesions))
