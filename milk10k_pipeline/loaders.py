"""
DataLoaders for the project (Milestone 1, B8).

    make_datasets(label_col)   -> train / val / test MILK10kImageDataset
    make_dataloaders(...)      -> train (augment + shuffle OR weighted sampler),
                                  val / test (eval_transform, no shuffle)

Imbalance handling, both computed from the TRAIN split only:
    (i)  class weights for the loss       -> outputs/class_weights.json
    (ii) WeightedRandomSampler for train  -> use_sampler=True
Use ONE of the two when training, not both (that would correct twice).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import config, labels
from .datasets import MILK10kImageDataset
from .transforms import eval_transform, train_transform


def seed_everything(seed: int = config.SEED) -> torch.Generator:
    """Seed python, numpy and torch; return a torch.Generator for DataLoader/sampler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _worker_init(worker_id: int) -> None:
    """Give every worker a different but reproducible numpy/python seed."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def split_csv(name: str) -> Path:
    """Path of outputs/splits/<name>.csv."""
    return config.SPLIT_DIR / f"{name}.csv"


def make_datasets(label_col: str = "y_primary", allow_missing: bool = False) -> Dict[str, MILK10kImageDataset]:
    """train (train_transform) + val/test (eval_transform) datasets from the split CSVs."""
    return {
        "train": MILK10kImageDataset(split_csv("train"), label_col=label_col,
                                     transform=train_transform(), allow_missing=allow_missing),
        "val": MILK10kImageDataset(split_csv("val"), label_col=label_col,
                                   transform=eval_transform(), allow_missing=allow_missing),
        "test": MILK10kImageDataset(split_csv("test"), label_col=label_col,
                                    transform=eval_transform(), allow_missing=allow_missing),
    }


def make_dataloaders(label_col: str = "y_primary", batch_size: int = config.BATCH_SIZE,
                     num_workers: int = config.NUM_WORKERS, use_sampler: bool = False,
                     seed: int = config.SEED, allow_missing: bool = False) -> Dict[str, DataLoader]:
    """Build the three DataLoaders.

    train : train_transform; shuffled, or drawn with a WeightedRandomSampler
            if ``use_sampler`` (shuffle and sampler are mutually exclusive)
    val/test : eval_transform, no shuffle (fixed order -> comparable runs)
    """
    g = seed_everything(seed)
    ds = make_datasets(label_col, allow_missing)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  worker_init_fn=_worker_init, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)

    if use_sampler:
        w = labels.sample_weights_for_sampler(ds["train"].labels)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True, generator=g)
        train_dl = DataLoader(ds["train"], sampler=sampler, **common)
    else:
        train_dl = DataLoader(ds["train"], shuffle=True, generator=g, **common)

    return {
        "train": train_dl,
        "val": DataLoader(ds["val"], shuffle=False, **common),
        "test": DataLoader(ds["test"], shuffle=False, **common),
    }


def compute_and_save_class_weights(train_csv: Optional[Path] = None,
                                   path: Optional[Path] = None, fine_cap: float = 20.0) -> Dict:
    """Class weights from the TRAIN split for both targets; saved to class_weights.json.

    The fine-target weights are capped at ``fine_cap`` so that OTHER_MALIGNANT
    (a handful of training images) does not dominate the loss.
    """
    train = pd.read_csv(train_csv or split_csv("train"))
    lm = labels.load_label_map() if config.LABEL_MAP_JSON.exists() else labels.build_label_map()
    out = {}
    for key, col in (("primary", "y_primary"), ("fine", "y_fine")):
        classes = lm[key]["classes"]
        cap = fine_cap if key == "fine" else None
        w = labels.balanced_class_weights(train[col], len(classes), cap=cap)
        counts = np.bincount(train[col], minlength=len(classes))
        out[key] = {
            "classes": classes,
            "train_image_counts": dict(zip(classes, counts.tolist())),
            "weights": dict(zip(classes, np.round(w, 4).tolist())),
            "formula": "N / (K * n_c)" + (f", capped at {cap}" if cap else ""),
        }
    path = Path(path or config.CLASS_WEIGHTS_JSON)
    path.write_text(json.dumps(out, indent=2))
    return out


def weights_tensor(key: str = "primary", path: Optional[Path] = None) -> torch.Tensor:
    """Load class weights as a tensor in class-index order, ready for nn.CrossEntropyLoss(weight=...)."""
    d = json.loads(Path(path or config.CLASS_WEIGHTS_JSON).read_text())[key]
    return torch.tensor([d["weights"][c] for c in d["classes"]], dtype=torch.float32)
