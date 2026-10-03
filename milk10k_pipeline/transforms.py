"""
Train / eval transforms (Milestone 1, B7). Importable so every milestone uses
exactly the same preprocessing.

    train_transform : random, label-preserving augmentation + normalisation
    eval_transform  : deterministic resize + centre crop + normalisation

Normalisation statistics are computed on the TRAIN split only
(scripts/06_preprocessing.py -> outputs/norm_stats.json). If that file does not
exist yet, ImageNet statistics are used as a fallback (with a warning).
"""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

from PIL import Image
from torchvision import transforms as T

from . import config

IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class RandomRot90:
    """Rotate by a random multiple of 90 degrees.

    Lossless (no interpolation, no black corners) and label-safe: a skin lesion
    has no canonical orientation in dermoscopy or close-up photos.
    Defined as a class (not a lambda) so it can be pickled by DataLoader
    workers on Windows.
    """

    def __call__(self, img):
        k = random.randint(0, 3)
        ops = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]
        return img if k == 0 else img.transpose(ops[k])

    def __repr__(self) -> str:
        return "RandomRot90()"


# One row per augmentation: (name, parameters, why it does not change the label).
AUGMENTATION_TABLE = [
    ("Resize + RandomResizedCrop", f"shorter side {config.RESIZE_SIZE}, crop {config.IMG_SIZE}, scale=(0.75, 1.0), ratio=(0.9, 1.1)",
     "Simulates different zoom/framing of the same lesion; >=75% of the area is kept so the lesion stays in view."),
    ("RandomHorizontalFlip", "p=0.5", "Lesions have no left/right orientation; a mirrored nevus is still a nevus."),
    ("RandomVerticalFlip", "p=0.5", "Same argument: dermoscopy has no 'up'."),
    ("RandomRot90", "k in {0,1,2,3}", "Orientation-free; lossless (no interpolation, no padding artefacts)."),
    ("ColorJitter", "brightness=0.1, contrast=0.1, saturation=0.05, hue=0",
     "Mimics lighting/camera differences. Hue is NOT jittered: the A3.5 audit shows that hue=0.02 already shifts lesion hue by "
     "3.2 deg (gap between classes: 3.45 deg) and hue>=0.05 exceeds the gap, and colour (blue-white veil, red vessels) is diagnostic."),
    ("Normalize", "train-split mean/std", "Not an augmentation: rescales intensities, identical for every image."),
]


def load_norm_stats(path: Optional[Path] = None) -> Dict[str, Sequence[float]]:
    """mean/std from outputs/norm_stats.json, or ImageNet stats if the file is missing."""
    path = Path(path or config.NORM_STATS_JSON)
    if path.exists():
        stats = json.loads(path.read_text())
        return {"mean": stats["mean"], "std": stats["std"]}
    warnings.warn(f"{path} not found -> using ImageNet normalisation stats")
    return IMAGENET_STATS


def train_transform(img_size: int = config.IMG_SIZE, resize_size: int = config.RESIZE_SIZE,
                    stats: Optional[Dict] = None) -> T.Compose:
    """Random augmentation pipeline used ONLY for the training split."""
    stats = stats or load_norm_stats()
    return T.Compose([
        T.Resize(resize_size),
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        RandomRot90(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),
        T.ToTensor(),
        T.Normalize(stats["mean"], stats["std"]),
    ])


def eval_transform(img_size: int = config.IMG_SIZE, resize_size: int = config.RESIZE_SIZE,
                   stats: Optional[Dict] = None) -> T.Compose:
    """Deterministic pipeline for val / test / inference."""
    stats = stats or load_norm_stats()
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(stats["mean"], stats["std"]),
    ])


def to_tensor_only(img_size: int = config.IMG_SIZE, resize_size: int = config.RESIZE_SIZE) -> T.Compose:
    """Resize + crop + ToTensor without normalisation (used to compute the train stats)."""
    return T.Compose([T.Resize(resize_size), T.CenterCrop(img_size), T.ToTensor()])


def augment_only(img_size: int = config.IMG_SIZE, resize_size: int = config.RESIZE_SIZE) -> T.Compose:
    """train_transform without ToTensor/Normalize -> returns a PIL image (for figures)."""
    return T.Compose(train_transform(img_size, resize_size, stats=IMAGENET_STATS).transforms[:-2])
