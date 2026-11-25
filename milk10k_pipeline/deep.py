"""
Deep-learning toolkit for Milestones 4 and 5 (Sessions 7, 9 and 10).

Everything that the M4 / M5 scripts and ``m5_final/predict.py`` share lives here,
so training, evaluation and inference cannot drift apart:

    load_frames()        my Milestone 1 split, one dermoscopic image per lesion,
                         with the 3-class label, the 11-class label (from
                         training_gt.csv), sex, age, site and skin tone
    load_image_224()     THE image transform used for training, validation, test
                         and inference (decode -> RGB -> 224 x 224). Used to build
                         the Session 7 cache and by predict(), so there is no
                         train / serve skew (Session 10).
    load_cache()         the decoded uint8 image tensors (same file as Sessions 7 and 9)
    prep() / augment()   ImageNet normalisation and random flips (Session 7)
    build_model()        ResNet-18 (fine-tuned) or frozen backbone + linear head
    fit()                the Session 7 training loop (class-weighted CE, Adam,
                         early stopping on validation macro-F1)
    embed() / train_head()  the Session 9 frozen-feature comparison
    leaderboard helpers  the official MILK10k metric (threshold 0.5, macro-F1 over
                         11 classes) and the post-processing that puts each class
                         threshold at 0.5
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score

from . import config
from .data import load_gt

PRIMARY = list(config.PRIMARY_CLASSES)          # Benign, Indeterminate, Malignant (label_map.json order)
FINE11 = list(config.FINE_CLASSES)              # training_gt.csv column order = submission column order

# Big files (decoded images, features) stay outside git and OneDrive, as in Sessions 7 and 9.
CACHE_DIR = Path(os.environ.get("MILK10K_CACHE_DIR", Path.home() / ".cache" / "milk10k"))
CHECKPOINT_DIR = config.REPO_ROOT / "checkpoints"
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_frames() -> Dict[str, pd.DataFrame]:
    """train / val / test of my Milestone 1 split, dermoscopic image only, one row per lesion.

    Columns added: y3 (0..2, Benign/Indeterminate/Malignant), y11 (0..10, training_gt.csv
    column order), skin_tone_class and site (from supplements/training_input.csv, per image;
    its body site is more complete than metadata.csv's anatom_site_general).
    """
    gt = load_gt()
    gt["y11"] = gt[FINE11].values.argmax(1)
    tone = _skin_tone_table()
    frames = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(config.SPLIT_DIR / f"{s}.csv")
        d = d[d["image_type"] == "dermoscopic"].reset_index(drop=True)
        assert d["lesion_id"].is_unique
        d["y3"] = d["diagnosis_1"].map({c: i for i, c in enumerate(PRIMARY)})
        d = d.merge(gt[["lesion_id", "y11"]], on="lesion_id", how="left", validate="one_to_one")
        assert (np.array(FINE11)[d["y11"]] == d["dx"].values).all(), "dx in split CSV != training_gt.csv"
        if tone is not None:
            d = d.merge(tone, on="isic_id", how="left", validate="one_to_one")
        frames[s] = d
    tr, va, te = (set(frames[s].lesion_id) for s in ("train", "val", "test"))
    assert not (tr & va or tr & te or va & te), "lesion leakage between splits"
    return frames


def _skin_tone_table() -> Optional[pd.DataFrame]:
    for p in (config.DATA_DIR / "supplements" / "training_input.csv", config.DATA_DIR / "training_input.csv"):
        if p.exists():
            return pd.read_csv(p, usecols=["isic_id", "skin_tone_class", "site"])
    return None


def load_image_224(path) -> torch.Tensor:
    """Decode one image exactly as for training: RGB, resized to 224 x 224 -> uint8 (3, 224, 224).

    This is the same call that built the Session 7 / 9 cache, so the model sees
    identical pixels at training and at inference time.
    """
    img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()


def load_cache(frames: Dict[str, pd.DataFrame]) -> Dict[str, torch.Tensor]:
    """The Session 7 cache (``~/.cache/milk10k/images_224_seed42.pt``); built if missing."""
    cache = CACHE_DIR / f"images_224_seed{config.SEED}.pt"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = torch.load(cache, weights_only=True)
    else:
        t0, data = time.perf_counter(), {}
        for s, d in frames.items():
            data[f"X_{s}"] = torch.stack([load_image_224(config.image_path(i)) for i in d["isic_id"]])
            data[f"y_{s}"] = torch.tensor(d["y3"].to_numpy())
        torch.save(data, cache)
        print(f"cache built in {time.perf_counter() - t0:.0f} s -> {cache}")
    for s, d in frames.items():   # the cache must line up with the split, row by row
        assert len(data[f"X_{s}"]) == len(d), f"cache {s} has {len(data[f'X_{s}'])} rows, split has {len(d)}"
        assert torch.equal(data[f"y_{s}"], torch.tensor(d["y3"].to_numpy())), f"cache {s} labels != split"
    return data


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def prep(xb: torch.Tensor) -> torch.Tensor:
    """uint8 (n, 3, 224, 224) -> float, ImageNet-normalised, on the device (Session 7)."""
    return (xb.to(device).float() / 255 - _MEAN.to(device)) / _STD.to(device)


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Random horizontal and vertical flips, one coin per image (Session 7)."""
    n = xb.size(0)
    flip_h = torch.rand(n, 1, 1, 1, device=xb.device) < 0.5
    xb = torch.where(flip_h, xb.flip(3), xb)
    flip_v = torch.rand(n, 1, 1, 1, device=xb.device) < 0.5
    return torch.where(flip_v, xb.flip(2), xb)


class FrozenBackboneClassifier(nn.Module):
    """ImageNet backbone (frozen, eval mode) + one trainable nn.Linear (Session 9)."""

    def __init__(self, backbone_name: str, n_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = frozen_backbone(backbone_name, pretrained=pretrained)
        self.head = nn.Linear(FEATURE_SIZE[backbone_name], n_classes)

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.head(f)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()           # the backbone never leaves eval mode
        return self


FEATURE_SIZE = {"resnet18": 512, "vit_b_16": 768}


def frozen_backbone(name: str, pretrained: bool = True) -> nn.Module:
    from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights
    if name == "resnet18":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()                  # 512 numbers after global average pooling
    elif name == "vit_b_16":
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        net.heads = nn.Identity()               # the class token: 768 numbers
    else:
        raise ValueError(name)
    for p in net.parameters():
        p.requires_grad = False
    return net.to(device).eval()


def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """arch: 'resnet18_ft' (all layers trainable) or 'resnet18_frozen' / 'vit_b_16_frozen'."""
    from torchvision.models import resnet18, ResNet18_Weights
    if arch == "resnet18_ft":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(512, n_classes)
        return m.to(device)
    if arch.endswith("_frozen"):
        return FrozenBackboneClassifier(arch[: -len("_frozen")], n_classes, pretrained).to(device)
    raise ValueError(arch)


def count(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


def save_checkpoint(model: nn.Module, arch: str, classes: Sequence[str], path: Path, extra: Optional[Dict] = None):
    """Frozen models: only the head is saved (the backbone is the public ImageNet one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.head.state_dict() if arch.endswith("_frozen") else model.state_dict()
    torch.save({"arch": arch, "classes": list(classes), "state_dict": state,
                "input": "RGB, resize to 224x224, /255, ImageNet mean/std", **(extra or {})}, path)


def load_checkpoint(path: Path) -> Tuple[nn.Module, Dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    arch, classes = ck["arch"], ck["classes"]
    if arch.endswith("_frozen"):
        model = build_model(arch, len(classes), pretrained=True)
        model.head.load_state_dict(ck["state_dict"])
    else:
        model = build_model(arch, len(classes), pretrained=False)
        model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


# --------------------------------------------------------------------------- #
# Training and prediction (Session 7 loop)
# --------------------------------------------------------------------------- #
def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """n / (K * n_c): mistakes on rare classes cost more (Sessions 3, 6, 7)."""
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    return len(y) / (n_classes * counts)


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(prep(X[i:i + batch_size])), 1).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def macro_f1(y, pred, n_classes: int) -> float:
    return float(f1_score(np.asarray(y), np.asarray(pred), average="macro",
                          labels=list(range(n_classes)), zero_division=0))


def fit(model, params, X_train, y_train, X_val, y_val, n_classes: int, epochs: int, lr: float,
        batch_size: int = 64, aug: bool = True, log=print):
    """Session 7 ``fit``: weighted CE + Adam, keeps the epoch with the best validation macro-F1."""
    model.to(device)
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes).to(device))
    history, best = [], {"f1": -1.0}
    for epoch in range(epochs):
        model.train()
        t0, total = time.perf_counter(), 0.0
        order = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = order[i:i + batch_size]
            xb, yb = prep(X_train[idx]), y_train[idx].to(device)
            if aug:
                xb = augment(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        train_s = time.perf_counter() - t0
        f1 = macro_f1(y_val, predict_proba(model, X_val).argmax(1), n_classes)
        history.append({"epoch": epoch, "train_loss": total / len(X_train), "val_f1": f1,
                        "train_s": train_s, "epoch_s": time.perf_counter() - t0})
        log(f"  epoch {epoch}  loss {total / len(X_train):.3f}  val macro-F1 {f1:.3f}  "
            f"({time.perf_counter() - t0:.0f} s)")
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return pd.DataFrame(history), best


# --------------------------------------------------------------------------- #
# Frozen features + linear head (Session 9)
# --------------------------------------------------------------------------- #
def embed(name: str, X: torch.Tensor, split: str, timings: Optional[Dict] = None) -> torch.Tensor:
    """Backbone features, cached with the same file name as my Session 9 notebook."""
    f = CACHE_DIR / f"s9_features_{name}_{split}_{len(X)}_seed{config.SEED}.pt"
    if not f.exists():
        net, t0 = frozen_backbone(name), time.perf_counter()
        with torch.no_grad():
            out = torch.cat([net(prep(X[i:i + 64])).cpu() for i in range(0, len(X), 64)])
        torch.save(out, f)
        if timings is not None:
            timings[f"{name}_{split}"] = round(time.perf_counter() - t0, 1)
        print(f"  {name} {split}: {tuple(out.shape)} in {time.perf_counter() - t0:.0f} s", flush=True)
    return torch.load(f, weights_only=True)


def train_head(F_train: torch.Tensor, y: torch.Tensor, n_classes: int, epochs: int = 25) -> nn.Linear:
    """Session 9 Exercise 4: one nn.Linear, class weights, Adam lr 1e-3, 25 epochs, batch 64."""
    torch.manual_seed(0)
    head = nn.Linear(F_train.shape[1], n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, n_classes))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(y))
        for i in range(0, len(y), 64):
            idx = order[i:i + 64]
            optimizer.zero_grad()
            loss_fn(head(F_train[idx]), y[idx]).backward()
            optimizer.step()
    return head


@torch.no_grad()
def head_proba(head: nn.Linear, F_: torch.Tensor) -> np.ndarray:
    return torch.softmax(head(F_), 1).numpy()


# --------------------------------------------------------------------------- #
# Uncertainty (Session 7 mini challenge)
# --------------------------------------------------------------------------- #
def bootstrap_ci(y, pred, n_classes: int, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = [macro_f1(y[i], pred[i], n_classes) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]


def paired_bootstrap(y, pred_a, pred_b, n_classes: int, n: int = 2000, seed: int = 0):
    """95% CI of macro-F1(b) - macro-F1(a) on the same resampled lesions, and P(b > a)."""
    rng = np.random.default_rng(seed)
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    diffs = np.empty(n)
    for k in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs[k] = macro_f1(y[i], pred_b[i], n_classes) - macro_f1(y[i], pred_a[i], n_classes)
    return [round(float(c), 4) for c in np.percentile(diffs, [2.5, 97.5])], round(float((diffs > 0).mean()), 3)


# --------------------------------------------------------------------------- #
# Official leaderboard metric and post-processing (Milestone 5)
# --------------------------------------------------------------------------- #
def leaderboard_f1(y_true, P: np.ndarray, threshold: float = 0.5, per_class: bool = False):
    """MILK10k metric: each probability >= 0.5 -> 'yes'; F1 per class; plain mean over the 11 classes."""
    y_true = np.asarray(y_true)
    Y = np.eye(P.shape[1], dtype=bool)[y_true]
    B = P >= threshold
    f1s = np.array([f1_score(Y[:, c], B[:, c], zero_division=0) for c in range(P.shape[1])])
    return (float(f1s.mean()), f1s) if per_class else float(f1s.mean())


def rescale_argmax(P: np.ndarray) -> np.ndarray:
    """Top class -> 0.5 + p/2 (>= 0.5), every other class -> p/2 (< 0.5).

    Monotone in each probability, values stay in [0, 1], and at the 0.5 threshold
    the leaderboard sees exactly the argmax decision (one class per lesion).
    """
    P = np.asarray(P, dtype=float)
    out = P / 2
    top = P.argmax(1)
    out[np.arange(len(P)), top] = 0.5 + P[np.arange(len(P)), top] / 2
    return out


def rescale_thresholds(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-class threshold t_c moved to 0.5 (piecewise linear, monotone, stays in [0, 1])."""
    P, t = np.asarray(P, dtype=float), np.asarray(t, dtype=float)[None, :]
    return np.where(P < t, 0.5 * P / t, 0.5 + 0.5 * (P - t) / (1 - t))


def tune_thresholds(y, P: np.ndarray, grid: Optional[np.ndarray] = None, min_pos: int = 3) -> np.ndarray:
    """One threshold per class maximising that class's F1. Classes with < min_pos positives keep 0.5."""
    grid = np.linspace(0.02, 0.98, 49) if grid is None else grid
    y = np.asarray(y)
    t = np.full(P.shape[1], 0.5)
    for c in range(P.shape[1]):
        pos = y == c
        if pos.sum() < min_pos:
            continue
        scores = [f1_score(pos, P[:, c] >= g, zero_division=0) for g in grid]
        t[c] = float(grid[int(np.argmax(scores))])
    return t


def crossfit_threshold_score(y, P: np.ndarray, repeats: int = 20, seed: int = 0) -> float:
    """Honest validation estimate of threshold tuning: tune on one half of val, score on the other."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = []
    for _ in range(repeats):
        idx = rng.permutation(len(y))
        a, b = idx[: len(y) // 2], idx[len(y) // 2:]
        for fit_i, ev_i in ((a, b), (b, a)):
            t = tune_thresholds(y[fit_i], P[fit_i])
            scores.append(leaderboard_f1(y[ev_i], rescale_thresholds(P[ev_i], t)))
    return float(np.mean(scores))


def postprocess(P: np.ndarray, strategy: str, thresholds: Optional[Sequence[float]] = None) -> np.ndarray:
    if strategy == "raw":
        return np.asarray(P, dtype=float)
    if strategy == "argmax":
        return rescale_argmax(P)
    if strategy == "thresholds":
        return rescale_thresholds(P, np.asarray(thresholds))
    raise ValueError(strategy)
