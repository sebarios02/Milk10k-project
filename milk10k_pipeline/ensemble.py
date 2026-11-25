"""
Milestone 5 improvement: a late-fusion ensemble for the 11-class leaderboard model.

Members (each one outputs 11 probabilities per lesion; they are averaged with weights):
    resnet_derm   my fine-tuned ResNet-18 (checkpoints/model_11class.pt) on the dermoscopic image,
                  optionally with flip test-time augmentation (TTA: mean of 4 flipped copies)
    vit_derm      frozen ViT-B/16 + linear head on the dermoscopic image (Session 9 features)
    vit_clin      frozen ViT-B/16 + linear head on the CLINICAL close-up (the view not used so far)
    meta          logistic regression on the official model-input metadata: age, sex, body site,
                  skin tone and 4 diagnostic MONET concepts of each view (ulceration/crust,
                  vessels, erythema, pigmentation)

Excluded on purpose (Milestone 1 shortcut rules): image_manipulation, the MONET pen-ink, gel and
hair concepts (acquisition artefacts, not lesion properties), attribution and licence.

Everything is chosen on VALIDATION; the cross-fitted estimate tells whether searching for weights
beats plain equal weights without fooling ourselves.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from . import config, deep

MEMBERS = ["resnet_derm", "vit_derm", "vit_clin", "meta"]
MONET_KEEP = ["MONET_ulceration_crust", "MONET_vasculature_vessels", "MONET_erythema", "MONET_pigmented"]
SITES = ["head_neck_face", "trunk", "upper_extremity", "lower_extremity", "hand", "foot", "genital"]


# --------------------------------------------------------------------------- metadata features
def model_input_table() -> pd.DataFrame:
    """The official per-image model-input metadata (same columns as MILK10k_Test_Metadata.csv)."""
    for p in (config.DATA_DIR / "supplements" / "training_input.csv", config.DATA_DIR / "training_input.csv"):
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("training_input.csv not found in the data folder")


def meta_features(meta: pd.DataFrame, lesions: Sequence[str]) -> np.ndarray:
    """One row per lesion, in the order of `lesions`. Works for the training and the test metadata."""
    derm = meta[meta["image_type"] == "dermoscopic"].drop_duplicates("lesion_id").set_index("lesion_id")
    clin = meta[meta["image_type"] != "dermoscopic"].drop_duplicates("lesion_id").set_index("lesion_id")
    d, c = derm.reindex(lesions), clin.reindex(lesions)
    X = pd.DataFrame(index=pd.Index(lesions))
    X["age"] = d["age_approx"].fillna(60).values           # 0.4 % missing; 60 = rounded train median
    X["male"] = (d["sex"] == "male").astype(float).values
    X["skin_tone"] = d["skin_tone_class"].fillna(3).values
    for s in SITES:
        X[f"site_{s}"] = (d["site"] == s).astype(float).values
    for m in MONET_KEEP:
        X[f"derm_{m}"] = d[m].values
        X[f"clin_{m}"] = c[m].values
    return X.fillna(0).to_numpy(dtype=float)


def fit_meta_model(X: np.ndarray, y: np.ndarray, C: float = 0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X, y)


# --------------------------------------------------------------------------- images
def clinical_isic_ids(split: str, lesions: Sequence[str]) -> List[str]:
    d = pd.read_csv(config.SPLIT_DIR / f"{split}.csv")
    clin = d[d["image_type"] != "dermoscopic"].drop_duplicates("lesion_id").set_index("lesion_id")["isic_id"]
    return clin.reindex(lesions).tolist()


def load_clinical_cache(frames: Dict[str, pd.DataFrame]) -> Dict[str, torch.Tensor]:
    """Clinical close-ups, decoded with the SAME function as the dermoscopic cache (deep.load_image_224)."""
    import time
    f = deep.CACHE_DIR / f"images_224_clinical_seed{config.SEED}.pt"
    if f.exists():
        data = torch.load(f, weights_only=True)
    else:
        t0, data = time.perf_counter(), {}
        for s, df in frames.items():
            ids = clinical_isic_ids(s, df["lesion_id"])
            data[f"X_{s}"] = torch.stack([deep.load_image_224(config.image_path(i)) for i in ids])
        torch.save(data, f)
        print(f"  clinical cache built in {time.perf_counter() - t0:.0f} s -> {f}", flush=True)
    for s, df in frames.items():
        assert len(data[f"X_{s}"]) == len(df)
    return data


@torch.no_grad()
def predict_tta(model, X: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    """Mean softmax over the image and its horizontal, vertical and double flips."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = deep.prep(X[i:i + batch_size])
        views = [xb, xb.flip(3), xb.flip(2), xb.flip(2).flip(3)]
        out.append(torch.stack([torch.softmax(model(v), 1) for v in views]).mean(0).cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- weights
def simplex_grid(k: int, step: float = 0.1) -> np.ndarray:
    n = int(round(1 / step))
    pts = [c for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]
    return np.array(pts, dtype=float) / n


def combine(Ps: Sequence[np.ndarray], w: Sequence[float]) -> np.ndarray:
    return sum(wi * P for wi, P in zip(w, Ps))


def best_weights(y, Ps: Sequence[np.ndarray], grid: np.ndarray):
    scores = [deep.macro_f1(y, combine(Ps, w).argmax(1), Ps[0].shape[1]) for w in grid]
    i = int(np.argmax(scores))
    return grid[i], float(scores[i])


def crossfit_weight_search(y, Ps: Sequence[np.ndarray], grid: np.ndarray, repeats: int = 10, seed: int = 0) -> float:
    """Honest estimate of 'search the weights on validation': fit on half of val, score on the other half."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    K = Ps[0].shape[1]
    out = []
    for _ in range(repeats):
        idx = rng.permutation(len(y))
        a, b = idx[: len(y) // 2], idx[len(y) // 2:]
        for fi, ev in ((a, b), (b, a)):
            w, _ = best_weights(y[fi], [P[fi] for P in Ps], grid)
            out.append(deep.macro_f1(y[ev], combine([P[ev] for P in Ps], w).argmax(1), K))
    return float(np.mean(out))
