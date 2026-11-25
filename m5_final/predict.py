"""predict(image_path) -> class probabilities for one new dermoscopic image (Milestone 5).

Loads the saved model and applies EXACTLY the transform used for training and
validation (milk10k_pipeline.deep.load_image_224 + ImageNet normalisation), so
there is no train / serve skew.

    python m5_final/predict.py path/to/ISIC_xxx.jpg                 # 11-class model (leaderboard)
    python m5_final/predict.py path/to/ISIC_xxx.jpg --model m4      # 3-class model (Benign/Indet./Malignant)

From Python:
    from m5_final.predict import predict
    predict("ISIC_0000000.jpg")   # {'AKIEC': 0.01, 'BCC': 0.83, ...}

Not a medical device: see model_card.md.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from milk10k_pipeline import deep  # noqa: E402

CHECKPOINTS = {"11class": deep.CHECKPOINT_DIR / "model_11class.pt",
               "m4": deep.CHECKPOINT_DIR / "model_m4.pt"}


@lru_cache(maxsize=2)
def load_model(which: str = "11class"):
    """Model + its class names, loaded once and kept in memory."""
    model, ck = deep.load_checkpoint(CHECKPOINTS[which])
    return model, ck["classes"]


@torch.no_grad()
def predict_batch(image_paths, which: str = "11class", batch_size: int = 32):
    """Softmax probabilities, one row per image (same order as image_paths)."""
    model, classes = load_model(which)
    out = []
    paths = list(image_paths)
    for i in range(0, len(paths), batch_size):
        xb = torch.stack([deep.load_image_224(p) for p in paths[i:i + batch_size]])
        out.append(deep.predict_proba(model, xb))
    import numpy as np
    return (np.concatenate(out) if out else np.zeros((0, len(classes)))), classes


def predict(image_path, which: str = "11class") -> dict:
    """One image -> {class name: probability}; the probabilities add up to 1."""
    P, classes = predict_batch([image_path], which)
    return {c: round(float(p), 4) for c, p in zip(classes, P[0])}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    which = "m4" if "--model" in sys.argv and sys.argv[sys.argv.index("--model") + 1] == "m4" else "11class"
    probs = predict(sys.argv[1], which)
    for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"{c:14s} {p:.4f}")
    print("->", max(probs, key=probs.get))
