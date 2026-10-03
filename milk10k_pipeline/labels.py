"""
Label strategy (Milestone 1, B3) and imbalance helpers.

Primary target : diagnosis_1 with 3 classes (Benign / Indeterminate / Malignant).
                 Indeterminate is KEPT as its own class (see the report).
Stretch target : the 11 dx classes collapsed to 8 "fine" classes:
                 the four rare BENIGN classes (BEN_OTH, DF, INF, VASC) are merged
                 into OTHER_BENIGN, MAL_OTH (9 lesions) is kept on its own as
                 OTHER_MALIGNANT. Benign and malignant rare classes are never
                 mixed in the same group.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

PRIMARY_TO_INDEX = {"Benign": 0, "Indeterminate": 1, "Malignant": 2}

FINE_MERGE = {
    "AKIEC": "AKIEC", "BCC": "BCC", "BKL": "BKL", "MEL": "MEL", "NV": "NV",
    "SCCKA": "SCCKA",
    "BEN_OTH": "OTHER_BENIGN", "DF": "OTHER_BENIGN", "INF": "OTHER_BENIGN",
    "VASC": "OTHER_BENIGN",
    "MAL_OTH": "OTHER_MALIGNANT",
}
FINE_CLASSES = ["AKIEC", "BCC", "BKL", "MEL", "NV", "SCCKA", "OTHER_BENIGN", "OTHER_MALIGNANT"]
FINE_TO_INDEX = {c: i for i, c in enumerate(FINE_CLASSES)}


def build_label_map() -> Dict:
    """The final label mapping, as saved to label_map.json and loaded by later milestones."""
    return {
        "primary": {
            "source_column": "diagnosis_1",
            "classes": list(PRIMARY_TO_INDEX),
            "to_index": PRIMARY_TO_INDEX,
            "note": "Indeterminate kept as a 3rd class (all of it is AKIEC); "
                    "for triage it is reported together with Malignant as 'needs work-up'.",
        },
        "fine": {
            "source_column": "dx",
            "merge": FINE_MERGE,
            "classes": FINE_CLASSES,
            "to_index": FINE_TO_INDEX,
            "note": "Rare benign classes merged into OTHER_BENIGN; MAL_OTH kept as "
                    "OTHER_MALIGNANT (never mixed with benign). Splits are still "
                    "stratified on the original 11 classes.",
        },
        # dx -> diagnosis_1 is NOT a function: AKIEC maps to two values.
        "dx_to_primary": {
            "AKIEC": ["Indeterminate", "Malignant"], "BCC": ["Malignant"],
            "BEN_OTH": ["Benign"], "BKL": ["Benign"], "DF": ["Benign"], "INF": ["Benign"],
            "MAL_OTH": ["Malignant"], "MEL": ["Malignant"], "NV": ["Benign"],
            "SCCKA": ["Malignant"], "VASC": ["Benign"],
        },
    }


def save_label_map(path: Optional[Path] = None) -> Path:
    """Write label_map.json (overwrites)."""
    path = Path(path or config.LABEL_MAP_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_label_map(), indent=2))
    return path


def load_label_map(path: Optional[Path] = None) -> Dict:
    """Read label_map.json."""
    return json.loads(Path(path or config.LABEL_MAP_JSON).read_text())


def add_label_columns(df: pd.DataFrame, label_map: Optional[Dict] = None) -> pd.DataFrame:
    """Add integer columns ``y_primary`` and ``y_fine`` (and string ``fine``) to an image/lesion table."""
    lm = label_map or build_label_map()
    out = df.copy()
    out["y_primary"] = out["diagnosis_1"].map(lm["primary"]["to_index"]).astype(int)
    out["fine"] = out["dx"].map(lm["fine"]["merge"])
    out["y_fine"] = out["fine"].map(lm["fine"]["to_index"]).astype(int)
    return out


def balanced_class_weights(y: Sequence[int], n_classes: int, cap: Optional[float] = None) -> np.ndarray:
    """w_c = N / (K * n_c)  (same formula as sklearn 'balanced'), optionally capped.

    Classes absent from ``y`` get weight 0 (they cannot contribute to the loss anyway).
    """
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.zeros(n_classes)
    present = counts > 0
    w[present] = len(y) / (present.sum() * counts[present])
    if cap is not None:
        w = np.minimum(w, cap)
    return w


def sample_weights_for_sampler(y: Sequence[int]) -> np.ndarray:
    """Per-sample weight 1/count(class): every class is drawn equally often on average."""
    y = np.asarray(y)
    counts = np.bincount(y)
    return 1.0 / counts[y]
