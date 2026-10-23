"""
Classical ML baseline (Milestone 2): feature sets, models, evaluation.

Design
------
* One sample = one LESION. The dermoscopic (``d_``) and clinical (``c_``)
  features of the lesion are concatenated (early fusion). Splits are the
  Milestone 1 lesion-level splits, so there is no leakage between views.
* Target: ``y_primary`` (Benign / Indeterminate / Malignant), the primary target
  fixed in Milestone 1. ``y_fine`` (8 classes) is the stretch target.
* Every model is a scikit-learn ``Pipeline``: median imputation -> standard
  scaling -> classifier. Imputer and scaler are fitted inside each CV fold, on
  training data only.
* Imbalance (Malignant : Indeterminate = 29.5 : 1): ``class_weight="balanced"``
  for every model that supports it; selection metric = balanced accuracy
  (mean per-class recall, the ISIC challenge metric).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config

PRIMARY = ["Benign", "Indeterminate", "Malignant"]
META_PREFIX = "meta_"

# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #
def add_metadata_features(les: pd.DataFrame) -> pd.DataFrame:
    """Non-leaky metadata -> numeric columns (age, sex one-hot, site one-hot).

    Categories are a fixed vocabulary (no statistics are learned here), missing
    values get their own indicator / NaN that the pipeline imputer handles.
    """
    out = les.copy()
    out[f"{META_PREFIX}age"] = out["age_approx"]
    for s in ("male", "female"):
        out[f"{META_PREFIX}sex_{s}"] = (out["sex"] == s).astype(float)
    # the only values present in MILK10k (37 % of lesions have no site -> indicator)
    sites = ["head/neck", "upper extremity", "lower extremity", "oral/genital"]
    for s in sites:
        out[f"{META_PREFIX}site_{s.replace('/', '_').replace(' ', '_')}"] = \
            (out["anatom_site_general"] == s).astype(float)
    out[f"{META_PREFIX}site_missing"] = out["anatom_site_general"].isna().astype(float)
    return out


def _cols(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


# name -> (column prefixes, description)
FEATURE_SETS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "F1_hist_derm":    (("d_hist_",), "Colour histograms RGB+HSV, dermoscopic"),
    "F2_colour_derm":  (("d_col_",), "Lesion vs skin colour statistics (RGB/HSV/Lab), dermoscopic"),
    "F3_texture_derm": (("d_tex_",), "Edge/texture filters (Sobel, LoG, local std, Canny), dermoscopic"),
    "F4_shape_derm":   (("d_shp_",), "Shape of the Otsu lesion mask (ABCD-inspired), dermoscopic"),
    "F5_all_derm":     (("d_hist_", "d_col_", "d_tex_", "d_shp_"), "F1-F4, dermoscopic only"),
    "F6_all_2views":   (("d_", "c_"), "F1-F4 for BOTH views (dermoscopic + clinical)"),
    "F7_all_2views_meta": (("d_", "c_", META_PREFIX), "F6 + age, sex, body site"),
    "F0_meta_only":    ((META_PREFIX,), "Metadata only (age, sex, body site) - reference"),
}


def feature_columns(df: pd.DataFrame, set_name: str) -> List[str]:
    prefixes, _ = FEATURE_SETS[set_name]
    cols = _cols(df, prefixes)
    if not cols:
        raise ValueError(f"feature set {set_name} matched no columns")
    return cols


def feature_group(col: str) -> str:
    """Column -> readable group, e.g. 'derm colour', 'clinical texture', 'metadata'."""
    if col.startswith(META_PREFIX):
        return "metadata"
    view = {"d_": "derm", "c_": "clinical"}[col[:2]]
    kind = {"hist": "histogram", "col": "colour", "tex": "texture", "shp": "shape"}[col[2:].split("_")[0]]
    return f"{view} {kind}"


# --------------------------------------------------------------------------- #
# Models + hyper-parameter grids
# --------------------------------------------------------------------------- #
def _pipe(clf) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", clf)])


def model_zoo(seed: int = config.SEED) -> Dict[str, Tuple[Pipeline, Dict]]:
    """name -> (pipeline, param grid). Grids are kept small on purpose (CPU budget)."""
    return {
        "LogReg": (_pipe(LogisticRegression(class_weight="balanced", max_iter=3000)),
                   {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "kNN": (_pipe(KNeighborsClassifier(weights="distance")),
                {"clf__n_neighbors": [5, 15, 31]}),
        "SVM_RBF": (_pipe(SVC(kernel="rbf", class_weight="balanced", probability=True,
                              random_state=seed)),
                    {"clf__C": [0.3, 1.0, 3.0, 10.0]}),
        "RandomForest": (_pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                      n_jobs=-1, random_state=seed)),
                         {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 5]}),
        "HistGB": (_pipe(HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                        random_state=seed)),
                   {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None]}),
    }


def dummy_models() -> Dict[str, Pipeline]:
    return {"Dummy_majority": _pipe(DummyClassifier(strategy="most_frequent")),
            "Dummy_stratified": _pipe(DummyClassifier(strategy="stratified", random_state=config.SEED))}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: Optional[np.ndarray] = None,
            classes: Sequence[str] = PRIMARY) -> Dict[str, float]:
    """Imbalance-aware metric set (same choice as in the Milestone 1 report)."""
    k = len(classes)
    labels = list(range(k))
    out = {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, c in enumerate(classes):
        out[f"recall_{c}"] = r[i]
        out[f"precision_{c}"] = p[i]
        out[f"f1_{c}"] = f[i]
    if proba is not None and len(np.unique(y_true)) == k:
        out["macro_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                             labels=labels)
    if list(classes) == PRIMARY:
        # triage view: Indeterminate + Malignant = "needs work-up" vs Benign
        t = y_true != 0
        pr = y_pred != 0
        out["workup_sensitivity"] = (t & pr).sum() / max(t.sum(), 1)
        out["benign_specificity"] = (~t & ~pr).sum() / max((~t).sum(), 1)
        if proba is not None:
            out["auc_benign_vs_rest"] = roc_auc_score(t, 1 - proba[:, 0])
            out["auc_malignant_vs_rest"] = roc_auc_score(y_true == 2, proba[:, 2])
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, proba: Optional[np.ndarray],
                 n_boot: int = 1000, seed: int = config.SEED,
                 keys: Sequence[str] = ("balanced_acc", "macro_f1", "recall_Malignant",
                                        "recall_Indeterminate", "recall_Benign",
                                        "workup_sensitivity", "macro_auc_ovr"),
                 classes: Sequence[str] = PRIMARY) -> pd.DataFrame:
    """Percentile 95 % CI by resampling lesions of the evaluation set with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics(y_true[idx], y_pred[idx], None if proba is None else proba[idx], classes)
        for k in keys:
            if k in m:
                vals[k].append(m[k])
    point = metrics(y_true, y_pred, proba, classes)
    rows = []
    for k in keys:
        if vals[k]:
            lo, hi = np.percentile(vals[k], [2.5, 97.5])
            rows.append({"metric": k, "value": point[k], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def confusion(y_true, y_pred, classes: Sequence[str] = PRIMARY) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return pd.DataFrame(cm, index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes])


# --------------------------------------------------------------------------- #
# Interpretation: grouped permutation importance
# --------------------------------------------------------------------------- #
def grouped_permutation_importance(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10,
                                   seed: int = config.SEED) -> pd.DataFrame:
    """Drop in balanced accuracy when ALL columns of a feature group are shuffled together.

    Shuffling a whole group (e.g. every derm colour column) avoids the problem that
    correlated single columns hide each other's importance.
    """
    rng = np.random.default_rng(seed)
    base = balanced_accuracy_score(y, model.predict(X))
    groups: Dict[str, List[str]] = {}
    for c in X.columns:
        groups.setdefault(feature_group(c), []).append(c)
    rows = []
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - balanced_accuracy_score(y, model.predict(Xp)))
        rows.append({"group": g, "n_features": len(cols), "drop_mean": np.mean(drops),
                     "drop_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("drop_mean", ascending=False).reset_index(drop=True)


def average_views(image_probs: np.ndarray, image_table: pd.DataFrame,
                  class_names: Sequence[str] = PRIMARY) -> pd.DataFrame:
    """Late fusion: average the image-level probabilities of each lesion's views.

    Same rule as ``datasets.aggregate_predictions`` (Milestone 1), re-implemented
    here so the classical baseline does not need PyTorch installed.
    """
    cols = [f"p_{c}" for c in class_names]
    df = pd.DataFrame(np.asarray(image_probs, float), columns=cols)
    df["lesion_id"] = image_table["lesion_id"].to_numpy()
    out = df.groupby("lesion_id")[cols].mean()
    out["pred"] = out[cols].to_numpy().argmax(axis=1)
    return out


def refit_on(model, X: pd.DataFrame, y: np.ndarray):
    m = clone(model)
    m.fit(X, y)
    return m
