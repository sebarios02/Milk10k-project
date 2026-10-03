"""
Data-quality report (Milestone 1, B4) and label-consistency checks (Part A, A1.1).

Every check returns data (DataFrames / dicts) so it can be printed in the
notebook AND written to the report file by scripts/04_quality_report.py.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from . import config

# One line per column with missing values: decision + reason.
MISSING_VALUE_DECISIONS = {
    "anatom_site_general": ("'unknown' category",
                            "37% missing and missingness is informative (A1.2 q4); keep it as its own level."),
    "age_approx": ("impute train median (inside the model Pipeline)",
                   "<0.5% missing; median fitted on train only to avoid leakage."),
    "anatom_site_special": ("drop", "98% missing, no usable signal."),
    "diagnosis_3": ("not used (label column)", "Part of the diagnosis hierarchy -> leaks the label."),
    "diagnosis_4": ("not used (label column)", "Part of the diagnosis hierarchy -> leaks the label."),
    "melanocytic": ("not used (label-derived)", "77% missing and it is a property of the diagnosis itself."),
}


def missing_table(images: pd.DataFrame) -> pd.DataFrame:
    """Missing counts/% per column + the handling decision."""
    miss = images.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    tab = pd.DataFrame({"n_missing": miss, "pct_missing": (100 * miss / len(images)).round(2)})
    tab["decision"] = [MISSING_VALUE_DECISIONS.get(c, ("?", ""))[0] for c in tab.index]
    tab["reason"] = [MISSING_VALUE_DECISIONS.get(c, ("?", ""))[1] for c in tab.index]
    return tab


def images_per_lesion_check(images: pd.DataFrame) -> Dict[str, int]:
    """Exactly 2 images per lesion, one of each image_type."""
    per = images.groupby("lesion_id")["image_type"].agg(["size", "nunique"])
    types = images.groupby("lesion_id")["image_type"].apply(frozenset)
    expected = frozenset({"dermoscopic", "clinical: close-up"})
    return {
        "n_lesions": int(len(per)),
        "lesions_not_2_images": int((per["size"] != 2).sum()),
        "lesions_not_2_types": int((per["nunique"] != 2).sum()),
        "lesions_wrong_type_pair": int((types != expected).sum()),
    }


def lesion_fields_consistency(images: pd.DataFrame,
                              cols=("age_approx", "sex", "anatom_site_general", "diagnosis_1",
                                    "diagnosis_2", "diagnosis_3", "diagnosis_4")) -> pd.Series:
    """Number of lesions whose two images DISAGREE on each field (NaN==NaN counts as agreeing)."""
    res = {}
    for c in cols:
        n_unique = images.groupby("lesion_id")[c].nunique(dropna=False)
        res[c] = int((n_unique > 1).sum())
    return pd.Series(res, name="lesions_with_disagreement")


def class_to_diagnosis1(images: pd.DataFrame) -> pd.DataFrame:
    """Lesion counts of the 11-class label vs diagnosis_1 (A1.1c)."""
    les = images.drop_duplicates("lesion_id")
    return pd.crosstab(les["dx"], les["diagnosis_1"])


def hierarchy_violations(images: pd.DataFrame, child: str, parent: str) -> pd.Series:
    """Child values that map to MORE than one parent value (empty Series = hierarchy OK)."""
    n_parents = images.dropna(subset=[child]).groupby(child)[parent].nunique()
    return n_parents[n_parents > 1]


def shortcut_crosstabs(images: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Row-normalised (%) cross-tabs of acquisition fields vs diagnosis_1."""
    out = {}
    for col in ("image_manipulation", "image_type", "diagnosis_confirm_type"):
        counts = pd.crosstab(images[col], images["diagnosis_1"])
        pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
        out[col] = counts.astype(str) + " (" + pct.astype(str) + "%)"
    return out


def leaky_columns_table() -> pd.DataFrame:
    """Columns that must NOT be model inputs, with the reason."""
    reasons = {
        "diagnosis_1": "the target itself",
        "diagnosis_2": "coarser/finer level of the same diagnosis (determines diagnosis_1)",
        "diagnosis_3": "finer level of the diagnosis (determines diagnosis_1 and dx)",
        "diagnosis_4": "finest level of the diagnosis",
        "diagnosis_confirm_type": "histopathology vs clinical assessment = whether the lesion was biopsied, "
                                  "decided because it looked suspicious; strongly tied to the label",
        "concomitant_biopsy": "same information as diagnosis_confirm_type",
        "melanocytic": "attribute of the diagnosis, only filled after diagnosis",
        "lesion_id / isic_id": "identifiers; only used for grouping",
        "image_manipulation": "acquisition artefact; kept out of the inputs (see shortcut check)",
    }
    return pd.DataFrame({"column": list(reasons), "reason_not_an_input": list(reasons.values())})
