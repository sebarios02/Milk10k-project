"""Tests for the hand-crafted features and the classical-baseline helpers (Milestone 2)."""
import numpy as np
import pandas as pd

from milk10k_pipeline import classical as C
from milk10k_pipeline import features as F


def _synthetic(h=450, w=600, seed=0):
    """Pinkish skin with a dark brown ellipse in the centre."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (215, 170, 160), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    inside = ((xx - w / 2) / 120) ** 2 + ((yy - h / 2) / 80) ** 2 <= 1
    img[inside] = (90, 55, 40)
    noise = rng.integers(-5, 6, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8), inside


def test_mask_finds_dark_centre_lesion():
    img, truth = _synthetic()
    mask, fallback = F.lesion_mask(img)
    assert not fallback
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou > 0.85


def test_features_are_finite_and_grouped():
    img, _ = _synthetic()
    f = F.extract_image_features(img)
    assert all(np.isfinite(v) for v in f.values())
    for g in F.FEATURE_GROUPS:
        assert any(k.startswith(g + "_") for k in f)
    # histograms are normalised
    assert abs(sum(v for k, v in f.items() if k.startswith("hist_R")) - 1) < 1e-6
    # an ellipse is fairly symmetric and convex
    assert f["shp_solidity"] > 0.95 and f["shp_asym_major"] < 0.1
    # the lesion is darker than the skin
    assert f["col_lab_L_contrast"] < 0


def test_metrics_dummy_balanced_accuracy_is_chance():
    y = np.array([0, 0, 1, 2, 2, 2])
    pred = np.full_like(y, 2)
    m = C.metrics(y, pred)
    assert abs(m["balanced_acc"] - 1 / 3) < 1e-9
    assert m["recall_Malignant"] == 1.0 and m["workup_sensitivity"] == 1.0


def test_feature_sets_select_expected_prefixes():
    df = pd.DataFrame(columns=["d_hist_R00", "d_col_x", "c_tex_y", "meta_age", "split"])
    assert C.feature_columns(df, "F1_hist_derm") == ["d_hist_R00"]
    assert set(C.feature_columns(df, "F7_all_2views_meta")) == {"d_hist_R00", "d_col_x", "c_tex_y", "meta_age"}
