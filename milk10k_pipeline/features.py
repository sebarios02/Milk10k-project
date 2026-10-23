"""
Hand-crafted image features for the classical ML baseline (Milestone 2).

Every feature is built from a concept seen in Sessions 2-4, so the baseline
is a direct test of "how far do the classic tools get us?":

* Session 2  - RGB channels, grayscale, histograms, normalisation
* Session 3  - colour spaces (HSV), lesion hue / brightness
* Session 4  - convolution, Gaussian blur (sigma = scale), Sobel gradients,
               Laplacian of Gaussian, local standard deviation (texture),
               Canny edges, Otsu lesion mask

Feature GROUPS (prefixes) so that feature sets can be compared later:

    hist_   colour histograms (RGB 16 bins/channel + HSV 16/8/8 bins), whole image
    col_    colour statistics: RGB / HSV / CIELab mean and std INSIDE the lesion,
            mean of the surrounding skin, lesion-minus-skin contrast, colour
            variegation (spread of the lesion colours)
    tex_    edge / texture filters on the grayscale image: Sobel magnitude,
            |LoG| at 3 scales, local std (9x9), Canny edge density, gray-level
            entropy - inside the lesion, on the border ring and on the whole image
    shp_    shape of the Otsu lesion mask (ABCD-inspired): area, compactness,
            solidity, eccentricity, asymmetry, border sharpness

All functions work on one RGB uint8 image (MILK10k: always 600 x 450).
``extract_image_features`` returns a flat ``dict[str, float]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

FEATURE_GROUPS = ("hist", "col", "tex", "shp")

# settings (kept here so the report can quote them)
MASK_SIGMA = 4.0          # Gaussian blur before Otsu (removes hair / dots, keeps lesion)
EDGE_SIGMA = 2.0          # pre-smoothing before Sobel / Canny (Session 4: 2-4 is the useful range)
LOG_SIGMAS = (1.0, 2.0, 4.0)
LOCAL_STD_SIZE = 9
CANNY_LOW, CANNY_HIGH = 0.08, 0.20   # relative to the max gradient, as in my Session 4 Canny
RING_WIDTH = 7            # px, width of the border ring (dilate - erode)
HIST_BINS_RGB = 16
HIST_BINS_HSV = (16, 8, 8)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# --------------------------------------------------------------------------- #
# Lesion mask (Session 4: Otsu threshold on the blurred grayscale image)
# --------------------------------------------------------------------------- #
def lesion_mask(rgb: np.ndarray, sigma: float = MASK_SIGMA) -> Tuple[np.ndarray, bool]:
    """Rough lesion segmentation (Session 4 Otsu idea, made colour-aware).

    1. Gaussian blur (sigma=4): hairs and pigment dots disappear, the lesion
       "valley" survives (Session 4, Figure 4)
    2. "lesion-ness" score per pixel = darker than the skin (CIELab L*) +
       redder than the skin (a*), both as z-scores w.r.t. a frame band that is
       assumed to be skin. The redness term matters: in grayscale pale pink BCCs
       are barely darker than the skin (Session 4 finding).
    3. Otsu threshold on the score, small morphological opening
    4. among the connected regions keep the one with the best
       area x centre-prior score (lesions are usually centred), fill holes

    Returns (mask[bool], fallback_used). If no region is found the central
    ellipse covering ~25 % of the image is used and ``fallback_used`` is True.
    """
    h, w = rgb.shape[:2]
    blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
    lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a = lab[..., 0], lab[..., 1]
    band = np.zeros((h, w), bool)
    b = max(h, w) // 12
    band[:b], band[-b:], band[:, :b], band[:, -b:] = True, True, True, True

    def z(x):
        return (x - np.median(x[band])) / (x[band].std() + 1e-6)

    score = -z(L) + np.maximum(z(a), 0)
    s8 = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab_cc, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    best, best_score = 0, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.002 * h * w:
            continue
        d = np.hypot((cent[i, 0] - w / 2) / w, (cent[i, 1] - h / 2) / h)
        sc = area * np.exp(-(d / 0.3) ** 2)
        if sc > best_score:
            best, best_score = i, sc
    if best == 0:
        mm = np.zeros((h, w), np.uint8)
        cv2.ellipse(mm, (w // 2, h // 2), (int(w * 0.28), int(h * 0.28)), 0, 0, 360, 1, -1)
        return mm.astype(bool), True
    comp = (lab_cc == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, -1)          # fill holes
    return filled.astype(bool), False


def border_ring(mask: np.ndarray, width: int = RING_WIDTH) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def _hist(values: np.ndarray, bins: int, rng: Tuple[float, float]) -> np.ndarray:
    h, _ = np.histogram(values, bins=bins, range=rng)
    return h / max(h.sum(), 1)


def histogram_features(rgb: np.ndarray, hsv: np.ndarray) -> Dict[str, float]:
    """Session 2: normalised per-channel histograms of the whole image."""
    f: Dict[str, float] = {}
    for c, name in enumerate("RGB"):
        for i, v in enumerate(_hist(rgb[..., c], HIST_BINS_RGB, (0, 256))):
            f[f"hist_{name}{i:02d}"] = v
    ranges = [(0, 180), (0, 256), (0, 256)]          # OpenCV hue is 0..179
    for c, name in enumerate("HSV"):
        for i, v in enumerate(_hist(hsv[..., c], HIST_BINS_HSV[c], ranges[c])):
            f[f"hist_{name}{i:02d}"] = v
    return f


def _circular_hue_stats(h: np.ndarray) -> Tuple[float, float]:
    """Hue is an angle: mean via sin/cos, spread = 1 - resultant length."""
    ang = h.astype(float) * (2 * np.pi / 180.0)
    s, c = np.sin(ang).mean(), np.cos(ang).mean()
    return float(np.arctan2(s, c)), float(1 - np.hypot(s, c))


def colour_features(rgb: np.ndarray, hsv: np.ndarray, lab: np.ndarray,
                    mask: np.ndarray) -> Dict[str, float]:
    """Session 3: colour inside the lesion vs. the surrounding skin, in 3 colour spaces."""
    f: Dict[str, float] = {}
    skin = ~cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if skin.sum() < 100:            # lesion fills the frame -> use the outermost frame band
        skin = np.zeros_like(mask)
        skin[:20], skin[-20:], skin[:, :20], skin[:, -20:] = True, True, True, True
    spaces = {"rgb": (rgb, "RGB"), "hsv": (hsv, "HSV"), "lab": (lab, "Lab")}
    for sp, (img, chans) in spaces.items():
        img = img.astype(float)
        for c, ch in enumerate(chans):
            if sp == "hsv" and ch == "H":
                continue                         # hue handled as an angle below
            les, sk = img[..., c][mask], img[..., c][skin]
            f[f"col_{sp}_{ch}_lesion_mean"] = les.mean()
            f[f"col_{sp}_{ch}_lesion_std"] = les.std()
            f[f"col_{sp}_{ch}_skin_mean"] = sk.mean()
            f[f"col_{sp}_{ch}_contrast"] = les.mean() - sk.mean()
    hm, hs = _circular_hue_stats(hsv[..., 0][mask])
    f["col_hsv_H_lesion_sin"], f["col_hsv_H_lesion_cos"] = np.sin(hm), np.cos(hm)
    f["col_hsv_H_lesion_spread"] = hs
    # colour difference lesion vs skin (Delta E in CIELab, OpenCV 8-bit Lab scaled back)
    L = lab.astype(float)
    L[..., 0] *= 100 / 255
    L[..., 1:] -= 128
    d = L[mask].mean(0) - L[skin].mean(0)
    f["col_deltaE_lesion_skin"] = float(np.linalg.norm(d))
    # colour variegation: how spread out the lesion colours are (trace of Lab covariance)
    f["col_variegation"] = float(np.sqrt(np.trace(np.cov(L[mask].T)))) if mask.sum() > 10 else 0.0
    # share of "red" (vascular / inflamed) and "blue-grey" (regression) lesion pixels
    hh, ss = hsv[..., 0][mask].astype(int), hsv[..., 1][mask].astype(int)
    f["col_frac_red"] = float(np.mean(((hh < 10) | (hh > 165)) & (ss > 60)))
    f["col_frac_bluegrey"] = float(np.mean((hh > 90) & (hh < 140)))
    return f


def _masked_stats(x: np.ndarray, region: np.ndarray, prefix: str, f: Dict[str, float],
                  pct: bool = False) -> None:
    v = x[region] if region.any() else x.ravel()
    f[f"{prefix}_mean"] = float(v.mean())
    if pct:
        f[f"{prefix}_std"] = float(v.std())
        f[f"{prefix}_p90"] = float(np.percentile(v, 90))


def _canny(gray01: np.ndarray, sigma: float) -> np.ndarray:
    """Canny with thresholds relative to the max gradient (same rule as my Session 4 version)."""
    g = cv2.GaussianBlur(gray01, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mmax = float(np.sqrt(gx ** 2 + gy ** 2).max()) or 1.0
    g8 = np.clip(g * 255, 0, 255).astype(np.uint8)
    # cv2.Canny expects thresholds on the 8-bit Sobel scale (x255)
    return cv2.Canny(g8, CANNY_LOW * mmax * 255, CANNY_HIGH * mmax * 255, L2gradient=True) > 0


def texture_features(gray01: np.ndarray, mask: np.ndarray, ring: np.ndarray) -> Dict[str, float]:
    """Session 4: filters as texture / edge descriptors."""
    f: Dict[str, float] = {}
    g = cv2.GaussianBlur(gray01, (0, 0), EDGE_SIGMA)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(gx ** 2 + gy ** 2)
    _masked_stats(sob, mask, "tex_sobel_lesion", f, pct=True)
    _masked_stats(sob, ring, "tex_sobel_border", f, pct=True)
    f["tex_sobel_image_mean"] = float(sob.mean())
    # Laplacian of Gaussian at 3 scales: dots / globules / blobs of increasing size
    for s in LOG_SIGMAS:
        gs = cv2.GaussianBlur(gray01, (0, 0), s)
        log = np.abs(cv2.Laplacian(gs, cv2.CV_32F, ksize=3)) * s ** 2   # scale-normalised
        _masked_stats(log, mask, f"tex_log{int(s)}_lesion", f)
        f[f"tex_log{int(s)}_image_mean"] = float(log.mean())
    # local standard deviation (box filters: E[x^2] - E[x]^2)
    k = (LOCAL_STD_SIZE, LOCAL_STD_SIZE)
    mu = cv2.blur(gray01, k)
    lstd = np.sqrt(np.maximum(cv2.blur(gray01 ** 2, k) - mu ** 2, 0))
    _masked_stats(lstd, mask, "tex_localstd_lesion", f, pct=True)
    skin = ~mask
    _masked_stats(lstd, skin if skin.any() else mask, "tex_localstd_skin", f)
    # Canny edge density
    edges = _canny(gray01, EDGE_SIGMA)
    f["tex_canny_lesion_density"] = float(edges[mask].mean()) if mask.any() else 0.0
    f["tex_canny_image_density"] = float(edges.mean())
    # gray-level entropy inside the lesion (Session 2 histogram)
    h = _hist(gray01[mask] if mask.any() else gray01.ravel(), 32, (0, 1))
    h = h[h > 0]
    f["tex_gray_entropy_lesion"] = float(-(h * np.log2(h)).sum())
    f["tex_gray_lesion_std"] = float(gray01[mask].std()) if mask.any() else 0.0
    return f


def shape_features(mask: np.ndarray, gray01: np.ndarray, fallback: bool) -> Dict[str, float]:
    """ABCD-inspired descriptors of the Otsu mask (Asymmetry, Border, Diameter)."""
    f: Dict[str, float] = {}
    h, w = mask.shape
    m8 = mask.astype(np.uint8)
    area = float(m8.sum())
    f["shp_area_frac"] = area / (h * w)
    f["shp_mask_fallback"] = float(fallback)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts or area < 10:
        for k in ("compactness", "solidity", "eccentricity", "asym_major", "asym_minor",
                  "extent", "centroid_offset"):
            f[f"shp_{k}"] = np.nan
        return f
    c = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(c, True)
    f["shp_compactness"] = perim ** 2 / (4 * np.pi * area)        # 1 = circle, >1 irregular border
    hull = cv2.convexHull(c)
    f["shp_solidity"] = area / max(cv2.contourArea(hull), 1.0)
    x, y, bw, bh = cv2.boundingRect(c)
    f["shp_extent"] = area / (bw * bh)
    # moments -> principal axes
    M = cv2.moments(m8, binaryImage=True)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    common = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    f["shp_eccentricity"] = float(np.sqrt(1 - l2 / l1)) if l1 > 0 else 0.0
    f["shp_centroid_offset"] = float(np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2))
    # asymmetry: rotate so the major axis is horizontal, flip, measure non-overlap
    theta = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    R = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    R[0, 2] += w / 2 - cx
    R[1, 2] += h / 2 - cy
    big = max(h, w) * 2
    R[0, 2] += (big - w) / 2
    R[1, 2] += (big - h) / 2
    rot = cv2.warpAffine(m8, R, (big, big), flags=cv2.INTER_NEAREST) > 0
    a = rot.sum()
    f["shp_asym_major"] = float((rot ^ rot[::-1, :]).sum() / (2 * a)) if a else np.nan
    f["shp_asym_minor"] = float((rot ^ rot[:, ::-1]).sum() / (2 * a)) if a else np.nan
    return f


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_image_features(rgb: np.ndarray, groups: Iterable[str] = FEATURE_GROUPS) -> Dict[str, float]:
    """All requested feature groups for one RGB uint8 image."""
    groups = set(groups)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray01 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask, fallback = lesion_mask(rgb)
    ring = border_ring(mask)
    f: Dict[str, float] = {}
    if "hist" in groups:
        f.update(histogram_features(rgb, hsv))
    if "col" in groups:
        f.update(colour_features(rgb, hsv, lab, mask))
    if "tex" in groups:
        f.update(texture_features(gray01, mask, ring))
    if "shp" in groups:
        f.update(shape_features(mask, gray01, fallback))
    return {k: float(v) for k, v in f.items()}


def extract_from_path(path: Path, groups: Iterable[str] = FEATURE_GROUPS) -> Dict[str, float]:
    return extract_image_features(load_rgb(Path(path)), groups)


# --------------------------------------------------------------------------- #
# Image table -> lesion table (one row per lesion, both views side by side)
# --------------------------------------------------------------------------- #
VIEW_PREFIX = {"dermoscopic": "d_", "clinical: close-up": "c_"}


def to_lesion_features(img_feats, image_table) -> "pd.DataFrame":
    """Pivot per-image features to one row per lesion with ``d_`` (dermoscopic)
    and ``c_`` (clinical close-up) prefixes.

    ``img_feats`` is indexed by isic_id; ``image_table`` has isic_id, lesion_id, image_type.
    """
    import pandas as pd
    t = image_table[["isic_id", "lesion_id", "image_type"]].merge(
        img_feats, left_on="isic_id", right_index=True, how="inner", validate="one_to_one")
    parts = []
    for view, pre in VIEW_PREFIX.items():
        v = t[t["image_type"] == view].drop(columns=["isic_id", "image_type"]).set_index("lesion_id")
        parts.append(v.add_prefix(pre))
    return pd.concat(parts, axis=1, join="inner")
