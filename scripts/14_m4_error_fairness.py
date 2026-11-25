"""Milestone 4, step 2: error analysis and fairness check of the final M4 model (test split).

Reads outputs/m4/test_predictions_3class.csv (written by 13_m4_best_model.py), so it
does not re-run any network.

Error analysis
  * confusion matrix (counts and row %) and recall per 11-class diagnosis
  * a gallery of the 10 most confident mistakes (dermoscopic + clinical view)
  * what the mistakes have in common: errors vs correct lesions on metadata and on
    the Milestone 2 / Session 8 Otsu lesion-mask features (area, asymmetry)
Fairness
  * macro-F1 (and malignant recall) per skin_tone_class (MILK10k scale: 0 = very dark ...
    5 = very light, not Fitzpatrick), per sex and per age group,
    with group size and a bootstrap 95% CI; groups with < 30 lesions are flagged.

Outputs: outputs/m4/{error_by_dx.csv, errors_vs_correct.csv, misclassified_gallery.csv, fairness.csv},
         figures/m4_confusion_rowpct.png, figures/m4_misclassified_gallery.png, figures/m4_fairness.png
"""
import _bootstrap  # noqa: F401

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

from milk10k_pipeline import config
from milk10k_pipeline import deep

SMOKE = "--smoke" in sys.argv
OUT = config.OUTPUT_DIR / ("m4_smoke" if SMOKE else "m4")
FIG = config.FIGURE_DIR
FIG.mkdir(parents=True, exist_ok=True)
K, C = 3, deep.PRIMARY
MIN_GROUP = 30

t = pd.read_csv(OUT / "test_predictions_3class.csv")
t["correct"] = t["y3"] == t["pred"]
t["p_true"] = t[[f"p_{c}" for c in C]].values[np.arange(len(t)), t["y3"]]
t["p_pred"] = t[[f"p_{c}" for c in C]].values[np.arange(len(t)), t["pred"]]
print(f"{len(t)} test lesions, {(~t.correct).sum()} errors ({(~t.correct).mean():.1%})")

# ----------------------------------------------------------------------------- confusion (row %)
cm = confusion_matrix(t.y3, t.pred, labels=[0, 1, 2])
pct = cm / cm.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(5, 4.3))
ax.imshow(pct, cmap="Oranges", vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                color="white" if pct[i, j] > 60 else "black")
ax.set_xticks(range(3)); ax.set_xticklabels(C); ax.set_yticks(range(3)); ax.set_yticklabels(C)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Final M4 model, TEST: counts and row %", fontsize=10)
fig.savefig(FIG / "m4_confusion_rowpct.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- recall per 11-class diagnosis
by_dx = pd.DataFrame({dx: {"n": len(g), "diagnosis_1": "/".join(sorted(g.diagnosis_1.unique())),
                           **{f"pred {c} %": round((g.pred == i).mean() * 100, 1) for i, c in enumerate(C)},
                           "correct %": round(g.correct.mean() * 100, 1)}
                      for dx, g in t.groupby("dx")}).T.sort_values("n", ascending=False)
by_dx.index.name = "dx"
by_dx.to_csv(OUT / "error_by_dx.csv")
print("\nby diagnosis:\n", by_dx.to_string())

# ----------------------------------------------------------------------------- errors vs correct
feat_path = config.OUTPUT_DIR / "features" / "lesion_features.csv.gz"
mask_cols = ["d_shp_area_frac", "d_shp_asym_major", "d_shp_asym_minor", "d_shp_mask_fallback"]
if feat_path.exists():
    fe = pd.read_csv(feat_path, usecols=["lesion_id"] + mask_cols)
    t = t.merge(fe, on="lesion_id", how="left")
rows = {}
for name, g in [("correct", t[t.correct]), ("all errors", t[~t.correct]),
                ("Malignant -> Benign", t[(t.y3 == 2) & (t.pred == 0)]),
                ("Benign -> Malignant", t[(t.y3 == 0) & (t.pred == 2)])]:
    r = {"n": len(g), "median age": g.age_approx.median(), "% male": round((g.sex == "male").mean() * 100, 1),
         "% head/neck": round((g["site"] == "head_neck_face").mean() * 100, 1),
         "mean confidence (p of predicted class)": round(g.p_pred.mean(), 3)}
    if "skin_tone_class" in g:
        r["% skin tone 0-2 (darker)"] = round(g.skin_tone_class.isin([0, 1, 2]).mean() * 100, 1)
    for c in mask_cols:
        if c in g:
            r[f"median {c}"] = round(g[c].median(), 3)
    rows[name] = r
evc = pd.DataFrame(rows).T
evc.to_csv(OUT / "errors_vs_correct.csv")
print("\nerrors vs correct:\n", evc.to_string())

# ----------------------------------------------------------------------------- gallery: 10 most confident mistakes
err = t[~t.correct].sort_values("p_pred", ascending=False)
pick = pd.concat([err[(err.y3 == 2) & (err.pred == 0)].head(4),      # missed malignant (the dangerous error)
                  err[(err.y3 == 0) & (err.pred == 2)].head(3),      # benign flagged as malignant
                  err[err.y3 == 1].head(2),                          # Indeterminate (AKIEC)
                  err[err.pred == 1].head(1)]).drop_duplicates("lesion_id").head(10)
split_imgs = pd.read_csv(config.SPLIT_DIR / "test.csv")
clin = split_imgs[split_imgs.image_type != "dermoscopic"].set_index("lesion_id")["isic_id"]
pick["clinical_isic_id"] = pick.lesion_id.map(clin)
cols = ["lesion_id", "isic_id", "clinical_isic_id", "dx", "diagnosis_1", "pred_label", "p_pred", "age_approx",
        "sex", "site"] + [c for c in ["skin_tone_class", "d_shp_area_frac"] if c in pick]
pick[cols].to_csv(OUT / "misclassified_gallery.csv", index=False)

n = len(pick)
if n:
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.6), squeeze=False)
    for j, (_, r) in enumerate(pick.iterrows()):
        for i, iid in enumerate([r.isic_id, r.clinical_isic_id]):
            ax = axes[i, j]
            p = config.image_path(iid) if isinstance(iid, str) else None
            if p is not None and p.exists():
                ax.imshow(Image.open(p).convert("RGB").resize((224, 224)))
            ax.set_xticks([]); ax.set_yticks([])
        tone = f" tone {int(r.skin_tone_class)}" if "skin_tone_class" in r and pd.notna(r.skin_tone_class) else ""
        axes[0, j].set_title(f"{r.dx} ({r.diagnosis_1[:5]})\n-> {r.pred_label[:5]} p={r.p_pred:.2f}", fontsize=7.5)
        axes[1, j].set_xlabel(f"{int(r.age_approx) if pd.notna(r.age_approx) else '?'} {str(r.sex)[:1]}"
                              f" {str(r.site)[:10]}{tone}", fontsize=6.5)
    axes[0, 0].set_ylabel("dermoscopic\n(model input)", fontsize=8); axes[1, 0].set_ylabel("clinical\n(not used)", fontsize=8)
    fig.suptitle("Final M4 model: the most confident mistakes on the test split", fontsize=10)
    plt.tight_layout(); fig.savefig(FIG / "m4_misclassified_gallery.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- fairness
t["age group"] = pd.cut(t.age_approx, [0, 49, 69, 200], labels=["<50", "50-69", "70+"])
groups = [("sex", "sex")]
if "skin_tone_class" in t:
    groups.insert(0, ("skin_tone_class", "skin tone"))
groups.append(("age group", "age group"))
fair = []
for col, label in groups:
    for val, g in t.groupby(col, observed=True):
        if len(g) == 0:
            continue
        mal = g[g.y3 == 2]
        fair.append({"attribute": label, "group": str(val), "n lesions": len(g),
                     "n Benign / Indet. / Malig.": "/".join(str(int((g.y3 == k).sum())) for k in range(3)),
                     "macro-F1": round(deep.macro_f1(g.y3, g.pred, K), 3),
                     "macro-F1 95% CI": deep.bootstrap_ci(g.y3.values, g.pred.values, K, n=1000) if len(g) >= 10 else None,
                     "malignant recall": round(float((mal.pred == 2).mean()), 3) if len(mal) else None,
                     "accuracy": round(float(g.correct.mean()), 3),
                     "small group (< 30)": len(g) < MIN_GROUP})
fair = pd.DataFrame(fair)
fair.to_csv(OUT / "fairness.csv", index=False)
print("\nfairness:\n", fair.to_string())

fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), squeeze=False)
overall = deep.macro_f1(t.y3, t.pred, K)
for ax, (_, label) in zip(axes[0], groups):
    g = fair[fair.attribute == label]
    lo = [v - (ci[0] if ci else v) for v, ci in zip(g["macro-F1"], g["macro-F1 95% CI"])]
    hi = [(ci[1] if ci else v) - v for v, ci in zip(g["macro-F1"], g["macro-F1 95% CI"])]
    colors = ["#BBBBBB" if s else "#2E6FBA" for s in g["small group (< 30)"]]
    ax.bar(range(len(g)), g["macro-F1"], yerr=[lo, hi], color=colors, capsize=3)
    ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{a}\nn={b}" for a, b in zip(g.group, g["n lesions"])], fontsize=8)
    ax.axhline(overall, color="k", ls=":", lw=1); ax.set_ylim(0, 1); ax.set_title(f"macro-F1 by {label}" + (" (0 = very dark, 5 = very light)" if label == "skin tone" else ""), fontsize=9)
axes[0, 0].set_ylabel("test macro-F1 (grey = n < 30)")
plt.tight_layout(); fig.savefig(FIG / "m4_fairness.png", dpi=150, bbox_inches="tight"); plt.close(fig)

res = json.loads((OUT / "results.json").read_text())
res["error_analysis"] = {"n_errors": int((~t.correct).sum()), "by_dx": by_dx.reset_index().to_dict(orient="records"),
                         "errors_vs_correct": evc.reset_index().to_dict(orient="records")}
res["fairness"] = fair.to_dict(orient="records")
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
print("done ->", OUT)
