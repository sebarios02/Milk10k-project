"""Milestone 2, step 3: final evaluation of the selected configuration.

The TEST split is used here ONCE, for the configuration chosen in step 2.

1. Refit the selected pipeline (same hyper-parameters) on TRAIN + VAL.
2. Test metrics with 95 % bootstrap CIs (1,000 lesion resamples), confusion
   matrix, per-class report, ROC curves. Majority / stratified dummies on the
   same test set as the reference.
3. Interpretation (on VAL, model fitted on TRAIN): grouped permutation importance.
4. Error analysis on TEST: recall per fine diagnosis, subgroups (sex, age band).
5. Design check (on VAL): early fusion (concatenated views, the selected one)
   vs. late fusion (one model per image, probabilities averaged per lesion with
   the same rule as ``aggregate_predictions`` from Milestone 1).

Outputs: outputs/classical/test_*.csv|json, figures/m2_*.png
Run:  python scripts/11_final_evaluation.py
"""
from __future__ import annotations

import json
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_curve

from milk10k_pipeline import classical as C
from milk10k_pipeline import config

OUT = config.OUTPUT_DIR / "classical"
FEAT_DIR = config.OUTPUT_DIR / "features"


def build_pipeline(sel: dict):
    pipe, _ = C.model_zoo()[sel["model"]]
    return clone(pipe).set_params(**{f"clf__{k}": v for k, v in sel["params"].items()})


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)  # SVC(probability=True) deprecation in sklearn>=1.9
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = json.loads((OUT / "selected.json").read_text())
    les = C.add_metadata_features(pd.read_csv(FEAT_DIR / "lesion_features.csv.gz", index_col="lesion_id"))
    cols = C.feature_columns(les, sel["feature_set"])
    tr, va, te = (les[les["split"] == s] for s in ("train", "val", "test"))
    trva = pd.concat([tr, va])
    y = "y_primary"
    print(f"selected: {sel['feature_set']} + {sel['model']} {sel['params']}")

    # ---------------- 1-2. final model on test ---------------- #
    final = build_pipeline(sel).fit(trva[cols], trva[y])
    pred, proba = final.predict(te[cols]), final.predict_proba(te[cols])
    yte = te[y].to_numpy()
    ci = C.bootstrap_ci(yte, pred, proba)
    ci.to_csv(OUT / "test_metrics_ci.csv", index=False, float_format="%.4f")
    full = C.metrics(yte, pred, proba)
    ref = {}
    for name, d in C.dummy_models().items():
        d.fit(trva[cols], trva[y])
        ref[name] = C.metrics(yte, d.predict(te[cols]), d.predict_proba(te[cols]))
    summary = pd.DataFrame({"selected_model": full, **ref})
    summary.to_csv(OUT / "test_metrics.csv", float_format="%.4f")
    cm = C.confusion(yte, pred)
    cm.to_csv(OUT / "test_confusion.csv")
    pd.DataFrame({"lesion_id": te.index, "y_true": yte, "y_pred": pred,
                  **{f"p_{c}": proba[:, i] for i, c in enumerate(C.PRIMARY)}}
                 ).to_csv(OUT / "test_predictions.csv", index=False, float_format="%.4f")
    print(ci.to_string(index=False))
    print(cm)

    # confusion matrix figure (counts + row-normalised)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmv = cm.to_numpy()
    for ax, mat, title, fmt in [(axes[0], cmv, "counts", "d"),
                                (axes[1], cmv / cmv.sum(1, keepdims=True), "row-normalised (= recall)", ".2f")]:
        ax.imshow(mat, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_xticks(range(3), C.PRIMARY)
        ax.set_yticks(range(3), C.PRIMARY)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Test confusion matrix, {title}")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "m2_test_confusion.png", dpi=130)
    plt.close(fig)

    # ROC curves one-vs-rest
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    from sklearn.metrics import roc_auc_score
    for i, c in enumerate(C.PRIMARY):
        fpr, tpr, _ = roc_curve(yte == i, proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} vs rest (AUC {roc_auc_score(yte == i, proba[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("Test ROC curves (one-vs-rest)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "m2_test_roc.png", dpi=130)
    plt.close(fig)

    # ---------------- 3. grouped permutation importance (VAL) ---------------- #
    m_tr = build_pipeline(sel).fit(tr[cols], tr[y])
    imp = C.grouped_permutation_importance(m_tr, va[cols], va[y].to_numpy(), n_repeats=10)
    imp.to_csv(OUT / "val_group_importance.csv", index=False, float_format="%.4f")
    print(imp.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(imp) + 1.2))
    ax.barh(imp["group"][::-1], imp["drop_mean"][::-1], xerr=imp["drop_std"][::-1], color="#2a5d9f", capsize=3)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("drop in VAL balanced accuracy when the group is shuffled")
    ax.set_title("Which feature groups does the model use?")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "m2_group_importance.png", dpi=130)
    plt.close(fig)

    # ---------------- 4. error analysis on TEST ---------------- #
    te_ = te.assign(pred=pred, correct=(pred == yte))
    te_["pred_name"] = te_["pred"].map(dict(enumerate(C.PRIMARY)))
    by_dx = (te_.groupby("dx")
                .apply(lambda g: pd.Series({"n": len(g), "diagnosis_1": g["diagnosis_1"].mode()[0],
                                            "pct_correct": g["correct"].mean(),
                                            **{f"pred_{c}": (g["pred_name"] == c).mean() for c in C.PRIMARY}}),
                       include_groups=False)
                .sort_values("n", ascending=False))
    by_dx.to_csv(OUT / "test_by_dx.csv", float_format="%.3f")
    print(by_dx)
    te_["age_band"] = pd.cut(te_["age_approx"], [0, 49, 69, 120], labels=["<50", "50-69", "70+"])
    sub = []
    for col in ("sex", "age_band"):
        for v, g in te_.groupby(col, observed=True):
            yy, pp = g[y].to_numpy(), g["pred"].to_numpy()
            sub.append({"attribute": col, "group": v, "n": len(g),
                        "balanced_acc": balanced_accuracy_score(yy, pp),
                        "malignant_recall": (pp[yy == 2] == 2).mean() if (yy == 2).any() else np.nan,
                        "benign_recall": (pp[yy == 0] == 0).mean() if (yy == 0).any() else np.nan,
                        "pct_malignant": (yy == 2).mean()})
    sub = pd.DataFrame(sub)
    sub.to_csv(OUT / "test_subgroups.csv", index=False, float_format="%.3f")
    print(sub)

    # ---------------- 5. early vs late fusion (VAL) ---------------- #
    img = pd.read_csv(FEAT_DIR / "image_features.csv.gz", index_col="isic_id")
    split_imgs = pd.concat([pd.read_csv(config.SPLIT_DIR / f"{s}.csv") for s in ("train", "val")])
    split_imgs = C.add_metadata_features(split_imgs.join(img, on="isic_id"))
    view_cols = [c for c in img.columns]          # same feature groups, no d_/c_ prefix
    if C.META_PREFIX in C.FEATURE_SETS[sel["feature_set"]][0]:
        view_cols += [c for c in split_imgs.columns if c.startswith(C.META_PREFIX)]
    split_imgs["is_derm"] = (split_imgs["image_type"] == "dermoscopic").astype(float)
    fusion_rows = []
    for name, use_cols in [("late fusion (1 model on both views, avg per lesion)", view_cols + ["is_derm"])]:
        itr = split_imgs[split_imgs["split"] == "train"]
        iva = split_imgs[split_imgs["split"] == "val"].reset_index(drop=True)
        m = build_pipeline(sel).fit(itr[use_cols], itr[y])
        agg = C.average_views(m.predict_proba(iva[use_cols]), iva)
        yv = iva.groupby("lesion_id")[y].first().loc[agg.index].to_numpy()
        fusion_rows.append({"design": name, **C.metrics(yv, agg["pred"].to_numpy(),
                                                        agg[[f"p_{c}" for c in C.PRIMARY]].to_numpy())})
    pv, pp = m_tr.predict(va[cols]), m_tr.predict_proba(va[cols])
    fusion_rows.insert(0, {"design": "early fusion (d_ + c_ concatenated) = selected", **C.metrics(va[y].to_numpy(), pv, pp)})
    fus = pd.DataFrame(fusion_rows)
    fus.to_csv(OUT / "val_fusion_comparison.csv", index=False, float_format="%.4f")
    print(fus[["design", "balanced_acc", "macro_f1", "recall_Malignant", "recall_Indeterminate"]])

    (OUT / "final_model.json").write_text(json.dumps({**sel, "trained_on": "train+val",
                                                      "n_train_lesions": len(trva),
                                                      "n_test_lesions": len(te),
                                                      "n_features": len(cols)}, indent=2))


if __name__ == "__main__":
    main()
