"""Milestone 2, step 2: compare feature sets x models and pick the best configuration.

Protocol (the TEST split is not touched here)
--------------------------------------------
1. For every (feature set, model): hyper-parameters tuned with 5-fold stratified
   CV on TRAIN (one row = one lesion, so folds never share a lesion);
   scoring = balanced accuracy.
2. The best hyper-parameters are refitted on all of TRAIN and scored on VAL.
3. The configuration with the highest VAL balanced accuracy is selected
   (ties -> higher macro-F1). Its name is written to outputs/classical/selected.json.

Outputs: outputs/classical/model_selection.csv, outputs/classical/selected.json,
         figures/m2_selection_heatmap.png, figures/m2_feature_set_comparison.png

Run:  python scripts/10_model_selection.py [--target y_primary] [--n-jobs 2]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from milk10k_pipeline import classical as C
from milk10k_pipeline import config

OUT = config.OUTPUT_DIR / "classical"
FEATS = config.OUTPUT_DIR / "features" / "lesion_features.csv.gz"


def load_lesions() -> pd.DataFrame:
    les = pd.read_csv(FEATS, index_col="lesion_id")
    return C.add_metadata_features(les)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="y_primary", choices=["y_primary", "y_fine"])
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--sets", nargs="*", default=list(C.FEATURE_SETS))
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)  # SVC(probability=True) deprecation in sklearn>=1.9
    OUT.mkdir(parents=True, exist_ok=True)

    les = load_lesions()
    tr, va = les[les["split"] == "train"], les[les["split"] == "val"]
    ytr, yva = tr[args.target].to_numpy(), va[args.target].to_numpy()
    classes = C.PRIMARY if args.target == "y_primary" else \
        json.loads(config.LABEL_MAP_JSON.read_text())["fine"]["classes"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    print(f"train {len(tr)} lesions, val {len(va)} lesions, target {args.target}")

    zoo = C.model_zoo()
    if args.models:
        zoo = {k: v for k, v in zoo.items() if k in args.models}
    rows = []
    for set_name in args.sets:
        cols = C.feature_columns(les, set_name)
        Xtr, Xva = tr[cols], va[cols]
        jobs = [(n, p, {}) for n, p in C.dummy_models().items()] if set_name == args.sets[0] else []
        jobs += [(n, p, g) for n, (p, g) in zoo.items()]
        for model_name, pipe, grid in jobs:
            t0 = time.time()
            gs = GridSearchCV(pipe, grid or {"clf__strategy": [pipe.get_params()["clf__strategy"]]},
                              scoring="balanced_accuracy", cv=cv, n_jobs=args.n_jobs, refit=True)
            gs.fit(Xtr, ytr)
            best = gs.best_estimator_
            pred = best.predict(Xva)
            proba = best.predict_proba(Xva)
            m = C.metrics(yva, pred, proba, classes)
            i = gs.best_index_
            row = {"feature_set": set_name if "Dummy" not in model_name else "-",
                   "n_features": len(cols) if "Dummy" not in model_name else 0,
                   "model": model_name,
                   "best_params": json.dumps({k.replace("clf__", ""): v for k, v in gs.best_params_.items()}),
                   "cv_bal_acc_mean": gs.cv_results_["mean_test_score"][i],
                   "cv_bal_acc_std": gs.cv_results_["std_test_score"][i],
                   **{f"val_{k}": v for k, v in m.items()},
                   "fit_seconds": time.time() - t0}
            rows.append(row)
            print(f"{row['feature_set']:>20s} {model_name:>16s}  CV {row['cv_bal_acc_mean']:.3f}"
                  f"±{row['cv_bal_acc_std']:.3f}  VAL bal_acc {m['balanced_acc']:.3f} "
                  f"macroF1 {m['macro_f1']:.3f}  ({row['fit_seconds']:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    suffix = "" if args.target == "y_primary" else "_fine"
    res.to_csv(OUT / f"model_selection{suffix}.csv", index=False, float_format="%.4f")

    cand = res[~res["model"].str.startswith("Dummy")]
    best = cand.sort_values(["val_balanced_acc", "val_macro_f1"], ascending=False).iloc[0]
    sel = {"target": args.target, "feature_set": best["feature_set"], "model": best["model"],
           "params": json.loads(best["best_params"]),
           "val_balanced_acc": float(best["val_balanced_acc"]),
           "val_macro_f1": float(best["val_macro_f1"]),
           "cv_bal_acc_mean": float(best["cv_bal_acc_mean"]),
           "rule": "highest VAL balanced accuracy (ties -> macro-F1); hyper-parameters from 5-fold CV on TRAIN"}
    (OUT / f"selected{suffix}.json").write_text(json.dumps(sel, indent=2))
    print("\nSELECTED:", json.dumps(sel, indent=2))

    if args.target == "y_primary":
        plot(res)


SHORT = {"F1_hist_derm": "F1 colour histograms (derm)", "F2_colour_derm": "F2 lesion/skin colour stats (derm)",
         "F3_texture_derm": "F3 edge/texture filters (derm)", "F4_shape_derm": "F4 mask shape (derm)",
         "F5_all_derm": "F5 = F1-F4 (derm)", "F6_all_2views": "F6 = F1-F4 (derm + clinical)",
         "F7_all_2views_meta": "F7 = F6 + metadata", "F0_meta_only": "F0 metadata only"}


def plot(res: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cand = res[~res["model"].str.startswith("Dummy")]
    piv = cand.pivot(index="feature_set", columns="model", values="val_balanced_acc")
    order_sets = [s for s in C.FEATURE_SETS if s in piv.index]
    order_models = [m for m in ["LogReg", "kNN", "SVM_RBF", "RandomForest", "HistGB"] if m in piv.columns]
    piv = piv.loc[order_sets, order_models]
    dummy = res.loc[res["model"] == "Dummy_majority", "val_balanced_acc"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=dummy, vmax=float(np.nanmax(piv.to_numpy())), aspect="auto")
    ax.set_xticks(range(len(order_models)), order_models, rotation=20)
    ax.set_yticks(range(len(order_sets)), [SHORT.get(s, s) for s in order_sets],
                  fontsize=8)
    best = np.unravel_index(np.nanargmax(piv.to_numpy()), piv.shape)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > (dummy + piv.to_numpy().max()) / 2 else "black",
                    fontweight="bold" if (i, j) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1, fill=False, ec="red", lw=2))
    fig.colorbar(im, ax=ax, label="validation balanced accuracy")
    ax.set_title(f"Model selection on VAL: balanced accuracy (majority dummy = {dummy:.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "m2_selection_heatmap.png", dpi=130)
    plt.close(fig)

    # best model per feature set, with the CV mean +- std on train
    best_per_set = cand.loc[cand.groupby("feature_set")["val_balanced_acc"].idxmax()]
    best_per_set = best_per_set.set_index("feature_set").loc[order_sets]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(best_per_set))
    ax.bar(x - 0.2, best_per_set["cv_bal_acc_mean"], 0.4, yerr=best_per_set["cv_bal_acc_std"],
           capsize=3, label="5-fold CV on train (mean ± std)", color="#9db9d9")
    ax.bar(x + 0.2, best_per_set["val_balanced_acc"], 0.4, label="validation", color="#2a5d9f")
    ax.axhline(dummy, color="grey", ls="--", lw=1, label=f"majority dummy = chance ({dummy:.2f})")
    ax.set_xticks(x, [f"{s}\n{m}" for s, m in zip(best_per_set.index, best_per_set["model"])], fontsize=7.5)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Best model per feature set")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "m2_feature_set_comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
