"""Milestone 2, stretch: the selected configuration on the 8-class fine target.

Same feature set and model family as the primary baseline; hyper-parameters
re-tuned with 5-fold CV on TRAIN for ``y_fine``; refit on TRAIN + VAL; scored
once on TEST. Outputs: outputs/classical/fine_*.csv, figures/m2_fine_confusion.png
"""
from __future__ import annotations

import json
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from milk10k_pipeline import classical as C
from milk10k_pipeline import config

OUT = config.OUTPUT_DIR / "classical"


def main() -> None:
    warnings.filterwarnings("ignore")  # sklearn UserWarnings on tiny classes, SVC FutureWarning
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = json.loads((OUT / "selected.json").read_text())
    classes = json.loads(config.LABEL_MAP_JSON.read_text())["fine"]["classes"]
    les = C.add_metadata_features(pd.read_csv(config.OUTPUT_DIR / "features" / "lesion_features.csv.gz",
                                              index_col="lesion_id"))
    cols = C.feature_columns(les, sel["feature_set"])
    tr, va, te = (les[les["split"] == s] for s in ("train", "val", "test"))
    pipe, grid = C.model_zoo()[sel["model"]]
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)   # OTHER_MALIGNANT has 7 train lesions -> 5 folds ok
    gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=cv, n_jobs=2).fit(tr[cols], tr["y_fine"])
    m_val = C.metrics(va["y_fine"].to_numpy(), gs.predict(va[cols]), gs.predict_proba(va[cols]), classes)

    final = clone(gs.best_estimator_).fit(pd.concat([tr, va])[cols], pd.concat([tr, va])["y_fine"])
    yte = te["y_fine"].to_numpy()
    pred, proba = final.predict(te[cols]), final.predict_proba(te[cols])
    m_te = C.metrics(yte, pred, proba, classes)
    ci = C.bootstrap_ci(yte, pred, proba, keys=("balanced_acc", "macro_f1", "macro_auc_ovr"), classes=classes)
    pd.DataFrame({"val": m_val, "test": m_te}).to_csv(OUT / "fine_metrics.csv", float_format="%.4f")
    ci.to_csv(OUT / "fine_test_metrics_ci.csv", index=False, float_format="%.4f")
    cm = C.confusion(yte, pred, classes)
    cm.to_csv(OUT / "fine_test_confusion.csv")
    print("params", gs.best_params_)
    print(ci.to_string(index=False))
    print(cm)

    mat = cm.to_numpy() / np.maximum(cm.to_numpy().sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{mat[i, j]:.2f}\n({cm.iat[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > .5 else "black")
    short = [c.replace("OTHER_", "OTH_") for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=45)
    ax.set_yticks(range(len(classes)), short)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"8-class test confusion (row-normalised), bal. acc {m_te['balanced_acc']:.3f}")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "m2_fine_confusion.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
