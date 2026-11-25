"""Milestone 5, step 1: the 11-class model for the leaderboard (Path A).

My analysis (M2 -> M4) is on diagnosis_1 (3 classes). The leaderboard only accepts
11 classes, so the BEST M4 ARCHITECTURE (read from checkpoints/model_m4.pt) is
re-trained once with 11 outputs: labels from training_gt.csv, new last layer,
same split, same recipe. Nothing else changes.

Then the post-processing for the official metric (each probability >= 0.5 counts
as "yes", macro-F1 over the 11 classes) is chosen on VALIDATION among:
    raw         the softmax as it is
    argmax      top class -> >= 0.5, all others < 0.5 (= one diagnosis per lesion)
    thresholds  one threshold per class tuned on validation, moved to 0.5
                (validation estimate by 2-fold cross-fitting, so it is not optimistic)
Finally my own test split gives the local estimate of the official score (opened once).

Outputs: checkpoints/model_11class.pt, outputs/m5/{results.json, postprocess.json,
         per_class_f1.csv, val_probs_11class.csv, test_probs_11class.csv}, figures/m5_per_class_f1.png
"""
import _bootstrap  # noqa: F401

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from milk10k_pipeline import config
from milk10k_pipeline import deep

SMOKE = "--smoke" in sys.argv
OUT = config.OUTPUT_DIR / ("m5_smoke" if SMOKE else "m5")
OUT.mkdir(parents=True, exist_ok=True)
config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CK_M4 = deep.CHECKPOINT_DIR / ("model_m4_smoke.pt" if SMOKE else "model_m4.pt")
CK = deep.CHECKPOINT_DIR / ("model_11class_smoke.pt" if SMOKE else "model_11class.pt")
K, C = 11, deep.FINE11
T0 = time.perf_counter()


def log(*a):
    print(f"[{time.perf_counter() - T0:6.0f} s]", *a, flush=True)


arch = torch.load(CK_M4, map_location="cpu", weights_only=False)["arch"]
log("best M4 architecture:", arch)

frames = deep.load_frames()
data = deep.load_cache(frames)
X = {s: data[f"X_{s}"] for s in ("train", "val", "test")}
y = {s: torch.tensor(frames[s]["y11"].to_numpy()) for s in ("train", "val", "test")}   # 11-class labels
if SMOKE:
    X["train"], y["train"] = X["train"][:200], y["train"][:200]
yv, yt = y["val"].numpy(), y["test"].numpy()
log("train 11-class counts:", dict(zip(C, torch.bincount(y["train"], minlength=K).tolist())))

R = {"arch": arch, "smoke": SMOKE, "classes": C, "n_train": int(len(y["train"])),
     "train_counts": dict(zip(C, torch.bincount(y["train"], minlength=K).tolist())),
     "val_counts": dict(zip(C, np.bincount(yv, minlength=K).tolist())),
     "test_counts": dict(zip(C, np.bincount(yt, minlength=K).tolist()))}

# ----------------------------------------------------------------------------- train (same recipe as M4)
if arch.endswith("_frozen"):
    name = arch[: -len("_frozen")]
    F = {s: deep.embed(name, X[s], s) for s in ("train", "val", "test")}      # cached: seconds
    head = deep.train_head(F["train"], y["train"], K)
    model = deep.build_model(arch, K)
    model.head.load_state_dict(head.state_dict())
    Pv, Pt = deep.head_proba(head, F["val"]), deep.head_proba(head, F["test"])
    R["recipe"] = "frozen backbone + nn.Linear, class-weighted CE, Adam 1e-3, 25 epochs (Session 9)"
else:
    torch.manual_seed(0)
    model = deep.build_model("resnet18_ft", K)                                 # model.fc = nn.Linear(512, 11)
    hist, best = deep.fit(model, model.parameters(), X["train"], y["train"], X["val"], y["val"], K,
                          epochs=1 if SMOKE else 6, lr=1e-4, log=log)
    hist.to_csv(OUT / "training_history.csv", index=False)
    R["recipe"] = "ResNet-18 all layers, class-weighted CE, Adam 1e-4, 6 epochs, flips, best epoch on val (Session 7)"
    R["best_epoch"], R["val_f1_per_epoch"] = int(best["epoch"]), [round(v, 4) for v in hist["val_f1"]]
    Pv = deep.predict_proba(model, X["val"])

R["val_macro_f1_argmax"] = round(deep.macro_f1(yv, Pv.argmax(1), K), 4)
log("validation macro-F1 (argmax, 11 classes):", R["val_macro_f1_argmax"])

# ----------------------------------------------------------------------------- post-processing, chosen on validation
thr = deep.tune_thresholds(yv, Pv)
val_scores = {"raw": deep.leaderboard_f1(yv, Pv),
              "argmax": deep.leaderboard_f1(yv, deep.rescale_argmax(Pv)),
              "thresholds": deep.crossfit_threshold_score(yv, Pv)}
strategy = max(val_scores, key=val_scores.get)
R["val_leaderboard_metric"] = {k: round(v, 4) for k, v in val_scores.items()}
R["postprocess"] = {"strategy": strategy, "thresholds": dict(zip(C, map(float, thr)))}
log("validation, official metric (threshold 0.5):", R["val_leaderboard_metric"], "-> chosen:", strategy)
(OUT / "postprocess.json").write_text(json.dumps({"strategy": strategy, "classes": C,
                                                  "thresholds": [float(v) for v in thr]}, indent=1))

deep.save_checkpoint(model, arch, C, CK, {"val_macro_f1_argmax": R["val_macro_f1_argmax"], "postprocess": strategy})
log("saved", CK.name)

# ----------------------------------------------------------------------------- my test split, once: local estimate
if not arch.endswith("_frozen"):
    Pt = deep.predict_proba(model, X["test"])
Pt_pp = deep.postprocess(Pt, strategy, thr)
score, f1s = deep.leaderboard_f1(yt, Pt_pp, per_class=True)
R["test_local_estimate"] = {"official_metric_with_postprocess": round(score, 4),
                            "official_metric_raw_softmax": round(deep.leaderboard_f1(yt, Pt), 4),
                            "macro_f1_argmax": round(deep.macro_f1(yt, Pt.argmax(1), K), 4)}
rng = np.random.default_rng(0)
boots = [deep.leaderboard_f1(yt[i], Pt_pp[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(1000))]
R["test_local_estimate"]["ci95"] = [round(float(v), 4) for v in np.percentile(boots, [2.5, 97.5])]
_, f1s_val = deep.leaderboard_f1(yv, deep.postprocess(Pv, strategy, thr), per_class=True)
pc = pd.DataFrame({"n train": R["train_counts"], "n val": R["val_counts"], "n test": R["test_counts"],
                   "F1 val": dict(zip(C, f1s_val.round(3))), "F1 test": dict(zip(C, f1s.round(3)))})
pc.index.name = "class"
pc.to_csv(OUT / "per_class_f1.csv")
log("TEST (my split) local estimate:", R["test_local_estimate"])
log("per class:\n" + pc.to_string())

fig, ax = plt.subplots(figsize=(9, 3.8))
xx = np.arange(K)
ax.bar(xx - 0.2, pc["F1 val"], 0.4, label="validation", color="#2E6FBA")
ax.bar(xx + 0.2, pc["F1 test"], 0.4, label="test (my split)", color="#F08200")
ax.set_xticks(xx); ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(C, pc["n test"])], fontsize=7.5)
ax.set_ylim(0, 1); ax.set_ylabel("F1 (threshold 0.5)"); ax.legend(fontsize=8)
ax.set_title(f"11-class model ({arch}), post-processing '{strategy}': F1 per class", fontsize=10)
plt.tight_layout(); fig.savefig(config.FIGURE_DIR / "m5_per_class_f1.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def table(df, P):
    t = df[["lesion_id", "isic_id", "dx", "y11"]].copy()
    for i, c in enumerate(C):
        t[c] = P[:, i].round(5)
    return t

table(frames["val"], Pv).to_csv(OUT / "val_probs_11class.csv", index=False)
table(frames["test"], Pt).to_csv(OUT / "test_probs_11class.csv", index=False)
R["minutes_total"] = round((time.perf_counter() - T0) / 60, 1)
(OUT / "results.json").write_text(json.dumps(R, indent=1))
log("done ->", OUT)
