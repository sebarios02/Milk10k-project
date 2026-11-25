"""Milestone 5, improvement: late-fusion ensemble for the 11-class leaderboard model.

After the first official submission (single ResNet-18, 0.420), the model is improved WITHOUT
looking at the leaderboard: every choice below is made on my validation split, and my test
split is opened once, for the chosen configuration only.

Members (milk10k_pipeline/ensemble.py): fine-tuned ResNet-18 on the dermoscopic image (with and
without flip TTA), frozen ViT-B/16 + linear head on the dermoscopic image, the same on the
CLINICAL close-up, and a logistic regression on the official metadata.

Candidates compared on validation (official metric with the argmax post-processing = macro-F1):
    each member alone, equal weights, and weights searched on a 0.1 simplex grid
    (the grid search is scored by 2-fold cross-fitting on validation, so it is not optimistic).

Outputs: outputs/m5_ensemble/{results.json, candidates.csv, per_class_f1.csv, val_probs.csv, test_probs.csv},
         checkpoints/ensemble_11class.pt (heads + weights + metadata model), figures/m5_ensemble.png

    python scripts/16_m5_ensemble.py         (~35-45 min on CPU, mostly ViT on the clinical images)
"""
import _bootstrap  # noqa: F401

import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from milk10k_pipeline import config, deep
from milk10k_pipeline import ensemble as ens

OUT = config.OUTPUT_DIR / "m5_ensemble"
OUT.mkdir(parents=True, exist_ok=True)
config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
K, C = 11, deep.FINE11
T0 = time.perf_counter()


def log(*a):
    print(f"[{time.perf_counter() - T0:6.0f} s]", *a, flush=True)


frames = deep.load_frames()
S = ("train", "val", "test")
y = {s: frames[s]["y11"].to_numpy() for s in S}
derm = deep.load_cache(frames)
log("dermoscopic cache:", {s: tuple(derm[f"X_{s}"].shape) for s in S})

# ----------------------------------------------------------------------------- member 1: fine-tuned ResNet-18
model, ck = deep.load_checkpoint(deep.CHECKPOINT_DIR / "model_11class.pt")
assert ck["classes"] == C
P = {"resnet_plain": {}, "resnet_derm": {}}
for s in ("val", "test"):
    P["resnet_plain"][s] = deep.predict_proba(model, derm[f"X_{s}"])
    P["resnet_derm"][s] = ens.predict_tta(model, derm[f"X_{s}"])
log("ResNet-18 val macro-F1: plain", round(deep.macro_f1(y["val"], P["resnet_plain"]["val"].argmax(1), K), 4),
    "| flip TTA", round(deep.macro_f1(y["val"], P["resnet_derm"]["val"].argmax(1), K), 4))

# ----------------------------------------------------------------------------- members 2-3: frozen ViT heads
clin = ens.load_clinical_cache(frames)
heads = {}
for member, X_, tag in (("vit_derm", derm, ""), ("vit_clin", clin, "clinical_")):
    F = {s: deep.embed("vit_b_16", X_[f"X_{s}"], f"{tag}{s}") for s in S}
    head = deep.train_head(F["train"], torch.tensor(y["train"]), K)
    heads[member] = head.state_dict()
    P[member] = {s: deep.head_proba(head, F[s]) for s in ("val", "test")}
    log(f"{member}: val macro-F1 {deep.macro_f1(y['val'], P[member]['val'].argmax(1), K):.4f}")

# ----------------------------------------------------------------------------- member 4: metadata
meta_table = ens.model_input_table()
Xm = {s: ens.meta_features(meta_table, frames[s]["lesion_id"]) for s in S}
meta_model = ens.fit_meta_model(Xm["train"], y["train"])
P["meta"] = {s: meta_model.predict_proba(Xm[s]) for s in ("val", "test")}
log(f"meta: val macro-F1 {deep.macro_f1(y['val'], P['meta']['val'].argmax(1), K):.4f}")

# ----------------------------------------------------------------------------- candidates, on validation only
yv = y["val"]
cand = {}
for m in ["resnet_plain"] + ens.MEMBERS:
    cand[f"{m} alone"] = {"weights": {m: 1.0}, "val": deep.macro_f1(yv, P[m]["val"].argmax(1), K)}
Pv = [P[m]["val"] for m in ens.MEMBERS]
eq = np.full(len(ens.MEMBERS), 1 / len(ens.MEMBERS))
cand["equal weights (4 members)"] = {"weights": dict(zip(ens.MEMBERS, eq)),
                                     "val": deep.macro_f1(yv, ens.combine(Pv, eq).argmax(1), K)}
grid = ens.simplex_grid(len(ens.MEMBERS), 0.1)
w_best, f_in = ens.best_weights(yv, Pv, grid)
f_cf = ens.crossfit_weight_search(yv, Pv, grid)
cand["weights searched on val (0.1 grid)"] = {"weights": dict(zip(ens.MEMBERS, w_best)), "val": f_cf,
                                              "val_in_sample": f_in}
log("candidates (val; grid search = cross-fitted estimate):")
for k, v in cand.items():
    log(f"   {k:40s} {v['val']:.4f}  weights {({m: round(float(w), 2) for m, w in v['weights'].items()})}")

choice = max(cand, key=lambda c: cand[c]["val"])
weights = {m: float(cand[choice]["weights"].get(m, 0.0)) for m in ["resnet_plain"] + ens.MEMBERS}
log("CHOSEN on validation:", choice, {m: round(w, 2) for m, w in weights.items() if w > 0})

def fused(split):
    return sum(w * P[m][split] for m, w in weights.items() if w > 0)

# ----------------------------------------------------------------------------- my test split, once
Pt = fused("test")
pt = Pt.argmax(1)
f_test = deep.macro_f1(y["test"], pt, K)
ci = deep.bootstrap_ci(y["test"], pt, K, n=1000)
f_test_single = deep.macro_f1(y["test"], P["resnet_plain"]["test"].argmax(1), K)
log(f"TEST (my split): chosen {f_test:.4f} {ci}  | single ResNet-18 (first submission) {f_test_single:.4f}")

_, f1v = deep.leaderboard_f1(yv, deep.rescale_argmax(fused("val")), per_class=True)
_, f1t = deep.leaderboard_f1(y["test"], deep.rescale_argmax(Pt), per_class=True)
_, f1t0 = deep.leaderboard_f1(y["test"], deep.rescale_argmax(P["resnet_plain"]["test"]), per_class=True)
pc = pd.DataFrame({"n val": np.bincount(yv, minlength=K), "n test": np.bincount(y["test"], minlength=K),
                   "F1 val (ensemble)": f1v.round(3), "F1 test (ensemble)": f1t.round(3),
                   "F1 test (single ResNet)": f1t0.round(3)}, index=C)
pc.index.name = "class"
pc.to_csv(OUT / "per_class_f1.csv")
log("per class:\n" + pc.to_string())

# ----------------------------------------------------------------------------- save
torch.save({"classes": C, "weights": weights, "resnet_checkpoint": "model_11class.pt",
            "vit_heads": heads, "tta": weights.get("resnet_derm", 0) > 0,
            "meta_model": pickle.dumps(meta_model), "meta_columns": "milk10k_pipeline.ensemble.meta_features",
            "postprocess": "argmax", "choice": choice},
           deep.CHECKPOINT_DIR / "ensemble_11class.pt")
pd.DataFrame([{"candidate": k, "val macro-F1": round(v["val"], 4),
               **{f"w_{m}": round(float(v["weights"].get(m, 0)), 2) for m in ["resnet_plain"] + ens.MEMBERS}}
              for k, v in cand.items()]).to_csv(OUT / "candidates.csv", index=False)
for split, Pf in (("val", fused("val")), ("test", Pt)):
    t = frames[split][["lesion_id", "dx", "y11"]].copy()
    t[C] = Pf.round(5)
    t.to_csv(OUT / f"{split}_probs.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 3.6))
xx = np.arange(K)
ax.bar(xx - 0.2, pc["F1 test (single ResNet)"], 0.4, color="#BBBBBB", label="single ResNet-18 (submission 1)")
ax.bar(xx + 0.2, pc["F1 test (ensemble)"], 0.4, color="#2E6FBA", label="ensemble (submission 2)")
ax.set_xticks(xx); ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(C, pc["n test"])], fontsize=7.5)
ax.set_ylim(0, 1); ax.set_ylabel("F1 on my test split"); ax.legend(fontsize=8)
ax.set_title(f"11-class: single ResNet-18 {f_test_single:.3f} vs ensemble {f_test:.3f} (macro-F1, my test split)", fontsize=10)
plt.tight_layout(); fig.savefig(config.FIGURE_DIR / "m5_ensemble.png", dpi=150, bbox_inches="tight"); plt.close(fig)

R = {"candidates_val": {k: {kk: (round(float(vv), 4) if not isinstance(vv, dict) else
                                {m: round(float(w), 2) for m, w in vv.items()}) for kk, vv in v.items()}
                        for k, v in cand.items()},
     "choice": choice, "weights": {m: round(w, 2) for m, w in weights.items()},
     "test_my_split": {"ensemble": round(f_test, 4), "ci95": ci, "single_resnet": round(f_test_single, 4)},
     "minutes": round((time.perf_counter() - T0) / 60, 1)}
(OUT / "results.json").write_text(json.dumps(R, indent=1))
log("done ->", OUT, "| next: python m5_final/make_submission_ensemble.py")
