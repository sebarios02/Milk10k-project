"""Milestone 4, step 1: best model, chosen fairly on validation, test opened once.

1. Fair comparison (Session 9, now with ALL 3,742 training lesions = FULL = True):
   ResNet-18 vs ViT-B/16, both frozen, same linear head, same split, same metric.
   Paired bootstrap on validation for ViT - ResNet.
2. Candidates for "best model": the two frozen FULL models + every Session 7 model
   (fine-tuned ResNet-18 variants, validation numbers from outputs/session_07/results.json).
   The winner is the highest VALIDATION macro-F1.
3. The winner is (re)trained with exactly its recipe and saved as checkpoints/model_m4.pt
   (Session 7 did not save weights). Its test macro-F1 is reported once.
4. Writes the M2 -> M4 scoreboard and the 3-class test predictions.

Outputs: outputs/m4/{results.json, scoreboard.csv, val_probs_3class.csv, test_predictions_3class.csv},
         figures/m4_scoreboard.png, figures/m4_best_test_confusion.png, checkpoints/model_m4.pt

    python scripts/13_m4_best_model.py            # full run (~1 h on CPU: ViT features + ResNet fine-tuning)
    python scripts/13_m4_best_model.py --smoke    # 2-minute check of the code path on 200 lesions
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
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from milk10k_pipeline import config
from milk10k_pipeline import deep

SMOKE = "--smoke" in sys.argv
OUT = config.OUTPUT_DIR / ("m4_smoke" if SMOKE else "m4")
OUT.mkdir(parents=True, exist_ok=True)
FIG = config.FIGURE_DIR
FIG.mkdir(parents=True, exist_ok=True)
CK = deep.CHECKPOINT_DIR / ("model_m4_smoke.pt" if SMOKE else "model_m4.pt")
K = 3
T0 = time.perf_counter()


def log(*a):
    print(f"[{time.perf_counter() - T0:6.0f} s]", *a, flush=True)


log(deep.device, torch.__version__, "| threads:", torch.get_num_threads(), "| smoke:", SMOKE)
log(config.describe())

# ----------------------------------------------------------------------------- data
frames = deep.load_frames()
data = deep.load_cache(frames)
X = {s: data[f"X_{s}"] for s in ("train", "val", "test")}
y = {s: data[f"y_{s}"] for s in ("train", "val", "test")}
if SMOKE:
    keep = torch.arange(200)
    X["train"], y["train"] = X["train"][keep], y["train"][keep]
log("train", tuple(X["train"].shape), "val", tuple(X["val"].shape), "test", tuple(X["test"].shape))
yv, yt = y["val"].numpy(), y["test"].numpy()

R = {"smoke": SMOKE, "n_train": int(len(y["train"])), "classes": deep.PRIMARY}

# ----------------------------------------------------------------------------- 1. frozen ResNet-18 vs ViT-B/16
log("1. frozen features (cached in", deep.CACHE_DIR, ")")
embed_s, frozen = {}, {}
for name in ("resnet18", "vit_b_16"):
    F = {s: deep.embed(name, X[s], s, embed_s) for s in ("train", "val", "test")}
    head = deep.train_head(F["train"], y["train"], K)
    Pv, Pt = deep.head_proba(head, F["val"]), deep.head_proba(head, F["test"])
    frozen[name] = {"head": head, "Pv": Pv, "Pt": Pt,
                    "val_f1": deep.macro_f1(yv, Pv.argmax(1), K), "test_f1": deep.macro_f1(yt, Pt.argmax(1), K)}
    log(f"   {name:9s} frozen + linear head: val macro-F1 {frozen[name]['val_f1']:.3f}")

ci, p_vit = deep.paired_bootstrap(yv, frozen["resnet18"]["Pv"].argmax(1), frozen["vit_b_16"]["Pv"].argmax(1), K)
R["frozen_comparison"] = {
    "val_f1": {n: round(f["val_f1"], 4) for n, f in frozen.items()},
    "val_delta_vit_minus_resnet": round(frozen["vit_b_16"]["val_f1"] - frozen["resnet18"]["val_f1"], 4),
    "val_delta_ci95": ci, "p_vit_better": p_vit, "embed_seconds_new": embed_s,
    "backbone_params": {"resnet18": 11176512, "vit_b_16": 85798656},
    "trainable_params": {n: int(sum(p.numel() for p in f["head"].parameters())) for n, f in frozen.items()},
}
log("   ViT - ResNet (val):", R["frozen_comparison"]["val_delta_vit_minus_resnet"], "CI", ci, "P(ViT better)", p_vit)

# ----------------------------------------------------------------------------- 2. candidates, chosen on validation
s7 = json.loads((config.OUTPUT_DIR / "session_07" / "results.json").read_text())
S7_RECIPES = {  # key -> (label, which parameters train)
    "full": ("ResNet-18 fully fine-tuned (all layers)", None),
    "layer4": ("ResNet-18 fine-tuned: layer4 + fc", ("layer4", "fc")),
    "layer3_4": ("ResNet-18 fine-tuned: layer3 + layer4 + fc", ("layer3", "layer4", "fc")),
}
cands = {f"S9 FULL - {n} frozen + linear head": {"kind": "frozen", "name": n, "val_f1": f["val_f1"]}
         for n, f in frozen.items()}
for k, (label, _) in S7_RECIPES.items():
    cands[f"S7 - {label}"] = {"kind": "s7", "key": k, "val_f1": s7[k]["best_val_f1"]}
winner = max(cands, key=lambda c: cands[c]["val_f1"])
R["candidates_val_f1"] = {c: round(v["val_f1"], 4) for c, v in cands.items()}
R["winner_on_val"] = winner
log("2. candidates (val macro-F1):", R["candidates_val_f1"])
log("   winner on validation:", winner)

# ----------------------------------------------------------------------------- 3. winner -> checkpoint, test once
w = cands[winner]
if w["kind"] == "frozen":
    arch = f"{w['name']}_frozen"
    model = deep.build_model(arch, K)
    model.head.load_state_dict(frozen[w["name"]]["head"].state_dict())
    hist = None
    Pv_best = frozen[w["name"]]["Pv"]
else:
    arch = "resnet18_ft"
    label, prefixes = S7_RECIPES[w["key"]]
    log(f"3. re-training the winner with its Session 7 recipe ({label}, 6 epochs, lr 1e-4, flips)")
    torch.manual_seed(0)                                   # same seed and order as my Session 7 notebook
    model = deep.build_model(arch, K)
    if prefixes:
        for n_, p in model.named_parameters():
            p.requires_grad = n_.startswith(prefixes)
    params = [p for p in model.parameters() if p.requires_grad]
    hist, best = deep.fit(model, params, X["train"], y["train"], X["val"], y["val"], K,
                          epochs=1 if SMOKE else 6, lr=1e-4, log=log)
    Pv_best = deep.predict_proba(model, X["val"])
    hist.to_csv(OUT / "winner_training_history.csv", index=False)
    R["winner_retrain"] = {"label": label, "best_epoch": int(best["epoch"]), "val_f1_per_epoch":
                           [round(v, 4) for v in hist["val_f1"]],
                           "session7_val_f1": s7[w["key"]]["best_val_f1"],
                           "sec_per_epoch": round(float(hist["epoch_s"].mean()), 1)}

val_f1_final = deep.macro_f1(yv, Pv_best.argmax(1), K)
deep.save_checkpoint(model, arch, deep.PRIMARY, CK, {"val_macro_f1": round(val_f1_final, 4), "winner": winner})
log(f"   saved {CK.name}  (arch {arch}, val macro-F1 {val_f1_final:.3f})")

# runner-up vs winner on validation (is the choice clear or within noise?)
runner_up = sorted(cands, key=lambda c: cands[c]["val_f1"])[-2]
ru = cands[runner_up]
ru_pred = frozen[ru["name"]]["Pv"].argmax(1) if ru["kind"] == "frozen" else np.array(s7[ru["key"]]["val_pred"])
ci_w, p_w = deep.paired_bootstrap(yv, ru_pred, Pv_best.argmax(1), K)
R["winner_vs_runner_up_val"] = {"runner_up": runner_up, "delta_ci95": ci_w, "p_winner_better": p_w}

# THE TEST SET, ONCE, for the final model
log("   opening the test set for the final model")
Pt_best = deep.predict_proba(model, X["test"]) if w["kind"] != "frozen" else frozen[w["name"]]["Pt"]
pt = Pt_best.argmax(1)
R["test"] = {"model": winner, "arch": arch, "macro_f1": round(deep.macro_f1(yt, pt, K), 4),
             "ci95": deep.bootstrap_ci(yt, pt, K), "accuracy": round(float(accuracy_score(yt, pt)), 4),
             "malignant_recall": round(float((pt[yt == 2] == 2).mean()), 4),
             "confusion": confusion_matrix(yt, pt, labels=[0, 1, 2]).tolist()}
log("   TEST:", R["test"])

fig, ax = plt.subplots(figsize=(4.8, 4.3))
ConfusionMatrixDisplay.from_predictions(yt, pt, display_labels=deep.PRIMARY, cmap="Oranges", ax=ax, colorbar=False)
ax.set_title(f"Final M4 model: {arch}\nTEST ({len(yt)} lesions), macro-F1 {R['test']['macro_f1']:.3f}", fontsize=9)
fig.savefig(FIG / "m4_best_test_confusion.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- 4. scoreboard M2 -> M4
sb = pd.read_csv(config.OUTPUT_DIR / "session_07" / "scoreboard.csv", index_col=0)[["val macro-F1", "test macro-F1"]]
for n, f in frozen.items():
    nice = {"resnet18": "ResNet-18", "vit_b_16": "ViT-B/16"}[n]
    sb.loc[f"M4 (S9, all 3,742 lesions) - {nice} frozen + linear head"] = [round(f["val_f1"], 3), round(f["test_f1"], 3)]
sb.loc[f"M4 FINAL = best on val ({arch}, re-trained, saved)"] = [round(val_f1_final, 3), R["test"]["macro_f1"]]
sb["milestone"] = ["M2", "M2", "M3", "M3", "M3", "M3", "M3", "M3", "M3"][:len(sb) - 3] + ["M4", "M4", "M4 final"]
sb.index.name = "model"
sb.to_csv(OUT / "scoreboard.csv")
R["scoreboard"] = sb.reset_index().to_dict(orient="records")
log("4. scoreboard\n" + sb.to_string())

fig, ax = plt.subplots(figsize=(10, 5.2))
s = sb.iloc[::-1]
yy = np.arange(len(s))
ax.barh(yy + 0.2, s["val macro-F1"], height=0.4, color="#2E6FBA", label="validation (749 lesions)")
ax.barh(yy - 0.2, s["test macro-F1"], height=0.4, color="#F08200", label="test (749 lesions, same for all)")
for y_, v, t in zip(yy, s["val macro-F1"], s["test macro-F1"]):
    ax.text(v + 0.003, y_ + 0.2, f"{v:.3f}", va="center", fontsize=7)
    ax.text(t + 0.003, y_ - 0.2, f"{t:.3f}", va="center", fontsize=7)
ax.set_yticks(yy); ax.set_yticklabels(s.index, fontsize=7.5)
ax.set_xlim(0.3, 0.8); ax.set_xlabel("macro-F1 (3 classes: Benign / Indeterminate / Malignant)")
ax.legend(loc="upper right", fontsize=8); ax.set_title("Scoreboard M2 -> M4: same split, same target, same metric")
plt.tight_layout(); fig.savefig(FIG / "m4_scoreboard.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- save predictions + results
def pred_table(df, P):
    t = df[["lesion_id", "isic_id", "diagnosis_1", "dx", "y3", "sex", "age_approx", "anatom_site_general"]
           + [c for c in ("skin_tone_class", "site") if c in df]].copy()
    t["pred"] = P.argmax(1)
    t["pred_label"] = np.array(deep.PRIMARY)[t["pred"]]
    for i, c in enumerate(deep.PRIMARY):
        t[f"p_{c}"] = P[:, i].round(5)
    return t

pred_table(frames["val"], Pv_best).to_csv(OUT / "val_probs_3class.csv", index=False)
pred_table(frames["test"], Pt_best).to_csv(OUT / "test_predictions_3class.csv", index=False)
R["minutes_total"] = round((time.perf_counter() - T0) / 60, 1)
(OUT / "results.json").write_text(json.dumps(R, indent=1))
log("done ->", OUT)
