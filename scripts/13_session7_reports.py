"""
Session 7 deliverables -> two PDFs for the Aula, built ONLY from the files the notebook saved.

    python scripts/13_session7_reports.py

Reads   outputs/session_07/results.json, outputs/session_07/scoreboard.csv,
        figures/session7_mini_challenge.png, figures/session7_scoreboard.png,
        figures/session7_best_test_confusion.png
Writes  reports/S7_MiniChallenge_Rios_Saucedo.pdf   (slide 29)
        reports/S7_Homework_Rios_Saucedo.pdf        (slide 30)

Only matplotlib is needed (already in the venv). Every number and every conclusion in the
PDFs is computed from the saved results, so the reports always match the notebook run.
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from milk10k_pipeline import config  # noqa: E402

OUT = config.OUTPUT_DIR / "session_07"
FIG = config.FIGURE_DIR
REPORTS = config.REPORT_DIR

AUTHOR = "Sebastián Ríos Saucedo"
COURSE = "Computer Vision & Speech Recognition · EADA · Session 7 (CNNs)"
REPO_URL = "https://github.com/sebarios02/Milk10k-project"
NB_URL = REPO_URL + "/blob/main/notebooks/Class%20Session%207.ipynb"
RESULTS_URL = REPO_URL + "/blob/main/outputs/session_07/results.json"
SCOREBOARD_URL = REPO_URL + "/blob/main/outputs/session_07/scoreboard.csv"
SCRIPT_URL = REPO_URL + "/blob/main/scripts/13_session7_reports.py"

ORANGE, BLUE, GREY = "#F08200", "#2E6FBA", "#555555"


# --------------------------------------------------------------------------- #
# A tiny top-to-bottom page layout on A4 with matplotlib
# --------------------------------------------------------------------------- #
class Doc:
    W, H = 8.27, 11.69                 # A4 in inches
    LM, RM, TM, BM = 0.75, 0.75, 0.7, 0.7

    def __init__(self, path: Path, title: str):
        self.pdf = PdfPages(path)
        self.title = title
        self.page_no = 0
        self._new_page()

    # coordinates: inches from the top of the page
    def _new_page(self):
        if self.page_no:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(self.W, self.H))
        self.page_no += 1
        self.y = self.TM

    def _footer(self):
        self.fig.text(self.LM / self.W, 0.35 / self.H, f"{AUTHOR} · {self.title}", fontsize=7, color=GREY)
        self.fig.text(1 - self.RM / self.W, 0.35 / self.H, str(self.page_no), fontsize=7, color=GREY, ha="right")

    def _need(self, h):
        if self.y + h > self.H - self.BM:
            self._new_page()

    def _text(self, s, size, weight="normal", color="black", url=None, indent=0.0, style="normal"):
        self.fig.text((self.LM + indent) / self.W, 1 - self.y / self.H, s, fontsize=size, weight=weight,
                      color=color, va="top", url=url, style=style, family="DejaVu Sans")

    def heading(self, s, size=13, color=ORANGE, space_before=0.18):
        self._need(space_before + size / 72 * 1.6 + 0.4)
        self.y += space_before
        self._text(s, size, "bold", color)
        self.y += size / 72 * 1.55

    def para(self, s, size=9.5, indent=0.0, color="black", bullet=None, weight="normal", style="normal"):
        width_in = self.W - self.LM - self.RM - indent - (0.18 if bullet else 0)
        chars = int(width_in * 72 / (size * (0.58 if weight == "bold" else 0.53)))
        lines = textwrap.wrap(s, chars) or [""]
        lh = size / 72 * 1.38
        for i, line in enumerate(lines):
            self._need(lh)
            if bullet and i == 0:
                self._text(bullet, size, indent=indent, color=color)
            self._text(line, size, weight, color, indent=indent + (0.18 if bullet else 0), style=style)
            self.y += lh
        self.y += lh * 0.35

    def link(self, label, url, size=9.5):
        lh = size / 72 * 1.45
        self._need(2 * lh)
        self._text(label, size, "bold")
        self.y += lh
        self._text(url, size - 1, color=BLUE, url=url, indent=0.2)    # own line: long URLs never overflow
        self.y += lh * 1.15

    def table(self, header, rows, col_w, size=8, bold_rows=(), row_h=0.24):
        h = row_h * (len(rows) + 1)
        self._need(h + 0.1)
        total = self.W - self.LM - self.RM
        widths = [w / sum(col_w) * total for w in col_w]
        y = self.y
        for r, row in enumerate([header] + rows):
            x = self.LM
            if r == 0:
                self.fig.patches.append(plt.Rectangle((self.LM / self.W, 1 - (y + row_h) / self.H), total / self.W,
                                                      row_h / self.H, transform=self.fig.transFigure,
                                                      color="#FBE3C8", zorder=0))
            elif r - 1 in bold_rows:
                self.fig.patches.append(plt.Rectangle((self.LM / self.W, 1 - (y + row_h) / self.H), total / self.W,
                                                      row_h / self.H, transform=self.fig.transFigure,
                                                      color="#E8F0FA", zorder=0))
            for c, (cell, w) in enumerate(zip(row, widths)):
                self.fig.text((x + 0.05) / self.W, 1 - (y + row_h / 2) / self.H, str(cell), fontsize=size,
                              va="center", ha="left",
                              weight="bold" if (r == 0 or r - 1 in bold_rows) else "normal")
                x += w
            y += row_h
            self.fig.add_artist(plt.Line2D([self.LM / self.W, (self.LM + total) / self.W],
                                           [1 - y / self.H] * 2, transform=self.fig.transFigure,
                                           color="#CCCCCC", lw=0.5))
        self.y = y + 0.15

    def image(self, path: Path, width_in=None, caption=None):
        if not path.exists():
            self.para(f"[figure missing: {path.name}, run the notebook first]", color="red")
            return
        img = plt.imread(path)
        width_in = width_in or (self.W - self.LM - self.RM)
        h = width_in * img.shape[0] / img.shape[1]
        self._need(h + 0.35)
        x0 = self.LM + (self.W - self.LM - self.RM - width_in) / 2
        ax = self.fig.add_axes([x0 / self.W, 1 - (self.y + h) / self.H, width_in / self.W, h / self.H])
        ax.imshow(img); ax.axis("off")
        self.y += h + 0.08
        if caption:
            self.para(caption, size=8, color=GREY, style="italic")

    def close(self):
        self._footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.pdf.close()


def header_block(doc: Doc, title: str, subtitle: str):
    doc._text(title, 17, "bold", ORANGE)
    doc.y += 0.36
    doc._text(subtitle, 10, color=GREY)
    doc.y += 0.25
    doc._text(f"{AUTHOR} · {COURSE} · {date.today():%d %B %Y}", 9, color=GREY)
    doc.y += 0.35


def f3(x):
    return f"{x:.3f}"


# --------------------------------------------------------------------------- #
# Load what the notebook saved
# --------------------------------------------------------------------------- #
def load():
    res_path = OUT / "results.json"
    if not res_path.exists():
        sys.exit(f"{res_path} not found: run notebooks/Class Session 7.ipynb first.")
    R = json.loads(res_path.read_text())
    for k in ("layer4", "layer3_4", "mini_challenge"):
        if k not in R:
            sys.exit(f"results.json has no '{k}': run the notebook up to section 10.")
    sb = pd.read_csv(OUT / "scoreboard.csv", index_col=0) if (OUT / "scoreboard.csv").exists() else None
    return R, sb


# --------------------------------------------------------------------------- #
# PDF 1: Mini challenge (slide 29)
# --------------------------------------------------------------------------- #
def mini_challenge_verdict(R):
    a, b, mc = R["layer4"], R["layer3_4"], R["mini_challenge"]
    d, (lo, hi), p = mc["delta_best_val_f1"], mc["delta_ci95"], mc["p_l34_better"]
    tr = mc["time_ratio_train"]
    extra = b["trainable_params"] - a["trainable_params"]
    cost = (f"Unfreezing layer3 adds {extra:,} trainable weights (×{mc['params_ratio']:.2f}) and makes each "
            f"training epoch take {b['sec_per_epoch_train']:.0f} s instead of {a['sec_per_epoch_train']:.0f} s "
            f"(×{tr:.2f}), because the gradient now has to travel back through layer3 and its weights must be updated.")
    gain = (f"Best validation macro-F1 goes from {f3(a['best_val_f1'])} (epoch {a['best_epoch']}) to "
            f"{f3(b['best_val_f1'])} (epoch {b['best_epoch']}), a difference of {d:+.3f}. The paired bootstrap on the "
            f"749 validation lesions gives a 95% interval of [{lo:+.3f}, {hi:+.3f}] for that difference "
            f"(layer3+4 is better in {p:.0%} of the resamples).")
    if lo > 0:
        verdict = (f"Verdict: worth it. The gain is larger than the validation noise (the whole interval is above 0), "
                   f"and the extra cost (×{tr:.2f} per epoch) is affordable for a 6-epoch run. layer3 learns mid-level "
                   f"patterns (textures, dots, pigment networks), which is exactly where dermoscopy differs most from "
                   f"ImageNet photos, so letting it adapt helps.")
        short = "worth it"
    elif hi < 0:
        verdict = (f"Verdict: not worth it. The larger model is worse on validation and the interval is entirely below 0, "
                   f"while each epoch costs ×{tr:.2f} more. With only 3,742 training lesions, the 2.1 M extra weights "
                   f"overfit faster than they adapt: layer4 alone is the better trade-off.")
        short = "not worth it"
    elif d > 0:
        verdict = (f"Verdict: not clearly worth it. The {d:+.3f} gain is inside the noise (the interval contains 0), "
                   f"so I can't claim layer3 helps, while the cost per epoch is certain (×{tr:.2f}). With 3,742 training "
                   f"lesions and only 18 Indeterminate lesions in validation, a single run can't separate a real "
                   f"improvement from luck. I keep layer4 only as the default (cheaper, fewer weights that can overfit); "
                   f"to change my mind I would need the gain to hold across several seeds or with cross-validation.")
        short = "not clearly worth it"
    else:
        verdict = (f"Verdict: not worth it. Unfreezing layer3 does not improve validation macro-F1 ({d:+.3f}, interval "
                   f"contains 0) and every epoch costs ×{tr:.2f} more. The ImageNet layer3 features are already good enough "
                   f"for this dataset size; the extra 2.1 M weights mostly add capacity to memorise 3,742 lesions.")
        short = "not worth it"
    return cost, gain, verdict, short


def build_mini_challenge(R):
    a, b = R["layer4"], R["layer3_4"]
    cost, gain, verdict, short = mini_challenge_verdict(R)
    path = REPORTS / "S7_MiniChallenge_Rios_Saucedo.pdf"
    doc = Doc(path, "S7 Mini Challenge")
    header_block(doc, "S7 Mini Challenge: Unfreeze More and Compare",
                 "Slide 29 · Is unfreezing more layers always worth the extra training cost?")

    doc.heading("Code (repository)", 12, space_before=0.05)
    doc.link("Repo: ", REPO_URL)
    doc.link("Notebook: ", NB_URL)
    doc.para("Mini challenge = section 10 of the notebook (sections 1–9 are the class code it builds on). "
             "Raw numbers: outputs/session_07/results.json and mini_challenge.csv.", size=8.5, color=GREY)

    doc.heading("Setup", 12)
    doc.para("Same lesion-level split as Sessions 5–6 (my Milestone 1 split, seed 42): 3,742 train / 749 validation / "
             "749 test lesions, one dermoscopic image per lesion, 224×224, ImageNet normalisation, random flips, "
             "class-weighted cross-entropy, Adam. Both runs start from a fresh ImageNet ResNet-18 with a new 512→3 head, "
             "seed 0, lr = 1e-4, 6 epochs, and keep the epoch with the best validation macro-F1. The only difference "
             "is which weights may change. The test set stays closed for this exercise.")

    doc.heading("Results (validation only)", 12)
    rows = []
    for r in (a, b):
        rows.append([r["label"].replace("ResNet-18 fine-tuned: ", ""), f"{r['trainable_params']:,}",
                     f"{r['trainable_params'] / r['total_params']:.0%}", f"{r['sec_per_epoch_train']:.0f}",
                     f"{r['sec_per_epoch_total']:.0f}", f"{f3(r['best_val_f1'])} (ep. {r['best_epoch']})"])
    doc.table(["Unfrozen", "Trainable params", "% of net", "s/ep. (train)", "s/ep. (+val)", "Best val macro-F1"],
              rows, [2.2, 1.4, 0.9, 1.2, 1.2, 1.6])
    doc.table(["Val macro-F1 per epoch", *[f"ep {i}" for i in range(len(a["val_f1_per_epoch"]))]],
              [["layer4 + fc", *map(f3, a["val_f1_per_epoch"])],
               ["layer3 + layer4 + fc", *map(f3, b["val_f1_per_epoch"])]],
              [2.2] + [1] * len(a["val_f1_per_epoch"]))
    if "frozen" in R:
        doc.para(f"Reference: frozen backbone + linear head (1,539 trainable weights) reaches {f3(R['frozen']['best_val_f1'])}.",
                 size=8.5, color=GREY)
    doc.image(FIG / "session7_mini_challenge.png",
              caption="Left: validation macro-F1 per epoch. Middle/right: validation confusion matrices of each model's "
                      "best epoch (rows = truth, columns = prediction).")

    doc.heading("Decision: worth it?", 12)
    doc.para(cost)
    doc.para(gain)
    doc.para(verdict, weight="bold")
    last = len(a["val_f1_per_epoch"]) - 1
    if a["best_epoch"] == last or b["best_epoch"] == last:
        doc.para("Caveat: " + ("both models" if a["best_epoch"] == b["best_epoch"] == last else "at least one model")
                 + f" reached its best score in the last epoch ({last}), so validation F1 was still rising when training "
                 "stopped. The comparison is fair (same 6 epochs, as the exercise asks), but it answers 'which is better "
                 "after 6 epochs'; with more epochs the model with more trainable weights could still catch up.", size=9)
    doc.para("Timings were measured on my laptop CPU (no GPU), so they move with other programs and thermal throttling; "
             "the ratio between runs is indicative, not exact.", size=8.5, color=GREY)
    doc.para("Why this is not automatic: unfreezing more layers gives the network more freedom to adapt ImageNet features "
             "to skin, but every extra weight must be learned from the same 3,742 lesions. The slide's rule of thumb "
             "(moderate data → partial fine-tuning) is a starting point; the measurement above is what decides. "
             "Validation is also small (749 lesions, 18 Indeterminate), so one lesion more or less in the rare class moves "
             "macro-F1 by about 0.01–0.02, which is why I report the bootstrap interval and not only the two numbers.",
             size=9)
    doc.close()
    return path, short


# --------------------------------------------------------------------------- #
# PDF 2: Homework (slide 30)
# --------------------------------------------------------------------------- #
def build_homework(R, sb):
    if sb is None or "scoreboard_best" not in R:
        sys.exit("No scoreboard yet: run the notebook up to section 12.")
    path = REPORTS / "S7_Homework_Rios_Saucedo.pdf"
    doc = Doc(path, "S7 Homework")
    header_block(doc, "Homework Session 7: Final Scoreboard",
                 "Slide 30 · Sessions 5–7, validation and test macro-F1 · best model · reading · full fine-tuning")

    doc.heading("Code (repository)", 12, space_before=0.05)
    doc.link("Repo: ", REPO_URL)
    doc.link("Notebook: ", NB_URL)
    doc.link("Scoreboard: ", SCOREBOARD_URL)
    doc.link("Report script: ", SCRIPT_URL)
    doc.para("Homework = notebook sections 11 (full fine-tuning), 12 (test set + scoreboard) and 13 (reading). "
             "Sessions 5–6 rows come from my earlier notebooks (outputs/session_06/scoreboard.csv).", size=8.5, color=GREY)

    # ---- 1. scoreboard
    doc.heading("1. Final scoreboard (same split, same metric)", 12)
    best = R["scoreboard_best"]
    rows, bold = [], []
    for i, (name, r) in enumerate(sb.iterrows()):
        rows.append([name, f3(r["val macro-F1"]), f3(r["test macro-F1"]), f"{r['test Δ vs my baseline']:+.3f}",
                     "★" if name == best["name"] else ""])
        if name == best["name"]:
            bold.append(i)
    doc.table(["Model", "Val macro-F1", "Test macro-F1", "Test Δ vs S5", "Best"],
              rows, [4.0, 1.1, 1.1, 1.1, 0.5], size=7.8, bold_rows=bold)
    doc.para("Validation = 749 lesions, test = 749 lesions (17–18 Indeterminate in each). The best model is chosen on "
             "validation; its test number is reported once, never used to choose.", size=8.5, color=GREY)
    doc.image(FIG / "session7_scoreboard.png", width_in=6.3)

    # ---- best model
    doc.heading("My best model and its number", 12)
    prev = sb[~sb.index.str.startswith("Session 7")]
    s7 = sb[sb.index.str.startswith("Session 7")]
    best_prev_name = prev["val macro-F1"].idxmax()
    line = (f"Best model (highest validation macro-F1): {best['name']}: validation {f3(best['val_macro_f1'])}, "
            f"TEST macro-F1 {f3(best['test_macro_f1'])}")
    if "test_ci95" in best:
        lo, hi = best["test_ci95"]
        line += f" (bootstrap 95% interval {f3(lo)}–{f3(hi)})"
    doc.para(line + ".", weight="bold")
    if len(s7):
        d_val = s7["val macro-F1"].max() - prev["val macro-F1"].max()
        d_test = best["test_macro_f1"] - prev.loc[best_prev_name, "test macro-F1"]
        if d_val > 0:
            doc.para(f"Does seeing the pixels beat the handcrafted numbers? Yes on this split: the best Session 7 network is "
                     f"{d_val:+.3f} above the best Session 5–6 model ({best_prev_name}, validation "
                     f"{f3(prev['val macro-F1'].max())}) on validation, and {d_test:+.3f} on test. The gain comes from "
                     f"transfer learning: an ImageNet backbone brings filters learned on 1.28 M photos, which 3,742 lesions "
                     f"could never teach from zero.")
        else:
            doc.para(f"Does seeing the pixels beat the handcrafted numbers? Not on this split: the best Session 7 network "
                     f"stays {d_val:+.3f} below {best_prev_name} on validation.")
        if "scratch" in R:
            sc = R["scratch"]
            doc.para(f"The small CNN trained from scratch reaches {f3(sc['best_val_f1'])} on validation and "
                     f"{f3(sc.get('test_macro_f1', float('nan')))} on test: with random filters and 306,147 weights "
                     f"(98% in the final Linear layer) it learns less than the pretrained backbones, which is the "
                     f"'why not train from scratch' slide measured on my data.", size=9)
    if "test_confusion" in best:
        cm = best["test_confusion"]
        recalls = [cm[i][i] / max(1, sum(cm[i])) for i in range(3)]
        doc.para(f"Per-class test recall of the best model: Benign {recalls[0]:.0%}, Indeterminate {recalls[1]:.0%} "
                 f"({cm[1][1]} of {sum(cm[1])}), Malignant {recalls[2]:.0%}. Malignant lesions predicted Benign "
                 f"(the clinically dangerous error): {cm[2][0]} of {sum(cm[2])}.", size=9)
        doc.image(FIG / "session7_best_test_confusion.png", width_in=3.0,
                  caption="Best model, confusion matrix on the test set (rows = truth).")

    # ---- Session 7 details
    doc.heading("Session 7 models in detail", 12)
    keys = [k for k in ("scratch", "frozen", "layer4", "layer3_4", "full") if k in R]
    doc.table(["Model", "Trainable", "s/epoch", "Best val F1 (ep.)", "Test F1", "Test acc."],
              [[R[k]["label"].replace("ResNet-18 ", ""), f"{R[k]['trainable_params']:,}",
                f"{R[k]['sec_per_epoch_train']:.1f}", f"{f3(R[k]['best_val_f1'])} ({R[k]['best_epoch']})",
                f3(R[k].get("test_macro_f1", float("nan"))), f3(R[k].get("test_accuracy", float("nan")))]
               for k in keys], [3.1, 1.2, 0.9, 1.5, 0.8, 0.8], size=7.8)
    if "frozen" in R:
        doc.para(f"The frozen backbone's {R['frozen']['sec_per_epoch_train']:.2f} s/epoch excludes the one-off "
                 f"{R['frozen'].get('embedding_seconds_once', 0):.0f} s needed to compute the 512 features of every image.",
                 size=8.5, color=GREY)

    # ---- 2. reading
    doc.heading("2. Reading: how do DenseNet and EfficientNet differ from ResNet?", 12)
    doc.para("DenseNet (Huang et al., 2017): instead of adding a block's input to its output like ResNet (F(x) + x), "
             "every layer in a dense block receives the concatenation of the feature maps of all previous layers, so "
             "features are reused instead of relearned and the network needs fewer parameters for the same accuracy.",
             bullet="•")
    doc.para("EfficientNet (Tan & Le, 2019): instead of only adding depth (ResNet-18 → 152), it scales depth, width and "
             "input resolution together with one fixed ratio (compound scaling), starting from a small base network "
             "built from inverted-residual MBConv blocks with depthwise convolutions and squeeze-and-excitation.",
             bullet="•")

    # ---- 3. optional
    doc.heading("3. Optional: full fine-tuning (strategy 3)", 12)
    if "full" in R:
        f, l4 = R["full"], R["layer4"]
        d = f["best_val_f1"] - l4["best_val_f1"]
        doc.para(f"All {f['trainable_params']:,} weights trained (lr 1e-4, 6 epochs): best validation macro-F1 "
                 f"{f3(f['best_val_f1'])} (epoch {f['best_epoch']}) vs {f3(l4['best_val_f1'])} for layer4 only "
                 f"({d:+.3f}), at {f['sec_per_epoch_train']:.0f} s per training epoch vs {l4['sec_per_epoch_train']:.0f} s "
                 f"(×{f['sec_per_epoch_train'] / l4['sec_per_epoch_train']:.2f}). Test macro-F1: "
                 f"{f3(f.get('test_macro_f1', float('nan')))}.")
        if abs(d) < 0.02:
            doc.para("The difference is within the ~0.02 that one or two rare-class lesions can move on this validation "
                     "set, so full fine-tuning does not pay for its cost here; it is meant for large datasets or images "
                     "far from natural photos.", size=9)
        elif d > 0:
            doc.para("Full fine-tuning helps here: dermoscopic images are far enough from ImageNet photos that even the "
                     "early filters benefit from adapting, and the small learning rate kept the pretrained weights from "
                     "being wrecked.", size=9)
        else:
            doc.para("Full fine-tuning is worse: with 3,742 lesions, letting all 11 M weights move overfits faster than it "
                     "adapts, which is why the slide recommends it only for large datasets.", size=9)
    else:
        doc.para("Not run (RUN_FULL_FT = False in the notebook).")
    doc.close()
    return path


if __name__ == "__main__":
    REPORTS.mkdir(parents=True, exist_ok=True)
    R, sb = load()
    p1, short = build_mini_challenge(R)
    print("written:", p1, "| verdict:", short)
    if sb is not None and "scoreboard_best" in R:
        print("written:", build_homework(R, sb))
    else:
        print("scoreboard not found yet: run notebook section 12, then re-run this script for the homework PDF.")
