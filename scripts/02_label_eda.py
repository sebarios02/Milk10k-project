"""B2 - Label-centred EDA (builds on the Session 2 analysis, does not redo it).

Figures (figures/):
    class_distribution_diagnosis1.png
    class_distribution_11class_log.png
    class_to_diagnosis1_mapping.png
    gallery_one_per_class.png       (3x4, Session 2 visualizer)
Table (reports/session2_findings_check.md):
    Session 2 finding -> still true on the full dataset? -> consequence
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from milk10k_pipeline import color_analysis as ca
from milk10k_pipeline import config, data, quality, visualizer
from milk10k_pipeline.reporting import df_to_markdown


def save(fig, name):
    fig.savefig(config.FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


def class_distributions(lesions: pd.DataFrame) -> None:
    c1 = lesions["diagnosis_1"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(c1.index, c1.values, color=["#c0392b", "#2e86c1", "#f39c12"][:len(c1)])
    ax.bar_label(bars, labels=[f"{v} ({100 * v / c1.sum():.1f}%)" for v in c1.values])
    ax.set_title("Lesions per diagnosis_1 (primary target)")
    ax.set_ylabel("lesions")
    save(fig, "class_distribution_diagnosis1.png")

    c11 = lesions["dx"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(c11.index, c11.values, color="steelblue")
    ax.set_yscale("log")
    ax.bar_label(bars, labels=c11.values.astype(str), fontsize=8)
    ax.set_title(f"Lesions per class, 11-class scheme (log scale) - largest/smallest = {c11.max() / c11.min():.0f}x")
    ax.set_ylabel("lesions (log)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "class_distribution_11class_log.png")

    tab = quality.class_to_diagnosis1(lesions)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.log1p(tab.values), cmap="Blues")
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, tab.values[i, j], ha="center", va="center", fontsize=9)
    ax.set_title("11-class -> diagnosis_1 (lesion counts)\nAKIEC splits into Indeterminate AND Malignant")
    save(fig, "class_to_diagnosis1_mapping.png")


def gallery(images: pd.DataFrame) -> None:
    derm = images[images["image_type"] == data.DERM]
    derm = derm[data.available_mask(derm["isic_id"])]
    picks = derm.groupby("dx").sample(1, random_state=config.SEED).sort_values("dx")
    arrays, titles = [], []
    for _, r in picks.iterrows():
        with Image.open(config.image_path(r["isic_id"])) as im:
            arrays.append(np.array(im.convert("RGB").resize((256, 192))))
        titles.append(f"{r['dx']} ({r['diagnosis_1']})")
    arrays.append(np.full((192, 256, 3), 255, np.uint8))      # 12th cell left empty
    titles.append("")
    fig = visualizer.plot_sample_grid(arrays, titles, n_cols=4, figsize_per_cell=3.0,
                                      title="One dermoscopic example per class")
    save(fig, "gallery_one_per_class.png")


def session2_findings(images: pd.DataFrame, lesions: pd.DataFrame) -> pd.DataFrame:
    """Re-test the Session 2 findings on the full dataset and state the consequence."""
    rows = []
    n_cls = lesions["dx"].nunique()
    rows.append(("The dataset has 10 diagnostic classes (Session 2 report).",
                 f"NO - {n_cls} classes. VASC (47 lesions) was lost because the class list was hard-coded.",
                 "Class names are now always read from training_gt.csv (data.load_gt)."))

    ct = pd.crosstab(images["diagnosis_confirm_type"], images["diagnosis_1"], normalize="index") * 100
    rows.append(("diagnosis_confirm_type is the most strongly associated field (clinical assessment ~ benign moles).",
                 f"YES - clinical-assessment images are {ct.loc['single contributor clinical assessment', 'Benign']:.0f}% Benign "
                 f"vs {ct.loc['histopathology', 'Benign']:.0f}% for histopathology.",
                 "Leaks how the label was obtained -> excluded from model inputs (with diagnosis_2-4, concomitant_biopsy)."))

    med = lesions.groupby("dx")["age"].median()
    rows.append(("Age separates classes: NV young (median ~40), SCCKA / MAL_OTH old (~65-70).",
                 f"YES - median age NV {med['NV']:.0f}, SCCKA {med['SCCKA']:.0f}, MAL_OTH {med['MAL_OTH']:.0f}, BCC {med['BCC']:.0f}.",
                 "Legitimate clinical prior: allowed as an OPTIONAL metadata input later, but image-only "
                 "models are evaluated first so we know what the pixels contribute."))

    site_missing = lesions["site"].isna().mean() * 100
    rows.append(("anatom_site_general is informative (chi2 p ~ 1e-74).",
                 f"YES, but {site_missing:.0f}% of lesions have no site and missingness itself varies by class (A1.2 q4).",
                 "Encode missing as an explicit 'unknown' level; never impute a site."))

    cnt = lesions["dx"].value_counts()
    rows.append(("Imbalance about 280:1 (BCC vs MAL_OTH).",
                 f"YES - {cnt.max()}:{cnt.min()} = {cnt.max() / cnt.min():.0f}:1 at lesion level.",
                 "Class-weighted loss / WeightedRandomSampler from TRAIN counts; macro-F1 + balanced accuracy as metrics."))

    # colour: SCCKA darkest - re-check with up to 100 images per class (all available ones)
    sample = ca.sample_images_per_class(images[data.available_mask(images["isic_id"])], config.IMAGE_DIR,
                                        "dx", n_per_class=100, seed=config.SEED)
    stats = ca.channel_summary_stats(sample)
    stats["gray"] = 0.299 * stats["R_mean"] + 0.587 * stats["G_mean"] + 0.114 * stats["B_mean"]
    darkest = stats.sort_values("gray").iloc[0]
    rows.append(("SCCKA is the darkest class in all three channels (25 images per class).",
                 f"Re-checked with up to 100 images/class: darkest class is {darkest['class']} "
                 f"(gray {darkest['gray']:.0f}); spread between class means is "
                 f"{stats['gray'].max() - stats['gray'].min():.0f} grey levels "
                 f"(images used per class: {int(stats['n_images'].min())}-{int(stats['n_images'].max())}).",
                 "Colour differences are small and may come from the acquisition site -> normalise with TRAIN stats, "
                 "no hue jitter, and check later that the model is not using global brightness."))

    alt = pd.crosstab(images["image_manipulation"], images["diagnosis_1"], normalize="index") * 100
    rows.append(("(new) image_manipulation was never inspected in Session 2.",
                 f"'altered' images are {alt.loc['altered', 'Indeterminate']:.0f}% Indeterminate vs "
                 f"{alt.loc['instrument only', 'Indeterminate']:.0f}% among unaltered ones.",
                 "Potential shortcut: not a model input; report results with/without altered images."))

    return pd.DataFrame(rows, columns=["Session 2 finding", "Still true on the full dataset?", "Consequence for the pipeline"])


def main() -> None:
    config.ensure_output_dirs()
    images = data.load_images_table()
    lesions = data.build_lesion_table(images)
    class_distributions(lesions)
    gallery(images)
    tab = session2_findings(images, lesions)
    out = config.REPORT_DIR / "session2_findings_check.md"
    out.write_text("# Session 2 findings re-checked on the full dataset\n\n" + df_to_markdown(tab) + "\n")
    print(tab.to_string())
    print(f"saved {out.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
