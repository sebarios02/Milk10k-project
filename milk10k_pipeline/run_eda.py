import matplotlib
matplotlib.use("Agg")  # safe for headless / script runs; drop this line to see plots interactively

from pathlib import Path
import pandas as pd

from . import metadata_analysis as ma
from . import color_analysis as ca
from . import preprocessing as pp
from .data_loader import MILK10kDataLoader
from . import visualizer as viz


DATA_DIR = Path("../../milk10k_project")        
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metadata() -> pd.DataFrame:
    df_metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    gt = pd.read_csv(DATA_DIR / "supplements" / "training_gt.csv")

    class_cols = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
                  "INF", "MAL_OTH", "MEL", "NV", "SCCKA"]
    gt["diagnosis_class"] = gt[class_cols].idxmax(axis=1)

    df = df_metadata.merge(gt[["lesion_id", "diagnosis_class"]], on="lesion_id", how="inner")
    return df


def main():
    df = load_metadata()
    print(f"Loaded metadata: {df.shape}")

    # ---------------- Metadata analysis ----------------
    print("\n--- Metadata survey ---")
    survey = ma.survey_columns(df, id_cols=["isic_id", "lesion_id"], target_col="diagnosis_class")
    survey.to_csv(OUTPUT_DIR / "column_survey.csv", index=False)
    print(survey)

    results = {}
    categorical_fields = ["sex", "anatom_site_general", "diagnosis_confirm_type"]
    numeric_fields = ["age_approx"]

    for field in categorical_fields:
        if field not in df.columns:
            continue
        res = ma.categorical_vs_target(df, field, "diagnosis_class", plot=True)
        plt_save(f"{field}_vs_target.png")
        results[field] = res
        print(f"{field}: p={res['p_value']:.4g} -> {res['verdict']}")

    for field in numeric_fields:
        if field not in df.columns:
            continue
        res = ma.numeric_vs_target(df, field, "diagnosis_class", plot=True)
        plt_save(f"{field}_vs_target.png")
        results[field] = res
        print(f"{field}: p={res['anova_p']:.4g} -> {res['verdict']}")

    summary = ma.summarize_associations(results)
    summary.to_csv(OUTPUT_DIR / "association_summary.csv", index=False)
    print(summary)

    # ---------------- Color/histogram analysis ----------------
    print("\n--- Color/histogram analysis ---")
    class_paths = ca.sample_images_per_class(
        df, IMAGE_DIR, "diagnosis_class", n_per_class=25
    )
    print({k: len(v) for k, v in class_paths.items()})

    fig, _ = ca.average_grayscale_histogram(class_paths)
    plt_save("avg_grayscale_histogram.png", fig)

    fig = ca.average_rgb_histogram(class_paths)
    plt_save("avg_rgb_histogram.png", fig)

    stats_df = ca.channel_summary_stats(class_paths)
    stats_df.to_csv(OUTPUT_DIR / "channel_summary_stats.csv", index=False)
    print(stats_df)

    fig, _ = ca.plot_channel_summary_boxplots(class_paths)
    plt_save("channel_boxplots.png", fig)

    # ---------------- Loader + visualizer sanity checks ----------------
    print("\n--- Data loader + visualizer ---")
    loader = MILK10kDataLoader(
        df, image_dir=IMAGE_DIR, label_col="diagnosis_class",
        batch_size=16, target_size=(128, 128), normalization="minmax",
    )
    print(f"Loader ready: {len(loader)} batches of up to {loader.batch_size}")

    images, labels = next(iter(loader))
    fig = viz.plot_sample_grid(images, labels, n_cols=4, title="Sample batch from MILK10kDataLoader")
    plt_save("sample_grid.png", fig)

    fig = viz.plot_class_balance(df["diagnosis_class"], title="MILK10k Class Balance (full available dataset)")
    plt_save("class_balance.png", fig)

    raw_batch = pp.process_batch(df.head(8), image_dir=IMAGE_DIR, label_col="diagnosis_class",
                                  target_size=(128, 128), normalization=None)
    proc_batch = pp.process_batch(df.head(8), image_dir=IMAGE_DIR, label_col="diagnosis_class",
                                   target_size=(128, 128), normalization="minmax")
    fig = viz.plot_batch_summary(raw_batch.images, proc_batch.images, n_examples=4)
    plt_save("batch_summary.png", fig)

    print(f"\nAll outputs saved to {OUTPUT_DIR.resolve()}")


def plt_save(filename, fig=None):
    import matplotlib.pyplot as plt
    target = fig if fig is not None else plt.gcf()
    target.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(target)


if __name__ == "__main__":
    main()
