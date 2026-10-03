"""B5 - Leak-free train / val / test split at lesion level, verified and saved.

Outputs: outputs/splits/{train,val,test}.csv (one row per IMAGE, with lesion_id,
labels and split), outputs/splits/split_info.json, outputs/splits/split_summary.csv
"""
import datetime as dt
import json

import pandas as pd

import _bootstrap  # noqa: F401
from milk10k_pipeline import config, data, labels, splits


def main() -> None:
    config.ensure_output_dirs()
    images = data.load_images_table()
    lesions = data.build_lesion_table(images)

    # Stratify on the 11 classes, refined by diagnosis_1: AKIEC is split into
    # AKIEC|Indeterminate and AKIEC|Malignant, so that the small Indeterminate
    # class (our primary target) is also balanced across splits.
    lesions["strat"] = splits.make_strat_key(lesions)
    tr, va, te = splits.split_lesions(lesions, config.VAL_SIZE, config.TEST_SIZE,
                                      seed=config.SEED, label_col="strat")
    sizes = splits.check_split(tr, va, te, len(lesions), config.VAL_SIZE, config.TEST_SIZE)

    lesions.drop(columns="strat").to_csv(config.LESIONS_CSV, index=False)
    lm = labels.build_label_map()
    img = splits.assign_split_column(images, tr, va, te)
    img = labels.add_label_columns(img, lm)
    keep = ["isic_id", "lesion_id", "image_type", "split", "diagnosis_1", "dx", "fine",
            "y_primary", "y_fine", "age_approx", "sex", "anatom_site_general", "image_manipulation"]
    img = img[keep]

    # ---- verification (printed) ----
    print("1) lesion overlap between splits")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = splits.leaked_lesions(img.loc[img.split == a, "lesion_id"], img.loc[img.split == b, "lesion_id"])
        print(f"   {a}-{b}: {n}")
        assert n == 0

    per = img.groupby(["split", "lesion_id"]).size()
    print(f"2) every lesion has exactly 2 images in its split: {bool((per == 2).all())}")
    assert (per == 2).all()

    n_les = img.drop_duplicates("lesion_id")["split"].value_counts()
    print("   lesions per split:", n_les.to_dict(), {k: f"{v:.2f}%" for k, v in sizes.items()})

    les_img = img.drop_duplicates("lesion_id")
    prop11 = splits.proportions_by_split(les_img, "dx")
    dev = (prop11[["train", "val", "test"]].sub(prop11["global"], axis=0)).abs()
    print("3) 11-class proportions per split (% of lesions):\n", prop11.to_string())
    print(f"   max deviation from global: {dev.values.max():.2f} pp "
          f"(class {dev.max(axis=1).idxmax()})")
    counts11 = pd.crosstab(les_img["dx"], les_img["split"])[["train", "val", "test"]]
    print("   lesion counts:\n", counts11.to_string())

    prop1 = splits.proportions_by_split(les_img, "diagnosis_1")
    print("4) diagnosis_1 proportions per split:\n", prop1.to_string())

    # ---- save ----
    for name in ("train", "val", "test"):
        img[img.split == name].to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
    summary = pd.concat({"pct_11class": prop11, "pct_diagnosis_1": prop1})
    summary.to_csv(config.SPLIT_DIR / "split_summary.csv")
    info = {
        "seed": config.SEED, "created": dt.date.today().isoformat(),
        "method": "StratifiedGroupKFold at lesion level (groups=lesion_id), stratified on 11-class dx refined by diagnosis_1 (AKIEC split in 2)",
        "requested": {"val": config.VAL_SIZE, "test": config.TEST_SIZE},
        "lesions": n_les.to_dict(), "images": img["split"].value_counts().to_dict(),
        "max_deviation_pp_11class": round(float(dev.values.max()), 3),
    }
    (config.SPLIT_DIR / "split_info.json").write_text(json.dumps(info, indent=2))
    print("saved outputs/splits/{train,val,test}.csv, split_summary.csv, split_info.json")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
