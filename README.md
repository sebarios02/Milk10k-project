# MILK10k skin-lesion classification: data pipeline (Milestone 1)

**Author:** Sebastián Ríos Saucedo · Computer Vision & Speech Recognition · EADA

## 1. What this project is

**Task.** Classify a skin lesion into **Benign / Indeterminate / Malignant** (`diagnosis_1`) from its images. Each lesion has two views, a dermoscopic image and a clinical close-up. Stretch goal: the finer diagnostic scheme (11 classes in the data, 8 after merging rare ones, see Section 5). Age, sex and body site may be added later as optional metadata. Columns that leak the label are never inputs (see `reports/data_quality_report.md`).

**Dataset.** MILK10k (ISIC Archive, *MILK study team*): **10,480 JPEG images of 5,240 lesions** (2 images per lesion), `metadata.csv` (per image) and `training_gt.csv` (one-hot, per lesion). License **CC-BY-NC 4.0**, attribution "MILK study team" (see `licenses/CC-BY-NC.txt` in the data folder). Non-commercial use only. The images are **not** redistributed in this repo.

**Clinical motivation.** Help prioritise which lesions a dermatologist should see or biopsy first. Because such models have historically underperformed on darker skin tones, fairness across skin tone, age and sex is part of the later evaluation.

**Goal of Milestone 1.** Turn the Session 2 exploration code into a **leak-free, reproducible data pipeline**: integrity checks on the full image set, label strategy, data-quality report, lesion-level splits, train-only normalisation statistics, augmentation, and PyTorch `Dataset`/`DataLoader` with imbalance handling.

## 2. Setup and how to run

* Python **3.10+** (developed in the project `venv`).
* Install: `python -m venv venv` → activate (`venv\Scripts\activate` on Windows) → `pip install -r requirements.txt`
* **Data location.** Put the MILK10k folder **next to the repo** (default):

```
Computer Vision & Speech Recognition/
├── Milk10k-project/        <- this repo
└── milk10k_project/        <- data (not in git)
    ├── metadata.csv
    ├── supplements/training_gt.csv
    └── images/ISIC_*.jpg   (10,480 files)
```

  or point the code anywhere else with an environment variable (read **only** in `milk10k_pipeline/config.py`):

```powershell
$env:MILK10K_DATA_DIR = "D:\data\milk10k_project"      # PowerShell
# optional: $env:MILK10K_IMAGES_DIR, $env:MILK10K_NUM_WORKERS
```

**Reproduce everything** (from the repo root, venv active):

```bash
python scripts/run_milestone1.py        # runs 01 -> 08 in order, stops at the first failure
# or step by step:
python scripts/01_integrity_check.py    # B1  all 10,480 files exist + Image.verify(); image_size_summary.csv
python scripts/02_label_eda.py          # B2  figures/ + reports/session2_findings_check.md
python scripts/03_label_strategy.py     # B3  outputs/label_map.json
python scripts/04_quality_report.py     # B4  reports/data_quality_report.md
python scripts/05_make_splits.py        # B5  outputs/splits/{train,val,test}.csv
python scripts/06_preprocessing.py      # B6  resolution evidence + outputs/norm_stats.json (train only)
python scripts/07_augmentation.py       # B7  determinism check, figures/augmentation_examples.png
python scripts/08_dataloader_checks.py  # B8  class_weights.json, sanity checks, batch figure
python -m pytest -q                     # tests (splits, labels, datasets, transforms)
```

Part A homework: open `notebooks/homework_part_a.ipynb` in VS Code → *Restart* → *Run All* → *Export* → PDF (`notebooks/homework_part_a.pdf`).

## 3. Repository structure

```
Milk10k-project/
├── README.md                     this file
├── SUBMISSION.md                 name, repo URL, direct links to every deliverable
├── requirements.txt              dependencies
├── .gitignore                    excludes venv, caches and ALL raw images
├── milk10k_pipeline/             importable package = all reusable logic
│   ├── config.py                 the ONLY place with paths, seed, image size, class names
│   ├── data.py                   load CSVs, image table + dx, lesion table (pivot/groupby)
│   ├── quality.py                consistency checks, missing-value decisions, shortcut cross-tabs
│   ├── integrity.py              file exists + Image.verify() + sizes (B1)
│   ├── labels.py                 label strategy, label_map.json, class/sampler weights (B3, B8)
│   ├── splits.py                 split_lesions(): lesion-level StratifiedGroupKFold (A3.2, B5)
│   ├── transforms.py             train_transform / eval_transform + augmentation table (B7)
│   ├── datasets.py               MILK10kImageDataset, LesionDataset, aggregate_predictions (A3.6, B8)
│   ├── loaders.py                DataLoaders, seeding, WeightedRandomSampler, class weights (B8)
│   ├── reporting.py              DataFrame -> Markdown helper
│   ├── metadata_analysis.py      Session 2: metadata vs target tests
│   ├── color_analysis.py         Session 2 colour analysis (+ Session 3 lesion hue/V)
│   ├── preprocessing.py          Session 2 numpy preprocessing (now with strict mode)
│   ├── data_loader.py            Session 2 numpy loader (now fails loudly on missing files)
│   ├── visualizer.py             Session 2 plotting utilities (reused for gallery and batch figures)
│   └── run_eda.py                Session 2 EDA runner -> eda_outputs/
├── scripts/                      thin entry points, one per Milestone-1 step (01 ... 08 + run_milestone1.py)
├── tests/                        pytest: split properties, label maps, datasets, transforms
├── notebooks/                    exploration + homework notebooks (Session 2, Session 3, homework_part_a)
├── outputs/                      GENERATED small files (committed): splits, label map, weights, stats
├── figures/                      GENERATED figures (EDA, augmentation, batch check)
├── reports/                      GENERATED / written reports (milestone report, data quality, ...)
├── eda_outputs/                  Session 2 figures and tables (kept as they were)
└── preprocessing/                Session 2 normalisation notebook
```

**Why this organisation.** Logic that is used more than once lives in **one importable package** (`milk10k_pipeline`), so the notebooks, the scripts and the future training code all call the same functions: the split, the transforms and the label mapping cannot drift between notebooks. `scripts/` only *orchestrate*: each one maps to one Milestone-1 requirement and can be re-run from a clean clone. Notebooks are for exploration and for answering the homework. Everything a script **generates** goes to `outputs/`, `figures/` or `reports/`, never next to the source code, so it is obvious what can be deleted and regenerated. All configuration (paths, seed, image size) is in `config.py`, so moving the data or changing the seed is a one-line change and no absolute path appears anywhere. I evolved the Session 2 package instead of starting over: the old modules are still there, upgraded (fail-loudly loader, VASC bug fixed, paths from config).

## 4. Data handling rules

* **Not committed:** raw images and the data folder (`.gitignore`), the venv, caches.
* **Committed:** split CSVs, `label_map.json`, `class_weights.json`, `norm_stats.json`, `lesions.csv`, `image_size_summary.csv`, figures and reports. All are small and needed to reproduce later milestones.
* Generated files go to `outputs/` (tables/JSON), `figures/` (PNG) and `reports/` (Markdown).
* **Seed:** `42` (`config.SEED`), used for splits, samplers and dataloaders.
* **Splits created:** 2026-10-03 (also stored in `outputs/splits/split_info.json`).

## 5. Key decisions and results so far

Full details in the **[Milestone 1 report](reports/milestone1_report.md)**.

* **Label strategy.** Primary target `diagnosis_1` with **Indeterminate kept as a 3rd class** (123 lesions, all AKIEC). Fine target: 11 → 8 classes (`BEN_OTH, DF, INF, VASC → OTHER_BENIGN`; `MAL_OTH → OTHER_MALIGNANT`; benign and malignant never mixed). AKIEC maps to two diagnosis_1 values, so diagnosis_1 cannot be derived from the 11-class prediction.
* **Splits.** Lesion-level StratifiedGroupKFold (70/15/15 ≈ 3,742 / 749 / 749 lesions), stratified on the 11 classes refined by diagnosis_1. Zero lesion overlap, both images of a lesion always in the same split, max class-proportion deviation 0.09 pp.
* **Preprocessing.** Resize shorter side to 256 → 224 crop. Normalisation with **train-only** mean/std. Option (a): each image is a sample with its lesion's label, and evaluation is **per lesion** (average of both views).
* **Imbalance.** Train imbalance Malignant : Indeterminate = **29.5 : 1** (11-class BCC : MAL_OTH = 257 : 1). Class-weighted loss **and** a WeightedRandomSampler are both implemented (use one at a time). Metrics: balanced accuracy, macro-F1, Malignant recall.
* **Leakage.** `diagnosis_confirm_type`, `diagnosis_2-4`, `concomitant_biopsy` and `melanocytic` are never inputs. `image_manipulation` is a potential shortcut and is not an input.
