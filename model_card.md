# Model card: MILK10k skin-lesion classifier (ResNet-18, fine-tuned)

**Author:** Sebastián Ríos Saucedo · EADA, Computer Vision & Speech Recognition · November 2026
**Code:** `milk10k_pipeline/deep.py`, `scripts/13–15`, `m5_final/predict.py` · **Report:** `report.pdf`

## Model details

| | 3-class model (`checkpoints/model_m4.pt`) | 11-class model (`checkpoints/model_11class.pt`) |
|---|---|---|
| Purpose | My analysis (M4): Benign / Indeterminate / Malignant | Official MILK10k leaderboard (M5) |
| Architecture | ResNet-18, ImageNet-1k pre-trained (torchvision `IMAGENET1K_V1`), all layers fine-tuned, `fc = Linear(512, 3)` | Same, `fc = Linear(512, 11)` (training_gt.csv column order) |
| Weights | 11.2 M, all trainable | 11.2 M, all trainable |
| Training | Adam lr 1e-4, batch 64, 6 epochs, class-weighted cross-entropy, random h/v flips, best epoch on validation macro-F1 | Same recipe |
| Input | **Dermoscopic image only**: RGB, resized to 224 × 224, /255, ImageNet mean/std (`deep.load_image_224` + `deep.prep`, identical at training and inference) | Same |
| Output | Softmax probabilities | Softmax probabilities; for the leaderboard the top class is rescaled to ≥ 0.5 and the others to < 0.5 (`outputs/m5/postprocess.json`) |

**How to run:** `python m5_final/predict.py path/to/image.jpg` (11-class) or `... --model m4` (3-class); from Python, `predict(image_path)` returns `{class: probability}`.

## Intended use

* Course project and research prototype.
* At most a **second reader** that ranks lesions **already judged suspicious by a clinician**, on dermoscopic images taken like MILK10k's, to help prioritise which to examine first.

## Must not be used for

* Diagnosing a lesion or **ruling out cancer**. A "Benign" output must never override clinical concern.
* Screening the general population, consumer or self-check apps, phone photos, clinical close-ups (the model never saw them as input).
* Any decision without a dermatologist, any commercial use (data licence CC-BY-NC).

## Training data

MILK10k (MILK study team, ISIC Archive, doi:10.34970/262082): 5,240 lesions from 5 centres, 96 % confirmed by histopathology. My lesion-level split (seed 42): 3,742 train / 749 validation / 749 test lesions. Class shares: Malignant 69 %, Benign 28 %, Indeterminate 2 %; BCC alone is 48 %, MAL_OTH has 7 training lesions. Darker skin tones (classes 0–2 of the MILK10k scale, 0 = very dark … 5 = very light) are about 11 % of the lesions.

## Performance (my held-out test split, 749 lesions, opened once)

| Metric | 3-class model |
|---|---|
| Macro-F1 | **0.654** [95 % CI 0.570–0.727] (validation 0.676) |
| Accuracy | 0.806 (always-"Malignant" baseline: 0.693) |
| Malignant recall | 0.838 |
| Melanoma recall (called Malignant) | **0.40** |
| Benign called Benign | 0.765 |

| Metric | 11-class model |
|---|---|
| Validation macro-F1 | 0.390 |
| Official metric on my test split (threshold 0.5) | 0.383 [0.327, 0.432] |
| Official MILK10k leaderboard (479 blind lesions) | **0.420** (submission 1) |

**Submission 2: 11-class ensemble** (`checkpoints/ensemble_11class.pt`, `scripts/16_m5_ensemble.py`): equal-weight average of the 11-class ResNet-18 with flip TTA (dermoscopic), a frozen ViT-B/16 + linear head on the dermoscopic image, the same on the clinical close-up, and a logistic regression on age, sex, site, skin tone and 4 diagnostic MONET concepts per view (image_manipulation and the pen-ink, gel and hair concepts are excluded). Chosen on validation (0.462 vs 0.390 for the single model).

| Metric | Ensemble |
|---|---|
| Validation macro-F1 | 0.462 |
| Official metric on my test split | 0.478 [0.408, 0.531] |
| Official MILK10k leaderboard | **0.449** |

The ensemble needs both images and the metadata of a lesion; `predict(image_path)` uses the single image-only model. The fairness analysis above was done on the 3-class model, not repeated for the ensemble.

## Fairness (3-class model, test split)

| Group | n | Malignant recall |
|---|---:|---:|
| Skin tone 0–2 (darker) | 83 | **0.49** (17 / 35) |
| Skin tone 3–5 (lighter) | 666 | 0.86 (418 / 484) |
| Female / male | 281 / 468 | 0.81 / 0.85 |
| Age < 50 / 50–69 / 70+ | 124 / 335 / 288 | 0.66 / 0.84 / 0.87 |

The skin-tone gap is statistically clear (Fisher p < 0.000001). It is partly explained by case mix (60 % of malignant lesions on darker skin are melanomas) but BCC recall is also lower there. No evidence of a difference by sex. Tones 0 and 1 have only 16 test lesions together, too few for any conclusion.

## Known limitations and risks

1. **Misses 58 % of melanomas**, mostly well-circumscribed pigmented ones that look like nevi.
2. **Lower sensitivity on darker skin**, which is under-represented in the training data.
3. **Overconfident:** errors with probability 0.99–1.00 exist; probabilities are not calibrated.
4. **Prevalence and selection:** trained and tested on biopsy-selected lesions (69 % malignant). Predictive values do not transfer to a clinic or a screening population.
5. **Rare classes are not learned** (MAL_OTH F1 = 0, BEN_OTH and INF ≤ 0.22).
6. **Possible shortcuts:** pen markings, image manipulation, centre-specific colour; only partly checked.
7. **No patient-level split** (no patient id in the data) and no external validation on other cameras or centres.

## Ethical considerations

A tool that is less sensitive on darker skin and on melanoma would, if used to *reassure* patients, delay diagnosis for exactly the groups and the disease where delays cost most. Any further use would need: calibration, a sensitivity-first operating point chosen with clinicians, external validation on a representative population with enough darker-skin cases, and monitoring per subgroup.

## Citation

MILK study team. *MILK10k Benchmark.* ISIC Archive, 2025. doi:10.34970/262082 · Tschandl P, et al. *J Invest Dermatol.* 2026;146(2):357-364. doi:10.1016/j.jid.2025.06.1594
