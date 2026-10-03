# Session 2 findings re-checked on the full dataset

| Session 2 finding | Still true on the full dataset? | Consequence for the pipeline |
|---|---|---|
| The dataset has 10 diagnostic classes (Session 2 report). | NO - 11 classes. VASC (47 lesions) was lost because the class list was hard-coded. | Class names are now always read from training_gt.csv (data.load_gt). |
| diagnosis_confirm_type is the most strongly associated field (clinical assessment ~ benign moles). | YES - clinical-assessment images are 87% Benign vs 26% for histopathology. | Leaks how the label was obtained -> excluded from model inputs (with diagnosis_2-4, concomitant_biopsy). |
| Age separates classes: NV young (median ~40), SCCKA / MAL_OTH old (~65-70). | YES - median age NV 40, SCCKA 70, MAL_OTH 75, BCC 65. | Legitimate clinical prior: allowed as an OPTIONAL metadata input later, but image-only models are evaluated first so we know what the pixels contribute. |
| anatom_site_general is informative (chi2 p ~ 1e-74). | YES, but 37% of lesions have no site and missingness itself varies by class (A1.2 q4). | Encode missing as an explicit 'unknown' level; never impute a site. |
| Imbalance about 280:1 (BCC vs MAL_OTH). | YES - 2522:9 = 280:1 at lesion level. | Class-weighted loss / WeightedRandomSampler from TRAIN counts; macro-F1 + balanced accuracy as metrics. |
| SCCKA is the darkest class in all three channels (25 images per class). | Re-checked with up to 100 images/class: darkest class is SCCKA (gray 135); spread between class means is 15 grey levels (images used per class: 18-100). | Colour differences are small and may come from the acquisition site -> normalise with TRAIN stats, no hue jitter, and check later that the model is not using global brightness. |
| (new) image_manipulation was never inspected in Session 2. | 'altered' images are 13% Indeterminate vs 2% among unaltered ones. | Potential shortcut: not a model input; report results with/without altered images. |
