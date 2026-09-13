## Clinical Task

This project addresses **skin lesion classification** using the MILK10k dataset,
a collection of dermatological images paired with clinical metadata (age, sex,
anatomic site, skin tone, and diagnosis).

The goal is to build a model that can classify a skin lesion image into one of
three broad categories — **Benign**, **Malignant**, or **Indeterminate** — and,
at a finer level, into 11 specific diagnostic classes (e.g., BCC, MEL, NV, AKIEC).

This kind of system could support **early skin cancer detection**, helping
prioritize which lesions should be reviewed by a dermatologist first. Because
skin cancer diagnosis models have historically underperformed on darker skin
tones, this project also considers **fairness** across skin tone, age, and sex
as part of the evaluation.
```