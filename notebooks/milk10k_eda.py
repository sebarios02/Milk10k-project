"""
MILK10k Exploratory Data Analysis
Assumes metadata.csv and training_gt.csv are in DATA_DIR (adjust path below).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
DATA_DIR = Path(".")  # adjust if needed

df_metadata = pd.read_csv(DATA_DIR / "metadata.csv")
gt = pd.read_csv(DATA_DIR / "supplements" / "training_gt.csv")

CLASS_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
              "INF", "MAL_OTH", "MEL", "NV", "SCCKA"]

# gt is one-hot encoded per class -> convert to single label column
gt["diagnosis_class"] = gt[CLASS_COLS].idxmax(axis=1)

# merge metadata + gt on lesion_id
df = df_metadata.merge(gt[["lesion_id", "diagnosis_class"]], on="lesion_id", how="inner")

# ---------------------------------------------------------
# 2. Samples per class
# ---------------------------------------------------------
class_counts = df["diagnosis_class"].value_counts()
print("Samples per class:\n", class_counts, "\n")

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar", color="steelblue")
plt.title("Number of Samples per Class")
plt.xlabel("Diagnosis Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("samples_per_class.png")
plt.show()

# ---------------------------------------------------------
# 3. Samples per subclass (diagnosis_1, most granular ISIC-DX label)
# ---------------------------------------------------------
if "diagnosis_1" in df.columns:
    subclass_counts = df["diagnosis_1"].value_counts()
    print("Samples per subclass:\n", subclass_counts, "\n")

    plt.figure(figsize=(10, 6))
    subclass_counts.plot(kind="bar", color="darkorange")
    plt.title("Number of Samples per Subclass")
    plt.xlabel("Subclass (diagnosis_1)")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("samples_per_subclass.png")
    plt.show()

# ---------------------------------------------------------
# 4. Age range per class
# ---------------------------------------------------------
age_col = "age_approx" if "age_approx" in df.columns else "age_approx"

plt.figure(figsize=(10, 6))
df.boxplot(column=age_col, by="diagnosis_class", rot=90)
plt.title("Age Range per Class")
plt.suptitle("")
plt.xlabel("Diagnosis Class")
plt.ylabel("Age (approx)")
plt.tight_layout()
plt.savefig("age_range_per_class.png")
plt.show()

# ---------------------------------------------------------
# 5. Sex distribution per class
# ---------------------------------------------------------
sex_dist = df.groupby(["diagnosis_class", "sex"]).size().unstack(fill_value=0)
print("Sex distribution per class:\n", sex_dist, "\n")

sex_dist.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Sex Distribution per Class")
plt.xlabel("Diagnosis Class")
plt.ylabel("Count")
plt.legend(title="Sex")
plt.tight_layout()
plt.savefig("sex_distribution_per_class.png")
plt.show()