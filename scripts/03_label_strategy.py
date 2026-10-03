"""B3 - Write the final label mapping to outputs/label_map.json and print the counts behind it."""
import json

import pandas as pd

import _bootstrap  # noqa: F401
from milk10k_pipeline import config, data, labels


def main() -> None:
    config.ensure_output_dirs()
    lesions = data.build_lesion_table()
    print("Lesions per diagnosis_1:\n", lesions["diagnosis_1"].value_counts().to_string(), "\n")
    print("Lesions per dx (11 classes):\n", lesions["dx"].value_counts().to_string(), "\n")
    fine = lesions["dx"].map(labels.FINE_MERGE)
    print("Lesions per fine class (after merge):\n", fine.value_counts().to_string(), "\n")
    path = labels.save_label_map()
    print(f"saved {path}")
    print(json.dumps(labels.load_label_map()["fine"]["merge"], indent=1))


if __name__ == "__main__":
    main()
