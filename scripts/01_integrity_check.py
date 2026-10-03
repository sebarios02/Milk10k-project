"""B1 - Full-dataset integrity check.

Checks that all 10,480 metadata rows resolve to an existing image, runs
Image.verify() on EVERY image and records the image sizes.

Outputs: outputs/image_sizes.csv, outputs/image_size_summary.csv
Exit code 1 if any file is missing or unreadable (fail loudly).
"""
import sys

import _bootstrap  # noqa: F401
from milk10k_pipeline import config, data, integrity


def main() -> int:
    config.ensure_output_dirs()
    print(config.describe())
    images = data.load_metadata()
    print(f"metadata rows: {len(images)}")

    scan = integrity.scan_images(images)
    scan.to_csv(config.IMAGE_SIZES_CSV, index=False)

    missing = scan[~scan["exists"]]
    unreadable = scan[scan["exists"] & ~scan["readable"]]
    print(f"existing files : {scan['exists'].sum()} / {len(scan)}")
    print(f"missing files  : {len(missing)}")
    if len(missing):
        print(missing["isic_id"].head(20).to_string(index=False))
    print(f"unreadable     : {len(unreadable)}")
    if len(unreadable):
        print(unreadable[["isic_id", "error"]].to_string(index=False))

    summary = integrity.size_summary(scan)
    summary.to_csv(config.IMAGE_SIZE_SUMMARY_CSV, index=False)
    print(summary.T.to_string())

    if len(missing) or len(unreadable):
        print("\nFAIL: the image set is incomplete or corrupted.")
        return 1
    print("\nOK: all rows resolve to a readable image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
