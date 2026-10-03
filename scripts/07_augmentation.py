"""B7 - Augmentation: determinism proof, figure and justification table.

Outputs: figures/augmentation_examples.png (1 original + 7 augmented, 3 classes
incl. a rare one), reports/augmentation_table.md
"""
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import _bootstrap  # noqa: F401
from milk10k_pipeline import config, transforms
from milk10k_pipeline.reporting import df_to_markdown


def prove_eval_deterministic(isic_id: str) -> None:
    tf = transforms.eval_transform()
    with Image.open(config.image_path(isic_id)) as im:
        img = im.convert("RGB")
    a, b = tf(img), tf(img)
    assert torch.equal(a, b), "eval_transform is not deterministic!"
    print(f"eval_transform deterministic on {isic_id}: torch.equal -> True, shape {tuple(a.shape)}")


def augmentation_figure(train: pd.DataFrame) -> None:
    derm = train[train["image_type"] == "dermoscopic"]
    picks = []
    for dx in ("BCC", "NV", "VASC"):                  # frequent, frequent, rare
        picks.append(derm[derm["dx"] == dx].sample(1, random_state=config.SEED).iloc[0])
    aug = transforms.augment_only()
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
    for r, row in enumerate(picks):
        with Image.open(config.image_path(row["isic_id"])) as im:
            img = im.convert("RGB")
        base = transforms.T.Compose([transforms.T.Resize(config.RESIZE_SIZE),
                                     transforms.T.CenterCrop(config.IMG_SIZE)])(img)
        axes[r, 0].imshow(base)
        axes[r, 0].set_title(f"{row['dx']} - original", fontsize=9)
        for c in range(1, 8):
            axes[r, c].imshow(aug(img))
            axes[r, c].set_title(f"aug {c}", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("train_transform (before ToTensor/Normalize): 1 original + 7 random draws")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "augmentation_examples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/augmentation_examples.png")


def main() -> None:
    config.ensure_output_dirs()
    train = pd.read_csv(config.SPLIT_DIR / "train.csv")
    prove_eval_deterministic(train["isic_id"].iloc[0])
    augmentation_figure(train)
    tab = pd.DataFrame(transforms.AUGMENTATION_TABLE,
                       columns=["augmentation", "parameters", "why the label does not change"])
    (config.REPORT_DIR / "augmentation_table.md").write_text(
        "# Augmentation table (train only)\n\n" + df_to_markdown(tab) + "\n")
    print(tab.to_string())
    print("\ntrain_transform:", transforms.train_transform())
    print("eval_transform:", transforms.eval_transform())


if __name__ == "__main__":
    main()
