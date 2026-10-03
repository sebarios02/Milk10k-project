"""B8 - Build the project DataLoaders and run the sanity checks.

- class weights from TRAIN -> outputs/class_weights.json
- batch shape / dtype / min / max
- label histogram of 20 train batches: plain shuffle vs WeightedRandomSampler
- time to load one epoch of the train set
- figures/train_batch_after_transforms.png (Session 2 visualizer, 16 images)
"""
import json
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from milk10k_pipeline import config, labels, loaders, visualizer


def label_hist(dl, n_batches, names):
    c = Counter()
    for i, (_, y, _) in enumerate(dl):
        c.update(y.tolist())
        if i + 1 == n_batches:
            break
    return {names[k]: c.get(k, 0) for k in range(len(names))}


def main() -> None:
    config.ensure_output_dirs()
    lm = labels.load_label_map()
    names = lm["primary"]["classes"]

    cw = loaders.compute_and_save_class_weights()
    print("class weights (train):", json.dumps(cw["primary"]["weights"]), "| fine:", json.dumps(cw["fine"]["weights"]))
    print("CrossEntropyLoss weight tensor:", loaders.weights_tensor("primary"))

    dls = loaders.make_dataloaders("y_primary", use_sampler=False)
    print({k: f"{len(v.dataset)} images / {len(v)} batches" for k, v in dls.items()})

    x, y, ids = next(iter(dls["train"]))
    print(f"batch: x {tuple(x.shape)} {x.dtype}, min {x.min():.3f}, max {x.max():.3f}, "
          f"mean {x.mean():.3f}; y {tuple(y.shape)} {y.dtype}; ids[:3] {list(ids[:3])}")
    xv, _, _ = next(iter(dls["val"]))
    print(f"val batch: {tuple(xv.shape)} {xv.dtype}")

    print("label histogram, 20 train batches, shuffle :", label_hist(dls["train"], 20, names))
    dls_s = loaders.make_dataloaders("y_primary", use_sampler=True)
    print("label histogram, 20 train batches, sampler :", label_hist(dls_s["train"], 20, names))

    t0 = time.perf_counter()
    n = 0
    for xb, _, _ in dls["train"]:
        n += len(xb)
    dt = time.perf_counter() - t0
    print(f"one train epoch: {n} images in {dt:.1f}s ({n / dt:.0f} img/s, "
          f"batch_size={config.BATCH_SIZE}, num_workers={config.NUM_WORKERS})")

    imgs = x[:16].permute(0, 2, 3, 1).numpy()     # CHW -> HWC for the Session 2 visualizer
    titles = [names[int(k)] for k in y[:16]]
    fig = visualizer.plot_sample_grid(imgs, titles, n_cols=4, title="Train batch after train_transform")
    fig.savefig(config.FIGURE_DIR / "train_batch_after_transforms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/train_batch_after_transforms.png")


if __name__ == "__main__":
    main()   # the __main__ guard is required for num_workers > 0 on Windows
