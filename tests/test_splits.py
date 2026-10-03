"""Property tests for split_lesions (Session 3 A3.2 / Milestone 1 B5). No images needed."""
import pytest

from milk10k_pipeline import data, splits

SEEDS = range(10)


@pytest.fixture(scope="module")
def lesions():
    return data.build_lesion_table()


def test_lesion_table_shape(lesions):
    assert len(lesions) == 5240
    assert lesions["lesion_id"].is_unique
    assert lesions[["derm_id", "clinical_id"]].notna().all().all()


@pytest.mark.parametrize("seed", SEEDS)
def test_no_overlap_and_sizes(lesions, seed):
    tr, va, te = splits.split_lesions(lesions, 0.15, 0.15, seed)
    splits.check_split(tr, va, te, len(lesions), 0.15, 0.15, tol_pp=1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_reproducible(lesions, seed):
    assert splits.split_lesions(lesions, 0.15, 0.15, seed) == splits.split_lesions(lesions, 0.15, 0.15, seed)


def test_different_seeds_differ(lesions):
    assert splits.split_lesions(lesions, 0.15, 0.15, 0)[2] != splits.split_lesions(lesions, 0.15, 0.15, 1)[2]


def test_rejects_image_level_table(lesions):
    images = data.load_images_table()
    with pytest.raises(ValueError):
        splits.split_lesions(images, 0.15, 0.15, 0)


def test_saved_split_files_are_leak_free():
    import pandas as pd
    from milk10k_pipeline import config
    paths = {n: config.SPLIT_DIR / f"{n}.csv" for n in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("run scripts/05_make_splits.py first")
    ids = {n: set(pd.read_csv(p)["lesion_id"]) for n, p in paths.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert sum(len(v) for v in ids.values()) == 5240
