"""Dataset / transform tests. Image-dependent tests are skipped if images are absent."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from milk10k_pipeline import config, data  # noqa: E402
from milk10k_pipeline.datasets import LesionDataset, MILK10kImageDataset, aggregate_predictions  # noqa: E402
from milk10k_pipeline.transforms import eval_transform  # noqa: E402


def test_aggregate_predictions_synthetic():
    table = pd.DataFrame({"lesion_id": ["L1", "L1", "L2", "L2"]})
    probs = np.array([[0.9, 0.1, 0.0],
                      [0.3, 0.7, 0.0],      # L1 mean = [0.6, 0.4, 0] -> class 0
                      [0.2, 0.2, 0.6],
                      [0.0, 0.6, 0.4]])     # L2 mean = [0.1, 0.4, 0.5] -> class 2
    out = aggregate_predictions(probs, table, ["B", "I", "M"])
    assert list(out.index) == ["L1", "L2"]
    assert np.allclose(out.loc["L1", ["p_B", "p_I", "p_M"]], [0.6, 0.4, 0.0])
    assert out["pred"].tolist() == [0, 2]
    assert out["n_images"].tolist() == [2, 2]


def test_missing_file_fails_loudly(tmp_path):
    table = pd.DataFrame({"lesion_id": ["L1"], "derm_id": ["NOPE_1"], "clinical_id": ["NOPE_2"],
                          "diagnosis_1": ["Benign"]})
    with pytest.raises(FileNotFoundError):
        LesionDataset(table, img_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        MILK10kImageDataset(pd.DataFrame({"isic_id": ["NOPE"], "y_primary": [0]}), img_dir=tmp_path)


def _available_lesions(n):
    les = data.build_lesion_table()
    ok = data.available_mask(les["derm_id"]) & data.available_mask(les["clinical_id"])
    les = les[ok]
    if len(les) < n:
        pytest.skip(f"need {n} lesions with both images on disk")
    return les.head(n)


def test_lesion_batch_shapes():
    les = _available_lesions(8)
    ds = LesionDataset(les, label_col="diagnosis_1", transform=eval_transform())
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8)))
    for v in ("derm", "clinical"):
        assert batch[v].shape == (8, 3, config.IMG_SIZE, config.IMG_SIZE)
    assert list(batch["lesion_id"]) == les["lesion_id"].tolist()


def test_eval_transform_deterministic():
    les = _available_lesions(1)
    from PIL import Image
    img = Image.open(config.image_path(les["derm_id"].iloc[0])).convert("RGB")
    tf = eval_transform()
    assert torch.equal(tf(img), tf(img))
