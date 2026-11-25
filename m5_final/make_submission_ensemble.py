"""Second leaderboard submission: the ensemble chosen on validation by scripts/16_m5_ensemble.py.

Same test data and checks as make_submission.py, but per lesion it uses BOTH images and the
official test metadata (the same columns as training_input.csv):

    python m5_final/make_submission_ensemble.py      -> m5_final/submission.csv
    (the first, single-model file is kept as m5_final/submission_v1_resnet.csv)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from milk10k_pipeline import deep  # noqa: E402
from milk10k_pipeline import ensemble as ens  # noqa: E402
from m5_final.make_submission import CLASSES, find_metadata, image_file  # noqa: E402

OUT = Path(__file__).resolve().parent / "submission.csv"


def images(meta: pd.DataFrame, lesions, derm: bool) -> torch.Tensor:
    sel = meta[(meta["image_type"] == "dermoscopic") == derm].drop_duplicates("lesion_id").set_index("lesion_id")
    return torch.stack([deep.load_image_224(image_file(l, sel.loc[l, "isic_id"])) for l in lesions])


@torch.no_grad()
def main():
    ck = torch.load(deep.CHECKPOINT_DIR / "ensemble_11class.pt", map_location="cpu", weights_only=False)
    assert ck["classes"] == CLASSES
    w = {m: v for m, v in ck["weights"].items() if v > 0}
    print("ensemble:", ck["choice"], {m: round(v, 2) for m, v in w.items()})

    meta = pd.read_csv(find_metadata())
    lesions = meta.drop_duplicates("lesion_id")["lesion_id"].tolist()
    print(f"{len(meta)} test images, {len(lesions)} lesions")

    P = {}
    if "resnet_derm" in w or "resnet_plain" in w or "vit_derm" in w:
        Xd = images(meta, lesions, derm=True)
    if "resnet_derm" in w or "resnet_plain" in w:
        model, _ = deep.load_checkpoint(deep.CHECKPOINT_DIR / ck["resnet_checkpoint"])
        if "resnet_derm" in w:
            P["resnet_derm"] = ens.predict_tta(model, Xd)
        if "resnet_plain" in w:
            P["resnet_plain"] = deep.predict_proba(model, Xd)
    if "vit_derm" in w or "vit_clin" in w:
        vit = deep.frozen_backbone("vit_b_16")
        for m, derm in (("vit_derm", True), ("vit_clin", False)):
            if m not in w:
                continue
            X = Xd if derm else images(meta, lesions, derm=False)
            F = torch.cat([vit(deep.prep(X[i:i + 64])).cpu() for i in range(0, len(X), 64)])
            head = torch.nn.Linear(F.shape[1], len(CLASSES))
            head.load_state_dict(ck["vit_heads"][m])
            P[m] = deep.head_proba(head, F)
            print(f"  {m}: done")
    if "meta" in w:
        meta_model = pickle.loads(ck["meta_model"])
        P["meta"] = meta_model.predict_proba(ens.meta_features(meta, lesions))

    fused = sum(v * P[m] for m, v in w.items())
    fused = deep.postprocess(fused, ck["postprocess"])

    sub = pd.DataFrame(fused, columns=CLASSES)
    sub.insert(0, "lesion_id", lesions)                   # the ISIC validator requires "lesion_id"
    assert list(sub.columns) == ["lesion_id"] + CLASSES
    assert len(sub) == 479 and sub.lesion_id.is_unique, f"{len(sub)} rows"
    assert sub[CLASSES].isna().sum().sum() == 0
    assert ((sub[CLASSES] >= 0) & (sub[CLASSES] <= 1)).all().all()

    old = OUT.with_name("submission_v1_resnet.csv")
    if OUT.exists() and not old.exists():
        OUT.rename(old)
        print("first submission kept as", old.name)
    sub.to_csv(OUT, index=False)
    print(f"OK -> {OUT}  ({len(sub)} rows x {sub.shape[1]} columns)")
    print("predicted classes (>= 0.5):", (sub[CLASSES] >= 0.5).sum().to_dict())


if __name__ == "__main__":
    main()
