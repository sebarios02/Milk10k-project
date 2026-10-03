# Augmentation table (train only)

| augmentation | parameters | why the label does not change |
|---|---|---|
| Resize + RandomResizedCrop | shorter side 256, crop 224, scale=(0.75, 1.0), ratio=(0.9, 1.1) | Simulates different zoom/framing of the same lesion; >=75% of the area is kept so the lesion stays in view. |
| RandomHorizontalFlip | p=0.5 | Lesions have no left/right orientation; a mirrored nevus is still a nevus. |
| RandomVerticalFlip | p=0.5 | Same argument: dermoscopy has no 'up'. |
| RandomRot90 | k in {0,1,2,3} | Orientation-free; lossless (no interpolation, no padding artefacts). |
| ColorJitter | brightness=0.1, contrast=0.1, saturation=0.05, hue=0 | Mimics lighting/camera differences. Hue is NOT jittered: the A3.5 audit shows that hue=0.02 already shifts lesion hue by 3.2 deg (gap between classes: 3.45 deg) and hue>=0.05 exceeds the gap, and colour (blue-white veil, red vessels) is diagnostic. |
| Normalize | train-split mean/std | Not an augmentation: rescales intensities, identical for every image. |
