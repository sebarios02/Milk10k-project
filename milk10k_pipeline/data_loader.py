from pathlib import Path
from typing import Optional, Iterator, Tuple, List

import numpy as np
import pandas as pd

from .preprocessing import process_batch


class MILK10kDataLoader:

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_dir: str,
        label_col: str,
        id_col: str = "isic_id",
        path_suffix: str = ".jpg",
        batch_size: int = 32,
        target_size: Tuple[int, int] = (128, 128),
        color_space: str = "rgb",
        normalization: Optional[str] = "minmax",
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ):
        self.image_dir = Path(image_dir)
        self.label_col = label_col
        self.id_col = id_col
        self.path_suffix = path_suffix
        self.batch_size = batch_size
        self.target_size = target_size
        self.color_space = color_space
        self.normalization = normalization
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

        self.metadata = self._filter_available(metadata)

    # -- Only draw from images actually available locally --------------------
    def _filter_available(self, metadata: pd.DataFrame) -> pd.DataFrame:
        exists_mask = metadata[self.id_col].apply(
            lambda x: (self.image_dir / f"{x}{self.path_suffix}").exists()
        )
        n_total = len(metadata)
        n_available = int(exists_mask.sum())
        if n_available < n_total:
            print(f"[MILK10kDataLoader] {n_total - n_available} of {n_total} rows "
                  f"have no matching image file in {self.image_dir} — excluded.")
        return metadata.loc[exists_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, List]]:
        n = len(self.metadata)
        order = self._rng.permutation(n) if self.shuffle else np.arange(n)

        for start in range(0, n, self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            batch_rows = self.metadata.iloc[batch_idx]

            result = process_batch(
                batch_rows,
                image_dir=self.image_dir,
                path_col=self.id_col,
                label_col=self.label_col,
                path_suffix=self.path_suffix,
                target_size=self.target_size,
                color_space=self.color_space,
                normalization=self.normalization,
            )
            # process_batch already skips/report bad files; a batch loader
            # just needs to hand back whatever loaded successfully.
            yield result.images, result.labels
