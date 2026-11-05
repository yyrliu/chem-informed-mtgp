from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset_core import DatasetCore
from .strategies import (
    FixedSizeChunking,
    GroupByColumnKFoldSplit,
    RowKFoldSplit,
    SplitStrategy,
)


@dataclass(slots=True)
class DataLoaderCore:
    """PyTorch-style data loader facade over DatasetCore."""

    dataset: DatasetCore

    def split_cv(
        self,
        *,
        group_col: str | None = None,
        strategy: SplitStrategy | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        df = self.dataset.get_df(normalized=False, copy=False)

        if strategy is None:
            strategy = (
                GroupByColumnKFoldSplit() if group_col is not None else RowKFoldSplit()
            )

        return [
            (tr.astype(int), te.astype(int))
            for tr, te in strategy.split(df, group_col=group_col)
        ]

    def iter_optimizer_chunks(
        self,
        source: Iterable[tuple[pd.DataFrame, np.ndarray]],
        *,
        chunk_size: int,
        n_batch: int | None = None,
        drop_remainder: bool = True,
    ) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
        strategy = FixedSizeChunking(drop_remainder=drop_remainder)
        multiple_of = n_batch if n_batch and n_batch > 0 else None
        yield from strategy.iter_chunks(
            source,
            chunk_size=chunk_size,
            multiple_of=multiple_of,
        )
