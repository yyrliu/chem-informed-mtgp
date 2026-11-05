from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


class SplitStrategy(Protocol):
    """Produce (train_row_indices, test_row_indices) splits."""

    def split(
        self,
        df: pd.DataFrame,
        *,
        group_col: str | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


@dataclass(slots=True)
class RowKFoldSplit:
    """Standard row-wise KFold split."""

    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = 42

    def split(
        self,
        df: pd.DataFrame,
        *,
        group_col: str | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        del group_col
        if len(df) < 2:
            raise ValueError("Need at least 2 rows for CV splitting.")
        n_splits = min(self.n_splits, len(df))
        if n_splits < 2:
            raise ValueError("Need at least 2 splits.")

        kf = KFold(
            n_splits=n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        index = np.arange(len(df), dtype=int)
        yield from kf.split(index)


@dataclass(slots=True)
class GroupByColumnKFoldSplit:
    """Group-aware KFold split keyed by a user-provided column."""

    n_splits: int = 5

    def split(
        self,
        df: pd.DataFrame,
        *,
        group_col: str | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if not group_col:
            raise ValueError("group_col is required for GroupByColumnKFoldSplit.")
        if group_col not in df.columns:
            raise KeyError(f"Column '{group_col}' not found in dataframe.")

        groups = df[group_col].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            raise ValueError(
                f"Need at least 2 unique groups in '{group_col}', found {unique_groups.size}."
            )

        n_splits = min(self.n_splits, unique_groups.size)
        if n_splits < 2:
            raise ValueError("Need at least 2 splits.")

        splitter = GroupKFold(n_splits=n_splits)
        index = np.arange(len(df), dtype=int)
        yield from splitter.split(index, groups=groups)


class ChunkStrategy(Protocol):
    """Chunk a stream of (DataFrame, ndarray) batches into optimizer-friendly pieces."""

    def iter_chunks(
        self,
        source: Iterable[tuple[pd.DataFrame, np.ndarray]],
        *,
        chunk_size: int,
        multiple_of: int | None = None,
    ) -> Iterator[tuple[pd.DataFrame, np.ndarray]]: ...


@dataclass(slots=True)
class FixedSizeChunking:
    """Chunk by rows while preserving order across source batches."""

    drop_remainder: bool = True

    def iter_chunks(
        self,
        source: Iterable[tuple[pd.DataFrame, np.ndarray]],
        *,
        chunk_size: int,
        multiple_of: int | None = None,
    ) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        target_size = chunk_size
        if multiple_of is not None and multiple_of > 0:
            target_size = (chunk_size // multiple_of) * multiple_of
            if target_size <= 0:
                raise ValueError(
                    "chunk_size must be >= multiple_of when multiple_of is provided."
                )

        df_parts: list[pd.DataFrame] = []
        x_parts: list[np.ndarray] = []
        buffered_rows = 0

        for df_batch, x_batch in source:
            if len(df_batch) != len(x_batch):
                raise ValueError("Dataframe and feature matrix row counts must match.")
            if len(df_batch) == 0:
                continue

            df_parts.append(df_batch)
            x_parts.append(x_batch)
            buffered_rows += len(df_batch)

            while buffered_rows >= target_size:
                df_all = pd.concat(df_parts, axis=0, ignore_index=False)
                x_all = np.vstack(x_parts)

                df_emit = df_all.iloc[:target_size].copy()
                x_emit = x_all[:target_size].copy()
                yield df_emit, x_emit

                df_keep = df_all.iloc[target_size:].copy()
                x_keep = x_all[target_size:].copy()

                df_parts = [df_keep] if len(df_keep) else []
                x_parts = [x_keep] if len(x_keep) else []
                buffered_rows = len(df_keep)

        if not self.drop_remainder and buffered_rows:
            df_all = pd.concat(df_parts, axis=0, ignore_index=False)
            x_all = np.vstack(x_parts)
            yield df_all.copy(), x_all.copy()
