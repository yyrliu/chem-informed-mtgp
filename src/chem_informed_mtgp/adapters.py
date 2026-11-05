from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Read tabular sample rows and optional modality datasets."""

    def query_samples(
        self,
        *,
        filters: Mapping[str, Any] | str | None = None,
    ) -> pd.DataFrame: ...

    def load_dataset(
        self,
        unique_ids: Sequence[str],
        subgroup: str,
        **kwargs: Any,
    ) -> Any: ...


@runtime_checkable
class SchemaAdapter(Protocol):
    """Map project-specific dataframes into a canonical view used by DatasetCore."""

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def validate(self, df: pd.DataFrame) -> None: ...


@runtime_checkable
class CandidateSourceAdapter(Protocol):
    """Yield candidate tables and their model-ready feature blocks in batches."""

    def iter_candidates(
        self,
        *,
        batch_size: int,
        n: int | None = None,
        replace: bool = False,
        random_state: int | None = None,
    ) -> tuple[Any, tuple[Any, Any]]: ...


@dataclass(slots=True)
class IdentitySchemaAdapter:
    """Default schema adapter for projects that already use canonical column names."""

    required_columns: tuple[str, ...] = ()

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def validate(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
