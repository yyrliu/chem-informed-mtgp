from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .adapters import CandidateSourceAdapter, DataSourceAdapter, SchemaAdapter
from .registries import (
    FeatureRegistry,
    InferenceRegistry,
    MetricContext,
    MetricRegistry,
    NaNPolicyChain,
)


class DatasetStage(IntEnum):
    EMPTY = 0
    LOADED = 1
    ENRICHED = 2
    CLEANED = 3
    NORMALIZED = 4
    FEATURIZED = 5


@dataclass(slots=True)
class DatasetCore:
    """Project-agnostic dataset pipeline with in-place mutable stages."""

    source: DataSourceAdapter
    schema: SchemaAdapter
    id_column: str = "sample_id"
    target_column: str | None = None
    metric_registry: MetricRegistry = field(default_factory=MetricRegistry)
    feature_registry: FeatureRegistry = field(default_factory=FeatureRegistry)
    inference_registry: InferenceRegistry = field(default_factory=InferenceRegistry)
    nan_policies: NaNPolicyChain = field(default_factory=NaNPolicyChain)

    _raw_df: pd.DataFrame | None = field(default=None, init=False)
    _df: pd.DataFrame | None = field(default=None, init=False)
    _normalized_df: pd.DataFrame | None = field(default=None, init=False)
    _features: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _scalers: dict[str, MinMaxScaler] = field(default_factory=dict, init=False)
    _normalization_columns: dict[str, list[str]] = field(
        default_factory=dict, init=False
    )
    _stage: DatasetStage = field(default=DatasetStage.EMPTY, init=False)

    @staticmethod
    def _ensure_outlier_column(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "is_outlier" not in out.columns:
            out["is_outlier"] = False
        out["is_outlier"] = out["is_outlier"].fillna(False).astype(bool)
        return out

    def _require_stage(self, min_stage: DatasetStage) -> None:
        if self._stage < min_stage:
            raise RuntimeError(
                f"Operation requires stage >= {min_stage.name}, current stage is {self._stage.name}."
            )

    @property
    def stage(self) -> DatasetStage:
        return self._stage

    def load_from_query(
        self,
        *,
        filters: Mapping[str, Any] | str | None = None,
    ) -> DatasetCore:
        df = self.source.query_samples(filters=filters)
        df = self.schema.to_canonical(df)
        self.schema.validate(df)
        df = self._ensure_outlier_column(df)

        self._raw_df = df.copy()
        self._df = df.copy()
        self._normalized_df = None
        self._features.clear()
        self._stage = DatasetStage.LOADED
        return self

    def add_rows(
        self, rows: pd.DataFrame, *, source_tag: str = "manual"
    ) -> DatasetCore:
        self._require_stage(DatasetStage.LOADED)
        if self._df is None:
            raise RuntimeError("Dataset is not loaded.")

        rows_canonical = self.schema.to_canonical(rows)
        rows_canonical = self._ensure_outlier_column(rows_canonical)
        rows_canonical["row_source"] = source_tag

        if "row_source" not in self._df.columns:
            self._df["row_source"] = "query"
        self._df = self._ensure_outlier_column(self._df)

        self._df = pd.concat(
            [self._df, rows_canonical], axis=0, ignore_index=False, sort=False
        )
        self._normalized_df = None
        self._features.clear()
        self._stage = max(self._stage, DatasetStage.ENRICHED)
        return self

    def apply_inference_rules(self) -> DatasetCore:
        self._require_stage(DatasetStage.LOADED)
        if self._df is None:
            raise RuntimeError("Dataset is not loaded.")

        self._df = self.inference_registry.apply(self._df)
        self._normalized_df = None
        self._features.clear()
        self._stage = max(self._stage, DatasetStage.ENRICHED)
        return self

    def apply_nan_policies(self) -> DatasetCore:
        self._require_stage(DatasetStage.LOADED)
        if self._df is None:
            raise RuntimeError("Dataset is not loaded.")

        self._df = self.nan_policies.apply(self._df)
        self._normalized_df = None
        self._features.clear()
        self._stage = max(self._stage, DatasetStage.CLEANED)
        return self

    def add_metric(self, metric_name: str, *, overwrite: bool = False) -> DatasetCore:
        self._require_stage(DatasetStage.LOADED)
        if self._df is None:
            raise RuntimeError("Dataset is not loaded.")

        spec = self.metric_registry.get(metric_name)

        def _load_modalities(unique_ids: Sequence[str], subgroup: str) -> Any:
            return self.source.load_dataset(unique_ids, subgroup)

        context = MetricContext(load_modalities=_load_modalities)
        metric_values = spec.fn(self._df, context)
        if not isinstance(metric_values, pd.Series):
            raise TypeError(
                f"Metric '{metric_name}' must return a pandas Series, got {type(metric_values)}."
            )
        if metric_values.name is None:
            metric_values.name = metric_name

        if metric_values.name in self._df.columns and not overwrite:
            raise KeyError(
                f"Column '{metric_values.name}' already exists. Use overwrite=True to replace it."
            )

        self._df[metric_values.name] = metric_values.reindex(self._df.index)
        self._normalized_df = None
        self._features.clear()
        self._stage = max(self._stage, DatasetStage.ENRICHED)
        return self

    def set_outliers(
        self,
        *,
        column: Mapping[Any, Any] | None = None,
        mask: Sequence[bool] | pd.Series | np.ndarray | None = None,
        append: bool = False,
    ) -> DatasetCore:
        self._require_stage(DatasetStage.LOADED)
        if self._df is None:
            raise RuntimeError("Dataset is not loaded.")

        self._df = self._ensure_outlier_column(self._df)
        if self._normalized_df is not None:
            self._normalized_df = self._ensure_outlier_column(self._normalized_df)

        if append:
            outlier_mask = self._df["is_outlier"].to_numpy(dtype=bool).copy()
        else:
            outlier_mask = np.zeros(len(self._df), dtype=bool)

        if column is not None:
            if len(column) != 1:
                raise NotImplementedError(
                    "set_outliers(column=...) currently supports exactly one column key."
                )

            col_name, col_values = next(iter(column.items()))
            if col_name not in self._df.columns:
                raise KeyError(
                    f"Column '{col_name}' is required to set outliers by column values."
                )

            if isinstance(col_values, (str, bytes)):
                value_candidates = [col_values]
            elif isinstance(col_values, Sequence):
                value_candidates = list(col_values)
            else:
                value_candidates = [col_values]

            outlier_mask |= (
                self._df[col_name].isin(value_candidates).to_numpy(dtype=bool)
            )

        if mask is not None:
            if isinstance(mask, pd.Series):
                mask_values = (
                    mask.reindex(self._df.index).fillna(False).to_numpy(dtype=bool)
                )
            else:
                mask_values = np.asarray(mask, dtype=bool)
                if mask_values.shape[0] != len(self._df):
                    raise ValueError(
                        f"Mask length {mask_values.shape[0]} must match dataset length {len(self._df)}."
                    )
            outlier_mask |= mask_values

        self._df.loc[:, "is_outlier"] = outlier_mask
        if self._normalized_df is not None:
            self._normalized_df.loc[:, "is_outlier"] = outlier_mask

        # Existing feature arrays may include outliers; rebuild after outlier updates.
        self._features.clear()
        return self

    def normalize(
        self,
        *,
        columns: Sequence[str],
        key: str = "default",
        fit_on: str = "ranges",
        ranges: Mapping[str, tuple[float, float]] | None = None,
        external_df: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        clip: tuple[float, float] | None = (0.0, 1.0),
    ) -> DatasetCore | pd.DataFrame:
        """Backward-compatible wrapper around fit/transform normalization APIs.

        Prefer:
        - fit_normalizer(...): fit scaler + normalize internal dataset.
        - transform_with_normalizer(...): normalize external data with fitted scaler.
        """
        if data is not None:
            return self.transform_with_normalizer(
                data=data,
                columns=columns,
                key=key,
                clip=clip,
            )

        return self.fit_normalizer(
            columns=columns,
            key=key,
            fit_on=fit_on,
            ranges=ranges,
            external_df=external_df,
            clip=clip,
        )

    def fit_normalizer(
        self,
        *,
        columns: Sequence[str],
        key: str = "default",
        fit_on: str = "ranges",
        ranges: Mapping[str, tuple[float, float]] | None = None,
        external_df: pd.DataFrame | None = None,
        clip: tuple[float, float] | None = (0.0, 1.0),
        overwrite: bool = False,
    ) -> DatasetCore:
        self._require_stage(DatasetStage.LOADED)

        if self._df is None:
            raise RuntimeError("Dataset is not loaded.")

        cols = list(columns)

        missing = [col for col in cols if col not in self._df.columns]
        if missing:
            raise KeyError(f"Missing columns for normalization: {missing}")

        if fit_on == "ranges":
            if ranges is None:
                raise ValueError("ranges is required when fit_on='ranges'.")
            scaler_fit_df = pd.DataFrame(
                [[ranges[col][0] for col in cols], [ranges[col][1] for col in cols]],
                columns=cols,
            )
        elif fit_on == "external":
            if external_df is None:
                raise ValueError("external_df is required when fit_on='external'.")
            scaler_fit_df = external_df.loc[:, cols]
        elif fit_on == "current":
            scaler_fit_df = self._df.loc[:, cols]
        else:
            raise ValueError("fit_on must be one of: 'ranges', 'external', 'current'.")

        scaler = MinMaxScaler()
        scaler.fit(scaler_fit_df)

        normalized_df = self._df.copy()
        transformed = scaler.transform(normalized_df.loc[:, cols])
        if clip is not None:
            transformed = np.clip(transformed, clip[0], clip[1])
        normalized_df.loc[:, cols] = transformed

        if overwrite or self._normalized_df is None:
            self._normalized_df = normalized_df

        else:
            # keep the self._normalized_df, only update the normalized columns
            self._normalized_df.loc[:, cols] = normalized_df
            print(
                f"Warning: Feature not yet complete. self._scalers['{key}'] is being overwritten with new columns {cols} without changing self._normalized_df reference."
            )

        self._scalers[key] = scaler
        self._normalization_columns[key] = cols
        self._features.clear()
        self._stage = max(self._stage, DatasetStage.NORMALIZED)
        return self

    def transform_with_normalizer(
        self,
        *,
        data: pd.DataFrame,
        columns: Sequence[str],
        key: str = "default",
        clip: tuple[float, float] | None = (0.0, 1.0),
    ) -> pd.DataFrame:
        """Transform external rows with an already-fitted normalizer."""
        self._require_stage(DatasetStage.LOADED)

        cols = list(columns)
        if key not in self._scalers:
            raise RuntimeError(
                f"No fitted scaler found for key '{key}'. Fit with fit_normalizer(...) first."
            )

        missing = [col for col in cols if col not in data.columns]
        if missing:
            raise KeyError(f"Missing columns for normalization: {missing}")

        scaler = self._scalers[key]
        scaler_cols = self._normalization_columns.get(key)
        if scaler_cols is None:
            raise RuntimeError(
                f"Normalization columns for key '{key}' are unavailable."
            )
        if list(scaler_cols) != cols:
            raise ValueError(
                f"Requested columns {cols} do not match fitted scaler columns {list(scaler_cols)} for key '{key}'."
            )

        out = data.copy()
        transformed = scaler.transform(out.loc[:, cols])
        if clip is not None:
            transformed = np.clip(transformed, clip[0], clip[1])
        out.loc[:, cols] = transformed
        return out

    def build_feature(self, name: str, *, from_normalized: bool = True) -> DatasetCore:
        self._require_stage(DatasetStage.LOADED)
        spec = self.feature_registry.get(name)
        df = self.get_df(normalized=from_normalized)
        feature = spec.fn(df)
        self._features[name] = np.asarray(feature)
        self._stage = max(self._stage, DatasetStage.FEATURIZED)
        return self

    def get_df(
        self,
        *,
        normalized: bool = False,
        copy: bool = True,
        include_outliers: bool = False,
    ) -> pd.DataFrame:
        self._require_stage(DatasetStage.LOADED)
        if normalized:
            if self._normalized_df is None:
                raise RuntimeError(
                    "Normalized dataframe is unavailable. Call normalize() first."
                )
            df = self._normalized_df
        else:
            if self._df is None:
                raise RuntimeError("Dataset is not loaded.")
            df = self._df

        df = self._ensure_outlier_column(df)
        if not include_outliers:
            df = df.loc[~df["is_outlier"].to_numpy(dtype=bool)]

        return df.copy() if copy else df

    def get_array(
        self,
        columns: Sequence[str],
        *,
        normalized: bool = False,
        include_outliers: bool = False,
        dtype: Any = np.float64,
    ) -> np.ndarray:
        df = self.get_df(
            normalized=normalized,
            copy=False,
            include_outliers=include_outliers,
        )
        return df.loc[:, list(columns)].to_numpy(dtype=dtype)

    def get_feature(self, name: str) -> np.ndarray:
        if name not in self._features:
            raise KeyError(
                f"Feature '{name}' is not built. Available: {sorted(self._features)}"
            )
        return self._features[name]

    def iter_candidate_batches(
        self,
        candidate_source: CandidateSourceAdapter,
        *,
        batch_size: int,
        n: int | None = None,
        replace: bool = False,
        random_state: int | None = None,
    ) -> tuple[Any, tuple[Any, Any]]:
        return candidate_source.iter_candidates(
            batch_size=batch_size,
            n=n,
            replace=replace,
            random_state=random_state,
        )
