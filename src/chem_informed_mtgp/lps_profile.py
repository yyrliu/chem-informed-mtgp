from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapters import CandidateSourceAdapter, DataSourceAdapter, IdentitySchemaAdapter
from .candidate_pool import iter_candidate_joint_batches
from .dataset_core import DatasetCore
from .fingerprint_kernel import assemble_feature_matrix, compute_a_site_fingerprints
from .registries import (
    FeatureBuilderSpec,
    FeatureRegistry,
    InferenceRegistry,
    InferenceRule,
    MetricRegistry,
    MetricSpec,
    NaNPolicyChain,
)


@dataclass(slots=True)
class LPSDataSourceAdapter(DataSourceAdapter):
    """Adapter around dash_lib.data_utils for the current LPS project."""

    db_path: str = "mydb.duckdb"

    def query_samples(
        self,
        *,
        filters: Mapping[str, Any] | str | None = None,
    ) -> pd.DataFrame:
        from dash_lib.data_utils import query_samples  # local import keeps core generic

        return query_samples(db_path=self.db_path, filters=filters)

    def load_dataset(
        self,
        unique_ids: Sequence[str],
        subgroup: str,
        **kwargs: Any,
    ) -> Any:
        from dash_lib.data_utils import load_dataset  # local import keeps core generic

        return load_dataset(
            unique_ids=unique_ids, subgroup=subgroup, db_path=self.db_path, **kwargs
        )


@dataclass(slots=True)
class LPSCandidateSourceAdapter(CandidateSourceAdapter):
    """Candidate source backed by iter_candidate_joint_batches."""

    param_config: Mapping[str, Mapping[str, Any]]
    a_site_to_smiles: Mapping[str, str]
    continuous_columns: Sequence[str]
    fingerprint_bits: int
    include_hash: bool = False
    snap: bool = True
    default_replace: bool = True
    default_seed: int | None = None

    def iter_candidates(
        self,
        *,
        batch_size: int,
        n: int | None = None,
        replace: bool = False,
        random_state: int | None = None,
    ) -> tuple[Any, tuple[Any, Any]]:
        actual_replace = replace if n is not None else self.default_replace
        actual_seed = self.default_seed if random_state is None else random_state

        return iter_candidate_joint_batches(
            param_config=self.param_config,
            a_site_to_smiles=self.a_site_to_smiles,
            continuous_columns=self.continuous_columns,
            fingerprint_bits=self.fingerprint_bits,
            batch_size=batch_size,
            include_hash=self.include_hash,
            snap=self.snap,
            n=n,
            replace=actual_replace,
            random_state=actual_seed,
        )


def make_lps_nan_policy_chain() -> NaNPolicyChain:
    chain = NaNPolicyChain()

    def _policy(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "asymmetry" in out.columns:
            out["asymmetry"] = out["asymmetry"].fillna(1e-4).clip(lower=1e-4)
        if "asymmetry_5FWHM" in out.columns:
            out["asymmetry_5FWHM"] = (
                out["asymmetry_5FWHM"].fillna(1e-4).clip(lower=1e-4) * 2.5
            )
        if "g_factor_max" in out.columns:
            out["g_factor_max"] = out["g_factor_max"].fillna(1e-5).clip(lower=1e-5)
        if "ldlb_area_ratio_gmax" in out.columns:
            out["ldlb_area_ratio_gmax"] = (
                out["ldlb_area_ratio_gmax"].fillna(0.0).clip(lower=0.005)
            )
        if "homogeneity" in out.columns:
            out["homogeneity"] = out["homogeneity"].fillna(0.0).clip(lower=0.001)
        return out

    chain.add(_policy)
    return chain


def make_lps_inference_registry(
    *,
    continuous_columns: Sequence[str],
    a_site_to_smiles: Mapping[str, str],
    fingerprint_bits: int,
    legacy_meombai_batches: Sequence[str] = ("3", "4", "5"),
) -> InferenceRegistry:
    registry = InferenceRegistry()

    def _backfill_a_site(df: pd.DataFrame) -> pd.DataFrame:
        if "a_site" not in df.columns or "sample_name" not in df.columns:
            return df
        out = df.copy()
        batch = out["sample_name"].astype(str).str.split("-").str[0]
        mask = out["a_site"].isna() & batch.isin(list(legacy_meombai_batches))
        out.loc[mask, "a_site"] = "MeOMBAI"
        return out

    def _infer_a_site_smiles(df: pd.DataFrame) -> pd.DataFrame:
        if "a_site" not in df.columns:
            return df
        out = df.copy()
        out["a_site_smiles"] = out["a_site"].map(dict(a_site_to_smiles))
        return out

    def _infer_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
        if "a_site_smiles" not in df.columns:
            return df
        out = df.copy()
        fp = compute_a_site_fingerprints(
            out,
            column="a_site_smiles",
            n_bits=fingerprint_bits,
        )
        out["a_site_fingerprint"] = list(fp)
        return out

    def _infer_config_hash(df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in continuous_columns if col not in df.columns]
        if missing:
            return df
        out = df.copy()
        if "config_hash" in out.columns and out["config_hash"].notna().all():
            return out

        def _hash_row(row: pd.Series) -> str:
            payload = {k: row[k] for k in continuous_columns}
            return sha256(pd.Series(payload).to_json().encode("utf-8")).hexdigest()

        out["config_hash"] = out.apply(_hash_row, axis=1)
        return out

    registry.register(
        InferenceRule(
            name="legacy_a_site_backfill",
            fn=_backfill_a_site,
            requires={"sample_name", "a_site"},
            provides={"a_site"},
        )
    )
    registry.register(
        InferenceRule(
            name="infer_a_site_smiles",
            fn=_infer_a_site_smiles,
            requires={"a_site"},
            provides={"a_site_smiles"},
        )
    )
    registry.register(
        InferenceRule(
            name="infer_a_site_fingerprint",
            fn=_infer_fingerprint,
            requires={"a_site_smiles"},
            provides={"a_site_fingerprint"},
        )
    )
    registry.register(
        InferenceRule(
            name="infer_config_hash",
            fn=_infer_config_hash,
            requires=set(continuous_columns),
            provides={"config_hash"},
        )
    )

    return registry


def make_lps_metric_registry(target_column: str = "score") -> MetricRegistry:
    registry = MetricRegistry()

    def _score(df: pd.DataFrame, _ctx: Any) -> pd.Series:
        series = (
            df["homogeneity"]
            * df["g_factor_max"]
            / (1.0 + df["ldlb_area_ratio_gmax"])
            * 1e5
        )
        return series.rename("score")

    def _cd_score(df: pd.DataFrame, _ctx: Any) -> pd.Series:
        series = df["g_factor_max"] / (1.0 + df["ldlb_area_ratio_gmax"]) * 5e3
        return series.rename("CD_score")

    def _homogeneity_score(df: pd.DataFrame, _ctx: Any) -> pd.Series:
        return (df["homogeneity"] * 100.0).rename("homogeneity_score")

    def _g_factor_score(df: pd.DataFrame, _ctx: Any) -> pd.Series:
        return (df["g_factor_max"] * 1e4).rename("g_factor_score")

    def _ldlb_score(df: pd.DataFrame, _ctx: Any) -> pd.Series:
        return (1.0 / (1.0 + df["ldlb_area_ratio_gmax"]) * 25.0).rename("ldlb_score")

    registry.register(MetricSpec(name="score", fn=_score))
    registry.register(MetricSpec(name="CD_score", fn=_cd_score))
    registry.register(MetricSpec(name="homogeneity_score", fn=_homogeneity_score))
    registry.register(MetricSpec(name="g_factor_score", fn=_g_factor_score))
    registry.register(MetricSpec(name="ldlb_score", fn=_ldlb_score))

    if target_column not in registry.names():
        raise KeyError(
            f"Unsupported target_column '{target_column}'. Supported: {registry.names()}"
        )

    return registry


def make_lps_feature_registry(
    *,
    continuous_columns: Sequence[str],
    fingerprint_bits: int,
    target_column: str,
) -> FeatureRegistry:
    registry = FeatureRegistry()

    def _joint_builder(df: pd.DataFrame) -> np.ndarray:
        return assemble_feature_matrix(
            df=df,
            continuous_columns=continuous_columns,
            fingerprint_column="a_site_smiles",
            n_bits=fingerprint_bits,
        )

    def _target_builder(df: pd.DataFrame) -> np.ndarray:
        return df[target_column].to_numpy(dtype=np.float64)

    registry.register(FeatureBuilderSpec(name="joint", fn=_joint_builder))
    registry.register(FeatureBuilderSpec(name="target", fn=_target_builder))
    return registry


@dataclass(slots=True)
class LPSProfileConfig:
    """Configuration bundle for creating a pre-wired LPS DatasetCore."""

    param_config: OrderedDict[str, Mapping[str, Any]]
    a_site_to_smiles: Mapping[str, str]
    continuous_columns: Sequence[str]
    fingerprint_bits: int
    target_column: str = "score"
    id_column: str = "sample_id"
    db_path: str = "mydb.duckdb"


def create_lps_dataset(config: LPSProfileConfig) -> DatasetCore:
    source = LPSDataSourceAdapter(db_path=config.db_path)
    schema = IdentitySchemaAdapter(required_columns=("sample_name",))

    dataset = DatasetCore(
        source=source,
        schema=schema,
        id_column=config.id_column,
        target_column=config.target_column,
        metric_registry=make_lps_metric_registry(config.target_column),
        feature_registry=make_lps_feature_registry(
            continuous_columns=config.continuous_columns,
            fingerprint_bits=config.fingerprint_bits,
            target_column=config.target_column,
        ),
        inference_registry=make_lps_inference_registry(
            continuous_columns=config.continuous_columns,
            a_site_to_smiles=config.a_site_to_smiles,
            fingerprint_bits=config.fingerprint_bits,
        ),
        nan_policies=make_lps_nan_policy_chain(),
    )
    return dataset


def build_manual_entries_from_mappings(
    *,
    manual_entries: Mapping[int, Sequence[int]],
    continuous_columns: Sequence[str],
    root_dir: str | Path,
    metric_defaults: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build manual rows from LPS-{batch}/data_mapping.json recipe entries.

    This mirrors the existing notebook pattern while keeping manual ingestion
    decoupled from DatasetCore itself.
    """

    defaults = {
        "homogeneity": 0.0,
        "asymmetry": 0.0,
        "g_factor_max": 0.0,
        "asymmetry_5FWHM": 0.0,
    }
    if metric_defaults:
        defaults.update(metric_defaults)

    rows: list[dict[str, Any]] = []
    root = Path(root_dir)

    for batch_num, sample_indices in manual_entries.items():
        mapping_path = root / f"LPS-{batch_num}" / "data_mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing data mapping file: {mapping_path}")

        with mapping_path.open("r", encoding="utf-8") as f:
            data_mapping = json.load(f)

        for sample_idx in sample_indices:
            entry = data_mapping.get(str(sample_idx))
            if entry is None:
                raise KeyError(
                    f"Sample {sample_idx} not found in mapping for batch {batch_num}."
                )

            recipe = entry["recipe"]
            row = {col: recipe.get(col) for col in continuous_columns}
            row["a_site"] = recipe.get("a_site")
            row["sample_name"] = f"{batch_num}-{int(sample_idx):02d}"
            row["session"] = f"LPS-{batch_num}"
            row["sample_number"] = int(sample_idx)
            row["manual_sample_id"] = f"manual-{row['sample_name']}"
            row["config_hash"] = sha256(
                json.dumps(recipe, sort_keys=True).encode("utf-8")
            ).hexdigest()
            row.update(defaults)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[*continuous_columns])

    manual_df = pd.DataFrame(rows)
    manual_df = manual_df.set_index("manual_sample_id", drop=False)
    return manual_df
