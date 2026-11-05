from __future__ import annotations

import numpy as np
import pandas as pd

from chem_informed_mtgp.adapters import IdentitySchemaAdapter
from chem_informed_mtgp.dataloader_core import DataLoaderCore
from chem_informed_mtgp.dataset_core import DatasetCore
from chem_informed_mtgp.registries import FeatureBuilderSpec, FeatureRegistry


class _DummySource:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def query_samples(self, *, filters=None):
        del filters
        return self._df.copy()

    def load_dataset(self, unique_ids, subgroup, **kwargs):
        del unique_ids, subgroup, kwargs
        return None


def test_dataset_core_external_normalization_and_feature_build():
    df = pd.DataFrame(
        {
            "sample_name": ["A-01", "A-02", "B-01"],
            "group": ["A", "A", "B"],
            "x1": [10.0, 20.0, 30.0],
            "x2": [100.0, 200.0, 300.0],
        }
    )
    source = _DummySource(df)
    schema = IdentitySchemaAdapter(required_columns=("sample_name", "x1", "x2"))
    feature_registry = FeatureRegistry()
    feature_registry.register(
        FeatureBuilderSpec(
            name="xy",
            fn=lambda d: d[["x1", "x2"]].to_numpy(dtype=np.float64),
        )
    )

    dataset = DatasetCore(
        source=source,
        schema=schema,
        feature_registry=feature_registry,
    )
    dataset.load_from_query()

    ref_df = pd.DataFrame({"x1": [0.0, 40.0], "x2": [0.0, 400.0]})
    dataset.fit_normalizer(columns=["x1", "x2"], fit_on="external", external_df=ref_df)

    normalized = dataset.get_df(normalized=True)
    assert np.isclose(normalized.loc[0, "x1"], 0.25)
    assert np.isclose(normalized.loc[2, "x2"], 0.75)

    dataset.build_feature("xy", from_normalized=True)
    xy = dataset.get_feature("xy")
    assert xy.shape == (3, 2)
    assert np.isclose(xy[1, 0], 0.5)


def test_dataset_core_normalize_external_data_with_stored_scaler():
    df = pd.DataFrame(
        {
            "sample_name": ["A-01", "A-02"],
            "x1": [10.0, 20.0],
            "x2": [100.0, 200.0],
        }
    )
    dataset = DatasetCore(
        source=_DummySource(df),
        schema=IdentitySchemaAdapter(required_columns=("sample_name", "x1", "x2")),
    ).load_from_query()

    # Fit scaler against declared ranges and cache it in dataset._scalers.
    dataset.fit_normalizer(
        columns=["x1", "x2"],
        fit_on="ranges",
        ranges={"x1": (0.0, 40.0), "x2": (0.0, 400.0)},
    )

    meshgrid_df = pd.DataFrame(
        {
            "x1": [0.0, 10.0, 40.0],
            "x2": [0.0, 100.0, 400.0],
            "other": [1, 2, 3],
        }
    )
    normalized_meshgrid = dataset.transform_with_normalizer(
        data=meshgrid_df,
        columns=["x1", "x2"],
    )

    assert np.allclose(normalized_meshgrid["x1"].to_numpy(), [0.0, 0.25, 1.0])
    assert np.allclose(normalized_meshgrid["x2"].to_numpy(), [0.0, 0.25, 1.0])
    assert np.allclose(meshgrid_df["x1"].to_numpy(), [0.0, 10.0, 40.0])


def test_dataloader_group_split_and_chunking():
    df = pd.DataFrame(
        {
            "sample_name": [f"S-{i:02d}" for i in range(10)],
            "cv_group": ["g1"] * 3 + ["g2"] * 3 + ["g3"] * 4,
            "x": np.arange(10, dtype=float),
        }
    )
    dataset = DatasetCore(
        source=_DummySource(df),
        schema=IdentitySchemaAdapter(required_columns=("sample_name", "cv_group", "x")),
    ).load_from_query()

    loader = DataLoaderCore(dataset)
    splits = loader.split_cv(group_col="cv_group")
    assert len(splits) >= 2

    for train_idx, test_idx in splits:
        train_groups = set(df.iloc[train_idx]["cv_group"])
        test_groups = set(df.iloc[test_idx]["cv_group"])
        assert train_groups.isdisjoint(test_groups)

    source_batches = [
        (
            pd.DataFrame({"sample_name": ["a", "b", "c"]}),
            np.arange(6, dtype=float).reshape(3, 2),
        ),
        (
            pd.DataFrame({"sample_name": ["d", "e", "f", "g"]}),
            np.arange(8, 16, dtype=float).reshape(4, 2),
        ),
    ]

    chunks = list(
        loader.iter_optimizer_chunks(
            source_batches,
            chunk_size=6,
            n_batch=3,
            drop_remainder=True,
        )
    )
    assert len(chunks) == 1
    chunk_df, chunk_x = chunks[0]
    assert len(chunk_df) == 6
    assert chunk_x.shape == (6, 2)
