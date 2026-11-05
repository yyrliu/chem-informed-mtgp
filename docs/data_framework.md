# DatasetCore and DataLoaderCore Guide

This document describes the new PyTorch-style data framework added in src/chem_informed_mtgp.

The framework is split into:
- Generic core classes for any project
- Pluggable adapters, registries, and strategies
- An LPS profile that wires the current project defaults

## 1. File Map

- src/chem_informed_mtgp/dataset_core.py
- src/chem_informed_mtgp/dataloader_core.py
- src/chem_informed_mtgp/adapters.py
- src/chem_informed_mtgp/registries.py
- src/chem_informed_mtgp/strategies.py
- src/chem_informed_mtgp/lps_profile.py

## 2. Design Goals

- Project-agnostic core: no hard-coded LPS schema in core classes
- In-place mutable workflow: easy notebook use
- Extensible behavior through policies and registries
- Support both tabular data and extra modalities
- Reusable in future projects with only adapter/profile changes

## 3. Core Concepts

### 3.1 DatasetCore

DatasetCore is the main stateful data pipeline class.

Lifecycle stages:
- EMPTY
- LOADED
- ENRICHED
- CLEANED
- NORMALIZED
- FEATURIZED

Main responsibilities:
- Load data from source adapters
- Add rows manually
- Apply inference rules
- Apply NaN policy chain
- Add derived metrics
- Normalize selected columns
- Build and store named feature arrays
- Return data as DataFrame or NumPy arrays
- Iterate candidate batches through a candidate adapter

### 3.2 DataLoaderCore

DataLoaderCore is a light facade over DatasetCore for:
- Cross-validation splitting
- Optimizer-friendly chunk iteration

Default behavior:
- If group_col is set, group-aware split is used
- If group_col is not set, row-wise KFold is used

### 3.3 Adapters

Adapters isolate project-specific I/O and schema.

- DataSourceAdapter:
  - query_samples
  - load_dataset
- SchemaAdapter:
  - to_canonical
  - validate
- CandidateSourceAdapter:
  - iter_candidates

### 3.4 Registries

- MetricRegistry: named metric functions
- FeatureRegistry: named feature builders
- InferenceRegistry: ordered inference rules
- NaNPolicyChain: ordered NaN handling functions

### 3.5 Strategies

- SplitStrategy:
  - RowKFoldSplit
  - GroupByColumnKFoldSplit
- ChunkStrategy:
  - FixedSizeChunking

## 4. Quick Start for LPS

Use the profile helpers from lps_profile.py to get notebook-parity defaults quickly.

```python
from collections import OrderedDict

from chem_informed_mtgp import (
    DataLoaderCore,
    LPSCandidateSourceAdapter,
    LPSProfileConfig,
    build_manual_entries_from_mappings,
    create_lps_dataset,
)

config = LPSProfileConfig(
    param_config=param_config,  # OrderedDict from notebook config
    a_site_to_smiles=A_SITE_TO_SMILES,
    continuous_columns=CONTINUOUS_COLUMNS,
    fingerprint_bits=FINGERPRINT_BITS,
    target_column=TARGET_COLUMN,
    db_path="mydb.duckdb",
)

dataset = create_lps_dataset(config)

# 1) Load base rows
dataset.load_from_query()

# 2) Optional: add manual rows from data_mapping.json files
manual_df = build_manual_entries_from_mappings(
    manual_entries={7: [3, 4], 11: [3]},
    continuous_columns=CONTINUOUS_COLUMNS,
    root_dir="chem-informed-mtgp-data-zenodo",  # folder containing LPS-<n>/data_mapping.json
)
dataset.add_rows(manual_df, source_tag="manual")

# 3) Inference + NaN cleanup
dataset.apply_inference_rules()
dataset.apply_nan_policies()

# 4) Mark known outliers once in the framework.
# Downstream get_df/get_array calls exclude outliers by default.
dataset.set_outliers(column={"sample_name": OUTLIER_SAMPLE_NAMES})

# 5) Add a target metric column by name (registered in LPS profile)
dataset.add_metric(TARGET_COLUMN, overwrite=True)

# 6) Normalize continuous parameters by declared ranges
ranges = {k: tuple(param_config[k]["range"]) for k in CONTINUOUS_COLUMNS}
dataset.fit_normalizer(columns=CONTINUOUS_COLUMNS, fit_on="ranges", ranges=ranges)

# 7) Build model features
dataset.build_feature("joint", from_normalized=True)
dataset.build_feature("target", from_normalized=False)

X_joint = dataset.get_feature("joint")
y = dataset.get_feature("target")

# 8) DataLoader usage
loader = DataLoaderCore(dataset)
folds = loader.split_cv(group_col="config_hash")

candidate_source = LPSCandidateSourceAdapter(
    param_config=param_config,
    a_site_to_smiles=A_SITE_TO_SMILES,
    continuous_columns=CONTINUOUS_COLUMNS,
    fingerprint_bits=FINGERPRINT_BITS,
)

candidate_iter, (bounds, norm_bounds) = dataset.iter_candidate_batches(
    candidate_source,
    batch_size=6000,
    n=12_000,
    replace=True,
    random_state=42,
)

for df_chunk, x_chunk in loader.iter_optimizer_chunks(
    candidate_iter,
    chunk_size=6000,
    n_batch=12,
    drop_remainder=True,
):
    # feed into optimizer
    pass
```

## 5. Outlier Handling

Outlier handling is built into DatasetCore and should be configured once near the start of the pipeline.

Key behavior:
- DatasetCore maintains an `is_outlier` boolean column.
- `get_df(...)` excludes outliers by default (`include_outliers=False`).
- `get_array(...)` excludes outliers by default (`include_outliers=False`).
- `set_outliers(...)` supports column-value based and mask-based updates.
- Exactly one column key is currently supported in `column={...}`.
- Use `append=True` to add new outlier flags without clearing existing ones.

Typical usage:

```python
# Set initial outliers by sample name.
dataset.set_outliers(column={"sample_name": ["3-21", "4-17"]})

# Optional: inspect all rows including outliers.
df_all = dataset.get_df(normalized=False, include_outliers=True)

# Default modeling view excludes outliers.
df_model = dataset.get_df(normalized=False)

# Add additional outliers by boolean mask (aligned to current dataset rows).
extra_mask = df_all["config_hash"].isin(["abc123", "def456"])
dataset.set_outliers(mask=extra_mask, append=True)
```

## 6. Using External DataFrame for Normalization

DatasetCore supports normalization against an external reference DataFrame.

```python
ref_df = some_reference_df[["annealing_temp", "annealing_time"]]
dataset.fit_normalizer(
    columns=["annealing_temp", "annealing_time"],
    fit_on="external",
    external_df=ref_df,
)

# Reuse the fitted scaler to normalize arbitrary external rows (e.g., candidate grids).
candidates_norm = dataset.transform_with_normalizer(
    data=candidates_df,
    columns=["annealing_temp", "annealing_time"],
)

# Backward compatibility: dataset.normalize(...) still works as a wrapper.
```

Supported fit_on modes:
- ranges
- external
- current

## 7. API Reference

### 7.1 DatasetCore

- load_from_query(filters=None)
- add_rows(rows, source_tag="manual")
- apply_inference_rules()
- apply_nan_policies()
- add_metric(metric_name, overwrite=False)
- set_outliers(column=None, mask=None, append=False)
- fit_normalizer(columns, key="default", fit_on="ranges", ranges=None, external_df=None, clip=(0.0, 1.0))
- transform_with_normalizer(data, columns, key="default", clip=(0.0, 1.0))
- normalize(columns, key="default", fit_on="ranges", ranges=None, external_df=None, data=None, clip=(0.0, 1.0))  # backward-compatible wrapper
- build_feature(name, from_normalized=True)
- get_df(normalized=False, copy=True, include_outliers=False)
- get_array(columns, normalized=False, include_outliers=False, dtype=np.float64)
- get_feature(name)
- iter_candidate_batches(candidate_source, batch_size, n=None, replace=False, random_state=None)

### 7.2 DataLoaderCore

- split_cv(group_col=None, strategy=None)
- iter_optimizer_chunks(source, chunk_size, n_batch=None, drop_remainder=True)

### 7.3 LPS Helpers

- create_lps_dataset(config)
- build_manual_entries_from_mappings(...)
- LPSCandidateSourceAdapter(...)

## 8. Creating a New Project Profile

For a new project, keep DatasetCore/DataLoaderCore unchanged.
Implement:

1. Data source adapter
- How tabular rows are queried
- How modality data is loaded

2. Schema adapter
- Map source column names to canonical names used by your project code
- Validate required fields

3. Registries and policy chain
- Inference rules
- NaN policies
- Custom metrics
- Feature builders

4. Candidate source adapter
- Wrap your own candidate generator while returning (DataFrame, ndarray) batches

Minimal skeleton:

```python
from dataclasses import dataclass
import pandas as pd

from chem_informed_mtgp.adapters import DataSourceAdapter, IdentitySchemaAdapter
from chem_informed_mtgp.dataset_core import DatasetCore

@dataclass(slots=True)
class MySource(DataSourceAdapter):
    def query_samples(self, *, filters=None) -> pd.DataFrame:
        ...
    def load_dataset(self, unique_ids, subgroup, **kwargs):
        ...

source = MySource()
schema = IdentitySchemaAdapter(required_columns=("sample_name",))

dataset = DatasetCore(source=source, schema=schema)
```

## 9. Migration Notes for Existing Notebooks

Suggested order:
1. Replace raw query + manual merge with DatasetCore.load_from_query and add_rows
2. Replace column inference and NaN logic with apply_inference_rules and apply_nan_policies
3. Move outlier logic to DatasetCore.set_outliers and stop doing manual DataFrame outlier slicing
4. Replace normalization cell with DatasetCore.normalize
5. Replace X_joint and y assembly with feature builders
6. Replace CV split and candidate chunking with DataLoaderCore

Keep GP and kernel code unchanged during initial migration.

## 10. Current Test Coverage

Implemented in tests/test_data_framework.py:
- external normalization behavior
- feature build behavior
- group-column split behavior
- optimizer chunking behavior

Run:

```bash
uv run pytest tests/test_data_framework.py -q
```

## 11. Known Limits in First Implementation

- LPS metric registry currently includes score, CD_score, homogeneity_score, g_factor_score, ldlb_score
- Advanced modality-driven metrics from notebooks can be added via MetricRegistry
- No GP/kernel refactor is included in this framework layer
