# Fingerprint Kernel Reference

This module supports three molecular-feature representations in Gaussian Process models:

- `bitvector`: legacy Morgan fingerprint bits (default `2048` bits)
- `indexed_tanimoto`: one categorical A-site index, with pairwise Tanimoto correlations precomputed between known A-sites
- `lkj_corr`: one categorical A-site index, with pairwise correlations *learned* through a Lewandowski-Kurowicka-Joe (LKJ) composed correlation matrix instead of Tanimoto similarity. Only supports exactly 3 categories.

## Kernel Entry Points

- `src/chem_informed_mtgp/fingerprint_kernel.py`
  - `joint_kernel_factory(...)`: NumPy-based hybrid kernel factory
  - `assemble_feature_matrix(...)`: builds model input matrix from continuous features plus fingerprint representation
- `src/chem_informed_mtgp/fingerprint_kernel_numba.py`
  - `joint_kernel_factory_numba(...)`: Numba-compiled hybrid kernel factory

## Continuous Kernels

The hybrid kernel uses one of:

- Matern 3/2 (`kernel="matern_3_2"`)
- Matern 5/2 (`kernel="matern_5_2"`)
- Squared Exponential (`kernel="squared_exponential"`)

## Fingerprint Modes

### 1) `bitvector`

- Feature shape: `(n_samples, cont_dims + fingerprint_bits)`
- Fingerprint similarity: bitwise Tanimoto

### 2) `indexed_tanimoto`

- Feature shape: `(n_samples, cont_dims + 1)`
- Last column stores A-site category index
- Fingerprint similarity is looked up from `FingerprintMetadata.tanimoto_correlations`

### 3) `lkj_corr`

- Feature shape: `(n_samples, cont_dims + 1)`, same categorical index encoding as `indexed_tanimoto`
- Requires `sigma_per_chemical=True` and exactly 3 A-site categories
- Fingerprint correlation matrix `C` is built at kernel-eval time from two learned parameters `r, s ∈ ℝ` (unconstrained, squashed through `tanh`) via a Cholesky factor:

  ```
  r' = tanh(r); s' = tanh(s); c = sqrt(1 - r'^2)
  G = [[1, 0, 0],
       [r', c, 0],
       [r', c*s', c*sqrt(1 - s'^2)]]
  C = G @ G.T
  ```

- `r` and `s` reuse the `hps[0]` (signal variance) and `hps[1+cont_dims]` (fingerprint weight) slots — in this mode those slots no longer mean variance/weight and must be bounded to `[-3.0, 3.0]` (see `get_init_hyperparamter_and_bounds(..., fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR)`), not their usual bounds.
- `K = sigma_i * sigma_j * K_continuous * C[idx1, idx2]` — there is no separate signal-variance multiplier since that slot is repurposed.
- **Master category**: with only 2 free parameters (`r`, `s`), `C` always has canonical category 0 sharing the *same* correlation (`r`) with categories 1 and 2, while the correlation between 1 and 2 is a derived quantity — one category is structurally special. Both `joint_kernel_factory` and `joint_kernel_factory_numba` accept `lkj_master_category` (an A-site name or integer index) to choose *which actual category* plays this "master" role; it defaults to index 0 (i.e. the first key in `a_site_to_smiles`) for backward compatibility. In `notebooks/transfer-learning.py` this is set via the `LKJ_MASTER_CATEGORY` env var, and swept via `lkj_master_category` in `notebooks/run_transfer_learning_sweep.py`.

#### Full (master-free) LKJ correlation: `lkj_master_category="none"`

Pass `lkj_master_category=LKJ_MASTER_NONE` (the string `"none"`) to use a fully general 3x3 correlation matrix with **3 independent parameters** `a, b, c ∈ ℝ` instead of 2 — no category is structurally special:

```
a' = tanh(a); b' = tanh(b); c' = tanh(c)
G = [[1,                    0,                          0                        ],
     [a',        sqrt(1 - a'^2),                         0                        ],
     [b',  sqrt(1 - b'^2)*c',  sqrt(1 - b'^2)*sqrt(1 - c'^2)]]
C = G @ G.T
```

- `a` and `b` reuse the same `hps[0]` and `hps[1+cont_dims]` slots as the 2-parameter version; `c` is a **new hyperparameter inserted immediately after** the `hps[1+cont_dims]` slot and before the per-chemical sigma block, i.e. at index `1+cont_dims+1`. Everything after it (sigma scales, noise, mean) shifts down by one slot. Bound it the same way, `[-3.0, 3.0]`.
- Backward compatible: `lkj_master_category=None` (default) or any A-site name/index keeps the original 2-parameter behavior and hyperparameter count unchanged. The extra parameter only appears when `lkj_master_category="none"` is explicitly requested.
- `get_init_hyperparamter_and_bounds(..., fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR, lkj_master_category=LKJ_MASTER_NONE)` inserts the extra `[-3.0, 3.0]` bounds row automatically.
- In `notebooks/transfer-learning.py`, set `LKJ_MASTER_CATEGORY=none` to select this mode.

## Supporting Functions

### In `src/chem_informed_mtgp/fingerprint_kernel.py`

- `build_fingerprint_metadata(a_site_to_smiles, n_bits=...)`
  - Builds deterministic A-site index maps and correlation matrix
- `encode_a_site_indices(df, metadata, ...)`
  - Maps each row to the categorical fingerprint index used by indexed mode
- `compute_a_site_fingerprints(df, ...)`
  - Builds/caches Morgan fingerprints for bitvector mode
- `tanimoto_kernel(x1, x2, amplitude=1.0)`
  - Dense Tanimoto kernel for bitvectors
- `indexed_tanimoto_kernel(x1_idx, x2_idx, tanimoto_correlations)`
  - Pairwise lookup kernel for indexed mode
- `exact_fingerprint_kernel(x1, x2, ...)`
  - Exact-match kernel for bitvectors

### In `src/chem_informed_mtgp/fingerprint_kernel_numba.py`

- `noise_factory_numba(...)`
  - Builds either constant noise or A-site-dependent noise functions
- `mean_factory_numba(...)`
  - Builds either constant mean or A-site-dependent mean functions

## Environment Variables

- `FINGERPRINT_BITS`: number of fingerprint bits (default `2048`)
- `CONTINUOUS_FEATURE_COUNT`: number of continuous dimensions (default `6`)
- `FINGERPRINT_MODE`: `bitvector` or `indexed_tanimoto`

## Hyperparameter Convention

For the hybrid kernel:

- `hps[0]`: signal variance
- `hps[1:1+cont_dims]`: continuous lengthscales
- `hps[1+cont_dims]`: fingerprint blend weight
- `hps[1+cont_dims+1:1+cont_dims+1+n_categories]`: per-chemical signal scales when `sigma_per_chemical=True`

In full `lkj_corr` mode (`lkj_master_category="none"`), one extra hyperparameter (the third LKJ correlation parameter) is inserted at index `1+cont_dims+1`, so the per-chemical signal scales shift to `hps[1+cont_dims+2:1+cont_dims+2+n_categories]`.

When `sigma_per_chemical=False`, the final kernel term is:

- `K = signal_var * K_continuous * (fp_weight * K_fp + (1 - fp_weight))`

When `sigma_per_chemical=True` in `indexed_tanimoto` mode:

- `K = signal_var * sqrt(sigma_i * sigma_j) * K_continuous * (fp_weight * K_fp + (1 - fp_weight))`

where `sigma_i` and `sigma_j` are looked up from the encoded A-site indices.

## Factory Notes

- `sigma_per_chemical=True` is only valid for `indexed_tanimoto` mode.
- NumPy kernels require `fingerprint_metadata` so the A-site ordering is deterministic.
- Numba kernels require `tanimoto_correlations` or an `extractor_fn` that resolves both the correlations and category count.
- Existing `hps` vectors without per-chemical scales remain valid unchanged.
