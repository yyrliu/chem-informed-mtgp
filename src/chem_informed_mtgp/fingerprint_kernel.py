"""Utility functions for molecular fingerprint augmentation in Gaussian processes.

This module now supports two fingerprint representations:
- bitvector: legacy Morgan fingerprints (cont_dims + fingerprint_bits)
- indexed_tanimoto: a single categorical index column (cont_dims + 1) with
  precomputed pairwise Tanimoto correlations between known A-site fingerprints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
from gpcam import kernels
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

FINGERPRINT_BITS = int(os.environ.get("FINGERPRINT_BITS", "2048"))
CONTINUOUS_FEATURE_COUNT = int(os.environ.get("CONTINUOUS_FEATURE_COUNT", "6"))
FINGERPRINT_MODE = os.environ.get("FINGERPRINT_MODE", "bitvector")

FINGERPRINT_MODE_BITVECTOR = "bitvector"
FINGERPRINT_MODE_INDEXED_TANIMOTO = "indexed_tanimoto"
FINGERPRINT_MODE_LKJ_CORR = "lkj_corr"

# Modes that encode the fingerprint as a single categorical index column
# (cont_dims + 1), as opposed to a full bitvector.
INDEXED_FINGERPRINT_MODES = frozenset(
    {FINGERPRINT_MODE_INDEXED_TANIMOTO, FINGERPRINT_MODE_LKJ_CORR}
)

# lkj_corr currently only supports the 3-category case, since it reuses the
# signal-variance and fingerprint-weight hyperparameter slots (2 free
# parameters) to parameterize a 3x3 LKJ-composed correlation matrix via its
# Cholesky factor. Supporting more categories would require n*(n-1)/2 free
# parameters.
LKJ_CORR_N_CATEGORIES = 3

# Sentinel for lkj_master_category: instead of reusing 2 existing hp slots
# (which forces one category to have equal correlation to both others), use
# 3 fully independent correlation parameters, one per category pair. Adds
# one extra hyperparameter versus the default "master category" scheme.
LKJ_MASTER_NONE = "none"

_EPSILON = 1e-12

# Global fingerprint cache: {(smiles, n_bits, radius): np.ndarray}
_FINGERPRINT_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


@dataclass(frozen=True)
class FingerprintMetadata:
    """Metadata for indexed_tanimoto fingerprint mode."""

    a_sites: tuple[str, ...]
    smiles: tuple[str, ...]
    fingerprints: np.ndarray
    tanimoto_correlations: np.ndarray
    a_site_to_index: Mapping[str, int]
    smiles_to_index: Mapping[str, int]


@lru_cache(maxsize=None)
def _get_morgan_generator(
    n_bits: int, radius: int, use_features: bool = False
) -> rdFingerprintGenerator.FingerprintGenerator64:
    if use_features:
        atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    else:
        atomInvariantsGenerator = None
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, atomInvariantsGenerator=atomInvariantsGenerator
    )


def _morgan_fingerprint(
    smiles: str, n_bits: int, radius: int = 2, use_features: bool = False
) -> np.ndarray:
    """Generate Morgan fingerprint with persistent caching."""
    cache_key = (smiles, n_bits, radius)

    if cache_key in _FINGERPRINT_CACHE:
        return _FINGERPRINT_CACHE[cache_key]

    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        arr = np.zeros((n_bits,), dtype=np.float64)
    else:
        generator = _get_morgan_generator(
            n_bits=n_bits, radius=radius, use_features=use_features
        )
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)

    _FINGERPRINT_CACHE[cache_key] = arr
    return arr


def build_fingerprint_metadata(
    a_site_to_smiles: Mapping[str, str],
    *,
    n_bits: int = FINGERPRINT_BITS,
    radius: int = 2,
    use_features: bool = False,
) -> FingerprintMetadata:
    """Build deterministic A-site index maps and pairwise Tanimoto correlations."""
    if not a_site_to_smiles:
        raise ValueError("a_site_to_smiles must contain at least one entry.")

    a_sites = tuple(a_site_to_smiles.keys())
    smiles = tuple((a_site_to_smiles[a_site] or "") for a_site in a_sites)
    fps = np.vstack(
        [
            _morgan_fingerprint(
                s, n_bits=n_bits, radius=radius, use_features=use_features
            )
            for s in smiles
        ]
    )

    tanimoto_correlations = tanimoto_kernel(fps, fps, amplitude=1.0)
    np.fill_diagonal(tanimoto_correlations, 1.0)

    a_site_to_index = {a_site: idx for idx, a_site in enumerate(a_sites)}
    smiles_to_index = {smiles_val: idx for idx, smiles_val in enumerate(smiles)}

    return FingerprintMetadata(
        a_sites=a_sites,
        smiles=smiles,
        fingerprints=fps,
        tanimoto_correlations=tanimoto_correlations,
        a_site_to_index=a_site_to_index,
        smiles_to_index=smiles_to_index,
    )


def _validate_fingerprint_mode(fingerprint_mode: str) -> None:
    if fingerprint_mode not in {
        FINGERPRINT_MODE_BITVECTOR,
        FINGERPRINT_MODE_INDEXED_TANIMOTO,
        FINGERPRINT_MODE_LKJ_CORR,
    }:
        raise ValueError(
            f"Unsupported fingerprint_mode '{fingerprint_mode}'. Choose from "
            f"'{FINGERPRINT_MODE_BITVECTOR}', '{FINGERPRINT_MODE_INDEXED_TANIMOTO}', "
            f"or '{FINGERPRINT_MODE_LKJ_CORR}'."
        )


def _ensure_metadata(metadata: FingerprintMetadata | None) -> FingerprintMetadata:
    if metadata is None:
        raise ValueError(
            "fingerprint_metadata is required for fingerprint_mode="
            f"'{FINGERPRINT_MODE_INDEXED_TANIMOTO}' or '{FINGERPRINT_MODE_LKJ_CORR}'."
        )
    return metadata


def _lkj_corr_matrix(r: float, s: float) -> np.ndarray:
    """3x3 correlation matrix from a Lewandowski-Kurowicka-Joe Cholesky factor.

    With only 2 free parameters (r, s), category 0 is structurally the
    "master": it has the same correlation (r) to categories 1 and 2, while
    the correlation between 1 and 2 is a derived quantity. Use
    `_lkj_category_order` to remap which actual category plays this role.
    """
    r = np.tanh(r)
    s = np.tanh(s)
    c = np.sqrt(1.0 - r * r)

    G = np.zeros((3, 3), dtype=np.float64)
    G[0, 0] = 1.0
    G[1, 0] = r
    G[1, 1] = c
    G[2, 0] = r
    G[2, 1] = c * s
    G[2, 2] = c * np.sqrt(1.0 - s * s)

    return G @ G.T


def _lkj_full_corr_matrix(a: float, b: float, c: float) -> np.ndarray:
    """3x3 correlation matrix from a fully general LKJ Cholesky factor.

    Unlike `_lkj_corr_matrix`, all 3 pairwise correlations are independent
    (no "master" category): corr(0,1) = tanh(a), corr(0,2) = tanh(b), and
    corr(1,2) is a smooth function of a, b, and c. Requires 3 free
    parameters instead of 2.
    """
    a = np.tanh(a)
    b = np.tanh(b)
    c = np.tanh(c)

    G = np.zeros((3, 3), dtype=np.float64)
    G[0, 0] = 1.0
    G[1, 0] = a
    G[1, 1] = np.sqrt(1.0 - a * a)
    G[2, 0] = b
    G[2, 1] = np.sqrt(1.0 - b * b) * c
    G[2, 2] = np.sqrt(1.0 - b * b) * np.sqrt(1.0 - c * c)

    return G @ G.T


def _lkj_category_order(n_categories: int, master_category: int) -> np.ndarray:
    """Map each actual category index to its canonical position in `_lkj_corr_matrix`.

    Canonical position 0 is the "master" category. The remaining actual
    categories fill positions 1 and 2 in ascending order of their actual
    index.
    """
    order = [master_category] + [i for i in range(n_categories) if i != master_category]
    inv = np.empty(n_categories, dtype=np.int64)
    for canonical_pos, actual_idx in enumerate(order):
        inv[actual_idx] = canonical_pos
    return inv


def _resolve_lkj_master_category(
    master_category: int | str | None,
    n_categories: int,
    metadata: FingerprintMetadata | None = None,
) -> int:
    """Resolve an lkj_master_category argument (name, index, or None) to an index."""
    if master_category is None:
        return 0

    if isinstance(master_category, str):
        if metadata is None:
            raise ValueError(
                "fingerprint_metadata is required to resolve a string lkj_master_category."
            )
        if master_category not in metadata.a_site_to_index:
            raise ValueError(
                f"Unknown lkj_master_category '{master_category}'; expected one of {metadata.a_sites}."
            )
        idx = metadata.a_site_to_index[master_category]
    else:
        idx = int(master_category)

    if not (0 <= idx < n_categories):
        raise ValueError(
            f"lkj_master_category index {idx} out of range for {n_categories} categories."
        )
    return idx


def compute_a_site_fingerprints(
    df, column: str = "a_site", n_bits: int = FINGERPRINT_BITS, radius: int = 2
) -> np.ndarray:
    """Generate RDKit Morgan fingerprints with persistent caching."""
    if column not in df:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    tokens = df[column].fillna("").astype(str).tolist()
    if not tokens:
        return np.empty((0, n_bits), dtype=np.float64)

    return np.vstack(
        [_morgan_fingerprint(smiles, n_bits=n_bits, radius=radius) for smiles in tokens]
    )


def encode_a_site_indices(
    df,
    *,
    metadata: FingerprintMetadata,
    a_site_column: str = "a_site",
    smiles_column: str = "a_site_smiles",
) -> np.ndarray:
    """Encode each row into a single float index for indexed_tanimoto mode."""
    if a_site_column not in df and smiles_column not in df:
        raise KeyError(
            f"Expected one of columns '{a_site_column}' or '{smiles_column}' to exist."
        )

    indices = np.full((len(df),), -1, dtype=np.int64)

    if a_site_column in df:
        a_sites = df[a_site_column].fillna("").astype(str).to_numpy()
        for i, a_site in enumerate(a_sites):
            if a_site in metadata.a_site_to_index:
                indices[i] = metadata.a_site_to_index[a_site]

    elif smiles_column in df:
        smiles_values = df[smiles_column].fillna("").astype(str).to_numpy()
        for i, smiles in enumerate(smiles_values):
            if indices[i] < 0 and smiles in metadata.smiles_to_index:
                indices[i] = metadata.smiles_to_index[smiles]

    if np.any(indices < 0):
        missing = np.where(indices < 0)[0][:5].tolist()
        raise ValueError(
            "Some rows could not be mapped to known fingerprint indices. "
            f"First unresolved row indices: {missing}"
        )

    return indices.astype(np.float64).reshape(-1, 1)


def clear_fingerprint_cache() -> int:
    """Clear the global fingerprint cache and return number of entries cleared."""
    count = len(_FINGERPRINT_CACHE)
    _FINGERPRINT_CACHE.clear()
    return count


def tanimoto_kernel(
    x1: np.ndarray, x2: np.ndarray, amplitude: float = 1.0
) -> np.ndarray:
    """Compute a scaled Tanimoto kernel between binary fingerprint matrices."""
    X1 = np.asarray(x1, dtype=np.float64)
    X2 = np.asarray(x2, dtype=np.float64)
    dot_product = X1 @ X2.T
    X1_norm = np.sum(X1, axis=1, keepdims=True)
    X2_norm = np.sum(X2, axis=1, keepdims=True)
    denominator = np.clip(X1_norm + X2_norm.T - dot_product, _EPSILON, None)
    tanimoto = dot_product / denominator
    return amplitude * tanimoto


def indexed_tanimoto_kernel(
    x1_idx: np.ndarray, x2_idx: np.ndarray, tanimoto_correlations: np.ndarray
) -> np.ndarray:
    """Lookup pairwise correlations for integer-like A-site indices."""
    idx1 = np.rint(np.asarray(x1_idx).reshape(-1)).astype(np.int32)
    idx2 = np.rint(np.asarray(x2_idx).reshape(-1)).astype(np.int32)

    n_cat = tanimoto_correlations.shape[0]
    if (
        np.any(idx1 < 0)
        or np.any(idx1 >= n_cat)
        or np.any(idx2 < 0)
        or np.any(idx2 >= n_cat)
    ):
        raise ValueError(
            f"Indexed fingerprint values must be in [0, {n_cat - 1}]. "
            f"Got idx1 range [{idx1.min()}, {idx1.max()}], idx2 range [{idx2.min()}, {idx2.max()}]."
        )

    return tanimoto_correlations[np.ix_(idx1, idx2)]


def exact_fingerprint_kernel(
    x1: np.ndarray, x2: np.ndarray, chunk_size: int = 100
) -> np.ndarray:
    n = x1.shape[0]
    m = x2.shape[0]

    p1 = np.packbits(x1.astype(np.uint8, copy=False), axis=1).view(np.uint64)
    p2 = np.packbits(x2.astype(np.uint8, copy=False), axis=1).view(np.uint64)
    w = p1.shape[1]

    result = np.empty((n, m), dtype=float)
    buf_xor = np.empty((n, chunk_size, w), dtype=np.uint64)
    buf_any = np.empty((n, chunk_size), dtype=np.bool_)

    for j in range(0, m, chunk_size):
        cj = min(chunk_size, m - j)

        xor = buf_xor[:, :cj, :]
        any_ = buf_any[:, :cj]

        np.bitwise_xor(p1[:, None, :], p2[None, j : j + cj, :], out=xor)
        np.any(xor, axis=2, out=any_)
        np.logical_not(any_, out=any_)
        result[:, j : j + cj] = any_

    return result


def matern_continuous_kernel_3_2(
    x1: np.ndarray, x2: np.ndarray, hps: Sequence[float], cont_dims: int
) -> np.ndarray:
    length_scales = hps[1 : 1 + cont_dims]
    distance_matrix = kernels.get_anisotropic_distance_matrix(x1, x2, length_scales)
    return kernels.matern_kernel_diff1(distance_matrix, 1)


def matern_continuous_kernel_5_2(
    x1: np.ndarray, x2: np.ndarray, hps: Sequence[float], cont_dims: int
) -> np.ndarray:
    length_scales = hps[1 : 1 + cont_dims]
    distance_matrix = kernels.get_anisotropic_distance_matrix(x1, x2, length_scales)
    return kernels.matern_kernel_diff2(distance_matrix, 1.0)


def squared_exponential_continuous_kernel(
    x1: np.ndarray, x2: np.ndarray, hps: Sequence[float], cont_dims: int
) -> np.ndarray:
    length_scales = hps[1 : 1 + cont_dims]
    distance_matrix = kernels.get_anisotropic_distance_matrix(x1, x2, length_scales)
    return kernels.squared_exponential_kernel(distance_matrix, 1.0)


def joint_kernel_factory(
    cont_dims: int = CONTINUOUS_FEATURE_COUNT,
    fingerprint_bits: int = FINGERPRINT_BITS,
    *,
    kernel: str = "matern_3_2",
    fingerprint_kernel: str = "tanimoto",
    fingerprint_mode: str = FINGERPRINT_MODE,
    fingerprint_metadata: FingerprintMetadata | None = None,
    sigma_per_chemical: bool = False,
    lkj_master_category: int | str | None = None,
):
    """Create hybrid continuous + fingerprint kernel in bitvector or indexed mode."""
    _validate_fingerprint_mode(fingerprint_mode)
    metadata = (
        _ensure_metadata(fingerprint_metadata)
        if fingerprint_mode in INDEXED_FINGERPRINT_MODES
        else None
    )

    lkj_category_order = None
    is_lkj_full = (
        fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR
        and lkj_master_category == LKJ_MASTER_NONE
    )
    if fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR:
        if not sigma_per_chemical:
            raise ValueError(
                f"fingerprint_mode='{FINGERPRINT_MODE_LKJ_CORR}' requires sigma_per_chemical=True "
                "(the signal variance and fingerprint weight slots are repurposed as LKJ parameters)."
            )
        if len(metadata.a_sites) != LKJ_CORR_N_CATEGORIES:
            raise ValueError(
                f"fingerprint_mode='{FINGERPRINT_MODE_LKJ_CORR}' only supports exactly "
                f"{LKJ_CORR_N_CATEGORIES} categories; got {len(metadata.a_sites)}."
            )
        if not is_lkj_full:
            master_idx = _resolve_lkj_master_category(
                lkj_master_category, len(metadata.a_sites), metadata
            )
            lkj_category_order = _lkj_category_order(len(metadata.a_sites), master_idx)

    if kernel == "matern_3_2":
        matern_continuous_kernel = matern_continuous_kernel_3_2
    elif kernel == "matern_5_2":
        matern_continuous_kernel = matern_continuous_kernel_5_2
    elif kernel == "squared_exponential":
        matern_continuous_kernel = squared_exponential_continuous_kernel
    else:
        raise ValueError(
            f"Unsupported kernel type '{kernel}'. Choose from 'matern_3_2', 'matern_5_2', or 'squared_exponential'."
        )

    if sigma_per_chemical and fingerprint_mode not in INDEXED_FINGERPRINT_MODES:
        raise ValueError(
            "sigma_per_chemical=True is only supported for fingerprint_mode="
            f"'{FINGERPRINT_MODE_INDEXED_TANIMOTO}' or '{FINGERPRINT_MODE_LKJ_CORR}'."
        )

    kernel_end = 1 + cont_dims + 1
    lkj_third_param_idx = kernel_end if is_lkj_full else None
    if is_lkj_full:
        kernel_end += (
            1  # extra LKJ correlation parameter, inserted before the sigma block
        )
    n_categories = len(metadata.a_sites) if metadata is not None else 0
    sigma_slice = slice(kernel_end, kernel_end + n_categories)

    def _parse_kernel_hps(hps: Sequence[float]) -> tuple[np.ndarray, np.ndarray | None]:
        hps_arr = np.asarray(hps, dtype=np.float64)
        if hps_arr.shape[0] < kernel_end:
            raise ValueError(
                f"Expected at least {kernel_end} kernel hyperparameters (signal variance, {cont_dims} lengthscales, fingerprint weight) but got {hps_arr.shape[0]}."
            )
        if sigma_per_chemical and hps_arr.shape[0] < sigma_slice.stop:
            raise ValueError(
                f"sigma_per_chemical=True requires at least {sigma_slice.stop} hyperparameters; got {hps_arr.shape[0]}."
            )
        return hps_arr[:kernel_end], hps_arr[
            sigma_slice
        ] if sigma_per_chemical else None

    def _fp_kernel(x1_fp: np.ndarray, x2_fp: np.ndarray, hps: np.ndarray) -> np.ndarray:
        if fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR:
            if is_lkj_full:
                C = _lkj_full_corr_matrix(
                    hps[0], hps[1 + cont_dims], hps[lkj_third_param_idx]
                )
                idx1 = np.rint(x1_fp.reshape(-1)).astype(np.int64)
                idx2 = np.rint(x2_fp.reshape(-1)).astype(np.int64)
            else:
                C = _lkj_corr_matrix(hps[0], hps[1 + cont_dims])
                idx1 = lkj_category_order[np.rint(x1_fp.reshape(-1)).astype(np.int64)]
                idx2 = lkj_category_order[np.rint(x2_fp.reshape(-1)).astype(np.int64)]
            return C[np.ix_(idx1, idx2)]

        if fingerprint_mode == FINGERPRINT_MODE_INDEXED_TANIMOTO:
            if fingerprint_kernel not in {"tanimoto", "exact"}:
                raise ValueError(
                    "indexed_tanimoto mode supports fingerprint_kernel in {'tanimoto', 'exact'}."
                )
            if fingerprint_kernel == "tanimoto":
                k_fp = indexed_tanimoto_kernel(
                    x1_fp,
                    x2_fp,
                    metadata.tanimoto_correlations,  # type: ignore[union-attr]
                )
                fp_weight = hps[1 + cont_dims]
                return fp_weight * k_fp + (1.0 - fp_weight)

            return (
                np.rint(x1_fp.reshape(-1, 1)).astype(np.int32)
                == np.rint(x2_fp.reshape(1, -1)).astype(np.int32)
            ).astype(np.float64)

        if fingerprint_kernel == "tanimoto":
            k_fp = tanimoto_kernel(x1_fp, x2_fp)
            fp_weight = hps[1 + cont_dims]
            return fp_weight * k_fp + (1.0 - fp_weight)

        if fingerprint_kernel == "exact":
            return exact_fingerprint_kernel(x1_fp, x2_fp)

        raise ValueError(
            f"Unsupported fingerprint kernel type '{fingerprint_kernel}'. Choose from 'tanimoto' or 'exact'."
        )

    expected_fp_dims = (
        1 if fingerprint_mode in INDEXED_FINGERPRINT_MODES else fingerprint_bits
    )

    def _kernel(
        x1: Sequence[Sequence[float]],
        x2: Sequence[Sequence[float]],
        hps: Sequence[float],
    ) -> np.ndarray:
        x1_arr = np.asarray(x1, dtype=np.float64)
        x2_arr = np.asarray(x2, dtype=np.float64)
        kernel_hps, sigma_values = _parse_kernel_hps(hps)

        expected_cols = cont_dims + expected_fp_dims
        if x1_arr.shape[1] != expected_cols or x2_arr.shape[1] != expected_cols:
            raise ValueError(
                f"Expected kernel inputs with {expected_cols} columns (cont={cont_dims}, fp={expected_fp_dims}) "
                f"for fingerprint_mode='{fingerprint_mode}', but got {x1_arr.shape[1]} and {x2_arr.shape[1]}."
            )

        x1_cont = x1_arr[:, :cont_dims]
        x2_cont = x2_arr[:, :cont_dims]
        x1_fp = x1_arr[:, cont_dims : cont_dims + expected_fp_dims]
        x2_fp = x2_arr[:, cont_dims : cont_dims + expected_fp_dims]

        # In lkj_corr mode, hps[0] is repurposed as an LKJ correlation parameter
        # (not an actual signal variance), so it must not also scale the kernel.
        signal_var = (
            1.0 if fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR else kernel_hps[0]
        )
        k_cont = matern_continuous_kernel(x1_cont, x2_cont, kernel_hps, cont_dims)
        k_fp_blend = _fp_kernel(x1_fp, x2_fp, kernel_hps)

        if sigma_values is None:
            return signal_var * k_cont * k_fp_blend

        idx1 = np.rint(x1_fp.reshape(-1)).astype(np.int64)
        idx2 = np.rint(x2_fp.reshape(-1)).astype(np.int64)
        sigma_scale = np.outer(sigma_values[idx1], sigma_values[idx2])
        return signal_var * sigma_scale * k_cont * k_fp_blend

    return _kernel


def assemble_feature_matrix(
    df,
    continuous_columns: Sequence[str],
    fingerprint_column: str = "a_site",
    n_bits: int = FINGERPRINT_BITS,
    *,
    fingerprint_mode: str = FINGERPRINT_MODE,
    fingerprint_metadata: FingerprintMetadata | None = None,
    a_site_column: str = "a_site",
) -> np.ndarray:
    """Concatenate continuous parameters with fingerprint or index representation."""
    _validate_fingerprint_mode(fingerprint_mode)
    if len(continuous_columns) == 0:
        raise ValueError("At least one continuous feature is required.")

    X_cont = df.loc[:, continuous_columns].to_numpy(dtype=np.float64)

    if fingerprint_mode in INDEXED_FINGERPRINT_MODES:
        metadata = _ensure_metadata(fingerprint_metadata)
        X_fp = encode_a_site_indices(
            df,
            metadata=metadata,
            a_site_column=a_site_column,
            smiles_column=fingerprint_column,
        )
    else:
        X_fp = compute_a_site_fingerprints(df, column=fingerprint_column, n_bits=n_bits)

    if X_cont.shape[0] != X_fp.shape[0]:
        raise ValueError(
            "Continuous and fingerprint feature blocks must have matching row counts."
        )

    return np.hstack([X_cont, X_fp])
