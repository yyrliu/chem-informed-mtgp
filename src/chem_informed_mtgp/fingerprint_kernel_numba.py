import numpy as np
from numba import njit, prange

from .fingerprint_kernel import (
    CONTINUOUS_FEATURE_COUNT,
    FINGERPRINT_BITS,
    FINGERPRINT_MODE,
    FINGERPRINT_MODE_BITVECTOR,
    FINGERPRINT_MODE_INDEXED_TANIMOTO,
    FINGERPRINT_MODE_LKJ_CORR,
    INDEXED_FINGERPRINT_MODES,
    LKJ_CORR_N_CATEGORIES,
    LKJ_MASTER_NONE,
    FingerprintMetadata,
    _lkj_category_order,
    _morgan_fingerprint,
    _resolve_lkj_master_category,
)


# --------------------------------------------------------------
# 1.1  Continuous kernels
# --------------------------------------------------------------
@njit(fastmath=True, inline="always")
def _matern_block_3_2(row1, row2, lengthscales):
    dist_sq = 0.0
    for k in range(len(lengthscales)):
        d = (row1[k] - row2[k]) / lengthscales[k]
        dist_sq += d * d
    sqrt3_r = np.sqrt(3.0) * np.sqrt(dist_sq)
    return (1.0 + sqrt3_r) * np.exp(-sqrt3_r)


@njit(fastmath=True, inline="always")
def _matern_block_5_2(row1, row2, lengthscales):
    dist_sq = 0.0
    for k in range(len(lengthscales)):
        d = (row1[k] - row2[k]) / lengthscales[k]
        dist_sq += d * d
    sqrt5_r = np.sqrt(5.0) * np.sqrt(dist_sq)
    return (1.0 + sqrt5_r + (5.0 / 3.0) * dist_sq) * np.exp(-sqrt5_r)


@njit(fastmath=True, inline="always")
def _squared_exponential_block(row1, row2, lengthscales):
    dist_sq = 0.0
    for k in range(len(lengthscales)):
        d = (row1[k] - row2[k]) / lengthscales[k]
        dist_sq += d * d
    return np.exp(-0.5 * dist_sq)


# --------------------------------------------------------------
# 1.2  Tanimoto on integer fingerprints
# --------------------------------------------------------------
@njit(fastmath=True, inline="always")
def _tanimoto_int_block(row1_int, row2_int):
    inter = 0
    sum1 = 0
    sum2 = 0
    for k in range(len(row1_int)):
        a = row1_int[k]
        b = row2_int[k]
        inter += a * b
        sum1 += a
        sum2 += b
    union = sum1 + sum2 - inter
    return float(inter) / float(union) if union != 0 else 0.0


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


def _extract_kernel_hps(
    hps: np.ndarray, kernel_end: int, sigma_stop: int | None
) -> tuple[np.ndarray, np.ndarray | None]:
    hps_arr = np.asarray(hps, dtype=np.float64)
    if hps_arr.shape[0] < kernel_end:
        raise ValueError(
            f"Expected at least {kernel_end} kernel hyperparameters (signal variance, lengthscales, fingerprint weight) but got {hps_arr.shape[0]}."
        )
    if sigma_stop is not None and hps_arr.shape[0] < sigma_stop:
        raise ValueError(
            f"sigma_per_chemical=True requires at least {sigma_stop} hyperparameters; got {hps_arr.shape[0]}."
        )
    kernel_hps = hps_arr[:kernel_end]
    sigma_values = hps_arr[kernel_end:sigma_stop] if sigma_stop is not None else None
    return kernel_hps, sigma_values


def joint_kernel_factory_numba(
    cont_dims: int = CONTINUOUS_FEATURE_COUNT,
    fingerprint_bits: int = FINGERPRINT_BITS,
    *,
    kernel: str = "matern_3_2",
    fingerprint_mode: str = FINGERPRINT_MODE,
    tanimoto_correlations: np.ndarray | None = None,
    sigma_per_chemical: bool = False,
    sigma_n_categories: int | None = None,
    fingerprint_metadata: FingerprintMetadata | None = None,
    extractor_fn=None,
    lkj_master_category: int | str | None = None,
):
    """
    Returns a Numba‑compiled hybrid kernel

        K = σ² *
            Matern3/2(continuous) *
            [ w * Tanimoto(fingerprint) + (1‑w) * 1 ]

    Parameters
    ----------
    cont_dims, fingerprint_bits : int
        Dimensionality of the continuous block and the fingerprint block.

    Expected hyper‑parameter vector (1‑D array)

        hps[0]                → signal variance σ²  (>0)
        hps[1:1+cont_dims]    → length‑scales ℓ₁ … ℓ_cont_dims
        hps[1+cont_dims]      → fingerprint weight w ∈ [0, 1]

    The returned function signature is

        (X1: np.ndarray, X2: np.ndarray, hps: np.ndarray) → np.ndarray

    where ``X*`` have shape ``(n_samples, cont_dims+fingerprint_bits)`` and are
    **float64** (the whole matrix can be float for convenience). Internally
    the fingerprint slice is cast to ``int32`` (or packed to ``uint64``) once per
    row, which is far cheaper than doing it inside the inner‑most loop.
    """
    _validate_fingerprint_mode(fingerprint_mode)

    if sigma_per_chemical and fingerprint_mode not in INDEXED_FINGERPRINT_MODES:
        raise ValueError(
            "sigma_per_chemical=True is only supported for fingerprint_mode="
            f"'{FINGERPRINT_MODE_INDEXED_TANIMOTO}' or '{FINGERPRINT_MODE_LKJ_CORR}'."
        )

    if fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR and not sigma_per_chemical:
        raise ValueError(
            f"fingerprint_mode='{FINGERPRINT_MODE_LKJ_CORR}' requires sigma_per_chemical=True "
            "(the signal variance and fingerprint weight slots are repurposed as LKJ parameters)."
        )

    if kernel == "matern_3_2":
        _matern_block = _matern_block_3_2
    elif kernel == "matern_5_2":
        _matern_block = _matern_block_5_2
    elif kernel == "squared_exponential":
        _matern_block = _squared_exponential_block
    else:
        raise ValueError(
            f"Unsupported kernel type '{kernel}'. Choose from 'matern_3_2', 'matern_5_2', or 'squared_exponential'."
        )

    if fingerprint_mode == FINGERPRINT_MODE_INDEXED_TANIMOTO:
        if tanimoto_correlations is None and extractor_fn is not None:
            if fingerprint_metadata is None:
                raise ValueError(
                    "fingerprint_metadata is required when extractor_fn is provided for indexed_tanimoto mode."
                )
            tanimoto_correlations, extracted_categories = extractor_fn(
                fingerprint_metadata
            )
            if sigma_n_categories is None:
                sigma_n_categories = extracted_categories

        if tanimoto_correlations is None:
            raise ValueError(
                "tanimoto_correlations must be provided for fingerprint_mode='indexed_tanimoto'."
            )

        tcorr = np.asarray(tanimoto_correlations, dtype=np.float64)
        if tcorr.ndim != 2 or tcorr.shape[0] != tcorr.shape[1]:
            raise ValueError("tanimoto_correlations must be a square 2D array.")

        if sigma_n_categories is None:
            sigma_n_categories = tcorr.shape[0]
        if sigma_n_categories != tcorr.shape[0]:
            raise ValueError(
                f"sigma_n_categories ({sigma_n_categories}) must match tanimoto_correlations.shape[0] ({tcorr.shape[0]})."
            )

        expected_cols = cont_dims + 1
        n_categories = tcorr.shape[0]
        kernel_end = 1 + cont_dims + 1
        sigma_stop = kernel_end + n_categories if sigma_per_chemical else None

        @njit(fastmath=True)
        def _indexed_kernel_impl(X1, X2, kernel_hps, sigma_values):
            n1, d1 = X1.shape
            n2, d2 = X2.shape

            if d1 != expected_cols or d2 != expected_cols:
                raise ValueError(
                    f"Expected X1 and X2 to have {expected_cols} columns (cont_dims + 1 index column) but got {d1} and {d2}."
                )

            K = np.empty((n1, n2), dtype=np.float64)
            signal_var = kernel_hps[0]
            ls_cont = kernel_hps[1 : 1 + cont_dims]
            fp_weight = kernel_hps[1 + cont_dims]

            for i in prange(n1):
                row1 = X1[i]
                row1_cont = row1[:cont_dims]
                idx1 = int(np.rint(row1[cont_dims]))
                if idx1 < 0 or idx1 >= n_categories:
                    raise ValueError(
                        f"Indexed fingerprint value out of range: {idx1}; expected [0, {n_categories - 1}]."
                    )

                for j in range(n2):
                    row2 = X2[j]
                    idx2 = int(np.rint(row2[cont_dims]))
                    if idx2 < 0 or idx2 >= n_categories:
                        raise ValueError(
                            f"Indexed fingerprint value out of range: {idx2}; expected [0, {n_categories - 1}]."
                        )

                    k_cont = _matern_block(row1_cont, row2[:cont_dims], ls_cont)
                    k_fp = tcorr[idx1, idx2]

                    if idx1 == idx2:
                        if k_fp < 1.0:
                            raise ValueError(
                                f"Tanimoto correlation for identical fingerprints (idx={idx1}) must be 1.0, but got {k_fp}."
                            )
                    k_fp = fp_weight * k_fp + (1.0 - fp_weight)

                    if sigma_per_chemical:
                        sigma_scale = sigma_values[idx1] * sigma_values[idx2]
                    else:
                        sigma_scale = 1.0
                    K[i, j] = signal_var * sigma_scale * k_cont * k_fp

            return K

        def hybrid_kernel(X1, X2, hps):
            kernel_hps, sigma_values = _extract_kernel_hps(hps, kernel_end, sigma_stop)
            if sigma_values is None:
                sigma_values = np.empty(0, dtype=np.float64)
            return _indexed_kernel_impl(X1, X2, kernel_hps, sigma_values)

        return hybrid_kernel

    if fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR:
        if sigma_n_categories is None and fingerprint_metadata is not None:
            sigma_n_categories = len(fingerprint_metadata.a_sites)

        if sigma_n_categories is None:
            raise ValueError(
                f"fingerprint_mode='{FINGERPRINT_MODE_LKJ_CORR}' requires sigma_n_categories "
                "or fingerprint_metadata to determine the number of categories."
            )

        if sigma_n_categories != LKJ_CORR_N_CATEGORIES:
            raise ValueError(
                f"fingerprint_mode='{FINGERPRINT_MODE_LKJ_CORR}' only supports exactly "
                f"{LKJ_CORR_N_CATEGORIES} categories; got {sigma_n_categories}."
            )

        n_categories = sigma_n_categories
        expected_cols = cont_dims + 1
        is_lkj_full = lkj_master_category == LKJ_MASTER_NONE

        if is_lkj_full:
            # 3 independent correlation parameters, no master category.
            kernel_end = 1 + cont_dims + 1 + 1
            sigma_stop = kernel_end + n_categories

            @njit(fastmath=True)
            def _lkj_full_corr_kernel_impl(X1, X2, kernel_hps, sigma_values):
                n1, d1 = X1.shape
                n2, d2 = X2.shape

                if d1 != expected_cols or d2 != expected_cols:
                    raise ValueError(
                        f"Expected X1 and X2 to have {expected_cols} columns (cont_dims + 1 index column) but got {d1} and {d2}."
                    )

                K = np.empty((n1, n2), dtype=np.float64)
                ls_cont = kernel_hps[1 : 1 + cont_dims]

                # hps[0], hps[1+cont_dims], and hps[2+cont_dims] are
                # repurposed as the 3 independent LKJ correlation parameters.
                a = np.tanh(kernel_hps[0])
                b = np.tanh(kernel_hps[1 + cont_dims])
                c = np.tanh(kernel_hps[2 + cont_dims])

                G = np.zeros((3, 3), dtype=np.float64)
                G[0, 0] = 1.0
                G[1, 0] = a
                G[1, 1] = np.sqrt(1.0 - a * a)
                G[2, 0] = b
                G[2, 1] = np.sqrt(1.0 - b * b) * c
                G[2, 2] = np.sqrt(1.0 - b * b) * np.sqrt(1.0 - c * c)

                C = G @ G.T

                for i in prange(n1):
                    row1 = X1[i]
                    row1_cont = row1[:cont_dims]
                    idx1 = int(np.rint(row1[cont_dims]))
                    if idx1 < 0 or idx1 >= n_categories:
                        raise ValueError(
                            f"Indexed fingerprint value out of range: {idx1}; expected [0, {n_categories - 1}]."
                        )

                    for j in range(n2):
                        row2 = X2[j]
                        idx2 = int(np.rint(row2[cont_dims]))
                        if idx2 < 0 or idx2 >= n_categories:
                            raise ValueError(
                                f"Indexed fingerprint value out of range: {idx2}; expected [0, {n_categories - 1}]."
                            )

                        k_cont = _matern_block(row1_cont, row2[:cont_dims], ls_cont)
                        k_fp = C[idx1, idx2]

                        sigma_scale = sigma_values[idx1] * sigma_values[idx2]
                        K[i, j] = sigma_scale * k_cont * k_fp

                return K

            def hybrid_kernel(X1, X2, hps):
                kernel_hps, sigma_values = _extract_kernel_hps(
                    hps, kernel_end, sigma_stop
                )
                return _lkj_full_corr_kernel_impl(X1, X2, kernel_hps, sigma_values)

            return hybrid_kernel

        kernel_end = 1 + cont_dims + 1
        sigma_stop = kernel_end + n_categories

        master_idx = _resolve_lkj_master_category(
            lkj_master_category, n_categories, fingerprint_metadata
        )
        # Maps each actual category index to its canonical position in the
        # LKJ correlation matrix C (canonical position 0 is the "master").
        category_order = _lkj_category_order(n_categories, master_idx)

        @njit(fastmath=True)
        def _lkj_corr_kernel_impl(X1, X2, kernel_hps, sigma_values, category_order):
            n1, d1 = X1.shape
            n2, d2 = X2.shape

            if d1 != expected_cols or d2 != expected_cols:
                raise ValueError(
                    f"Expected X1 and X2 to have {expected_cols} columns (cont_dims + 1 index column) but got {d1} and {d2}."
                )

            K = np.empty((n1, n2), dtype=np.float64)
            ls_cont = kernel_hps[1 : 1 + cont_dims]

            # hps[0] and hps[1+cont_dims] are repurposed as LKJ correlation
            # parameters (not signal variance / fingerprint weight).
            r = np.tanh(kernel_hps[0])
            s = np.tanh(kernel_hps[1 + cont_dims])
            c = np.sqrt(1.0 - r * r)

            G = np.zeros((3, 3), dtype=np.float64)
            G[0, 0] = 1.0
            G[1, 0] = r
            G[1, 1] = c
            G[2, 0] = r
            G[2, 1] = c * s
            G[2, 2] = c * np.sqrt(1.0 - s * s)

            C = G @ G.T

            for i in prange(n1):
                row1 = X1[i]
                row1_cont = row1[:cont_dims]
                actual_idx1 = int(np.rint(row1[cont_dims]))
                if actual_idx1 < 0 or actual_idx1 >= n_categories:
                    raise ValueError(
                        f"Indexed fingerprint value out of range: {actual_idx1}; expected [0, {n_categories - 1}]."
                    )
                idx1 = category_order[actual_idx1]

                for j in range(n2):
                    row2 = X2[j]
                    actual_idx2 = int(np.rint(row2[cont_dims]))
                    if actual_idx2 < 0 or actual_idx2 >= n_categories:
                        raise ValueError(
                            f"Indexed fingerprint value out of range: {actual_idx2}; expected [0, {n_categories - 1}]."
                        )
                    idx2 = category_order[actual_idx2]

                    k_cont = _matern_block(row1_cont, row2[:cont_dims], ls_cont)
                    k_fp = C[idx1, idx2]

                    sigma_scale = sigma_values[actual_idx1] * sigma_values[actual_idx2]
                    K[i, j] = sigma_scale * k_cont * k_fp

            return K

        def hybrid_kernel(X1, X2, hps):
            kernel_hps, sigma_values = _extract_kernel_hps(hps, kernel_end, sigma_stop)
            return _lkj_corr_kernel_impl(
                X1, X2, kernel_hps, sigma_values, category_order
            )

        return hybrid_kernel

    expected_cols = cont_dims + fingerprint_bits

    @njit(fastmath=True)
    def _bitvector_kernel_impl(X1, X2, kernel_hps):
        n1, d1 = X1.shape
        n2, d2 = X2.shape

        if d1 != expected_cols or d2 != expected_cols:
            raise ValueError(
                f"Expected X1 and X2 to have {expected_cols} columns (cont_dims + fingerprint_bits) but got {d1} and {d2}."
            )

        K = np.empty((n1, n2), dtype=np.float64)
        signal_var = kernel_hps[0]
        ls_cont = kernel_hps[1 : 1 + cont_dims]
        fp_weight = kernel_hps[1 + cont_dims]

        for i in prange(n1):
            row1 = X1[i]
            row1_cont = row1[:cont_dims]
            fp1 = row1[cont_dims:]

            for j in range(n2):
                row2 = X2[j]
                k_cont = _matern_block(row1_cont, row2[:cont_dims], ls_cont)
                k_fp = _tanimoto_int_block(fp1, row2[cont_dims:])
                k_fp = fp_weight * k_fp + (1.0 - fp_weight)
                K[i, j] = signal_var * k_cont * k_fp

        return K

    def hybrid_kernel(X1, X2, hps):
        kernel_hps, _ = _extract_kernel_hps(hps, 1 + cont_dims + 1, None)
        return _bitvector_kernel_impl(X1, X2, kernel_hps)

    return hybrid_kernel


def noise_factory_numba(
    hps_offset,
    num_noises=1,
    smiles_dict={},
    cont_dims=None,
    fingerprint_bits=None,
    *,
    fingerprint_mode: str = FINGERPRINT_MODE,
    fingerprint_metadata: FingerprintMetadata | None = None,
):
    # hps_offset is the index in the hyper-parameter vector where the noise parameters start.
    hps_slice = slice(hps_offset, hps_offset + num_noises)

    print(
        f"Noise will be assigned from hyper-parameters[{hps_slice.start}:{hps_slice.stop}] (total {num_noises} noise hyper-parameters)."
    )

    if num_noises == 1:
        noise_idx = hps_slice.start

        @njit(fastmath=True)
        def _constant_noise(x, hps):
            return np.ones(len(x)) * hps[noise_idx]

        return _constant_noise

    if num_noises != len(smiles_dict):
        raise ValueError(
            f"num_noises must be either 1 or matches the number of unique SMILES in the dataset (currently {len(smiles_dict)})."
        )

    if cont_dims is None:
        raise ValueError(
            "cont_dims must be provided when num_noises matches the number of unique SMILES."
        )

    _validate_fingerprint_mode(fingerprint_mode)

    a_sites, smiles_list = zip(*smiles_dict.items())

    if fingerprint_mode in INDEXED_FINGERPRINT_MODES:
        if fingerprint_metadata is None:
            raise ValueError(
                "fingerprint_metadata is required for fingerprint_mode="
                f"'{FINGERPRINT_MODE_INDEXED_TANIMOTO}' or '{FINGERPRINT_MODE_LKJ_CORR}'."
            )
        expected_indices = np.array(
            [fingerprint_metadata.a_site_to_index[a_site] for a_site in a_sites],
            dtype=np.int64,
        )

        print("Noise assignment: ")
        for i, a_site in enumerate(a_sites):
            print(
                f"  {a_site} (idx={expected_indices[i]}) -> noise_{i} (hps[{hps_slice.start + i}])"
            )

        expected_cols = cont_dims + 1

        @njit
        def _smiles_dependent_noise(x, hps):
            noise_hps = hps[hps_slice]
            noises = np.full(len(x), np.nan)
            if x.shape[1] != expected_cols:
                raise ValueError(
                    f"Input x has {x.shape[1]} columns but expected {expected_cols} (cont_dims + 1 index)."
                )
            for i in range(len(x)):
                fp_idx = int(np.rint(x[i, cont_dims]))
                for j in range(num_noises):
                    if fp_idx == expected_indices[j]:
                        noises[i] = noise_hps[j]
                        break
            if np.isnan(noises).any():
                raise ValueError(
                    "Some rows in x did not match any known indexed fingerprint. Check the input data and metadata."
                )
            return noises

        return _smiles_dependent_noise

    if fingerprint_bits is None:
        raise ValueError(
            "fingerprint_bits must be provided for bitvector mode when num_noises matches unique SMILES."
        )

    fps = np.array(
        [_morgan_fingerprint(smiles, fingerprint_bits) for smiles in smiles_list]
    )

    print("Noise assignment: ")
    for i, a_site in enumerate(a_sites):
        print(f"  {a_site} -> noise_{i} (hps[{hps_slice.start + i}])")

    expected_cols = cont_dims + fingerprint_bits

    @njit
    def _smiles_dependent_noise(x, hps):
        noise_hps = hps[hps_slice]
        noises = np.full(len(x), np.nan)
        if x.shape[1] != expected_cols:
            raise ValueError(
                f"Input x has {x.shape[1]} columns but expected {expected_cols} (cont_dims + fingerprint_bits)."
            )
        for i in range(len(x)):
            fp = x[i, cont_dims:]
            for j in range(num_noises):
                if np.array_equal(fp, fps[j]):
                    noises[i] = noise_hps[j]
                    break
        if np.isnan(noises).any():
            raise ValueError(
                "Some rows in x did not match any known SMILES fingerprint. Check the input data and smiles_dict."
            )
        return noises

    return _smiles_dependent_noise


my_noise = noise_factory_numba(hps_offset=-2, num_noises=1)


def mean_factory_numba(
    num_means=1,
    smiles_dict={},
    cont_dims=None,
    fingerprint_bits=None,
    *,
    fingerprint_mode: str = FINGERPRINT_MODE,
    fingerprint_metadata: FingerprintMetadata | None = None,
):
    mean_slice = slice(-num_means, None)
    print(
        f"Mean will be assigned from hyper-parameters[{mean_slice.start}:{mean_slice.stop}] (total {num_means} mean hyper-parameters)."
    )

    if num_means == 1:
        mean_idx = mean_slice.start

        @njit(fastmath=True)
        def _constant_mean(x, hps):
            return np.ones(len(x)) * hps[mean_idx]

        return _constant_mean

    if num_means != len(smiles_dict):
        raise ValueError(
            f"num_means must be either 1 or matches the number of unique SMILES in the dataset (currently {len(smiles_dict)})."
        )

    if cont_dims is None:
        raise ValueError(
            "cont_dims must be provided when num_means matches the number of unique SMILES."
        )

    _validate_fingerprint_mode(fingerprint_mode)

    a_sites, smiles_list = zip(*smiles_dict.items())

    if fingerprint_mode in INDEXED_FINGERPRINT_MODES:
        if fingerprint_metadata is None:
            raise ValueError(
                "fingerprint_metadata is required for fingerprint_mode="
                f"'{FINGERPRINT_MODE_INDEXED_TANIMOTO}' or '{FINGERPRINT_MODE_LKJ_CORR}'."
            )
        expected_indices = np.array(
            [fingerprint_metadata.a_site_to_index[a_site] for a_site in a_sites],
            dtype=np.int64,
        )

        print("Mean assignment: ")
        for i, a_site in enumerate(a_sites):
            print(
                f"  {a_site} (idx={expected_indices[i]}) -> mean_{i} (hps[{mean_slice.start + i}])"
            )

        expected_cols = cont_dims + 1

        @njit
        def _smiles_dependent_mean(x, hps):
            mean_hps = hps[mean_slice]
            means = np.full(len(x), np.nan)
            if x.shape[1] != expected_cols:
                raise ValueError(
                    f"Input x has {x.shape[1]} columns but expected {expected_cols} (cont_dims + 1 index)."
                )
            for i in range(len(x)):
                fp_idx = int(np.rint(x[i, cont_dims]))
                for j in range(num_means):
                    if fp_idx == expected_indices[j]:
                        means[i] = mean_hps[j]
                        break
            if np.isnan(means).any():
                raise ValueError(
                    "Some rows in x did not match any known indexed fingerprint. Check the input data and metadata."
                )
            return means

        return _smiles_dependent_mean

    if fingerprint_bits is None:
        raise ValueError(
            "fingerprint_bits must be provided for bitvector mode when num_means matches unique SMILES."
        )

    fps = np.array(
        [_morgan_fingerprint(smiles, fingerprint_bits) for smiles in smiles_list]
    )

    print("Mean assignment: ")
    for i, a_site in enumerate(a_sites):
        print(f"  {a_site} -> mean_{i} (hps[{mean_slice.start + i}])")

    expected_cols = cont_dims + fingerprint_bits

    @njit
    def _smiles_dependent_mean(x, hps):
        mean_hps = hps[mean_slice]
        means = np.full(len(x), np.nan)
        if x.shape[1] != expected_cols:
            raise ValueError(
                f"Input x has {x.shape[1]} columns but expected {expected_cols} (cont_dims + fingerprint_bits)."
            )
        for i in range(len(x)):
            fp = x[i, cont_dims:]
            for j in range(num_means):
                if np.array_equal(fp, fps[j]):
                    means[i] = mean_hps[j]
                    break
        if np.isnan(means).any():
            raise ValueError(
                "Some rows in x did not match any known SMILES fingerprint. Check the input data and smiles_dict."
            )
        return means

    return _smiles_dependent_mean


my_mean = mean_factory_numba(num_means=1)
