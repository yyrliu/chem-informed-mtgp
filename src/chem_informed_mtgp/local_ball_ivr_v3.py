# ------------------------------------------------------------
#  Imports
# ------------------------------------------------------------
import numpy as np
from numba import njit, prange
from scipy.sparse.csgraph import connected_components
from scipy.linalg import cholesky, solve_triangular
from joblib import Parallel, delayed
# ------------------------------------------------------------


# ------------------------------------------------------------
#  1.  Stable Cholesky + (optional) inverse
# ------------------------------------------------------------
def stable_cholesky_and_inverse(K, jitter_factor=1e-9, max_tries=5):
    """
    Return (L, K⁻¹, success) where L is the lower‑triangular Cholesky factor.
    If the matrix is not positive‑definite we add jitter to the diagonal.
    """
    jitter = jitter_factor * np.mean(np.diag(K))
    for _ in range(max_tries):
        try:
            K_j = K + np.eye(K.shape[0]) * jitter
            L = cholesky(K_j, lower=True, check_finite=False)
            # Inverse via triangular solves (no explicit matrix inversion)
            iL = solve_triangular(L, np.eye(K.shape[0]), lower=True, check_finite=False)
            K_inv = iL.T @ iL
            return L, K_inv, True
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return None, None, False


# ------------------------------------------------------------


# ------------------------------------------------------------
#  2.  Numba kernel – evaluates the reduction for ONE Monte‑Carlo set
# ------------------------------------------------------------
@njit(fastmath=True)
def _numba_calc_group_ivr_generic(
    samples, X_train, X_aug, L_aug, K_inv_train, hps, kernel_func, signal_var
):
    """
    ``samples`` – (S, d) points drawn inside the local ball.
    Returns Σ_i max(σ²_old(z_i) - σ²_new(z_i), 0).
    """
    n_samples = samples.shape[0]

    # --------------------------------------------------------
    # 2.1  Variance using the *training* data only
    # --------------------------------------------------------
    k_s_train = kernel_func(samples, X_train, hps)  # (S, N)
    term1 = k_s_train @ K_inv_train  # (S, N)

    var_current_red = np.empty(n_samples)
    for i in range(n_samples):
        dot = 0.0
        for j in range(term1.shape[1]):
            dot += term1[i, j] * k_s_train[i, j]
        var_current_red[i] = signal_var - dot

    # --------------------------------------------------------
    # 2.2  Variance after the candidate set is added
    # --------------------------------------------------------
    k_s_aug = kernel_func(samples, X_aug, hps)  # (S, N+M)
    Y = np.linalg.solve(L_aug, k_s_aug.T)  # (N+M, S)

    var_after_red = np.empty(n_samples)
    for i in range(n_samples):
        red = 0.0
        for j in range(Y.shape[0]):
            val = Y[j, i]
            red += val * val
        var_after_red[i] = signal_var - red

    # --------------------------------------------------------
    # 2.3  Positive reduction only (clip at zero)
    # --------------------------------------------------------
    total_diff = 0.0
    for i in range(n_samples):
        diff = var_current_red[i] - var_after_red[i]
        if diff > 0.0:
            total_diff += diff
    return total_diff


# ------------------------------------------------------------


@njit(parallel=True, fastmath=True)  # <‑‑ enable parallel execution
def _numba_calc_group_ivr_generic_parallel(
    samples, X_train, X_aug, L_aug, K_inv_train, hps, kernel_func, signal_var
):
    """
    Parallel version that loops over the Monte‑Carlo samples.
    The reduction `total_diff` is a **scalar reduction** – Numba recognises the
    += pattern and safely aggregates the partial sums from all threads.
    """
    n_samples = samples.shape[0]

    # -----------------------------------------------------------------
    # 1) Quantities that can be computed *once* (outside the parallel loop)
    # -----------------------------------------------------------------
    # k_s_train @ K_inv_train  → term1 (n_samples × n_train)
    k_s_train = kernel_func(samples, X_train, hps)
    term1 = k_s_train @ K_inv_train

    # Solve L_aug • Y = k_s_aug.T   →  Y (n_aug × n_samples)
    k_s_aug = kernel_func(samples, X_aug, hps)
    Y = np.linalg.solve(L_aug, k_s_aug.T)

    # -----------------------------------------------------------------
    # 2) Parallel reduction over the Monte‑Carlo draws
    # -----------------------------------------------------------------
    total_diff = 0.0  # <-- scalar reduction variable

    for i in prange(n_samples):  # <-- parallel loop
        # ---- current variance (training data only) -----------------
        dot = 0.0
        for j in range(term1.shape[1]):  # Σ_j term1[i,j] * k_s_train[i,j]
            dot += term1[i, j] * k_s_train[i, j]
        var_current = signal_var - dot

        # ---- variance after adding the candidate set ---------------
        red = 0.0
        for j in range(Y.shape[0]):  # Σ_j Y[j,i]^2
            val = Y[j, i]
            red += val * val
        var_after = signal_var - red

        # ---- positive reduction only --------------------------------
        diff = var_current - var_after
        if diff > 0.0:
            total_diff += diff  # <-- reduction (handled by Numba)

    return total_diff


# ------------------------------------------------------------
#  3.  Core routine that works on a **single** candidate set
# ------------------------------------------------------------
def _local_ball_ivr_one(
    x,
    gp_optimizer,
    bounds,
    kernel_func,
    ls_indices,
    noise_index,
    signal_index,
    continuous_indices,
    n_samples_per_group=2048,
    deterministic=False,
    seed=None,
    radius_multiplier=2.0,
    cache=None,
    rng=None,
    use_parallel=True,
):
    """
    Compute the Integrated‑Variance‑Reduction for ONE candidate set
    ``x`` (shape (n_cand, d)).  Returns a *float* – the IVR value for that
    set.  ``cache`` is a dict that stores the training‑only Cholesky factors;
    it is shared across batches when the outer wrapper calls this function
    repeatedly.
    """

    if use_parallel:
        _numba_calc_group_ivr = _numba_calc_group_ivr_generic_parallel
    else:
        _numba_calc_group_ivr = _numba_calc_group_ivr_generic

    if cache is None:
        cache = {}

    # --------------------------------------------------------
    # RNG – keep deterministic behaviour if requested
    # --------------------------------------------------------
    if rng is None:
        if deterministic:
            base = 42 if seed is None else int(seed) % (2**32)
            rng = np.random.RandomState(base)
        else:
            rng = np.random.RandomState()

    # --------------------------------------------------------
    # Ensure x is 2‑D
    # --------------------------------------------------------
    if x.ndim == 1:
        x = x.reshape(1, -1)  # (1, d)

    # --------------------------------------------------------
    # Grab training data & hyper‑parameters
    # --------------------------------------------------------
    hps = gp_optimizer.get_hyperparameters()
    X_train = gp_optimizer.x_data
    n_train = X_train.shape[0]

    # --------------------------------------------------------
    # ---- CACHE : training‑only inverse (K_inv_train) ----------
    # --------------------------------------------------------
    stale = True
    if "K_inv_train" in cache:
        if cache.get("n_train") == n_train and np.allclose(
            cache.get("hps", []), hps, atol=1e-9
        ):
            stale = False

    if stale:
        K_train = kernel_func(X_train, X_train, hps)
        K_train_noisy = K_train + np.eye(n_train) * hps[noise_index]
        _, K_inv_train, success = stable_cholesky_and_inverse(K_train_noisy)
        if not success:
            return -1.0
        cache.update(
            {
                "K_inv_train": K_inv_train,
                "K_train_noisy": K_train_noisy,
                "n_train": n_train,
                "hps": hps.copy(),
            }
        )
    else:
        K_inv_train = cache["K_inv_train"]
        K_train_noisy = cache["K_train_noisy"]

    # --------------------------------------------------------
    # ---- BUILD augmented matrix K_aug (train + candidates) ----
    # --------------------------------------------------------
    n_cand = x.shape[0]
    K_cand = kernel_func(x, x, hps) + np.eye(n_cand) * hps[noise_index]
    K_cross = kernel_func(X_train, x, hps)  # (n_train, n_cand)

    K_aug = np.block([[K_train_noisy, K_cross], [K_cross.T, K_cand]])
    L_aug, _, success_aug = stable_cholesky_and_inverse(K_aug)
    if not success_aug:
        return -1.0

    X_aug = np.vstack([X_train, x])  # (n_train+n_cand, d)

    # --------------------------------------------------------
    # ---- GROUPING of the candidate points
    # --------------------------------------------------------
    ls_vals = hps[ls_indices]
    cont_radii = ls_vals * radius_multiplier
    X_cont = x[:, continuous_indices]

    # Scaled Euclidean distances in the continuous sub‑space
    diff = X_cont[:, None, :] - X_cont[None, :, :]  # (n_cand, n_cand, n_cont)
    safe_radii = np.maximum(cont_radii, 1e-6)
    distances_scaled = np.linalg.norm(diff / safe_radii, axis=2)

    # Discrete dimensions: enforce an infinite distance whenever any bit differs
    all_dims = np.arange(x.shape[1])
    discrete_indices = np.setdiff1d(all_dims, continuous_indices)
    if discrete_indices.size:
        X_disc = x[:, discrete_indices]
        disc_mismatch = (X_disc[:, None, :] != X_disc[None, :, :]).any(axis=2)
        distances_scaled[disc_mismatch] = np.inf

    n_groups, labels = connected_components(distances_scaled < 2.0, directed=False)

    # --------------------------------------------------------
    # ---- MONTE‑CARLO sampling + variance reduction per group
    # --------------------------------------------------------
    total_ivr = 0.0
    sigma_f2 = hps[signal_index]

    for gid in range(n_groups):
        mask = labels == gid
        group_pts = x[mask]  # (n_g, d)
        centre_ref = group_pts[0]

        if group_pts.shape[0] == 1:
            # ---------- SINGLE‑POINT GROUP (hypersphere) ----------
            n_cont = len(continuous_indices)

            # Guard against the pathological case “no continuous dims”
            if n_cont == 0:
                # Nothing to perturb – every sample is just the centre copy
                samples = np.tile(centre_ref, (n_samples_per_group, 1))
            else:
                # 1. random direction on the unit sphere in ℝⁿᶜ
                direction = rng.randn(n_samples_per_group, n_cont)
                direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9

                # 2. radius that yields a *uniform* distribution in the ball
                radius = rng.uniform(0.0, 1.0, n_samples_per_group) ** (1.0 / n_cont)

                # 3. perturbation = radius * direction * per‑dim radii
                perturb = direction * (radius[:, None] * cont_radii)

                # 4. start from the centre (full dimensional) and overwrite cont. cols
                samples = np.tile(centre_ref, (n_samples_per_group, 1))
                samples[:, continuous_indices] += perturb  # only cont. part changes

        else:
            # ---------- MULTI‑POINT GROUP (axis‑aligned hyper‑rectangle) ----------
            n_cont = len(continuous_indices)

            # If there are *no* continuous dimensions we just copy the centre.
            if n_cont == 0:
                samples = np.tile(centre_ref, (n_samples_per_group, 1))
            else:
                # 1. compute per‑dim lower / upper bounds that include the whole group
                lo_c = np.min(
                    group_pts[:, continuous_indices] - cont_radii, axis=0
                )  # (n_cont,)
                hi_c = np.max(
                    group_pts[:, continuous_indices] + cont_radii, axis=0
                )  # (n_cont,)

                # 2. allocate the full matrix by tiling the centre row
                samples = np.tile(centre_ref, (n_samples_per_group, 1))

                # 3. fill only the continuous slice with a uniform draw
                samples[:, continuous_indices] = rng.uniform(
                    lo_c, hi_c, size=(n_samples_per_group, n_cont)
                )
        # -----------------------------------------------------------------
        # At this point `samples` has shape (n_samples_per_group, total_dims)
        # *continuous* columns contain a valid Monte‑Carlo draw,
        # *fingerprint* columns are an exact copy of `centre_ref`.
        # -----------------------------------------------------------------

        # -------  b) enforce user bounds ---------------------------------
        cont_samples = samples[:, continuous_indices]
        bound_arr = bounds[continuous_indices]  # (n_cont, 2)
        inside = np.all(
            (cont_samples >= bound_arr[:, 0]) & (cont_samples <= bound_arr[:, 1]),
            axis=1,
        )
        valid = samples[inside]
        if valid.shape[0] < 10:  # not enough points
            continue

        # -------  c) variance reduction for the *valid* MC points -------
        group_sum = _numba_calc_group_ivr(
            valid,
            X_train,
            X_aug,
            L_aug,
            K_inv_train,
            hps,
            kernel_func,
            sigma_f2,
        )
        total_ivr += group_sum / n_samples_per_group

    return total_ivr


# ------------------------------------------------------------


# ------------------------------------------------------------
#  4.  Public entry point – batched version (Option A)
# ------------------------------------------------------------
def local_ball_ivr(
    x,
    gp_optimizer,
    bounds,
    kernel_func,
    ls_indices,
    noise_index,
    signal_index,
    continuous_indices,
    n_samples_per_group=2048,
    deterministic=False,
    seed=None,
    radius_multiplier=2.0,
    use_parallel=True,
    use_batch_parallel=True,
    n_jobs=-1,
    _cache=None,
):
    """
    Integrated‑Variance‑Reduction for one or many candidate sets.

    Parameters
    ----------
    x : ndarray
        Either shape (n_candidates, d)   – a single set,
        or shape (n_batches, n_candidates, d) – multiple independent sets.
        The function treats the first dimension as a *batch* dimension.
    gp_optimizer : object
        Must provide ``x_data`` (training inputs) and ``get_hyperparameters()``.
    bounds : ndarray (n_continuous, 2)
        Lower / upper bounds for the continuous dimensions.
    kernel_func : callable
        ``kernel_func(X1, X2, hps) → covariance matrix``.
    ls_indices, noise_index, signal_index, continuous_indices : int / list[int]
        Indexing into the hyper‑parameter vector as used in the original code.
    n_samples_per_group : int, default 2048
        Number of Monte‑Carlo points drawn per group.
    deterministic : bool, default False
        If True the random number generator is seeded (42 or ``seed``) so that
        results are reproducible.
    seed : int or None, default None
        Optional user seed; ignored when ``deterministic=False``.
    radius_multiplier : float, default 2.0
        Multiplies the length‑scale to obtain the ball radius.
    use_parallel : bool, default True
        Whether to use the parallel Numba implementation (prange) for Monte Carlo
        variance reduction. If False, uses sequential Numba implementation.
    use_batch_parallel : bool, default True
        Whether to use parallel batch evaluation via joblib. If True and n_batches > 1,
        batches are evaluated in parallel using multiple processes. If is a joblib.Parallel instance, it will be used directly (allows custom backends). If False, batches are processed sequentially.
    n_jobs : int, default -1
        Number of parallel jobs for batch processing (-1 uses all CPUs).
        Only used when use_batch_parallel=True and n_batches > 1.
    _cache : dict or None, default None
        Private cache that stores the training‑only Cholesky factor and its
        inverse.  Supplying the same dict across multiple calls re‑uses the
        factorisation (identical to the mutable‑default behaviour of the
        original implementation).

    Returns
    -------
    np.ndarray
        Shape (n_batches,) – the IVR value for each candidate set.
        If a single set was supplied the shape will be ``(1,)`` (compatible
        with the original return type ``np.array([value])``).
    """
    # --------------------------------------------------------
    # Normalise input to a 3‑D array
    # --------------------------------------------------------
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:  # single set → make a batch of size 1
        x = x[None, ...]  # shape (1, n_cand, d)
    elif x.ndim != 3:
        raise ValueError("x must be 2‑D (n_cand, d) or 3‑D (batch, n_cand, d)")
    n_batches = x.shape[0]

    # --------------------------------------------------------
    # Prepare a deterministic RNG if requested – we will give each batch
    # its own RNG that is a deterministic offset of the base seed.
    # --------------------------------------------------------
    if deterministic:
        base_seed = 42 if seed is None else int(seed) % (2**32)
        base_rng = np.random.RandomState(base_seed)
    else:
        base_rng = None  # will be ignored

    # --------------------------------------------------------
    # Shared cache for the training‑only factorisation
    # --------------------------------------------------------
    cache = _cache if _cache is not None else {}

    # --------------------------------------------------------
    # Loop over batches – each batch is completely independent
    # --------------------------------------------------------

    # Decide whether to use batch-level parallelization
    if not use_batch_parallel or n_batches == 1 or n_jobs == 1:
        # Sequential processing (original behavior)
        ivr_vals = np.empty(n_batches, dtype=np.float64)

        for b in range(n_batches):
            # Create a per‑batch RNG (deterministic offset if needed)
            if deterministic:
                rng = np.random.RandomState()
                rng.set_state(base_rng.get_state())
            else:
                rng = np.random.RandomState()

            ivr_vals[b] = _local_ball_ivr_one(
                x[b],  # (n_cand, d)
                gp_optimizer,
                bounds,
                kernel_func,
                ls_indices,
                noise_index,
                signal_index,
                continuous_indices,
                n_samples_per_group=n_samples_per_group,
                deterministic=deterministic,
                seed=seed,
                radius_multiplier=radius_multiplier,
                cache=cache,
                rng=rng,
                use_parallel=use_parallel,
            )

        return ivr_vals

    # --------------------------------------------------------
    # Parallel processing across batches using joblib
    # --------------------------------------------------------
    def evaluate_batch(batch_idx):
        """Worker function for parallel batch evaluation."""
        # Each worker gets its own RNG with offset seed for deterministic mode
        if deterministic:
            worker_seed = (base_seed + batch_idx) % (2**32)
            worker_rng = np.random.RandomState(worker_seed)
        else:
            worker_rng = np.random.RandomState()

        # Use the pre-computed cache (copied at fork time) to avoid recomputing training factorization
        # Workers only READ from cache (no writes if already populated), so no race conditions
        worker_cache = cache.copy() if cache else {}

        ivr_val = _local_ball_ivr_one(
            x[batch_idx],
            gp_optimizer,
            bounds,
            kernel_func,
            ls_indices,
            noise_index,
            signal_index,
            continuous_indices,
            n_samples_per_group=n_samples_per_group,
            deterministic=deterministic,
            seed=seed,
            radius_multiplier=radius_multiplier,
            cache=worker_cache,  # Pre-computed cache (read-only)
            rng=worker_rng,
            use_parallel=use_parallel,
        )

        return ivr_val

    if isinstance(use_batch_parallel, Parallel):
        ivr_vals = use_batch_parallel(
            delayed(evaluate_batch)(b) for b in range(n_batches)
        )
    else:
        # Execute in parallel
        ivr_vals = Parallel()(delayed(evaluate_batch)(b) for b in range(n_batches))

    return np.array(ivr_vals, dtype=np.float64)


# ------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------
    #  Example (uses the hybrid kernel you posted elsewhere)
    # -----------------------------------------------------------------

    # Dummy objects that mimic the optimizer API used by the function
    class DummyGP:
        def __init__(self, X, hps):
            self.x_data = X
            self._hps = hps

        def get_hyperparameters(self):
            return self._hps.copy()

    # --- 1. Helper: Continuous Logic (Matern 3/2) ---
    @njit(fastmath=True, inline="always")
    def _matern_block(row1, row2, lengthscales):
        """
        Calculates Matern 3/2 correlation for the first 2 dimensions.
        """
        # Calculate Anisotropic Euclidean Distance
        dist_sq = 0.0
        for k in range(2):  # Iterate over continuous dims (0 and 1)
            diff = (row1[k] - row2[k]) / lengthscales[k]
            dist_sq += diff * diff

        dist = np.sqrt(dist_sq)

        # Matern 3/2 Formula
        sqrt3_d = 1.73205080757 * dist
        return (1.0 + sqrt3_d) * np.exp(-sqrt3_d)

    # --- 2. Helper: Discrete Logic (Tanimoto with Padding) ---
    @njit(fastmath=True, inline="always")
    def _tanimoto_block(row1, row2, n_total_dims):
        """
        Calculates Tanimoto similarity for dims 2+, adding 6 virtual '1' bits.
        """
        intersection = 0.0
        sum1 = 0.0
        sum2 = 0.0

        # Iterate from dim 2 to the end
        for k in range(2, n_total_dims):
            bit1 = row1[k]
            bit2 = row2[k]

            intersection += bit1 * bit2
            sum1 += bit1
            sum2 += bit2

        # Add the "Virtual Padding" (6 bits of 1s)
        # This mimics: np.append(fp, np.ones(6))
        PADDING = 6.0
        intersection += PADDING
        sum1 += PADDING
        sum2 += PADDING

        union = sum1 + sum2 - intersection

        if union > 1e-9:
            return intersection / union
        return 0.0

    # --- 3. Main Kernel Entry Point ---
    @njit(fastmath=True, cache=True)
    def my_hybrid_kernel(X1, X2, hps):
        """
        Numba-optimized Hybrid Kernel.
        Structure: Signal * Matern32(Continuous) * Tanimoto(Discrete)
        """
        n1, n_dims = X1.shape
        n2 = X2.shape[0]

        # Pre-allocate Output
        K = np.zeros((n1, n2))

        # Extract Hyperparameters for readability
        signal_var = hps[0]
        ls_cont = hps[1:3]  # Lengthscales for dims 0, 1

        # Parallel Loop
        for i in prange(n1):
            for j in range(n2):
                # A. Continuous contribution
                k_cont = _matern_block(X1[i], X2[j], ls_cont)

                # B. Discrete contribution
                k_disc = _tanimoto_block(X1[i], X2[j], n_dims)

                # C. Combine
                K[i, j] = signal_var * k_cont * k_disc

        return K

    # Toy data
    np.random.seed(0)
    X_train = np.random.rand(30, 8)  # 30 training points, 8 dims
    hps = np.array([1.0, 0.5, 0.7, 0.01, 0.0])  # [signal, ls0, ls1, noise, mean]

    gp = DummyGP(X_train, hps)

    # Candidate sets – two independent batches, each with 5 points
    candidates_batch = np.random.rand(2, 5, 8)  # (batch, n_cand, d)

    # Continuous bounds (only the first two dimensions are continuous)
    CONTINUOUS_BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])

    # Call the batched IVR routine
    ivr_per_batch = local_ball_ivr(
        candidates_batch,
        gp,
        CONTINUOUS_BOUNDS,
        kernel_func=my_hybrid_kernel,  # or my_hybrid_kernel_fp_weight
        ls_indices=[1, 2],
        noise_index=3,
        signal_index=0,
        continuous_indices=[0, 1],
        n_samples_per_group=1024,
        deterministic=True,
        seed=123,
        radius_multiplier=2.0,
    )

    print("IVR values for the two batches:", ivr_per_batch)
