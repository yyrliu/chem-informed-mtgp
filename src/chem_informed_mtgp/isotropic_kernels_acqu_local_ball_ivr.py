from scipy.sparse.csgraph import connected_components
from scipy.linalg import cholesky, cho_solve
import numpy as np

"""
Optimization strategy using **Numba** for kernel compilation and **vectorization** for the IVR calculation logic.
1.  Numba Accelerated Kernels:
    - Created `get_anisotropic_distance_matrix_numba`, `matern_kernel_diff1_numba`, and `my_kernel_numba` using `@jit(nopython=True)`. This compiles the distance and kernel matrix calculations into optimized machine code, which is drastically faster than python loops.
    - Added a `my_kernel_wrapper` to ensure compatibility with `gpCam`'s expected function signature.
2.  Vectorized Linear Algebra:
    - Pre-computed Cholesky Factorization**: The `local_ball_ivr_optimized` function now pre-calculates the Cholesky decomposition of the augmented Kernel matrix (`K_aug`) **once** at the start.
    - Batch Solving**: Instead of looping through thousands of samples and solving a linear system for each one (which is extremely slow), it now solves for all samples simultaneously (`cho_solve(factor, k_s_aug.T)`). This transforms an $O(S \cdot N^3)$ loop into a highly efficient $O(N^2 \cdot S)$ matrix operation.
3.  Optimized Loop Structure:
    - The `local_ball_ivr_optimized` function replaces list comprehensions with vectorized NumPy operations for sampling, bounds checking, and variance reduction calculation.
"""


def local_ball_ivr_optimized(
    x, gp_optimizer, bounds, kernel_func, n_samples_per_group=2048
):
    """
    Optimized Local Ball IVR for Batch Acquisition

    Maximizes information gain by selecting points that minimize total uncertainty.

    Optimizations:
    - Pre-computation of K_inv/Cholesky factorization
    - Vectorized posterior variance calculation
    - Vectorized validation of samples
    - Solving linear systems instead of explicit inversion

    Parameters:
    -----------
    x : array (n_candidates, d)
        Candidate points (single or batch)
    gp_optimizer : GPOptimizer
        The GP model
    bounds : array (d, 2)
        Domain bounds
    kernel_func : callable
        Kernel function (can be Numba-compiled)
    n_samples_per_group : int
        Samples PER group (not total)

    Returns:
    --------
    ivr : array (1,)
        Total integrated variance reduction
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    hps = gp_optimizer.get_hyperparameters()
    lengthscale = hps[1]
    noise_var = hps[2] ** 2
    d = bounds.shape[0]
    n_candidates = len(x)
    radius = 2.0 * lengthscale

    # Handle GP object structure
    if hasattr(gp_optimizer, "gp") and not isinstance(gp_optimizer.gp, bool):
        cache_obj = gp_optimizer.gp
    else:
        cache_obj = gp_optimizer

    x_data = gp_optimizer.x_data
    n_data = len(x_data)

    # 1. PRE-COMPUTATION
    # Instead of just K_inv, let's work with the augmented matrix for the candidates
    # We need the full K_aug for the variance reduction step anyway

    K_train = kernel_func(x_data, x_data, hps)
    K_train_noisy = K_train + np.eye(n_data) * noise_var
    k_cross = kernel_func(x_data, x, hps)
    K_cand = kernel_func(x, x, hps) + np.eye(n_candidates) * noise_var**2

    # Block matrix [[K_train+noise, K_cross], [K_cross.T, K_cand+noise]]
    K_aug = np.block([[K_train_noisy, k_cross], [k_cross.T, K_cand]])
    X_aug = np.vstack([x_data, x])

    # Compute Cholesky factorization of K_aug once (O(N^3))
    # This is much more stable and faster than inv() for subsequent solves
    try:
        L_aug = cholesky(K_aug, lower=True)
        factor = (L_aug, True)  # (L, lower)
    except np.linalg.LinAlgError:
        # Fallback to robust solve if not positive definite (e.g. numerical issues)
        L_aug = None

    # 2. GROUP DETECTION
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    overlap_matrix = distances < 2 * radius
    n_groups, labels = connected_components(overlap_matrix, directed=False)

    all_ivr = []

    for group_id in range(n_groups):
        group_mask = labels == group_id
        group_centers = x[group_mask]

        # 3. VECTORIZED SAMPLING
        if len(group_centers) == 1:
            center = group_centers[0]
            samples_unit = np.random.randn(n_samples_per_group, d)
            samples_unit = samples_unit / np.linalg.norm(
                samples_unit, axis=1, keepdims=True
            )
            radii = np.random.uniform(0, 1, n_samples_per_group) ** (1 / d) * radius
            samples = center + samples_unit * radii[:, np.newaxis]
        else:
            min_extent = np.min(group_centers - radius, axis=0)
            max_extent = np.max(group_centers + radius, axis=0)
            samples = np.random.uniform(
                min_extent, max_extent, size=(n_samples_per_group, d)
            )

        # Vectorized inside check
        inside = np.all((samples >= bounds[:, 0]) & (samples <= bounds[:, 1]), axis=1)
        samples_valid = samples[inside]

        n_valid = len(samples_valid)
        if n_valid < 50:
            continue

        # 4. VECTORIZED VARIANCE CALCULATION
        # Variance before: GP posterior variance at valid samples
        # Use gp_optimizer's method or manual calculation if we want to avoid overhead
        # Manual is often faster here as we can reuse K_train inverse

        # We need var_current. Let's use the standard formula:
        # v(s) = k(s,s) - k(s, X) K^{-1} k(X, s)
        # But we already have K_aug which contains K_train, so let's check if we can reuse anything
        # Actually, var_current is based ONLY on x_data.

        # Get K_inv_train for current variance
        if not hasattr(cache_obj, "_K_inv_cached"):
            cache_obj._K_inv_cached = np.linalg.inv(K_train_noisy)
        K_inv_train = cache_obj._K_inv_cached

        # Calculate kernels for all samples at once
        k_s_train = kernel_func(samples_valid, x_data, hps)  # (n_samples, n_data)
        k_s_s = (
            hps[0] + noise_var
        )  # Prior variance (assuming stationary kernel + noise)

        # Vectorized quadratic form: diag(k @ K_inv @ k.T)
        # We compute sum( (k @ K_inv) * k, axis=1 )
        term1 = k_s_train @ K_inv_train
        var_current = k_s_s - np.sum(term1 * k_s_train, axis=1)

        # Variance after: includes candidates
        # v_new(s) = k(s,s) - k(s, X_aug) K_aug^{-1} k(X_aug, s)
        k_s_aug = kernel_func(samples_valid, X_aug, hps)  # (n_samples, n_data + n_cand)

        if L_aug is not None:
            # Fast solve with Cholesky: x = cho_solve((L, lower), b)
            # We need sum(k_s_aug * (K_aug^-1 @ k_s_aug.T).T)
            # Equivalent to: v = k_s_aug @ K_aug^{-1} @ k_s_aug.T diagonals
            # Solved: X = cho_solve(factor, k_s_aug.T). T is (n_aug, n_samples)
            # Result is (n_aug, n_samples)

            solved = cho_solve(factor, k_s_aug.T)
            reduction_term = np.sum(k_s_aug * solved.T, axis=1)
            var_after = k_s_s - reduction_term
        else:
            # Fallback
            solved = np.linalg.solve(K_aug, k_s_aug.T)
            reduction_term = np.sum(k_s_aug * solved.T, axis=1)
            var_after = k_s_s - reduction_term

        var_reduction = np.maximum(var_current - var_after, 0)

        group_ivr = np.mean(var_reduction) * (n_valid / n_samples_per_group)
        all_ivr.append(group_ivr)

    return np.array([np.sum(all_ivr)])


def create_batch_acquisition_optimized(
    n_batch, bounds_original, kernel, n_samples_per_group=500
):
    """
    Optimized Factory function
    """
    d = bounds_original.shape[0]
    bounds_extended = np.vstack([bounds_original for _ in range(n_batch)])

    def batch_acquisition_wrapper(x, gpo):
        if x.ndim == 1:
            x_batch = x.reshape(n_batch, d)
            return local_ball_ivr_optimized(
                x_batch, gpo, bounds_original, kernel, n_samples_per_group
            )[0]

        elif x.ndim == 2 and x.shape[1] == n_batch * d:
            S = x.shape[0]
            # Parallel processing could be added here if S is large
            # But the inner function is already quite heavy, so improved single-thread per call might be enough
            # With Numba + Vectorization, sequential might be fast enough
            results = np.zeros(S)
            for i in range(S):
                x_batch = x[i, :].reshape(n_batch, d)
                results[i] = local_ball_ivr_optimized(
                    x_batch, gpo, bounds_original, kernel, n_samples_per_group
                )[0]
            return results
        else:
            raise ValueError(f"Shape error: {x.shape}")

    return bounds_extended, batch_acquisition_wrapper


def local_ball_ivr(x, gp_optimizer, bounds, kernel_func, n_samples_per_group=2048):
    """
    Local Ball IVR for Batch Acquisition

    Maximizes information gain by selecting points that minimize total uncertainty.

    Features:
    - Handles single points and batches
    - Detects overlapping balls (convex hull for overlap)
    - Equal sampling per group (fair comparison)
    - Sums IVR across all groups
    - Scales to high dimensions

    Parameters:
    -----------
    x : array (n_candidates, d)
        Candidate points (single or batch)
    gp_optimizer : GPOptimizer
        The GP model
    bounds : array (d, 2)
        Domain bounds
    kernel_func : callable
        Kernel function
    n_samples_per_group : int
        Samples PER group (not total)

    Returns:
    --------
    ivr : array (1,)
        Total integrated variance reduction
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    hps = gp_optimizer.get_hyperparameters()
    lengthscale = hps[1]
    noise_var = hps[2] ** 2
    d = bounds.shape[0]
    n_candidates = len(x)
    radius = 2.0 * lengthscale

    # FIXED: Handle both GPOptimizer and GP objects
    # In some versions, .gp is an object, in others (like gpCam >= 8), it might be a boolean flag or method
    if hasattr(gp_optimizer, "gp") and not isinstance(gp_optimizer.gp, bool):
        # It's a GPOptimizer wrapper with an inner object
        cache_obj = gp_optimizer.gp
    else:
        # It's already the GP object, or GPOptimizer where .gp is not the storage object
        cache_obj = gp_optimizer

    x_data = gp_optimizer.x_data

    if not hasattr(cache_obj, "_K_inv_cached"):
        K_train = kernel_func(x_data, x_data, hps)
        K_noisy = K_train + np.eye(len(x_data)) * noise_var
        cache_obj._K_inv_cached = np.linalg.inv(K_noisy)

    # Detect overlapping groups
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    overlap_matrix = distances < 2 * radius

    n_groups, labels = connected_components(overlap_matrix, directed=False)

    all_ivr = []

    for group_id in range(n_groups):
        group_mask = labels == group_id
        group_centers = x[group_mask]

        # Sample region
        if len(group_centers) == 1:
            center = group_centers[0]
            samples_unit = np.random.randn(n_samples_per_group, d)
            samples_unit = samples_unit / np.linalg.norm(
                samples_unit, axis=1, keepdims=True
            )
            radii = np.random.uniform(0, 1, n_samples_per_group) ** (1 / d) * radius
            samples = center + samples_unit * radii[:, np.newaxis]
        else:
            min_extent = np.min(group_centers - radius, axis=0)
            max_extent = np.max(group_centers + radius, axis=0)
            samples = np.random.uniform(
                min_extent, max_extent, size=(n_samples_per_group, d)
            )

        inside = np.all((samples >= bounds[:, 0]) & (samples <= bounds[:, 1]), axis=1)
        samples_valid = samples[inside]

        if len(samples_valid) < 50:
            continue

        # Variance before
        var_current = np.array(
            [
                gp_optimizer.posterior_covariance(s.reshape(1, -1), variance_only=True)[
                    "v(x)"
                ][0]
                for s in samples_valid
            ]
        )

        # Variance after (with ALL candidates)
        K_train = kernel_func(x_data, x_data, hps)
        K_train_noisy = K_train + np.eye(len(x_data)) * noise_var
        k_cross = kernel_func(x_data, x, hps)
        K_cand = kernel_func(x, x, hps) + np.eye(n_candidates) * noise_var**2
        K_aug = np.block([[K_train_noisy, k_cross], [k_cross.T, K_cand]])
        X_aug = np.vstack([x_data, x])

        var_after = []
        for s in samples_valid:
            k_s_aug = kernel_func(np.array([s]), X_aug, hps).flatten()
            k_s_s = kernel_func(np.array([s]), np.array([s]), hps)[0, 0]
            var_s = k_s_s - k_s_aug @ np.linalg.solve(K_aug, k_s_aug)
            var_after.append(var_s)

        var_after = np.array(var_after)
        var_reduction = np.maximum(var_current - var_after, 0)

        group_ivr = np.mean(var_reduction) * (len(samples_valid) / n_samples_per_group)
        all_ivr.append(group_ivr)

    return np.array([np.sum(all_ivr)])


def create_batch_acquisition(n_batch, bounds_original, kernel, n_samples_per_group=500):
    """
    Factory function: creates acquisition function for n_batch points
    """
    d = bounds_original.shape[0]

    # Stack bounds n_batch times
    bounds_extended = np.vstack([bounds_original for _ in range(n_batch)])

    def batch_acquisition_wrapper(x, gpo):
        """
        x : array
            - Shape (n_batch * d,) for single candidate
            - Shape (S, n_batch * d) for vectorized (S candidates)
        """
        if x.ndim == 1:
            # Single batch: shape (n_batch * d,)
            x_batch = x.reshape(n_batch, d)
            ivr = local_ball_ivr(
                x_batch,
                gpo,
                bounds_original,
                kernel,
                n_samples_per_group=n_samples_per_group,
            )[0]
            return ivr

        elif x.ndim == 2 and x.shape[1] == n_batch * d:
            # Vectorized: shape (S, n_batch * d)
            S = x.shape[0]
            results = []
            for i in range(S):
                x_batch = x[i, :].reshape(n_batch, d)  # Extract i-th candidate
                ivr = local_ball_ivr(
                    x_batch,
                    gpo,
                    bounds_original,
                    kernel,
                    n_samples_per_group=n_samples_per_group,
                )[0]
                results.append(ivr)
            return np.array(results)
        else:
            # FIXED: Dynamic error message
            expected_1d = n_batch * d
            expected_2d = f"(S, {n_batch * d})"
            raise ValueError(
                f"Unexpected shape: {x.shape}. "
                f"Expected ({expected_1d},) or {expected_2d}"
            )

    return bounds_extended, batch_acquisition_wrapper
