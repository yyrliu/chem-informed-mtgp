from scipy.sparse.csgraph import connected_components
from scipy.linalg import cholesky, cho_solve
import numpy as np


def safe_cholesky(K, max_tries):
    """Cholesky decomposition with automatic jitter"""
    K_try = K.copy()
    jitter = 1e-8 * np.mean(np.diag(K))

    for i in range(max_tries):
        try:
            return cholesky(K_try, lower=True)
        except np.linalg.LinAlgError:
            if i == max_tries - 1:
                raise
            K_try = K + np.eye(len(K)) * jitter
            jitter *= 10


def local_ball_ivr(x, gp_optimizer, bounds, kernel_func, n_samples_per_group=2048):
    """
    Optimized Local Ball IVR for Batch Acquisition

    Maximizes information gain by selecting points that minimize total uncertainty.

    Optimizations:
    - Pre-computation of Cholesky factorization
    - Vectorized posterior variance calculation
    - Batch solving for all samples simultaneously

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

    # Get dimensionality
    d = bounds.shape[0]

    hps = gp_optimizer.get_hyperparameters()

    # Extract hyperparameters correctly
    signal_var = hps[0]  # Signal variance
    lengthscales = np.array(hps[1 : 1 + d])  # Lengthscales (array of length d)
    noise_var = hps[1 + d]  # Noise variance

    n_candidates = len(x)

    # Ellipsoid radii: different per dimension
    radii = 2.0 * lengthscales  # Shape: (d,)

    # Handle GP object
    if hasattr(gp_optimizer, "gp") and not isinstance(gp_optimizer.gp, bool):
        cache_obj = gp_optimizer.gp
    else:
        cache_obj = gp_optimizer

    x_data = gp_optimizer.x_data
    n_data = len(x_data)

    # PRE-COMPUTATION
    K_train = kernel_func(x_data, x_data, hps)
    K_train_noisy = K_train + np.eye(n_data) * noise_var
    k_cross = kernel_func(x_data, x, hps)
    K_cand = kernel_func(x, x, hps) + np.eye(n_candidates) * noise_var

    K_aug = np.block([[K_train_noisy, k_cross], [k_cross.T, K_cand]])
    X_aug = np.vstack([x_data, x])

    # Cholesky factorization with jitter
    try:
        L_aug = safe_cholesky(K_aug, 5)
        factor = (L_aug, True)
        use_cholesky = True
    except np.linalg.LinAlgError:
        use_cholesky = False
        print("Warning: Falling back to direct solve")

    # GROUP DETECTION (using scaled/normalized distance)
    # Two ellipsoids overlap if their normalized distance < 2
    diff = x[:, None, :] - x[None, :, :]  # Shape: (n_cand, n_cand, d)
    scaled_diff = diff / lengthscales  # Scale by lengthscale per dimension
    distances_scaled = np.linalg.norm(scaled_diff, axis=2)
    overlap_matrix = distances_scaled < 2.0

    n_groups, labels = connected_components(overlap_matrix, directed=False)

    all_ivr = []

    for group_id in range(n_groups):
        group_mask = labels == group_id
        group_centers = x[group_mask]

        # SAMPLING IN ELLIPSOID
        if len(group_centers) == 1:
            # Single ellipsoid: sample from unit sphere, then scale by radii
            center = group_centers[0]
            samples_unit = np.random.randn(n_samples_per_group, d)
            samples_unit = samples_unit / np.linalg.norm(
                samples_unit, axis=1, keepdims=True
            )
            # Uniform radial distribution, then scale each dimension
            r = np.random.uniform(0, 1, n_samples_per_group) ** (1 / d)
            samples = center + samples_unit * (r[:, np.newaxis] * radii)
        else:
            # Multiple overlapping ellipsoids: use bounding box
            min_extent = np.min(group_centers - radii, axis=0)
            max_extent = np.max(group_centers + radii, axis=0)
            samples = np.random.uniform(
                min_extent, max_extent, size=(n_samples_per_group, d)
            )

        inside = np.all((samples >= bounds[:, 0]) & (samples <= bounds[:, 1]), axis=1)
        samples_valid = samples[inside]
        n_valid = len(samples_valid)

        if n_valid < 50:
            continue

        # VECTORIZED VARIANCE CALCULATION

        # Cache K_inv for current variance (with cache invalidation)
        if (
            not hasattr(cache_obj, "_K_inv_cached")
            or len(cache_obj._K_inv_cached) != n_data
        ):
            cache_obj._K_inv_cached = np.linalg.inv(K_train_noisy)
        K_inv_train = cache_obj._K_inv_cached

        # Variance before (current GP)
        k_s_train = kernel_func(samples_valid, x_data, hps)
        k_s_s = signal_var

        term1 = k_s_train @ K_inv_train
        var_current = k_s_s - np.sum(term1 * k_s_train, axis=1)

        # Variance after (with candidates)
        k_s_aug = kernel_func(samples_valid, X_aug, hps)

        if use_cholesky:
            solved = cho_solve(factor, k_s_aug.T)
        else:
            solved = np.linalg.solve(K_aug, k_s_aug.T)

        reduction_term = np.sum(k_s_aug * solved.T, axis=1)
        var_after = k_s_s - reduction_term

        var_reduction = np.maximum(var_current - var_after, 0)
        group_ivr = np.mean(var_reduction) * (n_valid / n_samples_per_group)
        all_ivr.append(group_ivr)

    return np.array([np.sum(all_ivr)])


def create_batch_acquisition(n_batch, bounds_original, kernel, n_samples_per_group=500):
    """
    Factory function: creates acquisition function for n_batch points

    Parameters:
    -----------
    n_batch : int
        Number of points to acquire simultaneously
    bounds_original : array (d, 2)
        Original domain bounds
    kernel : callable
        Kernel function
    n_samples_per_group : int
        Number of MC samples per group

    Returns:
    --------
    bounds_extended : array (n_batch * d, 2)
        Extended bounds for optimizer
    batch_acquisition_wrapper : callable
        Acquisition function compatible with optimizer
    """
    d = bounds_original.shape[0]
    bounds_extended = np.vstack([bounds_original for _ in range(n_batch)])

    def batch_acquisition_wrapper(x, gpo):
        """
        x : array
            - Shape (n_batch * d,) for single candidate
            - Shape (S, n_batch * d) for vectorized evaluation
        """
        if x.ndim == 1:
            x_batch = x.reshape(n_batch, d)
            return local_ball_ivr(
                x_batch, gpo, bounds_original, kernel, n_samples_per_group
            )[0]

        elif x.ndim == 2 and x.shape[1] == n_batch * d:
            S = x.shape[0]
            results = np.zeros(S)
            for i in range(S):
                x_batch = x[i, :].reshape(n_batch, d)
                results[i] = local_ball_ivr(
                    x_batch, gpo, bounds_original, kernel, n_samples_per_group
                )[0]
            return results
        else:
            raise ValueError(f"Shape error: {x.shape}")

    return bounds_extended, batch_acquisition_wrapper
