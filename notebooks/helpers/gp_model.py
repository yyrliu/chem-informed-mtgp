import numpy as np

from chem_informed_mtgp.fingerprint_kernel import (
    FINGERPRINT_MODE_LKJ_CORR,
    LKJ_MASTER_NONE,
)

# Bounds for the LKJ correlation parameters that repurpose the signal
# variance and fingerprint weight hyperparameter slots in lkj_corr mode.
LKJ_CORR_PARAM_BOUNDS = [-3.0, 3.0]


def get_init_hyperparamter_and_bounds(
    target_column_name,
    n_features,
    n_signal_var=1,
    individual_length_scale_bounds_df=None,
    fingerprint_mode=None,
    lkj_master_category=None,
):

    if individual_length_scale_bounds_df is not None:
        raise NotImplementedError("Individual length scale bounds not yet implemented")

    else:
        lengthscale_bounds_per_param = None

    is_lkj_corr = fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR
    is_lkj_full = is_lkj_corr and lkj_master_category == LKJ_MASTER_NONE

    if target_column_name.endswith("_boxcox"):
        hps_variance_bounds = np.array([[1e-1, 1e3]])
        hps_mean_bounds = np.array([[-50.0, 50.0]])
        hps_noise_bounds = np.array([[0.1, 50.0]])
    elif target_column_name.endswith("_norm") or target_column_name == "g_log_score":
        hps_variance_bounds = np.array([[1e-4, 10.0]])
        hps_mean_bounds = np.array([[0.0, 1.0]])
        hps_noise_bounds = np.array([[1e-4, 1.0]])
    else:
        hps_variance_bounds = np.array([[1e-1, 1e3]])
        hps_mean_bounds = np.array([[0.0, 50.0]])
        hps_noise_bounds = np.array([[0.1, 50.0]])

    sigma_bounds = (
        np.tile(np.array([[1e-1, 1e3]]), (n_signal_var, 1))
        if n_signal_var > 1
        else np.empty((0, 2))
    )

    if is_lkj_corr:
        # hps[0] (normally signal variance) is repurposed as an LKJ correlation
        # parameter.
        hps_variance_bounds = np.array([LKJ_CORR_PARAM_BOUNDS])
    elif n_signal_var > 1:
        hps_variance_bounds = np.array([[1.0, 1.0]])

    length_scale_bounds = (
        lengthscale_bounds_per_param
        if lengthscale_bounds_per_param is not None
        else np.tile(np.array([[0.05, 1.5]]), (n_features, 1))
    )

    # FP weight slot: fixed at 1.0 normally, or repurposed as the second LKJ
    # correlation parameter in lkj_corr mode.
    fp_weight_bounds = (
        np.array([LKJ_CORR_PARAM_BOUNDS]) if is_lkj_corr else np.array([[1.0, 1.0]])
    )

    # Full lkj_corr (lkj_master_category="none") uses 3 independent
    # correlation parameters instead of 2, so it needs one extra hyperparameter
    # inserted right after the fp_weight slot, before the sigma block.
    lkj_third_param_bounds = (
        np.array([LKJ_CORR_PARAM_BOUNDS]) if is_lkj_full else np.empty((0, 2))
    )

    hps_bounds = np.vstack(
        [
            hps_variance_bounds,  # Kernel variance
            length_scale_bounds,  # Lengthscales
            fp_weight_bounds,  # FP weight (or LKJ correlation param in lkj_corr mode)
            lkj_third_param_bounds,  # Third LKJ correlation param (full lkj_corr only)
            sigma_bounds,  # Per-chemical signal variance scales
            hps_noise_bounds,  # Noise
            hps_mean_bounds,  # Mean
        ]
    )

    init_hyperparameters = np.mean(hps_bounds, axis=1)
    return init_hyperparameters, hps_bounds
