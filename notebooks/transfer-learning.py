import os
import sys
import time
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpcam import GPOptimizer
from chem_informed_mtgp.fingerprint_kernel import (
    CONTINUOUS_FEATURE_COUNT,
    FINGERPRINT_BITS,
    FINGERPRINT_MODE_INDEXED_TANIMOTO,
    FINGERPRINT_MODE_LKJ_CORR,
    INDEXED_FINGERPRINT_MODES,
    LKJ_MASTER_NONE,
    assemble_feature_matrix,
    build_fingerprint_metadata,
)
from chem_informed_mtgp.fingerprint_kernel_numba import (
    joint_kernel_factory_numba,
    my_noise,
    my_mean,
)
from configs.plt_rc import params_basic
from configs.exp_config import NOTEBOOK_CONFIG
from configs.paths import CV_RESULTS_DIR_BASE, GP_TRAIN_RESULT_FILE
from helpers.dataset_cleanup import (
    load_manual_entries,
    clean_dataset,
    normalize_features,
)
from helpers.gp_model import get_init_hyperparamter_and_bounds
from helpers.cross_validation import (
    get_cv_df,
    get_cv_split,
    get_picked_hashes,
)
from helpers.transfer_learning_paths import (
    get_cv_results_dir,
    get_base_save_name,
    compute_configs_info,
    get_output_file_names,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dash_lib.data_utils import query_samples  # noqa: E402

########## set up plotting parameters ##########
plt.rcParams.update(params_basic)
################################################

########## CV configuration ##########
SAVE_RESULTS_WITH_TIMESTAMP = False  # whether to include timestamp in output filenames; can set to False for easier comparison between runs when configs are unchanged
NUM_CV_FOLDS = 5
CV_STRATEGY = "stratified"  # options: "kfold", "stratified", "roundrobin"
ASITE_AWARE_SPLITTING = True  # set to False to disable a-site-aware splitting
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))
FEW_SHOT_TARGET_ASITE = os.environ.get(
    "FEW_SHOT_TARGET_ASITE", "ClMBAI"
)  # a_site value to enrich in the training set for few-shot ablation
FEW_SHOT_ONLY = (
    os.environ.get("FEW_SHOT_ONLY", "False").lower() == "true"
)  # if True, remove all non-target a_site samples from training set instead of just a fraction
FEW_SHOT_RATIO = float(
    os.environ.get("FEW_SHOT_RATIO", "1.0")
)  # fraction of MBAI hashes to keep in training set
FEW_SHOT_REPEAT_INDEX = os.environ.get("FEW_SHOT_REPEAT_INDEX")
FEW_SHOT_REPEAT_INDEX = (
    int(FEW_SHOT_REPEAT_INDEX) if FEW_SHOT_REPEAT_INDEX is not None else None
)  # index of this repeat draw for FEW_SHOT_RATIO sweeps; same RANDOM_SEED (CV split) but a different few-shot hash pick per index. None reproduces the original single-draw behavior.
SKIP_IF_EXISTS = (
    os.environ.get("SKIP_IF_EXISTS", "false").lower() == "true"
)  # if True, a pre-existing output file exits(0) early instead of raising — lets a repeat sweep resume without re-erroring on already-completed repeats.
FP_KERNEL_CONF = os.environ.get(
    "FP_KERNEL_CONF", "fp2"
)  # options: "total-correl", "no-correl", "fp2"
VAR_PER_ASITE = (
    os.environ.get("VAR_PER_ASITE", "True").lower() == "true"
)  # whether to use separate signal variance for each chemical category in the fingerprint kernel; only applicable for indexed tanimoto kernel
FINGERPRINT_MODE = os.environ.get(
    "FINGERPRINT_MODE", NOTEBOOK_CONFIG["fingerprint_mode"]
)
LKJ_MASTER_CATEGORY = os.environ.get(
    "LKJ_MASTER_CATEGORY", None
)  # a_site name that gets equal LKJ correlation to both other a_sites; set to "none" for the full 3-parameter LKJ correlation (no master); only applicable for lkj_corr mode
######################################
CV_RESULTS_DIR = get_cv_results_dir(CV_RESULTS_DIR_BASE, RANDOM_SEED)

########## setup macro ##########
param_config = NOTEBOOK_CONFIG["param_config"]
outliers_by_sample_name = NOTEBOOK_CONFIG["outliers_by_sample_name"]
INITIAL_TRAIN_BATCHES = NOTEBOOK_CONFIG["train_batches"]
CV_BATCHES = NOTEBOOK_CONFIG["cv_batches"]
DEFAULT_TARGET_COLUMN = NOTEBOOK_CONFIG["default_target_column"]
A_SITE_TO_SMILES = NOTEBOOK_CONFIG["a_site_to_smiles"]
POSTERIOR_MEAN_THRESHOLDS_BY_A_SITE = NOTEBOOK_CONFIG[
    "posterior_mean_thresholds_by_a_site"
]
CONDITION_OUTLIER_RSD_THRESHOLD = NOTEBOOK_CONFIG["condition_outlier_rsd_threshold"]
KERNEL_TYPE = NOTEBOOK_CONFIG["kernel_type"]

RANDOM_POOL_SIZE = 2_500_000
# RANDOM_POOL_SIZE = 1_000_000
# RANDOM_POOL_SIZE = 10_000

N_CPU = 16
N_NEW_POINTS = 8
N_IVR_EVAL_BATCH_SIZE = 5000 // N_NEW_POINTS * N_CPU
FINGERPRINT_BITS = int(os.environ.get("FINGERPRINT_BITS", FINGERPRINT_BITS))
CONTINUOUS_FEATURE_COUNT = min(
    len(param_config),
    int(os.environ.get("CONTINUOUS_FEATURE_COUNT", CONTINUOUS_FEATURE_COUNT)),
)
CONTINUOUS_COLUMNS = list(param_config.keys())[:CONTINUOUS_FEATURE_COUNT]
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", DEFAULT_TARGET_COLUMN)
################################################

print(f"Using random seed: {RANDOM_SEED}")

########## Build fingerprint metadata and kernel function ##########

if FINGERPRINT_MODE == FINGERPRINT_MODE_LKJ_CORR:
    print(
        f"FINGERPRINT_MODE='{FINGERPRINT_MODE_LKJ_CORR}': FP_KERNEL_CONF is ignored "
        "(correlations are learned via the LKJ parameterization instead of Tanimoto)."
    )
else:
    assert FP_KERNEL_CONF in [
        "fp2",
        "no-correl",
        "total-correl",
    ], f"Unsupported FINGERPRINT_MODE: {FP_KERNEL_CONF}"

fingerprint_metadata = build_fingerprint_metadata(
    A_SITE_TO_SMILES,
    n_bits=FINGERPRINT_BITS,
    radius=2,
    use_features=False,
)

if FP_KERNEL_CONF == "fp2":
    # use the default tanimoto correlations from the dataset
    pass
elif FP_KERNEL_CONF == "no-correl":
    # override tanimoto correlations with identity matrix (no correlation between different a_sites)
    from dataclasses import replace

    fingerprint_metadata = replace(
        fingerprint_metadata,
        tanimoto_correlations=np.eye(len(fingerprint_metadata.a_sites)),
    )
elif FP_KERNEL_CONF == "total-correl":
    # override tanimoto correlations with total correlation (all ones matrix)
    from dataclasses import replace

    fingerprint_metadata = replace(
        fingerprint_metadata,
        tanimoto_correlations=np.ones(
            (len(fingerprint_metadata.a_sites), len(fingerprint_metadata.a_sites))
        ),
    )

print("Tanimoto correlation matrix:")
print(fingerprint_metadata.tanimoto_correlations)

if FINGERPRINT_MODE in INDEXED_FINGERPRINT_MODES:
    FP_FEATURE_DIMS = 1
else:
    FP_FEATURE_DIMS = FINGERPRINT_BITS

sigma_per_chemical = VAR_PER_ASITE  # whether to use separate lengthscale/sigma for each chemical category in the fingerprint kernel; only applicable for indexed tanimoto / lkj_corr kernels

if FINGERPRINT_MODE == FINGERPRINT_MODE_LKJ_CORR and not sigma_per_chemical:
    raise ValueError(
        f"FINGERPRINT_MODE='{FINGERPRINT_MODE_LKJ_CORR}' requires VAR_PER_ASITE=True "
        "(the signal variance and fingerprint weight slots are repurposed as LKJ parameters)."
    )

kernel_fn_numba = joint_kernel_factory_numba(
    cont_dims=CONTINUOUS_FEATURE_COUNT,
    fingerprint_bits=FP_FEATURE_DIMS,
    kernel=KERNEL_TYPE,
    fingerprint_mode=FINGERPRINT_MODE,
    tanimoto_correlations=(
        fingerprint_metadata.tanimoto_correlations
        if FINGERPRINT_MODE == FINGERPRINT_MODE_INDEXED_TANIMOTO
        else None
    ),
    sigma_per_chemical=sigma_per_chemical,
    sigma_n_categories=(
        len(fingerprint_metadata.a_sites) if sigma_per_chemical else None
    ),
    fingerprint_metadata=fingerprint_metadata,
    lkj_master_category=LKJ_MASTER_CATEGORY,
)

print(
    f"Fingerprint mode={FINGERPRINT_MODE}, FP_FEATURE_DIMS={FP_FEATURE_DIMS}, A-site categories={len(fingerprint_metadata.a_sites)}"
    + (
        (
            ", LKJ correlation: full (3 independent parameters)"
            if LKJ_MASTER_CATEGORY == LKJ_MASTER_NONE
            else f", LKJ master category={LKJ_MASTER_CATEGORY or fingerprint_metadata.a_sites[0]}"
        )
        if FINGERPRINT_MODE == FINGERPRINT_MODE_LKJ_CORR
        else ""
    )
)

################################################

########## Load dataset and prepare manual entries ##########
manual_entries_df = load_manual_entries(
    NOTEBOOK_CONFIG["manual_entries"], continuous_columns=CONTINUOUS_COLUMNS
)

db_path = Path(__file__).resolve().parent.parent / "mydb.duckdb"

df = query_samples(db_path=db_path)
df = df[df["sample_name"].str.split("-").str[0].isin(INITIAL_TRAIN_BATCHES)]

if not manual_entries_df.empty:
    df = pd.concat([df, manual_entries_df], sort=False)

df = clean_dataset(
    df,
    outliers_by_sample_name,
    continuous_columns=CONTINUOUS_COLUMNS,
    target_column=TARGET_COLUMN,
)

df_normalized = normalize_features(df, param_config)

# %%

# %%


# Assemble the (continuous + fingerprint)-dimensional design matrix and targets
feature_df = df_normalized.copy().reset_index(drop=True)
X_joint = assemble_feature_matrix(
    df=feature_df,
    continuous_columns=CONTINUOUS_COLUMNS,
    fingerprint_column="a_site_smiles",
    n_bits=FINGERPRINT_BITS,
    fingerprint_mode=FINGERPRINT_MODE,
    fingerprint_metadata=fingerprint_metadata,
    a_site_column="a_site",
)

y = feature_df[TARGET_COLUMN].to_numpy(dtype=np.float64)

######## Setup model & verify with saved hyperparameters ##########
# %%
init_hps, hps_bounds = get_init_hyperparamter_and_bounds(
    target_column_name=TARGET_COLUMN,
    n_features=CONTINUOUS_FEATURE_COUNT,
    n_signal_var=len(fingerprint_metadata.a_sites) if sigma_per_chemical else 1,
    fingerprint_mode=FINGERPRINT_MODE,
    lkj_master_category=LKJ_MASTER_CATEGORY,
)

if FINGERPRINT_MODE == FINGERPRINT_MODE_LKJ_CORR:
    print(
        f"FINGERPRINT_MODE='{FINGERPRINT_MODE_LKJ_CORR}': skipping verification against "
        "the saved indexed_tanimoto training result (no matching baseline artifact)."
    )
else:
    train_result_npz = np.load(GP_TRAIN_RESULT_FILE)

    print("Loaded training results:")
    print("Train batches:", train_result_npz["train_batches"])
    print("Log Likelihood:", train_result_npz["log_likelihood"])
    print(
        "Hyperparameters:",
        np.array2string(
            train_result_npz["hyperparameters"], suppress_small=True, precision=4
        ),
    )

    print("Re-loaded training results to `kernel_fn`")
    # Initialize and train the GP prior to Bayesian Optimization
    fingerprint_gp = GPOptimizer(
        X_joint,
        y,
        init_hyperparameters=train_result_npz["hyperparameters"],
        kernel_function=kernel_fn_numba,
        prior_mean_function=my_mean,
        noise_function=my_noise,
    )

    gp_hps, gp_log_likelihood = (
        fingerprint_gp.get_hyperparameters(),
        fingerprint_gp.log_likelihood(),
    )

    if np.isclose(gp_hps, train_result_npz["hyperparameters"]).all():
        print("Hyperparameters successfully loaded to `fingerprint_gp`.")
    else:
        print("Warning: Loaded hyperparameters do not match the initialized values.")

    if np.isclose(gp_log_likelihood, train_result_npz["log_likelihood"]):
        print("Log likelihood of `fingerprint_gp` matches the loaded training result.")
    else:
        print(
            f"Warning: Log likelihood of `fingerprint_gp` does not match the loaded training result.\nFingerprint GP log likelihood: {gp_log_likelihood}, loaded log likelihood: {train_result_npz['log_likelihood']}"
        )
    train_result_npz.close()

######## Setup CV dataset ##########

df_cross_validation, unique_hashes = get_cv_df(
    df_normalized,
    validate_batches=CV_BATCHES,
    outliers_by_sample_name=outliers_by_sample_name,
    sampling_rate=1.0,
    single_asite=FEW_SHOT_TARGET_ASITE if FEW_SHOT_ONLY else None,
)

CV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df_cross_validation = df_cross_validation[
    [col for col in df_cross_validation.columns if not col.endswith("path")]
]
df_cross_validation["sample_number"] = df_cross_validation["sample_number"].astype(int)
cv_df_save_path = CV_RESULTS_DIR / "df_cross_validation.csv"

# FEW_SHOT_ONLY is a subset, skip saving
if FEW_SHOT_ONLY:
    if not cv_df_save_path.exists():
        raise FileNotFoundError(
            f"FEW_SHOT_ONLY is enabled, but df_cross_validation CSV not found at {cv_df_save_path}. Please run with FEW_SHOT_ONLY disabled to generate the full CV dataframe first."
        )
else:
    if cv_df_save_path.exists():
        existing_cv_df = pd.read_csv(cv_df_save_path, index_col=0)
        pd.testing.assert_frame_equal(
            df_cross_validation, existing_cv_df, check_dtype=False
        )
        print(
            f"df_cross_validation already exists at {cv_df_save_path} and matches the current dataframe. No need to overwrite."
        )
    else:
        # File-safe write with retry logic for parallel execution
        # Multiple threads may try to write this file simultaneously
        lock_file = cv_df_save_path.with_suffix(".lock")
        max_retries = 10
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                # Try to create lock file atomically
                lock_file.touch(exist_ok=False)
                try:
                    df_cross_validation.to_csv(cv_df_save_path, index=True)
                    print(f"Saved df_cross_validation to {cv_df_save_path}")
                finally:
                    # Always remove lock file
                    if lock_file.exists():
                        lock_file.unlink()
                break
            except FileExistsError:
                # Another process has the lock, wait and retry
                if attempt < max_retries - 1:
                    print(
                        f"  (Waiting for lock on df_cross_validation.csv, attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    print(
                        f"  ⚠ Could not acquire lock after {max_retries} retries. Skipping file write."
                    )
            except Exception as e:
                print(f"  ⚠ Error writing df_cross_validation: {e}")
                break

# %%

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

base_save_name = get_base_save_name(
    TARGET_COLUMN, KERNEL_TYPE, CV_STRATEGY, ASITE_AWARE_SPLITTING
)

# stable_configs_info_str excludes the fewshot/repeat suffix -- it names the
# FEW_SHOT_RATIO=1.0 baseline of this exact config, used to locate a
# warm-start hps file.
stable_configs_info_str, configs_info_str = compute_configs_info(
    var_per_asite=VAR_PER_ASITE,
    fingerprint_mode=FINGERPRINT_MODE,
    fp_kernel_conf=FP_KERNEL_CONF,
    lkj_master_category=LKJ_MASTER_CATEGORY,
    few_shot_only=FEW_SHOT_ONLY,
    few_shot_target_asite=FEW_SHOT_TARGET_ASITE,
    few_shot_ratio=FEW_SHOT_RATIO,
    few_shot_repeat_index=FEW_SHOT_REPEAT_INDEX,
)
print(f"configs_info_str: {configs_info_str}")

if len(unique_hashes) < 2:
    raise ValueError("Need at least two unique hashes for cross‑validation.")

k = min(NUM_CV_FOLDS, len(unique_hashes))

cv_fold_splits, unique_hashes = get_cv_split(
    df_cross_validation,
    strategy=CV_STRATEGY,
    a_site_aware=ASITE_AWARE_SPLITTING,
    k=k,
    random_state=RANDOM_SEED,
    target_column=TARGET_COLUMN,
)


# ------------------------------------------------------------------
#  Warm start: reuse the FEW_SHOT_RATIO=1.0 baseline's per-fold
#  hyperparameters (same config, just no few-shot subsampling) as the
#  init point instead of the bounds midpoint, to speed up convergence.
# ------------------------------------------------------------------
baseline_hps_by_fold = None
if FEW_SHOT_RATIO < 1.0:
    baseline_hps_file_name, _ = get_output_file_names(
        base_save_name, stable_configs_info_str
    )
    baseline_hps_path = CV_RESULTS_DIR / baseline_hps_file_name
    if baseline_hps_path.exists():
        baseline_hps_df = pd.read_csv(baseline_hps_path)
        baseline_hps_cols = sorted(
            (c for c in baseline_hps_df.columns if c.startswith("hps_")),
            key=lambda c: int(c.split("_", 1)[1]),
        )
        if len(baseline_hps_cols) == len(init_hps):
            baseline_hps_by_fold = {
                int(row["fold"]): row[baseline_hps_cols].to_numpy(dtype=np.float64)
                for _, row in baseline_hps_df.iterrows()
            }
            print(
                f"Warm-starting hyperparameters from baseline: {baseline_hps_path.name}"
            )
        else:
            print(
                f"Baseline hps at {baseline_hps_path.name} has "
                f"{len(baseline_hps_cols)} params, expected {len(init_hps)}; "
                "skipping warm start."
            )
    else:
        print(
            f"No baseline hps found at {baseline_hps_path.name} for warm start; "
            "using default init."
        )


# ------------------------------------------------------------------
#  run_fold — unchanged regardless of strategy
# ------------------------------------------------------------------
def run_fold(fold_no, train_idx, test_idx):
    # ----- training data -------------------------------------------------
    train_hashes = unique_hashes[train_idx]

    # remove some hashes from the training set to simulate few shot learning for MBAI
    if FEW_SHOT_RATIO < 1.0:
        ratio_to_remove = 1.0 - FEW_SHOT_RATIO
        mbai_hashes = get_picked_hashes(
            unique_hashes=train_hashes,
            df_cross_validation=df_cross_validation,
            a_site_filter=FEW_SHOT_TARGET_ASITE,
            sample_fraction=ratio_to_remove,
            # different seed per fold for variability; when repeating a
            # FEW_SHOT_RATIO sweep point, folding in FEW_SHOT_REPEAT_INDEX
            # draws a different few-shot subset while RANDOM_SEED (and thus
            # the CV split itself) stays fixed across repeats.
            seed=(fold_no, FEW_SHOT_REPEAT_INDEX)
            if FEW_SHOT_REPEAT_INDEX is not None
            else fold_no,
        )
        train_hashes = np.setdiff1d(train_hashes, mbai_hashes)
        print(
            f"Fold {fold_no}: Removing {len(mbai_hashes)} {FEW_SHOT_TARGET_ASITE} hashes from training set to simulate few-shot learning (ratio={ratio_to_remove:.0%})"
        )

    train_mask = df_cross_validation["config_hash"].isin(train_hashes)
    train_rows = df_cross_validation.loc[train_mask, "feature_idx"].to_numpy()
    x_train = X_joint[train_rows]
    y_train = y[train_rows]

    print(len(x_train))

    # ----- test data -----------------------------------------------------
    test_hashes = unique_hashes[test_idx]
    test_rows = df_cross_validation[
        df_cross_validation["config_hash"].isin(test_hashes)
    ]
    x_test = X_joint[test_rows["feature_idx"].to_numpy()]
    y_test = test_rows[TARGET_COLUMN].to_numpy()
    test_ids = test_rows[TARGET_COLUMN].index.to_list()

    # ----- GP fitting ----------------------------------------------------
    fold_init_hps = init_hps
    if baseline_hps_by_fold is not None and fold_no in baseline_hps_by_fold:
        fold_init_hps = np.clip(
            baseline_hps_by_fold[fold_no], hps_bounds[:, 0], hps_bounds[:, 1]
        )

    gp = GPOptimizer(
        x_train,
        y_train,
        init_hyperparameters=fold_init_hps,
        kernel_function=kernel_fn_numba,
        prior_mean_function=my_mean,
        noise_function=my_noise,
    )
    gp.train(
        hyperparameter_bounds=hps_bounds,
        init_hyperparameters=fold_init_hps,
        method="global",
        max_iter=2000,
    )

    # ----- prediction & metric -------------------------------------------
    y_pred = gp.posterior_mean(x_test)["m(x)"]
    y_var = gp.posterior_covariance(x_test, variance_only=True)["v(x)"]
    y_var_with_noise = gp.posterior_covariance(
        x_test, variance_only=True, add_noise=True
    )["v(x)"]
    rmse = gp.rmse(x_test, y_test)
    hps = gp.get_hyperparameters()
    return (
        fold_no,
        rmse,
        y_test.tolist(),
        y_pred.tolist(),
        test_ids,
        hps.tolist(),
        y_var.tolist(),
        y_var_with_noise.tolist(),
    )


# %%

hps_file_name, cv_results_file_name = get_output_file_names(
    base_save_name, configs_info_str
)

if SAVE_RESULTS_WITH_TIMESTAMP:
    hps_file_name = f"{timestamp}_{hps_file_name}"
    cv_results_file_name = f"{timestamp}_{cv_results_file_name}"

if not SAVE_RESULTS_WITH_TIMESTAMP:
    # check if files already exist, if so, raise an error to avoid overwriting previous results
    existing_hps_path = CV_RESULTS_DIR / hps_file_name
    existing_cv_results_path = CV_RESULTS_DIR / cv_results_file_name

    if existing_hps_path.exists() or existing_cv_results_path.exists():
        if SKIP_IF_EXISTS:
            print(
                f"Output already exists ({existing_hps_path.name}); "
                "SKIP_IF_EXISTS=true, skipping this run."
            )
            sys.exit(0)
        existing_path = (
            existing_hps_path
            if existing_hps_path.exists()
            else existing_cv_results_path
        )
        raise FileExistsError(
            f"File {existing_path} already exists. Please set SAVE_RESULTS_WITH_TIMESTAMP=True to avoid overwriting previous results."
        )

# ----- parallel execution -------------------------------------------------
results = Parallel(backend="loky", n_jobs=-2)(
    delayed(run_fold)(i, tr, te) for i, (tr, te) in enumerate(cv_fold_splits, 1)
)

cross_val_df = pd.DataFrame(
    results,
    columns=[
        "fold",
        "rmse",
        "y_true",
        "y_pred",
        "test_ids",
        "hyperparameters",
        "y_var",
        "y_var_with_noise",
    ],
).sort_values("fold")

# Offset saved fold numbers by repeat index so that repeated FEW_SHOT_RATIO
# draws occupy distinct fold-number blocks instead of colliding on 1..k —
# downstream per-fold pairing (transfer-learning-plot.py) then treats each
# repeat as an extra independent sample rather than silently averaging over
# unrelated few-shot draws that happen to share a fold number.
if FEW_SHOT_REPEAT_INDEX is not None:
    cross_val_df["fold"] = cross_val_df["fold"] + FEW_SHOT_REPEAT_INDEX * k

# %%

print(cross_val_df[["fold", "rmse"]].to_markdown(tablefmt="github"))

hps_df = pd.DataFrame(cross_val_df["hyperparameters"].tolist()).add_prefix("hps_")
hps_df["fold"] = cross_val_df["fold"]
print("\nLearned hyperparameters by fold:")
print(hps_df.set_index("fold").to_markdown(tablefmt="github", floatfmt=".4f"))


# CV_STRATEGY is included in both output filenames
hps_save_path = Path(CV_RESULTS_DIR) / hps_file_name
hps_df.to_csv(hps_save_path, index=False)
print(f"Saved CV hyperparameters to {hps_save_path.parent.name}/{hps_save_path.name}")

all_true = np.concatenate(cross_val_df["y_true"].to_numpy())
all_pred = np.concatenate(cross_val_df["y_pred"].to_numpy())
test_ids = np.concatenate(cross_val_df["test_ids"].to_numpy())
all_pred_var = np.concatenate(cross_val_df["y_var"].to_numpy())
all_pred_var_with_noise = np.concatenate(cross_val_df["y_var_with_noise"].to_numpy())
fold_numbers = np.concatenate(
    [
        [fold] * len(true)
        for fold, true in zip(cross_val_df["fold"], cross_val_df["y_true"])
    ]
)

rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
print(f"\nOverall RMSE: {rmse:.3f}")

cv_results_path = Path(CV_RESULTS_DIR) / cv_results_file_name
cv_results_df = pd.DataFrame(
    {
        "y_true": all_true,
        "y_pred": all_pred,
        "test_ids": test_ids,
        "pred_var": all_pred_var,
        "pred_var_with_noise": all_pred_var_with_noise,
        "fold": fold_numbers,
    }
)

cv_results_df.to_csv(cv_results_path, index=False)
print(f"Saved CV results to {cv_results_path.parent.name}/{cv_results_path.name}")
