# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from configs.exp_config import NOTEBOOK_CONFIG
from configs.nature_mpl_rc import DOUBLE_COLUMN_WIDTH, nature_rc
from configs.paths import CV_RESULTS_DIR_BASE, GP_TRAIN_RESULT_FILE
from gpcam import GPOptimizer
from helpers.dataset_cleanup import (
    clean_dataset,
    load_manual_entries,
    normalize_features,
)
from helpers.gp_model import get_init_hyperparamter_and_bounds
from matplotlib.colors import Normalize

from chem_informed_mtgp.fingerprint_kernel import (
    CONTINUOUS_FEATURE_COUNT,
    FINGERPRINT_BITS,
    FINGERPRINT_MODE_INDEXED_TANIMOTO,
    assemble_feature_matrix,
    build_fingerprint_metadata,
)
from chem_informed_mtgp.fingerprint_kernel_numba import (
    joint_kernel_factory_numba,
    my_mean,
    my_noise,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dash_lib.data_utils import query_samples  # noqa: E402

########## set up plotting parameters ##########
plt.rcParams.update(nature_rc)
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
FP_KERNEL_CONF = os.environ.get(
    "FP_KERNEL_CONF", "fp2"
)  # options: "total-correl", "no-correl", "fp2"
VAR_PER_ASITE = (
    os.environ.get("VAR_PER_ASITE", "True").lower() == "true"
)  # whether to use separate signal variance for each chemical category in the fingerprint kernel; only applicable for indexed tanimoto kernel
REMOVE_SIGNAL_VARIANCE_FROM_COVARIANCE = (
    os.environ.get("REMOVE_SIGNAL_VARIANCE_FROM_COVARIANCE", "False").lower() == "true"
)  # whether to divide K by signal variance for the covariance plot
COVARIANCE_PLOT_SAMPLES_PER_ASITE = os.environ.get(
    "COVARIANCE_PLOT_SAMPLES_PER_ASITE", None
)  # if set, randomly sample N points per a-site for visualization (e.g. "10")
if COVARIANCE_PLOT_SAMPLES_PER_ASITE is not None:
    COVARIANCE_PLOT_SAMPLES_PER_ASITE = int(COVARIANCE_PLOT_SAMPLES_PER_ASITE)
configs_info = [
    "var-per-asite" if VAR_PER_ASITE else "",
    FP_KERNEL_CONF,
    f"{FEW_SHOT_TARGET_ASITE}-only" if FEW_SHOT_ONLY else "",
]
######################################
configs_info = [info for info in configs_info if info]  # filter out empty strings
CV_RESULTS_DIR = CV_RESULTS_DIR_BASE.with_name(
    f"{CV_RESULTS_DIR_BASE.name}_rseed={RANDOM_SEED}"
)

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
FINGERPRINT_MODE = NOTEBOOK_CONFIG["fingerprint_mode"]

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

if FINGERPRINT_MODE == FINGERPRINT_MODE_INDEXED_TANIMOTO:
    FP_FEATURE_DIMS = 1
else:
    FP_FEATURE_DIMS = FINGERPRINT_BITS

sigma_per_chemical = VAR_PER_ASITE  # whether to use separate lengthscale/sigma for each chemical category in the fingerprint kernel; only applicable for indexed tanimoto kernel

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
)

print(
    f"Fingerprint mode={FINGERPRINT_MODE}, FP_FEATURE_DIMS={FP_FEATURE_DIMS}, A-site categories={len(fingerprint_metadata.a_sites)}"
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
)

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

######## Plot covariance in gram martix, group by a-site ##########


# %%
# ==== Plot full covariance gram matrix with block structure by a-site ====

REMOVE_SIGNAL_VARIANCE_FROM_COVARIANCE = (
    True  # whether to divide K by signal variance for the covariance plot
)
COVARIANCE_PLOT_SAMPLES_PER_ASITE = None

# Get the prior covariance matrix from the GP
prior_pdf = fingerprint_gp.get_prior_pdf()
K = prior_pdf["prior covariance (K)"]

# Optionally remove signal variance (amplitude scaling)
if REMOVE_SIGNAL_VARIANCE_FROM_COVARIANCE:
    hps = fingerprint_gp.get_hyperparameters()
    if sigma_per_chemical:
        # Per-a-site signal variances start at index: 1 + n_features + 1
        sigma_start_idx = 1 + CONTINUOUS_FEATURE_COUNT + 1
        sigma_end_idx = sigma_start_idx + len(fingerprint_metadata.a_sites)
        signal_variances = hps[sigma_start_idx:sigma_end_idx]
        print(
            f"Per-a-site signal variances (indices {sigma_start_idx}:{sigma_end_idx}): {signal_variances}"
        )
    else:
        # Single signal variance at index 0
        signal_variances = [hps[0]] * len(fingerprint_metadata.a_sites)
        print(f"Single signal variance (index 0): {signal_variances[0]:.6f}")

    # Will divide each a-site block by its variance after reordering
    print("Signal variance will be removed per a-site block\n")
else:
    signal_variances = None
    print("Signal variance included in covariance matrix\n")

# Deduplicate by config_hash (keep first occurrence of each unique config)
unique_indices = feature_df.drop_duplicates(
    subset="config_hash", keep="first"
).index.tolist()

# Create mapping of a-site to indices (ordered by a-site name), filtered to unique configs
a_sites_list = fingerprint_metadata.a_sites
asite_to_indices = {}
reorder_idx = []
block_boundaries = []
cumulative = 0

for a_site in a_sites_list:
    indices = feature_df[
        (feature_df["a_site"] == a_site) & (feature_df.index.isin(unique_indices))
    ].index.tolist()
    original_count = len(indices)

    # Optionally subsample for visualization
    if COVARIANCE_PLOT_SAMPLES_PER_ASITE is not None:
        n_samples = min(COVARIANCE_PLOT_SAMPLES_PER_ASITE, len(indices))
        indices = np.random.choice(indices, size=n_samples, replace=False).tolist()
        print(f"Subsampled {a_site}: {len(indices)} / {original_count}")

    asite_to_indices[a_site] = indices
    reorder_idx.extend(indices)
    cumulative += len(indices)
    block_boundaries.append(cumulative)

# Remove last boundary (it's the end of the matrix)
block_boundaries = block_boundaries[:-1]

# Reorder K matrix to group by a-site
K_reordered = K[np.ix_(reorder_idx, reorder_idx)]

# Optionally divide each a-site block by its signal variance
if signal_variances is not None:
    K_reordered_adjusted = K_reordered.copy()
    for i, a_site_i in enumerate(a_sites_list):
        idx_i_start = sum(len(asite_to_indices[a_sites_list[j]]) for j in range(i))
        idx_i_end = idx_i_start + len(asite_to_indices[a_site_i])
        for j, a_site_j in enumerate(a_sites_list):
            idx_j_start = sum(len(asite_to_indices[a_sites_list[k]]) for k in range(j))
            idx_j_end = idx_j_start + len(asite_to_indices[a_site_j])
            # Divide by geometric mean of signal variances from both blocks
            var_scaling = signal_variances[i] * signal_variances[j]
            K_reordered_adjusted[idx_i_start:idx_i_end, idx_j_start:idx_j_end] /= (
                var_scaling
            )
    K_reordered = K_reordered_adjusted

# Normalize to [0, 1] for visualization
K_norm = (K_reordered - K_reordered.min()) / (K_reordered.max() - K_reordered.min())

# Create figure with block visualization
fig, ax = plt.subplots(
    figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.9),
    dpi=300,
    constrained_layout=True,
)
# fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)


norm = Normalize(vmax=np.percentile(K_norm, 92))
# norm = LogNorm(vmin=np.percentile(K_norm, 1), vmax=np.percentile(K_norm, 99))
im = ax.imshow(K_norm, cmap="viridis", aspect="equal", norm=norm)

# Add white block dividers
for boundary in block_boundaries:
    ax.axhline(boundary - 0.5, color="white", linewidth=1.5)
    ax.axvline(boundary - 0.5, color="white", linewidth=1.5)

# Set axis ticks at block centers
n_per = [len(asite_to_indices[a_site]) for a_site in a_sites_list]
tick_positions = []
cumulative = 0
for n in n_per:
    tick_positions.append(cumulative + n / 2 - 0.5)
    cumulative += n

ax.set_xticks(tick_positions)
ax.set_xticklabels(a_sites_list, fontsize="large")
ax.set_yticks(tick_positions)
ax.set_yticklabels(a_sites_list, rotation=90, va="center", fontsize="large")
ax.tick_params(bottom=False, left=False, top=False, right=False)

# title_suffix = " (signal variance removed)" if REMOVE_SIGNAL_VARIANCE_FROM_COVARIANCE else ""
# ax.set_title(f"Tanimoto Gram matrix\n(block structure by a-site){title_suffix}", fontsize=13, fontweight="bold")
# ax.set_xlabel("A-site")
# ax.set_ylabel("A-site")

# Add colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
# tick inwards to avoid overlapping with the figure edge
cbar.ax.tick_params(direction="in")
cbar.set_label(r"covariance (w/o signal variance $\sigma_f^2$)", fontsize="large")

filename = "covariance_gram_matrix"
if REMOVE_SIGNAL_VARIANCE_FROM_COVARIANCE:
    filename += "_no_signal_var"
if COVARIANCE_PLOT_SAMPLES_PER_ASITE is not None:
    filename += f"_subsample_{COVARIANCE_PLOT_SAMPLES_PER_ASITE}"
output_path = Path("../manuscript_assets/si", f"3_{filename}.svg")
# plt.savefig(output_path, transparent=True, dpi=600)
plt.savefig(output_path.with_suffix(".png"), dpi=300)
print(f"\nSaved block-structured covariance gram matrix to:\n  {output_path}")
print(f"Block sizes: {dict(zip(a_sites_list, n_per))}")
print(f"Matrix shape: {K_norm.shape}\n")
