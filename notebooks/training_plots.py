# %%
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpcam import GPOptimizer
from chem_informed_mtgp.fingerprint_kernel import (
    CONTINUOUS_FEATURE_COUNT,
    FINGERPRINT_BITS,
    FINGERPRINT_MODE_INDEXED_TANIMOTO,
    assemble_feature_matrix,
    build_fingerprint_metadata,
)
from chem_informed_mtgp.fingerprint_kernel_numba import (
    joint_kernel_factory_numba,
    my_noise,
    my_mean,
)
from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH
from configs.exp_config import NOTEBOOK_CONFIG
from configs.paths import BO_RESULTS_DIR
from helpers.dataset_cleanup import (
    load_manual_entries,
    clean_dataset,
    normalize_features,
)
from helpers.gp_model import get_init_hyperparamter_and_bounds

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dash_lib.data_utils import query_samples  # noqa: E402

########## set up plotting parameters ##########
plt.rcParams.update(nature_rc)
################################################
VAR_PER_ASITE = (
    os.environ.get("VAR_PER_ASITE", "True").lower() == "true"
)  # whether to use separate signal variance for each chemical category in the fingerprint kernel; only applicable for indexed tanimoto kernel
configs_info = [
    "var-per-asite" if VAR_PER_ASITE else "shared-var",
]
######################################

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

FINGERPRINT_BITS = int(os.environ.get("FINGERPRINT_BITS", FINGERPRINT_BITS))
CONTINUOUS_FEATURE_COUNT = min(
    len(param_config),
    int(os.environ.get("CONTINUOUS_FEATURE_COUNT", CONTINUOUS_FEATURE_COUNT)),
)
CONTINUOUS_COLUMNS = list(param_config.keys())[:CONTINUOUS_FEATURE_COUNT]
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", DEFAULT_TARGET_COLUMN)
################################################

########## Build fingerprint metadata and kernel function ##########

fingerprint_metadata = build_fingerprint_metadata(
    A_SITE_TO_SMILES,
    n_bits=FINGERPRINT_BITS,
    radius=2,
    use_features=False,
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

############## Plot cumulative counts of unique config_hashes by a-site and batch ##############

df["batch"] = df["sample_name"].map(lambda x: x.split("-")[0]).astype(int)
SMILES_TO_ASITE = {v: k for k, v in A_SITE_TO_SMILES.items()}
df["a-site"] = df["a_site_smiles"].map(SMILES_TO_ASITE)
counts = (
    df.drop_duplicates(subset=["config_hash"])
    .groupby(["a-site", "batch"])
    .size()
    .reset_index(name="count")
)
counts = counts.sort_values(["a-site", "batch"])
counts["cum_count"] = counts.groupby("a-site")["count"].cumsum()

fig, ax = plt.subplots(
    figsize=(DOUBLE_COLUMN_WIDTH * 0.5, DOUBLE_COLUMN_WIDTH * 0.5 * 0.67),
    dpi=300,
    constrained_layout=True,
)

for a_site, group in counts.groupby("a-site"):
    ax.plot(group["batch"], group["cum_count"], marker="o", label=a_site)

ax.legend()
ax.set_xlabel("Batch")
ax.set_ylabel("Cumulative Count")

fig.savefig("../manuscript_assets/si/2_cumulative_counts_by_asite.png", dpi=300)

#####################################

################### Plot target distribution by batch ###################
# %%

plot_target = TARGET_COLUMN
df_barplot = df.copy()
df_barplot["session_number"] = df_barplot["session"].str.split("-").str[1].astype(int)
df_barplot.groupby("session_number")


colors_count = df_barplot["session_number"].nunique()

cmap = plt.get_cmap("viridis_r")
colors = [cmap(i / colors_count) for i in range(colors_count)]

target_labels = {
    "score": "Score (a.u.)",
    "g_factor_max": r"g-factor ($\hat{g}$)",
    "homogeneity": r"Homogeneity ($H_{PL}$, a.u.)",
    "ldlb_area_ratio_asym": r"LDLB area ratio ($R_{LDLB}$)",
}

for target in target_labels:
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH * 0.5, DOUBLE_COLUMN_WIDTH * 0.5 * 0.4),
        dpi=300,
        constrained_layout=True,
    )

    for (session_number, group), color in zip(
        df_barplot.groupby("session_number"), colors
    ):
        jitter_xs = np.random.normal(loc=0.0, scale=0.07, size=len(group))
        ax.scatter(
            group["session_number"]
            + jitter_xs,  # jitter x position by session number and index
            group[target],
            alpha=0.3,
            color=color,
            s=10,
        )

    ax.set_xlabel("Batch")
    ax.set_xticks(sorted(df_barplot["session_number"].unique()))
    ax.set_ylabel(target_labels[target])

    fig.savefig(
        f"../manuscript_assets/si/2_{target}_distribution_by_batch.png", dpi=300
    )
    plt.show()

##########################################


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

# Initialize and train the GP prior to Bayesian Optimization
fingerprint_gp = GPOptimizer(
    X_joint,
    y,
    init_hyperparameters=init_hps,
    kernel_function=kernel_fn_numba,
    prior_mean_function=my_mean,
    noise_function=my_noise,
)

fingerprint_gp.train(
    hyperparameter_bounds=hps_bounds,
    init_hyperparameters=init_hps,
    method="global",
    max_iter=4000,
)

print("Training complete.")
print(
    "Log Likelihood:",
    fingerprint_gp.log_likelihood(fingerprint_gp.get_hyperparameters()),
)
print(
    "Learned hyperparameters:",
    np.array2string(
        fingerprint_gp.get_hyperparameters(), suppress_small=True, precision=4
    ),
)

results_dir = BO_RESULTS_DIR
if not results_dir.exists():
    results_dir.mkdir(parents=True)

file_name = f"gp_train_results_batch={INITIAL_TRAIN_BATCHES[0]}-{INITIAL_TRAIN_BATCHES[-1]}_target={TARGET_COLUMN}_{'sigma_per_chemical' if sigma_per_chemical else ''}.npz"

save_path = results_dir / file_name

np.savez_compressed(
    save_path,
    train_batches=np.array(INITIAL_TRAIN_BATCHES, dtype=str),
    train_samples=feature_df["sample_name"].to_numpy(dtype=str),
    log_likelihood=fingerprint_gp.log_likelihood(),
    hyperparameters=fingerprint_gp.get_hyperparameters(),
    config_info=np.array(configs_info, dtype=str),
)

print(f"Saved GP training results to {save_path}")
