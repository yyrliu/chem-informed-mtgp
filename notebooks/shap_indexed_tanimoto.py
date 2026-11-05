# %%
import json
import os
import pickle
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import shapiq
from gpcam import GPOptimizer
from sklearn.preprocessing import MinMaxScaler

from chem_informed_mtgp.fingerprint_kernel import (
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

from configs.exp_config import NOTEBOOK_CONFIG  # noqa: E402
from configs.paths import GP_TRAIN_RESULT_FILE, data_mapping_path  # noqa: E402
from dash_lib.data_utils import query_samples  # noqa: E402

########## configuration ##########
RANDOM_SEED = 42
SIGMA_PER_CHEMICAL = True  # Toggle per-chemistry signal scaling in GP

param_config = NOTEBOOK_CONFIG["param_config"]
outliers_by_sample_name = NOTEBOOK_CONFIG["outliers_by_sample_name"]
INITIAL_TRAIN_BATCHES = NOTEBOOK_CONFIG["train_batches"]
A_SITE_TO_SMILES = NOTEBOOK_CONFIG["a_site_to_smiles"]
manual_entries = NOTEBOOK_CONFIG["manual_entries"]

CONTINUOUS_COLUMNS = list(param_config.keys())
TARGET_COLUMN = NOTEBOOK_CONFIG["default_target_column"]

GP_TRAIN_RESULT_PATH = GP_TRAIN_RESULT_FILE

SHAP_CACHE_PATH = Path(SCRIPT_DIR)
###################################

# %%

########## manual entries ##########
manual_entry_rows = []
for batch_num, sample_indices in manual_entries.items():
    mapping_path = data_mapping_path(batch_num)
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing LPS-{batch_num} data mapping at {mapping_path}"
        )

    with mapping_path.open("r") as f:
        data_mapping = json.load(f)

    for sample_idx in sample_indices:
        entry = data_mapping.get(str(sample_idx))
        if entry is None:
            raise KeyError(
                f"Sample {sample_idx} not found in LPS-{batch_num} data mapping."
            )

        recipe = entry["recipe"]
        row = {col: recipe[col] for col in CONTINUOUS_COLUMNS}
        row["config_hash"] = sha256(
            json.dumps(recipe, sort_keys=True).encode("utf-8")
        ).hexdigest()
        row["a_site"] = recipe.get("a_site")
        sample_name = f"{batch_num}-{int(sample_idx):02d}"
        row["sample_name"] = sample_name
        row["session"] = f"LPS-{batch_num}"
        row["sample_number"] = int(sample_idx)
        row["homogeneity"] = 0.0
        row["asymmetry"] = 0.0
        row["g_factor_max"] = 0.0
        row["asymmetry_5FWHM"] = 0.0
        row["manual_sample_id"] = f"manual-{sample_name}"
        manual_entry_rows.append(row)

    print(f"Added manual entry for LPS-{batch_num} samples {sample_indices}")

manual_entries_df = (
    pd.DataFrame(manual_entry_rows).set_index("manual_sample_id", drop=False)
    if manual_entry_rows
    else pd.DataFrame()
)
####################################

# %%

########## data loading & preprocessing ##########
db_path = str(Path(SCRIPT_DIR).parent / "mydb.duckdb")
df = query_samples(db_path=db_path)

if "sample_name" in df.columns:
    df = df[df["sample_name"].str.split("-").str[0].isin(INITIAL_TRAIN_BATCHES)]

if not manual_entries_df.empty:
    df = pd.concat([df, manual_entries_df], sort=False)

if outliers_by_sample_name:
    outliers_mask = df["sample_name"].isin(outliers_by_sample_name)
    print(
        f"Excluding {outliers_mask.sum()} outliers: {df[outliers_mask]['sample_name'].tolist()}"
    )
    df = df[~outliers_mask]

if "config_hash" not in df.columns:
    df["config_hash"] = (
        df[CONTINUOUS_COLUMNS]
        .astype(str)
        .agg("-".join, axis=1)
        .str.replace(".", "", regex=False)
    )

if "asymmetry" in df.columns:
    df["asymmetry"] = df["asymmetry"].fillna(1e-4).clip(lower=1e-4)
if "g_factor_max" in df.columns:
    df["g_factor_max"] = df["g_factor_max"].fillna(1e-5).clip(lower=1e-5)
if "ldlb_area_ratio_gmax" in df.columns:
    df["ldlb_area_ratio_gmax"] = (
        df["ldlb_area_ratio_gmax"].fillna(0.0).clip(lower=0.005)
    )
if "homogeneity" in df.columns:
    df["homogeneity"] = df["homogeneity"].fillna(0.0).clip(lower=0.001)

df[TARGET_COLUMN] = (
    df["homogeneity"] * df["g_factor_max"] / (1 + df["ldlb_area_ratio_gmax"]) * 1e5
)

df.loc[
    df["a_site"].isna() & df["sample_name"].str.split("-").str[0].isin(["3", "4", "5"]),
    "a_site",
] = "MeOMBAI"

df = df[df["a_site"].isin(A_SITE_TO_SMILES.keys())].copy()
df = df.dropna(subset=CONTINUOUS_COLUMNS + [TARGET_COLUMN] + ["a_site"]).reset_index(
    drop=True
)

print(f"\nData preprocessing complete: {len(df)} rows")
##################################################

# %%

########## feature matrices ##########
fingerprint_metadata = build_fingerprint_metadata(
    A_SITE_TO_SMILES, n_bits=FINGERPRINT_BITS, radius=2, use_features=False
)

print(f"A-sites: {fingerprint_metadata.a_sites}")
print(f"Tanimoto correlations:\n{fingerprint_metadata.tanimoto_correlations}")

# Unscaled feature matrix (for shapiq — raw continuous values + a_site index)
X = assemble_feature_matrix(
    df=df,
    continuous_columns=CONTINUOUS_COLUMNS,
    fingerprint_column="a_site",
    fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
    fingerprint_metadata=fingerprint_metadata,
    a_site_column="a_site",
)
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + ["a_site_code"]
CONTINUOUS_FEATURE_COUNT = len(CONTINUOUS_COLUMNS)

assert not np.isnan(X).any(), "NaN values found in feature matrix"
print(f"\nFeature matrix shape: {X.shape}, features: {FEATURE_COLUMNS}")
######################################

# %%

########## GP model setup ##########
df_norm = df.copy()
range_df = pd.DataFrame(
    [
        [param_config[col]["range"][0] for col in CONTINUOUS_COLUMNS],
        [param_config[col]["range"][1] for col in CONTINUOUS_COLUMNS],
    ],
    columns=CONTINUOUS_COLUMNS,
)
scaler = MinMaxScaler()
scaler.fit(range_df)
df_norm[CONTINUOUS_COLUMNS] = scaler.transform(df_norm[CONTINUOUS_COLUMNS])
df_norm["a_site_smiles"] = df_norm["a_site"].map(A_SITE_TO_SMILES)

X_gp = assemble_feature_matrix(
    df=df_norm,
    continuous_columns=CONTINUOUS_COLUMNS,
    fingerprint_column="a_site_smiles",
    fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
    fingerprint_metadata=fingerprint_metadata,
    a_site_column="a_site",
)
y_gp = df[TARGET_COLUMN].to_numpy(dtype=np.float64)

kernel_fn_numba = joint_kernel_factory_numba(
    cont_dims=CONTINUOUS_FEATURE_COUNT,
    fingerprint_bits=1,  # indexed_tanimoto: 1 categorical index
    fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
    fingerprint_metadata=fingerprint_metadata,
    tanimoto_correlations=fingerprint_metadata.tanimoto_correlations,
    sigma_per_chemical=SIGMA_PER_CHEMICAL,
)

if not GP_TRAIN_RESULT_PATH.exists():
    raise FileNotFoundError(
        f"Could not find training result file: {GP_TRAIN_RESULT_PATH}"
    )

train_result_npz = np.load(GP_TRAIN_RESULT_PATH)
print("Loaded training results:")
print("Train batches:", train_result_npz["train_batches"])
print("Log Likelihood:", train_result_npz["log_likelihood"])
print(
    "Hyperparameters:",
    np.array2string(
        train_result_npz["hyperparameters"], suppress_small=True, precision=4
    ),
)

gp_model = GPOptimizer(
    X_gp,
    y_gp,
    init_hyperparameters=train_result_npz["hyperparameters"],
    kernel_function=kernel_fn_numba,
    prior_mean_function=my_mean,
    noise_function=my_noise,
)

if np.isclose(
    gp_model.get_hyperparameters(), train_result_npz["hyperparameters"]
).all():
    print("Hyperparameters successfully loaded to `gp_model`.")
else:
    print("Warning: loaded hyperparameters do not match initialized values.")

loaded_ll = float(train_result_npz["log_likelihood"])
gp_ll = float(gp_model.log_likelihood())
if np.isclose(gp_ll, loaded_ll, atol=1):
    print("Log likelihood matches loaded training result.")
else:
    print(
        f"Warning: log likelihood mismatch — GP: {gp_ll:.3f}, loaded: {loaded_ll:.3f}"
    )

train_result_npz.close()
####################################

# %%

########## GP prediction wrapper for shapiq ##########
CODE_TO_A_SITE = {idx: site for idx, site in enumerate(fingerprint_metadata.a_sites)}


def gp_predict_wrapper(X_7d: np.ndarray) -> np.ndarray:
    """GP posterior mean over 7D feature space [6 continuous + 1 a_site index]."""
    X_7d = np.atleast_2d(np.asarray(X_7d, dtype=float))
    x_df = pd.DataFrame(X_7d, columns=FEATURE_COLUMNS)
    x_df[CONTINUOUS_COLUMNS] = x_df[CONTINUOUS_COLUMNS].astype(float)

    max_code = len(CODE_TO_A_SITE) - 1
    x_df["a_site_code_clipped"] = np.clip(
        np.rint(x_df["a_site_code"]).astype(int), 0, max_code
    )
    x_df["a_site"] = x_df["a_site_code_clipped"].map(CODE_TO_A_SITE)

    if x_df["a_site"].isna().any():
        bad = sorted(x_df.loc[x_df["a_site"].isna(), "a_site_code_clipped"].unique())
        raise ValueError(f"Invalid a_site_code values: {bad}")

    x_df_norm = x_df.copy()
    x_df_norm[CONTINUOUS_COLUMNS] = scaler.transform(x_df_norm[CONTINUOUS_COLUMNS])
    x_df_norm["a_site_smiles"] = x_df_norm["a_site"].map(A_SITE_TO_SMILES)

    if x_df_norm["a_site_smiles"].isna().any():
        missing = sorted(
            x_df_norm.loc[x_df_norm["a_site_smiles"].isna(), "a_site"].unique()
        )
        raise ValueError(f"Missing SMILES mapping: {missing}")

    X_gp_pred = assemble_feature_matrix(
        df=x_df_norm,
        continuous_columns=CONTINUOUS_COLUMNS,
        fingerprint_column="a_site_smiles",
        fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
        fingerprint_metadata=fingerprint_metadata,
        a_site_column="a_site",
    )
    return np.asarray(gp_model.posterior_mean(X_gp_pred)["m(x)"], dtype=float).reshape(
        -1
    )


print("GP prediction wrapper created")
######################################################

# %%

########## shapiq k-SII computation (orders 1–3) ##########
# Deduplicate by config_hash before explaining
unique_configs, unique_indices = np.unique(df["config_hash"], return_index=True)
X_shapiq = X[unique_indices]
y_shapiq = df.loc[unique_indices, TARGET_COLUMN].to_numpy(dtype=float)
print(f"Unique configs: {len(unique_configs)}, X_shapiq shape: {X_shapiq.shape}")

feature_names = FEATURE_COLUMNS

for max_order in range(1, 4):
    print(f"Setting up shapiq TabularExplainer (order-{max_order} k-SII)...")
    explainer = shapiq.TabularExplainer(
        model=gp_predict_wrapper,
        data=X_shapiq,
        index="k-SII",
        max_order=max_order,
        categorical_features=[6],
        random_state=RANDOM_SEED,
        sample_size=len(X_shapiq),
    )

    # budget >= 128 guarantees exact values for 7 features
    shapiq_values = explainer.explain_X(X_shapiq, budget=256, n_jobs=-1)
    print(f"Order-{max_order} k-SII values computed")

    cache = {
        "shapiq_values": shapiq_values,
        "X_shapiq": X_shapiq,
        "y_shapiq": y_shapiq,
        "feature_names": feature_names,
        "a_site_names": list(fingerprint_metadata.a_sites),
    }

    pkl_path = Path(
        SHAP_CACHE_PATH,
        f"order_{max_order}_shap_indexed_tanimoto_cache_rs={RANDOM_SEED}.pkl",
    )

    with pkl_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved SHAP cache ({len(shapiq_values)} instances) to {pkl_path}")
########################################################

# %%
