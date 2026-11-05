from configs.exp_config import NOTEBOOK_CONFIG
from configs.paths import data_mapping_path
from hashlib import sha256
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

CONTINUOUS_FEATURE_COUNT = len(NOTEBOOK_CONFIG["param_config"])
CONTINUOUS_COLUMNS = list(NOTEBOOK_CONFIG["param_config"].keys())[
    :CONTINUOUS_FEATURE_COUNT
]
manual_entries = NOTEBOOK_CONFIG["manual_entries"]


def load_manual_entries(manual_entries, continuous_columns=CONTINUOUS_COLUMNS):
    manual_entry_rows = []
    for batch_num, sample_indices in manual_entries.items():
        # Construct the path to the data mapping file
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
            row = {col: recipe[col] for col in continuous_columns}
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

        print(
            f"Added manual entry for LPS-{batch_num} samples {sample_indices} (n={len(sample_indices)})"
        )

    manual_entries_df = (
        pd.DataFrame(manual_entry_rows)
        # .drop_duplicates(subset="config_hash")
        .set_index("manual_sample_id", drop=False)
    )
    print(
        f"Prepared {len(manual_entries_df)} total manual entries from {len(manual_entries)} batch(es): {list(manual_entries.keys())}"
    )

    return manual_entries_df


def add_a_site_column(df):
    df.loc[
        (df["a_site"].isna())
        & (df["sample_name"].str.split("-").str[0].isin(["3", "4", "5"])),
        "a_site",
    ] = "MeOMBAI"
    return df


def remove_outliers(df, outliers_by_sample_name):
    if len(outliers_by_sample_name):
        outliers_mask = df["sample_name"].isin(outliers_by_sample_name)
        print(
            f"Excluding Outliers excluded from training set: {df[outliers_mask]['sample_name'].tolist()}"
        )
        df = df[~outliers_mask]
    return df


def fill_empty_config_hash(df, continuous_columns=CONTINUOUS_COLUMNS):
    if "config_hash" not in df.columns:
        df["config_hash"] = (
            df[continuous_columns]
            .astype(str)
            .agg("-".join, axis=1)
            .str.replace(".", "", regex=False)
        )
    return df


def fill_and_clip_metrics(df):
    if "asymmetry" in df.columns:
        df["asymmetry"] = df["asymmetry"].fillna(1e-4).clip(lower=1e-4)

    if "asymmetry_5FWHM" in df.columns:
        # asymmetry uses 2.0x FWHM, therefore multiply the asymmetry_5FWHM by 2.5 of baseline of asymmetry
        df["asymmetry_5FWHM"] = (
            df["asymmetry_5FWHM"].fillna(1e-4).clip(lower=1e-4) * 2.5
        )

    if "g_factor_max" in df.columns:
        df["g_factor_max"] = df["g_factor_max"].fillna(1e-5).clip(lower=1e-5)

    if "ldlb_area_ratio_gmax" in df.columns:
        df["ldlb_area_ratio_gmax"] = (
            df["ldlb_area_ratio_gmax"].fillna(0.0).clip(lower=0.005)
        )

    if "homogeneity" in df.columns:
        df["homogeneity"] = df["homogeneity"].fillna(0.0).clip(lower=0.001)

    return df


def add_target_column(df, target_column=NOTEBOOK_CONFIG["default_target_column"]):
    if target_column not in df.columns:
        if target_column.startswith("score"):
            df[target_column] = (
                df["homogeneity"]
                * df["g_factor_max"]
                / (1 + df["ldlb_area_ratio_gmax"])
                * 1e5
            )
        if target_column.startswith("g_log_score"):
            df[target_column] = (
                df["homogeneity"]
                * (np.log10(df["g_factor_max"]) + 5)
                / (1 + df["ldlb_area_ratio_gmax"])
            )
        if target_column.startswith("CD_score"):
            df[target_column] = (
                df["g_factor_max"] / (1 + df["ldlb_area_ratio_gmax"]) * 5e3
            )
            print(
                f"Derived target column '{target_column}' from g_factor_max and ldlb_area_ratio_gmax"
            )
        if target_column == "homogeneity_score":
            df[target_column] = df["homogeneity"] * 100.0
            print(f"Using homogeneity as the target column '{target_column}'")
        if target_column == "g_factor_score":
            df[target_column] = df["g_factor_max"] * 1e4
            print(f"Using g_factor_max as the target column '{target_column}'")
        if target_column == "ldlb_score":
            df[target_column] = 1 / (1 + df["ldlb_area_ratio_gmax"]) * 25.0
            print(f"Derived target column '{target_column}' from ldlb_area_ratio_gmax")

    return df


def clean_dataset(
    df,
    outliers_by_sample_name,
    continuous_columns=CONTINUOUS_COLUMNS,
    target_column=NOTEBOOK_CONFIG["default_target_column"],
):
    df = add_a_site_column(df)
    df = remove_outliers(df, outliers_by_sample_name).copy()
    df = fill_empty_config_hash(df, continuous_columns=continuous_columns)
    df = fill_and_clip_metrics(df)
    df = add_target_column(df, target_column=target_column)

    required_columns = set(
        continuous_columns
        + [
            "config_hash",
            "a_site",
            "sample_name",
            "session",
            "sample_number",
            target_column,
        ]
    )
    missing_fields = required_columns - set(df.columns)
    if missing_fields:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing_fields)}")

    df["a_site_smiles"] = df["a_site"].map(NOTEBOOK_CONFIG["a_site_to_smiles"])

    df_no_smiles = df[df["a_site_smiles"].isna()]
    if not df_no_smiles.empty:
        ValueError(
            f"Warning: The following samples have missing a_site_smiles and will be dropped:\n{df_no_smiles[['sample_name', 'a_site']].to_string(index=False)}"
        )

    return df


def normalize_features(df, param_config):

    cols_to_scale = list(param_config.keys())

    range_df = pd.DataFrame(
        [
            [param_config[col]["range"][0] for col in cols_to_scale],
            [param_config[col]["range"][1] for col in cols_to_scale],
        ],
        columns=cols_to_scale,
    )

    scaler = MinMaxScaler()
    scaler.fit(range_df)

    df_normalized = df.copy()
    df_normalized[cols_to_scale] = scaler.transform(df_normalized[cols_to_scale])

    return df_normalized
