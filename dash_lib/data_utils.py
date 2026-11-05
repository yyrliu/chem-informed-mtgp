import duckdb
import xarray as xr
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from natsort import natsort_keygen
from pathlib import Path


def query_samples(db_path="mydb.duckdb", filters=None):
    con = duckdb.connect(db_path, read_only=True)

    # Support two filter styles for backward compatibility:
    # 1) dict: {column: value or [values]} (existing behavior)
    # 2) str: raw SQL where clause, with or without the leading "WHERE"
    where_clause = ""
    if isinstance(filters, str):
        raw = filters.strip()
        raw_lower = raw.lower()
        if raw_lower.startswith("where "):
            where_clause = raw[6:].strip()
        elif raw_lower == "where":
            where_clause = ""
        else:
            where_clause = raw
    else:
        clauses = []
        for k, v in (filters or {}).items():
            if isinstance(v, list):
                # Create a comma-separated list of quoted strings for the IN clause
                in_values = ", ".join([f"'{item}'" for item in v])
                clauses.append(f"{k} IN ({in_values})")
            else:
                # Handle single value with =
                clauses.append(f"{k} = '{v}'")
        where_clause = " AND ".join(clauses)

    query = f"SELECT * FROM samples {('WHERE ' + where_clause) if where_clause else ''}"

    # Set index to unique_id for easier lookup later
    df = (
        con.execute(query)
        .fetchdf()
        .set_index("unique_id")
        .sort_values(by="sample_name", key=natsort_keygen())
    )

    # Rename sample_name by removing "LPS-" prefix and pad the sample_number with leading zeros
    if "sample_name" in df.columns:
        df["sample_name"] = df["sample_name"].str.replace(r"^LPS-", "", regex=True)
        df["sample_name"] = df["sample_name"].str.replace(
            r"(\d+)-(\d+)",
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}",
            regex=True,
        )

    con.close()
    return df


def load_single_dataset(path):
    """Load a single dataset file - works for both sync and async modes"""
    with xr.open_dataset(path, engine="hidefix") as ds:
        return ds.load()


def load_dataset(
    unique_ids,
    subgroup,
    db_path="mydb.duckdb",
    async_mode=False,
    open_mfdataset=True,
    no_load=False,
    _dss={},
):
    if async_mode:
        print(
            "Warning: async_mode is deprecated and will be removed in future versions. Please use open_mfdataset=True for efficient multi-file loading instead."
        )

    if isinstance(unique_ids, str):
        unique_ids = [unique_ids]

    combined_ds = _dss.get(subgroup)

    if combined_ds is not None:
        missing_sample_ids = set(unique_ids) - set(combined_ds.sample_id.values)
        if not missing_sample_ids:
            return combined_ds.sel(sample_id=unique_ids)

        all_meta_df = query_samples(
            db_path, filters={"unique_id": list(missing_sample_ids)}
        )
        if all_meta_df.empty:
            return combined_ds.sel(sample_id=unique_ids)

    else:
        all_meta_df = query_samples(db_path, filters={"unique_id": unique_ids})
        if all_meta_df.empty:
            return None

    path_key = f"{subgroup}_path"
    if path_key not in all_meta_df.columns:
        return None

    ds_paths = all_meta_df[path_key].dropna()
    # resolve path from db_path, assuming paths in the database are relative to the db_path directory
    db_dir = Path(db_path).parent
    ds_paths = ds_paths.apply(lambda p: (db_dir / p).resolve())
    ds_paths = ds_paths[ds_paths.apply(lambda p: p.is_file())]

    if ds_paths.empty:
        if combined_ds is not None:
            return combined_ds.where(
                combined_ds["sample_id"].isin(unique_ids), drop=True
            )
        else:
            return None

    if open_mfdataset:
        retrieved_combined_ds = xr.open_mfdataset(
            ds_paths.tolist(),
            combine="nested",
            concat_dim="sample_id",
            parallel=True,
            engine="hidefix",
        )
        retrieved_combined_ds = retrieved_combined_ds.assign_coords(
            sample_id=("sample_id", ds_paths.index)
        )
    else:
        loaded_datasets = [load_single_dataset(path) for path in ds_paths.tolist()]
        retrieved_combined_ds = xr.concat(
            loaded_datasets, dim=pd.Index(ds_paths.index, name="sample_id")
        )

    if not no_load:
        retrieved_combined_ds.load()

    for col in all_meta_df.columns:
        if "path" not in col.lower():
            # Create a DataArray for the metadata column, indexed by sample_id
            coord_da = xr.DataArray(all_meta_df[col], dims=["sample_id"])
            retrieved_combined_ds = retrieved_combined_ds.assign_coords({col: coord_da})

    if combined_ds is not None:
        combined_ds = xr.concat([combined_ds, retrieved_combined_ds], dim="sample_id")
    else:
        combined_ds = retrieved_combined_ds

    _dss[subgroup] = combined_ds

    return combined_ds.where(combined_ds["sample_id"].isin(unique_ids), drop=True)


def analyze_spectrum(ds, var="absorption_avg", subtract_bg=False):
    if var not in ds:
        return {"peaks": [], "processed_spectrum": np.array([])}
    spectrum = ds[var].values
    if subtract_bg:
        spectrum -= np.min(spectrum)
    peaks, _ = find_peaks(spectrum, height=0.1, distance=10)
    return {
        "peaks": ds["wavelength"].values[peaks] if "wavelength" in ds else peaks,
        "processed_spectrum": spectrum,
    }
