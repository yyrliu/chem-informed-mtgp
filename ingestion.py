import os
import argparse
import duckdb
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import hashlib
from pathlib import Path
from datetime import datetime
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from chem_informed_mtgp.xr_g_factor_analysis import (
    calculate_asymmetry as xr_calculate_asymmetry,
    find_g_factor_max as xr_find_g_factor_max,
    calculate_ldlb_area_ratio as xr_calculate_ldlb_area_ratio,
)
from joblib import Parallel, delayed


HOMOGENITY_X_RANGE = (1000, 3100)
HOMOGENITY_Y_RANGE = (300, 2450)

# Define the wavelength range for g-factor calculation in nm
G_FACTOR_MAX_WAVELENGTH_RANGE = (470, 520)
G_FACTOR_MAX_FWHM_WINDOW = "both"
ASYMMETRY_WIDTH_FOR_HEIGHT_REL = 0.5
ASYMMETRY_WAVELENGTH_RANGE = (480, 520)
ASYMMETRY_MIN_PEAK_HEIGHT = 0.05


def _calculate_homogeneity(image_data, x_range, y_range):
    """
    Calculates a homogeneity score for an image.
    The score is the mean entropy of the HSV channels.
    """
    # 1. Create a square mask for the range of interest
    image_data = image_data.values.squeeze()
    image_data = image_data[y_range[0] : y_range[1], x_range[0] : x_range[1], :]

    # 2. Convert RGB to HSV
    hsv_img = img_as_ubyte(rgb2hsv(image_data))

    # 3. Use skimage.feature.graycomatrix and graycoprops on the V channel (hsv_img[..., 2])
    v_channel = hsv_img[..., 2]
    # Compute GLCM (gray-level co-occurrence matrix)
    glcm = graycomatrix(
        v_channel,
        distances=[1, 3, 5],
        angles=[i * np.pi / 4 for i in range(4)],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast").mean()
    energy = graycoprops(glcm, "energy").mean()
    homogeneity = 1.0 / (contrast + 1e-8) * energy * 100

    return xr.DataArray(homogeneity)


def _calculate_g_factor_max(cd_dataset, wl_range, fwhm_window):
    """
    Wrapper for centralized logic in xr_g_factor_analysis.py.
    Find |g-factor| g_max from 2*HWHM window of absorption peak.

    Returns:
        tuple: (g_factor_max, g_factor_max_wavelength, wavelength_range)
               g_factor_max and g_factor_max_wavelength are np.nan if calculation fails.
               wavelength_range is the actual search range used (tuple), or input wl_range if failed.
    """

    # baseline_correction set to false for compatibility with ingested batch 3-20.
    res = xr_find_g_factor_max(
        cd_dataset,
        wl_range=wl_range,
        fwhm_window=fwhm_window,
        baseline_correction=False,
    )

    # Return NaN for any non-success status, but preserve the input range
    if res["status"] != "success":
        return np.nan, np.nan, wl_range

    # Return the actual search range that was used
    actual_range = res["search_range"]
    return res["max_g_factor"], res["max_wavelength"], actual_range


def _calculate_asymmetry(cd_dataset, peak_height, rel_height, wl_range):
    """
    Wrapper for centralized logic in xr_g_factor_analysis.py.
    Calculate asymmetry score (integrated |g-factor| over 2*HWHM window).

    Returns:
        tuple: (asymmetry_score, wavelength_range)
               asymmetry_score is np.nan if calculation fails.
               wavelength_range is the actual integration range used (tuple), or input wl_range if failed.
    """
    res_asymmetry = xr_calculate_asymmetry(
        cd_dataset,
        min_peak_height=peak_height,
        rel_height=rel_height,
        wl_range=wl_range,
    )

    # Return NaN for any non-success status, but preserve the input range
    if res_asymmetry["status"] != "success":
        return np.nan, wl_range

    # Return the actual integration range that was used
    actual_range = res_asymmetry["integration_range"]
    return res_asymmetry["asymmetry_score"], actual_range


def _calculate_ldlb_ratio(cd_dataset, wl_range):
    """
    Wrapper for centralized logic in xr_g_factor_analysis.py.
    Calculate the ratio of LDLB-CD to general-CD integrated areas.

    Returns:
        float: ldlb_area_ratio, or np.nan if calculation fails.
    """
    res = xr_calculate_ldlb_area_ratio(cd_dataset, wl_range=wl_range)

    # Return NaN for any non-success status
    if res["status"] != "success":
        return np.nan

    return res["ldlb_ratio"]


def compute_process_hash(process_params):
    if not process_params:
        return None
    param_str = str(sorted(process_params.items()))
    return hashlib.sha256(param_str.encode()).hexdigest()


def _ingest_h5_file(path, parquet_dir="data_parquet"):
    print(f"Ingesting {Path(path).name}...")
    path = Path(path)
    with h5py.File(path, "r") as f:
        # Extract metadata
        session = f["/hardware/mf_crucible_spinbot/settings"].attrs.get("session_name")
        if session is None or not session.startswith("LPS-"):
            session = path.stem.split("_")[1]

        sample_number = f["/app/settings"].attrs.get("sample", "")

        meta = {
            "unique_id": f["/"].attrs["unique_id"],  # Use filename without extension
            "session": session,
            "sample_number": sample_number,
            "sample_name": f"{session}-{sample_number}",
            "filename": path.name,
            "filepath": str(path),
            "timestamp": datetime.fromtimestamp(f.attrs.get("time_id", 0)).isoformat(
                timespec="seconds"
            ),
            "cd_path": "",
            "cbox_uv_vis_path": "",
            "cbox_pl_path": "",
            "xrd_path": "",
            "xrd_xy_path": "",
            "cbox_photo_path": "",
        }

        process_param_keys = [
            "annealing_temp",
            "annealing_time",
            "concentration",
            "config_hash",
            "dispense_speed",
            "dispense_volume",
            "spin_speed",
            "a_site",
            "wash",
        ]

        if "/process_params" in f:
            for key in process_param_keys:
                meta[key] = f["/process_params"].attrs.get(key, None)
        else:
            for key in process_param_keys:
                meta[key] = None

        # Process results into xarray
        subgroups = {}
        if "results/cd" in f:
            g = f["results/cd"]
            ds = xr.Dataset(
                {
                    "front_CD": ("wavelength", g["front_CD"][:]),
                    "back_CD": ("wavelength", g["back_CD"][:]),
                    "front_abs": ("wavelength", g["front_abs"][:]),
                    "back_abs": ("wavelength", g["back_abs"][:]),
                    "gen_CD": ("wavelength", g["gen_CD"][:]),
                    "ldlb_CD": ("wavelength", g["ldlb_CD"][:]),
                    "abs_avg": ("wavelength", g["abs_avg"][:]),
                    "g_factor": ("wavelength", g["g_factor"][:]),
                },
                coords={"wavelength": g["wavelength"][:]},
            )
            subgroups["cd"] = ds

        if "results/cbox/uv_vis" in f:
            g = f["results/cbox/uv_vis"]
            ds = xr.Dataset(
                {
                    "absorption": (["position", "wavelength"], g["absorption"][:]),
                    "absorption_avg": ("wavelength", g["absorption_avg"][:]),
                    "transmission": (
                        ["position", "wavelength"],
                        g["transmission"][:],
                    ),
                    "transmission_avg": ("wavelength", g["transmission_avg"][:]),
                },
                coords={
                    "wavelength": g["wavelength"][:],
                    "position": np.arange(g["absorption"].shape[0]),
                },
            )
            subgroups["cbox_uv_vis"] = ds

        if "results/cbox/pl" in f:
            g = f["results/cbox/pl"]
            ds = xr.Dataset(
                {
                    "intensity": (["position", "wavelength"], g["intensity"][:]),
                    "intensity_avg": ("wavelength", g["intensity_avg"][:]),
                },
                coords={
                    "wavelength": g["wavelength"][:],
                    "position": np.arange(g["intensity"].shape[0]),
                },
            )
            subgroups["cbox_pl"] = ds

        if "results/xrd" in f:
            g = f["results/xrd"]
            ds = xr.Dataset(
                {"intensity": ("2theta", g["int"][:])},
                coords={"2theta": g["2theta_Cu"][:]},
            )
            subgroups["xrd"] = ds

        if "results/xrd_xy" in f:
            g = f["results/xrd_xy"]
            ds = xr.Dataset(
                {
                    "intensity": ("2theta", g["int"][:]),
                    "2theta_Co": ("2theta", g["2theta_Co"][:]),
                },
                coords={"2theta": g["2theta_Cu"][:]},
            )
            subgroups["xrd_xy"] = ds

        skip_photo = os.environ.get("LPS_SKIP_PHOTO") == "1"
        if "results/cbox/photo" in f and not skip_photo:
            g = f["results/cbox/photo"]
            ds = xr.Dataset(
                {"image": (["y", "x", "channel"], g["image"][:])},
                coords={
                    "y": np.arange(g["image"].shape[0]),
                    "x": np.arange(g["image"].shape[1]),
                    "channel": np.arange(g["image"].shape[2]),
                },
            )
            subgroups["cbox_photo"] = ds

        # Calculate homogeneity if photo data exists
        if "cbox_photo" in subgroups:
            homogeneity_range = (HOMOGENITY_X_RANGE, HOMOGENITY_Y_RANGE)
            # if int(meta["session"].split("-")[1]) >= 15:
            #     # shift y range up by 150 pixels for batch 15 and later to account for changed camera position
            #     homogeneity_range = (
            #         HOMOGENITY_X_RANGE,
            #         (HOMOGENITY_Y_RANGE[0] - 150, HOMOGENITY_Y_RANGE[1] - 150),
            #     )

            homogeneity = _calculate_homogeneity(
                subgroups["cbox_photo"]["image"],
                x_range=homogeneity_range[0],
                y_range=homogeneity_range[1],
            )
            meta["homogeneity"] = homogeneity.item()
        else:
            meta["homogeneity"] = np.nan

        # Calculate CD-based metrics if cd data exists
        if "cd" in subgroups:
            # Calculate g_factor_max and wavelength
            gval, gwl, g_range = _calculate_g_factor_max(
                subgroups["cd"],
                wl_range=G_FACTOR_MAX_WAVELENGTH_RANGE,
                fwhm_window=G_FACTOR_MAX_FWHM_WINDOW,
            )
            meta["g_factor_max"] = gval
            meta["g_factor_max_wavelength"] = gwl

            # Calculate LDLB ratio for g_factor (using g_factor's wavelength range)
            ldlb_ratio_gmax = _calculate_ldlb_ratio(
                subgroups["cd"],
                wl_range=g_range,
            )
            meta["ldlb_area_ratio_gmax"] = ldlb_ratio_gmax

            # Calculate asymmetry area
            asymmetry, asym_range = _calculate_asymmetry(
                subgroups["cd"],
                peak_height=ASYMMETRY_MIN_PEAK_HEIGHT,
                rel_height=ASYMMETRY_WIDTH_FOR_HEIGHT_REL,
                wl_range=ASYMMETRY_WAVELENGTH_RANGE,
            )
            meta["asymmetry"] = asymmetry

            # Calculate LDLB ratio for asymmetry (using asymmetry's wavelength range)
            ldlb_ratio_asym = _calculate_ldlb_ratio(
                subgroups["cd"],
                wl_range=asym_range,
            )
            meta["ldlb_area_ratio_asym"] = ldlb_ratio_asym
        else:
            meta["g_factor_max"] = np.nan
            meta["g_factor_max_wavelength"] = np.nan
            meta["ldlb_area_ratio_gmax"] = np.nan
            meta["asymmetry"] = np.nan
            meta["ldlb_area_ratio_asym"] = np.nan

        # Export to netCDF (compatible with xarray)
        for subgroup, ds in subgroups.items():
            netcdf_path = f"{parquet_dir}/{subgroup}/{meta['unique_id']}.nc"
            os.makedirs(os.path.dirname(netcdf_path), exist_ok=True)
            ds.to_netcdf(netcdf_path, engine="h5netcdf")
            meta[f"{subgroup}_path"] = netcdf_path

    return meta


def _ingest_h5_file_with_retry(path, parquet_dir="data_parquet", max_retries=3):
    for attempt in range(max_retries):
        try:
            return _ingest_h5_file(path, parquet_dir)
        except Exception as e:
            print(f"Error ingesting {path} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"Failed to ingest {path} after {max_retries} attempts.")
                raise


def ingest_h5_files(
    h5_paths, parquet_dir="data_parquet", db_path="mydb.duckdb", append=False
):
    os.makedirs(parquet_dir, exist_ok=True)
    con = duckdb.connect(db_path)

    data_list = Parallel(n_jobs=-3)(
        delayed(_ingest_h5_file_with_retry)(path, parquet_dir=parquet_dir)
        for path in h5_paths
    )

    # Ingest metadata to DuckDB
    df_meta = pd.DataFrame(data_list)
    # Ensure wash column is string type
    if "wash" in df_meta.columns:
        df_meta["wash"] = df_meta["wash"].astype("string")
    if "a_site" in df_meta.columns:
        df_meta["a_site"] = df_meta["a_site"].astype("string")

    con.register("temp_table", df_meta)

    # Check if table exists
    table_exists = (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='samples'"
        ).fetchone()
        is not None
    )

    if append and table_exists:
        con.execute("INSERT OR REPLACE INTO samples SELECT * FROM temp_table")
    else:
        # Create fresh table with constraints
        con.execute("CREATE OR REPLACE TABLE samples AS SELECT * FROM temp_table")
        con.execute("ALTER TABLE samples ALTER unique_id SET NOT NULL")
        con.execute("ALTER TABLE samples ADD PRIMARY KEY (unique_id)")

    con.close()


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest H5 files into database.")
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to existing table instead of replacing",
    )
    parser.add_argument("paths", nargs="*", help="H5 file paths or directory")

    args = parser.parse_args()

    if args.paths:
        first_path = Path(args.paths[0])
        if first_path.is_dir():
            h5_paths = list(first_path.rglob("*.h5"))
        else:
            h5_paths = args.paths
    else:
        parser.error("provide at least one preprocessed H5 file or directory")
    ingest_h5_files(h5_paths, append=args.append)
