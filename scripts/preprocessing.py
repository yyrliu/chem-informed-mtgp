"""
LPS Preprocessing Pipeline

This module processes H5 files from CBox measurements and integrates associated CD and XRD data
into a unified HDF5 structure for comprehensive analysis.

XRD Processing:
===============
XRD data can come from:
- XY files: Processed directly from tabular XRD measurements
- GFRM files: 2D detector images integrated using pyFAI with a PONI calibration file

When both XY and GFRM data are available, GFRM is stored as primary XRD data and XY is
nested under the 'xy_data' key for comparison or validation purposes.

H5 Data Structure:
==================

Root level:
├── measurement/                 # Original measurement data (preserved)
├── results/                     # New processed results group
│   ├── cbox/                   # CBox spectroscopy results
│   │   ├── uv_vis/
│   │   │   ├── wavelength      # 1D array [n_wavelengths]
│   │   │   ├── energy          # 1D array [n_wavelengths]
│   │   │   ├── absorption      # 2D array [N, n_wavelengths] (individual spectra)
│   │   │   ├── absorption_avg  # 1D array [n_wavelengths] (averaged spectrum)
│   │   │   ├── absorption_jacobian  # 2D array [N, n_wavelengths] (individual with Jacobian)
│   │   │   ├── absorption_jacobian_avg # 1D array [n_wavelengths] (averaged with Jacobian)
│   │   │   ├── transmission    # 2D array [N, n_wavelengths] (individual spectra)
│   │   │   ├── transmission_avg # 1D array [n_wavelengths] (averaged spectrum)
│   │   │   └── num_spectra     # Scalar (number of individual spectra, typically 8)
│   │   │
│   │   ├── pl/
│   │   │   ├── wavelength      # 1D array [n_wavelengths]
│   │   │   ├── energy          # 1D array [n_wavelengths]
│   │   │   ├── intensity       # 2D array [N, n_wavelengths] (individual spectra)
│   │   │   ├── intensity_avg   # 1D array [n_wavelengths] (averaged spectrum)
│   │   │   ├── intensity_jacobian   # 2D array [N, n_wavelengths] (individual with Jacobian)
│   │   │   ├── intensity_jacobian_avg # 1D array [n_wavelengths] (averaged with Jacobian)
│   │   │   └── num_spectra     # Scalar (number of individual spectra)
│   │   │
│   │   └── photo/
│   │       ├── image           # 2D array [height, width]
│   │       └── exposure        # Scalar (adj_photo_exposure)
│   │
│   ├── cd/                     # Circular dichroism results
│   │   ├── wavelength          # 1D array
│   │   ├── front_CD            # 1D array
│   │   ├── back_CD             # 1D array
│   │   ├── front_abs           # 1D array
│   │   ├── back_abs            # 1D array
│   │   ├── gen_CD              # 1D array (genuine CD)
│   │   ├── ldlb_CD             # 1D array (linear dichroism + birefringence)
│   │   ├── abs_avg             # 1D array (average absorbance)
│   │   ├── g_factor            # 1D array (g-factor: gen_CD / (abs_avg * 32980))
│   │   └── file_info/          # Metadata about source files
│   │       ├── front_file      # String (source filename)
│   │       └── back_file       # String (source filename)
│   │
│   ├── xrd/                    # XRD results (GFRM data when available, else falls back to XY)
│   │   ├── 2theta_Cu           # 1D array (Cu K-alpha scale)
│   │   ├── 2theta_Co           # 1D array (original Co K-alpha scale)
│   │   ├── int                 # 1D array (intensity)
│   │   └── file_info (attrs)   # Source file metadata
│   │
│   └── xrd_xy/                 # XY XRD results (alternate location when both GFRM and XY exist)
│       ├── 2theta_Cu           # 1D array
│       ├── 2theta_Co           # 1D array
│       ├── int                 # 1D array
│       └── file_info (attrs)   # Source file metadata
│
└── process_params/             # Process parameters from recipes.json (ROOT LEVEL)
    ├── annealing_temp          # Scalar (annealing temperature)
    ├── annealing_time          # Scalar (annealing time)
    ├── dispense_volume         # Scalar (dispense volume)
    ├── spin_speed              # Scalar (spin speed)
    ├── dispense_speed          # Scalar (dispense speed)
    └── concentration           # Scalar (concentration)
"""

import cv2
import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Optional
from hashlib import sha256
from joblib import Parallel, delayed, cpu_count
import pyFAI
import fabio

# Import from existing modules
from chem_informed_mtgp.h5_analysis import (
    filter_h5_files,
    read_h5_file,
    process_uv_vis_data_raw,
    process_pl_data_raw,
)
from chem_informed_mtgp.jws_converter import find_and_convert_cd_file


def find_associated_files(
    h5_file_path: str,
    base_dir: str,
    xrd_filename_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[str]]:
    """
    Find CD and XRD files associated with an H5 file.

    Args:
        h5_file_path: Path to the H5 file
        base_dir: Base directory containing subdirectories (CBox, CD, XRD)
        xrd_filename_mapping: Pre-computed mapping of sample[:13] to XRD filenames

    Returns:
        Dictionary with paths to associated files
    """
    # Extract sample[:13] from H5 filename according to notebook specification
    # Format: {batch_name}_{sample_position}_{sample[:13]}_run{n}_spec_run
    h5_path = Path(h5_file_path)
    *batch_name, sample_position, sample_id, run_string = h5_path.name.removesuffix(
        "_spec_run.h5"
    ).split("_")
    h5_path = Path(h5_file_path)

    if not sample_id:
        print(f"Warning: Could not extract sample[:13] from filename: {h5_path.stem}")
        return {"cd_front": None, "cd_back": None, "xrd": None}

    # Ensure we use only first 13 characters as specified in notebook
    sample_id = sample_id[:13]

    associated_files = {
        "cd_front": None,
        "cd_back": None,
        "xrd": None,
        "xrd_gfrm": None,
    }

    base_path = Path(base_dir)

    # Search for CD files in {base_dir}/CD/**/*.txt or **/*.jws
    # and convert .jws to .txt if necessary.
    cd_dir = base_path / "CD"
    if cd_dir.exists():
        associated_files["cd_front"] = find_and_convert_cd_file(
            cd_dir, sample_id, "front"
        )
        associated_files["cd_back"] = find_and_convert_cd_file(
            cd_dir, sample_id, "back"
        )

    # Search for XRD files using pre-computed mapping
    # After remapping, XRD xy files follow {sample[:13]}.xy pattern and the gfrm files follow {sample[:13]}.gfrm pattern
    if xrd_filename_mapping and sample_id in xrd_filename_mapping:
        xrd_file_path = xrd_filename_mapping[sample_id]
        if Path(xrd_file_path).exists():
            associated_files["xrd_gfrm"] = xrd_file_path
            associated_files["xrd"] = xrd_file_path.replace("000.gfrm", "merge.xy")

    return associated_files


def create_xrd_filename_mapping(base_dir: str) -> Dict[str, str]:
    """
    Create a mapping of sample ID to XRD filenames based on well plate positions.

    The well plate layout is 6 rows (A-F) × 4 columns (01-04) = 24 positions.
    Sample ID = (row_index * 4) + column_index + 1
    where row A=0, B=1, C=2, D=3, E=4, F=5 and column 01=0, 02=1, 03=2, 04=3

    Args:
        base_dir: Base directory containing XRD subdirectory

    Returns:
        Dictionary mapping sample ID (as string) to XRD file paths
    """
    base_path = Path(base_dir)
    xrd_dir = base_path / "XRD"
    mapping = {}

    if xrd_dir.exists():
        # Find all first XRD .gfrm files of each sample
        xrd_files = list(xrd_dir.rglob("*_000.gfrm"))

        for xrd_file in xrd_files:
            # Extract well position from filename (e.g., LPS_4_A03_merge.xy -> A03)
            if xrd_file.is_file():
                filename = xrd_file.name
                # Split by underscores and find the well position part
                parts = filename.split("_")
                well_pos = None
                for part in parts:
                    # Look for pattern like A03, B04, etc.
                    if len(part) == 3 and part[0].isalpha() and part[1:].isdigit():
                        well_pos = part
                        break

                if well_pos:
                    # Parse well position to get sample ID
                    row_letter = well_pos[0]
                    col_digits = well_pos[1:]

                    # Convert row letter to index (A=0, B=1, etc.)
                    row_index = ord(row_letter.upper()) - ord("A")

                    # Convert column digits to index (01=0, 02=1, etc.)
                    col_index = int(col_digits) - 1

                    # Calculate sample ID using well plate formula
                    sample_id = (row_index * 4) + col_index + 1

                    # Store mapping using sample ID as string key
                    mapping[str(sample_id)] = str(xrd_file)
                    print(
                        f"XRD mapping: {filename} -> {well_pos} -> sample {sample_id}"
                    )
                else:
                    print(f"Warning: Could not extract well position from {filename}")

    return mapping


def load_recipes_json(
    base_dir: str, h5_files: Optional[List[str]] = None
) -> Optional[Dict[int, Dict]]:
    """
    Load and validate recipes.json file from the base directory.
    Creates a mapping from sample ID to recipe parameters.

    Args:
        base_dir: Base directory containing recipes.json file
        h5_files: List of H5 file paths to extract sample IDs from (optional)

    Returns:
        Dictionary mapping sample ID to recipe parameters if successful, None if failed
    """
    base_path = Path(base_dir)

    # Find the first *recipes.json file in the base directory
    recipes_files = list(base_path.glob("*recipes.json"))

    if not recipes_files:
        print(f"Warning: No *recipes.json file found in {base_dir}")
        return None

    recipes_path = recipes_files[0]

    if len(recipes_files) > 1:
        print(
            f"Warning: Multiple *recipes.json files found in {base_dir}. Using: {recipes_path.name}"
        )

    try:
        with open(recipes_path, "r", encoding="utf-8") as f:
            recipes = json.load(f)

        if not isinstance(recipes, list):
            print(f"Error: recipes.json should contain a list, got {type(recipes)}")
            return None

        print(f"Successfully loaded recipes.json with {len(recipes)} entries")

        # Create mapping from sample ID (1-based) to recipe parameters
        recipe_mapping = {i: recipe for i, recipe in enumerate(recipes, start=1)}

        # If H5 files provided, check which sample IDs are present and missing
        if h5_files:
            present_sample_ids = set()
            for h5_file in h5_files:
                try:
                    h5_path = Path(h5_file)
                    *batch_name, sample_position, sample_id_str, run_string = (
                        h5_path.name.removesuffix("_spec_run.h5").split("_")
                    )
                    sample_id = int(sample_id_str)
                    present_sample_ids.add(sample_id)
                except (ValueError, IndexError) as e:
                    print(
                        f"Warning: Could not extract sample ID from {h5_path.name}: {e}"
                    )

            all_sample_ids = set(range(1, len(recipes) + 1))
            missing_sample_ids = sorted(all_sample_ids - present_sample_ids)
            present_sample_ids = sorted(present_sample_ids)

            print(f"Recipe mapping created for samples 1-{len(recipes)}")
            print(f"Present H5 files for samples: {present_sample_ids}")

            if missing_sample_ids:
                print(f"Missing H5 files for samples: {missing_sample_ids}")
                print("Note: Recipes for missing samples will be available but unused")

            print(
                f"Total recipes: {len(recipes)}, Present H5 files: {len(present_sample_ids)}"
            )

        return recipe_mapping

    except json.JSONDecodeError as e:
        print(f"Error parsing recipes.json: {e}")
        return None
    except Exception as e:
        print(f"Error reading recipes.json: {e}")
        return None


def process_cbox_data(h5_data: Dict) -> Dict:
    """
    Process CBox UV-Vis and PL data with individual and averaged spectra.

    Args:
        h5_data: Dictionary containing H5 file data

    Returns:
        Dictionary with processed CBox data
    """
    cbox_results = {}

    # Process UV-Vis data
    if "wl_data" in h5_data:
        try:
            uv_vis_results = process_uv_vis_data_raw(h5_data["wl_data"])

            cbox_results["uv_vis"] = {
                "wavelength": uv_vis_results["wavelength"],
                "energy": uv_vis_results["energy"],
                "absorption": uv_vis_results["absorption"],
                "absorption_avg": np.mean(uv_vis_results["absorption"], axis=0),
                "absorption_jacobian": uv_vis_results["absorption_jacobian"],
                "absorption_jacobian_avg": np.mean(
                    uv_vis_results["absorption_jacobian"], axis=0
                ),
                "transmission": uv_vis_results["transmission"],
                "transmission_avg": np.mean(uv_vis_results["transmission"], axis=0),
                "num_spectra": uv_vis_results["num_spectra"],
            }

        except Exception as e:
            raise Exception(f"Error processing UV-Vis data: {e}")

    # Process PL data
    if "pl_data" in h5_data:
        try:
            pl_results = process_pl_data_raw(h5_data["pl_data"])

            cbox_results["pl"] = {
                "wavelength": pl_results["wavelength"],
                "energy": pl_results["energy"],
                "intensity": pl_results["intensity"],
                "intensity_avg": np.mean(pl_results["intensity"], axis=0),
                "intensity_jacobian": pl_results["intensity_jacobian"],
                "intensity_jacobian_avg": np.mean(
                    pl_results["intensity_jacobian"], axis=0
                ),
                "num_spectra": pl_results["num_spectra"],
            }

        except Exception:
            # Continue without PL data if processing fails
            pass

    # Process photo data - prepare metadata but don't copy large image data
    if "photo" in h5_data:
        photo_data = h5_data["photo"]
        cbox_results["photo"] = {
            "exposure": photo_data["adj_photo_exposure"],
            "_link_image": False,
            "image": cv2.cvtColor(photo_data["adj_photo"], cv2.COLOR_BAYER_RG2BGR),
        }

    return cbox_results


def process_cd_data(cd_front_file: str, cd_back_file: str) -> Dict:
    """
    Process CD data from front and back measurement files.

    Args:
        cd_front_file: Path to front CD measurement file
        cd_back_file: Path to back CD measurement file

    Returns:
        Dictionary with processed CD data
    """
    # Load CD data files with correct format (4 columns: wavelength, CD, HT VOLTAGE, absorbance)
    # Skip the first 3 rows (header information)
    df_front = pd.read_csv(
        cd_front_file, sep="\t", skiprows=3, names=["wavelength", "CD", "HT", "abs"]
    )
    df_back = pd.read_csv(
        cd_back_file, sep="\t", skiprows=3, names=["wavelength", "CD", "HT", "abs"]
    )

    # Ensure same wavelength grid
    wavelength = df_front["wavelength"].values
    front_CD = df_front["CD"].values
    back_CD = df_back["CD"].values
    front_abs = df_front["abs"].values
    back_abs = df_back["abs"].values

    # Calculate derived quantities
    gen_CD = (front_CD + back_CD) / 2  # Genuine CD
    ldlb_CD = (front_CD - back_CD) / 2  # Linear dichroism + birefringence
    abs_avg = (front_abs + back_abs) / 2  # Average absorbance

    # Calculate g-factor: gen_CD / (abs_avg * 32980)
    # Handle division by zero by setting g_factor to 0 where abs_avg is 0
    g_factor = np.divide(
        gen_CD, abs_avg * 32980, out=np.zeros_like(gen_CD), where=(abs_avg != 0)
    )

    return {
        "wavelength": wavelength,
        "front_CD": front_CD,
        "back_CD": back_CD,
        "front_abs": front_abs,
        "back_abs": back_abs,
        "gen_CD": gen_CD,
        "ldlb_CD": ldlb_CD,
        "abs_avg": abs_avg,
        "g_factor": g_factor,
        "file_info": {"front_file": cd_front_file, "back_file": cd_back_file},
    }


def process_xrd_data(xrd_file: str) -> Dict:
    """
    Process XRD data from XY file.

    Args:
        xrd_file: Path to XRD .xy file

    Returns:
        Dictionary with processed XRD data
    """
    # Load XRD data
    df = pd.read_csv(xrd_file, sep="\t", comment="#", names=["2theta_Co", "int"])

    # Convert Co K-alpha to Cu K-alpha
    Co_K_alpha = 1.7902
    Cu_K_alpha = 1.5406
    df["2theta_Cu"] = df["2theta_Co"] * Cu_K_alpha / Co_K_alpha

    # Filter 2theta range
    df_filtered = df[(df["2theta_Cu"] > 4) & (df["2theta_Cu"] < 50.5)]

    return {
        "2theta_Cu": df_filtered["2theta_Cu"].values,
        "2theta_Co": df_filtered["2theta_Co"].values,
        "int": df_filtered["int"].values,
        "file_info": {"source_file": xrd_file},
    }


def process_xrd_gfrm_data(gfrm_file: str, poni_file: str) -> Dict:
    """
    Process XRD GFRM data from .gfrm file.

    Args:
        gfrm_file: Path to XRD .gfrm file
    Returns:
        Dictionary with processed XRD GFRM data
    """
    # Load GFRM data & AzimuthalIntegrator
    ai = pyFAI.load(poni_file)
    with fabio.open(gfrm_file) as gfrm:
        img_array = gfrm.data
        res = ai.integrate1d_ng(
            img_array,
            600,
            unit="2th_deg",
            radial_range=(5, 35),
            azimuth_range=(-18, 18),
            method=("full", "histogram", "cython"),
            filename=Path(gfrm_file).with_suffix(".dat").as_posix(),
        )

        if hasattr(gfrm, "header"):
            gfrm_headers = dict(gfrm.header)
        else:
            gfrm_headers = {}

    df = pd.DataFrame({"2theta_Co": res.radial, "int": res.intensity})
    # Convert Co K-alpha to Cu K-alpha
    Co_K_alpha = 1.7902
    Cu_K_alpha = 1.5406
    df["2theta_Cu"] = df["2theta_Co"] * Cu_K_alpha / Co_K_alpha
    # Filter 2theta range
    df_filtered = df[(df["2theta_Cu"] > 4) & (df["2theta_Cu"] < 50.5)]

    return {
        "2theta_Cu": df_filtered["2theta_Cu"].values,
        "2theta_Co": df_filtered["2theta_Co"].values,
        "int": df_filtered["int"].values,
        "file_info": {"source_file": gfrm_file, **gfrm_headers, "poni_file": poni_file},
    }


def save_results_to_h5(
    h5_file_path: str,
    output_h5_path: str,
    cbox_data: Dict,
    cd_data: Optional[Dict],
    xrd_data: Optional[Dict],
    process_params: Optional[Dict] = None,
):
    """
    Save processed results to a new H5 file in the 'results' group.

    Args:
        h5_file_path: Path to the original H5 file
        output_h5_path: Path where the processed H5 file will be saved
        cbox_data: Processed CBox data
        cd_data: Processed CD data (optional)
        xrd_data: Processed XRD data (optional)
        process_params: Process parameters from recipes.json (optional)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_h5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy original H5 file to output location
    import shutil

    shutil.copy2(h5_file_path, output_h5_path)

    # Open the copied file and add results
    with h5py.File(output_h5_path, "a") as hf:
        # Create or recreate results group
        if "results" in hf:
            del hf["results"]
        results_grp = hf.create_group("results")

        # Save CBox data
        cbox_grp = results_grp.create_group("cbox")

        # UV-Vis data
        if "uv_vis" in cbox_data:
            uv_vis_grp = cbox_grp.create_group("uv_vis")
            uv_vis = cbox_data["uv_vis"]
            for key, value in uv_vis.items():
                uv_vis_grp.create_dataset(key, data=value)

        # PL data
        if "pl" in cbox_data:
            pl_grp = cbox_grp.create_group("pl")
            pl = cbox_data["pl"]
            for key, value in pl.items():
                pl_grp.create_dataset(key, data=value)

        # Photo data - use softlinks to avoid redundancy
        if "photo" in cbox_data:
            photo_grp = cbox_grp.create_group("photo")
            photo = cbox_data["photo"]

            # Handle image data with softlinking
            if photo.get("_link_image", False):
                # Check if the original adj_photo exists in the file
                if "measurement/spec_run/adj_photo" in hf:
                    # Create a proper soft link instead of copying the data
                    photo_grp["image"] = h5py.SoftLink(
                        "/measurement/spec_run/adj_photo"
                    )
                    print(
                        "  Created softlink: /results/cbox/photo/image -> /measurement/spec_run/adj_photo"
                    )
                else:
                    print(
                        "  Warning: Original adj_photo not found, cannot create softlink"
                    )

            # Handle other photo data normally (exposure is small)
            for key, value in photo.items():
                if key not in ["_link_image"]:  # Skip internal flags
                    # Use chunks for large datasets
                    photo_grp.create_dataset(key, data=value, chunks=(key == "image"))

        # Save CD data
        if cd_data:
            cd_grp = results_grp.create_group("cd")
            for key, value in cd_data.items():
                if key == "file_info":
                    info_grp = cd_grp.create_group("file_info")
                    for info_key, info_value in value.items():
                        info_grp.create_dataset(
                            info_key, data=info_value.encode("utf-8")
                        )
                else:
                    cd_grp.create_dataset(key, data=value)

        # Save XRD data
        if xrd_data:
            if "xy_data" in xrd_data:
                xy_data = xrd_data.pop("xy_data")
                xy_data_grp = results_grp.create_group("xrd_xy")
                for key, value in xy_data.items():
                    if key == "file_info":
                        print(value)
                        xy_data_grp.attrs.update(value)
                    else:
                        xy_data_grp.create_dataset(key, data=value)

            xrd_grp = results_grp.create_group("xrd")
            for key, value in xrd_data.items():
                if key == "file_info":
                    xrd_grp.attrs.update(value)
                else:
                    xrd_grp.create_dataset(key, data=value)

        # Save process parameters at root level (not under results)
        if process_params:
            # Delete existing process_params group if it exists
            if "process_params" in hf:
                del hf["process_params"]
            process_grp = hf.create_group("process_params")

            process_params["config_hash"] = sha256(
                json.dumps(process_params, sort_keys=True).encode("utf-8")
            ).hexdigest()

            for key, value in process_params.items():
                process_grp.attrs[key] = value


def process_single_h5_file(
    h5_file_path: str,
    base_dir: str,
    xrd_filename_mapping: Optional[Dict[str, str]] = None,
    recipes: Optional[Dict[int, Dict]] = None,
    poni_file: Optional[str] = None,
    verbose: bool = True,
    associated_files: Optional[Dict[str, Optional[str]]] = None,
    process_params: Optional[Dict] = None,
) -> bool:
    """
    Process a single H5 file and integrate associated CD and XRD data.

    Processes CBox spectroscopy data, CD measurements, and XRD data from both XY files
    and GFRM (2D detector) files. When both XY and GFRM data are available, they are
    combined with GFRM as primary and XY nested under 'xy_data' key.

    Args:
        h5_file_path: Path to the H5 file
        base_dir: Base directory containing subdirectories (CBox, CD, XRD)
        xrd_filename_mapping: Pre-computed mapping of sample[:13] to XRD filenames
        recipes: Dictionary mapping sample ID to process parameters from recipes.json (optional)
        poni_file: Path to PONI file for GFRM processing (required when XRD GFRM files exist)
        verbose: Whether to print progress information
        associated_files: Dictionary with paths to associated files (optional, bypasses discovery)
        process_params: Dictionary with process parameters (optional, bypasses recipe lookup)

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        if verbose:
            print(f"Processing {Path(h5_file_path).name}")

        # Read H5 file
        if verbose:
            print("  Reading H5 file...")
        h5_data = read_h5_file(h5_file_path)
        if h5_data is None:
            if verbose:
                print(f"  Failed to read H5 file: {h5_file_path}")
            return False
        if verbose:
            print("  H5 file read successfully")

        # Create output path in preprocessed subdirectory
        h5_path = Path(h5_file_path)
        base_path = Path(base_dir)

        output_h5_path = Path(base_path, "preprocessed", h5_path.name)

        # Find associated files
        if associated_files is None:
            if verbose:
                print("  Finding associated files...")
            associated_files = find_associated_files(
                h5_file_path, base_dir, xrd_filename_mapping
            )

        if verbose:
            print(f"    Associated files: {associated_files}")

        # Process CBox data
        if verbose:
            print("  Processing CBox data...")
        cbox_data = process_cbox_data(h5_data)
        if verbose:
            print("    CBox data processed successfully")

        # Process CD data if available
        cd_data = None
        if associated_files["cd_front"] and associated_files["cd_back"]:
            try:
                if verbose:
                    print("  Processing CD data...")
                cd_data = process_cd_data(
                    associated_files["cd_front"], associated_files["cd_back"]
                )
                if verbose:
                    print("    CD data processed successfully")
            except Exception as e:
                if verbose:
                    print(f"    Failed to process CD data: {e}")

        # Process XRD data if available
        xrd_data = None
        xy_xrd_data = None
        gfrm_xrd_data = None
        if associated_files["xrd"]:
            try:
                if verbose:
                    print("  Processing XRD data...")
                xy_xrd_data = process_xrd_data(associated_files["xrd"])
                if verbose:
                    print("    XRD data processed successfully")
            except Exception as e:
                if verbose:
                    print(f"    Failed to process XRD data: {e}")

        if associated_files["xrd_gfrm"]:
            try:
                if verbose:
                    print("  Processing XRD GFRM data...")
                if not poni_file:
                    raise ValueError(
                        "PONI file is required for GFRM processing; pass --poni <path>."
                    )
                gfrm_xrd_data = process_xrd_gfrm_data(
                    associated_files["xrd_gfrm"], poni_file
                )
                if verbose:
                    print("    XRD GFRM data processed successfully")
            except Exception as e:
                if verbose:
                    print(f"    Failed to process XRD GFRM data: {e}")

        if xy_xrd_data and gfrm_xrd_data:
            xrd_data = {**gfrm_xrd_data, "xy_data": xy_xrd_data}
        elif xy_xrd_data:
            xrd_data = xy_xrd_data
        elif gfrm_xrd_data:
            xrd_data = gfrm_xrd_data

        # Extract process parameters for this file based on sample ID
        if process_params is None and recipes:
            # Extract sample ID from H5 filename (reuse the logic from find_associated_files)
            h5_path = Path(h5_file_path)

            try:
                *batch_name, sample_position, sample_id_str, run_string = (
                    h5_path.name.removesuffix("_spec_run.h5").split("_")
                )
            except ValueError as e:
                raise ValueError(
                    f"Could not extract sample ID from filename: {h5_path.stem}"
                ) from e

            # Convert sample ID to integer
            try:
                sample_id = int(sample_id_str)
            except ValueError as e:
                raise ValueError(
                    f"Could not convert sample ID '{sample_id_str}' to integer"
                ) from e

            # Look up recipe parameters using sample ID as key
            if sample_id in recipes:
                process_params = recipes[sample_id]
                if verbose:
                    print(f"    Using process parameters for sample ID {sample_id}")
            else:
                if verbose:
                    print(
                        f"    Warning: No recipe found for sample ID {sample_id}. Processing without recipe parameters."
                    )

        # Save results to new H5 file in preprocessed directory
        if verbose:
            print("  Saving results...")
        save_results_to_h5(
            h5_file_path,
            str(output_h5_path),
            cbox_data,
            cd_data,
            xrd_data,
            process_params,
        )

        if verbose:
            print(f"    Results saved to {Path(output_h5_path).name}")

        return True

    except Exception as e:
        if verbose:
            print(f"Error processing {Path(h5_file_path).name}: {e}")
            import traceback

            traceback.print_exc()
        return False


def process_single_h5_file_wrapper(h5_file_path: str, **kwargs) -> tuple[bool, str]:
    """
    Wrapper for process_single_h5_file that handles exceptions and logging for multiprocessing.
    Returns tuple of (success, log_message) for clean output collection.
    """
    try:
        # Disable verbose logging in workers to avoid output chaos
        kwargs["verbose"] = False

        success = process_single_h5_file(h5_file_path, **kwargs)

        if success:
            return True, f"✓ Completed: {Path(h5_file_path).name}"
        else:
            return False, f"✗ Failed: {Path(h5_file_path).name}"

    except Exception as e:
        return False, f"✗ Error processing {Path(h5_file_path).name}: {str(e)}"


def check_file_associations(
    base_dirs: List[str], verbose: bool = True
) -> Dict[str, Dict]:
    """
    Check file associations for all H5 files without processing them.

    Args:
        base_dirs: List of base directories containing CBox, CD, and XRD subdirectories
        verbose: Whether to print detailed output

    Returns:
        Dictionary with association results for each directory
    """
    results = {}

    for base_dir in base_dirs:
        print(f"\n=== Checking file associations for {base_dir} ===")

        # Create XRD filename mapping
        if verbose:
            print("Creating XRD filename mapping...")
        xrd_mapping = create_xrd_filename_mapping(base_dir)

        if verbose and xrd_mapping:
            print(f"Found {len(xrd_mapping)} XRD files:")
            for key, value in list(xrd_mapping.items())[:5]:  # Show first 5
                print(f"  {key} -> {Path(value).name}")
            if len(xrd_mapping) > 5:
                print(f"  ... and {len(xrd_mapping) - 5} more")

        # Find H5 files
        base_path = Path(base_dir)
        cbox_dir = base_path / "CBox"
        h5_files = []

        if cbox_dir.exists():
            h5_files = list(cbox_dir.rglob("*.h5"))
            print(f"Found {len(h5_files)} H5 files")
        else:
            print(f"Warning: CBox directory does not exist: {cbox_dir}")
            continue

        # Filter H5 files
        if h5_files:
            filtered_files_dict, ignored_files = filter_h5_files(
                [str(f) for f in h5_files]
            )
            filtered_h5_files = [
                info["file_path"] for info in filtered_files_dict.values()
            ]

            if ignored_files:
                print(
                    f"Ignored {len(ignored_files)} files: {[Path(f).name for f in ignored_files]}"
                )

            print(f"Processing {len(filtered_h5_files)} filtered H5 files")
        else:
            filtered_h5_files = []

        # Check associations for each H5 file
        dir_results = {
            "total_files": len(filtered_h5_files),
            "with_cd": 0,
            "with_xrd": 0,
            "with_both": 0,
            "with_neither": 0,
            "associations": [],
        }

        for i, h5_file in enumerate(filtered_h5_files):
            h5_name = Path(h5_file).name

            # Find associated files
            associated_files = find_associated_files(h5_file, base_dir, xrd_mapping)

            has_cd = associated_files["cd_front"] and associated_files["cd_back"]
            has_xrd = associated_files["xrd"] is not None

            if has_cd:
                dir_results["with_cd"] += 1
            if has_xrd:
                dir_results["with_xrd"] += 1
            if has_cd and has_xrd:
                dir_results["with_both"] += 1
            if not has_cd and not has_xrd:
                dir_results["with_neither"] += 1

            association_result = {
                "h5_file": h5_name,
                "cd_front": Path(associated_files["cd_front"]).name
                if associated_files["cd_front"]
                else None,
                "cd_back": Path(associated_files["cd_back"]).name
                if associated_files["cd_back"]
                else None,
                "xrd": Path(associated_files["xrd"]).name
                if associated_files["xrd"]
                else None,
                "has_cd": has_cd,
                "has_xrd": has_xrd,
            }
            dir_results["associations"].append(association_result)

            if verbose:
                status = []
                if has_cd:
                    status.append("CD")
                if has_xrd:
                    status.append("XRD")
                status_str = f"[{', '.join(status) if status else 'none'}]"
                print(f"  {i + 1:2d}. {h5_name} {status_str}")

                if verbose and (
                    associated_files["cd_front"]
                    or associated_files["cd_back"]
                    or associated_files["xrd"]
                ):
                    if associated_files["cd_front"]:
                        print(
                            f"      CD front: {Path(associated_files['cd_front']).name}"
                        )
                    if associated_files["cd_back"]:
                        print(
                            f"      CD back:  {Path(associated_files['cd_back']).name}"
                        )
                    if associated_files["xrd"]:
                        print(f"      XRD:      {Path(associated_files['xrd']).name}")
                    if associated_files["xrd_gfrm"]:
                        print(
                            f"      XRD GFRM: {Path(associated_files['xrd_gfrm']).name}"
                        )

        # Summary for this directory
        total = dir_results["total_files"]
        print(f"\n--- Summary for {base_dir} ---")
        print(f"Total H5 files: {total}")
        print(
            f"With CD data:   {dir_results['with_cd']} ({dir_results['with_cd'] / total * 100:.1f}%)"
        )
        print(
            f"With XRD data:  {dir_results['with_xrd']} ({dir_results['with_xrd'] / total * 100:.1f}%)"
        )
        print(
            f"With both:      {dir_results['with_both']} ({dir_results['with_both'] / total * 100:.1f}%)"
        )
        print(
            f"With neither:   {dir_results['with_neither']} ({dir_results['with_neither'] / total * 100:.1f}%)"
        )

        results[base_dir] = dir_results

    return results


def process_h5_files_batch(
    base_dirs: List[str],
    verbose: bool = True,
    ignore_duplicates: bool = True,
    max_files: Optional[int] = None,
    max_workers: Optional[int] = None,
    poni_file: Optional[str] = None,
) -> List[str]:
    """
    Process all H5 files found in the specified base directories.

    Args:
        base_dirs: List of base directories containing CBox, CD, and XRD subdirectories
        verbose: Whether to print progress information
        ignore_duplicates: Whether to ignore duplicate H5 files
        max_files: Maximum number of files to process (for testing)
        max_workers: Number of parallel workers (default: CPU count - 1, None = serial processing)
        poni_file: Path to PONI file for GFRM processing (required when XRD GFRM files are present)

    Returns:
        List of successfully processed H5 files
    """
    # Set up multiprocessing parameters
    if max_workers is None:
        # Default: use all cores minus 1, but at least 1
        max_workers = max(1, cpu_count() - 1)
    elif max_workers <= 0:
        # If 0 or negative, disable multiprocessing (serial processing)
        max_workers = 1

    use_multiprocessing = max_workers > 1

    if verbose:
        if use_multiprocessing:
            print(f"Using {max_workers} parallel workers for processing")
        else:
            print("Using serial processing (single worker)")
    # Create XRD filename mappings and load recipes for each directory
    xrd_mappings = {}
    recipes_by_dir = {}

    for base_dir in base_dirs:
        if verbose:
            print(f"Creating XRD filename mapping for {base_dir}...")
        xrd_mappings[base_dir] = create_xrd_filename_mapping(base_dir)

        # Find H5 files for this directory to get count and pass to recipe loading
        base_path = Path(base_dir)
        if base_path.exists():
            cbox_dir = base_path / "CBox"
            if cbox_dir.exists():
                h5_files = list(cbox_dir.rglob("*.h5"))
                filtered_files_dict_temp, _ = filter_h5_files(
                    [str(f) for f in h5_files]
                )
                h5_file_paths = [
                    info["file_path"] for info in filtered_files_dict_temp.values()
                ]

                if verbose:
                    print(
                        f"Loading recipes for {base_dir} (found {len(h5_file_paths)} H5 files)..."
                    )
                recipes_by_dir[base_dir] = load_recipes_json(base_dir, h5_file_paths)
            else:
                recipes_by_dir[base_dir] = None

    # Find all H5 files in CBox subdirectories
    all_h5_files = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: Base directory does not exist: {base_dir}")
            continue

        cbox_dir = base_path / "CBox"
        if cbox_dir.exists():
            h5_files = list(cbox_dir.rglob("*.h5"))
            if verbose:
                print(f"Found {len(h5_files)} H5 files in {cbox_dir}")
            all_h5_files.extend([str(f) for f in h5_files])
        else:
            print(f"Warning: CBox directory does not exist: {cbox_dir}")

    if ignore_duplicates:
        # Remove duplicates while preserving order
        seen = set()
        unique_h5_files = []
        for f in all_h5_files:
            if f not in seen:
                seen.add(f)
                unique_h5_files.append(f)
        all_h5_files = unique_h5_files

    # Filter H5 files using existing filter function
    filtered_files_dict, ignored_files = filter_h5_files(all_h5_files)

    # Extract file paths from the filtered dictionary
    filtered_h5_files = [info["file_path"] for info in filtered_files_dict.values()]

    # Limit number of files if max_files is specified
    if max_files is not None and max_files > 0:
        original_count = len(filtered_h5_files)
        filtered_h5_files = filtered_h5_files[:max_files]
        if verbose:
            print(
                f"Limiting to first {len(filtered_h5_files)} files out of {original_count} total files"
            )

    if verbose:
        print(f"Found {len(filtered_h5_files)} H5 files to process")

    # Process files grouped by directory to maintain proper indexing for recipes
    processed_files = []
    failed_files = []

    # Group files by their base directory
    files_by_dir = {}
    for h5_file in filtered_h5_files:
        h5_path = Path(h5_file)
        base_dir = None

        for bd in base_dirs:
            bd_path = Path(bd)
            try:
                h5_path.relative_to(bd_path)
                base_dir = bd
                break
            except ValueError:
                continue

        if base_dir is None:
            print(f"Warning: Could not determine base directory for {h5_file}")
            failed_files.append(h5_file)
            continue

        if base_dir not in files_by_dir:
            files_by_dir[base_dir] = []
        files_by_dir[base_dir].append(h5_file)

    # Process files for each directory
    for base_dir, dir_files in files_by_dir.items():
        if verbose:
            print(f"\nProcessing {len(dir_files)} files from {base_dir}")

        recipes = recipes_by_dir.get(base_dir)
        xrd_mapping = xrd_mappings.get(base_dir)

        if use_multiprocessing and len(dir_files) > 1:
            # Parallel processing with joblib - no misleading progress bar
            if verbose:
                print(f"Processing files in parallel with {max_workers} workers...")
                print(f"Processing {len(dir_files)} files...")

            # Use joblib.Parallel for clean parallel execution
            results = Parallel(
                n_jobs=max_workers,
                verbose=1 if verbose else 0,  # Use joblib's built-in progress reporting
                backend="loky",  # Best backend for CPU-bound tasks
            )(
                delayed(process_single_h5_file_wrapper)(
                    h5_file,
                    base_dir=base_dir,
                    xrd_filename_mapping=xrd_mapping,
                    recipes=recipes,
                    poni_file=poni_file,
                    verbose=False,  # Disable verbose in workers
                )
                for h5_file in dir_files
            )

            # Process results and show completion messages
            completed_count = 0
            for h5_file, (success, message) in zip(dir_files, results):
                completed_count += 1
                if success:
                    processed_files.append(h5_file)
                else:
                    failed_files.append(h5_file)

                if verbose:
                    progress = f"[{completed_count}/{len(dir_files)}]"
                    print(f"{progress} {message}")
        else:
            # Serial processing with detailed logging
            if verbose:
                print("Processing files sequentially...")
                print(f"Processing {len(dir_files)} files...")

            for file_index, h5_file in enumerate(dir_files):
                if verbose:
                    print(
                        f"\n[{file_index + 1}/{len(dir_files)}] Processing: {Path(h5_file).name}"
                    )

                success = process_single_h5_file(
                    h5_file,
                    base_dir,
                    xrd_filename_mapping=xrd_mapping,
                    recipes=recipes,
                    poni_file=poni_file,
                    verbose=verbose,
                )
                if success:
                    processed_files.append(h5_file)
                    if verbose:
                        print(f"✓ Completed: {Path(h5_file).name}")
                else:
                    failed_files.append(h5_file)
                    if verbose:
                        print(f"✗ Failed: {Path(h5_file).name}")

    if verbose:
        print("\nProcessing complete:")
        print(f"  Successfully processed: {len(processed_files)}")
        print(f"  Failed: {len(failed_files)}")
        if failed_files:
            print(f"  Failed files: {failed_files}")

    return processed_files


def main():
    """
    Main function to run the LPS preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="LPS Preprocessing Pipeline - Process H5 files and integrate CD/XRD data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure Expected:
  {dir_path}/
  ├── CBox/**/*.h5           # H5 files to process
  ├── CD/**/*.txt            # CD measurement files  
  ├── XRD/**/*_merge.xy      # XRD measurement files
  └── preprocessed/          # Output directory (created automatically)

Example usage:
  # Check file associations only (fast, good for debugging)
  python preprocessing.py "G:\\My Drive\\LPS\\LPS-1" --check-only
  
  # Process all files with parallel processing (default: CPU cores - 1)
  python preprocessing.py "G:\\My Drive\\LPS\\LPS-1"
  
  # Process with specific number of workers (uses joblib + progress bars)
  python preprocessing.py "G:\\My Drive\\LPS\\LPS-1" --workers 4
  
  # Serial processing with detailed logging (good for debugging)
  python preprocessing.py "G:\\My Drive\\LPS\\LPS-1" --workers 1
  
  # Process only first 3 files (useful for testing)
  python preprocessing.py "G:\\My Drive\\LPS\\LPS-1" --max-files 3 --workers 2
  
  # Process multiple directories with parallel processing
  python preprocessing.py "dir1" "dir2" --workers 6
        """,
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more base directories containing CBox, CD, and XRD subdirectories",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Disable verbose output"
    )

    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow processing of duplicate H5 files (default: ignore duplicates)",
    )

    parser.add_argument(
        "--check-only",
        "-c",
        action="store_true",
        help="Only check file associations without processing (useful for debugging and prerequisite testing)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit processing to first N files (useful for testing). Only applies when not using --check-only",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1, 0 or negative = serial processing)",
    )

    parser.add_argument(
        "--poni",
        type=str,
        default=None,
        help="Path to the PONI file for XRD GFRM data processing",
    )

    parser.add_argument(
        "--skip-gfrm", action="store_true", help="Skip processing of XRD GFRM files"
    )

    args = parser.parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet
    ignore_duplicates = not args.allow_duplicates

    # Validate directories
    valid_dirs = []
    for dir_path in args.directories:
        path = Path(dir_path)
        if not path.exists():
            print(f"Error: Directory does not exist: {dir_path}")
            sys.exit(1)
        if not path.is_dir():
            print(f"Error: Path is not a directory: {dir_path}")
            sys.exit(1)
        valid_dirs.append(str(path.resolve()))

    # try to find the poni file at ../*.poni if not provided
    if args.poni is None:
        parent_dir = Path(valid_dirs[0]).parent
        poni_files = list(parent_dir.glob("*.poni"))
        if poni_files:
            args.poni = str(poni_files[0].resolve())

    if args.poni and not Path(args.poni).exists():
        print(f"Error: PONI file does not exist: {args.poni}")
        sys.exit(1)

    if verbose:
        print(f"Processing directories: {valid_dirs}")
        print("Expected structure for each directory:")
        print("  CBox/**/*.h5           # H5 files to process")
        print("  CD/**/*.txt            # CD measurement files")
        print("  XRD/**/*_merge.xy      # XRD measurement files")
        print("  preprocessed/          # Output directory for processed H5 files")
        print()

    # Process H5 files or just check associations
    try:
        if args.check_only:
            print("=== FILE ASSOCIATION CHECK MODE ===")
            print("Checking file associations without processing...")
            print()

            check_results = check_file_associations(valid_dirs, verbose=verbose)

            # Overall summary
            total_files = sum(
                result["total_files"] for result in check_results.values()
            )
            total_with_cd = sum(result["with_cd"] for result in check_results.values())
            total_with_xrd = sum(
                result["with_xrd"] for result in check_results.values()
            )
            total_with_both = sum(
                result["with_both"] for result in check_results.values()
            )
            total_with_neither = sum(
                result["with_neither"] for result in check_results.values()
            )

            print("\n=== OVERALL SUMMARY ===")
            print(f"Total H5 files across all directories: {total_files}")
            print(
                f"With CD data:   {total_with_cd} ({total_with_cd / total_files * 100:.1f}%)"
            )
            print(
                f"With XRD data:  {total_with_xrd} ({total_with_xrd / total_files * 100:.1f}%)"
            )
            print(
                f"With both:      {total_with_both} ({total_with_both / total_files * 100:.1f}%)"
            )
            print(
                f"With neither:   {total_with_neither} ({total_with_neither / total_files * 100:.1f}%)"
            )

            if total_with_neither > 0:
                print(
                    f"\nNote: {total_with_neither} files have no associated CD or XRD data."
                )
                print(
                    "This is normal if not all samples have corresponding measurements."
                )

            if total_files > 0:
                print("\nFile association check completed successfully!")
                print("You can now run without --check-only to process the files.")
            else:
                print("No H5 files found to check.")

        else:
            processed_files = process_h5_files_batch(
                valid_dirs,
                verbose=verbose,
                ignore_duplicates=ignore_duplicates,
                max_files=args.max_files,
                max_workers=args.workers,
                poni_file=args.poni,
            )

            if verbose:
                print("\nFinal Summary:")
            print(f"Successfully processed {len(processed_files)} H5 files")

            if not processed_files:
                print("No files were processed. Please check:")
                print("1. H5 files exist in CBox subdirectories")
                print("2. H5 files match the expected naming format")
                print("3. Directory structure matches expected layout")
                print("4. Run with --check-only flag to verify file associations")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
