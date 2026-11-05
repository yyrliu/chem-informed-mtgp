"""
H5 File Analysis for UV-Vis and PL Spectroscopy

This module contains the essential functionality extracted from the Master_search notebooks for:
- Reading measurements from H5 files
- Data preprocessing
- Plotting UV-Vis and PL spectra
- Analysis of absorption, transmission, and photoluminescence data

Data source: CBox folder containing spec_run H5 files
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import re


def find_nearest(array, value):
    """Find the index of the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def index_of(arrval, value):
    """Return index of array *at or below* value."""
    if value < min(arrval):
        return 0
    return max(np.where(arrval <= value)[0])


def wavelength_to_energy(wavelength_nm):
    """Convert wavelength in nm to energy in eV."""
    energy_eV = 1240 / wavelength_nm
    return energy_eV


def apply_jacobian(signal, wavelength_nm):
    """Apply Jacobian transformation for energy scale conversion."""
    energy_eV = wavelength_to_energy(wavelength_nm)
    jacobian = (1240) / (energy_eV**2)
    transformed_signal = signal * jacobian
    return transformed_signal


def find_h5_files(base_dirs):
    """Find all H5 files in the CBox directory structures."""
    h5_files = []

    for base_dir in base_dirs:
        h5_files.extend(Path(base_dir).glob("**/*.h5"))

    return sorted(h5_files)


def filter_h5_files(h5_files):
    """Filter H5 files to keep only the highest run number for each sample."""
    filtered_files = {}
    ignored_files = []

    # Regex pattern to match the expected filename format
    # Pattern: XXX_run{n}_spec_run.h5
    # Groups: (sample_id)(run_number)
    pattern = r"^(.+)_run(\d+)_spec_run\.h5$"

    for file_path in h5_files:
        filename = Path(file_path).name

        # Use regex to match the expected pattern
        match = re.match(pattern, filename)

        if not match:
            ignored_files.append(
                {
                    "filename": filename,
                    "file_path": file_path,
                    "reason": "file_name_mismatch",
                    "message": "Does not follow pattern XXX_run{n}_spec_run.h5",
                }
            )
            continue

        # Extract sample identifier and run number from regex groups
        sample_id = match.group(1)  # Everything before _run
        run_number = int(match.group(2))  # The number after _run

        # Keep the file with highest run number for this sample
        if (
            sample_id not in filtered_files
            or run_number > filtered_files[sample_id]["run_number"]
        ):
            # If we're replacing a file, add the old one to ignored list
            if sample_id in filtered_files:
                old_file_path = Path(filtered_files[sample_id]["file_path"])
                old_filename = old_file_path.name
                ignored_files.append(
                    {
                        "filename": old_filename,
                        "file_path": old_file_path,
                        "reason": "duplicate_run_number",
                        "message": f"Lower run number ({filtered_files[sample_id]['run_number']}) than {filename} (run {run_number})",
                    }
                )

            filtered_files[sample_id] = {
                "file_path": file_path,
                "run_number": run_number,
            }
        else:
            ignored_files.append(
                {
                    "filename": filename,
                    "file_path": file_path,
                    "reason": "duplicate_run_number",
                    "message": f"Lower run number ({run_number}) than existing file with run {filtered_files[sample_id]['run_number']}",
                }
            )

    return filtered_files, ignored_files


def discover_h5_files(data_dirs, verbose=True, ignore_duplicates=True):
    """Discover and filter H5 files from the specified directories."""
    # Find all H5 files
    all_h5_files = find_h5_files(data_dirs)

    # Filter to keep only highest run number for each sample
    filtered_files, ignored_files = filter_h5_files(all_h5_files)

    # Extract the filtered file paths
    filtered_h5_files = [info["file_path"] for info in filtered_files.values()]

    if not ignore_duplicates:
        filtered_h5_files.extend(
            [
                file["file_path"]
                for file in ignored_files
                if file["reason"] == "duplicate_run_number"
            ]
        )
        ignored_files = [
            file for file in ignored_files if file["reason"] != "duplicate_run_number"
        ]

    if verbose:
        print(f"Found {len(filtered_h5_files)} H5 files after filtering:")

        if ignored_files:
            print(f"\nIgnored {len(ignored_files)} files:")
            for ignored_file in ignored_files:
                print(f"  - {ignored_file['filename']}: {ignored_file['message']}")
            print()

        for i, file in enumerate(filtered_h5_files[:10]):  # Show first 10
            print(f"{i + 1:>2}. {os.path.basename(file)}")
        if len(filtered_h5_files) > 10:
            print(f"... and {len(filtered_h5_files) - 10} more files")

    return filtered_h5_files


def read_h5_file(file_path):
    # file_name format:
    # f"{batch_name}_{self.settings['sample_position']}_{self.app.settings['sample'][0:13]}_run{n}_spec_run.h5"
    # batch_name format: f"{username}_{session_name}", could contain underscores (e.g. yrliu98_S-pMeMBAI-pre-3)
    *batch_name, sample_position, sample_name, run_string = (
        Path(file_path).name.removesuffix("_spec_run.h5").split("_")
    )
    batch_name = "_".join(batch_name)
    run = re.search(r"run(\d+)", run_string).group(0)

    # If sample_name is too short, it is not a uuid, so we use the file name stem as sample_id
    # Otherwise, we assume it is a uuid and use the sample_name directly
    sample_id = (
        Path(file_path).stem.removesuffix("_spec_run")
        if len(sample_name) < 13
        else sample_name
    )

    metadata = {
        "batch_name": batch_name,
        "sample_name": sample_name,
        "sample_position": sample_position,
        "sample_id": sample_id,
        "run": run,
    }

    """Read H5 file and extract spectroscopy data."""
    try:
        with h5py.File(file_path, "r") as hf:
            # UV-Vis/WL data
            wl_data = {
                "wl_dark_int_time": hf["measurement/spec_run/wl_dark_int_time"][()],
                "wl_dark_spectrum": hf["measurement/spec_run/wl_dark_spectrum"][:],
                "wl_samp_int_times": hf["measurement/spec_run/wl_spectra_int_times"][:],
                "wl_samp_spectrum": hf["measurement/spec_run/wl_spectra"][:],
                "wl_ref_int_time": hf["measurement/spec_run/wl_ref_int_time"][()],
                "wl_ref_spectrum": hf["measurement/spec_run/wl_ref_spectrum"][:],
                "wl_wave_axis": hf["measurement/spec_run/wl_wls"][:],
            }

            # PL data
            if "measurement/spec_run/pl_spectra" in hf:
                pl_data = {
                    "pl_dark_int_time": hf["measurement/spec_run/pl_dark_int_time"][()],
                    "pl_dark_spectrum": hf["measurement/spec_run/pl_dark_spectrum"][:],
                    "pl_spec_int_times": hf[
                        "measurement/spec_run/pl_spectra_int_times"
                    ][:],
                    "pl_spectra": hf["measurement/spec_run/pl_spectra"][:],
                    "pl_ref_int_time": hf["measurement/spec_run/pl_ref_int_time"][()],
                    "pl_ref_spectrum": hf["measurement/spec_run/pl_ref_spectrum"][:],
                    "pl_wls": hf["measurement/spec_run/pl_wls"][:],
                }
            else:
                pl_data = None

            if "measurement/spec_run/photo" in hf:
                photo = {
                    "photo": hf["measurement/spec_run/photo"][:],
                    "adj_photo": hf["measurement/spec_run/adj_photo"][:],
                    "dark_photo": hf["measurement/spec_run/dark_photo"][:],
                    "adj_photo_exposure": hf["measurement/spec_run/adj_photo_exposure"][
                        ()
                    ],
                }
            else:
                photo = None

            return {
                "metadata": metadata,
                "run": run,
                "photo": photo,
                "wl_data": wl_data,
                "pl_data": pl_data,
                "file_path": file_path,
            }

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_uv_vis_data(wl_data, wls_range=None):
    """Process UV-Vis spectroscopy data with averaging (wrapper for backward compatibility)."""
    # Get individual spectra
    raw_results = process_uv_vis_data_raw(wl_data, wls_range)

    # Calculate averages
    transmittance_avg = np.mean(raw_results["transmission"], axis=0)
    absorption_avg = np.mean(raw_results["absorption"], axis=0)
    absorption_jacobian_avg = np.mean(raw_results["absorption_jacobian"], axis=0)

    # Calculate transmittance metrics
    wl0, wl1 = 550, 800
    wavelength = raw_results["wavelength"]
    int_slice = (wavelength >= wl0) & (wavelength <= wl1)
    integrated_transmittance = np.sum(transmittance_avg[int_slice])
    mean_transmittance = np.mean(transmittance_avg[int_slice])

    return {
        "wavelength": raw_results["wavelength"],
        "energy": raw_results["energy"],
        "transmission": transmittance_avg,
        "absorption": absorption_avg,
        "absorption_jacobian": absorption_jacobian_avg,
        "mean_transmittance": mean_transmittance,
        "integrated_transmittance": integrated_transmittance,
    }


def process_uv_vis_data_raw(wl_data, wls_range=None):
    """Process UV-Vis spectroscopy data without averaging - returns individual spectra."""
    # Extract data
    wl_dark_int_time = wl_data["wl_dark_int_time"]
    wl_dark_spectrum = wl_data["wl_dark_spectrum"]
    wl_samp_int_times = wl_data["wl_samp_int_times"]
    wl_samp_spectrum = wl_data["wl_samp_spectrum"]
    wl_ref_int_time = wl_data["wl_ref_int_time"]
    wl_ref_spectrum = wl_data["wl_ref_spectrum"]
    wl_wave_axis = wl_data["wl_wave_axis"]

    if wls_range is None:
        wls_range = (min(wl_wave_axis), max(wl_wave_axis))

    # Remove sample spectra outside the specified wavelength range
    valid_indices = (wl_wave_axis >= wls_range[0]) & (wl_wave_axis <= wls_range[1])
    wl_wave_axis_filtered = wl_wave_axis[valid_indices]
    wl_dark_spectrum_filtered = wl_dark_spectrum[valid_indices]
    wl_samp_spectrum_filtered = wl_samp_spectrum[:, valid_indices]
    wl_ref_spectrum_filtered = wl_ref_spectrum[valid_indices]

    # Calculate individual transmitted intensities
    time_normalized_transmitted_intensity = (
        wl_samp_spectrum_filtered / wl_samp_int_times.reshape((-1, 1))
        - wl_dark_spectrum_filtered / wl_dark_int_time
    )

    time_normalized_reference = (
        wl_ref_spectrum_filtered / wl_ref_int_time
        - wl_dark_spectrum_filtered / wl_dark_int_time
    )

    # Individual transmittance (2D)
    transmittance_2d = time_normalized_transmitted_intensity / time_normalized_reference

    # Individual absorption (2D)
    absorption_2d = -np.log10(np.clip(transmittance_2d, 1e-4, None))

    # Individual absorption with Jacobian (2D)
    absorption_jacobian_2d = np.array(
        [
            apply_jacobian(absorption_2d[i], wl_wave_axis_filtered)
            for i in range(absorption_2d.shape[0])
        ]
    )

    # Energy scale
    wl_energy_axis = wavelength_to_energy(wl_wave_axis_filtered)

    return {
        "wavelength": wl_wave_axis_filtered,
        "energy": wl_energy_axis,
        "absorption": absorption_2d,
        "absorption_jacobian": absorption_jacobian_2d,
        "transmission": transmittance_2d,
        "num_spectra": absorption_2d.shape[0],
    }


def process_pl_data(pl_data, wls_range=None):
    """Process PL spectroscopy data with averaging (wrapper for backward compatibility)."""
    # Get individual spectra
    raw_results = process_pl_data_raw(pl_data, wls_range)

    # Calculate averages
    intensity_avg = np.mean(raw_results["intensity"], axis=0)
    intensity_jacobian_avg = np.mean(raw_results["intensity_jacobian"], axis=0)

    # Create processed_spectra list for backward compatibility
    processed_spectra = []
    for i in range(raw_results["num_spectra"]):
        processed_spectra.append(
            {
                "wavelength": raw_results["wavelength"],
                "energy": raw_results["energy"],
                "intensity": raw_results["intensity"][i],
                "intensity_jacobian": raw_results["intensity_jacobian"][i],
            }
        )

    # Create averaged spectrum
    average_spectrum = {
        "wavelength": raw_results["wavelength"],
        "energy": raw_results["energy"],
        "intensity": intensity_avg,
        "intensity_jacobian": intensity_jacobian_avg,
    }

    return {
        "full_wavelength": pl_data["pl_wls"],  # Full original wavelength axis
        "processed_spectra": processed_spectra,
        "num_spectra": raw_results["num_spectra"],
        "average_spectrum": average_spectrum,
    }


def process_pl_data_raw(pl_data, wls_range=None):
    """Process PL spectroscopy data without averaging - returns individual spectra."""
    # Extract data
    dark_int_time = pl_data["pl_dark_int_time"]
    dark_spectrum = pl_data["pl_dark_spectrum"]
    spec_int_times = pl_data["pl_spec_int_times"]
    spectra = pl_data["pl_spectra"]
    wls = pl_data["pl_wls"]

    if wls_range is None:
        wls_range = (min(wls), max(wls))

    # Select wavelength range for analysis
    lowQ_idx = find_nearest(wls, wls_range[0])
    highQ_idx = find_nearest(wls, wls_range[1])

    wavelength_filtered = wls[lowQ_idx:highQ_idx]
    dark_spectrum_filtered = dark_spectrum[lowQ_idx:highQ_idx]
    spectra_filtered = spectra[:, lowQ_idx:highQ_idx]

    # Process individual spectra
    individual_spectra = []
    individual_jacobian_spectra = []

    for i in range(spectra_filtered.shape[0]):
        # Time normalization
        time_normalized_intensity = np.divide(
            spectra_filtered[i], spec_int_times[i]
        ) - np.divide(dark_spectrum_filtered, dark_int_time)

        individual_spectra.append(time_normalized_intensity)

        # Convert to energy scale and apply Jacobian
        energies = wavelength_to_energy(wavelength_filtered)
        transformed_signal = apply_jacobian(
            time_normalized_intensity, wavelength_filtered
        )
        individual_jacobian_spectra.append(transformed_signal)

    # Convert to 2D arrays
    intensity_2d = np.array(individual_spectra)
    intensity_jacobian_2d = np.array(individual_jacobian_spectra)
    energies = wavelength_to_energy(wavelength_filtered)

    return {
        "wavelength": wavelength_filtered,
        "energy": energies,
        "intensity": intensity_2d,
        "intensity_jacobian": intensity_jacobian_2d,
        "num_spectra": intensity_2d.shape[0],
    }


def plot_uv_vis_spectra(uv_vis_data, sample_id, save_path=None):
    """Plot UV-Vis transmission and absorption spectra."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Transmission spectrum
    ax1.plot(uv_vis_data["wavelength"], uv_vis_data["transmission"], "b-", linewidth=2)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Transmission")
    ax1.set_title(f"Transmission Spectrum - {sample_id}")
    ax1.grid(True, alpha=0.3)

    # Absorption spectrum
    ax2.plot(uv_vis_data["wavelength"], uv_vis_data["absorption"], "r-", linewidth=2)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Absorption")
    ax2.set_title(f"Absorption Spectrum - {sample_id}")
    ax2.grid(True, alpha=0.3)

    # Absorption spectrum with Jacobian (energy scale)
    ax3.plot(
        uv_vis_data["energy"], uv_vis_data["absorption_jacobian"], "g-", linewidth=2
    )
    ax3.set_xlabel("Energy (eV)")
    ax3.set_ylabel("Absorption (Jacobian)")
    ax3.set_title(f"Absorption vs Energy - {sample_id}")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_pl_spectra(pl_data, sample_id, save_path=None):
    """Plot PL spectra."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, pl_data["num_spectra"]))

    # Wavelength scale
    for i, spectrum in enumerate(pl_data["processed_spectra"]):
        ax1.plot(
            spectrum["wavelength"],
            spectrum["intensity"],
            color=colors[i],
            alpha=0.7,
            label=f"Position {i + 1}",
        )

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("PL Intensity")
    ax1.set_title(f"PL Spectra - {sample_id}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Energy scale with Jacobian
    for i, spectrum in enumerate(pl_data["processed_spectra"]):
        ax2.plot(
            spectrum["energy"],
            spectrum["intensity_jacobian"],
            color=colors[i],
            alpha=0.7,
            label=f"Position {i + 1}",
        )

    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("PL Intensity (Jacobian)")
    ax2.set_title(f"PL Spectra vs Energy - {sample_id}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_sample_photo(photo, sample_id, save_path=None):
    """Plot sample photo."""
    plt.figure(figsize=(10, 8))

    # Handle both 2D and 3D photo data
    if photo.ndim == 3:
        # If 3D (color image), sum along color axis
        photo_display = photo.sum(axis=2)
    elif photo.ndim == 2:
        # If 2D (grayscale), use as is
        photo_display = photo
    else:
        print(f"Unexpected photo dimensions: {photo.ndim}")
        return

    plt.imshow(photo_display, cmap="gray")
    plt.colorbar()
    plt.title(f"Sample Photo - {sample_id}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def analyze_h5_file(
    file_path, plot_results=True, save_plots=False, uv_vis_range=None, pl_range=None
):
    """Complete analysis of a single H5 file."""
    print(f"Analyzing: {os.path.basename(file_path)}")

    # Read the file
    data = read_h5_file(file_path)
    if data is None:
        print("Failed to read file")
        return None

    # Process UV-Vis data
    uv_vis_results = process_uv_vis_data(data["wl_data"], wls_range=uv_vis_range)

    # Process PL data
    pl_results = process_pl_data(data["pl_data"], wls_range=pl_range)

    # Print key metrics
    print(f"Sample ID: {data['sample_id']}")
    print(
        f"Mean Transmittance (550-800 nm): {uv_vis_results['mean_transmittance']:.4f}"
    )
    print(f"Number of PL spectra: {pl_results['num_spectra']}")

    # Plot results if requested
    if plot_results:
        plot_uv_vis_spectra(uv_vis_results, data["sample_id"])
        plot_pl_spectra(pl_results, data["sample_id"])
        plot_sample_photo(data["photo"], data["sample_id"])

    return {
        "sample_id": data["sample_id"],
        "file_path": file_path,
        "uv_vis_results": uv_vis_results,
        "pl_results": pl_results,
        "photo": data["photo"],
    }


def analyze_multiple_files(file_list, max_files=5, uv_vis_range=None, pl_range=None):
    """Analyze multiple H5 files."""
    results = []

    for i, file_path in enumerate(file_list[:max_files]):
        print(f"\n--- Analysis {i + 1}/{min(len(file_list), max_files)} ---")
        result = analyze_h5_file(
            file_path, plot_results=False, uv_vis_range=uv_vis_range, pl_range=pl_range
        )
        if result:
            results.append(result)

    return results


def create_summary_dataframe(results):
    """Create a summary DataFrame from analysis results."""
    summary_data = []
    for result in results:
        summary_data.append(
            {
                "Sample ID": result["sample_id"],
                "File Name": os.path.basename(result["file_path"]),
                "Mean Transmittance": result["uv_vis_results"]["mean_transmittance"],
                "Integrated Transmittance": result["uv_vis_results"][
                    "integrated_transmittance"
                ],
                "Num PL Spectra": result["pl_results"]["num_spectra"],
            }
        )

    return pd.DataFrame(summary_data)


# Default data directories (can be overridden)
DEFAULT_DATA_DIRS = [r"g:\My Drive\LPS\20250709_S_MeOMBAI_prestudy_2\CBox"]


if __name__ == "__main__":
    # Example usage
    print("H5 Analysis Module")
    print("==================")

    # Discover files
    h5_files = discover_h5_files(DEFAULT_DATA_DIRS)

    if len(h5_files) > 0:
        print(f"\nFound {len(h5_files)} H5 files ready for analysis.")
        print("Example usage:")
        print("  from h5_analysis import analyze_h5_file, discover_h5_files")
        print("  files = discover_h5_files(['path/to/your/data'])")
        print("  result = analyze_h5_file(files[0])")
    else:
        print("No H5 files found in the default directories.")
        print(
            "Please update DEFAULT_DATA_DIRS or use discover_h5_files() with your data paths."
        )
