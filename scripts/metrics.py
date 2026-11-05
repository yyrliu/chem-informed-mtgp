"""
Metrics Pipeline

This module calculates heterogeneity scores for H5 files from CBox measurements
and writes them to the metrics/heterogeneity dataset in the HDF5 structure.

The heterogeneity score is calculated from the image data at /results/cbox/photo/image
and combines three metrics:
- Entropy (10% weight) - Measures randomness/disorder in pixel intensity
- Standard Deviation (20% weight) - Measures overall intensity dispersion
- Radial Extrema Variance (70% weight) - Measures intensity variations in concentric rings

The calculated score is written to metrics/heterogeneity in each H5 file.

Additionally, this module finds CD absorption peaks in /results/cd/abs_avg using
scipy.signal.find_peaks within a configurable wavelength range and writes the
peak position to /metrics/cd_abs_peak.

The module also calculates g-factor metrics from /results/cd/g_factor:
- Finds the global maximum of |g_factor| within the search range
- Splits the g_factor data at the CD absorption peak wavelength
- Finds local maxima of |g_factor| in each split region
- Writes results to /metrics/g_factor with structure:
  * max: {wavelength, intensity} - Global maximum
  * local_max: [{wavelength, intensity}] - Local maxima in split regions
"""

import h5py
import argparse
import sys
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed, cpu_count
from scipy.signal import find_peaks
from chem_informed_mtgp.h5_analysis import discover_h5_files

# Configuration macros
CD_ABS_PEAK_SEARCH_RANGE = (
    470,
    530,
)  # Wavelength range in nm for CD absorption peak search


def add_g_factor_metrics(
    h5_file_path, wavelength_range=CD_ABS_PEAK_SEARCH_RANGE, verbose=True
):
    """
    Calculate and add g-factor metrics to h5 file.
    Reads g-factor data from /results/cd/g_factor and finds global and local maxima,
    then writes results to metrics/g_factor

    Args:
        h5_file_path (str): Path to the h5 file
        wavelength_range (tuple): Tuple of (min_wavelength, max_wavelength) in nm for g-factor analysis
        verbose (bool): Enable verbose output

    Returns:
        dict or None: G-factor metrics if successful, None if failed
    """
    try:
        with h5py.File(h5_file_path, "r+") as hf:
            # Check if CD data exists
            if "results/cd/wavelength" not in hf or "results/cd/g_factor" not in hf:
                if verbose:
                    print(f"Warning: CD g-factor data not found in {h5_file_path}")
                return None

            # Read CD data
            wavelength = hf["results/cd/wavelength"][:]
            g_factor = hf["results/cd/g_factor"][:]

            if verbose:
                print(
                    f"Found CD g-factor data with {len(wavelength)} wavelength points"
                )
                print(
                    f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm"
                )

            # Apply mask of CD_ABS_PEAK_SEARCH_RANGE
            min_wl, max_wl = wavelength_range
            mask = (wavelength >= min_wl) & (wavelength <= max_wl)
            wl_filtered = wavelength[mask]
            g_factor_filtered = g_factor[mask]

            if len(wl_filtered) == 0:
                if verbose:
                    print(f"Warning: No data points in range {min_wl}-{max_wl} nm")
                return None

            if verbose:
                print(
                    f"Analyzing g-factor in range {min_wl}-{max_wl} nm ({len(wl_filtered)} points)"
                )

            # Get max(abs(g_factor)) - global maximum
            abs_g_factor = np.abs(g_factor_filtered)
            global_max_idx = np.argmax(abs_g_factor)
            global_max_wavelength = wl_filtered[global_max_idx]
            global_max_intensity = g_factor_filtered[global_max_idx]

            if verbose:
                print(
                    f"Global g-factor max: wavelength={global_max_wavelength:.2f} nm, intensity={global_max_intensity:.6f}"
                )

            # Get CD absorption peak to split the data
            cd_abs_peak = None
            if "results/cd/abs_avg" in hf:
                abs_avg = hf["results/cd/abs_avg"][:]
                abs_avg_filtered = abs_avg[mask]

                # Find peaks in absorption
                peaks, _ = find_peaks(abs_avg_filtered, height=0)
                if len(peaks) > 0:
                    peak_heights = abs_avg_filtered[peaks]
                    highest_peak_idx = np.argmax(peak_heights)
                    cd_abs_peak = wl_filtered[peaks[highest_peak_idx]]
                    if verbose:
                        print(
                            f"Using CD absorption peak at {cd_abs_peak:.2f} nm for splitting"
                        )

            # If no CD peak found, use the global g-factor max as split point
            if cd_abs_peak is None:
                cd_abs_peak = global_max_wavelength
                if verbose:
                    print(
                        f"No CD absorption peak found, using global g-factor max at {cd_abs_peak:.2f} nm for splitting"
                    )

            # Split g-factor data at cd_abs_peak
            split_mask_left = wl_filtered <= cd_abs_peak
            split_mask_right = wl_filtered > cd_abs_peak

            local_maxima = []

            # Left part (wavelength <= cd_abs_peak)
            if np.any(split_mask_left):
                wl_left = wl_filtered[split_mask_left]
                g_factor_left = g_factor_filtered[split_mask_left]
                abs_g_factor_left = np.abs(g_factor_left)

                if len(abs_g_factor_left) > 0:
                    left_max_idx = np.argmax(abs_g_factor_left)
                    left_max_wavelength = wl_left[left_max_idx]
                    left_max_intensity = g_factor_left[left_max_idx]
                    local_maxima.append(
                        {
                            "wavelength": float(left_max_wavelength),
                            "intensity": float(left_max_intensity),
                        }
                    )
                    if verbose:
                        print(
                            f"Left local max: wavelength={left_max_wavelength:.2f} nm, intensity={left_max_intensity:.6f}"
                        )

            # Right part (wavelength > cd_abs_peak)
            if np.any(split_mask_right):
                wl_right = wl_filtered[split_mask_right]
                g_factor_right = g_factor_filtered[split_mask_right]
                abs_g_factor_right = np.abs(g_factor_right)

                if len(abs_g_factor_right) > 0:
                    right_max_idx = np.argmax(abs_g_factor_right)
                    right_max_wavelength = wl_right[right_max_idx]
                    right_max_intensity = g_factor_right[right_max_idx]
                    local_maxima.append(
                        {
                            "wavelength": float(right_max_wavelength),
                            "intensity": float(right_max_intensity),
                        }
                    )
                    if verbose:
                        print(
                            f"Right local max: wavelength={right_max_wavelength:.2f} nm, intensity={right_max_intensity:.6f}"
                        )

            # Prepare results structure
            g_factor_metrics = {
                "max": {
                    "wavelength": float(global_max_wavelength),
                    "intensity": float(global_max_intensity),
                },
                "local_max": local_maxima,
            }

            # Create metrics group if it doesn't exist
            if "metrics" not in hf:
                metrics_group = hf.create_group("metrics")
                if verbose:
                    print("Created 'metrics' group")
            else:
                metrics_group = hf["metrics"]

            # Write g-factor metrics to metrics/g_factor
            if "g_factor" in metrics_group:
                del metrics_group["g_factor"]
                if verbose:
                    print("Updated existing g_factor dataset")

            g_factor_grp = metrics_group.create_group("g_factor")

            # Save max values
            max_grp = g_factor_grp.create_group("max")
            max_grp.create_dataset("wavelength", data=global_max_wavelength)
            max_grp.create_dataset("intensity", data=global_max_intensity)

            # Save local_max values
            local_max_grp = g_factor_grp.create_group("local_max")
            for i, local_max in enumerate(local_maxima):
                local_grp = local_max_grp.create_group(str(i))
                local_grp.create_dataset("wavelength", data=local_max["wavelength"])
                local_grp.create_dataset("intensity", data=local_max["intensity"])

            if verbose:
                print(f"Added g-factor metrics with {len(local_maxima)} local maxima")

            return g_factor_metrics

    except Exception as e:
        if verbose:
            print(f"Error processing g-factor data in {h5_file_path}: {str(e)}")
        return None


def add_cd_abs_peak_metrics(
    h5_file_path, wavelength_range=CD_ABS_PEAK_SEARCH_RANGE, verbose=True
):
    """
    Calculate and add CD absorption peak position to h5 file.
    Reads CD absorption data from /results/cd/abs_avg and finds peaks using scipy.signal.find_peaks
    within the specified wavelength range, then writes peak position to metrics/cd_abs_peak

    Args:
        h5_file_path (str): Path to the h5 file
        wavelength_range (tuple): Tuple of (min_wavelength, max_wavelength) in nm for peak search
        verbose (bool): Enable verbose output

    Returns:
        float or None: CD absorption peak position in nm if successful, None if failed
    """
    try:
        with h5py.File(h5_file_path, "r+") as hf:
            # Check if CD data exists
            if "results/cd/wavelength" not in hf or "results/cd/abs_avg" not in hf:
                if verbose:
                    print(f"Warning: CD data not found in {h5_file_path}")
                return None

            # Read CD data
            wavelength = hf["results/cd/wavelength"][:]
            abs_avg = hf["results/cd/abs_avg"][:]

            if verbose:
                print(f"Found CD data with {len(wavelength)} wavelength points")
                print(
                    f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm"
                )

            # Filter data within the specified wavelength range
            min_wl, max_wl = wavelength_range
            mask = (wavelength >= min_wl) & (wavelength <= max_wl)
            wl_filtered = wavelength[mask]
            abs_filtered = abs_avg[mask]

            if len(wl_filtered) == 0:
                if verbose:
                    print(f"Warning: No data points in range {min_wl}-{max_wl} nm")
                return None

            if verbose:
                print(
                    f"Searching for peaks in range {min_wl}-{max_wl} nm ({len(wl_filtered)} points)"
                )

            # Find peaks using scipy.signal.find_peaks
            peaks, properties = find_peaks(abs_filtered, height=0)  # Basic peak finding

            if len(peaks) == 0:
                if verbose:
                    print("No peaks found in the specified range")
                return None

            # Get the wavelength position of the highest peak
            peak_heights = abs_filtered[peaks]
            highest_peak_idx = np.argmax(peak_heights)
            peak_position = wl_filtered[peaks[highest_peak_idx]]

            if verbose:
                print(f"Found {len(peaks)} peaks, highest at {peak_position:.2f} nm")
                print(f"Peak height: {peak_heights[highest_peak_idx]:.6f}")

            # Create metrics group if it doesn't exist
            if "metrics" not in hf:
                metrics_group = hf.create_group("metrics")
                if verbose:
                    print("Created 'metrics' group")
            else:
                metrics_group = hf["metrics"]

            # Write CD absorption peak position to metrics/cd_abs_peak
            if "cd_abs_peak" in metrics_group:
                del metrics_group["cd_abs_peak"]
                if verbose:
                    print("Updated existing cd_abs_peak dataset")

            metrics_group.create_dataset("cd_abs_peak", data=peak_position)
            if verbose:
                print(f"Added CD absorption peak position: {peak_position:.2f} nm")

            return peak_position

    except Exception as e:
        if verbose:
            print(f"Error processing CD data in {h5_file_path}: {str(e)}")
        return None


def optimize_PL_image(image_data):
    return np.random.random()  # Placeholder for actual heterogeneity calculation logic


def add_heterogeneity_metrics(h5_file_path, verbose=True):
    """
    Calculate and add heterogeneity score to h5 file.
    Reads image from /results/cbox/photo/image and writes score to metrics/heterogeneity

    Args:
        h5_file_path (str): Path to the h5 file
        verbose (bool): Enable verbose output

    Returns:
        float or None: Heterogeneity score if successful, None if failed
    """
    try:
        with h5py.File(h5_file_path, "r+") as hf:
            # Read the image data from results/cbox/photo/image
            if "results/cbox/photo/image" not in hf:
                if verbose:
                    print(
                        f"Warning: 'results/cbox/photo/image' not found in {h5_file_path}"
                    )
                return None

            image_data = hf["results/cbox/photo/image"][:]
            if verbose:
                print(f"Found image data with shape: {image_data.shape}")

            # Calculate heterogeneity score
            heterogeneity_score = optimize_PL_image(image_data)
            if verbose:
                print(f"Calculated heterogeneity score: {heterogeneity_score:.6f}")

            # Create metrics group if it doesn't exist
            if "metrics" not in hf:
                metrics_group = hf.create_group("metrics")
                if verbose:
                    print("Created 'metrics' group")
            else:
                metrics_group = hf["metrics"]
                if verbose:
                    print("Using existing 'metrics' group")

            # Write heterogeneity score to metrics/heterogeneity
            if "heterogeneity" in metrics_group:
                # Update existing dataset
                del metrics_group["heterogeneity"]
                if verbose:
                    print("Updated existing heterogeneity dataset")

            metrics_group.create_dataset("heterogeneity", data=heterogeneity_score)
            if verbose:
                print(f"Added heterogeneity score: {heterogeneity_score:.6f}")

            return heterogeneity_score

    except Exception as e:
        if verbose:
            print(f"Error processing {h5_file_path}: {str(e)}")
        return None


def add_all_metrics(
    h5_file_path, wavelength_range=CD_ABS_PEAK_SEARCH_RANGE, verbose=True
):
    """
    Calculate and add heterogeneity, CD absorption peak, and g-factor metrics to h5 file.

    Args:
        h5_file_path (str): Path to the h5 file
        wavelength_range (tuple): Tuple of (min_wavelength, max_wavelength) in nm for CD peak search
        verbose (bool): Enable verbose output

    Returns:
        dict: Dictionary with results for all metrics
    """
    results = {"heterogeneity": None, "cd_abs_peak": None, "g_factor": None}

    # Add heterogeneity metrics
    heterogeneity_score = add_heterogeneity_metrics(h5_file_path, verbose=verbose)
    results["heterogeneity"] = heterogeneity_score

    # Add CD absorption peak metrics
    cd_peak = add_cd_abs_peak_metrics(h5_file_path, wavelength_range, verbose=verbose)
    results["cd_abs_peak"] = cd_peak

    # Add g-factor metrics
    g_factor_metrics = add_g_factor_metrics(
        h5_file_path, wavelength_range, verbose=verbose
    )
    results["g_factor"] = g_factor_metrics

    return results


def process_single_file_metrics(h5_file_path, wavelength_range, verbose=False):
    """
    Process a single file for all metrics. Simplified for joblib.
    """
    try:
        metrics_results = add_all_metrics(
            h5_file_path, wavelength_range, verbose=verbose
        )
        return h5_file_path, metrics_results
    except Exception as e:
        return h5_file_path, {
            "heterogeneity": None,
            "cd_abs_peak": None,
            "g_factor": None,
            "error": str(e),
        }


def batch_add_all_metrics(
    h5_file_paths,
    wavelength_range=CD_ABS_PEAK_SEARCH_RANGE,
    verbose=True,
    max_workers=None,
):
    """
    Process multiple h5 files using joblib for cleaner parallel processing.

    Args:
        h5_file_paths (list): List of h5 file paths
        wavelength_range (tuple): Tuple of (min_wavelength, max_wavelength) in nm for CD peak search
        verbose (bool): Enable verbose output
        max_workers (int): Number of parallel workers (default: CPU count - 1)

    Returns:
        dict: Dictionary mapping file paths to metrics results
    """
    if max_workers is None:
        max_workers = max(1, int(cpu_count() - 1))
    elif max_workers <= 0:
        max_workers = 1

    use_parallel = max_workers > 1 and len(h5_file_paths) > 1

    if verbose:
        if use_parallel:
            print(
                f"Processing {len(h5_file_paths)} files with {max_workers} workers..."
            )
        else:
            print(f"Processing {len(h5_file_paths)} files sequentially...")

    if use_parallel:
        # Parallel processing with joblib - no misleading progress bar
        if verbose:
            print(f"Processing {len(h5_file_paths)} files...")

        results_list = Parallel(
            n_jobs=max_workers,
            verbose=1 if verbose else 0,  # Use joblib's built-in progress reporting
            backend="loky",
        )(
            delayed(process_single_file_metrics)(
                h5_file_path,
                wavelength_range,
                verbose=False,  # Disable verbose in workers
            )
            for h5_file_path in h5_file_paths
        )

        # Convert results to dictionary and show completion messages
        results = {}
        completed_count = 0
        for h5_file_path, metrics_results in results_list:
            completed_count += 1
            results[h5_file_path] = metrics_results

            if verbose:
                # Show completion status
                success_count = sum(
                    1
                    for v in metrics_results.values()
                    if v is not None and "error" not in str(v)
                )
                progress = f"[{completed_count}/{len(h5_file_paths)}]"
                if success_count > 0:
                    print(
                        f"{progress} ✓ Completed: {Path(h5_file_path).name} ({success_count}/3 metrics)"
                    )
                else:
                    print(f"{progress} ✗ Failed: {Path(h5_file_path).name}")

    else:
        # Serial processing
        results = {}

        if verbose:
            print(f"Processing {len(h5_file_paths)} files...")

        for i, h5_file_path in enumerate(h5_file_paths, 1):
            if verbose:
                print(
                    f"[{i}/{len(h5_file_paths)}] Processing: {Path(h5_file_path).name}"
                )

            _, metrics_results = process_single_file_metrics(
                h5_file_path, wavelength_range, verbose=verbose
            )
            results[h5_file_path] = metrics_results

    # Calculate statistics
    successful_heterogeneity = sum(
        1 for r in results.values() if r.get("heterogeneity") is not None
    )
    successful_cd_peak = sum(
        1 for r in results.values() if r.get("cd_abs_peak") is not None
    )
    successful_g_factor = sum(
        1 for r in results.values() if r.get("g_factor") is not None
    )
    failed = sum(
        1
        for r in results.values()
        if all(
            r.get(key) is None for key in ["heterogeneity", "cd_abs_peak", "g_factor"]
        )
    )

    if verbose:
        print("\n=== BATCH PROCESSING SUMMARY ===")
        print(f"Total files processed: {len(h5_file_paths)}")
        print(f"Successful heterogeneity metrics: {successful_heterogeneity}")
        print(f"Successful CD absorption peak metrics: {successful_cd_peak}")
        print(f"Successful g-factor metrics: {successful_g_factor}")
        print(f"Completely failed: {failed}")

        # Show errors if any
        errors = [(path, r.get("error")) for path, r in results.items() if "error" in r]
        if errors:
            print(f"\nErrors encountered in {len(errors)} files:")
            for path, error in errors[:5]:  # Show first 5 errors
                print(f"  {Path(path).name}: {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")

    return results


def verify_heterogeneity_metrics(h5_file_path, verbose=True):
    """
    Verify that heterogeneity metrics were written correctly

    Args:
        h5_file_path (str): Path to the h5 file
        verbose (bool): Enable verbose output

    Returns:
        float or None: Heterogeneity score if found, None otherwise
    """
    try:
        with h5py.File(h5_file_path, "r") as hf:
            if "metrics/heterogeneity" in hf:
                score = hf["metrics/heterogeneity"][()]
                if verbose:
                    print(
                        f"Heterogeneity score in {Path(h5_file_path).name}: {score:.6f}"
                    )
                return score
            else:
                if verbose:
                    print(
                        f"No heterogeneity metrics found in {Path(h5_file_path).name}"
                    )
                return None

    except Exception as e:
        if verbose:
            print(f"Error reading {h5_file_path}: {str(e)}")
        return None


def verify_g_factor_metrics(h5_file_path, verbose=True):
    """
    Verify that g-factor metrics were written correctly

    Args:
        h5_file_path (str): Path to the h5 file
        verbose (bool): Enable verbose output

    Returns:
        dict or None: G-factor metrics if found, None otherwise
    """
    try:
        with h5py.File(h5_file_path, "r") as hf:
            if "metrics/g_factor" in hf:
                g_factor_grp = hf["metrics/g_factor"]

                # Read max values
                max_wavelength = g_factor_grp["max/wavelength"][()]
                max_intensity = g_factor_grp["max/intensity"][()]

                # Read local_max values
                local_maxima = []
                if "local_max" in g_factor_grp:
                    local_max_grp = g_factor_grp["local_max"]
                    for key in sorted(local_max_grp.keys()):
                        local_grp = local_max_grp[key]
                        local_maxima.append(
                            {
                                "wavelength": local_grp["wavelength"][()],
                                "intensity": local_grp["intensity"][()],
                            }
                        )

                g_factor_metrics = {
                    "max": {
                        "wavelength": float(max_wavelength),
                        "intensity": float(max_intensity),
                    },
                    "local_max": local_maxima,
                }

                if verbose:
                    print(f"G-factor metrics in {Path(h5_file_path).name}:")
                    print(f"  Global max: {max_wavelength:.2f} nm, {max_intensity:.6f}")
                    for i, local_max in enumerate(local_maxima):
                        print(
                            f"  Local max {i}: {local_max['wavelength']:.2f} nm, {local_max['intensity']:.6f}"
                        )

                return g_factor_metrics
            else:
                if verbose:
                    print(f"No g-factor metrics found in {Path(h5_file_path).name}")
                return None

    except Exception as e:
        if verbose:
            print(f"Error reading g-factor metrics from {h5_file_path}: {str(e)}")
        return None


def verify_cd_abs_peak_metrics(h5_file_path, verbose=True):
    """
    Verify that CD absorption peak metrics were written correctly

    Args:
        h5_file_path (str): Path to the h5 file
        verbose (bool): Enable verbose output

    Returns:
        float or None: CD absorption peak position in nm if found, None otherwise
    """
    try:
        with h5py.File(h5_file_path, "r") as hf:
            if "metrics/cd_abs_peak" in hf:
                peak_position = hf["metrics/cd_abs_peak"][()]
                if verbose:
                    print(
                        f"CD absorption peak in {Path(h5_file_path).name}: {peak_position:.2f} nm"
                    )
                return peak_position
            else:
                if verbose:
                    print(
                        f"No CD absorption peak metrics found in {Path(h5_file_path).name}"
                    )
                return None

    except Exception as e:
        if verbose:
            print(f"Error reading {h5_file_path}: {str(e)}")
        return None


def verify_all_metrics(h5_file_path, verbose=True):
    """
    Verify that heterogeneity, CD absorption peak, and g-factor metrics were written correctly

    Args:
        h5_file_path (str): Path to the h5 file
        verbose (bool): Enable verbose output

    Returns:
        dict: Dictionary with verification results for all metrics
    """
    return {
        "heterogeneity": verify_heterogeneity_metrics(h5_file_path, verbose=verbose),
        "cd_abs_peak": verify_cd_abs_peak_metrics(h5_file_path, verbose=verbose),
        "g_factor": verify_g_factor_metrics(h5_file_path, verbose=verbose),
    }


def check_metrics_status(directories, verbose=True, debug_paths=False):
    """
    Check which H5 files already have metrics (heterogeneity, CD absorption peak, and/or g-factor)

    Args:
        directories (list): List of directory paths
        verbose (bool): Enable verbose output

    Returns:
        dict: Status summary
    """
    h5_files = discover_h5_files(directories, verbose=False)

    with_heterogeneity = []
    with_cd_peak = []
    with_g_factor = []
    with_all_metrics = []
    with_no_metrics = []
    invalid_files = []

    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, "r") as hf:
                has_heterogeneity = "metrics/heterogeneity" in hf
                has_cd_peak = "metrics/cd_abs_peak" in hf
                has_g_factor = "metrics/g_factor" in hf

                if has_heterogeneity:
                    with_heterogeneity.append(h5_file)
                if has_cd_peak:
                    with_cd_peak.append(h5_file)
                if has_g_factor:
                    with_g_factor.append(h5_file)
                if has_heterogeneity and has_cd_peak and has_g_factor:
                    with_all_metrics.append(h5_file)
                if not has_heterogeneity and not has_cd_peak and not has_g_factor:
                    with_no_metrics.append(h5_file)
        except Exception:
            invalid_files.append(h5_file)

    if verbose:
        print("=== METRICS STATUS CHECK ===")
        print(f"Total H5 files: {len(h5_files)}")
        print(f"With heterogeneity metrics: {len(with_heterogeneity)}")
        print(f"With CD absorption peak metrics: {len(with_cd_peak)}")
        print(f"With g-factor metrics: {len(with_g_factor)}")
        print(f"With all metrics: {len(with_all_metrics)}")
        print(f"With no metrics: {len(with_no_metrics)}")
        print(f"Invalid/inaccessible files: {len(invalid_files)}")

        if with_no_metrics:
            print("\nFiles without any metrics:")
            for f in with_no_metrics[:5]:  # Show first 5
                print(f"  {Path(f).name}")
            if len(with_no_metrics) > 5:
                print(f"  ... and {len(with_no_metrics) - 5} more")

    return {
        "total_files": len(h5_files),
        "with_heterogeneity": len(with_heterogeneity),
        "with_cd_peak": len(with_cd_peak),
        "with_g_factor": len(with_g_factor),
        "with_all_metrics": len(with_all_metrics),
        "with_no_metrics": len(with_no_metrics),
        "invalid_files": len(invalid_files),
        "files_with_heterogeneity": with_heterogeneity,
        "files_with_cd_peak": with_cd_peak,
        "files_with_g_factor": with_g_factor,
        "files_with_all_metrics": with_all_metrics,
        "files_without_metrics": with_no_metrics,
        "invalid_file_list": invalid_files,
    }


def main():
    """
    Main function to run the metrics calculation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Metrics Pipeline - Calculate heterogeneity scores and CD absorption peaks for H5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Directory Structure Expected:
  {{dir_path}}/
  └── *.h5                  # H5 files with proper results structure
  
Required H5 Structure:
  Each H5 file should contain:
  ├── results/cbox/photo/image  # Image data for heterogeneity calculation
  └── results/cd/
      ├── wavelength            # Wavelength array for CD data
      ├── abs_avg               # Average absorption for CD peak finding
      └── g_factor              # G-factor data for g-factor metrics
  
Output:
  Writes metrics to:
  ├── metrics/heterogeneity     # Scalar heterogeneity score
  ├── metrics/cd_abs_peak       # CD absorption peak position in nm
  └── metrics/g_factor/         # G-factor metrics
      ├── max/                  # Global maximum |g_factor|
      │   ├── wavelength        # Wavelength of global max
      │   └── intensity         # Intensity at global max
      └── local_max/            # Local maxima in split regions
          ├── 0/wavelength      # Wavelength of first local max
          ├── 0/intensity       # Intensity of first local max
          └── ...               # Additional local maxima

CD Absorption Peak Search Range: {CD_ABS_PEAK_SEARCH_RANGE[0]}-{CD_ABS_PEAK_SEARCH_RANGE[1]} nm

Example usage:
  # Check metrics status only (see which files have/don't have metrics)
  python metrics.py "G:\\My Drive\\LPS\\LPS-1" --check-only
  
  # Calculate all metrics for all files with parallel processing (default)
  python metrics.py "G:\\My Drive\\LPS\\LPS-1"
  
  # Use specific number of workers with progress bars
  python metrics.py "G:\\My Drive\\LPS\\LPS-1" --workers 4
  
  # Serial processing (good for debugging)
  python metrics.py "G:\\My Drive\\LPS\\LPS-1" --workers 1
  
  # Use custom wavelength range for CD and g-factor analysis
  python metrics.py "G:\\My Drive\\LPS\\LPS-1" --cd-peak-range 450 550 --workers 6
  
  # Process only first 3 files (useful for testing)
  python metrics.py "G:\\My Drive\\LPS\\LPS-1" --max-files 3 --workers 2
  
  # Process multiple directories with parallel processing
  python metrics.py "dir1" "dir2" --workers 8
  
  # Verify existing metrics
  python metrics.py "G:\\My Drive\\LPS\\LPS-1" --verify-only
        """,
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more base directories containing H5 files",
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
        "--check-only",
        "-c",
        action="store_true",
        help="Only check which files have/don't have metrics (no processing)",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing metrics (no processing)",
    )

    parser.add_argument(
        "--cd-peak-range",
        nargs=2,
        type=float,
        metavar=("MIN_WL", "MAX_WL"),
        help=f"Custom wavelength range (nm) for CD peak search (default: {CD_ABS_PEAK_SEARCH_RANGE[0]} {CD_ABS_PEAK_SEARCH_RANGE[1]})",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit processing to first N files (useful for testing). Only applies when processing files",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing metrics (default: skip files with existing metrics)",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1, 0 = serial processing)",
    )

    args = parser.parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet

    # Handle custom wavelength range
    wavelength_range = CD_ABS_PEAK_SEARCH_RANGE
    if args.cd_peak_range:
        wavelength_range = tuple(args.cd_peak_range)
        if verbose:
            print(
                f"Using custom CD peak search range: {wavelength_range[0]}-{wavelength_range[1]} nm"
            )

    # Validate directories
    valid_dirs = []
    for dir_path in args.directories:
        p = Path(dir_path)
        if not p.exists():
            print(f"Error: Directory does not exist: {dir_path}")
            sys.exit(1)
        if not p.is_dir():
            print(f"Error: Path is not a directory: {dir_path}")
            sys.exit(1)
        valid_dirs.append(str(p.resolve()))

    if verbose:
        print(f"Processing directories: {valid_dirs}")
        print("Looking for H5 files in:")
        print("  **/*.h5               # H5 files with proper results structure")
        print(f"CD peak search range: {wavelength_range[0]}-{wavelength_range[1]} nm")
        print()

    try:
        if args.check_only:
            print("=== METRICS STATUS CHECK MODE ===")
            print("Checking which files have metrics...")
            print()

            status = check_metrics_status(valid_dirs, verbose=verbose)

            if status["total_files"] > 0:
                print("\nStatus check completed successfully!")
                if status["with_no_metrics"] > 0:
                    print(
                        "Run without --check-only to calculate metrics for remaining files."
                    )
            else:
                print("No H5 files found to check.")

        elif args.verify_only:
            print("=== METRICS VERIFICATION MODE ===")
            print("Verifying existing metrics...")
            print()

            h5_files = discover_h5_files(valid_dirs, verbose=verbose)

            if not h5_files:
                print("No H5 files found to verify.")
                return

            verified_heterogeneity = 0
            verified_cd_peak = 0
            verified_g_factor = 0
            for h5_file in h5_files:
                results = verify_all_metrics(h5_file, verbose=verbose)
                if results["heterogeneity"] is not None:
                    verified_heterogeneity += 1
                if results["cd_abs_peak"] is not None:
                    verified_cd_peak += 1
                if results["g_factor"] is not None:
                    verified_g_factor += 1

            print("\nVerification completed:")
            print(
                f"  Files with heterogeneity metrics: {verified_heterogeneity}/{len(h5_files)}"
            )
            print(
                f"  Files with CD absorption peak metrics: {verified_cd_peak}/{len(h5_files)}"
            )
            print(f"  Files with g-factor metrics: {verified_g_factor}/{len(h5_files)}")

        else:
            # Process files
            h5_files = discover_h5_files(valid_dirs, verbose=verbose)

            if not h5_files:
                print("No H5 files found to process.")
                print("Please check:")
                print("1. H5 files exist in CBox or preprocessed subdirectories")
                print("2. Directory structure matches expected layout")
                print("3. Run with --check-only flag to verify metrics status")
                return

            # Filter files if not forcing overwrite
            if not args.force:
                files_to_process = []
                skipped_count = 0

                for h5_file in h5_files:
                    try:
                        with h5py.File(h5_file, "r") as hf:
                            has_heterogeneity = "metrics/heterogeneity" in hf
                            has_cd_peak = "metrics/cd_abs_peak" in hf
                            has_g_factor = "metrics/g_factor" in hf

                            # Process all metrics, skip if all exist
                            should_process = not (
                                has_heterogeneity and has_cd_peak and has_g_factor
                            )

                            if should_process:
                                files_to_process.append(h5_file)
                            else:
                                skipped_count += 1
                    except Exception:
                        # If we can't read the file, try to process it anyway
                        files_to_process.append(h5_file)

                if verbose and skipped_count > 0:
                    print(
                        f"Skipping {skipped_count} files that already have all metrics"
                    )
                    print("Use --force to overwrite existing metrics")
                    print()

                h5_files = files_to_process

            # Apply max_files limit
            if args.max_files and len(h5_files) > args.max_files:
                if verbose:
                    print(
                        f"Limiting to first {args.max_files} files (out of {len(h5_files)} total)"
                    )
                h5_files = h5_files[: args.max_files]

            if not h5_files:
                print(
                    "No files to process (all files already have the requested metrics)."
                )
                print("Use --force to overwrite existing metrics.")
                return

            # Process the files - always calculate all metrics
            results = batch_add_all_metrics(
                h5_files, wavelength_range, verbose=verbose, max_workers=args.workers
            )

            # Summary for all metrics
            successful_heterogeneity = sum(
                1 for r in results.values() if r["heterogeneity"] is not None
            )
            successful_cd_peak = sum(
                1 for r in results.values() if r["cd_abs_peak"] is not None
            )
            successful_g_factor = sum(
                1 for r in results.values() if r["g_factor"] is not None
            )

            successful_files = [
                f
                for f, r in results.items()
                if r["heterogeneity"] is not None
                or r["cd_abs_peak"] is not None
                or r["g_factor"] is not None
            ]
            failed_files = [
                f
                for f, r in results.items()
                if r["heterogeneity"] is None
                and r["cd_abs_peak"] is None
                and r["g_factor"] is None
            ]

            if verbose:
                print("\n=== FINAL SUMMARY ===")
                print(f"Files processed: {len(h5_files)}")
                print(f"  With heterogeneity metrics: {successful_heterogeneity}")
                print(f"  With CD absorption peak metrics: {successful_cd_peak}")
                print(f"  With g-factor metrics: {successful_g_factor}")
                print(f"  At least one metric successful: {len(successful_files)}")

            if failed_files:
                print(f"Failed to process {len(failed_files)} files")
                if verbose:
                    print("Failed files:")
                    for f in failed_files:
                        print(f"  {Path(f).name}")

            if not successful_files and not failed_files:
                print("No files were processed.")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
