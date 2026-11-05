"""
G-factor analysis module for peak detection and width-based calculations using xarray.

This module provides functions for analyzing absorption peaks and determining
g-factor maximum values within peak width ranges. Uses xarray-native operations
throughout, avoiding manipulation of numpy arrays directly.
"""

import numpy as np
import xarray as xr
import logging
from .xr_spectral_analysis import (
    baseline_correction as correct_baseline,
    find_peaks_in_window,
    calculate_peak_widths,
)

# Configure logger
logger = logging.getLogger(__name__)


def calculate_asymmetry(
    cd_dataset,
    min_peak_height=0.05,
    rel_height=0.5,
    wl_range=(480, 520),
    int_width_mult=2.0,
):
    """
    Calculates the asymmetry area under the g-factor curve within the 2*HWHM range
    of the main absorption peak.

    Workflow:
    1. Find the main absorption peak in the UV-Vis data (within wl_range)
    2. Calculate HWHM (half-width at half-maximum) at the specified rel_height
    3. Define integration window as: peak ± (HWHM * int_width_mult)
    4. Integrate the absolute g-factor values within this window

    Parameters
    ----------
    cd_dataset : xarray.Dataset
        Dataset containing 'g_factor', 'abs_avg', and 'wavelength' data
    min_peak_height : float, optional
        Minimum height for peak detection in absorption data. Default is 0.05.
    rel_height : float, optional
        Relative height (0-1) for calculating peak width. Default is 0.5 (FWHM).
    wl_range : tuple, optional
        (min_wl, max_wl) to restrict where to search for the absorption peak.
        Does not limit the g-factor integration window. Default is (480, 520).
    int_width_mult : float, optional
        Multiplier for HWHM to define integration window width.
        Window = peak ± (HWHM * int_width_mult). Default is 2.0.

    Returns
    -------
    dict
        Dictionary containing asymmetry score and analysis details.
    """
    # Validate required variables
    if not {"g_factor", "abs_avg", "wavelength"}.issubset(
        set(cd_dataset.data_vars) | set(cd_dataset.coords)
    ):
        return {
            "asymmetry_score": np.nan,
            "uv_vis_peak": None,
            "integration_range": None,
            "status": "missing_vars",
        }

    # Sort dataset by wavelength from low to high
    cd_dataset = cd_dataset.copy().sortby("wavelength", ascending=True)

    wl = cd_dataset["wavelength"]
    abs_da = cd_dataset["abs_avg"]
    g_factor = cd_dataset["g_factor"]

    # Early return on empty data
    if wl.size == 0 or abs_da.size == 0 or g_factor.size == 0:
        return {
            "asymmetry_score": np.nan,
            "uv_vis_peak": None,
            "integration_range": None,
            "status": "empty_data",
        }

    try:
        # Apply baseline correction to absorption data using xarray-native function
        corrected_abs, _ = correct_baseline(abs_da, sample="UV-Vis")

        # Find peaks in the specified wavelength range using window parameters
        wl_min, wl_max = wl_range
        target = (wl_min + wl_max) / 2
        window_size = wl_max - wl_min
        peaks_x, peaks_intensities, _ = find_peaks_in_window(
            corrected_abs,
            x="wavelength",
            target=target,
            window_size=window_size,
            height=min_peak_height,
            width=1,
        )

        if len(peaks_x) == 0:
            # Set area to zero if no peaks found
            return {
                "asymmetry_score": 0.0,
                "uv_vis_peak": None,
                "integration_range": None,
                "status": "no_peaks_found",
            }

        # Use the highest peak found
        max_peak_idx = np.argmax(peaks_intensities)
        peak_wavelength = peaks_x[max_peak_idx]

        # Calculate peak widths
        res_width, *_ = calculate_peak_widths(
            corrected_abs,
            x="wavelength",
            peaks_x=[peak_wavelength],
            rel_height=rel_height,
        )

        # Get width information
        width, width_height, left_ip, right_ip = res_width

        # Convert interpolation points to wavelengths
        wl_values = wl.values
        fwhm_left_wl = float(np.interp(left_ip, range(len(wl_values)), wl_values))
        fwhm_right_wl = float(np.interp(right_ip, range(len(wl_values)), wl_values))

        # Determine HWHM from the FWHM bounds relative to peak
        hwhm = (peak_wavelength - fwhm_left_wl, fwhm_right_wl - peak_wavelength)

        # Area integration range: peak +/- (HWHM * int_width_mult)
        int_window = [
            peak_wavelength - hwhm[0] * int_width_mult,
            peak_wavelength + hwhm[1] * int_width_mult,
        ]

        # Calculate area under g-factor curve using xarray integration
        g_slice = g_factor.sel(wavelength=slice(*int_window))
        area = np.abs(g_slice).integrate("wavelength")

        uv_vis_peak = {
            "wavelength": float(peak_wavelength),
            "fwhm_wl": (float(fwhm_left_wl), float(fwhm_right_wl)),
            "hwhm": hwhm,
            "width_height": float(width_height),
            "width": float(width),
        }

        return {
            "asymmetry_score": float(area.values),
            "uv_vis_peak": uv_vis_peak,
            "integration_range": (float(int_window[0]), float(int_window[1])),
            "status": "success",
        }

    except Exception as e:
        raise ValueError("Asymmetry calculation failed") from e


def find_g_factor_max(
    cd_dataset,
    wl_range=None,
    rel_height=0.3,
    fwhm_window="both",
    int_width_mult=2.0,
    baseline_correction=True,
):
    """
    Finds the maximum absolute g-factor value within a width-based window around
    the main absorption peak.

    Workflow:
    1. Find the main absorption peak in the UV-Vis data (within wl_range if specified)
    2. Calculate HWHM (half-width at half-maximum) at the specified rel_height
    3. Define search window based on fwhm_window:
       - 'both': peak ± (HWHM * int_width_mult)
       - 'left': [peak - HWHM * int_width_mult, peak]
       - 'right': [peak, peak + HWHM * int_width_mult]
    4. Find the maximum absolute g-factor value within this window

    Args:
        cd_dataset (xarray.Dataset): Dataset containing at least the variables
            'g_factor', 'abs_avg', and 'wavelength'.
        wl_range (tuple, optional): (min_wl, max_wl) to restrict where to search
            for the absorption peak. If None, searches the full wavelength range.
            Does not limit the g-factor search window.
        rel_height (float, optional): Relative height (0-1) for calculating peak
            width. Default is 0.5 (FWHM).
        fwhm_window (str, optional): Which side(s) of the peak to search:
            'left', 'right', or 'both'. Default is 'both'.
        int_width_mult (float, optional): Multiplier for HWHM to define search
            window width. Window = peak ± (HWHM * int_width_mult). Default is 2.0.
        baseline_correction (bool or callable, optional): If True, applies a
            default baseline correction to the absorption data before peak detection.

    Returns:
        dict: A dictionary containing:
            - 'max_g_factor' (float): Maximum absolute g-factor value found
              within the specified window.
            - 'max_wavelength' (float, optional): Wavelength at which the
              maximum g-factor occurs (if found).
            - 'uv_vis_peak_wavelength' (float, optional): Wavelength of the
              main absorption peak.
            - 'fwhm_range' (tuple, optional): Tuple of (left, right) FWHM
              window bounds.
            - 'search_range' (tuple, optional): Actual search window used for
              g-factor maximum.
            - 'status' (str): Status string indicating success or reason for failure.
    """
    # Validate required variables
    if not {"g_factor", "abs_avg", "wavelength"}.issubset(
        set(cd_dataset.data_vars) | set(cd_dataset.coords)
    ):
        return {
            "max_g_factor": np.nan,
            "max_wavelength": np.nan,
            "uv_vis_peak_wavelength": np.nan,
            "fwhm_range": None,
            "search_range": None,
            "status": "missing_vars",
        }

    # Sort dataset by wavelength from low to high
    cd_dataset = cd_dataset.copy().sortby("wavelength", ascending=True)

    wl = cd_dataset["wavelength"]
    abs_avg = cd_dataset["abs_avg"]
    gfac = cd_dataset["g_factor"]

    # Early return on empty data
    if wl.size == 0 or abs_avg.size == 0 or gfac.size == 0 or gfac.isnull().all():
        return {
            "max_g_factor": np.nan,
            "max_wavelength": np.nan,
            "uv_vis_peak_wavelength": np.nan,
            "fwhm_range": None,
            "search_range": None,
            "status": "empty_data",
        }

    try:
        if baseline_correction:
            corrected_abs, _ = correct_baseline(abs_avg, sample="UV-Vis")
        else:
            corrected_abs = abs_avg

        # Find peaks with optional wavelength range
        if wl_range is not None:
            wl_min, wl_max = wl_range
            target = (wl_min + wl_max) / 2
            window_size = wl_max - wl_min
            peaks_x, peaks_intensities, _ = find_peaks_in_window(
                corrected_abs,
                x="wavelength",
                target=target,
                window_size=window_size,
                height=0.1,
                width=1,
            )
        else:
            peaks_x, peaks_intensities, _ = find_peaks_in_window(
                corrected_abs, x="wavelength", height=0.1, width=1
            )

        if len(peaks_x) == 0:
            logger.warning("No peaks found in absorption data.")
            return {
                "max_g_factor": 0,
                "max_wavelength": np.nan,
                "uv_vis_peak_wavelength": np.nan,
                "fwhm_range": None,
                "search_range": None,
                "status": "no_peaks_found",
            }

        # Use the highest peak found
        max_peak_idx = int(np.argmax(peaks_intensities))
        uv_vis_peak_wavelength = float(peaks_x[max_peak_idx])

        # Calculate peak widths
        widths = calculate_peak_widths(
            corrected_abs, x="wavelength", peaks_x=peaks_x, rel_height=rel_height
        )
        width, width_height, left_ip, right_ip = widths[max_peak_idx]

        # Convert interpolation points to wavelengths
        wl_values = wl.values
        fwhm_left_wl = float(np.interp(left_ip, range(len(wl_values)), wl_values))
        fwhm_right_wl = float(np.interp(right_ip, range(len(wl_values)), wl_values))

        # Determine HWHM from the FWHM bounds relative to peak
        hwhm = (
            uv_vis_peak_wavelength - fwhm_left_wl,
            fwhm_right_wl - uv_vis_peak_wavelength,
        )

        # Apply fwhm_window option with int_width_mult
        if fwhm_window == "left":
            window_left = uv_vis_peak_wavelength - hwhm[0] * int_width_mult
            window_right = uv_vis_peak_wavelength
        elif fwhm_window == "right":
            window_left = uv_vis_peak_wavelength
            window_right = uv_vis_peak_wavelength + hwhm[1] * int_width_mult
        elif fwhm_window == "both":
            window_left = uv_vis_peak_wavelength - hwhm[0] * int_width_mult
            window_right = uv_vis_peak_wavelength + hwhm[1] * int_width_mult
        else:
            raise ValueError(f"Invalid fwhm_window option: {fwhm_window}")

        # Search g-factor in this window using xarray slicing
        g_factor_window = gfac.sel(wavelength=slice(window_left, window_right))

        if g_factor_window.size > 0:
            # Find maximum absolute g-factor value
            abs_g = np.abs(g_factor_window)

            # Get maximum value and its wavelength
            gfac_max_val = float(abs_g.max().values)
            max_wl_idx = abs_g.argmax().values
            gfac_max_wl = float(g_factor_window.wavelength[max_wl_idx].values)

            return {
                "max_g_factor": gfac_max_val,
                "max_wavelength": gfac_max_wl,
                "uv_vis_peak_wavelength": float(uv_vis_peak_wavelength),
                "fwhm_range": (float(fwhm_left_wl), float(fwhm_right_wl)),
                "search_range": (float(window_left), float(window_right)),
                "status": "success",
            }
        else:
            logger.warning("FWHM mask resulted in empty selection.")
            return {
                "max_g_factor": 0,
                "max_wavelength": np.nan,
                "uv_vis_peak_wavelength": float(uv_vis_peak_wavelength),
                "fwhm_range": (float(fwhm_left_wl), float(fwhm_right_wl)),
                "search_range": (float(window_left), float(window_right)),
                "status": "fwhm_empty",
            }

    except Exception as e:
        raise ValueError("G-factor max calculation failed") from e


def calculate_ldlb_area_ratio(cd_dataset: xr.Dataset, wl_range=(480, 520)):
    if not {"gen_CD", "ldlb_CD"}.issubset(
        set(cd_dataset.data_vars) | set(cd_dataset.coords)
    ):
        return {
            "ldlb_ratio": np.nan,
            "cd_area": np.nan,
            "ldlb_area": np.nan,
            "status": "missing_vars",
        }

    cd_dataset = cd_dataset.copy().sortby("wavelength", ascending=True)
    cd_dataset = cd_dataset.sel(wavelength=slice(*wl_range))

    cd = cd_dataset["gen_CD"]
    ldlb = cd_dataset["ldlb_CD"]

    cd_area = np.abs(cd).integrate("wavelength")
    ldlb_area = np.abs(ldlb).integrate("wavelength")

    try:
        return {
            "ldlb_ratio": float(np.abs(ldlb_area / cd_area).mean().values),
            "cd_area": float(cd_area.values),
            "ldlb_area": float(ldlb_area.values),
            "status": "success",
        }
    except Exception as e:
        raise ValueError("CD to LDLB ratio calculation failed") from e
