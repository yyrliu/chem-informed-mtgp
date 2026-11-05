"""
G-factor analysis module for peak detection and width-based calculations.

This module provides functions for analyzing absorption peaks and determining
g-factor maximum values within peak width ranges. Uses scipy.signal.peak_widths
for robust peak width calculations at specified relative heights.
"""

import numpy as np
from .spectral_analysis import (
    correct_baseline,
    find_peaks_in_range,
)


def calculate_asymmetry(
    cd_dataset,
    min_peak_height=0.05,
    rel_height=0.5,
    wl_range=(480, 520),
    int_width_mult=2,
):
    """
    Calculates the asymmetry area under the g-factor curve within the 2*HWHM range
    of the main absorption peak.

    Parameters
    ----------
    cd_dataset : xarray.Dataset
        Dataset containing 'g_factor', 'abs_avg', and 'wavelength' data
    peak_height : float, optional
        Minimum height for peak detection in absorption data. Default is 0.05.
    rel_height : float, optional
        Relative height for peak width calculation. Default is 0.5.
    wl_range : tuple, optional
        Wavelength range (min, max) for peak finding. Default is (480, 520).
    int_width_mult : float, optional
        Multiplier for HWHM to define integration width. Default is 2.

    Returns
    -------
    dict
        Dictionary containing asymmetry score and analysis details.
    """
    # Validate required variables
    if not {"g_factor", "abs_avg", "wavelength"}.issubset(
        set(cd_dataset.data_vars) | set(cd_dataset.coords)
    ):
        return {"asymmetry_score": np.nan, "status": "missing_vars"}

    # sort dataset by wavelength from low to high
    cd_dataset = cd_dataset.copy().sortby("wavelength", ascending=True)

    wl = cd_dataset["wavelength"].values
    abs = cd_dataset["abs_avg"].values
    g_factor = cd_dataset["g_factor"].values

    # Early return on empty data
    if wl.size == 0 or abs.size == 0 or g_factor.size == 0:
        return {"asymmetry_score": np.nan, "status": "empty_data"}

    try:
        # Apply baseline correction to absorption data
        corrected_abs, _ = correct_baseline(
            cd_dataset["wavelength"], cd_dataset["abs_avg"]
        )

        # Find peaks in the corrected data
        peaks, widths, _ = find_peaks_in_range(
            wl,
            corrected_abs,
            min_peak_height=min_peak_height,
            wl_range=wl_range,
            rel_height=rel_height,
            width=1,
        )

        if not len(peaks):
            # Set area to zero if no peaks found
            return {"asymmetry_score": 0.0, "status": "no_peaks_found"}

        # Use the highest peak found
        max_peak_idx = np.argmax(corrected_abs[peaks].values)
        peak_wavelength = wl[peaks[max_peak_idx]]
        width, width_height, left_ip, right_ip = list(zip(*widths))[max_peak_idx]

        # Use unified peak width calculation
        fwhm_left_wl, fwhm_right_wl = np.interp([left_ip, right_ip], range(len(wl)), wl)

        # Determine HWHM from the FWHM bounds relative to peak
        # (left_hwhm, right_hwhm)
        hwhm = (peak_wavelength - fwhm_left_wl, fwhm_right_wl - peak_wavelength)

        # Area integration range: peak +/- (HWHM * int_width_mult)
        int_window = [
            peak_wavelength - hwhm[0] * int_width_mult,
            peak_wavelength + hwhm[1] * int_width_mult,
        ]

        # Calculate and return area under g-factor curve
        # xarray integration
        g_slice = cd_dataset["g_factor"].sel(wavelength=slice(*int_window))
        area = np.abs(np.abs(g_slice).integrate("wavelength"))

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


def find_g_factor_max(cd_dataset, wl_range=None, rel_height=0.3, fwhm_window="both"):
    """
    Calculates the maximum absolute value of the g-factor within the FWHM (full width at half maximum) window of the main absorption peak in a given dataset.
    Args:
        cd_dataset (xarray.Dataset): Dataset containing at least the variables 'g_factor', 'abs_avg', and 'wavelength'.
        wavelength_range (tuple, optional): Tuple of (min_wavelength, max_wavelength) to restrict the search range. If None, uses the full range.
        rel_height (float, optional): Relative height (between 0 and 1) for FWHM calculation. Default is 0.3.
        fwhm_window (str, optional): Which side of the FWHM window to use for searching ('left', 'right', or 'both'). Default is 'both'.
        dict: A dictionary containing:
            - 'max_g_factor' (float): Maximum absolute g-factor value found within the specified window.
            - 'max_wavelength' (float, optional): Wavelength at which the maximum g-factor occurs (if found).
            - 'uv_vis_peak_wavelength' (float, optional): Wavelength of the main absorption peak.
            - 'fwhm_range' (tuple, optional): Tuple of (left, right) FWHM window bounds.
            - 'search_range' (tuple, optional): Actual search window used for g-factor maximum.
            - 'status' (str): Status string indicating success or reason for failure.
            - 'error' (str, optional): Error message if an exception occurred.
    """
    # Validate required variables
    if not {"g_factor", "abs_avg", "wavelength"}.issubset(
        set(cd_dataset.data_vars) | set(cd_dataset.coords)
    ):
        return {"max_g_factor": np.nan, "status": "missing_vars"}

    # sort dataset by wavelength from low to high
    cd_dataset = cd_dataset.copy().sortby("wavelength", ascending=True)

    wl = cd_dataset["wavelength"].values
    abs_avg = cd_dataset["abs_avg"].values
    gfac = cd_dataset["g_factor"].values

    # Early return on empty data
    if wl.size == 0 or abs_avg.size == 0 or gfac.size == 0 or np.all(np.isnan(gfac)):
        return {"max_g_factor": np.nan, "status": "empty_data"}

    try:
        # Full range search
        peaks, widths, _ = find_peaks_in_range(
            wl, abs_avg, min_peak_height=0.1, wl_range=wl_range
        )

        if len(peaks) == 0:
            print("Warring: No peaks found in absorption data.")
            return {"max_g_factor": 0, "status": "no_peaks_found"}

        # Use the highest peak found
        max_peak_idx = np.argmax(abs_avg[peaks])
        uv_vis_peak_wavelength = wl[peaks[max_peak_idx]]

        width, width_height, left_ip, right_ip = list(zip(*widths))[max_peak_idx]
        fwhm_left_wl, fwhm_right_wl = np.interp([left_ip, right_ip], range(len(wl)), wl)

        window_left, window_right = fwhm_left_wl, fwhm_right_wl
        # Apply fwhm_window option
        if fwhm_window == "left":
            window_right = uv_vis_peak_wavelength
        elif fwhm_window == "right":
            window_left = uv_vis_peak_wavelength
        elif fwhm_window == "both":
            pass  # keep both sides
        else:
            raise ValueError(f"Invalid fwhm_window option: {fwhm_window}")

        # Search g-factor in this width
        fwhm_mask = (wl >= window_left) & (wl <= window_right)

        if np.any(fwhm_mask):
            g_factor_fwhm = gfac[fwhm_mask]
            wl_fwhm = wl[fwhm_mask]

            abs_g = np.abs(g_factor_fwhm)
            max_idx = np.argmax(abs_g)

            gfac_max_val = float(abs_g[max_idx])
            gfac_max_wl = float(wl_fwhm[max_idx])

            return {
                "max_g_factor": gfac_max_val,
                "max_wavelength": gfac_max_wl,
                "uv_vis_peak_wavelength": float(uv_vis_peak_wavelength),
                "fwhm_range": (float(fwhm_left_wl), float(fwhm_right_wl)),
                "search_range": (float(window_left), float(window_right)),
                "status": "success",
            }
        else:
            print("Warring: FWHM mask resulted in empty selection.")
            return {"max_g_factor": 0, "status": "fwhm_empty"}

    except Exception as e:
        raise ValueError("G-factor max calculation failed") from e
