"""
Spectral analysis for xarray DataArrays, including baseline correction, peak finding, Voigt fitting, and area calculation.
"""

import numpy as np
import xarray as xr
import logging
from scipy.signal import peak_widths, find_peaks
from scipy.optimize import curve_fit, brentq
from scipy.special import wofz
from pybaselines import Baseline

# Configure logger
logger = logging.getLogger(__name__)


def baseline_correction(da: xr.DataArray, sample=None, method=None, **kwargs):
    """Perform baseline correction on an xarray DataArray using specified method.
    Parameters:
        da (xr.DataArray): Input data array to be baseline corrected.
        sample (str, optional): Type of sample to determine default method and parameters.
        method (str, optional): Baseline correction method to use.
    Returns:
        corrected_da (xr.DataArray): Baseline corrected data array.
        baseline_da (xr.DataArray): Calculated baseline data array.
    """

    if sample is not None and method is not None:
        raise ValueError("Either method or sample type must be specified, not both.")

    if method is None:
        if sample == "PXRD":
            method = "asls"
            kwargs = {"lam": 5e3}
        elif sample == "UV-Vis":
            method = "aspls"
            kwargs = {"lam": 7e5}
        else:
            raise ValueError("Unknown sample type. Please specify method directly.")

    baseline_finder = Baseline()
    func = getattr(baseline_finder, method)
    baseline, _ = func(da.values, **kwargs)
    corrected = da.values - baseline

    corrected_da = xr.DataArray(corrected, coords=da.coords, dims=da.dims)
    baseline_da = xr.DataArray(baseline, coords=da.coords, dims=da.dims)
    return corrected_da, baseline_da


def voigt_profile_func(x, amp, cen, sigma, gamma, slope, offset):
    """
    Voigt profile function with linear background.
    """
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    return (
        amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        + slope * (x - cen)
        + offset
    )


def find_peaks_in_window(
    da: xr.DataArray, x: str, target=None, window_size=None, **find_peaks_kwargs
):
    """
    Find peaks in an xarray DataArray, optionally restricting to a window around a target.
    """
    if x not in da.dims and x not in da.coords:
        raise ValueError(f"{x} must be a dimension or coordinate in the DataArray.")

    if target is not None and window_size is not None:
        target_da = da.sel(
            **{x: slice(target - window_size / 2, target + window_size / 2)}
        )
        da = target_da

    peaks_indices, properties = find_peaks(da.values, **find_peaks_kwargs)

    # convert indices back to original x values (wavelength or twoTheta)
    peaks_x = da[x].values[peaks_indices]
    # get intensities at peak positions
    peaks_intensities = da.values[peaks_indices]

    # Convert properties from dict of arrays to array of dicts
    properties = [dict(zip(properties, t)) for t in zip(*properties.values())]

    return peaks_x, peaks_intensities, properties


def calculate_peak_widths(
    da: xr.DataArray, x: str, peaks_x=None, peaks_indices=None, rel_height=0.5
):
    """
    Calculate peak widths for peaks in an xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with peak data
    x : str
        Name of the coordinate dimension (e.g., 'wavelength', 'twoTheta')
    peaks_x : array-like, optional
        Peak positions in x-coordinates. Either peaks_x or peaks_indices must be provided.
    peaks_indices : array-like, optional
        Indices of peaks in the data. Either peaks_x or peaks_indices must be provided.
    rel_height : float, optional
        Relative height for width calculation. Default is 0.5.

    Returns
    -------
    list of tuples
        List of (width, width_height, left_ip, right_ip) for each peak
    """
    if peaks_x is None and peaks_indices is None:
        raise ValueError("Either peaks_x or peaks_indices must be provided")

    # Convert peaks_x to indices if needed
    if peaks_indices is None:
        x_values = da[x].values
        peaks_indices = np.searchsorted(x_values, peaks_x)

    if len(peaks_indices) == 0:
        return []

    # Calculate widths using scipy on the values
    widths_data = peak_widths(da.values, peaks_indices, rel_height=rel_height)

    # widths_data is a tuple of (widths, width_heights, left_ips, right_ips)
    # Pack into list of tuples for each peak
    results = list(zip(widths_data[0], widths_data[1], widths_data[2], widths_data[3]))

    return results


def voigt_fit(
    da: xr.DataArray, x: str, peak_pos: float, window_size: float, no_slope=True
):
    """
    Fit a Voigt profile to a peak in the DataArray.
    """
    # set bounds for parameters: amp > 0, sigma > 0, gamma > 0
    # amp, cen, sigma, gamma, slope, offset
    if no_slope:
        bounds = (
            [0, peak_pos - 0.5, 0, 0, -1e-9, -np.inf],
            [np.inf, peak_pos + 0.5, np.inf, np.inf, 1e-9, np.inf],
        )
    else:
        bounds = (
            [0, peak_pos - 0.5, 0, 0, -np.inf, -np.inf],
            [np.inf, peak_pos + 0.5, np.inf, np.inf, np.inf, np.inf],
        )

    peak_intensity = da.sel(**{x: peak_pos}, method="nearest").values
    fit_da = da.sel(
        **{x: slice(peak_pos - window_size / 2, peak_pos + window_size / 2)}
    )
    x_data = fit_da[x].values
    y_data = fit_da.values

    # guess: amp, cen, sigma, gamma, slope, offset
    initial_guess = [peak_intensity, peak_pos, 0.1, 0.1, 0, np.min(y_data)]
    try:
        popt, pcov = curve_fit(
            voigt_profile_func, x_data, y_data, p0=initial_guess, bounds=bounds
        )
    except RuntimeError as e:
        logger.error(f"Voigt fit failed for peak at {peak_pos:.2f}: {e}")
        raise

    fitted_curve = voigt_profile_func(da[x].values, *popt)
    fitted_da = xr.DataArray(fitted_curve, coords=da.coords, dims=da.dims)

    return fitted_da, popt


def get_confidence_bounds(amp, cen, sigma, gamma, confidence=0.95, cut_off=2):
    """
    Calculate the x-values that bound `confidence` (fraction) of the Voigt profile area.
    """
    # Estimate FWHM to define a sufficient grid
    fwhm_L = 2 * gamma
    fwhm_G = 2.3548 * sigma
    fwhm = 0.5346 * fwhm_L + np.sqrt(0.2166 * fwhm_L**2 + fwhm_G**2)

    # Create a grid wide enough to capture the tails (20 * FWHM)
    window = 20 * fwhm
    if window == 0:
        window = 1.0

    x_grid = np.linspace(cen - window, cen + window, 5000)
    y_grid = voigt_profile_func(x_grid, amp, cen, sigma, gamma, 0, 0)

    cum_area = np.cumsum(y_grid) * (x_grid[1] - x_grid[0])
    total_area = cum_area[-1]
    if total_area == 0:
        return cen - 0.1, cen + 0.1

    alpha = 1.0 - confidence
    lower_q = alpha / 2.0
    upper_q = 1.0 - (alpha / 2.0)

    norm_cum = cum_area / total_area
    idx_low = np.searchsorted(norm_cum, lower_q)
    idx_high = np.searchsorted(norm_cum, upper_q)

    l, r = x_grid[idx_low], x_grid[idx_high]
    l, r = np.clip(l, cen - cut_off * fwhm, cen), np.clip(r, cen, cen + cut_off * fwhm)

    return l, r


def calculate_peak_areas(
    da: xr.DataArray,
    x: str,
    fit_results: list,
    confidence=0.95,
    tolerance=0.025,
    integration_max_span=3.0,
):
    """
    Calculate peak areas using non-overlapping integration ranges based on Voigt fits.
    """
    sorted_results = sorted(fit_results, key=lambda r: r["popt"][1])
    processed_results = []
    x_min, x_max = da[x].min().item(), da[x].max().item()

    intersections = {}
    for i in range(len(sorted_results) - 1):
        popt_curr = sorted_results[i]["popt"]
        popt_next = sorted_results[i + 1]["popt"]

        c_amp, c_cen, c_sigma, c_gamma = popt_curr[:4]
        n_amp, n_cen, n_sigma, n_gamma = popt_next[:4]

        def diff_func(val):
            val_curr = voigt_profile_func(val, c_amp, c_cen, c_sigma, c_gamma, 0, 0)
            val_next = voigt_profile_func(val, n_amp, n_cen, n_sigma, n_gamma, 0, 0)
            return val_curr - val_next

        try:
            root = brentq(diff_func, c_cen, n_cen)
            intersections[i] = root
        except ValueError:
            intersections[i] = (c_cen + n_cen) / 2

    for i, res in enumerate(sorted_results):
        popt = res["popt"]
        if len(popt) == 6:
            amp, cen, sigma, gamma, slope, offset = popt
        else:
            amp, cen, sigma, gamma = popt[:4]
            slope, offset = 0, 0

        l_conf, r_conf = get_confidence_bounds(
            amp, cen, sigma, gamma, confidence=confidence
        )

        if i == 0:
            l_final = max(l_conf, x_min)
        else:
            l_final = max(intersections[i - 1], l_conf)

        if i == len(sorted_results) - 1:
            r_final = min(r_conf, x_max)
        else:
            r_final = min(intersections[i], r_conf)

        if (r_final - l_final) > integration_max_span:
            half_span = integration_max_span / 2
            l_prop = cen - half_span
            r_prop = cen + half_span
            if i > 0:
                l_prop = max(l_prop, intersections[i - 1])
            if i < len(sorted_results) - 1:
                r_prop = min(r_prop, intersections[i])
            l_final, r_final = l_prop, r_prop

        if l_final >= r_final:
            l_final = cen - 0.05
            r_final = cen + 0.05

        l_integ = max(l_final, x_min)
        r_integ = min(r_final, x_max)

        subset_x = da.sel(**{x: slice(l_integ, r_integ)})[x].values

        if (
            len(subset_x) < 2
            or np.abs(subset_x[0] - l_integ) > tolerance
            or np.abs(subset_x[-1] - r_integ) > tolerance
        ):
            grid_x = np.unique(np.concatenate(([l_integ], subset_x, [r_integ])))
            grid_x = grid_x[(grid_x >= l_integ) & (grid_x <= r_integ)]
            if len(grid_x) > 1:
                extended_y = da.interp({x: grid_x}).values
                subset = xr.DataArray(extended_y, coords={x: grid_x}, dims=[x])
            else:
                subset = xr.DataArray([], coords={x: []}, dims=[x])
        else:
            subset = da.sel(**{x: subset_x})

        if subset.sizes[x] > 1:
            background = slope * (subset[x] - cen) + offset
            corrected_subset = subset - background
            area = corrected_subset.integrate(coord=x).item()
        else:
            area = 0.0

        res_copy = res.copy()
        res_copy.update(
            {
                "area": area,
                "integration_range": (l_integ, r_integ),
                "offset_used": offset,
                "slope_used": slope,
            }
        )
        processed_results.append(res_copy)

    return processed_results
