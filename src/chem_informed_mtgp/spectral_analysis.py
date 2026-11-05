import numpy as np
from scipy.signal import peak_widths, find_peaks
from scipy.optimize import curve_fit
from scipy.special import wofz
import xarray as xr
from pybaselines import Baseline


# Voigt profile definition
def voigt(x, amp, pos, sigma, gamma):
    """
    Voigt profile function.
    sigma: Gaussian standard deviation
    gamma: Lorentzian half-width at half-maximum (HWHM)
    """
    z = ((x - pos) + 1j * gamma) / (sigma * np.sqrt(2))
    # wofz is the Faddeeva function
    return amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))


def fit_peak(x_data, y_data, center_guess=None):
    """
    Fit a peak to a Voigt profile.

    Parameters
    ----------
    x_data : array-like
        The x values of the peak region.
    y_data : array-like
        The intensity values of the peak region.
    center_guess : float, optional
        Initial guess for the peak center.

    Returns
    -------
    dict
        Dictionary containing fitted parameters ('amp', 'pos', 'sigma', 'gamma')
        and the calculated 'fwhm'. Returns None if fit fails.
    """
    if len(x_data) < 4:  # Need enough points to fit
        return None

    # Initial guesses
    if center_guess is None:
        center_guess = x_data[np.argmax(y_data)]

    amp_guess = np.max(y_data)
    sigma_guess = (x_data[-1] - x_data[0]) / 10.0
    gamma_guess = sigma_guess

    p0 = [amp_guess, center_guess, sigma_guess, gamma_guess]

    # bounds: amp > 0, pos within range, sigma > 0, gamma > 0
    bounds = ([0, x_data[0], 0, 0], [np.inf, x_data[-1], np.inf, np.inf])

    try:
        popt, pcov = curve_fit(voigt, x_data, y_data, p0=p0, bounds=bounds, maxfev=2000)
    except (RuntimeError, ValueError):
        return None

    amp, pos, sigma, gamma = popt

    # Calculate FWHM for Voigt profile (approximate formula by Olivero and Longbothum)
    fwhm_g = 2.35482 * sigma
    fwhm_l = 2 * gamma
    fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)

    return {
        "amp": amp,
        "pos": pos,
        "sigma": sigma,
        "gamma": gamma,
        "fwhm": fwhm,
        "fitted_curve": voigt(x_data, *popt),
    }


def correct_baseline(x_data, intensity, offset=0, min_clip=None, method=None, **kwargs):
    """
    Correct the baseline of the intensity data using the specified method.

    Parameters
    ----------
    x_data : array-like
        The x values (e.g., wavelength or 2θ) corresponding to the intensity data.
    intensity : array-like
        The intensity data to correct.
    offset : float, optional
        Offset to add after baseline correction. Default is 0.
    min_clip : float, optional
        Minimum value to clip the corrected intensity to. Default is None.
    method : str, optional
        The method to use for baseline correction. Default is "aspls".
    **kwargs : dict
        Additional parameters for the baseline correction method.

    Returns
    -------
    tuple
        (corrected_intensity, (calc_baseline, params))
        - corrected_intensity: The baseline-corrected intensity data
        - calc_baseline: The calculated baseline
        - params: Parameters from the baseline fitting
    """

    if method is None:
        method = "aspls"
        if not kwargs:
            kwargs = {
                "lam": 1e7
            }  # synced with w-h_plot default usage or ingest default? Ingest has 7e5, wh has 1e7. Let's stick to argument handling.
            # wait, ingestion uses defaults if not provided. logic below handles it if kwargs is empty.
            if not kwargs:
                kwargs = {"lam": 7e5}  # Default from ingestion.py

    baseline_fitter = Baseline(x_data=x_data, check_finite=False)
    func = getattr(baseline_fitter, method)
    calc_baseline, params = func(intensity, **kwargs)
    corrected_intensity = (
        np.clip(intensity - calc_baseline, a_min=min_clip, a_max=None) + offset
    )

    return corrected_intensity, (calc_baseline, params)


def find_peaks_in_range(
    x_data, intensity, min_peak_height=None, wl_range=None, rel_height=0.5, width=None
):
    """
    Find peaks in the intensity data, optionally within a specified range.

    Parameters
    ----------
    x_data : array-like
        The x values corresponding to the intensity data.
    intensity : array-like
        The intensity data to analyze.
    peak_height : float, optional
        The minimum height for peak detection. Default is 0.1.
    wl_range : tuple, optional
        Range (min, max) to mask data outside for peak finding. If None, uses full data.
    rel_height : float, optional
        Relative height for peak width calculation. Default is 0.5.

    Returns
    -------
    tuple
        (peaks, results_half)
        - peaks: Indices of detected peaks
        - widths: Peak width results at relative height
    """

    # Apply masking if wl_range is provided
    if wl_range is not None:
        # Check if intensity is an xarray object, otherwise use numpy indexing
        if isinstance(intensity, (xr.DataArray, xr.Dataset)):
            intensity_masked = intensity.where(
                (x_data >= wl_range[0]) & (x_data <= wl_range[1]), 0
            )
        else:
            # Assume numpy array
            mask = (x_data >= wl_range[0]) & (x_data <= wl_range[1])
            intensity_masked = np.where(mask, intensity, 0)
    else:
        intensity_masked = intensity

    # Find peaks
    peaks, props = find_peaks(intensity_masked, height=min_peak_height, width=width)
    widths = peak_widths(intensity, peaks, rel_height=rel_height)

    return peaks, widths, props
