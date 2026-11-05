import numpy as np
from scipy.stats import linregress
from scipy.signal import peak_prominences
from .spectral_analysis import correct_baseline, find_peaks_in_range, fit_peak
import xarray as xr


def calculate_williamson_hall(peak_positions, fwhms, wavelength=1.5406):
    """
    Perform Williamson-Hall analysis.

    Parameters
    ----------
    peak_positions : list or array
        Peak positions in 2theta (degrees).
    fwhms : list or array
        Full width at half maximum (degrees).
    wavelength : float
        X-ray wavelength (Angstrom). Default is Cu K-alpha (1.5406 A).

    Returns
    -------
    dict
        Dictionary containing WH results or None if insufficient points.
    """
    if len(peak_positions) < 2:
        return None

    # Convert to radians
    theta = np.deg2rad(np.array(peak_positions) / 2)
    beta = np.deg2rad(np.array(fwhms))

    # Williamson-Hall equation: beta * cos(theta) = (k * lambda / D) + (4 * epsilon * sin(theta))
    y_wh = beta * np.cos(theta)
    x_wh = 4 * np.sin(theta)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_wh, y_wh)

    # Calculate physical parameters
    k_shape_factor = 0.9
    strain = slope  # dimensionless

    # Grain size (D = k * lambda / intercept)
    if intercept > 0:
        grain_size_angstrom = (k_shape_factor * wavelength) / intercept
        grain_size_nm = grain_size_angstrom / 10.0
    else:
        grain_size_nm = np.nan

    return {
        "strain": strain,
        "grain_size_nm": grain_size_nm,
        "r_squared": r_value**2,
        "slope": slope,
        "intercept": intercept,
        "x_wh": x_wh,
        "y_wh": y_wh,
        "std_err": std_err,
    }


def williamson_hall_analysis(
    ds, target_peaks, width_method="vogit", window=2.0, rel_height=0.5, baseline_lam=1e7
):
    """
    Process XRD data: baseline correction, peak finding, fitting, and W-H analysis.

    Args:
        ds: xarray Dataset or DataArray with '2theta' and 'intensity'
        target_peaks: List of expected peak positions
        width_method: Method for peak width calculation ('vogit' or 'integral_breadth')
        window: Search window around target peaks (degrees)
        rel_height: Relative height for peak width initial estimation
        baseline_lam: Lambda parameter for baseline correction

    Returns
    -------
    dict
        Dictionary containing all analysis results.
    """
    x = ds["2theta"]
    y = ds["intensity"]

    # 1. Baseline correction
    y_corr, (baseline, _) = correct_baseline(x, y, lam=baseline_lam)

    detected_peaks = []
    for target_peak in target_peaks:
        wl_range = (target_peak - window, target_peak + window)

        peaks, widths, props = find_peaks_in_range(
            x, y_corr, min_peak_height=0.1, wl_range=wl_range, rel_height=0.9, width=2
        )

        # Calculate prominences to get bases for integration
        if len(peaks) > 0:
            prominences, left_bases, right_bases = peak_prominences(
                y_corr.values, peaks
            )
            props["left_bases"] = left_bases.astype(int)
            props["right_bases"] = right_bases.astype(int)

        result = {
            "target": target_peak,
            "window": wl_range,
        }

        if len(peaks) > 0:
            peak_xs = x.values[peaks]
            closest_idx = np.argmin(np.abs(peak_xs - target_peak))
            peak_2theta = x[peaks[closest_idx]]
            width, width_height, left_ip, right_ip = list(zip(*widths))[closest_idx]
            # convert dict of lists -> list of dicts then access by index
            prop = list(dict(zip(props, t)) for t in zip(*props.values()))[closest_idx]
            result.update(
                {
                    "detected_x": float(peak_2theta.values),
                    "detected_y": float(prop["peak_heights"]),
                    "prop": prop,
                    "window": x.isel(
                        **{"2theta": [int(left_ip), int(right_ip)]}
                    ).values,
                }
            )
        else:
            result.update(
                {
                    "detected_x": np.nan,
                    "detected_y": np.nan,
                    "prop": None,
                }
            )

        detected_peaks.append(result)

    fitting_results = []
    # supersample x for better fitting resolution
    fitted_y_x = np.linspace(x.min().values, x.max().values, num=len(x) * 5)
    fitted_y = xr.DataArray(
        np.zeros_like(fitted_y_x), coords=[fitted_y_x], dims=["2theta"]
    )

    for peak in detected_peaks:
        if np.isnan(peak["detected_x"]):
            fitting_results.append(None)
            continue

        wl_range = peak["window"]
        target_mask = (x >= wl_range[0]) & (x <= wl_range[1])

        fit_res = fit_peak(
            x[target_mask], y_corr[target_mask], center_guess=peak["detected_x"]
        )
        from chem_informed_mtgp.spectral_analysis import voigt  # Needed for plotting

        fitted_y += voigt(
            fitted_y.coords["2theta"],
            fit_res["amp"],
            fit_res["pos"],
            fit_res["sigma"],
            fit_res["gamma"],
        )

        if width_method == "vogit":
            fitting_results.append(fit_res)

        elif width_method == "integral_breadth":
            one_percent_height = np.percentile(
                fitted_y.sel(**{"2theta": slice(*wl_range)}), 5
            )

            one_percent_x = (
                y_corr.sel(**{"2theta": slice(*wl_range)})
                .where(y_corr >= one_percent_height)["2theta"]
                .values
            )

            import matplotlib.pyplot as plt

            plt.plot(x, y_corr, label="Corrected Data")
            plt.plot(fitted_y.coords["2theta"], fitted_y, label="Fitted Voigt")
            plt.axhline(
                one_percent_height, color="red", linestyle="--", label="1% Height"
            )
            plt.scatter(
                one_percent_x,
                [one_percent_height] * len(one_percent_x),
                color="green",
                label="1% Intersection Points",
            )
            plt.legend()
            plt.ylim(-0.1, 0.5)
            plt.xlim(peak["detected_x"] - 3, peak["detected_x"] + 3)
            plt.show()

            print(one_percent_x)
            if len(one_percent_x) < 2:
                pass

            # use xarray integration to get area under peak
            left_base = peak["prop"]["left_bases"]
            right_base = peak["prop"]["right_bases"]

            int_mask = (x >= x[peak["prop"]["left_bases"]]) & (
                x <= x[peak["prop"]["right_bases"]]
            )

            # Use isel for integer indexing
            dim_name = y_corr.dims[0]
            subset = y_corr.isel({dim_name: slice(left_base, right_base)})

            area = y_corr[int_mask].integrate("2theta").values
            height = peak["detected_y"]
            if height > 0:
                fwhm = area / height
                fit_res = {
                    "pos": peak["detected_x"],
                    "fwhm": fwhm,
                    "integral_area": area,
                    "height": height,
                    "integration_window": (left_base, right_base),
                }
                fitting_results.append(fit_res)
            else:
                fitting_results.append(None)

        else:
            raise ValueError(f"Unknown width_method: {width_method}")

    # Williamson-Hall Analysis
    valid_peaks_pos = []
    valid_peaks_fwhm = []
    for fit_res in fitting_results:
        if fit_res is not None:
            valid_peaks_pos.append(fit_res["pos"])
            valid_peaks_fwhm.append(fit_res["fwhm"])
    wh_res = calculate_williamson_hall(valid_peaks_pos, valid_peaks_fwhm)
    return {
        "sample_name": str(ds["sample_name"].values)
        if "sample_name" in ds
        else "Unknown",
        "x": x,
        "y": y,
        "y_corr": y_corr,
        "baseline": baseline,
        "detected_peaks": detected_peaks,
        "fitting_results": fitting_results,
        "wh_res": wh_res,
        "target_peaks_count": len(target_peaks),
    }


def process_xrd_pattern(ds, target_peaks, window=2.0, rel_height=0.5, baseline_lam=1e7):
    """
    Process XRD data: baseline correction, peak finding, fitting, and W-H analysis.

    Args:
        ds: xarray Dataset or DataArray with '2theta' and 'intensity'
        target_peaks: List of expected peak positions
        window: Search window around target peaks (degrees)
        rel_height: Relative height for peak width initial estimation
        baseline_lam: Lambda parameter for baseline correction

    Returns
    -------
    dict
        Dictionary containing all analysis results.
    """
    x = ds["2theta"]
    y = ds["intensity"]

    # 1. Baseline correction
    y_corr, (baseline, _) = correct_baseline(x, y, lam=baseline_lam)

    detected_peaks_data = []
    valid_peaks_pos = []
    valid_peaks_fwhm = []

    for i, target in enumerate(target_peaks):
        wl_range = (target - window, target + window)

        # skip if range outside data
        if wl_range[1] < x.min() or wl_range[0] > x.max():
            continue

        mask = (x >= wl_range[0]) & (x <= wl_range[1])
        if not mask.any():
            continue

        local_max = y_corr.where(mask).max().item()

        if local_max < 0.2:
            continue

        peaks, results_half = find_peaks_in_range(
            x.values,
            y_corr.values,
            min_peak_height=local_max * 0.2,
            wl_range=wl_range,
            rel_height=rel_height,
        )

        if len(peaks) > 0:
            peak_locs = x.values[peaks]
            closest_idx = np.argmin(np.abs(peak_locs - target))
            peak_idx = peaks[closest_idx]
            detected_x = x.values[peak_idx]

            fit_mask = (x >= wl_range[0]) & (x <= wl_range[1])
            x_fit = x.values[fit_mask]
            y_fit = y_corr.values[fit_mask]

            fit_res = fit_peak(x_fit, y_fit, center_guess=detected_x)

            peak_info = {
                "target": target,
                "target_index": i,
                "detected_x": detected_x,
                "detected_y": float(y_corr.values[peak_idx]),
                "fit_res": fit_res,
                "is_fit_good": False,
            }

            if fit_res:
                fit_pos = fit_res["pos"]
                if abs(fit_pos - detected_x) < 0.5:
                    peak_info["is_fit_good"] = True
                    # Add to W-H data
                    valid_peaks_pos.append(fit_pos)
                    valid_peaks_fwhm.append(fit_res["fwhm"])
                    peak_info["x_fit"] = x_fit

            detected_peaks_data.append(peak_info)

    # Williamson-Hall Analysis
    wh_res = calculate_williamson_hall(valid_peaks_pos, valid_peaks_fwhm)

    return {
        "sample_name": str(ds["sample_name"].values)
        if "sample_name" in ds
        else "Unknown",
        "x": x,
        "y": y,
        "y_corr": y_corr,
        "baseline": baseline,
        "detected_peaks": detected_peaks_data,
        "wh_res": wh_res,
        "target_peaks_count": len(target_peaks),
    }
