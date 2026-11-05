import os
import sys

# import matplotlib.transforms as mtransforms
# from matplotlib.lines import Line2D
from pathlib import Path
from natsort import natsort_keygen
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd

from chem_informed_mtgp.xr_spectral_analysis import (  # noqa: E402
    baseline_correction,
    find_peaks_in_window,
)

from chem_informed_mtgp.xr_g_factor_analysis import (  # noqa: E402
    calculate_asymmetry,
    find_g_factor_max,
    calculate_ldlb_area_ratio,
)


from configs.plt_rc import params_basic, figure_zoom_multiplier
from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dash_lib.data_utils import query_samples, load_dataset  # noqa: E402

# %%

# %%

########## set up plotting parameters ##########
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
################################################

db_path = Path(__file__).resolve().parent.parent / "mydb.duckdb"


samples = query_samples(db_path=db_path)
sorted_ids = samples.index.tolist()
ds = load_dataset(sorted_ids, "cd", async_mode=False, db_path=db_path)

mask = ds["sample_name"].isin(["14-17", "16-13"])
ds = ds.where(mask, drop=True)

# %%


def process_cd_data(
    ds,
    wl_range=(450, 550),
    peak_search_range=(480, 520),
    rel_height=0.5,
    peak_height=0.03,
):
    """
    Process CD data: baseline correction, peak finding, and integration.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing wavelength, gen_CD, g_factor, abs_avg
    wl_range : tuple
        Wavelength range for data extraction
    peak_search_range : tuple
        Wavelength range for searching peaks
    rel_height : float
        Relative height for peak width calculation

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    # Apply baseline correction to absorption data
    corrected_abs_avg, calc_baseline = baseline_correction(
        ds["abs_avg"], sample="UV-Vis"
    )

    # Find peaks in the corrected data
    wl_min, wl_max = wl_range
    target = (wl_min + wl_max) / 2
    window_size = wl_max - wl_min
    peaks, peaks_intensities, _ = find_peaks_in_window(
        corrected_abs_avg,
        x="wavelength",
        target=target,
        window_size=window_size,
        height=peak_height,
        width=1,
    )

    asymmetry_results = calculate_asymmetry(
        ds, wl_range=wl_range, rel_height=rel_height, int_width_mult=2.0
    )
    g_factor_max_results = find_g_factor_max(
        ds, wl_range=wl_range, fwhm_window="both", rel_height=rel_height
    )

    # Use the integration range from asymmetry analysis if available, otherwise use wl_range for LDLB ratio calculation
    if asymmetry_results and "integration_range" in asymmetry_results:
        int_range = asymmetry_results["integration_range"]
    else:
        int_range = wl_range

    ldlb_ratio_resutls = calculate_ldlb_area_ratio(ds, wl_range=int_range)

    return {
        "ds": ds,
        "corrected_abs_avg": corrected_abs_avg,
        "calc_baseline": calc_baseline,
        "peaks": peaks,
        "peak_analysis": asymmetry_results,
        "wl_range": wl_range,
        "rel_height": rel_height,
        "g_factor_max": g_factor_max_results["max_g_factor"],  # Current max g-factor
        "g_factor_max_wavelength": g_factor_max_results["max_wavelength"],
        "g_factor_max_range": g_factor_max_results["search_range"],
        "cd_area": ldlb_ratio_resutls["cd_area"],
        "ldlb_area": ldlb_ratio_resutls["ldlb_area"],
        "ldlb_area_ratio": ldlb_ratio_resutls["ldlb_ratio"],
    }


color_map = {
    "CD": "C0",
    "LDLB": "C1",
    "g-factor": "C2",
    "Abs": "C4",
    "Abs_corrected": "C4",
    "Abs_baseline": "C5",
}


def adjust_hsv(color, h_shift=0, s_shift=0, v_shift=0):
    """
    Adjust the HSV values of a given color.

    Parameters
    ----------
    color : str or tuple
        Color in any format recognized by matplotlib.
    h_shift : float
        Shift in hue (0 to 1).
    s_shift : float
        Shift in saturation (0 to 1).
    v_shift : float
        Shift in value (0 to 1).

    Returns
    -------
    tuple
        Adjusted color in RGB format.
    """
    rgb = mpl.colors.to_rgb(color)
    hsv = mpl.colors.rgb_to_hsv(rgb)
    hsv[0] = (hsv[0] + h_shift) % 1.0  # Hue shift with wrap-around
    hsv[1] = np.clip(hsv[1] + s_shift, 0, 1)  # Saturation shift with clipping
    hsv[2] = np.clip(hsv[2] + v_shift, 0, 1)  # Value shift with clipping

    # format as string for printing, .2f
    org_hsv_formatted = np.array2string(
        mpl.colors.rgb_to_hsv(rgb), precision=2, separator=", ", suppress_small=True
    )
    hsv_formatted = np.array2string(
        hsv, precision=2, separator=", ", suppress_small=True
    )
    # warning if clipping occurs, print original color and the adjusted hsv values
    if (hsv[1] + s_shift) < 0 or (hsv[1] + s_shift) > 1:
        print(
            f"Warning: Saturation shift clippinged for color {color}. Original HSV: {org_hsv_formatted}, Adjusted HSV: {hsv_formatted}"
        )
    if (hsv[2] + v_shift) < 0 or (hsv[2] + v_shift) > 1:
        print(
            f"Warning: Value shift resulted in clipping for color {color}. Original HSV: {org_hsv_formatted}, Adjusted HSV: {hsv_formatted}"
        )

    return mpl.colors.hsv_to_rgb(hsv)


# %%


def plot_cd_analysis(analysis_results, cd_abs_max=150, g_factor_abs_max=17e-4):
    """
    Visualize CD analysis results.
    """
    ds = analysis_results["ds"]
    corrected_abs_avg = analysis_results["corrected_abs_avg"]
    rel_height = analysis_results["rel_height"]
    wl_range = analysis_results["wl_range"]

    peak_analysis = analysis_results["peak_analysis"]
    ldlb_area_ratio = analysis_results["ldlb_area_ratio"]

    # Create figure with subplots
    fig, ((ax_cd, ax_g, ax_abs)) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(DOUBLE_COLUMN_WIDTH * 0.25, DOUBLE_COLUMN_WIDTH * 0.42),
        dpi=300,
        height_ratios=(1.25, 1, 1),
        constrained_layout=True,
    )

    # Plot CD data
    ax_cd.plot(ds["wavelength"], ds["gen_CD"], color=color_map["CD"])
    ax_cd.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax_ldlb = ax_cd.inset_axes([0.1, 0.095, 0.35, 0.35], zorder=10)
    ds["ldlb_CD"].sel(wavelength=slice(*wl_range)).plot.line(
        ax=ax_ldlb, color=color_map["LDLB"], linewidth=0.8
    )
    ax_ldlb.set_title(
        r"$R_{LDLB}$" + f"={ldlb_area_ratio:.1%}", y=1.0, pad=-9, fontsize="small"
    )
    ldlb_abs_max = np.abs(ds["ldlb_CD"].sel(wavelength=slice(*wl_range))).max()
    ax_ldlb.set_ylim(-ldlb_abs_max * 1.1, ldlb_abs_max * 1.1)
    ax_ldlb.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_ldlb.set_xlabel(None)
    ax_ldlb.set_ylabel(None)
    ax_ldlb.tick_params(axis="both", which="major", length=2, pad=1.5)
    # set bg alpha to 0.5

    # Plot g-factor data
    ax_g.plot(ds["wavelength"], ds["g_factor"] * 1e4, color=color_map["g-factor"])
    ax_g.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Plot absorption data
    ax_abs.plot(
        ds["wavelength"],
        corrected_abs_avg,
        color=color_map["Abs_corrected"],
    )

    if peak_analysis:
        uv_vis_peak = peak_analysis["uv_vis_peak"]
        peak_wavelength = uv_vis_peak["wavelength"]
        peak_width_height = uv_vis_peak["width_height"]
        two_hwhm = peak_analysis["integration_range"]
        hwhm = uv_vis_peak["hwhm"]
        fwhm_range = uv_vis_peak["fwhm_wl"]

        # Mark peak with vertical lines
        for ax in [ax_abs, ax_cd, ax_g]:
            ax.vlines(
                peak_wavelength,
                0,
                1,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                transform=ax.get_xaxis_transform(),
            )

        # Annotate peak on CD plot
        ax_cd.text(
            peak_wavelength + 2,
            0.82,
            f"UV-Vis peak\n{peak_wavelength} nm",
            color="dimgray",
            horizontalalignment="left",
            transform=ax_cd.get_xaxis_transform(),
            bbox=dict(
                facecolor=mpl.colors.to_rgba("white", 0.7), edgecolor="none", pad=0
            ),
        )

        # Mark half maximum with horizontal line
        # ax_abs.hlines(peak_width_height, fwhm_range[0], fwhm_range[1], color="red")

        for fwhm_wl in fwhm_range:
            ax_abs.annotate(
                "",
                xy=(peak_wavelength, peak_width_height),
                xytext=(fwhm_wl, peak_width_height),
                ha="center",
                va="center",
                fontsize="small",
                color="red",
                arrowprops=dict(
                    arrowstyle="<->",
                    color="red",
                    lw=0.5,
                ),
            )

        # Annotate HWHM values
        ax_abs.text(
            peak_wavelength - hwhm[0] / 2,
            peak_width_height - 0.06,
            f"{hwhm[0]:.1f} nm",
            color="red",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize="small",
        )

        ax_abs.text(
            peak_wavelength + hwhm[1] / 2,
            peak_width_height - 0.06,
            f"  {hwhm[1]:.1f} nm",
            color="red",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize="small",
        )

        ax_abs.text(
            peak_wavelength - 2,
            peak_width_height + 0.06,
            f"FW({(1 - rel_height):.1f}H)M",
            color="red",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize="small",
            bbox=dict(
                facecolor=mpl.colors.to_rgba("white", 0.7), edgecolor="none", pad=0
            ),
        )

        # Annotate 2*HWHM range
        annotate_y = 1.38  # Adjust this value to position the annotation higher

        ann_fwm = ax_abs.annotate(
            f"2×FW({(1 - rel_height):.1f}H)M\n{two_hwhm[0]:.0f}-{two_hwhm[1]:.0f} nm",
            xy=(two_hwhm[0], annotate_y),
            xytext=((two_hwhm[0] + two_hwhm[1]) / 2, annotate_y),
            ha="center",
            va="center",
            fontsize="small",
            color="red",
            xycoords="data",
            arrowprops=dict(
                arrowstyle="->,head_length=0.5, head_width=0.4",
                linestyle=":",
                color="red",
                lw=1.0,
            ),
            bbox=dict(
                facecolor=mpl.colors.to_rgba("white", 0.7), edgecolor="none", pad=0
            ),
        )

        ax_abs.annotate(
            "",
            xy=(two_hwhm[1], annotate_y),
            textcoords=ann_fwm,
            xytext=(1, 0.5),
            ha="center",
            va="center",
            fontsize="small",
            color="red",
            arrowprops=dict(
                arrowstyle="->,head_length=0.5, head_width=0.4",
                linestyle=":",
                color="red",
                lw=1.0,
            ),
        )

        for wl in two_hwhm:
            ax_abs.plot(
                [wl, wl],
                [0, 3.25],
                transform=ax_abs.get_xaxis_transform(which="grid"),
                color="red",
                ls=":",
                clip_on=False,
                zorder=-1,
            )

        # mark the g_factor_max_wavelength and g_factor_max on the g-factor plot
        g_factor_max = (
            ds["g_factor"]
            .sel(wavelength=analysis_results["g_factor_max_wavelength"])
            .values
        )

        ax_g_darkgreen = adjust_hsv(color_map["g-factor"], s_shift=0.1, v_shift=-0.2)

        ax_g.plot(
            analysis_results["g_factor_max_wavelength"],
            g_factor_max * 1e4,
            marker="o",
            color=ax_g_darkgreen,
        )

        ax_g.annotate(
            r"$\^g$" + ": max(|g-factor|)\n" + f"{np.abs(g_factor_max):.2e}",
            xy=(analysis_results["g_factor_max_wavelength"], g_factor_max * 1e4),
            xytext=(
                analysis_results["g_factor_max_wavelength"] - 32,
                g_factor_max * 1e4 + (20 if g_factor_max < 0 else -20),
            ),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color=ax_g_darkgreen,
                connectionstyle="angle,angleA=0,angleB=90",
            ),
            bbox=dict(
                boxstyle="round,pad=0.3",
                linewidth=1,
                edgecolor=ax_g_darkgreen,
                facecolor=mpl.colors.to_rgba(
                    adjust_hsv(color_map["g-factor"], s_shift=0.1), 0.5
                ),
            ),
            horizontalalignment="center",
            verticalalignment=("bottom" if g_factor_max < 0 else "top"),
        )

        ax_cd.fill_between(
            ds["wavelength"],
            ds["gen_CD"],
            where=(ds["wavelength"] >= (two_hwhm[0] - 1))
            & (ds["wavelength"] <= two_hwhm[1]),
            color=color_map["CD"],
            alpha=0.3,
        )

        ax_cd.text(
            485,
            75,
            r"$A_{|CD|}$" + f"={analysis_results['cd_area']:.0f}",
            color=adjust_hsv(color_map["CD"], v_shift=0.1, h_shift=-0.05),
            ha="center",
        )

        ax_ldlb.fill_between(
            ds["wavelength"],
            ds["ldlb_CD"],
            where=(ds["wavelength"] >= two_hwhm[0]) & (ds["wavelength"] <= two_hwhm[1]),
            color=color_map["LDLB"],
            alpha=0.3,
        )

        ax_ldlb.scatter(
            512,
            -3.5,
            color=color_map["LDLB"],
            marker="o",
            s=10,
            zorder=10,
        )

        ax_ldlb.annotate(
            r"$A_{|LDLB|}$" + f"={analysis_results['ldlb_area']:.0f}",
            xy=(511, -3.5),
            xycoords="data",
            xytext=(500, -390),
            textcoords=ax_cd.transData,
            arrowprops=dict(arrowstyle="-", color=color_map["LDLB"]),
            bbox=dict(
                boxstyle="round,pad=0.3",
                linewidth=1,
                edgecolor=color_map["LDLB"],
                facecolor=mpl.colors.to_rgba(color_map["LDLB"], 0.4),
            ),
            ha="center",
        )

    # Calculate Y-limit safely
    start_wl = min(wl_range)
    end_wl = max(wl_range)
    if ds["wavelength"][0] > ds["wavelength"][-1]:
        wl_slice = slice(end_wl, start_wl)
    else:
        wl_slice = slice(start_wl, end_wl)

    # Configure CD plot
    ax_cd.set_ylim(-cd_abs_max * 1.05, cd_abs_max * 1.05)
    ax_cd.set_ylabel("CD (mdeg)")

    # Set title with sample name in CD plot
    # sample_name = ds["sample_name"].values
    # ax_cd.set_title(f"{sample_name}", y=1.0, pad=-12, x=0.08)

    # Configure absorption plot
    # ax_abs.legend(loc="best", ncols=1, frameon=True, fontsize="small").get_frame().set_linewidth(0.0)

    subset_abs = corrected_abs_avg.sel(wavelength=wl_slice)
    if subset_abs.size > 0:
        y_max = subset_abs.max().values
    else:
        y_max = 1.0

    ax_abs.set_ylim(-0.14, max(y_max * 1.35, 1.1))
    ax_abs.set_xlabel("Wavelength (nm)")
    ax_abs.set_ylabel("Absorption (a.u.)")

    # Configure g-factor plot
    ax_g.set_xlim(wl_range[0], wl_range[1])
    ax_g.set_ylim(-g_factor_abs_max * 1.1 * 1e4, g_factor_abs_max * 1.1 * 1e4)
    offset_text_y = ax_g.yaxis.get_offset_text()
    offset_text_y.set_position((-0.125, 0))
    ax_g.set_ylabel("g-factor (x10⁴)")

    # Remove space between subplots
    fig.get_layout_engine().set(hspace=0, h_pad=0)

    return fig


metadata = pd.DataFrame(
    {k: v.values for k, v in ds.coords.items() if k not in ds.dims},
    index=ds["sample_id"].values,
)

metadata = metadata.sort_values(by="sample_name", key=natsort_keygen())
sorted_ids = (
    metadata.groupby("config_hash")["sample_name"]
    .transform("first")
    .sort_values(kind="stable")
    .index.tolist()
)

wl_range = (450, 538)

# Results list for DataFrame
cd_results_list = []

ds = ds.sortby(["wavelength"])
cd_abs_max = np.abs(ds["gen_CD"].sel(wavelength=slice(*wl_range))).max()
g_factor_abs_max = np.abs(ds["g_factor"].sel(wavelength=slice(*wl_range))).max()

# Create and save the plot loop over samples in the ds
for id in sorted_ids[:1]:
    sample_ds = ds.sel(sample_id=id)
    save_path = Path(SCRIPT_DIR).parent / "manuscript_assets/2a_cd.svg"

    # Process - Now using default rel_height=0.7
    results = process_cd_data(
        sample_ds, wl_range=wl_range, peak_search_range=(480, 520), rel_height=0.7
    )

    # Plot
    fig = plot_cd_analysis(
        results,
        cd_abs_max=cd_abs_max,
        g_factor_abs_max=g_factor_abs_max,
    )
    # plt.close(fig)

    fig.suptitle("a)", x=0.0, y=1, horizontalalignment="left", verticalalignment="top")
    fig.align_labels()

    # Save figure if path provided
    if save_path:
        print(f"  Figure saved to: {save_path}")
        fig.savefig(save_path, dpi=300)


######################################### Old plots ##############################################
# %%
plt.rcdefaults()
plt.rcParams.update(params_basic)


def plot_cd_analysis_3_panel(analysis_results, cd_abs_max=150, g_factor_abs_max=17e-4):
    """
    Visualize CD analysis results.
    """
    ds = analysis_results["ds"]
    corrected_abs_avg = analysis_results["corrected_abs_avg"]
    calc_baseline = analysis_results["calc_baseline"]
    rel_height = analysis_results["rel_height"]
    wl_range = analysis_results["wl_range"]

    peak_analysis = analysis_results["peak_analysis"]
    ldlb_area_ratio = analysis_results["ldlb_area_ratio"]

    # Create figure with subplots
    fig, ((ax_cd, ax_g, ax_abs)) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(2.0 * figure_zoom_multiplier, 4 * figure_zoom_multiplier),
        dpi=300,
        height_ratios=(1.25, 1, 1.5),
    )

    # Plot CD data
    ax_cd.plot(ds["wavelength"], ds["gen_CD"], color=color_map["CD"])
    ax_cd.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax_ldlb = ax_cd.inset_axes([0.1, 0.095, 0.35, 0.35], zorder=10)
    ds["ldlb_CD"].sel(wavelength=slice(*wl_range)).plot.line(
        ax=ax_ldlb, color=color_map["LDLB"], linewidth=0.8
    )
    ax_ldlb.set_title(
        r"$R_{LDLB}$" + f"={ldlb_area_ratio:.1%}", y=1.0, pad=-9, fontsize="small"
    )
    ldlb_abs_max = np.abs(ds["ldlb_CD"].sel(wavelength=slice(*wl_range))).max()
    ax_ldlb.set_ylim(-ldlb_abs_max * 1.1, ldlb_abs_max * 1.1)
    ax_ldlb.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_ldlb.set_xlabel(None)
    ax_ldlb.set_ylabel(None)
    ax_ldlb.tick_params(axis="both", which="major", length=2, pad=1.5)
    # set bg alpha to 0.5

    # Plot g-factor data
    ax_g.plot(ds["wavelength"], ds["g_factor"] * 1e4, color=color_map["g-factor"])
    ax_g.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Plot absorption data
    ax_abs.plot(ds["wavelength"], ds["abs_avg"], label="Abs", color=color_map["Abs"])
    ax_abs.plot(
        ds["wavelength"],
        corrected_abs_avg,
        color=color_map["Abs_corrected"],
        label="Corrected Abs",
    )
    ax_abs.plot(
        ds["wavelength"],
        calc_baseline,
        color=color_map["Abs_baseline"],
        linestyle="--",
        label="Baseline",
    )

    if peak_analysis:
        uv_vis_peak = peak_analysis["uv_vis_peak"]
        peak_wavelength = uv_vis_peak["wavelength"]
        peak_width_height = uv_vis_peak["width_height"]
        two_hwhm = peak_analysis["integration_range"]
        hwhm = uv_vis_peak["hwhm"]
        fwhm_range = uv_vis_peak["fwhm_wl"]

        # Mark peak with vertical lines
        for ax in [ax_abs, ax_cd, ax_g]:
            ax.vlines(
                peak_wavelength,
                0,
                1,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                transform=ax.get_xaxis_transform(),
            )

        # Annotate peak on CD plot
        ax_cd.text(
            peak_wavelength + 1,
            0.82,
            f"UV-Vis peak\n{peak_wavelength} nm",
            color="dimgray",
            horizontalalignment="left",
            fontsize="small",
            transform=ax_cd.get_xaxis_transform(),
        )

        # Mark half maximum with horizontal line
        ax_abs.hlines(peak_width_height, fwhm_range[0], fwhm_range[1], color="red")

        # Annotate HWHM values
        ax_abs.text(
            peak_wavelength - hwhm[0] / 2,
            peak_width_height - 0.08,
            f"{hwhm[0]:.1f} nm",
            color="red",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize="small",
            # fontsize=8,
        )

        ax_abs.text(
            peak_wavelength + hwhm[1] / 2,
            peak_width_height - 0.08,
            f"  {hwhm[1]:.1f} nm",
            color="red",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize="small",
            # fontsize=8,
        )

        ax_abs.text(
            peak_wavelength - 2,
            peak_width_height + 0.08,
            f"FW({(1 - rel_height):.1f}H)M",
            color="red",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize="small",
            # fontsize=10,
        )

        # Annotate 2*HWHM range

        ax_abs.annotate(
            f"2×FW({(1 - rel_height):.1f}H)M\n{two_hwhm[0]:.0f}-{two_hwhm[1]:.0f} nm",
            xy=((two_hwhm[0] + two_hwhm[1]) / 2, 1.75),
            xytext=(515, 2.4),
            ha="center",
            fontsize="small",
            color="red",
            arrowprops=dict(
                arrowstyle="-[,widthB=9.4,lengthB=1",
                linestyle=":",
                color="red",
                lw=1.0,
                connectionstyle="angle,angleA=0,angleB=90,rad=10",
            ),
        )

        # Highlight 2*HWHM range on absorption plot
        # ax_abs.fill_between(
        #     ds["wavelength"][
        #         (ds["wavelength"] >= two_hwhm[0]) & (ds["wavelength"] <= two_hwhm[1])
        #     ],
        #     0,
        #     1,
        #     color=color_map["Abs_corrected"],
        #     alpha=0.3,
        #     transform=ax_abs.get_xaxis_transform(),
        # )

        for wl in two_hwhm:
            ax_abs.plot(
                [wl, wl],
                [0, 2.6],
                transform=ax_abs.get_xaxis_transform(which="grid"),
                color="red",
                ls=":",
                clip_on=False,
                zorder=-1,
            )

        # g_factor_max_range = analysis_results["g_factor_max_range"]
        # ax_g.fill_between(
        #     ds["wavelength"][
        #         (ds["wavelength"] >= g_factor_max_range[0])
        #         & (ds["wavelength"] <= g_factor_max_range[1])
        #     ],
        #     0,
        #     1,
        #     color=color_map["g-factor"],
        #     alpha=0.3,
        #     transform=ax_g.get_xaxis_transform(),
        # )

        # mark the g_factor_max_wavelength and g_factor_max on the g-factor plot
        g_factor_max = (
            ds["g_factor"]
            .sel(wavelength=analysis_results["g_factor_max_wavelength"])
            .values
        )
        ax_g.plot(
            analysis_results["g_factor_max_wavelength"],
            g_factor_max * 1e4,
            marker="o",
            color="darkgreen",
        )

        ax_g.annotate(
            r"$\^g$" + ": max(|g-factor|)\n" + f"{np.abs(g_factor_max):.2e}",
            xy=(analysis_results["g_factor_max_wavelength"], g_factor_max * 1e4),
            xytext=(
                analysis_results["g_factor_max_wavelength"] - 32,
                g_factor_max * 1e4 + (20 if g_factor_max < 0 else -20),
            ),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color="darkgreen",
                connectionstyle="angle,angleA=0,angleB=90",
            ),
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=mpl.colors.to_rgba("white", 0.7)
            ),
            horizontalalignment="center",
            verticalalignment=("bottom" if g_factor_max < 0 else "top"),
        )

        ax_cd.fill_between(
            ds["wavelength"],
            ds["gen_CD"],
            where=(ds["wavelength"] >= (two_hwhm[0] - 1))
            & (ds["wavelength"] <= two_hwhm[1]),
            color=color_map["CD"],
            alpha=0.3,
        )

        ax_cd.text(
            485,
            75,
            r"$A_{|CD|}$" + f"={analysis_results['cd_area']:.0f}",
            color="mediumblue",
            ha="center",
        )

        ax_ldlb.fill_between(
            ds["wavelength"],
            ds["ldlb_CD"],
            where=(ds["wavelength"] >= two_hwhm[0]) & (ds["wavelength"] <= two_hwhm[1]),
            color=color_map["LDLB"],
            alpha=0.3,
        )

        ax_ldlb.annotate(
            r"$A_{|LDLB|}$" + f"={analysis_results['ldlb_area']:.0f}",
            xy=(510, -3.5),
            xycoords="data",
            xytext=(500, -390),
            textcoords=ax_cd.transData,
            # color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange"),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=mpl.colors.to_rgba(color_map["LDLB"], 0.8),
            ),
            ha="center",
            # va="center",
            # fontsize=8,
        )

    # abs_g_factor_max = analysis_results["g_factor_max"]
    # # Add a compact inset bar plot for the two summary metrics.
    # if g_factor_max < 0:
    #     inset_ax = ax_g.inset_axes([0.05, 0.85, 0.9, 0.1])
    # else:
    #     inset_ax = ax_g.inset_axes([0.05, 0.15, 0.9, 0.1])

    # inset_ax.barh(
    #     [0],
    #     [np.clip(abs_g_factor_max / 0.0015, 0.0, 1.0)],
    #     color="tab:cyan",
    #     alpha=0.75,
    #     height=0.55,
    # )
    # inset_ax.set_yticks([0], labels=[f"g-factor max: {abs_g_factor_max:.2e}"])
    # inset_ax.xaxis.set_visible(False)
    # inset_ax.tick_params(axis="y", direction="in", pad=-120, length=0)
    # inset_ax.set_yticklabels(inset_ax.get_yticklabels(), va="center")
    # for spine in inset_ax.spines.values():
    #     spine.set_visible(False)
    # inset_ax.set_facecolor(mpl.colors.to_rgba("lightgray", alpha=0.7))

    # Calculate Y-limit safely
    start_wl = min(wl_range)
    end_wl = max(wl_range)
    if ds["wavelength"][0] > ds["wavelength"][-1]:
        wl_slice = slice(end_wl, start_wl)
    else:
        wl_slice = slice(start_wl, end_wl)

    # Configure CD plot
    ax_cd.set_ylim(-cd_abs_max * 1.05, cd_abs_max * 1.05)
    ax_cd.set_ylabel("CD (mdeg)")

    # Set title with sample name in CD plot
    sample_name = ds["sample_name"].values
    ax_cd.set_title(f"{sample_name}", y=1.0, pad=-12, x=0.08)

    # Configure absorption plot
    ax_abs.legend(ncols=1, frameon=True).get_frame().set_linewidth(0.0)

    subset_abs = ds["abs_avg"].sel(wavelength=wl_slice)
    if subset_abs.size > 0:
        y_max = subset_abs.max().values
    else:
        y_max = 1.0

    ax_abs.set_ylim(-0.14, max(y_max * 1.3, 1.1))
    ax_abs.set_xlabel("Wavelength (nm)")
    ax_abs.set_ylabel("Absorption (a.u.)")

    # Configure g-factor plot
    ax_g.set_xlim(wl_range[0], wl_range[1])
    ax_g.set_ylim(-g_factor_abs_max * 1.1 * 1e4, g_factor_abs_max * 1.1 * 1e4)
    offset_text_y = ax_g.yaxis.get_offset_text()
    offset_text_y.set_position((-0.125, 0))
    ax_g.set_ylabel("g-factor (x10⁴)")

    # Remove space between subplots
    fig.subplots_adjust(hspace=0)

    return fig


cd_plots_dir = Path(SCRIPT_DIR).parent / "manuscript_assets/cd_plots"
# create directory if not exists
cd_plots_dir.mkdir(parents=True, exist_ok=True)

metadata = pd.DataFrame(
    {k: v.values for k, v in ds.coords.items() if k not in ds.dims},
    index=ds["sample_id"].values,
)

metadata = metadata.sort_values(by="sample_name", key=natsort_keygen())
sorted_ids = (
    metadata.groupby("config_hash")["sample_name"]
    .transform("first")
    .sort_values(kind="stable")
    .index.tolist()
)

wl_range = (450, 530)

# Results list for DataFrame
cd_results_list = []

ds = ds.sortby(["wavelength"])
cd_abs_max = np.abs(ds["gen_CD"].sel(wavelength=slice(*wl_range))).max()
g_factor_abs_max = np.abs(ds["g_factor"].sel(wavelength=slice(*wl_range))).max()

# Create and save the plot loop over samples in the ds
for id in sorted_ids[:1]:
    sample_ds = ds.sel(sample_id=id)
    save_path = Path(cd_plots_dir, f"{sample_ds['sample_name'].values}_3_panel.png")

    # Process - Now using default rel_height=0.7
    results = process_cd_data(
        sample_ds, wl_range=wl_range, peak_search_range=(480, 520), rel_height=0.7
    )

    # Plot
    fig = plot_cd_analysis_3_panel(
        results,
        cd_abs_max=cd_abs_max,
        g_factor_abs_max=g_factor_abs_max,
    )
    # plt.close(fig)

    fig.suptitle(
        "a)", x=0.01, y=0.995, horizontalalignment="left", verticalalignment="top"
    )
    fig.align_labels()
    fig.set_constrained_layout(True)

    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"  Figure saved to: {save_path}")
        fig.savefig(save_path.with_suffix(".svg"), dpi=300)

# %%
