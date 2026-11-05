# %%
import os
import sys


from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from pybaselines import Baseline


from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH  # noqa: E402
from configs.paths import batch_dir  # noqa: E402
from tricontour_utils import (  # noqa: E402
    create_faceted_tricontour,
    load_shap_cache,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))  # noqa: E402


########## set up plotting parameters ##########
plt.rcdefaults()
# plt.rcParams.update(params_basic)
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    }
)
################################################

cmap = plt.get_cmap("RdBu_r")


def label(ds_sample):
    temp = ds_sample.annealing_temp.values
    conc = ds_sample.concentration.values
    score = ds_sample.SHAP_Combined_Env.values
    try:
        return f"{temp:.0f}°C, {conc:.2f} M ({score:.2f})"
    except TypeError:
        return "Ref"


# %%

BATCH_32_DIR = batch_dir(32)
RECIPE_JSON_PATH = BATCH_32_DIR / "total_correlation_recipes.json"

########## shared recipe dataframe (CD, XRD, photo, and heatmap all key off this) ##########
# set True to drop sample 2 (SI-1) from every plot below, not just the heatmap
EXCLUDE_SI1_SAMPLE = True

annotation_mappings = {
    # "name": {1: "A", 2: "SI-1", 3: "B", 4: "C", 5: "D", 6: "E", 7: "F", 8: "G"},
    "name": {1: "D", 2: "SI-1", 3: "C", 4: "B", 5: "A", 6: "G", 7: "F", 8: "E"},
    "text_color": {1: "w", 2: "k", 3: "k", 4: "w", 5: "w", 6: "w", 7: "k", 8: "k"},
}


def get_shortest_str(x, g_precision=3, f_precision=2):
    """Format a number with whichever of general/fixed notation is shorter."""
    if not isinstance(x, (int, float)) or pd.isna(x):
        return str(x)
    g_format = f"{x:.{g_precision}g}"
    f_format = f"{x:.{f_precision}f}"
    return g_format if len(g_format) <= len(f_format) else f_format


df_samples = pd.read_json(RECIPE_JSON_PATH)
df_samples["sample_number"] = (df_samples.index + 1).astype(str)
df_samples["sample_name"] = (
    df_samples["sample_number"].astype(int).map(annotation_mappings["name"])
)
df_samples["annotate_text_color"] = (
    df_samples["sample_number"].astype(int).map(annotation_mappings["text_color"])
)
df_samples["sample_text"] = df_samples["sample_name"]
df_samples["sample_text_with_SHAP"] = (
    df_samples["sample_name"]
    + "("
    + df_samples["SHAP_Combined_Env"].map(get_shortest_str)
    + ")"
)

if EXCLUDE_SI1_SAMPLE:
    df_samples = df_samples[~df_samples["sample_text"].str.startswith("SI-")]


def align_recipe_to_ds(ds, recipe, sample_number_col="sample_number"):
    """Restrict `ds` to the sample_numbers present in `recipe`, then assign every
    other column of `recipe` as a coordinate on the sample_number dimension.

    This is how CD/XRD/photo plots stay in sync with df_samples: a sample missing
    from `recipe` (e.g. SI-1 when EXCLUDE_SI1_SAMPLE=True) is dropped from `ds`
    too, the same way the heatmap only annotates what's in df_samples.
    """
    keep = [s for s in ds.sample_number.values if s in set(recipe[sample_number_col])]
    ds = ds.sel(sample_number=keep)

    aligned = recipe.set_index(sample_number_col).reindex(keep)
    coords_to_assign = {
        col: ("sample_number", aligned[col].values) for col in aligned.columns
    }
    return ds.assign_coords(**coords_to_assign)


##############################################################################################

# %%
cd_dir = BATCH_32_DIR / "CD"
txt_files = list(cd_dir.glob("*.txt"))

parsed_dfs = []
for txt_file in txt_files:
    df = pd.read_csv(
        txt_file, sep="\t", skiprows=3, names=["wavelength", "CD", "HT", "abs"]
    )
    side, sample_number = txt_file.stem.split("-")
    df["side"] = side
    df["sample_number"] = sample_number
    parsed_dfs.append(df)

raw_cd_df = pd.concat(parsed_dfs, ignore_index=True)

cd_df = pd.pivot_table(
    raw_cd_df,
    index=["wavelength", "sample_number"],
    columns=["side"],
    values=["CD", "abs"],
)

# convert to xarray for easier manipulation
cd_ds = cd_df.to_xarray()


cd_ds["cd_gen"] = (cd_ds[("CD", "front")] + cd_ds[("CD", "back")]) / 2
cd_ds["cd_ldlb"] = (cd_ds[("CD", "front")] - cd_ds[("CD", "back")]) / 2
cd_ds["abs_avg"] = (cd_ds[("abs", "front")] + cd_ds[("abs", "back")]) / 2

cd_ds["g_factor"] = np.divide(
    cd_ds["cd_gen"],
    (cd_ds["abs_avg"]) * 32980,
    out=np.zeros_like(cd_ds["cd_gen"]),
    where=(cd_ds["abs_avg"] > 0.0),
)


# %%

# restrict cd_ds to samples present in df_samples and attach recipe columns as coords
ds = align_recipe_to_ds(cd_ds, df_samples)


# %%
fig, axes = plt.subplots(
    ncols=3,
    nrows=3,
    figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.6),
    sharex="col",
    sharey="row",
    constrained_layout=True,
    gridspec_kw={"height_ratios": [1.5, 1, 0.8]},
)

# group by a_site
for (a_site, grouped), ax_row in zip(ds.groupby("a_site"), axes.T):
    ax_cd, ax_g, ax_abs = ax_row
    ax_cd.set_title(a_site, pad=-10, y=1, x=0.05, ha="left")
    grouped = grouped.sortby("SHAP_Combined_Env", ascending=False)
    colors = [0.9, 0.75, 0.1] if len(grouped.sample_number.values) == 3 else [0.9, 0.1]

    handles = []
    labels = []

    for sample_number, color in zip(grouped.sample_number.values, colors):
        ds_sample = grouped.sel(sample_number=sample_number, wavelength=slice(450, 550))
        color = cmap(color)
        label_str = str(ds_sample.sample_text.values)

        da_cd = ds_sample["cd_gen"]
        (cd_line,) = ax_cd.plot(
            da_cd.wavelength, da_cd.values, label=label_str, color=color
        )

        da_g = ds_sample["g_factor"]
        ax_g.plot(da_g.wavelength, da_g.values * 10e4, label=label_str, color=color)

        da_abs = ds_sample["abs_avg"]
        ax_abs.plot(da_abs.wavelength, da_abs.values, label=label_str, color=color)

        da_ldlb = ds_sample["cd_ldlb"]
        (ldlb_line,) = ax_cd.plot(
            da_ldlb.wavelength,
            da_ldlb.values,
            label=label_str,
            color=color,
            ls="--",
            # alpha=0,
            linewidth=0.5,
        )

        ax_g.axhline(0, ls="--", color="gray")
        ax_cd.axhline(0, ls="--", color="gray")

        handles.append((cd_line, ldlb_line))
        labels.append(label_str)

    # additional legend for the Gen/LDLB lines, showing the dashed line style for LDLB
    linestyle_legend = [
        plt.Line2D([0], [0], color="black", lw=1, ls="-"),
        plt.Line2D([0], [0], color="black", lw=1, ls="--"),
    ]
    linestyle_labels = ["Gen", "LDLB"]

    legend = ax_cd.legend(
        handles,
        labels,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        handlelength=3,
        loc="upper right",
    )

    ax_cd.add_artist(legend)

    # linestyle legend for Gen/LDLB at the bottom of the CD panel
    ax_cd.legend(
        linestyle_legend, linestyle_labels, loc="lower right", fontsize="small", ncol=2
    )

axes[0, 0].set_ylim(-60, 130)
axes[2, 0].set_ylim(-0.2, None)

axes[0, 0].set_ylabel("CD (mdeg)")
axes[1, 0].set_ylabel(r"g-factor ($\times 10^{-4}$)")
axes[2, 0].set_ylabel("Abs")

fig.supxlabel("Wavelength (nm)")
fig.align_ylabels()
fig.suptitle(
    "b", x=0.0, y=1.0, va="top", ha="left", fontsize="large", fontweight="bold"
)

fig.get_layout_engine().set(hspace=0.0, h_pad=0.02, w_pad=0.02)
fig.savefig(
    Path("../manuscript_assets/verification", "5b_cd_g_abs.svg"),
    dpi=300,
)
# %%


########################## XRD ##########################

# ref_metadata: maps ref sample_number → {a_site, label}
ref_metadata = {
    "ref_R-ClMBA2PbI4-ic20627": {"a_site": "ClMBAI", "label": "Ref"},
    "ref_S-MBA2PbI4": {"a_site": "MBAI", "label": "Ref"},
    # "ref_S-MBAPbI3-1D-607737": {"a_site": "MBAI", "label": "Ref (1D)"},
}


def xrd_baseline(
    da: xr.DataArray, method="arpls", flatten_only=True, **kwargs
) -> tuple[xr.DataArray, xr.DataArray]:
    """Baseline correction for XRD DataArrays using pybaselines.

    Defaults to arpls (lam=1e5), which handles the broad amorphous
    background and air-scatter ramp common in thin-film PXRD.
    Good alternatives: iarpls (lam=1e5), drpls (lam=1e5, eta=0.5),
    snip (max_half_window=40).
    """
    fitter = Baseline()
    baseline, _ = getattr(fitter, method)(da.values, **kwargs)

    if flatten_only:
        # flatten the baseline but not shift the intensity values
        corrected = da.values - baseline + np.min(baseline)
    else:
        corrected = da.values - baseline

    corrected_da = xr.DataArray(corrected, coords=da.coords, dims=da.dims)
    baseline_da = xr.DataArray(baseline, coords=da.coords, dims=da.dims)
    return corrected_da, baseline_da


xrd_dir = BATCH_32_DIR / "rigaku"
csv_files = list(xrd_dir.glob("*.csv"))

parsed_dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file, sep=",", skiprows=2, names=["two_theta", "intensity"])
    df["sample_number"] = csv_file.stem.split("_")[-1]
    parsed_dfs.append(df)


ref_dir = BATCH_32_DIR / "xrd_ref"
ref_files = list(ref_dir.glob("*.txt"))

for ref_file in ref_files:
    df = pd.read_csv(ref_file, sep="\s+")
    df["sample_number"] = f"ref_{ref_file.stem}"
    df = df.rename(columns={df.columns[7]: "two_theta", df.columns[8]: "intensity"})
    parsed_dfs.append(df[["sample_number", "two_theta", "intensity"]])


raw_xrd_df = pd.concat(parsed_dfs, ignore_index=True)

# %%

xrd_df = pd.pivot_table(
    raw_xrd_df,
    index=["two_theta", "sample_number"],
    values=["intensity"],
)

# convert to xarray for easier manipulation
xrd_ds = xrd_df.to_xarray()

# %%

# add reference samples (not present in df_samples/the recipe json) as their own rows,
# then align_recipe_to_ds keeps df_samples' own EXCLUDE_SI1_SAMPLE filtering for the rest
ref_samples = list(filter(lambda x: x.startswith("ref_"), xrd_ds.sample_number.values))

ref_recipe_df = pd.DataFrame(
    np.nan,
    index=range(len(ref_samples)),
    columns=df_samples.columns,
)
ref_recipe_df["sample_number"] = ref_samples
ref_recipe_df["SHAP_Combined_Env"] = -np.inf
ref_recipe_df["a_site"] = [ref_metadata.get(s, {}).get("a_site") for s in ref_samples]

recipe_with_refs = pd.concat([df_samples, ref_recipe_df], ignore_index=True)

ds = align_recipe_to_ds(xrd_ds, recipe_with_refs)

# %%

peaks = {
    "ClMBAI": {
        5.51: "(002)",
        11.04: "(004)",
        14.10: "(110)",
        16.60: "(006)",
        22.10: "(008)",
    },
    "MBAI": {
        6.16: "(002)",
        12.37: "(004)",
        13.84: "(110)",
        18.56: "(006)",
        19.21: "(020)",
        24.84: "(008)",
        # 8.27: "(002) 1D",
        # 10.95: "(011) 1D",
        # 16.0: "(013) 1D",
        # 16.7: "(004) 1D",
        # 23.52: "(105) 1D",
    },
    "MeOMBAI": {
        5.28: "(002)",
        10.56: "(004)",
        15.84: "(006)",
        21.12: "(008)",
        26.45: "(0010)",
    },
    "All": {
        13.05: "*",
    },
}

subplot_mode = "col"  # "row" or "col"

if subplot_mode == "row":
    subplot_conf = (3, 1)
elif subplot_mode == "col":
    subplot_conf = (1, 3)
else:
    raise ValueError(f"Invalid subplot_mode: {subplot_mode}")

fig, axes = plt.subplots(
    *subplot_conf,
    sharex="col",
    sharey="row",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.5, DOUBLE_COLUMN_WIDTH)
    if subplot_mode == "row"
    else (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.35),
    constrained_layout=True,
)

# group by a_site
for (a_site, grouped), ax in zip(ds.groupby("a_site"), axes):
    if subplot_mode == "row":
        ax.set_title(a_site, pad=-14, y=1, x=0.02, ha="left")
    else:
        ax.set_title(a_site, pad=-10, y=1)

    grouped = grouped.sortby("SHAP_Combined_Env", ascending=False)

    all_samples = grouped.sample_number.values
    non_ref_samples = [s for s in all_samples if not str(s).startswith("ref_")]
    ref_samples_in_group = [s for s in all_samples if str(s).startswith("ref_")]

    colors = [0.9, 0.75, 0.1] if len(non_ref_samples) == 3 else [0.9, 0.1]

    lines = []
    for i, (sample_number, color) in enumerate(zip(non_ref_samples, colors)):
        ds_sample = grouped.sel(sample_number=sample_number)
        color = cmap(color)
        label_str = str(ds_sample.sample_text.values)

        da_xrd = ds_sample["intensity"].dropna(dim="two_theta")
        da_xrd_corrected, _ = xrd_baseline(da_xrd, method="snip", max_half_window=50)

        da_xrd_corrected = (
            da_xrd_corrected + 1e3
        )  # background suppression for log scale
        da_relative = da_xrd_corrected / da_xrd_corrected.max()

        da_relative = da_relative.sel(two_theta=slice(3.2, 27))
        offset = (
            6 ** (len(non_ref_samples) - i - 1) / da_relative.min().values
        )  # scale each curve for visibility

        if not len(ref_samples_in_group):
            offset *= 0.2  # if no ref samples, scale down the top curve to make space for title

        (line,) = ax.plot(
            da_relative.two_theta,
            da_relative.values * offset,
            label=label_str,
            color=color,
        )
        lines.append(line)

    ax.set_yscale("log")

    y_min, y_max = ax.get_ylim()
    legend = ax.legend(handles=lines, loc="upper right")
    ax.set_ylim(None, y_max * 3.5)  # make space for title

    # get linexy data, find max y for each x, and plot the peak labels at those positions
    line_data = [line.get_xydata() for line in lines]
    line_x = line_data[0][:, 0]  # assume all lines have the same x values
    line_max_y = np.max([data[:, 1] for data in line_data], axis=0)

    peaks_to_plot = peaks.get(a_site, {}) | peaks.get("All", {})

    for peak_pos, peak_label in peaks_to_plot.items():
        # find the max y at peak_pos +/- 0.5 degrees
        mask = (line_x >= peak_pos - 0.3) & (line_x <= peak_pos + 0.3)
        if not np.any(mask):
            y = np.interp(peak_pos, line_x, line_max_y)
        else:
            y = np.max(line_max_y[mask])

        if (
            peak_label in peaks.get("All", {}).values()
            or len(ref_samples_in_group) == 0
        ):
            color = "black"
        elif "1D" in peak_label:
            color = "C4"
        else:
            color = "black"

        ax.text(
            peak_pos,
            y * 1.5,
            peak_label,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize="medium" if peak_label == "*" else "small",
            color=color,
        )

    if len(ref_samples_in_group) > 0:
        ax.add_artist(legend)
        ref_dynamic_range = 8
        # ref_ymax = y_min * 1.8
        ref_ymax = 1.47
        ref_ymin = ref_ymax / ref_dynamic_range
        print(
            f"ref_ymin: {ref_ymin}, ref_ymax: {ref_ymax}, y_min: {y_min}, y_max: {y_max}"
        )
        ax.set_ylim(ref_ymin * 0.6, None)
        ax.axhline(ref_ymin, lw=0.5, color="black")

        # order ref_samples_in_group by ref_metadata.keys() to ensure consistent color assignment
        ref_samples_in_group = sorted(
            ref_samples_in_group, key=lambda s: list(ref_metadata.keys()).index(s)
        )

        ref_bars = []
        for j, (sample_number, color) in enumerate(
            zip(ref_samples_in_group, ["black", "C4"])
        ):
            ds_sample = grouped.sel(sample_number=sample_number)
            label_str = ref_metadata.get(str(sample_number), {}).get(
                "label", str(sample_number)
            )

            da_xrd = ds_sample["intensity"].dropna(dim="two_theta")
            da_relative = (da_xrd / da_xrd.max()) * ref_ymax
            da_relative = da_relative.sel(two_theta=slice(3.2, 27))

            bar = ax.bar(
                da_relative.two_theta,
                da_relative.values,
                bottom=ref_ymin,
                label=label_str,
                width=0.1,
                color=color,
            )
            ref_bars.append(bar)

        legend_y = ref_ymax * 0.7

        if len(ref_bars) > 1:
            ax.legend(
                [tuple(ref_bars)],
                ["Ref 2D/1D"],
                numpoints=1,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
                bbox_to_anchor=(1, legend_y),
                loc="upper right",
                bbox_transform=ax.get_yaxis_transform(),
            )

        else:
            ax.legend(
                handles=ref_bars,
                fontsize="small",
                bbox_to_anchor=(1, legend_y),
                loc="upper right",
                bbox_transform=ax.get_yaxis_transform(),
            )

    ax.tick_params(axis="both", which="major", length=4, width=0.5)
    ax.minorticks_on()

if subplot_mode == "row":
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.0, w_pad=0.02)
else:
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.02, w_pad=0.02)


fig.supxlabel(r"$2\theta$ (°)")
fig.supylabel("Relative Intensity (a.u.)")

fig.savefig(
    Path("../manuscript_assets/si", "5_xrd.png"),
    dpi=300,
)

# %%

########################## Photo ##########################

# a coordinate-only dataset, restricted/aligned to df_samples, so we can reuse
# a_site/sample_text for titles the same way the CD and XRD plots do
# note: if EXCLUDE_SI1_SAMPLE is True, sample 2's grid position below will be
# missing from ds and the verification plot loop will raise a KeyError on it
photo_coord_ds = xr.Dataset(coords={"sample_number": df_samples["sample_number"]})
ds = align_recipe_to_ds(photo_coord_ds, df_samples)

photo_path = BATCH_32_DIR / "photo" / "DSC_0001.jpg"

# ROI enclosing the full 2x4 sample grid, in original-image pixel coordinates.
# Edit these to adjust the overall grid crop.
roi_x, roi_y, roi_w, roi_h = 1765, 650, 2690, 1347

sample_grid = [[1, 2, 3, 4], [5, 6, 7, 8]]
n_rows = len(sample_grid)
n_cols = len(sample_grid[0])
cell_w = roi_w / n_cols
cell_h = roi_h / n_rows


def fine_crop_box(sample_number, dx_left=120, dx_right=-15, dy_top=110, dy_bottom=-10):
    """Return a (left, top, right, bottom) box for one sample, nudged from its
    even grid-division box by the given pixel offsets.

    Use this later to trim blank margins left over from dividing the ROI
    evenly by 8 -- e.g. fine_crop_box(3, dx_left=20, dy_top=-15).
    """
    for row_idx, row in enumerate(sample_grid):
        if sample_number in row:
            col_idx = row.index(sample_number)
            break
    else:
        raise ValueError(f"sample_number {sample_number} not in sample_grid")

    left = roi_x + col_idx * cell_w
    top = roi_y + row_idx * cell_h
    right = left + cell_w
    bottom = top + cell_h

    return (
        left + dx_left,
        top + dy_top,
        right + dx_right,
        bottom + dy_bottom,
    )


photo_img = Image.open(photo_path)

sample_photos = {}
for row in sample_grid:
    for sample_number in row:
        box = fine_crop_box(sample_number)
        sample_photos[str(sample_number)] = photo_img.crop(box)


# Verification plot: show all 8 crops in their grid layout to confirm the ROI
# and per-sample boxes line up with the actual vials before using them further.
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(DOUBLE_COLUMN_WIDTH * 0.5, DOUBLE_COLUMN_WIDTH * 0.25),
    constrained_layout=True,
)
for row_idx, row in enumerate(sample_grid):
    for col_idx, sample_number in enumerate(row):
        ax = axes[row_idx, col_idx]
        ax.imshow(sample_photos[str(sample_number)])

        if str(sample_number) not in ds.sample_number.values:
            ax.set_title(f"sample {sample_number}\n(excluded)")
        else:
            ds_sample = ds.sel(sample_number=str(sample_number))
            a_site = ds_sample.a_site.values
            ax.set_title(f"{a_site}: {ds_sample.sample_text.values}")

        ax.set_xticks([])
        ax.set_yticks([])

# turn of spines for all axes
for ax in axes.flatten():
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.savefig(
    Path("../manuscript_assets/verification", "5c_sample_photos_check.svg"), dpi=300
)
# fig.savefig(os.path.join(SCRIPT_DIR, "sample_photos_check.png"), dpi=100)
fig.show()
# %%

# %%

########## heatmap: aggregate A-Site interaction, with recipe samples ##########
# Reproduces shap_indexed_tanimoto_plot.py:1391-1444, with the 8 synthesised
# samples from total_correlation_recipes.json annotated on top.


SHAP_CACHE_PATH = Path(SCRIPT_DIR) / "order_3_shap_indexed_tanimoto_cache_rs=42.pkl"

_c = load_shap_cache(SHAP_CACHE_PATH)
shapiq_3rd_values = _c["shapiq_values"]
X_shapiq = _c["X_shapiq"]
sample_category_strings = _c["sample_category_strings"]
idx_concentration = _c["idx_concentration"]
idx_temp = _c["idx_temp"]
idx_asite = _c["idx_asite"]

environmental_interaction_array = np.array(
    [
        iv[(idx_concentration,)]
        + iv[(idx_temp,)]
        + iv[(idx_asite,)]
        + iv[(idx_concentration, idx_temp)]
        + iv[(idx_asite, idx_concentration)]
        + iv[(idx_asite, idx_temp)]
        + iv[(idx_asite, idx_concentration, idx_temp)]
        for iv in shapiq_3rd_values
    ]
)
df_heatmap = pd.DataFrame(
    {
        "concentration": X_shapiq[:, idx_concentration],
        "annealing_temp": X_shapiq[:, idx_temp],
        "SHAP_Combined_Env": environmental_interaction_array,
        "A-Site": sample_category_strings,
    }
)

# annotate with the shared df_samples (defined near the top of this file), keyed
# to the same facet column as the heatmap; respects EXCLUDE_SI1_SAMPLE
df_samples_heatmap = df_samples.rename(columns={"a_site": "A-Site"})

fig, ax_by_asite = create_faceted_tricontour(
    df=df_heatmap,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_Combined_Env",
    facet_col="A-Site",
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc,A-Site}}^{(1,2,3)}$)",
    figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.33),
    y_lim=(0.21, 1.54),
    x_lim=(36.7, None),
    return_ax=True,
)

# annotate the synthesised samples onto the matching A-Site panel
for a_site, grouped in df_samples_heatmap.groupby("A-Site"):
    ax = ax_by_asite[a_site]
    ax.scatter(
        grouped["annealing_temp"],
        grouped["concentration"],
        s=30,
        c="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        clip_on=False,
    )
    for _, r in grouped.iterrows():
        ax.annotate(
            r["sample_text_with_SHAP"],
            (r["annealing_temp"], r["concentration"]),
            textcoords="offset points",
            xytext=(-5, 5),
            ha="center",
            fontsize="medium",
            fontweight="semibold",
            c=r["annotate_text_color"],
            zorder=6,
        )

fig.supxlabel("Annealing Temperature (°C)")
fig.supylabel("Concentration (M)")
title_text = fig.suptitle(
    "a ",
    x=0.0,
    y=1.0,
    va="top",
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)
fig.get_layout_engine().set(wspace=0.02, h_pad=0.03, w_pad=0.02)
for ax_candidate in fig.axes:
    if hasattr(ax_candidate, "_colorbar"):
        cbar = ax_candidate._colorbar
        break
vmin, vmax = cbar.mappable.get_clim()
cbar.set_ticks([vmin * 0.99, 0, vmax * 0.99])
cbar.set_ticklabels(["low", "0", "high"])
cbar.set_label("SHAP Value", labelpad=-10)

fig.savefig(
    os.path.join("../manuscript_assets/verification", "5a_heatmap_annotated.svg"),
    dpi=300,
)
plt.show()

# %%
