# %%
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configs.nature_mpl_rc import nature_rc  # noqa: E402

########## plotting parameters ##########
plt.rcdefaults()
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
#########################################

########## configuration ##########
SHAP_CACHE_DIR = Path(SCRIPT_DIR)
RANDOM_SEED = 42
FIGURE_OUTPUT_DIR = Path(SCRIPT_DIR).parent / "manuscript_assets/shap"
FIGURE_PREFIX = ""

feature_names_map = {
    "annealing_temp": {
        "short": "Temp (°C)",
        "long": "Annealing Temperature (°C)",
        "short_unitless": "Temp",
        "long_unitless": "Annealing Temperature",
    },
    "annealing_time": {
        "short": "Time (s)",
        "long": "Annealing Time (s)",
        "short_unitless": "Time",
        "long_unitless": "Annealing Time",
    },
    "dispense_volume": {
        "short": "Dvol (µL)",
        "long": "Dispense Volume (µL)",
        "short_unitless": "Dvol",
        "long_unitless": "Dispense Volume",
    },
    "spin_speed": {
        "short": "Rpm",
        "long": "Spin Speed (rpm)",
        "short_unitless": "Rpm",
        "long_unitless": "Spin Speed",
    },
    "dispense_speed": {
        "short": "Dspd (µL/s)",
        "long": "Dispense Speed (µL/s)",
        "short_unitless": "Dspd",
        "long_unitless": "Dispense Speed",
    },
    "concentration": {
        "short": "Conc (M)",
        "long": "Concentration (M)",
        "short_unitless": "Conc",
        "long_unitless": "Concentration",
    },
    "a_site_code": {
        "short": "A-Site",
        "long": "A-Site",
        "short_unitless": "A-Site",
        "long_unitless": "A-Site",
    },
}
###################################


def _save_fig(fig, name: str) -> None:
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT_DIR / f"{FIGURE_PREFIX}{name}.svg", dpi=300)


# %%

########## load shared data from order-3 cache ##########
_base_cache_path = (
    SHAP_CACHE_DIR / f"order_3_shap_indexed_tanimoto_cache_rs={RANDOM_SEED}.pkl"
)
with _base_cache_path.open("rb") as f:
    _base_cache = pickle.load(f)
X_shapiq = _base_cache["X_shapiq"]
feature_names = _base_cache["feature_names"]
a_site_names = _base_cache["a_site_names"]
print(f"Loaded base data from {_base_cache_path}")
#########################################################

# %%

########## shared setup ##########
sample_category_strings = [a_site_names[int(val)] for val in X_shapiq[:, -1]]

scatter_constants = {
    "annealing_temp": 0.009,
    "dispense_speed": 0.015,
    "spin_speed": 0.005,
    "annealing_time": 0.01,
    "dispense_volume": 0.01,
    "a_site_code": 0.05,
    "concentration": 0.01,
}

_ASITE_PALETTE = {"MBAI": "#4C72B0", "MeOMBAI": "#55A868", "ClMBAI": "#C44E52"}
_ASITE_STRINGS = np.array(sample_category_strings)
_asite_colors = [_ASITE_PALETTE[name] for name in a_site_names]
_asite_cmap = ListedColormap(_asite_colors, name="a_site")
_asite_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

SHOW_ORDER_1_PLUS_2 = True  # True = main effect + interaction; False = 2nd-order only

# Per-column (feat_a) axis limits; "percentile(N)" is resolved against x_vals.
_scatter_overrides: dict[str, dict] = {
    "dispense_volume": {"xmax": 160},
    # "concentration":   {"xmax": "percentile(80)", "xmin": "percentile(10)"},
    # "annealing_temp":  {"ymax": "percentile(99.5)"},
    # "annealing_time":  {"xmax": 1250},
}

manual_global_ylim = {
    1: (-9, 11),
    2: (-8, 11),
}

# Per-row (feat_b) color limits; overrides the auto percentile range for that feature.
_color_overrides: dict[str, dict] = {
    "concentration": {"vmax": 1.6},
}

_n = len(feature_names)

# Precompute per-feature color ranges so all rows sharing the same color feature
# map to the same colorbar scale.
_color_ranges = {
    j: (
        float(np.percentile(X_shapiq[:, j], 2)),
        float(np.percentile(X_shapiq[:, j], 98)),
    )
    for j, feat in enumerate(feature_names)
    if feat != "a_site_code"
}
##################################


def _resolve_lim(val, data):
    if val is None:
        return None
    if isinstance(val, str) and val.startswith("percentile("):
        return float(np.percentile(data, float(val[len("percentile(") : -1])))
    return float(val)


def _truncate_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


_scatter_cmap = _truncate_cmap(plt.get_cmap("viridis"))

# %%

########## SHAP scatter matrix (orders 1–3) ##########
for max_order in range(1, 4):
    pkl_path = (
        SHAP_CACHE_DIR
        / f"order_{max_order}_shap_indexed_tanimoto_cache_rs={RANDOM_SEED}.pkl"
    )
    with pkl_path.open("rb") as f:
        cache = pickle.load(f)
    shapiq_values = cache["shapiq_values"]
    print(f"Loaded order-{max_order} cache: {len(shapiq_values)} instances")

    fig, axes = plt.subplots(
        _n,
        _n,
        figsize=(13.5, 10.8),
        layout="constrained",
        sharex="col",
        dpi=300,
        sharey="all",
    )

    for i, feat_a in enumerate(feature_names):
        for j, feat_b in enumerate(feature_names):
            ax = axes[j, i]
            cbar = None
            pair = tuple(sorted((i, j)))

            if i == j:  # diagonal: 1st-order only
                y_vals = np.array([vals[(i,)] for vals in shapiq_values])
            elif SHOW_ORDER_1_PLUS_2:
                y_vals = np.array([vals[(i,)] + vals[pair] for vals in shapiq_values])
            else:
                y_vals = np.array([vals[pair] for vals in shapiq_values])

            x_vals = X_shapiq[:, i]
            scale = scatter_constants.get(feat_a, 0.001) * (x_vals.max() - x_vals.min())
            x_jittered = x_vals + np.random.normal(0, scale, size=len(x_vals))

            if i == j and feat_a != "a_site_code":  # diagonal: gray, no colormap
                ax.scatter(x_jittered, y_vals, c="0.5", s=10, linewidths=0, alpha=0.7)
            elif feat_b == "a_site_code":
                sc_asite = ax.scatter(
                    x_jittered,
                    y_vals,
                    c=X_shapiq[:, j],
                    cmap=_asite_cmap,
                    norm=_asite_norm,
                    s=10,
                    linewidths=0,
                    alpha=0.7,
                )
                if i == _n - 1:
                    cbar = fig.colorbar(
                        sc_asite, ax=ax, shrink=0.8, pad=0.03, ticks=[0, 1, 2]
                    )
                    cbar.set_ticklabels(a_site_names)
                    cbar.ax.tick_params(which="both", length=0)
                    cbar.set_label(
                        "A-Site", rotation=270, labelpad=10, fontsize="small"
                    )
            else:
                vmin, vmax = _color_ranges[j]
                ov_c = _color_overrides.get(feat_b, {})
                vmin = float(ov_c.get("vmin", vmin))
                vmax = float(ov_c.get("vmax", vmax))
                sc = ax.scatter(
                    x_jittered,
                    y_vals,
                    c=X_shapiq[:, j],
                    cmap=_scatter_cmap,
                    s=10,
                    linewidths=0,
                    alpha=0.7,
                    vmin=vmin,
                    vmax=vmax,
                )
                if (
                    i == _n - 1
                ):  # rightmost column: add colorbar for this row's color feature
                    feat_b_label = (feature_names_map.get(feat_b) or {}).get(
                        "short"
                    ) or feat_b
                    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.03)
                    cbar.set_label(
                        feat_b_label, rotation=270, labelpad=10, fontsize="small"
                    )

            if cbar is not None:
                # thinner frame for colorbar
                cbar.outline.set_linewidth(0.5)

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

            # Apply per-column axis limit overrides
            ov = _scatter_overrides.get(feat_a, {})
            xlim_lo = _resolve_lim(ov.get("xmin"), x_vals)
            xlim_hi = _resolve_lim(ov.get("xmax"), x_vals)
            ylim_lo = _resolve_lim(ov.get("ymin"), y_vals)
            ylim_hi = _resolve_lim(ov.get("ymax"), y_vals)
            if xlim_lo is not None or xlim_hi is not None:
                ax.set_xlim(xlim_lo, xlim_hi)
            if ylim_lo is not None or ylim_hi is not None:
                ax.set_ylim(ylim_lo, ylim_hi)

            if feat_a == "a_site_code":
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(list(_ASITE_PALETTE.keys()), rotation=0, ha="center")

            ax.set_ylabel(None)
            feat_a_label = (feature_names_map.get(feat_a) or {}).get("short") or feat_a
            ax.set_xlabel(feat_a_label if j == _n - 1 else None)

    global_ylim = manual_global_ylim.get(max_order)
    if global_ylim is not None:
        for ax in axes[:, 0]:
            ax.set_ylim(global_ylim)

    fig.supylabel("SHAP value")
    fig.align_labels()
    _save_fig(fig, f"si_shap_scatter_matrix_order_{max_order}")
    plt.show()
######################################################################

# %%
