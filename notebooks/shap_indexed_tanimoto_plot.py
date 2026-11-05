# %%
import os
import pickle
import sys
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from shapiq import InteractionValues
from shapiq.plot import stacked_bar_plot
from matplotlib.tri import (
    TriAnalyzer,
    Triangulation,
    UniformTriRefiner,
    LinearTriInterpolator,
    CubicTriInterpolator,
)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH  # noqa: E402

########## plotting parameters ##########
plt.rcdefaults()
# plt.rcParams.update(params_basic)
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
#########################################

########## configuration ##########
SHAP_CACHE_PATH = Path(SCRIPT_DIR) / "order_3_shap_indexed_tanimoto_cache_rs=42.pkl"
FIGURE_OUTPUT_DIR = Path(SCRIPT_DIR).parent / "manuscript_assets/shap"
FIGURE_PREFIX = ""  # prepended to every saved filename, e.g. "v2_"

shap_pink = "#ff0054"

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
    """Save figure as <FIGURE_PREFIX><name>.svg at dpi=300 inside FIGURE_OUTPUT_DIR."""
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # check if name has a extension, if not, add .svg, else, keep the extension
    if not Path(name).suffix:
        name += ".svg"

    fig.savefig(FIGURE_OUTPUT_DIR / f"{FIGURE_PREFIX}{name}", dpi=300)


# %%

########## load SHAP cache ##########
with SHAP_CACHE_PATH.open("rb") as f:
    cache = pickle.load(f)
shapiq_3rd_values = cache["shapiq_values"]
X_shapiq = cache["X_shapiq"]
y_shapiq = cache["y_shapiq"]
feature_names = cache["feature_names"]
a_site_names = cache["a_site_names"]
print(f"Loaded SHAP cache from {SHAP_CACHE_PATH}")
print(f"  {len(shapiq_3rd_values)} instances, X_shapiq shape: {X_shapiq.shape}")
####################################

# %%

########## helper setup for plots ##########
sample_category_strings = [a_site_names[int(val)] for val in X_shapiq[:, -1]]

idx_concentration = feature_names.index("concentration")
idx_temp = feature_names.index("annealing_temp")
idx_asite = feature_names.index("a_site_code")

scatter_constants = {
    "annealing_temp": 0.009,
    "dispense_speed": 0.015,
    "spin_speed": 0.005,
    "annealing_time": 0.01,
    "dispense_volume": 0.01,
    "concentration": 0.01,
}
############################################

# %%


########## scatter: continuous feature × A-Site interaction ##########
def _plot_scatter_asite(
    index_tuples,
    x_feature,
    ax,
    hue_data=None,
    hue_palette=None,
    sc_kwargs=dict(),
):
    """scatter plot where SHAP value = sum of shapiq_3rd_values keys in index_tuples."""
    cont_idx = feature_names.index(x_feature)
    interaction_values_array = [
        sum(vals[t] for t in index_tuples) for vals in shapiq_3rd_values
    ]
    df_plot = pd.DataFrame(
        {
            x_feature: X_shapiq[:, cont_idx],
            "SHAP_Interaction": interaction_values_array,
        }
    )
    if hue_data is not None:
        df_plot["_hue"] = hue_data

    scale_factor = scatter_constants.get(x_feature, 0.001) * (
        df_plot[x_feature].max() - df_plot[x_feature].min()
    )
    df_plot["x_jittered"] = df_plot[x_feature] + np.random.normal(
        loc=0.0, scale=scale_factor, size=len(df_plot)
    )

    sns.scatterplot(
        data=df_plot,
        x="x_jittered",
        y="SHAP_Interaction",
        hue="_hue" if hue_data is not None else None,
        palette=hue_palette,
        ax=ax,
        **(dict(linewidth=0.3, s=20, edgecolor="white") | sc_kwargs),
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    sns.despine(ax=ax)
    return fig, ax


# %%
########## spin_speed × A-Site interaction ##########

continuous_feature = "spin_speed"
cont_idx = feature_names.index(continuous_feature)
interaction_pair = tuple(sorted((cont_idx, idx_asite)))
fig, ax = plt.subplots(
    figsize=(SINGLE_COLUMN_WIDTH * 0.6, SINGLE_COLUMN_WIDTH * 0.6),
    constrained_layout=True,
)

_plot_scatter_asite(
    index_tuples=[(cont_idx,), interaction_pair],
    x_feature=continuous_feature,
    ax=ax,
    hue_data=sample_category_strings,
    hue_palette={"MBAI": "#4C72B0", "MeOMBAI": "#55A868", "ClMBAI": "#C44E52"},
)

ax.text(
    0.05,
    0.985,
    r"$\Phi_{\text{rpm},\text{A-Site}}^{(1,2)} = \phi_{\text{rpm}}^{(1)} + \phi_{\text{rpm},\text{A-Site}}^{(2)}$",
    transform=ax.transAxes,
    fontsize="small",
    va="top",
    ha="left",
)

# fig.suptitle(
#     f"{feature_names_map[continuous_feature]['long_unitless']} × {feature_names_map['a_site_code']['long_unitless']}",
#     fontsize="large",
#     fontweight="bold",
#     ha="left",
#     x=0.01,
#     # y=1,
# )


ax.set_xlabel(feature_names_map[continuous_feature]["long"])
ax.set_ylabel("SHAP value")
ax.legend(loc="best", frameon=False)
ax.set_ylim(None, ax.get_ylim()[1] * 1.05)

_save_fig(fig, f"../si/4_scatter_asite_{continuous_feature}_x_A-Site.png")
plt.show()

# %%
########## spin_speed only ##########

continuous_feature = "spin_speed"
cont_idx = feature_names.index(continuous_feature)
fig, ax = plt.subplots(
    figsize=(DOUBLE_COLUMN_WIDTH * 0.25, DOUBLE_COLUMN_WIDTH * 0.24),
    constrained_layout=True,
)

_plot_scatter_asite(
    index_tuples=[
        (cont_idx,),
    ],
    x_feature=continuous_feature,
    ax=ax,
    sc_kwargs=dict(color="0.25", linewidth=0.1),
)

ax.set_xlabel(feature_names_map[continuous_feature]["long"])
ax.set_ylabel("SHAP value")

ax.text(
    0.05,
    0.985,
    r"$\phi_{\text{rpm}}^{(1)}$",
    transform=ax.transAxes,
    fontsize="small",
    va="top",
    ha="left",
)

sup_titile_text = fig.suptitle(
    "f",
    x=0.01,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)

text = fig.axes[0].annotate(
    f"  {feature_names_map[continuous_feature]['long_unitless']} (1st-order)",
    xycoords=sup_titile_text,
    xy=(1, 0.5),
    verticalalignment="center",
    fontsize="medium",
)

text.set_in_layout(False)

_save_fig(fig, f"4f_scatter_asite_{continuous_feature}")
plt.show()

######################################################################

# %%


########## scatter: continuous × continuous interaction ##########
def _plot_scatter_cont(
    index_tuples,
    x_feature,
    color_feature,
    ax,
    cmap="viridis",
):
    """Scatter colored by color_feature; SHAP value = sum of shapiq_3rd_values keys in index_tuples."""
    idx_x = feature_names.index(x_feature)
    idx_c = feature_names.index(color_feature)
    interaction_values_array = [
        sum(vals[t] for t in index_tuples) for vals in shapiq_3rd_values
    ]
    df_plot = pd.DataFrame(
        {
            x_feature: X_shapiq[:, idx_x],
            color_feature: X_shapiq[:, idx_c],
            "SHAP_Interaction": interaction_values_array,
        }
    )
    scale_factor = scatter_constants.get(x_feature, 0.001) * (
        df_plot[x_feature].max() - df_plot[x_feature].min()
    )
    df_plot["x_jittered"] = df_plot[x_feature] + np.random.normal(
        loc=0.0, scale=scale_factor, size=len(df_plot)
    )
    sc = ax.scatter(
        x=df_plot["x_jittered"],
        y=df_plot["SHAP_Interaction"],
        c=df_plot[color_feature],
        cmap=cmap,
        vmin=float(np.percentile(df_plot[color_feature], 0.5)),
        vmax=float(np.percentile(df_plot[color_feature], 99.5)),
        s=15,
        edgecolors="white",
        linewidths=0.3,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    cbar = ax.get_figure().colorbar(sc)
    cbar.set_label(
        feature_names_map[color_feature]["long"],
        rotation=270,
        labelpad=15,
        fontsize="small",
    )
    sns.despine(ax=ax)
    return sc, cbar


feat1 = "annealing_temp"
feat2 = "spin_speed"
idx1 = feature_names.index(feat1)
idx2 = feature_names.index(feat2)
interaction_pair = tuple(sorted((idx1, idx2)))
fig, ax = plt.subplots(
    figsize=(DOUBLE_COLUMN_WIDTH * 0.3, DOUBLE_COLUMN_WIDTH * 0.24),
    constrained_layout=True,
)
sc, cbar = _plot_scatter_cont(
    index_tuples=[interaction_pair],
    x_feature=feat1,
    color_feature=feat2,
    ax=ax,
)
# stat_text = (
#     f"Mean SHAP: {np.mean(df_plot['SHAP_Interaction']):.4f}\n"
#     f"Mean |SHAP|: {np.mean(np.abs(df_plot['SHAP_Interaction'])):.4f}\n"
#     f"Std SHAP: {np.std(df_plot['SHAP_Interaction']):.4f}"
# )
# ax.text(0.4, 0.98, stat_text, transform=ax.transAxes, fontsize="small", va="top")

ax.text(
    0.05,
    0.985,
    r"$\phi_{\text{rpm},\text{temp}}^{(2)}$",
    transform=ax.transAxes,
    fontsize="small",
    va="top",
    ha="left",
)

sup_titile_text = fig.suptitle(
    "g",
    x=0.01,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)

text = fig.axes[0].annotate(
    f"  {feature_names_map[feat1]['long_unitless']} × {feature_names_map[feat2]['long_unitless']}",
    xycoords=sup_titile_text,
    xy=(1, 0.5),
    verticalalignment="center",
    fontsize="medium",
)

text.set_in_layout(False)


ax.set_xlabel(feature_names_map[feat1]["long"])
ax.set_ylabel("SHAP value")

# ax.set_ylim(-2.9, None)

_save_fig(fig, f"4_scatter_{feat1}_x_{feat2}")
plt.show()
##################################################################

# %%
########## heatmap: temp × spin_speed 2nd-order interaction (tricontour) ##########
HEATMAP_LEVELS = 15

feat1 = "dispense_speed"
feat2 = "concentration"

# feat1 = "annealing_temp"
# feat2 = "spin_speed"

idx1 = feature_names.index(feat1)
idx2 = feature_names.index(feat2)
interaction_pair = tuple(sorted((idx1, idx2)))

interaction_values_array = np.array([iv[interaction_pair] for iv in shapiq_3rd_values])
df_heatmap_h = pd.DataFrame(
    {
        feat1: X_shapiq[:, idx1],
        feat2: X_shapiq[:, idx2],
        "SHAP_Interaction": interaction_values_array,
    }
)

_xv, _yv = df_heatmap_h[[feat1, feat2]].to_numpy().T
xv, yv, zv = (
    df_heatmap_h.groupby([feat1, feat2], as_index=False)
    .agg({"SHAP_Interaction": "mean"})[[feat1, feat2, "SHAP_Interaction"]]
    .to_numpy()
    .T
)

# v_min, v_max = np.percentile(zv, [5, 95])
v_min, v_max = np.percentile(zv, [2.5, 97.5])
level_edges = np.linspace(v_min, v_max, HEATMAP_LEVELS + 1)

sx, sy = np.ptp(xv), np.ptp(yv)
tri_n = Triangulation(xv / sx, yv / sy)
mask = TriAnalyzer(tri_n).get_flat_tri_mask(0.01)
tri_n.set_mask(mask)
refiner = UniformTriRefiner(tri_n)
interp = CubicTriInterpolator(tri_n, zv, kind="geom")
tri_refi_n, z_refi = refiner.refine_field(zv, subdiv=3, triinterpolator=interp)
tri = Triangulation(
    tri_refi_n.x * sx, tri_refi_n.y * sy, triangles=tri_refi_n.triangles
)
if tri_refi_n.mask is not None:
    tri.set_mask(tri_refi_n.mask)

fig, ax = plt.subplots(
    figsize=(DOUBLE_COLUMN_WIDTH * 0.3, DOUBLE_COLUMN_WIDTH * 0.24),
    constrained_layout=True,
    dpi=300,
)
ax.set_box_aspect(1)
cs = ax.tricontourf(
    tri,
    z_refi,
    levels=level_edges,
    cmap="coolwarm",
    vmin=v_min,
    vmax=v_max,
    extend="both",
)
ax.scatter(_xv, _yv, c="k", s=4, alpha=0.15, edgecolors="none")
ax.tick_params(axis="both", which="both", direction="in")

ax.set_xlabel(feature_names_map[feat1]["long"])
ax.set_ylabel(feature_names_map[feat2]["long"])

if feat1 == "concentration":
    ax.set_xlim(0.23, 1.53)
if feat2 == "concentration":
    ax.set_ylim(0.23, 1.53)

feat1_unitless = feature_names_map[feat1]["short_unitless"]
feat2_unitless = feature_names_map[feat2]["short_unitless"]

ax.text(
    0.05,
    0.985,
    rf"$\phi_{{\text{{{feat1_unitless}}},\text{{{feat2_unitless}}}}}^{{(2)}}$",
    transform=ax.transAxes,
    fontsize="small",
    va="top",
    ha="left",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.2),
)

sup_titile_text = fig.suptitle(
    "h",
    x=0.0,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)

sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=v_min, vmax=v_max))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, format="%.1f")
cbar.set_label("SHAP value", rotation=270, labelpad=15, fontsize="small")

_save_fig(fig, f"4_heatmap_{feat1}_x_{feat2}")
plt.show()
####################################################################################

# %%


########## helper: extract and rank interaction scores ##########
def _compute_population_scores(shapiq_values, metric="std"):
    """Aggregate per-sample interaction values into a population-level score per coalition."""
    iv_sample = shapiq_values[0]
    interaction_keys = None
    for attr in ["dict_values", "interaction_lookup", "lookup"]:
        if hasattr(iv_sample, attr):
            obj = getattr(iv_sample, attr)
            if isinstance(obj, dict):
                interaction_keys = list(obj.keys())
                break
    if interaction_keys is None:
        interaction_keys = list(iv_sample.keys())

    scores = []
    for k in interaction_keys:
        if len(k) == 0:
            continue
        if metric == "mean_abs":
            score = np.mean([np.abs(iv[k]) for iv in shapiq_values])
        elif metric == "std":
            score = np.std([iv[k] for iv in shapiq_values])
        elif metric == "mean":
            score = np.mean([iv[k] for iv in shapiq_values])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        scores.append((k, score))
    return scores


def _format_and_rank(
    population_scores,
    feature_names,
    feature_names_map,
    feature_important=None,
    rank_index_only=False,
    reverse=True,
):
    if feature_important is None:
        feature_important = []

    def bold_mathtex_wrapper(name):
        abbr = feature_names_map[name]["short_unitless"]
        if name in feature_important:
            return f"$\\mathrm{{\\bf{{{abbr}}}}}$"
        else:
            # put everything in math mode to avoid unicode minus sign issues in matplotlib
            return f"$\\mathrm{{{abbr}}}$"

    formatted = [
        (
            " x ".join(bold_mathtex_wrapper(feature_names[i]) for i in k).replace(
                "-", "‐"
            ),
            score,
        )
        for k, score in population_scores
    ]
    if rank_index_only:
        ranked_indices = sorted(
            range(len(formatted)), key=lambda i: formatted[i][1], reverse=reverse
        )
        return formatted, ranked_indices

    else:
        formatted.sort(key=lambda x: x[1], reverse=reverse)
        return formatted


################################################################

# %%

########## top features: horizontal bar chart (std) ##########
metric = "std"
metrics_name_map = {
    "mean_abs": "Mean(|SHAP|)",
    "std": "Standard deviation of SHAP",
    "mean": "Mean(SHAP)",
}
max_display = 10
feature_important = ["annealing_temp", "concentration", "a_site_code"]

population_scores = _compute_population_scores(shapiq_3rd_values, metric=metric)
formatted_data = _format_and_rank(
    population_scores, feature_names, feature_names_map, feature_important
)

top_features = formatted_data[: max_display - 1]
remaining_features = formatted_data[max_display - 1 :]
if remaining_features:
    avg_remaining = np.mean([item[1] for item in remaining_features])
    top_features.append(
        (
            f"Mean({'SD(SHAP)' if metric == 'std' else metrics_name_map[metric]}) of\n{len(remaining_features)} other features",
            avg_remaining,
        )
    )

labels = [item[0] for item in top_features]
scores = [item[1] for item in top_features]
labels.reverse()
scores.reverse()

with plt.rc_context(
    {
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.bf": "Arial:bold",
    }
):
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH * 0.25, DOUBLE_COLUMN_WIDTH * 0.22),
        dpi=300,
        constrained_layout=True,
    )
    bars = ax.barh(labels, scores, color=shap_pink, height=0.65)
    max_score = max(scores)

    x_max = ax.get_xlim()[1]

    for bar in bars:
        width = bar.get_width()
        if width <= 0:
            continue

        if width / x_max > 0.9:
            label_x = width - max_score * 0.008
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2.0 + 0.01,
                f"{width:.2f}",
                va="center_baseline",
                ha="right",
                color="white",
                fontsize="small",
                # bbox=dict(facecolor='none', edgecolor='black', pad=0.1)
            )
        else:
            label_x = width + max_score * 0.01
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2.0 + 0.01,
                f"{width:.2f}",
                va="center_baseline",
                ha="left",
                color=shap_pink,
                fontsize="small",
            )

    ax.tick_params(axis="both", labelsize="small", length=0)
    ticklabels = ax.get_yticklabels()
    dy = -16 / fig.dpi
    offset = mtransforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
    for label in ticklabels:
        if label.get_text().startswith("Mean"):
            label.set_transform(label.get_transform() + offset)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_xlabel(metrics_name_map[metric])
    title_text = fig.suptitle(
        "a",
        x=0.01,
        y=0.99,
        fontsize="large",
        fontweight="bold",
        ha="left",
    )

    # fig.axes[0].annotate(
    #     "  SHAP Attributions",
    #     xycoords=title_text,
    #     xy=(1, 0),
    #     verticalalignment="bottom",
    #     fontsize="medium",
    # ).set_in_layout(False)

    # ann = ax.annotate(
    # "SHAP Attributions", xycoords=title_text, xy=(1, 0.03), verticalalignment="bottom", fontsize="medium")
    # ann.set_in_layout(False)

    _save_fig(fig, "4a_shap_top_features_order")
    plt.show()
############################################################################


# %%

########## top features: vertical bar chart variants (max_display=35) ##########
max_display = 35

mean_scores = _compute_population_scores(shapiq_3rd_values, metric="mean_abs")
_, mean_abs_ordered_index = _format_and_rank(
    mean_scores, feature_names, feature_names_map, rank_index_only=True
)

for metric in ["mean_abs", "mean", "std"]:
    population_scores = _compute_population_scores(shapiq_3rd_values, metric=metric)
    formatted_data, _ = _format_and_rank(
        population_scores, feature_names, feature_names_map, rank_index_only=True
    )

    formatted_data = [formatted_data[i] for i in mean_abs_ordered_index]

    top_features = formatted_data[: max_display - 1]
    remaining_features = formatted_data[max_display - 1 :]
    if remaining_features:
        avg_remaining = np.mean([item[1] for item in remaining_features])
        top_features.append(
            (
                f"Mean({metrics_name_map[metric]}) of\n{len(remaining_features)} other features",
                avg_remaining,
            )
        )

    labels = [item[0] for item in top_features]
    scores = [item[1] for item in top_features]

    with plt.rc_context(
        {
            "xtick.direction": "out",
            "xtick.major.size": 2.5,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.bf": "Arial:bold",
        }
    ):
        fig, ax = plt.subplots(
            figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.38),
            dpi=300,
            constrained_layout=True,
        )
        bars = ax.bar(labels, scores, color=shap_pink, width=0.65)
        max_score = max(scores)
        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                y = 0.04
            else:
                y = height + max_score * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 3,
                y,
                f"{height:.2f}",
                va="center",
                ha="left",
                color=shap_pink,
                rotation=37,
                rotation_mode="anchor",
                fontsize="small",
            )
        ax.set_xticks(range(len(labels)))
        ticklabels = ax.set_xticklabels(
            labels, rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize="small")
        dy = -12 / fig.dpi
        offset = mtransforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
        for label in ticklabels:
            if label.get_text().startswith("Mean") or label.get_text().startswith(
                "SHAP"
            ):
                label.set_fontsize("small")
                label.set_transform(label.get_transform() + offset)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(-0.8, max_display - 1 + 0.8)
        ax.set_ylabel(metrics_name_map[metric])
        _save_fig(
            fig, f"../si/4_shap_top_features_display={max_display}_metric={metric}.png"
        )

plt.show()

########################################################################################

# %%

########## global stacked bar (mean absolute shapiq InteractionValues) ##########
all_instance_values = np.stack([iv.values for iv in shapiq_3rd_values])
global_mean_absolute_values = np.mean(np.abs(all_instance_values), axis=0)
global_shapiq_values = InteractionValues(
    values=global_mean_absolute_values,
    index=shapiq_3rd_values[0].index,
    min_order=shapiq_3rd_values[0].min_order,
    max_order=shapiq_3rd_values[0].max_order,
    n_players=shapiq_3rd_values[0].n_players,
    baseline_value=0.0,
)
stacked_bar_plot(
    global_shapiq_values,
    feature_names=feature_names,
    title="GP Surrogate - Global Absolute Interaction Breakdown",
    show=False,
)
# _save_fig(plt.gcf(), "global_stacked_bar")
plt.show()
#################################################################################

# %%

########## stacked bar: absolute multi-order decomposition (A-Site split) ##########
# Two distinct corrections are applied to each order-k term:
#   (i)  symmetry inside the tensor slice. o2[i, :] lists each pair {i, j} once,
#        but o3[i, :, :] lists each triplet {i, j, k} twice, at (j, k) and (k, j).
#        Correction: 1 for order 2, 1/2 for order 3.
#   (ii) sharing across bars. An order-k term is summed into the bar of each of
#        the k variables it contains. Correction: 1/k.
# Net divisors: order 1 -> 1, order 2 -> 2, order 3 -> 6.
NORMALIZE_BY_ORDER = True

SYM3 = 0.5  # (i) triplet appears twice in slice i
w1 = 1.0
w2 = 1.0 / 2.0 if NORMALIZE_BY_ORDER else 1.0  # (ii)
w3 = 1.0 / 3.0 if NORMALIZE_BY_ORDER else 1.0  # (ii)

n_samples = len(shapiq_3rd_values)
n_features = len(feature_names)

abs_o1 = np.zeros((n_samples, n_features))
abs_o2_asite = np.zeros((n_samples, n_features))
abs_o2_non_asite = np.zeros((n_samples, n_features))
abs_o3_asite = np.zeros((n_samples, n_features))
abs_o3_non_asite = np.zeros((n_samples, n_features))

# entries of slice o3[i] whose triplet {i, j, k} contains A-Site
mask3_asite = np.zeros((n_features, n_features), dtype=bool)
mask3_asite[idx_asite, :] = True
mask3_asite[:, idx_asite] = True

for s, iv in enumerate(shapiq_3rd_values):
    o1 = np.abs(iv.get_n_order_values(1))
    o2 = np.abs(iv.get_n_order_values(2))
    o3 = np.abs(iv.get_n_order_values(3))

    # --- order 1: no correction ---------------------------------------------
    abs_o1[s] = w1 * o1

    # --- order 2: row i lists each pair {i, j} once; term shared by 2 bars ----
    o2_tot = o2.sum(axis=1)
    o2_as = o2[:, idx_asite].copy()
    o2_non = o2_tot - o2_as
    o2_as[idx_asite] = o2_tot[idx_asite]  # all pairs of A-Site contain A-Site
    o2_non[idx_asite] = 0.0
    abs_o2_asite[s] = w2 * o2_as
    abs_o2_non_asite[s] = w2 * o2_non

    # --- order 3: slice i lists each triplet twice; term shared by 3 bars -----
    o3_tot = SYM3 * o3.sum(axis=(1, 2))
    o3_as = SYM3 * (o3 * mask3_asite[None, :, :]).sum(axis=(1, 2))
    o3_non = o3_tot - o3_as
    o3_as[idx_asite] = o3_tot[idx_asite]  # all triplets of A-Site contain A-Site
    o3_non[idx_asite] = 0.0
    abs_o3_asite[s] = w3 * o3_as
    abs_o3_non_asite[s] = w3 * o3_non

    # --- check: with k-normalization the bars partition the total magnitude ---
    if NORMALIZE_BY_ORDER and s == 0:
        bars_total = (
            abs_o1[s].sum()
            + abs_o2_asite[s].sum()
            + abs_o2_non_asite[s].sum()
            + abs_o3_asite[s].sum()
            + abs_o3_non_asite[s].sum()
        )
        unique_total = o1.sum() + o2.sum() / 2.0 + o3.sum() / 6.0
        assert np.isclose(bars_total, unique_total), (bars_total, unique_total)

mean_abs_o1 = np.mean(abs_o1, axis=0)
mean_abs_o2_non_asite = np.mean(abs_o2_non_asite, axis=0)
mean_abs_o2_asite = np.mean(abs_o2_asite, axis=0)
mean_abs_o3_non_asite = np.mean(abs_o3_non_asite, axis=0)
mean_abs_o3_asite = np.mean(abs_o3_asite, axis=0)

accessible_tab20_colors = [
    "#0072B2",
    "#80B9DB",  # Pair 1: Dark Blue / Light Blue
    "#E69F00",
    "#F3CE80",  # Pair 2: Orange / Light Orange
    "#009E73",
    "#80CFB9",  # Pair 3: Bluish Green / Mint
    "#CC79A7",
    "#E6BCCC",  # Pair 4: Reddish Purple / Pink
    "#D55E00",
    "#EA9F80",  # Pair 5: Vermillion / Coral
    "#56B4E9",
    "#ABDAF4",  # Pair 6: Sky Blue / Soft Sky
    "#F0E442",
    "#F7F2A1",  # Pair 7: Yellow / Cream
    "#005AB5",
    "#80ADDA",  # Pair 8: Deep Blue / Pale Blue
    "#994F00",
    "#CCA780",  # Pair 9: Dark Brown / Tan
    "#111111",
    "#888888",  # Pair 10: Charcoal / Gray
]

c_o1, c_o2_non_asite, c_o2_asite, c_o3_non_asite, c_o3_asite = np.array(
    accessible_tab20_colors
)[[0, 3, 2, 5, 4]]

fig, ax = plt.subplots(
    figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.8),
    constrained_layout=True,
    dpi=300,
)
x = np.arange(n_features)
bar_width = 0.8

segments = [
    (mean_abs_o1, "Order 1", c_o1),
    (mean_abs_o2_non_asite, "Order 2 (Other Pairs)", c_o2_non_asite),
    (mean_abs_o2_asite, "Order 2 (with A-Site)", c_o2_asite),
    (mean_abs_o3_non_asite, "Order 3 (Other Triplets)", c_o3_non_asite),
    (mean_abs_o3_asite, "Order 3 (with A-Site)", c_o3_asite),
]

bottom = np.zeros(n_features)
for values, label, color in segments:
    ax.bar(
        x,
        values,
        bar_width,
        bottom=bottom,
        label=label,
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )
    bottom = bottom + values

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(
    [feature_names_map[n]["short"] for n in feature_names],
    rotation=45,
    ha="right",
    rotation_mode="anchor",
)
ax.set_ylabel("Order-normalized Mean(|SHAP interaction|)")
handles, lbls = ax.get_legend_handles_labels()
ax.legend(
    dict(zip(lbls, handles)).values(),
    dict(zip(lbls, handles)).keys(),
    loc="upper center",
    ncol=1,
    frameon=False,
    fontsize="small",
)
ax.tick_params(axis="x", direction="out")
_save_fig(fig, "../si/4_stacked_bar_multi_order.png")
plt.show()
####################################################################################

# %%


########## helper: create_faceted_heatmap ##########
def create_faceted_heatmap(
    df,
    x_col,
    y_col,
    value_col,
    facet_col=None,
    x_binning_strategy="quantile",
    n_bins=8,
    title=None,
    cbar_label="Value",
    figsize=None,
):
    """Single or multi-panel heatmaps with symmetric color scaling."""
    df_plot = df.copy()

    if pd.api.types.is_numeric_dtype(df_plot[x_col]):
        if x_binning_strategy == "quantile":
            df_plot["x_bin"] = pd.cut(df_plot[x_col], bins=n_bins, labels=False)
            bin_centers = (
                df_plot.groupby("x_bin")[x_col].mean().round(0).astype(int).to_dict()
            )
            df_plot["x_binned"] = df_plot["x_bin"].map(bin_centers)
            x_col_plot = "x_binned"
        elif x_binning_strategy == "equal":
            bin_edges = np.linspace(
                df_plot[x_col].min(), df_plot[x_col].max(), n_bins + 1
            )
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
            df_plot["x_bin"] = pd.cut(
                df_plot[x_col], bins=bin_edges, labels=False, include_lowest=True
            )
            df_plot["x_binned"] = df_plot["x_bin"].map(
                lambda i: int(np.floor(bin_mids[int(i)] + 0.5)) if pd.notna(i) else i
            )
            x_col_plot = "x_binned"
        else:
            raise ValueError(
                f"Invalid x_binning_strategy: {x_binning_strategy!r}. Choose 'quantile' or 'equal'."
            )
    else:
        x_col_plot = x_col

    if facet_col:
        global_x = sorted(df_plot[x_col_plot].unique())
        global_y = sorted(df_plot[y_col].unique(), reverse=True)
        cell_vals = df_plot.groupby([facet_col, y_col, x_col_plot])[value_col].mean()
    else:
        global_x = sorted(df_plot[x_col_plot].unique())
        global_y = sorted(df_plot[y_col].unique(), reverse=True)
        cell_vals = df_plot.groupby([y_col, x_col_plot])[value_col].mean()

    v_max = np.percentile(cell_vals.abs(), 95)
    v_min = -v_max

    if facet_col:
        categories = sorted(df_plot[facet_col].unique())
        n_panels = len(categories)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize or (4 * n_panels, 5),
            sharey=True,
            layout="constrained",
            dpi=300,
        )
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(figsize=figsize or (6, 5), layout="constrained", dpi=300)
        axes = [ax]
        categories = [None]

    for i, cat in enumerate(categories):
        ax = axes[i]
        df_sub = df_plot[df_plot[facet_col] == cat] if facet_col else df_plot
        if facet_col:
            ax.set_title(
                f"{cat}", pad=-8, x=0.04, ha="left", y=0.96, fontweight="semibold"
            )

        pivot = df_sub.pivot_table(
            index=y_col, columns=x_col_plot, values=value_col, aggfunc="mean"
        ).reindex(index=global_y, columns=global_x)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="coolwarm",
            vmin=v_min,
            vmax=v_max,
            cbar=False,
            annot=False,
            linewidths=0.5,
            square=True,
        )
        ax.set_xlabel(None)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )
        ax.set_ylabel(None)
        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_facecolor(to_rgba("lightgray", alpha=0.3))

    fig.supxlabel(x_col)
    fig.supylabel(y_col)
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=v_min, vmax=v_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm, format="%.1f", ax=axes if facet_col else axes[0], shrink=0.8, pad=0.01
    )
    cbar.set_label(cbar_label, rotation=270, verticalalignment="bottom")
    return fig


####################################################

# %%

########## heatmap: agnostic 2nd-order (conc + temp + conc×temp) ##########
general_trend_array = np.array(
    [
        iv[(idx_concentration,)] + iv[(idx_temp,)] + iv[(idx_concentration, idx_temp)]
        for iv in shapiq_3rd_values
    ]
)
df_heatmap = pd.DataFrame(
    {
        "concentration": X_shapiq[:, idx_concentration],
        "annealing_temp": X_shapiq[:, idx_temp],
        "SHAP_General_Trend": general_trend_array,
    }
)

_temp = feature_names_map["annealing_temp"]["short_unitless"]
_conc = feature_names_map["concentration"]["short_unitless"]
fig = create_faceted_heatmap(
    df=df_heatmap,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_General_Trend",
    n_bins=8,
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc}}^{(1,2)}$)",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.26, DOUBLE_COLUMN_WIDTH * 0.26),
)
fig.axes[0].set_title(
    "Agnostic", pad=-13, x=0.04, ha="left", y=1, fontweight="semibold"
)
fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"])
fig.suptitle(
    "c) 1st/2nd-order interaction w/o A-Site\n",
    x=0.0,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)
fig.text(
    0.06,
    0.875,
    r"$\Phi_{\text{temp,conc}}^{(1,2)} = \phi_{\text{temp}}^{(1)} + \phi_{\text{conc}}^{(1)} + \phi_{\text{temp,conc}}^{(2)}$",
    ha="left",
    fontsize="medium",
)


fig.get_layout_engine().set(h_pad=0.03, w_pad=0.02)
_save_fig(fig, "3c_heatmap_agnostic")
plt.show()

# publication version: strip title and colorbar
fig.suptitle("")
for ax in fig.axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
fig.delaxes(fig.axes[-1])
_save_fig(fig, "1c_heatmap_agnostic_no_cbar")
plt.show()
###########################################################################

# %%

########## heatmap: A-Site × (conc + temp) + 3rd-order ##########
environmental_interaction_array = np.array(
    [
        iv[(idx_asite, idx_concentration)]
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

_temp = feature_names_map["annealing_temp"]["short_unitless"]
_conc = feature_names_map["concentration"]["short_unitless"]
_asite = feature_names_map["a_site_code"]["short_unitless"]
fig = create_faceted_heatmap(
    df=df_heatmap,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_Combined_Env",
    facet_col="A-Site",
    n_bins=8,
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)}$)",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.655, DOUBLE_COLUMN_WIDTH * 0.26),
)
fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"])
fig.suptitle(
    "d) 2nd & 3rd-order interaction with A-Sites\n",
    x=0.005,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)
fig.text(
    0.0315,
    0.875,
    r"$\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)} = \phi_{\text{A-Site,temp}}^{(2)} + \phi_{\text{A-Site,conc}}^{(2)} + \phi_{\text{A-Site,temp,conc}}^{(3)}$",
    ha="left",
    fontsize="medium",
)
fig.get_layout_engine().set(wspace=0.03, h_pad=0.03, w_pad=0.05)
_save_fig(fig, "3d_heatmap_asite")
plt.show()

# publication version: simplified colorbar ticks
fig.suptitle("")
for ax in fig.axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
for ax_candidate in fig.axes:
    if hasattr(ax_candidate, "_colorbar"):
        cbar = ax_candidate._colorbar
        break
vmin, vmax = cbar.mappable.get_clim()
cbar.set_ticks([vmin, 0, vmax])
cbar.set_ticklabels(["low", "0", "high"])
cbar.set_label("interaction value", labelpad=-10)
_save_fig(fig, "1c_heatmap_asite_simplified_cbar")
plt.show()

##################################################################

# %%

########## heatmap: aggregate (all conc + temp + a_site terms) ##########
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

_temp = feature_names_map["annealing_temp"]["short_unitless"]
_conc = feature_names_map["concentration"]["short_unitless"]
_asite = feature_names_map["a_site_code"]["short_unitless"]
fig = create_faceted_heatmap(
    df=df_heatmap,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_Combined_Env",
    facet_col="A-Site",
    n_bins=8,
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc,A-Site}}^{(1,2,3)}$)",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.655, DOUBLE_COLUMN_WIDTH * 0.26),
)
fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"])
fig.suptitle(
    "b) Aggregate Interaction Across A-Sites\n",
    x=0.01,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)
fig.text(
    0.0315,
    0.875,
    r"$\Phi_{\text{temp,conc,A-Site}}^{(1,2,3)} = \Phi_{\text{temp,conc}}^{(1,2)} + \Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)} + \phi_{\text{A-site}}^{(1)}$",
    ha="left",
    fontsize="medium",
)

fig.get_layout_engine().set(wspace=0.03, h_pad=0.03, w_pad=0.05)
_save_fig(fig, "3b_heatmap_asite_aggregate")
plt.show()

for a_site in df_heatmap["A-Site"].unique():
    subset = df_heatmap[df_heatmap["A-Site"] == a_site]
    top = subset.sort_values(by="SHAP_Combined_Env", ascending=False).head(10)
    bottom = subset.sort_values(by="SHAP_Combined_Env", ascending=True).head(3)
    extreme_values = pd.concat([top, bottom])
    print(f"Extreme SHAP values for A-Site '{a_site}':")
    print(extreme_values)

# publication version: strip colorbar
fig.suptitle("")
for ax in fig.axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
fig.delaxes(fig.axes[-1])
_save_fig(fig, "1c_heatmap_asite_aggregate_no_cbar")
#########################################################################

# %%

########## violin: A-Site 1st-order main effect ##########
environmental_interaction_array = np.array(
    [iv[(idx_asite,)] for iv in shapiq_3rd_values]
)
df_heatmap = pd.DataFrame(
    {
        "concentration": X_shapiq[:, idx_concentration],
        "annealing_temp": X_shapiq[:, idx_temp],
        "SHAP_Combined_Env": environmental_interaction_array,
        "A-Site": sample_category_strings,
    }
)
a_site_order = list(df_heatmap.groupby("A-Site").groups.keys())

fig, ax = plt.subplots(
    figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.8),
    constrained_layout=True,
    dpi=300,
)
sns.violinplot(
    x="A-Site",
    y="SHAP_Combined_Env",
    data=df_heatmap,
    hue="A-Site",
    ax=ax,
    order=a_site_order,
)
ax.set_ylabel("SHAP Value")
ax.set_xlabel("A-Site")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
# fig.suptitle("1st-Order A-Site Effects", x=0.01, horizontalalignment="left")
_save_fig(fig, "../si/4_violin_asite.png")
plt.show()
##########################################################

# %%


def refine_tricontour(
    triangulation, z, subdiv=3, min_circle_ratio=0.01, triinterpolator="cubic"
):
    """Refine a triangular mesh for tricontour plots.
    Parameters
    ----------
    triangulation : matplotlib.tri.Triangulation
        Triangulation object.
    z : array-like
        Values at the data points.
    subdiv : int, optional
         Number of recursive subdivisions of the initial mesh for smooth plots.
         Values >3 might result in a very high number of triangles for the refine mesh: new triangles numbering = (4**subdiv)*ntri
    min_circle_ratio : float, optional
        Minimum circle ratio - border triangles with circle ratio below this will be
        masked if they touch a border. Suggested value 0.01; use -1 to keep all triangles.
    Returns
    -------
    tri_refi : matplotlib.tri.Triangulation
        Refined triangulation object.
    z_refi : array-like
        Values at the refined triangulation points.
    """

    # ----------------------------------------------------------------------------
    # Improving the triangulation before high-res plots: removing flat triangles
    # ----------------------------------------------------------------------------
    # masking badly shaped triangles at the border of the triangular mesh.
    mask = TriAnalyzer(triangulation).get_flat_tri_mask(min_circle_ratio)
    triangulation.set_mask(mask)

    # refining the data
    refiner = UniformTriRefiner(triangulation)
    if triinterpolator == "linear":
        interp = LinearTriInterpolator(triangulation, z)
    elif triinterpolator == "cubic":
        interp = CubicTriInterpolator(triangulation, z, kind="geom")
    else:
        raise ValueError(
            f"Invalid triinterpolator: {triinterpolator!r}. "
            "Choose 'linear', 'cubic', or pass a "
            "TriInterpolator instance."
        )
    tri_refi, z_refi = refiner.refine_field(z, subdiv=subdiv, triinterpolator=interp)

    return tri_refi, z_refi


########## helper: create_faceted_tricontour ##########
def create_faceted_tricontour(
    df,
    x_col,
    y_col,
    value_col,
    facet_col=None,
    levels=14,
    title=None,
    cbar_label="Value",
    figsize=None,
    filled=True,
    show_points=True,
    refine=True,
    x_lim=None,
    y_lim=None,
):
    """Single or multi-panel tricontour plots with symmetric color scaling."""
    df_plot = df.copy().dropna(subset=[x_col, y_col, value_col])

    v_min, v_max = np.percentile(df_plot[value_col], [5, 95])
    level_edges = np.linspace(v_min, v_max, levels + 1)
    x_lim = (
        x_lim
        if isinstance(x_lim, tuple)
        else (df_plot[x_col].min() * 0.95, df_plot[x_col].max() * 1.02)
    )
    y_lim = (
        y_lim
        if isinstance(y_lim, tuple)
        else (df_plot[y_col].min(), df_plot[y_col].max())
    )

    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=v_min, vmax=v_max)
    )

    if facet_col:
        categories = sorted(df_plot[facet_col].unique())
        n_panels = len(categories)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize or (4 * n_panels, 5),
            sharey=True,
            layout="constrained",
            dpi=150,
        )
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(figsize=figsize or (6, 5), layout="constrained", dpi=150)
        axes = [ax]
        categories = [None]

    cs = None
    for i, cat in enumerate(categories):
        ax = axes[i]
        # keep ax aspect ratio equal to avoid distortion of triangulation
        ax.set_box_aspect(1)

        df_sub = (df_plot[df_plot[facet_col] == cat] if facet_col else df_plot).copy()
        if facet_col:
            ax.set_title(
                f"{cat}", pad=-8, x=0.04, ha="left", y=0.96, fontweight="semibold"
            )

        # average duplicate points (same x and y) to avoid issues with triangulation
        df_sub_avg = df_sub.groupby([x_col, y_col], as_index=False).agg(
            {value_col: "mean"}
        )
        xv, yv, zv = df_sub_avg[[x_col, y_col, value_col]].to_numpy().T

        if len(df_sub_avg) < 3:
            ax.text(
                0.5,
                0.5,
                "Not enough points",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            fn = ax.tricontourf if filled else ax.tricontour
            sx, sy = np.ptp(xv), np.ptp(yv)
            tri_n = Triangulation(xv / sx, yv / sy)  # Delaunay, isotropic

            if isinstance(refine, bool) and not refine:
                tri = Triangulation(xv, yv, triangles=tri_n.triangles)
                zv_refi = zv

            else:
                refine_kwargs = refine if isinstance(refine, dict) else {}
                tri_refi_n, zv_refi = refine_tricontour(tri_n, zv, **refine_kwargs)

                tri = Triangulation(
                    tri_refi_n.x * sx, tri_refi_n.y * sy, triangles=tri_refi_n.triangles
                )

                if tri_refi_n.mask is not None:
                    tri.set_mask(tri_refi_n.mask)

                lin = LinearTriInterpolator(tri_n, zv)
                z_lin = lin(tri_refi_n.x, tri_refi_n.y)
                pad = 0.05 * np.ptp(zv)
                bad = (zv_refi > zv.max() + pad) | (zv_refi < zv.min() - pad)
                if bad.any():
                    print(
                        f"repaired {bad.sum()} out of {len(zv_refi)} node(s), max |cubic| = {np.abs(zv_refi[bad]).max():.3g}"
                    )
                    zv_refi[bad] = z_lin[bad]

            cs = fn(
                tri,
                zv_refi,
                levels=level_edges,
                cmap="coolwarm",
                vmin=v_min,
                vmax=v_max,
                extend="both",
            )

            if not filled:
                ax.clabel(cs, inline=True, fontsize="x-small", fmt="%.1f")
            if show_points:
                xv, yv, zv = df_sub[[x_col, y_col, value_col]].to_numpy().T
                interp = LinearTriInterpolator(tri, zv_refi)
                zv_pred = interp(xv, yv)
                zv_err = zv_pred - zv
                print(f"RMSE of interpolation: {np.sqrt(np.mean(zv_err**2)):.4f}")

                # sort by error magnitude for plotting, so that the largest errors are on the bottom and not covering smaller points
                sort_idx = np.argsort(np.abs(zv_err))[::-1]
                xv, yv, zv, zv_pred = (
                    xv[sort_idx],
                    yv[sort_idx],
                    zv[sort_idx],
                    zv_pred[sort_idx],
                )

                # colors = sm.to_rgba(zv)
                # ax.scatter(xv, yv, c=colors, s=np.abs(zv_pred-zv)**2*10 + 5, ec='k')
                # ax.scatter(xv, yv, c=colors, s=8, ec='k', lw=0.5, alpha=0.5)
                ax.scatter(xv, yv, c="k", s=1, alpha=0.15)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.tick_params(axis="both", which="both", direction="in")

    fig.supxlabel(x_col)
    fig.supylabel(y_col)
    if title:
        fig.suptitle(title)

    cbar = fig.colorbar(
        sm, ax=axes if facet_col else axes[0], shrink=0.8, pad=0.01, format="%.0f"
    )
    cbar.set_label(cbar_label, rotation=270, verticalalignment="bottom")
    return fig


#######################################################

# %%

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

fig = create_faceted_tricontour(
    df=df_heatmap,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_Combined_Env",
    facet_col="A-Site",
    # levels=30,
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc,A-Site}}^{(1,2,3)}$)",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.65, DOUBLE_COLUMN_WIDTH * 0.23),
    y_lim=(0.23, 1.52),
)
fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"])
sup_titile_text = fig.suptitle(
    "c",
    x=0.01,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)

text = fig.axes[0].annotate(
    "  Aggregate interaction across A-Sites  ",
    xycoords=sup_titile_text,
    xy=(1, 0.5),
    verticalalignment="center",
    fontsize="medium",
)

text.set_in_layout(False)

text = (
    fig.axes[0]
    .annotate(
        r"$\Phi_{\text{temp,conc,A-Site}}^{(1,2,3)} = \Phi_{\text{temp,conc}}^{(1,2)} + \Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)} + \phi_{\text{A-site}}^{(1)}$",
        xycoords=text,
        xy=(1, 0.5),
        verticalalignment="center",
        fontsize="medium",
    )
    .set_in_layout(False)
)

fig.get_layout_engine().set(wspace=0.03, h_pad=0.03, w_pad=0.03)
_save_fig(fig, "4c_heatmap_asite_aggregate_tricontour")
plt.show()


# %%
########## heatmap: agnostic 2nd-order (conc + temp + conc×temp) ##########
general_trend_array = np.array(
    [
        iv[(idx_concentration,)] + iv[(idx_temp,)] + iv[(idx_concentration, idx_temp)]
        for iv in shapiq_3rd_values
    ]
)
df_heatmap = pd.DataFrame(
    {
        "concentration": X_shapiq[:, idx_concentration],
        "annealing_temp": X_shapiq[:, idx_temp],
        "SHAP_General_Trend": general_trend_array,
    }
)

_temp = feature_names_map["annealing_temp"]["short_unitless"]
_conc = feature_names_map["concentration"]["short_unitless"]
fig = create_faceted_tricontour(
    df=df_heatmap,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_General_Trend",
    # levels=15,
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc}}^{(1,2)}$)",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.25, DOUBLE_COLUMN_WIDTH * 0.24),
    y_lim=(0.23, 1.53),
)
fig.axes[0].set_title(
    "Agnostic", pad=-13, x=0.04, ha="left", y=1, fontweight="semibold"
)
fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"])
sup_titile_text = fig.suptitle(
    "d\n",
    x=0.01,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)

text = fig.axes[0].annotate(
    "  1st/2nd-order interaction w/o A-Site",
    xycoords=sup_titile_text,
    xy=(1, 0.75),
    verticalalignment="center",
    fontsize="medium",
)

text.set_in_layout(False)

fig.axes[0].annotate(
    r"  $\Phi_{\text{temp,conc}}^{(1,2)} = \phi_{\text{temp}}^{(1)} + \phi_{\text{conc}}^{(1)} + \phi_{\text{temp,conc}}^{(2)}$",
    xycoords=text,
    xy=(0, -0.15),
    verticalalignment="top",
    fontsize="medium",
).set_in_layout(False)

fig.get_layout_engine().set(h_pad=0.03, w_pad=0.02)
_save_fig(fig, "4d_heatmap_agnostic_tricontour")
plt.show()


# publication version: strip title and colorbar
# fig.suptitle("")
# for ax in fig.axes:
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_color("black")
#         spine.set_linewidth(1.0)
# fig.delaxes(fig.axes[-1])
# _save_fig(fig, "1c_heatmap_agnostic_no_cbar_tricontour")
# plt.show()


# %%
environmental_interaction_array = np.array(
    [
        iv[(idx_asite, idx_concentration)]
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

_temp = feature_names_map["annealing_temp"]["short_unitless"]
_conc = feature_names_map["concentration"]["short_unitless"]
_asite = feature_names_map["a_site_code"]["short_unitless"]
# fig = create_faceted_tricontour(
#     df=df_heatmap,
#     x_col="annealing_temp",
#     y_col="concentration",
#     value_col="SHAP_Combined_Env",
#     facet_col="A-Site",
#     levels=10,
#     cbar_label=r"SHAP value ($\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)}$)",
#     figsize=(DOUBLE_COLUMN_WIDTH * 0.655, DOUBLE_COLUMN_WIDTH * 0.26),
#     y_lim=(0.23, 1.53),
#     # refine=dict(subdiv=3, min_circle_ratio=0.01)
# )
# fig.supxlabel(feature_names_map["annealing_temp"]["long"])
# fig.supylabel(feature_names_map["concentration"]["long"])
# fig.suptitle(
#     "d) 2nd & 3rd-order interaction with A-Sites\n",
#     x=0.005,
#     horizontalalignment="left",
#     fontsize="large",
#     fontweight="bold",
# )
# fig.text(
#     0.0315,
#     0.875,
#     r"$\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)} = \phi_{\text{A-Site,temp}}^{(2)} + \phi_{\text{A-Site,conc}}^{(2)} + \phi_{\text{A-Site,temp,conc}}^{(3)}$",
#     ha="left",
#     fontsize="medium",
# )
# fig.get_layout_engine().set(wspace=0.03, h_pad=0.03, w_pad=0.05)
# _save_fig(fig, "3d_heatmap_asite_tricontour")
# plt.show()

# # publication version: simplified colorbar ticks
# fig.suptitle("")
# for ax in fig.axes:
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_color("black")
#         spine.set_linewidth(1.0)
# for ax_candidate in fig.axes:
#     if hasattr(ax_candidate, "_colorbar"):
#         cbar = ax_candidate._colorbar
#         break
# vmin, vmax = cbar.mappable.get_clim()
# cbar.set_ticks([vmin, 0, vmax])
# cbar.set_ticklabels(["low", "0", "high"])
# cbar.set_label("interaction value", labelpad=-10)
# _save_fig(fig, "1c_heatmap_asite_simplified_cbar_tricontour")
# plt.show()

# %%

df_heatmap_MeO_only = df_heatmap[df_heatmap["A-Site"] == "MeOMBAI"]

fig = create_faceted_tricontour(
    df=df_heatmap_MeO_only,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_Combined_Env",
    facet_col="A-Site",
    levels=10,
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)}$)",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.25, DOUBLE_COLUMN_WIDTH * 0.24),
    y_lim=(0.23, 1.53),
    # refine=dict(subdiv=3, min_circle_ratio=0.01)
)
fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"])

sup_titile_text = fig.suptitle(
    "e\n",
    x=0.01,
    horizontalalignment="left",
    fontsize="large",
    fontweight="bold",
)

text = fig.axes[0].annotate(
    "  2nd & 3rd-order interaction with A-Sites",
    xycoords=sup_titile_text,
    xy=(1, 0.75),
    verticalalignment="center",
    fontsize="medium",
)

text.set_in_layout(False)

fig.axes[0].annotate(
    r"  $\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)} = \phi_{\text{A-Site,temp}}^{(2)} + \phi_{\text{A-Site,conc}}^{(2)} + \phi_{\text{A-Site,temp,conc}}^{(3)}$",
    xycoords=text,
    xy=(0, -0.14),
    verticalalignment="top",
    fontsize="small",
).set_in_layout(False)


fig.get_layout_engine().set(h_pad=0.03, w_pad=0.02)
_save_fig(fig, "4e_heatmap_MeO_tricontour")
plt.show()

# %%

########## merged figure: panels c, d, e combined into a single 1x5 grid, shared colorbar ##########
MERGED_LEVELS = 15
MERGED_Y_LIM = (0.23, 1.53)
_temp_all = X_shapiq[:, idx_temp]
MERGED_X_LIM = (_temp_all.min() * 0.95, _temp_all.max() * 1.02)


def _tricontour_on_ax(ax, _xv, _yv, _zv, level_edges, v_min, v_max, refine=True):
    """Plot a single tricontourf panel onto ax using a shared level/color scale."""
    ax.set_box_aspect(1)
    df_avg = (
        pd.DataFrame({"x": _xv, "y": _yv, "z": _zv})
        .groupby(["x", "y"], as_index=False)
        .agg({"z": "mean"})
    )
    xv, yv, zv = df_avg[["x", "y", "z"]].to_numpy().T

    sx, sy = np.ptp(xv), np.ptp(yv)
    tri_n = Triangulation(xv / sx, yv / sy)

    if isinstance(refine, bool) and not refine:
        tri = Triangulation(xv, yv, triangles=tri_n.triangles)
        zv_refi = zv
    else:
        refine_kwargs = refine if isinstance(refine, dict) else {}
        tri_refi_n, zv_refi = refine_tricontour(tri_n, zv, **refine_kwargs)
        tri = Triangulation(
            tri_refi_n.x * sx, tri_refi_n.y * sy, triangles=tri_refi_n.triangles
        )
        if tri_refi_n.mask is not None:
            tri.set_mask(tri_refi_n.mask)

    tri = ax.tricontourf(
        tri,
        zv_refi,
        levels=level_edges,
        cmap="coolwarm",
        vmin=v_min,
        vmax=v_max,
        extend="both",
    )

    ax.scatter(_xv, _yv, c="k", s=4, alpha=0.15, edgecolors="none")

    return ax


# panel c: 1st/2nd/3rd-order incl. A-Site, faceted by A-Site
c_values = np.array(
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

# panel d: agnostic 1st/2nd-order (no A-Site), single panel
d_values = np.array(
    [
        iv[(idx_concentration,)] + iv[(idx_temp,)] + iv[(idx_concentration, idx_temp)]
        for iv in shapiq_3rd_values
    ]
)

# panel e: A-Site interaction terms only, MeOMBAI subset
e_values = np.array(
    [
        iv[(idx_asite, idx_concentration)]
        + iv[(idx_asite, idx_temp)]
        + iv[(idx_asite, idx_concentration, idx_temp)]
        for iv in shapiq_3rd_values
    ]
)

conc_all = X_shapiq[:, idx_concentration]
temp_all = X_shapiq[:, idx_temp]
a_site_arr = np.array(sample_category_strings)
e_mask = a_site_arr == "MeOMBAI"

v_min, v_max = np.percentile(
    np.concatenate([c_values, d_values, e_values[e_mask]]), [5, 95]
)
level_edges = np.linspace(v_min, v_max, MERGED_LEVELS + 1)

fig, axes = plt.subplots(
    1,
    5,
    figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.235),
    sharey=True,
    layout="constrained",
    dpi=300,
)

a_site_order = ["ClMBAI", "MBAI", "MeOMBAI"]
for i, cat in enumerate(a_site_order):
    mask = a_site_arr == cat
    _tricontour_on_ax(
        axes[i],
        temp_all[mask],
        conc_all[mask],
        c_values[mask],
        level_edges,
        v_min,
        v_max,
    )
    axes[i].set_title(cat, pad=-8, x=0.04, ha="left", y=0.96, fontweight="semibold")

_tricontour_on_ax(axes[3], temp_all, conc_all, d_values, level_edges, v_min, v_max)
axes[3].set_title("Agnostic", pad=-8, x=0.04, ha="left", y=0.96, fontweight="semibold")

_tricontour_on_ax(
    axes[4],
    temp_all[e_mask],
    conc_all[e_mask],
    e_values[e_mask],
    level_edges,
    v_min,
    v_max,
)
title = axes[4].set_title(
    "MeOMBAI ", pad=-8, x=0.04, ha="left", y=0.96, fontweight="semibold"
)
axes[4].annotate(
    r"(A-site interaction only)",
    xycoords=title,
    xy=(1, 0.5),
    verticalalignment="center",
    fontsize="small",
).set_in_layout(False)


for i, ax in enumerate(axes):
    ax.set_xlim(MERGED_X_LIM)
    ax.set_ylim(MERGED_Y_LIM)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis="both", which="both", direction="in")
    if i > 0:
        ax.tick_params(axis="y", labelleft=False)

fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"], x=-0.007, ha="right")

for letter, ax in zip("cde", [axes[0], axes[3], axes[4]]):
    ann = ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-8, 0) if letter != "c" else (-26, 0),
        textcoords="offset points",
        fontsize="large",
        fontweight="bold",
        ha="center",
        va="top",
    )

    if letter == "c":
        ann.set_in_layout(False)

sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=v_min, vmax=v_max))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.01, format="%.0f")
cbar.set_label("SHAP value", rotation=270, verticalalignment="baseline")

fig.get_layout_engine().set(wspace=0.01, h_pad=0.01, w_pad=0.01)

_save_fig(fig, "4cde_heatmap_merged_tricontour")
plt.show()
#######################################################################################################

# %%

fig = create_faceted_tricontour(
    df=df_heatmap,
    x_col="annealing_temp",
    y_col="concentration",
    value_col="SHAP_Combined_Env",
    facet_col="A-Site",
    levels=10,
    cbar_label=r"SHAP value ($\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)}$)",
    figsize=(DOUBLE_COLUMN_WIDTH * 0.655, DOUBLE_COLUMN_WIDTH * 0.235),
    y_lim=(0.23, 1.53),
    # refine=dict(subdiv=3, min_circle_ratio=0.01)
)
fig.supxlabel(feature_names_map["annealing_temp"]["long"])
fig.supylabel(feature_names_map["concentration"]["long"])
fig_title_text = fig.suptitle(
    "2nd & 3rd-order interaction with A-Sites",
    x=0.005,
    horizontalalignment="left",
    fontsize="medium",
)

annotation = fig.axes[0].annotate(
    r" $\Phi_{\text{temp,conc}}^{\text{A-Site}(2,3)} = \phi_{\text{A-Site,temp}}^{(2)} + \phi_{\text{A-Site,conc}}^{(2)} + \phi_{\text{A-Site,temp,conc}}^{(3)}$",
    xycoords=fig_title_text,
    xy=(1, 0.5),
    verticalalignment="center",
)

annotation.set_in_layout(False)


fig.get_layout_engine().set(wspace=0.03, h_pad=0.03, w_pad=0.05)
_save_fig(fig, "../si/4_heatmap_asite_tricontour.png")
plt.show()


fig.suptitle("")
for ax in fig.axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
for ax_candidate in fig.axes:
    if hasattr(ax_candidate, "_colorbar"):
        cbar = ax_candidate._colorbar
        break
vmin, vmax = cbar.mappable.get_clim()
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels(["low", "high"])
cbar.set_label("interaction value", labelpad=-10)
_save_fig(fig, "1c_heatmap_asite_simplified_cbar")
plt.show()
