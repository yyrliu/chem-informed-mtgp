# %%
import os
import sys

import seaborn as sns

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


from chem_informed_mtgp.fingerprint_kernel import (
    CONTINUOUS_FEATURE_COUNT,
    FINGERPRINT_BITS,
)


from configs.exp_config import NOTEBOOK_CONFIG
from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH
from configs.paths import CV_RESULTS_DIR_BASE
from helpers.cv_cache import load_cv_results, run_dir

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))  # noqa: E402


########## set up plotting parameters ##########
plt.rcdefaults()
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.bf": "Arial:bold",
    }
)
################################################

########## setup macro ##########
param_config = NOTEBOOK_CONFIG["param_config"]
outliers_by_sample_name = NOTEBOOK_CONFIG["outliers_by_sample_name"]
INITIAL_TRAIN_BATCHES = NOTEBOOK_CONFIG["train_batches"]
CV_BATCHES = NOTEBOOK_CONFIG["cv_batches"]
DEFAULT_TARGET_COLUMN = NOTEBOOK_CONFIG["default_target_column"]
A_SITE_TO_SMILES = NOTEBOOK_CONFIG["a_site_to_smiles"]
POSTERIOR_MEAN_THRESHOLDS_BY_A_SITE = NOTEBOOK_CONFIG[
    "posterior_mean_thresholds_by_a_site"
]
CONDITION_OUTLIER_RSD_THRESHOLD = NOTEBOOK_CONFIG["condition_outlier_rsd_threshold"]
KERNEL_TYPE = NOTEBOOK_CONFIG["kernel_type"]
FINGERPRINT_MODE = NOTEBOOK_CONFIG["fingerprint_mode"]

RANDOM_POOL_SIZE = 2_500_000
# RANDOM_POOL_SIZE = 1_000_000
# RANDOM_POOL_SIZE = 10_000

N_CPU = 16
N_NEW_POINTS = 8
N_IVR_EVAL_BATCH_SIZE = 5000 // N_NEW_POINTS * N_CPU
FINGERPRINT_BITS = int(os.environ.get("FINGERPRINT_BITS", FINGERPRINT_BITS))
CONTINUOUS_FEATURE_COUNT = min(
    len(param_config),
    int(os.environ.get("CONTINUOUS_FEATURE_COUNT", CONTINUOUS_FEATURE_COUNT)),
)
CONTINUOUS_COLUMNS = list(param_config.keys())[:CONTINUOUS_FEATURE_COUNT]
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", DEFAULT_TARGET_COLUMN)

metric_name_mapping = {
    "crps_mean": "CRPS",
    "mae": "MAE",
    "picp": "PICP",
}

fp_label_mapping = {
    "fp2": "Tanimoto",
    "no-correl": "identity\n(lengthscales only)",
    "total-correl": "constant\n(fully pooled)",
}

################################################

# %%
run_dirs = [run_dir(s) for s in range(42, 42 + 30)]

validation_summary_df, foldwise_df = load_cv_results(run_dirs)

cv_sample_df = pd.read_csv(
    CV_RESULTS_DIR_BASE.with_name(f"{CV_RESULTS_DIR_BASE.name}_rseed=44")
    / "df_cross_validation.csv"
)

# # %%
# bar_plot_df = validation_summary_df.copy()[
#     validation_summary_df["a-site-aware"] & validation_summary_df["var-per-asite"]
# ]

# for metric in ["crps_mean", "rmse", "picp"]:
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=bar_plot_df, x="a_site", y=metric, hue="file", palette="tab20")
#     plt.title(f"{metric} by a_site and CV result file")
#     plt.ylabel(metric)
#     plt.xlabel("a_site")
#     plt.legend(title="CV Result File", bbox_to_anchor=(1.05, 1), loc="upper left")
#     if metric == "picp":
#         plt.axhline(0.95, color="red", linestyle="--", label="95% Interval")
# # %%

# print(
#     "\n".join(
#         [
#             k
#             for k in validation_summary_df[
#                 validation_summary_df["few_shot_ratio"] == 100
#             ]
#             .groupby("file")
#             .groups.keys()
#         ]
#     )
# )

# bar_plot_df = validation_summary_df.copy()[
#     validation_summary_df["a-site-aware"]
#     & (validation_summary_df["few_shot_ratio"] == 100)
#     & ~validation_summary_df["a_site"].isin(["Others"])
# ]

# a_site_order = [
#     "Overall",
#     "ClMBAI",
#     "MBAI",
#     "MeOMBAI",
# ]
# # ensure a_site is categorical with the specified order in string format for plotting
# a_site_order_in_data = [
#     site for site in a_site_order if site in bar_plot_df["a_site"].unique()
# ]

# legend_label_mapping = {
#     "fp2": "baseline",
#     "fp2_no-correl": "no-correl",
#     "var-per-asite_total-correl": "total-correl",
#     "var-per-asite_fp2": "var-per-asite",
#     "var-per-asite_fp2_no-correl": "var-per-asite no-correl",
# }

# name_mapping = {
#     "crps_mean": "CRPS",
#     "rmse": "RMSE",
#     "picp": "PICP",
# }

# # %%
# bar_plot_df["legend_label"] = bar_plot_df.apply(
#     lambda row: legend_label_mapping.get(row["file"].split(" (")[0]), axis=1
# )
# bar_plot_df["a_site"] = pd.Categorical(
#     bar_plot_df["a_site"], categories=a_site_order_in_data, ordered=True
# )

# for metric in ["crps_mean", "rmse", "picp"]:
#     plt.figure(
#         figsize=(2.8 * figure_zoom_multiplier, 2 * figure_zoom_multiplier),
#         constrained_layout=True,
#         dpi=300,
#     )
#     sns.barplot(
#         data=bar_plot_df, x="a_site", y=metric, hue="legend_label", palette="tab20"
#     )
#     plt.title(f"Post-loop model tuning: {name_mapping[metric]}")
#     plt.ylabel(name_mapping[metric])
#     plt.xlabel("a_site")

#     if metric == "picp":
#         plt.axhline(0.95, color="red", linestyle="--")
#         plt.ylim(0, 1.23)
#         plt.legend(loc="upper right", ncol=2)
#     else:
#         plt.legend(loc="upper right")
#         plt.ylim(
#             None, plt.gca().get_ylim()[1] * 1.1
#         )  # set y-axis lower limit to 0 for better comparison

# # %%

# bar_plot_df = validation_summary_df.copy()[
#     validation_summary_df["a-site-aware"] & validation_summary_df["var-per-asite"]
# ]

# for metric in ["crps_mean", "rmse", "picp"]:
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=bar_plot_df, x="a_site", y=metric, hue="file", palette="tab20")
#     plt.title(f"{metric} by a_site and CV result file")
#     plt.ylabel(metric)
#     plt.xlabel("a_site")
#     plt.legend(title="CV Result File", bbox_to_anchor=(1.05, 1), loc="upper left")
#     if metric == "picp":
#         plt.axhline(0.95, color="red", linestyle="--", label="95% Interval")

# %%
LKJ_MASTER_CATEGORY = "none"

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    sharex="col",
    figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.7),
    constrained_layout=True,
    dpi=300,
    sharey="row",
    height_ratios=[1, 1, 0.6],
)

for i, a_site in enumerate(["ClMBAI", "MBAI", "MeOMBAI"]):
    # build from foldwise_df instead of validation_summary_df
    line_plot_df = foldwise_df.copy()[
        (foldwise_df["a-site-aware"] & foldwise_df["var-per-asite"])
        & foldwise_df["a_site"].isin([a_site, "Others", "Overall"])
        & (
            (foldwise_df["few_shot_a_site"] == a_site)
            | foldwise_df["few_shot_a_site"].isna()
        )
        & (
            (foldwise_df["fp_type"] != "lkj_corr")
            | (foldwise_df["lkj_master_category"] == LKJ_MASTER_CATEGORY)
        )
    ]

    line_plot_df["A-site"] = (
        line_plot_df["a_site"]
        .astype(str)
        .replace({a_site: "Target", "Others": "Others", "Overall": "Overall"})
    )
    line_plot_df["Model"] = line_plot_df["fp_type"].replace(
        {
            "fp2": "Tanimoto",
            "no-correl": "Identity",
            "total-correl": "Constant",
            "lkj_corr": "Learned",
        }
    )

    for j, (metric, ax) in enumerate(
        zip(["crps_mean", "mae", "picp"], axes.T[i].flatten())
    ):
        sns.lineplot(
            data=line_plot_df,
            x="few_shot_ratio",
            y=metric,
            hue="A-site",
            style="Model",
            markers=True,
            style_order=["Tanimoto", "Identity", "Constant", "Learned"],
            hue_order=["Others", "Overall"],
            errorbar=None,
            ax=ax,
        )

        ax.set_xlabel("Few-Shot Ratio (%)")
        ax.set_ylabel(metric_name_mapping[metric])

        ax.set_xlabel(f"Target ({a_site}) data points (count, fraction %)", labelpad=8)

        ax.legend().set_visible(False)

        if metric == "picp":
            ax.axhline(0.95, color="red", linestyle="--")
            ax.set_ylim(None, 1.01)
            if j == 3:
                ax.text(2.5, 0.96, "95% Interval", color="red", ha="right", va="bottom")

        if j == 0:
            ax.set_title(a_site, y=1.0, pad=-12)

        target_a_site_num = cv_sample_df[cv_sample_df["a_site"] == a_site][
            "sample_name"
        ].nunique()

        def _fn_tick_labels(x, pos, target_a_site_num=target_a_site_num):
            return str(int(x / 100 * target_a_site_num)) + "\n" + f"({x:.0f}%)"

        ax.xaxis.set_major_formatter(FuncFormatter(_fn_tick_labels))

handles, labels = axes.flatten()[-1].get_legend_handles_labels()
handles.insert(3, Line2D([1], [1], linestyle="none", marker="none"))
labels.insert(3, "")

fig.legend(handles, labels, loc="outside center right", frameon=False)

fig.align_ylabels(axes)

fig.savefig(
    "../manuscript_assets/si/3_negative_transfer.png",
    dpi=300,
)

## %%
# LINE_PLOT_A_SITE = "MBAI"

# # build from foldwise_df instead of validation_summary_df
# line_plot_df = foldwise_df.copy()[
#     (
#         (
#             foldwise_df["a-site-aware"]
#             #  & foldwise_df["var-per-asite"]
#         )
#         | (foldwise_df[f"{LINE_PLOT_A_SITE}-only"])
#     )
#     & foldwise_df["a_site"].isin([LINE_PLOT_A_SITE, "Others", "Overall"])
#     & (
#         (foldwise_df["few_shot_a_site"] == LINE_PLOT_A_SITE)
#         | foldwise_df["few_shot_a_site"].isna()
#     )
# ]

# with plt.rc_context({"ytick.minor.visible": True}):
#     for metric in ["crps_mean", "rmse", "picp"]:
#         fig, axes = plt.subplots(
#             nrows=1,
#             ncols=2,
#             figsize=(4.5 * figure_zoom_multiplier, 1.8 * figure_zoom_multiplier),
#             constrained_layout=True,
#             dpi=300,
#             sharey=True,
#         )

#         a_site_only_df = line_plot_df[line_plot_df[f"{LINE_PLOT_A_SITE}-only"]]
#         line_plot_df_target_only = line_plot_df[
#             (line_plot_df["a_site"] == LINE_PLOT_A_SITE)
#             & ~line_plot_df[f"{LINE_PLOT_A_SITE}-only"]
#         ]

#         var_per_asite_groups = line_plot_df_target_only.groupby("var-per-asite")

#         for ax, (var_per_asite, df) in zip(axes, var_per_asite_groups):
#             df_plot = pd.concat([df, a_site_only_df])

#             sns.lineplot(
#                 data=df_plot,
#                 x="few_shot_ratio",
#                 y=metric,
#                 hue="fp_type",
#                 marker="o",
#                 # style="var-per-asite",
#                 # marker=True,
#                 # hue_order=["fp2", "no-correl", "total-correl", f"{LINE_PLOT_A_SITE}-only"],
#                 errorbar="se",  # <-- across-fold 95% CI band; use "sd" or "se" if preferred
#                 # err_style="bars",     # or "bars"
#                 # errorbar=None,
#                 ax=ax,
#             )

#             if var_per_asite:
#                 ax.set_title("individual signal variance", pad=-12, y=1.0)
#             else:
#                 ax.set_title("shared signal variance", pad=-12, y=1.0)

#             ax.set_xlabel(None)
#             ax.set_ylabel(None)

#             (lines, labels) = ax.get_legend_handles_labels()
#             lines.insert(4, Line2D([1], [1], linestyle="none", marker="none"))
#             labels.insert(4, "")

#             if metric == "picp":
#                 # ax.set_ylim(ax.get_ylim()[0], 1.05)
#                 ax.axhline(0.95, color="red", linestyle="--", label="95% Interval")
#                 ax.legend(lines, labels, loc="lower right")
#             else:
#                 # ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.1)
#                 ax.legend(lines, labels, bbox_to_anchor=(1, 0.92), loc="upper right")

#         # 1. Calculate minimum and maximum limits across all axes
#         min_y = min(ax.get_ylim()[0] for ax in axes)
#         max_y = max(ax.get_ylim()[1] for ax in axes)

#         # 2. Apply those limits to each axis
#         for ax in axes:
#             if metric == "picp":
#                 ax.set_ylim(min_y, 1.05)
#             else:
#                 ax.set_ylim(min_y, max_y * 1.0)

#         fig.suptitle(f"{LINE_PLOT_A_SITE} Few-Shot Learning: {name_mapping[metric]}")
#         fig.supxlabel("Few-Shot Ratio (%)")
#         fig.supylabel(name_mapping[metric])

################################## Fig 3c #####################################
# %%
LINE_PLOT_A_SITE = "MBAI"
LKJ_MASTER_CATEGORY = "none"  # lkj_corr rows with a different master category are excluded to avoid pooling across configs
METRIC = "crps_mean"

fp_label_mapping = {
    "fp2": "Chemical similarity" + r" ($\mathbf{B}$ = Tanimoto)",
    "lkj_corr": "Task correlations" + r" ($\mathbf{B}$ = learned)",
    "no-correl": "No inter-task interpolation" + r" ($\mathbf{B}$ = $\mathbf{I}$)",
    "total-correl": "Full interpolation" + r" ($\mathbf{B}$ = $\mathbf{J}$)",
}

# build from foldwise_df instead of validation_summary_df
line_plot_df = foldwise_df.copy()[
    (
        (foldwise_df["a-site-aware"]) & foldwise_df["var-per-asite"]
        | (foldwise_df[f"{LINE_PLOT_A_SITE}-only"])
    )
    & foldwise_df["a_site"].isin([LINE_PLOT_A_SITE])
    & (
        (foldwise_df["few_shot_a_site"] == LINE_PLOT_A_SITE)
        | foldwise_df["few_shot_a_site"].isna()
    )
    & (
        (foldwise_df["fp_type"] != "lkj_corr")
        | (foldwise_df["lkj_master_category"] == LKJ_MASTER_CATEGORY)
    )
]

line_plot_df["label"] = (
    line_plot_df["fp_type"]
    + "_"
    + line_plot_df["var-per-asite"].map({True: "var-per-asite", False: ""})
)

line_plot_df["legend_label"] = line_plot_df["fp_type"].map(fp_label_mapping)

with plt.rc_context({"ytick.minor.visible": True}):
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH * 1 / 3, DOUBLE_COLUMN_WIDTH * 1 / 3 * 0.75),
        constrained_layout=True,
        dpi=300,
        sharey=True,
    )

    sns.lineplot(
        data=line_plot_df[~line_plot_df[f"{LINE_PLOT_A_SITE}-only"]],
        x="few_shot_ratio",
        y=METRIC,
        hue="legend_label",
        marker="o",
        # marker=True,
        # hue_order=["fp2", "no-correl", "total-correl", f"{LINE_PLOT_A_SITE}-only"],
        errorbar="se",  # <-- across-fold 95% CI band; use "sd" or "se" if preferred
        # err_style="bars",     # or "bars"
        # errorbar=None,
        ax=ax,
    )

    ax.set_xlabel(f"Target ({LINE_PLOT_A_SITE}) data points (count, fraction %)")
    ax.set_ylabel("CRPS")
    ax.legend(
        loc="upper right",
        frameon=False,
        title="Coregionalization $\\mathbf{B}$",
        alignment="left",
    )

    # from MBAI-only, few-shot ratio 100%, draw a horizontal line
    axhline_y = line_plot_df[
        (line_plot_df["few_shot_ratio"] == 100)
        & (line_plot_df[f"{LINE_PLOT_A_SITE}-only"])
    ][METRIC].mean()
    ax.axhline(y=axhline_y, linestyle="--", color="gray", zorder=-1)
    y_scale = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(
        -1,
        axhline_y - y_scale * 0.03,
        "Baseline (w/o MTGP)",
        color="gray",
        fontsize="x-small",
        ha="left",
        va="top",
    )

    ax.set_ylim(*ax.get_ylim())

    # draw vline at few-shot ratio 35% to indicate the best performance of fp2
    ax.axvline(x=35, linestyle="--", color="C0", zorder=-1)
    ax.text(
        35,
        ax.get_ylim()[0] - 0.03,
        "35%",
        color="C0",
        fontweight="bold",
        fontsize="small",
        ha="center",
        va="top",
        clip_on=False,
    )

    target_a_site_num = cv_sample_df[cv_sample_df["a_site"] == LINE_PLOT_A_SITE][
        "sample_name"
    ].nunique()

    def _fn_tick_labels(x, pos):
        return str(int(x / 100 * target_a_site_num)) + "\n" + f"({x:.0f}%)"

    ax.xaxis.set_major_formatter(FuncFormatter(_fn_tick_labels))

    fig.suptitle("c", x=0.0, y=1, horizontalalignment="left", verticalalignment="top")
    fig.savefig(
        f"../manuscript_assets/MTGP/3c_{LINE_PLOT_A_SITE}_few_shot_learning_{METRIC}.svg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# %%
################################### Fig 3c SI #####################################

LKJ_MASTER_CATEGORY = "none"
fp_label_mapping = {
    "fp2": "Chemical similarity" + r" ($\mathbf{B}$ = Tanimoto)",
    "lkj_corr": "Task correlations" + r" ($\mathbf{B}$ = learned)",
    "no-correl": "No inter-task interpolation" + r" ($\mathbf{B}$ = $\mathbf{I}$)",
    "total-correl": "Full interpolation" + r" ($\mathbf{B}$ = $\mathbf{J}$)",
}

with plt.rc_context({"ytick.minor.visible": True}):
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.75),
        constrained_layout=True,
        sharex="col",
        dpi=300,
    )

    for i, metric in enumerate(["crps_mean", "mae", "picp"]):
        for j, (a_site, ax) in enumerate(
            zip(["ClMBAI", "MBAI", "MeOMBAI"], axes[i].flatten())
        ):
            line_plot_df = foldwise_df.copy()[
                (
                    (foldwise_df["a-site-aware"]) & foldwise_df["var-per-asite"]
                    | (foldwise_df[f"{a_site}-only"])
                )
                & foldwise_df["a_site"].isin([a_site])
                & (
                    (foldwise_df["few_shot_a_site"] == a_site)
                    | foldwise_df["few_shot_a_site"].isna()
                )
                & (
                    (foldwise_df["fp_type"] != "lkj_corr")
                    | (foldwise_df["lkj_master_category"] == LKJ_MASTER_CATEGORY)
                )
            ]

            line_plot_df["label"] = (
                line_plot_df["fp_type"]
                + "_"
                + line_plot_df["var-per-asite"].map({True: "var-per-asite", False: ""})
            )

            line_plot_df["legend_label"] = line_plot_df["fp_type"].map(fp_label_mapping)

            sns.lineplot(
                data=line_plot_df[~line_plot_df[f"{a_site}-only"]],
                x="few_shot_ratio",
                y=metric,
                hue="legend_label",
                marker="o",
                errorbar="se",  # <-- across-fold 95% CI band; use "sd" or "se" if preferred
                ax=ax,
            )

            ax.legend().set_visible(False)

            ax.set_xlabel(
                f"Target ({a_site}) data points (count, fraction %)", labelpad=8
            )
            ax.set_ylabel(metric_name_mapping[metric])
            # ax.legend(loc="best", frameon=False)

            if metric == "picp":
                ax.axhline(0.95, color="red", linestyle="--")
                ax.set_ylim(0.69, 1.01)
                if j == 2:
                    ax.text(
                        100, 0.955, "95% Interval", color="red", ha="right", va="bottom"
                    )

            axhline_y = line_plot_df[
                (line_plot_df["few_shot_ratio"] == 100)
                & (line_plot_df[f"{a_site}-only"])
            ][metric].mean()
            ax.axhline(y=axhline_y, linestyle="--", color="gray", zorder=-1)
            y_scale = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(
                (-1 if metric != "picp" else 70),
                axhline_y - y_scale * 0.03,
                "Baseline (w/o MTGP)",
                color="gray",
                fontsize="small",
                ha="left",
                va="top",
            )

            if i == 0:
                ax.set_title(a_site, y=1.0, pad=-10)

            target_a_site_num = cv_sample_df[cv_sample_df["a_site"] == a_site][
                "sample_name"
            ].nunique()

            def _fn_tick_labels(x, pos, target_a_site_num=target_a_site_num):
                return str(int(x / 100 * target_a_site_num)) + "\n" + f"({x:.0f}%)"

            ax.xaxis.set_major_formatter(FuncFormatter(_fn_tick_labels))

    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncol=4)
    fig.align_ylabels()

    fig.savefig(
        "../manuscript_assets/si/3c_full.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

# %%
################################### SI variance modes #####################################

single_task_masks = []
for LINE_PLOT_A_SITE in ["ClMBAI", "MBAI", "MeOMBAI"]:
    single_task_mask = foldwise_df[f"{LINE_PLOT_A_SITE}-only"]
    single_task_masks.append(single_task_mask)

# merge to a single mask
single_task_mask = np.logical_or.reduce(single_task_masks)
with plt.rc_context({"ytick.minor.visible": True, "xtick.direction": "out"}):
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.75),
        constrained_layout=True,
        dpi=300,
        sharey="row",
    )

    for i, metric in enumerate(["crps_mean", "mae", "picp"]):
        for j, (a_site, ax) in enumerate(
            zip(["ClMBAI", "MBAI", "MeOMBAI", "Overall"], axes[i].flatten())
        ):
            compare_var_per_asite = foldwise_df.copy()[
                (foldwise_df["a-site-aware"])
                & (~single_task_mask)
                & (foldwise_df["fp_type"] != "lkj_corr")
                & (foldwise_df["a_site"] == a_site)
                & (foldwise_df["few_shot_ratio"] == 100)
            ]

            compare_var_per_asite["var-per-asite"] = compare_var_per_asite[
                "var-per-asite"
            ].map({True: "individual", False: "shared"})
            compare_var_per_asite["legend_label"] = compare_var_per_asite[
                "fp_type"
            ].map(fp_label_mapping)

            sns.barplot(
                data=compare_var_per_asite,
                x="legend_label",
                y=metric,
                hue="var-per-asite",
                errorbar="se",
                ax=ax,
            )

            ax.set_xlabel(None)
            if j == 0:
                ax.set_ylabel(metric_name_mapping[metric])

            ax.legend().set_visible(False)
            ax.set_title(a_site, y=1.0, pad=-12)

            if metric == "picp":
                ax.axhline(0.95, color="red", linestyle="--")
                ax.set_ylim(0, 1.13)
                if j == 3:
                    ax.text(
                        2.5, 0.96, "95% Interval", color="red", ha="right", va="bottom"
                    )

    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside center right", title="Variance", frameon=False
    )
    fig.supxlabel("Model task correlation", y=0, va="top")
    fig.align_ylabels()
    fig.get_layout_engine().set(h_pad=0.15)

    fig.savefig(
        "../manuscript_assets/si/3_variance_mode.png",
        dpi=300,
    )

# %%
# ============================================================
# Paired analysis of the coregionalization ablation
# All models share identical folds / splits / reference data,
# so model-vs-model comparisons should be made WITHIN fold.
# ============================================================


LINE_PLOT_A_SITE = "ClMBAI"
LKJ_MASTER_CATEGORY = "none"
METRIC = "crps_mean"  # lower is better
REFERENCE_MODEL = "fp2"  # Tanimoto = the proposed model
N_BOOT = 10_000
RNG = np.random.default_rng(0)

# ------------------------------------------------------------
# 0. Identify the fold column
# ------------------------------------------------------------
_fold_candidates = [
    c
    for c in foldwise_df.columns
    if c.lower() in {"fold", "fold_id", "fold_idx", "split", "repeat", "seed"}
]
if not _fold_candidates:
    raise KeyError(
        f"Could not find a fold column. Available columns:\n{list(foldwise_df.columns)}\n"
        "Set FOLD_COL manually below."
    )
FOLD_COL = _fold_candidates[0]
print(f"Using fold column: '{FOLD_COL}'")

# ------------------------------------------------------------
# 1. Same filtering as before
# ------------------------------------------------------------
line_plot_df = foldwise_df.copy()[
    (
        (foldwise_df["a-site-aware"]) & foldwise_df["var-per-asite"]
        | (foldwise_df[f"{LINE_PLOT_A_SITE}-only"])
    )
    & foldwise_df["a_site"].isin([LINE_PLOT_A_SITE])
    & (
        (foldwise_df["few_shot_a_site"] == LINE_PLOT_A_SITE)
        | foldwise_df["few_shot_a_site"].isna()
    )
    & (
        (foldwise_df["fp_type"] != "lkj_corr")
        | (foldwise_df["lkj_master_category"] == LKJ_MASTER_CATEGORY)
    )
]

fp_label_mapping = {
    "fp2": r"$\mathbf{B}$ = Tanimoto",
    "no-correl": r"$\mathbf{B}$ = identity (lengthscales only)",
    "total-correl": r"$\mathbf{B}$ = constant (fully pooled)",
    "lkj_corr": r"$\mathbf{B}$ = learned (LKJ)",
}
line_plot_df["legend_label"] = line_plot_df["fp_type"].map(fp_label_mapping)

mtgp_df = line_plot_df[~line_plot_df[f"{LINE_PLOT_A_SITE}-only"]].copy()
baseline_df = line_plot_df[
    (line_plot_df[f"{LINE_PLOT_A_SITE}-only"]) & (line_plot_df["few_shot_ratio"] == 100)
].copy()

# ------------------------------------------------------------
# 2. Wide table: one row per (fold, fraction), one column per model
#    This is what makes the pairing possible.
# ------------------------------------------------------------
wide = mtgp_df.pivot_table(
    index=[FOLD_COL, "few_shot_ratio"], columns="fp_type", values=METRIC
).sort_index()

n_missing = wide.isna().sum().sum()
if n_missing:
    print(
        f"WARNING: {n_missing} missing model/fold combinations — pairing will drop these rows."
    )

# per-fold baseline (single value per fold, independent of fraction)
baseline_by_fold = baseline_df.groupby(FOLD_COL)[METRIC].mean()
baseline_mean = baseline_by_fold.mean()

comparators = [
    m for m in ["no-correl", "lkj_corr", "total-correl"] if m in wide.columns
]
fractions = sorted(wide.index.get_level_values("few_shot_ratio").unique())

# ------------------------------------------------------------
# 3. Paired statistics
# ------------------------------------------------------------
rows = []
for frac in fractions:
    sub = wide.xs(frac, level="few_shot_ratio")
    for comp in comparators:
        pair = sub[[REFERENCE_MODEL, comp]].dropna()
        if len(pair) < 2:
            continue
        # positive => comparator has HIGHER CRPS => Tanimoto is better
        d = pair[comp] - pair[REFERENCE_MODEL]
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        tstat, pval = stats.ttest_rel(pair[comp], pair[REFERENCE_MODEL])
        try:
            _, p_wilcox = stats.wilcoxon(pair[comp], pair[REFERENCE_MODEL])
        except ValueError:
            p_wilcox = np.nan
        rho = pair[REFERENCE_MODEL].corr(pair[comp])
        marginal_se = pair[REFERENCE_MODEL].std(ddof=1) / np.sqrt(n)
        rows.append(
            dict(
                few_shot_ratio=frac,
                comparator=comp,
                n_folds=n,
                mean_delta=d.mean(),
                se_delta=se,
                ci_lo=d.mean() - 1.96 * se,
                ci_hi=d.mean() + 1.96 * se,
                p_paired_t=pval,
                p_wilcoxon=p_wilcox,
                wins=(d > 0).sum(),
                fold_corr=rho,
                se_ratio=se / marginal_se if marginal_se > 0 else np.nan,
            )
        )

paired_stats = pd.DataFrame(rows)

print("\n" + "=" * 78)
print("PAIRED COMPARISON vs B = Tanimoto  (positive delta = Tanimoto better)")
print("=" * 78)
print(
    paired_stats.assign(
        summary=lambda t: t.apply(
            lambda r: (
                f"{r.mean_delta:+.3f} ± {r.se_delta:.3f}  "
                f"(p={r.p_paired_t:.4f}, wins {int(r.wins)}/{int(r.n_folds)}, "
                f"fold-corr={r.fold_corr:.2f}, SE shrunk to {r.se_ratio:.0%})"
            ),
            axis=1,
        )
    )[["few_shot_ratio", "comparator", "summary"]].to_string(index=False)
)

print(
    f"\nMean fold-level correlation between arms: {paired_stats.fold_corr.mean():.3f}"
)
print("  -> the closer to 1, the more the marginal error bars overstate the")
print("     uncertainty of a model-vs-model comparison.")


# ------------------------------------------------------------
# 4. Bootstrap CI for the crossing point vs the single-task baseline
# ------------------------------------------------------------
def crossing_fraction(mean_by_frac, fracs):
    """First fraction at which mean paired difference crosses zero (linear interp)."""
    y = np.asarray(mean_by_frac, float)
    x = np.asarray(fracs, float)
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1]:
            return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
    return np.nan


tan = wide[REFERENCE_MODEL].unstack("few_shot_ratio")  # folds x fractions
common_folds = tan.index.intersection(baseline_by_fold.index)
tan = tan.loc[common_folds]
base = baseline_by_fold.loc[common_folds]

# D_f(r) = Tanimoto_f(r) - baseline_f ; crossing where mean D hits 0
D = tan.sub(base, axis=0)
point_estimate = crossing_fraction(D.mean(axis=0).values, D.columns.values)

boot = []
folds_arr = np.array(common_folds)
for _ in range(N_BOOT):
    idx = RNG.choice(len(folds_arr), size=len(folds_arr), replace=True)
    boot.append(crossing_fraction(D.iloc[idx].mean(axis=0).values, D.columns.values))
boot = np.array([b for b in boot if np.isfinite(b)])

print("\n" + "=" * 78)
print("CROSSING POINT vs single-task baseline (paired, bootstrapped over folds)")
print("=" * 78)
if np.isfinite(point_estimate) and boot.size:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Crossing fraction : {point_estimate:.1f}%  (95% CI {lo:.1f}–{hi:.1f}%)")
    print(
        f"  Data reduction    : {100 - point_estimate:.0f}%  (95% CI {100 - hi:.0f}–{100 - lo:.0f}%)"
    )
    print("  --> quote this interval in the abstract instead of a bare '65%'")
else:
    print("  No zero-crossing found on the sampled grid.")

# ------------------------------------------------------------
# 5. Figure: (c) learning curves, (d) paired differences
# ------------------------------------------------------------
palette = {
    fp_label_mapping["fp2"]: "#E69F00",  # orange
    fp_label_mapping["lkj_corr"]: "#56B4E9",  # sky blue
    fp_label_mapping["no-correl"]: "#009E73",  # green
    fp_label_mapping[
        "total-correl"
    ]: "#CC79A7",  # magenta (replaces low-contrast yellow)
}
dashes = {k: (None, None) for k in palette}
markers = {
    fp_label_mapping["fp2"]: "o",
    fp_label_mapping["lkj_corr"]: "s",
    fp_label_mapping["no-correl"]: "^",
    fp_label_mapping["total-correl"]: "D",
}

with plt.rc_context({"ytick.minor.visible": True}):
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH * 2 / 3, DOUBLE_COLUMN_WIDTH * 1 / 3 * 0.72),
        constrained_layout=True,
        dpi=300,
    )

    # ---------------- panel c: absolute CRPS ----------------
    sns.lineplot(
        data=mtgp_df,
        x="few_shot_ratio",
        y=METRIC,
        hue="legend_label",
        style="legend_label",
        markers=markers,
        dashes=dashes,
        palette=palette,
        errorbar="se",
        err_kws={"alpha": 0.10, "linewidth": 0},
        ax=ax,
    )
    ax.set_xlabel("Target (MBAI) data fraction (%)")
    ax.set_ylabel("CRPS")
    ax.legend(loc="upper right", frameon=False, fontsize="x-small")
    ax.axhline(y=baseline_mean, linestyle="--", color="gray", zorder=-1)
    ax.text(
        -1,
        baseline_mean - 0.07,
        "Baseline (w/o MTGP)\ntrained with 100%\nMBAI dataset",
        color="gray",
        fontsize="x-small",
        ha="left",
        va="top",
    )
    ax.set_ylim(*ax.get_ylim())

    if np.isfinite(point_estimate):
        ax.axvline(
            x=point_estimate,
            linestyle="--",
            color=palette[fp_label_mapping["fp2"]],
            zorder=-1,
        )
        ax.text(
            point_estimate,
            ax.get_ylim()[0] - 0.02,
            f"{point_estimate:.0f}",
            color=palette[fp_label_mapping["fp2"]],
            fontweight="bold",
            fontsize="medium",
            ha="center",
            va="top",
            clip_on=False,
        )
        star_y = wide[REFERENCE_MODEL].unstack("few_shot_ratio").mean(axis=0)
        nearest = min(fractions, key=lambda f: abs(f - point_estimate))
        ax.scatter(
            point_estimate,
            star_y[nearest],
            marker="*",
            color="gold",
            s=150,
            zorder=5,
            edgecolor="black",
        )

    # ---------------- panel d: paired differences ----------------
    ax2.axhline(0, color="k", lw=1, zorder=1)
    ax2.annotate(
        "Tanimoto better",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        fontsize="x-small",
        color="gray",
        va="top",
    )
    ax2.annotate(
        "Tanimoto worse",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        fontsize="x-small",
        color="gray",
        va="bottom",
    )

    for comp in comparators:
        s = paired_stats[paired_stats.comparator == comp].sort_values("few_shot_ratio")
        lbl = fp_label_mapping[comp]
        ax2.plot(
            s.few_shot_ratio,
            s.mean_delta,
            marker=markers[lbl],
            color=palette[lbl],
            label=lbl,
            ms=4,
        )
        ax2.fill_between(
            s.few_shot_ratio,
            s.mean_delta - s.se_delta,
            s.mean_delta + s.se_delta,
            color=palette[lbl],
            alpha=0.22,
            lw=0,
        )
        sig = s[s.p_paired_t < 0.05]
        ax2.scatter(
            sig.few_shot_ratio,
            sig.mean_delta + sig.se_delta + 0.04,
            marker="*",
            s=22,
            color=palette[lbl],
            zorder=6,
        )

    ax2.set_xlabel("Target (MBAI) data fraction (%)")
    ax2.set_ylabel(r"$\Delta$CRPS (comparator $-$ Tanimoto)")
    ax2.legend(loc="upper right", frameon=False, fontsize="x-small")

    fig.suptitle("c", x=0.0, y=1.0, ha="left", va="top")
    ax2.set_title("d", loc="left", x=-0.18, y=1.0, fontweight="bold")

plt.show()
