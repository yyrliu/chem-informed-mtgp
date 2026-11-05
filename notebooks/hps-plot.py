# %%
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt


from configs.exp_config import NOTEBOOK_CONFIG
from helpers.cv_cache import load_hps, run_dir
from configs.nature_mpl_rc import nature_rc, SINGLE_COLUMN_WIDTH

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))  # noqa: E402

########## set up plotting parameters ##########
plt.rcdefaults()
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
################################################

# %%
########## setup macro (mirrors transfer-learning-plot.py) ##########
param_config = NOTEBOOK_CONFIG["param_config"]
A_SITE_TO_SMILES = NOTEBOOK_CONFIG["a_site_to_smiles"]
A_SITES = list(A_SITE_TO_SMILES.keys())

CONTINUOUS_FEATURE_COUNT = int(os.environ.get("CONTINUOUS_FEATURE_COUNT", 6))
CONTINUOUS_COLUMNS = list(param_config.keys())[:CONTINUOUS_FEATURE_COUNT]

RUN_DIRS = [run_dir(s) for s in range(42, 42 + 30)]
#######################################################################

# %%
print("Loading cv_hps_*.csv from all run dirs (cached)...")
HPS_DF = load_hps(CONTINUOUS_COLUMNS, A_SITES, RUN_DIRS)
print(f"Loaded {HPS_DF['file'].nunique()} files, {len(HPS_DF)} rows.")

# %%
HPS_DF["fp_type"].unique()
# %%

LINE_PLOT_A_SITE = "MBAI"
LKJ_MASTER_CATEGORY = "none"

line_plot_df = HPS_DF.copy()[
    ((HPS_DF["few_shot_a_site"] == LINE_PLOT_A_SITE) | HPS_DF["few_shot_a_site"].isna())
    & (
        (HPS_DF["fp_type"] != "lkj_corr")
        | (HPS_DF["lkj_master_category"] == LKJ_MASTER_CATEGORY)
    )
    & ((HPS_DF["group"] == "lkj") & HPS_DF["name"].str.contains("corr"))
]

fig, ax = plt.subplots(
    figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.7),
    constrained_layout=True,
    dpi=300,
)

sns.boxplot(
    data=line_plot_df,
    x="few_shot_ratio",
    y="value",
    hue="name",
    flierprops=dict(markersize=1, alpha=0.5),
    # width=1,
    gap=0.2,
    ax=ax,
)

ax.set_xlabel(f"Target ({LINE_PLOT_A_SITE}) data fraction (%)")
ax.set_ylabel("Correlation")

# move legend to the bottom right
sns.move_legend(ax, loc="lower right", title=None, frameon=False)

fig.savefig(f"../manuscript_assets/si/3c_{LINE_PLOT_A_SITE}_LKJ_mat.png", dpi=300)
# line_plot_df.groupby(["name", "few_shot_ratio"])["value"].describe()

# %%

LINE_PLOT_A_SITE = "MeOMBAI"
LKJ_MASTER_CATEGORY = "none"

line_plot_df = HPS_DF.copy()[
    ((HPS_DF["few_shot_a_site"] == LINE_PLOT_A_SITE) | HPS_DF["few_shot_a_site"].isna())
    & (
        (HPS_DF["fp_type"] != "lkj_corr")
        | (HPS_DF["lkj_master_category"] == LKJ_MASTER_CATEGORY)
    )
    # & (HPS_DF["fp_type"] == "fp2")
    & (HPS_DF["group"] == "lengthscale")
    & (HPS_DF["name"].str.contains("spin_speed"))
]

# sns.lineplot(
#     data=line_plot_df,
#     x="few_shot_ratio",
#     y="value",
#     hue="name",
#     errorbar="sd",
# )

# plt.show()

ax = sns.violinplot(
    data=line_plot_df,
    x="few_shot_ratio",
    y="value",
    hue="name",
    inner="quart",
    # width=1,
    # gap=0.,
    saturation=0.5,
)
