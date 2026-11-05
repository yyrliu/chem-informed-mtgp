# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH
from configs.paths import BO_RESULTS_DIR
from matplotlib.colors import ListedColormap, to_rgba

plt.rcdefaults()
plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "axes.spines.right": True,
    }
)

csvs = [
    BO_RESULTS_DIR / "20260518_165719_kl_div_results.csv",
]

results_dfs = []
for csv in csvs:
    df = pd.read_csv(csv, skiprows=4)
    results_dfs.append(df)

results_df = pd.concat(results_dfs, ignore_index=True)

# skip the first 2 rows (LPS-5-8 are random sampling)
results_df = results_df[2:].reset_index(drop=True)

results_df["kl_div_norm"] = results_df["kl_div"] / results_df["kl_div"].max()
results_df["kl_div_sem_norm"] = results_df["kl_div_sem"] / results_df["kl_div"].max()

# x labels and positions for grouped bars
labels = results_df["batch_b"].str.split("-").str[1].tolist()
x = np.arange(len(labels))
width = 0.8

# %%

fig, ax1 = plt.subplots(
    figsize=(DOUBLE_COLUMN_WIDTH * 1 / 3, DOUBLE_COLUMN_WIDTH * 1 / 3 * 0.72),
    dpi=300,
    constrained_layout=True,
)
ax2 = ax1.twinx()

cm_order = [0, 4, 5, 2, 3, 4, 5]

colors = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in cm_order]

cm = ListedColormap(colors)

bars1 = ax1.bar(
    # x - width / 2,
    x,
    results_df["kl_div_norm"],
    width=width,
    yerr=results_df["kl_div_sem_norm"],
    color=cm.colors[0],
    label="KL Div",
)

results_df["kl_div_cumsum"] = results_df["kl_div_norm"].cumsum()
results_df["kl_div_cumsum_norm"] = (
    results_df["kl_div_cumsum"] / results_df["kl_div_cumsum"].max()
)

line1 = ax2.plot(
    x,
    results_df["kl_div_cumsum_norm"],
    marker="o",
    markersize=4,
    color=cm.colors[1],
    label="Cumulative KL Div",
)

ax1.set_ylim(0, 1.26)
ax2.set_ylim(0, 1.04)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
ax1.set_xlim(-0.8, len(labels) - 0.2)
ax1.set_xlabel("Batch")
ax1.set_ylabel("KL Div (rel.)")
ax2.set_ylabel("Cumulative KL Div (rel.)", color=cm.colors[1])
ax2.tick_params(axis="y", labelcolor=cm.colors[1])

# annote with 90 degree arrow pointing to the bar at x=11 with text "New high score discovery"
ax1.annotate(
    "New high score\ndiscovery",
    xy=(11, 0.41),
    xytext=(11, 0.51),
    ha="center",
    fontsize="small",
    color=cm.colors[2],
    arrowprops=dict(
        arrowstyle="-[, lengthB=0.4",
        color=cm.colors[2],
        lw=1,
        shrinkA=0,
        # connectionstyle="angle,angleA=0,angleB=90,rad=10",
    ),
)
rect = ax1.axvspan(13.5, 18.5, ymax=0.995, color=to_rgba(cm.colors[3], 0.15), zorder=0)
rect.set(edgecolor=cm.colors[3], linewidth=1)

ax1.text(
    16.0,
    0.78,
    "Targeted\nExploration",
    fontsize="small",
    color=cm.colors[3],
    ha="center",
    va="center",
)

# combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

fig.suptitle("b", x=0.0, y=1.0, horizontalalignment="left", verticalalignment="top")

fig.savefig(
    "../manuscript_assets/3b_kl_div.svg",
    dpi=300,
    bbox_inches="tight",
)

# %%
