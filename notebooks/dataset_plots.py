# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH

import matplotlib as mpl
from cycler import cycler

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dash_lib.data_utils import query_samples, load_dataset  # noqa: E402


plt.rcParams.update(nature_rc)

plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)

# %%

y_label_mapping = {
    "gen_CD": r"$CD_{gen}$ (mdeg)",
    "g_factor_10e4": r"$g$-factor $\times 10^{-4}$",
    "ldlb_CD": r"$CD_{LDLB}$ (mdeg)",
    "abs_avg": r"Abs",
}

db_path = Path(__file__).resolve().parent.parent / "mydb.duckdb"
print(f"Loading dataset from {db_path}")
# %%

samples = query_samples(db_path=str(db_path))
sorted_ids = samples.index.tolist()
ds = load_dataset(sorted_ids, "cd", async_mode=False, db_path=str(db_path))

# %%

fig, axes = plt.subplots(
    nrows=4,
    ncols=3,
    figsize=(DOUBLE_COLUMN_WIDTH * 0.8, DOUBLE_COLUMN_WIDTH),
    dpi=300,
    sharex=True,
    sharey="row",
    constrained_layout=True,
)

for a_site, rol_axes in zip(["MBAI", "MeOMBAI", "ClMBAI"], axes.T):
    mask = ds["a_site"] == a_site
    ds_plot = ds.where(mask, drop=True)

    colors = mpl.colormaps["viridis"](np.linspace(0, 1, len(ds_plot["sample_id"])))
    custom_cycler = cycler(color=colors)

    ds_plot["g_factor_10e4"] = ds_plot["g_factor"] * 1e4

    for ax, col in zip(rol_axes, y_label_mapping.keys()):
        ax.set_prop_cycle(custom_cycler)

        ds_plot[col].sel(wavelength=slice(550, 400)).plot.line(
            x="wavelength", ax=ax, add_legend=False, lw=0.3, alpha=0.5
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        # hline at y=0
        if col != "abs_avg":
            ax.axhline(0, color="black", lw=0.5, ls="--")

    rol_axes[0].set_title(
        a_site,
        fontsize="medium",
        fontweight="regular",
        y=1.0,
        pad=-10,
    )

for ax, col in zip(axes[:, 0], y_label_mapping.keys()):
    ax.set_ylabel(y_label_mapping[col])

fig.supxlabel("Wavelength (nm)")
fig.align_ylabels()
fig.get_layout_engine().set(h_pad=0.05, hspace=0.0)

fig.savefig("../manuscript_assets/si/2_cd_overview.png", dpi=300)
# %%
