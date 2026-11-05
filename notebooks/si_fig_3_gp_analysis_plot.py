# %%
from natsort import natsorted
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from configs.nature_mpl_rc import nature_rc, DOUBLE_COLUMN_WIDTH
from configs.paths import BO_RESULTS_DIR

plt.rcParams.update(nature_rc)
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)

# %%
results_dir = BO_RESULTS_DIR

posterior_files = natsorted(
    filter(
        lambda f: "old" not in f.name, results_dir.glob("gp_train_results_batch=*.npz")
    )
)
posterior_files = list(
    filter(
        lambda f: not f.name.endswith("score_sigma_per_chemical.npz"), posterior_files
    )
)
if len(posterior_files) < 2:
    raise FileNotFoundError(
        f"Need at least 2 posterior result files in {results_dir}; found {len(posterior_files)}."
    )

analysis_data = []
for file in posterior_files:
    data = np.load(file, allow_pickle=True)
    analysis_data.append(
        {
            "file": file.name,
            "train_batches": data["train_batches"].astype(str).tolist(),
            "log_likelihood": float(data["log_likelihood"]),
            "hyperparameters": np.asarray(data["hyperparameters"], dtype=np.float64),
        }
    )
    data.close()

analysis_df = pd.DataFrame(analysis_data)
print("Loaded batch metadata from:")
print(analysis_df.to_markdown(tablefmt="github", index=False))

# %%


hps_names = {
    "variance": "Signal variance",
    "ls_anneal_temp": "Lengthscale: Temp",
    "ls_anneal_time": "Lengthscale: Time",
    "ls_volume": "Lengthscale: Dvol",
    "ls_spin_speed": "Lengthscale: Rpm",
    "ls_dispense_speed": "Lengthscale: Dspd",
    "ls_conc": "Lengthscale: Conc",
    "ls_a_site": "ls_a_site",
    "noise": "Noise",
    "mean": "Mean",
}

analysis_df[list(hps_names.keys())] = pd.DataFrame(
    analysis_df["hyperparameters"].tolist(), index=analysis_df.index
)
analysis_df["x"] = analysis_df["train_batches"].apply(
    lambda batches: int(batches[-1])
)  # Use the last batch number as x-axis

analysis_df = analysis_df[analysis_df["x"] >= 8]

show_log_likelihood = False

if show_log_likelihood:
    hps_names = {"log_likelihood": "Log Likelihood"} | hps_names

subplot_keys = list(hps_names.keys())

subplot_keys.pop(subplot_keys.index("ls_a_site"))  # Remove ls_a_site from the plot

# plot log-likelihood and hyperparameters vs. x
fig, axes = plt.subplots(
    nrows=5,
    ncols=2,
    figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH),
    sharex="row",
    layout="constrained",
)
axes = axes.flatten()
for i, col in enumerate(subplot_keys):
    axes[i].plot(analysis_df["x"], analysis_df[col], marker="o")

    # x tick labels as int
    axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    # move title into the plot area
    axes[i].title.set_position([0.5, 0.85])
    axes[i].set_ylabel(hps_names[col].split(":")[0])
    if col.startswith("ls_"):
        axes[i].set_ylim(0, 1.82)

    axes[i].set_title(
        hps_names[col],
        y=1.0,
        va="top",
        pad=-10,
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    )

axes[-1].set_axis_off()  # Hide the last empty subplot

axes[-2].set_xlabel("Batch")
axes[-3].set_xlabel("Batch")

fig.align_ylabels()
fig.savefig("../manuscript_assets/si/3_gp_hps_over_batches.png", dpi=300)
# %%
