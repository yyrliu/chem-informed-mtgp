# Chemistry-Informed Transfer and Interaction-Resolved Learning Towards Data-Efficient Materials Exploration

Code and analysis scripts for the manuscript of the same name (Liu, Alghalayini, Lee, Noack, Lüer, Brabec, Sutter-Fella). The repository contains the chemistry-informed multi-task GP and acquisition code, the Bayesian optimization (BO) notebook, the scripts that generate the manuscript and SI figures, and the data preprocessing pipeline. The experimental data are on Zenodo (see [Data](#data)).

## Repository layout

| Path | Contents |
| --- | --- |
| `BO_fingerprint_sigma_per_chemical.ipynb` | Main notebook: loads the campaign data, trains the fingerprint-kernel GP, proposes the next experiment batch, and computes the KL divergence between sequentially trained models. |
| `notebooks/` | Scripts (VS Code `# %%` cells) that generate the figures, run the cross-validation / transfer-learning benchmarks, and run the SHAP-IQ analysis. See [Reproducing the figures](#reproducing-the-figures). |
| `notebooks/configs/`, `notebooks/helpers/` | Shared paths, experiment configuration, plot styles, CV and hyperparameter utilities. |
| `src/chem_informed_mtgp/` | Python package: fingerprint kernels (NumPy/Numba), acquisition function, candidate pool, KL divergence, and CD/PL/XRD analysis. |
| `robot_recipe_prep.ipynb` | Turns selected BO candidates into batched robot recipes. |
| `scripts/` | Command-line tools for data preprocessing. |
| `dash_lib/`, `app.py`, `ingestion.py` | DuckDB ingestion and an optional Dash data browser. |
| `manuscript_assets/` | Figure outputs (SVG/PNG) written by the scripts in `notebooks/`. |
| `docs/` | [Data preparation](docs/DATA_PREPARATION.md), [data mapping](docs/DATA_MAPPING.md), and [data framework](docs/data_framework.md). |
| `tests/` | Unit tests for the kernels, acquisition function, candidate pool and data framework. |

The acquisition function described in the manuscript as qFC-LVR (Fingerprint-Conditioned Local Variance Reduction) is implemented as `local_ball_ivr_v3.py` in `src/chem_informed_mtgp/`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/yyrliu/chem-informed-mtgp.git
cd chem-informed-mtgp
uv sync
```

PyTorch is optional and selected through an extra (`cpu`, `cu128` or `xpu`), e.g. `uv sync --extra cpu`. All dependencies are specified in `pyproject.toml`. Run the tests with `uv run pytest`.

## Data

The experimental data are archived on Zenodo (LPS-3 to LPS-27 and LPS-32, BO results, and cross-validation results). Download and extract the archive to `chem-informed-mtgp-data-zenodo/` in the repository root, or point the `LPS_DATA_ROOT` environment variable at its location:

```
# PowerShell
$env:LPS_DATA_ROOT = "D:\data\chem-informed-mtgp-data-zenodo"
# bash
export LPS_DATA_ROOT=/data/chem-informed-mtgp-data-zenodo
```

`transfer_learning_cv/` ships as `please_extract.zip` and must be extracted in place before running the transfer-learning scripts. The BO notebook additionally reads processed measurements from `mydb.duckdb`, which is built by the preprocessing and ingestion steps. See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for the full workflow, folder layout and extraction commands.

## Bayesian optimization notebook

Open `BO_fingerprint_sigma_per_chemical.ipynb` with the project environment (`uv run jupyter lab`, or select the `.venv` kernel in VS Code) and run the cells in order: load and normalize the data, train the GP (or load the saved result from `BO_results_outlier_removed/`), build the candidate pool and select the next batch, and optionally compute the KL divergence between models trained on successive batches. Outputs are written to `BO_results_outlier_removed/`.

## Reproducing the figures

Scripts in `notebooks/` write figures to `manuscript_assets/`. Run them from the repository root, e.g. `uv run python notebooks/kl_div.py`. Some scripts are cell-based (`# %%`) and are best run interactively. Output file prefixes (e.g. `3c_`, `4_`) follow an earlier figure numbering and may not match the final figure numbers.

| Content | Script(s) |
| --- | --- |
| Framework schematics | `fig_1_dummy_plots.py` |
| Example CD spectra and PL-image homogeneity (GLCM) maps | `fig2a_plot.py`, `fig2b_plot.py` |
| Campaign overview: CD spectra, score distributions, cumulative counts | `dataset_plots.py`, `training_plots.py` |
| KL divergence over the campaign | `kl_div.py` |
| Knowledge-transfer benchmark (five-fold CV, few-shot) | `transfer-learning.py` (sweeps via `run_transfer_learning_sweep.py`), `transfer-learning-plot.py` |
| GP hyperparameters and covariance matrices | `hps-plot.py`, `si_fig_3_gp_analysis_plot.py`, `si_fig_3_covariance_matrix.py` |
| SHAP-IQ analysis | `shap_indexed_tanimoto.py` (computes the cached SHAP values), `shap_indexed_tanimoto_plot.py`, `shap_indexed_tanimoto_scatter_matrix.py` |
| Experimental testing of SHAP-derived conditions | `shap_verification_plots.py` |
| 7D parameter-space volume plots | `6d_visualization.py` |

`training.py` trains the GP for the scripts above, and the `*.pkl` files in `notebooks/` are cached SHAP results.

## Data preparation

The pipeline that links raw UV-Vis/PL/CD/XRD files to recipes, preprocesses them and ingests them into DuckDB is documented in [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md). An optional Dash app (`uv run python app.py`) lets you browse the processed data.

## Citation

Citation and Zenodo DOI to be added.
