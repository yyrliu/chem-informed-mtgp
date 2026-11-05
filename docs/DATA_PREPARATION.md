# Data Preparation

This document describes how the raw experimental data (UV-Vis/PL H5 files, CD spectra, optional XRD, and recipes) are turned into the processed dataset used by the analysis notebooks. The pipeline is: **link files to recipes → preprocess H5 → ingest into DuckDB**. See [DATA_MAPPING.md](DATA_MAPPING.md) for how files are matched.

If you only want to reproduce the figures, the Zenodo archive already contains the inputs the scripts need. You need this pipeline only to re-process raw data, or to build `mydb.duckdb`, which `BO_fingerprint_sigma_per_chemical.ipynb` reads through `dash_lib.data_utils.query_samples`.

## Setup

```
git clone https://github.com/yyrliu/chem-informed-mtgp.git
cd chem-informed-mtgp
uv sync
```

## Data layout

Each experiment batch (e.g. `LPS-27`) is a folder with the following layout. Only `CBox/` and the recipes JSON are required; `CD/` and `XRD/` are optional.

```
LPS-27/
├── total_correlation_recipes.json   # one recipe per sample, in sample order (any *recipe*.json)
├── CBox/                            # raw UV-Vis/PL H5 files: {user}_{batch}_{pos}_{sample_id}_run{n}_spec_run.h5
├── CD/                              # front-{id}.jws|txt, back-{id}.jws|txt
└── XRD/                             # (optional) *_{well}_*.gfrm or .xy
```

The Zenodo archive ships the raw `CBox/` H5 files, `CD/` files, recipes and `data_mapping.json` for LPS-3 to LPS-27 and LPS-32. Raw XRD `.gfrm` frames are not included. LPS-32 additionally includes its exported Rigaku XRD data in `rigaku/` and the reference XRD patterns in `xrd_ref/` (both used by `notebooks/shap_verification_plots.py`).

## Pipeline

### 1. Generate the sample mapping

```
uv run python scripts/generate_mapping_from_recipes.py BASE_DIR [--recipes RECIPES.json] [--output data_mapping.json] [-n]
```

- Links every recipe to its measurement files and writes `data_mapping.json` into `BASE_DIR`.
- The recipes file must be a JSON list; the *i*-th recipe (1-indexed) is **Sample ID** *i*. If `--recipes` is omitted, the single `*recipe*.json` in `BASE_DIR` is used (or `recipes.json` if several exist).
- Files are matched by Sample ID:
    - H5: `CBox/*_{id}_run*.h5`
    - CD: `CD/*front-{id}.(txt|jws)` and `CD/*back-{id}.(txt|jws)`
    - XRD: well position from the Sample ID (`row = (id-1) // 4`, `col = (id-1) % 4 + 1`, e.g. sample 1 → `A01`), matched as `XRD/*_{well}_*.gfrm`, falling back to `.xy`
- Missing files are written as empty strings; `-n/--no-verify` instead writes predicted filenames.
- Each entry of `data_mapping.json` contains `sample_id`, `recipe`, `files` (`h5`, `cd_front`, `cd_back`, `xrd`), and `derived_metadata` (`well_position`). The file can be edited by hand to correct individual links before the next step.
- The paths stored in `data_mapping.json` are relative to the folder containing the file (e.g. `CBox/..._run1_spec_run.h5`, `CD/front-1.jws`), so the mapping stays valid wherever the dataset is moved or downloaded, including the Zenodo archive. Re-run this step only if you add or rename files.
- Example: `uv run python scripts/generate_mapping_from_recipes.py chem-informed-mtgp-data-zenodo/LPS-27 --recipes total_correlation_recipes.json`

### 2. Preprocess H5 data from the mapping

```
uv run python scripts/preprocessing_from_mapping.py BASE_DIR/data_mapping.json [--poni PONI_FILE] [-n N_JOBS]
```

- For each mapping entry, this step:
    - Locates the H5 file in `CBox/` (exact name, otherwise the highest `run{n}` for that sample, otherwise any file with that Sample ID).
    - Converts CD `.jws` files to `.txt` (skipped if the `.txt` already exists).
    - Attaches the CD (and, if present, XRD) files and stores the recipe as `process_params`.
    - Computes the UV-Vis, PL and CD results and saves a unified H5 file to `BASE_DIR/preprocessed/` (same filename as the raw file).
- `--poni` is only needed for XRD `.gfrm` files (pyFAI integration). Samples are processed in parallel; `-n` sets the number of workers (default: all CPUs).
- Samples whose files cannot be found are skipped with a warning; a summary (`N/M files processed successfully`) is printed at the end.
- On Windows, set `PYTHONUTF8=1` (e.g. `PYTHONUTF8=1 uv run python ...`) so the per-file log, which contains `✓`/`✗`, prints without a `UnicodeEncodeError`.
- Example: `uv run python scripts/preprocessing_from_mapping.py chem-informed-mtgp-data-zenodo/LPS-27/data_mapping.json -n 4`
- *Alternative (no mapping/recipes)*: `uv run python scripts/preprocessing.py [DATA_DIRECTORIES...]` discovers and links the files directly from the filenames and writes to the same `preprocessed/` folder.

### 3. Ingest the preprocessed H5 files into DuckDB

```
uv run python ingestion.py [PREPROCESSED_H5_PATHS...]
```

- Processes the H5 files from the `preprocessed` folder and populates `mydb.duckdb` (metadata) and `data_parquet/` (measurement arrays).
- Example: `uv run python ingestion.py "chem-informed-mtgp-data-zenodo/LPS-27/preprocessed/*.h5"`

### 4. (Optional) Browse the data

```
uv run python app.py
```

Launches a Dash app at http://127.0.0.1:8050/ for querying samples by metadata and viewing UV-Vis, CD and XRD spectra. This is a convenience tool for inspecting the processed data and is not needed to reproduce the manuscript figures.

## Data location for the notebooks

The scripts in `notebooks/` (and `BO_fingerprint_sigma_per_chemical.ipynb`) read data from a single root, defined in `notebooks/configs/paths.py`. It defaults to `chem-informed-mtgp-data-zenodo/` inside the repository. Set the `LPS_DATA_ROOT` environment variable to use a different location (the folder that contains `LPS-3/`, `LPS-4/`, ..., `BO_results_outlier_removed/`, `transfer_learning_cv/`):

```
# PowerShell
$env:LPS_DATA_ROOT = "D:\data\chem-informed-mtgp-data-zenodo"
# bash
export LPS_DATA_ROOT=/data/chem-informed-mtgp-data-zenodo
```

## Cross-validation results

`transfer_learning_cv/` in the archive contains a single `please_extract.zip`. Extract it in place, so the per-seed folders (`transfer_learning_cv_27_stratified_k=5_rseed=<N>/`) sit directly inside `transfer_learning_cv/` and no extra folder level is created:

```
# PowerShell
Expand-Archive chem-informed-mtgp-data-zenodo\transfer_learning_cv\please_extract.zip -DestinationPath chem-informed-mtgp-data-zenodo\transfer_learning_cv
# bash
unzip chem-informed-mtgp-data-zenodo/transfer_learning_cv/please_extract.zip -d chem-informed-mtgp-data-zenodo/transfer_learning_cv
```

The transfer-learning scripts (`transfer-learning-plot.py`, `si_fig_3_covariance_matrix.py`, ...) read these folders from `<LPS_DATA_ROOT>/transfer_learning_cv/`.
