"""Data locations shared by the notebooks/scripts.

Point LPS_DATA_ROOT at the folder that contains LPS-3, LPS-4, ..., LPS-32,
BO_results_outlier_removed/ and transfer_learning_cv/ (the extracted Zenodo
archive). If unset, a `chem-informed-mtgp-data-zenodo` folder at the repository root is used.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("LPS_DATA_ROOT", REPO_ROOT / "chem-informed-mtgp-data-zenodo"))

BO_RESULTS_DIR = DATA_ROOT / "BO_results_outlier_removed"
GP_TRAIN_RESULT_FILE = (
    BO_RESULTS_DIR / "gp_train_results_batch=3-27_target=score_sigma_per_chemical.npz"
)

# Per-seed runs live next to this as "<name>_rseed={seed}"
CV_RESULTS_DIR_BASE = (
    DATA_ROOT / "transfer_learning_cv" / "transfer_learning_cv_27_stratified_k=5"
)


def batch_dir(batch_num) -> Path:
    return DATA_ROOT / f"LPS-{batch_num}"


def data_mapping_path(batch_num) -> Path:
    return batch_dir(batch_num) / "data_mapping.json"
