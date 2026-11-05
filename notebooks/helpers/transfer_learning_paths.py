"""Shared output-path naming logic for transfer-learning.py.

transfer-learning.py isn't import-safe as a whole -- it's a flat script that
runs its full pipeline (data loading, GP fitting, cross-validation, saving)
at module scope, kept that way on purpose so its `# %%` cells can still be
run one at a time in an interactive window. The one piece of that script
other tooling needs (run_transfer_learning_sweep.py predicting a run's output
filename so it can skip an already-completed run before spawning a
subprocess) is pulled out here, so there is a single source of truth for the
naming formula instead of two copies that can drift apart.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from chem_informed_mtgp.fingerprint_kernel import (
    FINGERPRINT_MODE_LKJ_CORR,
    LKJ_MASTER_NONE,
)


def get_cv_results_dir(cv_results_dir_base: Path, random_seed: int) -> Path:
    return cv_results_dir_base.with_name(
        f"{cv_results_dir_base.name}_rseed={random_seed}"
    )


def get_base_save_name(
    target_column: str,
    kernel_type: str,
    cv_strategy: str,
    asite_aware_splitting: bool,
) -> str:
    return (
        f"target={target_column}_kernel={kernel_type}_split={cv_strategy}"
        f"{'_asite_aware' if asite_aware_splitting else ''}"
    )


def compute_configs_info(
    *,
    var_per_asite: bool,
    fingerprint_mode: str,
    fp_kernel_conf: str,
    lkj_master_category: Optional[str],
    few_shot_only: bool,
    few_shot_target_asite: str,
    few_shot_ratio: float,
    few_shot_repeat_index: Optional[int],
) -> Tuple[str, str]:
    """Return (stable_configs_info_str, configs_info_str).

    stable_configs_info_str excludes the few-shot-ratio suffix -- it names
    the FEW_SHOT_RATIO=1.0 baseline of this exact config, which
    transfer-learning.py uses to locate a warm-start hps file.
    configs_info_str is the full name, including that suffix when
    few_shot_ratio < 1.0.
    """
    configs_info: List[str] = [
        "var-per-asite" if var_per_asite else "",
        fingerprint_mode
        if fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR
        else fp_kernel_conf,
        (
            "full"
            if lkj_master_category == LKJ_MASTER_NONE
            else f"master={lkj_master_category}"
        )
        if fingerprint_mode == FINGERPRINT_MODE_LKJ_CORR and lkj_master_category
        else "",
        f"{few_shot_target_asite}-only" if few_shot_only else "",
    ]
    configs_info = [info for info in configs_info if info]
    stable_configs_info_str = "_".join(configs_info)

    if few_shot_ratio < 1.0:
        configs_info.append(
            f"fewshot-{few_shot_target_asite}-{int(few_shot_ratio * 100)}pct"
            + (
                f"-rep{few_shot_repeat_index}"
                if few_shot_repeat_index is not None
                else ""
            )
        )
    configs_info_str = "_".join(configs_info)

    return stable_configs_info_str, configs_info_str


def get_output_file_names(save_name: str, configs_info_str: str) -> Tuple[str, str]:
    """Return (hps_file_name, cv_results_file_name), without any timestamp
    prefix (see transfer-learning.py's SAVE_RESULTS_WITH_TIMESTAMP)."""
    return (
        f"cv_hps_{save_name}__{configs_info_str}.csv",
        f"cv_results_{save_name}__{configs_info_str}.csv",
    )
