#!/usr/bin/env python
"""
Orchestrator script to run transfer-learning.py with multiple configuration combinations.

Runs three sets of configurations:
1. FEW_SHOT_RATIO = 1.0 with all combinations of FP_KERNEL_CONF and VAR_PER_ASITE
2. FEW_SHOT_RATIO in [0.01, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8] with few-shot target variations
3. FEW_SHOT_ONLY = True with specific configurations
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

import psutil

from configs.exp_config import NOTEBOOK_CONFIG
from configs.paths import CV_RESULTS_DIR_BASE
from helpers.transfer_learning_paths import (
    get_cv_results_dir,
    get_base_save_name,
    compute_configs_info,
    get_output_file_names,
)

# Number of parallel workers for execution
N_WORKERS = 3

# Number of independent few-shot hash draws per (config, few_shot_ratio<1.0)
# combination. Each repeat keeps RANDOM_SEED (and thus the CV fold split)
# fixed but picks a different few-shot subset via FEW_SHOT_REPEAT_INDEX,
# averaging out the noise that a single small hash draw produces at low
# ratios. Combinations with few_shot_ratio >= 1.0 (or no few_shot_ratio
# param at all) are unaffected and run exactly once, as before.
N_FEW_SHOT_REPEATS = 5

# All configuration sets defined in one place
# Format: {set_name: {"params": {...}, "env_overrides": {...}}}
CONFIG_SETS = {
    "Set 1: FEW_SHOT_RATIO = 1.0": {
        "params": {
            "fp_kernel_conf": ["total-correl", "no-correl", "fp2"],
            "var_per_asite": [True, False],
            "random_seed": list(range(48, 48 + 24)),
        },
        "env_overrides": {"FEW_SHOT_RATIO": "1.0"},
    },
    "Set 1b: FEW_SHOT_RATIO = 1.0, lkj_corr fingerprint mode": {
        "params": {
            "fingerprint_mode": ["lkj_corr"],
            "lkj_master_category": ["none"],
            "var_per_asite": [True],
            "random_seed": list(range(48, 48 + 24)),
        },
        "env_overrides": {"FEW_SHOT_RATIO": "1.0"},
    },
    "Set 2: FEW_SHOT_RATIO sweep with few-shot target variations": {
        "params": {
            "few_shot_ratio": [0.01, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
            "few_shot_target_asite": ["ClMBAI", "MBAI", "MeOMBAI"],
            "fp_kernel_conf": ["total-correl", "no-correl", "fp2"],
            "var_per_asite": [True, False],
            "random_seed": [42, 43, 44, 45, 46, 47],
        },
        "env_overrides": {},
    },
    "Set 2b: FEW_SHOT_RATIO sweep with few-shot target variations": {
        "params": {
            "few_shot_ratio": [0.01, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
            "few_shot_target_asite": ["ClMBAI", "MBAI", "MeOMBAI"],
            "lkj_master_category": ["none"],
            "fingerprint_mode": ["lkj_corr"],
            "var_per_asite": [True],
            "random_seed": [42, 43, 44, 45, 46, 47],
        },
        "env_overrides": {},
    },
    "Set 3: FEW_SHOT_ONLY = True (remove non-target a-sites from training)": {
        "params": {
            "few_shot_ratio": [0.2, 0.35, 0.5, 0.65, 0.8, 1.0],
            "few_shot_target_asite": ["ClMBAI", "MBAI", "MeOMBAI"],
            "fp_kernel_conf": ["fp2"],
            "var_per_asite": [False],
            "random_seed": [42, 43, 44],
        },
        "env_overrides": {"FEW_SHOT_ONLY": "true"},
    },
    "Set 4: FEW_SHOT_ONLY = True, min ratio (len(hps) + 3 samples)": {
        "params": {
            "few_shot_ratio": [0.0559, 0.1],
            "few_shot_target_asite": ["ClMBAI"],
            "fp_kernel_conf": ["fp2"],
            "var_per_asite": [False],
            "random_seed": [42, 43, 44],
        },
        "env_overrides": {"FEW_SHOT_ONLY": "true"},
    },
    "Set 5: FEW_SHOT_ONLY = True, min ratio (len(hps) + 3 samples)": {
        "params": {
            "few_shot_ratio": [0.0482, 0.1],
            "few_shot_target_asite": ["MeOMBAI"],
            "fp_kernel_conf": ["fp2"],
            "var_per_asite": [False],
            "random_seed": [42, 43, 44],
        },
        "env_overrides": {"FEW_SHOT_ONLY": "true"},
    },
    "Set 6: FEW_SHOT_ONLY = True, min ratio (len(hps) + 3 samples)": {
        "params": {
            "few_shot_ratio": [0.134],
            "few_shot_target_asite": ["MBAI"],
            "fp_kernel_conf": ["fp2"],
            "var_per_asite": [False],
            "random_seed": [42, 43, 44],
        },
        "env_overrides": {"FEW_SHOT_ONLY": "true"},
    },
}

# Get the path to transfer-learning.py
SCRIPT_DIR = Path(__file__).parent
TRANSFER_LEARNING_SCRIPT = SCRIPT_DIR / "transfer-learning.py"

if not TRANSFER_LEARNING_SCRIPT.exists():
    print(f"Error: Could not find transfer-learning.py at {TRANSFER_LEARNING_SCRIPT}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Output-path prediction, using transfer-learning.py's own naming logic (via
# helpers/transfer_learning_paths.py) to predict a run's output path *before*
# paying for a subprocess spawn. transfer-learning.py keeps its own
# equivalent check as a redundant safety net for runs launched any other way.
# ---------------------------------------------------------------------------
CV_STRATEGY = "stratified"
ASITE_AWARE_SPLITTING = True
KERNEL_TYPE = NOTEBOOK_CONFIG["kernel_type"]


def get_expected_output_paths(env_overrides: Dict[str, str]) -> Tuple[Path, Path]:
    """Predict the (hps_path, cv_results_path) transfer-learning.py would write
    to for the given environment-variable overrides. Only meaningful when
    SAVE_RESULTS_WITH_TIMESTAMP=False there (the default).
    """
    random_seed = int(env_overrides.get("RANDOM_SEED", 42))
    few_shot_target_asite = env_overrides.get("FEW_SHOT_TARGET_ASITE", "ClMBAI")
    few_shot_only = env_overrides.get("FEW_SHOT_ONLY", "False").lower() == "true"
    few_shot_ratio = float(env_overrides.get("FEW_SHOT_RATIO", "1.0"))
    few_shot_repeat_index_raw = env_overrides.get("FEW_SHOT_REPEAT_INDEX")
    few_shot_repeat_index = (
        int(few_shot_repeat_index_raw)
        if few_shot_repeat_index_raw is not None
        else None
    )
    fp_kernel_conf = env_overrides.get("FP_KERNEL_CONF", "fp2")
    var_per_asite = env_overrides.get("VAR_PER_ASITE", "True").lower() == "true"
    fingerprint_mode = env_overrides.get(
        "FINGERPRINT_MODE", NOTEBOOK_CONFIG["fingerprint_mode"]
    )
    lkj_master_category = env_overrides.get("LKJ_MASTER_CATEGORY")
    target_column = env_overrides.get(
        "TARGET_COLUMN", NOTEBOOK_CONFIG["default_target_column"]
    )

    _, configs_info_str = compute_configs_info(
        var_per_asite=var_per_asite,
        fingerprint_mode=fingerprint_mode,
        fp_kernel_conf=fp_kernel_conf,
        lkj_master_category=lkj_master_category,
        few_shot_only=few_shot_only,
        few_shot_target_asite=few_shot_target_asite,
        few_shot_ratio=few_shot_ratio,
        few_shot_repeat_index=few_shot_repeat_index,
    )
    base_save_name = get_base_save_name(
        target_column, KERNEL_TYPE, CV_STRATEGY, ASITE_AWARE_SPLITTING
    )
    cv_results_dir = get_cv_results_dir(CV_RESULTS_DIR_BASE, random_seed)
    hps_file_name, cv_results_file_name = get_output_file_names(
        base_save_name, configs_info_str
    )

    return cv_results_dir / hps_file_name, cv_results_dir / cv_results_file_name


def find_existing_output(env_overrides: Dict[str, str]) -> Optional[Path]:
    """Return the first output path that already exists for this config, or
    None. Only called when SKIP_IF_EXISTS=true, matching the one case in
    transfer-learning.py where an existing output is treated as a
    skip-worthy no-op rather than a FileExistsError worth surfacing.
    """
    hps_path, cv_results_path = get_expected_output_paths(env_overrides)
    if hps_path.exists():
        return hps_path
    if cv_results_path.exists():
        return cv_results_path
    return None


# ---------------------------------------------------------------------------
# Subprocess lifecycle tracking, so a Ctrl+C can force-kill every subprocess
# tree (uv run -> python transfer-learning.py -> joblib loky workers) that's
# currently running, instead of waiting for them to finish naturally.
# ---------------------------------------------------------------------------
_active_procs_lock = threading.Lock()
_active_procs: set = set()
_stop_requested = threading.Event()


def _register_proc(proc: subprocess.Popen) -> None:
    with _active_procs_lock:
        _active_procs.add(proc)


def _unregister_proc(proc: subprocess.Popen) -> None:
    with _active_procs_lock:
        _active_procs.discard(proc)


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Force-kill proc and all of its descendants.

    `uv run` spawns the real python process as a child of itself, which in
    turn spawns joblib "loky" worker processes; killing only the top-level
    `uv` process would leave those running as orphans.
    """
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)


def _kill_all_active() -> None:
    with _active_procs_lock:
        procs = list(_active_procs)
    for proc in procs:
        _kill_proc_tree(proc)


def _wait_for_process(proc: subprocess.Popen) -> int:
    """Block until proc exits, polling instead of an unbounded wait().

    On Windows, Popen.wait()/subprocess.run() with no timeout blocks inside a
    raw WaitForSingleObject call that never hands control back to the
    interpreter, so a pending Ctrl+C only gets delivered once the child exits
    on its own -- which is why the sweep couldn't be interrupted. Polling with
    a short timeout returns control regularly so a KeyboardInterrupt in the
    main thread is raised promptly, and lets worker threads notice
    _stop_requested and tear down their subprocess instead of waiting on it.
    """
    try:
        while True:
            try:
                return proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                if _stop_requested.is_set():
                    _kill_proc_tree(proc)
                    return proc.wait()
    except KeyboardInterrupt:
        # Only the main thread ever sees this (sequential N_WORKERS=1 path).
        _stop_requested.set()
        _kill_proc_tree(proc)
        raise


def generate_run_configs(
    config_params: Dict[str, List[Any]], env_overrides: Dict[str, str] | None = None
) -> List[Tuple[str, Dict[str, str]]]:
    """
    Generate a list of (config_str, env_overrides_dict) tuples from configuration parameters.

    Args:
        config_params: Dict with parameter names as keys and lists of values as values.
                      The dict can also contain 'param_order' to specify the order of
                      iteration (for cleaner output). Otherwise, keys are used in order.
        env_overrides: Optional dict of environment variable overrides to include in all runs.

    Returns:
        List of (config_str, env_overrides_dict) tuples ready for execution.
        Note: env_overrides_dict contains ONLY the overrides, not a full copy of os.environ.
    """
    base_env_overrides = env_overrides or {}

    # Extract parameter order if provided
    param_order = config_params.pop("param_order", None)
    if param_order is None:
        param_order = [k for k in config_params.keys()]

    # Generate all combinations
    param_names = param_order
    param_values = [config_params[name] for name in param_names]

    runs = []
    for values in product(*param_values):
        config_dict = dict(zip(param_names, values))

        # Build environment overrides (not a full copy yet)
        env_overrides_dict = base_env_overrides.copy()

        # Convert boolean values to string for environment
        for key, val in config_dict.items():
            if isinstance(val, bool):
                env_overrides_dict[key.upper()] = "true" if val else "false"
            else:
                env_overrides_dict[key.upper()] = str(val)

        # Format config string for display
        config_str = ", ".join(f"{k}={v}" for k, v in config_dict.items())

        # Few-shot ratio sweeps are noisy at low ratios because only a small
        # number of hashes get picked for training. Repeat those combinations
        # N_FEW_SHOT_REPEATS times with a different FEW_SHOT_REPEAT_INDEX each
        # (same RANDOM_SEED / CV split, different few-shot hash draw) so the
        # noise averages out. Combinations at ratio >= 1.0 (or without a
        # few_shot_ratio param) run once, unchanged.
        few_shot_ratio = float(env_overrides_dict.get("FEW_SHOT_RATIO", "1.0"))
        if few_shot_ratio < 1.0:
            for repeat_index in range(N_FEW_SHOT_REPEATS):
                repeat_env_overrides_dict = env_overrides_dict.copy()
                repeat_env_overrides_dict["FEW_SHOT_REPEAT_INDEX"] = str(repeat_index)
                # Repeat 0 lines up with a pre-existing (pre-repeat-feature)
                # result that's been renamed to carry a "-rep0" suffix — let
                # transfer-learning.py detect that and skip instead of
                # erroring, so old single-draw data is reused as repeat 0
                # and only repeats 1..N-1 actually run.
                repeat_env_overrides_dict["SKIP_IF_EXISTS"] = "true"
                repeat_config_str = (
                    f"{config_str}, few_shot_repeat_index={repeat_index}"
                )
                runs.append((repeat_config_str, repeat_env_overrides_dict))
        else:
            runs.append((config_str, env_overrides_dict))

    return runs


def run_single_config(
    script_path: Path,
    config_str: str,
    env_overrides: Dict[str, str],
    run_number: int,
    total_runs: int,
) -> Tuple[int, str, bool]:
    """
    Run a single configuration.

    Args:
        script_path: Path to the transfer-learning.py script
        config_str: Configuration string for display
        env_overrides: Environment variable overrides (merged with current environment in worker thread)
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)

    Returns:
        Tuple of (run_number, config_str, success)
    """
    print(f"[{run_number}/{total_runs}] Running: {config_str}")

    if _stop_requested.is_set():
        print("  Skipped: sweep is stopping")
        return (run_number, config_str, False)

    # SKIP_IF_EXISTS mirrors the check transfer-learning.py itself does right
    # before writing output, but performed here so an already-completed
    # repeat is skipped without paying for a subprocess spawn (uv resolve +
    # python startup + heavy imports). transfer-learning.py keeps its own
    # check too, as a redundant safety net for runs launched any other way.
    if env_overrides.get("SKIP_IF_EXISTS", "false").lower() == "true":
        existing_path = find_existing_output(env_overrides)
        if existing_path is not None:
            print(
                f"  Output already exists ({existing_path.name}); "
                "SKIP_IF_EXISTS=true, skipping without launching subprocess."
            )
            return (run_number, config_str, True)

    # Create environment in worker thread to avoid threading issues
    env_dict = os.environ.copy()
    env_dict.update(env_overrides)

    # Debug: Print environment variables to verify they're correct
    env_vars = {
        k: v for k, v in env_overrides.items() if k.isupper() and not k.startswith("_")
    }
    print(f"  Environment vars: {env_vars}")

    try:
        proc = subprocess.Popen(
            ["uv", "run", str(script_path)],
            env=env_dict,
            cwd=script_path.parent,
        )
    except Exception as e:
        print(f"  ✗ Error running script: {e}")
        return (run_number, config_str, False)

    _register_proc(proc)
    try:
        returncode = _wait_for_process(proc)
    finally:
        _unregister_proc(proc)

    if _stop_requested.is_set():
        print("  ✗ Run cancelled")
        return (run_number, config_str, False)
    if returncode != 0:
        print(f"  ⚠ Run failed with exit code {returncode}")
        return (run_number, config_str, False)
    else:
        print("  ✓ Run completed successfully")
        return (run_number, config_str, True)


def run_config_set(
    set_name: str,
    config_params: Dict[str, List[Any]],
    env_overrides: Dict[str, str] | None = None,
    start_run: int = 1,
    total_runs: int | None = None,
) -> Tuple[int, List[bool]]:
    """
    Run a set of configurations either sequentially or in parallel.

    Args:
        set_name: Name of the configuration set (for display)
        config_params: Configuration parameters (passed to generate_run_configs)
        env_overrides: Environment variable overrides for all runs in this set
        start_run: Starting run number (for display)
        total_runs: Total number of runs across all sets (for display)

    Returns:
        Tuple of (next_run_number, results) where results is a list of booleans
        indicating success/failure for each run.
    """
    print(f"\n### {set_name} ###\n")

    # Generate all configurations for this set
    runs = generate_run_configs(config_params, env_overrides)

    if total_runs is None:
        total_runs = start_run + len(runs) - 1

    results = []
    run_number = start_run

    if N_WORKERS == 1:
        # Sequential execution
        for config_str, env_overrides in runs:
            _, _, success = run_single_config(
                TRANSFER_LEARNING_SCRIPT,
                config_str,
                env_overrides,
                run_number,
                total_runs,
            )
            results.append(success)
            print("-" * 80)
            run_number += 1
    else:
        # Parallel execution.
        #
        # Deliberately not a `with ThreadPoolExecutor(...) as executor:` block:
        # on KeyboardInterrupt, a `with` block's __exit__ (executor.shutdown)
        # runs *before* any surrounding except clause, which would block the
        # interpreter waiting for worker threads still stuck on a running
        # subprocess -- exactly the hang this is meant to fix. Killing active
        # subprocesses first, then shutting the executor down, avoids that.
        executor = ThreadPoolExecutor(max_workers=N_WORKERS)
        futures = {}
        for i, (config_str, env_overrides) in enumerate(runs):
            future = executor.submit(
                run_single_config,
                TRANSFER_LEARNING_SCRIPT,
                config_str,
                env_overrides,
                start_run + i,
                total_runs,
            )
            futures[future] = i

        # Collect results as they complete
        results = [None] * len(runs)
        try:
            for future in as_completed(futures):
                run_num, config_str, success = future.result()
                idx = futures[future]
                results[idx] = success
                print("-" * 80)
        except KeyboardInterrupt:
            print("\nInterrupted -- terminating running subprocesses...")
            _stop_requested.set()
            _kill_all_active()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    return (run_number, results)


def calculate_total_runs() -> int:
    """Calculate total number of runs across all sets.

    Delegates to generate_run_configs (on a shallow copy of each set's params,
    so CONFIG_SETS itself is left untouched) rather than a manual product, so
    the count stays accurate even as few_shot_ratio<1.0 combinations expand
    into N_FEW_SHOT_REPEATS runs each.
    """
    total = 0
    for set_config in CONFIG_SETS.values():
        runs = generate_run_configs(
            dict(set_config["params"]), set_config.get("env_overrides")
        )
        total += len(runs)
    return total


if __name__ == "__main__":
    total_runs = calculate_total_runs()
    print(f"Total runs to execute: {total_runs}")
    print(f"Workers: {N_WORKERS}")
    print("=" * 80)

    run_count = 1
    all_results = []
    interrupted = False

    # Run all configuration sets
    try:
        for set_name, set_config in CONFIG_SETS.items():
            run_count, results = run_config_set(
                set_name,
                set_config["params"],
                env_overrides=set_config["env_overrides"],
                start_run=run_count,
                total_runs=total_runs,
            )
            all_results.extend(results)
    except KeyboardInterrupt:
        interrupted = True
        _stop_requested.set()
        _kill_all_active()
        print("\nSweep interrupted by user.")

    # Summary
    print("\n" + "=" * 80)
    successful_runs = sum(all_results)
    failed_runs = len(all_results) - successful_runs
    print(f"Summary: {successful_runs}/{len(all_results)} runs completed successfully")
    if failed_runs > 0:
        print(f"⚠ {failed_runs} runs failed")
    print("=" * 80)

    if interrupted:
        sys.exit(130)
