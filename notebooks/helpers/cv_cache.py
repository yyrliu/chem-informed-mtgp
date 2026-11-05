"""
Disk-cached loaders for the CV result / hyperparameter dataframes shared by
transfer-learning-plot.py and hps_stability_app.py:

    validation_summary_df, foldwise_df  <- helpers.cross_validation.collect_cv_results
    HPS_DF                              <- helpers.hyperparameters.collect_cv_hps

Both dataframes are built from `cv_results_*.csv` / `cv_hps_*.csv` files that
live on a network drive (BASE_CV_DIR, one directory per random seed), so
computing them is slow. This module caches the result to parquet under
notebooks/.cache/ keyed on the inputs that determine the output (run dirs,
continuous columns, a-sites); pass force_refresh=True to bypass a stale cache.

Run this file directly to (re)populate both caches ahead of time:

    uv run python -m notebooks.helpers.cv_cache [--force]
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

if __package__:
    from .cross_validation import collect_cv_results
    from .hyperparameters import collect_cv_hps
else:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from helpers.cross_validation import collect_cv_results
    from helpers.hyperparameters import collect_cv_hps

from configs.paths import CV_RESULTS_DIR_BASE  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

BASE_CV_DIR = CV_RESULTS_DIR_BASE


def run_dir(seed):
    if seed is None:
        return BASE_CV_DIR
    return BASE_CV_DIR.parent / f"{BASE_CV_DIR.name}_rseed={seed}"


RUN_DIRS = [run_dir(s) for s in (42, 43, 44, 45, 46, 47)]


def _hash_key(*parts):
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _load_from_cache(files, meta_file, key):
    if not meta_file.exists() or any(not f.exists() for f in files):
        return None
    meta = json.loads(meta_file.read_text())
    if meta.get("key") != key:
        return None
    return [pd.read_parquet(f) for f in files]


def _write_cache(dfs, files, meta_file, key):
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    for df, f in zip(dfs, files):
        df.to_parquet(f)
    meta_file.write_text(json.dumps({"key": key}))


def load_cv_results(
    run_dirs=RUN_DIRS, cache_dir=CACHE_DIR, force_refresh=False, n_jobs=-2
):
    """
    Return (validation_summary_df, foldwise_df), computed once and cached to
    parquet under `cache_dir`. Recomputes automatically if `run_dirs` changes;
    pass force_refresh=True to bypass the cache unconditionally.
    """
    run_dirs = [Path(d) for d in run_dirs]
    key = _hash_key([str(d) for d in run_dirs])
    vs_file = cache_dir / "validation_summary_df.parquet"
    fw_file = cache_dir / "foldwise_df.parquet"
    meta_file = cache_dir / "cv_results_meta.json"

    if not force_refresh:
        cached = _load_from_cache([vs_file, fw_file], meta_file, key)
        if cached is not None:
            return tuple(cached)

    def _collect_one_run_dir(d):
        return collect_cv_results(
            d, pd.read_csv(d / "df_cross_validation.csv", index_col=0)
        )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_collect_one_run_dir)(d) for d in run_dirs
    )
    validation_summary_df = pd.concat([r[0] for r in results], ignore_index=True)
    foldwise_df = pd.concat(
        [fw.assign(run=i) for i, (_, fw) in enumerate(results)], ignore_index=True
    )
    foldwise_df["fold"] = (
        foldwise_df.groupby("run").ngroup() * foldwise_df["fold"].max()
        + foldwise_df["fold"]
    )
    foldwise_df = foldwise_df.drop(columns=["run"])

    _write_cache(
        [validation_summary_df, foldwise_df], [vs_file, fw_file], meta_file, key
    )
    return validation_summary_df, foldwise_df


def load_hps(
    continuous_columns,
    a_sites,
    run_dirs=RUN_DIRS,
    cache_dir=CACHE_DIR,
    force_refresh=False,
    n_jobs=-1,
):
    """
    Return HPS_DF (see `collect_cv_hps`), computed once and cached to parquet
    under `cache_dir`. Recomputes automatically if any input changes; pass
    force_refresh=True to bypass the cache unconditionally.
    """
    run_dirs = [Path(d) for d in run_dirs]
    key = _hash_key([str(d) for d in run_dirs], list(continuous_columns), list(a_sites))
    hps_file = cache_dir / "hps_df.parquet"
    meta_file = cache_dir / "hps_meta.json"

    if not force_refresh:
        cached = _load_from_cache([hps_file], meta_file, key)
        if cached is not None:
            return cached[0]

    hps_df = collect_cv_hps(run_dirs, continuous_columns, a_sites, n_jobs=n_jobs)

    _write_cache([hps_df], [hps_file], meta_file, key)
    return hps_df


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs.exp_config import NOTEBOOK_CONFIG  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Ignore any existing cache and recompute."
    )
    args = parser.parse_args()

    param_config = NOTEBOOK_CONFIG["param_config"]
    continuous_columns = list(param_config.keys())[:6]
    a_sites = list(NOTEBOOK_CONFIG["a_site_to_smiles"].keys())

    print("Loading cv_results (validation_summary_df, foldwise_df)...")
    validation_summary_df, foldwise_df = load_cv_results(force_refresh=args.force)
    print(
        f"  {len(validation_summary_df)} summary rows, {len(foldwise_df)} foldwise rows."
    )

    print("Loading cv_hps (HPS_DF)...")
    hps_df = load_hps(continuous_columns, a_sites, force_refresh=args.force)
    print(f"  {hps_df['file'].nunique()} files, {len(hps_df)} rows.")
