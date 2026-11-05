"""
Decode raw `cv_hps_*.csv` hyperparameter vectors (as written by
transfer-learning.py) into named groups — lengthscales, signal variance, and
LKJ correlation hyperparameters — so that hyperparameter *stability* across
CV folds can be compared across model configs (fp_type, var-per-asite,
lkj_master_category, few-shot ratio, ...).

Column layout of a raw `hps` vector (see
src/chem_informed_mtgp/fingerprint_kernels.md, "Hyperparameter Convention"):

    [0]                      signal variance      (lkj_corr: LKJ param r/a)
    [1 : 1+cont_dims]        lengthscales
    [1+cont_dims]            fingerprint weight   (lkj_corr: LKJ param s/b)
    [1+cont_dims+1]          LKJ param c          (only lkj_master_category="none")
    [... : ...+n_categories] per-chemical signal scale (only if var-per-asite)
    [-2]                     noise
    [-1]                     mean

This module builds that layout from the parsed filename config, decodes each
row's raw values into named columns, and (for lkj_corr) reconstructs the
actual 3x3 correlation matrix from the tanh/Cholesky parameterization,
labeling entries by A-site name pairs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from chem_informed_mtgp.fingerprint_kernel import (
    _lkj_category_order,
    _lkj_corr_matrix,
    _lkj_full_corr_matrix,
)

from .cross_validation import add_derived_config_cols, build_file_label

LKJ_MASTER_NONE = "none"


# --------------------------------------------------------------------------- #
# Layout: name every slot of a raw hps vector
# --------------------------------------------------------------------------- #
def hps_layout(
    cont_dims,
    continuous_columns,
    a_sites,
    *,
    fp_type,
    var_per_asite,
    lkj_master_category=None,
):
    """
    Return (names, groups): parallel lists naming every slot of a raw `hps`
    vector for this config, and grouping each into "lengthscale",
    "signal_variance", "lkj", or "other" (fp_weight / noise / mean).
    """
    is_lkj = fp_type == "lkj_corr"
    is_lkj_full = is_lkj and lkj_master_category == LKJ_MASTER_NONE

    names, groups = [], []

    names.append("lkj_r" if is_lkj else "signal_variance")
    groups.append("lkj" if is_lkj else "signal_variance")

    for col in continuous_columns[:cont_dims]:
        names.append(f"lengthscale_{col}")
        groups.append("lengthscale")

    names.append("lkj_s" if is_lkj else "fp_weight")
    groups.append("lkj" if is_lkj else "other")

    if is_lkj_full:
        names.append("lkj_c")
        groups.append("lkj")

    if var_per_asite:
        for site in a_sites:
            names.append(f"sigma_{site}")
            groups.append("signal_variance")

    names.append("noise")
    groups.append("other")
    names.append("mean")
    groups.append("other")

    return names, groups


# --------------------------------------------------------------------------- #
# LKJ: raw (r, s[, c]) -> constructed correlation matrix, keyed by A-site pair
# --------------------------------------------------------------------------- #
def lkj_actual_correlation_matrix(r, s, a_sites, *, master_category=None, c=None):
    """
    Reconstruct the 3x3 correlation matrix in *actual* A-site order (i.e.
    `a_sites` order) from the raw unconstrained hyperparameters.

    `c=None` selects the 2-parameter "master category" mode (see
    `_lkj_corr_matrix`); passing `c` selects the full 3-parameter mode (see
    `_lkj_full_corr_matrix`), which needs no reordering since it has no
    structurally-special category.
    """
    n = len(a_sites)
    if c is not None:
        return _lkj_full_corr_matrix(r, s, c)

    if master_category is None or master_category == "":
        master_idx = 0
    elif isinstance(master_category, str):
        master_idx = list(a_sites).index(master_category)
    else:
        master_idx = int(master_category)

    order = _lkj_category_order(n, master_idx)  # actual_idx -> canonical_pos
    C_canonical = _lkj_corr_matrix(r, s)
    return C_canonical[np.ix_(order, order)]


def lkj_correlation_pairs(C, a_sites):
    """{"corr[siteA-siteB]": value} for every unordered pair of A-sites."""
    pairs = {}
    for i in range(len(a_sites)):
        for j in range(i + 1, len(a_sites)):
            pairs[f"corr[{a_sites[i]}-{a_sites[j]}]"] = C[i, j]
    return pairs


# --------------------------------------------------------------------------- #
# Batched (vectorized) LKJ reconstruction, used by collect_cv_hps so an
# entire file's rows are decoded in one shot instead of a per-row Python loop
# --------------------------------------------------------------------------- #
def _lkj_corr_matrix_batch(r, s):
    """Batched `_lkj_corr_matrix`: r, s shape (n,) -> C shape (n, 3, 3)."""
    r = np.tanh(r)
    s = np.tanh(s)
    c = np.sqrt(1.0 - r**2)
    n = r.shape[0]
    G = np.zeros((n, 3, 3))
    G[:, 0, 0] = 1.0
    G[:, 1, 0] = r
    G[:, 1, 1] = c
    G[:, 2, 0] = r
    G[:, 2, 1] = c * s
    G[:, 2, 2] = c * np.sqrt(1.0 - s**2)
    return G @ G.transpose(0, 2, 1)


def _lkj_full_corr_matrix_batch(a, b, c):
    """Batched `_lkj_full_corr_matrix`: a, b, c shape (n,) -> C shape (n, 3, 3)."""
    a = np.tanh(a)
    b = np.tanh(b)
    c = np.tanh(c)
    n = a.shape[0]
    G = np.zeros((n, 3, 3))
    G[:, 0, 0] = 1.0
    G[:, 1, 0] = a
    G[:, 1, 1] = np.sqrt(1.0 - a**2)
    G[:, 2, 0] = b
    G[:, 2, 1] = np.sqrt(1.0 - b**2) * c
    G[:, 2, 2] = np.sqrt(1.0 - b**2) * np.sqrt(1.0 - c**2)
    return G @ G.transpose(0, 2, 1)


def lkj_actual_correlation_matrix_batch(r, s, a_sites, *, master_category=None, c=None):
    """
    Batched `lkj_actual_correlation_matrix`: r, s (and optionally c) are
    arrays of shape (n,) — one row per CV fold — instead of scalars.
    Returns C of shape (n, 3, 3) in actual A-site order.
    """
    n_sites = len(a_sites)
    if c is not None:
        return _lkj_full_corr_matrix_batch(r, s, c)

    if master_category is None or master_category == "":
        master_idx = 0
    elif isinstance(master_category, str):
        master_idx = list(a_sites).index(master_category)
    else:
        master_idx = int(master_category)

    order = _lkj_category_order(n_sites, master_idx)
    C_canonical = _lkj_corr_matrix_batch(r, s)
    return C_canonical[np.ix_(np.arange(C_canonical.shape[0]), order, order)]


def lkj_correlation_pairs_batch(C, a_sites):
    """{"corr[siteA-siteB]": array of shape (n,)} for every unordered A-site pair."""
    pairs = {}
    for i in range(len(a_sites)):
        for j in range(i + 1, len(a_sites)):
            pairs[f"corr[{a_sites[i]}-{a_sites[j]}]"] = C[:, i, j]
    return pairs


# --------------------------------------------------------------------------- #
# Collection: cv_hps_*.csv files -> tidy long-form dataframe
# --------------------------------------------------------------------------- #
def _collect_one_hps_file(file, continuous_columns, a_sites, cv_dir=None):
    """Decode a single `cv_hps_*.csv` file into its long-form rows."""
    cont_dims = len(continuous_columns)
    raw_df = pd.read_csv(file)
    hps_cols = sorted(
        (c for c in raw_df.columns if c.startswith("hps_")),
        key=lambda c: int(c.split("_", 1)[1]),
    )

    file_label = build_file_label(file)
    cfg = add_derived_config_cols(
        pd.DataFrame({"file": [file_label]}), a_sites=a_sites
    ).iloc[0]
    fp_type = cfg["fp_type"]
    var_per_asite = bool(cfg["var-per-asite"])
    lkj_master_category = cfg["lkj_master_category"]
    if pd.isna(lkj_master_category):
        lkj_master_category = None

    names, groups = hps_layout(
        cont_dims,
        continuous_columns,
        a_sites,
        fp_type=fp_type,
        var_per_asite=var_per_asite,
        lkj_master_category=lkj_master_category,
    )
    if len(names) != len(hps_cols):
        raise ValueError(
            f"{file.name}: expected {len(names)} hyperparameters for "
            f"fp_type={fp_type!r}, var_per_asite={var_per_asite}, "
            f"lkj_master_category={lkj_master_category!r}, got {len(hps_cols)}."
        )

    named_df = pd.DataFrame(raw_df[hps_cols].to_numpy(dtype=np.float64), columns=names)
    named_df["fold"] = raw_df["fold"].to_numpy()
    group_of = dict(zip(names, groups))

    is_lkj = fp_type == "lkj_corr"
    is_lkj_full = is_lkj and lkj_master_category == LKJ_MASTER_NONE
    if is_lkj:
        C = lkj_actual_correlation_matrix_batch(
            named_df["lkj_r"].to_numpy(),
            named_df["lkj_s"].to_numpy(),
            a_sites,
            master_category=lkj_master_category,
            c=named_df["lkj_c"].to_numpy() if is_lkj_full else None,
        )
        for name, values in lkj_correlation_pairs_batch(C, a_sites).items():
            named_df[name] = values
            group_of[name] = "lkj"

    long_df = named_df.melt(id_vars="fold", var_name="name", value_name="value")
    long_df["group"] = long_df["name"].map(group_of)
    long_df["file"] = file_label
    for col, value in cfg.items():
        if col == "file":
            continue
        long_df[col] = value
    if cv_dir is not None:
        long_df["cv_dir"] = str(cv_dir)

    return long_df


def collect_cv_hps(cv_dirs, continuous_columns, a_sites, n_jobs=-1):
    """
    Load every `cv_hps_*.csv` in `cv_dirs`, decode its raw `hps_0..hps_N`
    columns using `hps_layout`, and (for lkj_corr configs) add the
    constructed pairwise correlation values.

    `cv_dirs` may be a single directory or an iterable of directories (e.g.
    one per random seed); files across all of them are processed together.
    Each file is read and decoded independently, so files (across all dirs)
    are processed in parallel via joblib — this is where wall-clock time is
    dominated by CSV I/O (often on a network drive). Pass `n_jobs=1` to
    disable parallelism (e.g. for debugging).

    Returns a tidy long-form DataFrame with one row per (file, fold, hp
    name): columns `file`, `fold`, `group` ("lengthscale", "signal_variance",
    "lkj", "other"), `name`, `value`, plus the config columns from
    `add_derived_config_cols` (fp_type, lkj_master_category, var-per-asite,
    a-site-aware, few_shot_ratio, few_shot_a_site, ...), and `cv_dir` when
    more than one directory was passed.
    """
    if isinstance(cv_dirs, (str, Path)):
        cv_dirs = [cv_dirs]
    multi_dir = len(cv_dirs) > 1

    files = [
        (cv_dir, file)
        for cv_dir in cv_dirs
        for file in Path(cv_dir).glob("*cv_hps*.csv")
    ]
    if not files:
        raise FileNotFoundError(f"No cv_hps_*.csv files found in {list(cv_dirs)}")

    per_file_dfs = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_collect_one_hps_file)(
            file, continuous_columns, a_sites, cv_dir=cv_dir if multi_dir else None
        )
        for cv_dir, file in files
    )
    return pd.concat(per_file_dfs, ignore_index=True)


# --------------------------------------------------------------------------- #
# Stability summary: how much does each hyperparameter move across folds?
# --------------------------------------------------------------------------- #
def summarize_hps_stability(hps_long_df, group_cols=("file",)):
    """
    For every (group_cols..., group, name), summarize the spread of `value`
    across CV folds: mean, std, and coefficient of variation (std / |mean|).

    Correlation entries ("corr[siteA-siteB]") are bounded in [-1, 1], so raw
    `std` is directly comparable across configs; other groups vary hugely in
    scale, so `cv` is the more useful comparison there.
    """
    summary = (
        hps_long_df.groupby([*group_cols, "group", "name"])["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["cv"] = summary["std"] / summary["mean"].abs()
    return summary
