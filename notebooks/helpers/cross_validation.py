import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import norm
from pathlib import Path
import re


def get_cv_df(
    df_normalized,
    validate_batches,
    outliers_by_sample_name,
    sampling_rate=1.0,
    single_asite=None,
):
    valid_mask = (
        df_normalized["sample_name"].str.split("-").str[0].isin(validate_batches)
        & ~df_normalized["sample_name"].isin(outliers_by_sample_name)
        # & (df_normalized["a_site"] == "MBAI")
    )

    if single_asite is not None:
        valid_mask &= df_normalized["a_site"] == single_asite

    print(
        f"Outliers excluded from CV: {df_normalized[~valid_mask]['sample_name'].tolist()}"
    )

    feature_indices = np.where(valid_mask.to_numpy())[0]
    df_cross_validation = df_normalized.loc[valid_mask].copy()
    df_cross_validation["feature_idx"] = pd.Series(
        feature_indices, index=df_cross_validation.index
    )

    sampling_rate = 1.0

    unique_hashes = df_cross_validation["config_hash"].unique()
    if sampling_rate < 1.0:
        size = int(len(unique_hashes) * sampling_rate)
        unique_hashes = np.random.default_rng(42).choice(
            unique_hashes, size=size, replace=False
        )

    print(f"Valid rows for CV: {len(df_cross_validation)}")
    print(
        f"Selected rows: {len(df_cross_validation[df_cross_validation['config_hash'].isin(unique_hashes)])}"
    )
    print(f"Unique config hashes: {len(unique_hashes)}")

    return df_cross_validation, unique_hashes


# --------------------------------------------------------------------------- #
# Hash sampling
# --------------------------------------------------------------------------- #
def get_picked_hashes(
    unique_hashes,
    df_cross_validation,
    a_site_filter="MBAI",
    sample_fraction=0.5,
    seed=42,
):
    """
    Return a random sample of hashes filtered by a_site value.

    Args:
        unique_hashes: Array of unique hashes to select from.
        df_cross_validation: DataFrame with 'config_hash' and 'a_site' columns.
        a_site_filter: The a_site value to filter on. Defaults to "MBAI".
        sample_fraction: Fraction of filtered hashes to sample. Defaults to 0.5.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray of randomly picked hashes.
    """
    hash_to_a_site = df_cross_validation.set_index("config_hash")["a_site"].to_dict()
    hashes_filtered = [
        h for h in unique_hashes if hash_to_a_site.get(h, None) == a_site_filter
    ]

    if sample_fraction == 1.0:
        return np.array(hashes_filtered)

    return np.random.default_rng(seed).choice(
        hashes_filtered,
        size=min(int(len(hashes_filtered) * sample_fraction), len(hashes_filtered) - 1),
        replace=False,
    )


# --------------------------------------------------------------------------- #
# Binning helpers (stratified strategy only)
# --------------------------------------------------------------------------- #
def _choose_n_bins(n, k, max_bins=10, members_per_bin=2):
    """
    Pick a bin count in the usable range for StratifiedKFold.

    StratifiedKFold requires every bin to have >= k members, so the hard
    ceiling is n // k. We aim for ~`members_per_bin * k` members per bin to
    leave margin for `duplicates='drop'` collapsing edges, and keep at least
    2 bins (fewer makes stratification pointless).
    """
    ceiling = n // k
    target = n // (members_per_bin * k)
    return max(2, min(target, ceiling, max_bins))


def _safe_qcut(values, n_bins, retbins=False):
    """
    Quantile-bin `values` into up to `n_bins` bins.

    Uses duplicates='drop', which merges duplicate *bin edges* (caused by ties)
    but never drops data points — every value still receives a label. The
    result may have fewer than `n_bins` distinct bins, and a heavily tied value
    can produce one bin much larger than the others (fine: StratifiedKFold
    spreads it proportionally across folds).

    Binning is on raw values; since qcut bins by rank, this is identical to
    binning on any monotonic transform (e.g. log).

    Returns (labels, effective_n_bins) or (labels, effective_n_bins, edges)
    if retbins=True. Labels are contiguous integers 0..m-1.
    """
    values = np.asarray(values, dtype=float)

    if retbins:
        binned, edges = pd.qcut(
            values, q=n_bins, labels=False, duplicates="drop", retbins=True
        )
    else:
        binned = pd.qcut(
            values, q=n_bins, labels=False, duplicates="drop", retbins=False
        )
        edges = None

    binned = np.asarray(binned)
    n_effective = 0 if np.isnan(binned).any() else len(np.unique(binned))
    binned = binned.astype(int) if n_effective > 0 else binned
    return (binned, n_effective, edges) if retbins else (binned, n_effective)


# --------------------------------------------------------------------------- #
# Per-strategy split functions: (local_idx, values) -> [(train, test), ...]
# Each returns exactly k (train, test) pairs of indices drawn from local_idx.
# --------------------------------------------------------------------------- #
def _kfold_splits(local_idx, values, k, random_state):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    return [(local_idx[tr], local_idx[te]) for tr, te in kf.split(local_idx)]


def _roundrobin_splits(local_idx, values, k, random_state):
    """
    Deterministic sort + modulo assignment.

    WARNING: this creates systematically imbalanced folds (fold f always gets
    rank-f of each block of k). Prefer 'stratified' for skewed targets; kept
    for comparison only.
    """
    order = np.argsort(values, kind="stable")
    folds = np.empty(len(local_idx), dtype=int)
    folds[order] = np.arange(len(local_idx)) % k
    return [(local_idx[folds != f], local_idx[folds == f]) for f in range(k)]


def _stratified_splits(local_idx, values, k, random_state, max_bins=10):
    """
    Stratify on quantile bins of `values`. No data points are ever dropped.

    Raises ValueError if the data cannot be stratified into k folds, i.e. when:
      - qcut produces NaN labels, or
      - fewer than 2 distinct bins result (target effectively constant/tied), or
      - any bin has fewer than k members (StratifiedKFold can't fill k folds).
    """
    n = len(local_idx)
    n_bins = _choose_n_bins(n, k, max_bins=max_bins)
    binned, n_effective = _safe_qcut(values, n_bins)

    if n_effective < 2:
        raise ValueError(
            f"Cannot stratify n={n} points into k={k} folds: quantile binning "
            f"produced {n_effective} distinct bin(s). The target is likely "
            f"constant or heavily tied. Reduce k, use strategy='kfold', or "
            f"pool/merge this group."
        )

    _, counts = np.unique(binned, return_counts=True)
    if counts.min() < k:
        raise ValueError(
            f"Cannot stratify n={n} points into k={k} folds: smallest quantile "
            f"bin has {counts.min()} member(s) (< k={k}). Either reduce k (e.g. "
            f"k={counts.min()}), increase data, or use strategy='kfold'. "
            f"Bin sizes: {sorted(counts.tolist())}."
        )

    largest_frac = counts.max() / n
    if largest_frac > 0.5:
        print(
            f"  WARNING: largest bin holds {largest_frac:.0%} of points "
            f"(n={counts.max()}/{n}); target is heavily tied, so stratification "
            f"is coarse. Bin sizes: {sorted(counts.tolist())}."
        )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    return [(local_idx[tr], local_idx[te]) for tr, te in skf.split(local_idx, binned)]


STRATEGIES = {
    "kfold": _kfold_splits,
    "stratified": _stratified_splits,
    "roundrobin": _roundrobin_splits,
}


# --------------------------------------------------------------------------- #
# A-site-aware fold stitching
# --------------------------------------------------------------------------- #
def asite_aware_splits(a_site_labels, unique_a_sites, k, split_fn_per_group):
    """
    Build folds independently per a-site group, then stitch fold f across groups.

    split_fn_per_group(group_mask) -> list of k (train, test) pairs of global
    indices for that group.

    Note: "fold f" is only a label per group; the underlying splitters do not
    coordinate which samples land in fold f across groups. Each resulting fold
    is still a valid held-out set, which is all CV requires.
    """
    per_group = {}
    for site in unique_a_sites:
        group_mask = np.where(a_site_labels == site)[0]
        if len(group_mask) < k:
            raise ValueError(
                f"A-site '{site}' has only {len(group_mask)} hashes, fewer than "
                f"k={k}. Reduce k or merge rare sites."
            )
        try:
            per_group[site] = split_fn_per_group(group_mask)
        except ValueError as e:
            raise ValueError(f"Split failed for a-site '{site}': {e}") from e

    return [
        (
            np.concatenate([per_group[s][f][0] for s in unique_a_sites]),
            np.concatenate([per_group[s][f][1] for s in unique_a_sites]),
        )
        for f in range(k)
    ]


def print_asite_diagnostics(fold_splits, a_site_labels, unique_a_sites):
    print(
        f"A-sites found: { {str(s): int((a_site_labels == s).sum()) for s in unique_a_sites} }"
    )
    for f, (_, te) in enumerate(fold_splits):
        counts = {str(s): int((a_site_labels[te] == s).sum()) for s in unique_a_sites}
        print(f"  Fold {f + 1:2d} test size={len(te):3d}  per-site: {counts}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def get_cv_split(
    df, strategy, a_site_aware=False, k=10, random_state=42, target_column="target"
):
    """
    Build k cross-validation folds over unique config hashes.

    strategy:
        'kfold'      - shuffled KFold.
        'stratified' - StratifiedKFold on quantile bins of the median target
                       (raises if the data can't be safely stratified).
        'roundrobin' - deterministic sort+modulo (biased; for comparison only).

    a_site_aware: if True, folds are built independently per a-site group and
                  stitched, so every fold keeps each a-site's proportions.

    Returns (fold_splits, unique_hashes), where fold_splits is a list of
    (train_idx, test_idx) into unique_hashes.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES)}."
        )
    split_fn = STRATEGIES[strategy]

    unique_hashes = df["config_hash"].unique()
    hash_to_a_site = df.set_index("config_hash")["a_site"].to_dict()
    a_site_labels = np.array([hash_to_a_site.get(h, "Unknown") for h in unique_hashes])

    # Median target per hash, used by stratified binning (raw values: qcut bins
    # by rank, so a log transform would give identical folds). If the *model*
    # trains on log targets, transform there — not here.
    values = (
        df.groupby("config_hash")[target_column].median().reindex(unique_hashes).values
    )

    if a_site_aware:
        unique_a_sites = np.unique(a_site_labels)
        fold_splits = asite_aware_splits(
            a_site_labels,
            unique_a_sites,
            k,
            lambda mask: split_fn(mask, values[mask], k, random_state),
        )
        print_asite_diagnostics(fold_splits, a_site_labels, unique_a_sites)
    else:
        fold_splits = split_fn(np.arange(len(unique_hashes)), values, k, random_state)

    print(
        f"CV: strategy={strategy}, k={k}, a_site_aware={a_site_aware}, "
        f"n={len(unique_hashes)}"
    )
    if strategy == "stratified" and not a_site_aware:
        _, n_eff, edges = _safe_qcut(
            values, _choose_n_bins(len(values), k), retbins=True
        )
        print(f"  {n_eff} bins; edges in y scale: {np.round(edges, 2)}")

    return fold_splits, unique_hashes


def rmsle(y_true, y_pred):
    # y_pred is an unclipped GP posterior mean and can dip slightly below 0;
    # clip to 0 so log1p stays defined (y_true is always > 0 for this target).
    y_pred_clipped = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def crps_s(x, mu, sigma):
    res = abs(
        sigma
        * (
            (1.0 / np.sqrt(np.pi))
            - 2.0 * norm.pdf((x - mu) / sigma)
            - (((x - mu) / sigma) * (2.0 * norm.cdf((x - mu) / sigma) - 1.0))
        )
    )
    return np.mean(res), np.sqrt(np.var(res))


def picp(x, mu, sigma_with_noise, interval=0.95):
    """
    Computes the Prediction Interval Coverage Probability (PICP)
    for a Gaussian Process posterior.
    """

    z = norm.ppf(1 - (1 - interval) / 2)
    lower_bounds = mu - z * sigma_with_noise
    upper_bounds = mu + z * sigma_with_noise

    inside = (x >= lower_bounds) & (x <= upper_bounds)
    picp = np.mean(inside)

    return picp


FOLD_COL = "fold"  # <-- change if your CSV uses a different column name


# --------------------------------------------------------------------------- #
# Filename -> config parsing (shared by cv_results and cv_hps collection)
# --------------------------------------------------------------------------- #
def build_file_label(file):
    """
    Build the human-readable config label used to tag rows collected from a
    `cv_results_*.csv` / `cv_hps_*.csv` file, e.g.
    "var-per-asite_lkj_corr_master=MBAI (stratified, a-site-aware)".

    Strips the shared "target=...kernel=...split=..." prefix (everything
    before "__") and appends the split strategy / a-site-awareness parsed
    from the full filename.
    """
    split_strategy = re.search(r"split=([^_]+)", file.stem).group(1)
    asite_aware = "asite_aware" in file.stem
    file_spec = file.stem.split("__")[-1]
    return (
        file_spec + f" ({split_strategy}" + (", a-site-aware)" if asite_aware else ")")
    )


def add_derived_config_cols(df, a_sites=None):
    """
    Parse model-config flags out of a `file` column (as produced by
    `build_file_label`) into their own columns: few_shot_ratio,
    few_shot_a_site, fp_type, lkj_master_category, a-site-aware,
    var-per-asite, and one `{a_site}-only` boolean per a-site.

    `a_sites`: explicit list of a-site names to check for "-only" filenames.
    Defaults to the unique values of df["a_site"] (requires that column to
    exist), which is what `collect_cv_results` provides.
    """
    df = df.copy()
    df["few_shot_ratio"] = (
        df["file"]
        .str.extract(r"-(\d{1,2})pct")
        .infer_objects(copy=False)
        .fillna(100)
        .astype(int)
    )
    df["few_shot_a_site"] = df["file"].str.extract(r"fewshot-([A-Za-z]+)")
    df["few_shot_repeat_index"] = df["file"].str.extract(r"-rep(\d+)").astype("Int64")

    fp_type_mapping = {
        "no-correl": "no-correl",
        "fp2": "fp2",
        "total-correl": "total-correl",
        "lkj_corr": "lkj_corr",
    }
    df["fp_type"] = df["file"].apply(
        lambda x: next((v for k, v in fp_type_mapping.items() if k in x), "correl")
    )
    df["lkj_master_category"] = df["file"].str.extract(r"master=([A-Za-z]+)")
    df.loc[df["file"].str.contains("_full"), "lkj_master_category"] = "none"
    df["a-site-aware"] = df["file"].str.contains("a-site-aware")
    df["var-per-asite"] = df["file"].str.contains("var-per-asite")

    if a_sites is None:
        a_sites = df["a_site"].astype(str).unique()
    for a_site in a_sites:
        a_site = str(a_site)
        df[f"{a_site}-only"] = df["file"].str.contains(f"_{a_site}-only")
        if "fp_type" in df:
            df.loc[df[f"{a_site}-only"], "fp_type"] = f"{a_site}-only"

    df = df.sort_values(["file"])
    if "a_site" in df:
        a_site_order = ["ClMBAI", "MBAI", "MeOMBAI", "Others", "Overall"]
        df["a_site"] = pd.Categorical(
            df["a_site"], categories=a_site_order, ordered=True
        )
    return df


def collect_cv_results(cv_dir, cv_df, few_shot_a_site="MBAI"):
    result_files = list(Path(cv_dir).glob("*cv_results*.csv"))

    validation_summary = []
    foldwise_rows = []  # NEW: accumulate per-fold long-form records

    for file in result_files:
        long_path = rf"\\?\{str(file.absolute())}"
        cv_results_df = pd.read_csv(long_path)
        cv_res_with_name = (
            cv_results_df.copy()
            .set_index("test_ids")
            .join(cv_df[["sample_name", "a_site"]], on="test_ids")
        )

        others_cv = cv_res_with_name[cv_res_with_name["a_site"] != few_shot_a_site]

        rmse_by_a_site = (
            cv_res_with_name.groupby("a_site")
            .apply(
                lambda df: np.sqrt(np.mean((df["y_true"] - df["y_pred"]) ** 2)),
                include_groups=False,
            )
            .to_frame(name="rmse")
        )

        rmse_others = np.sqrt(np.mean((others_cv["y_true"] - others_cv["y_pred"]) ** 2))
        rmse_by_a_site.loc["Others"] = rmse_others

        overall_rmse = np.sqrt(
            np.mean((cv_results_df["y_true"] - cv_results_df["y_pred"]) ** 2)
        )
        rmse_by_a_site.loc["Overall"] = overall_rmse

        rmsle_by_a_site = (
            cv_res_with_name.groupby("a_site")
            .apply(
                lambda df: rmsle(df["y_true"], df["y_pred"]),
                include_groups=False,
            )
            .to_frame(name="rmsle")
        )
        rmsle_by_a_site.loc["Others"] = rmsle(others_cv["y_true"], others_cv["y_pred"])
        rmsle_by_a_site.loc["Overall"] = rmsle(
            cv_results_df["y_true"], cv_results_df["y_pred"]
        )

        mae_by_a_site = (
            cv_res_with_name.groupby("a_site")
            .apply(
                lambda df: mae(df["y_true"], df["y_pred"]),
                include_groups=False,
            )
            .to_frame(name="mae")
        )
        mae_by_a_site.loc["Others"] = mae(others_cv["y_true"], others_cv["y_pred"])
        mae_by_a_site.loc["Overall"] = mae(
            cv_results_df["y_true"], cv_results_df["y_pred"]
        )

        crps_by_a_site, picp_by_a_site = None, None
        has_unc = (
            "pred_var" in cv_results_df.columns
            and "y_true" in cv_results_df.columns
            and "y_pred" in cv_results_df.columns
        )
        if has_unc:
            crps_by_a_site = (
                cv_res_with_name.groupby("a_site")
                .apply(
                    lambda df: crps_s(
                        df["y_true"], df["y_pred"], np.sqrt(df["pred_var"])
                    ),
                    include_groups=False,
                )
                .to_frame(name="crps")
            )

            others_crps = crps_s(
                others_cv["y_true"], others_cv["y_pred"], np.sqrt(others_cv["pred_var"])
            )
            crps_by_a_site.loc["Others", "crps"] = tuple(others_crps)

            overall_crps = crps_s(
                cv_results_df["y_true"],
                cv_results_df["y_pred"],
                np.sqrt(cv_results_df["pred_var"]),
            )
            crps_by_a_site.loc["Overall", "crps"] = tuple(overall_crps)
            crps_by_a_site[["crps_mean", "crps_std"]] = pd.DataFrame(
                crps_by_a_site["crps"].tolist(), index=crps_by_a_site.index
            )
            crps_by_a_site = crps_by_a_site.drop(columns=["crps"])

            picp_by_a_site = (
                cv_res_with_name.groupby("a_site")
                .apply(
                    lambda df: picp(
                        df["y_true"], df["y_pred"], np.sqrt(df["pred_var_with_noise"])
                    ),
                    include_groups=False,
                )
                .to_frame(name="picp")
            )

            others_picp = picp(
                others_cv["y_true"],
                others_cv["y_pred"],
                np.sqrt(others_cv["pred_var_with_noise"]),
            )
            picp_by_a_site.loc["Others"] = others_picp
            overall_picp = picp(
                cv_results_df["y_true"],
                cv_results_df["y_pred"],
                np.sqrt(cv_results_df["pred_var_with_noise"]),
            )
            picp_by_a_site.loc["Overall"] = overall_picp

        file_label = build_file_label(file)

        validation_summary.append(
            {
                "file": file_label,
                "a_site": rmse_by_a_site.index,
                "rmse": rmse_by_a_site["rmse"].values,
                "rmsle": rmsle_by_a_site["rmsle"].values,
                "mae": mae_by_a_site["mae"].values,
                "crps_mean": crps_by_a_site["crps_mean"].values
                if crps_by_a_site is not None
                else None,
                "crps_std": crps_by_a_site["crps_std"].values
                if crps_by_a_site is not None
                else None,
                "picp": picp_by_a_site["picp"].values
                if picp_by_a_site is not None
                else None,
            }
        )

        # ---------- NEW: per-fold metrics ----------
        if FOLD_COL not in cv_res_with_name.columns:
            raise KeyError(
                f"Expected a '{FOLD_COL}' column in {file.name} to compute "
                f"fold-wise metrics. Set FOLD_COL to the correct name."
            )

        def _fold_metrics(sub):
            rmse = np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))
            out = {
                "rmse": rmse,
                "rmsle": rmsle(sub["y_true"], sub["y_pred"]),
                "mae": mae(sub["y_true"], sub["y_pred"]),
                "n": len(sub),
            }
            if has_unc:
                # crps_s returns (mean, std) over the points; per fold we want the
                # mean only — the spread across folds is the error bar we care about.
                out["crps_mean"] = crps_s(
                    sub["y_true"], sub["y_pred"], np.sqrt(sub["pred_var"])
                )[0]
                out["picp"] = picp(
                    sub["y_true"],
                    sub["y_pred"],
                    np.sqrt(sub["pred_var_with_noise"]),
                )
            return pd.Series(out)

        # per (a_site, fold)
        for a_site_val, a_grp in cv_res_with_name.groupby("a_site"):
            for fold_val, sub in a_grp.groupby(FOLD_COL):
                m = _fold_metrics(sub)
                foldwise_rows.append(
                    {"file": file_label, "a_site": a_site_val, "fold": fold_val, **m}
                )

        # "Others" per fold
        for fold_val, sub in others_cv.groupby(FOLD_COL):
            m = _fold_metrics(sub)
            foldwise_rows.append(
                {"file": file_label, "a_site": "Others", "fold": fold_val, **m}
            )

        # "Overall" per fold
        for fold_val, sub in cv_res_with_name.groupby(FOLD_COL):
            m = _fold_metrics(sub)
            foldwise_rows.append(
                {"file": file_label, "a_site": "Overall", "fold": fold_val, **m}
            )
        # -------------------------------------------

    validation_summary_df = pd.DataFrame(
        {
            "file": np.repeat(
                [entry["file"] for entry in validation_summary],
                [len(entry["a_site"]) for entry in validation_summary],
            ),
            "a_site": np.concatenate([entry["a_site"] for entry in validation_summary]),
            "rmse": np.concatenate([entry["rmse"] for entry in validation_summary]),
            "rmsle": np.concatenate([entry["rmsle"] for entry in validation_summary]),
            "mae": np.concatenate([entry["mae"] for entry in validation_summary]),
            "crps_mean": np.concatenate(
                [entry["crps_mean"] for entry in validation_summary]
            )
            if "crps_mean" in validation_summary[0]
            else None,
            "crps_std": np.concatenate(
                [entry["crps_std"] for entry in validation_summary]
            )
            if "crps_std" in validation_summary[0]
            else None,
            "picp": np.concatenate([entry["picp"] for entry in validation_summary])
            if "picp" in validation_summary[0]
            else None,
        }
    )

    foldwise_df = pd.DataFrame(foldwise_rows)

    # ---- apply the SAME derived columns to both dataframes ----
    validation_summary_df = add_derived_config_cols(validation_summary_df)
    foldwise_df = add_derived_config_cols(foldwise_df)

    return validation_summary_df, foldwise_df
