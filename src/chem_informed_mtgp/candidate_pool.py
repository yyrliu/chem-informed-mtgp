# --------------------------------------------------------------
#  iter_candidate_joint_batches  (grid‑+‑random‑+‑snapping)
# --------------------------------------------------------------
from __future__ import annotations

from itertools import product
from typing import Iterator, Sequence, Any, Mapping


import numpy as np
import pandas as pd

from chem_informed_mtgp.fingerprint_kernel import (
    FINGERPRINT_BITS as DEFAULT_FINGERPRINT_BITS,
    FINGERPRINT_MODE,
    FingerprintMetadata,
    assemble_feature_matrix,
)


# ------------------------------------------------------------------
#  Helper functions (vectorised, memory‑efficient)
# ------------------------------------------------------------------
def _snap_to_nearest(series: pd.Series, allowed: np.ndarray) -> pd.Series:
    """Snap each element of *series* to the nearest value in *allowed*."""
    allowed = np.asarray(allowed, dtype=series.dtype)
    # find insertion point that would keep *allowed* sorted
    idx = np.searchsorted(allowed, series.values, side="left")
    left = np.clip(idx - 1, 0, len(allowed) - 1)
    right = np.clip(idx, 0, len(allowed) - 1)

    left_dist = np.abs(series.values - allowed[left])
    right_dist = np.abs(series.values - allowed[right])

    nearest = np.where(left_dist <= right_dist, allowed[left], allowed[right])
    return pd.Series(nearest, index=series.index, name=series.name)


def _snap_to_step(
    series: pd.Series,
    step: float,
    lower: float,
    upper: float,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Snap ``series`` to the nearest multiple of ``step`` anchored at ``lower``.
    The result is clipped to ``[lower, upper]`` (tiny epsilon avoids floating‑point
    overshoot).
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    normalized = (series - lower) / step
    rounded = np.rint(normalized)  # round to nearest integer
    snapped = rounded * step + lower
    snapped = np.clip(snapped, lower - eps, upper + eps)
    return pd.Series(snapped, index=series.index, name=series.name)


# ------------------------------------------------------------------
#  snap_dataframe
# ------------------------------------------------------------------
def snap_dataframe(
    df: pd.DataFrame,
    param_config: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Snap DataFrame columns to their configured grids.

    Columns are snapped based on ``param_config`` structure:
    - If a column has ``"values"``, snap to nearest allowed value.
    - If a column has ``"range"`` and ``"step"``, snap to nearest step multiple.
    - Columns in the DataFrame not present in ``param_config`` are preserved unchanged.

    Parameters
    ----------
    df
        DataFrame with columns to snap. All keys in ``param_config`` must be present.

    param_config
        Mapping describing each continuous variable.  For each key you may
        provide either a ``"range"``/``"step"`` pair *or* a concrete
        ``"values"`` 1‑D iterable of allowed values.  When ``"values"``
        is present it **must be accompanied by** ``"range"`` for validation.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with columns specified in ``param_config`` snapped to
        their configured grids. Other columns are preserved unchanged.

    Raises
    ------
    ValueError
        If any column in ``param_config`` is missing from ``df``, or if
        ``param_config`` has invalid structure (e.g., ``"values"`` without
        ``"range"``).

    Examples
    --------
    >>> from collections import OrderedDict
    >>> config = OrderedDict([
    ...     ("temp", {"range": (40, 150), "step": 5}),
    ...     ("conc", {"range": (0.1, 2.0), "values": [0.25, 0.5, 1.0, 1.75]}),
    ... ])
    >>> df = pd.DataFrame({"temp": [42.3, 147.8], "conc": [0.26, 1.73]})
    >>> snap_dataframe(df, config)
       temp  conc
    0  40.0  0.25
    1 150.0  1.75
    """
    # ---- Validate that all param_config keys are in the DataFrame ----
    missing_cols = set(param_config.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"The following columns from param_config are missing from the input DataFrame: "
            f"{sorted(missing_cols)}"
        )

    # ---- Validate param_config structure ----
    for col, cfg in param_config.items():
        if "values" in cfg:
            if "range" not in cfg:
                raise ValueError(
                    f"Column {col!r} has 'values' but is missing 'range'. "
                    "The 'range' is required when using discrete 'values'."
                )
        else:
            if "range" not in cfg or "step" not in cfg:
                raise ValueError(
                    f"Column {col!r} must contain either a 'values' entry "
                    "or a ('range', 'step') pair."
                )

    # ---- Perform snapping ----
    snapped = df.copy()
    for col, cfg in param_config.items():
        if "values" in cfg:
            snapped[col] = _snap_to_nearest(
                snapped[col], np.asarray(cfg["values"], dtype=float)
            )
        else:
            lo, hi = cfg["range"]
            snapped[col] = _snap_to_step(snapped[col], cfg["step"], lo, hi)

    return snapped


# ------------------------------------------------------------------
def iter_candidate_joint_batches(
    param_config: Mapping[str, Mapping[str, Any]],
    a_site_to_smiles: Mapping[str, str],
    *,
    continuous_columns: Sequence[str] | None = None,
    fingerprint_bits: int | None = None,
    fingerprint_mode: str = FINGERPRINT_MODE,
    fingerprint_metadata: FingerprintMetadata | None = None,
    batch_size: int = 50_000,
    include_hash: bool = True,
    snap: bool = False,  # ← new flag
    n: int | None = None,  # ← random‑sample size
    replace: bool = False,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[Iterator[tuple[pd.DataFrame, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """
    Yield batches of (normalised DataFrame, joint feature matrix).

    Parameters
    ----------
    param_config
        Mapping describing each continuous variable.  For each key you may
        provide either a ``"range"``/``"step"`` pair *or* a concrete
        ``"values"`` 1‑D iterable of allowed values.  When ``"values"``
        is present it **overrides** the range/step definition for grid
        creation, but the entry is still kept for normalisation and,
        if ``snap=True``, for snapping.

    a_site_to_smiles
        Mapping of site identifiers → SMILES strings.

    continuous_columns
        Sub‑set of columns to treat as continuous.  If ``None`` (default) all
        keys of ``param_config`` are used.

    fingerprint_bits
        Number of bits for the fingerprint column (defaults to the library’s
        constant).

    batch_size
        Number of rows per yielded batch.

    include_hash
        When ``True`` a ``config_hash`` column is added.

    snap
        Controls how random candidates (``n`` is not ``None``) are generated.

        ``False`` (default) – values are drawn uniformly from each parameter's
        continuous ``range``; ``step`` and ``values`` are ignored.  The
        returned DataFrame contains raw floating‑point samples.

        ``True`` – flat grid indices are drawn uniformly so that every
        distinct grid point is equally probable.  Values are taken directly
        from ``value_grids``, giving the same distribution as full enumeration
        but limited to ``n`` samples (with replacement).

        Has no effect during full sequential enumeration (``n=None``), which
        always iterates exact grid points.

    n
        ``None`` → enumerate the *entire* grid sequentially.
        Positive integer → draw *n* candidates uniformly at random from the
        continuous parameter ranges (snapped to the grid when ``snap=True``).

    replace
        Unused in the current implementation (continuous sampling inherently
        allows any ``n``).  Kept for API compatibility.

    random_state
        Integer seed, ``np.random.Generator`` or ``None`` (default) for
        reproducibility when ``n`` is supplied.

    Returns
    -------
    (Iterator, (ndarray, ndarray)) – a tuple containing:
        - Iterator[tuple[DataFrame, ndarray]]: generator yielding batches of
          candidates and their corresponding joint feature matrices
        - tuple of two ndarrays:
            * Original bounds: shape (n_params, 2) with [min, max] for each
              parameter in the same order as `effective_columns`
            * Normalized bounds: shape (n_params, 2).  When ``snap=False``
              and ``n`` is given, all rows are ``[0, 1]`` (draws span the full
              declared range).  Otherwise rows reflect the actual grid
              endpoints normalised to the declared range (may be a strict
              subset of ``[0, 1]`` for ``"values"`` columns whose allowed
              values do not reach the range boundaries).
    """
    # ------------------------------------------------------------------
    # 1️⃣  Basic validation
    # ------------------------------------------------------------------
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    effective_columns = list(continuous_columns or param_config.keys())
    if not effective_columns:
        raise ValueError("continuous_columns cannot be empty.")

    if fingerprint_bits is None:
        fingerprint_bits = DEFAULT_FINGERPRINT_BITS

    # Deterministic ordering of the sites (important for reproducible hashes)
    site_items = sorted(a_site_to_smiles.items())
    if not site_items:
        raise ValueError("a_site_to_smiles must contain at least one entry.")

    # ------------------------------------------------------------------
    # 2️⃣  Build the per‑column value grids and bookkeeping for normalisation
    # ------------------------------------------------------------------
    value_grids: list[np.ndarray] = []  # one ndarray per column
    grid_lengths: list[int] = []  # length of each grid
    param_min: dict[str, float] = {}  # for normalisation
    param_span: dict[str, float] = {}  # for normalisation
    param_bounds: list[tuple[float, float]] = []  # bounds for each column

    for col in effective_columns:
        meta = param_config[col]

        # ----- 2a – explicit list of allowed values (e.g. concentration) -----
        if "values" in meta:
            values = np.asarray(meta["values"], dtype=float)
            # sort + deduplicate for deterministic behaviour
            values = np.unique(np.sort(values))
            if values.size == 0:
                raise ValueError(f"'values' for column {col!r} is empty.")

            # When "values" is present, "range" MUST also be provided for normalization
            if "range" not in meta:
                raise KeyError(
                    f"Column {col!r} has 'values' but is missing 'range'. "
                    "The 'range' is required for normalization bounds even when "
                    "using discrete 'values' for the grid."
                )

            value_grids.append(values)
            grid_lengths.append(int(values.size))

            # Always use "range" for normalization bounds (as per docstring)
            start, stop = meta["range"]
            param_min[col] = float(start)
            param_span[col] = float(stop - start)
            param_bounds.append((float(start), float(stop)))
            continue  # skip the range/step handling

        # ----- 2b – classic range/step definition -----
        if "range" not in meta or "step" not in meta:
            raise KeyError(
                f"Column {col!r} must contain either a 'values' entry "
                "or a ('range', 'step') pair."
            )
        start, stop = meta["range"]
        step = meta["step"]
        # Guard against floating‑point truncation when the stop is not exactly
        # reachable with the step size.
        values = np.arange(start, stop + 0.5 * step, step, dtype=float)
        values = values[values <= stop + 1e-9]
        values = np.round(values, 10)

        if values.size == 0:
            raise ValueError(
                f"No values generated for column {col!r} with "
                f"range={meta['range']} step={step}"
            )
        value_grids.append(values)
        grid_lengths.append(int(values.size))

        # Normalisation uses the declared range (not the truncated values)
        param_min[col] = float(start)
        param_span[col] = float(stop - start)
        param_bounds.append((float(start), float(stop)))

    total_candidates = int(np.prod(grid_lengths)) * len(site_items)
    if total_candidates == 0:
        raise ValueError("Could not form any candidate combinations.")
    if verbose:
        print(f"Total candidate combinations (before batching): {total_candidates:,}")

    # Convert bounds to numpy array (shape: n_params x 2)
    bounds_array = np.array(param_bounds, dtype=float)

    # Calculate normalized bounds based on actual grid endpoints
    # (value_grids are already sorted/unique, so just index first/last)
    normalized_param_bounds: list[tuple[float, float]] = []
    for i, col in enumerate(effective_columns):
        grid_values = value_grids[i]
        grid_min = float(grid_values[0])  # First element of sorted grid
        grid_max = float(grid_values[-1])  # Last element of sorted grid

        # Normalize using the declared range
        span = param_span[col]
        if span > 0:
            norm_min = (grid_min - param_min[col]) / span
            norm_max = (grid_max - param_min[col]) / span
        else:
            norm_min = 0.0
            norm_max = 1.0

        normalized_param_bounds.append((norm_min, norm_max))

    # When snap=False and n is given, random sampling draws from the full
    # declared range [lo, hi], not just the discrete grid endpoints.
    # After normalisation that maps to [0, 1], so override accordingly.
    # For snap=True (grid sampling) or n=None (full enumeration), the
    # grid-endpoint bounds computed above are correct.
    if not snap and n is not None:
        normalized_param_bounds = [(0.0, 1.0)] * len(effective_columns)

    normalized_bounds_array = np.array(normalized_param_bounds, dtype=float)

    # Series used later for vectorised normalisation
    min_series = pd.Series(param_min)
    span_series = pd.Series(param_span)

    # ------------------------------------------------------------------
    # 3️⃣  Batch finalisation (includes optional snapping)
    # ------------------------------------------------------------------
    def finalize(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray]:
        batch_df = pd.DataFrame(rows)

        # ---- normalisation to [0, 1] --------------------------------
        # Values in each batch are already on the correct grid (sequential
        # enumeration and snap=True sampling both pull directly from
        # value_grids).  snap=False draws are continuous floats within the
        # declared range.  Either way, no extra snapping step is needed here.
        normalized = batch_df.copy()
        normalized[effective_columns] = (
            (normalized[effective_columns] - min_series)
            .divide(span_series)
            .clip(0.0, 1.0)
        )

        # ---- 3c – assemble the joint feature matrix -----------------
        X_chunk = assemble_feature_matrix(
            df=normalized,
            continuous_columns=effective_columns,
            fingerprint_column="a_site_smiles",
            n_bits=fingerprint_bits,
            fingerprint_mode=fingerprint_mode,
            fingerprint_metadata=fingerprint_metadata,
            a_site_column="a_site",
        )
        return batch_df, X_chunk

    # ------------------------------------------------------------------
    # 4️⃣  Define the generator function
    # ------------------------------------------------------------------
    def generate_batches() -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
        nonlocal n
        if n is None:
            # --------------------------------------------------------------
            # 4a – Full sequential enumeration
            # --------------------------------------------------------------
            buffer: list[dict[str, Any]] = []
            for combo in product(*value_grids):
                base = dict(zip(effective_columns, combo))
                hash_core = "|".join(
                    f"{col}={base[col]:g}" for col in effective_columns
                )

                for a_site, smiles in site_items:
                    candidate = base.copy()
                    candidate["a_site"] = a_site
                    candidate["a_site_smiles"] = smiles
                    if include_hash:
                        candidate["config_hash"] = f"grid|{a_site}|{hash_core}"
                    buffer.append(candidate)

                    if len(buffer) >= batch_size:
                        yield finalize(buffer)
                        buffer = []

            if buffer:
                yield finalize(buffer)
            return

        # ------------------------------------------------------------------
        # 4b – Random‑sampling mode
        # ------------------------------------------------------------------
        n_samples = int(n)
        if n_samples <= 0:
            raise ValueError("n (number of random candidates) must be > 0.")
        if verbose:
            mode = (
                "grid-snapped (uniform over grid points)"
                if snap
                else "continuous range"
            )
            print(f"Randomly sampling {n_samples:,} candidates ({mode})")

        # ---- RNG setup --------------------------------------------------
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        if snap:
            # ---- snap=True: draw flat grid indices → every grid point equally
            # probable.  This is the original behaviour and avoids the edge-
            # undersampling that would occur if we snapped continuous draws.
            shape = tuple(grid_lengths) + (len(site_items),)
            flat_indices = rng.choice(total_candidates, size=n_samples, replace=True)

            buffer: list[dict[str, Any]] = []
            for flat_idx in flat_indices:
                multi = np.unravel_index(int(flat_idx), shape)
                param_idx = multi[:-1]
                site_idx = multi[-1]
                cand: dict[str, Any] = {
                    col: float(value_grids[col_i][param_idx[col_i]])
                    for col_i, col in enumerate(effective_columns)
                }
                a_site, smiles = site_items[site_idx]
                cand["a_site"] = a_site
                cand["a_site_smiles"] = smiles
                if include_hash:
                    hash_core = "|".join(f"{c}={cand[c]:g}" for c in effective_columns)
                    cand["config_hash"] = f"grid|{a_site}|{hash_core}"
                buffer.append(cand)
                if len(buffer) >= batch_size:
                    yield finalize(buffer)
                    buffer = []
            if buffer:
                yield finalize(buffer)

        else:
            # ---- snap=False: draw uniformly from each parameter's continuous
            # range; step and values are ignored.
            param_arrays: dict[str, np.ndarray] = {
                col: rng.uniform(
                    param_min[col], param_min[col] + param_span[col], size=n_samples
                )
                for col in effective_columns
            }
            site_indices = rng.integers(0, len(site_items), size=n_samples)

            buffer = []
            for i in range(n_samples):
                cand = {col: float(param_arrays[col][i]) for col in effective_columns}
                a_site, smiles = site_items[int(site_indices[i])]
                cand["a_site"] = a_site
                cand["a_site_smiles"] = smiles
                if include_hash:
                    hash_core = "|".join(f"{c}={cand[c]:g}" for c in effective_columns)
                    cand["config_hash"] = f"random|{a_site}|{hash_core}"
                buffer.append(cand)
                if len(buffer) >= batch_size:
                    yield finalize(buffer)
                    buffer = []
            if buffer:
                yield finalize(buffer)

    # Return the generator and bounds (original + normalized) as a tuple
    return generate_batches(), (bounds_array, normalized_bounds_array)
