import numpy as np
from itertools import combinations
import pandas as pd
from functools import lru_cache


def batch_by_temp_then_time(
    df: pd.DataFrame,
    batch_size=4,
    temp_col="annealing_temp",
    time_col="annealing_time",
    max_temp_range=50,
):
    """
    Partition rows into batches of size `batch_size`.
    Constraint:
      - Temperatures in a batch must satisfy the `max_temp_range` constraint (hard limit).
      - If max_temp_range is scalar: range <= scalar.
      - If max_temp_range is list: batch must fit entirely within one of the (min, max) tuples.
    Objective:
      - Minimize global sum of time costs.
    """
    n = len(df)
    if n % batch_size != 0:
        raise ValueError(f"Row count {n} not divisible by batch_size={batch_size}.")

    idx = np.arange(n)
    all_temps = df[temp_col].values
    all_times = df[time_col].values

    # Helper to run the solver with specific constraints
    def solve_partition(current_constraints):
        # 1. Precompute candidates
        candidates = []

        # Validation helper
        def is_valid_batch(t_indices):
            batch_temps = all_temps[list(t_indices)]
            if isinstance(current_constraints, list):
                t_min, t_max = batch_temps.min(), batch_temps.max()
                for low, high in current_constraints:
                    if t_min >= low and t_max <= high:
                        return True
                return False
            else:
                return (batch_temps.max() - batch_temps.min()) <= current_constraints

        def calc_cost(t_indices):
            batch_times = all_times[list(t_indices)]
            return max(batch_times.max(), batch_times.min() + 360)

        for combo in combinations(idx, batch_size):
            if is_valid_batch(combo):
                candidates.append(
                    (
                        tuple(sorted(combo)),
                        calc_cost(combo),
                        all_temps[list(combo)].min(),
                    )
                )

        if not candidates:
            return None

        # Index candidates
        batches_by_pivot = {i: [] for i in idx}
        for cand in candidates:
            combo = cand[0]
            for member in combo:
                batches_by_pivot[member].append(cand)

        # 2. DP Solver
        @lru_cache(None)
        def search(mask):
            if not mask:
                return 0, []

            pivot = mask[0]
            best_cost = float("inf")
            best_part = None

            # Options using pivot
            options = [c for c in batches_by_pivot[pivot] if set(c[0]).issubset(mask)]

            # Simple heuristic optimization: try smaller costs first
            options.sort(key=lambda x: x[1])

            for cand in options:
                rem_mask = tuple(sorted(set(mask) - set(cand[0])))
                rem_cost, rem_part = search(rem_mask)

                if rem_cost != float("inf"):
                    total = cand[1] + rem_cost
                    if total < best_cost:
                        best_cost = total
                        best_part = [cand] + rem_part

            return best_cost, best_part

        cost, solution = search(tuple(sorted(idx)))

        if cost == float("inf") or solution is None:
            return None

        return solution

    # Attempt 1: Strict constraints
    solution = solve_partition(max_temp_range)

    if solution is None:
        # Infeasible. Generate diagnostics and suggestions.
        msg = "Infeasible: Unable to partition data into valid batches with strictly enforcing the constraints.\n"

        # Analyzing distribution
        msg += f"Data Temperatures: {sorted(all_temps.tolist())}\n"

        # Suggestion Search
        if isinstance(max_temp_range, list):
            found_relaxed = False
            for pad in [5, 10, 15, 20, 30, 50]:
                relaxed_ranges = [(l - pad, h + pad) for l, h in max_temp_range]
                if solve_partition(relaxed_ranges):
                    msg += f"SUGGESTION: The constraints are too tight. A solution exists if you relax ranges by +/- {pad}°C.\n"
                    msg += f"Try using: max_temp_range={relaxed_ranges}"
                    found_relaxed = True
                    break

            if not found_relaxed:
                msg += "Could not find a feasible solution even with relaxed ranges (up to +/- 50°C)."
        else:
            msg += f"Try increasing the max_temp_range (Current: {max_temp_range})."

        raise ValueError(msg)

    # Final Formatting
    solution.sort(key=lambda x: x[2])  # Order by min temp
    ordered_indices = []
    for combo, _, _ in solution:
        batch_times = all_times[list(combo)]
        local_order = np.argsort(batch_times)[::-1]
        ordered_indices.extend([combo[i] for i in local_order])

    return df.iloc[ordered_indices].reset_index(drop=True)
