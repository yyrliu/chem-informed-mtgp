"""
Numba-accelerated cost functions for Bayesian Optimization.

This module provides JIT-compiled cost functions that are compatible with
gpcam's GPOptimizer while maintaining high performance through Numba.
"""

import numpy as np
from numba import njit
from typing import Dict, OrderedDict as OrderedDictType, Optional, Union, Any
from collections import OrderedDict


# Cost function type codes
COST_CONSTANT = 0
COST_RAMP = 1


def normalize_param_costs(param_costs: dict, param_config: dict) -> dict:
    """
    Normalize cost parameters to the [0, 1] range based on parameter ranges.

    Parameters
    ----------
    param_costs : dict
        Dictionary of cost configurations for each parameter
    param_config : dict
        Dictionary of parameter configurations with 'range' key

    Returns
    -------
    dict
        Normalized cost parameters
    """
    from copy import deepcopy

    normalized = deepcopy(param_costs)
    for k, v in normalized.items():
        param = param_costs[k]
        if v["function"] == "ramp":
            p_range = param_config[k]["range"]
            v["start"] = (param["start"] - p_range[0]) / (p_range[1] - p_range[0])
            v["slope"] = param["slope"] / (
                1 - (param["start"] - p_range[0]) / (p_range[1] - p_range[0])
            )
        elif v["function"] == "constant":
            continue
        else:
            raise ValueError(f"Unknown cost function {v['function']} for parameter {k}")

    return normalized


def encode_param_costs(param_costs_dict: dict) -> np.ndarray:
    """
    Encode parameter costs as a structured numpy array for Numba compatibility.

    The array has shape (n_params, 4) where each row contains:
    [function_type, start, slope, base]

    - function_type: 0 = constant, 1 = ramp
    - start: starting point for ramp function (ignored for constant)
    - slope: slope of ramp function (ignored for constant)
    - base: base cost value (1.0 for constant, configurable for ramp)

    Parameters
    ----------
    param_costs_dict : dict
        Dictionary of cost configurations (should be normalized)

    Returns
    -------
    np.ndarray
        Encoded cost parameters as (n_params, 4) array
    """
    n_params = len(param_costs_dict)
    encoded = np.zeros((n_params, 4), dtype=np.float64)

    for i, cost_config in enumerate(param_costs_dict.values()):
        fn = cost_config.get("function", "constant")

        if fn == "constant":
            encoded[i, 0] = COST_CONSTANT
            encoded[i, 3] = 1.0  # base
        elif fn == "ramp":
            encoded[i, 0] = COST_RAMP
            encoded[i, 1] = cost_config["start"]
            encoded[i, 2] = cost_config["slope"]
            encoded[i, 3] = cost_config.get("base", 1.0)
        else:
            raise ValueError(f"Unknown cost function: {fn}")

    return encoded


def decode_param_costs(
    encoded: np.ndarray, param_names: Optional[list] = None
) -> Dict[str, dict]:
    """
    Decode the encoded parameter costs back to a dict of config objects.

    Parameters
    ----------
    encoded : np.ndarray
        Array of shape (n_params, 4) with rows: [type, start, slope, base]
    param_names : list, optional
        Names of parameters to use as keys. If None, keys will be indices.

    Returns
    -------
    dict
        Mapping of param name -> cost config dict
    """
    n_params = encoded.shape[0]
    out: Dict[str, dict] = {}
    for i in range(n_params):
        name = str(i) if param_names is None else param_names[i]
        t = int(encoded[i, 0])
        if t == COST_CONSTANT:
            out[name] = {"function": "constant"}
        elif t == COST_RAMP:
            out[name] = {
                "function": "ramp",
                "start": float(encoded[i, 1]),
                "slope": float(encoded[i, 2]),
                "base": float(encoded[i, 3]),
            }
        else:
            raise ValueError(f"Unknown encoded function type {t} for index {i}")
    return out


@njit
def _compute_dimension_costs(x: np.ndarray, param_costs: np.ndarray) -> np.ndarray:
    """
    Core dimensional cost accumulation (pre-global scaling and inversion).

    Returns per-point summed raw cost matching original logic prior to
    applying global slope/offset and inversion.
    """
    n_points = x.shape[0]
    n_dims = x.shape[1]
    cost = np.zeros((n_points, n_dims), dtype=np.float64)
    for i in range(n_dims):
        fn_type = param_costs[i, 0]
        if fn_type == COST_CONSTANT:
            cost[:, i] = 1.0
        elif fn_type == COST_RAMP:
            start = param_costs[i, 1]
            slope = param_costs[i, 2]
            base = param_costs[i, 3]
            for j in range(n_points):
                if x[j, i] < start:
                    cost[j, i] = base
                else:
                    cost[j, i] = base + slope * (x[j, i] - start)
    return np.sum(cost, axis=1)


def compute_cost_numba(
    origin: np.ndarray,
    x: np.ndarray,
    param_costs: np.ndarray,
    global_slope: float,
    global_offset: float,
    reduce: bool,
) -> Union[np.ndarray, float]:
    """
    Wrapper implementing original reduction semantics:

    Original sequence:
      cost_dims -> cost_sum_per_point = sum(axis=1)
      if reduce: total = sum(cost_sum_per_point)
      apply global scaling: global_slope * value + global_offset
      invert: 1 / value

    Therefore when reduce=True the inversion is applied AFTER the total sum,
    not on individual points prior to summation.
    """
    cost_sum_per_point = _compute_dimension_costs(x, param_costs)
    if reduce:
        total = np.sum(cost_sum_per_point)
        scaled = global_slope * total + global_offset
        return 1.0 / scaled
    else:
        scaled = global_slope * cost_sum_per_point + global_offset
        return 1.0 / scaled


class CostFunctionFactory:
    """
    Factory for creating Numba-accelerated cost functions compatible with gpcam.

    This factory creates closures that wrap the JIT-compiled cost function
    while maintaining compatibility with GPOptimizer's expected signature.
    """

    def __init__(
        self,
        param_config: Union[dict, OrderedDictType],
        global_params: Optional[dict] = None,
        reduce: bool = True,
        override_param_costs: Optional[Union[str, dict]] = None,
    ):
        """
        Initialize the cost function factory.

        Parameters
        ----------
        param_config : dict or OrderedDict
            Parameter configuration with 'cost' key for each parameter
        global_params : dict, optional
            Global cost parameters with 'slope' and 'offset' keys
        reduce : bool, optional
            If True, sum costs across all points; otherwise return per-point
        override_param_costs : str or dict, optional
            Override cost configurations. Use 'standard' for predefined config
        """
        if global_params is None:
            global_params = {"slope": 1.0, "offset": 0.0}

        # Cache names for serialization
        self.param_names = list(param_config.keys())

        # Build parameter costs configuration
        param_costs_config = OrderedDict(
            [
                (k, v.get("cost", {"function": "constant"}))
                for k, v in param_config.items()
            ]
        )

        # Handle overrides
        if override_param_costs is None:
            override_param_costs = {}
        elif isinstance(override_param_costs, str):
            if override_param_costs == "standard":
                override_param_costs = OrderedDict(
                    [
                        ("annealing_temp", {"function": "constant"}),
                        (
                            "annealing_time",
                            {
                                "function": "ramp",
                                "start": 600.0,
                                "slope": 1.0,
                                "base": 1.0,
                            },
                        ),
                        (
                            "dispense_volume",
                            {
                                "function": "ramp",
                                "start": 75.0,
                                "slope": 1.0,
                                "base": 1.0,
                            },
                        ),
                        ("spin_speed", {"function": "constant"}),
                        ("dispense_speed", {"function": "constant"}),
                        (
                            "concentration",
                            {
                                "function": "ramp",
                                "start": 1.0,
                                "slope": 1.0,
                                "base": 1.0,
                            },
                        ),
                    ]
                )
            else:
                raise ValueError("override_param_costs string must be 'standard'")
        elif isinstance(override_param_costs, dict):
            # Validate override parameters
            for k, v in override_param_costs.items():
                if k not in param_config:
                    raise ValueError(f"Unknown parameter {k} in override_param_costs")
                if "function" not in v:
                    raise ValueError(
                        f"Missing 'function' key in override_param_costs for parameter {k}"
                    )
        else:
            raise ValueError("override_param_costs must be None, 'standard', or a dict")

        # Apply overrides
        param_costs_config.update(override_param_costs)

        # Store unnormalized config for reconstruction (user-friendly)
        self.param_costs_config = param_costs_config
        self.global_params = {
            "slope": float(global_params["slope"]),
            "offset": float(global_params["offset"]),
        }
        self.reduce = bool(reduce)

        # Normalize (internal) and encode (internal)
        self.normalized_param_costs = normalize_param_costs(
            param_costs_config, param_config
        )
        self.encoded_param_costs = encode_param_costs(self.normalized_param_costs)
        self.global_slope = self.global_params["slope"]
        self.global_offset = self.global_params["offset"]

    def __call__(self, origin: np.ndarray, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute cost function values.

        Parameters
        ----------
        origin : np.ndarray
            Origin point(s), shape (1, n_dims) or (n_points, n_dims)
        x : np.ndarray
            Points to evaluate, shape (n_points, n_dims)

        Returns
        -------
        float or np.ndarray
            Cost value(s)
        """
        # Ensure inputs are 2D
        origin = np.atleast_2d(origin)
        x = np.atleast_2d(x)

        # Broadcast origin if needed
        if origin.shape[0] == 1 and x.shape[0] > 1:
            origin = np.repeat(origin, x.shape[0], axis=0)

        return compute_cost_numba(
            origin,
            x,
            self.encoded_param_costs,
            self.global_slope,
            self.global_offset,
            self.reduce,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the cost function for future reconstruction.

        Notes:
        - Keeps normalized costs internal. We dump original param_costs_config
          and global_params for human-readable reconstruction.
        - Encoded costs are also included to allow exact reproducibility if desired.
        """
        return {
            "param_names": self.param_names,
            "param_costs": dict(self.param_costs_config),
            "global_params": dict(self.global_params),
            "reduce": self.reduce,
            # Internal state for reproducibility/debug (optional consumers):
            "encoded_param_costs": self.encoded_param_costs.tolist(),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any], param_config: Union[dict, OrderedDictType]):
        """
        Reconstruct a CostFunctionFactory from a serialized dict.

        Parameters
        ----------
        payload : dict
            Dictionary produced by `to_dict()`
        param_config : dict
            Parameter configuration including ranges. Needed to re-normalize.
        """
        param_costs = payload.get("param_costs", {})
        global_params = payload.get("global_params", {"slope": 1.0, "offset": 0.0})
        reduce = bool(payload.get("reduce", True))
        return CostFunctionFactory(
            param_config=param_config,
            global_params=global_params,
            reduce=reduce,
            override_param_costs=param_costs,
        )


def cost_function_factory(
    param_config: Union[dict, OrderedDictType],
    global_params: Optional[dict] = None,
    reduce: bool = True,
    override_param_costs: Optional[Union[str, dict]] = None,
):
    """
    Create a Numba-accelerated cost function for use with gpcam.

    This is a convenience function that creates and returns a callable
    CostFunctionFactory instance.

    Parameters
    ----------
    param_config : dict or OrderedDict
        Parameter configuration with 'cost' key for each parameter
    global_params : dict, optional
        Global cost parameters with 'slope' and 'offset' keys
    reduce : bool, optional
        If True, sum costs across all points; otherwise return per-point
    override_param_costs : str or dict, optional
        Override cost configurations. Use 'standard' for predefined config

    Returns
    -------
    callable
        Cost function with signature (origin, x) -> cost

    Examples
    --------
    >>> from collections import OrderedDict
    >>> param_config = OrderedDict([
    ...     ("param1", {"range": (0, 10), "cost": {"function": "constant"}}),
    ...     ("param2", {"range": (0, 100), "cost": {"function": "ramp",
    ...                 "start": 50.0, "slope": 0.5, "base": 1.0}})
    ... ])
    >>> cost_fn = cost_function_factory(param_config)
    >>> x = np.array([[0.5, 0.5], [0.7, 0.8]])
    >>> costs = cost_fn(0, x)
    """
    return CostFunctionFactory(
        param_config, global_params, reduce, override_param_costs
    )
