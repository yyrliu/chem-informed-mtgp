from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


MetricFn = Callable[[pd.DataFrame, "MetricContext"], pd.Series]
InferenceFn = Callable[[pd.DataFrame], pd.DataFrame]
NaNPolicyFn = Callable[[pd.DataFrame], pd.DataFrame]
FeatureBuilderFn = Callable[[pd.DataFrame], Any]


@dataclass(slots=True)
class MetricContext:
    """Runtime context passed to custom metric functions."""

    load_modalities: Callable[[Sequence[str], str], Any]


@dataclass(slots=True)
class InferenceRule:
    name: str
    fn: InferenceFn
    requires: set[str] = field(default_factory=set)
    provides: set[str] = field(default_factory=set)


@dataclass(slots=True)
class MetricSpec:
    name: str
    fn: MetricFn
    required_modalities: tuple[str, ...] = ()


@dataclass(slots=True)
class FeatureBuilderSpec:
    name: str
    fn: FeatureBuilderFn


@dataclass(slots=True)
class MetricRegistry:
    _specs: dict[str, MetricSpec] = field(default_factory=dict)

    def register(self, spec: MetricSpec, *, overwrite: bool = False) -> None:
        if not overwrite and spec.name in self._specs:
            raise KeyError(f"Metric '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> MetricSpec:
        if name not in self._specs:
            raise KeyError(f"Metric '{name}' is not registered.")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs.keys())


@dataclass(slots=True)
class FeatureRegistry:
    _specs: dict[str, FeatureBuilderSpec] = field(default_factory=dict)

    def register(self, spec: FeatureBuilderSpec, *, overwrite: bool = False) -> None:
        if not overwrite and spec.name in self._specs:
            raise KeyError(f"Feature builder '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureBuilderSpec:
        if name not in self._specs:
            raise KeyError(f"Feature builder '{name}' is not registered.")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs.keys())


@dataclass(slots=True)
class InferenceRegistry:
    _rules: list[InferenceRule] = field(default_factory=list)

    def register(self, rule: InferenceRule) -> None:
        self._rules.append(rule)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        for rule in self._rules:
            if rule.requires and not rule.requires.issubset(set(out.columns)):
                continue
            out = rule.fn(out)
        return out

    def names(self) -> list[str]:
        """Return sorted list of registered rule names."""
        return sorted({rule.name for rule in self._rules})


@dataclass(slots=True)
class NaNPolicyChain:
    _policies: list[NaNPolicyFn] = field(default_factory=list)

    def add(self, policy: NaNPolicyFn) -> None:
        self._policies.append(policy)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        for policy in self._policies:
            out = policy(out)
        return out
