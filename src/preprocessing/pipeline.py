"""Build configurable, unfitted preprocessing independently of classifiers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .log2 import Log2Transformer
from .mad_filter import MADFilter
from .missing_value_filter import MissingValueFilter
from .pca import make_pca
from .z_score import make_z_score_scaler

STEP_ORDER = ("log2", "missing", "mad", "impute", "z_score", "pca")


def resolve_preprocessing_profile(
    config: dict[str, Any], profile: str
) -> dict[str, dict[str, Any]]:
    """Merge a profile's step overrides without duplicating common parameters."""
    if profile not in config["profiles"]:
        raise ValueError(f"Unknown preprocessing profile: {profile}")
    settings = deepcopy(config["steps"])
    overrides = config["profiles"][profile]
    if not isinstance(overrides, dict):
        raise TypeError(f"Preprocessing profile must be a mapping: {profile}")
    unknown = (set(settings) | set(overrides)) - set(STEP_ORDER)
    if unknown:
        raise ValueError(f"Unknown preprocessing steps: {sorted(unknown)}")
    if set(settings) != set(STEP_ORDER):
        raise ValueError(
            "Define all preprocessing steps; use enabled: false to disable a step"
        )
    if any(not isinstance(parameters, dict) for parameters in settings.values()):
        raise TypeError("Each preprocessing step must be a parameter mapping")
    for name, override in overrides.items():
        if not isinstance(override, dict):
            raise TypeError(f"Step override must be a mapping: {name}")
        settings[name].update(override)
    for name, parameters in settings.items():
        if not isinstance(parameters.get("enabled"), bool):
            raise TypeError(f"{name}.enabled must be a boolean")
    # Log2 is the declared-scale contract, not a learned dimensionality choice.
    if not settings["log2"]["enabled"]:
        raise ValueError("Log2 harmonization must remain the first preprocessing step")
    return settings


def build_preprocessor(
    config: dict[str, Any],
    *,
    profile: str,
    input_scale: str,
    random_state: int | None = None,
) -> Pipeline:
    """Create fresh library/project transformers; never read data or fit here."""
    settings = resolve_preprocessing_profile(config, profile)
    factories = {
        "log2": lambda **kwargs: Log2Transformer(input_scale=input_scale, **kwargs),
        "missing": MissingValueFilter,
        "mad": MADFilter,
        "impute": lambda **kwargs: SimpleImputer(**kwargs).set_output(
            transform="pandas"
        ),
        "z_score": lambda **kwargs: make_z_score_scaler(**kwargs),
        "pca": lambda **kwargs: make_pca(random_state=random_state, **kwargs),
    }
    steps = []
    for name in STEP_ORDER:
        parameters = settings[name].copy()
        if parameters.pop("enabled"):
            steps.append((name, factories[name](**parameters)))
    return Pipeline(steps)


def inspect_preprocessor(fitted: Pipeline) -> tuple[dict[str, Any], pd.DataFrame]:
    """Describe fitted stages without assuming missing/MAD/PCA are enabled.

    Protein identities refer to the inputs before PCA, not component names.
    Disabled filters pass their input count through; unavailable MAD values and
    non-PCA explained variance are recorded as NaN rather than invented values.
    """
    selected = fitted.feature_names_in_
    missing = fitted.named_steps.get("missing")
    mad = fitted.named_steps.get("mad")
    pca = fitted.named_steps.get("pca")
    if missing is not None:
        selected = missing.selected_features_
    n_after_missing = len(selected)
    if mad is not None:
        selected = mad.selected_features_
    diagnostics = {
        "n_features_after_missing": n_after_missing,
        "n_features_after_mad": len(selected),
        "n_pca_components": pca.n_components_ if pca is not None else 0,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())
        if pca is not None
        else np.nan,
        "mad_threshold": mad.mad_threshold_ if mad is not None else np.nan,
    }
    features = pd.DataFrame(
        {
            "feature_id": selected,
            "train_mad": mad.mad_scores_.reindex(selected).to_numpy()
            if mad is not None
            else np.nan,
        }
    )
    return diagnostics, features
