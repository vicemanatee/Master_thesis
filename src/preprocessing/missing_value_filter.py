"""Filter features according to their missing-value rate.

The expected matrix layout follows scikit-learn conventions: rows are samples
and columns are features.  ``MissingValueFilter`` should be fitted only on a
training split and then applied unchanged to validation or test data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import load_preprocessing_config


def _validate_dataframe(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one sample and one feature")
    if not data.columns.is_unique:
        raise ValueError("feature names (DataFrame columns) must be unique")


def _validate_fraction(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("max_missing_fraction must be between 0 and 1")


class MissingValueFilter(TransformerMixin, BaseEstimator):
    """Keep features whose training-set missing rate is below a threshold.

    Parameters
    ----------
    max_missing_fraction:
        Maximum allowed fraction of missing observations in a feature. None
        reads missing_value_filter.max_missing_fraction from YAML at fit time.
        Features exactly on the boundary are retained.
    config_path:
        Optional path to a complete configuration. None uses the project's
        configs/preprocessing.yaml. Explicit parameter values override YAML.
    """

    def __init__(
        self,
        max_missing_fraction: float | None = None,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        self.max_missing_fraction = max_missing_fraction
        self.config_path = config_path

    def fit(self, X: pd.DataFrame, y: Any = None) -> MissingValueFilter:
        """Learn retained feature names from the training data."""
        _validate_dataframe(X)
        max_missing_fraction = self.max_missing_fraction
        if max_missing_fraction is None:
            config = load_preprocessing_config(self.config_path)
            max_missing_fraction = config["missing_value_filter"][
                "max_missing_fraction"
            ]
        _validate_fraction(max_missing_fraction)

        self.max_missing_fraction_ = max_missing_fraction
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.missing_fraction_ = X.isna().mean(axis=0)
        self.support_mask_ = (
            self.missing_fraction_ <= self.max_missing_fraction_
        ).to_numpy()
        self.selected_features_ = self.feature_names_in_[self.support_mask_]

        if self.selected_features_.size == 0:
            raise ValueError(
                "No features remain after missing-value filtering; "
                "increase max_missing_fraction"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the training-derived feature selection to another dataset."""
        _validate_dataframe(X)
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("MissingValueFilter must be fitted before transform")

        missing_columns = set(self.selected_features_) - set(X.columns)
        if missing_columns:
            raise ValueError(
                f"Input data is missing fitted features: {sorted(missing_columns)}"
            )
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the retained feature names."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "MissingValueFilter must be fitted before requesting feature names"
            )
        return self.selected_features_.copy()


def filter_missing_values(
    data: pd.DataFrame,
    *,
    max_missing_fraction: float | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Filter a DataFrame using missingness calculated from that DataFrame.

    Omitted parameters are read from YAML; explicit values override them.
    For cross-validation, prefer ``MissingValueFilter`` explicitly so the
    filter is fitted on the training fold and reused on the validation fold.
    """
    return MissingValueFilter(
        max_missing_fraction=max_missing_fraction, config_path=config_path
    ).fit_transform(data)
