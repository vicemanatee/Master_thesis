"""Feature filtering by missing-value rate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _validate_dataframe(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one sample and one feature")
    if not data.columns.is_unique:
        raise ValueError("feature names must be unique")


class MissingValueFilter(TransformerMixin, BaseEstimator):
    """Keep features observed in enough training samples."""

    def __init__(self, max_missing_fraction: float = 0.30) -> None:
        self.max_missing_fraction = max_missing_fraction

    def fit(self, X: pd.DataFrame, y: Any = None) -> MissingValueFilter:
        _validate_dataframe(X)
        if (
            isinstance(self.max_missing_fraction, bool)
            or not 0.0 <= self.max_missing_fraction <= 1.0
        ):
            raise ValueError("max_missing_fraction must be between 0 and 1")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.max_missing_fraction_ = self.max_missing_fraction
        self.missing_fraction_ = X.isna().mean(axis=0)
        self.support_mask_ = (
            self.missing_fraction_ <= self.max_missing_fraction_
        ).to_numpy()
        self.selected_features_ = self.feature_names_in_[self.support_mask_]
        if self.selected_features_.size == 0:
            raise ValueError("No features remain after missing-value filtering")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_dataframe(X)
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("MissingValueFilter must be fitted before transform")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(
                f"Input data is missing fitted features: {sorted(missing)}"
            )
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("MissingValueFilter must be fitted before transform")
        return self.selected_features_.copy()


def filter_missing_values(
    data: pd.DataFrame, *, max_missing_fraction: float = 0.30
) -> pd.DataFrame:
    """Fit and apply missing-value filtering to one matrix."""
    return MissingValueFilter(max_missing_fraction).fit_transform(data)
