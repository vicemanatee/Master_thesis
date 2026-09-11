"""Filter features according to their missing-value rate.

The expected matrix layout follows scikit-learn conventions: rows are samples
and columns are features.  ``MissingValueFilter`` should be fitted only on a
training split and then applied unchanged to validation or test data.
"""

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
        raise ValueError("feature names (DataFrame columns) must be unique")


def _validate_fraction(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("max_missing_fraction must be between 0 and 1")


class MissingValueFilter(TransformerMixin, BaseEstimator):
    """Keep features whose training-set missing rate is below a threshold.

    Parameters
    ----------
    max_missing_fraction:
        Maximum allowed fraction of missing observations in a feature.  The
        default of 0.30 retains features observed in at least 70% of training
        samples.  Features exactly on the boundary are retained.
    """

    def __init__(self, max_missing_fraction: float = 0.30) -> None:
        self.max_missing_fraction = max_missing_fraction

    def fit(self, X: pd.DataFrame, y: Any = None) -> MissingValueFilter:
        """Learn retained feature names from the training data."""
        _validate_dataframe(X)
        _validate_fraction(self.max_missing_fraction)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.missing_fraction_ = X.isna().mean(axis=0)
        self.support_mask_ = (
            self.missing_fraction_ <= self.max_missing_fraction
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
                "Input data is missing fitted features: "
                f"{sorted(missing_columns)}"
            )
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        """Return the retained feature names."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "MissingValueFilter must be fitted before requesting feature names"
            )
        return self.selected_features_.copy()


def filter_missing_values(
    data: pd.DataFrame,
    *,
    max_missing_fraction: float = 0.30,
) -> pd.DataFrame:
    """Filter a DataFrame using missingness calculated from that DataFrame.

    For cross-validation, prefer ``MissingValueFilter`` explicitly so the
    filter is fitted on the training fold and reused on the validation fold.
    """
    return MissingValueFilter(max_missing_fraction=max_missing_fraction).fit_transform(
        data
    )
