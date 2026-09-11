"""Median absolute deviation (MAD) feature filtering.

The expected matrix layout follows scikit-learn conventions: rows are samples
and columns are features.  The selection rule matches the source study's R
implementation: retain features with MAD strictly above a chosen quantile.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _validate_numeric_dataframe(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one sample and one feature")
    if not data.columns.is_unique:
        raise ValueError("feature names (DataFrame columns) must be unique")

    non_numeric = [
        column
        for column in data.columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise TypeError(f"MAD requires numeric features; found: {non_numeric}")


def _validate_quantile(value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError("quantile must satisfy 0 <= quantile < 1")


def calculate_mad(
    data: pd.DataFrame,
    *,
    scale: float = 1.4826,
) -> pd.Series:
    """Calculate each feature's scaled median absolute deviation.

    ``scale=1.4826`` matches the default consistency constant used by R's
    ``stats::mad``.  Missing values are ignored.

    Returns
    -------
    pandas.Series
        MAD scores indexed by feature name.  An entirely missing feature gets
        a NaN score and will not be retained by ``MADFilter``.
    """
    _validate_numeric_dataframe(data)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")

    def column_mad(column: pd.Series) -> float:
        observed = column.dropna().to_numpy(dtype=float)
        if observed.size == 0:
            return float("nan")
        median = np.median(observed)
        return float(scale * np.median(np.abs(observed - median)))

    return data.apply(column_mad, axis=0).rename("MAD")


class MADFilter(TransformerMixin, BaseEstimator):
    """Retain features above a training-set MAD quantile.

    Parameters
    ----------
    quantile:
        MAD quantile used as the strict lower boundary.  For example, 0.75
        retains approximately the top 25% most variable features, while 0.50
        retains approximately the top 50%.  Ties at the boundary are excluded,
        matching the original R implementation's ``MAD > quantile(MAD, q)``.
    scale:
        MAD consistency constant.  The default 1.4826 matches R's
        ``stats::mad``.
    """

    def __init__(self, quantile: float = 0.75, scale: float = 1.4826) -> None:
        self.quantile = quantile
        self.scale = scale

    def fit(self, X: pd.DataFrame, y: Any = None) -> MADFilter:
        """Learn the MAD threshold and retained features from training data."""
        _validate_quantile(self.quantile)
        self.mad_scores_ = calculate_mad(X, scale=self.scale)

        finite_scores = self.mad_scores_.dropna()
        if finite_scores.empty:
            raise ValueError("MAD cannot be calculated for any feature")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.mad_threshold_ = float(finite_scores.quantile(self.quantile))
        self.support_mask_ = (
            self.mad_scores_ > self.mad_threshold_
        ).to_numpy()
        self.selected_features_ = self.feature_names_in_[self.support_mask_]

        if self.selected_features_.size == 0:
            raise ValueError(
                "No features remain after MAD filtering; lower the quantile "
                "or inspect tied/constant features"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the training-derived MAD feature selection."""
        _validate_numeric_dataframe(X)
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("MADFilter must be fitted before transform")

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
                "MADFilter must be fitted before requesting feature names"
            )
        return self.selected_features_.copy()


def filter_by_mad(
    data: pd.DataFrame,
    *,
    quantile: float = 0.75,
    scale: float = 1.4826,
) -> pd.DataFrame:
    """Filter a DataFrame using MAD scores calculated from that DataFrame.

    For cross-validation, prefer ``MADFilter`` explicitly so the filter is
    fitted on the training fold and reused on the validation fold.
    """
    return MADFilter(quantile=quantile, scale=scale).fit_transform(data)
