"""Median absolute deviation (MAD) feature filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.base import BaseEstimator, TransformerMixin


def _validate_numeric_dataframe(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one sample and one feature")
    if not data.columns.is_unique:
        raise ValueError("feature names must be unique")
    non_numeric = [
        column
        for column in data.columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise TypeError(f"MAD requires numeric features; found: {non_numeric}")


def calculate_mad(data: pd.DataFrame, *, scale: float = 1.4826) -> pd.Series:
    """Calculate the scaled MAD of each feature while ignoring missing values."""
    _validate_numeric_dataframe(data)
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")

    values = data.to_numpy(dtype=float, na_value=np.nan)
    all_missing = np.isnan(values).all(axis=0)
    scores = np.full(values.shape[1], np.nan, dtype=float)
    if (~all_missing).any():
        scores[~all_missing] = (
            median_abs_deviation(values[:, ~all_missing], axis=0, nan_policy="omit")
            * scale
        )
    return pd.Series(scores, index=data.columns, name="MAD")


class MADFilter(TransformerMixin, BaseEstimator):
    """Keep features above a training-set MAD quantile."""

    def __init__(self, quantile: float = 0.75, scale: float = 1.4826) -> None:
        self.quantile = quantile
        self.scale = scale

    def fit(self, X: pd.DataFrame, y: Any = None) -> MADFilter:
        if isinstance(self.quantile, bool) or not 0.0 <= self.quantile < 1.0:
            raise ValueError("quantile must satisfy 0 <= quantile < 1")
        self.mad_scores_ = calculate_mad(X, scale=self.scale)
        finite_scores = self.mad_scores_.dropna()
        if finite_scores.empty:
            raise ValueError("MAD cannot be calculated for any feature")

        self.quantile_ = self.quantile
        self.scale_ = self.scale
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.mad_threshold_ = float(finite_scores.quantile(self.quantile_))
        self.support_mask_ = (self.mad_scores_ > self.mad_threshold_).to_numpy()
        self.selected_features_ = self.feature_names_in_[self.support_mask_]
        if self.selected_features_.size == 0:
            raise ValueError("No features remain after MAD filtering")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_numeric_dataframe(X)
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("MADFilter must be fitted before transform")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(
                f"Input data is missing fitted features: {sorted(missing)}"
            )
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("MADFilter must be fitted before transform")
        return self.selected_features_.copy()


def filter_by_mad(
    data: pd.DataFrame, *, quantile: float = 0.75, scale: float = 1.4826
) -> pd.DataFrame:
    """Fit and apply MAD filtering to one matrix."""
    return MADFilter(quantile=quantile, scale=scale).fit_transform(data)
