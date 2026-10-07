"""Selectable missing-value imputation for log2 abundance matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


class LowAbundanceImputer(TransformerMixin, BaseEstimator):
    """Fill NaNs with each feature's training quantile minus a log2 offset.

    Inputs must already be log2 abundances. The default offset of 1 represents
    halving the linear abundance corresponding to the chosen quantile. All-
    missing training features must be removed by the upstream missingness filter.
    """

    def __init__(self, quantile: float = 0.01, offset: float = 1.0) -> None:
        self.quantile = quantile
        self.offset = offset

    @staticmethod
    def _values(X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Imputation requires a pandas DataFrame")
        if X.empty or not X.columns.is_unique:
            raise ValueError("Input must be non-empty with unique feature names")
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise TypeError("Imputation requires numeric features")
        values = X.to_numpy(dtype=float, na_value=np.nan, copy=True)
        if np.isinf(values).any():
            raise ValueError("Input contains infinite abundances")
        return values

    def fit(self, X: pd.DataFrame, y: Any = None) -> LowAbundanceImputer:
        if isinstance(self.quantile, bool) or not 0 <= self.quantile <= 1:
            raise ValueError("quantile must be between 0 and 1")
        if (
            isinstance(self.offset, bool)
            or not np.isfinite(self.offset)
            or self.offset <= 0
        ):
            raise ValueError("offset must be a positive finite log2 value")
        values = self._values(X)
        all_missing = np.isnan(values).all(axis=0)
        if all_missing.any():
            raise ValueError(
                "Remove all-missing training features before imputation: "
                f"{X.columns[all_missing].tolist()}"
            )
        self.statistics_ = np.nanquantile(values, self.quantile, axis=0) - self.offset
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "statistics_")
        values = self._values(X)
        if not X.columns.equals(pd.Index(self.feature_names_in_)):
            raise ValueError("Input features must match the fitted feature order")
        result = np.where(np.isnan(values), self.statistics_, values)
        return pd.DataFrame(result, index=X.index, columns=X.columns)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "statistics_")
        return self.feature_names_in_.copy()


def make_nan_preprocessor(
    strategy: str = "median", **kwargs: Any
) -> SimpleImputer | LowAbundanceImputer:
    """Select sklearn imputation or the training-derived low-abundance method."""
    if strategy == "low_abundance":
        return LowAbundanceImputer(**kwargs)
    return SimpleImputer(strategy=strategy, **kwargs).set_output(transform="pandas")
