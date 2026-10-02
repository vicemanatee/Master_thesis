"""Convert declared abundance scales to log2 before feature preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def to_log2(data: pd.DataFrame, *, input_scale: str) -> pd.DataFrame:
    """Convert linear abundances, or copy values already on the log2 scale.

    Linear zeros become missing values; negative linear abundances are errors.
    Existing missing values and row/column labels are preserved. The scale must
    be declared by the caller rather than inferred from the observed values.
    """
    if input_scale not in {"linear", "log2"}:
        raise ValueError("input_scale must be linear or log2")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty or not data.columns.is_unique:
        raise ValueError("data must be non-empty with unique feature names")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise TypeError("log2 conversion requires numeric features")
    values = data.to_numpy(dtype=float, na_value=np.nan, copy=True)
    if np.isinf(values).any():
        raise ValueError("Abundances contain infinite values")
    if input_scale == "log2":
        return data.copy()
    if (values < 0).any():
        raise ValueError("Linear abundances must be non-negative")
    values[values == 0] = np.nan
    return pd.DataFrame(np.log2(values), index=data.index, columns=data.columns)


class Log2Transformer(TransformerMixin, BaseEstimator):
    """Start a Pipeline with a common log2 scale using LoadedMatrix.scale."""

    def __init__(self, input_scale: str) -> None:
        self.input_scale = input_scale

    def fit(self, X: pd.DataFrame, y: Any = None) -> Log2Transformer:
        to_log2(X, input_scale=self.input_scale)
        self.input_scale_ = self.input_scale
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "input_scale_"):
            raise RuntimeError("Log2Transformer must be fitted before transform")
        if not isinstance(X, pd.DataFrame) or not X.columns.equals(
            pd.Index(self.feature_names_in_)
        ):
            raise ValueError("Input features must match the fitted feature order")
        return to_log2(X, input_scale=self.input_scale_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("Log2Transformer must be fitted before use")
        return self.feature_names_in_.copy()
