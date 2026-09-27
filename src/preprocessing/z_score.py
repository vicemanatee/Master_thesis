"""Z-score standardization backed by scikit-learn.

The expected matrix layout follows scikit-learn conventions: rows are samples
and columns are features. Fit the scaler on a training split and reuse it for
validation or test data to prevent data leakage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


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
        raise TypeError(
            f"Z-score scaling requires numeric features; found: {non_numeric}"
        )


def make_z_score_scaler() -> StandardScaler:
    """Create a pandas-preserving Z-score scaler for use in a Pipeline.

    ``StandardScaler`` learns feature means and population standard deviations
    during ``fit``. Missing values are ignored while fitting and preserved by
    ``transform``.
    """
    return StandardScaler().set_output(transform="pandas")


def z_score(data: pd.DataFrame) -> pd.DataFrame:
    """Fit and apply Z-score standardization to one DataFrame.

    For model evaluation and cross-validation, prefer
    :func:`make_z_score_scaler` so validation and test data reuse statistics
    learned only from the corresponding training fold.
    """
    _validate_numeric_dataframe(data)
    return make_z_score_scaler().fit_transform(data)
