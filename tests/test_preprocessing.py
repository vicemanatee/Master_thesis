import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    MADFilter,
    MissingValueFilter,
    calculate_mad,
    filter_by_mad,
    filter_missing_values,
)


def test_filter_missing_values_keeps_boundary() -> None:
    data = pd.DataFrame(
        {
            "complete": [1.0, 2.0, 3.0, 4.0],
            "boundary": [1.0, 2.0, 3.0, np.nan],
            "too_sparse": [1.0, np.nan, 3.0, np.nan],
        }
    )

    filtered = filter_missing_values(data, max_missing_fraction=0.25)

    assert filtered.columns.tolist() == ["complete", "boundary"]


def test_missing_filter_uses_training_selection_on_test_data() -> None:
    train = pd.DataFrame({"keep": [1.0, 2.0], "drop": [np.nan, 1.0]})
    test = pd.DataFrame({"keep": [np.nan], "drop": [5.0]})

    transformer = MissingValueFilter(max_missing_fraction=0.25).fit(train)

    assert transformer.transform(test).columns.tolist() == ["keep"]


def test_calculate_mad_ignores_missing_values() -> None:
    data = pd.DataFrame(
        {
            "constant": [2.0, 2.0, 2.0, np.nan],
            "variable": [1.0, 3.0, 5.0, 7.0],
        }
    )

    scores = calculate_mad(data)

    assert scores["constant"] == 0.0
    assert scores["variable"] == pytest.approx(2.0 * 1.4826)


def test_mad_filter_retains_features_above_quantile() -> None:
    data = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "moderate": [1.0, 2.0, 3.0, 4.0],
            "variable": [1.0, 3.0, 5.0, 7.0],
        }
    )

    filtered = filter_by_mad(data, quantile=0.50)

    assert filtered.columns.tolist() == ["variable"]


def test_mad_filter_uses_training_selection_on_test_data() -> None:
    train = pd.DataFrame(
        {
            "selected": [1.0, 3.0, 5.0, 7.0],
            "not_selected": [1.0, 1.0, 1.0, 1.0],
        }
    )
    test = pd.DataFrame(
        {
            "selected": [10.0, 10.0],
            "not_selected": [0.0, 100.0],
        }
    )

    transformer = MADFilter(quantile=0.50).fit(train)

    assert transformer.transform(test).columns.tolist() == ["selected"]


def test_mad_rejects_non_numeric_features() -> None:
    with pytest.raises(TypeError, match="numeric"):
        calculate_mad(pd.DataFrame({"label": ["a", "b"]}))
