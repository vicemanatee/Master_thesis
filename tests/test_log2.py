import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from preprocessing import Log2Transformer, MissingValueFilter, to_log2


def test_linear_log2_preserves_labels_missing_values_and_input():
    data = pd.DataFrame(
        {"P1": [1.0, 2.0, 4.0], "P2": [0.0, np.nan, 0.5]},
        index=pd.Index(["run1", "run2", "run3"], name="run_id"),
    )
    original = data.copy(deep=True)
    result = to_log2(data, input_scale="linear")
    expected = pd.DataFrame(
        {"P1": [0.0, 1.0, 2.0], "P2": [np.nan, np.nan, -1.0]},
        index=data.index,
    )
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(data, original)
    # Missingness filtering must see zero-derived missing values.
    assert MissingValueFilter(0.3).fit_transform(result).columns.tolist() == ["P1"]


def test_existing_log2_is_unchanged_including_nonpositive_values():
    data = pd.DataFrame({"P1": [-2.0, 0.0, 3.0, np.nan]})
    result = to_log2(data, input_scale="log2")
    pd.testing.assert_frame_equal(result, data)
    assert result is not data


@pytest.mark.parametrize(
    ("value", "scale"), [(-1.0, "linear"), (np.inf, "linear"), (-np.inf, "log2")]
)
def test_invalid_abundances_raise(value, scale):
    with pytest.raises(ValueError):
        to_log2(pd.DataFrame({"P1": [value]}), input_scale=scale)


def test_unknown_scale_is_not_guessed():
    with pytest.raises(ValueError, match="input_scale"):
        to_log2(pd.DataFrame({"P1": [2.0]}), input_scale="unknown")


def test_transform_reuses_declared_scale_and_supports_clone():
    transformer = clone(Log2Transformer(input_scale="linear"))
    transformer.fit(pd.DataFrame({"P1": [2.0, 4.0]}))
    transformer.input_scale = "log2"
    result = transformer.transform(pd.DataFrame({"P1": [8.0]}))
    assert result.loc[0, "P1"] == 3.0
    assert transformer.get_feature_names_out().tolist() == ["P1"]
