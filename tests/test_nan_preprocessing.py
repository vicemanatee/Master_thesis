from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone
from sklearn.impute import SimpleImputer

from preprocessing import (
    LowAbundanceImputer,
    build_preprocessor,
    make_nan_preprocessor,
)


def test_low_abundance_fits_per_feature_and_reuses_training_values():
    train = pd.DataFrame({"A": [2.0, 4.0, np.nan], "B": [-3.0, -1.0, 1.0]})
    original = train.copy(deep=True)
    imputer = clone(LowAbundanceImputer(quantile=0.0, offset=1.0)).fit(train)
    np.testing.assert_allclose(imputer.statistics_, [1.0, -4.0])
    filled_train = imputer.transform(train)
    assert filled_train.loc[2, "A"] == 1.0
    pd.testing.assert_frame_equal(train, original)

    test = pd.DataFrame(
        {"A": [100.0, np.nan], "B": [np.nan, -100.0]}, index=["s1", "s2"]
    )
    expected = pd.DataFrame({"A": [100.0, 1.0], "B": [-4.0, -100.0]}, index=test.index)
    pd.testing.assert_frame_equal(imputer.transform(test), expected)
    np.testing.assert_allclose(imputer.statistics_, [1.0, -4.0])


def test_low_quantile_offset_and_nullable_values():
    data = pd.DataFrame({"A": pd.Series([0.0, 10.0, pd.NA], dtype="Float64")})
    result = LowAbundanceImputer(quantile=0.1, offset=0.5).fit_transform(data)
    np.testing.assert_allclose(result.A, [0.0, 10.0, 0.5])


def test_all_missing_feature_fails_without_inventing_abundance():
    with pytest.raises(ValueError, match="all-missing.*A"):
        LowAbundanceImputer().fit(pd.DataFrame({"A": [np.nan, np.nan]}))


@pytest.mark.parametrize(
    "parameters",
    [{"quantile": -0.1}, {"quantile": np.nan}, {"offset": 0}, {"offset": np.inf}],
)
def test_invalid_parameters_raise(parameters):
    with pytest.raises(ValueError):
        LowAbundanceImputer(**parameters).fit(pd.DataFrame({"A": [1.0]}))


def test_factory_preserves_median_and_low_abundance_is_optional():
    data = pd.DataFrame({"A": [2.0, 4.0, np.nan]})
    median = make_nan_preprocessor()
    assert isinstance(median, SimpleImputer)
    assert median.fit_transform(data).loc[2, "A"] == 3.0
    assert isinstance(make_nan_preprocessor("low_abundance"), LowAbundanceImputer)


@pytest.mark.parametrize("input_scale", ["linear", "log2"])
def test_profile_preserves_filter_order_and_imputes_after_log2(input_scale):
    path = Path(__file__).resolve().parents[1] / "configs/preprocessing.yaml"
    with path.open() as stream:
        config = yaml.safe_load(stream)
    config["profiles"]["low_abundance"].update(
        missing={"max_missing_fraction": 0.5},
        mad={"enabled": False},
        z_score={"enabled": False},
    )
    config["profiles"]["low_abundance"]["impute"].update(quantile=0.0)
    train = pd.DataFrame({"A": [2.0, 4.0, np.nan], "empty": [np.nan] * 3})
    test = pd.DataFrame({"A": [np.nan], "empty": [100.0]})
    if input_scale == "linear":
        train = np.exp2(train)
        test = np.exp2(test)
    pipeline = build_preprocessor(
        config, profile="low_abundance", input_scale=input_scale
    )
    assert list(pipeline.named_steps) == ["log2", "missing", "impute"]
    result = pipeline.fit_transform(train)
    assert result.columns.tolist() == ["A"]
    np.testing.assert_allclose(result.A, [2.0, 4.0, 1.0])
    assert pipeline.transform(test).loc[0, "A"] == 1.0
