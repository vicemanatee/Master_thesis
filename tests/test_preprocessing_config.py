from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import (
    MADFilter,
    MissingValueFilter,
    calculate_mad,
    compare_preprocessing,
    filter_by_mad,
    filter_missing_values,
    load_preprocessing_config,
    summarize_data,
)
from preprocessing import config as config_module


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    config = load_preprocessing_config()
    config["missing_value_filter"]["max_missing_fraction"] = 0.50
    config["mad_filter"].update(quantile=0.25, scale=2.0)
    config["diagnostics"].update(
        name="omics", before_name="input", after_name="filtered"
    )
    path = tmp_path / "preprocessing.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "low": [0.0, 1.0, 2.0, 3.0],
            "medium": [0.0, 2.0, 4.0, 6.0],
            "high": [0.0, 3.0, 6.0, 9.0],
        },
        index=["P1", "P2", "P3", "P4"],
    )


def test_default_config_is_independent_of_working_directory(tmp_path, monkeypatch):
    expected = load_preprocessing_config()
    monkeypatch.chdir(tmp_path)

    assert load_preprocessing_config() == expected


def test_missing_filter_and_function_use_yaml(config_path):
    data = pd.DataFrame({"complete": [1.0] * 4, "half": [1.0, 2.0, np.nan, np.nan]})
    transformer = MissingValueFilter(config_path=config_path).fit(data)

    assert transformer.max_missing_fraction_ == 0.50
    pd.testing.assert_frame_equal(transformer.transform(data), data)
    pd.testing.assert_frame_equal(
        filter_missing_values(data, config_path=config_path), data
    )
    assert filter_missing_values(
        data, max_missing_fraction=0.0, config_path=config_path
    ).columns.tolist() == ["complete"]


def test_mad_functions_and_transformer_use_yaml(config_path, data):
    scores = calculate_mad(data, config_path=config_path)
    np.testing.assert_allclose(scores, [0.0, 2.0, 4.0, 6.0])

    transformer = MADFilter(config_path=config_path).fit(data)
    assert transformer.quantile_ == 0.25
    assert transformer.scale_ == 2.0
    assert transformer.mad_threshold_ == 1.5
    expected = data[["low", "medium", "high"]]
    pd.testing.assert_frame_equal(transformer.transform(data), expected)
    pd.testing.assert_frame_equal(
        filter_by_mad(data, config_path=config_path), expected
    )


def test_explicit_parameters_override_yaml_individually(config_path, data):
    transformer = MADFilter(quantile=0.75, config_path=config_path).fit(data)
    assert transformer.quantile_ == 0.75
    assert transformer.scale_ == 2.0
    assert transformer.transform(data).columns.tolist() == ["high"]

    transformer = MADFilter(scale=1.0, config_path=config_path).fit(data)
    assert transformer.quantile_ == 0.25
    assert transformer.scale_ == 1.0
    np.testing.assert_allclose(transformer.mad_scores_, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(calculate_mad(data, scale=1.0), [0.0, 1.0, 2.0, 3.0])
    assert filter_by_mad(
        data, quantile=0.75, scale=1.0, config_path=config_path
    ).columns.tolist() == ["high"]


def test_config_is_read_at_fit_and_transform_keeps_learned_selection(
    config_path, data, monkeypatch
):
    # Simulate editing the default file in an already-running Python session.
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    mad = MADFilter()
    missing = MissingValueFilter()
    missing_data = data.copy()
    missing_data.loc[["P1", "P2"], "medium"] = np.nan

    config = load_preprocessing_config()
    config["mad_filter"].update(quantile=0.75, scale=1.0)
    config["missing_value_filter"]["max_missing_fraction"] = 0.0
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    mad.fit(data)
    missing.fit(missing_data)
    assert mad.quantile_ == 0.75
    assert mad.scale_ == 1.0
    assert missing.max_missing_fraction_ == 0.0

    # Transform must not reread YAML or refit statistics on validation data.
    config_path.write_text("", encoding="utf-8")
    assert mad.transform(data).columns.tolist() == ["high"]
    assert missing.transform(data).columns.tolist() == ["constant", "low", "high"]


def test_diagnostics_names_come_from_yaml_and_can_be_overridden(config_path, data):
    assert summarize_data(data, config_path=config_path)["name"] == "omics"
    assert (
        summarize_data(data, name="train", config_path=config_path)["name"] == "train"
    )
    comparison = compare_preprocessing(data, data[["high"]], config_path=config_path)
    assert comparison["name"].tolist() == ["input", "filtered"]
    assert comparison["n_features"].tolist() == [4, 1]
    comparison = compare_preprocessing(
        data, data, before_name="train", config_path=config_path
    )
    assert comparison["name"].tolist() == ["train", "filtered"]


def test_configured_estimators_support_clone_and_grid_search(config_path, data):
    pipeline = Pipeline(
        [
            ("missing", MissingValueFilter(config_path=config_path)),
            ("mad", MADFilter(config_path=config_path)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )
    cloned = clone(pipeline)
    assert cloned["mad"].config_path == config_path
    assert cloned["mad"].quantile is None
    search = GridSearchCV(
        pipeline,
        param_grid={"mad__quantile": [0.25, 0.75]},
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=12),
        error_score="raise",
    ).fit(data, [0, 0, 1, 1])

    assert (
        search.best_estimator_["mad"].quantile_ == search.best_params_["mad__quantile"]
    )
    assert search.best_estimator_["mad"].scale_ == 2.0
    assert search.best_estimator_["missing"].max_missing_fraction_ == 0.5


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("missing_value_filter", "max_missing_fraction", -0.1),
        ("missing_value_filter", "max_missing_fraction", 1.1),
        ("missing_value_filter", "max_missing_fraction", True),
        ("mad_filter", "quantile", 1.0),
        ("mad_filter", "quantile", "0.75"),
        ("mad_filter", "quantile", None),
        ("mad_filter", "scale", 0.0),
        ("mad_filter", "scale", float("inf")),
        ("mad_filter", "scale", float("nan")),
        ("diagnostics", "name", ""),
        ("diagnostics", "name", 5),
    ],
)
def test_invalid_config_values_raise(config_path, section, key, value):
    config = load_preprocessing_config(config_path)
    config[section][key] = value
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        load_preprocessing_config(config_path)


@pytest.mark.parametrize("content", ["", "[]", "mad_filter: [", "mad_filter: {}"])
def test_empty_malformed_or_incomplete_config_raises(config_path, content):
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="preprocessing"):
        load_preprocessing_config(config_path)


@pytest.mark.parametrize("section", [None, "mad_filter"])
def test_unknown_config_keys_raise(config_path, section):
    config = load_preprocessing_config(config_path)
    target = config if section is None else config[section]
    target["typo"] = 0.25
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        load_preprocessing_config(config_path)


def test_missing_config_has_no_silent_fallback(tmp_path, data):
    with pytest.raises(FileNotFoundError):
        MissingValueFilter(config_path=tmp_path / "missing.yaml").fit(data)
