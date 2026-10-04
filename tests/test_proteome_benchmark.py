"""Check identity alignment, component boundaries and CV leakage safety."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.benchmark.experiment import load_experiment, make_model_pipelines
from src.benchmark.models import make_classifier
from src.benchmark.results import save_results
from src.benchmark.runner import evaluate_models, make_cv_splits
from src.data.clinical import align_clinical_labels, load_clinical_labels
from src.data.dataset import ModelingDataset


def clinical_table():
    return (
        pd.DataFrame(
            {
                "Sample": ["S000001", "S000941"],
                "Patient": ["patient_a", "patient_b"],
                "LN": ["1", "0"],
                "LN.spec": ["1to3", "N0"],
            }
        )
        .astype("string")
        .set_index("Sample")
    )


def test_alignment_follows_runs_and_applies_correction():
    metadata = pd.DataFrame(
        {
            "sample_id": ["S000940", "S000001", "S000001"],
            "is_pool": [False, False, False],
        },
        index=["run_3", "run_1", "run_2"],
    )
    result = align_clinical_labels(
        metadata,
        clinical_table().iloc[::-1],
        sample_id_corrections={"S000940": "S000941"},
    )
    assert result.index.equals(metadata.index)
    assert result.LN.tolist() == [0, 1, 1]
    assert result.Patient.tolist() == ["patient_b", "patient_a", "patient_a"]
    assert result.iloc[0].original_sample_id == "S000940"
    assert result.iloc[0].sample_id == "S000941"


@pytest.mark.parametrize("failure", ["absent", "missing", "conflict", "duplicate"])
def test_alignment_rejects_invalid_labels(failure):
    clinical = clinical_table()
    metadata = pd.DataFrame(
        {"sample_id": ["S000001"], "is_pool": [False]}, index=["run"]
    )
    if failure == "absent":
        metadata["sample_id"] = "S999999"
    elif failure == "missing":
        clinical.loc["S000001", "Patient"] = pd.NA
    elif failure == "conflict":
        clinical.loc["S000001", "LN"] = "0"
    else:
        clinical = pd.concat([clinical, clinical.iloc[:1]])
    with pytest.raises(ValueError):
        align_clinical_labels(metadata, clinical)


def test_clinical_repeated_records_must_agree(monkeypatch):
    table = clinical_table().reset_index()
    records = pd.concat([table, table.iloc[:1]], ignore_index=True)
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: records.copy())
    assert len(load_clinical_labels("unused.xlsx")) == 2
    records.loc[2, "Patient"] = "different_patient"
    with pytest.raises(ValueError, match="Conflicting"):
        load_clinical_labels("unused.xlsx")


def as_dataset(X, metadata):
    return ModelingDataset(X, metadata.LN, metadata.Patient, metadata, "log2")


def full_experiment_config():
    """Use explicit combinations instead of depending on the active run list."""
    config = load_experiment()
    config["runs"] = deepcopy(
        [
            {"name": "svm", "preprocessing": "standard", "model": "svm"},
            {
                "name": "random_forest",
                "preprocessing": "unscaled",
                "model": "random_forest",
            },
            {
                "name": "logistic_regression",
                "preprocessing": "standard",
                "model": "logistic_l2",
            },
            {
                "name": "pca_logistic_regression",
                "preprocessing": "with_pca",
                "model": "logistic_l2",
            },
        ]
    )
    return config


def test_preprocessing_uses_training_statistics_and_handles_test_nan():
    config = full_experiment_config()
    config["preprocessing"]["steps"]["mad"]["quantile"] = 0.0
    X_train = pd.DataFrame(
        {
            "variable": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "train_missing": [1.0, np.nan, 4.0, 8.0, 16.0, 32.0],
            "constant": [4.0] * 6,
        }
    )
    pipeline = make_model_pipelines(input_scale="log2", config=config)["svm"]
    pipeline.fit(X_train, [0, 0, 0, 1, 1, 1])
    preprocessor = pipeline["preprocessing"]
    assert preprocessor["missing"].selected_features_.tolist() == [
        "variable",
        "constant",
    ]
    assert preprocessor["mad"].selected_features_.tolist() == ["variable"]
    imputer_value = preprocessor["impute"].statistics_.copy()
    scaler_mean = preprocessor["z_score"].mean_.copy()
    X_test = pd.DataFrame(
        {
            "variable": [np.nan, 1e9],
            "train_missing": [100.0, 100.0],
            "constant": [np.nan, 1e9],
        }
    )
    transformed = preprocessor.transform(X_test)
    assert np.isfinite(transformed.to_numpy()).all()
    np.testing.assert_array_equal(preprocessor["impute"].statistics_, imputer_value)
    np.testing.assert_array_equal(preprocessor["z_score"].mean_, scaler_mean)
    np.testing.assert_allclose(imputer_value, [np.median(X_train.variable)])


def test_grouped_cv_predictions_and_result_exports(tmp_path):
    config = full_experiment_config()
    config["benchmark"]["models"]["random_forest"]["params"].update(
        n_estimators=10, n_jobs=1
    )
    config["preprocessing"]["steps"]["pca"]["n_components"] = 2
    rng = np.random.default_rng(10)
    X = pd.DataFrame(rng.normal(size=(40, 12)), index=[f"run_{i}" for i in range(40)])
    groups = np.repeat([f"patient_{i}" for i in range(20)], 2)
    y = np.repeat(np.arange(20) % 2, 2)
    metadata = pd.DataFrame(
        {
            "Patient": groups,
            "LN": y,
            "sample_id": [f"sample_{i}" for i in range(40)],
        },
        index=X.index,
    )
    dataset = as_dataset(X, metadata)
    splits = make_cv_splits(dataset, config=config["benchmark"])
    np.testing.assert_array_equal(
        np.sort(np.concatenate([s[1] for s in splits])), np.arange(40)
    )
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
    pipelines = make_model_pipelines(input_scale=dataset.scale, config=config)
    result = evaluate_models(dataset, pipelines, config=config["benchmark"])
    n_models = len(config["runs"])
    assert len(result.fold_metrics) == 5 * n_models
    assert len(result.predictions) == len(X) * n_models
    assert set(result.summary.model) == {run["name"] for run in config["runs"]}
    assert result.fold_assignments.groupby("Patient").fold.nunique().eq(1).all()
    assert result.predictions.groupby(["model", "sample_id"]).size().eq(1).all()
    assert np.isfinite(result.fold_metrics.filter(regex="^test_").to_numpy()).all()
    counts = result.fold_metrics.pivot(
        index="fold", columns="model", values="n_features_after_mad"
    )
    assert counts.eq(counts.svm, axis=0).all().all()
    pca_metrics = result.fold_metrics.query("model == 'pca_logistic_regression'")
    assert pca_metrics.n_pca_components.eq(2).all()
    assert pca_metrics.n_features_classifier_input.eq(2).all()
    assert pca_metrics.pca_explained_variance_ratio_sum.between(0, 1).all()
    other_metrics = result.fold_metrics.query("model != 'pca_logistic_regression'")
    assert other_metrics.n_pca_components.eq(0).all()
    assert other_metrics.n_features_classifier_input.equals(
        other_metrics.n_features_after_mad.rename("n_features_classifier_input")
    )
    assert other_metrics.pca_explained_variance_ratio_sum.isna().all()
    folder = save_results(result, tmp_path / "run")
    assert (folder / "run.json").exists()
    assert len(pd.read_csv(folder / "predictions.csv")) == len(X) * n_models
    with pytest.raises(FileExistsError):
        save_results(result, folder)


def test_component_parameters_and_preprocessing_profiles_are_independent():
    config = full_experiment_config()
    config["benchmark"]["models"]["logistic_l2"]["params"]["C"] = 0.02
    config["preprocessing"]["steps"]["pca"]["n_components"] = 3
    pipelines = make_model_pipelines(input_scale="log2", config=config)
    assert "z_score" not in pipelines["random_forest"]["preprocessing"].named_steps
    assert "pca" not in pipelines["logistic_regression"]["preprocessing"].named_steps
    assert list(pipelines["pca_logistic_regression"]["preprocessing"].named_steps) == [
        "log2",
        "missing",
        "mad",
        "impute",
        "z_score",
        "pca",
    ]
    for name in ("logistic_regression", "pca_logistic_regression"):
        assert isinstance(pipelines[name]["classifier"], LogisticRegression)
        assert pipelines[name]["classifier"].l1_ratio == 0.0
        assert pipelines[name]["classifier"].C == 0.02
    assert (
        pipelines["pca_logistic_regression"]["preprocessing"]["pca"].n_components == 3
    )
    assert (
        pipelines["logistic_regression"]["classifier"]
        is not pipelines["pca_logistic_regression"]["classifier"]
    )


@pytest.mark.parametrize("model", ["svm", "random_forest", "logistic_l2"])
def test_pca_can_be_combined_with_any_classifier(model):
    config = load_experiment()
    config["preprocessing"]["steps"]["pca"]["n_components"] = 2
    config["preprocessing"]["steps"]["mad"]["quantile"] = 0.5
    config["benchmark"]["models"]["random_forest"]["params"].update(
        n_estimators=5, n_jobs=1
    )
    config["runs"] = [
        {"name": "arbitrary_run_name", "preprocessing": "with_pca", "model": model}
    ]
    pipeline = make_model_pipelines(input_scale="log2", config=config)[
        "arbitrary_run_name"
    ]
    assert "pca" in pipeline["preprocessing"].named_steps
    assert "z_score" in pipeline["preprocessing"].named_steps
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(40, 12)))
    pipeline.fit(X, np.arange(40) % 2)
    assert pipeline["classifier"].n_features_in_ == 2
    assert pipeline.predict(X).shape == (40,)


def test_experiment_can_select_only_svm_rf_without_enabling_pca():
    config = full_experiment_config()
    config["runs"] = config["runs"][:2]
    pipelines = make_model_pipelines(input_scale="log2", config=config)
    assert list(pipelines) == ["svm", "random_forest"]
    assert all(
        "pca" not in model["preprocessing"].named_steps for model in pipelines.values()
    )


def test_classifier_factory_rejects_combination_as_classifier_type():
    with pytest.raises(ValueError, match="Unknown classifier"):
        make_classifier({"type": "pca_logistic_regression", "params": {}})


def test_pca_is_fitted_inside_each_cv_training_fold(monkeypatch):
    config = full_experiment_config()
    config["runs"] = config["runs"][-1:]
    config["preprocessing"]["steps"]["pca"]["n_components"] = 2
    rng = np.random.default_rng(23)
    X = pd.DataFrame(rng.normal(size=(40, 20)), index=[f"run_{i}" for i in range(40)])
    metadata = pd.DataFrame(
        {
            "Patient": [f"patient_{i}" for i in range(40)],
            "LN": np.arange(40) % 2,
            "sample_id": [f"sample_{i}" for i in range(40)],
        },
        index=X.index,
    )
    dataset = as_dataset(X, metadata)
    seen_indices, fitted_reducers = [], []
    original_fit_transform = PCA.fit_transform

    def track_fit_transform(self, data, y=None):
        seen_indices.append(data.index.copy())
        transformed = original_fit_transform(self, data, y)
        fitted_reducers.append((self, self.components_.copy(), self.mean_.copy()))
        return transformed

    monkeypatch.setattr(PCA, "fit_transform", track_fit_transform)
    result = evaluate_models(
        dataset,
        make_model_pipelines(input_scale=dataset.scale, config=config),
        config=config["benchmark"],
    )
    splits = make_cv_splits(dataset, config=config["benchmark"])
    assert len(seen_indices) == len(splits)
    for seen, (train, test) in zip(seen_indices, splits, strict=True):
        assert seen.equals(X.iloc[train].index)
        assert not set(seen) & set(X.iloc[test].index)
    for reducer, components, mean in fitted_reducers:
        np.testing.assert_array_equal(reducer.components_, components)
        np.testing.assert_array_equal(reducer.mean_, mean)
    assert len(result.predictions) == len(X)
    assert result.predictions.groupby("sample_id").size().eq(1).all()


def test_pca_pipeline_reuses_training_statistics_with_held_out_nan():
    config = full_experiment_config()
    config["preprocessing"]["steps"]["mad"]["quantile"] = 0.0
    config["preprocessing"]["steps"]["pca"]["n_components"] = 2
    rng = np.random.default_rng(8)
    train = pd.DataFrame(rng.normal(size=(20, 8)), columns=[f"p{i}" for i in range(8)])
    test = train.iloc[:2].copy()
    test.index = ["held_out_nan", "held_out_extreme"]
    test.iloc[0, :] = np.nan
    test.iloc[1, :] = 1e9
    pipeline = make_model_pipelines(input_scale="log2", config=config)[
        "pca_logistic_regression"
    ]
    pipeline.fit(train, np.arange(20) % 2)
    preprocessor = pipeline["preprocessing"]
    components = preprocessor["pca"].components_.copy()
    pca_mean = preprocessor["pca"].mean_.copy()
    scaler_mean = preprocessor["z_score"].mean_.copy()
    imputer_values = preprocessor["impute"].statistics_.copy()
    transformed = preprocessor.transform(test)
    assert transformed.shape == (2, 2)
    assert transformed.index.equals(test.index)
    assert np.isfinite(transformed.to_numpy()).all()
    np.testing.assert_array_equal(preprocessor["pca"].components_, components)
    np.testing.assert_array_equal(preprocessor["pca"].mean_, pca_mean)
    np.testing.assert_array_equal(preprocessor["z_score"].mean_, scaler_mean)
    np.testing.assert_array_equal(preprocessor["impute"].statistics_, imputer_values)
