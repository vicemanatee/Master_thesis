"""Check identity alignment and leakage boundaries, not estimator internals."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.benchmark.proteome_benchmark import (
    BENCHMARK_CONFIG,
    evaluate_models,
    make_cv_splits,
    make_model_pipelines,
    read_config,
    save_results,
)
from src.data.clinical import align_clinical_labels, load_clinical_labels


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


def test_preprocessing_uses_training_statistics_and_handles_test_nan():
    config = read_config(BENCHMARK_CONFIG)
    config["preprocessing"]["mad_quantile"] = 0.0
    X_train = pd.DataFrame(
        {
            "variable": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            "train_missing": [1.0, np.nan, 4.0, 8.0, 16.0, 32.0],
            "constant": [4.0] * 6,
        }
    )
    pipeline = make_model_pipelines(input_scale="log2", config=config)["svm"]
    pipeline.fit(X_train, [0, 0, 0, 1, 1, 1])
    assert pipeline["missing"].selected_features_.tolist() == ["variable", "constant"]
    assert pipeline["mad"].selected_features_.tolist() == ["variable"]
    imputer_value = pipeline["impute"].statistics_.copy()
    scaler_mean = pipeline["z_score"].mean_.copy()
    # An extreme held-out value and a held-out NaN cannot change fitted state.
    X_test = pd.DataFrame(
        {
            "variable": [np.nan, 1e9],
            "train_missing": [100.0, 100.0],
            "constant": [np.nan, 1e9],
        }
    )
    transformed = pipeline[:-1].transform(X_test)
    assert np.isfinite(transformed.to_numpy()).all()
    np.testing.assert_array_equal(pipeline["impute"].statistics_, imputer_value)
    np.testing.assert_array_equal(pipeline["z_score"].mean_, scaler_mean)
    np.testing.assert_allclose(imputer_value, [np.median(X_train.variable)])


def test_grouped_cv_predictions_and_result_exports(tmp_path):
    config = deepcopy(read_config(BENCHMARK_CONFIG))
    config["models"]["random_forest"].update(n_estimators=10, n_jobs=1)
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
    splits = make_cv_splits(X, metadata, config=config)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([s[1] for s in splits])), np.arange(40)
    )
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
    result = evaluate_models(X, metadata, input_scale="log2", config=config)
    assert len(result.fold_metrics) == 10
    assert len(result.predictions) == 80
    assert result.fold_assignments.groupby("Patient").fold.nunique().eq(1).all()
    assert result.predictions.groupby(["model", "sample_id"]).size().eq(1).all()
    assert np.isfinite(result.fold_metrics.filter(regex="^test_").to_numpy()).all()
    counts = result.fold_metrics.pivot(
        index="fold", columns="model", values="n_features_after_mad"
    )
    assert counts.svm.equals(counts.random_forest.rename("svm"))
    folder = save_results(result, tmp_path / "run")
    assert (folder / "run.json").exists()
    assert len(pd.read_csv(folder / "predictions.csv")) == 80
    with pytest.raises(FileExistsError):
        save_results(result, folder)
