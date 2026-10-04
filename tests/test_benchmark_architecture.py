"""Module-owned configs, explicit composition, and disabled-step diagnostics."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.benchmark import experiment as experiment_module
from src.benchmark.experiment import (
    EXPERIMENT_CONFIG,
    load_experiment,
    make_model_pipelines,
    run_benchmark,
)
from src.benchmark.results import save_results
from src.benchmark.runner import evaluate_models
from src.config import read_config
from src.data import dataset as dataset_module
from src.data.dataset import ModelingDataset, load_dataset_config, prepare_dataset
from src.data.loader import LoadedMatrix
from src.preprocessing import build_preprocessor, resolve_preprocessing_profile


def write_yaml(path, mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")


@pytest.fixture
def configuration_tree(tmp_path):
    folder = tmp_path / "config" / "modules"
    project_configs = EXPERIMENT_CONFIG.parent
    data = read_config(project_configs / "data.yaml")
    data["datasets"]["proteome_ln"]["paths_config"] = "../paths.yaml"
    benchmark = read_config(project_configs / "benchmark.yaml")
    benchmark["output_root"] = "../../runs"
    write_yaml(folder / "data.yaml", data)
    write_yaml(
        folder / "preprocessing.yaml",
        read_config(project_configs / "preprocessing.yaml"),
    )
    write_yaml(folder / "benchmark.yaml", benchmark)
    write_yaml(folder.parent / "paths.yaml", read_config(project_configs / "path.yaml"))
    experiment = read_config(EXPERIMENT_CONFIG)
    # Architecture tests exercise reference validation and result snapshots with
    # several combinations; they must not depend on the user's active run list.
    experiment["runs"] = [
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
    for module in ("data", "preprocessing", "benchmark"):
        experiment[f"{module}_config"] = f"config/modules/{module}.yaml"
    source = tmp_path / "experiment.yaml"
    write_yaml(source, experiment)
    return source


def test_references_resolve_against_each_owner_not_cwd(
    configuration_tree, monkeypatch, tmp_path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_experiment(configuration_tree)
    assert Path(config["data"]["paths_config"]) == tmp_path / "config" / "paths.yaml"
    assert Path(config["benchmark"]["output_root"]) == tmp_path / "runs"
    assert all(Path(value).is_absolute() for value in config["config_paths"].values())


def test_previous_config_path_is_reference_only_and_resolves_to_same_experiment():
    alias = EXPERIMENT_CONFIG.parent / "proteome_benchmark.yaml"
    assert set(read_config(alias)) == {"experiment_config"}
    assert load_experiment(alias) == load_experiment()


def test_data_config_can_be_loaded_without_any_experiment(configuration_tree):
    source = configuration_tree.parent / "config" / "modules" / "data.yaml"
    settings = load_dataset_config(source, dataset="proteome_ln")
    assert settings == load_experiment(configuration_tree)["data"]
    assert Path(settings["paths_config"]) == source.parent.parent / "paths.yaml"


@pytest.mark.parametrize(
    "failure", ["dataset", "model", "profile", "duplicate", "empty", "run_parameter"]
)
def test_bad_experiment_references_are_rejected(configuration_tree, failure):
    config = read_config(configuration_tree)
    if failure == "dataset":
        config["dataset"] = "not_a_dataset"
    elif failure == "model":
        config["runs"][0]["model"] = "not_a_model"
    elif failure == "profile":
        config["runs"][0]["preprocessing"] = "not_a_profile"
    elif failure == "duplicate":
        config["runs"][1]["name"] = config["runs"][0]["name"]
    elif failure == "empty":
        config["runs"] = []
    else:
        config["runs"][0]["C"] = 1.0
    write_yaml(configuration_tree, config)
    with pytest.raises(ValueError):
        load_experiment(configuration_tree)


def test_cyclic_compatibility_reference_is_rejected(tmp_path):
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    write_yaml(first, {"experiment_config": "second.yaml"})
    write_yaml(second, {"experiment_config": "first.yaml"})
    with pytest.raises(ValueError, match="Cyclic"):
        load_experiment(first)


def test_factories_do_not_fit_or_mutate_configs():
    config = load_experiment()
    snapshot = deepcopy(config)
    pipelines = make_model_pipelines(input_scale="log2", config=config)
    assert config == snapshot
    for pipeline in pipelines.values():
        assert not hasattr(pipeline["classifier"], "classes_")
        assert not hasattr(pipeline["preprocessing"]["log2"], "feature_names_in_")
    profile = resolve_preprocessing_profile(config["preprocessing"], "with_pca")
    profile["pca"]["n_components"] = 999
    assert config == snapshot


@pytest.mark.parametrize("failure", ["unknown_step", "non_boolean", "disable_log2"])
def test_invalid_preprocessing_settings_fail_explicitly(failure):
    config = load_experiment()["preprocessing"]
    if failure == "unknown_step":
        config["profiles"]["standard"]["typo"] = {"enabled": True}
    elif failure == "non_boolean":
        config["steps"]["mad"]["enabled"] = "false"
    else:
        config["steps"]["log2"]["enabled"] = False
    with pytest.raises((ValueError, TypeError)):
        build_preprocessor(config, profile="standard", input_scale="log2")


@pytest.mark.parametrize("only_log2", [False, True])
def test_disabled_filters_and_pca_do_not_break_evaluation_or_exports(only_log2):
    config = load_experiment()
    overrides = {"missing": {"enabled": False}, "mad": {"enabled": False}}
    if only_log2:
        overrides.update(
            {name: {"enabled": False} for name in ("impute", "z_score", "pca")}
        )
    config["preprocessing"]["profiles"]["no_filters"] = overrides
    config["runs"] = [
        {"name": "l2_no_filters", "preprocessing": "no_filters", "model": "logistic_l2"}
    ]
    config["benchmark"]["scoring"] = ["roc_auc", "accuracy"]
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(40, 12)), index=[f"run_{i}" for i in range(40)])
    if not only_log2:
        X.iloc[0, 0] = np.nan
    metadata = pd.DataFrame(
        {
            "Patient": [f"patient_{i}" for i in range(40)],
            "LN": np.arange(40) % 2,
            "sample_id": [f"sample_{i}" for i in range(40)],
        },
        index=X.index,
    )
    dataset = ModelingDataset(X, metadata.LN, metadata.Patient, metadata, "log2")
    pipelines = make_model_pipelines(input_scale=dataset.scale, config=config)
    result = evaluate_models(dataset, pipelines, config=config["benchmark"])
    assert len(result.fold_metrics) == 5
    assert result.fold_metrics.n_features_after_missing.eq(12).all()
    assert result.fold_metrics.n_features_after_mad.eq(12).all()
    assert result.fold_metrics.n_features_classifier_input.eq(12).all()
    assert result.fold_metrics.mad_threshold.isna().all()
    assert result.fold_metrics.n_pca_components.eq(0).all()
    assert result.selected_features.train_mad.isna().all()
    assert len(result.selected_features) == 5 * 12
    assert "test_f1" not in result.fold_metrics
    assert len(result.predictions) == len(X)


def test_dataset_assembly_preserves_abundances_and_separates_labels(
    configuration_tree, monkeypatch
):
    config = load_experiment(configuration_tree)["data"]
    X = pd.DataFrame(
        {
            "P1": [1.0, 4.0, np.nan],
            "P2": [3.0, 8.0, 5.0],
            "Cont_ABCDEF": [6.0, 9.0, 7.0],
        },
        index=["run_2", "run_1", "run_3"],
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["S000001", "S000940", "S000001"],
            "is_pool": [False] * 3,
        },
        index=X.index,
    )
    loaded = LoadedMatrix(
        X=X,
        sample_metadata=metadata,
        feature_metadata=pd.DataFrame({"Protein.Group": X.columns}, index=X.columns),
        omics="proteome",
        source_path=configuration_tree.parent / "pg.tsv",
        scale="linear",
        processing_stage="diann",
    )
    clinical = (
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
    calls = []

    def fake_load_matrix(omics, **kwargs):
        calls.append((omics, kwargs))
        return loaded

    monkeypatch.setattr(dataset_module, "load_matrix", fake_load_matrix)
    monkeypatch.setattr(
        dataset_module, "load_clinical_labels", lambda *args, **kwargs: clinical
    )
    result = prepare_dataset(config)
    assert calls[0][0] == "proteome"
    assert calls[0][1]["config_path"] == Path(config["paths_config"])
    assert result.scale == "linear"
    assert result.X.columns.tolist() == ["P1", "P2"]
    assert result.X.index.tolist() == ["run_2", "run_3", "run_1"]
    pd.testing.assert_frame_equal(result.X, X.loc[result.X.index, ["P1", "P2"]])
    assert result.X.isna().sum().sum() == 1
    assert result.y.tolist() == [1, 1, 0]
    assert result.groups.tolist() == ["patient_a", "patient_a", "patient_b"]
    assert "LN" not in result.X
    assert metadata.sample_id.tolist() == ["S000001", "S000940", "S000001"]


def test_dataset_rejects_misaligned_labels():
    X = pd.DataFrame({"p": [1.0, 2.0]}, index=["r1", "r2"])
    metadata = pd.DataFrame({"LN": [0, 1], "Patient": ["a", "b"]}, index=X.index)
    with pytest.raises(ValueError, match="identical run indices"):
        ModelingDataset(X, metadata.LN.iloc[::-1], metadata.Patient, metadata, "log2")


def test_effective_overrides_and_component_snapshots_are_saved(
    configuration_tree, monkeypatch, tmp_path
):
    module_folder = configuration_tree.parent / "config" / "modules"
    preprocessing = read_config(module_folder / "preprocessing.yaml")
    preprocessing["steps"]["pca"]["n_components"] = 2
    preprocessing["steps"]["mad"]["quantile"] = 0.5
    write_yaml(module_folder / "preprocessing.yaml", preprocessing)
    benchmark = read_config(module_folder / "benchmark.yaml")
    benchmark["models"]["random_forest"]["params"].update(n_estimators=5, n_jobs=1)
    write_yaml(module_folder / "benchmark.yaml", benchmark)
    paths_override = tmp_path / "overrides" / "paths.yaml"
    write_yaml(
        paths_override,
        {
            "matrices": {"proteome_raw": "custom.tsv"},
            "clinical": {"path": "clinical.xlsx", "sheet_name": "SCANB.9206"},
        },
    )
    rng = np.random.default_rng(18)
    X = pd.DataFrame(
        rng.lognormal(size=(40, 12)), index=[f"run_{i}" for i in range(40)]
    )
    metadata = pd.DataFrame(
        {
            "Patient": [f"patient_{i}" for i in range(40)],
            "LN": np.arange(40) % 2,
            "sample_id": [f"sample_{i}" for i in range(40)],
        },
        index=X.index,
    )
    seen = []

    def prepare_without_io(config):
        seen.append(deepcopy(config))
        return ModelingDataset(
            X,
            metadata.LN,
            metadata.Patient,
            metadata,
            "linear",
            {"processing_stage": "diann", "input_scale": "linear"},
        )

    monkeypatch.setattr(experiment_module, "prepare_dataset", prepare_without_io)
    result = run_benchmark(
        config_path=configuration_tree, paths_path=paths_override, input_format="diann"
    )
    folder = save_results(result, tmp_path / "saved_run")
    snapshot = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert seen[0]["input_format"] == "diann"
    assert seen[0]["paths_config"] == str(paths_override.resolve())
    assert snapshot["config"]["data"] == seen[0]
    assert snapshot["config"]["paths"] == read_config(paths_override)
    assert snapshot["config"]["preprocessing"] == preprocessing
    assert snapshot["config"]["benchmark"]["models"] == benchmark["models"]
    assert snapshot["config_paths"]["paths"] == str(paths_override.resolve())
    assert load_experiment(configuration_tree)["data"]["input_format"] == "rollup"
    assert len(result.predictions) == len(X) * 4
