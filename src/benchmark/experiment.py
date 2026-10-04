"""Composition boundary: resolve module configs and assemble named experiments."""

from __future__ import annotations

import platform
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from ..config import read_config, resolve_config_path
from ..data.dataset import load_dataset_config, prepare_dataset
from ..preprocessing.pipeline import build_preprocessor, resolve_preprocessing_profile
from .models import make_classifier
from .results import BenchmarkResult
from .runner import evaluate_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
# Preserve the previous public constant as a compatibility experiment entry.
BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "proteome_benchmark.yaml"


def load_experiment(path: str | Path = EXPERIMENT_CONFIG) -> dict[str, Any]:
    """Resolve YAML references; module files own parameters, runs own choices."""
    source = Path(path).resolve()
    visited = set()
    while True:
        if source in visited:
            raise ValueError("Cyclic experiment_config reference")
        visited.add(source)
        experiment = read_config(source)
        if "experiment_config" not in experiment:
            break
        source = resolve_config_path(experiment["experiment_config"], owner=source)
    modules = {}
    config_paths = {"experiment": str(source)}
    for module in ("data", "preprocessing", "benchmark"):
        module_path = resolve_config_path(experiment[f"{module}_config"], owner=source)
        if module != "data":
            modules[module] = read_config(module_path)
        config_paths[module] = str(module_path)
    dataset_name = experiment["dataset"]
    data = load_dataset_config(config_paths["data"], dataset=dataset_name)
    runs = experiment["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("Experiment requires a nonempty runs list")
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("Experiment runs must be mappings")
        if set(run) != {"name", "preprocessing", "model"}:
            raise ValueError(
                "Each run defines only name, preprocessing and model references"
            )
    names = [run["name"] for run in runs]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Experiment run names must be nonempty and unique")
    for run in runs:
        if run["model"] not in modules["benchmark"]["models"]:
            raise ValueError(f"Unknown model reference: {run['model']}")
        resolve_preprocessing_profile(modules["preprocessing"], run["preprocessing"])
    modules["benchmark"]["output_root"] = str(
        resolve_config_path(
            modules["benchmark"]["output_root"], owner=config_paths["benchmark"]
        )
    )
    return {
        "dataset": dataset_name,
        "data": data,
        "preprocessing": modules["preprocessing"],
        "benchmark": modules["benchmark"],
        "runs": deepcopy(runs),
        "config_paths": config_paths,
    }


def make_model_pipelines(
    *, input_scale: str, config: dict[str, Any]
) -> dict[str, Pipeline]:
    """Assemble choices without coupling a classifier's type to preprocessing."""
    benchmark = config["benchmark"]
    pipelines = {}
    for run in config["runs"]:
        preprocessor = build_preprocessor(
            config["preprocessing"],
            profile=run["preprocessing"],
            input_scale=input_scale,
            random_state=benchmark["random_state"],
        )
        classifier = make_classifier(
            benchmark["models"][run["model"]],
            random_state=benchmark["random_state"],
        )
        pipelines[run["name"]] = Pipeline(
            [("preprocessing", preprocessor), ("classifier", classifier)]
        )
    return pipelines


def run_benchmark(
    *,
    config_path: str | Path = EXPERIMENT_CONFIG,
    paths_path: str | Path | None = None,
    input_format: str | None = None,
) -> BenchmarkResult:
    """Thin orchestration: load aligned data, build components, evaluate once."""
    config = load_experiment(config_path)
    if paths_path is not None:
        config["data"]["paths_config"] = str(Path(paths_path).resolve())
    if input_format is not None:
        config["data"]["input_format"] = input_format
    config["config_paths"]["paths"] = config["data"]["paths_config"]
    config["paths"] = read_config(config["data"]["paths_config"])
    dataset = prepare_dataset(config["data"])
    print(
        f"{dataset.provenance['processing_stage']}: {len(dataset.X)} runs, "
        f"{dataset.groups.nunique()} patients, {dataset.X.shape[1]} features; "
        f"LN counts {dataset.y.value_counts().sort_index().to_dict()}",
        flush=True,
    )
    pipelines = make_model_pipelines(input_scale=dataset.scale, config=config)
    result = evaluate_models(dataset, pipelines, config=config["benchmark"])
    result.provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "proteome LN-negative (0) versus LN-positive (1)",
        **dataset.provenance,
        "config_path": str(Path(config_path).resolve()),
        "config_paths": config["config_paths"],
        "feature_selection_scale": "log2",
        "config": config,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "evaluation": "Fixed parameters; grouped CV; no separate test set; sample-level metrics.",
        "uncertainty": "Summary std is across folds, not a confidence interval.",
    }
    return result
