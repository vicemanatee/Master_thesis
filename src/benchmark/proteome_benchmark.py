"""Proteome LN classification with fixed SVM/RF and patient-grouped CV.

Run from the project root: uv run python -m src.benchmark.proteome_benchmark
All learned downstream preprocessing lives inside sklearn Pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, make_scorer, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ..data.clinical import align_clinical_labels, load_clinical_labels
from ..data.loader import CONFIG_PATH, load_matrix
from ..preprocessing import (
    Log2Transformer,
    MADFilter,
    MissingValueFilter,
    make_z_score_scaler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "proteome_benchmark.yaml"
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
    "f1": "f1",
    "sensitivity": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
}


@dataclass
class BenchmarkResult:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    selected_features: pd.DataFrame
    provenance: dict[str, Any]


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return config


def make_model_pipelines(
    *, input_scale: str, config: dict[str, Any]
) -> dict[str, Pipeline]:
    """Build fresh library estimators; CV clones them for each training fold."""
    preprocessing = config["preprocessing"]
    models = {
        "svm": SVC(random_state=config["random_state"], **config["models"]["svm"]),
        "random_forest": RandomForestClassifier(
            random_state=config["random_state"], **config["models"]["random_forest"]
        ),
    }
    pipelines = {}
    for name, classifier in models.items():
        steps = [
            ("log2", Log2Transformer(input_scale=input_scale)),
            ("missing", MissingValueFilter(preprocessing["max_missing_fraction"])),
            (
                "mad",
                MADFilter(
                    quantile=preprocessing["mad_quantile"],
                    scale=preprocessing["mad_scale"],
                ),
            ),
            (
                "impute",
                SimpleImputer(strategy=preprocessing["imputer_strategy"]).set_output(
                    transform="pandas"
                ),
            ),
        ]
        if name == "svm":
            steps.append(("z_score", make_z_score_scaler()))
        steps.append(("classifier", classifier))
        pipelines[name] = Pipeline(steps)
    return pipelines


def make_cv_splits(
    X: pd.DataFrame, metadata: pd.DataFrame, *, config: dict[str, Any]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Share one LN-stratified, patient-grouped partition across classifiers."""
    if not X.index.is_unique or not X.index.equals(metadata.index):
        raise ValueError("X and metadata must have identical unique run indices")
    y, groups = metadata["LN"], metadata["Patient"]
    if set(y.unique()) != {0, 1} or groups.isna().any():
        raise ValueError("Benchmark requires both LN classes and valid Patient IDs")
    cv = config["cv"]
    if metadata.groupby("LN")["Patient"].nunique().min() < cv["n_splits"]:
        raise ValueError("Too few patient groups per LN class for requested CV")
    splitter = StratifiedGroupKFold(
        n_splits=cv["n_splits"],
        shuffle=cv["shuffle"],
        random_state=config["random_state"] if cv["shuffle"] else None,
    )
    splits = list(splitter.split(X, y, groups))
    for train, test in splits:
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise ValueError("Patient overlap between train and held-out fold")
        if y.iloc[train].nunique() != 2 or y.iloc[test].nunique() != 2:
            raise ValueError("Both LN classes must occur in every fold")
    return splits


def evaluate_models(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    input_scale: str,
    config: dict[str, Any],
) -> BenchmarkResult:
    """Use library CV; retain held-out predictions and fold-specific features.

    Metrics are sample-level; repeated samples stay in one patient group.
    Parameters are fixed. There is no separate test set or tuning in this run.
    """
    splits = make_cv_splits(X, metadata, config=config)
    y = metadata["LN"]
    assignments = metadata.copy()
    assignments["fold"] = 0
    for fold, (_, test) in enumerate(splits, start=1):
        assignments.iloc[test, assignments.columns.get_loc("fold")] = fold

    metrics, predictions, features = [], [], []
    pipelines = make_model_pipelines(input_scale=input_scale, config=config)
    for name, pipeline in pipelines.items():
        print(f"Evaluating {name}: {len(splits)} patient-grouped folds", flush=True)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splits,
            scoring=SCORING,
            n_jobs=config["cv"]["n_jobs"],
            return_train_score=True,
            return_estimator=True,
            error_score="raise",
        )
        for index, ((train, test), fitted) in enumerate(
            zip(splits, scores["estimator"], strict=True)
        ):
            fold = index + 1
            X_test = X.iloc[test]
            predicted = fitted.predict(X_test)
            if hasattr(fitted, "decision_function"):
                continuous = fitted.decision_function(X_test)
                score_type = "decision_function"
            else:
                continuous = fitted.predict_proba(X_test)[:, 1]
                score_type = "probability_LN_positive"
            tn, fp, fn, tp = confusion_matrix(
                y.iloc[test], predicted, labels=[0, 1]
            ).ravel()
            record = {
                "model": name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "n_train_patients": metadata.iloc[train]["Patient"].nunique(),
                "n_test_patients": metadata.iloc[test]["Patient"].nunique(),
                "n_test_ln_negative": int(y.iloc[test].eq(0).sum()),
                "n_test_ln_positive": int(y.iloc[test].eq(1).sum()),
                "n_features_after_missing": len(fitted["missing"].selected_features_),
                "n_features_after_mad": len(fitted["mad"].selected_features_),
                "mad_threshold": fitted["mad"].mad_threshold_,
                "fit_time": scores["fit_time"][index],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            for metric in SCORING:
                record[f"test_{metric}"] = scores[f"test_{metric}"][index]
                record[f"train_{metric}"] = scores[f"train_{metric}"][index]
            metrics.append(record)
            prediction = metadata.iloc[test].copy()
            prediction["model"] = name
            prediction["fold"] = fold
            prediction["y_true"] = y.iloc[test]
            prediction["y_pred"] = predicted
            prediction["score"] = continuous
            prediction["score_type"] = score_type
            predictions.append(prediction)
            selected = fitted["mad"].selected_features_
            features.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold,
                        "feature_id": selected,
                        "train_mad": fitted["mad"]
                        .mad_scores_.reindex(selected)
                        .to_numpy(),
                    }
                )
            )

    fold_metrics = pd.DataFrame(metrics)
    metric_columns = [f"test_{metric}" for metric in SCORING]
    summary = fold_metrics.groupby("model", sort=False)[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return BenchmarkResult(
        fold_metrics,
        summary.reset_index(),
        pd.concat(predictions),
        assignments,
        pd.concat(features, ignore_index=True),
        {},
    )


def run_benchmark(
    *,
    config_path: str | Path = BENCHMARK_CONFIG,
    paths_path: str | Path = CONFIG_PATH,
    input_format: str | None = None,
) -> BenchmarkResult:
    """Load once, align sample identities to clinical labels, then evaluate."""
    config = read_config(config_path)
    config["input_format"] = input_format or config["input_format"]
    paths_path = Path(paths_path).resolve()
    paths = read_config(paths_path)
    clinical_path = Path(paths["clinical"]["path"])
    if not clinical_path.is_absolute():
        clinical_path = (paths_path.parent / clinical_path).resolve()
    loaded = load_matrix(
        "proteome", config_path=paths_path, input_format=config["input_format"]
    )
    X = loaded.X
    if loaded.processing_stage == "diann":
        contamination = loaded.feature_metadata["Protein.Group"].str.match(
            config["cohort"]["diann_contaminant_pattern"]
        )
        X = X.loc[:, ~contamination.to_numpy()]
    clinical = load_clinical_labels(
        clinical_path, sheet_name=paths["clinical"]["sheet_name"]
    )
    metadata = align_clinical_labels(
        loaded.sample_metadata,
        clinical,
        sample_id_corrections=config["cohort"]["sample_id_corrections"],
    )
    # Canonical order gives identical partitions for rollup/diann comparisons.
    metadata = metadata.sort_values("sample_id", kind="stable")
    X = X.loc[metadata.index]
    print(
        f"{loaded.processing_stage}: {len(X)} runs, {metadata.Patient.nunique()} patients, "
        f"{X.shape[1]} features; LN counts {metadata.LN.value_counts().sort_index().to_dict()}",
        flush=True,
    )
    result = evaluate_models(X, metadata, input_scale=loaded.scale, config=config)
    result.provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "proteome LN-negative (0) versus LN-positive (1)",
        "matrix_path": str(loaded.source_path),
        "sample_metadata_path": str(loaded.sample_metadata_path)
        if loaded.sample_metadata_path
        else None,
        "clinical_path": str(clinical_path),
        "clinical_sheet": paths["clinical"]["sheet_name"],
        "config_path": str(Path(config_path).resolve()),
        "paths_path": str(paths_path),
        "input_scale": loaded.scale,
        "feature_selection_scale": "log2",
        "processing_stage": loaded.processing_stage,
        "n_samples": len(X),
        "n_patients": metadata.Patient.nunique(),
        "n_input_features": X.shape[1],
        "label_counts": metadata.LN.value_counts().sort_index().to_dict(),
        "config": config,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "evaluation": "Fixed parameters; grouped CV; no separate test set; sample-level metrics.",
        "uncertainty": "Summary std is across folds, not a confidence interval.",
        "upstream_scope": (
            "CycLoess and RRollup were computed across the full cohort including Pool. "
            "CV uses a fixed, jointly processed matrix, not a fully inductive upstream pipeline."
            if loaded.processing_stage == "rollup"
            else "DIA-NN quantification/normalization used the original experiment. "
            "Upstream mass-spectrometry processing was not rerun per fold."
        ),
    }
    return result


def save_results(result: BenchmarkResult, output_dir: str | Path) -> Path:
    """Export readable tables and resolved configuration without replacing a run."""
    folder = Path(output_dir).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("fold_metrics", "summary", "selected_features"):
        getattr(result, name).to_csv(folder / f"{name}.csv", index=False)
    for name in ("predictions", "fold_assignments"):
        getattr(result, name).to_csv(folder / f"{name}.csv", index_label="run_id")
    with (folder / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(result.provenance, stream, indent=2, ensure_ascii=False)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BENCHMARK_CONFIG)
    parser.add_argument("--paths", type=Path, default=CONFIG_PATH)
    parser.add_argument("--input-format", choices=("rollup", "diann"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "results"
        / "proteome_benchmark"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}")
    result = run_benchmark(
        config_path=args.config, paths_path=args.paths, input_format=args.input_format
    )
    folder = save_results(result, output_dir)
    print(result.summary.to_string(index=False))
    print(f"Results saved to {folder}")


if __name__ == "__main__":
    main()
