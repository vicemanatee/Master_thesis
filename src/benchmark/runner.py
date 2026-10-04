"""Patient-grouped evaluation of supplied pipelines, without model assembly."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, get_scorer, make_scorer, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

from ..data.dataset import ModelingDataset
from ..preprocessing.pipeline import inspect_preprocessor
from .results import BenchmarkResult


def make_cv_splits(
    dataset: ModelingDataset, *, config: dict[str, Any]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use labels and patient groups, not any specific data-file format."""
    X, y, groups = dataset.X, dataset.y, dataset.groups
    if set(y.unique()) != {0, 1}:
        raise ValueError("Benchmark requires both binary classes")
    cv = config["cv"]
    if groups.groupby(y).nunique().min() < cv["n_splits"]:
        raise ValueError("Too few patient groups per class for requested CV")
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
            raise ValueError("Both classes must occur in every fold")
    return splits


def make_scoring(names: list[str]) -> dict[str, Any]:
    if not names or len(names) != len(set(names)):
        raise ValueError("Scoring must contain nonempty, unique metric names")
    return {
        name: make_scorer(recall_score, pos_label=0)
        if name == "specificity"
        else get_scorer("recall" if name == "sensitivity" else name)
        for name in names
    }


def evaluate_models(
    dataset: ModelingDataset,
    pipelines: dict[str, Pipeline],
    *,
    config: dict[str, Any],
) -> BenchmarkResult:
    """Evaluate fresh composite pipelines using shared folds and metric settings.

    The evaluator receives no data paths or component parameters. Fitting the
    complete pipeline inside cross_validate keeps all learned transforms local
    to the training fold. Scores are sample-level, not patient-aggregated.
    """
    if not pipelines:
        raise ValueError("At least one experiment pipeline is required")
    splits = make_cv_splits(dataset, config=config)
    X, y, groups, metadata = dataset.X, dataset.y, dataset.groups, dataset.metadata
    scoring = make_scoring(config["scoring"])
    assignments = metadata.copy()
    assignments["fold"] = 0
    for fold, (_, test) in enumerate(splits, start=1):
        assignments.iloc[test, assignments.columns.get_loc("fold")] = fold

    metrics, predictions, features = [], [], []
    for name, pipeline in pipelines.items():
        print(f"Evaluating {name}: {len(splits)} patient-grouped folds", flush=True)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splits,
            scoring=scoring,
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
            diagnostics, selected = inspect_preprocessor(fitted["preprocessing"])
            record = {
                "model": name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "n_train_patients": groups.iloc[train].nunique(),
                "n_test_patients": groups.iloc[test].nunique(),
                "n_test_ln_negative": int(y.iloc[test].eq(0).sum()),
                "n_test_ln_positive": int(y.iloc[test].eq(1).sum()),
                "n_features_after_missing": diagnostics["n_features_after_missing"],
                "n_features_after_mad": diagnostics["n_features_after_mad"],
                "n_features_classifier_input": fitted["classifier"].n_features_in_,
                "n_pca_components": diagnostics["n_pca_components"],
                "pca_explained_variance_ratio_sum": diagnostics[
                    "pca_explained_variance_ratio_sum"
                ],
                "mad_threshold": diagnostics["mad_threshold"],
                "fit_time": scores["fit_time"][index],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            for metric in scoring:
                record[f"test_{metric}"] = scores[f"test_{metric}"][index]
                record[f"train_{metric}"] = scores[f"train_{metric}"][index]
            metrics.append(record)
            prediction = metadata.iloc[test].copy()
            prediction["model"], prediction["fold"] = name, fold
            prediction["y_true"], prediction["y_pred"] = y.iloc[test], predicted
            prediction["score"], prediction["score_type"] = continuous, score_type
            predictions.append(prediction)
            selected.insert(0, "fold", fold)
            selected.insert(0, "model", name)
            features.append(selected)

    fold_metrics = pd.DataFrame(metrics)
    metric_columns = [f"test_{metric}" for metric in scoring]
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
