"""Fit independent pathway pipelines on patient-grouped training folds."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer

from preprocessing import inspect_preprocessor


def validate_cv_splits(X, y, groups, splits):
    """Check alignment, binary labels, patient isolation and test coverage."""
    if not X.index.equals(y.index) or not X.index.equals(groups.index):
        raise ValueError('X, y and groups must have identical sample indices.')
    if X.empty or not X.index.is_unique:
        raise ValueError('X must contain samples with unique run indices.')
    if y.isna().any() or groups.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError('CV requires binary 0/1 labels and non-missing patient IDs.')
    if not splits:
        raise ValueError('CV must contain at least one split.')

    test_count = np.zeros(len(X), dtype=int)
    for train_idx, test_idx in splits:
        train_patients = set(groups.iloc[train_idx])
        test_patients = set(groups.iloc[test_idx])
        if not train_patients.isdisjoint(test_patients):
            raise ValueError('A patient occurs in both the training and test fold.')
        if y.iloc[train_idx].nunique() != 2 or y.iloc[test_idx].nunique() != 2:
            raise ValueError('Both LN classes must occur in every training and test fold.')
        np.add.at(test_count, test_idx, 1)

    if not np.all(test_count == 1):
        raise ValueError('Each run must occur in exactly one test fold.')


def run_cv(
        pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        splits,
        scoring: list[str],
        metadata: pd.DataFrame | None = None,
    ):
    """Return train/test metrics and out-of-fold predictions without reading files."""
    validate_cv_splits(X, y, groups, splits)
    if not scoring or len(scoring) != len(set(scoring)):
        raise ValueError('Scoring must contain non-empty, unique metric names.')
    scorers = {name: get_scorer(name) for name in scoring}

    if metadata is None:
        metadata = pd.DataFrame(index=X.index)
    if not X.index.equals(metadata.index):
        raise ValueError('Metadata must have the same sample indices as X.')

    fold_assignments = metadata.copy()
    fold_assignments['patient_id'] = groups
    fold_assignments['y_true'] = y
    fold_assignments['fold'] = 0
    metrics, predictions, selected_features, pathway_coverage = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fitted = clone(pipeline)
        start = perf_counter()
        fitted.fit(X_train, y_train)
        fit_time = perf_counter() - start

        predicted = fitted.predict(X_test)
        positive_column = list(fitted.classes_).index(1)
        probability = fitted.predict_proba(X_test)[:, positive_column]

        diagnostics, features = inspect_preprocessor(
            fitted.named_steps['protein_preprocessing']
        )
        membership = fitted.named_steps['pathway'].membership_
        record = {
            'fold': fold,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_train_patients': groups.iloc[train_idx].nunique(),
            'n_test_patients': groups.iloc[test_idx].nunique(),
            'n_train_ln_positive': int(y_train.eq(1).sum()),
            'n_test_ln_positive': int(y_test.eq(1).sum()),
            'n_features_after_missing': diagnostics['n_features_after_missing'],
            'n_features_after_mad': diagnostics['n_features_after_mad'],
            'n_pathways': membership.shape[1],
            'fit_time': fit_time,
        }
        for name, scorer in scorers.items():
            record[f'train_{name}'] = float(scorer(fitted, X_train, y_train))
            # Keep the existing metric columns for held-out test scores.
            record[name] = float(scorer(fitted, X_test, y_test))
        metrics.append(record)

        fold_assignments.loc[X_test.index, 'fold'] = fold
        prediction = metadata.iloc[test_idx].copy()
        prediction['patient_id'] = groups.iloc[test_idx]
        prediction['fold'] = fold
        prediction['y_true'] = y_test
        prediction['y_pred'] = predicted
        prediction['probability_ln_positive'] = probability
        predictions.append(prediction)

        features.insert(0, 'fold', fold)
        selected_features.append(features)
        coverage = pd.DataFrame({
            'pathway': membership.columns,
            'n_members': membership.ne(0).sum(axis=0).to_numpy(),
            'weight_sum': membership.sum(axis=0).to_numpy(),
        })
        coverage.insert(0, 'fold', fold)
        pathway_coverage.append(coverage)

        scores = ', '.join(
            f'{name}: train={record[f"train_{name}"]:.3f}/test={record[name]:.3f}'
            for name in scoring
        )
        print(f'Fold {fold}/{len(splits)}: {scores}', flush=True)

    fold_metrics = pd.DataFrame(metrics)
    summary = []
    for name in scoring:
        for split, column in [('train', f'train_{name}'), ('test', name)]:
            summary.append({
                'split': split,
                'metric': name,
                'mean': fold_metrics[column].mean(),
                'std': fold_metrics[column].std(),
            })

    return {
        'fold_metrics': fold_metrics,
        'summary': pd.DataFrame(summary),
        'predictions': pd.concat(predictions).reindex(X.index),
        'fold_assignments': fold_assignments,
        'selected_features': pd.concat(selected_features, ignore_index=True),
        'pathway_coverage': pd.concat(pathway_coverage, ignore_index=True),
    }
