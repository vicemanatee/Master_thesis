"""Standalone classifier factory: no data paths or preprocessing decisions."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

CLASSIFIERS = {
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def make_classifier(
    config: dict[str, Any], *, random_state: int | None = None
) -> BaseEstimator:
    kind = config["type"]
    if kind not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier type: {kind}")
    return CLASSIFIERS[kind](random_state=random_state, **config["params"])
