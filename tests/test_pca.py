"""Check the reusable library PCA's labels and train/held-out contract."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA

from src.preprocessing import make_pca


def test_pca_matches_sklearn_and_preserves_sample_indices():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        rng.normal(size=(12, 5)),
        index=[f"run_{i}" for i in range(12)],
        columns=[f"protein_{i}" for i in range(5)],
    )
    reducer = make_pca(n_components=2)
    transformed = reducer.fit_transform(train)
    reference = PCA(n_components=2, svd_solver="full").fit_transform(train)
    assert isinstance(reducer, PCA)
    assert transformed.index.equals(train.index)
    assert transformed.columns.tolist() == ["pca0", "pca1"]
    np.testing.assert_allclose(transformed.to_numpy(), reference)
    assert clone(reducer).get_params() == reducer.get_params()


def test_pca_transform_does_not_learn_from_held_out_data():
    train = pd.DataFrame({"a": [1.0, 3.0, 7.0, 4.0], "b": [2.0, 8.0, 1.0, 5.0]})
    reducer = make_pca(n_components=1).fit(train)
    components = reducer.components_.copy()
    mean = reducer.mean_.copy()
    test = pd.DataFrame({"a": [1e9], "b": [-1e9]}, index=["held_out"])
    transformed = reducer.transform(test)
    assert transformed.index.equals(test.index)
    assert transformed.shape == (1, 1)
    np.testing.assert_array_equal(reducer.components_, components)
    np.testing.assert_array_equal(reducer.mean_, mean)
    np.testing.assert_allclose(
        transformed.to_numpy(), (test.to_numpy() - mean) @ components.T
    )


def test_pca_rejects_component_count_above_training_dimensions():
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 1.0]})
    with pytest.raises(ValueError, match="n_components"):
        make_pca(n_components=3).fit(data)
