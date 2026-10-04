"""PCA backed by scikit-learn, for training-fold fit/held-out transform."""

from __future__ import annotations

from sklearn.decomposition import PCA


def make_pca(
    *,
    n_components: float | None,
    svd_solver: str = "full",
    whiten: bool = False,
    random_state: int | None = None,
) -> PCA:
    """Create a pandas-preserving PCA step without fitting any data.

    Place after imputation and scaling inside the classifier Pipeline. PCA
    centers features but does not standardize them. Fit only on the training
    fold; validation/test data must reuse the fitted axes via ``transform``.
    Integer component counts must fit every training fold's sample/feature
    dimensions; scikit-learn raises rather than silently changing the count.
    """
    return PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        whiten=whiten,
        random_state=random_state,
    ).set_output(transform="pandas")
