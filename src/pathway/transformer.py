'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-10-12 20:09:01
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-12 20:22:39
FilePath: /Master_Thesis/src/pathway/transformer.py
Description: A transformer for converting feature-level data to pathway-level data. 
'''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd

from .aggregation import aggregate_feature_to_pathway

class PathwayTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, membership):
        self.membership = membership
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer to data. 
        X: DataFrame of shape (n_samples, n_features)
        y: Ignored
        """
        self.feature_names_in_ = X.columns
        aligned_membership = self.membership.loc[self.feature_names_in_]
        weight_sum = aligned_membership.sum(axis=0)
        self.membership_ = aligned_membership.loc[:, weight_sum.ne(0)].copy()

        if self.membership_.empty:
            raise ValueError("No pathways with non-zero total membership weight after aligning to X features.")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aggregate samples using the PGs and pathways retained during fit."""
        check_is_fitted(self, ["feature_names_in_", "membership_"])
        X_aligned = X.loc[:, self.feature_names_in_]
        return aggregate_feature_to_pathway(X_aligned, self.membership_)
