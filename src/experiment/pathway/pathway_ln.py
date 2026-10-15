'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-10-12 21:40:22
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-13 14:18:40
FilePath: /Master_Thesis/src/experiment/pathway_ln.py
Description: building a pathway-level pipeline for classification
'''

import pandas as pd
from sklearn.pipeline import Pipeline

from pathway import PathwayTransformer
from preprocessing import make_z_score_scaler


def build_pathway_pipeline(
        preprocessor,
        membership: pd.DataFrame,
        classifier,
        *,
        standardize_pathway: bool = True,
)->Pipeline:
    """
    Build a pathway-level pipeline for classification.
    """
    steps = [
        ('protein_preprocessing', preprocessor),
        ('pathway', PathwayTransformer(membership)),
    ]
    if standardize_pathway:
        steps.append(('pathway_standardization', make_z_score_scaler()))
    steps.append(('classifier', classifier))

    return Pipeline(steps)
