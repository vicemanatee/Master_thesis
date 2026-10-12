'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-10-12 18:37:17
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-12 19:44:52
FilePath: /Master_Thesis/src/pathway/aggregation.py
Description: buildiing pathway-level membership matrix from feature-level membership matrix and feature-to-gene mapping
'''

import pandas as pd

def aggregate_feature_to_pathway(
        X: pd.DataFrame,    # sample * PG_after_filter
        membership: pd.DataFrame,   # PG * pathway
) -> pd.DataFrame:
    
    """
    Aggregate feature-level data to pathway-level data using a membership matrix.
    """

    pathway_data = pd.DataFrame(0.0, index = X.index, columns = membership.columns)

    aligned_membership = membership.loc[X.columns]

    pathway_data = X.dot(aligned_membership)
    weight_sum = aligned_membership.sum(axis=0)
    zero_weight_pathways = weight_sum.index[weight_sum.eq(0)].tolist()
    if zero_weight_pathways:
        raise ValueError(
            "Cannot aggregate pathways with zero total membership weight "
            f"after aligning to X features: {zero_weight_pathways}"
        )
    pathway_data = pathway_data.div(weight_sum, axis = 'columns')

    return pathway_data


