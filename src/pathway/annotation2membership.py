'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-10-12 12:02:01
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-12 15:54:32
FilePath: /Master_Thesis/src/pathway/annotation2membership.py
Description: Convert annotation to membership

default mapping: Unknown and None to 0, Partial and All to 1
'''
import pandas as pd

DEFAULT_STATE_WEIGHT = {
    "Unknown": 0.0,
    "None": 0.0,
    "Partial": 1.0,
    "All": 1.0,
}

def annotation2membership_default(
        annotaion: pd.DataFrame,
        state_weight: dict[str, float]= DEFAULT_STATE_WEIGHT
):
    """
    Convert annotation DataFrame to membership DataFrame 
    """
    membership = pd.DataFrame(0.0,
                              index=annotaion.index,
                              columns=annotaion.columns)

    for feature_id, row in annotaion.iterrows():
        for set_name, state in row.items():
            membership.loc[feature_id, set_name] = state_weight[state]
    return membership
