'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-10-02 16:05:16
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-02 16:05:46
FilePath: /Master_Thesis/tests/test_preprocessing_pipeline.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from preprocessing import (
    MADFilter,
    MissingValueFilter,
    make_z_score_scaler,
)


def test_preprocessing_pipeline():
    X_train = pd.DataFrame({
        "protein_A": [1.0, 2.0, 3.0, 4.0],
        "protein_B": [1.0, np.nan, 3.0, np.nan],
        "protein_C": [1.0, 3.0, 5.0, 7.0],
        "protein_D": [5.0, 5.0, 5.0, 5.0],
    })

    pipeline = Pipeline([
        (
            "missing",
            MissingValueFilter(
                max_missing_fraction=0.25
            ),
        ),
        (
            "mad",
            MADFilter(
                quantile=0.0
            ),
        ),
        (
            "impute",
            SimpleImputer(
                strategy="median"
            ).set_output(transform="pandas"),
        ),
        (
            "z_score",
            make_z_score_scaler(),
        ),
    ])

    result = pipeline.fit_transform(X_train)

    assert result.shape[0] == X_train.shape[0]

    assert not result.isna().any().any()

    assert np.isfinite(
        result.to_numpy()
    ).all()