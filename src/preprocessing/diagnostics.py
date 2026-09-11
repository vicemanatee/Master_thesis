'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-09-11 21:46:03
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-09-11 21:53:26
FilePath: /Master_Thesis/src/preprocessing/diagnostics.py
Description: Diagnostic functions for data preprocessing.
'''
import numpy as np
import pandas as pd

def summarize_data(
    data: pd.DataFrame,
    *,
    name: str = "data",
)-> dict:
  return {
    "name": name,
    "n_samples": data.shape[0],
    "n_features": data.shape[1],
    "missing_fraction": data.isna().mean(axis=0).mean(),
    "n_missing_features": int(data.isna().sum().sum()),
  }

def compare_preprocessing(
  before: pd.DataFrame,
  after: pd.DataFrame,
)-> pd.DataFrame:
  before_summary = summarize_data(before, name="before")
  after_summary = summarize_data(after, name="after")

  return pd.DataFrame(
    [before_summary, after_summary]
  )