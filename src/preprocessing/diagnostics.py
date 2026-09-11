"""
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-09-11 21:46:03
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-09-11 21:53:26
FilePath: /Master_Thesis/src/preprocessing/diagnostics.py
Description: Diagnostic functions for data preprocessing.
"""

from pathlib import Path

import pandas as pd

from .config import load_preprocessing_config


def summarize_data(
    data: pd.DataFrame,
    *,
    name: str | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """Summarize a matrix; an omitted name is read from diagnostics.name in YAML."""
    if name is None:
        name = load_preprocessing_config(config_path)["diagnostics"]["name"]
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
    *,
    before_name: str | None = None,
    after_name: str | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare matrices using names from YAML unless explicitly overridden."""
    if before_name is None or after_name is None:
        config = load_preprocessing_config(config_path)["diagnostics"]
        before_name = config["before_name"] if before_name is None else before_name
        after_name = config["after_name"] if after_name is None else after_name
    before_summary = summarize_data(before, name=before_name)
    after_summary = summarize_data(after, name=after_name)

    return pd.DataFrame([before_summary, after_summary])
