from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import pandas as pd
import yaml

def load_yaml(path: str | Path = Path(__file__).resolve().parents[2] / "configs/path.yaml")->dict[str, Any]:
  '''load yaml configuration file'''

  path = Path(path)

  if not path.exists():
    raise FileNotFoundError(f"Configuration file not found: {path}")

  with open("r", encoding = 'utf-8') as f:
    config = yaml.safe_load(f)

  if config is None:
    raise ValueError(f"Configuration file is empty: {path}")

  return config


def get_columns(path: str | Path)->list [str]:
  '''Read only the column names without loading the full dataset. '''

  path = Path(path)

  if not path.exists():
    raise FileNotFoundError(f"Configuration file not found: {path}")

  return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()


def load_data(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
)->pd.DataFrame:
    '''Load a tabular dataset
    Parameters
    ----------
    path : str | Path
        The path to the dataset file.
    columns : Iterable[str] | None, optional
        The columns to load. If None, all columns are loaded.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    '''

    path = Path(path)

    usecols = list(columns) if columns is not None else None

    return pd.read_csv(
      path, 
      sep='\t',
      usecols=usecols,
    )

  