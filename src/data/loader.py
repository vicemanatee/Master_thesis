"""Load the DIA-NN matrices used by the classification experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "path.yaml"

_MATRIX_SPECS = {
    "proteome": {
        "feature_id": "Protein.Group",
        "annotations": (
            "Protein.Group",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "First.Protein.Description",
        ),
    },
    "phosphoproteome": {
        "feature_id": "Precursor.Id",
        "annotations": (
            "Protein.Group",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "First.Protein.Description",
            "Proteotypic",
            "Stripped.Sequence",
            "Modified.Sequence",
            "Precursor.Charge",
            "Precursor.Id",
        ),
    },
}


@dataclass
class LoadedMatrix:
    """A samples-by-features matrix with aligned annotations."""

    X: pd.DataFrame
    feature_metadata: pd.DataFrame
    sample_metadata: pd.DataFrame
    omics: str
    source_path: Path


def _default_path(omics: str, config_path: str | Path | None) -> Path:
    config_path = Path(config_path or CONFIG_PATH).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        value = config["matrices"][omics]
    except (KeyError, TypeError) as error:
        raise ValueError(f"No matrix path configured for {omics}") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid matrix path configured for {omics}")
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _sample_identity(run: str) -> tuple[str | None, bool]:
    basename = re.split(r"[\\/]", run)[-1]
    matches = re.findall(r"S\d{6}", basename)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous sample ID in run name: {run}")
    is_pool = re.search(r"(?i)(?:^|[-_])pool(?:[-_]|$)", basename) is not None
    return (matches[0] if matches else None), is_pool


def load_matrix(
    omics: str,
    path: str | Path | None = None,
    *,
    include_pool: bool = False,
    config_path: str | Path | None = None,
) -> LoadedMatrix:
    """Load a proteome PG matrix or phosphoproteome precursor matrix.

    Both formats are returned as samples x features. Phosphopeptide charge
    states remain separate features; this reader does not aggregate them.
    """
    if omics not in _MATRIX_SPECS:
        raise ValueError(f"omics must be one of {sorted(_MATRIX_SPECS)}")
    if not isinstance(include_pool, bool):
        raise TypeError("include_pool must be a boolean")

    source_path = (
        Path(path).resolve() if path is not None else _default_path(omics, config_path)
    )
    spec = _MATRIX_SPECS[omics]
    header = pd.read_csv(source_path, sep="\t", nrows=0, encoding="utf-8-sig").columns
    if not header.is_unique:
        raise ValueError(f"Duplicate columns in {source_path}")

    annotations = list(spec["annotations"])
    missing = set(annotations) - set(header)
    if missing:
        raise ValueError(f"Missing {omics} annotation columns: {sorted(missing)}")
    sample_columns = [column for column in header if column not in annotations]
    if not sample_columns:
        raise ValueError(f"No sample columns found in {source_path}")

    data = pd.read_csv(
        source_path,
        sep="\t",
        dtype={
            column: "string" for column in annotations if column != "Precursor.Charge"
        },
        keep_default_na=False,
        na_values={
            column: ["", "NA", "NaN", "nan", "N/A"] for column in sample_columns
        },
        encoding="utf-8-sig",
    )
    feature_id = spec["feature_id"]
    ids = data[feature_id]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{feature_id} values must be non-empty and unique")

    quantities = data.loc[:, sample_columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(quantities.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Matrix quantities contain infinite values")

    identities = [_sample_identity(run) for run in sample_columns]
    keep = [include_pool or not identity[1] for identity in identities]
    selected_runs = [run for run, selected in zip(sample_columns, keep) if selected]
    selected_identities = [
        identity for identity, selected in zip(identities, keep) if selected
    ]
    if not selected_runs:
        raise ValueError("No sample runs remain after Pool exclusion")

    features = pd.Index(ids, name=feature_id)
    X = quantities.loc[:, selected_runs].T
    X.index.name = "run_id"
    X.columns = features

    feature_metadata = data.loc[:, annotations].copy()
    feature_metadata.index = features
    sample_metadata = pd.DataFrame(
        {
            "sample_id": [identity[0] for identity in selected_identities],
            "is_pool": [identity[1] for identity in selected_identities],
        },
        index=X.index.copy(),
    )
    return LoadedMatrix(X, feature_metadata, sample_metadata, omics, source_path)


def load_proteome_matrix(
    path: str | Path | None = None,
    *,
    include_pool: bool = False,
    config_path: str | Path | None = None,
) -> LoadedMatrix:
    """Load the protein-group matrix used by the proteome benchmark."""
    return load_matrix(
        "proteome", path, include_pool=include_pool, config_path=config_path
    )


def load_phosphoproteome_matrix(
    path: str | Path | None = None,
    *,
    include_pool: bool = False,
    config_path: str | Path | None = None,
) -> LoadedMatrix:
    """Load precursors while preserving modification and charge metadata."""
    return load_matrix(
        "phosphoproteome", path, include_pool=include_pool, config_path=config_path
    )
