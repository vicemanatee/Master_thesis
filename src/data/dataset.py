"""Assemble aligned modeling data without fitting any statistical transform."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import read_config, resolve_config_path
from .clinical import align_clinical_labels, load_clinical_labels
from .loader import load_matrix


@dataclass
class ModelingDataset:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    metadata: pd.DataFrame
    scale: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.empty or not self.X.index.is_unique:
            raise ValueError("X must be nonempty with unique run indices")
        if any(
            not self.X.index.equals(item.index)
            for item in (self.y, self.groups, self.metadata)
        ):
            raise ValueError(
                "X, y, groups and metadata must have identical run indices"
            )
        if self.y.isna().any() or self.groups.isna().any():
            raise ValueError("Dataset labels and patient groups cannot be missing")


def load_dataset_config(path: str | Path, *, dataset: str) -> dict[str, Any]:
    """Read this module's YAML independently of any benchmark/experiment."""
    definitions = read_config(path)["datasets"]
    if dataset not in definitions:
        raise ValueError(f"Unknown dataset: {dataset}")
    config = deepcopy(definitions[dataset])
    config["paths_config"] = str(
        resolve_config_path(config["paths_config"], owner=path)
    )
    return config


def prepare_dataset(config: dict[str, Any]) -> ModelingDataset:
    """Consume only resolved data settings, returning X/labels/groups separately.

    Source-specific contaminant removal is a fixed annotation rule, not a
    training-derived feature filter. Missingness/MAD/PCA belong downstream.
    """
    paths_path = Path(config["paths_config"]).resolve()
    paths = read_config(paths_path)
    clinical_path = resolve_config_path(paths["clinical"]["path"], owner=paths_path)
    if config["target"] != "LN":
        raise ValueError("The current clinical adapter supports target LN only")
    loaded = load_matrix(
        config["omics"], config_path=paths_path, input_format=config["input_format"]
    )
    X = loaded.X
    pattern = config.get("diann_contaminant_pattern")
    if loaded.processing_stage == "diann" and config["omics"] == "proteome" and pattern:
        contamination = loaded.feature_metadata["Protein.Group"].str.match(pattern)
        X = X.loc[:, ~contamination.to_numpy()]
    clinical = load_clinical_labels(
        clinical_path, sheet_name=paths["clinical"]["sheet_name"]
    )
    metadata = align_clinical_labels(
        loaded.sample_metadata,
        clinical,
        sample_id_corrections=config.get("sample_id_corrections", {}),
    ).sort_values("sample_id", kind="stable")
    X = X.loc[metadata.index]
    provenance = {
        "matrix_path": str(loaded.source_path),
        "sample_metadata_path": str(loaded.sample_metadata_path)
        if loaded.sample_metadata_path
        else None,
        "clinical_path": str(clinical_path),
        "clinical_sheet": paths["clinical"]["sheet_name"],
        "paths_path": str(paths_path),
        "input_scale": loaded.scale,
        "processing_stage": loaded.processing_stage,
        "n_samples": len(X),
        "n_patients": metadata[config["group_column"]].nunique(),
        "n_input_features": X.shape[1],
        "label_counts": metadata[config["target"]]
        .value_counts()
        .sort_index()
        .to_dict(),
        "upstream_scope": (
            "CycLoess and RRollup were computed across the full cohort including Pool. "
            "CV uses a fixed, jointly processed matrix, not a fully inductive upstream pipeline."
            if loaded.processing_stage == "rollup"
            else "DIA-NN quantification/normalization used the original experiment. "
            "Upstream mass-spectrometry processing was not rerun per fold."
        ),
    }
    return ModelingDataset(
        X,
        metadata[config["target"]].copy(),
        metadata[config["group_column"]].copy(),
        metadata,
        loaded.scale,
        provenance,
    )
