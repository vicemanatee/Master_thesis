"""Read rollup modeling inputs and explicit DIA-NN source matrices."""

from __future__ import annotations

import csv
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

_ROLLUP_ANNOTATIONS = (
    "Protein",
    "Protein.Names",
    "First.Protein.Description",
    "Genes",
    "pep_count",
    "rollup_score",
)
_ROLLUP_SAMPLE_FIELDS = ("file_name", "sample_id", "ms_run_order", "sample_prep_batch")


@dataclass
class LoadedMatrix:
    """A samples-by-features matrix with aligned annotations."""

    X: pd.DataFrame
    feature_metadata: pd.DataFrame
    sample_metadata: pd.DataFrame
    omics: str
    source_path: Path
    scale: str = "linear"
    processing_stage: str = "diann"
    sample_metadata_path: Path | None = None


def _default_path(key: str, config_path: str | Path | None) -> Path:
    config_path = Path(config_path or CONFIG_PATH).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        value = config["matrices"][key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"No matrix path configured for {key}") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid matrix path configured for {key}")
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _header(path: Path) -> list[str]:
    """Check original TSV headers before pandas can rename duplicate fields."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream, delimiter="\t"), [])
    if not columns or any(not c.strip() for c in columns):
        raise ValueError(f"Empty file or blank columns in {path}")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate columns in {path}")
    return columns


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
    input_format: str | None = None,
    sample_metadata_path: str | Path | None = None,
) -> LoadedMatrix:
    """Dispatch to the DIA-NN (default) or proteome rollup reader.

    A direct path does not change the format. Labels and statistical operations
    belong downstream. Phosphopeptide charge states remain separate features.
    """
    if omics not in _MATRIX_SPECS:
        raise ValueError(f"omics must be one of {sorted(_MATRIX_SPECS)}")
    if not isinstance(include_pool, bool):
        raise TypeError("include_pool must be a boolean")

    if input_format is None:
        input_format = "diann"
    if input_format not in {
        #"rollup",
          "diann"}:
        raise ValueError("input_format must be rollup or diann")
    if input_format == "rollup":
        if omics != "proteome":
            raise ValueError("Rollup input is currently supported only for proteome")
        return load_proteome_rollup(
            path,
            sample_metadata_path=sample_metadata_path,
            include_pool=include_pool,
            config_path=config_path,
        )
    if sample_metadata_path is not None:
        raise ValueError("sample_metadata_path is supported only for rollup input")

    return load_diann_matrix(
        omics, path, include_pool=include_pool, config_path=config_path
    )


def load_diann_matrix(
    omics: str,
    path: str | Path | None = None,
    *,
    include_pool: bool = False,
    config_path: str | Path | None = None,
) -> LoadedMatrix:
    """Read a DIA-NN matrix and return aligned abundances and annotations.

    Both proteome and phosphoproteome inputs use linear abundances. Sample IDs
    are extracted from run names; phosphopeptide charge states remain separate.
    """
    if omics not in _MATRIX_SPECS:
        raise ValueError(f"omics must be one of {sorted(_MATRIX_SPECS)}")
    if not isinstance(include_pool, bool):
        raise TypeError("include_pool must be a boolean")

    source_path = (
        Path(path).resolve()
        if path is not None
        else _default_path(
            "proteome_raw" if omics == "proteome" else omics, config_path
        )
    )
    spec = _MATRIX_SPECS[omics]
    header = _header(source_path)

    annotations = list(spec["annotations"])
    missing = set(annotations) - set(header)
    if missing:
        raise ValueError(f"Missing {omics} annotation columns: {sorted(missing)}")
    sample_columns = [column for column in header if column not in annotations]
    if not sample_columns:
        raise ValueError(f"No sample columns found in {source_path}")
    unknown = [
        c
        for c in sample_columns
        if not re.fullmatch(r"(?i)(?:.*\.mzML(?:\.dia)?|S[0-9]{6})", c)
    ]
    if unknown:
        raise ValueError(f"Unrecognised DIA-NN sample columns: {unknown}")

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
    sample_metadata_path: str | Path | None = None,
) -> LoadedMatrix:
    """Load proteome data using the default DIA-NN format."""
    return load_matrix(
        "proteome",
        path,
        include_pool=include_pool,
        config_path=config_path,
        sample_metadata_path=sample_metadata_path,
    )


def load_proteome_rollup(
    path: str | Path | None = None,
    *,
    sample_metadata_path: str | Path | None = None,
    include_pool: bool = False,
    config_path: str | Path | None = None,
) -> LoadedMatrix:
    """Read annotated CycLoess/RRollup output and its authoritative run mapping.

    The metadata sample_id selects matrix columns, including corrected IDs.
    X uses file_name as run_id and retains matrix column order. Pool IDs stay
    in metadata when requested. Abundances are already log2; no transformation
    is performed by this reader. A custom matrix path uses sibling metadata
    unless sample_metadata_path is supplied explicitly.
    """
    if not isinstance(include_pool, bool):
        raise TypeError("include_pool must be a boolean")
    source_path = (
        Path(path).resolve()
        if path is not None
        else _default_path("proteome", config_path)
    )
    metadata_path = (
        Path(sample_metadata_path).resolve()
        if sample_metadata_path is not None
        else source_path.parent / "sample_metadata.tsv"
        if path is not None
        else _default_path("proteome_sample_metadata", config_path)
    )
    header = _header(source_path)
    missing = set(_ROLLUP_ANNOTATIONS) - set(header)
    if missing:
        raise ValueError(f"Missing rollup annotation columns: {sorted(missing)}")
    samples = [c for c in header if c not in _ROLLUP_ANNOTATIONS]
    if not samples:
        raise ValueError("Rollup matrix has no sample columns")

    metadata_header = _header(metadata_path)
    if set(metadata_header) != set(_ROLLUP_SAMPLE_FIELDS):
        raise ValueError(f"Rollup sample metadata must contain {_ROLLUP_SAMPLE_FIELDS}")
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    for field in _ROLLUP_SAMPLE_FIELDS:
        if metadata[field].str.strip().eq("").any():
            raise ValueError(f"Blank rollup sample metadata field: {field}")
    for field in ("sample_id", "file_name"):
        if metadata[field].duplicated().any():
            raise ValueError(f"Duplicate rollup sample metadata {field}")
    valid_ids = metadata["sample_id"].str.fullmatch(r"S[0-9]{6}|Pool_B[12]_[0-9]+")
    if not valid_ids.all():
        raise ValueError("Invalid rollup sample_id; expected Sxxxxxx or Pool_B1/B2_run")
    if set(samples) != set(metadata["sample_id"]):
        raise ValueError("Rollup matrix sample columns do not match sample metadata")
    # Align by explicit IDs, never by metadata row position or file-name regex.
    metadata = metadata.set_index("sample_id", drop=False).loc[samples].copy()
    metadata["is_pool"] = metadata["sample_id"].str.startswith("Pool_")

    data = pd.read_csv(
        source_path,
        sep="\t",
        dtype={c: "string" for c in _ROLLUP_ANNOTATIONS[:4]},
        keep_default_na=False,
        na_values={c: ["", "NA", "NaN", "nan", "N/A"] for c in samples},
        encoding="utf-8-sig",
    )
    ids = data["Protein"]
    if ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Protein values must be non-empty and unique")
    for field in ("pep_count", "rollup_score"):
        values = pd.to_numeric(data[field], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"Rollup {field} must contain finite numeric values")
        if field == "pep_count":
            if ((values < 2) | (values % 1 != 0)).any():
                raise ValueError("Rollup pep_count must contain integers >= 2")
            values = values.astype("int64")
        data[field] = values
    quantities = data.loc[:, samples].apply(pd.to_numeric, errors="raise")
    if np.isinf(quantities.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Rollup quantities contain infinite values")
    if not include_pool:
        metadata = metadata.loc[~metadata["is_pool"]].copy()
    if metadata.empty:
        raise ValueError("No sample runs remain after Pool exclusion")

    features = pd.Index(ids, name="Protein")
    X = quantities.loc[:, metadata["sample_id"].tolist()].T.copy()
    X.index = pd.Index(metadata["file_name"].tolist(), name="run_id")
    X.columns = features
    metadata.index = X.index.copy()
    feature_metadata = data.loc[:, list(_ROLLUP_ANNOTATIONS)].copy()
    feature_metadata.index = features
    return LoadedMatrix(
        X,
        feature_metadata,
        metadata,
        "proteome",
        source_path,
        scale="log2",
        processing_stage="rollup",
        sample_metadata_path=metadata_path,
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
