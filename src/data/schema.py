"""Layer 1: inspect declared omics/file schemas without reading abundances."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .loader import CONFIG_DIR, get_columns, load_yaml


@dataclass(frozen=True)
class DataSchema:
    path: Path
    omics: str
    file_type: str
    layout: str
    columns: tuple[str, ...]
    annotation_columns: tuple[str, ...]
    sample_columns: tuple[str, ...]
    feature_id: str | None
    run_column: str | None
    identity_columns: tuple[str, ...]
    string_columns: tuple[str, ...]
    sample_id_pattern: str
    pool_pattern: str


def get_schema_spec(
    omics: str,
    file_type: str,
    *,
    schema_path: str | Path | None = None,
) -> dict:
    """Resolve a separately declared (omics, file_type) schema from YAML."""
    config = load_yaml(schema_path or CONFIG_DIR / "data_schema.yaml")
    try:
        spec = config["schemas"][omics][file_type]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown schema: {omics}/{file_type}") from error
    keys = {
        "layout",
        "path_key",
        "feature_id",
        "run_column",
        "identity_columns",
        "required_columns",
        "optional_columns",
        "string_columns",
    }
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValueError(
            f"Invalid schema keys for {omics}/{file_type}; expected {sorted(keys)}"
        )
    if spec["layout"] not in {"matrix", "table"}:
        raise ValueError("Schema layout must be matrix or table")
    for key in (
        "identity_columns",
        "required_columns",
        "optional_columns",
        "string_columns",
    ):
        value = spec[key]
        if (
            not isinstance(value, list)
            or any(not isinstance(c, str) or not c.strip() for c in value)
            or len(value) != len(set(value))
        ):
            raise ValueError(f"Schema {key} must be a list of unique field names")
    required = set(spec["required_columns"])
    known = required | set(spec["optional_columns"])
    if required & set(spec["optional_columns"]):
        raise ValueError("Required and optional schema fields overlap")
    if (
        not set(spec["identity_columns"]) <= required
        or not set(spec["string_columns"]) <= known
    ):
        raise ValueError("Identity/string fields must be declared in the schema")
    for key in ("feature_id", "run_column"):
        if spec[key] is not None and spec[key] not in spec["identity_columns"]:
            raise ValueError(f"Schema {key} must be an identity column or null")
    if spec["layout"] == "matrix" and (
        spec["feature_id"] is None or spec["run_column"] is not None
    ):
        raise ValueError("Matrix schemas need a feature_id and no run_column")
    patterns = config.get("patterns", {})
    if not isinstance(patterns, dict):
        raise TypeError("Schema patterns must be a mapping")
    for key in ("sample_column", "sample_id", "pool"):
        if not isinstance(patterns.get(key), str):
            raise TypeError(f"Schema pattern must be a string: {key}")
        try:
            re.compile(patterns[key])
        except re.error as error:
            raise ValueError(f"Invalid schema pattern {key}: {error}") from error
    return {**spec, "patterns": patterns}


def inspect_schema(
    path: str | Path,
    *,
    omics: str,
    file_type: str,
    schema_path: str | Path | None = None,
) -> DataSchema:
    """Validate the actual header against the explicit omics/file type.

    Unknown fields are errors, not implicit sample columns. Matching physical
    matrix layouts cannot establish which assay generated a file: omics must
    be supplied correctly by the caller/path configuration.
    """
    spec = get_schema_spec(omics, file_type, schema_path=schema_path)
    path = Path(path).resolve()
    columns = get_columns(path)
    missing = set(spec["required_columns"]) - set(columns)
    if missing:
        raise ValueError(f"Missing fields for {omics}/{file_type}: {sorted(missing)}")
    known = set(spec["required_columns"]) | set(spec["optional_columns"])
    samples = [
        c
        for c in columns
        if c not in known
        and spec["layout"] == "matrix"
        and re.fullmatch(spec["patterns"]["sample_column"], c)
    ]
    unknown = set(columns) - known - set(samples)
    if unknown:
        raise ValueError(
            f"Unrecognised fields for {omics}/{file_type}: {sorted(unknown)}"
        )
    if spec["layout"] == "matrix" and not samples:
        raise ValueError(f"No sample quantity columns found in {path}")
    return DataSchema(
        path,
        omics,
        file_type,
        spec["layout"],
        tuple(columns),
        tuple(c for c in columns if c in known),
        tuple(samples),
        spec["feature_id"],
        spec["run_column"],
        tuple(spec["identity_columns"]),
        tuple(spec["string_columns"]),
        spec["patterns"]["sample_id"],
        spec["patterns"]["pool"],
    )


def sample_identity(run: str, schema: DataSchema) -> tuple[str | None, bool]:
    """Extract an uncorrected ID from a Windows/POSIX basename; flag Pool runs."""
    basename = re.split(r"[\\/]", run)[-1]
    is_pool = re.search(schema.pool_pattern, basename) is not None
    matches = list(re.finditer(schema.sample_id_pattern, basename))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous sample ID in run name: {run}")
    return (matches[0].group(0) if matches else None), is_pool


def get_feature_ids(schema: DataSchema, *, chunksize: int = 100_000) -> list[str]:
    """Scan only the feature-ID field, not the abundance matrix.

    Long reports/libraries repeat IDs; return unique IDs in first-seen order.
    This is an optional catalogue operation, not a prerequisite for loading.
    """
    if schema.feature_id is None:
        raise ValueError(f"{schema.file_type} has no biological feature-ID field")
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    seen: dict[str, None] = {}
    with pd.read_csv(
        schema.path,
        sep="\t",
        usecols=[schema.feature_id],
        dtype={schema.feature_id: "string"},
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8-sig",
    ) as reader:
        for chunk in reader:
            values = chunk[schema.feature_id]
            if values.isna().any() or values.str.strip().eq("").any():
                raise ValueError(f"Blank feature ID in {schema.path}")
            seen.update(dict.fromkeys(values))
    return list(seen)
