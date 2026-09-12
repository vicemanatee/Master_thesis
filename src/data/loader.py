from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

if TYPE_CHECKING:
    from .selection import ReadPlan

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: str | Path = CONFIG_DIR / "path.yaml") -> dict[str, Any]:
    """Load a YAML mapping. Relative paths supplied by callers use their cwd."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Configuration must be a non-empty mapping: {path}")
    return config


def get_columns(path: str | Path) -> list[str]:
    """Read the original header; reject duplicates before pandas renames them."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream, delimiter="\t"), [])
    if not columns or any(not column.strip() for column in columns):
        raise ValueError(f"Empty file or blank column name: {path}")
    duplicates = [name for name, count in Counter(columns).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate columns in {path}: {duplicates}")
    return columns


def load_data(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    dtype: Mapping[str, str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load TSV fields without transposition, filtering or statistical processing.

    Requested column order is preserved. Empty cells are missing; literal text
    such as the gene identifier 'NA' is not automatically changed to missing.
    Use a ReadPlan for schema-aware types and numeric missing-value tokens.
    """
    available = get_columns(path)
    selected = list(columns) if columns is not None else available
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("columns must be non-empty and contain no duplicates")
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Columns not found in {path}: {sorted(missing)}")
    data = pd.read_csv(
        path,
        sep="\t",
        usecols=selected,
        dtype=dtype,
        nrows=nrows,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8-sig",
    )
    return data.loc[:, selected]


def iter_from_plan(plan: ReadPlan) -> Iterator[pd.DataFrame]:
    """Read selected rows in bounded chunks, in source order.

    The iterator must be exhausted to validate that every requested feature
    and sample occurs in the selected records. No PTM/QC thresholds are applied.
    """
    from .schema import sample_identity

    schema = plan.schema
    if tuple(get_columns(schema.path)) != schema.columns:
        raise ValueError("File header changed after planning; rebuild the ReadPlan")
    dtypes = {c: "string" for c in schema.string_columns if c in plan.columns}
    dtypes.update({c: "float64" for c in plan.sample_columns})
    missing_values = {
        c: ["", "NA", "NaN", "nan", "N/A"] if c in plan.sample_columns else [""]
        for c in plan.columns
    }
    seen_features: set[str] = set()
    seen_samples: set[str] = set()
    with pd.read_csv(
        schema.path,
        sep="\t",
        usecols=list(plan.columns),
        dtype=dtypes,
        chunksize=plan.chunksize,
        keep_default_na=False,
        na_values=missing_values,
        encoding="utf-8-sig",
    ) as reader:
        for chunk in reader:
            if schema.run_column is not None:
                run = chunk[schema.run_column]
                # Resolve each distinct run once, not once per precursor row.
                identities = {
                    value: sample_identity(str(value), schema)
                    for value in run.dropna().unique()
                }
                if not plan.include_pool:
                    pools = run.map({k: v[1] for k, v in identities.items()})
                    chunk = chunk.loc[~pools.fillna(False).astype(bool)]
                if plan.sample_ids is not None:
                    ids = chunk[schema.run_column].map(
                        {k: v[0] for k, v in identities.items()}
                    )
                    chunk = chunk.loc[ids.isin(plan.sample_ids)]
            if plan.feature_ids is not None:
                chunk = chunk.loc[chunk[schema.feature_id].isin(plan.feature_ids)]
                seen_features.update(chunk[schema.feature_id].dropna())
            if plan.sample_ids is not None and schema.run_column is not None:
                seen_samples.update(
                    sample_identity(str(value), schema)[0]
                    for value in chunk[schema.run_column].dropna().unique()
                )
            if not chunk.empty:
                yield chunk.loc[:, list(plan.columns)]
    if plan.feature_ids is not None:
        missing = set(plan.feature_ids) - seen_features
        if missing:
            raise ValueError(
                f"Feature IDs absent from selected data: {sorted(missing)}"
            )
    if plan.sample_ids is not None and schema.run_column is not None:
        missing = set(plan.sample_ids) - seen_samples
        if missing:
            raise ValueError(f"Sample IDs absent from selected data: {sorted(missing)}")


def load_from_plan(plan: ReadPlan) -> pd.DataFrame:
    """Materialize a plan as a source-oriented table; large reports need RAM.

    Prefer iter_from_plan for report/library workflows that can consume chunks.
    """
    chunks = list(iter_from_plan(plan))
    if not chunks:
        raise ValueError(f"No records selected from {plan.schema.path}")
    return pd.concat(chunks, ignore_index=True)


@dataclass
class LoadedMatrix:
    """Numeric run-by-feature matrix with aligned, separate annotations."""

    X: pd.DataFrame
    feature_metadata: pd.DataFrame
    sample_metadata: pd.DataFrame
    plan: ReadPlan


def load_matrix(plan: ReadPlan) -> LoadedMatrix:
    """Explicitly convert a wide DIA-NN matrix into runs x features.

    Original run labels remain the index. Sample IDs are metadata, so repeated
    injections are never silently merged. No imputation or sample-ID correction.
    """
    from .schema import sample_identity

    schema = plan.schema
    if schema.layout != "matrix":
        raise ValueError(f"{schema.file_type} is not a quantitative matrix")
    data = load_from_plan(plan)
    ids = data[schema.feature_id]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(
            f"Matrix feature IDs must be non-empty and unique: {schema.feature_id}"
        )
    features = pd.Index(ids, name=schema.feature_id)
    X = data.loc[:, list(plan.sample_columns)].T.copy()
    X.columns = features
    X.index.name = "run_id"
    if X.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Matrix quantities contain infinite values")
    metadata_columns = [c for c in plan.columns if c not in plan.sample_columns]
    feature_metadata = data.loc[:, metadata_columns].copy()
    feature_metadata.index = features
    identities = [sample_identity(run, schema) for run in X.index]
    sample_metadata = pd.DataFrame(
        {
            "original_run": list(X.index),
            "sample_id": [identity[0] for identity in identities],
            "is_pool": [identity[1] for identity in identities],
        },
        index=X.index.copy(),
    )
    return LoadedMatrix(X, feature_metadata, sample_metadata, plan)
