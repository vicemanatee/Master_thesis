"""Persist validated run-by-feature matrices as reusable data bundles."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .loader import LoadedMatrix

BUNDLE_VERSION = 1
MATRIX_FILE = "X.npy"
SAMPLE_METADATA_FILE = "sample_metadata.tsv"
FEATURE_METADATA_FILE = "feature_metadata.tsv"
MANIFEST_FILE = "manifest.yaml"


@dataclass
class MatrixBundle:
    """A label-free run-by-feature matrix and its aligned metadata."""

    X: pd.DataFrame
    sample_metadata: pd.DataFrame
    feature_metadata: pd.DataFrame
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _validate_loaded_matrix(matrix: LoadedMatrix) -> np.ndarray:
    if not matrix.sample_metadata.index.equals(matrix.X.index):
        raise ValueError("Sample metadata index is not aligned with matrix rows")
    if not matrix.feature_metadata.index.equals(matrix.X.columns):
        raise ValueError("Feature metadata index is not aligned with matrix columns")
    if matrix.X.index.has_duplicates:
        raise ValueError("Run IDs must be unique before saving a matrix bundle")
    if matrix.X.columns.has_duplicates:
        raise ValueError("Feature IDs must be unique before saving a matrix bundle")
    if matrix.X.index.isna().any() or matrix.X.columns.isna().any():
        raise ValueError("Run and feature IDs must be non-missing")
    if "matrix_row" in matrix.sample_metadata or "run_id" in matrix.sample_metadata:
        raise ValueError("Sample metadata uses a reserved bundle field")
    if (
        "matrix_column" in matrix.feature_metadata
        or "feature_id" in matrix.feature_metadata
    ):
        raise ValueError("Feature metadata uses a reserved bundle field")
    array = matrix.X.to_numpy(copy=False)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Matrix bundle values must have a numeric dtype")
    if np.isinf(array).any():
        raise ValueError("Matrix bundle values contain infinite values")
    return array


def save_matrix_bundle(matrix: LoadedMatrix, output_dir: str | Path) -> Path:
    """Atomically save a label-free run-by-feature matrix bundle.

    The destination must not already exist. Source TSVs remain authoritative;
    this bundle stores only the numeric matrix, aligned metadata and provenance.
    """
    array = _validate_loaded_matrix(matrix)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Matrix bundle destination already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        np.save(temporary / MATRIX_FILE, array, allow_pickle=False)

        sample_metadata = matrix.sample_metadata.copy()
        sample_metadata.insert(0, "run_id", matrix.X.index.tolist())
        sample_metadata.insert(0, "matrix_row", range(len(sample_metadata)))
        sample_metadata.to_csv(
            temporary / SAMPLE_METADATA_FILE,
            sep="\t",
            index=False,
            encoding="utf-8",
            na_rep="",
        )

        feature_metadata = matrix.feature_metadata.copy()
        feature_metadata.insert(0, "feature_id", matrix.X.columns.tolist())
        feature_metadata.insert(0, "matrix_column", range(len(feature_metadata)))
        feature_metadata.to_csv(
            temporary / FEATURE_METADATA_FILE,
            sep="\t",
            index=False,
            encoding="utf-8",
            na_rep="",
        )

        source_path = matrix.plan.schema.path
        source: dict[str, Any] = {"path": str(source_path)}
        if source_path.is_file():
            source.update(_file_record(source_path))
        manifest = {
            "format": "run-feature-matrix-bundle",
            "format_version": BUNDLE_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "matrix": {
                "layout": "run_x_feature",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "row_index_name": matrix.X.index.name,
                "row_index_dtype": str(matrix.X.index.dtype),
                "column_index_name": matrix.X.columns.name,
                "column_index_dtype": str(matrix.X.columns.dtype),
            },
            "source": source,
            "schema": {
                "omics": matrix.plan.schema.omics,
                "file_type": matrix.plan.schema.file_type,
                "feature_id": matrix.plan.schema.feature_id,
            },
            "selection": {
                "feature_ids": list(matrix.plan.feature_ids)
                if matrix.plan.feature_ids is not None
                else None,
                "sample_ids": list(matrix.plan.sample_ids)
                if matrix.plan.sample_ids is not None
                else None,
                "include_pool": matrix.plan.include_pool,
            },
            "files": {
                name: _file_record(temporary / name)
                for name in (
                    MATRIX_FILE,
                    SAMPLE_METADATA_FILE,
                    FEATURE_METADATA_FILE,
                )
            },
            "labels_included": False,
            "preprocessing_applied": False,
        }
        with (temporary / MANIFEST_FILE).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                manifest,
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def _load_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / MANIFEST_FILE
    with path.open(encoding="utf-8") as stream:
        try:
            manifest = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid matrix bundle manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise TypeError("Matrix bundle manifest must be a mapping")
    if manifest.get("format") != "run-feature-matrix-bundle":
        raise ValueError("Unrecognised matrix bundle format")
    if manifest.get("format_version") != BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported matrix bundle version: {manifest.get('format_version')}"
        )
    return manifest


def _verify_bundle_files(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise TypeError("Matrix bundle manifest has no file records")
    for name in (MATRIX_FILE, SAMPLE_METADATA_FILE, FEATURE_METADATA_FILE):
        record = records.get(name)
        path = bundle_dir / name
        if not path.is_file() or not isinstance(record, dict):
            raise ValueError(f"Matrix bundle file is missing: {name}")
        if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get(
            "sha256"
        ):
            raise ValueError(f"Matrix bundle checksum mismatch: {name}")


def load_matrix_bundle(
    bundle_dir: str | Path,
    *,
    mmap_mode: str | None = "r",
    verify: bool = True,
) -> MatrixBundle:
    """Load a saved matrix bundle with optional memory mapping and verification."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest = _load_manifest(bundle_dir)
    if verify:
        _verify_bundle_files(bundle_dir, manifest)

    array = np.load(bundle_dir / MATRIX_FILE, mmap_mode=mmap_mode, allow_pickle=False)
    matrix_spec = manifest.get("matrix")
    if not isinstance(matrix_spec, dict):
        raise TypeError("Matrix bundle manifest has no matrix specification")
    expected_shape = tuple(matrix_spec.get("shape", ()))
    if array.shape != expected_shape:
        raise ValueError(
            f"Matrix shape {array.shape} does not match manifest {expected_shape}"
        )
    if str(array.dtype) != matrix_spec.get("dtype"):
        raise ValueError("Matrix dtype does not match the bundle manifest")

    sample_metadata = pd.read_csv(
        bundle_dir / SAMPLE_METADATA_FILE,
        sep="\t",
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8",
    )
    feature_metadata = pd.read_csv(
        bundle_dir / FEATURE_METADATA_FILE,
        sep="\t",
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8",
    )
    if sample_metadata["matrix_row"].tolist() != list(range(array.shape[0])):
        raise ValueError("Sample metadata row positions do not match the matrix")
    if feature_metadata["matrix_column"].tolist() != list(range(array.shape[1])):
        raise ValueError("Feature metadata column positions do not match the matrix")
    if sample_metadata["run_id"].isna().any() or sample_metadata["run_id"].duplicated().any():
        raise ValueError("Run IDs in sample metadata must be non-missing and unique")
    if (
        feature_metadata["feature_id"].isna().any()
        or feature_metadata["feature_id"].duplicated().any()
    ):
        raise ValueError("Feature IDs in feature metadata must be non-missing and unique")

    run_index = pd.Index(
        sample_metadata.pop("run_id"),
        dtype=matrix_spec.get("row_index_dtype"),
        name=matrix_spec.get("row_index_name"),
    )
    feature_index = pd.Index(
        feature_metadata.pop("feature_id"),
        dtype=matrix_spec.get("column_index_dtype"),
        name=matrix_spec.get("column_index_name"),
    )
    sample_metadata = sample_metadata.drop(columns="matrix_row")
    feature_metadata = feature_metadata.drop(columns="matrix_column")
    sample_metadata.index = run_index
    feature_metadata.index = feature_index
    X = pd.DataFrame(array, index=run_index, columns=feature_index, copy=False)
    return MatrixBundle(X, sample_metadata, feature_metadata, manifest)
