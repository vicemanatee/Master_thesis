"""Layer 2: resolve configuration into an explicit, header-validated read plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .loader import CONFIG_DIR, load_yaml
from .schema import DataSchema, get_schema_spec, inspect_schema, sample_identity


@dataclass(frozen=True)
class ReadPlan:
    schema: DataSchema
    columns: tuple[str, ...]
    sample_columns: tuple[str, ...]
    feature_ids: tuple[str, ...] | None
    sample_ids: tuple[str, ...] | None
    include_pool: bool
    chunksize: int


def _selection_list(value: object, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(v, str) or not v.strip() for v in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must be null or a non-empty list of unique strings")
    return tuple(value)


def build_read_plan(
    omics: str,
    *,
    file_type: str | None = None,
    source: str | None = None,
    path: str | Path | None = None,
    config_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    paths_path: str | Path | None = None,
) -> ReadPlan:
    """Combine dataset choices, file locations and an omics-specific schema.

    file_type/source/path arguments override their configured defaults. Other
    selection choices live in dataset.yaml. A supplied path bypasses path.yaml.
    Relative paths inside path.yaml resolve against that configuration's folder.
    """
    config = load_yaml(config_path or CONFIG_DIR / "dataset.yaml")
    try:
        selection = config["datasets"][omics]
    except (KeyError, TypeError) as error:
        raise ValueError(f"No dataset configuration for {omics}") from error
    keys = {
        "source",
        "file_type",
        "columns",
        "feature_ids",
        "sample_ids",
        "include_pool",
        "chunksize",
    }
    if not isinstance(selection, dict) or set(selection) != keys:
        raise ValueError(f"Dataset {omics} must contain exactly {sorted(keys)}")
    for key in ("source", "file_type"):
        if not isinstance(selection[key], str) or not selection[key].strip():
            raise ValueError(f"Dataset {key} must be a non-empty string")
    selected_columns = _selection_list(selection["columns"], "columns")
    feature_ids = _selection_list(selection["feature_ids"], "feature_ids")
    sample_ids = _selection_list(selection["sample_ids"], "sample_ids")
    include_pool = selection["include_pool"]
    chunksize = selection["chunksize"]
    if not isinstance(include_pool, bool):
        raise TypeError("include_pool must be a boolean")
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    file_type = file_type if file_type is not None else selection["file_type"]
    source = source if source is not None else selection["source"]
    spec = get_schema_spec(omics, file_type, schema_path=schema_path)
    if path is None:
        paths_path = Path(paths_path or CONFIG_DIR / "path.yaml").resolve()
        paths = load_yaml(paths_path)
        try:
            location = paths[source][omics][spec["path_key"]]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"No configured path for {source}/{omics}/{file_type}"
            ) from error
        if not isinstance(location, str) or not location.strip():
            raise ValueError("Configured dataset path must be a non-empty string")
        path = Path(location)
        if not path.is_absolute():
            path = paths_path.parent / path
    schema = inspect_schema(
        path, omics=omics, file_type=file_type, schema_path=schema_path
    )
    if feature_ids is not None and schema.feature_id is None:
        raise ValueError(f"{file_type} does not support biological feature selection")
    if (
        schema.layout != "matrix"
        and schema.run_column is None
        and (sample_ids is not None or not include_pool)
    ):
        raise ValueError(f"{file_type} has no sample axis for sample/Pool selection")

    # For matrices, columns selects annotations; sample_ids selects quantities.
    available = (
        schema.annotation_columns if schema.layout == "matrix" else schema.columns
    )
    requested = selected_columns if selected_columns is not None else available
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(
            f"Requested fields unavailable in {omics}/{file_type}: {sorted(missing)}"
        )
    # Preserve requested/source field order, appending any omitted identity fields.
    columns = list(dict.fromkeys([*requested, *schema.identity_columns]))
    samples = []
    matched_ids = set()
    for column in schema.sample_columns:
        sample_id, pool = sample_identity(column, schema)
        if (not include_pool and pool) or (
            sample_ids is not None and sample_id not in sample_ids
        ):
            continue
        samples.append(column)
        matched_ids.add(sample_id)
    if schema.layout == "matrix":
        if sample_ids is not None and (missing_ids := set(sample_ids) - matched_ids):
            raise ValueError(
                f"Sample IDs absent from selected columns: {sorted(missing_ids)}"
            )
        if not samples:
            raise ValueError("No sample columns selected")
        columns.extend(samples)
    return ReadPlan(
        schema,
        tuple(columns),
        tuple(samples),
        feature_ids,
        sample_ids,
        include_pool,
        chunksize,
    )
