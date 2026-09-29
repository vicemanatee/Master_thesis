"""Three-layer readers for proteome and phosphoproteome DIA-NN tables."""

from .bundle import MatrixBundle, load_matrix_bundle, save_matrix_bundle
from .loader import (
    LoadedMatrix,
    get_columns,
    iter_from_plan,
    load_data,
    load_from_plan,
    load_matrix,
    load_yaml,
)
from .schema import DataSchema, get_feature_ids, inspect_schema
from .selection import ReadPlan, build_read_plan

__all__ = [
    "DataSchema",
    "LoadedMatrix",
    "MatrixBundle",
    "ReadPlan",
    "build_read_plan",
    "get_columns",
    "get_feature_ids",
    "inspect_schema",
    "iter_from_plan",
    "load_data",
    "load_from_plan",
    "load_matrix",
    "load_matrix_bundle",
    "load_yaml",
    "save_matrix_bundle",
]
