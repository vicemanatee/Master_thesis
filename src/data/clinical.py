"""Read clinical targets once and align them to runs by explicit sample IDs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

CLINICAL_FIELDS = ("Sample", "Patient", "LN", "LN.spec")
LN_MAPPING = {"N0": 0, "1to3": 1, "4toX": 1}


def load_clinical_labels(
    path: str | Path, *, sheet_name: str = "SCANB.9206"
) -> pd.DataFrame:
    """Validate and collapse repeated GEX rows into one record per Sample."""
    table = pd.read_excel(
        path, sheet_name=sheet_name, usecols=list(CLINICAL_FIELDS), dtype="string"
    )
    if table["Sample"].isna().any() or table["Sample"].str.strip().eq("").any():
        raise ValueError("Clinical table contains blank Sample IDs")
    counts = table.groupby("Sample", sort=False)[["Patient", "LN", "LN.spec"]].nunique(
        dropna=False
    )
    conflicts = counts.index[counts.gt(1).any(axis=1)].tolist()
    if conflicts:
        raise ValueError(f"Conflicting clinical records for samples: {conflicts}")
    return table.drop_duplicates("Sample").set_index("Sample")


def align_clinical_labels(
    sample_metadata: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    sample_id_corrections: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Align LN and Patient to run order, rejecting missing/ambiguous matches.

    Keeps original IDs for traceability. No clinical values enter X. Repeated
    runs of one sample remain separate observations with the same Patient.
    """
    if not sample_metadata.index.is_unique or not clinical.index.is_unique:
        raise ValueError("Run and clinical sample indices must be unique")
    if sample_metadata.empty or sample_metadata["is_pool"].any():
        raise ValueError("Label alignment requires biological runs without Pool")
    metadata = sample_metadata.copy()
    metadata["original_sample_id"] = metadata["sample_id"]
    metadata["sample_id"] = metadata["sample_id"].replace(
        dict(sample_id_corrections or {})
    )
    ids = metadata["sample_id"]
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError("Missing sample IDs in proteome metadata")
    missing = pd.Index(ids).difference(clinical.index).tolist()
    if missing:
        raise ValueError(f"Clinical labels not found for samples: {missing}")
    labels = clinical.reindex(ids.tolist()).copy()
    labels.index = metadata.index
    if (
        labels.isna().any().any()
        or labels.astype("string")
        .apply(lambda column: column.str.strip().eq(""))
        .any()
        .any()
    ):
        raise ValueError("Matched samples have missing clinical labels or Patient IDs")
    mapped = labels["LN.spec"].map(LN_MAPPING)
    if mapped.isna().any():
        raise ValueError("Matched samples have unsupported LN.spec values")
    recorded = pd.to_numeric(labels["LN"], errors="raise")
    if not recorded.eq(mapped).all():
        raise ValueError("Clinical LN contradicts LN.spec")
    labels["LN"] = mapped.astype("int64")
    return metadata.join(labels, validate="one_to_one")
