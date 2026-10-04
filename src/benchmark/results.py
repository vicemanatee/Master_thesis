"""Result containers and exports independent of data loading/model assembly."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BenchmarkResult:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    selected_features: pd.DataFrame
    provenance: dict[str, Any]


def save_results(result: BenchmarkResult, output_dir: str | Path) -> Path:
    """Export the existing tables without replacing any prior run."""
    folder = Path(output_dir).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("fold_metrics", "summary", "selected_features"):
        getattr(result, name).to_csv(folder / f"{name}.csv", index=False)
    for name in ("predictions", "fold_assignments"):
        getattr(result, name).to_csv(folder / f"{name}.csv", index_label="run_id")
    with (folder / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(result.provenance, stream, indent=2, ensure_ascii=False)
    return folder
