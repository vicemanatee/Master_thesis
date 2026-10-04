"""Small YAML/path utilities shared by otherwise independent modules."""

from pathlib import Path
from typing import Any

import yaml


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return config


def resolve_config_path(value: str | Path, *, owner: str | Path) -> Path:
    """Resolve a reference against the YAML containing it, not the shell cwd."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Invalid path reference in {owner}: {value!r}")
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (Path(owner).resolve().parent / path).resolve()
    )
