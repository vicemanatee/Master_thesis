"""Read preprocessing defaults from YAML without depending on the working directory."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/preprocessing.yaml"


def load_preprocessing_config(
    config_path: str | Path | None = None,
) -> dict[str, dict[str, float | str]]:
    """Load and validate a complete preprocessing configuration.

    The default path is relative to the project, not the current working
    directory. A custom path is used as supplied. No defaults are duplicated
    in Python, and the file is read on each call so edits take effect without
    restarting Python. Missing files, missing/unknown keys, and invalid values
    raise an error rather than silently using another configuration.
    """
    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    with path.open(encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid preprocessing YAML in {path}: {error}"
            ) from error

    required_keys = {
        "missing_value_filter": {"max_missing_fraction"},
        "mad_filter": {"quantile", "scale"},
        "diagnostics": {"name", "before_name", "after_name"},
    }
    if not isinstance(config, dict) or config.keys() != required_keys.keys():
        raise ValueError(
            f"{path}: expected exactly these sections: {list(required_keys)}"
        )
    for section, keys in required_keys.items():
        if not isinstance(config[section], dict) or config[section].keys() != keys:
            raise ValueError(f"{path}: {section} must contain exactly {sorted(keys)}")

    for section in ("missing_value_filter", "mad_filter"):
        for name, value in config[section].items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{path}: {section}.{name} must be a finite number")

    if not 0 <= config["missing_value_filter"]["max_missing_fraction"] <= 1:
        raise ValueError(f"{path}: max_missing_fraction must be between 0 and 1")
    if not 0 <= config["mad_filter"]["quantile"] < 1:
        raise ValueError(f"{path}: quantile must satisfy 0 <= quantile < 1")
    if config["mad_filter"]["scale"] <= 0:
        raise ValueError(f"{path}: scale must be positive")
    for name, value in config["diagnostics"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: diagnostics.{name} must be a non-empty string")

    return config
