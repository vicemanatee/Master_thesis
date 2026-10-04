"""Proteome LN benchmark: resolve experiment YAML and run patient-grouped CV.

Run from the project root: uv run python -m src.benchmark.proteome_benchmark
All learned downstream preprocessing stays inside the fitted sklearn Pipeline.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from ..config import read_config
from .experiment import (
    BENCHMARK_CONFIG,
    EXPERIMENT_CONFIG,
    load_experiment,
    make_model_pipelines,
    run_benchmark,
)
from .results import BenchmarkResult, save_results
from .runner import evaluate_models, make_cv_splits

# Keep convenient imports at the previous entry point while implementation
# lives in independently usable modules. Programmatic evaluation now consumes
# a ModelingDataset and supplied pipelines rather than a monolithic config.
__all__ = [
    "BENCHMARK_CONFIG",
    "BenchmarkResult",
    "evaluate_models",
    "load_experiment",
    "make_cv_splits",
    "make_model_pipelines",
    "read_config",
    "run_benchmark",
    "save_results",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_CONFIG,
        help="Experiment YAML (module parameters are referenced, not duplicated)",
    )
    parser.add_argument(
        "--paths", type=Path, help="Override the selected dataset's path YAML"
    )
    parser.add_argument("--input-format", choices=("rollup", "diann"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    result = run_benchmark(
        config_path=args.config, paths_path=args.paths, input_format=args.input_format
    )
    output_dir = args.output_dir or (
        Path(result.provenance["config"]["benchmark"]["output_root"])
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    folder = save_results(result, output_dir)
    print(result.summary.to_string(index=False))
    print(f"Results saved to {folder}")


if __name__ == "__main__":
    main()
