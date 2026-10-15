"""Save pathway experiment tables and configuration separately from fitting."""

from datetime import UTC, datetime
from pathlib import Path

import yaml


def save_pathway_results(results: dict, output_root: str | Path, config: dict):
    """Create a separate directory for each run and return its path."""
    run_name = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    output_dir = Path(output_root).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    for name, table in results.items():
        if name in ['predictions', 'fold_assignments']:
            table.to_csv(output_dir / f'{name}.csv', index_label='run_id')
        else:
            table.to_csv(output_dir / f'{name}.csv', index=False)

    with (output_dir / 'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

    return output_dir
