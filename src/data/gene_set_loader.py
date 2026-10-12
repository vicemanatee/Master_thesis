"""Read fixed gene-set annotations independently of omics abundance data."""

from pathlib import Path
from typing import TypedDict


class GeneSet(TypedDict):
    """Description and unique gene symbols for one named gene set."""

    description: str
    genes: set[str]


def load_gmt(path: str | Path) -> dict[str, GeneSet]:
    """Read a tab-delimited GMT file, including files with a .txt suffix.

    Each nonblank line contains a unique set name, a description (possibly
    empty), and at least one gene symbol. Preserve file order and symbol case;
    deduplicate genes within each set. No identifier mapping or scoring occurs.
    Malformed rows, duplicate names, and empty files raise ValueError.
    """
    source = Path(path)
    gene_sets: dict[str, GeneSet] = {}
    with source.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            location = f"{source}: line {line_number}"
            if len(fields) < 3:
                raise ValueError(f"{location}: expected name, description, and genes")
            name = fields[0].strip()
            description = fields[1].strip()
            genes = {gene.strip() for gene in fields[2:]}
            if not name:
                raise ValueError(f"{location}: gene-set name must not be empty")
            if "" in genes:
                raise ValueError(f"{location}: gene symbols must not be empty")
            if name in gene_sets:
                raise ValueError(f"{location}: duplicate gene-set name: {name}")
            gene_sets[name] = {"description": description, "genes": genes}
    if not gene_sets:
        raise ValueError(f"{source}: no gene sets found")
    return gene_sets
