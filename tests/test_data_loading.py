import csv
from io import StringIO

import pandas as pd
import pytest

from data import load_phosphoproteome_matrix, load_proteome_matrix

RUN = r"D:\raw\230502_S000001.mzML.dia"
POOL = r"D:\raw\230502_Pool_B1.mzML.dia"
PROTEOME_ANNOTATIONS = [
    "Protein.Group",
    "Protein.Ids",
    "Protein.Names",
    "Genes",
    "First.Protein.Description",
]
PHOSPHO_ANNOTATIONS = [
    *PROTEOME_ANNOTATIONS,
    "Proteotypic",
    "Stripped.Sequence",
    "Modified.Sequence",
    "Precursor.Charge",
    "Precursor.Id",
]


def write_tsv(path, columns, rows):
    stream = StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")
    return path


def test_load_proteome_matrix_returns_aligned_samples_by_features(tmp_path):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PROTEOME_ANNOTATIONS, POOL, RUN],
        [
            ["P2", "P2", "name", "NA", "description", 3, 0],
            ["P1", "P1", "name", "GENE1", "description", 4, "NA"],
        ],
    )

    loaded = load_proteome_matrix(path)

    assert loaded.X.shape == (1, 2)
    assert loaded.X.index.tolist() == [RUN]
    assert loaded.X.columns.tolist() == ["P2", "P1"]
    assert loaded.X.loc[RUN, "P2"] == 0
    assert pd.isna(loaded.X.loc[RUN, "P1"])
    assert loaded.sample_metadata.loc[RUN, "sample_id"] == "S000001"
    assert not loaded.sample_metadata.loc[RUN, "is_pool"]
    assert loaded.feature_metadata.loc["P2", "Genes"] == "NA"
    assert loaded.feature_metadata.index.equals(loaded.X.columns)

    with_pool = load_proteome_matrix(path, include_pool=True)
    assert with_pool.X.index.tolist() == [POOL, RUN]
    assert with_pool.sample_metadata.loc[POOL, "is_pool"]


def test_phosphoproteome_uses_same_interface_and_preserves_charge(tmp_path):
    rows = []
    for charge in (2, 3):
        values = {
            "Protein.Group": "P1",
            "Protein.Ids": "P1",
            "Protein.Names": "name",
            "Genes": "GENE1",
            "First.Protein.Description": "description",
            "Proteotypic": 1,
            "Stripped.Sequence": "ASK",
            "Modified.Sequence": "AS(UniMod:21)K",
            "Precursor.Charge": charge,
            "Precursor.Id": f"AS(UniMod:21)K{charge}",
        }
        rows.append([values[column] for column in PHOSPHO_ANNOTATIONS] + [10])
    path = write_tsv(tmp_path / "pr.tsv", [*PHOSPHO_ANNOTATIONS, RUN], rows)

    loaded = load_phosphoproteome_matrix(path)

    assert loaded.X.shape == (1, 2)
    assert loaded.omics == "phosphoproteome"
    assert loaded.feature_metadata["Precursor.Charge"].tolist() == [2, 3]
    assert loaded.feature_metadata["Modified.Sequence"].nunique() == 1


@pytest.mark.parametrize("feature_id", ["P1", ""])
def test_duplicate_or_blank_feature_ids_raise(tmp_path, feature_id):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PROTEOME_ANNOTATIONS, RUN],
        [
            ["P1", "P1", "name", "gene", "description", 1],
            [feature_id, "P1", "name", "gene", "description", 2],
        ],
    )
    with pytest.raises(ValueError, match="non-empty and unique"):
        load_proteome_matrix(path)


@pytest.mark.parametrize("quantity", ["invalid", "inf"])
def test_invalid_quantities_raise(tmp_path, quantity):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PROTEOME_ANNOTATIONS, RUN],
        [["P1", "P1", "name", "gene", "description", quantity]],
    )
    with pytest.raises(ValueError):
        load_proteome_matrix(path)
