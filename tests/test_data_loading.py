import csv
from functools import partial
from io import StringIO

import pandas as pd
import pytest

from data import (
    load_diann_matrix,
    load_matrix,
    load_phosphoproteome_matrix,
    load_proteome_matrix,
    load_proteome_rollup,
)

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


@pytest.mark.parametrize(
    "reader",
    [load_proteome_matrix, partial(load_diann_matrix, "proteome")],
    ids=["wrapper", "diann"],
)
def test_load_proteome_matrix_returns_aligned_samples_by_features(tmp_path, reader):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PROTEOME_ANNOTATIONS, POOL, RUN],
        [
            ["P2", "P2", "name", "NA", "description", 3, 0],
            ["P1", "P1", "name", "GENE1", "description", 4, "NA"],
        ],
    )

    loaded = reader(path)

    assert loaded.X.shape == (1, 2)
    assert loaded.X.index.tolist() == [RUN]
    assert loaded.X.columns.tolist() == ["P2", "P1"]
    assert loaded.X.loc[RUN, "P2"] == 0
    assert pd.isna(loaded.X.loc[RUN, "P1"])
    assert loaded.sample_metadata.loc[RUN, "sample_id"] == "S000001"
    assert not loaded.sample_metadata.loc[RUN, "is_pool"]
    assert loaded.feature_metadata.loc["P2", "Genes"] == "NA"
    assert loaded.feature_metadata.index.equals(loaded.X.columns)

    with_pool = reader(path, include_pool=True)
    assert with_pool.X.index.tolist() == [POOL, RUN]
    assert with_pool.sample_metadata.loc[POOL, "is_pool"]


@pytest.mark.parametrize(
    "reader",
    [load_phosphoproteome_matrix, partial(load_diann_matrix, "phosphoproteome")],
    ids=["wrapper", "diann"],
)
def test_phosphoproteome_uses_same_interface_and_preserves_charge(tmp_path, reader):
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

    loaded = reader(path)

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
        load_diann_matrix("proteome", path)


@pytest.mark.parametrize("quantity", ["invalid", "inf"])
def test_invalid_quantities_raise(tmp_path, quantity):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PROTEOME_ANNOTATIONS, RUN],
        [["P1", "P1", "name", "gene", "description", quantity]],
    )
    with pytest.raises(ValueError):
        load_diann_matrix("proteome", path)


ROLLUP_ANNOTATIONS = [
    "Protein",
    "Protein.Names",
    "First.Protein.Description",
    "Genes",
    "pep_count",
    "rollup_score",
]
METADATA_FIELDS = ["file_name", "sample_id", "ms_run_order", "sample_prep_batch"]


@pytest.fixture
def rollup(tmp_path):
    matrix = write_tsv(
        tmp_path / "cycloess_Rollup_Genes.tsv",
        [*ROLLUP_ANNOTATIONS, "Pool_B1_1", "S000941", "S000015"],
        [
            ["P2;P3", "name", "description", "NA", 2, 0.1, 20, 21, "NA"],
            ["P1", "name", "description", "GENE1", 4, 0.2, 22, 0, 23],
        ],
    )
    # Deliberately shuffled, with a corrected ID unlike its original file name.
    metadata = write_tsv(
        tmp_path / "sample_metadata.tsv",
        METADATA_FIELDS,
        [
            ["230503_S000015", "S000015", "G2", "2"],
            ["230502_S000940", "S000941", "G1", "1"],
            ["230502_Pool_B1", "Pool_B1_1", "G1", "1"],
        ],
    )
    return matrix, metadata


def test_rollup_preserves_corrected_mapping_and_metadata(rollup):
    matrix, _ = rollup
    loaded = load_proteome_rollup(matrix)
    assert loaded.X.shape == (2, 2)
    assert loaded.X.index.tolist() == ["230502_S000940", "230503_S000015"]
    assert loaded.X.index.name == "run_id"
    assert loaded.X.columns.tolist() == ["P2;P3", "P1"]
    assert loaded.sample_metadata.index.equals(loaded.X.index)
    assert loaded.feature_metadata.index.equals(loaded.X.columns)
    assert loaded.sample_metadata["sample_id"].tolist() == ["S000941", "S000015"]
    assert loaded.sample_metadata["ms_run_order"].tolist() == ["G1", "G2"]
    assert loaded.sample_metadata["sample_prep_batch"].tolist() == ["1", "2"]
    assert loaded.X.loc["230502_S000940", "P1"] == 0
    assert loaded.X.loc["230502_S000940", "P2;P3"] == 21
    assert pd.isna(loaded.X.loc["230503_S000015", "P2;P3"])
    assert loaded.feature_metadata.loc["P2;P3", "Genes"] == "NA"
    assert loaded.feature_metadata["pep_count"].dtype == "int64"
    assert loaded.feature_metadata["rollup_score"].dtype.kind == "f"
    assert loaded.scale == "log2"
    assert loaded.processing_stage == "rollup"
    included = load_matrix("proteome", matrix, include_pool=True, input_format="rollup")
    assert included.X.shape == (3, 2)
    assert included.sample_metadata.iloc[0]["sample_id"] == "Pool_B1_1"
    assert included.sample_metadata.iloc[0]["is_pool"]


def test_rollup_config_paths_resolve_independently_of_cwd(
    tmp_path, rollup, monkeypatch
):
    matrix, metadata = rollup
    config = tmp_path / "path.yaml"
    config.write_text(
        f"matrices:\n  proteome: {matrix.name}\n"
        f"  proteome_sample_metadata: {metadata.name}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path.parent)
    loaded = load_proteome_rollup(config_path=config)
    assert loaded.source_path == matrix.resolve()
    assert loaded.sample_metadata_path == metadata.resolve()


@pytest.mark.parametrize(
    "change", ["duplicate_sample", "duplicate_run", "missing", "blank"]
)
def test_rollup_rejects_invalid_sample_mapping(rollup, change):
    matrix, metadata_path = rollup
    metadata = pd.read_csv(metadata_path, sep="\t", dtype="string")
    if change == "duplicate_sample":
        metadata.loc[0, "sample_id"] = metadata.loc[1, "sample_id"]
    elif change == "duplicate_run":
        metadata.loc[0, "file_name"] = metadata.loc[1, "file_name"]
    elif change == "missing":
        metadata = metadata.iloc[:-1]
    else:
        metadata.loc[0, "sample_id"] = ""
    metadata.to_csv(metadata_path, sep="\t", index=False)
    with pytest.raises(ValueError):
        load_proteome_rollup(matrix)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Protein", ""),
        ("Protein", "P1"),
        ("pep_count", 1.5),
        ("rollup_score", "inf"),
        ("S000941", "invalid"),
        ("S000941", "inf"),
    ],
)
def test_rollup_rejects_invalid_features_or_values(rollup, field, value):
    matrix, _ = rollup
    with matrix.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    rows[1][rows[0].index(field)] = value
    write_tsv(matrix, rows[0], rows[1:])
    with pytest.raises(ValueError):
        load_proteome_rollup(matrix)


def test_rollup_rejects_duplicate_header_before_pandas_renaming(rollup):
    matrix, _ = rollup
    with matrix.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    rows[0][-1] = rows[0][-2]
    write_tsv(matrix, rows[0], rows[1:])
    with pytest.raises(ValueError, match="Duplicate columns"):
        load_proteome_rollup(matrix)


def test_default_and_explicit_diann_keep_linear_scale(tmp_path):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PROTEOME_ANNOTATIONS, RUN],
        [["P1", "P1", "name", "gene", "description", 10]],
    )
    raw = load_matrix("proteome", path, input_format="diann")
    assert raw.scale == "linear"
    assert raw.processing_stage == "diann"
    for loaded in (load_matrix("proteome", path), load_proteome_matrix(path)):
        pd.testing.assert_frame_equal(loaded.X, raw.X)
        assert loaded.scale == "linear"
        assert loaded.processing_stage == "diann"
    with pytest.raises(ValueError, match="rollup annotation"):
        load_matrix("proteome", path, input_format="rollup")
