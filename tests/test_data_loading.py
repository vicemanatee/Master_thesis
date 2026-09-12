"""Reader tests use small synthetic TSVs; no experiments are changed or loaded."""

import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
import yaml

from data import (
    build_read_plan,
    get_columns,
    get_feature_ids,
    inspect_schema,
    iter_from_plan,
    load_data,
    load_from_plan,
    load_matrix,
    load_yaml,
)
from data.loader import CONFIG_DIR
from data.schema import get_schema_spec

PG_FIELDS = [
    "Protein.Group",
    "Protein.Ids",
    "Protein.Names",
    "Genes",
    "First.Protein.Description",
]
RUN1 = r"D:\raw\230502_S000001.mzML.dia"
RUN2 = r"D:\raw\230502_S000940.mzML.dia"
POOL = r"D:\raw\230502_Pool_B1.mzML.dia"


def write_tsv(path, columns, rows):
    stream = StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")
    return path


def configure(tmp_path, omics="proteome", **changes):
    config = load_yaml(CONFIG_DIR / "dataset.yaml")
    config["datasets"][omics].update(changes)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def pg_path(tmp_path):
    return write_tsv(
        tmp_path / "pg.tsv",
        [*PG_FIELDS, POOL, RUN2, RUN1],
        [
            ["P2", "P2", "P2_HUMAN", "NA", "description", "3", "", "0"],
            ["P1", "P1", "P1_HUMAN", "GENE1", "description", "4", "NA", "10"],
        ],
    )


def make_table(tmp_path, omics, kind, *, raw=False):
    spec = get_schema_spec(omics, kind)
    fields = spec["required_columns"] + (spec["optional_columns"] if raw else [])
    rows = []
    for sample, feature in [(RUN1, "PEPTIDE2"), (RUN2, "PEPTIDE2"), (POOL, "OTHER3")]:
        values = {field: "0.5" for field in fields}
        for field in spec["string_columns"]:
            if field in values:
                values[field] = "text"
        if spec["feature_id"]:
            values[spec["feature_id"]] = feature
        if spec["run_column"]:
            values[spec["run_column"]] = sample
        rows.append([values[field] for field in fields])
    return write_tsv(tmp_path / f"{omics}_{kind}.tsv", fields, rows)


def test_yaml_uses_requested_path(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("value: 7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_yaml(path) == {"value": 7}
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


@pytest.mark.parametrize("text", ["", "[]", "{}", "a: ["])
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(path)


@pytest.mark.parametrize("header", ["", "a\t\n", "a\ta\n"])
def test_bad_headers(tmp_path, header):
    path = tmp_path / "bad.tsv"
    path.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError):
        get_columns(path)


def test_existing_loader_interface_and_column_order(pg_path):
    result = load_data(pg_path, columns=["Genes", "Protein.Group"])
    assert result.columns.tolist() == ["Genes", "Protein.Group"]
    assert result["Genes"].tolist() == ["NA", "GENE1"]
    assert len(load_data(pg_path, nrows=1)) == 1
    with pytest.raises(ValueError, match="not found"):
        load_data(pg_path, columns=["absent"])


def test_matrix_output_alignment_and_preserved_values(pg_path):
    plan = build_read_plan("proteome", path=pg_path)
    table = load_from_plan(plan)
    assert table.columns.tolist() == get_columns(pg_path)
    result = load_matrix(plan)
    assert result.X.shape == (3, 2)
    assert result.X.columns.tolist() == ["P2", "P1"]
    assert result.X.index.tolist() == [POOL, RUN2, RUN1]
    assert result.X.loc[RUN1, "P2"] == 0
    assert pd.isna(result.X.loc[RUN2, "P2"])
    assert pd.isna(result.X.loc[RUN2, "P1"])
    assert result.feature_metadata.index.equals(result.X.columns)
    assert result.sample_metadata.index.equals(result.X.index)
    assert result.feature_metadata.loc["P2", "Genes"] == "NA"
    assert result.sample_metadata.loc[RUN2, "sample_id"] == "S000940"
    assert result.sample_metadata.loc[POOL, "is_pool"]
    assert pd.isna(result.sample_metadata.loc[POOL, "sample_id"])


def test_schema_is_field_based_not_positional(pg_path):
    table = load_data(pg_path)
    reordered = list(reversed(table.columns))
    table.to_csv(pg_path, sep="\t", index=False, columns=reordered)
    schema = inspect_schema(pg_path, omics="proteome", file_type="pg_matrix")
    assert schema.sample_columns == (RUN1, RUN2, POOL)
    assert set(schema.annotation_columns) == set(PG_FIELDS)


def test_configured_columns_sample_and_feature_selection(tmp_path, pg_path):
    config = configure(
        tmp_path,
        columns=["Genes"],
        feature_ids=["P1"],
        sample_ids=["S000001"],
        include_pool=False,
        chunksize=1,
    )
    plan = build_read_plan("proteome", path=pg_path, config_path=config)
    assert plan.columns == ("Genes", "Protein.Group", RUN1)
    result = load_matrix(plan)
    assert result.X.shape == (1, 1)
    assert result.X.loc[RUN1, "P1"] == 10


def test_pool_exclusion_is_explicit(tmp_path, pg_path):
    config = configure(tmp_path, include_pool=False)
    result = load_matrix(build_read_plan("proteome", path=pg_path, config_path=config))
    assert result.X.index.tolist() == [RUN2, RUN1]


@pytest.mark.parametrize(
    "settings",
    [
        {"feature_ids": []},
        {"sample_ids": "S000001"},
        {"columns": ["Genes", "Genes"]},
        {"include_pool": "false"},
        {"chunksize": True},
        {"chunksize": 0},
        {"file_type": None},
        {"typo": True},
    ],
)
def test_invalid_selection_config(tmp_path, pg_path, settings):
    config = configure(tmp_path, **settings)
    with pytest.raises((ValueError, TypeError)):
        build_read_plan("proteome", path=pg_path, config_path=config)


def test_unavailable_field_and_sample_fail_early(tmp_path, pg_path):
    for settings in ({"columns": ["PTM.Site.Confidence"]}, {"sample_ids": ["S999999"]}):
        config = configure(tmp_path, **settings)
        with pytest.raises(ValueError):
            build_read_plan("proteome", path=pg_path, config_path=config)


def test_missing_requested_feature_fails_after_scan(tmp_path, pg_path):
    config = configure(tmp_path, feature_ids=["P1", "NOT_PRESENT"], chunksize=1)
    plan = build_read_plan("proteome", path=pg_path, config_path=config)
    with pytest.raises(ValueError, match="NOT_PRESENT"):
        load_from_plan(plan)


def test_feature_catalogue_is_not_header_names(pg_path):
    schema = inspect_schema(pg_path, omics="proteome", file_type="pg_matrix")
    assert get_feature_ids(schema, chunksize=1) == ["P2", "P1"]


@pytest.mark.parametrize("omics", ["proteome", "phosphoproteome"])
@pytest.mark.parametrize("kind", ["report", "library", "stats"])
@pytest.mark.parametrize("raw", [False, True])
def test_table_schemas_and_chunked_loading(tmp_path, omics, kind, raw):
    path = make_table(tmp_path, omics, kind, raw=raw)
    config = configure(tmp_path, omics, chunksize=1)
    plan = build_read_plan(omics, path=path, file_type=kind, config_path=config)
    assert len(list(iter_from_plan(plan))) == 3
    result = load_from_plan(plan)
    assert result.columns.tolist() == get_columns(path)
    with pytest.raises(ValueError, match="not a quantitative matrix"):
        load_matrix(plan)
    if kind != "stats":
        assert get_feature_ids(plan.schema, chunksize=1) == ["PEPTIDE2", "OTHER3"]


def test_phospho_report_cannot_use_proteome_schema(tmp_path):
    path = make_table(tmp_path, "phosphoproteome", "report")
    with pytest.raises(ValueError, match="Unrecognised.*proteome/report"):
        inspect_schema(path, omics="proteome", file_type="report")
    plain = make_table(tmp_path, "proteome", "report")
    with pytest.raises(ValueError, match="PTM.Site.Confidence"):
        inspect_schema(plain, omics="phosphoproteome", file_type="report")


def test_phospho_precursor_preserves_modification_and_charge(tmp_path):
    fields = get_schema_spec("phosphoproteome", "pr_matrix")["required_columns"]
    rows = []
    for charge in [2, 3]:
        record = dict.fromkeys(fields, "text")
        record.update(
            {
                "Precursor.Id": f"AS(UniMod:21)K{charge}",
                "Modified.Sequence": "AS(UniMod:21)K",
                "Precursor.Charge": charge,
            }
        )
        rows.append([record[field] for field in fields] + [10])
    path = write_tsv(tmp_path / "phospho_pr.tsv", [*fields, RUN1], rows)
    config = configure(tmp_path, "phosphoproteome", columns=["Genes"])
    result = load_matrix(
        build_read_plan("phosphoproteome", path=path, config_path=config)
    )
    assert result.X.shape == (1, 2)  # No implicit aggregation of charge states.
    assert "Modified.Sequence" in result.feature_metadata
    assert result.feature_metadata["Precursor.Charge"].tolist() == [2, 3]


def test_report_selection_uses_run_and_precursor(tmp_path):
    path = make_table(tmp_path, "phosphoproteome", "report")
    config = configure(
        tmp_path,
        "phosphoproteome",
        file_type="report",
        chunksize=1,
        columns=["Precursor.Normalised", "PTM.Site.Confidence"],
        sample_ids=["S000940"],
        feature_ids=["PEPTIDE2"],
        include_pool=False,
    )
    plan = build_read_plan("phosphoproteome", path=path, config_path=config)
    result = load_from_plan(plan)
    assert len(result) == 1
    assert result.iloc[0]["Run"] == RUN2
    assert "Modified.Sequence" in result


def test_run_matching_ignores_parent_directory_names(tmp_path):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PG_FIELDS, r"D:\S999999_Pool\S000001.mzML.dia"],
        [["P1", "P1", "name", "gene", "description", 10]],
    )
    result = load_matrix(build_read_plan("proteome", path=path))
    assert result.sample_metadata.iloc[0]["sample_id"] == "S000001"
    assert not result.sample_metadata.iloc[0]["is_pool"]


@pytest.mark.parametrize(
    "name",
    [
        "230511_Pool-B2.mzML.dia",
        "230502_Pool_B1.mzML.dia",
        "230717_phospho_POOL_B2.mzML.dia",
        "230512-Pool-B1.mzML.dia",
    ],
)
def test_pool_separator_variants(tmp_path, name):
    run = "D:\\raw\\" + name
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PG_FIELDS, run, RUN1],
        [["P1", "P1", "name", "gene", "description", 10, 20]],
    )
    included = load_matrix(build_read_plan("proteome", path=path))
    assert included.sample_metadata.loc[run, "is_pool"]
    config = configure(tmp_path, include_pool=False)
    excluded = load_matrix(build_read_plan("proteome", path=path, config_path=config))
    assert excluded.X.index.tolist() == [RUN1]


def test_repeated_sample_ids_keep_distinct_runs(tmp_path):
    runs = [RUN1, r"D:\raw\230503_S000001.mzML.dia"]
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PG_FIELDS, *runs],
        [["P1", "P1", "name", "gene", "description", 1, 2]],
    )
    result = load_matrix(build_read_plan("proteome", path=path))
    assert result.X.shape == (2, 1)
    assert result.sample_metadata["sample_id"].tolist() == ["S000001", "S000001"]


@pytest.mark.parametrize("identifier", ["P1", ""])
def test_bad_matrix_feature_ids_raise(tmp_path, identifier):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PG_FIELDS, RUN1],
        [
            ["P1", "P1", "name", "gene", "description", 1],
            [identifier, "P1", "name", "gene", "description", 2],
        ],
    )
    with pytest.raises(ValueError, match="non-empty and unique"):
        load_matrix(build_read_plan("proteome", path=path))


@pytest.mark.parametrize("quantity", ["invalid", "inf"])
def test_invalid_quantities_are_not_silently_missing(tmp_path, quantity):
    path = write_tsv(
        tmp_path / "pg.tsv",
        [*PG_FIELDS, RUN1],
        [["P1", "P1", "name", "gene", "description", quantity]],
    )
    with pytest.raises(ValueError):
        load_matrix(build_read_plan("proteome", path=path))


def test_unknown_matrix_column_is_not_a_sample(tmp_path, pg_path):
    data = load_data(pg_path)
    data["Unexpected.Score"] = 1
    data.to_csv(pg_path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Unexpected.Score"):
        inspect_schema(pg_path, omics="proteome", file_type="pg_matrix")


def test_stale_header_requires_replanning(pg_path):
    plan = build_read_plan("proteome", path=pg_path)
    data = load_data(pg_path)
    data.rename(columns={"Genes": "Changed"}).to_csv(pg_path, sep="\t", index=False)
    with pytest.raises(ValueError, match="header changed"):
        load_from_plan(plan)


def test_paths_are_relative_to_path_config(tmp_path, pg_path, monkeypatch):
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        yaml.safe_dump(
            {"prepared_data": {"proteome": {"protein group": pg_path.name}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(Path("/private/tmp"))
    plan = build_read_plan("proteome", paths_path=paths)
    assert plan.schema.path == pg_path.resolve()


@pytest.mark.parametrize("omics", ["proteome", "phosphoproteome"])
@pytest.mark.parametrize(
    "kind",
    [
        "pg_matrix",
        "gg_matrix",
        "pr_matrix",
        "unique_genes_matrix",
        "report",
        "stats",
        "library",
    ],
)
def test_every_configured_schema_has_both_source_paths(omics, kind):
    paths = load_yaml(CONFIG_DIR / "path.yaml")
    spec = get_schema_spec(omics, kind)
    for source in ("raw_data", "prepared_data"):
        assert isinstance(paths[source][omics][spec["path_key"]], str)


def test_sample_selection_rejected_for_library(tmp_path):
    path = make_table(tmp_path, "proteome", "library")
    config = configure(tmp_path, file_type="library", include_pool=False)
    with pytest.raises(ValueError, match="no sample axis"):
        build_read_plan("proteome", path=path, config_path=config)
