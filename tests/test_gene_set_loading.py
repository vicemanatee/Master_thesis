"""GMT parsing without abundance inputs or pathway matching."""

import pytest

from data import load_gmt


def test_gmt_reads_variable_length_sets_and_preserves_symbols(tmp_path):
    path = tmp_path / "sets.gmt.txt"
    path.write_text(
        "\ufeffSET_A\thttps://example.org/a\tGENE1\tGENE2\tGENE1\r\n"
        "\r\n"
        "SET_B\t\tMixedCaseGene\r\n",
        encoding="utf-8",
    )

    result = load_gmt(path)

    assert list(result) == ["SET_A", "SET_B"]
    assert result["SET_A"] == {
        "description": "https://example.org/a",
        "genes": {"GENE1", "GENE2"},
    }
    assert result["SET_B"] == {"description": "", "genes": {"MixedCaseGene"}}
    assert load_gmt(str(path)) == result


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("SET_A\tdescription\n", "expected name, description, and genes"),
        ("\tdescription\tGENE1\n", "name must not be empty"),
        ("SET_A\tdescription\tGENE1\t\n", "gene symbols must not be empty"),
        ("SET_A\tdescription\tGENE1\nSET_A\tdescription\tGENE2\n", "duplicate"),
    ],
)
def test_gmt_rejects_invalid_rows_with_source_location(tmp_path, content, message):
    path = tmp_path / "invalid.gmt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as error:
        load_gmt(path)

    assert str(path) in str(error.value)
    assert "line " in str(error.value)


def test_gmt_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.gmt"
    path.write_text("\n \n", encoding="utf-8")

    with pytest.raises(ValueError, match="no gene sets found"):
        load_gmt(path)
