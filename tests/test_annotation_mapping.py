from pathlib import Path

from pathway import map_annotation
from data import load_diann_matrix, load_hallmark

def test_mapping_annotation_real_data():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "path.yaml"
    )

    feature_matrix = load_diann_matrix("proteome", config_path=config_path)
    gene_sets = load_hallmark(config_path=config_path)

    result = map_annotation.mapping_annotation(
        feature_metadata=feature_matrix.feature_metadata,
        gene_sets=gene_sets,
    )

    print("Size of the result DataFrame:", result.shape)

    assert result.index.equals(feature_matrix.feature_metadata.index), (
        "feature id or order in the mapping result does not match the feature metadata"
    )

    assert result.columns.tolist() == list(gene_sets), (
        "pathway id or order in the mapping result does not match the feature metadata"
    )

    allowed_annotations = {'Unknown', 'None', 'Partial', 'All'}

    assert result.isin(allowed_annotations).all().all(), (
        "mapping result contains unexpected annotation values"
    )

    feature_id = "A0A7P0TA47"
    pathway = "HALLMARK_ADIPOGENESIS"

    assert feature_matrix.feature_metadata.loc[feature_id, "Genes"] == "DLAT"
    assert "DLAT" in gene_sets[pathway]["genes"]

    assert result.loc[feature_id, pathway] == "All", (
        f"Expected 'All' for feature {feature_id} and pathway {pathway}, but got {result.loc[feature_id, pathway]}"
    )

    feature_id = "O14920;O15111"
    pathway = "HALLMARK_ADIPOGENESIS"

    assert feature_matrix.feature_metadata.loc[feature_id, "Genes"] == "CHUK;IKBKB"
    assert "CHUK" in gene_sets[pathway]["genes"]
    assert "IKBKB" not in gene_sets[pathway]["genes"]

    assert result.loc[feature_id, pathway] == "Partial", (
        f"Expected 'Partial' for feature {feature_id} and pathway {pathway}, but got {result.loc[feature_id, pathway]}"
    )

    feature_id = "A0A0G2JRQ6"

    assert result.loc[feature_id].eq("Unknown").all(),(
        f"Expected all 'Unknown' for feature {feature_id}, but got {result.loc[feature_id]}"
    )

    feature_id = "A0A024R1R8;Q9Y2S6"
    pathway = "HALLMARK_ADIPOGENESIS"

    assert result.loc[feature_id, pathway] == "None", (
        f"Expected 'None' for feature {feature_id} and pathway {pathway},but got {result.loc[feature_id, pathway]}"
    )

if __name__ == "__main__":
    test_mapping_annotation_real_data()

