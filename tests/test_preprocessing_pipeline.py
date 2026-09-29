from pathlib import Path

import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from data import build_read_plan, load_matrix, load_yaml
from data.loader import CONFIG_DIR
from preprocessing import (
    MADFilter,
    MissingValueFilter,
    load_preprocessing_config,
    summarize_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "preprocessing_pipeline_test.csv"
OMICS = ("proteome", "phosphoproteome")


def test_real_preprocessing_pipeline(tmp_path: Path) -> None:
    """Run loading -> NaN filter -> MAD filter on both real omics matrices."""
    dataset_config = load_yaml(CONFIG_DIR / "dataset.yaml")
    for omics in OMICS:
        dataset_config["datasets"][omics]["include_pool"] = False

    dataset_config_path = tmp_path / "dataset_no_pool.yaml"
    dataset_config_path.write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
    )

    preprocessing_config = load_preprocessing_config()
    max_missing_fraction = preprocessing_config["missing_value_filter"][
        "max_missing_fraction"
    ]
    mad_quantile = preprocessing_config["mad_filter"]["quantile"]
    mad_scale = preprocessing_config["mad_filter"]["scale"]
    records: list[dict] = []

    for omics in OMICS:
        plan = build_read_plan(omics, config_path=dataset_config_path)
        loaded = load_matrix(plan)
        assert not loaded.sample_metadata["is_pool"].any()

        pipeline = Pipeline(
            [
                (
                    "nan_filter",
                    MissingValueFilter(
                        max_missing_fraction=max_missing_fraction
                    ),
                ),
                (
                    "mad_filter",
                    MADFilter(quantile=mad_quantile, scale=mad_scale),
                ),
            ]
        )
        after_mad = pipeline.fit_transform(loaded.X)
        after_nan = pipeline["nan_filter"].transform(loaded.X)
        assert (after_nan.isna().mean(axis=0) <= max_missing_fraction).all()

        stages = (
            ("loaded", loaded.X),
            ("after_nan_filter", after_nan),
            ("after_mad_filter", after_mad),
        )
        previous_n_features: int | None = None
        loaded_n_features = loaded.X.shape[1]

        for stage, data in stages:
            summary = summarize_data(data, name=stage)
            n_missing_values = summary.pop("n_missing_features")
            n_features = summary["n_features"]
            records.append(
                {
                    "test_name": "preprocessing pipeline test",
                    "omics": omics,
                    "stage": summary.pop("name"),
                    **summary,
                    "n_missing_values": n_missing_values,
                    "features_removed_from_previous": (
                        0
                        if previous_n_features is None
                        else previous_n_features - n_features
                    ),
                    "feature_retention_fraction": n_features / loaded_n_features,
                    "max_missing_fraction": max_missing_fraction,
                    "mad_quantile": mad_quantile,
                    "mad_scale": mad_scale,
                    "pool_included": False,
                    "source_file": str(plan.schema.path),
                }
            )
            previous_n_features = n_features

        assert loaded.X.shape[0] == after_nan.shape[0] == after_mad.shape[0]
        assert loaded.X.shape[1] >= after_nan.shape[1] >= after_mad.shape[1] > 0
        assert after_nan.columns.isin(loaded.X.columns).all()
        assert after_mad.columns.isin(after_nan.columns).all()

    results = pd.DataFrame.from_records(records)
    assert len(results) == len(OMICS) * 3
    assert results.groupby("omics")["stage"].apply(list).to_dict() == {
        omics: ["loaded", "after_nan_filter", "after_mad_filter"]
        for omics in OMICS
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)
    assert RESULT_PATH.is_file()
