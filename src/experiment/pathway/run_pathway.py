'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-10-13 11:29:41
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-15 06:22:05
FilePath: /Master_Thesis/src/experiment/pathway/run_pathway.py
Description: pathway experiment runner
'''
import platform
from datetime import UTC, datetime
from pathlib import Path

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from config import read_config, resolve_config_path
from data import (
    LoadedMatrix,
    align_clinical_labels,
    load_clinical_labels,
    load_hallmark,
    load_matrix,
)
from pathway import annotation2membership_default, mapping_annotation
from preprocessing import build_preprocessor, resolve_preprocessing_profile

from .evaluation import run_cv, validate_cv_splits
from .pathway_ln import build_pathway_pipeline
from .results import save_pathway_results


def load_pathway_configs(
        config_path:Path|str,
    ):

    """
    Load pathway-related configurations from a YAML file.
    """

    config_path = Path(config_path).resolve()
    experiment_config = read_config(config_path)

    preprocessing_path = resolve_config_path(
        experiment_config['preprocessing']['config'],
        owner = config_path
    )

    data_path = resolve_config_path(
        experiment_config['data']['config'],
        owner = config_path
    )

    preprocessing_config = read_config(preprocessing_path)

    data_config = read_config(data_path)['datasets'][
        experiment_config['data']['dataset']
    ].copy()

    data_config['paths_config']= str(
        resolve_config_path(
            data_config['paths_config'],
            owner = data_path
        )
    )

    return experiment_config, preprocessing_config, data_config

def build_pathway_preprocessor(
        preprocessing_config: dict,
        experiment_config: dict,
        loaded: LoadedMatrix,
    ):
    """
    Build an unfitted protein preprocessor before pathway aggregation.
    """

    preprocessor = build_preprocessor(
        preprocessing_config,
        profile = experiment_config['preprocessing']['profile'],
        input_scale = loaded.scale,
    )
    if 'pca' in preprocessor.named_steps:
        raise ValueError('PCA must be disabled before pathway aggregation.')
    return preprocessor


def build_classifier(config: dict):
    """
    Build a classifier for pathway-level data.
    """

    if config['name'] not in ['logistic_regression']:
        raise ValueError(
            f"Unknown classifier: {config['name']}"
        )
    return LogisticRegression(**config['params'])


def make_cv_splitter(X, y, groups, cv_config: dict):
    """
    Create a cross-validation splitter for pathway-level data.
    """

    parameters = cv_config.copy()
    if not parameters.get('shuffle', False):
        parameters['random_state'] = None
    splitter = StratifiedGroupKFold(**parameters)
    splits = list(splitter.split(X, y, groups))
    validate_cv_splits(X, y, groups, splits)

    return splits

def filter_pathway_features(loaded: LoadedMatrix, data_config: dict):
    """Apply the configured fixed contaminant rule without fitting statistics."""
    X = loaded.X.copy()
    pattern = data_config.get('diann_contaminant_pattern')
    if loaded.processing_stage == 'diann' and loaded.omics == 'proteome' and pattern:
        contamination = loaded.feature_metadata['Protein.Group'].str.match(
            pattern, na=False
        )
        X = X.loc[:, ~contamination]
    if X.shape[1] == 0:
        raise ValueError('No features remain after fixed contaminant filtering.')
    return X


def main(config_path: str | Path | None = None):
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[3]
            / 'configs'
            / 'pathway.yaml'
        )
    config_path = Path(config_path).resolve()

    experiment_config, preprocessing_config, data_config = load_pathway_configs(config_path)
    loaded = load_matrix(
        data_config['omics'],
        config_path = data_config['paths_config'],
        input_format = data_config['input_format']
    )
    X = filter_pathway_features(loaded, data_config)

    if experiment_config['pathway']['gene_sets'] == 'hallmark':
        gene_sets = load_hallmark(config_path = data_config['paths_config'])
    else:
        raise ValueError(
            f"Unknown gene sets: {experiment_config['pathway']['gene_sets']}"
        )


    annotation = mapping_annotation(
        feature_metadata = loaded.feature_metadata.loc[X.columns],
        gene_sets = gene_sets
    )
    membership = annotation2membership_default(annotation)


    paths_config = read_config(data_config['paths_config'])
    clinical_path = resolve_config_path(
        paths_config['clinical']['path'],
        owner = data_config['paths_config']
        )
    clinical_labels = load_clinical_labels(
        clinical_path,
        sheet_name = paths_config['clinical']['sheet_name'],
    )
    aligned_labels = align_clinical_labels(
        sample_metadata = loaded.sample_metadata,
        clinical = clinical_labels,
        sample_id_corrections = data_config.get('sample_id_corrections', {}),
    )

    y = aligned_labels[data_config["target"]]
    groups = aligned_labels[data_config["group_column"]]

    preprocessor = build_pathway_preprocessor(
        preprocessing_config, experiment_config, loaded
    )
    classifier = build_classifier(experiment_config['classifier'])
    pipeline = build_pathway_pipeline(
        preprocessor = preprocessor,
        membership = membership,
        classifier = classifier,
        standardize_pathway = experiment_config['pathway']['standardize'],
    )
    splits = make_cv_splitter(X, y, groups, experiment_config['cv'])

    print(
        f'Pathway LN: {len(X)} runs, {groups.nunique()} patients, '
        f'{X.shape[1]} PGs after fixed filtering, {len(gene_sets)} pathways.',
        flush=True,
    )
    results = run_cv(
        pipeline, X, y, groups, splits,
        scoring = experiment_config['scoring'],
        metadata = aligned_labels,
    )

    effective_config = {
        'experiment': experiment_config,
        'preprocessing': preprocessing_config,
        'resolved_preprocessing': resolve_preprocessing_profile(
            preprocessing_config, experiment_config['preprocessing']['profile']
        ),
        'data': data_config,
        'paths': paths_config,
        'provenance': {
            'created_utc': datetime.now(UTC).isoformat(),
            'experiment_config_path': str(config_path),
            'matrix_path': str(loaded.source_path),
            'clinical_path': str(clinical_path),
            'gene_set_path': str(resolve_config_path(
                paths_config['gene_sets'][experiment_config['pathway']['gene_sets']],
                owner=data_config['paths_config'],
            )),
            'input_scale': loaded.scale,
            'processing_stage': loaded.processing_stage,
            'n_samples': len(X),
            'n_patients': int(groups.nunique()),
            'n_features_before_fixed_filter': loaded.X.shape[1],
            'n_features_after_fixed_filter': X.shape[1],
            'scoring_unit': 'run',
            'positive_class': 1,
            'python_version': platform.python_version(),
            'sklearn_version': sklearn.__version__,
            'upstream_scope': (
                'DIA-NN quantification/normalization used the original experiment. '
                'Upstream mass-spectrometry processing was not rerun per fold.'
                if loaded.processing_stage == 'diann'
                else 'CycLoess and RRollup were computed across the full cohort. '
                'Upstream processing was not rerun per fold.'
            ),
        },
    }
    output_root = resolve_config_path(
        experiment_config['output_root'], owner=config_path
    )
    output_dir = save_pathway_results(results, output_root, effective_config)
    print(results['summary'].to_string(index=False), flush=True)
    print(f'Results saved to: {output_dir}', flush=True)
    return results, output_dir


if __name__ == "__main__":
    main()
