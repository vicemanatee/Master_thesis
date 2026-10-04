# Master Thesis

Breast-cancer classification using proteome and phosphoproteome data.

The default proteome benchmark input is the annotated CycLoess/RRollup output. The
phosphoproteome reader is kept behind the same interface for later experiments.

## Data loading

```python
from src.data import load_phosphoproteome_matrix, load_proteome_matrix

proteome = load_proteome_matrix()
phosphoproteome = load_phosphoproteome_matrix()

X_proteome = proteome.X
X_phosphoproteome = phosphoproteome.X
```

Both readers return `LoadedMatrix` with:

- `X`: samples (runs) x features;
- `sample_metadata`: run/sample mapping and Pool flags, indexed exactly as `X`;
- `feature_metadata`: protein or precursor annotations aligned with `X.columns`;
- `omics`, `source_path`, `scale` and `processing_stage`: the input identity and scale;
- `sample_metadata_path`: the mapping file for rollup input.

The proteome uses `Protein` features from
`BC_Data/processed/proteome_rollup/cycloess_Rollup_Genes.tsv`. This key comes from
sorted `Protein.Ids` and differs from DIA-NN `Protein.Group`. `pep_count` and
`rollup_score` remain feature metadata, with numeric types. Abundances are already
log2 after CycLoess normalization and RRollup: do not log-transform them again.

The rollup reader joins `sample_metadata.tsv` by its explicit `sample_id`, checks
that matrix columns and mapping IDs match exactly, and keeps matrix column order.
`X.index` contains the original `file_name` with index name `run_id`. Sample metadata
retains `file_name`, `sample_id`, `ms_run_order`, `sample_prep_batch` and `is_pool`.
Corrected design IDs are authoritative: `230502_S000940` maps to `S000941`.
Duplicate mapping IDs or file names are errors. A custom matrix path uses the
sibling `sample_metadata.tsv`, unless `sample_metadata_path` is supplied.

For source inspection, reproducibility or processing-method comparisons, read the
original DIA-NN PG matrix explicitly:

```python
from src.data import load_matrix, load_raw_proteome_matrix

raw = load_raw_proteome_matrix()  # BC_Data/raw/.../230814_report.pg_matrix.tsv
# Equivalent: load_matrix("proteome", input_format="diann")
```

This raw entry point uses `Protein.Group`, returns linear quantities and extracts
uncorrected sample IDs from run names. `load_matrix("proteome")` defaults to rollup;
passing a custom path never implicitly changes the input format. The
phosphoproteome uses `Precursor.Id` features from `pr_matrix` and retains
`Modified.Sequence` and `Precursor.Charge`. Charge states are not aggregated by
the reader. Default input paths are stored in `configs/path.yaml`; either reader
also accepts a direct `path` argument.

Pool runs are excluded by default. Set `include_pool=True` only when Pool data
are intentionally needed. The reader preserves missing quantities and does not
attach clinical labels or perform statistical preprocessing.

The existing `BC_Data/processed/proteome` NPY bundle was generated from the earlier
DIA-NN PG input; it is separate from the rollup data and is not the default input.

## Preprocessing

The reusable preprocessing steps follow the scikit-learn `fit`/`transform`
interface:

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.preprocessing import (
    Log2Transformer,
    MADFilter,
    MissingValueFilter,
    make_z_score_scaler,
)

preprocessing = Pipeline(
    [
        ("log2", Log2Transformer(input_scale=proteome.scale)),
        ("missing", MissingValueFilter(max_missing_fraction=0.30)),
        ("mad", MADFilter(quantile=0.75)),
        ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("z_score", make_z_score_scaler()),
    ]
)

X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)
```

Always pass the loaded input's `scale` to the first step (use `raw.scale` or
`phosphoproteome.scale` for those inputs). Linear abundances are converted to
log2; the rollup's existing log2 values are preserved, including negative values.
Linear zeros become missing values before missing-rate filtering; negative linear
values raise an error. No pseudocount is added. Readers retain the original scale
and values; the pipeline output has subsequently been filtered, imputed and
standardized, so its final scale is z-score rather than log2.

Rows are observations and columns are omics features. Put preprocessing and the
classifier inside the cross-validation pipeline so every fold learns filtering
and scaling only from its training data. Clinical labels and patient groups are
joined downstream by explicit IDs; repeated samples from one patient must stay
in the same cross-validation group.

`src.preprocessing.make_pca(n_components=10)` creates an unfitted sklearn PCA
with pandas output and the same `fit`/`transform` interface. Place it **after
imputation and standardization**, inside the full classifier pipeline. It learns
axes only from training samples; it must not be fitted on the full cohort before
CV. PCA centers but does not standardize features itself. Component counts are
not silently reduced if a training fold has too few retained proteins or samples;
sklearn raises an error so the configured experiment remains explicit.

The current rollup was generated with Cyclic Loess normalization over the full
cohort. CV-local filtering, imputation and scaling do not undo that prior use of
all samples. Record this normalization scope when reporting baseline CV results;
strict evaluation on new patients requires separately reviewing which upstream
normalization/rollup parameters must be learned from training data.

## Proteome LN benchmark

Run from `Code/Master_Thesis`:

```bash
uv run python -m src.benchmark.proteome_benchmark
# Compare DIA-NN using the same patient partition:
uv run python -m src.benchmark.proteome_benchmark --input-format diann
# Run a different experiment list:
uv run python -m src.benchmark.proteome_benchmark --config configs/experiment.yaml
```

The command and options remain `--config`, `--paths`, `--input-format`, and
`--output-dir`. `--config` now selects an **experiment YAML**, whose references
load the separate module settings. `--paths` overrides the selected dataset's
path YAML; `--input-format` overrides its input format. The old
`configs/proteome_benchmark.yaml` is a reference-only compatibility entry pointing
to `experiment.yaml`; it no longer owns component parameters. Old monolithic
custom configs must be migrated to the module/reference schema.

### Configuration ownership

| File | Owns | Does not own |
| --- | --- | --- |
| `configs/path.yaml` | Matrix/clinical paths and clinical sheet | Preprocessing or models |
| `configs/data.yaml` | Named datasets, format, target/groups, cohort/ID rules; path-config reference | MAD/PCA or CV settings |
| `configs/preprocessing.yaml` | Step parameters, enabled flags, named profiles | Classifier selection or CV |
| `configs/benchmark.yaml` | Standalone classifiers, random seed, CV, scoring, output root | Data rules or preprocessing |
| `configs/experiment.yaml` | Dataset selection and named preprocessing/model combinations | Duplicated component parameters |

Every reference resolves relative to the YAML containing it, not the current
working directory. The data module's `paths_config` points to the path YAML.
`output_root` resolves relative to `benchmark.yaml`. By default, each run exports
to a new timestamped folder under `results/proteome_benchmark`; nonempty folders
are never overwritten.

Preprocessing `steps` define parameters once. Profiles override only the step
settings that differ: `standard` uses the base steps, `unscaled` disables z-score,
and `with_pca` enables PCA. Each step has an `enabled` flag; missingness, MAD,
imputation, scaling and PCA can be controlled independently. The fixed scientific
order is log2 -> missingness -> MAD -> imputation -> z-score -> PCA. Declared-scale
log2 harmonization remains required; already-log2 inputs pass through. Disabled
steps do not fit or transform data. Disabling imputation with NaNs still present,
or choosing too many PCA components, may cause sklearn to reject the experiment;
the runner does not silently substitute a different preprocessing method.

Each profile starts from `steps` and then applies only its overrides. Consequently,
`standard: {}` inherits every base flag. If base `pca.enabled` is changed to
`true`, PCA is also enabled for `standard` and `unscaled`; `with_pca` then becomes
an equivalent explicit override rather than the only PCA-enabled profile.

The available classifier types are `svm`, `random_forest` and
`logistic_regression`.
`logistic_l2` is a named LogisticRegression parameter set, using `l1_ratio=0.0`
(pure L2), `C=0.1`, `solver=lbfgs`, `max_iter=2000` and balanced class weights.
In sklearn >=1.8, `penalty='l2'` is deprecated; `l1_ratio=0.0` is the current API.
PCA defaults to 10 components with `svd_solver=full` and `whiten=false`.

The experiment list determines combinations, not Python branches on model names.
The active `experiment.yaml` currently selects PCA with SVM:

```yaml
- name: pca_svm
  preprocessing: with_pca
  model: svm
```

A run's `name` is its output identity, not its classifier type. It may be arbitrary
but must be unique. To omit an experiment, remove it from `runs`; to adjust PCA,
edit its preprocessing settings; to adjust regularization, edit the classifier
parameters. A profile can override an individual step parameter as well as its
enabled flag, without copying all shared settings.

### Code boundaries

- `src/data/dataset.py`: `load_dataset_config(path, dataset=...)` reads the data
  module independently; `prepare_dataset(data_config)` loads and aligns data;
  returns `ModelingDataset(X, y, groups, metadata, scale, provenance)`. Readers
  remain label-free; labels and groups are assembled downstream of reading.
- `src/preprocessing/pipeline.py`: `build_preprocessor(...)` creates an **unfitted**
  transformer pipeline; it knows neither data files nor classifier types.
  `inspect_preprocessor(...)` owns optional-stage diagnostics.
- `src/benchmark/models.py`: `make_classifier(...)` creates an unfitted standalone
  classifier; it does not choose PCA/scaling.
- `src/benchmark/runner.py`: `evaluate_models(dataset, pipelines, config=...)`
  evaluates supplied pipelines; it does not load data or select combinations.
- `src/benchmark/experiment.py`: resolves module references and assembles
  preprocessing + classifiers. `proteome_benchmark.py` is the CLI entry point.
- `src/benchmark/results.py`: result container and exports, preserving existing
  table names and columns.

Programmatic usage makes these boundaries explicit:

```python
from src.benchmark.experiment import load_experiment, make_model_pipelines
from src.benchmark.runner import evaluate_models
from src.data.dataset import prepare_dataset

config = load_experiment()
dataset = prepare_dataset(config["data"])
pipelines = make_model_pipelines(input_scale=dataset.scale, config=config)
result = evaluate_models(dataset, pipelines, config=config["benchmark"])
```

Components are separate in code, but preprocessing and the classifier are still
combined into a sklearn Pipeline **during training**. All learned filtering,
imputation, scaling and PCA are fitted only inside each CV training fold, not
on the whole cohort before CV.

### Data and evaluation contract

The default input is Rollup. Clinical labels are read from `SCANB.9206`.
`load_clinical_labels` verifies repeated GEX records before deduplication;
`align_clinical_labels` applies `S000940 -> S000941` and aligns by sample ID.
Missing labels, conflicts and contradictory `LN`/`LN.spec` values are errors.
Clinical fields remain outside X. `N0 -> 0`, `1to3/4toX -> 1`. Pool is excluded;
all labeled biological samples, including Group2, are retained. The five extra
exclusions from some author analyses are not applied. DIA-NN additionally uses
the fixed author pure-contaminant rule configured in `data.yaml`.

The cohort has 182 samples from 180 patients, with 94 LN-negative and 88
LN-positive samples. All experiments share shuffled `StratifiedGroupKFold`
splits, stratifying LN while keeping each Patient in one held-out fold. Samples
are sorted by corrected ID so Rollup/DIA-NN use the same partitions. This remains
fixed-parameter five-fold CV: no extra 70/30 split, independent test set,
hyperparameter search or nested CV. Metrics count samples, not unique patients.
RF's internal bootstrap is not bootstrap uncertainty estimation.

The refactor preserved the current settings, including MAD quantile 0.8,
SVM C=0.001 and RF min_samples_leaf=5. The chosen preprocessing profile controls
scaling/PCA, not classifier type. Median imputation remains useful because a
protein complete in training may be missing in a held-out sample. PCA maximizes
variance rather than LN separation and neither PCA nor L2 guarantees improved
generalization.

### Saved outputs

- `summary.csv`: held-out ROC-AUC, average precision, balanced accuracy, accuracy,
  F1, sensitivity and specificity by default; mean and sample standard deviation
  across folds. Fold standard deviation is not a confidence interval.
- `fold_metrics.csv`: train/held-out scores, sample/patient counts, feature counts,
  fit times, confusion matrices and PCA diagnostics. Disabled feature filters
  pass their input count through; unavailable MAD/PCA statistics are blank.
- `predictions.csv`: one held-out prediction per sample per experiment.
  SVM/LR use decision-function scores; RF uses LN-positive probabilities.
  `score_type` records the convention and `model` records the experiment name.
- `fold_assignments.csv`: identities and shared held-out folds.
- `selected_features.csv`: proteins selected before PCA and training MAD by
  experiment/fold. If MAD is disabled, `train_mad` is blank; these are protein IDs,
  not principal-component names.
- `run.json`: resolved component configs, experiment combinations, path settings,
  effective CLI overrides, config-file references, sources, versions and scale.
  Thus later edits to shared YAMLs do not erase a completed run's configuration.

Rollup uses full-cohort CycLoess/RRollup, and DIA-NN remains a fixed upstream
quantification output. Fold-local downstream preprocessing does not make those
upstream processes independent of held-out samples; their scope remains recorded
in `run.json`.
