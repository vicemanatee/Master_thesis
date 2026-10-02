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
    Log2Transformer, MADFilter, MissingValueFilter, make_z_score_scaler,
)

preprocessing = Pipeline([
    ("log2", Log2Transformer(input_scale=proteome.scale)),
    ("missing", MissingValueFilter(max_missing_fraction=0.30)),
    ("mad", MADFilter(quantile=0.75)),
    ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
    ("z_score", make_z_score_scaler()),
])

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

The current rollup was generated with Cyclic Loess normalization over the full
cohort. CV-local filtering, imputation and scaling do not undo that prior use of
all samples. Record this normalization scope when reporting baseline CV results;
strict evaluation on new patients requires separately reviewing which upstream
normalization/rollup parameters must be learned from training data.

## Proteome LN benchmark

Run from `Code/Master_Thesis`:

```bash
uv run python -m src.benchmark.proteome_benchmark
# Compare the DIA-NN PG input using the same patient partition:
uv run python -m src.benchmark.proteome_benchmark --input-format diann
```

`configs/proteome_benchmark.yaml` owns the fixed model parameters, preprocessing
thresholds, random seed and CV settings. `configs/path.yaml` owns matrix and
clinical file paths; its relative paths resolve against that YAML file. Optional
arguments: `--config`, `--paths`, `--input-format`, `--output-dir`. Each run creates
a new timestamped folder under `results/proteome_benchmark`; nonempty output
folders are never overwritten.

The default input is Rollup. Clinical labels are read once from `SCANB.9206` of
the public supplementary workbook. `data.clinical.load_clinical_labels` verifies
that repeated GEX records have identical Patient/LN fields before deduplication.
`align_clinical_labels` applies the explicit `S000940 -> S000941` correction and
aligns by sample ID in run order. Missing labels, conflicts and contradictory
`LN`/`LN.spec` values are errors. Clinical fields remain separate from X.
`N0 -> 0`, `1to3/4toX -> 1`. Pool is excluded; all labeled biological samples
are retained, including Group2. The five extra sample exclusions from some of
the author's downstream analyses are not applied in this baseline. DIA-NN input
additionally uses the author's pure-contaminant removal rule.

The current cohort has 182 samples from 180 Patient IDs, with 94 LN-negative
and 88 LN-positive samples after the ID correction. Both classifiers share the
same shuffled `StratifiedGroupKFold` splits, stratifying LN while keeping each
Patient in a single held-out fold. Samples are sorted by corrected ID before
splitting so the two input formats use the same partition. Five-fold CV leaves
approximately 20% out per fold; there is no additional 70/30 split, independent
test set or hyperparameter search. Metrics count samples, not unique patients.

Each training fold fits a sklearn Pipeline:

```text
declared scale -> log2 (Rollup passes through)
 -> missing-rate feature filter (default 0% missing in training)
 -> MAD above the 80th percentile (~top 20%)
 -> sklearn median imputation
 -> sklearn StandardScaler (SVM only)
 -> sklearn SVC / RandomForestClassifier
```

Validation reuses the fitted feature lists, medians and scaling. Imputation
remains necessary because a protein complete in training may be missing in the
held-out fold. MAD and missing-rate implementations reuse existing project
transformers; MAD uses SciPy, log2 uses NumPy. CV, classifiers, imputation,
scaling and metrics use scikit-learn. RF uses bootstrap internally; this is not
bootstrap resampling for estimating uncertainty. SVM uses a linear kernel;
RF uses 500 trees and `min_samples_leaf=2`. Parameters are fixed before evaluation;
selecting parameters from these scores would require separate nested CV.

Saved outputs:

- `summary.csv`: held-out ROC-AUC, average precision, balanced accuracy, accuracy,
  F1, sensitivity and specificity, averaged across folds with sample standard
  deviation. Fold standard deviation is not a confidence interval.
- `fold_metrics.csv`: train/held-out scores, sample/patient counts, retained
  feature counts, fit times and held-out confusion matrices.
- `predictions.csv`: one held-out prediction per sample per classifier; SVM
  scores are decision margins, RF scores are LN-positive probabilities.
- `fold_assignments.csv`: run/sample/Patient identities and shared held-out folds.
- `selected_features.csv`: selected protein IDs and training MAD by model/fold.
- `run.json`: resolved configuration, sources, versions, scale and upstream
  normalization scope. The current Rollup uses full-cohort CycLoess/RRollup;
  fold-local downstream preprocessing does not make that upstream processing
  independent of held-out samples. DIA-NN also remains a fixed upstream output.
