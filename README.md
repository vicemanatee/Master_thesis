<!--
 * @Author: Li Yangqing liyangqingbuaa@gmail.com
 * @Date: 2026-09-11 00:32:53
 * @LastEditors: Li Yangqing liyangqingbuaa@gmail.com
 * @LastEditTime: 2026-09-11 11:40:05
 * @FilePath: /Code/Master_Thesis/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Master_thesis
A project for my master thesis, which is about the research of classification for Breast Cancer by using proteomics and phosphoproteomics data.

## Data loading

The reader extends the existing `src/data/loader.py` with two layers:

- `schema.py`: `inspect_schema` reads the header and validates an explicitly
  selected **omics + file type**. `get_feature_ids` optionally scans just the
  biological-ID column; header fields are not model feature IDs.
- `selection.py`: `build_read_plan` resolves configuration and selects fields,
  sample columns and explicit feature IDs before loading abundances.
- `loader.py`: `load_from_plan` reads a source-oriented table; `iter_from_plan`
  streams selected records in chunks. The original `load_yaml`, `get_columns`
  and `load_data(path, columns=...)` interfaces remain available.

Configuration has separate responsibilities:

| File | Responsibility |
| --- | --- |
| `configs/path.yaml` | Raw and Key_col file locations; existing path keys are retained. |
| `configs/data_schema.yaml` | Separate Proteome/Phosphoproteome schemas, required/optional fields and run-name patterns. |
| `configs/dataset.yaml` | Default file type, selected columns, feature IDs, sample IDs, Pool inclusion and chunk size. |
| `configs/preprocessing.yaml` | Statistical preprocessing, not reading. |

Supported file types are `pg_matrix`, `gg_matrix`, `pr_matrix`,
`unique_genes_matrix`, `report`, `stats` and `library`. Matrix layouts overlap,
but assay-specific report/library PTM fields are explicitly validated. The raw
report/library schemas also allow the additional fields omitted from Key_col.
Unknown columns fail validation; they are never guessed to be sample quantities.
When matrix headers are identical, the header alone cannot prove assay origin:
the configured omics/path pairing remains the source of that information.

From a Python session in the repository root:

```python
from src.data import build_read_plan, get_feature_ids, load_from_plan, load_matrix

proteome_plan = build_read_plan("proteome")         # prepared_data / pg_matrix
phospho_plan = build_read_plan("phosphoproteome")   # prepared_data / pr_matrix

# Optional inspection without loading abundance values:
print(phospho_plan.schema.annotation_columns)
# feature_ids = get_feature_ids(phospho_plan.schema)  # scans the ID column

proteome = load_matrix(proteome_plan)
phospho = load_matrix(phospho_plan)
X_proteome = proteome.X
X_phospho = phospho.X
# For an unchanged orientation instead: table = load_from_plan(proteome_plan)
```

`load_matrix` is an explicit format conversion, returning `X` (runs x features),
`feature_metadata`, `sample_metadata` and the resolved `plan`. Feature annotations
are indexed exactly like `X.columns`; sample annotations exactly like `X.index`.
The index retains original run labels; `sample_metadata.sample_id` contains the
extracted Sxxxxxx identifier and `is_pool` flags mixed-sample runs. Repeated
injections remain distinct. Matrix features use `Protein.Group`, `Precursor.Id`
or `Genes` according to the schema; duplicate/blank IDs fail rather than being
silently aggregated. Precursor modification sequence and charge remain available.
Report/stats/library tables cannot be passed to `load_matrix`.

In `dataset.yaml`, `columns` selects matrix annotations (sample quantities are
added separately) or fields of a report/stats/library. Necessary identity fields
are always included. `feature_ids` filters exact biological IDs in **rows**;
`sample_ids` filters Sxxxxxx IDs, retaining every matching run in source order.
`null` means all, whereas an empty list is an error. Explicit argument overrides
for `file_type`, `source` or `path` leave the other dataset selection settings
active, so ensure those settings apply to the selected table.

For example, set `include_pool: false`, `sample_ids: ["S000001"]` or a chosen
`feature_ids` list under the appropriate omics section to restrict an experiment.
Non-matrix sample selections apply to the report's `Run` or stats' `File.Name`;
libraries have no sample axis. PTM confidence thresholds and phosphorylation-only
filtering belong to subsequent preprocessing, not the generic reader.

For large main reports, use bounded chunks rather than materializing the table:

```python
from src.data import iter_from_plan

report_plan = build_read_plan("phosphoproteome", file_type="report")
for chunk in iter_from_plan(report_plan):
    # Consume the selected records here, e.g. for run-specific PTM QC.
    pass
```

Exhaust the iterator to validate that all explicitly requested feature/sample
IDs occur in the selected records. Filtering TSV rows still scans the file.
`load_from_plan` materializes all selected chunks and can require substantial RAM.

Defaults preserve all features and Pool runs. No loader imputes, normalizes,
applies MAD/FDR/PTM thresholds, aggregates charges, or corrects S000940 to S000941.
Zero quantities remain zero. Matrix numeric missing tokens become missing values,
while text identifiers such as `NA` are preserved. Clinical-label matching,
documented sample-ID corrections and cross-omics alignment must be explicit
downstream steps; never join datasets by row position.

Default config locations are repository-relative and independent of cwd. Supply
`config_path`, `schema_path` and `paths_path` to use other configurations; relative
data paths *inside path.yaml* resolve against that YAML file's directory. A direct
`path` argument resolves against the caller's cwd. No reader writes experiment data.

## Preprocessing configuration

`configs/preprocessing.yaml` is the single source of defaults for missing-value
filtering, MAD filtering/calculation, and diagnostic output names. Edit this file
to change experiment defaults. Explicit function or estimator arguments override
the corresponding YAML values; omitted arguments (`None`) use the configuration.

The default configuration path is resolved relative to this repository, so it
does not depend on the working directory. To use a different complete configuration,
pass `config_path="/path/to/preprocessing.yaml"`. Invalid or missing configuration
raises an error instead of silently reverting to hard-coded defaults.

From a Python session in the repository root:

```python
from sklearn.pipeline import Pipeline
from src.preprocessing import MADFilter, MissingValueFilter, compare_preprocessing

preprocessing = Pipeline([
    ("missing", MissingValueFilter()),
    ("mad", MADFilter()),
])
X_train_filtered = preprocessing.fit_transform(X_train)
X_test_filtered = preprocessing.transform(X_test)
print(compare_preprocessing(X_train, X_train_filtered))
```

Rows must be patients and columns must be omics features. The estimators read
configuration at `fit` time. `transform` reuses the learned feature selection,
even if the YAML file changes afterward. Resolved parameters are recorded in
`max_missing_fraction_`, `quantile_`, and `scale_` on the fitted estimators.

For cross-validation, place these steps **and the model** inside the pipeline
passed to the CV routine, so each training fold fits its own preprocessing.
Parameters such as `mad__quantile` remain available to `GridSearchCV`. Keep the
configuration unchanged during one experiment, and save its settings with the
experiment results for reproducibility.
