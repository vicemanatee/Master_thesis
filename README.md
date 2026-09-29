# Master Thesis

Breast-cancer classification using proteome and phosphoproteome data.

The current goal is a proteome benchmark based on DIA-NN's `pg_matrix`. The
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
- `sample_metadata`: extracted sample IDs and Pool flags;
- `feature_metadata`: protein or precursor annotations aligned with `X.columns`;
- `omics` and `source_path`: the input identity.

The proteome uses `Protein.Group` features from `pg_matrix`. The
phosphoproteome uses `Precursor.Id` features from `pr_matrix` and retains
`Modified.Sequence` and `Precursor.Charge`. Charge states are not aggregated by
the reader. Default input paths are stored in `configs/path.yaml`; either reader
also accepts a direct `path` argument.

Pool runs are excluded by default. Set `include_pool=True` only when Pool data
are intentionally needed. The reader preserves missing quantities and does not
attach clinical labels or perform statistical preprocessing.

## Preprocessing

The reusable preprocessing steps follow the scikit-learn `fit`/`transform`
interface:

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.preprocessing import MADFilter, MissingValueFilter, make_z_score_scaler

preprocessing = Pipeline([
    ("missing", MissingValueFilter(max_missing_fraction=0.30)),
    ("mad", MADFilter(quantile=0.75)),
    ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
    ("z_score", make_z_score_scaler()),
])

X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)
```

Rows are observations and columns are omics features. Put preprocessing and the
classifier inside the cross-validation pipeline so every fold learns filtering
and scaling only from its training data. Clinical labels and patient groups are
joined downstream by explicit IDs; repeated samples from one patient must stay
in the same cross-validation group.
