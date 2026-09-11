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
