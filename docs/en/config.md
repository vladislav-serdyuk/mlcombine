# YAML Configuration Reference

mlcombine uses a single YAML file to describe the entire pipeline.
The file is validated through Pydantic models with `extra="forbid"`.

## Minimal Example

```yaml
version: "1.0"
data:
  train_df: "train.csv"
  test_df: "test.csv"
  target_col: "target"
model:
  - provider: "sklearn"
    params:
      backbone: "random_forest"
```

Default provider is `sklearn`, default backbone is `random_forest`.

## Full Example (Titanic + CatBoost + URL dataset)

```yaml
version: "1.0"
plugins:
  - "./my_extensions.py"
data:
  train_df: "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
  test_df: "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
  target_col: "Survived"
  sep: ","
  force_prepare_dataset: false
force_types:
  Pclass: "category"
  Sex: "category"
  Embarked: "category"
model:
  - provider: "catboost"
    params:
      objective: "F1"
      iterations: 3000
      depth: 8
handling:
  numbers:
    impute: "median"
    scale: "robust"
  categories:
    encode: "onehot"
    smoothing: 10.0
trainer:
  output_dir: "./outputs"
  output_file: "./outputs/submission.csv"
environment:
  auto_install: true
```

---

## Config Sections

### `plugins` (optional)

List of Python files or modules to import before the pipeline runs.
Each file can register extensions via `@registry.*` decorators.

```yaml
plugins:
  - "./my_extensions.py"
  - "my_package.plugin_module"
```

Supported:
- Path to `.py` file (absolute or relative to CWD)
- Python module name (e.g. `my_package.my_plugin`)

---

### `data` (required)

| Field | Type | Default | Description |
|---|---|---|---|
| `train_df` | `str` | **required** | Path to CSV/TSV/Parquet/XLSX/XLS or **URL** |
| `test_df` | `str` | **required** | Path or URL to testing file |
| `target_col` | `str \| list[str] \| dict[str,str]` | **required** | Target column name (or list for multitask, or dict) |
| `sep` | `str` | `","` | CSV separator |
| `treatment_col` | `str \| None` | `None` | Treatment column (for uplift modeling) |
| `drop_columns` | `list[str]` | `[]` | Columns to drop before training |
| `force_prepare_dataset` | `bool` | `false` | Force re-download/extract |
| `task_type` | `str \| None` | `None` | Explicit task type (`classification`, `regression`, `uplift`, `multitask`) |

`train_df` and `test_df` support **HTTP/HTTPS URLs**. The file is downloaded automatically:

```yaml
data:
  train_df: "https://example.com/datasets/train.csv"
  test_df: "https://example.com/datasets/test.csv"
```

Archives `.zip`, `.tar`, `.tar.gz`, `.gz`, `.bz2`, `.xz` are supported — auto-extracted:

```yaml
data:
  train_df: "https://example.com/datasets/train.zip"
```

File format is determined by extension:
- `.csv` → CSV
- `.tsv` → TSV (tab)
- `.parquet` → Parquet
- `.xlsx` / `.xls` → Excel

---

### `force_types` (optional)

Override column types (bypass auto-detection).

```yaml
force_types:
  Pclass: "category"
  PassengerId: "number"
```

Allowed values: `number`, `category`, `datetime`, `image_path`, `sequence_token`, `text`, `unknown`, plus any custom types registered via `FeatureHandler`.

---

### `model` — model DAG (list)

`model` is now a list of nodes (`ModelNode`). Each node is one provider.
The last node in the list is the final model used for fit/predict.

```yaml
model:
  - provider: "sklearn"
    params:
      backbone: "random_forest"
      n_estimators: 300
```

#### Node fields

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | **required** | Provider (see table below) |
| `id` | `str \| None` | `None` | Identifier for cross-references from other nodes |
| `model` | `str \| None` | `None` | Reference to a single node (by `id`) — for meta-providers |
| `models` | `list[str]` | `[]` | References to multiple nodes (by `id`) |
| `params` | `dict` | `{}` | Provider parameters (backbone, layers, etc.) |

#### Providers

| Value | Description |
|---|---|
| `"sklearn"` | Scikit-learn (RandomForest, SVM, MLP, GradientBoosting, LogisticRegression) |
| `"catboost"` | CatBoost |
| `"lightgbm"` | LightGBM |
| `"pytorch"` | PyTorch (custom architecture via `params.layers`) |
| `"hybrid"` | Hybrid: images + text → late-fusion |
| `"tuner"` | Optuna hyperparameter tuning |
| `"fold_ensemble"` | K-Fold training with OOF predictions + target encoding |
| `"ensemble"` | Weighted averaging of multiple models |
| `"stacking"` | Stacking with meta-model on OOF |
| `"t_learner"` | Uplift T-learner |
| `"s_learner"` | Uplift S-learner |
| `"<custom>"` | Any custom provider registered via `registry.model_provider()` |

#### DAG config examples

**Ensemble (CatBoost + LightGBM):**

```yaml
model:
  - id: "cb"
    provider: "catboost"
    params:
      iterations: 3000
  - id: "lgb"
    provider: "lightgbm"
    params:
      num_iterations: 3000
  - provider: "ensemble"
    models: ["cb", "lgb"]
    params:
      weights: [0.6, 0.4]
```

**FoldEnsemble with target-encoding:**

```yaml
model:
  - id: "base"
    provider: "sklearn"
    params:
      backbone: "random_forest"
      n_estimators: 200
  - provider: "fold_ensemble"
    model: "base"
    params:
      n_folds: 5
      stratified: true
      target_encode_cols: ["cat_col1"]
      target_encode_smoothing: 10.0
      vote: "hard"
```

**Optuna tuning for CatBoost:**

```yaml
model:
  - provider: "tuner"
    params:
      n_trials: 50
      target_provider: "catboost"
      target_params:
        iterations: 3000
      search_space:
        depth: { type: "int", low: 4, high: 10 }
        learning_rate: { type: "float", low: 0.01, high: 0.3, log: true }
      evaluator: "cv"
      tune_metric: "f1"
      evaluator_params:
        n_folds: 3
        stratified: true
```

**PyTorch with custom layers:**

```yaml
model:
  - provider: "pytorch"
    params:
      objective: "F1"
      layers:
        - type: "linear"
          out_features: 256
        - type: "batch_norm1d"
        - type: "relu"
        - type: "dropout"
          p: 0.3
        - type: "linear"
          out_features: 128
        - type: "relu"
        - type: "linear"
      constants:
        d_model: 256
      epochs: 20
      batch_size: 64
      lr: 0.001
```

#### `objective` and `num_classes`

| objective | `num_classes` |
|---|---|
| `"MAPE"`, `"RMSE"` | Regression |
| `"F1"`, `"accuracy"`, `"precision"`, `"recall"` | Classification (2 classes unless specified) |
| `"AUC"`, `"logloss"` | Classification |

#### `layers` (PyTorch)

See the dedicated page: [PyTorch Layers](layers.md).

The `layers` section goes inside `params`:

```yaml
model:
  - provider: "pytorch"
    params:
      layers:
        - type: "linear"
          out_features: 128
        - type: "relu"
        - type: "dropout"
          p: 0.2
```

Each layer can also carry an `activation` field for post-activation:

```yaml
- type: "linear"
  out_features: 128
  activation: "gelu"
```

#### `constants`

`${var}` interpolation works in all layer and block fields:

```yaml
model:
  - provider: "pytorch"
    params:
      constants:
        d_model: 256
      layers:
        - type: "linear"
          out_features: "${d_model}"
```

#### `blocks` and `block_dirs`

```yaml
model:
  - provider: "pytorch"
    params:
      constants:
        d_model: 256
        nhead: 8
      layers:
        - type: "linear"
          out_features: "${d_model}"
        - type: "block"
          ref: "encoder_block"
          params:
            d_model: "${d_model}"
            nhead: "${nhead}"
        - type: "mean_pool"
        - type: "linear"
      block_dirs:
        - "./my_blocks"
```

---

### `handling`

Preprocessing: imputation, encoding, scaling.

#### `numbers`

| Field | Type | Default | Description |
|---|---|---|---|
| `impute` | `str` | `"median"` | Strategy: `mean`, `median`, `most_frequent`, `constant` |
| `scale` | `str` | `"robust"` | Scaling: `standard`, `robust`, `minmax`, `none` |

#### `categories`

| Field | Type | Default | Description |
|---|---|---|---|
| `encode` | `str` | `"onehot"` | Encoding: `ordinal`, `onehot`, `target`, `none` |
| `smoothing` | `float` | `10.0` | Smoothing for target encoding |

#### `columns`

Per-column overrides — unset fields inherit the global strategies above.
`none` skips the column entirely (no imputation/encoding/scaling).

| Field | Type | Description |
|---|---|---|
| `encode` | `str` | Encoding override for this column: `ordinal`, `onehot`, `none` |
| `impute` | `str` | Imputation override: `mean`, `median`, `most_frequent`, `constant`, `none` |
| `scale` | `str` | Scaling override: `standard`, `robust`, `minmax`, `none` |
| `fill_value` | `float` | Fill value for `impute: "constant"` |

```yaml
handling:
  numbers: { impute: "median", scale: "robust" }
  categories: { encode: "onehot" }
  columns:
    cat1: { encode: "ordinal" }       # one column as ordinal, rest one-hot
    cat2: { encode: "none" }          # keep raw (CatBoost handles it itself)
    num1: { scale: "none" }           # no scaling for this column
    num2: { impute: "constant", fill_value: 0 }
```

#### `sequence_token`

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"word2vec"` | Embedding model |
| `embedding_dim` | `int` | `64` | Embedding dimension |
| `aggregation` | `str` | `"mean"` | Strategy: `mean`, `max`, `min`, `sum`, `nunique`, `mode`, `datetime_stats` |

#### `images`

| Field | Type | Default | Description |
|---|---|---|---|
| `processing_mode` | `str` | `"standard"` | Mode: `standard`, `patches` |
| `patch_size` | `int` | `512` | Patch size |
| `augmentation` | `str` | `"none"` | Augmentation: `none`, `hard_noise`, `standard` |

---

### `step_config` (optional)

Additional pipeline steps: split, cross-encoder, pairwise similarity, text embedding.

#### `split`

```yaml
step_config:
  split:
    val_fraction: 0.2
    stratified: true
```

| Field | Type | Default | Description |
|---|---|---|---|
| `val_fraction` | `float` | `0.2` | Validation fraction |
| `stratified` | `bool` | `false` | Stratified split |

#### `cross_encoder`

Cross-encoder for text pair scoring (sentence-transformers).

```yaml
step_config:
  cross_encoder:
    model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    pairs:
      - ["left_title", "right_title"]
      - ["left_content", "right_content"]
    batch_size: 64
    max_length: 128
    predict_chunk: 500
    drop_source: false
    device: null
    cache_dir: "./cache/cross_encoder"
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | HuggingFace model |
| `pairs` | `list[list[str]]` | `[]` | Text column pairs |
| `batch_size` | `int` | `64` | Batch size |
| `max_length` | `int \| None` | `None` | Max token length |
| `predict_chunk` | `int` | `512` | Rows per predict call |
| `drop_source` | `bool` | `false` | Drop source text columns after scoring |
| `device` | `str \| None` | `None` | `"cuda"` / `"cpu"` / `null` (auto) |
| `cache_dir` | `str \| None` | `None` | Cache directory (SHA-256 keyed) |

#### `pairwise_similarity`

Bi-encoder cosine similarity for text pairs.

```yaml
step_config:
  pairwise_similarity:
    model_name: "all-MiniLM-L6-v2"
    pairs:
      - ["left_title", "right_title"]
    max_length: 64
    batch_size: 64
    predict_chunk: 512
    drop_source: false
    device: null
```

#### `text_embedding`

Text embeddings via sentence-transformers.

```yaml
step_config:
  text_embedding:
    model_name: "all-MiniLM-L6-v2"
    max_length: 128
    batch_size: 64
    cache_dir: "./cache/embeddings"
```

#### `save_predictions`

Prediction output file format. `id_col` controls which column is used as
the row id in the submission.

```yaml
step_config:
  save_predictions:
    id_col: "id"          # column used as prediction row id (default: none)
    target_col: "prediction"  # name of the prediction column
    sep: ","              # CSV separator (auto: "," / "\t" for .tsv)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `id_col` | `str \| None` | `None` | Column to use as id; when unset, index is used |
| `target_col` | `str` | `"prediction"` | Name of the prediction column |
| `sep` | `str \| None` | `None` | Separator (`","` or `"\t"` for `.tsv`) |

#### `evaluate`

Metrics on the holdout split (or OOF for meta-providers).

```yaml
step_config:
  evaluate:
    metrics: ["f1", "accuracy"]
```

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | `list[str] \| None` | `None` | Subset of `f1`, `f1_macro`, `accuracy`, `rmse`, `auc`, `logloss`, `mae`, `mape`; `None` = all applicable |

#### `feature_generation`

Count / frequency encoding for high-cardinality categorical columns.

```yaml
step_config:
  feature_generation:
    count_encode: true
    freq_encode: true
    max_unique: 1000
```

| Field | Type | Default | Description |
|---|---|---|---|
| `count_encode` | `bool` | `true` | Add count-encoded columns |
| `freq_encode` | `bool` | `true` | Add frequency-encoded columns |
| `max_unique` | `int` | `1000` | Only encode columns with ≤ this many unique values |

#### `column_length`

Adds `{col}_len` columns with string lengths.

```yaml
step_config:
  column_length:
    columns: ["title", "description"]
```

| Field | Type | Default | Description |
|---|---|---|---|
| `columns` | `list[str]` | `[]` | Columns to compute lengths for |

#### `diff_ratio`

Adds `{a}_{b}_diff_ratio` columns — `(a - b) / max(|a|, |b|)` per pair.

```yaml
step_config:
  diff_ratio:
    pairs: [["price_a", "price_b"]]
```

| Field | Type | Default | Description |
|---|---|---|---|
| `pairs` | `list[list[str]]` | `[]` | Numeric column pairs |

#### `same_value`

Adds `{a}_{b}_same` flags — 1 when both columns are equal.

```yaml
step_config:
  same_value:
    pairs: [["cat_a", "cat_b"]]
    drop_source: false
```

| Field | Type | Default | Description |
|---|---|---|---|
| `pairs` | `list[list[str]]` | `[]` | Column pairs to compare |
| `drop_source` | `bool` | `false` | Drop source columns after comparison |

#### `text_overlap`

Overlap features between text column pairs (char n-grams / token overlap).

```yaml
step_config:
  text_overlap:
    pairs: [["title_a", "title_b"]]
    char_ngram: 3
    drop_source: false
```

| Field | Type | Default | Description |
|---|---|---|---|
| `pairs` | `list[list[str]]` | `[]` | Text column pairs |
| `char_ngram` | `int` | `0` | Character n-gram size for overlap (`0` = tokens only) |
| `token_pattern` | `str` | `\w+` | Regex for tokenization |
| `drop_source` | `bool` | `false` | Drop source columns after scoring |

#### `reference_join`

Merge pair data against reference tables (e.g. item metadata), optionally
filtering training labels.

```yaml
step_config:
  reference_join:
    joins:
      - reference_path: "items.parquet"
        left_on: "leftItemId"
        suffix: "_left"
    keep_labels: ["no_relevant", "relevant"]
```

| Field | Type | Description |
|---|---|---|
| `joins` | `list[dict]` | Join configs: `reference_path`, `left_on`, `suffix` |
| `keep_labels` | `list[str] \| None` | When set, train rows with other labels are dropped |

Shorthand (single join, no `keep_labels`):

```yaml
step_config:
  reference_join:
    reference_path: "items.parquet"
    left_on: "leftItemId"
    suffix: "_left"
```

---

### `trainer`

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | `"./outputs"` | Artifact output directory |
| `output_file` | `str` | `"./outputs/submission.csv"` | Prediction output file |
| `fallback_on_error` | `bool` | `true` | Continue on error (placeholder) |

---

### `environment`

| Field | Type | Default | Description |
|---|---|---|---|
| `auto_install` | `bool` | `true` | Auto-install missing backends (`uv add` / `pip install`) |

---

### `evaluator` (optional)

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"holdout"` | `"holdout"` or `"cv"` |
| `params` | `dict` | `{}` | Parameters (`val_fraction`, `stratified`, `n_folds`) |
| `metrics` | `list[str]` | `[]` | Metrics: `f1`, `accuracy`, `rmse`, `auc`, `logloss`, `mae`, `mape` |

```yaml
evaluator:
  name: "cv"
  params:
    n_folds: 5
    stratified: true
  metrics:
    - "f1"
    - "accuracy"
```

---

## Enum Reference

### `FeatureType`

| Value | Description |
|---|---|
| `"number"` | Numeric column |
| `"category"` | Categorical column |
| `"datetime"` | Date/time |
| `"image_path"` | Image path |
| `"sequence_token"` | Tokenized sequence |
| `"text"` | Text |
| `"unknown"` | Unknown type |

### `TaskType`

| Value | Description |
|---|---|
| `"classification"` | Binary classification |
| `"regression"` | Regression |
| `"uplift"` | Uplift modeling |
| `"multitask"` | Multi-task (multiple targets) |

### `ModelObjective`

`"MAPE"`, `"F1"`, `"accuracy"`, `"precision"`, `"recall"`, `"AUC"`, `"logloss"`, `"RMSE"`

### `ImputeStrategy`

`"mean"`, `"median"`, `"most_frequent"`, `"constant"`

### `ScaleStrategy`

`"standard"`, `"robust"`, `"minmax"`, `"none"`

### `EncodeStrategy`

`"target"`, `"ordinal"`, `"onehot"`, `"none"`

### `AggregationStrategy`

`"mean"`, `"max"`, `"min"`, `"sum"`, `"nunique"`, `"mode"`, `"datetime_stats"`
