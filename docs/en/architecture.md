# mlcombine Architecture

## High-Level Flow

```
YAML Config → MLCombineConfig (Pydantic validation)
                      ↓
         PipelineEngine.from_config()
                      ↓
         PipelineContext() created
                      ↓
    ┌───── 10-step training pipeline ─────┐
    │ 1. PrepareDatasetStep   (download)  │
    │ 2. DataLoaderStep       (CSV→DF)    │
    │ 3. TypeDetectStep       (types +    │
    │      task detection)                │
    │ 4. PrepareEnvironmentStep (auto-install)│
    │ 5. ImputeStep           (fill NaN)  │
    │ 6. EncodeScaleStep      (encode +   │
    │      scale)                         │
    │ 7. CreateModelStep      (ModelFactory)│
    │ 8. WrapUpliftStep?      (TLearner/  │
    │      SLearner)                      │
    │ 9. ModelFitStep         (fit())     │
    │10. SaveArtifactsStep    (joblib)    │
    └─────────────────────────────────────┘
                      ↓
         Context.data + Context.artifacts
         Model saved to ./outputs/model.joblib
```

For inference, steps 4-10 are replaced by:
```
 4. LoadArtifactsStep     (load model)
 5. DropTargetColumnsStep (drop target)
 6. SavePredictionsStep   (predict → CSV)
```

---

## PipelineEngine

`PipelineEngine[ContextType]` — orchestrator managing a sequence of steps.

### Construction

```python
engine = PipelineEngine()
engine.add_step(MyStep())
engine.add_step_before("ImputeStep", MyOtherStep())
```

### From config

```python
engine = PipelineEngine.from_config(cfg)
```

This method:
1. Loads plugins from `cfg.plugins`
2. Creates a `ModelFactory` (with providers from `registry`)
3. Builds the 10-step training pipeline or 6-step inference pipeline

### Execution

```python
ctx = PipelineContext()
result = engine.run_all(ctx)   # returns mutated ctx
```

Each step is logged with timing.

### Step manipulation

```python
engine.add_step(step)                  # append
engine.add_step_before("name", step)   # insert before
engine.add_step_after("name", step)    # insert after
```

Step names are auto-deduplicated with `_N` suffix.

---

## Pipeline Steps

### 1. `PrepareDatasetStep`

Downloads the dataset from URL (if `train_df` starts with `http`),
extracts archives (`.zip`, `.tar`, `.gz`, `.bz2`, `.xz`).

**Input:** `cfg.data.train_df`, `cfg.data.test_df`
**Output:** `ctx.data.train_df_path`, `ctx.data.test_df_path`

### 2. `DataLoaderStep`

Loads CSV/TSV/Parquet/XLSX/XLS into `pd.DataFrame`. Format detected by file extension.

**Input:** file paths
**Output:** `ctx.data.train_df`, `ctx.data.test_df`

### 3. `TypeDetectStep`

Auto-detects column types (`FeatureType`) and task type (`TaskType`).

**Logic for numeric columns:**
- int with `unique_ratio < 5%` → `CATEGORY`
- otherwise → `NUMBER`

**Logic for string/object columns:**
1. Try parse as datetime → `DATETIME`
2. Check image extensions → `IMAGE_PATH`
3. Find hex sequences → `SEQUENCE_TOKEN`
4. Average length > 50 chars → `TEXT`
5. Custom `FeatureHandler` from `registry` → custom type
6. `unique_ratio < 0.5` → `CATEGORY`
7. Otherwise → `TEXT`

**Task type:**
- Multiple target columns → `MULTITASK`
- Int target with `unique_count <= 20` → `CLASSIFICATION`
- Float target → `REGRESSION`
- Object target → `CLASSIFICATION`

### 4. `PrepareEnvironmentStep`

Auto-installs missing Python packages via `uv add` (falls back to `pip install`).

**Backend → package mapping:**
- `catboost` → `catboost`
- `lightgbm` → `lightgbm`
- `pytorch` / `hybrid` → `torch`

Reloads provider modules via `importlib.reload()` after install.

### 5. `ImputeStep`

`SimpleImputer` on numeric columns (by `FeatureType.NUMBER`).

### 6. `EncodeScaleStep`

Two stages:
1. **Encoding:** `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` per categorical column
2. **Scaling:** `StandardScaler` / `RobustScaler` / `MinMaxScaler` on numeric columns

`EncodeStrategy.TARGET` raises `NotImplementedError` (needs OOF isolation).

### 7. `CreateModelStep`

Creates model via `ModelFactory.create_model()`.

Auto `input_size`: if `model.layers` is set but `input_size` is omitted,
counts features (excluding target and treatment).

### 8. `WrapUpliftStep`

Wraps model in `TLearner` or `SLearner` if `model.uplift_method != None`.

### 9. `ModelFitStep`

Extracts X, y, treatment from `train_df`, calls `model.fit()`.

### 10. `SaveArtifactsStep`

Saves model via `joblib.dump` to `{output_dir}/model.joblib`.

---

## ModelFactory and Providers

```
ModelFactory
  │
  ├── SklearnProvider    (RandomForest, SVM, MLP, etc.)
  ├── CatBoostProvider   (CatBoostClassifier / CatBoostRegressor)
  ├── LightGBMProvider   (LGBMClassifier / LGBMRegressor)
  ├── PyTorchProvider    (dynamic YAML architecture)
  └── HybridModelProvider(images + text → late-fusion)
```

### ModelFactory

```python
factory = ModelFactory()
factory.register_provider("my_provider", MyProvider())
factory.register_custom_model("my_model", pretrained_model)
model = factory.create_model(config, task_type, num_classes, input_size)
```

Resolution order:
1. `config.provider == "auto"` → `"sklearn"`
2. Check `custom_models`
3. Look up provider in `_providers` (built-in + registry)
4. `provider.create(backbone, objective, task_type, num_classes, **kwargs)`

Providers from `registry.model_providers` are automatically merged into `ModelFactory` on creation.

### SklearnProvider

| backbone | Classification | Regression |
|---|---|---|
| `random_forest` | `RandomForestClassifier` | `RandomForestRegressor` |
| `gradient_boosting` | `GradientBoostingClassifier` | `GradientBoostingRegressor` |
| `svm` | `SVC` | `SVR` |
| `logistic_regression` | `LogisticRegression` | fallback RandomForest |
| `mlp` | `MLPClassifier` | `MLPRegressor` |

All models use `random_state=42`.

### CatBoostProvider

- `thread_count=-1` (all cores)
- `loss_function`: `MultiClass` (>2 classes), `Logloss` (binary), `RMSE` (regression)
- `eval_metric` from `objective`

### LightGBMProvider

- `objective`: `multiclass`, `binary`, `regression`
- `metric` from `objective`

### PyTorchProvider

Builds `nn.Sequential` or `_LayerGraph` from YAML layer descriptions.

**Training:** Adam, 10 epochs, batch_size=32.
- Regression: `MSELoss`
- Classification (>2 classes): `CrossEntropyLoss`
- Binary: `BCEWithLogitsLoss`

**Default architecture** (when `layers` is omitted):
```yaml
- linear (128) → relu → dropout (0.2) → linear (64) → relu → dropout (0.2) → linear
```

### HybridModelProvider

Late-fusion: two branches (images → Linear→256→ReLU→Dropout→128, text — same) → concat → fusion (Linear→128→ReLU→Dropout→Linear→out).

---

## Uplift Meta-Estimators

### TLearner

Two copies of the base model (via `deepcopy`):
- `model_t` trained on treatment=1
- `model_c` trained on treatment=0

`predict()` = `P(Y|T=1) - P(Y|T=0)`

### SLearner

Single copy, treatment added as an extra feature.

`fit()` requires `treatment` kwarg.
`predict()` = `predict([X, 1]) - predict([X, 0])`

---

## Type System

### `PipelineContext`

```
PipelineContext
  ├── data: PipelineData
  │     ├── train_df, test_df (DataFrame)
  │     ├── train_df_path, test_df_path (Path)
  │     ├── detected_types (dict[str, FeatureType | str])
  │     ├── task_type (TaskType)
  │     └── target_col, treatment_col
  └── artifacts: PipelineArtifacts
        ├── model (MLModelProtocol | None)
        ├── imputer (SimpleImputer | None)
        ├── encoders (dict[str, OrdinalEncoder])
        ├── scaler (StandardScaler | RobustScaler | MinMaxScaler | None)
        ├── feature_names (list[str] | None)
        ├── target_mean (float | None)
        └── target_mapping (dict | None)
```

### FeatureType → FeatureMap

`FeatureType` — a StrEnum with 7 built-in types, extendable via `registry.feature_handler`.

`FeatureMap = dict[ColumnName, FeatureType | str]` — stores auto-detection results.

Types are used by preprocessing steps to select columns:
- `ImputeStep` processes `NUMBER`
- `EncodeScaleStep` processes `NUMBER` (scale) and `CATEGORY` (encode)
- Custom types are ignored by built-in steps (pass through "as-is")

---

## UnifiedTensor and Adapters

```
BaseAdapter[T] (ABC)
  ├── NumpyAdapter[np.ndarray]
  └── TorchAdapter[torch.Tensor] (if torch is installed)
```

`UnifiedTensor[T]` — wraps raw arrays, delegates operations to the adapter.

**Supported operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `@`, `**`, unary `-`
- Math: `abs`, `sqrt`, `exp`, `log`, `clip`
- Reductions: `sum`, `mean`, `min`, `max`, `argmin`, `argmax`
- Shape: `reshape`, `flatten`, `transpose`, `squeeze`, `unsqueeze`
- Properties: `.T`, `.ndim`, `.shape`, `.dtype`, `.device`
- Conversion: `.numpy()`, `.tolist()`, `.item()`, `.copy()`

`_resolve_adapter(backend)` — checks `registry._tensor_adapters` first, then built-in.

---

## Registry System

Detailed: [Extensions](extensions.md)

In short: `ExtensionRegistry` is a singleton with 6 extension containers:

| Container | Purpose |
|---|---|
| `_steps` | Custom pipeline steps |
| `_feature_handlers` | Custom column type detectors |
| `_tensor_adapters` | UnifiedTensor backends |
| `_model_providers` | Custom model providers |
| `_layer_builders` | Custom PyTorch layer types |
| `_activations` | Custom activation functions |

Built-in PyTorch layers and activations auto-register when `mlcombine.models.providers.pytorch` is imported.
