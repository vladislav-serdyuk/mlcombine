# mlcombine

![Version](https://img.shields.io/badge/Version-0.1.0-blue.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://www.python.org/)

![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![Mypy](https://img.shields.io/badge/mypy-strict-blue)
![Pytest](https://img.shields.io/badge/pytest-passed-brightgreen)

![Pydantic](https://img.shields.io/badge/pydantic-v2-E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-5A9FD4?style=for-the-badge&logo=optuna&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

![CatBoost](https://img.shields.io/badge/CatBoost-FCC800?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-00B050?style=for-the-badge)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

Declarative Low-Code/No-Code framework for ML competitions.
Just write a YAML config — the framework handles the rest.

```bash
pip install mlcombine
mlcombine train --config config.yaml
```

## Quick Start

With `uv` (recommended):

```bash
git clone https://github.com/your-org/mlcombine.git
cd mlcombine
uv sync --group dev
```

With `pip`:

```bash
git clone https://github.com/your-org/mlcombine.git
cd mlcombine
pip install -e ".[dev]"
```

Run the Titanic baseline:

```bash
mlcombine train --config examples/configs/titanic_baseline.yaml
```

Or from Python:

```python
from mlcombine.core.pipeline import PipelineEngine
from mlcombine.core.schemas.config import MLCombineConfig
from mlcombine.core.types import PipelineContext

cfg = MLCombineConfig(**{
    "data": {
        "train_df": "train.csv",
        "test_df": "test.csv",
        "target_col": "Survived",
    },
    "model": [
        {"provider": "sklearn", "params": {"backbone": "gradient_boosting"}},
    ],
    "trainer": {"output_dir": "outputs", "output_file": "outputs/submission.csv"},
})
engine = PipelineEngine.from_config(cfg)
engine.run_all(PipelineContext())
```

---

## Documentation

| Language | |
|---|---|
| English | [README.md](README.md) · [config.yaml](docs/en/config.md) · [Extensions](docs/en/extensions.md) · [Architecture](docs/en/architecture.md) · [PyTorch Layers](docs/en/layers.md) |
| Russian | [README-RU.md](README-RU.md) · [config.yaml](docs/ru/config.md) · [Расширения](docs/ru/extensions.md) · [Архитектура](docs/ru/architecture.md) · [Слои PyTorch](docs/ru/layers.md) |

### Example YAML configs

| Use-case | File |
|---|---|
| Basic baseline (sklearn) | [titanic_baseline.yaml](examples/configs/titanic_baseline.yaml) |
| CV + Optuna tuning | [tabular_cv.yaml](examples/configs/tabular_cv.yaml) |
| Ensemble CatBoost + LightGBM | [ensemble_blend.yaml](examples/configs/ensemble_blend.yaml) |
| Custom PyTorch architecture | [pytorch_layers.yaml](examples/configs/pytorch_layers.yaml) |
| Remote dataset (URL) | [remote_data.yaml](examples/configs/remote_data.yaml) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Configuration | `YAML` + `Pydantic v2` |
| CLI | `Click` |
| Tabular models | `CatBoost`, `LightGBM`, `scikit-learn` |
| Neural networks | `PyTorch` (optional) |
| Hyperparameter search | `Optuna` |
| Pipeline engine | Custom (BaseStep → PipelineEngine) |
| Type checking | `mypy --strict` |
| Testing | `pytest` |

---

## Project Structure

```
mlcombine/
├── src/mlcombine/
│   ├── cli/main.py              # Click CLI (train / predict)
│   ├── core/
│   │   ├── pipeline.py          # PipelineEngine + _BASE_ORDER + plugin loader
│   │   ├── builder.py           # ModelBuilder — topological DAG sort
│   │   ├── registry.py          # ExtensionRegistry (steps, providers, metrics...)
│   │   ├── types.py             # BaseStep, PipelineContext, PipelineData
│   │   ├── metric.py            # Built-in metrics (f1, auc, rmse...)
│   │   ├── evaluator.py         # BaseArchitectureValidator ABC
│   │   ├── tensor/              # UnifiedTensor (numpy/torch adapters)
│   │   └── schemas/
│   │       ├── config.py        # MLCombineConfig, ModelNode, DataConfig...
│   │       ├── blueprint.py     # ModelBlueprint — lazy model construction
│   │       └── step_configs.py  # StepConfigs (extra="allow")
│   ├── models/
│   │   ├── providers/           # CatBoost, LightGBM, sklearn, PyTorch, hybrid
│   │   └── meta/                # ensemble, fold_ensemble, stacking, tuner, uplift
│   ├── steps/                   # 27 pipeline steps (load, fit, predict, evaluate...)
│   └── evaluators/              # cv.py, holdout.py
├── tests/                       # tests
├── docs/                        # Documentation (ru/en)
├── examples/configs/            # Example YAML configs
└── pyproject.toml
```

---

## Architecture

```mermaid
flowchart TB
    YAML["YAML config"] --> PipelineEngine

    subgraph PipelineEngine["PipelineEngine"]
        direction TB
        Resolve["_resolve_order()"]
        Steps["Step sequence"]
        Resolve --> Steps
    end

    PipelineEngine --> Step1["DataLoaderStep"]
    Step1 --> Step2["TypeDetectStep"]
    Step2 --> Step3["FeatureGenerationStep<br/>DateTimeFeatureStep<br/>TextEmbeddingStep"]
    Step3 --> Step4["SplitStep"]
    Step4 --> Step5["ImputeStep / EncodeScaleStep"]
    Step5 --> CreateModel["CreateModelStep"]

    CreateModel --> ModelBuilder["ModelBuilder.build_all()"]
    ModelBuilder --> DAG

    subgraph DAG["Model DAG"]
        CB["catboost<br/>provider: catboost"]
        FE["fold_ensemble<br/>model: cb<br/>provider: fold_ensemble"]
        FE --> CB
    end

    DAG --> ModelFit["ModelFitStep"]
    ModelFit --> Evaluate["EvaluateStep"]
    Evaluate --> SaveArtifacts["SaveArtifactsStep"]
    SaveArtifacts --> Predict["ModelPredictStep"]
    Predict --> Save["SavePredictionsStep"]
```

### How it works

1. **PipelineEngine** reads the YAML config, loads plugins, resolves step order
2. Each step is a `BaseStep` with `train`/`predict` flags — train and predict pipelines can differ
3. **ModelBuilder** resolves the model DAG: leaf providers (CatBoost, sklearn) + meta-providers (ensemble, stacking, fold_ensemble, tuner)
4. Dependencies between nodes are resolved via `id` → `model`/`models` references
5. In predict mode, only steps with `predict=True` run; artifacts are loaded from disk

### Default step order

```
PrepareEnvironment → PrepareDataset → DataLoader → TypeDetect → FeatureGeneration →
LoadArtifacts → Split → Impute → EncodeScale → CreateModel → ModelFit →
Evaluate → SaveArtifacts → AlignFeatures → ModelPredict → DropTargetColumns → SavePredictions
```

Steps can be reordered via `pipeline.order` or extended via `@registry.step()` with `before`/`after`.

---

## Configuration

### Basic baseline

```yaml
# titanic_baseline.yaml
data:
  train_df: "train.csv"
  test_df: "test.csv"
  target_col: "Survived"
  task_type: classification
model:
  - provider: "sklearn"
    params:
      backbone: "gradient_boosting"
trainer:
  output_dir: outputs
  output_file: outputs/submission.csv
```

### CatBoost + Fold Ensemble

```yaml
model:
  - id: "cb"
    provider: "catboost"
    params:
      iterations: 5000
      depth: 10
      learning_rate: 0.07
  - provider: "fold_ensemble"
    model: "cb"
    params:
      n_folds: 5
      stratified: true
```

### Ensemble CatBoost + LightGBM

```yaml
model:
  - id: "cb"
    provider: "catboost"
    params: { iterations: 3000, depth: 8 }
  - id: "lgb"
    provider: "lightgbm"
    params: { num_iterations: 3000, max_depth: 8 }
  - provider: "ensemble"
    models: ["cb", "lgb"]
    params:
      weights: [0.6, 0.4]
```

### Optuna Tuning

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
```

---

## Model Providers

| Provider | Type | Description |
|---|---|---|
| `sklearn` | leaf | Random Forest, Gradient Boosting, SVM, Logistic Regression, MLP |
| `catboost` | leaf | CatBoost (GPU via `gpu: true`) |
| `lightgbm` | leaf | LightGBM |
| `pytorch` | leaf | Custom architectures via YAML blocks |
| `hybrid` | leaf | Multi-modal PyTorch (image + text) |
| `ensemble` | meta | Weighted average (hard/soft vote) |
| `fold_ensemble` | meta | K-fold CV + fold model averaging |
| `stacking` | meta | Stacking with meta-model (logistic/ridge), OOF-support |
| `tuner` | meta | Optuna hyperparameter search |
| `t_learner` / `s_learner` | meta | Uplift modeling |

---

## Extending: Plugins

Any step or provider can be added through the plugin system.

### Custom Step

```python
# my_step.py
from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

@registry.step("MyStep", before="TypeDetectStep")
class MyStep(BaseStep[PipelineContext]):
    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict=False, weights=None):
        self._param = cfg.step_config.my_step.get("param", 42)

    def run(self, context: PipelineContext) -> PipelineContext:
        # mutate context.data.train_df / test_df
        return context
```

Register in YAML:

```yaml
plugins: ["path/to/my_step.py"]
step_config:
  my_step:
    param: 100
```

### Custom Model Provider

```python
@registry.model_provider("my_model")
def my_provider(backbone=None, task_type=None, objective=None, **params):
    # params come from YAML + task_type/num_classes from ModelBuilder
    return MyModel(**params)
```

---

## Testing & Linting

```bash
# Tests
python -m pytest tests/ -q
python -m pytest tests/ -q -m "not optuna"  # skip slow optuna tests

# Linter
ruff check .

# Type checker
mypy --strict src

# Formatting
ruff format .
```