# Extensions

mlcombine is designed to be extensible. You can add custom pipeline steps,
column types, model providers, tensor adapters, PyTorch layers, and activation
functions — without modifying the library source code.

## Registration System

The central entry point is the global `registry` singleton:

```python
from mlcombine.core.registry import registry
```

### 1. Custom Pipeline Steps

Subclass `BaseStep[PipelineContext]` and add the step via `engine.add_step*`:

```python
from mlcombine.core.types import BaseStep, PipelineContext
from mlcombine.core.registry import registry


class AuditStep(BaseStep[PipelineContext]):
    """Logs dataframe shape after each step."""

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.data.train_df is not None:
            logger.info(f"Train shape: {context.data.train_df.shape}")
        if context.data.test_df is not None:
            logger.info(f"Test shape: {context.data.test_df.shape}")
        return context
```

```python
engine = PipelineEngine.from_config(cfg)
engine.add_step_after("TypeDetectStep", AuditStep())
```

### 2. Custom Column Types (FeatureHandler)

Add a new column type that is auto-detected by `TypeDetectStep`.

```python
import pandas as pd
from mlcombine.core.registry import FeatureHandler, registry


@registry.feature_handler("audio")
class AudioHandler(FeatureHandler):
    """Detects columns containing audio file paths."""

    def detect(self, series: pd.Series) -> bool:
        return bool(
            series.astype(str)
            .str.contains(r"\.(mp3|wav|flac|ogg)$", regex=True)
            .mean() > 0.5
        )

    def preprocess(self, series: pd.Series, config: dict | None = None) -> pd.Series:
        """Optional preprocessing. Default is a no-op."""
        return series
```

After registration, `TypeDetectStep` will return `"audio"` for such columns.
Custom types are ignored by built-in preprocessing steps (they pass through "as-is").

### 3. Custom Model Backends

Providers are functions, not classes. Register a function via `@registry.model_provider`:

```python
from mlcombine.core.registry import registry


class MyCustomModel:
    """A simple model conforming to MLModelProtocol."""

    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        return self

    def predict(self, X):
        import numpy as np
        return np.zeros(len(X))


@registry.model_provider("my_provider")
def my_provider_fn(backbone="rf", task_type=None, objective=None,
                   num_classes=None, input_size=None, **params):
    """Provider function. Receives params from YAML as **params."""
    return MyCustomModel()
```

A provider can accept `model=` or `models=` keyword arguments for meta-providers (see built-in `cv`, `ensemble`, `tuner`).

Now in YAML (DAG format):

```yaml
model:
  - provider: "my_provider"
    params:
      backbone: "rf"
```

### 4. Custom Tensor Adapters

To support new tensor backends (JAX, MLX, etc.):

```python
from mlcombine.core.tensor.base import BaseAdapter
from mlcombine.core.registry import registry


@registry.tensor_adapter("jax")
class JaxAdapter(BaseAdapter):
    """Adapter for JAX arrays."""

    def convert(self, data, dtype=None, device=None):
        import jax.numpy as jnp
        return jnp.array(data, dtype=dtype)

    def to_numpy(self, data):
        import jax.numpy as jnp
        return jnp.asarray(data)

    # ... remaining BaseAdapter methods
```

### 5. Custom PyTorch Layers

```python
import torch.nn as nn
from mlcombine.core.registry import registry


@registry.layer_builder("my_custom_layer")
def build_my_layer(cfg, prev_dim, **kwargs):
    """Builds a custom layer from config.

    Args:
        cfg: dict with parameters from YAML (type already extracted).
        prev_dim: input dimension (auto-detected).
    """
    return nn.Sequential(
        nn.Linear(prev_dim, cfg.get("hidden_dim", 64)),
        nn.GELU(),
        nn.Dropout(cfg.get("dropout", 0.1)),
    )
```

Now in YAML:

```yaml
model:
  - provider: "pytorch"
    params:
      layers:
        - type: "linear"
          out_features: 128
        - type: "my_custom_layer"
          hidden_dim: 256
          dropout: 0.2
        - type: "linear"
            out_features: 10
```

### 6. Custom Activation Functions

```python
import torch.nn as nn
from mlcombine.core.registry import registry


@registry.activation("swish")
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()
```

Now available in any layer:

```yaml
model:
  - provider: "pytorch"
    params:
      layers:
        - type: "linear"
          out_features: 128
          activation: "swish"
```

---

## Loading Plugins via YAML

Add a `plugins` section to your config:

```yaml
plugins:
  - "./my_extensions.py"         # path to .py file
  - "my_package.plugin_module"   # Python module name
```

Plugins are loaded before the pipeline is built. Inside the file, use `@registry.*` decorators.

Example `my_extensions.py`:

```python
"""My mlcombine extensions."""

import pandas as pd
from mlcombine.core.registry import FeatureHandler, registry


@registry.feature_handler("geo_coord")
class GeoHandler(FeatureHandler):
    def detect(self, series: pd.Series) -> bool:
        return bool(
            series.astype(str)
            .str.contains(r"^-?\d+\.\d+,-?\d+\.\d+$", regex=True)
            .mean() > 0.5
        )


@registry.model_provider("my_ensemble")
def my_ensemble_provider(backbone="rf", task_type=None, objective=None,
                         num_classes=None, input_size=None, **params):
    from sklearn.ensemble import VotingClassifier
    return VotingClassifier(...)
```

---

## Loading Plugins Programmatically

```python
from mlcombine.core.pipeline import _load_plugins

_load_plugins(["./my_extensions.py"])
```

The function supports:
- Path to `.py` file (absolute or relative to CWD)
- Python module name (via `importlib.import_module`)
