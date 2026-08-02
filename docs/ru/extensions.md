# Расширения (Extensions)

mlcombine спроектирован как расширяемая система. Вы можете добавлять
кастомные шаги пайплайна, типы колонок, бэкенды моделей, тензорные адаптеры,
слои PyTorch и функции активации — без модификации исходного кода библиотеки.

## Регистрационная система

Центральная точка входа — глобальный синглтон `registry`:

```python
from mlcombine.core.registry import registry
```

### 1. Кастомные шаги пайплайна

Наследуйте `BaseStep[PipelineContext]` и добавьте шаг через `engine.add_step*`:

```python
from mlcombine.core.types import BaseStep, PipelineContext
from mlcombine.core.registry import registry


class AuditStep(BaseStep[PipelineContext]):
    """Логирует размер датафрейма после каждого шага."""

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

### 2. Кастомные типы колонок (FeatureHandler)

Позволяет добавить новый тип колонки, который автоопределяется `TypeDetectStep`.

```python
import pandas as pd
from mlcombine.core.registry import FeatureHandler, registry


@registry.feature_handler("audio")
class AudioHandler(FeatureHandler):
    """Определяет колонки с путями к аудиофайлам."""

    def detect(self, series: pd.Series) -> bool:
        return bool(
            series.astype(str)
            .str.contains(r"\.(mp3|wav|flac|ogg)$", regex=True)
            .mean() > 0.5
        )

    def preprocess(self, series: pd.Series, config: dict | None = None) -> pd.Series:
        """Опциональная предобработка. По умолчанию — no-op."""
        return series
```

После регистрации `TypeDetectStep` будет возвращать `"audio"` для таких колонок.
Кастомные типы игнорируются встроенными шагами препроцессинга (проходят "as-is").

### 3. Кастомные бэкенды моделей

Провайдеры — функции, а не классы. Зарегистрируйте функцию через `@registry.model_provider`:

```python
from mlcombine.core.registry import registry


class MyCustomModel:
    """Простая модель, совместимая с MLModelProtocol."""

    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        return self

    def predict(self, X):
        import numpy as np
        return np.zeros(len(X))


@registry.model_provider("my_provider")
def my_provider_fn(backbone="rf", task_type=None, objective=None,
                   num_classes=None, input_size=None, **params):
    """Функция-провайдер. Получает params из YAML как **params."""
    return MyCustomModel()
```

Провайдер может принимать `model=` или `models=` для meta-провайдеров (см. встроенные `fold_ensemble`, `ensemble`, `tuner`).

Теперь в YAML (DAG-формат):

```yaml
model:
  - provider: "my_provider"
    params:
      backbone: "rf"
```

### 4. Кастомные тензорные адаптеры

Для поддержки новых тензорных бэкендов (JAX, MLX, etc.):

```python
from mlcombine.core.tensor.base import BaseAdapter
from mlcombine.core.registry import registry


@registry.tensor_adapter("jax")
class JaxAdapter(BaseAdapter):
    """Адаптер для JAX-массивов."""

    def convert(self, data, dtype=None, device=None):
        import jax.numpy as jnp
        return jnp.array(data, dtype=dtype)

    def to_numpy(self, data):
        import jax.numpy as jnp
        return jnp.asarray(data)

    # ... остальные методы BaseAdapter
```

### 5. Кастомные слои PyTorch

```python
import torch.nn as nn
from mlcombine.core.registry import registry


@registry.layer_builder("my_custom_layer")
def build_my_layer(cfg, prev_dim, **kwargs):
    """Строит кастомный слой из конфига.

    Args:
        cfg: dict с параметрами из YAML (type уже извлечён).
        prev_dim: размерность входа (автоопределяется).
    """
    return nn.Sequential(
        nn.Linear(prev_dim, cfg.get("hidden_dim", 64)),
        nn.GELU(),
        nn.Dropout(cfg.get("dropout", 0.1)),
    )
```

Теперь в YAML:

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

### 6. Кастомные функции активации

```python
import torch.nn as nn
from mlcombine.core.registry import registry


@registry.activation("swish")
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()
```

Теперь доступна в любом слое:

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

## Подключение плагинов через YAML

Добавьте секцию `plugins` в конфиг:

```yaml
plugins:
  - "./my_extensions.py"         # путь к .py файлу
  - "my_package.plugin_module"   # имя Python-модуля
```

Плагины загружаются до создания пайплайна. Внутри файла используйте `@registry.*` декораторы.

Пример файла `my_extensions.py`:

```python
"""Мои расширения для mlcombine."""

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

## Загрузка плагинов программно

```python
from mlcombine.core.pipeline import _load_plugins

_load_plugins(["./my_extensions.py"])
```

Функция поддерживает:
- Путь к `.py` файлу (абсолютный или от CWD)
- Имя Python-модуля (через `importlib.import_module`)
