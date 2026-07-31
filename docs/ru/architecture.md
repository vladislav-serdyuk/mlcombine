# Архитектура mlcombine

## Общая схема

```
YAML Config → MLCombineConfig (Pydantic)
                      ↓
         PipelineEngine.from_config()
                      ↓
         PipelineContext() created
                      ↓
    ┌───── Пайплайн из 10 шагов ──────────┐
    │ 1. PrepareDatasetStep   (скачивание,│
    │      распаковка архивов)            │
    │ 2. DataLoaderStep       (CSV→DataFrame)│
    │ 3. TypeDetectStep       (типы колонок,│
    │      тип задачи)                     │
    │ 4. PrepareEnvironmentStep (автоустановка)│
    │ 5. ImputeStep           (заполнение NaN)│
    │ 6. EncodeScaleStep      (кодирование + │
    │      масштабирование)                │
    │ 7. CreateModelStep      (ModelFactory) │
    │ 8. WrapUpliftStep?      (TLearner/   │
    │      SLearner)                       │
    │ 9. ModelFitStep         (fit())      │
    │10. SaveArtifactsStep    (joblib dump)│
    └─────────────────────────────────────┘
                      ↓
         Context.data + Context.artifacts
         Модель сохранена в ./outputs/model.joblib
```

Для инференса шаги 4-10 заменяются на:
```
 4. LoadArtifactsStep     (загрузка модели)
 5. DropTargetColumnsStep (удаление таргета)
 6. SavePredictionsStep   (predict → CSV)
```

---

## PipelineEngine

`PipelineEngine[ContextType]` — оркестратор, управляющий последовательностью шагов.

### Создание

```python
engine = PipelineEngine()
engine.add_step(MyStep())
engine.add_step_before("ImputeStep", MyOtherStep())
```

### Из конфига

```python
engine = PipelineEngine.from_config(cfg)
```

Этот метод:
1. Загружает плагины из `cfg.plugins`
2. Создаёт `ModelFactory` (с провайдерами из `registry`)
3. Строит пайплайн из 10 шагов (тренировка) или 6 шагов (инференс)

### Выполнение

```python
ctx = PipelineContext()
result = engine.run_all(ctx)   # возвращает мутированный ctx
```

Каждый шаг логируется с таймингом.

### Манипуляция шагами

```python
engine.add_step(step)                  # в конец
engine.add_step_before("name", step)   # перед указанным
engine.add_step_after("name", step)    # после указанного
```

Имена шагов автоматически дедуплицируются (добавлением суффикса `_N`).

---

## Шаги пайплайна

### 1. `PrepareDatasetStep`

Скачивает датасет по URL (если `train_df` начинается с `http`), распаковывает архивы (`.zip`, `.tar`, `.gz`, `.bz2`, `.xz`).

**Вход:** `cfg.data.train_df`, `cfg.data.test_df`
**Выход:** `ctx.data.train_df_path`, `ctx.data.test_df_path`

### 2. `DataLoaderStep`

Загружает CSV/TSV/Parquet/XLSX/XLS в `pd.DataFrame`. Определяет формат по расширению файла.

**Вход:** пути к файлам
**Выход:** `ctx.data.train_df`, `ctx.data.test_df`

### 3. `TypeDetectStep`

Автоопределяет тип каждой колонки (`FeatureType`) и тип задачи (`TaskType`).

**Логика для числовых колонок:**
- int с `unique_ratio < 5%` → `CATEGORY`
- иначе → `NUMBER`

**Логика для строковых/object колонок:**
1. Попытка распарсить как дату → `DATETIME`
2. Проверка расширений изображений → `IMAGE_PATH`
3. Поиск hex-последовательностей → `SEQUENCE_TOKEN`
4. Средняя длина > 50 символов → `TEXT`
5. Кастомные `FeatureHandler` из `registry` → кастомный тип
6. `unique_ratio < 0.5` → `CATEGORY`
7. Иначе → `TEXT`

**Тип задачи:**
- Несколько таргет-колонок → `MULTITASK`
- int-таргет с `unique_count <= 20` → `CLASSIFICATION`
- float-таргет → `REGRESSION`
- object-таргет → `CLASSIFICATION`

### 4. `PrepareEnvironmentStep`

Автоустановка缺失ющих Python-пакетов через `uv add` (с фолбеком на `pip install`).

**Маппинг бэкенд → пакет:**
- `catboost` → `catboost`
- `lightgbm` → `lightgbm`
- `pytorch` / `hybrid` → `torch`

После установки перезагружает модули провайдеров через `importlib.reload()`.

### 5. `ImputeStep`

`SimpleImputer` на числовых колонках (по `FeatureType.NUMBER`).

### 6. `EncodeScaleStep`

Two этапа:
1. **Кодирование:** `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` на каждую категориальную колонку
2. **Масштабирование:** `StandardScaler` / `RobustScaler` / `MinMaxScaler` на числовые колонки

`EncodeStrategy.TARGET` пока выбрасывает `NotImplementedError` (нужна OOF-изоляция).

### 7. `CreateModelStep`

Создаёт модель через `ModelFactory.create_model()`.

Автоопределение `input_size`: если `model.layers` задан, но `input_size` не указан — считается по количеству признаков (исключая таргет и treatment).

### 8. `WrapUpliftStep`

Оборачивает модель в `TLearner` или `SLearner`, если `model.uplift_method` != `None`.

### 9. `ModelFitStep`

Извлекает X, y, treatment из `train_df`, вызывает `model.fit()`.

### 10. `SaveArtifactsStep`

Сохраняет модель через `joblib.dump` в `{output_dir}/model.joblib`.

---

## ModelFactory и провайдеры

```
ModelFactory
  │
  ├── SklearnProvider    (RandomForest, SVM, MLP, etc.)
  ├── CatBoostProvider   (CatBoostClassifier / CatBoostRegressor)
  ├── LightGBMProvider   (LGBMClassifier / LGBMRegressor)
  ├── PyTorchProvider    (динамическая архитектура из YAML)
  └── HybridModelProvider(изображения + текст → late-fusion)
```

### ModelFactory

```python
factory = ModelFactory()
factory.register_provider("my_provider", MyProvider())
factory.register_custom_model("my_model", pretrained_model)
model = factory.create_model(config, task_type, num_classes, input_size)
```

Порядок разрешения:
1. `config.provider == "auto"` → `"sklearn"`
2. Проверка `custom_models`
3. Поиск провайдера в `_providers` (built-in + registry)
4. `provider.create(backbone, objective, task_type, num_classes, **kwargs)`

Провайдеры из `registry.model_providers` автоматически мержатся в `ModelFactory` при создании.

### SklearnProvider

| backbone | Класс классификации | Класс регрессии |
|---|---|---|
| `random_forest` | `RandomForestClassifier` | `RandomForestRegressor` |
| `gradient_boosting` | `GradientBoostingClassifier` | `GradientBoostingRegressor` |
| `svm` | `SVC` | `SVR` |
| `logistic_regression` | `LogisticRegression` | fallback RandomForest |
| `mlp` | `MLPClassifier` | `MLPRegressor` |

Все модели с `random_state=42`.

### CatBoostProvider

- `thread_count=-1` (все ядра)
- `loss_function`: `MultiClass` (>2 классов), `Logloss` (2 класса), `RMSE` (регрессия)
- `eval_metric` по `objective`

### LightGBMProvider

- `objective`: `multiclass`, `binary`, `regression`
- `metric` по `objective`

### PyTorchProvider

Строит `nn.Sequential` или `_LayerGraph` из YAML-описания слоёв.

**Тренировка:** Adam, 10 эпох, batch_size=32.
- Регрессия: `MSELoss`
- Классификация (>2 классов): `CrossEntropyLoss`
- Бинарная: `BCEWithLogitsLoss`

**Дефолтная архитектура** (когда `layers` не указан):
```yaml
- linear (128) → relu → dropout (0.2) → linear (64) → relu → dropout (0.2) → linear
```

### HybridModelProvider

Late-fusion: две ветки (изображения → Linear→256→ReLU→Dropout→128, текст — аналогично) → concat → fusion (Linear→128→ReLU→Dropout→Linear→out).

---

## Uplift-мета-оценщики

### TLearner

Две копии базовой модели (через `deepcopy`):
- `model_t`: обучена на treatment=1
- `model_c`: обучена на treatment=0

`predict()` = `P(Y|T=1) - P(Y|T=0)`

### SLearner

Одна копия базовой модели, treatment добавляется как дополнительный признак.

`fit()` требует `treatment` kwarg.
`predict()` = `predict([X, 1]) - predict([X, 0])`

---

## Система типов

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

`FeatureType` — StrEnum с 7 встроенными типами + возможность кастомных (через `registry.feature_handler`).

`FeatureMap = dict[ColumnName, FeatureType | str]` — хранит результат автоопределения.

Типы используются шагами препроцессинга для выбора колонок:
- `ImputeStep` обрабатывает `NUMBER`
- `EncodeScaleStep` обрабатывает `NUMBER` (scale) и `CATEGORY` (encode)
- Кастомные типы игнорируются встроенными шагами (проходят сквозь пайплайн "as-is")

---

## UnifiedTensor и адаптеры

```
BaseAdapter[T] (ABC)
  ├── NumpyAdapter[np.ndarray]
  └── TorchAdapter[torch.Tensor] (если torch установлен)
```

`UnifiedTensor[T]` — обёртка над сырым массивом, делегирующая операции адаптеру.

**Поддерживаемые операции:**
- Арифметика: `+`, `-`, `*`, `/`, `@`, `**`, unary `-`
- Математика: `abs`, `sqrt`, `exp`, `log`, `clip`
- Редукции: `sum`, `mean`, `min`, `max`, `argmin`, `argmax`
- Форма: `reshape`, `flatten`, `transpose`, `squeeze`, `unsqueeze`
- Свойства: `.T`, `.ndim`, `.shape`, `.dtype`, `.device`
- Конверсия: `.numpy()`, `.tolist()`, `.item()`, `.copy()`

`_resolve_adapter(backend)` — сначала проверяет `registry._tensor_adapters`, потом built-in.

---

## Регистрационная система (Registry)

Подробно: [Расширения](extensions.md)

Коротко: `ExtensionRegistry` — синглтон с 6 контейнерами расширений:

| Контейнер | Что регистрирует |
|---|---|
| `_steps` | Кастомные шаги пайплайна |
| `_feature_handlers` | Детекторы кастомных типов колонок |
| `_tensor_adapters` | Бэкенды для UnifiedTensor |
| `_model_providers` | Кастомные провайдеры моделей |
| `_layer_builders` | Кастомные типы слоёв PyTorch |
| `_activations` | Кастомные функции активации |

Встроенные слои и активации PyTorch авторегистрируются при импорте `mlcombine.models.providers.pytorch`.
