# Конфигурация YAML

mlcombine использует единый YAML-файл для описания всего пайплайна.
Файл парсится через Pydantic-модели, все поля со строгой валидацией (`extra="forbid"`).

## Минимальный пример

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

Backend по умолчанию — `sklearn`, backbone — `random_forest`.

## Полный пример (Titanic + CatBoost + URL-датасет)

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

## Секции конфига

### `plugins` (опционально)

Список Python-файлов или модулей для импорта до запуска пайплайна.
Каждый файл может регистрировать расширения через `@registry.*` декораторы.

```yaml
plugins:
  - "./my_extensions.py"
  - "my_package.plugin_module"
```

Поддерживаются:
- Путь к `.py` файлу (абсолютный или относительный от CWD)
- Имя Python-модуля (например, `my_package.my_plugin`)

---

### `data` (обязательно)

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `train_df` | `str` | **обязательно** | Путь к CSV/TSV/Parquet/XLSX/XLS или **URL** |
| `test_df` | `str` | **обязательно** | Путь к файлу или URL |
| `target_col` | `str \| list[str] \| dict[str,str]` | **обязательно** | Имя целевой колонки (или список для мультитаска, или словарь) |
| `sep` | `str` | `","` | Разделитель в CSV |
| `treatment_col` | `str \| None` | `None` | Колонка с treatment (для uplift-моделирования) |
| `drop_columns` | `list[str]` | `[]` | Колонки для удаления перед обучением |
| `force_prepare_dataset` | `bool` | `false` | Принудительно перезагрузить/распаковать датасет |
| `task_type` | `str \| None` | `None` | Явное указание типа задачи (`classification`, `regression`, `uplift`, `multitask`) |

`train_df` и `test_df` поддерживают **HTTP/HTTPS URL**. Файл скачивается автоматически:

```yaml
data:
  train_df: "https://example.com/datasets/train.csv"
  test_df: "https://example.com/datasets/test.csv"
```

Поддерживаются архивы `.zip`, `.tar`, `.tar.gz`, `.gz`, `.bz2`, `.xz` — автоматически распаковываются:

```yaml
data:
  train_df: "https://example.com/datasets/train.zip"
```

Формат файла определяется по расширению:
- `.csv` — CSV
- `.tsv` — TSV (табуляция)
- `.parquet` — Parquet
- `.xlsx` / `.xls` — Excel

---

### `force_types` (опционально)

Принудительно указать типы колонок (в обход автоопределения).

```yaml
force_types:
  Pclass: "category"
  PassengerId: "number"
```

Допустимые значения: `number`, `category`, `datetime`, `image_path`, `sequence_token`, `text`, `unknown`, а также любые кастомные типы, зарегистрированные через `FeatureHandler`.

---

### `model` — DAG моделей (список)

`model` теперь — список узлов (`ModelNode`). Каждый узел — один провайдер.
Последний узел в списке — финальная модель для fit/predict.

```yaml
model:
  - provider: "sklearn"
    params:
      backbone: "random_forest"
      n_estimators: 300
```

#### Поля узла

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `provider` | `str` | **обязательно** | Провайдер (см. таблицу ниже) |
| `id` | `str \| None` | `None` | Идентификатор для ссылок из других узлов |
| `model` | `str \| None` | `None` | Ссылка на один узел (по `id`) — для meta-провайдеров |
| `models` | `list[str]` | `[]` | Ссылки на несколько узлов (по `id`) |
| `params` | `dict` | `{}` | Параметры провайдера (backbone, layers, и т.д.) |

#### Провайдеры

| Значение | Описание |
|---|---|
| `"sklearn"` | Scikit-learn (RandomForest, SVM, MLP, GradientBoosting, LogisticRegression) |
| `"catboost"` | CatBoost |
| `"lightgbm"` | LightGBM |
| `"pytorch"` | PyTorch (кастомная архитектура через `params.layers`) |
| `"hybrid"` | Гибрид: изображения + текст → late-fusion |
| `"tuner"` | Optuna-тюнинг гиперпараметров |
| `"fold_ensemble"` | K-Fold обучение с OOF-предсказаниями + target encoding |
| `"ensemble"` | Взвешенное усреднение нескольких моделей |
| `"stacking"` | Stacking с мета-моделью на OOF |
| `"t_learner"` | Uplift T-learner |
| `"s_learner"` | Uplift S-learner |
| `"<custom>"` | Любой кастомный провайдер, зарегистрированный через `registry.model_provider()` |

#### Примеры DAG-конфигов

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

**FoldEnsemble с target-encoding:**

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

**Optuna-тюнинг CatBoost:**

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

**PyTorch с кастомными слоями:**

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

#### `objective` и `num_classes`

| objective | `num_classes` |
|---|---|
| `"MAPE"`, `"RMSE"` | Регрессия |
| `"F1"`, `"accuracy"`, `"precision"`, `"recall"` | Классификация (2 класса, если не указано иное) |
| `"AUC"`, `"logloss"` | Классификация |

#### `layers` (PyTorch)

Подробный справочник по слоям — в отдельной статье: [Слои PyTorch](layers.md).

Секция `layers` указывается внутри `params`:

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

Каждый слой может содержать поле `activation` для пост-активации:

```yaml
- type: "linear"
  out_features: 128
  activation: "gelu"
```

#### `constants`

Интерполяция `${var}` работает во всех полях слоёв и блоков:

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

#### `blocks` и `block_dirs`

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

Управление препроцессингом: импутация, кодирование, масштабирование.

#### `numbers`

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `impute` | `str` | `"median"` | Стратегия импутации: `mean`, `median`, `most_frequent`, `constant` |
| `scale` | `str` | `"robust"` | Масштабирование: `standard`, `robust`, `minmax`, `none` |

#### `categories`

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `encode` | `str` | `"onehot"` | Кодирование: `ordinal`, `onehot`, `target`, `none` |
| `smoothing` | `float` | `10.0` | Сглаживание для target-encoding |

#### `columns`

По-колоночные переопределения — не заданные поля наследуют глобальные стратегии выше.
`none` полностью пропускает колонку (без импутации/кодирования/масштабирования).

| Поле | Тип | Описание |
|---|---|---|
| `encode` | `str` | Переопределение кодирования: `ordinal`, `onehot`, `none` |
| `impute` | `str` | Переопределение импутации: `mean`, `median`, `most_frequent`, `constant`, `none` |
| `scale` | `str` | Переопределение масштабирования: `standard`, `robust`, `minmax`, `none` |
| `fill_value` | `float` | Значение для `impute: "constant"` |

```yaml
handling:
  numbers: { impute: "median", scale: "robust" }
  categories: { encode: "onehot" }
  columns:
    cat1: { encode: "ordinal" }       # одна колонка ordinal, остальные one-hot
    cat2: { encode: "none" }          # оставить как есть (CatBoost сам обработает)
    num1: { scale: "none" }           # без масштабирования для этой колонки
    num2: { impute: "constant", fill_value: 0 }
```

#### `sequence_token`

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `model` | `str` | `"word2vec"` | Модель эмбеддингов |
| `embedding_dim` | `int` | `64` | Размерность эмбеддинга |
| `aggregation` | `str` | `"mean"` | Стратегия агрегации: `mean`, `max`, `min`, `sum`, `nunique`, `mode`, `datetime_stats` |

#### `images`

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `processing_mode` | `str` | `"standard"` | Режим: `standard`, `patches` |
| `patch_size` | `int` | `512` | Размер патча |
| `augmentation` | `str` | `"none"` | Аугментация: `none`, `hard_noise`, `standard` |

---

### `step_config` (опционально)

Дополнительные шаги пайплайна: split, cross-encoder, pairwise similarity, text embedding.

#### `split`

```yaml
step_config:
  split:
    val_fraction: 0.2
    stratified: true
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `val_fraction` | `float` | `0.2` | Доля валидационной выборки |
| `stratified` | `bool` | `false` | Стратифицированное разбиение |

#### `cross_encoder`

Cross-encoder для скоринга текстовых пар (sentence-transformers).

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

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `model_name` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | HuggingFace модель |
| `pairs` | `list[list[str]]` | `[]` | Пары текстовых колонок |
| `batch_size` | `int` | `64` | Размер батча |
| `max_length` | `int \| None` | `None` | Макс. длина токенов |
| `predict_chunk` | `int` | `512` | Строк на вызов predict |
| `drop_source` | `bool` | `false` | Удалить исходные колонки |
| `device` | `str \| None` | `None` | `"cuda"` / `"cpu"` / `null` (авто) |
| `cache_dir` | `str \| None` | `None` | Директория кэша (SHA-256 ключ) |

#### `pairwise_similarity`

Bi-encoder косинусная схожесть для текстовых пар.

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

Эмбеддинги текста через sentence-transformers.

```yaml
step_config:
  text_embedding:
    model_name: "all-MiniLM-L6-v2"
    max_length: 128
    batch_size: 64
    cache_dir: "./cache/embeddings"
```

#### `save_predictions`

Формат файла с предиктами. `id_col` задаёт колонку с id строк в сабмишене.

```yaml
step_config:
  save_predictions:
    id_col: "id"              # колонка-ид строк предикта (default: нет)
    target_col: "prediction"  # название колонки с предиктами
    sep: ","                  # разделитель CSV (auto: "," / "\t" для .tsv)
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `id_col` | `str \| None` | `None` | Колонка для id; без неё используется индекс |
| `target_col` | `str` | `"prediction"` | Название колонки с предиктами |
| `sep` | `str \| None` | `None` | Разделитель (`","` или `"\t"` для `.tsv`) |

#### `evaluate`

Метрики на holdout-сплите (или OOF для мета-провайдеров).

```yaml
step_config:
  evaluate:
    metrics: ["f1", "accuracy"]
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `metrics` | `list[str] \| None` | `None` | Подмножество `f1`, `f1_macro`, `accuracy`, `rmse`, `auc`, `logloss`, `mae`, `mape`; `None` = все применимые |

#### `feature_generation`

Count / frequency-кодирование категориальных колонок с высокой кардинальностью.

```yaml
step_config:
  feature_generation:
    count_encode: true
    freq_encode: true
    max_unique: 1000
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `count_encode` | `bool` | `true` | Добавлять count-кодированные колонки |
| `freq_encode` | `bool` | `true` | Добавлять frequency-кодированные колонки |
| `max_unique` | `int` | `1000` | Кодировать колонки не более чем с этим числом уникальных значений |

#### `column_length`

Добавляет колонки `{col}_len` с длиной строк.

```yaml
step_config:
  column_length:
    columns: ["title", "description"]
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `columns` | `list[str]` | `[]` | Колонки для вычисления длины |

#### `diff_ratio`

Добавляет колонки `{a}_{b}_diff_ratio` — `(a - b) / max(|a|, |b|)` для каждой пары.

```yaml
step_config:
  diff_ratio:
    pairs: [["price_a", "price_b"]]
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `pairs` | `list[list[str]]` | `[]` | Пары числовых колонок |

#### `same_value`

Добавляет флаги `{a}_{b}_same` — 1, когда обе колонки равны.

```yaml
step_config:
  same_value:
    pairs: [["cat_a", "cat_b"]]
    drop_source: false
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `pairs` | `list[list[str]]` | `[]` | Пары колонок для сравнения |
| `drop_source` | `bool` | `false` | Удалять исходные колонки после сравнения |

#### `text_overlap`

Фичи пересечения для пар текстовых колонок (char n-grams / token overlap).

```yaml
step_config:
  text_overlap:
    pairs: [["title_a", "title_b"]]
    char_ngram: 3
    drop_source: false
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `pairs` | `list[list[str]]` | `[]` | Пары текстовых колонок |
| `char_ngram` | `int` | `0` | Размер char n-gram для overlap (`0` = только токены) |
| `token_pattern` | `str` | `\w+` | Регэксп для токенизации |
| `drop_source` | `bool` | `false` | Удалять исходные колонки после подсчёта |

#### `reference_join`

Объединение данных с reference-таблицами (например, метаданные товаров),
опционально фильтруя тренировочные метки.

```yaml
step_config:
  reference_join:
    joins:
      - reference_path: "items.parquet"
        left_on: "leftItemId"
        suffix: "_left"
    keep_labels: ["no_relevant", "relevant"]
```

| Поле | Тип | Описание |
|---|---|---|
| `joins` | `list[dict]` | Конфиги join'ов: `reference_path`, `left_on`, `suffix` |
| `keep_labels` | `list[str] \| None` | Если задано — строки трейна с другими метками удаляются |

Сокращённая запись (один join, без `keep_labels`):

```yaml
step_config:
  reference_join:
    reference_path: "items.parquet"
    left_on: "leftItemId"
    suffix: "_left"
```

---

### `trainer`

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `output_dir` | `str` | `"./outputs"` | Директория для сохранения артефактов |
| `output_file` | `str` | `"./outputs/submission.csv"` | Файл для предикшнов |
| `fallback_on_error` | `bool` | `true` | Продолжить при ошибке (заглушка) |

---

### `environment`

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `auto_install` | `bool` | `true` | Автоустановка缺失ющих бэкендов (`uv add` / `pip install`) |

---

### `evaluator` (опционально)

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `name` | `str` | `"holdout"` | `"holdout"` или `"cv"` |
| `params` | `dict` | `{}` | Параметры (`val_fraction`, `stratified`, `n_folds`) |
| `metrics` | `list[str]` | `[]` | Метрики: `f1`, `accuracy`, `rmse`, `auc`, `logloss`, `mae`, `mape` |

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

## Полный справочник enum'ов

### `FeatureType`

| Значение | Описание |
|---|---|
| `"number"` | Числовая колонка |
| `"category"` | Категориальная колонка |
| `"datetime"` | Дата/время |
| `"image_path"` | Путь к изображению |
| `"sequence_token"` | Токенизированная последовательность |
| `"text"` | Текст |
| `"unknown"` | Неопределённый тип |

### `TaskType`

| Значение | Описание |
|---|---|
| `"classification"` | Бинарная классификация |
| `"regression"` | Регрессия |
| `"uplift"` | Uplift-моделирование |
| `"multitask"` | Мультитаск (несколько таргетов) |

### `ModelObjective`

| Значение |
|---|
| `"MAPE"`, `"F1"`, `"accuracy"`, `"precision"`, `"recall"`, `"AUC"`, `"logloss"`, `"RMSE"` |

### `ImputeStrategy`

`"mean"`, `"median"`, `"most_frequent"`, `"constant"`

### `ScaleStrategy`

`"standard"`, `"robust"`, `"minmax"`, `"none"`

### `EncodeStrategy`

`"target"`, `"ordinal"`, `"onehot"`, `"none"`

### `AggregationStrategy`

`"mean"`, `"max"`, `"min"`, `"sum"`, `"nunique"`, `"mode"`, `"datetime_stats"`
