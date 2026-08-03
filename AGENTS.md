# AGENTS.md — команды и контекст для opencode

## Пакетный менеджер: uv (только!)
Все установки — через `uv add` / `uv add --dev`, синхронизация — `uv sync` / `uv lock`.
**НЕ использовать pip** — `uv sync` удалит pip-установленные пакеты (так терялись catboost/torch/pyarrow).
Dev-зависимости (catboost, torch, pyarrow, pre-commit, pytest, ruff, mypy) зафиксированы в `uv.lock`.
Запуск без активации venv: `.venv/bin/python -m ...` (ruff/mypy/pytest не в PATH).

## Тестирование
```bash
.venv/bin/python -m pytest tests/ -q           # 257 passed, 1 skip (sklearn text)
.venv/bin/python -m pytest tests/test_registry.py -q
.venv/bin/python -m pytest tests/test_factory.py -q
.venv/bin/python -m pytest tests/test_tuner.py -q
.venv/bin/python -m pytest tests/ -q -W error  # предупреждения должны отсутствовать
```

## Линтер / type checker / pre-commit
```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m mypy --strict src          # ВАЖНО: только src, в tests pre-existing ошибки
.venv/bin/pre-commit run --all-files
```
`.pre-commit-config.yaml`: ruff (pinned v0.15.16, `--fix`), ruff-format, mypy local hook.
Хук mypy обязан быть `local` с `entry: .venv/bin/python -m mypy --strict src` — изолированный env
pre-commit не видит зависимостей проекта.

## Docker
`Dockerfile.test` — по гайду uv:
- `COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/` (версию uv поднимать вручную)
- split-слои: `uv sync --locked --group dev --no-install-project` до `COPY . .`, повторный sync после
- cache mount `/root/.cache/uv`, `UV_LINK_MODE=copy`, `UV_COMPILE_BYTECODE=1`
- CMD: `uv run pytest tests/ -q`

```bash
docker build -f Dockerfile.test -t mlcombine-test . && docker run --rm mlcombine-test
```
Без GPU в контейнере: 242 passed, 6 skipped (GPU-тест скипается через реальный fit на GPU,
не конструктор — конструктор не ловит "CUDA driver version is insufficient").

## Архитектура
`MLCombineConfig.model` — `list[ModelNode]` (DAG). Каждый node: `id` (для ссылок), `provider`,
`model`/`models` (ссылки на другие node'ы), `params`. Последний node — финальная модель.

```yaml
model:
  - provider: "catboost"
    params: { ... }
  - provider: "tuner"
    params: { ... }
```

Все провайдеры (catboost, sklearn, lightgbm, pytorch, hybrid, t_learner, s_learner, tuner,
ensemble, stacking, fold_ensemble) зарегистрированы через `@registry.model_provider("name")`
и являются **функциями**: `def provider_fn(backbone=..., task_type=..., objective=..., num_classes=None, input_size=None, **params)`.
Meta-провайдеры получают `model=` или `models=`. `ModelBuilder.build_all(nodes, ...)` строит
node'ы с топологической сортировкой по ссылкам.

### Tuner (optuna)
В `params`: `target_provider`, `target_params`, `search_space`, `n_trials` (default 50).
```yaml
model:
  - provider: "tuner"
    params:
      n_trials: 50
      target_provider: "catboost"
      target_params:
        backbone: "gradient_boosting"
        iterations: 3000
      search_space:
        depth: { type: "int", low: 4, high: 10 }
        learning_rate: { type: "float", low: 0.01, high: 0.3, log: true }
```

### Meta-провайдеры
- `fold_ensemble` — K-fold обучение вложенной модели, OOF-safe target encoding
  (`target_encode_cols`, `target_encode_smoothing` default 10.0), сохраняет
  `fold_models_`/`oof_preds_`; predict — усреднение; `vote: "hard"|"soft"` (default `"hard"`,
  hard дал лучший скор на практике). Если задан `group_col` — `StratifiedGroupKFold`.
- `ensemble` — взвешенное усреднение predict/predict_proba (`weights`), fit — no-op.
- `stacking` — дорогой: N базовых моделей + мета-модель на OOF (6 CatBoost fit'ов — десятки минут).
- `EvaluateStep` — метрики (f1, accuracy, rmse, auc, logloss, mae, mape) на OOF или holdout.
- Multi-model: `PipelineArtifacts.models: dict[str, object]` — `model_<id>.joblib`.
- `TextEmbeddingStep` кэширует эмбеддинги в `cache_dir` (ключ SHA-256), авто-вставляется
  перед `CreateModelStep`.

## Известные проблемы
- Метрики в реестре (`@registry.metric(name, direction=...)`) хранят `MetricDirection`
  (`MINIMIZE`/`MAXIMIZE`) — источник истины для направления optuna-тунера
  (`TunerWrapper._study_direction`). Fallback — `_MINIMIZE_METRICS` в `tuner.py`
  (loss-метрики без direction). `registry.metric.get(name)` возвращает только
  `(fn, kwargs)`; метаданные — через `get_meta(name)`. Метрики в `core/metric.py`
  зарегистрированы с direction; при `registry.metric.clear()`/reload — перерегистрируются.
- Ссылки `model`/`models` резолвятся по ключу `node.id or node.provider`
  (`ModelBuilder._resolve_deps`, `_topological_sort`). Если в конфиге два node'а
  с одним провайдером — ссылка по имени провайдера молча резолвится в последний
  построенный (last-wins), поэтому ссылающимся node'ам нужны уникальные `id`.
- `sklearn._rf` использует `rf_kwargs.update(kwargs)` вместо
  `n_estimators=100` в сигнатуре — иначе optuna-тюнинг конфликтует.
- GPU-режим CatBoost: достаточно добавить `gpu: true` в `params` или
  `target_params`. Провайдер вызовет `params.pop("gpu")` и установит
  `task_type="GPU"` в конструкторе CatBoost.
- CatBoost `objective` в `params` может быть строкой (напр. `"MAE"`) — она
  задаёт `loss_function` и `eval_metric` (если не заданы явно). Строкой
  считается только raw str (`type(x) is str`), StrEnum — это метаинформация
  для дефолтных eval_metric. Regression: RMSE/MAE/MAPE/Huber/Quantile;
  classification: Logloss/CrossEntropy/MultiClass (валидные для n_classes).
- NaN в text колонках: CatBoost сам работает с NaN в text features.
  На predict `_ensure_text_as_str` / `_cast_text_features_as_str` чинят
  позиционные text feature индексы. `_predict_with_text_fallback` —
  fallback на blanket str cast при dtype error.
- Column mismatch train vs predict: `CatBoostWrapper._align_columns()` реордерит
  DataFrame по `feature_names_` перед predict, чтобы позиционные text feature
  индексы CatBoost совпадали. `EvaluateStep._run_holdout/_run_insample` тоже
  дропают `drop_columns` для консистентности с `ModelFitStep`.
- Predict-mode EncodeScaleStep: если `encode: none` и `scale: none` — артефакты
  не загружаются (их просто нет). Если encode/scale активны а артефактов нет —
  `RuntimeError`. `_fit_on_predict_data` удалён (был data leakage).
- Per-column handling (`handling.columns`): значения `ColumnHandlingConfig` —
  partial overrides, `none` пропускает колонку. Только encode/impute имеют
  per-column логику в `EncodeScaleStep`/`ImputeStep`; scale тоже per-column
  (`self.scalers_: dict[str, scaler]`, артефакт `scalers.joblib`, вместо
  одного `scaler_`/`scaler.joblib`). Целевая колонка исключается из scaling
  и imputation.
- `data.id_col` — глобальная колонка-ид: исключается из preprocessing
  (ImputeStep/EncodeScaleStep), из фич модели (ModelFitStep, EvaluateStep,
  ModelPredictStep) и сохраняется в сабмишен. `AlignFeaturesStep` (predict)
  кладёт её в `prediction_ids` ДО `reindex(columns=feature_names)`, иначе
  колонка теряется.
- Default кодирование — `onehot` (`EncodeStrategy.ONEHOT` в конфиге),
  не `ordinal`!
- PyTorch: использовать `torch.tensor(...)`, а НЕ `torch.from_numpy(...)` —
  from_numpy требует writable numpy array (warning при read-only данных, `-W error` ловит).
- Тесты скипаются по модульным флагам `HAS_*` (catboost/torch/lightgbm) —
  детект через try-import, для GPU — обязателен реальный fit.

## Документация
`README.md` (English) и `README-RU.md` (Russian) — дублируют друг друга по структуре.
При изменении API обновлять оба.
