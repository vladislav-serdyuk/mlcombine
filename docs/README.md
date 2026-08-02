# mlcombine

Декларативный Low-Code/No-Code фреймворк для автоматизации ML-соревнований.

```bash
pip install mlcombine
mlcombine train --config config.yaml
```

## Документация

| Language | Ссылка |
|---|---|
| Русский | [config.yaml](ru/config.md) · [Расширения](ru/extensions.md) · [Архитектура](ru/architecture.md) · [Слои PyTorch](ru/layers.md) |
| English | [config.yaml](en/config.md) · [Extensions](en/extensions.md) · [Architecture](en/architecture.md) · [PyTorch Layers](en/layers.md) |

## Примеры YAML-конфигов

| Use-case | Файл |
|---|---|
| Базовый baseline (sklearn) | [titanic_baseline.yaml](../examples/configs/titanic_baseline.yaml) |
| Табличные данные + Optuna-тюнинг | [tabular_cv.yaml](../examples/configs/tabular_cv.yaml) |
| Ensemble CatBoost + LightGBM | [ensemble_blend.yaml](../examples/configs/ensemble_blend.yaml) |
| PyTorch кастомная архитектура | [pytorch_layers.yaml](../examples/configs/pytorch_layers.yaml) |
| Загрузка датасета по URL | [remote_data.yaml](../examples/configs/remote_data.yaml) |

## Что нового (2026-06)

- **Model DAG** — `model` теперь список `ModelNode` с поддержкой зависимостей (`model`/`models`), топологической сортировкой
- **ModelBlueprint** — ленивое описание модели вместо прямого вызова провайдера
- **Meta-провайдеры:** `fold_ensemble` (K-Fold + OOF target encoding), `ensemble` (взвешенное усреднение), `stacking` (мета-модель на OOF)
- **Tuner** — Optuna-тюнинг с `search_space` в YAML
- **CrossEncoderStep / PairwiseSimilarityStep** — скоринг текстовых пар (sentence-transformers)
- **URL-датасеты** — `train_df`/`test_df` принимают HTTP/HTTPS URL, автоматическое скачивание и распаковка архивов
- **Кэширование эмбеддингов** — TextEmbeddingStep сохраняет результаты в `cache_dir`
- **`from __future__ import annotations`** — во всех source-файлах
