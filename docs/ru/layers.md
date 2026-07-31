# Слои PyTorch

Справочник всех встроенных слоёв для описания архитектуры нейросети.
Слои задаются через `params.layers` внутри узла `model` в YAML или программно.

```yaml
model:
  - provider: "pytorch"
    params:
      layers:
        - type: "linear"
          out_features: 128
        - type: "relu"
        - type: "linear"
      epochs: 10
      batch_size: 32
      lr: 0.001
```

## Базовые слои

### `linear`

Полносвязный слой (`nn.Linear`).

```yaml
- type: "linear"
  out_features: 128
  bias: true           # опционально, по умолчанию true
  activation: "gelu"   # опционально, пост-активация
```

Если `in_features` не указан, берётся из предыдущего слоя (или `input_size`).

### Активации

```yaml
- type: "relu"
- type: "tanh"
- type: "sigmoid"
- type: "gelu"
```

Все 4 активации доступны и как отдельные слои, и как пост-активация через поле `activation`.

### `dropout`

```yaml
- type: "dropout"
  p: 0.5
```

### Нормализация

```yaml
- type: "batch_norm1d"
  num_features: 128    # опционально, по умолчанию из prev_dim
  eps: 1e-5

- type: "layer_norm"
  num_features: 128
  eps: 1e-5
```

---

## Свёрточные слои (1D)

### `conv1d`

```yaml
- type: "conv1d"
  out_channels: 64
  kernel_size: 3
  stride: 1
  padding: 0
```

### Пулдинг

```yaml
- type: "max_pool1d"
  kernel_size: 2
  stride: 2           # опционально, по умолчанию kernel_size

- type: "avg_pool1d"
  kernel_size: 2
  stride: 2
```

---

## Операции с формой

```yaml
- type: "flatten"
- type: "identity"
- type: "unsqueeze"
  dim: 1
- type: "squeeze"
  dim: 1
```

---

## Трансформер

### `positional_encoding`

Синусоидальное позиционное кодирование.

```yaml
- type: "positional_encoding"
  d_model: 256        # опционально, по умолчанию prev_dim
  max_len: 5000
```

### `multihead_attention`

Self-attention (без кросс-аттеншна).

```yaml
- type: "multihead_attention"
  embed_dim: 256      # опционально, по умолчанию prev_dim
  num_heads: 4
  dropout: 0.1
```

### `transformer_encoder_layer`

Один слой TransformerEncoder.

```yaml
- type: "transformer_encoder_layer"
  d_model: 256
  nhead: 4
  dim_feedforward: 2048
  dropout: 0.1
  activation: "relu"
```

### `transformer_encoder`

Стэк из `num_layers` энкодеров.

```yaml
- type: "transformer_encoder"
  d_model: 256
  nhead: 4
  num_layers: 2
  dim_feedforward: 2048
  dropout: 0.1
  activation: "relu"
```

---

## Графовые операции (Graph Ops)

Графовый режим включается автоматически, если хотя бы один слой содержит поле `inputs`
или использует графовые операторы (`add`, `concat`, `take_last`, etc.).

В графовом режиме каждый слой может ссылаться на выходы предыдущих слоёв по имени.

### `add`

Покомпонентное сложение двух или более тензоров.

```yaml
- type: "add"
  inputs: ["layer_a", "layer_b"]
```

### `concat`

Конкатенация по измерению `dim` (по умолчанию 1).

```yaml
- type: "concat"
  inputs: ["branch_1", "branch_2"]
  dim: 1
```

### Операции извлечения (3D → 2D)

Для последовательностей (batch, seq_len, features):

```yaml
- type: "take_last"     # берёт последний токен x[:, -1, :]
- type: "take_first"    # берёт первый токен x[:, 0, :]
- type: "mean_pool"      # среднее по seq_len
- type: "max_pool_seq"   # максимум по seq_len
- type: "sum_pool_seq"   # сумма по seq_len
```

### `inputs` — графовый режим для любого слоя

Любой слой может явно указать входы:

```yaml
layers:
  - type: "linear"
    out_features: 64
    name: "shared"
  - type: "linear"
    out_features: 32
    name: "branch_a"
    inputs: ["shared"]
  - type: "linear"
    out_features: 32
    name: "branch_b"
    inputs: ["shared"]
  - type: "concat"
    inputs: ["branch_a", "branch_b"]
  - type: "linear"
    out_features: 10
```

Вход `"@input"` ссылается на исходные данные.

---

## Блоки

Блоки — переиспользуемые фрагменты архитектуры, определённые в YAML.

### Встроенные блоки

Пакет поставляется со встроенными блоками в `src/mlcombine/models/blocks/`:

| Файл | Блок | Описание |
|---|---|---|
| `ffn.yaml` | `ffn` | Linear → activation → Linear (params: d_model, ff_dim, activation, dropout) |
| `attention.yaml` | `multi_head_attn` | MHA → residual add → layer norm |
| `transformer.yaml` | `encoder_block` | Полный энкодер (MHA → add/norm → FFN → add/norm) |
| `pooling.yaml` | `global_avg_pool`, `global_max_pool`, `take_last`, `take_first` | Пулдинг |
| `positional.yaml` | `positional_encoding` | Позиционное кодирование |

### Использование блоков

```yaml
layers:
  - type: "block"
    ref: "encoder_block"
    repeat: 2
    params:
      d_model: 256
      nhead: 8
```

Поле `params` переопределяет параметры блока по умолчанию.

### Кастомные блоки

Создайте YAML-файл:

```yaml
# my_blocks/my_block.yaml
blocks:
  my_custom_block:
    params:
      hidden_dim: 128
      dropout: 0.1
    layers:
      - type: "linear"
        out_features: "${hidden_dim}"
        name: "expand"
      - type: "gelu"
      - type: "dropout"
        p: "${dropout}"
      - type: "linear"
        out_features: "@input"
        name: "project"
```

Укажите директорию в конфиге:

```yaml
model:
  block_dirs:
    - "./my_blocks"
```

### Интерполяция `${var}`

```yaml
model:
  constants:
    d_model: 256
  layers:
    - type: "linear"
      out_features: "${d_model}"
```

Работает во всех полях слоёв и блоков.

### `include` в блоках

YAML-файлы блоков могут включать другие YAML-файлы через `include`:

```yaml
include:
  - "./shared/attention.yaml"
  - "./shared/ffn.yaml"

blocks:
  my_encoder:
    layers:
      - type: "block"
        ref: "multi_head_attn"
      - type: "block"
        ref: "ffn"
```

---

## Sequential vs Graph

### Sequential (fast path)

Используется, когда нет графовых признаков (ни `inputs`, ни графовых операторов).

```yaml
layers:
  - type: "linear"
  - type: "relu"
  - type: "linear"
```

Собирается в `nn.Sequential`. Максимальная производительность.

### Graph (DAG path)

Автоматически включается при наличии `inputs` или графовых операторов.

```yaml
layers:
  - type: "linear"
    name: "shared"
  - type: "add"
    inputs: ["shared", "_input"]
```

Собирается в `_LayerGraph` — кастомный `nn.Module` с кэшем промежуточных тензоров.

---

## Все активации (доступны через поле `activation`)

| Имя | Класс |
|---|---|
| `relu` | `nn.ReLU` |
| `tanh` | `nn.Tanh` |
| `sigmoid` | `nn.Sigmoid` |
| `gelu` | `nn.GELU` |
| `leaky_relu` | `nn.LeakyReLU` |
| `elu` | `nn.ELU` |
| `selu` | `nn.SELU` |
| `silu` | `nn.SiLU` |
| `mish` | `nn.Mish` |
| `softmax` | `nn.Softmax` (требует `dim`) |

```yaml
- type: "linear"
  out_features: 10
  activation: "silu"
```

Для `softmax`:

```yaml
- type: "linear"
  out_features: 10
  activation: "softmax"
  # dim по умолчанию 1
```

---

## Полный пример: трансформер для текстовой классификации

```yaml
version: "1.0"
data:
  train_df: "train.csv"
  test_df: "test.csv"
  target_col: "label"
  task_type: classification
model:
  - provider: "pytorch"
    params:
      objective: "F1"
      epochs: 20
      batch_size: 32
      lr: 0.0005
      constants:
        d_model: 128
        nhead: 4
        ff_dim: 512
        dropout: 0.1
      layers:
        - type: "linear"
          out_features: "${d_model}"
        - type: "positional_encoding"
          d_model: "${d_model}"
          max_len: 512
        - type: "transformer_encoder"
          d_model: "${d_model}"
          nhead: "${nhead}"
          num_layers: 3
          dim_feedforward: "${ff_dim}"
          dropout: "${dropout}"
        - type: "mean_pool"
        - type: "linear"
          out_features: 64
          activation: "gelu"
        - type: "dropout"
          p: "${dropout}"
        - type: "linear"
handling:
  numbers:
    impute: median
    scale: standard
trainer:
  output_dir: "./outputs"
  output_file: "./outputs/submission.csv"
```

## Полный пример: ResNet-подобная архитектура (DAG)

Графовый режим с skip-connection через `add`:

```yaml
model:
  - provider: "pytorch"
    params:
      constants:
        dim: 128
      layers:
        - type: "linear"
          out_features: "${dim}"
          name: "proj"
        - type: "linear"
          out_features: "${dim}"
          name: "branch_a"
          inputs: ["proj"]
        - type: "relu"
          name: "act_a"
          inputs: ["branch_a"]
        - type: "linear"
          out_features: "${dim}"
          name: "branch_b"
          inputs: ["act_a"]
        - type: "add"
          inputs: ["proj", "branch_b"]
        - type: "relu"
        - type: "linear"
          out_features: 64
        - type: "relu"
        - type: "linear"
```

## Полный пример: мульти-вход (concat двух веток)

```yaml
model:
  - provider: "pytorch"
    params:
      constants:
        d: 64
      layers:
        - type: "linear"
          out_features: "${d}"
          name: "branch_num"
        - type: "relu"
          name: "act_num"
          inputs: ["branch_num"]
        - type: "linear"
          out_features: "${d}"
          name: "branch_cat"
        - type: "relu"
          name: "act_cat"
          inputs: ["branch_cat"]
        - type: "concat"
          inputs: ["act_num", "act_cat"]
          dim: 1
        - type: "linear"
          out_features: 64
        - type: "relu"
        - type: "linear"
```
