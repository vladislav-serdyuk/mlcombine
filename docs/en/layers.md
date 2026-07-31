# PyTorch Layers

Reference of all built-in layer types for describing neural network architectures.
Layers are specified via `params.layers` inside a `model` node in YAML or programmatically.

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

## Core Layers

### `linear`

Fully connected layer (`nn.Linear`).

```yaml
- type: "linear"
  out_features: 128
  bias: true           # optional, default true
  activation: "gelu"   # optional, post-activation
```

If `in_features` is omitted, it is taken from the previous layer (or `input_size`).

### Activations

```yaml
- type: "relu"
- type: "tanh"
- type: "sigmoid"
- type: "gelu"
```

All 4 activations are available both as standalone layers and as post-activation
via the `activation` field on any layer.

### `dropout`

```yaml
- type: "dropout"
  p: 0.5
```

### Normalization

```yaml
- type: "batch_norm1d"
  num_features: 128    # optional, defaults to prev_dim
  eps: 1e-5

- type: "layer_norm"
  num_features: 128
  eps: 1e-5
```

---

## Convolutional Layers (1D)

### `conv1d`

```yaml
- type: "conv1d"
  out_channels: 64
  kernel_size: 3
  stride: 1
  padding: 0
```

### Pooling

```yaml
- type: "max_pool1d"
  kernel_size: 2
  stride: 2           # optional, defaults to kernel_size

- type: "avg_pool1d"
  kernel_size: 2
  stride: 2
```

---

## Shape Operations

```yaml
- type: "flatten"
- type: "identity"
- type: "unsqueeze"
  dim: 1
- type: "squeeze"
  dim: 1
```

---

## Transformer

### `positional_encoding`

Sinusoidal positional encoding.

```yaml
- type: "positional_encoding"
  d_model: 256        # optional, defaults to prev_dim
  max_len: 5000
```

### `multihead_attention`

Self-attention (no cross-attention).

```yaml
- type: "multihead_attention"
  embed_dim: 256      # optional, defaults to prev_dim
  num_heads: 4
  dropout: 0.1
```

### `transformer_encoder_layer`

Single TransformerEncoder layer.

```yaml
- type: "transformer_encoder_layer"
  d_model: 256
  nhead: 4
  dim_feedforward: 2048
  dropout: 0.1
  activation: "relu"
```

### `transformer_encoder`

Stack of `num_layers` encoders.

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

## Graph Operations

Graph mode is enabled automatically when any layer has an `inputs` field
or uses graph operators (`add`, `concat`, `take_last`, etc.).

In graph mode, each layer can reference named outputs of previous layers.

### `add`

Element-wise sum of two or more tensors.

```yaml
- type: "add"
  inputs: ["layer_a", "layer_b"]
```

### `concat`

Concatenation along dimension `dim` (default 1).

```yaml
- type: "concat"
  inputs: ["branch_1", "branch_2"]
  dim: 1
```

### Extraction ops (3D → 2D)

For sequences (batch, seq_len, features):

```yaml
- type: "take_last"     # x[:, -1, :]
- type: "take_first"    # x[:, 0, :]
- type: "mean_pool"      # mean over seq_len
- type: "max_pool_seq"   # max over seq_len
- type: "sum_pool_seq"   # sum over seq_len
```

### `inputs` — graph mode for any layer

Any layer can explicitly declare its inputs:

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

The special input `"@input"` refers to the raw input data.

---

## Blocks

Blocks are reusable architecture fragments defined in YAML.

### Built-in Blocks

The package ships with built-in blocks in `src/mlcombine/models/blocks/`:

| File | Block | Description |
|---|---|---|
| `ffn.yaml` | `ffn` | Linear → activation → Linear (params: d_model, ff_dim, activation, dropout) |
| `attention.yaml` | `multi_head_attn` | MHA → residual add → layer norm |
| `transformer.yaml` | `encoder_block` | Full encoder (MHA → add/norm → FFN → add/norm) |
| `pooling.yaml` | `global_avg_pool`, `global_max_pool`, `take_last`, `take_first` | Pooling ops |
| `positional.yaml` | `positional_encoding` | Positional encoding |

### Using Blocks

```yaml
layers:
  - type: "block"
    ref: "encoder_block"
    repeat: 2
    params:
      d_model: 256
      nhead: 8
```

The `params` field overrides the block's default parameters.

### Custom Blocks

Create a YAML file:

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

Then reference it:

```yaml
model:
  block_dirs:
    - "./my_blocks"
```

### `${var}` Interpolation

```yaml
model:
  constants:
    d_model: 256
  layers:
    - type: "linear"
      out_features: "${d_model}"
```

Works in all layer and block fields.

### `include` in Blocks

Block YAML files can include other YAML files:

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

Used when there are no graph features (no `inputs`, no graph operators).

```yaml
layers:
  - type: "linear"
  - type: "relu"
  - type: "linear"
```

Built as `nn.Sequential`. Maximum performance.

### Graph (DAG path)

Automatically enabled when any layer has `inputs` or uses graph operators.

```yaml
layers:
  - type: "linear"
    name: "shared"
  - type: "add"
    inputs: ["shared", "_input"]
```

Built as `_LayerGraph` — a custom `nn.Module` with a forward-pass cache.

---

## All Activations (available via the `activation` field)

| Name | Class |
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
| `softmax` | `nn.Softmax` (requires `dim`) |

```yaml
- type: "linear"
  out_features: 10
  activation: "silu"
```

For `softmax`:

```yaml
- type: "linear"
  out_features: 10
  activation: "softmax"
  # dim defaults to 1
```

---

## Full Example: Transformer for Text Classification

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

## Full Example: ResNet-like Architecture (DAG)

Graph mode with skip-connection via `add`:

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

## Full Example: Multi-Input (concat two branches)

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
