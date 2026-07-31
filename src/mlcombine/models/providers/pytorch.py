"""PyTorch provider — builds dynamic architectures from YAML layer descriptions.

Supports MLP (Linear, ReLU, Dropout, BatchNorm), CNN (Conv1d, pooling),
Transformer (encoder, encoder layer, multi-head attention, positional encoding),
and arbitrary ``nn.Module`` chains via ``build_sequential()``.

When no ``layers`` config is provided, a default 3-layer MLP is used
(backward compatible with the old ``SimpleNN``).
"""

from __future__ import annotations

import math
import re
import logging
from pathlib import Path
from typing import Any, Callable, Self, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import yaml

from mlcombine.core.tensor import UnifiedTensor
from mlcombine.core.enums import ModelObjective, TaskType
from mlcombine.core.registry import registry
from mlcombine.core.protocols import SupportedModel

logger = logging.getLogger(__name__)

# ── constants interpolation ──────────────────────────────────────────

_CONST_RE = re.compile(r"\$\{(\w+)\}")


def _interpolate_value(value: Any, constants: dict[str, Any]) -> Any:
    """Recursively interpolate ``${var}`` placeholders with ``constants``."""
    if isinstance(value, str):

        def _replace(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in constants:
                raise ValueError(f"Unknown constant: {key!r}. Available: {sorted(constants)}")
            val = constants[key]
            return str(val)

        return _CONST_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_value(v, constants) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(v, constants) for v in value]
    return value


def interpolate_constants(
    obj: Any,
    constants: dict[str, Any] | None = None,
) -> Any:
    """Interpolate ``${var}`` placeholders throughout *obj* (dict, list, str, …)."""
    if constants is None:
        return obj
    return _interpolate_value(obj, constants)


# ── block loading ────────────────────────────────────────────────────


def _load_blocks_from_file(filepath: str, visited: set[str] | None = None) -> dict[str, Any]:
    """Load all blocks from a YAML file, recursing into ``include:`` entries."""
    visited_set: set[str]
    if visited is None:
        visited_set = set()
    else:
        visited_set = visited

    resolved = str(Path(filepath).resolve())
    if resolved in visited_set:
        raise ValueError(f"Circular include detected: {filepath}")
    visited_set.add(resolved)

    with open(resolved) as f:
        raw = yaml.safe_load(f) or {}

    blocks: dict[str, Any] = dict(raw.get("blocks", {}))
    base_dir = Path(resolved).parent

    for include_path in raw.get("include", []):
        included_file = str((base_dir / include_path).resolve())
        included_blocks = _load_blocks_from_file(included_file, visited_set)
        for name, block in included_blocks.items():
            if name not in blocks:
                blocks[name] = block

    return blocks


def load_blocks(
    blocks_raw: dict[str, Any] | None,
    block_dirs: list[str] | None,
    config_base_path: Path | None,
) -> dict[str, Any]:
    """Load blocks: built-in → *block_dirs* → inline (highest priority)."""
    all_blocks: dict[str, Any] = {}

    # 1. Built-in blocks from package
    builtin_dir = Path(__file__).resolve().parent / "blocks"
    if builtin_dir.is_dir():
        for f in sorted(builtin_dir.glob("*.yaml")):
            try:
                all_blocks.update(_load_blocks_from_file(str(f)))
            except yaml.YAMLError, OSError:
                logger.warning("Failed to load built-in block file %s", f, exc_info=True)

    # 2. User block_dirs
    if config_base_path and block_dirs:
        for d in block_dirs:
            resolved = (config_base_path / d).resolve()
            if resolved.is_dir():
                for f in sorted(resolved.glob("*.yaml")):
                    try:
                        all_blocks.update(_load_blocks_from_file(str(f)))
                    except yaml.YAMLError, OSError:
                        logger.warning("Failed to load block file %s", f, exc_info=True)

    # 3. Inline blocks (highest priority)
    if blocks_raw:
        all_blocks.update(blocks_raw)

    return all_blocks


# ── block expansion ─────────────────────────────────────────────────


def _last_named_layer(layers: list[dict[str, Any]]) -> str | None:
    """Return the ``name`` of the last layer that has one, or ``None``."""
    for layer in reversed(layers):
        if "name" in layer:
            return layer["name"]  # type: ignore[no-any-return]
    return None


def _expand_single_block(
    block: dict[str, Any],
    prefix: str,
    instance_input: str | None,
    params: dict[str, Any],
    constants: dict[str, Any],
    all_blocks: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand one block *instance* into a flat list of layer configs.

    Args:
        block: The block definition (``layers`` + ``params``).
        prefix: Unique prefix for all names inside the block.
        instance_input: Name of the tensor feeding into ``@input``.
        params: Override params for this instance (already merged with block defaults).
        constants: Global constants (fallback for param resolution).
        all_blocks: Full blocks registry for resolving nested block refs.

    Returns:
        Flattened layer config list.
    """
    scope: dict[str, Any] = {**constants, **params}
    expanded: list[dict[str, Any]] = []

    # names defined inside this block (will get the prefix)
    own_names: set[str] = {layer["name"] for layer in block.get("layers", []) if "name" in layer}

    for layer in block.get("layers", []):
        layer = dict(layer)
        layer = interpolate_constants(layer, scope)

        # resolve inputs: @input → instance_input, own names → prefixed, rest unchanged
        if "inputs" in layer:
            layer["inputs"] = [instance_input if n == "@input" else f"{prefix}_{n}" if n in own_names else n for n in layer["inputs"]]

        # prefix own names
        if "name" in layer:
            layer["name"] = f"{prefix}_{layer['name']}"

        if layer.get("type") == "block":
            sub_layers = expand_blocks([layer], all_blocks, constants)
            expanded.extend(sub_layers)
        else:
            expanded.append(layer)

    return expanded


def expand_blocks(
    layers: list[dict[str, Any]],
    blocks: dict[str, Any] | None = None,
    constants: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Replace ``type: block`` layers with their expanded layer lists."""
    blocks = blocks or {}
    constants = constants or {}
    result: list[dict[str, Any]] = []
    prev_seq: str | None = None

    for layer in layers:
        if layer.get("type") == "block":
            ref: str = layer.get("ref", "")
            block = blocks.get(ref)
            if block is None:
                raise ValueError(f"Unknown block: {ref!r}. Available: {sorted(blocks)}")

            repeat: int = int(layer.get("repeat", 1))
            block_defaults: dict[str, Any] = block.get("params", {})
            instance_params: dict[str, Any] = dict(block_defaults)
            instance_params.update(layer.get("params", {}))

            for ri in range(repeat):
                pfx = ref if repeat == 1 else f"{ref}_{ri}"
                exp = _expand_single_block(
                    block,
                    pfx,
                    prev_seq,
                    instance_params,
                    constants,
                    blocks,
                )
                result.extend(exp)
                last_n = _last_named_layer(exp)
                if last_n:
                    prev_seq = last_n  # noqa: PLW2901  (already prefixed)
        else:
            result.append(layer)
            if "name" in layer:
                prev_seq = layer["name"]  # noqa: PLW2901

    return result


# ── validation ──────────────────────────────────────────────────────

_FALLBACK_OPS: set[str] = {
    "linear",
    "relu",
    "tanh",
    "sigmoid",
    "gelu",
    "dropout",
    "batch_norm1d",
    "layer_norm",
    "conv1d",
    "max_pool1d",
    "avg_pool1d",
    "flatten",
    "identity",
    "unsqueeze",
    "squeeze",
    "positional_encoding",
    "multihead_attention",
    "transformer_encoder_layer",
    "transformer_encoder",
    "add",
    "concat",
    "take_last",
    "take_first",
    "mean_pool",
    "max_pool_seq",
    "sum_pool_seq",
    "take_slice",
    "cosine_similarity",
    "fourier_encoder_layer",
    "fourier_encoder",
}


def _validate_layer_cfg(layers: list[dict[str, Any]], blocks: dict[str, Any], constants: dict[str, Any]) -> None:
    """Pre-validate a *flattened* layer list (already expanded).

    Valid types come from the global registry when torch is installed
    (built-in and custom layers), falling back to a hardcoded set otherwise.
    """
    known_ops: set[str] = set(registry.layer_builders) or _FALLBACK_OPS
    for i, layer in enumerate(layers):
        t = layer.get("type", "")
        if t not in known_ops:
            raise ValueError(f"Layer {i}: unknown type {t!r}. Available: {sorted(known_ops)}")


def validate_config(
    layers: list[dict[str, Any]],
    blocks: dict[str, Any],
    constants: dict[str, Any],
) -> None:
    """Full pre-flight validation before building the model.

    Checks:
      - All ``${var}`` in *layers* are resolvable from *constants*.
      - All ``type: block`` references exist in *blocks*.
      - All primitive layer types are known.
    """
    # 1. constants interpolation (catches missing keys)
    interpolate_constants(layers, constants)

    # 2. expand blocks (catches unknown refs)
    expanded = expand_blocks(layers, blocks, constants)

    # 3. validate expanded layer types
    _validate_layer_cfg(expanded, blocks, constants)


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ── activation registry ────────────────────────────────────────

    _ACTIVATION_MAP: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "softmax": nn.Softmax,
    }
    # Sync built-in activations into the global registry
    for _act_name, _act_cls in _ACTIVATION_MAP.items():
        registry.activation(_act_name)(_act_cls)

    def _wrap_activation(module: nn.Module, activation_name: str, dim: int | None = None) -> nn.Module:
        """Wrap *module* with an activation layer if *activation_name* is set.

        All resolutions go through the global registry — custom activations
        registered via ``@registry.activation()`` can override built-ins.
        """
        if not activation_name:
            return module
        act_cls = registry.get_activation(activation_name)
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation_name!r}. Available: {sorted(registry.activations)}")
        if activation_name == "softmax":
            return nn.Sequential(module, cast(nn.Module, act_cls(dim=dim or 1)))
        return nn.Sequential(module, cast(nn.Module, act_cls()))

    # ── graph operation modules ────────────────────────────────────

    class _Add(nn.Module):
        """Element-wise sum of multiple tensors."""

        def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
            if not tensors:
                return torch.tensor(0.0)
            return sum(tensors[1:], start=tensors[0])

    class _Concat(nn.Module):
        """Concatenate multiple tensors along *dim*."""

        def __init__(self, dim: int = 1) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
            return torch.cat(tensors, dim=self.dim)

    class _TakeLast(nn.Module):
        """Extract the last token ``x[:, -1, :]`` — 3D → 2D."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1, :]

    class _TakeFirst(nn.Module):
        """Extract the first token ``x[:, 0, :]`` — 3D → 2D."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, 0, :]

    class _MeanPool(nn.Module):
        """Average over the sequence dimension — 3D → 2D."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=1)

    class _MaxPoolSeq(nn.Module):
        """Max over the sequence dimension — 3D → 2D."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.max(dim=1).values

    class _SumPoolSeq(nn.Module):
        """Sum over the sequence dimension — 3D → 2D."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.sum(dim=1)

    # ── new DAG layers ─────────────────────────────────────────────

    class _TakeSlice(nn.Module):
        """Extract a slice along the feature dimension.

        Given a 2D tensor ``(batch, features)``, returns ``x[:, start:end]``.
        """

        def __init__(self, start: int = 0, end: int | None = None) -> None:
            super().__init__()
            self.start = start
            self.end = end

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, self.start : self.end]

    class _CosineSimilarity(nn.Module):
        """Compute cosine similarity between two tensors *a* and *b*.

        Expects two named inputs via the DAG graph and returns
        ``(batch, 1)``.
        """

        def __init__(self, dim: int = 1) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.cosine_similarity(a, b, dim=self.dim).unsqueeze(1)

    class _FourierEncoderLayer(nn.Module):
        """FNet-style encoder layer using FFT instead of self-attention.

        Applies 2D real FFT (over sequence *and* feature dims), then
        a feed-forward block — no trainable attention weights.
        """

        def __init__(
            self,
            d_model: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            act: type[nn.Module] = nn.GELU if activation == "gelu" else nn.ReLU
            self.fft_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                act(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            self.ffn_norm = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            fft_out = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
            x = self.fft_norm(x + fft_out)
            x = self.ffn_norm(x + self.ffn(x))
            return x

    class _FourierEncoder(nn.Module):
        """Stacked Fourier encoder (FNet) — a sequence of Fourier layers."""

        def __init__(
            self,
            d_model: int,
            num_layers: int = 1,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_FourierEncoderLayer(d_model, dim_feedforward, dropout, activation) for _ in range(num_layers)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    # ── small helper modules ──────────────────────────────────────

    class Unsqueeze(nn.Module):
        """Adds a dimension at position ``dim`` (e.g. for CNN sequence dim)."""

        def __init__(self, dim: int = 1) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(self.dim)

    class Squeeze(nn.Module):
        """Removes a dimension at position ``dim``."""

        def __init__(self, dim: int = 1) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(self.dim)

    # ── transformer helpers ───────────────────────────────────────

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for transformer sequence data.

        Expects ``(batch, seq_len, d_model)`` when ``batch_first=True``.
        """

        pe: torch.Tensor

        def __init__(self, d_model: int, max_len: int = 5000) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1), :]

    class _MultiheadAttention(nn.Module):
        """Self-attention wrapper that discards attention weights for Sequential."""

        def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.attn(x, x, x)[0]
            return out

    class _TransformerEncoder(nn.Module):
        """Stacked Transformer encoder layers, usable inside ``nn.Sequential``."""

        def __init__(
            self,
            d_model: int,
            nhead: int = 4,
            num_layers: int = 1,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.encoder(x)
            return out

    # ── layer registry ────────────────────────────────────────────

    def _linear(cfg: dict[str, Any], prev_dim: int, out_dim: int | None = None) -> nn.Module:  # noqa: ANN202
        return nn.Linear(
            int(cfg.get("in_features") or prev_dim),
            int(cfg.get("out_features") or out_dim or prev_dim),
            bias=cfg.get("bias", True),
        )

    def _conv1d(cfg: dict[str, Any], prev_dim: int, **_: Any) -> nn.Module:  # noqa: ANN202
        return nn.Conv1d(
            in_channels=int(cfg.get("in_channels") or prev_dim),
            out_channels=int(cfg.get("out_channels") or prev_dim),
            kernel_size=int(cfg.get("kernel_size", 3)),
            stride=int(cfg.get("stride", 1)),
            padding=int(cfg.get("padding", 0)),
        )

    def _dropout(cfg: dict[str, Any], **_: Any) -> nn.Module:  # noqa: ANN202
        return nn.Dropout(p=float(cfg.get("p", 0.5)))

    def _norm(kind: str) -> Callable[..., nn.Module]:  # noqa: ANN202
        def _build(cfg: dict[str, Any], prev_dim: int, **_: Any) -> nn.Module:
            num = int(cfg.get("num_features") or prev_dim)
            eps = cfg.get("eps", 1e-5)
            if kind == "BatchNorm1d":
                return nn.BatchNorm1d(num, eps=eps)
            return nn.LayerNorm(num, eps=eps)

        return _build

    def _pool(kind: str) -> Callable[..., nn.Module]:  # noqa: ANN202
        def _build(cfg: dict[str, Any], **_: Any) -> nn.Module:
            ks = int(cfg.get("kernel_size") or cfg.get("pool_kernel_size", 2))
            st_raw = cfg.get("stride") or cfg.get("pool_stride")
            st: int | None = int(st_raw) if st_raw is not None else None
            if st is None:
                if kind == "MaxPool1d":
                    return nn.MaxPool1d(ks)
                return nn.AvgPool1d(ks)
            if kind == "MaxPool1d":
                return nn.MaxPool1d(ks, stride=st)
            return nn.AvgPool1d(ks, stride=st)

        return _build

    _LAYER_BUILDERS: dict[str, Callable[..., nn.Module]] = {
        "linear": _linear,
        "relu": lambda cfg, **_: nn.ReLU(),
        "tanh": lambda cfg, **_: nn.Tanh(),
        "sigmoid": lambda cfg, **_: nn.Sigmoid(),
        "gelu": lambda cfg, **_: nn.GELU(),
        "dropout": _dropout,
        "batch_norm1d": _norm("BatchNorm1d"),
        "layer_norm": _norm("LayerNorm"),
        "conv1d": _conv1d,
        "max_pool1d": _pool("MaxPool1d"),
        "avg_pool1d": _pool("AvgPool1d"),
        "flatten": lambda cfg, **_: nn.Flatten(),
        "identity": lambda cfg, **_: nn.Identity(),
        "unsqueeze": lambda cfg, **_: Unsqueeze(dim=int(cfg.get("dim", 1))),
        "squeeze": lambda cfg, **_: Squeeze(dim=int(cfg.get("dim", 1))),
        # ── transformer ────────────────────────────────────────────
        "positional_encoding": lambda cfg, prev_dim, **_: _PositionalEncoding(
            d_model=int(cfg.get("d_model", prev_dim)),
            max_len=int(cfg.get("max_len", 5000)),
        ),
        "multihead_attention": lambda cfg, prev_dim, **_: _MultiheadAttention(
            embed_dim=int(cfg.get("embed_dim", prev_dim)),
            num_heads=int(cfg.get("num_heads", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
        ),
        "transformer_encoder_layer": lambda cfg, prev_dim, **_: nn.TransformerEncoderLayer(
            d_model=int(cfg.get("d_model", prev_dim)),
            nhead=int(cfg.get("nhead", 4)),
            dim_feedforward=int(cfg.get("dim_feedforward", 2048)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "relu")),
            batch_first=True,
        ),
        "transformer_encoder": lambda cfg, prev_dim, **_: _TransformerEncoder(
            d_model=int(cfg.get("d_model", prev_dim)),
            nhead=int(cfg.get("nhead", 4)),
            num_layers=int(cfg.get("num_layers", 1)),
            dim_feedforward=int(cfg.get("dim_feedforward", 2048)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "relu")),
        ),
        # ── graph operations ─────────────────────────────────────────
        "add": lambda cfg, **_: _Add(),
        "concat": lambda cfg, **_: _Concat(dim=int(cfg.get("dim", 1))),
        "take_last": lambda cfg, **_: _TakeLast(),
        "take_first": lambda cfg, **_: _TakeFirst(),
        "mean_pool": lambda cfg, **_: _MeanPool(),
        "max_pool_seq": lambda cfg, **_: _MaxPoolSeq(),
        "sum_pool_seq": lambda cfg, **_: _SumPoolSeq(),
        # ── new DAG layers ────────────────────────────────────────────
        "take_slice": lambda cfg, prev_dim, **_: _TakeSlice(
            start=int(cfg.get("start", 0)),
            end=cfg.get("end"),
        ),
        "cosine_similarity": lambda cfg, prev_dim, **_: _CosineSimilarity(
            dim=int(cfg.get("dim", 1)),
        ),
        "fourier_encoder_layer": lambda cfg, prev_dim, **_: _FourierEncoderLayer(
            d_model=int(cfg.get("d_model", prev_dim)),
            dim_feedforward=int(cfg.get("dim_feedforward", 2048)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "relu")),
        ),
        "fourier_encoder": lambda cfg, prev_dim, **_: _FourierEncoder(
            d_model=int(cfg.get("d_model", prev_dim)),
            num_layers=int(cfg.get("num_layers", 1)),
            dim_feedforward=int(cfg.get("dim_feedforward", 2048)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=str(cfg.get("activation", "relu")),
        ),
    }
    # Sync built-in layer builders into the global registry
    for _lb_name, _lb_fn in _LAYER_BUILDERS.items():
        registry.layer_builder(_lb_name)(_lb_fn)

    def _build_module(
        cfg: dict[str, Any],
        prev_dim: int,
        out_dim: int | None = None,
    ) -> nn.Module:
        """Create a module from *cfg*, wrapping with activation if present.

        All resolutions go through the global registry — custom layer builders
        registered via ``@registry.layer_builder()`` can override built-ins.
        """
        t = cfg.get("type", "")
        builder = registry.get_layer_builder(t)
        if builder is None:
            raise ValueError(f"Unknown layer type: {t!r}. Available: {sorted(registry.layer_builders)}")
        module = cast(nn.Module, builder(cfg, prev_dim=prev_dim, out_dim=out_dim))
        activation = cfg.get("activation")
        if activation:
            module = _wrap_activation(module, activation)
        return module

    _DEFAULT_LAYERS: list[dict[str, Any]] = [
        {"type": "linear", "out_features": 128},
        {"type": "relu"},
        {"type": "dropout", "p": 0.2},
        {"type": "linear", "out_features": 64},
        {"type": "relu"},
        {"type": "dropout", "p": 0.2},
        {"type": "linear"},
    ]

    def _resolve_output_dim(
        t: str,
        cfg: dict[str, Any],
        input_dim: int,
        out_dim: int | None,
        is_last: bool,
    ) -> int:
        """Compute the feature dimension after applying layer type *t*."""
        if t == "linear":
            return int(cfg.get("out_features") or (out_dim if is_last else input_dim))  # type: ignore[arg-type]
        if t == "conv1d":
            return int(cfg.get("out_channels") or input_dim)
        if t in ("transformer_encoder_layer", "transformer_encoder", "multihead_attention"):
            return int(cfg.get("d_model") or cfg.get("embed_dim") or input_dim)
        if t == "positional_encoding":
            return int(cfg.get("d_model") or input_dim)
        if t == "take_slice":
            end: Any = cfg.get("end")
            start: int = int(cfg.get("start", 0))
            return int(end) - start if end is not None else input_dim
        if t == "cosine_similarity":
            return 1
        if t in ("fourier_encoder_layer", "fourier_encoder"):
            return int(cfg.get("d_model") or input_dim)
        # graph ops that change dim
        if t == "concat":
            # concat dim; output dim = sum of all input dims along dim
            return input_dim  # callers should overwrite with actual sum
        # everything else preserves dim
        return input_dim

    def build_sequential(
        layers: list[dict[str, Any]],
        input_size: int,
        out_dim: int,
    ) -> nn.Sequential:
        """Build an ``nn.Sequential`` from a list of layer descriptions.

        Args:
            layers: YAML-derived layer configs (each has ``type``).
            input_size: Number of input features.
            out_dim: Number of output units (used for the last linear layer).

        Returns:
            A compiled ``nn.Sequential`` module.
        """
        modules: list[nn.Module] = []
        prev_dim = input_size

        for i, cfg in enumerate(layers):
            t = cfg.get("type", "")
            is_last_linear = t == "linear" and i == len(layers) - 1
            module = _build_module(cfg, prev_dim, out_dim=out_dim if is_last_linear else None)
            modules.append(module)

            prev_dim = _resolve_output_dim(t, cfg, prev_dim, out_dim, is_last_linear)

        return nn.Sequential(*modules)

    # ── LayerGraph (DAG execution engine) ─────────────────────────

    class _LayerNode:
        """A single node in the layer graph."""

        def __init__(
            self,
            name: str,
            module: nn.Module,
            input_names: list[str],
            output_dim: int,
        ) -> None:
            self.name = name
            self.module = module
            self.input_names = input_names
            self.output_dim = output_dim

    class _LayerGraph(nn.Module):
        """DAG model: nodes can reference named outputs of previous nodes via ``input_names``.

        If *input_names* is empty the node is sequential — it takes the immediately
        preceding node's output (like ``nn.Sequential``).
        """

        def __init__(self, configs: list[dict[str, Any]], input_size: int, out_dim: int) -> None:
            super().__init__()
            self.nodes: list[_LayerNode] = []
            self.last_name: str = ""

            auto_counter: int = 0

            for i, cfg in enumerate(configs):
                t = cfg.get("type", "")
                input_names: list[str] = cfg.get("inputs", [])

                # determine input dim for this node
                if input_names:
                    # graph mode: look up dims from already-registered nodes
                    input_dims: list[int] = []
                    for inp in input_names:
                        if inp == "_input":
                            input_dims.append(input_size)
                        else:
                            for node in self.nodes:
                                if node.name == inp:
                                    input_dims.append(node.output_dim)
                                    break
                            else:
                                input_dims.append(input_size)  # fallback
                    if t == "add":
                        in_dim = input_dims[0] if input_dims else input_size
                    elif t == "concat":
                        in_dim = sum(input_dims)  # sum along dim=1
                        total_dim = sum(input_dims)
                        cfg = {**cfg, "_concat_sum_dim": total_dim}
                        in_dim = total_dim
                    else:
                        in_dim = input_dims[0] if input_dims else input_size
                    # prev dim for the builder (single input)
                    prev_dim = in_dim
                else:
                    # sequential mode
                    name: str = cfg.get("name", "")
                    if not name:
                        auto_counter += 1
                        name = f"_auto_{auto_counter}"
                        cfg = {**cfg, "name": name}
                    prev_dim = input_size if i == 0 and not self.nodes else (self.nodes[-1].output_dim if self.nodes else input_size)
                    input_names = []  # will be resolved at forward time

                is_last_linear = t == "linear" and i == len(configs) - 1
                module = _build_module(cfg, prev_dim, out_dim=out_dim if is_last_linear else None)

                # resolve output dim
                out_dim_i = _resolve_output_dim(t, cfg, prev_dim, out_dim, is_last_linear)
                if t == "concat":
                    out_dim_i = cfg.get("_concat_sum_dim", prev_dim)

                node = _LayerNode(
                    name=cfg.get("name", f"_auto_{auto_counter}"),
                    module=module,
                    input_names=input_names,
                    output_dim=out_dim_i,
                )
                self.nodes.append(node)

            self.last_name = self.nodes[-1].name if self.nodes else "_input"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cache: dict[str, torch.Tensor] = {"_input": x}
            prev: str = "_input"

            for node in self.nodes:
                if node.input_names:
                    inputs = [cache[n] for n in node.input_names]
                    out = node.module(*inputs) if len(inputs) > 1 else node.module(inputs[0])
                else:
                    out = node.module(cache[prev])
                cache[node.name] = out
                prev = node.name

            return cache[self.last_name]

    # ── model entry point ─────────────────────────────────────────

    _GRAPH_OPS: frozenset[str] = frozenset(
        {
            "add",
            "concat",
            "take_last",
            "take_first",
            "mean_pool",
            "max_pool_seq",
            "sum_pool_seq",
        }
    )

    def _has_graph_features(layers: list[dict[str, Any]]) -> bool:
        """Check if any layer uses graph features (inputs, graph ops)."""
        for layer in layers:
            if "inputs" in layer:
                return True
            if layer.get("type") in _GRAPH_OPS:
                return True
        return False

    def build_model(
        layers: list[dict[str, Any]],
        input_size: int,
        out_dim: int,
        constants: dict[str, Any] | None = None,
        blocks: dict[str, Any] | None = None,
        block_dirs: list[str] | None = None,
        config_base_path: Path | None = None,
    ) -> nn.Module:
        """Build a model from YAML layer descriptions.

        Dispatches to ``nn.Sequential`` (fast path) or ``_LayerGraph`` (DAG path)
        based on whether graph features are present.
        """
        # 1. constants interpolation
        layers = interpolate_constants(layers, constants or {})

        # 2. load and expand blocks
        all_blocks = load_blocks(blocks, block_dirs, config_base_path)
        layers = expand_blocks(layers, all_blocks, constants or {})

        # 3. validate
        _validate_layer_cfg(layers, all_blocks, constants or {})

        # 4. choose graph or sequential
        if _has_graph_features(layers):
            return _LayerGraph(layers, input_size, out_dim)
        return build_sequential(layers, input_size, out_dim)

    # ── wrapper ───────────────────────────────────────────────────

    class PyTorchWrapper:
        """Wrapper making a PyTorch ``nn.Module`` conform to ``MLModelProtocol``."""

        def __init__(
            self,
            model: nn.Module,
            task_type: TaskType,
            out_dim: int,
        ) -> None:
            self.model = model
            self.task_type = task_type
            self.out_dim = out_dim
            self.is_fitted = False

        def fit(
            self,
            x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
            y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
            **kwargs: Any,
        ) -> Self:
            epochs = int(kwargs.get("epochs", 10))
            lr = float(kwargs.get("lr", 0.001))
            batch_size = int(kwargs.get("batch_size", 32))

            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            if isinstance(y, UnifiedTensor):
                y = y.numpy()
            if isinstance(x, pd.DataFrame):
                X_arr = x.to_numpy()
            else:
                X_arr = np.asarray(x)
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy()
            else:
                y_arr = np.asarray(y)

            X_tensor = torch.tensor(X_arr).float()
            y_tensor = torch.tensor(y_arr).float()
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)

            criterion: nn.Module
            if self.task_type == TaskType.REGRESSION:
                criterion = nn.MSELoss()
            elif self.task_type == TaskType.CLASSIFICATION and self.out_dim > 1:
                criterion = nn.CrossEntropyLoss()
                y_tensor = y_tensor.squeeze(1).long()
            else:
                criterion = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.model.train()
            for _ in range(epochs):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    output = self.model(batch_X)
                    if self.task_type == TaskType.CLASSIFICATION and self.out_dim > 1:
                        output = output.squeeze(1)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()

            self.is_fitted = True
            logger.info("PyTorch fitted: %d epochs, final loss=%.4f", epochs, loss.item())
            return self

        def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[Any]:
            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            if isinstance(x, pd.Series):
                x = x.to_frame().T
            if isinstance(x, pd.DataFrame):
                X_arr = x.to_numpy()
            else:
                X_arr = np.asarray(x)

            self.model.eval()
            with torch.no_grad():
                output = self.model(torch.tensor(X_arr).float())

            if self.task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK):
                if self.out_dim > 1:
                    return cast(np.ndarray, output.argmax(dim=1, keepdim=True).numpy())
                return (torch.sigmoid(output) >= 0.5).numpy().astype(int)
            return cast(np.ndarray, output.numpy())

        def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            if isinstance(x, pd.Series):
                x = x.to_frame().T
            if isinstance(x, pd.DataFrame):
                X_arr = x.to_numpy()
            else:
                X_arr = np.asarray(x)

            self.model.eval()
            with torch.no_grad():
                output = self.model(torch.tensor(X_arr).float())

            if self.out_dim > 1:
                return torch.softmax(output, dim=1).numpy()
            sigmoid_output = torch.sigmoid(output)
            return torch.cat([1 - sigmoid_output, sigmoid_output], dim=1).numpy()

except ImportError:
    pass


@registry.model_provider("pytorch", package="torch", module="mlcombine.models.providers.pytorch")
def pytorch_provider(
    backbone: str = "mlp",
    task_type: TaskType = TaskType.REGRESSION,
    objective: ModelObjective = ModelObjective.RMSE,
    num_classes: int | None = None,
    input_size: int | None = None,
    **params: Any,
) -> SupportedModel:
    """Create a PyTorch model using ``build_model()``.

    Args:
        backbone: Ignored (architecture is controlled via ``layers``).
        task_type: Regression or classification.
        objective: Objective metric.
        num_classes: Number of classes for classification tasks.
        input_size: Input feature count (passed by builder, may also be in **params).
        **params: May contain ``layers``, ``constants``, ``blocks``,
            ``block_dirs``, ``config_base_path``.
    """
    try:
        import torch  # noqa: F401 — fail fast before accessing names defined inside try: import torch

        model_class_path: str | None = params.get("model_class_path")
        input_size = int(input_size or params.get("input_size", 100))
        final_out = num_classes or 1
        if task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK):
            final_out = num_classes or 2

        if model_class_path:
            import importlib.util
            from pathlib import Path

            spec = importlib.util.spec_from_file_location(
                "custom_model",
                str(Path(model_class_path).resolve()),
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load model_class_path: {model_class_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            build_fn = getattr(mod, "build_model", None)
            if build_fn is None:
                raise ImportError(f"model_class_path {model_class_path} must export a 'build_model' function")
            model_params: dict[str, Any] = dict(params.get("params", {}))
            model = build_fn(input_size=input_size, num_classes=final_out, **model_params)
            return PyTorchWrapper(model, task_type, final_out)

        layers_raw: list[dict[str, Any]] | None = params.get("layers")
        constants: dict[str, Any] = dict(params.get("constants") or {})
        blocks: dict[str, Any] = dict(params.get("blocks") or {})
        block_dirs: list[str] | None = params.get("block_dirs")
        config_base_path: Path | None = params.get("config_base_path")

        layer_list = layers_raw if layers_raw else _DEFAULT_LAYERS
        model = build_model(
            layer_list,
            input_size=input_size,
            out_dim=final_out,
            constants=constants,
            blocks=blocks,
            block_dirs=block_dirs,
            config_base_path=config_base_path,
        )
        return PyTorchWrapper(model, task_type, final_out)
    except ImportError:
        logger.error("PyTorch is not installed. Install with: uv add torch")
        raise
