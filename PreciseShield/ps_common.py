from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(REPO_ROOT / "PreciseShield") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PreciseShield"))

from cttn.data import SPLITS, SUBSETS, TASK_TYPES, select_label_balanced
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.when2tool_bridge import load_model_module, load_utils


STAGE_VERSION = 1
PS_METHOD = "precise_shield"
INTERMEDIATE_MODULE = "ffn_intermediate"


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def precise_root() -> Path:
    return data_root() / "precise_shield"


def ps_default_root(kind: str) -> Path:
    mapping = {
        "activations": "activations",
        "neurons": "neurons",
        "visualizations": "visualizations",
        "checkpoints": "checkpoints",
        "outputs": "outputs",
        "causal": "causal_validation",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown PreciseShield root kind: {kind}")
    return precise_root() / mapping[kind]


def ps_resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else ps_default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def split_values(value: str) -> list[str]:
    return list(SPLITS) if value == "all" else [value]


def dataset_manifest(model_dataset: Path) -> dict[str, Any]:
    path = model_dataset / "manifest.json"
    return read_json(path) if path.exists() else {}


def select_records(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    *,
    strategy: str = "balanced",
    require_per_type_labels: bool = True,
) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(rows):
        return list(rows)
    if strategy == "first":
        selected = list(rows[:count])
    elif strategy == "balanced":
        selected = select_label_balanced(
            rows,
            count,
            seed,
            require_per_type_labels=require_per_type_labels,
        )
    else:
        raise ValueError(f"Unknown sample strategy: {strategy}")
    return selected


def clean_path(path: Path) -> None:
    clean_directory(path, data_root())


def should_skip(
    out_dir: Path,
    params: dict[str, Any],
    expected_files: Iterable[Path],
    *,
    overwrite: bool,
    clean: bool,
) -> bool:
    if clean:
        clean_path(out_dir)
        return False
    manifest_path = out_dir / "manifest.json"
    files = [manifest_path, *expected_files]
    if overwrite or not all(path.exists() for path in files):
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing output: {out_dir}")
        return True
    return False


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    *,
    enable_thinking: bool | None = False,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs: Any,
) -> Any:
    call_kwargs = dict(kwargs)
    call_kwargs.update({"tokenize": tokenize, "add_generation_prompt": add_generation_prompt})
    if tools:
        call_kwargs["tools"] = tools
    if enable_thinking is not None:
        call_kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **call_kwargs)
    except TypeError:
        call_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **call_kwargs)


def build_current_prompt(
    task: dict[str, Any],
    *,
    tokenizer: Any,
    w2t_utils: Any,
    system_prompt: str,
    tool_format: str,
) -> str:
    state = w2t_utils.init_state(
        task,
        system_prompt,
        record_mode="lite",
        prompt_mode="current",
        require_reasoning=False,
        tool_format=tool_format,
        tokenizer=tokenizer,
    )
    return apply_chat_template(
        tokenizer,
        state["messages"],
        state["tools"],
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=False,
    )


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def get_submodule_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@dataclass(frozen=True)
class FFNIntermediateLayer:
    layer: int
    mlp_name: str
    gate_name: str
    up_name: str
    down_name: str
    gate: nn.Module
    up: nn.Module
    down: nn.Module
    dim: int

    @property
    def key(self) -> str:
        return self.down_name


def discover_ffn_intermediate_layers(model: nn.Module) -> list[FFNIntermediateLayer]:
    named = dict(model.named_modules())
    layers: list[FFNIntermediateLayer] = []
    for name, module in named.items():
        if not name.endswith(".mlp.down_proj"):
            continue
        parts = name.split(".")
        try:
            layer_index = int(parts[-3])
        except (ValueError, IndexError):
            continue
        mlp_name = ".".join(parts[:-1])
        gate_name = f"{mlp_name}.gate_proj"
        up_name = f"{mlp_name}.up_proj"
        if gate_name not in named or up_name not in named:
            continue
        down = module
        gate = named[gate_name]
        up = named[up_name]
        if not hasattr(down, "in_features") or not hasattr(gate, "out_features") or not hasattr(up, "out_features"):
            continue
        dim = int(down.in_features)
        if int(gate.out_features) != dim or int(up.out_features) != dim:
            raise ValueError(f"FFN dim mismatch at layer {layer_index}: {gate_name}, {up_name}, {name}")
        layers.append(
            FFNIntermediateLayer(
                layer=layer_index,
                mlp_name=mlp_name,
                gate_name=gate_name,
                up_name=up_name,
                down_name=name,
                gate=gate,
                up=up,
                down=down,
                dim=dim,
            )
        )
    layers.sort(key=lambda item: item.layer)
    if not layers:
        raise RuntimeError("No FFN intermediate layers found: expected *.mlp.gate_proj/up_proj/down_proj modules")
    return layers


def module_meta_from_layers(layers: list[FFNIntermediateLayer]) -> list[dict[str, Any]]:
    return [
        {
            "key": layer.key,
            "layer": int(layer.layer),
            "module": INTERMEDIATE_MODULE,
            "dim": int(layer.dim),
            "mlp_name": layer.mlp_name,
            "gate_name": layer.gate_name,
            "up_name": layer.up_name,
            "down_name": layer.down_name,
        }
        for layer in layers
    ]


def down_weight_norms(layers: list[FFNIntermediateLayer]) -> dict[str, torch.Tensor]:
    norms: dict[str, torch.Tensor] = {}
    for layer in layers:
        weight = layer.down.weight.detach().float().cpu()
        if weight.ndim != 2 or weight.shape[1] != layer.dim:
            raise ValueError(f"Unexpected down_proj weight shape at {layer.down_name}: {tuple(weight.shape)}")
        norms[layer.key] = weight.norm(p=2, dim=0).contiguous()
    return norms


def ps_neuron_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["layer"]), int(row["index"])


def rows_by_layer(rows: list[dict[str, Any]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for row in rows:
        layer, index = ps_neuron_key(row)
        grouped.setdefault(layer, []).append(index)
    for layer in grouped:
        grouped[layer] = sorted(set(grouped[layer]))
    return grouped


def layer_dims_from_model(model: nn.Module) -> dict[int, int]:
    return {layer.layer: layer.dim for layer in discover_ffn_intermediate_layers(model)}


def validate_ps_neuron_rows(rows: list[dict[str, Any]], model: nn.Module) -> None:
    dims = layer_dims_from_model(model)
    bad_missing = []
    bad_index = []
    for row in rows:
        layer, index = ps_neuron_key(row)
        dim = dims.get(layer)
        if dim is None:
            bad_missing.append(layer)
        elif index < 0 or index >= dim:
            bad_index.append((layer, index, dim))
    if bad_missing:
        raise ValueError(f"Neuron rows reference missing layers: {sorted(set(bad_missing))[:10]}")
    if bad_index:
        raise ValueError(f"Neuron rows contain out-of-range indices: {bad_index[:10]}")


class PSMaskedLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, mask: torch.Tensor, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_layer)!r}")
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        self.register_buffer("ps_mask", mask.detach().float().view(-1), persistent=True)
        self.reset_parameters()
        self.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        mask_shape = [1] * max(update.ndim - 1, 0) + [-1]
        mask = self.ps_mask.to(device=update.device, dtype=update.dtype).view(*mask_shape)
        return base + update * mask


def ps_module_masks(model: nn.Module, neuron_rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    validate_ps_neuron_rows(neuron_rows, model)
    grouped = rows_by_layer(neuron_rows)
    layers = discover_ffn_intermediate_layers(model)
    masks: dict[str, torch.Tensor] = {}
    for layer_info in layers:
        indices = grouped.get(layer_info.layer, [])
        if not indices:
            continue
        for module_name, module in [(layer_info.gate_name, layer_info.gate), (layer_info.up_name, layer_info.up)]:
            mask = torch.zeros(layer_info.dim, dtype=torch.float32, device=module.weight.device)
            mask[torch.tensor(indices, dtype=torch.long, device=mask.device)] = 1.0
            masks[module_name] = mask
    return masks


def apply_ps_masked_lora(
    model: nn.Module,
    neuron_rows: list[dict[str, Any]],
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> dict[str, Any]:
    masks = ps_module_masks(model, neuron_rows)
    if not masks:
        raise ValueError("PreciseShield neuron set is empty; masked LoRA has no trainable target")
    replaced: list[dict[str, Any]] = []
    for module_name, mask in masks.items():
        parent, child_name = get_submodule_parent(model, module_name)
        base = getattr(parent, child_name)
        if isinstance(base, PSMaskedLoRALinear):
            base = base.base_layer
        wrapped = PSMaskedLoRALinear(base, mask, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapped)
        replaced.append(
            {
                "module_key": module_name,
                "active_neurons": int(mask.sum().item()),
                "out_features": int(base.out_features),
            }
        )
    return {
        "method": PS_METHOD,
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "total_active_rows": int(sum(item["active_neurons"] for item in replaced)),
        "target_module_count": len(replaced),
        "target_modules": replaced,
    }


def ps_trainable_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, PSMaskedLoRALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()


def mark_only_ps_lora_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in ps_trainable_lora_parameters(model):
        param.requires_grad = True


def ps_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    keep = {}
    for name, value in model.state_dict().items():
        if ".lora_A." in name or ".lora_B." in name or name.endswith(".ps_mask"):
            keep[name] = value.detach().cpu()
    return keep


def ps_module_masks_for_config(model: nn.Module, neuron_rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    validate_ps_neuron_rows(neuron_rows, model)
    grouped = rows_by_layer(neuron_rows)
    layers = discover_ffn_intermediate_layers(model)
    result: dict[str, list[int]] = {}
    for layer_info in layers:
        indices = grouped.get(layer_info.layer, [])
        if not indices:
            continue
        result[layer_info.gate_name] = [int(i) for i in indices]
        result[layer_info.up_name] = [int(i) for i in indices]
    return result


def save_ps_masked_lora_adapter(adapter_dir: Path, model: nn.Module, config: dict[str, Any]) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ps_lora_state_dict(model), adapter_dir / "adapter_model.pt")
    write_json(adapter_dir / "adapter_config.json", config)


def load_ps_masked_lora_adapter(model: nn.Module, adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    state_path = adapter_dir / "adapter_model.pt"
    if not config_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Missing PreciseShield adapter files under {adapter_dir}")
    config = read_json(config_path)
    module_masks = config.get("module_masks", {})
    if not module_masks:
        raise ValueError(f"Adapter config has no module_masks: {config_path}")

    rows: list[dict[str, Any]] = []
    layers = discover_ffn_intermediate_layers(model)
    module_to_layer = {layer.gate_name: layer.layer for layer in layers}
    module_to_layer.update({layer.up_name: layer.layer for layer in layers})
    seen: set[tuple[int, int]] = set()
    for module_name, indices in module_masks.items():
        if module_name not in module_to_layer:
            raise KeyError(f"Adapter references missing module: {module_name}")
        layer = module_to_layer[module_name]
        for index in indices:
            key = (int(layer), int(index))
            if key not in seen:
                rows.append({"layer": key[0], "index": key[1]})
                seen.add(key)

    apply_ps_masked_lora(
        model,
        rows,
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_unexpected = [key for key in unexpected if ".lora_" not in key and not key.endswith(".ps_mask")]
    if bad_unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {bad_unexpected[:10]}")
    config["missing_lora_keys"] = [key for key in missing if ".lora_" in key or key.endswith(".ps_mask")]
    return config


@contextmanager
def ps_activation_mask(model: nn.Module, neuron_rows: list[dict[str, Any]]) -> Iterator[None]:
    if not neuron_rows:
        yield
        return
    validate_ps_neuron_rows(neuron_rows, model)
    grouped = rows_by_layer(neuron_rows)
    layers = discover_ffn_intermediate_layers(model)
    handles = []

    def make_hook(indices: list[int]):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs:
                return inputs
            hidden = inputs[0]
            if not torch.is_tensor(hidden):
                return inputs
            idx = torch.tensor(indices, dtype=torch.long, device=hidden.device)
            masked = hidden.clone()
            masked.index_fill_(-1, idx, 0)
            return (masked, *inputs[1:])

        return hook

    for layer_info in layers:
        indices = grouped.get(layer_info.layer, [])
        if indices:
            handles.append(layer_info.down.register_forward_pre_hook(make_hook(indices)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def sample_random_like_intermediate(
    reference_rows: list[dict[str, Any]],
    model: nn.Module,
    *,
    seed: int,
    exclude_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    dims = layer_dims_from_model(model)
    grouped = rows_by_layer(reference_rows)
    excluded = rows_by_layer(exclude_rows or [])
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for layer, indices in sorted(grouped.items()):
        dim = dims[layer]
        excluded_set = set(excluded.get(layer, []))
        pool = [idx for idx in range(dim) if idx not in excluded_set]
        if len(pool) < len(indices):
            pool = list(range(dim))
        sampled = rng.sample(pool, k=min(len(indices), len(pool)))
        out.extend({"layer": int(layer), "module": INTERMEDIATE_MODULE, "index": int(idx)} for idx in sampled)
    return out


def remove_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
