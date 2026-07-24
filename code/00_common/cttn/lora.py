from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F

from .modeling import find_ffn_target_modules


NeuronKey = tuple[int, str, int]


def neuron_key(row: dict[str, Any]) -> NeuronKey:
    return int(row["layer"]), str(row["module"]), int(row["index"])


def rows_to_keys(rows: list[dict[str, Any]]) -> set[NeuronKey]:
    return {neuron_key(row) for row in rows}


def rows_by_module(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[int]]:
    grouped: dict[tuple[int, str], list[int]] = {}
    for row in rows:
        layer, module, index = neuron_key(row)
        grouped.setdefault((layer, module), []).append(index)
    for key in grouped:
        grouped[key] = sorted(set(grouped[key]))
    return grouped


def module_dims_from_model(model: Any) -> dict[tuple[int, str], int]:
    dims: dict[tuple[int, str], int] = {}
    for _name, layer, module_name, module in find_ffn_target_modules(model):
        dims[(layer, module_name)] = int(module.out_features)
    return dims


def module_name_map(model: Any) -> dict[tuple[int, str], tuple[str, nn.Module]]:
    return {
        (layer, module_name): (name, module)
        for name, layer, module_name, module in find_ffn_target_modules(model)
    }


def validate_neuron_rows(rows: list[dict[str, Any]], model: Any) -> None:
    dims = module_dims_from_model(model)
    missing = []
    bad_index = []
    for row in rows:
        layer, module, index = neuron_key(row)
        dim = dims.get((layer, module))
        if dim is None:
            missing.append((layer, module))
        elif index < 0 or index >= dim:
            bad_index.append((layer, module, index, dim))
    if missing:
        raise ValueError(f"Neuron rows reference missing FFN modules: {sorted(set(missing))[:10]}")
    if bad_index:
        raise ValueError(f"Neuron rows contain out-of-range indices: {bad_index[:10]}")


def build_module_masks(
    rows: list[dict[str, Any]],
    model: Any,
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    validate_neuron_rows(rows, model)
    targets = module_name_map(model)
    grouped = rows_by_module(rows)
    masks: dict[str, torch.Tensor] = {}
    for key, indices in grouped.items():
        module_name, module = targets[key]
        out_features = int(module.out_features)
        mask = torch.zeros(out_features, dtype=torch.float32, device=device or module.weight.device)
        if indices:
            mask[torch.tensor(indices, dtype=torch.long, device=mask.device)] = 1.0
        masks[module_name] = mask
    return masks


def get_submodule_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class MaskedLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, mask: torch.Tensor, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        self.register_buffer("ctd_mask", mask.detach().to(dtype=torch.float32).view(1, 1, -1), persistent=True)
        self.reset_parameters()
        self.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        mask = self.ctd_mask.to(device=update.device, dtype=update.dtype)
        return base + update * mask


def apply_masked_lora(
    model: nn.Module,
    neuron_rows: list[dict[str, Any]],
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> dict[str, Any]:
    masks = build_module_masks(neuron_rows, model)
    if not masks:
        raise ValueError("CTD neuron set is empty; masked LoRA has no trainable target")

    replaced: list[dict[str, Any]] = []
    for module_name, mask in masks.items():
        parent, child_name = get_submodule_parent(model, module_name)
        base = getattr(parent, child_name)
        if isinstance(base, MaskedLoRALinear):
            base = base.base_layer
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {module_name}, got {type(base)!r}")
        wrapped = MaskedLoRALinear(base, mask, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapped)
        replaced.append(
            {
                "module_key": module_name,
                "active_neurons": int(mask.sum().item()),
                "out_features": int(base.out_features),
            }
        )
    return {
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "total_active_neurons": int(sum(item["active_neurons"] for item in replaced)),
        "target_module_count": len(replaced),
        "target_modules": replaced,
    }


def trainable_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, MaskedLoRALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()


def mark_only_lora_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in trainable_lora_parameters(model):
        param.requires_grad = True


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    keep = {}
    for name, value in model.state_dict().items():
        if ".lora_A." in name or ".lora_B." in name or name.endswith(".ctd_mask"):
            keep[name] = value.detach().cpu()
    return keep


def save_masked_lora_adapter(adapter_dir: Path, model: nn.Module, config: dict[str, Any]) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), adapter_dir / "adapter_model.pt")
    with (adapter_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_masked_lora_adapter(model: nn.Module, adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    state_path = adapter_dir / "adapter_model.pt"
    if not config_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Missing masked LoRA adapter files under {adapter_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    rows = []
    for module_name, indices in config.get("module_masks", {}).items():
        layer = None
        module = None
        for _name, target_layer, target_module, _target in find_ffn_target_modules(model):
            if _name == module_name:
                layer = target_layer
                module = target_module
                break
        if layer is None or module is None:
            raise KeyError(f"Adapter references missing module: {module_name}")
        rows.extend({"layer": layer, "module": module, "index": int(i)} for i in indices)
    apply_masked_lora(
        model,
        rows,
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_unexpected = [key for key in unexpected if ".lora_" not in key and not key.endswith(".ctd_mask")]
    if bad_unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {bad_unexpected[:10]}")
    config["missing_lora_keys"] = [key for key in missing if ".lora_" in key or key.endswith(".ctd_mask")]
    return config


def module_masks_for_config(model: nn.Module, rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    validate_neuron_rows(rows, model)
    grouped = rows_by_module(rows)
    targets = module_name_map(model)
    result: dict[str, list[int]] = {}
    for key, indices in grouped.items():
        module_name, _module = targets[key]
        result[module_name] = sorted(int(i) for i in indices)
    return result


def sample_random_like(
    reference_rows: list[dict[str, Any]],
    model: Any,
    *,
    seed: int,
    exclude_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    dims = module_dims_from_model(model)
    grouped = rows_by_module(reference_rows)
    excluded = rows_by_module(exclude_rows or [])
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for (layer, module), indices in sorted(grouped.items()):
        dim = dims[(layer, module)]
        excluded_set = set(excluded.get((layer, module), []))
        pool = [idx for idx in range(dim) if idx not in excluded_set]
        if len(pool) < len(indices):
            pool = list(range(dim))
        sampled = rng.sample(pool, k=min(len(indices), len(pool)))
        out.extend({"layer": layer, "module": module, "index": int(idx)} for idx in sampled)
    return out


@contextmanager
def activation_mask(model: nn.Module, neuron_rows: list[dict[str, Any]]) -> Iterator[None]:
    if not neuron_rows:
        yield
        return
    masks = build_module_masks(neuron_rows, model)
    handles = []

    def make_hook(mask: torch.Tensor):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            out = output[0] if isinstance(output, tuple) else output
            idx = torch.nonzero(mask.to(out.device) > 0, as_tuple=False).flatten()
            if idx.numel() == 0:
                return output
            masked = out.clone()
            masked.index_fill_(-1, idx, 0)
            if isinstance(output, tuple):
                return (masked, *output[1:])
            return masked

        return hook

    name_to_module = {name: module for name, _layer, _module_name, module in find_ffn_target_modules(model)}
    for module_name, mask in masks.items():
        handles.append(name_to_module[module_name].register_forward_hook(make_hook(mask)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
