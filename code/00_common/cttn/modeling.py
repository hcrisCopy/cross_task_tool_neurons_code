from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .paths import load_models_config, resolve_path


VALID_MODEL_ALIASES = (
    "qwen3-1.7b",
    "qwen3-4b-instruct",
    "qwen3-14b",
    "qwen3-32b",
    "llama3.1-8b",
    "llama3.3-70b",
)


def get_model_info(model_alias: str) -> dict[str, Any]:
    cfg = load_models_config().get("models", {})
    if model_alias not in cfg:
        raise KeyError(f"Unknown model alias {model_alias}. Valid: {', '.join(VALID_MODEL_ALIASES)}")
    return cfg[model_alias]


def resolve_model_path(model_alias: str, override: str | None = None) -> Path:
    if override:
        return resolve_path(override)
    info = get_model_info(model_alias)
    return resolve_path(info["local_path"])


def infer_tool_format(model_alias: str, model_path: str | Path) -> str:
    key = f"{model_alias} {model_path}".lower()
    return "native" if "llama" in key else "xml"


def parse_ffn_module_name(name: str) -> tuple[int, str] | None:
    match = re.search(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$", name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def find_ffn_target_modules(model: Any) -> list[tuple[str, int, str, Any]]:
    targets: list[tuple[str, int, str, Any]] = []
    for name, module in model.named_modules():
        parsed = parse_ffn_module_name(name)
        if parsed is not None:
            layer, module_name = parsed
            targets.append((name, layer, module_name, module))
    targets.sort(key=lambda item: (item[1], item[2], item[0]))
    if not targets:
        raise RuntimeError("No FFN target modules found: expected mlp.gate_proj/up_proj/down_proj")
    return targets
