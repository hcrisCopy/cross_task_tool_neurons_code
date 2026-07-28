from __future__ import annotations

import csv
import hashlib
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from cttn.data import SUBSETS, select_label_balanced
from cttn.eval_metrics import build_comparison_with_base
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.when2tool_bridge import load_model_module, load_utils


PP_STAGE_VERSION = 1
PP_METHOD = "CTD-Probe&Prefill"
PROBE_METHOD_SAFETY_KERNEL = "safety_kernel"
PROBE_METHOD_SAFETY_KERNEL_UNION = "safety_kernel_union"
PROBE_METHOD_SAFETY_KERNEL_NOABC = "safety_kernel_noabc"
PROBE_METHOD_PRECISE_SHIELD = "precise_shield"
PROBE_METHOD_PRECISE_SHIELD_UNION = "precise_shield_union"
PROBE_METHOD_PRECISE_SHIELD_NOABC = "precise_shield_noabc"
SUPPORTED_PROBE_METHODS = (
    PROBE_METHOD_SAFETY_KERNEL,
    PROBE_METHOD_SAFETY_KERNEL_UNION,
    PROBE_METHOD_SAFETY_KERNEL_NOABC,
    PROBE_METHOD_PRECISE_SHIELD,
    PROBE_METHOD_PRECISE_SHIELD_UNION,
    PROBE_METHOD_PRECISE_SHIELD_NOABC,
)
PROBE_METHOD_CONFIGS = {
    PROBE_METHOD_SAFETY_KERNEL: {
        "namespace": "safety_kernel",
        "label": "Safety Kernel",
        "shared_label": "CTD",
        "single_label": "TDN",
        "shared_filename": "CTD_neurons.jsonl",
        "single_filename": "TDN_neurons.jsonl",
        "feature_set": "CTD",
        "feature_description": "Safety Kernel CTD FFN output last-token activations",
        "feature_definition": "stage4 last-input-token FFN output activation restricted to stage6 CTD neurons",
        "train_probe_method": "CTD logistic probe",
        "probe_prefill_method": PP_METHOD,
    },
    PROBE_METHOD_SAFETY_KERNEL_UNION: {
        "namespace": "safety_kernel_union",
        "label": "SafetyKernel_Union",
        "shared_label": "CTD_Union",
        "single_label": "TDN",
        "shared_filename": "CTD_Union_neurons.jsonl",
        "single_filename": "TDN_neurons.jsonl",
        "feature_set": "CTD_Union",
        "feature_description": "SafetyKernel_Union FFN output last-token activations",
        "feature_definition": "stage4 last-input-token FFN output activation restricted to SafetyKernel_Union stage6 CTD_Union neurons",
        "train_probe_method": "CTD_Union logistic probe",
        "probe_prefill_method": "SafetyKernel_Union-CTD_Union-Probe&Prefill",
    },
    PROBE_METHOD_SAFETY_KERNEL_NOABC: {
        "namespace": "safety_kernel_noabc",
        "label": "SafetyKernel_noABC",
        "shared_label": "SK_noABC_TDN",
        "single_label": "SK_noABC_TDN",
        "shared_filename": "SK_noABC_TDN_neurons.jsonl",
        "single_filename": "SK_noABC_TDN_neurons.jsonl",
        "feature_set": "SK_noABC_TDN",
        "feature_description": "SafetyKernel_noABC global tool-decision FFN output last-token activations",
        "feature_definition": "stage4 last-input-token FFN output activation restricted to noABC TopK SCAR(tool_necessary=1 vs 0) neurons",
        "train_probe_method": "SK_noABC_TDN logistic probe",
        "probe_prefill_method": "SafetyKernel_noABC-SK_noABC_TDN-Probe&Prefill",
    },
    PROBE_METHOD_PRECISE_SHIELD: {
        "namespace": "precise_shield",
        "label": "PreciseShield",
        "shared_label": "PS_CTD",
        "single_label": "PS_TDN",
        "shared_filename": "PS_CTD_neurons.jsonl",
        "single_filename": "PS_TDN_neurons.jsonl",
        "feature_set": "PS_CTD",
        "feature_description": "PreciseShield PS-CTD FFN intermediate last-token activations",
        "feature_definition": "PreciseShield last-input-token FFN intermediate h before down_proj restricted to stage6 PS_CTD neurons",
        "train_probe_method": "PS_CTD logistic probe",
        "probe_prefill_method": "PreciseShield-PS_CTD-Probe&Prefill",
    },
    PROBE_METHOD_PRECISE_SHIELD_UNION: {
        "namespace": "precise_shield_union",
        "label": "PreciseShield_Union",
        "shared_label": "PS_CTD_Union",
        "single_label": "PS_TDN",
        "shared_filename": "PS_CTD_Union_neurons.jsonl",
        "single_filename": "PS_TDN_neurons.jsonl",
        "feature_set": "PS_CTD_Union",
        "feature_description": "PreciseShield_Union PS-CTD_Union FFN intermediate last-token activations",
        "feature_definition": "PreciseShield last-input-token FFN intermediate h before down_proj restricted to PreciseShield_Union stage6 PS_CTD_Union neurons",
        "train_probe_method": "PS_CTD_Union logistic probe",
        "probe_prefill_method": "PreciseShield_Union-PS_CTD_Union-Probe&Prefill",
    },
    PROBE_METHOD_PRECISE_SHIELD_NOABC: {
        "namespace": "precise_shield_noabc",
        "label": "PreciseShield_noABC",
        "shared_label": "PS_noABC_TDN",
        "single_label": "PS_noABC_TDN",
        "shared_filename": "PS_noABC_TDN_neurons.jsonl",
        "single_filename": "PS_noABC_TDN_neurons.jsonl",
        "feature_set": "PS_noABC_TDN",
        "feature_description": "PreciseShield_noABC global tool-decision FFN intermediate last-token activations",
        "feature_definition": "PreciseShield last-input-token FFN intermediate h before down_proj restricted to noABC TopK(S_tool_necessary) minus TopK(S_tool_unnecessary) neurons",
        "train_probe_method": "PS_noABC_TDN logistic probe",
        "probe_prefill_method": "PreciseShield_noABC-PS_noABC_TDN-Probe&Prefill",
    },
}
PP_SUBDIRS = {
    "features": "probe_features",
    "probes": "probes",
    "outputs": "outputs",
    "causal": "causal_validation",
    "reports": "reports",
}
PP_LEGACY_SAFETY_KERNEL_SUBDIRS = tuple(PP_SUBDIRS.values())

__all__ = [
    "PP_METHOD",
    "PP_STAGE_VERSION",
    "PROBE_METHOD_PRECISE_SHIELD",
    "PROBE_METHOD_PRECISE_SHIELD_NOABC",
    "PROBE_METHOD_PRECISE_SHIELD_UNION",
    "PROBE_METHOD_SAFETY_KERNEL",
    "PROBE_METHOD_SAFETY_KERNEL_NOABC",
    "PROBE_METHOD_SAFETY_KERNEL_UNION",
    "SUPPORTED_PROBE_METHODS",
    "classification_metrics",
    "clean_path",
    "compare_summaries_to_base",
    "compute_prefills",
    "default_method_activations_dir",
    "default_method_neurons_dir",
    "default_prefill_mode",
    "features_from_rows",
    "flatten_probe_predictions",
    "grouped_classification_metrics",
    "infer_tool_format",
    "load_ctd_rows",
    "load_shared_neuron_rows",
    "load_model_module",
    "load_stage_activations",
    "load_tdn_rows",
    "load_utils",
    "method_feature_definition",
    "method_feature_description",
    "method_feature_set",
    "method_label",
    "method_neuron_identity",
    "method_probe_prefill_name",
    "method_train_probe_name",
    "normalize_probe_method",
    "parse_thresholds",
    "probe_method_choices",
    "probe_method_root",
    "path_from_config",
    "print_subset_plan",
    "prepare_probe_method_root",
    "pp_subdir",
    "private_rows",
    "probe_prefill_root",
    "read_json",
    "read_jsonl",
    "remove_files",
    "resolve_model_path",
    "resolve_path",
    "sample_random_like_rows",
    "select_meta_indices",
    "should_skip",
    "sigmoid_temperature",
    "stable_sha256",
    "subset_values",
    "summarize_labels",
    "validate_records_cover_task_ids",
    "write_csv",
    "write_json",
    "write_jsonl",
    "copy_probe_artifacts",
    "is_dp_parent",
    "make_dp_run_root",
    "namespace_to_cli",
    "parse_gpus",
    "prepare_feature_meta_shard",
    "prepare_feature_tensor_shard",
    "run_data_parallel_workers",
    "set_single_process_cuda_visible",
    "shard_indices",
    "shard_items",
    "sort_records_by_task_ids",
    "task_id",
]
PREFILL_TEMPLATES = {
    "soft": {
        "xml": {
            "no_tool": "I can solve this directly without using a tool.\n",
            "use_tool": "I need to use a tool for this question.\n",
        },
        "native": {
            "no_tool": "I can solve this directly without using a tool.\n",
            "use_tool": "I need to use a tool for this question.\n",
        },
    },
    "hard": {
        "xml": {
            "no_tool": "\\boxed{",
            "use_tool": "<tool_call>\n",
        },
        "native": {
            "no_tool": "\\boxed{",
            "use_tool": '{"name": "',
        },
    },
}


def normalize_probe_method(value: str | None) -> str:
    method = (value or PROBE_METHOD_SAFETY_KERNEL).strip().lower().replace("-", "_")
    aliases = {
        "ctd": PROBE_METHOD_SAFETY_KERNEL,
        "safety": PROBE_METHOD_SAFETY_KERNEL,
        "safetykernel": PROBE_METHOD_SAFETY_KERNEL,
        "safety_kernel_ctd": PROBE_METHOD_SAFETY_KERNEL,
        "sku": PROBE_METHOD_SAFETY_KERNEL_UNION,
        "ctd_union": PROBE_METHOD_SAFETY_KERNEL_UNION,
        "safetykernelunion": PROBE_METHOD_SAFETY_KERNEL_UNION,
        "safety_kernel_union_ctd": PROBE_METHOD_SAFETY_KERNEL_UNION,
        "skna": PROBE_METHOD_SAFETY_KERNEL_NOABC,
        "sk_noabc": PROBE_METHOD_SAFETY_KERNEL_NOABC,
        "sk_no_abc": PROBE_METHOD_SAFETY_KERNEL_NOABC,
        "safetykernelnoabc": PROBE_METHOD_SAFETY_KERNEL_NOABC,
        "safety_kernel_no_abc": PROBE_METHOD_SAFETY_KERNEL_NOABC,
        "ps": PROBE_METHOD_PRECISE_SHIELD,
        "preciseshield": PROBE_METHOD_PRECISE_SHIELD,
        "ps_ctd": PROBE_METHOD_PRECISE_SHIELD,
        "psu": PROBE_METHOD_PRECISE_SHIELD_UNION,
        "preciseshieldunion": PROBE_METHOD_PRECISE_SHIELD_UNION,
        "precise_shield_union_ctd": PROBE_METHOD_PRECISE_SHIELD_UNION,
        "ps_ctd_union": PROBE_METHOD_PRECISE_SHIELD_UNION,
        "psna": PROBE_METHOD_PRECISE_SHIELD_NOABC,
        "ps_noabc": PROBE_METHOD_PRECISE_SHIELD_NOABC,
        "ps_no_abc": PROBE_METHOD_PRECISE_SHIELD_NOABC,
        "preciseshieldnoabc": PROBE_METHOD_PRECISE_SHIELD_NOABC,
        "precise_shield_no_abc": PROBE_METHOD_PRECISE_SHIELD_NOABC,
    }
    method = aliases.get(method, method)
    if method not in PROBE_METHOD_CONFIGS:
        raise ValueError(f"Unknown probe method: {value}. Valid: {', '.join(SUPPORTED_PROBE_METHODS)}")
    return method


def probe_method_choices() -> tuple[str, ...]:
    return SUPPORTED_PROBE_METHODS


def _method_config(probe_method: str | None) -> dict[str, str]:
    return PROBE_METHOD_CONFIGS[normalize_probe_method(probe_method)]


def probe_method_root(root: Path, probe_method: str | None) -> Path:
    cfg = _method_config(probe_method)
    namespace = cfg["namespace"]
    if root.name == namespace:
        return root
    return root / namespace


def _reuse_legacy_safety_kernel_outputs(base_root: Path, method_root: Path) -> None:
    if not base_root.exists() or base_root == method_root:
        return
    copied: list[tuple[Path, Path]] = []
    for name in PP_LEGACY_SAFETY_KERNEL_SUBDIRS:
        src = base_root / name
        dst = method_root / name
        if not src.exists() or dst.exists():
            continue
        ensure_dir(method_root)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)
        copied.append((src, dst))
    if copied:
        print("Reused legacy Safety Kernel ProbePrefill outputs under method namespace:", flush=True)
        for src, dst in copied:
            print(f"  {src} -> {dst}", flush=True)


def prepare_probe_method_root(root: Path, probe_method: str | None) -> Path:
    method = normalize_probe_method(probe_method)
    method_root = probe_method_root(root, method)
    if method == PROBE_METHOD_SAFETY_KERNEL:
        _reuse_legacy_safety_kernel_outputs(root, method_root)
    return method_root


def default_method_activations_dir(probe_method: str | None) -> Path:
    method = normalize_probe_method(probe_method)
    if method == PROBE_METHOD_SAFETY_KERNEL_NOABC:
        return data_root() / "safety_kernel_noabc" / "activations"
    if method in {PROBE_METHOD_PRECISE_SHIELD, PROBE_METHOD_PRECISE_SHIELD_UNION, PROBE_METHOD_PRECISE_SHIELD_NOABC}:
        return data_root() / "precise_shield" / "activations"
    return path_from_config("activations_dir")


def default_method_neurons_dir(probe_method: str | None) -> Path:
    method = normalize_probe_method(probe_method)
    if method == PROBE_METHOD_PRECISE_SHIELD:
        return data_root() / "precise_shield" / "neurons"
    if method == PROBE_METHOD_PRECISE_SHIELD_UNION:
        return data_root() / "precise_shield_union" / "neurons"
    if method == PROBE_METHOD_PRECISE_SHIELD_NOABC:
        return data_root() / "precise_shield_noabc" / "neurons"
    if method == PROBE_METHOD_SAFETY_KERNEL_UNION:
        return data_root() / "safety_kernel_union" / "neurons"
    if method == PROBE_METHOD_SAFETY_KERNEL_NOABC:
        return data_root() / "safety_kernel_noabc" / "neurons"
    return path_from_config("neurons_dir")


def method_label(probe_method: str | None) -> str:
    return _method_config(probe_method)["label"]


def method_feature_set(probe_method: str | None) -> str:
    return _method_config(probe_method)["feature_set"]


def method_feature_description(probe_method: str | None) -> str:
    return _method_config(probe_method)["feature_description"]


def method_feature_definition(probe_method: str | None) -> str:
    return _method_config(probe_method)["feature_definition"]


def method_train_probe_name(probe_method: str | None) -> str:
    return _method_config(probe_method)["train_probe_method"]


def method_probe_prefill_name(probe_method: str | None) -> str:
    return _method_config(probe_method)["probe_prefill_method"]


def method_neuron_identity(row: dict[str, Any]) -> dict[str, Any]:
    out = {"layer": int(row["layer"]), "index": int(row["index"])}
    if "module" in row:
        out["module"] = str(row["module"])
    return out


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def probe_prefill_root(value: str | None = None) -> Path:
    if value:
        return resolve_path(value)
    return data_root() / "probe_prefill"


def pp_subdir(root: Path, kind: str) -> Path:
    if kind not in PP_SUBDIRS:
        raise KeyError(f"Unknown ProbePrefill output kind: {kind}")
    return root / PP_SUBDIRS[kind]


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def print_subset_plan(value: str, *, stage: str, model_alias: str) -> list[str]:
    subsets = subset_values(value)
    print(f"{stage} {model_alias}: subset order = {' -> '.join(subsets)}", flush=True)
    return subsets


def parse_thresholds(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    for value in values:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
    return values


def default_prefill_mode(model_alias: str, tool_format: str) -> str:
    return "hard" if tool_format == "native" or "llama" in model_alias.lower() else "soft"


def clean_path(path: Path, *, allowed_root: Path | None = None) -> None:
    clean_directory(path, allowed_root or probe_prefill_root())


def should_skip(
    out_dir: Path,
    params: dict[str, Any],
    expected_files: Iterable[Path],
    *,
    overwrite: bool,
    clean: bool,
    allowed_root: Path | None = None,
) -> bool:
    if clean:
        clean_path(out_dir, allowed_root=allowed_root)
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


def maybe_read_manifest(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_meta_indices(
    meta_rows: list[dict[str, Any]],
    count: int,
    seed: int,
    *,
    strategy: str = "balanced",
    require_per_type_labels: bool = True,
) -> list[int]:
    if count <= 0 or count >= len(meta_rows):
        return list(range(len(meta_rows)))
    if strategy == "first":
        selected = list(meta_rows[:count])
    elif strategy == "balanced":
        selected = select_label_balanced(
            list(meta_rows),
            count,
            seed,
            require_per_type_labels=require_per_type_labels,
        )
    else:
        raise ValueError(f"Unknown sample strategy: {strategy}")
    selected_ids = {str(row["id"]) for row in selected}
    indices = [idx for idx, row in enumerate(meta_rows) if str(row["id"]) in selected_ids]
    if len(indices) != len(selected_ids):
        raise ValueError("Could not map all selected metadata rows back to activation indices")
    return indices


def load_stage_activations(activations_dir: Path, model_alias: str, subset: str, split: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    act_dir = activations_dir / model_alias / subset / split
    activation_path = act_dir / "activations.pt"
    meta_path = act_dir / "meta.jsonl"
    manifest_path = act_dir / "manifest.json"
    if not activation_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing stage 4 activation outputs: {act_dir}")
    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    meta_rows = read_jsonl(meta_path)
    manifest = maybe_read_manifest(manifest_path)
    return payload, meta_rows, manifest


def load_shared_neuron_rows(
    neurons_dir: Path,
    model_alias: str,
    subset: str,
    probe_method: str | None = PROBE_METHOD_SAFETY_KERNEL,
) -> list[dict[str, Any]]:
    cfg = _method_config(probe_method)
    path = neurons_dir / model_alias / "shared_by_subset" / subset / cfg["shared_filename"]
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{cfg['shared_label']} neuron set is empty: {path}")
    return rows


def load_ctd_rows(neurons_dir: Path, model_alias: str, subset: str) -> list[dict[str, Any]]:
    return load_shared_neuron_rows(neurons_dir, model_alias, subset, PROBE_METHOD_SAFETY_KERNEL)


def load_tdn_rows(
    neurons_dir: Path,
    model_alias: str,
    subset: str,
    task_type: str,
    probe_method: str | None = PROBE_METHOD_SAFETY_KERNEL,
) -> list[dict[str, Any]]:
    cfg = _method_config(probe_method)
    path = neurons_dir / model_alias / "single_type_by_subset" / subset / task_type / cfg["single_filename"]
    return read_jsonl(path)


def neuron_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["layer"]), str(row["module"]), int(row["index"])


def rows_to_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[int, str, int]]:
    return {neuron_key(row) for row in rows}


def private_rows(tdn_rows: list[dict[str, Any]], ctd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctd = rows_to_keys(ctd_rows)
    return [row for row in tdn_rows if neuron_key(row) not in ctd]


def module_key_map(module_meta: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    mapping: dict[tuple[int, str], str] = {}
    for meta in module_meta:
        mapping[(int(meta["layer"]), str(meta["module"]))] = str(meta["key"])
    return mapping


def module_dims(module_meta: list[dict[str, Any]]) -> dict[tuple[int, str], int]:
    return {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}


def order_neuron_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get("rank", 10**9)), int(row["layer"]), str(row["module"]), int(row["index"])))


def features_from_rows(
    payload: dict[str, Any],
    meta_rows: list[dict[str, Any]],
    neuron_rows: list[dict[str, Any]],
    indices: list[int] | None = None,
) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    module_meta = payload["module_meta"]
    activations: dict[str, torch.Tensor] = payload["activations"]
    key_by_coord = module_key_map(module_meta)
    ordered_rows = order_neuron_rows(neuron_rows)
    selected_indices = list(range(len(meta_rows))) if indices is None else list(indices)
    if not ordered_rows:
        raise ValueError("Cannot build probe features from an empty neuron set")

    selected_idx_tensor = torch.tensor(selected_indices, dtype=torch.long)
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, row in enumerate(ordered_rows):
        coord = (int(row["layer"]), str(row["module"]))
        module_key = str(row.get("module_key") or key_by_coord.get(coord, ""))
        if not module_key or module_key not in activations:
            raise KeyError(f"Cannot map neuron row to activation tensor: {row}")
        index = int(row["index"])
        tensor = activations[module_key]
        if index < 0 or index >= tensor.shape[1]:
            raise IndexError(f"Neuron index out of range for {module_key}: index={index}, dim={tensor.shape[1]}")
        groups[module_key].append((position, index))

    feature_columns: list[torch.Tensor | None] = [None] * len(ordered_rows)
    for module_key, positions_and_indices in groups.items():
        positions = [pos for pos, _idx in positions_and_indices]
        col_indices = torch.tensor([idx for _pos, idx in positions_and_indices], dtype=torch.long)
        block = activations[module_key].index_select(0, selected_idx_tensor).index_select(1, col_indices).float().cpu()
        for local_col, global_pos in enumerate(positions):
            feature_columns[global_pos] = block[:, local_col]
    if any(column is None for column in feature_columns):
        raise RuntimeError("Internal error while building feature columns")
    features = torch.stack([column for column in feature_columns if column is not None], dim=1).contiguous()
    selected_meta = [meta_rows[idx] for idx in selected_indices]
    labels = np.array([int(row["tool_necessary"]) for row in selected_meta], dtype=np.int64)
    return features, labels, selected_meta, ordered_rows


def summarize_labels(meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(meta_rows),
        "by_tool_necessary": dict(Counter(str(int(row["tool_necessary"])) for row in meta_rows)),
        "by_task_type": dict(Counter(str(row.get("task_type", "unknown")) for row in meta_rows)),
        "by_difficulty": dict(Counter(str(row.get("difficulty", "unknown")) for row in meta_rows)),
        "by_env": dict(Counter(str(row.get("env_name", "unknown")) for row in meta_rows)),
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() if len(y_true) else [[0, 0], [0, 0]],
        "n_tool_necessary": int(y_true.sum()) if len(y_true) else 0,
    }
    metrics["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else None
    return metrics


def grouped_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    meta_rows: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    result = {"overall": classification_metrics(y_true, y_prob, threshold)}
    group_specs = {
        "by_task_type": "task_type",
        "by_env": "env_name",
        "by_difficulty": "difficulty",
    }
    for section, field in group_specs.items():
        result[section] = {}
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(meta_rows):
            groups[str(row.get(field, "unknown"))].append(idx)
        for name, idxs in sorted(groups.items()):
            idx_arr = np.array(idxs, dtype=np.int64)
            result[section][name] = classification_metrics(y_true[idx_arr], y_prob[idx_arr], threshold)
    return result


def sigmoid_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("--temperature must be > 0")
    scaled = np.clip(logits / float(temperature), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def compute_prefills(
    *,
    task_ids: list[str],
    probabilities: np.ndarray,
    threshold: float,
    prefill_mode: str,
    tool_format: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    templates = PREFILL_TEMPLATES[prefill_mode][tool_format]
    prefills: dict[str, str] = {}
    stats = {"skip_tool": 0, "use_tool": 0, "threshold": threshold, "prefill_mode": prefill_mode, "tool_format": tool_format}
    for task_id, prob in zip(task_ids, probabilities):
        if float(prob) < threshold:
            prefills[str(task_id)] = templates["no_tool"]
            stats["skip_tool"] += 1
        else:
            prefills[str(task_id)] = templates["use_tool"]
            stats["use_tool"] += 1
    return prefills, stats


def flatten_probe_predictions(
    meta_rows: list[dict[str, Any]],
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for meta, prob in zip(meta_rows, y_prob):
        pred = int(float(prob) >= threshold)
        rows.append(
            {
                "id": str(meta["id"]),
                "subset": meta.get("subset", ""),
                "split": meta.get("split", ""),
                "task_type": meta.get("task_type", ""),
                "env_name": meta.get("env_name", ""),
                "difficulty": meta.get("difficulty", "unknown"),
                "tool_necessary": int(meta["tool_necessary"]),
                "probe_probability": float(prob),
                "probe_prediction": pred,
            }
        )
    return rows


def sample_random_like_rows(
    reference_rows: list[dict[str, Any]],
    module_meta: list[dict[str, Any]],
    *,
    seed: int,
    exclude_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    dims = module_dims(module_meta)
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    excluded: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in reference_rows:
        layer, module, index = neuron_key(row)
        grouped[(layer, module)].append(index)
    for row in exclude_rows or []:
        layer, module, index = neuron_key(row)
        excluded[(layer, module)].add(index)

    rng = random.Random(seed)
    out = []
    for key, indices in sorted(grouped.items()):
        dim = dims[key]
        pool = [idx for idx in range(dim) if idx not in excluded[key]]
        if len(pool) < len(indices):
            pool = list(range(dim))
        sampled = rng.sample(pool, k=min(len(indices), len(pool)))
        layer, module = key
        out.extend({"layer": layer, "module": module, "index": int(idx)} for idx in sampled)
    return out


def compare_summaries_to_base(
    *,
    base_summary_path: Path,
    probe_summary_path: Path,
    out_csv: Path,
    out_manifest: Path,
    model_alias: str,
    subset: str,
    method: str,
    params: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite and out_csv.exists() and out_manifest.exists():
        manifest = read_json(out_manifest)
        if manifest.get("params") == params:
            if _comparison_csv_has_printable_overall(out_csv):
                print(f"Skip existing comparison: {out_csv}")
                return manifest
            print(f"Rebuild stale comparison missing printable overall: {out_csv}")
    rows = build_comparison_with_base(
        base_summary=read_json(base_summary_path),
        trained_summary=read_json(probe_summary_path),
        model_alias=model_alias,
        subset=subset,
        method=method,
    )
    write_csv(out_csv, rows)
    manifest = {"params": params, "rows": len(rows), "path": str(out_csv)}
    write_json(out_manifest, manifest)
    return manifest


def _comparison_csv_has_printable_overall(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            required = {"group_kind", "group_name", "delta_acc_pp", "delta_avg_tool_calls"}
            if not required.issubset(fieldnames):
                return False
            return any(row.get("group_kind") == "overall" and row.get("group_name") == "overall" for row in reader)
    except OSError:
        return False


def remove_files(paths: Iterable[Path], *, allowed_root: Path) -> None:
    root = allowed_root.resolve()
    for path in paths:
        if path.exists():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Refusing to remove outside allowed root: {resolved}")
            if path.is_dir():
                clean_directory(path, root)
            else:
                path.unlink()


def parse_gpus(value: str | None) -> list[str]:
    if value is None:
        return ["0"]
    gpus = [item.strip() for item in str(value).split(",") if item.strip()]
    return gpus or ["0"]


def is_dp_parent(args: Any) -> bool:
    return int(getattr(args, "_worker_index", -1)) < 0 and len(parse_gpus(getattr(args, "gpus", "0"))) > 1


def set_single_process_cuda_visible(gpus: str | None) -> None:
    parsed = parse_gpus(gpus)
    if len(parsed) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = parsed[0]


def shard_indices(total: int, num_shards: int) -> list[list[int]]:
    return [list(range(shard, total, num_shards)) for shard in range(num_shards)]


def shard_items(items: list[Any], indices: list[int]) -> list[Any]:
    return [items[idx] for idx in indices]


def namespace_to_cli(args: Any, *, exclude: set[str] | None = None) -> list[str]:
    excluded = set(exclude or set())
    cli: list[str] = []
    for key, value in vars(args).items():
        if key.startswith("_") or key in excluded:
            continue
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(option)
        elif value is not None:
            cli.extend([option, str(value)])
    return cli


def make_dp_run_root(root: Path, *, stage: str, model_alias: str, subset: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = root / "_dp_shards" / stage / model_alias / subset / f"{stamp}_{os.getpid()}"
    ensure_dir(run_root)
    return run_root


def prepare_feature_meta_shard(
    src_root: Path,
    dst_root: Path,
    *,
    model_alias: str,
    subset: str,
    indices: list[int],
    split: str = "test",
) -> list[dict[str, Any]]:
    src_dir = pp_subdir(src_root, "features") / model_alias / subset
    dst_dir = pp_subdir(dst_root, "features") / model_alias / subset
    ensure_dir(dst_dir)
    rows = read_jsonl(src_dir / f"{split}_meta.jsonl")
    selected = shard_items(rows, indices)
    write_jsonl(dst_dir / f"{split}_meta.jsonl", selected)
    summary_path = src_dir / f"{split}_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        summary["data_parallel_shard"] = {"size": len(selected), "source_size": len(rows)}
        write_json(dst_dir / f"{split}_summary.json", summary)
    return selected


def prepare_feature_tensor_shard(
    src_root: Path,
    dst_root: Path,
    *,
    model_alias: str,
    subset: str,
    indices: list[int],
    split: str = "test",
) -> None:
    src_dir = pp_subdir(src_root, "features") / model_alias / subset
    dst_dir = pp_subdir(dst_root, "features") / model_alias / subset
    ensure_dir(dst_dir)
    payload_path = src_dir / f"{split}_features.pt"
    if payload_path.exists():
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        idx = torch.tensor(indices, dtype=torch.long)
        payload = dict(payload)
        payload["features"] = payload["features"].index_select(0, idx).contiguous()
        torch.save(payload, dst_dir / f"{split}_features.pt")


def copy_probe_artifacts(src_root: Path, dst_root: Path, *, model_alias: str, subset: str) -> None:
    src = pp_subdir(src_root, "probes") / model_alias / subset
    dst = pp_subdir(dst_root, "probes") / model_alias / subset
    if not src.exists():
        raise FileNotFoundError(f"Missing probe artifacts: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def task_id(record: dict[str, Any]) -> str:
    for key in ("id", "task_id"):
        if key in record:
            return str(record[key])
    raise KeyError(f"Cannot find task id in record keys: {sorted(record)}")


def sort_records_by_task_ids(records: list[dict[str, Any]], task_ids: list[str]) -> list[dict[str, Any]]:
    order = {str(task_id): idx for idx, task_id in enumerate(task_ids)}
    return sorted(records, key=lambda row: order.get(task_id(row), len(order)))


def validate_records_cover_task_ids(records: list[dict[str, Any]], task_ids: list[str], *, label: str) -> None:
    expected = Counter(str(item) for item in task_ids)
    got_ids: list[str] = []
    for index, record in enumerate(records):
        try:
            got_ids.append(task_id(record))
        except KeyError as exc:
            raise ValueError(f"{label}: record {index} is missing id/task_id") from exc
    got = Counter(got_ids)
    if got == expected:
        return
    missing = list((expected - got).elements())[:10]
    extra = list((got - expected).elements())[:10]
    raise RuntimeError(
        f"{label}: merged records do not cover expected test ids exactly "
        f"(expected={sum(expected.values())}, got={sum(got.values())}, "
        f"missing_sample={missing}, extra_sample={extra})"
    )


def run_data_parallel_workers(
    *,
    script_path: Path,
    args: Any,
    gpus: list[str],
    subset: str,
    worker_roots: list[Path],
    total_progress: int,
    desc: str,
    shard_sizes: list[int] | None = None,
    extra_cli: list[str] | None = None,
) -> None:
    base_cli = namespace_to_cli(args, exclude={"gpus", "output_root", "subset"})
    processes = []
    lines: queue.Queue[tuple[int, str | None]] = queue.Queue()

    def reader(index: int, pipe: Any) -> None:
        try:
            for line in iter(pipe.readline, ""):
                lines.put((index, line.rstrip("\n")))
        finally:
            lines.put((index, None))

    size_text = f", shard_sizes={shard_sizes}" if shard_sizes is not None else ""
    print(
        f"{desc}: launching {len(gpus)} data-parallel workers on GPUs {','.join(gpus)}"
        f"{size_text}, total_progress={total_progress} tasks",
        flush=True,
    )
    for index, (gpu, worker_root) in enumerate(zip(gpus, worker_roots)):
        cmd = [
            sys.executable,
            str(script_path),
            *base_cli,
            "--subset",
            subset,
            "--output-root",
            str(worker_root),
            "--gpus",
            gpu,
            "--_worker-index",
            str(index),
            "--_num-workers",
            str(len(gpus)),
            *(extra_cli or []),
        ]
        print(f"{desc}: worker {index} -> GPU {gpu}, root={worker_root}", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["TQDM_DISABLE"] = "1"
        env["CTTN_PROGRESS_MARKER"] = "1"
        env["WHEN2TOOL_QUIET_PROGRESS"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(REPO_ROOT),
        )
        processes.append(proc)
        threading.Thread(target=reader, args=(index, proc.stdout), daemon=True).start()  # type: ignore[arg-type]

    marker = re.compile(r"CTTN_PROGRESS \+(\d+)")
    finished = 0
    last_lines: list[str] = []
    try:
        from tqdm.auto import tqdm as _tqdm
    except ModuleNotFoundError:
        _tqdm = None

    bar = _tqdm(total=total_progress, desc=desc, unit="task", dynamic_ncols=True) if _tqdm else None
    done = 0
    while finished < len(processes):
        index, line = lines.get()
        if line is None:
            finished += 1
            continue
        match = marker.search(line)
        if match:
            inc = int(match.group(1))
            done += inc
            if bar is not None:
                bar.update(inc)
            else:
                print(f"{desc}: {done}/{total_progress}", flush=True)
            continue
        if line.strip():
            rendered = f"[worker {index}] {line}"
            last_lines.append(rendered)
            last_lines = last_lines[-80:]
            print(rendered, flush=True)
    if bar is not None:
        bar.close()

    failed = [idx for idx, proc in enumerate(processes) if proc.wait() != 0]
    if failed:
        tail = "\n".join(last_lines[-30:])
        raise RuntimeError(f"Data-parallel workers failed: {failed}\n{tail}")
