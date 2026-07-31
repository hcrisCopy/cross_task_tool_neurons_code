"""Discover pathway-enhanced ToolKnowledgeNeurons for ProbePrefill.

This stage reads TKN activations and TKN_CTD candidates, builds directional
tool/direct pathways on the train split, and writes an isolated neuron root for
the unchanged ProbePrefill training code.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPO_ROOT / "code" / "00_common", REPO_ROOT / "PreciseShield"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cttn.data import SUBSETS, TASK_TYPES
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import progress
from cttn.when2tool_bridge import load_utils
from ps_common import build_current_prompt, discover_ffn_intermediate_layers, torch_dtype


STAGE_VERSION = 2
METHOD_NAME = "ToolKnowledgePathways"
FEATURE_SET = "TKP_TKN_CTD"
TKN_FILENAME = "TKN_CTD_neurons.jsonl"
TKP_FILENAME = "TKP_TKN_CTD_neurons.jsonl"
EDGE_FILENAME = "TKP_TKN_path_edges.jsonl"
INTERMEDIATE_MODULE = "ffn_intermediate"
DIRECTIONS = ("tool_high", "direct_high")
LABEL_FOR_DIRECTION = {"tool_high": 1, "direct_high": 0}
SIGN_FOR_DIRECTION = {"tool_high": 1.0, "direct_high": -1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TKN pathway discovery: directional co-activation plus sampled masking validation."
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None, help="Optional local model path override for causal masking.")
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--activations-dir", default=None, help="TKN activation root.")
    parser.add_argument("--tkn-neurons-dir", default=None, help="Original TKN neuron root.")
    parser.add_argument("--output-neurons-dir", default=None, help="Isolated TKP neuron output root.")
    parser.add_argument("--visualizations-dir", default=None, help="TKP visualization output root.")
    parser.add_argument("--dataset-dir", default=None, help="Modified When2Tool dataset root.")
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--gpus", default="0", help="Single GPU id for this process.")
    parser.add_argument("--device", default="cuda:0", help="Tensor statistics device: cpu, cuda, or cuda:0.")
    parser.add_argument("--candidate-per-direction-per-layer", type=int, default=256)
    parser.add_argument("--anchor-per-direction-per-layer", type=int, default=96)
    parser.add_argument("--final-per-direction-per-layer", type=int, default=192)
    parser.add_argument("--max-layer-gap", type=int, default=4)
    parser.add_argument("--activation-quantile", type=float, default=0.70)
    parser.add_argument("--min-target-phi", type=float, default=0.02)
    parser.add_argument("--generic-penalty", type=float, default=0.5)
    parser.add_argument("--min-edge-score", type=float, default=-0.01)
    parser.add_argument("--edge-top-k", type=int, default=4)
    parser.add_argument("--min-samples-per-class", type=int, default=4)
    parser.add_argument("--causal-mode", choices=["none", "sampled_mask"], default="sampled_mask")
    parser.add_argument("--causal-sources-per-layer-direction", type=int, default=3)
    parser.add_argument("--causal-targets-per-source", type=int, default=4)
    parser.add_argument("--causal-samples-per-task", type=int, default=12)
    parser.add_argument("--causal-batch-size", type=int, default=2)
    parser.add_argument("--min-causal-effect", type=float, default=0.0)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def default_activation_root() -> Path:
    return data_root() / "tool_knowledge_neurons" / "activations"


def default_tkn_root() -> Path:
    return data_root() / "tool_knowledge_neurons" / "neurons"


def default_output_root() -> Path:
    return data_root() / "tool_knowledge_pathways" / "neurons"


def default_viz_root() -> Path:
    return data_root() / "tool_knowledge_pathways" / "visualizations"


def activation_dir(root: Path, model_alias: str, subset: str) -> Path:
    return root / model_alias / subset / "train"


def tkn_path(root: Path, model_alias: str, subset: str) -> Path:
    return root / model_alias / "shared_by_subset" / subset / TKN_FILENAME


def output_dir(root: Path, model_alias: str, subset: str) -> Path:
    return root / model_alias / "shared_by_subset" / subset


def viz_dir(root: Path, model_alias: str, subset: str) -> Path:
    return root / model_alias / "shared_by_subset" / subset


def set_single_gpu(value: str | None) -> None:
    gpus = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]


def resolve_compute_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return device


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_row_id(row: dict[str, Any]) -> tuple[int, str, int, str]:
    return int(row["layer"]), str(row.get("module", INTERMEDIATE_MODULE)), int(row["index"]), str(row.get("direction", ""))


def load_inputs(
    activation_root: Path,
    tkn_root: Path,
    model_alias: str,
    subset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    act_dir = activation_dir(activation_root, model_alias, subset)
    activation_path = act_dir / "activations.pt"
    meta_path = act_dir / "meta.jsonl"
    manifest_path = act_dir / "manifest.json"
    candidates_path = tkn_path(tkn_root, model_alias, subset)
    if not activation_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing TKN train activations under {act_dir}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing TKN candidate file: {candidates_path}")
    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    meta_rows = read_jsonl(meta_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    tkn_rows = read_jsonl(candidates_path)
    if not meta_rows or not tkn_rows:
        raise ValueError(f"{subset}: empty TKN activations or candidates")
    return payload, meta_rows, manifest, tkn_rows


def choose_candidates(
    rows: list[dict[str, Any]],
    *,
    per_direction_per_layer: int,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    if per_direction_per_layer < 1:
        raise ValueError("--candidate-per-direction-per-layer must be >= 1")
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {direction: defaultdict(list) for direction in DIRECTIONS}
    for row in rows:
        direction = str(row.get("direction", ""))
        if direction not in grouped:
            continue
        item = dict(row)
        item.setdefault("module", INTERMEDIATE_MODULE)
        item.setdefault("direction_sign", int(SIGN_FOR_DIRECTION[direction]))
        grouped[direction][int(item["layer"])].append(item)
    out: dict[str, dict[int, list[dict[str, Any]]]] = {direction: {} for direction in DIRECTIONS}
    for direction in DIRECTIONS:
        for layer, values in grouped[direction].items():
            values.sort(
                key=lambda item: (
                    -safe_float(item.get("tkn_shared_score", item.get("score", 0.0))),
                    int(item["index"]),
                )
            )
            kept = [dict(row) for row in values[:per_direction_per_layer]]
            for rank, row in enumerate(kept, start=1):
                row["tkp_candidate_rank_in_layer_direction"] = rank
            out[direction][layer] = kept
    return out


def phi_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape[0] != right.shape[0]:
        raise ValueError("Phi matrices need the same example count")
    n = max(int(left.shape[0]), 1)
    left = left.float()
    right = right.float()
    p_left = left.mean(dim=0)
    p_right = right.mean(dim=0)
    joint = left.transpose(0, 1).matmul(right) / n
    numerator = joint - p_left[:, None] * p_right[None, :]
    denom = torch.sqrt(
        (p_left * (1.0 - p_left)).clamp_min(0.0)[:, None]
        * (p_right * (1.0 - p_right)).clamp_min(0.0)[None, :]
    )
    return torch.where(denom > 1.0e-8, numerator / denom, torch.zeros_like(numerator))


def module_key_by_layer(payload: dict[str, Any]) -> dict[int, str]:
    return {int(row["layer"]): str(row["key"]) for row in payload["module_meta"]}


def module_dim_by_layer(payload: dict[str, Any]) -> dict[int, int]:
    return {int(row["layer"]): int(row["dim"]) for row in payload["module_meta"]}


def build_activity_cache(
    payload: dict[str, Any],
    candidates: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    quantile: float,
    device: torch.device,
) -> dict[str, dict[int, dict[str, Any]]]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("--activation-quantile must be in (0, 1)")
    activations: dict[str, torch.Tensor] = payload["activations"]
    key_by_layer = module_key_by_layer(payload)
    cache: dict[str, dict[int, dict[str, Any]]] = {direction: {} for direction in DIRECTIONS}
    for direction in DIRECTIONS:
        for layer, rows in progress(
            sorted(candidates[direction].items()),
            desc=f"cache/{direction}",
            unit="layer",
        ):
            key = key_by_layer[layer]
            indices = torch.tensor([int(row["index"]) for row in rows], dtype=torch.long)
            signs = torch.tensor(
                [safe_float(row.get("direction_sign"), SIGN_FOR_DIRECTION[direction]) for row in rows],
                dtype=torch.float32,
                device=device,
            )
            values = activations[key].index_select(1, indices).to(device=device, dtype=torch.float32)
            aligned = values * signs.view(1, -1)
            thresholds = torch.quantile(aligned, quantile, dim=0, keepdim=True)
            cache[direction][layer] = {
                "active": aligned > thresholds,
                "rows": rows,
                "module_key": key,
                "indices": indices,
                "signs": signs.detach().cpu(),
            }
            del values, aligned, thresholds
    return cache


def class_indices(meta_rows: list[dict[str, Any]], task_type: str, label: int) -> torch.Tensor:
    values = [
        idx
        for idx, row in enumerate(meta_rows)
        if str(row.get("task_type")) == task_type and int(row.get("tool_necessary", -1)) == label
    ]
    return torch.tensor(values, dtype=torch.long)


def discover_coactivation_edges(
    meta_rows: list[dict[str, Any]],
    cache: dict[str, dict[int, dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.max_layer_gap < 1:
        raise ValueError("--max-layer-gap must be >= 1")
    if args.edge_top_k < 1:
        raise ValueError("--edge-top-k must be >= 1")
    edges: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        target_label = LABEL_FOR_DIRECTION[direction]
        generic_label = 1 - target_label
        layers = sorted(cache[direction])
        layer_set = set(layers)
        pairs: list[tuple[int, int]] = []
        for src_layer in layers:
            for gap in range(1, args.max_layer_gap + 1):
                dst_layer = src_layer + gap
                if dst_layer in layer_set:
                    pairs.append((src_layer, dst_layer))
        for src_layer, dst_layer in progress(pairs, desc=f"edges/{direction}", unit="pair"):
            src_block = cache[direction][src_layer]
            dst_block = cache[direction][dst_layer]
            src_active = src_block["active"]
            dst_active = dst_block["active"]
            target_phi_by_task: dict[str, torch.Tensor] = {}
            generic_phi_by_task: dict[str, torch.Tensor] = {}
            enough = True
            for task_type in TASK_TYPES:
                target_idx = class_indices(meta_rows, task_type, target_label).to(src_active.device)
                generic_idx = class_indices(meta_rows, task_type, generic_label).to(src_active.device)
                if target_idx.numel() < args.min_samples_per_class or generic_idx.numel() < args.min_samples_per_class:
                    enough = False
                    break
                target_phi_by_task[task_type] = phi_matrix(
                    src_active.index_select(0, target_idx),
                    dst_active.index_select(0, target_idx),
                )
                generic_phi_by_task[task_type] = phi_matrix(
                    src_active.index_select(0, generic_idx),
                    dst_active.index_select(0, generic_idx),
                )
            if not enough:
                continue
            target_consensus = torch.stack([target_phi_by_task[t] for t in TASK_TYPES]).amin(dim=0)
            generic_worst = torch.stack([generic_phi_by_task[t] for t in TASK_TYPES]).amax(dim=0).clamp_min(0.0)
            edge_score = target_consensus - float(args.generic_penalty) * generic_worst
            src_rows = src_block["rows"]
            dst_rows = dst_block["rows"]
            for source_pos, source in enumerate(src_rows):
                scores = edge_score[source_pos]
                eligible = torch.nonzero(
                    (target_consensus[source_pos] >= float(args.min_target_phi))
                    & (scores >= float(args.min_edge_score)),
                    as_tuple=False,
                ).flatten()
                if eligible.numel() == 0:
                    continue
                selected_count = min(int(args.edge_top_k), int(eligible.numel()))
                selected_scores = scores.index_select(0, eligible)
                _values, local_order = torch.topk(selected_scores, k=selected_count, largest=True)
                selected_positions = eligible.index_select(0, local_order).detach().cpu().tolist()
                for target_pos in selected_positions:
                    target = dst_rows[int(target_pos)]
                    row = {
                        "model_alias": args.model_alias,
                        "subset": "",
                        "direction": direction,
                        "target_label": target_label,
                        "generic_label": generic_label,
                        "source_layer": int(source["layer"]),
                        "source_module": str(source.get("module", INTERMEDIATE_MODULE)),
                        "source_module_key": str(source.get("module_key", src_block["module_key"])),
                        "source_index": int(source["index"]),
                        "target_layer": int(target["layer"]),
                        "target_module": str(target.get("module", INTERMEDIATE_MODULE)),
                        "target_module_key": str(target.get("module_key", dst_block["module_key"])),
                        "target_index": int(target["index"]),
                        "layer_gap": int(dst_layer - src_layer),
                        "source_tkn_score": safe_float(source.get("tkn_shared_score", source.get("score", 0.0))),
                        "target_tkn_score": safe_float(target.get("tkn_shared_score", target.get("score", 0.0))),
                        "phi_consensus": float(target_consensus[source_pos, target_pos].detach().cpu()),
                        "generic_phi_worst": float(generic_worst[source_pos, target_pos].detach().cpu()),
                        "coactivation_score": float(edge_score[source_pos, target_pos].detach().cpu()),
                        "edge_source": "directional_phi_minus_opposite_label_generic_control",
                        "causal_validated": False,
                    }
                    for task_type in TASK_TYPES:
                        row[f"target_phi_{task_type}"] = float(target_phi_by_task[task_type][source_pos, target_pos].detach().cpu())
                        row[f"generic_phi_{task_type}"] = float(generic_phi_by_task[task_type][source_pos, target_pos].detach().cpu())
                    edges.append(row)
            del target_consensus, generic_worst, edge_score
    edges.sort(
        key=lambda row: (
            -safe_float(row.get("coactivation_score")),
            str(row["direction"]),
            int(row["source_layer"]),
            int(row["layer_gap"]),
            int(row["source_index"]),
            int(row["target_index"]),
        )
    )
    return edges


def edge_identity(edge: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(edge["direction"]),
        int(edge["source_layer"]),
        int(edge["source_index"]),
        int(edge["target_layer"]),
        int(edge["target_index"]),
    )


def select_edges_for_causal(edges: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.causal_mode == "none":
        return []
    if args.causal_sources_per_layer_direction < 1 or args.causal_targets_per_source < 1:
        return []
    grouped_by_layer_direction: dict[tuple[int, str], dict[tuple[str, int, int], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for edge in edges:
        source_key = (str(edge["direction"]), int(edge["source_layer"]), int(edge["source_index"]))
        grouped_by_layer_direction[(int(edge["source_layer"]), str(edge["direction"]))][source_key].append(edge)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for (_layer, _direction), by_source in sorted(grouped_by_layer_direction.items()):
        source_summaries = []
        for source_key, values in by_source.items():
            best = max(values, key=lambda row: safe_float(row.get("coactivation_score")))
            source_summaries.append(
                (
                    source_key,
                    safe_float(best.get("coactivation_score")) * (1.0 + safe_float(best.get("source_tkn_score"))),
                    values,
                )
            )
        source_summaries.sort(key=lambda item: (-item[1], item[0]))
        for _source_key, _score, values in source_summaries[: int(args.causal_sources_per_layer_direction)]:
            values = sorted(
                values,
                key=lambda row: (
                    -safe_float(row.get("coactivation_score")),
                    int(row["layer_gap"]),
                    int(row["target_index"]),
                ),
            )
            for edge in values[: int(args.causal_targets_per_source)]:
                ident = edge_identity(edge)
                if ident not in seen:
                    selected.append(dict(edge))
                    seen.add(ident)
    selected.sort(key=lambda row: (str(row["direction"]), int(row["source_layer"]), int(row["source_index"]), int(row["target_layer"])))
    return selected


def selected_train_positions(
    meta_rows: list[dict[str, Any]],
    *,
    task_type: str,
    label: int,
    limit: int,
) -> list[int]:
    indices = [
        idx
        for idx, row in enumerate(meta_rows)
        if str(row.get("task_type")) == task_type and int(row.get("tool_necessary", -1)) == label
    ]
    indices.sort(key=lambda idx: str(meta_rows[idx].get("id", "")))
    return indices[: max(1, int(limit))]


def load_model_context(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[int, Any], str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(args.model_alias, args.model_path)
    tool_format = infer_tool_format(args.model_alias, model_path)
    utils = load_utils(args.when2tool_repo)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map=args.device_map,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    layers = {item.layer: item for item in discover_ffn_intermediate_layers(model)}
    return model, tokenizer, utils, layers, tool_format


def causal_mask_edges(
    edges: list[dict[str, Any]],
    payload: dict[str, Any],
    meta_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    subset: str,
) -> list[dict[str, Any]]:
    if not edges:
        return []
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    train_path = dataset_root / args.model_alias / subset / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset for causal masking: {train_path}")
    tasks = {str(row["id"]): row for row in read_jsonl(train_path)}
    model, tokenizer, utils, layers, tool_format = load_model_context(args)
    system_prompt = utils.get_system_prompt(tool_format)
    activations: dict[str, torch.Tensor] = payload["activations"]
    first_device = next(model.parameters()).device

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        groups[(str(edge["direction"]), int(edge["source_layer"]), int(edge["source_index"]))].append(edge)

    updated: list[dict[str, Any]] = []
    for (direction, source_layer, source_index), outgoing in progress(
        sorted(groups.items()),
        desc=f"{subset}/causal-mask",
        unit="source",
    ):
        target_by_layer: dict[int, list[int]] = defaultdict(list)
        target_key_by_layer: dict[int, str] = {}
        target_sign_by_layer: dict[int, dict[int, float]] = defaultdict(dict)
        for edge in outgoing:
            target_layer = int(edge["target_layer"])
            target_index = int(edge["target_index"])
            target_by_layer[target_layer].append(target_index)
            target_key_by_layer[target_layer] = str(edge["target_module_key"])
            target_sign_by_layer[target_layer][target_index] = SIGN_FOR_DIRECTION[direction]
        target_indices_by_layer = {layer: sorted(set(indices)) for layer, indices in target_by_layer.items()}
        target_pos_by_layer = {
            layer: {index: pos for pos, index in enumerate(indices)}
            for layer, indices in target_indices_by_layer.items()
        }
        effect_by_task: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
        condition_label = LABEL_FOR_DIRECTION[direction]
        for task_type in TASK_TYPES:
            positions = selected_train_positions(
                meta_rows,
                task_type=task_type,
                label=condition_label,
                limit=args.causal_samples_per_task,
            )
            if not positions:
                continue
            prompt_rows = []
            baseline_positions = []
            for position in positions:
                task = tasks.get(str(meta_rows[position]["id"]))
                if task is None:
                    raise KeyError(f"Activation id missing from dataset: {meta_rows[position]['id']}")
                prompt_rows.append(
                    build_current_prompt(
                        task,
                        tokenizer=tokenizer,
                        w2t_utils=utils,
                        system_prompt=system_prompt,
                        tool_format=tool_format,
                    )
                )
                baseline_positions.append(position)

            captured: dict[int, list[torch.Tensor]] = defaultdict(list)
            last_indices: torch.Tensor | None = None

            def source_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                if last_indices is None or not inputs:
                    return inputs
                hidden = inputs[0]
                changed = hidden.clone()
                batch = torch.arange(changed.shape[0], device=changed.device)
                changed[batch, last_indices.to(changed.device), int(source_index)] = 0
                return (changed, *inputs[1:])

            def make_target_hook(target_layer: int, target_tensor: torch.Tensor):
                def target_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    if last_indices is None or not inputs:
                        return
                    hidden = inputs[0]
                    batch = torch.arange(hidden.shape[0], device=hidden.device)
                    captured[target_layer].append(
                        hidden[batch, last_indices.to(hidden.device), :]
                        .index_select(1, target_tensor.to(hidden.device))
                        .detach()
                        .cpu()
                    )

                return target_hook

            handles = [layers[source_layer].down.register_forward_pre_hook(source_hook)]
            for target_layer, target_indices in target_indices_by_layer.items():
                target_tensor = torch.tensor(target_indices, dtype=torch.long)
                handles.append(layers[target_layer].down.register_forward_pre_hook(make_target_hook(target_layer, target_tensor)))
            try:
                for start in range(0, len(prompt_rows), int(args.causal_batch_size)):
                    encoded = tokenizer(
                        prompt_rows[start : start + int(args.causal_batch_size)],
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                    )
                    last_indices = encoded["attention_mask"].sum(dim=1) - 1
                    encoded = {name: value.to(first_device) for name, value in encoded.items()}
                    with torch.inference_mode():
                        model(**encoded, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()

            effect_by_task[task_type] = {}
            baseline_index_tensor = torch.tensor(baseline_positions, dtype=torch.long)
            for target_layer, target_indices in target_indices_by_layer.items():
                if target_layer not in captured or not captured[target_layer]:
                    continue
                changed = torch.cat(captured[target_layer], dim=0).float()
                target_tensor = torch.tensor(target_indices, dtype=torch.long)
                baseline = (
                    activations[target_key_by_layer[target_layer]]
                    .index_select(0, baseline_index_tensor)
                    .index_select(1, target_tensor)
                    .float()
                )
                signs = torch.tensor(
                    [target_sign_by_layer[target_layer][idx] for idx in target_indices],
                    dtype=torch.float32,
                )
                aligned_delta = (baseline - changed) * signs.view(1, -1)
                denominator = (baseline * signs.view(1, -1)).std(dim=0, unbiased=False).clamp_min(1.0e-6)
                signed_effect = aligned_delta.mean(dim=0) / denominator
                abs_effect = (baseline - changed).abs().mean(dim=0) / denominator
                for target_index, local_pos in target_pos_by_layer[target_layer].items():
                    signed_value = float(signed_effect[local_pos])
                    abs_value = float(abs_effect[local_pos])
                    effect_by_task[task_type][(target_layer, target_index)] = {
                        "signed": signed_value,
                        "positive": max(signed_value, 0.0),
                        "abs": abs_value,
                    }

        for edge in outgoing:
            item = dict(edge)
            positives = []
            abs_values = []
            for task_type in TASK_TYPES:
                values = effect_by_task.get(task_type, {}).get((int(edge["target_layer"]), int(edge["target_index"])))
                if values is None:
                    continue
                item[f"causal_signed_effect_{task_type}"] = values["signed"]
                item[f"causal_positive_effect_{task_type}"] = values["positive"]
                item[f"causal_abs_effect_{task_type}"] = values["abs"]
                positives.append(values["positive"])
                abs_values.append(values["abs"])
            item["causal_validated"] = True
            item["causal_effect_consensus"] = min(positives) if positives else 0.0
            item["causal_abs_effect_mean"] = sum(abs_values) / len(abs_values) if abs_values else 0.0
            item["causal_pass"] = bool(item["causal_effect_consensus"] >= float(args.min_causal_effect))
            item["causal_edge_score"] = max(safe_float(item.get("coactivation_score")), 0.0) * (
                1.0 + safe_float(item.get("causal_effect_consensus"))
            )
            item["intervention"] = "mask_source_ffn_intermediate_at_last_input_token_measure_directional_target_drop"
            updated.append(item)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return updated


def merge_causal_edges(edges: list[dict[str, Any]], causal_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {edge_identity(edge): edge for edge in causal_edges}
    merged = []
    for edge in edges:
        merged.append(by_id.get(edge_identity(edge), edge))
    merged.sort(
        key=lambda row: (
            -safe_float(row.get("causal_edge_score", row.get("coactivation_score"))),
            str(row["direction"]),
            int(row["source_layer"]),
            int(row["layer_gap"]),
            int(row["source_index"]),
            int(row["target_index"]),
        )
    )
    return merged


def pathway_score(row: dict[str, Any], stats: dict[str, Any]) -> float:
    base = safe_float(row.get("tkn_shared_score", row.get("score", 0.0)))
    coact = safe_float(stats.get("max_coactivation_score"))
    causal = safe_float(stats.get("max_causal_effect"))
    degree = int(stats.get("in_degree", 0)) + int(stats.get("out_degree", 0))
    return base * (1.0 + 2.0 * max(coact, 0.0) + 2.0 * max(causal, 0.0)) + 0.01 * math.log1p(degree)


def build_pathway_nodes(
    candidates: dict[str, dict[int, list[dict[str, Any]]]],
    edges: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    subset: str,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_lookup: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    by_layer_direction: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for direction in DIRECTIONS:
        for layer, rows in candidates[direction].items():
            for row in rows:
                key = stable_row_id(row)
                row_lookup[key] = row
                by_layer_direction[(layer, direction)].append(row)

    node_stats: dict[tuple[int, str, int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "in_degree": 0,
            "out_degree": 0,
            "max_coactivation_score": 0.0,
            "max_causal_effect": 0.0,
            "max_causal_edge_score": 0.0,
            "causal_pass_edges": 0,
            "reasons": set(),
        }
    )
    for edge in edges:
        direction = str(edge["direction"])
        source_key = (int(edge["source_layer"]), str(edge.get("source_module", INTERMEDIATE_MODULE)), int(edge["source_index"]), direction)
        target_key = (int(edge["target_layer"]), str(edge.get("target_module", INTERMEDIATE_MODULE)), int(edge["target_index"]), direction)
        for key, degree_name in [(source_key, "out_degree"), (target_key, "in_degree")]:
            stats = node_stats[key]
            stats[degree_name] += 1
            stats["max_coactivation_score"] = max(stats["max_coactivation_score"], safe_float(edge.get("coactivation_score")))
            stats["max_causal_effect"] = max(stats["max_causal_effect"], safe_float(edge.get("causal_effect_consensus")))
            stats["max_causal_edge_score"] = max(stats["max_causal_edge_score"], safe_float(edge.get("causal_edge_score")))
            if edge.get("causal_pass"):
                stats["causal_pass_edges"] += 1
                stats["reasons"].add("causal")
            else:
                stats["reasons"].add("coactivation")

    selected_keys: set[tuple[int, str, int, str]] = set()
    anchor_count = max(0, int(args.anchor_per_direction_per_layer))
    final_cap = max(0, int(args.final_per_direction_per_layer))
    for (layer, direction), rows in sorted(by_layer_direction.items()):
        anchors = rows[: anchor_count if final_cap == 0 else min(anchor_count, final_cap)]
        for row in anchors:
            key = stable_row_id(row)
            selected_keys.add(key)
            node_stats[key]["reasons"].add("tkn_anchor")
        touched = [row for row in rows if stable_row_id(row) in node_stats]
        touched.sort(key=lambda row: (-pathway_score(row, node_stats[stable_row_id(row)]), int(row["index"])))
        ordered: list[dict[str, Any]] = []
        seen: set[tuple[int, str, int, str]] = set()
        for row in [*anchors, *touched]:
            key = stable_row_id(row)
            if key not in seen:
                ordered.append(row)
                seen.add(key)
        if final_cap > 0:
            ordered = ordered[:final_cap]
        for row in ordered:
            selected_keys.add(stable_row_id(row))

    module_dims = module_dim_by_layer(payload)
    rows_out: list[dict[str, Any]] = []
    for key in selected_keys:
        original = dict(row_lookup[key])
        stats = node_stats[key]
        reasons = sorted(str(item) for item in stats["reasons"]) or ["tkn_anchor"]
        score = pathway_score(original, stats)
        original.update(
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "method": METHOD_NAME,
                "feature_set": FEATURE_SET,
                "module": str(original.get("module", INTERMEDIATE_MODULE)),
                "module_dim": int(original.get("module_dim", module_dims.get(int(original["layer"]), 0))),
                "score": score,
                "tkp_pathway_score": score,
                "initial_tkn_score": safe_float(original.get("tkn_shared_score", original.get("score", 0.0))),
                "path_in_degree": int(stats["in_degree"]),
                "path_out_degree": int(stats["out_degree"]),
                "path_max_coactivation_score": safe_float(stats["max_coactivation_score"]),
                "path_max_causal_effect": safe_float(stats["max_causal_effect"]),
                "path_max_causal_edge_score": safe_float(stats["max_causal_edge_score"]),
                "path_causal_pass_edges": int(stats["causal_pass_edges"]),
                "path_selection_reason": "+".join(reasons),
                "pathway_method": "directional_skip_layer_tkn_pathways",
                "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
            }
        )
        rows_out.append(original)
    rows_out.sort(
        key=lambda row: (
            -safe_float(row.get("tkp_pathway_score")),
            str(row.get("direction", "")),
            int(row["layer"]),
            int(row["index"]),
        )
    )
    for rank, row in enumerate(rows_out, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank

    layer_rows: list[dict[str, Any]] = []
    selected_by_layer_direction = Counter((int(row["layer"]), str(row.get("direction", ""))) for row in rows_out)
    candidate_by_layer_direction = Counter()
    for direction in DIRECTIONS:
        for layer, rows in candidates[direction].items():
            candidate_by_layer_direction[(layer, direction)] = len(rows)
    for layer in sorted({layer for layer, _direction in candidate_by_layer_direction}):
        for direction in DIRECTIONS:
            selected = [row for row in rows_out if int(row["layer"]) == layer and str(row.get("direction")) == direction]
            layer_rows.append(
                {
                    "model_alias": args.model_alias,
                    "subset": subset,
                    "layer": layer,
                    "module": INTERMEDIATE_MODULE,
                    "direction": direction,
                    "candidate_neurons": int(candidate_by_layer_direction.get((layer, direction), 0)),
                    "selected_neurons": int(selected_by_layer_direction.get((layer, direction), 0)),
                    "score_mean": sum(safe_float(row.get("tkp_pathway_score")) for row in selected) / max(len(selected), 1),
                    "score_max": max((safe_float(row.get("tkp_pathway_score")) for row in selected), default=0.0),
                    "anchor_nodes": sum(1 for row in selected if "tkn_anchor" in str(row.get("path_selection_reason", ""))),
                    "causal_nodes": sum(1 for row in selected if "causal" in str(row.get("path_selection_reason", ""))),
                }
            )
    return rows_out, layer_rows


def plot_visualizations(rows: list[dict[str, Any]], edges: list[dict[str, Any]], layer_rows: list[dict[str, Any]], out_dir: Path, subset: str) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Visualization skipped: matplotlib unavailable ({exc})", flush=True)
        return {}

    ensure_dir(out_dir)
    paths: dict[str, str] = {}

    layers = sorted({int(row["layer"]) for row in rows})
    direction_colors = {"tool_high": "#2563eb", "direct_high": "#f97316"}

    count_path = out_dir / f"tkp_node_counts_by_layer_{subset}.png"
    fig, ax = plt.subplots(figsize=(10, max(4, len(layers) * 0.24)))
    left = [0] * len(layers)
    for direction in DIRECTIONS:
        values = [sum(1 for row in rows if int(row["layer"]) == layer and str(row.get("direction")) == direction) for layer in layers]
        ax.barh([str(layer) for layer in layers], values, left=left, label=direction, color=direction_colors[direction])
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlabel("Selected TKP neurons")
    ax.set_ylabel("Layer")
    ax.set_title(f"{subset}: pathway-enhanced TKN nodes")
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(count_path, dpi=180)
    plt.close(fig)
    paths["node_counts"] = str(count_path)

    edge_path = out_dir / f"tkp_path_edges_{subset}.png"
    selected = {(int(row["layer"]), int(row["index"]), str(row.get("direction"))) for row in rows}
    fig, ax = plt.subplots(figsize=(12, 7))
    for edge in edges:
        direction = str(edge["direction"])
        src = (int(edge["source_layer"]), int(edge["source_index"]), direction)
        dst = (int(edge["target_layer"]), int(edge["target_index"]), direction)
        if src not in selected or dst not in selected:
            continue
        strength = max(safe_float(edge.get("causal_edge_score", edge.get("coactivation_score"))), 0.0)
        ax.plot(
            [src[0], dst[0]],
            [src[1], dst[1]],
            color=direction_colors[direction],
            alpha=min(0.65, 0.10 + strength),
            linewidth=0.6,
        )
    for direction in DIRECTIONS:
        points = [row for row in rows if str(row.get("direction")) == direction]
        ax.scatter(
            [int(row["layer"]) for row in points],
            [int(row["index"]) for row in points],
            s=8,
            alpha=0.55,
            color=direction_colors[direction],
            label=direction,
        )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("FFN intermediate index")
    ax.set_title(f"{subset}: directional skip-layer pathway graph")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(edge_path, dpi=180)
    plt.close(fig)
    paths["path_edges"] = str(edge_path)

    gap_path = out_dir / f"tkp_edge_scores_by_gap_{subset}.png"
    gap_scores: dict[int, list[float]] = defaultdict(list)
    for edge in edges:
        gap_scores[int(edge["layer_gap"])].append(safe_float(edge.get("coactivation_score")))
    fig, ax = plt.subplots(figsize=(7, 4))
    gaps = sorted(gap_scores)
    means = [sum(gap_scores[gap]) / max(len(gap_scores[gap]), 1) for gap in gaps]
    ax.bar([str(gap) for gap in gaps], means, color="#0f766e")
    ax.set_xlabel("Layer gap")
    ax.set_ylabel("Mean coactivation score")
    ax.set_title(f"{subset}: edge score by layer gap")
    fig.tight_layout()
    fig.savefig(gap_path, dpi=180)
    plt.close(fig)
    paths["edge_gap_scores"] = str(gap_path)

    retention_path = out_dir / f"tkp_candidate_retention_{subset}.png"
    fig, ax = plt.subplots(figsize=(11, max(4, len(layers) * 0.24)))
    labels = []
    values = []
    for layer in layers:
        for direction in DIRECTIONS:
            row = next((item for item in layer_rows if int(item["layer"]) == layer and str(item["direction"]) == direction), None)
            if row is None or int(row["candidate_neurons"]) == 0:
                continue
            labels.append(f"L{layer} {direction.replace('_high', '')}")
            values.append(float(row["selected_neurons"]) / float(row["candidate_neurons"]))
    ax.barh(labels, values, color="#7c3aed")
    ax.set_xlabel("Selected / candidate")
    ax.set_title(f"{subset}: retention by layer and direction")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(retention_path, dpi=180)
    plt.close(fig)
    paths["candidate_retention"] = str(retention_path)
    return paths


def expected_paths(out_dir: Path) -> list[Path]:
    return [
        out_dir / TKP_FILENAME,
        out_dir / EDGE_FILENAME,
        out_dir / "layer_summary.csv",
        out_dir / "summary.json",
        out_dir / "manifest.json",
    ]


def params_for_subset(
    args: argparse.Namespace,
    *,
    subset: str,
    activation_root: Path,
    tkn_root: Path,
    output_root: Path,
    visualizations_root: Path,
    activation_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "tkp_01_tkn_directional_pathway_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "feature_set": FEATURE_SET,
        "model_alias": args.model_alias,
        "model_path": args.model_path,
        "subset": subset,
        "activation_root": str(activation_root),
        "tkn_neurons_root": str(tkn_root),
        "output_neurons_root": str(output_root),
        "visualizations_root": str(visualizations_root),
        "activation_manifest_params": activation_manifest.get("params", {}),
        "candidate_per_direction_per_layer": args.candidate_per_direction_per_layer,
        "anchor_per_direction_per_layer": args.anchor_per_direction_per_layer,
        "final_per_direction_per_layer": args.final_per_direction_per_layer,
        "max_layer_gap": args.max_layer_gap,
        "activation_quantile": args.activation_quantile,
        "min_target_phi": args.min_target_phi,
        "generic_penalty": args.generic_penalty,
        "min_edge_score": args.min_edge_score,
        "edge_top_k": args.edge_top_k,
        "min_samples_per_class": args.min_samples_per_class,
        "causal_mode": args.causal_mode,
        "causal_sources_per_layer_direction": args.causal_sources_per_layer_direction,
        "causal_targets_per_source": args.causal_targets_per_source,
        "causal_samples_per_task": args.causal_samples_per_task,
        "causal_batch_size": args.causal_batch_size,
        "min_causal_effect": args.min_causal_effect,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "neuron_identity": "(layer, ffn_intermediate, index, direction)",
        "score_definition": (
            "Use train split only. Keep top TKN candidates per layer and direction. "
            "Build direction-aligned binary activity from sign*h at the last input token. "
            "Add skip-layer edges when A/B/C target-label phi coactivation survives an opposite-label generic penalty. "
            "Run sampled source masking only on a stratified edge shortlist, then rank nodes by TKN score plus path support. "
            "High TKN anchors are retained per layer/direction so the pathway stage augments rather than erases TKN."
        ),
    }


def should_skip_subset(out_dir: Path, viz_out_dir: Path, params: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.clean:
        if out_dir.exists():
            clean_directory(out_dir, data_root())
        if viz_out_dir.exists():
            clean_directory(viz_out_dir, data_root())
        return False
    if args.overwrite:
        return False
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists() or not all(path.exists() for path in expected_paths(out_dir)):
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing ToolKnowledgePathways output: {out_dir}", flush=True)
        return True
    return False


def summarize_rows(rows: list[dict[str, Any]], edges: list[dict[str, Any]], causal_shortlist: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer = Counter(int(row["layer"]) for row in rows)
    by_direction = Counter(str(row.get("direction", "")) for row in rows)
    edge_by_gap = Counter(int(edge["layer_gap"]) for edge in edges)
    causal_edges = [edge for edge in edges if edge.get("causal_validated")]
    causal_pass = [edge for edge in causal_edges if edge.get("causal_pass")]
    return {
        "selected_neurons": len(rows),
        "selected_by_direction": dict(sorted(by_direction.items())),
        "selected_layer_min": min(by_layer) if by_layer else None,
        "selected_layer_max": max(by_layer) if by_layer else None,
        "selected_layer_coverage": len(by_layer),
        "selected_by_layer": dict(sorted(by_layer.items())),
        "coactivation_edges": len(edges),
        "edge_by_gap": dict(sorted(edge_by_gap.items())),
        "causal_shortlist_edges": len(causal_shortlist),
        "causal_validated_edges": len(causal_edges),
        "causal_pass_edges": len(causal_pass),
        "score_stats": {
            "min": min((safe_float(row.get("tkp_pathway_score")) for row in rows), default=0.0),
            "mean": sum(safe_float(row.get("tkp_pathway_score")) for row in rows) / max(len(rows), 1),
            "max": max((safe_float(row.get("tkp_pathway_score")) for row in rows), default=0.0),
        },
    }


def run_subset(
    args: argparse.Namespace,
    *,
    subset: str,
    activation_root: Path,
    tkn_root: Path,
    output_root: Path,
    visualizations_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    payload, meta_rows, activation_manifest, tkn_rows = load_inputs(activation_root, tkn_root, args.model_alias, subset)
    params = params_for_subset(
        args,
        subset=subset,
        activation_root=activation_root,
        tkn_root=tkn_root,
        output_root=output_root,
        visualizations_root=visualizations_root,
        activation_manifest=activation_manifest,
    )
    out_dir = output_dir(output_root, args.model_alias, subset)
    viz_out_dir = viz_dir(visualizations_root, args.model_alias, subset)
    if should_skip_subset(out_dir, viz_out_dir, params, args):
        return read_json(out_dir / "summary.json")

    candidates = choose_candidates(tkn_rows, per_direction_per_layer=args.candidate_per_direction_per_layer)
    candidate_counts = {
        direction: {layer: len(rows) for layer, rows in sorted(candidates[direction].items())}
        for direction in DIRECTIONS
    }
    print(f"{subset}: candidate counts by direction/layer = {candidate_counts}", flush=True)

    cache = build_activity_cache(payload, candidates, quantile=args.activation_quantile, device=device)
    edges = discover_coactivation_edges(meta_rows, cache, args)
    for edge in edges:
        edge["subset"] = subset
    causal_shortlist = select_edges_for_causal(edges, args)
    if args.causal_mode == "sampled_mask":
        print(f"{subset}: causal shortlist edges = {len(causal_shortlist)} / {len(edges)}", flush=True)
        causal_edges = causal_mask_edges(causal_shortlist, payload, meta_rows, args, subset=subset)
        edges = merge_causal_edges(edges, causal_edges)
    rows, layer_rows = build_pathway_nodes(candidates, edges, args, subset=subset, payload=payload)
    if not rows:
        raise ValueError(f"{subset}: TKP_TKN_CTD is empty. Lower thresholds or increase candidate caps.")

    ensure_dir(out_dir)
    write_jsonl(out_dir / TKP_FILENAME, rows)
    write_jsonl(out_dir / EDGE_FILENAME, edges)
    write_csv_rows(out_dir / "layer_summary.csv", layer_rows)
    write_csv_rows(out_dir / "top_neurons.csv", rows[: min(len(rows), 1000)])
    visualizations = {} if args.no_visualizations else plot_visualizations(rows, edges, layer_rows, viz_out_dir, subset)
    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "feature_set": FEATURE_SET,
        "neuron_file": str(out_dir / TKP_FILENAME),
        "edge_file": str(out_dir / EDGE_FILENAME),
        "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
        "candidate_counts": candidate_counts,
        "summary": summarize_rows(rows, edges, causal_shortlist),
        "visualizations": visualizations,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    print(
        f"{subset}: wrote {len(rows)} {FEATURE_SET} neurons, {len(edges)} edges "
        f"(causal validated={summary['summary']['causal_validated_edges']}) to {out_dir}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    set_single_gpu(args.gpus)
    if args.causal_batch_size < 1 or args.causal_samples_per_task < 1:
        raise ValueError("Causal batch size and samples per task must be positive")
    if args.anchor_per_direction_per_layer < 0 or args.final_per_direction_per_layer < 0:
        raise ValueError("Anchor/final caps must be >= 0")
    activation_root = resolve_path(args.activations_dir) if args.activations_dir else default_activation_root()
    tkn_root = resolve_path(args.tkn_neurons_dir) if args.tkn_neurons_dir else default_tkn_root()
    output_root = resolve_path(args.output_neurons_dir) if args.output_neurons_dir else default_output_root()
    visualizations_root = resolve_path(args.visualizations_dir) if args.visualizations_dir else default_viz_root()
    device = resolve_compute_device(args.device)
    print(f"{METHOD_NAME}: model={args.model_alias}, subsets={' -> '.join(subset_values(args.subset))}", flush=True)
    print(f"{METHOD_NAME}: tensor device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)

    root_manifest: dict[str, Any] = {
        "stage": "tkp_01_tkn_directional_pathway_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "feature_set": FEATURE_SET,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for subset in subset_values(args.subset):
        summary = run_subset(
            args,
            subset=subset,
            activation_root=activation_root,
            tkn_root=tkn_root,
            output_root=output_root,
            visualizations_root=visualizations_root,
            device=device,
        )
        root_manifest["subsets"][subset] = summary
        compact = summary["summary"]
        summary_rows.append(
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "method": METHOD_NAME,
                "feature_set": FEATURE_SET,
                "selected_neurons": compact["selected_neurons"],
                "layer_coverage": compact["selected_layer_coverage"],
                "layer_min": compact["selected_layer_min"],
                "layer_max": compact["selected_layer_max"],
                "coactivation_edges": compact["coactivation_edges"],
                "causal_validated_edges": compact["causal_validated_edges"],
                "causal_pass_edges": compact["causal_pass_edges"],
                "score_mean": compact["score_stats"]["mean"],
                "score_max": compact["score_stats"]["max"],
            }
        )

    model_root = output_root / args.model_alias / "shared_by_subset"
    ensure_dir(model_root)
    write_json(model_root / "manifest.json", root_manifest)
    write_csv_rows(model_root / "shared_summary.csv", summary_rows)
    print(f"Wrote ToolKnowledgePathways manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
