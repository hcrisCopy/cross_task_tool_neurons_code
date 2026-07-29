from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cttn.data import SUBSETS, TASK_TYPES
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, resolve_path
from cttn.progress import progress


STAGE_VERSION = 1
METHOD_NAME = "ToolRoutingNeurons"
TDN_FILENAME = "TRN_TDN_neurons.jsonl"
CTD_FILENAME = "TRN_CTD_neurons.jsonl"
MODULE_ORDER = ["attn_q", "attn_k", "attn_v", "attn_o_in"]
LAYER_TOP_SCORE_RATIO = 0.01
MAX_TYPE_TOP_RATIO = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRN-5: discover A/B/C shared attention-routing neurons from train-split "
            "Q/K/V output and O-input activations."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="TRN-4 activation root; defaults to ../cross_task_tool_neurons_data/tool_routing_neurons/activations.",
    )
    parser.add_argument(
        "--neurons-dir",
        default=None,
        help="TRN neuron output root; defaults to ../cross_task_tool_neurons_data/tool_routing_neurons/neurons.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help="TRN visualization root; defaults to ../cross_task_tool_neurons_data/tool_routing_neurons/visualizations.",
    )
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument(
        "--modules",
        default="attn_q,attn_k,attn_v,attn_o_in",
        help="Comma-separated attention neuron spaces to score: attn_q,attn_k,attn_v,attn_o_in.",
    )
    parser.add_argument(
        "--type-top-ratio",
        type=float,
        default=0.10,
        help="Per task-type, per layer/module candidate ratio before the A/B/C intersection. Must be <= 0.10.",
    )
    parser.add_argument("--min-neurons-per-module", type=int, default=1)
    parser.add_argument("--min-shared-score", type=float, default=0.0)
    parser.add_argument("--min-consensus-z", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument(
        "--floor-ratio",
        type=float,
        default=0.05,
        help="paired_shift floor = max(epsilon, median(valid std_delta) * floor_ratio).",
    )
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all deterministic label pairs.")
    parser.add_argument(
        "--no-proj-norm",
        action="store_true",
        help="Disable projection weight norm weighting even when TRN-4 saved projection_weight_norms.",
    )
    parser.add_argument(
        "--support-power",
        type=float,
        default=0.5,
        help="Exponent for the activation support factor mean(abs(delta) > floor). Use 0 to disable support weighting.",
    )
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--device", default="auto", help="Tensor statistics device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_root(kind: str) -> Path:
    mapping = {
        "activations": data_root() / "tool_routing_neurons" / "activations",
        "neurons": data_root() / "tool_routing_neurons" / "neurons",
        "visualizations": data_root() / "tool_routing_neurons" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown ToolRoutingNeurons root kind: {kind}")
    return mapping[kind]


def resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def shared_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "shared_by_subset"


def single_type_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "single_type_by_subset"


def parse_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--modules must contain at least one module")
    unknown = [item for item in modules if item not in MODULE_ORDER]
    if unknown:
        raise ValueError(f"Unknown modules: {unknown}. Valid: {MODULE_ORDER}")
    return sorted(set(modules), key=MODULE_ORDER.index)


def resolve_compute_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    return device


def top_count(dim: int, ratio: float, minimum: int) -> int:
    if not 0 < ratio <= MAX_TYPE_TOP_RATIO:
        raise ValueError(f"--type-top-ratio must be in (0, {MAX_TYPE_TOP_RATIO}]")
    if minimum < 1:
        raise ValueError("--min-neurons-per-module must be >= 1")
    return min(dim, max(int(math.ceil(float(ratio) * dim)), int(minimum)))


def selected_module_meta(module_meta: list[dict[str, Any]], modules: list[str]) -> list[dict[str, Any]]:
    selected = [meta for meta in module_meta if str(meta["module"]) in modules]
    module_rank = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    selected.sort(key=lambda meta: (int(meta["layer"]), module_rank.get(str(meta["module"]), 99), str(meta["module"])))
    if not selected:
        raise ValueError(f"No requested modules found in activation payload: {modules}")
    return selected


def sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("env_name", "")), str(row.get("difficulty", "")), str(row["id"]))


def make_label_pairs(
    meta_rows: list[dict[str, Any]],
    task_type: str,
    *,
    min_pairs: int,
    max_pairs: int,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    call_indices = [idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 1]
    direct_indices = [idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 0]
    call_indices.sort(key=lambda idx: sort_key(meta_rows[idx]))
    direct_indices.sort(key=lambda idx: sort_key(meta_rows[idx]))
    pair_count = min(len(call_indices), len(direct_indices))
    if max_pairs > 0:
        pair_count = min(pair_count, max_pairs)
    if pair_count < min_pairs:
        raise ValueError(
            f"{task_type}: need at least {min_pairs} tool/direct pairs, got "
            f"tool_necessary=1: {len(call_indices)}, tool_necessary=0: {len(direct_indices)}, pairs={pair_count}"
        )
    call_indices = call_indices[:pair_count]
    direct_indices = direct_indices[:pair_count]
    pair_rows = []
    for pair_rank, (call_idx, direct_idx) in enumerate(zip(call_indices, direct_indices), start=1):
        call = meta_rows[call_idx]
        direct = meta_rows[direct_idx]
        pair_rows.append(
            {
                "pair_rank": pair_rank,
                "task_type": task_type,
                "call_row_index": call_idx,
                "direct_row_index": direct_idx,
                "call_id": str(call["id"]),
                "direct_id": str(direct["id"]),
                "call_env_name": call.get("env_name", ""),
                "direct_env_name": direct.get("env_name", ""),
                "call_difficulty": call.get("difficulty", "unknown"),
                "direct_difficulty": direct.get("difficulty", "unknown"),
            }
        )
    return call_indices, direct_indices, pair_rows


def paired_shift_score(mean_delta: torch.Tensor, std_delta: torch.Tensor, eps: float, floor_ratio: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    valid = std_delta[torch.isfinite(std_delta) & (std_delta > eps)]
    floor = max(eps, float(torch.median(valid).item()) * floor_ratio) if valid.numel() else 1.0
    denom = torch.sqrt(std_delta.square() + floor * floor)
    signed = mean_delta / denom
    score = signed.abs()
    signed = torch.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score, signed, floor


def zscore(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    center = tensor.mean()
    scale = tensor.std(unbiased=False).clamp_min(eps)
    out = (tensor - center) / scale
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def norm_factor_for_key(
    projection_norms: dict[str, torch.Tensor] | None,
    key: str,
    device: torch.device,
    eps: float,
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled or not projection_norms or key not in projection_norms:
        return None
    norm = projection_norms[key].to(device).float()
    return torch.nan_to_num(norm / norm.mean().clamp_min(eps), nan=1.0, posinf=1.0, neginf=1.0)


def compute_one_module(
    *,
    key: str,
    activations: dict[str, torch.Tensor],
    projection_norms: dict[str, torch.Tensor] | None,
    call_idx: torch.Tensor,
    direct_idx: torch.Tensor,
    device: torch.device,
    eps: float,
    floor_ratio: float,
    use_proj_norm: bool,
    support_power: float,
) -> tuple[str, dict[str, torch.Tensor | float | bool]]:
    call_x = activations[key].index_select(0, call_idx).to(device, non_blocking=True).float()
    direct_x = activations[key].index_select(0, direct_idx).to(device, non_blocking=True).float()
    delta = call_x - direct_x
    mean_delta = delta.mean(dim=0)
    std_delta = delta.std(dim=0, unbiased=True)
    paired_shift, signed_shift, floor = paired_shift_score(mean_delta, std_delta, eps, floor_ratio)
    support = (delta.abs() > floor).float().mean(dim=0)
    if support_power != 0.0:
        support_factor = support.clamp_min(eps).pow(float(support_power))
    else:
        support_factor = torch.ones_like(support)
    norm_factor = norm_factor_for_key(projection_norms, key, device, eps, use_proj_norm)
    if norm_factor is not None:
        weighted_shift = paired_shift * norm_factor * support_factor
        weighted_signed = signed_shift * norm_factor * support_factor
        used_proj_norm = True
    else:
        weighted_shift = paired_shift * support_factor
        weighted_signed = signed_shift * support_factor
        used_proj_norm = False
    z_signed = zscore(weighted_signed, eps)
    task_top_score = torch.nan_to_num(weighted_shift * z_signed.abs().clamp_min(eps), nan=0.0, posinf=0.0, neginf=0.0)
    out: dict[str, torch.Tensor | float | bool] = {
        "paired_shift_score": paired_shift.detach().cpu(),
        "weighted_shift_score": weighted_shift.detach().cpu(),
        "signed_shift": signed_shift.detach().cpu(),
        "weighted_signed_shift": weighted_signed.detach().cpu(),
        "z_signed_shift": z_signed.detach().cpu(),
        "task_top_score": task_top_score.detach().cpu(),
        "activation_support": support.detach().cpu(),
        "mean_delta": mean_delta.detach().cpu(),
        "std_delta": std_delta.detach().cpu(),
        "std_floor": float(floor),
        "used_proj_norm": bool(used_proj_norm),
    }
    del (
        call_x,
        direct_x,
        delta,
        mean_delta,
        std_delta,
        paired_shift,
        signed_shift,
        support,
        support_factor,
        weighted_shift,
        weighted_signed,
        z_signed,
        task_top_score,
    )
    if norm_factor is not None:
        del norm_factor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return key, out


def compute_scores_for_modules(
    *,
    activations: dict[str, torch.Tensor],
    projection_norms: dict[str, torch.Tensor] | None,
    module_meta: list[dict[str, Any]],
    call_indices: list[int],
    direct_indices: list[int],
    device: torch.device,
    eps: float,
    floor_ratio: float,
    use_proj_norm: bool,
    support_power: float,
    desc: str,
) -> dict[str, dict[str, torch.Tensor | float | bool]]:
    call_idx = torch.tensor(call_indices, dtype=torch.long)
    direct_idx = torch.tensor(direct_indices, dtype=torch.long)
    score_pack: dict[str, dict[str, torch.Tensor | float | bool]] = {}
    for meta in progress(module_meta, desc=desc, unit="module"):
        key, pack = compute_one_module(
            key=str(meta["key"]),
            activations=activations,
            projection_norms=projection_norms,
            call_idx=call_idx,
            direct_idx=direct_idx,
            device=device,
            eps=eps,
            floor_ratio=floor_ratio,
            use_proj_norm=use_proj_norm,
            support_power=support_power,
        )
        score_pack[key] = pack
    return score_pack


def consensus_for_module(
    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float | bool]]],
    key: str,
) -> dict[str, torch.Tensor]:
    packs = [score_by_task[task_type][key] for task_type in TASK_TYPES]
    z_values = [pack["z_signed_shift"] for pack in packs]
    shifts = [pack["weighted_shift_score"] for pack in packs]
    top_scores = [pack["task_top_score"] for pack in packs]
    signed = [pack["weighted_signed_shift"] for pack in packs]
    supports = [pack["activation_support"] for pack in packs]
    if not all(isinstance(value, torch.Tensor) for value in [*z_values, *shifts, *top_scores, *signed, *supports]):
        raise TypeError(f"Missing tensor scores for {key}")
    z_a, z_b, z_c = z_values
    shift_a, shift_b, shift_c = shifts
    top_a, top_b, top_c = top_scores
    signed_a, signed_b, signed_c = signed
    support_a, support_b, support_c = supports
    positive = torch.minimum(torch.minimum(z_a, z_b), z_c)
    negative = torch.minimum(torch.minimum(-z_a, -z_b), -z_c)
    consensus_z = torch.maximum(positive, negative)
    direction_sign = torch.where(positive >= negative, torch.ones_like(consensus_z), -torch.ones_like(consensus_z))
    min_shift = torch.minimum(torch.minimum(shift_a, shift_b), shift_c)
    mean_shift = torch.stack([shift_a, shift_b, shift_c], dim=0).mean(dim=0)
    min_top_score = torch.minimum(torch.minimum(top_a, top_b), top_c)
    mean_top_score = torch.stack([top_a, top_b, top_c], dim=0).mean(dim=0)
    shared_strength = torch.sqrt(torch.clamp(min_top_score, min=0.0) * torch.clamp(mean_top_score, min=0.0))
    shared_score = torch.clamp(consensus_z, min=0.0) * shared_strength
    mean_support = torch.stack([support_a, support_b, support_c], dim=0).mean(dim=0)
    return {
        "score": torch.nan_to_num(shared_score, nan=0.0, posinf=0.0, neginf=0.0),
        "consensus_z": consensus_z,
        "direction_sign": direction_sign,
        "positive_consensus": positive,
        "negative_consensus": negative,
        "shared_strength": shared_strength,
        "min_weighted_shift": min_shift,
        "mean_weighted_shift": mean_shift,
        "min_task_top_score": min_top_score,
        "mean_task_top_score": mean_top_score,
        "mean_activation_support": mean_support,
        "z_A": z_a,
        "z_B": z_b,
        "z_C": z_c,
        "weighted_shift_A": shift_a,
        "weighted_shift_B": shift_b,
        "weighted_shift_C": shift_c,
        "weighted_signed_A": signed_a,
        "weighted_signed_B": signed_b,
        "weighted_signed_C": signed_c,
        "task_top_score_A": top_a,
        "task_top_score_B": top_b,
        "task_top_score_C": top_c,
    }


def top_set_by_task_score(score: torch.Tensor, k: int) -> set[int]:
    if score.numel() == 0:
        return set()
    k = min(k, int(score.numel()))
    values = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if k <= 0:
        return set()
    return {int(idx) for idx in torch.topk(values, k=k, largest=True, sorted=False).indices.tolist()}


def sort_intersection_indices(
    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float | bool]]],
    consensus: dict[str, torch.Tensor],
    *,
    key: str,
    k: int,
    min_shared_score: float,
    min_consensus_z: float,
) -> list[int]:
    task_sets = []
    for task_type in TASK_TYPES:
        task_score = score_by_task[task_type][key]["task_top_score"]
        if not isinstance(task_score, torch.Tensor):
            raise TypeError(f"Missing task_top_score for {task_type}/{key}")
        task_sets.append(top_set_by_task_score(task_score, k))
    shared = set.intersection(*task_sets) if task_sets else set()
    scores = consensus["score"]
    consensus_z = consensus["consensus_z"]
    strength = consensus["shared_strength"]
    filtered = [
        idx
        for idx in shared
        if float(scores[idx]) > min_shared_score and float(consensus_z[idx]) > min_consensus_z
    ]
    filtered.sort(
        key=lambda idx: (
            -float(scores[idx]),
            -float(consensus_z[idx]),
            -float(strength[idx]),
            int(idx),
        )
    )
    return [int(idx) for idx in filtered]


def selected_rows(
    *,
    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float | bool]]],
    module_meta: list[dict[str, Any]],
    model_alias: str,
    subset: str,
    type_top_ratio: float,
    min_neurons_per_module: int,
    min_shared_score: float,
    min_consensus_z: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    consensus_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for meta in module_meta:
        key = str(meta["key"])
        dim = int(meta["dim"])
        k = top_count(dim, type_top_ratio, min_neurons_per_module)
        consensus = consensus_for_module(score_by_task, key)
        consensus_by_key[key] = consensus
        top_indices = sort_intersection_indices(
            score_by_task,
            consensus,
            key=key,
            k=k,
            min_shared_score=min_shared_score,
            min_consensus_z=min_consensus_z,
        )
        selected_scores = [float(consensus["score"][idx]) for idx in top_indices]
        layer_rows.append(
            {
                "model_alias": model_alias,
                "subset": subset,
                "layer": int(meta["layer"]),
                "module": str(meta["module"]),
                "module_key": key,
                "module_dim": dim,
                "selected_neurons": len(top_indices),
                "abc_intersection_candidate_limit": k,
                "type_top_ratio": type_top_ratio,
                "min_shared_score": min_shared_score,
                "min_consensus_z": min_consensus_z,
                "score_mean": sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
                "score_min": min(selected_scores) if selected_scores else 0.0,
                "score_max": max(selected_scores) if selected_scores else 0.0,
            }
        )
        for rank_in_module, idx in enumerate(top_indices, start=1):
            direction_sign = int(float(consensus["direction_sign"][idx]))
            row: dict[str, Any] = {
                "model_alias": model_alias,
                "subset": subset,
                "layer": int(meta["layer"]),
                "module": str(meta["module"]),
                "module_key": key,
                "index": int(idx),
                "rank_in_module": rank_in_module,
                "type_top_ratio": type_top_ratio,
                "module_dim": dim,
                "selected_neurons_in_module": len(top_indices),
                "score": float(consensus["score"][idx]),
                "trn_shared_score": float(consensus["score"][idx]),
                "consensus_z": float(consensus["consensus_z"][idx]),
                "direction_sign": direction_sign,
                "direction": "tool_high" if direction_sign > 0 else "direct_high",
                "positive_consensus": float(consensus["positive_consensus"][idx]),
                "negative_consensus": float(consensus["negative_consensus"][idx]),
                "shared_strength": float(consensus["shared_strength"][idx]),
                "min_weighted_shift": float(consensus["min_weighted_shift"][idx]),
                "mean_weighted_shift": float(consensus["mean_weighted_shift"][idx]),
                "min_task_top_score": float(consensus["min_task_top_score"][idx]),
                "mean_task_top_score": float(consensus["mean_task_top_score"][idx]),
                "mean_activation_support": float(consensus["mean_activation_support"][idx]),
                "score_source": "attention_routing_paired_shift_weighted_signed_abc_intersection",
                "activation_definition": "last_input_token_attention_qkv_output_or_o_input_coordinate",
                "neuron_definition": "Fei-Shen-style attention projection row/column routing coordinate",
            }
            for task_type in TASK_TYPES:
                pack = score_by_task[task_type][key]
                for src_name, out_name in [
                    ("z_signed_shift", "z"),
                    ("weighted_shift_score", "weighted_shift"),
                    ("paired_shift_score", "paired_shift"),
                    ("weighted_signed_shift", "weighted_signed"),
                    ("signed_shift", "signed_shift"),
                    ("mean_delta", "mean_delta"),
                    ("std_delta", "std_delta"),
                    ("task_top_score", "task_top_score"),
                    ("activation_support", "activation_support"),
                ]:
                    value = pack[src_name]
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(f"Missing {src_name} tensor for {task_type}/{key}")
                    row[f"{out_name}_{task_type}"] = float(value[idx])
                row[f"std_floor_{task_type}"] = float(pack["std_floor"])
                row[f"used_proj_norm_{task_type}"] = bool(pack["used_proj_norm"])
            rows.append(row)
    rows.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["consensus_z"]),
            -float(item["shared_strength"]),
            int(item["layer"]),
            str(item["module"]),
            int(item["index"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
    return rows, layer_rows, consensus_by_key


def rows_by_task_direction(rows: list[dict[str, Any]], task_type: str) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["task_type"] = task_type
        item["task_score"] = row.get(f"task_top_score_{task_type}")
        item["task_z"] = row.get(f"z_{task_type}")
        item["task_signed_shift"] = row.get(f"signed_shift_{task_type}")
        copied.append(item)
    copied.sort(
        key=lambda item: (
            -float(item.get("task_score") or 0.0),
            -abs(float(item.get("task_z") or 0.0)),
            int(item["layer"]),
            str(item["module"]),
            int(item["index"]),
        )
    )
    for rank, item in enumerate(copied, start=1):
        item["rank"] = rank
    return copied


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def write_counts_csv(rows: list[dict[str, Any]], path: Path, field: str) -> None:
    ensure_dir(path.parent)
    counts = Counter(row[field] for row in rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field, "count"])
        for key, count in sorted(counts.items()):
            writer.writerow([key, count])


def write_empty_plot(out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title(title)
    ax.text(0.5, 0.5, "No neurons", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def module_groups(module_meta: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [(int(meta["layer"]), str(meta["module"])) for meta in module_meta]


def plot_density(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    groups = module_groups(module_meta)
    layers = sorted({layer for layer, _module in groups})
    modules = [module for module in MODULE_ORDER if any(group_module == module for _layer, group_module in groups)]
    if not layers or not modules:
        write_empty_plot(out_path, "TRN-CTD Shared Attention Routing Neurons")
        return
    dims = {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}
    counts = Counter((int(row["layer"]), str(row["module"])) for row in rows)
    matrix = []
    for layer in layers:
        matrix.append([counts.get((layer, module), 0) / max(dims.get((layer, module), 1), 1) for module in modules])
    fig, ax = plt.subplots(figsize=(5.4, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("TRN-CTD Shared Attention Routing Neurons")
    ax.set_xticks(range(len(modules)), modules, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("Attention routing neuron space")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="selected ratio")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_heatmap(rows: list[dict[str, Any]], out_path: Path, top_n: int, title: str) -> None:
    selected = rows[: max(1, min(top_n, len(rows)))]
    if not selected:
        write_empty_plot(out_path, title)
        return
    group_order = sorted({(int(row["layer"]), str(row["module"])) for row in selected})
    y_by_group = {group: idx for idx, group in enumerate(group_order)}
    matrix = [[float("nan") for _ in selected] for _ in group_order]
    for x, row in enumerate(selected):
        matrix[y_by_group[(int(row["layer"]), str(row["module"]))]][x] = float(row["score"])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("TRN-CTD rank")
    ax.set_ylabel("Layer / attention routing module")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{module}" for layer, module in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="TRN shared score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_score(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], out_path: Path, title: str) -> None:
    groups = module_groups(module_meta)
    layers = sorted({layer for layer, _module in groups})
    modules = [module for module in MODULE_ORDER if any(group_module == module for _layer, group_module in groups)]
    if not layers or not modules:
        write_empty_plot(out_path, title)
        return
    by_group: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        by_group.setdefault((int(row["layer"]), str(row["module"])), []).append(float(row["score"]))
    matrix = []
    for layer in layers:
        matrix.append([sum(by_group.get((layer, module), [0.0])) / max(len(by_group.get((layer, module), [])), 1) for module in modules])
    fig, ax = plt.subplots(figsize=(5.4, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(modules)), modules, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("Attention routing neuron space")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean selected score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_score_heatmap(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    groups = module_groups(module_meta)
    dims = {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}
    row_values: list[list[float]] = []
    row_labels: list[str] = []
    max_cols = 0
    for layer, module in groups:
        candidates = [float(row["score"]) for row in rows if int(row["layer"]) == layer and str(row["module"]) == module]
        dim = max(dims.get((layer, module), len(candidates)), 1)
        k = max(1, int(dim * LAYER_TOP_SCORE_RATIO))
        candidates.sort(reverse=True)
        values = candidates[: min(k, len(candidates))]
        row_values.append(values)
        row_labels.append(f"L{layer}.{module}")
        max_cols = max(max_cols, k, len(values))

    if not row_values or not any(row_values) or max_cols <= 0:
        write_empty_plot(out_path, "TRN-CTD top 1% score by layer/module")
        return

    matrix = [[float("nan") for _ in range(max_cols)] for _ in row_values]
    for row_idx, values in enumerate(row_values):
        for col_idx, value in enumerate(values):
            matrix[row_idx][col_idx] = value

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#f3f4f6")
    fig_width = max(10, min(42, max_cols * 0.018))
    fig_height = max(6, len(row_labels) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title("TRN-CTD: top 1% shared score by layer/module")
    ax.set_xlabel("Neuron rank within top 1% of each layer/module")
    ax.set_ylabel("Layer / attention routing module")
    ticks = list(range(0, max_cols, max(1, max_cols // 10)))
    if ticks and ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="TRN shared score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    modules: list[str],
    neurons_root: Path,
    viz_root: Path,
    use_proj_norm: bool,
) -> dict[str, Any]:
    return {
        "stage": "trn_05_attention_routing_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activation_dir": str(activation_dir),
        "activation_manifest": activation_manifest,
        "neurons_root": str(neurons_root),
        "visualizations_root": str(viz_root),
        "modules": modules,
        "type_top_ratio": args.type_top_ratio,
        "min_neurons_per_module": args.min_neurons_per_module,
        "min_shared_score": args.min_shared_score,
        "min_consensus_z": args.min_consensus_z,
        "epsilon": args.epsilon,
        "floor_ratio": args.floor_ratio,
        "min_pairs": args.min_pairs,
        "max_pairs": args.max_pairs,
        "use_proj_norm": use_proj_norm,
        "support_power": args.support_power,
        "heatmap_top_n": args.heatmap_top_n,
        "layer_top_score_ratio": LAYER_TOP_SCORE_RATIO,
        "score_definition": (
            "For task type c in A/B/C, deterministically pair train rows with tool_necessary=1 and 0. "
            "In each attention-routing neuron space, delta=a_tool-a_direct and "
            "signed_shift=mean(delta)/sqrt(std(delta)^2+floor^2). "
            "task_top_score=abs(signed_shift)*normalized_projection_norm*support^support_power*abs(z_signed_shift). "
            "Each task type first keeps only the top type_top_ratio neurons per layer/module; TRN_CTD is the "
            "strict A/B/C intersection. The final score keeps only direction-consistent neurons: "
            "shared_score=relu(max(min(z_A,z_B,z_C), min(-z_A,-z_B,-z_C))) * "
            "sqrt(min(task_top_score_A,B,C) * mean(task_top_score_A,B,C))."
        ),
        "neuron_identity": "(layer, attn_q|attn_k|attn_v|attn_o_in, index)",
    }


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"trn_ctd_density_heatmap_{subset}.png",
        viz_dir / f"trn_ctd_score_heatmap_{subset}.png",
        viz_dir / f"trn_ctd_layer_score_heatmap_{subset}.png",
        viz_dir / f"trn_ctd_layer_top1pct_score_heatmap_{subset}.png",
    ]


def expected_outputs(out_dir: Path, single_root: Path, viz_dir: Path, subset: str) -> list[Path]:
    paths = [
        out_dir / CTD_FILENAME,
        out_dir / "layer_summary.csv",
        out_dir / "top_neurons.csv",
        out_dir / "layer_counts.csv",
        out_dir / "module_counts.csv",
        out_dir / "summary.json",
        out_dir / "manifest.json",
        single_root / subset / "manifest.json",
        single_root / subset / "module_meta.json",
        *expected_visualizations(viz_dir, subset),
    ]
    for task_type in TASK_TYPES:
        paths.append(single_root / subset / task_type / TDN_FILENAME)
    return paths


def remove_existing_files(paths: Iterable[Path]) -> None:
    root = data_root().resolve()
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Refusing to remove outside data root: {resolved}")
        path.unlink()


def clean_outputs(out_dir: Path, single_subset_root: Path, viz_dir: Path, subset: str) -> None:
    clean_directory(out_dir, data_root())
    clean_directory(single_subset_root, data_root())
    remove_existing_files(expected_visualizations(viz_dir, subset))


def should_skip(
    out_dir: Path,
    single_root: Path,
    viz_dir: Path,
    subset: str,
    params: dict[str, Any],
    *,
    overwrite: bool,
    clean: bool,
) -> bool:
    single_subset_root = single_root / subset
    if clean:
        clean_outputs(out_dir, single_subset_root, viz_dir, subset)
        return False
    if overwrite:
        return False
    expected = expected_outputs(out_dir, single_root, viz_dir, subset)
    if not all(path.exists() for path in expected):
        return False
    manifest = read_json(out_dir / "manifest.json")
    if manifest.get("params") == params:
        print(f"Skip existing ToolRoutingNeurons shared neurons: {out_dir}", flush=True)
        return True
    return False


def summarize_class_balance(meta_rows: list[dict[str, Any]], model_alias: str, subset: str) -> list[dict[str, Any]]:
    counts = Counter((str(row["task_type"]), int(row["tool_necessary"])) for row in meta_rows)
    rows: list[dict[str, Any]] = []
    for task_type in TASK_TYPES:
        for label in [0, 1]:
            rows.append(
                {
                    "model_alias": model_alias,
                    "subset": subset,
                    "task_type": task_type,
                    "tool_necessary": label,
                    "count": counts.get((task_type, label), 0),
                }
            )
    return rows


def run_subset(
    args: argparse.Namespace,
    *,
    subset: str,
    activation_root: Path,
    neurons_root: Path,
    viz_root: Path,
    modules: list[str],
    device: torch.device,
) -> dict[str, Any]:
    act_dir = activation_root / args.model_alias / subset / "train"
    activation_path = act_dir / "activations.pt"
    meta_path = act_dir / "meta.jsonl"
    manifest_path = act_dir / "manifest.json"
    if not activation_path.exists() or not meta_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing ToolRoutingNeurons train activations for {subset}: {act_dir}")

    activation_manifest = read_json(manifest_path)
    shared_dir = shared_root(neurons_root, args.model_alias)
    single_root = single_type_root(neurons_root, args.model_alias)
    out_dir = shared_dir / subset
    viz_dir = viz_root / args.model_alias / "shared_by_subset"

    use_proj_norm = not args.no_proj_norm
    params = expected_params(
        args,
        subset=subset,
        activation_dir=act_dir,
        activation_manifest=activation_manifest,
        modules=modules,
        neurons_root=neurons_root,
        viz_root=viz_root,
        use_proj_norm=use_proj_norm,
    )
    if should_skip(out_dir, single_root, viz_dir, subset, params, overwrite=args.overwrite, clean=args.clean):
        return read_json(out_dir / "summary.json")

    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations: dict[str, torch.Tensor] = payload["activations"]
    full_module_meta: list[dict[str, Any]] = payload["module_meta"]
    module_meta = selected_module_meta(full_module_meta, modules)
    projection_norms = payload.get("projection_weight_norms")
    if use_proj_norm and not projection_norms:
        raise ValueError(
            "TRN requires projection_weight_norms for the default score. "
            "Rerun TRN-4, or pass --no-proj-norm to use activation-only paired shifts."
        )
    meta_rows = read_jsonl(meta_path)

    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float | bool]]] = {}
    class_summaries: dict[str, Any] = {}
    pair_rows_by_type: dict[str, list[dict[str, Any]]] = {}
    for task_type in TASK_TYPES:
        call_indices, direct_indices, pair_rows = make_label_pairs(
            meta_rows,
            task_type,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
        )
        score_by_task[task_type] = compute_scores_for_modules(
            activations=activations,
            projection_norms=projection_norms,
            module_meta=module_meta,
            call_indices=call_indices,
            direct_indices=direct_indices,
            device=device,
            eps=args.epsilon,
            floor_ratio=args.floor_ratio,
            use_proj_norm=use_proj_norm,
            support_power=args.support_power,
            desc=f"{subset}/type {task_type} TRN paired shift",
        )
        class_summaries[task_type] = {
            "task_type": task_type,
            "n_tool": len(call_indices),
            "n_direct": len(direct_indices),
            "n_pairs": len(pair_rows),
        }
        pair_rows_by_type[task_type] = pair_rows

    rows, layer_rows, consensus_by_key = selected_rows(
        score_by_task=score_by_task,
        module_meta=module_meta,
        model_alias=args.model_alias,
        subset=subset,
        type_top_ratio=args.type_top_ratio,
        min_neurons_per_module=args.min_neurons_per_module,
        min_shared_score=args.min_shared_score,
        min_consensus_z=args.min_consensus_z,
    )

    ensure_dir(out_dir)
    ensure_dir(single_root / subset)
    write_jsonl(out_dir / CTD_FILENAME, rows)
    write_csv_rows(out_dir / "top_neurons.csv", rows)
    write_csv_rows(out_dir / "layer_summary.csv", layer_rows)
    write_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
    write_counts_csv(rows, out_dir / "module_counts.csv", "module")
    write_csv_rows(single_root / subset / "class_balance.csv", summarize_class_balance(meta_rows, args.model_alias, subset))
    write_json(single_root / subset / "module_meta.json", module_meta)

    for task_type in TASK_TYPES:
        type_dir = single_root / subset / task_type
        ensure_dir(type_dir)
        type_rows = rows_by_task_direction(rows, task_type)
        write_jsonl(type_dir / TDN_FILENAME, type_rows)
        write_csv_rows(type_dir / "top_neurons.csv", type_rows)
        write_counts_csv(type_rows, type_dir / "layer_counts.csv", "layer")
        write_counts_csv(type_rows, type_dir / "module_counts.csv", "module")
        write_jsonl(type_dir / "label_pairs.jsonl", pair_rows_by_type[task_type])
        write_json(
            type_dir / "summary.json",
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "task_type": task_type,
                "method": METHOD_NAME,
                "neuron_set": "TRN_TDN",
                "selected_neurons": len(type_rows),
                "class_summary": class_summaries[task_type],
            },
        )

    density_path = viz_dir / f"trn_ctd_density_heatmap_{subset}.png"
    score_path = viz_dir / f"trn_ctd_score_heatmap_{subset}.png"
    layer_score_path = viz_dir / f"trn_ctd_layer_score_heatmap_{subset}.png"
    layer_top1pct_path = viz_dir / f"trn_ctd_layer_top1pct_score_heatmap_{subset}.png"
    plot_density(rows, module_meta, density_path)
    plot_score_heatmap(rows, score_path, args.heatmap_top_n, f"{subset} TRN-CTD ABC-intersection routing score")
    plot_layer_score(rows, module_meta, layer_score_path, f"{subset} TRN-CTD mean selected score")
    plot_layer_top_score_heatmap(rows, module_meta, layer_top1pct_path)

    torch.save(
        {
            "method": METHOD_NAME,
            "subset": subset,
            "modules": modules,
            "type_top_ratio": args.type_top_ratio,
            "use_proj_norm": use_proj_norm,
            "support_power": args.support_power,
            "module_meta": module_meta,
            "score_by_task": {
                task_type: {
                    key: {name: value.to(torch.float32) if isinstance(value, torch.Tensor) else value for name, value in pack.items()}
                    for key, pack in task_scores.items()
                }
                for task_type, task_scores in score_by_task.items()
            },
            "consensus_by_key": {key: {name: value.to(torch.float32) for name, value in pack.items()} for key, pack in consensus_by_key.items()},
        },
        out_dir / "trn_scores.pt",
    )

    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_file": str(out_dir / CTD_FILENAME),
        "neuron_set": "TRN_CTD",
        "selected_neurons": len(rows),
        "modules": modules,
        "type_top_ratio": args.type_top_ratio,
        "use_proj_norm": use_proj_norm,
        "support_power": args.support_power,
        "class_summaries": class_summaries,
        "score_stats": {
            "min": min((float(row["score"]) for row in rows), default=0.0),
            "mean": sum(float(row["score"]) for row in rows) / max(len(rows), 1),
            "max": max((float(row["score"]) for row in rows), default=0.0),
        },
        "strength_stats": {
            "min_weighted_shift_mean": sum(float(row["min_weighted_shift"]) for row in rows) / max(len(rows), 1),
            "shared_strength_mean": sum(float(row["shared_strength"]) for row in rows) / max(len(rows), 1),
            "mean_activation_support": sum(float(row["mean_activation_support"]) for row in rows) / max(len(rows), 1),
        },
        "top_layers": Counter(int(row["layer"]) for row in rows).most_common(10),
        "top_modules": Counter(str(row["module"]) for row in rows).most_common(),
        "visualizations": {
            "density_heatmap": str(density_path),
            "score_heatmap": str(score_path),
            "layer_score_heatmap": str(layer_score_path),
            "layer_top1pct_score_heatmap": str(layer_top1pct_path),
        },
    }
    if not rows:
        summary["warning"] = "TRN_CTD is empty. Increase --type-top-ratio within <=0.10, lower thresholds, or check train label coverage."
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    write_json(single_root / subset / "manifest.json", {"params": params, "summary": summary})
    print(
        f"{subset}: TRN_CTD={len(rows)}, modules={','.join(modules)}, "
        f"type_top_ratio={args.type_top_ratio}, score_mean={summary['score_stats']['mean']:.4f}, "
        f"score_max={summary['score_stats']['max']:.4f}, use_proj_norm={use_proj_norm}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    activation_root = resolve_root(args.activations_dir, "activations")
    neurons_root = resolve_root(args.neurons_dir, "neurons")
    viz_root = resolve_root(args.visualizations_dir, "visualizations")
    modules = parse_modules(args.modules)
    device = resolve_compute_device(args.device)
    print(f"ToolRoutingNeurons compute device: {device}", flush=True)
    print(f"ToolRoutingNeurons modules: {','.join(modules)}", flush=True)
    print(f"ToolRoutingNeurons subset order = {' -> '.join(subset_values(args.subset))}", flush=True)

    root_manifest: dict[str, Any] = {
        "stage": "trn_05_attention_routing_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for subset in progress(subset_values(args.subset), desc=f"TRN-5 {args.model_alias}", unit="subset"):
        summary = run_subset(
            args,
            subset=subset,
            activation_root=activation_root,
            neurons_root=neurons_root,
            viz_root=viz_root,
            modules=modules,
            device=device,
        )
        root_manifest["subsets"][subset] = summary
        summary_rows.append(
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "method": METHOD_NAME,
                "neuron_set": "TRN_CTD",
                "selected_neurons": summary["selected_neurons"],
                "score_mean": summary["score_stats"]["mean"],
                "score_max": summary["score_stats"]["max"],
                "use_proj_norm": summary["use_proj_norm"],
            }
        )

    model_root = shared_root(neurons_root, args.model_alias)
    ensure_dir(model_root)
    write_csv_rows(
        model_root / "shared_summary.csv",
        summary_rows,
        fieldnames=[
            "model_alias",
            "subset",
            "method",
            "neuron_set",
            "selected_neurons",
            "score_mean",
            "score_max",
            "use_proj_norm",
        ],
    )
    write_json(model_root / "manifest.json", root_manifest)
    print(f"Wrote ToolRoutingNeurons shared manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
