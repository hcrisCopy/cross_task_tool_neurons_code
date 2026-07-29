from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
PRECISE_DIR = REPO_ROOT / "PreciseShield"
for candidate in (COMMON_DIR, PRECISE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cttn.data import SUBSETS, TASK_TYPES
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, resolve_path
from cttn.progress import progress
from ps_common import INTERMEDIATE_MODULE


STAGE_VERSION = 1
METHOD_NAME = "ToolKnowledgeNeurons"
TDN_FILENAME = "TKN_TDN_neurons.jsonl"
CTD_FILENAME = "TKN_CTD_neurons.jsonl"
LAYER_TOP_SCORE_RATIO = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ToolKnowledgeNeurons TKN-5: discover shared tool-decision neurons in "
            "PreciseShield FFN-intermediate coordinates with paired-shift signed consensus."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="TKN-4 activation root; defaults to ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations.",
    )
    parser.add_argument(
        "--neurons-dir",
        default=None,
        help="TKN neuron output root; defaults to ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help="TKN visualization root; defaults to ../cross_task_tool_neurons_data/tool_knowledge_neurons/visualizations.",
    )
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument(
        "--selection",
        choices=["top_ratio", "knee"],
        default="top_ratio",
        help="Select each layer by fixed top ratio, or by a DNA-style elbow capped by --top-ratio.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.12,
        help="Maximum per-layer selected ratio by TKN shared score.",
    )
    parser.add_argument("--min-neurons-per-layer", type=int, default=64)
    parser.add_argument("--min-shared-score", type=float, default=0.0)
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
        "--no-down-norm",
        action="store_true",
        help="Disable PreciseShield-style down_proj column norm weighting even when norms are in the activation payload.",
    )
    parser.add_argument(
        "--refine-with-linear-probe",
        action="store_true",
        help="Use a train-only temporary linear probe weight as a DNA-style scoring factor before writing TKN_CTD.",
    )
    parser.add_argument(
        "--refine-keep-ratio",
        type=float,
        default=0.50,
        help="When --refine-with-linear-probe is enabled, keep this global fraction of the broad TKN candidates.",
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=0,
        help="When >0, overrides --refine-keep-ratio with an absolute number of neurons to keep per subset.",
    )
    parser.add_argument("--refine-reg", type=float, default=10000.0)
    parser.add_argument("--refine-max-iter", type=int, default=2000)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--device", default="auto", help="Tensor statistics device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_root(kind: str) -> Path:
    mapping = {
        "activations": data_root() / "tool_knowledge_neurons" / "activations",
        "neurons": data_root() / "tool_knowledge_neurons" / "neurons",
        "visualizations": data_root() / "tool_knowledge_neurons" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown ToolKnowledgeNeurons root kind: {kind}")
    return mapping[kind]


def resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def shared_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "shared_by_subset"


def single_type_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "single_type_by_subset"


def resolve_compute_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    return device


def top_count(dim: int, ratio: float, minimum: int) -> int:
    if not 0 < ratio <= 1:
        raise ValueError("--top-ratio must be in (0, 1]")
    if minimum < 1:
        raise ValueError("--min-neurons-per-layer must be >= 1")
    return min(dim, max(int(math.ceil(float(ratio) * dim)), int(minimum)))


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
    down_norms: dict[str, torch.Tensor] | None,
    key: str,
    device: torch.device,
    eps: float,
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled or not down_norms or key not in down_norms:
        return None
    norm = down_norms[key].to(device).float()
    return torch.nan_to_num(norm / norm.mean().clamp_min(eps), nan=1.0, posinf=1.0, neginf=1.0)


def compute_one_layer(
    *,
    key: str,
    activations: dict[str, torch.Tensor],
    down_norms: dict[str, torch.Tensor] | None,
    call_idx: torch.Tensor,
    direct_idx: torch.Tensor,
    device: torch.device,
    eps: float,
    floor_ratio: float,
    use_down_norm: bool,
) -> tuple[str, dict[str, torch.Tensor | float | bool]]:
    call_x = activations[key].index_select(0, call_idx).to(device, non_blocking=True).float()
    direct_x = activations[key].index_select(0, direct_idx).to(device, non_blocking=True).float()
    delta = call_x - direct_x
    mean_delta = delta.mean(dim=0)
    std_delta = delta.std(dim=0, unbiased=True)
    paired_shift, signed_shift, floor = paired_shift_score(mean_delta, std_delta, eps, floor_ratio)
    norm_factor = norm_factor_for_key(down_norms, key, device, eps, use_down_norm)
    if norm_factor is not None:
        weighted_shift = paired_shift * norm_factor
        weighted_signed = signed_shift * norm_factor
        used_down_norm = True
    else:
        weighted_shift = paired_shift
        weighted_signed = signed_shift
        used_down_norm = False
    z_signed = zscore(weighted_signed, eps)
    out: dict[str, torch.Tensor | float | bool] = {
        "paired_shift_score": paired_shift.detach().cpu(),
        "weighted_shift_score": weighted_shift.detach().cpu(),
        "signed_shift": signed_shift.detach().cpu(),
        "weighted_signed_shift": weighted_signed.detach().cpu(),
        "z_signed_shift": z_signed.detach().cpu(),
        "mean_delta": mean_delta.detach().cpu(),
        "std_delta": std_delta.detach().cpu(),
        "std_floor": float(floor),
        "used_down_norm": bool(used_down_norm),
    }
    del call_x, direct_x, delta, mean_delta, std_delta, paired_shift, signed_shift, weighted_shift, weighted_signed, z_signed
    if norm_factor is not None:
        del norm_factor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return key, out


def compute_scores_for_layers(
    *,
    activations: dict[str, torch.Tensor],
    down_norms: dict[str, torch.Tensor] | None,
    module_meta: list[dict[str, Any]],
    call_indices: list[int],
    direct_indices: list[int],
    device: torch.device,
    eps: float,
    floor_ratio: float,
    use_down_norm: bool,
    desc: str,
) -> dict[str, dict[str, torch.Tensor | float | bool]]:
    call_idx = torch.tensor(call_indices, dtype=torch.long)
    direct_idx = torch.tensor(direct_indices, dtype=torch.long)
    score_pack: dict[str, dict[str, torch.Tensor | float | bool]] = {}
    for meta in progress(module_meta, desc=desc, unit="layer"):
        key, pack = compute_one_layer(
            key=str(meta["key"]),
            activations=activations,
            down_norms=down_norms,
            call_idx=call_idx,
            direct_idx=direct_idx,
            device=device,
            eps=eps,
            floor_ratio=floor_ratio,
            use_down_norm=use_down_norm,
        )
        score_pack[key] = pack
    return score_pack


def consensus_for_layer(
    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float | bool]]],
    key: str,
) -> dict[str, torch.Tensor]:
    packs = [score_by_task[task_type][key] for task_type in TASK_TYPES]
    z_values = [pack["z_signed_shift"] for pack in packs]
    shifts = [pack["weighted_shift_score"] for pack in packs]
    signed = [pack["weighted_signed_shift"] for pack in packs]
    if not all(isinstance(value, torch.Tensor) for value in [*z_values, *shifts, *signed]):
        raise TypeError(f"Missing tensor scores for {key}")
    z_a, z_b, z_c = z_values
    shift_a, shift_b, shift_c = shifts
    signed_a, signed_b, signed_c = signed
    positive = torch.minimum(torch.minimum(z_a, z_b), z_c)
    negative = torch.minimum(torch.minimum(-z_a, -z_b), -z_c)
    consensus_z = torch.maximum(positive, negative)
    direction_sign = torch.where(positive >= negative, torch.ones_like(consensus_z), -torch.ones_like(consensus_z))
    min_shift = torch.minimum(torch.minimum(shift_a, shift_b), shift_c)
    mean_shift = torch.stack([shift_a, shift_b, shift_c], dim=0).mean(dim=0)
    shared_strength = torch.sqrt(torch.clamp(min_shift, min=0.0) * torch.clamp(mean_shift, min=0.0))
    shared_score = torch.clamp(consensus_z, min=0.0) * shared_strength
    return {
        "score": torch.nan_to_num(shared_score, nan=0.0, posinf=0.0, neginf=0.0),
        "consensus_z": consensus_z,
        "direction_sign": direction_sign,
        "positive_consensus": positive,
        "negative_consensus": negative,
        "shared_strength": shared_strength,
        "min_weighted_shift": min_shift,
        "mean_weighted_shift": mean_shift,
        "z_A": z_a,
        "z_B": z_b,
        "z_C": z_c,
        "weighted_shift_A": shift_a,
        "weighted_shift_B": shift_b,
        "weighted_shift_C": shift_c,
        "weighted_signed_A": signed_a,
        "weighted_signed_B": signed_b,
        "weighted_signed_C": signed_c,
    }


def knee_count(sorted_scores: torch.Tensor, maximum: int, minimum: int) -> int:
    n = int(sorted_scores.numel())
    if n <= 0:
        return 0
    if n <= minimum:
        return n
    if n == 1 or float(sorted_scores[0] - sorted_scores[-1]) <= 1.0e-12:
        return min(n, maximum)
    y = (sorted_scores - sorted_scores[-1]) / (sorted_scores[0] - sorted_scores[-1]).clamp_min(1.0e-12)
    x = torch.linspace(0.0, 1.0, steps=n)
    distance = y - (1.0 - x)
    raw = int(torch.argmax(distance).item()) + 1
    return max(1, min(n, maximum, max(minimum, raw)))


def sort_selected_indices(
    consensus: dict[str, torch.Tensor],
    *,
    max_count: int,
    min_count: int,
    selection: str,
    min_shared_score: float,
) -> list[int]:
    scores = consensus["score"]
    if scores.numel() == 0:
        return []
    candidate = torch.nonzero(scores > min_shared_score, as_tuple=False).flatten()
    if candidate.numel() == 0:
        return []
    candidate_scores = scores.index_select(0, candidate)
    sorted_values, sorted_local = torch.sort(candidate_scores, descending=True)
    if selection == "knee":
        count = knee_count(sorted_values, max_count, min_count)
    else:
        count = min(int(candidate.numel()), max_count)
    selected = candidate.index_select(0, sorted_local[:count]).tolist()
    selected.sort(
        key=lambda idx: (
            -float(scores[idx]),
            -float(consensus["consensus_z"][idx]),
            -float(consensus["shared_strength"][idx]),
            int(idx),
        )
    )
    return [int(idx) for idx in selected]


def selected_rows(
    *,
    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float | bool]]],
    module_meta: list[dict[str, Any]],
    model_alias: str,
    subset: str,
    selection: str,
    top_ratio: float,
    min_neurons_per_layer: int,
    min_shared_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    consensus_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for meta in module_meta:
        key = str(meta["key"])
        dim = int(meta["dim"])
        max_count = top_count(dim, top_ratio, min_neurons_per_layer)
        consensus = consensus_for_layer(score_by_task, key)
        consensus_by_key[key] = consensus
        top_indices = sort_selected_indices(
            consensus,
            max_count=max_count,
            min_count=min_neurons_per_layer,
            selection=selection,
            min_shared_score=min_shared_score,
        )
        selected_scores = [float(consensus["score"][idx]) for idx in top_indices]
        layer_rows.append(
            {
                "model_alias": model_alias,
                "subset": subset,
                "layer": int(meta["layer"]),
                "module": INTERMEDIATE_MODULE,
                "module_key": key,
                "module_dim": dim,
                "selected_neurons": len(top_indices),
                "selection": selection,
                "top_ratio": top_ratio,
                "min_shared_score": min_shared_score,
                "score_mean": sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
                "score_min": min(selected_scores) if selected_scores else 0.0,
                "score_max": max(selected_scores) if selected_scores else 0.0,
            }
        )
        for rank_in_layer, idx in enumerate(top_indices, start=1):
            direction_sign = int(float(consensus["direction_sign"][idx]))
            row: dict[str, Any] = {
                "model_alias": model_alias,
                "subset": subset,
                "layer": int(meta["layer"]),
                "module": INTERMEDIATE_MODULE,
                "module_key": key,
                "index": int(idx),
                "rank_in_layer": rank_in_layer,
                "selection": selection,
                "top_ratio": top_ratio,
                "module_dim": dim,
                "selected_neurons_in_layer": len(top_indices),
                "score": float(consensus["score"][idx]),
                "tkn_shared_score": float(consensus["score"][idx]),
                "consensus_z": float(consensus["consensus_z"][idx]),
                "direction_sign": direction_sign,
                "direction": "tool_high" if direction_sign > 0 else "direct_high",
                "positive_consensus": float(consensus["positive_consensus"][idx]),
                "negative_consensus": float(consensus["negative_consensus"][idx]),
                "shared_strength": float(consensus["shared_strength"][idx]),
                "min_weighted_shift": float(consensus["min_weighted_shift"][idx]),
                "mean_weighted_shift": float(consensus["mean_weighted_shift"][idx]),
                "score_source": "paired_shift_signed_abc_consensus",
                "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
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
                ]:
                    value = pack[src_name]
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(f"Missing {src_name} tensor for {task_type}/{key}")
                    row[f"{out_name}_{task_type}"] = float(value[idx])
                row[f"std_floor_{task_type}"] = float(pack["std_floor"])
            rows.append(row)
    rows.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["consensus_z"]),
            -float(item["shared_strength"]),
            int(item["layer"]),
            int(item["index"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
    return rows, layer_rows, consensus_by_key


def build_candidate_feature_matrix(
    activations: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
) -> torch.Tensor:
    if not rows:
        raise ValueError("Cannot refine an empty TKN candidate set")
    feature_columns: list[torch.Tensor | None] = [None] * len(rows)
    groups: dict[str, list[tuple[int, int]]] = {}
    for position, row in enumerate(rows):
        groups.setdefault(str(row["module_key"]), []).append((position, int(row["index"])))
    for module_key, positions_and_indices in groups.items():
        positions = [pos for pos, _idx in positions_and_indices]
        indices = torch.tensor([idx for _pos, idx in positions_and_indices], dtype=torch.long)
        block = activations[module_key].index_select(1, indices).float().cpu()
        for local_col, global_pos in enumerate(positions):
            feature_columns[global_pos] = block[:, local_col]
    if any(column is None for column in feature_columns):
        raise RuntimeError("Internal error while building TKN refine feature columns")
    return torch.stack([column for column in feature_columns if column is not None], dim=1).contiguous()


def refine_rows_with_linear_probe(
    *,
    rows: list[dict[str, Any]],
    activations: dict[str, torch.Tensor],
    meta_rows: list[dict[str, Any]],
    keep_ratio: float,
    top_k: int,
    reg: float,
    max_iter: int,
    subset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < keep_ratio <= 1:
        raise ValueError("--refine-keep-ratio must be in (0, 1]")
    if reg <= 0:
        raise ValueError("--refine-reg must be positive")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = build_candidate_feature_matrix(activations, rows)
    y = np.array([int(row["tool_necessary"]) for row in meta_rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise ValueError(f"{subset}: refine labels contain only one class")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.numpy())
    clf = LogisticRegression(C=1.0 / reg, solver="lbfgs", max_iter=max_iter, random_state=42)
    clf.fit(X_scaled, y)
    coef_abs = np.abs(clf.coef_[0])
    train_prob = clf.predict_proba(X_scaled)[:, 1]
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score

        train_auroc = float(roc_auc_score(y, train_prob))
        train_acc = float(accuracy_score(y, (train_prob >= 0.5).astype(np.int64)))
    except Exception:
        train_auroc = 0.0
        train_acc = 0.0
    for position, row in enumerate(rows):
        coef = float(clf.coef_[0][position])
        coef_mag = float(coef_abs[position])
        row["initial_rank"] = int(row.get("rank", position + 1))
        row["initial_tkn_score"] = float(row["score"])
        row["refine_linear_coef"] = coef
        row["refine_abs_coef"] = coef_mag
        row["refine_score"] = coef_mag * max(float(row["score"]), 0.0)
        row["score"] = row["refine_score"]
        row["tkn_refined_score"] = row["refine_score"]
        row["selection"] = f"{row.get('selection', 'top_ratio')}+linear_probe_refine"
    keep_count = int(top_k) if top_k > 0 else int(math.ceil(len(rows) * keep_ratio))
    keep_count = max(1, min(len(rows), keep_count))
    rows.sort(
        key=lambda item: (
            -float(item["refine_score"]),
            -float(item["refine_abs_coef"]),
            -float(item["initial_tkn_score"]),
            int(item["layer"]),
            int(item["index"]),
        )
    )
    kept = [dict(row) for row in rows[:keep_count]]
    for rank, row in enumerate(kept, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
    summary = {
        "enabled": True,
        "candidate_neurons": len(rows),
        "kept_neurons": len(kept),
        "keep_ratio": keep_ratio,
        "top_k": top_k,
        "reg": reg,
        "C": 1.0 / reg,
        "max_iter": max_iter,
        "train_auroc": train_auroc,
        "train_accuracy": train_acc,
        "score_definition": "refine_score=abs(train_only_linear_probe_coef)*initial_TKN_score",
    }
    return kept, summary


def layer_summary_from_rows(
    *,
    rows: list[dict[str, Any]],
    module_meta: list[dict[str, Any]],
    model_alias: str,
    subset: str,
    selection: str,
    top_ratio: float,
    min_shared_score: float,
) -> list[dict[str, Any]]:
    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_layer.setdefault(int(row["layer"]), []).append(row)
    layer_rows: list[dict[str, Any]] = []
    for meta in module_meta:
        layer = int(meta["layer"])
        selected = rows_by_layer.get(layer, [])
        scores = [float(row["score"]) for row in selected]
        layer_rows.append(
            {
                "model_alias": model_alias,
                "subset": subset,
                "layer": layer,
                "module": INTERMEDIATE_MODULE,
                "module_key": str(meta["key"]),
                "module_dim": int(meta["dim"]),
                "selected_neurons": len(selected),
                "selection": selection,
                "top_ratio": top_ratio,
                "min_shared_score": min_shared_score,
                "score_mean": sum(scores) / len(scores) if scores else 0.0,
                "score_min": min(scores) if scores else 0.0,
                "score_max": max(scores) if scores else 0.0,
            }
        )
    return layer_rows


def rows_by_task_direction(rows: list[dict[str, Any]], task_type: str) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["task_type"] = task_type
        item["task_score"] = row.get(f"z_{task_type}")
        item["task_weighted_shift"] = row.get(f"weighted_shift_{task_type}")
        copied.append(item)
    copied.sort(
        key=lambda item: (
            -abs(float(item.get("task_score") or 0.0)),
            -float(item.get("task_weighted_shift") or 0.0),
            int(item["layer"]),
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


def plot_density(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    layers = [int(meta["layer"]) for meta in module_meta]
    if not layers:
        write_empty_plot(out_path, "TKN-CTD Shared Neurons")
        return
    dims = {int(meta["layer"]): int(meta["dim"]) for meta in module_meta}
    counts = Counter(int(row["layer"]) for row in rows)
    matrix = [[counts.get(layer, 0) / max(dims.get(layer, 1), 1)] for layer in layers]
    fig, ax = plt.subplots(figsize=(4.2, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("TKN-CTD Shared Neurons")
    ax.set_xticks([0], [INTERMEDIATE_MODULE], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN neuron space")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_heatmap(rows: list[dict[str, Any]], out_path: Path, top_n: int, title: str) -> None:
    selected = rows[: max(1, min(top_n, len(rows)))]
    if not selected:
        write_empty_plot(out_path, title)
        return
    group_order = sorted({int(row["layer"]) for row in selected})
    y_by_group = {group: idx for idx, group in enumerate(group_order)}
    matrix = [[float("nan") for _ in selected] for _ in group_order]
    for x, row in enumerate(selected):
        matrix[y_by_group[int(row["layer"])]][x] = float(row["score"])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("TKN-CTD rank")
    ax.set_ylabel("Layer / FFN neuron space")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{INTERMEDIATE_MODULE}" for layer in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="TKN shared score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_score(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], out_path: Path, title: str) -> None:
    layers = [int(meta["layer"]) for meta in module_meta]
    if not layers:
        write_empty_plot(out_path, title)
        return
    by_layer: dict[int, list[float]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer"]), []).append(float(row["score"]))
    matrix = [[sum(by_layer.get(layer, [0.0])) / max(len(by_layer.get(layer, [])), 1)] for layer in layers]
    fig, ax = plt.subplots(figsize=(4.2, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([0], [INTERMEDIATE_MODULE], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN neuron space")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, label="mean selected score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_score_heatmap(
    rows: list[dict[str, Any]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    *,
    score_field: str,
    score_label: str,
    title: str,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    layers = [int(meta["layer"]) for meta in module_meta]
    if not layers:
        layers = sorted({int(row["layer"]) for row in rows})
    dims = {int(meta["layer"]): int(meta["dim"]) for meta in module_meta}

    row_values: list[list[float]] = []
    row_labels: list[str] = []
    max_cols = 0
    for layer in layers:
        candidates = [
            float(row[score_field])
            for row in rows
            if int(row["layer"]) == layer and row.get(score_field) is not None
        ]
        dim = max(dims.get(layer, len(candidates)), 1)
        k = max(1, int(dim * ratio))
        candidates.sort(reverse=True)
        values = candidates[: min(k, len(candidates))]
        row_values.append(values)
        row_labels.append(f"L{layer}.{INTERMEDIATE_MODULE}")
        max_cols = max(max_cols, k, len(values))

    if not row_values or not any(row_values) or max_cols <= 0:
        write_empty_plot(out_path, title)
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
    ax.set_title(title)
    ax.set_xlabel(f"Neuron rank within top {int(ratio * 100)}% of each layer")
    ax.set_ylabel("Layer / FFN neuron space")
    ticks = list(range(0, max_cols, max(1, max_cols // 10)))
    if ticks and ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label=score_label)
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
    neurons_root: Path,
    viz_root: Path,
    use_down_norm: bool,
) -> dict[str, Any]:
    return {
        "stage": "tkn_05_paired_shift_signed_consensus_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activation_dir": str(activation_dir),
        "activation_manifest": activation_manifest,
        "neurons_root": str(neurons_root),
        "visualizations_root": str(viz_root),
        "selection": args.selection,
        "top_ratio": args.top_ratio,
        "min_neurons_per_layer": args.min_neurons_per_layer,
        "min_shared_score": args.min_shared_score,
        "epsilon": args.epsilon,
        "floor_ratio": args.floor_ratio,
        "min_pairs": args.min_pairs,
        "max_pairs": args.max_pairs,
        "use_down_norm": use_down_norm,
        "refine_with_linear_probe": args.refine_with_linear_probe,
        "refine_keep_ratio": args.refine_keep_ratio,
        "refine_top_k": args.refine_top_k,
        "refine_reg": args.refine_reg,
        "refine_max_iter": args.refine_max_iter,
        "heatmap_top_n": args.heatmap_top_n,
        "layer_top_score_ratio": LAYER_TOP_SCORE_RATIO,
        "score_definition": (
            "For task c in A/B/C, pair tool_necessary=1 and 0 train rows deterministically, "
            "delta=a_tool-a_direct in PreciseShield FFN-intermediate h. "
            "paired_shift=abs(mean(delta))/sqrt(std(delta)^2+floor^2). "
            "signed_shift is optionally weighted by normalized down_proj column norm, then layer-zscored. "
            "shared_score=relu(max(min(z_A,z_B,z_C), min(-z_A,-z_B,-z_C))) * "
            "sqrt(min(weighted_shift_A,B,C) * mean(weighted_shift_A,B,C)). "
            "If refine_with_linear_probe is true, broad candidates are reranked by "
            "abs(train-only logistic probe coefficient) * shared_score."
        ),
        "neuron_identity": "(layer, ffn_intermediate, index)",
    }


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"tkn_ctd_density_heatmap_{subset}.png",
        viz_dir / f"tkn_ctd_score_heatmap_{subset}.png",
        viz_dir / f"tkn_ctd_layer_score_heatmap_{subset}.png",
        viz_dir / f"tkn_ctd_layer_top1pct_score_heatmap_{subset}.png",
    ]


def expected_outputs(out_dir: Path, single_root: Path, viz_dir: Path, subset: str) -> list[Path]:
    paths = [
        out_dir / CTD_FILENAME,
        out_dir / "layer_summary.csv",
        out_dir / "top_neurons.csv",
        out_dir / "layer_counts.csv",
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
        print(f"Skip existing ToolKnowledgeNeurons shared neurons: {out_dir}", flush=True)
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
    device: torch.device,
) -> dict[str, Any]:
    act_dir = activation_root / args.model_alias / subset / "train"
    activation_path = act_dir / "activations.pt"
    meta_path = act_dir / "meta.jsonl"
    manifest_path = act_dir / "manifest.json"
    if not activation_path.exists() or not meta_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing ToolKnowledgeNeurons train activations for {subset}: {act_dir}")

    activation_manifest = read_json(manifest_path)
    shared_dir = shared_root(neurons_root, args.model_alias)
    single_root = single_type_root(neurons_root, args.model_alias)
    out_dir = shared_dir / subset
    viz_dir = viz_root / args.model_alias / "shared_by_subset"

    use_down_norm = not args.no_down_norm
    params = expected_params(
        args,
        subset=subset,
        activation_dir=act_dir,
        activation_manifest=activation_manifest,
        neurons_root=neurons_root,
        viz_root=viz_root,
        use_down_norm=use_down_norm,
    )
    if should_skip(out_dir, single_root, viz_dir, subset, params, overwrite=args.overwrite, clean=args.clean):
        return read_json(out_dir / "summary.json")

    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations: dict[str, torch.Tensor] = payload["activations"]
    module_meta: list[dict[str, Any]] = payload["module_meta"]
    down_norms = payload.get("down_weight_norms")
    if use_down_norm and not down_norms:
        raise ValueError(
            "TKN requires PreciseShield-style down_weight_norms for the default score. "
            "Rerun TKN-4, or pass --no-down-norm to use activation-only paired shifts."
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
        score_by_task[task_type] = compute_scores_for_layers(
            activations=activations,
            down_norms=down_norms,
            module_meta=module_meta,
            call_indices=call_indices,
            direct_indices=direct_indices,
            device=device,
            eps=args.epsilon,
            floor_ratio=args.floor_ratio,
            use_down_norm=use_down_norm,
            desc=f"{subset}/type {task_type} TKN paired shift",
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
        selection=args.selection,
        top_ratio=args.top_ratio,
        min_neurons_per_layer=args.min_neurons_per_layer,
        min_shared_score=args.min_shared_score,
    )
    refine_summary = {"enabled": False}
    if args.refine_with_linear_probe:
        rows, refine_summary = refine_rows_with_linear_probe(
            rows=rows,
            activations=activations,
            meta_rows=meta_rows,
            keep_ratio=args.refine_keep_ratio,
            top_k=args.refine_top_k,
            reg=args.refine_reg,
            max_iter=args.refine_max_iter,
            subset=subset,
        )
        layer_rows = layer_summary_from_rows(
            rows=rows,
            module_meta=module_meta,
            model_alias=args.model_alias,
            subset=subset,
            selection="top_ratio+linear_probe_refine",
            top_ratio=args.top_ratio,
            min_shared_score=args.min_shared_score,
        )

    ensure_dir(out_dir)
    ensure_dir(single_root / subset)
    write_jsonl(out_dir / CTD_FILENAME, rows)
    write_csv_rows(out_dir / "top_neurons.csv", rows)
    write_csv_rows(out_dir / "layer_summary.csv", layer_rows)
    write_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
    write_csv_rows(single_root / subset / "class_balance.csv", summarize_class_balance(meta_rows, args.model_alias, subset))
    write_json(single_root / subset / "module_meta.json", module_meta)

    for task_type in TASK_TYPES:
        type_dir = single_root / subset / task_type
        ensure_dir(type_dir)
        type_rows = rows_by_task_direction(rows, task_type)
        write_jsonl(type_dir / TDN_FILENAME, type_rows)
        write_csv_rows(type_dir / "top_neurons.csv", type_rows)
        write_counts_csv(type_rows, type_dir / "layer_counts.csv", "layer")
        write_jsonl(type_dir / "label_pairs.jsonl", pair_rows_by_type[task_type])
        write_json(
            type_dir / "summary.json",
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "task_type": task_type,
                "method": METHOD_NAME,
                "neuron_set": "TKN_TDN",
                "selected_neurons": len(type_rows),
                "class_summary": class_summaries[task_type],
            },
        )

    density_path = viz_dir / f"tkn_ctd_density_heatmap_{subset}.png"
    score_path = viz_dir / f"tkn_ctd_score_heatmap_{subset}.png"
    layer_score_path = viz_dir / f"tkn_ctd_layer_score_heatmap_{subset}.png"
    layer_top1pct_path = viz_dir / f"tkn_ctd_layer_top1pct_score_heatmap_{subset}.png"
    plot_density(rows, module_meta, density_path)
    plot_score_heatmap(rows, score_path, args.heatmap_top_n, f"{subset} TKN-CTD paired-shift signed consensus")
    plot_layer_score(rows, module_meta, layer_score_path, f"{subset} TKN-CTD mean selected score")
    plot_layer_top_score_heatmap(
        rows,
        module_meta,
        layer_top1pct_path,
        score_field="score",
        score_label="TKN shared score",
        title=f"{subset} TKN-CTD: top 1% shared score by layer",
    )

    torch.save(
        {
            "method": METHOD_NAME,
            "subset": subset,
            "selection": args.selection,
            "top_ratio": args.top_ratio,
            "use_down_norm": use_down_norm,
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
        out_dir / "tkn_scores.pt",
    )

    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_file": str(out_dir / CTD_FILENAME),
        "neuron_set": "TKN_CTD",
        "selected_neurons": len(rows),
        "selection": args.selection,
        "top_ratio": args.top_ratio,
        "use_down_norm": use_down_norm,
        "refine": refine_summary,
        "class_summaries": class_summaries,
        "score_stats": {
            "min": min((float(row["score"]) for row in rows), default=0.0),
            "mean": sum(float(row["score"]) for row in rows) / max(len(rows), 1),
            "max": max((float(row["score"]) for row in rows), default=0.0),
        },
        "strength_stats": {
            "min_weighted_shift_mean": sum(float(row["min_weighted_shift"]) for row in rows) / max(len(rows), 1),
            "shared_strength_mean": sum(float(row["shared_strength"]) for row in rows) / max(len(rows), 1),
        },
        "top_layers": Counter(int(row["layer"]) for row in rows).most_common(10),
        "visualizations": {
            "density_heatmap": str(density_path),
            "score_heatmap": str(score_path),
            "layer_score_heatmap": str(layer_score_path),
            "layer_top1pct_score_heatmap": str(layer_top1pct_path),
        },
    }
    if not rows:
        summary["warning"] = "TKN_CTD is empty. Increase --top-ratio or lower --min-shared-score."
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    write_json(single_root / subset / "manifest.json", {"params": params, "summary": summary})
    print(
        f"{subset}: TKN_CTD={len(rows)}, selection={args.selection}, top_ratio={args.top_ratio}, "
        f"score_mean={summary['score_stats']['mean']:.4f}, score_max={summary['score_stats']['max']:.4f}, "
        f"use_down_norm={use_down_norm}, refine={refine_summary.get('enabled', False)}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    activation_root = resolve_root(args.activations_dir, "activations")
    neurons_root = resolve_root(args.neurons_dir, "neurons")
    viz_root = resolve_root(args.visualizations_dir, "visualizations")
    device = resolve_compute_device(args.device)
    print(f"ToolKnowledgeNeurons compute device: {device}", flush=True)
    print(f"ToolKnowledgeNeurons subset order = {' -> '.join(subset_values(args.subset))}", flush=True)

    root_manifest: dict[str, Any] = {
        "stage": "tkn_05_paired_shift_signed_consensus_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for subset in progress(subset_values(args.subset), desc=f"TKN-5 {args.model_alias}", unit="subset"):
        summary = run_subset(
            args,
            subset=subset,
            activation_root=activation_root,
            neurons_root=neurons_root,
            viz_root=viz_root,
            device=device,
        )
        root_manifest["subsets"][subset] = summary
        summary_rows.append(
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "method": METHOD_NAME,
                "neuron_set": "TKN_CTD",
                "selected_neurons": summary["selected_neurons"],
                "score_mean": summary["score_stats"]["mean"],
                "score_max": summary["score_stats"]["max"],
                "use_down_norm": summary["use_down_norm"],
                "refine_enabled": summary.get("refine", {}).get("enabled", False),
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
            "use_down_norm",
            "refine_enabled",
        ],
    )
    write_json(model_root / "manifest.json", root_manifest)
    print(f"Wrote ToolKnowledgeNeurons shared manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
