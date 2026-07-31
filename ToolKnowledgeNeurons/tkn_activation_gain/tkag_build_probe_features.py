from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cttn.data import SUBSETS, TASK_TYPES, select_label_balanced
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, resolve_path
from cttn.progress import progress


STAGE_VERSION = 1
METHOD_NAME = "TKNActivationGain"
FEATURE_SET = "TKN_AG"
CTD_FILENAME = "TKAG_CTD_neurons.jsonl"
TDN_FILENAME = "TKAG_TDN_neurons.jsonl"
TKN_CTD_FILENAME = "TKN_CTD_neurons.jsonl"
INTERMEDIATE_MODULE = "ffn_intermediate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ProbePrefill features from ToolKnowledgeNeurons with train-only "
            "direction-aligned nonlinear activation gain."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--tkn-neurons-dir", default=None)
    parser.add_argument("--output-neurons-dir", default=None)
    parser.add_argument("--output-probe-root", default=None)
    parser.add_argument("--visualizations-dir", default=None)
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "balanced"], default="first")
    parser.add_argument("--require-per-type-labels", action="store_true")
    parser.add_argument("--seed", type=int, choices=[2026, 42, 123456], default=2026)
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--single-keep-ratio", type=float, default=0.0)
    parser.add_argument("--multi-keep-ratio", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 keeps the ratio-selected neurons.")
    parser.add_argument("--min-neurons-per-layer", type=int, default=1)
    parser.add_argument("--gain-lambda", type=float, default=1.5)
    parser.add_argument("--single-gain-lambda", type=float, default=-1.0)
    parser.add_argument("--multi-gain-lambda", type=float, default=-1.0)
    parser.add_argument("--evidence-power", type=float, default=2.0)
    parser.add_argument(
        "--feature-mode",
        choices=[
            "signed_z_gain",
            "gain_only",
            "signed_z",
            "raw_gain",
            "binary",
            "augmented_gain",
            "dual_evidence",
            "gaussian_llr",
            "augmented_llr",
        ],
        default="signed_z_gain",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["midpoint", "direct_quantile", "zero"],
        default="midpoint",
    )
    parser.add_argument("--direct-quantile", type=float, default=0.90)
    parser.add_argument("--tkn-score-power", type=float, default=0.5)
    parser.add_argument("--label-score-power", type=float, default=1.0)
    parser.add_argument("--llr-clip", type=float, default=20.0, help="0 disables clipping for Gaussian LLR features.")
    parser.add_argument("--append-layer-pool", action="store_true")
    parser.add_argument("--pool-reducers", default="mean,max", help="Comma-separated reducers from: mean,max.")
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_root(kind: str) -> Path:
    mapping = {
        "activations": data_root() / "tool_knowledge_neurons" / "activations",
        "tkn_neurons": data_root() / "tool_knowledge_neurons" / "neurons",
        "output_neurons": data_root() / "tkn_activation_gain" / "neurons",
        "output_probe": data_root() / "probe_prefill" / "tkn_activation_gain",
        "visualizations": data_root() / "tkn_activation_gain" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown root kind: {kind}")
    return mapping[kind]


def resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def resolve_compute_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    return device


def pp_features_root(output_probe_root: Path) -> Path:
    return output_probe_root / "probe_features"


def feature_dir(output_probe_root: Path, model_alias: str, subset: str) -> Path:
    return pp_features_root(output_probe_root) / model_alias / subset


def shared_dir(output_neurons_root: Path, model_alias: str, subset: str) -> Path:
    return output_neurons_root / model_alias / "shared_by_subset" / subset


def single_type_dir(output_neurons_root: Path, model_alias: str, subset: str) -> Path:
    return output_neurons_root / model_alias / "single_type_by_subset" / subset


def viz_dir(visualizations_root: Path, model_alias: str) -> Path:
    return visualizations_root / model_alias / "shared_by_subset"


def load_stage_activations(
    activations_root: Path,
    model_alias: str,
    subset: str,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    act_dir = activations_root / model_alias / subset / split
    payload_path = act_dir / "activations.pt"
    meta_path = act_dir / "meta.jsonl"
    manifest_path = act_dir / "manifest.json"
    if not payload_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing TKN activation outputs: {act_dir}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return payload, read_jsonl(meta_path), manifest


def load_tkn_rows(tkn_root: Path, model_alias: str, subset: str) -> list[dict[str, Any]]:
    path = tkn_root / model_alias / "shared_by_subset" / subset / TKN_CTD_FILENAME
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Missing or empty TKN_CTD rows: {path}")
    rows.sort(key=lambda row: (int(row.get("rank", 10**9)), int(row["layer"]), int(row["index"])))
    return rows


def select_meta_indices(
    meta_rows: list[dict[str, Any]],
    max_samples: int,
    seed: int,
    *,
    strategy: str,
    require_per_type_labels: bool,
) -> list[int]:
    if max_samples <= 0 or max_samples >= len(meta_rows):
        return list(range(len(meta_rows)))
    if strategy == "first":
        selected = list(meta_rows[:max_samples])
    elif strategy == "balanced":
        selected = select_label_balanced(
            meta_rows,
            max_samples,
            seed,
            require_per_type_labels=require_per_type_labels,
        )
    else:
        raise ValueError(f"Unknown sample strategy: {strategy}")
    selected_ids = {str(row["id"]) for row in selected}
    return [idx for idx, row in enumerate(meta_rows) if str(row["id"]) in selected_ids]


def summarize_labels(meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(meta_rows),
        "by_tool_necessary": dict(Counter(str(int(row["tool_necessary"])) for row in meta_rows)),
        "by_task_type": dict(Counter(str(row.get("task_type", "unknown")) for row in meta_rows)),
        "by_difficulty": dict(Counter(str(row.get("difficulty", "unknown")) for row in meta_rows)),
        "by_env": dict(Counter(str(row.get("env_name", "unknown")) for row in meta_rows)),
    }


def module_meta_map(module_meta: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(meta["key"]): meta for meta in module_meta}


def group_row_positions(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for pos, row in enumerate(rows):
        grouped[str(row["module_key"])].append(pos)
    return grouped


def validate_quantile(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("--direct-quantile must be in [0, 1]")


def keep_ratio_for_subset(args: argparse.Namespace, subset: str) -> float:
    ratio = args.single_keep_ratio if subset == "single_hop" else args.multi_keep_ratio
    return float(ratio if ratio > 0 else args.keep_ratio)


def gain_lambda_for_subset(args: argparse.Namespace, subset: str) -> float:
    value = args.single_gain_lambda if subset == "single_hop" else args.multi_gain_lambda
    return float(value if value >= 0 else args.gain_lambda)


def selected_count(total: int, ratio: float, top_k: int) -> int:
    if total <= 0:
        return 0
    if top_k > 0:
        return max(1, min(total, int(top_k)))
    if not 0 < ratio <= 1:
        raise ValueError("--keep-ratio and subset keep ratios must be in (0, 1]")
    return max(1, min(total, int(math.ceil(float(ratio) * total))))


def compute_train_statistics(
    *,
    payload: dict[str, Any],
    meta_rows: list[dict[str, Any]],
    selected_indices: list[int],
    tkn_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    subset: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    validate_quantile(args.direct_quantile)
    activations: dict[str, torch.Tensor] = payload["activations"]
    module_meta = module_meta_map(payload["module_meta"])
    labels = torch.tensor([int(meta_rows[idx]["tool_necessary"]) for idx in selected_indices], dtype=torch.bool, device=device)
    if labels.sum().item() == 0 or labels.sum().item() == labels.numel():
        raise ValueError(f"{subset}: selected train rows contain only one class")
    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    rows = [dict(row) for row in tkn_rows]
    grouped = group_row_positions(rows)
    gain_lambda = gain_lambda_for_subset(args, subset)

    for module_key in progress(sorted(grouped), desc=f"TKAG stats {subset}", unit="layer"):
        if module_key not in activations:
            raise KeyError(f"TKN row references missing activation module: {module_key}")
        if module_key not in module_meta:
            raise KeyError(f"TKN row references missing module_meta entry: {module_key}")
        positions = grouped[module_key]
        col_indices = torch.tensor([int(rows[pos]["index"]) for pos in positions], dtype=torch.long)
        block = activations[module_key].index_select(0, index_tensor).index_select(1, col_indices).to(device).float()
        base_sign = torch.tensor(
            [float(rows[pos].get("direction_sign", 1.0) or 1.0) for pos in positions],
            dtype=torch.float32,
            device=device,
        ).sign()
        base_sign = torch.where(base_sign == 0, torch.ones_like(base_sign), base_sign)
        oriented = block * base_sign.view(1, -1)
        pos_values = oriented[labels]
        neg_values = oriented[~labels]
        pos_mean = pos_values.mean(dim=0)
        neg_mean = neg_values.mean(dim=0)
        gap = pos_mean - neg_mean
        flip = torch.where(gap >= 0, torch.ones_like(gap), -torch.ones_like(gap))
        effective_sign = base_sign * flip
        oriented = block * effective_sign.view(1, -1)
        pos_values = oriented[labels]
        neg_values = oriented[~labels]
        pos_mean = pos_values.mean(dim=0)
        neg_mean = neg_values.mean(dim=0)
        gap = pos_mean - neg_mean
        std = oriented.std(dim=0, unbiased=False).clamp_min(float(args.epsilon))
        pos_std = pos_values.std(dim=0, unbiased=False).clamp_min(float(args.epsilon))
        neg_std = neg_values.std(dim=0, unbiased=False).clamp_min(float(args.epsilon))
        if args.threshold_mode == "midpoint":
            threshold = 0.5 * (pos_mean + neg_mean)
        elif args.threshold_mode == "direct_quantile":
            threshold = torch.quantile(neg_values, float(args.direct_quantile), dim=0)
        else:
            threshold = torch.zeros_like(pos_mean)
        label_z = gap / std
        tkn_score = torch.tensor(
            [float(rows[pos].get("tkn_shared_score", rows[pos].get("score", 0.0)) or 0.0) for pos in positions],
            dtype=torch.float32,
            device=device,
        ).clamp_min(0.0)
        selection_score = (tkn_score + float(args.epsilon)).pow(float(args.tkn_score_power)) * (
            label_z.abs() + float(args.epsilon)
        ).pow(float(args.label_score_power))

        for local, pos in enumerate(positions):
            row = rows[pos]
            row["method"] = METHOD_NAME
            row["neuron_set"] = "TKAG_CTD"
            row["source_neuron_set"] = "TKN_CTD"
            row["source_rank"] = int(row.get("rank", pos + 1))
            row["source_tkn_score"] = float(row.get("tkn_shared_score", row.get("score", 0.0)) or 0.0)
            row["original_direction_sign"] = int(float(row.get("direction_sign", 1.0) or 1.0))
            row["feature_direction_sign"] = int(float(effective_sign[local].item()))
            row["feature_mode"] = args.feature_mode
            row["gain_lambda"] = float(gain_lambda)
            row["evidence_power"] = float(args.evidence_power)
            row["llr_clip"] = float(args.llr_clip)
            row["threshold_mode"] = args.threshold_mode
            row["feature_threshold"] = float(threshold[local].item())
            row["feature_scale"] = float(std[local].item())
            row["train_tool_mean_aligned"] = float(pos_mean[local].item())
            row["train_direct_mean_aligned"] = float(neg_mean[local].item())
            row["train_tool_std_aligned"] = float(pos_std[local].item())
            row["train_direct_std_aligned"] = float(neg_std[local].item())
            row["train_label_gap"] = float(gap[local].item())
            row["train_label_z"] = float(label_z[local].item())
            row["selection_score"] = float(selection_score[local].item())
            row["score"] = float(selection_score[local].item())
            row["rank"] = int(pos + 1)
        del block, oriented, pos_values, neg_values
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def enforce_layer_minimum(
    rows: list[dict[str, Any]],
    *,
    keep_count: int,
    min_neurons_per_layer: int,
) -> list[dict[str, Any]]:
    if min_neurons_per_layer <= 0:
        return sorted(rows, key=lambda row: -float(row["selection_score"]))[:keep_count]
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)
    chosen_ids: set[tuple[int, int]] = set()
    chosen: list[dict[str, Any]] = []
    for layer in sorted(by_layer):
        layer_rows = sorted(by_layer[layer], key=lambda row: -float(row["selection_score"]))
        for row in layer_rows[: min(min_neurons_per_layer, len(layer_rows))]:
            key = (int(row["layer"]), int(row["index"]))
            chosen_ids.add(key)
            chosen.append(row)
    ranked = sorted(rows, key=lambda row: -float(row["selection_score"]))
    for row in ranked:
        if len(chosen) >= keep_count:
            break
        key = (int(row["layer"]), int(row["index"]))
        if key not in chosen_ids:
            chosen_ids.add(key)
            chosen.append(row)
    chosen = sorted(chosen, key=lambda row: -float(row["selection_score"]))[:keep_count]
    return chosen


def feature_components(feature_mode: str) -> list[str]:
    if feature_mode == "augmented_gain":
        return ["signed_z", "signed_z_gain"]
    if feature_mode == "dual_evidence":
        return ["signed_z", "positive_gain", "negative_gain"]
    if feature_mode == "augmented_llr":
        return ["signed_z", "gaussian_llr"]
    return [feature_mode]


def expand_feature_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    components = feature_components(args.feature_mode)
    expanded: list[dict[str, Any]] = []
    for source_pos, row in enumerate(rows, start=1):
        for component in components:
            item = dict(row)
            item["feature_mode"] = args.feature_mode
            item["feature_component"] = component
            item["source_feature_rank"] = int(row.get("rank", source_pos))
            item["feature_multiplier"] = len(components)
            expanded.append(item)
    for rank, row in enumerate(expanded, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
        row["feature_rank"] = rank
        row["feature_dim"] = len(expanded)
        row["selected_source_neurons"] = len(rows)
    return expanded


def parse_pool_reducers(value: str) -> list[str]:
    reducers = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"mean", "max"}
    invalid = [item for item in reducers if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported --pool-reducers: {invalid}; allowed={sorted(allowed)}")
    return reducers or ["mean", "max"]


def make_layer_pool_rows(base_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.append_layer_pool:
        return []
    reducers = parse_pool_reducers(args.pool_reducers)
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        by_layer[int(row["layer"])].append(row)
    pool_rows: list[dict[str, Any]] = []
    for layer in sorted(by_layer):
        layer_rows = by_layer[layer]
        components = sorted({str(row.get("feature_component", row["feature_mode"])) for row in layer_rows})
        template = dict(layer_rows[0])
        for component in components:
            component_scores = [
                float(row.get("selection_score", 0.0))
                for row in layer_rows
                if str(row.get("feature_component", row["feature_mode"])) == component
            ]
            for reducer in reducers:
                row = dict(template)
                row["index"] = -1
                row["method"] = METHOD_NAME
                row["neuron_set"] = "TKAG_LAYER_POOL"
                row["feature_pool"] = True
                row["pool_source_component"] = component
                row["pool_reducer"] = reducer
                row["feature_component"] = f"pool_{component}_{reducer}"
                row["source_feature_rank"] = ""
                row["selection_score"] = sum(component_scores) / max(len(component_scores), 1)
                row["score"] = row["selection_score"]
                pool_rows.append(row)
    start_rank = len(base_rows) + 1
    all_count = len(base_rows) + len(pool_rows)
    for offset, row in enumerate(pool_rows):
        rank = start_rank + offset
        row["rank"] = rank
        row["shared_rank"] = rank
        row["feature_rank"] = rank
        row["feature_dim"] = all_count
    for row in base_rows:
        row["feature_dim"] = all_count
    return pool_rows


def select_feature_rows(rows: list[dict[str, Any]], args: argparse.Namespace, subset: str) -> list[dict[str, Any]]:
    count = selected_count(len(rows), keep_ratio_for_subset(args, subset), args.top_k)
    selected = enforce_layer_minimum(rows, keep_count=count, min_neurons_per_layer=args.min_neurons_per_layer)
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
        row["selected_neurons"] = len(selected)
        row["keep_ratio"] = keep_ratio_for_subset(args, subset)
    return expand_feature_rows(selected, args)


def transform_features(z: torch.Tensor, aligned: torch.Tensor, *, mode: str, gain_lambda: float, power: float) -> torch.Tensor:
    evidence = torch.clamp(z, min=0.0)
    if power != 1.0:
        evidence = evidence.pow(float(power))
    if mode == "signed_z_gain":
        return z + float(gain_lambda) * evidence
    if mode == "positive_gain":
        return float(gain_lambda) * evidence
    if mode == "negative_gain":
        negative = torch.clamp(-z, min=0.0)
        if power != 1.0:
            negative = negative.pow(float(power))
        return float(gain_lambda) * negative
    if mode == "gain_only":
        return evidence
    if mode == "signed_z":
        return z
    if mode == "raw_gain":
        return aligned * (1.0 + float(gain_lambda) * evidence)
    if mode == "binary":
        return (z > 0).float()
    raise ValueError(f"Unknown feature mode: {mode}")


def build_features_for_split(
    *,
    payload: dict[str, Any],
    meta_rows: list[dict[str, Any]],
    selected_indices: list[int],
    feature_rows: list[dict[str, Any]],
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]]]:
    activations: dict[str, torch.Tensor] = payload["activations"]
    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    selected_meta = [meta_rows[idx] for idx in selected_indices]
    labels = np.array([int(row["tool_necessary"]) for row in selected_meta], dtype=np.int64)
    groups = group_row_positions(feature_rows)
    features = torch.empty((len(selected_indices), len(feature_rows)), dtype=torch.float32)

    for module_key in progress(sorted(groups), desc=desc, unit="layer"):
        positions = groups[module_key]
        col_indices = torch.tensor([int(feature_rows[pos]["index"]) for pos in positions], dtype=torch.long)
        block = activations[module_key].index_select(0, index_tensor).index_select(1, col_indices).to(device).float()
        signs = torch.tensor([float(feature_rows[pos]["feature_direction_sign"]) for pos in positions], device=device).float()
        thresholds = torch.tensor([float(feature_rows[pos]["feature_threshold"]) for pos in positions], device=device).float()
        scales = torch.tensor([float(feature_rows[pos]["feature_scale"]) for pos in positions], device=device).float().clamp_min(1.0e-12)
        lambdas = torch.tensor([float(feature_rows[pos]["gain_lambda"]) for pos in positions], device=device).float()
        tool_means = torch.tensor([float(feature_rows[pos]["train_tool_mean_aligned"]) for pos in positions], device=device).float()
        direct_means = torch.tensor([float(feature_rows[pos]["train_direct_mean_aligned"]) for pos in positions], device=device).float()
        tool_stds = torch.tensor([float(feature_rows[pos].get("train_tool_std_aligned", feature_rows[pos]["feature_scale"])) for pos in positions], device=device).float().clamp_min(1.0e-12)
        direct_stds = torch.tensor([float(feature_rows[pos].get("train_direct_std_aligned", feature_rows[pos]["feature_scale"])) for pos in positions], device=device).float().clamp_min(1.0e-12)
        llr_clips = torch.tensor([float(feature_rows[pos].get("llr_clip", 20.0)) for pos in positions], device=device).float()
        powers = {float(feature_rows[pos]["evidence_power"]) for pos in positions}
        if len(powers) != 1:
            raise ValueError("Feature rows in one TKAG build must share evidence_power")
        aligned = block * signs.view(1, -1)
        z = (aligned - thresholds.view(1, -1)) / scales.view(1, -1)
        transformed = torch.empty_like(z)
        components = [str(feature_rows[pos].get("feature_component", feature_rows[pos]["feature_mode"])) for pos in positions]
        for component in sorted(set(components)):
            mask = torch.tensor([name == component for name in components], dtype=torch.bool, device=device)
            if component == "gaussian_llr":
                local_aligned = aligned[:, mask]
                pm = tool_means[mask].view(1, -1)
                nm = direct_means[mask].view(1, -1)
                ps = tool_stds[mask].view(1, -1)
                ns = direct_stds[mask].view(1, -1)
                llr = torch.log(ns / ps) + 0.5 * ((local_aligned - nm) / ns).pow(2) - 0.5 * (
                    (local_aligned - pm) / ps
                ).pow(2)
                clip_values = llr_clips[mask]
                if len(set(clip_values.cpu().tolist())) != 1:
                    raise ValueError("Feature rows in one TKAG build must share llr_clip")
                clip_value = float(clip_values[0].item())
                if clip_value > 0:
                    llr = llr.clamp(min=-clip_value, max=clip_value)
                transformed[:, mask] = llr
                continue
            local_lambdas = lambdas.index_select(0, mask.nonzero(as_tuple=False).flatten())
            if len(set(local_lambdas.cpu().tolist())) == 1:
                transformed[:, mask] = transform_features(
                    z[:, mask],
                    aligned[:, mask],
                    mode=component,
                    gain_lambda=float(local_lambdas[0].item()),
                    power=next(iter(powers)),
                )
            else:
                positive = torch.clamp(z[:, mask], min=0.0).pow(next(iter(powers)))
                if component == "signed_z_gain":
                    transformed[:, mask] = z[:, mask] + local_lambdas.view(1, -1) * positive
                elif component == "positive_gain":
                    transformed[:, mask] = local_lambdas.view(1, -1) * positive
                else:
                    raise ValueError(f"Per-feature lambdas are unsupported for component: {component}")
        features[:, positions] = transformed.detach().cpu()
        del block, aligned, z, transformed
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return features.contiguous(), labels, selected_meta


def append_layer_pool_features(
    features: torch.Tensor,
    base_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> torch.Tensor:
    if not pool_rows:
        return features.contiguous()
    columns: list[torch.Tensor] = []
    for pool_row in pool_rows:
        layer = int(pool_row["layer"])
        source_component = str(pool_row["pool_source_component"])
        positions = [
            idx
            for idx, row in enumerate(base_rows)
            if int(row["layer"]) == layer
            and str(row.get("feature_component", row["feature_mode"])) == source_component
        ]
        if not positions:
            column = torch.zeros((features.shape[0],), dtype=features.dtype)
        else:
            values = features[:, positions]
            reducer = str(pool_row["pool_reducer"])
            if reducer == "mean":
                column = values.mean(dim=1)
            elif reducer == "max":
                column = values.max(dim=1).values
            else:
                raise ValueError(f"Unsupported pool reducer: {reducer}")
        columns.append(column)
    pooled = torch.stack(columns, dim=1)
    return torch.cat([features, pooled], dim=1).contiguous()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    import csv

    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames or rows[0].keys())
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def layer_summary(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], model_alias: str, subset: str) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)
    out = []
    for meta in module_meta:
        layer = int(meta["layer"])
        layer_rows = by_layer.get(layer, [])
        scores = [float(row["selection_score"]) for row in layer_rows]
        out.append(
            {
                "model_alias": model_alias,
                "subset": subset,
                "layer": layer,
                "module": INTERMEDIATE_MODULE,
                "module_key": str(meta["key"]),
                "module_dim": int(meta["dim"]),
                "selected_neurons": len(layer_rows),
                "selection_score_mean": sum(scores) / len(scores) if scores else 0.0,
                "selection_score_max": max(scores) if scores else 0.0,
                "abs_label_z_mean": sum(abs(float(row["train_label_z"])) for row in layer_rows) / len(layer_rows) if layer_rows else 0.0,
            }
        )
    return out


def write_single_type_rows(root: Path, rows: list[dict[str, Any]], subset: str) -> None:
    for task_type in TASK_TYPES:
        type_dir = root / task_type
        ensure_dir(type_dir)
        copied = []
        for row in rows:
            item = dict(row)
            item["task_type"] = task_type
            item["task_score"] = row.get(f"z_{task_type}", row.get("train_label_z", 0.0))
            copied.append(item)
        write_jsonl(type_dir / TDN_FILENAME, copied)
        write_json(
            type_dir / "summary.json",
            {
                "subset": subset,
                "task_type": task_type,
                "method": METHOD_NAME,
                "neuron_set": "TKAG_TDN",
                "selected_neurons": len(copied),
            },
        )


def write_visualizations(rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], out_dir: Path, subset: str) -> dict[str, str]:
    ensure_dir(out_dir)
    count_path = out_dir / f"tkag_layer_counts_{subset}.png"
    score_path = out_dir / f"tkag_layer_score_{subset}.png"
    hist_path = out_dir / f"tkag_selection_score_hist_{subset}.png"

    layers = [int(row["layer"]) for row in layer_rows]
    counts = [int(row["selected_neurons"]) for row in layer_rows]
    scores = [float(row["selection_score_mean"]) for row in layer_rows]
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.35), 4.5))
    ax.bar(layers, counts, color="#4c78a8")
    ax.set_title(f"{subset} TKAG selected neurons by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selected neurons")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(count_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.35), 4.5))
    ax.plot(layers, scores, marker="o", linewidth=2, color="#f58518")
    ax.set_title(f"{subset} TKAG mean selection score by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean selection score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(score_path, dpi=180)
    plt.close(fig)

    values = [float(row["selection_score"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=50, color="#54a24b", alpha=0.9)
    ax.set_title(f"{subset} TKAG selection score distribution")
    ax.set_xlabel("Selection score")
    ax.set_ylabel("Neuron count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)

    return {
        "layer_counts": str(count_path),
        "layer_score": str(score_path),
        "selection_score_hist": str(hist_path),
    }


def safe_clean(path: Path) -> None:
    if path.exists():
        clean_directory(path, data_root())


def should_skip(out_dir: Path, params: dict[str, Any], expected: list[Path], *, overwrite: bool, clean: bool) -> bool:
    if clean:
        safe_clean(out_dir)
        return False
    manifest = out_dir / "manifest.json"
    files = [manifest, *expected]
    if overwrite or not all(path.exists() for path in files):
        return False
    data = read_json(manifest)
    if data.get("params") == params:
        print(f"Skip existing TKAG output: {out_dir}", flush=True)
        return True
    return False


def split_max_samples(args: argparse.Namespace, split: str) -> int:
    return args.max_train_samples if split == "train" else args.max_test_samples


def run_subset(
    args: argparse.Namespace,
    *,
    subset: str,
    activations_root: Path,
    tkn_root: Path,
    output_neurons_root: Path,
    output_probe_root: Path,
    visualizations_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_payload, train_meta, train_manifest = load_stage_activations(activations_root, args.model_alias, subset, "train")
    tkn_rows = load_tkn_rows(tkn_root, args.model_alias, subset)
    train_indices = select_meta_indices(
        train_meta,
        args.max_train_samples,
        args.seed,
        strategy=args.sample_strategy,
        require_per_type_labels=args.require_per_type_labels,
    )
    source_hash = stable_sha256(
        [{"layer": row["layer"], "module": row.get("module", ""), "index": row["index"], "rank": row.get("rank")} for row in tkn_rows]
    )
    params = {
        "stage": "tkag_01_build_probe_features",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activations_root": str(activations_root),
        "tkn_root": str(tkn_root),
        "output_neurons_root": str(output_neurons_root),
        "output_probe_root": str(output_probe_root),
        "visualizations_root": str(visualizations_root),
        "source_tkn_count": len(tkn_rows),
        "source_tkn_sha256": source_hash,
        "activation_manifest_params": train_manifest.get("params", {}),
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "sample_strategy": args.sample_strategy,
        "require_per_type_labels": bool(args.require_per_type_labels),
        "seed": args.seed,
        "keep_ratio": args.keep_ratio,
        "single_keep_ratio": args.single_keep_ratio,
        "multi_keep_ratio": args.multi_keep_ratio,
        "top_k": args.top_k,
        "min_neurons_per_layer": args.min_neurons_per_layer,
        "gain_lambda": args.gain_lambda,
        "single_gain_lambda": args.single_gain_lambda,
        "multi_gain_lambda": args.multi_gain_lambda,
        "evidence_power": args.evidence_power,
        "feature_mode": args.feature_mode,
        "threshold_mode": args.threshold_mode,
        "direct_quantile": args.direct_quantile,
        "tkn_score_power": args.tkn_score_power,
        "label_score_power": args.label_score_power,
        "llr_clip": args.llr_clip,
        "append_layer_pool": bool(args.append_layer_pool),
        "pool_reducers": args.pool_reducers,
        "epsilon": args.epsilon,
    }
    features_out = feature_dir(output_probe_root, args.model_alias, subset)
    neurons_out = shared_dir(output_neurons_root, args.model_alias, subset)
    single_out = single_type_dir(output_neurons_root, args.model_alias, subset)
    subset_viz = viz_dir(visualizations_root, args.model_alias)
    expected = [
        features_out / "train_features.pt",
        features_out / "train_meta.jsonl",
        features_out / "train_summary.json",
        features_out / "test_features.pt",
        features_out / "test_meta.jsonl",
        features_out / "test_summary.json",
        neurons_out / CTD_FILENAME,
        neurons_out / "summary.json",
    ]
    if args.clean:
        safe_clean(features_out)
        safe_clean(neurons_out)
        safe_clean(single_out)
    if should_skip(features_out, params, expected, overwrite=args.overwrite, clean=False):
        return read_json(neurons_out / "summary.json")

    stat_rows = compute_train_statistics(
        payload=train_payload,
        meta_rows=train_meta,
        selected_indices=train_indices,
        tkn_rows=tkn_rows,
        args=args,
        subset=subset,
        device=device,
    )
    base_feature_rows = select_feature_rows(stat_rows, args, subset)
    pool_rows = make_layer_pool_rows(base_feature_rows, args)
    feature_rows = [*base_feature_rows, *pool_rows]
    train_features, train_labels, selected_train_meta = build_features_for_split(
        payload=train_payload,
        meta_rows=train_meta,
        selected_indices=train_indices,
        feature_rows=base_feature_rows,
        device=device,
        desc=f"TKAG train features {subset}",
    )
    train_features = append_layer_pool_features(train_features, base_feature_rows, pool_rows)
    del train_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_payload, test_meta, test_manifest = load_stage_activations(activations_root, args.model_alias, subset, "test")
    test_indices = select_meta_indices(
        test_meta,
        args.max_test_samples,
        args.seed + 17,
        strategy=args.sample_strategy,
        require_per_type_labels=args.require_per_type_labels,
    )
    test_features, test_labels, selected_test_meta = build_features_for_split(
        payload=test_payload,
        meta_rows=test_meta,
        selected_indices=test_indices,
        feature_rows=base_feature_rows,
        device=device,
        desc=f"TKAG test features {subset}",
    )
    test_features = append_layer_pool_features(test_features, base_feature_rows, pool_rows)
    del test_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ensure_dir(features_out)
    ensure_dir(neurons_out)
    ensure_dir(single_out)
    torch.save(
        {
            "features": train_features,
            "labels": torch.tensor(train_labels, dtype=torch.long),
            "neuron_rows": feature_rows,
            "feature_shape": list(train_features.shape),
        },
        features_out / "train_features.pt",
    )
    torch.save(
        {
            "features": test_features,
            "labels": torch.tensor(test_labels, dtype=torch.long),
            "neuron_rows": feature_rows,
            "feature_shape": list(test_features.shape),
        },
        features_out / "test_features.pt",
    )
    write_jsonl(features_out / "train_meta.jsonl", selected_train_meta)
    write_jsonl(features_out / "test_meta.jsonl", selected_test_meta)
    write_jsonl(neurons_out / CTD_FILENAME, feature_rows)
    write_csv(neurons_out / "top_neurons.csv", feature_rows)
    layer_rows = layer_summary(feature_rows, test_payload_meta_from_rows(feature_rows), args.model_alias, subset)
    write_csv(neurons_out / "layer_summary.csv", layer_rows)
    write_single_type_rows(single_out, feature_rows, subset)
    viz_paths = write_visualizations(feature_rows, layer_rows, subset_viz, subset)

    train_summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "split": "train",
        "probe_method": "tkn_activation_gain",
        "feature_set": FEATURE_SET,
        "feature_shape": list(train_features.shape),
        "label_summary": summarize_labels(selected_train_meta),
        "shared_neuron_count": len(feature_rows),
        "source_tkn_count": len(tkn_rows),
    }
    test_summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "split": "test",
        "probe_method": "tkn_activation_gain",
        "feature_set": FEATURE_SET,
        "feature_shape": list(test_features.shape),
        "label_summary": summarize_labels(selected_test_meta),
        "shared_neuron_count": len(feature_rows),
        "source_tkn_count": len(tkn_rows),
    }
    write_json(features_out / "train_summary.json", train_summary)
    write_json(features_out / "test_summary.json", test_summary)
    write_json(features_out / "train" / "manifest.json", {"params": {**params, "split": "train"}, "summary": train_summary})
    write_json(features_out / "test" / "manifest.json", {"params": {**params, "split": "test", "test_activation_manifest_params": test_manifest.get("params", {})}, "summary": test_summary})

    score_values = [float(row["selection_score"]) for row in feature_rows]
    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_set": "TKAG_CTD",
        "feature_set": FEATURE_SET,
        "source_neuron_set": "TKN_CTD",
        "source_tkn_count": len(tkn_rows),
        "selected_neurons": len(feature_rows),
        "feature_mode": args.feature_mode,
        "gain_lambda": gain_lambda_for_subset(args, subset),
        "evidence_power": args.evidence_power,
        "llr_clip": args.llr_clip,
        "append_layer_pool": bool(args.append_layer_pool),
        "pool_reducers": parse_pool_reducers(args.pool_reducers) if args.append_layer_pool else [],
        "threshold_mode": args.threshold_mode,
        "keep_ratio": keep_ratio_for_subset(args, subset),
        "score_stats": {
            "min": min(score_values) if score_values else 0.0,
            "mean": sum(score_values) / len(score_values) if score_values else 0.0,
            "max": max(score_values) if score_values else 0.0,
        },
        "label_z_abs_mean": sum(abs(float(row["train_label_z"])) for row in feature_rows) / max(len(feature_rows), 1),
        "feature_outputs": {
            "train_features": str(features_out / "train_features.pt"),
            "test_features": str(features_out / "test_features.pt"),
        },
        "neuron_file": str(neurons_out / CTD_FILENAME),
        "visualizations": viz_paths,
    }
    write_json(neurons_out / "summary.json", summary)
    write_json(neurons_out / "manifest.json", {"params": params, "summary": summary})
    write_json(single_out / "manifest.json", {"params": params, "summary": summary})
    write_json(features_out / "manifest.json", {"params": params, "summary": {"train": train_summary, "test": test_summary}})
    print(
        f"{subset}: TKAG features dim={len(feature_rows)} from TKN={len(tkn_rows)}, "
        f"mode={args.feature_mode}, lambda={gain_lambda_for_subset(args, subset)}, "
        f"score_mean={summary['score_stats']['mean']:.4f}",
        flush=True,
    )
    return summary


def test_payload_meta_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["module_key"])
        current = by_key.get(key)
        dim = int(row.get("module_dim", 0))
        if current is None:
            by_key[key] = {
                "key": key,
                "layer": int(row["layer"]),
                "module": str(row["module"]),
                "dim": dim,
            }
        elif dim > int(current.get("dim", 0)):
            current["dim"] = dim
    return sorted(by_key.values(), key=lambda item: int(item["layer"]))


def main() -> None:
    args = parse_args()
    if args.evidence_power <= 0:
        raise ValueError("--evidence-power must be positive")
    if args.gain_lambda < 0:
        raise ValueError("--gain-lambda must be non-negative")
    activations_root = resolve_root(args.activations_dir, "activations")
    tkn_root = resolve_root(args.tkn_neurons_dir, "tkn_neurons")
    output_neurons_root = resolve_root(args.output_neurons_dir, "output_neurons")
    output_probe_root = resolve_root(args.output_probe_root, "output_probe")
    visualizations_root = resolve_root(args.visualizations_dir, "visualizations")
    device = resolve_compute_device(args.device)
    print(f"TKAG compute device: {device}", flush=True)
    print(f"TKAG subset order = {' -> '.join(subset_values(args.subset))}", flush=True)

    root_manifest = {
        "stage": "tkag_01_build_probe_features",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "probe_method": "tkn_activation_gain",
        "feature_set": FEATURE_SET,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows = []
    for subset in progress(subset_values(args.subset), desc=f"TKAG {args.model_alias}", unit="subset"):
        summary = run_subset(
            args,
            subset=subset,
            activations_root=activations_root,
            tkn_root=tkn_root,
            output_neurons_root=output_neurons_root,
            output_probe_root=output_probe_root,
            visualizations_root=visualizations_root,
            device=device,
        )
        root_manifest["subsets"][subset] = summary
        summary_rows.append(
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "method": METHOD_NAME,
                "feature_set": FEATURE_SET,
                "selected_neurons": summary["selected_neurons"],
                "source_tkn_count": summary["source_tkn_count"],
                "feature_mode": summary["feature_mode"],
                "gain_lambda": summary["gain_lambda"],
                "test_feature_path": summary["feature_outputs"]["test_features"],
            }
        )

    model_neuron_root = output_neurons_root / args.model_alias / "shared_by_subset"
    ensure_dir(model_neuron_root)
    write_csv(model_neuron_root / "shared_summary.csv", summary_rows)
    write_json(model_neuron_root / "manifest.json", root_manifest)
    model_feature_root = pp_features_root(output_probe_root) / args.model_alias
    ensure_dir(model_feature_root)
    write_json(model_feature_root / "manifest.json", root_manifest)
    print(f"Wrote TKAG manifest: {model_neuron_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
