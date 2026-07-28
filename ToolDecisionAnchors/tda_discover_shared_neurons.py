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
METHOD_NAME = "ToolDecisionAnchors"
TDN_FILENAME = "TDA_TDN_neurons.jsonl"
CTD_FILENAME = "TDA_CTD_neurons.jsonl"
MODULE_ORDER = ["gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ToolDecisionAnchors TDA-5: discover direction-consistent A/B/C shared "
            "tool-decision neurons from train-split FFN output activations."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="TDA activation root; defaults to ../cross_task_tool_neurons_data/tool_decision_anchors/activations.",
    )
    parser.add_argument(
        "--neurons-dir",
        default=None,
        help="TDA neuron output root; defaults to ../cross_task_tool_neurons_data/tool_decision_anchors/neurons.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help="TDA visualization root; defaults to ../cross_task_tool_neurons_data/tool_decision_anchors/visualizations.",
    )
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument(
        "--modules",
        default="gate_proj,up_proj",
        help="Comma-separated FFN output modules to score. Use gate_proj,up_proj,down_proj for all modules.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.70,
        help="Per selected module top ratio by signed ABC consensus score.",
    )
    parser.add_argument("--min-neurons-per-module", type=int, default=1)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument(
        "--device",
        default="auto",
        help="Tensor statistics device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_root(kind: str) -> Path:
    mapping = {
        "activations": data_root() / "tool_decision_anchors" / "activations",
        "neurons": data_root() / "tool_decision_anchors" / "neurons",
        "visualizations": data_root() / "tool_decision_anchors" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown ToolDecisionAnchors root kind: {kind}")
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
    if not 0 < ratio <= 1:
        raise ValueError("--top-ratio must be in (0, 1]")
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


def task_indices(
    meta_rows: list[dict[str, Any]],
    task_type: str,
    *,
    min_class_count: int,
) -> tuple[list[int], list[int]]:
    call = [idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 1]
    direct = [idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 0]
    if len(call) < min_class_count or len(direct) < min_class_count:
        raise ValueError(
            f"{task_type}: need at least {min_class_count} examples per class, got "
            f"tool_necessary=1: {len(call)}, tool_necessary=0: {len(direct)}"
        )
    return call, direct


def zscore(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    center = tensor.mean()
    scale = tensor.std(unbiased=False).clamp_min(eps)
    out = (tensor - center) / scale
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_one_module(
    *,
    key: str,
    activations: dict[str, torch.Tensor],
    call_indices: list[int],
    direct_indices: list[int],
    device: torch.device,
    eps: float,
) -> dict[str, torch.Tensor | float]:
    call_idx = torch.tensor(call_indices, dtype=torch.long)
    direct_idx = torch.tensor(direct_indices, dtype=torch.long)
    if device.type == "cuda":
        with torch.cuda.device(device):
            call_x = activations[key].index_select(0, call_idx).to(device, non_blocking=True).float()
            direct_x = activations[key].index_select(0, direct_idx).to(device, non_blocking=True).float()
            mu_call = call_x.mean(dim=0)
            mu_direct = direct_x.mean(dim=0)
            var_call = call_x.var(dim=0, unbiased=False)
            var_direct = direct_x.var(dim=0, unbiased=False)
            signed_effect = (mu_call - mu_direct) / torch.sqrt(var_call + var_direct + eps)
            signed_effect = torch.nan_to_num(signed_effect, nan=0.0, posinf=0.0, neginf=0.0)
            z_effect = zscore(signed_effect, eps)
            out: dict[str, torch.Tensor | float] = {
                "signed_effect": signed_effect.detach().cpu(),
                "z_effect": z_effect.detach().cpu(),
                "mean_signed_effect": float(signed_effect.mean().detach().cpu().item()),
                "std_signed_effect": float(signed_effect.std(unbiased=False).detach().cpu().item()),
            }
            del call_x, direct_x, mu_call, mu_direct, var_call, var_direct, signed_effect, z_effect
            torch.cuda.empty_cache()
            return out

    call_x = activations[key].index_select(0, call_idx).to(device).float()
    direct_x = activations[key].index_select(0, direct_idx).to(device).float()
    mu_call = call_x.mean(dim=0)
    mu_direct = direct_x.mean(dim=0)
    var_call = call_x.var(dim=0, unbiased=False)
    var_direct = direct_x.var(dim=0, unbiased=False)
    signed_effect = (mu_call - mu_direct) / torch.sqrt(var_call + var_direct + eps)
    signed_effect = torch.nan_to_num(signed_effect, nan=0.0, posinf=0.0, neginf=0.0)
    z_effect = zscore(signed_effect, eps)
    return {
        "signed_effect": signed_effect.detach().cpu(),
        "z_effect": z_effect.detach().cpu(),
        "mean_signed_effect": float(signed_effect.mean().item()),
        "std_signed_effect": float(signed_effect.std(unbiased=False).item()),
    }


def compute_task_scores(
    *,
    activations: dict[str, torch.Tensor],
    module_meta: list[dict[str, Any]],
    meta_rows: list[dict[str, Any]],
    task_type: str,
    min_class_count: int,
    device: torch.device,
    eps: float,
    desc: str,
) -> tuple[dict[str, dict[str, torch.Tensor | float]], dict[str, Any]]:
    call_indices, direct_indices = task_indices(meta_rows, task_type, min_class_count=min_class_count)
    scores: dict[str, dict[str, torch.Tensor | float]] = {}
    for meta in progress(module_meta, desc=desc, unit="module"):
        key = str(meta["key"])
        scores[key] = compute_one_module(
            key=key,
            activations=activations,
            call_indices=call_indices,
            direct_indices=direct_indices,
            device=device,
            eps=eps,
        )
    return scores, {"task_type": task_type, "n_tool": len(call_indices), "n_direct": len(direct_indices)}


def consensus_for_module(
    score_by_task: dict[str, dict[str, torch.Tensor | float]],
    key: str,
) -> dict[str, torch.Tensor]:
    z_a = score_by_task["A"][key]["z_effect"]
    z_b = score_by_task["B"][key]["z_effect"]
    z_c = score_by_task["C"][key]["z_effect"]
    if not isinstance(z_a, torch.Tensor) or not isinstance(z_b, torch.Tensor) or not isinstance(z_c, torch.Tensor):
        raise TypeError(f"Missing tensor scores for {key}")
    positive = torch.minimum(torch.minimum(z_a, z_b), z_c)
    negative = torch.minimum(torch.minimum(-z_a, -z_b), -z_c)
    shared = torch.maximum(positive, negative)
    direction_sign = torch.where(positive >= negative, torch.ones_like(shared), -torch.ones_like(shared))
    mean_abs = torch.stack([z_a.abs(), z_b.abs(), z_c.abs()], dim=0).mean(dim=0)
    return {
        "score": shared,
        "direction_sign": direction_sign,
        "mean_abs_z": mean_abs,
        "z_A": z_a,
        "z_B": z_b,
        "z_C": z_c,
        "positive_consensus": positive,
        "negative_consensus": negative,
    }


def sort_selected_indices(consensus: dict[str, torch.Tensor], k: int) -> list[int]:
    scores = consensus["score"]
    mean_abs = consensus["mean_abs_z"]
    if scores.numel() == 0:
        return []
    top_k = min(k, int(scores.numel()))
    rough = torch.topk(scores, k=top_k, largest=True, sorted=False).indices.tolist()
    rough.sort(key=lambda idx: (-float(scores[idx]), -float(mean_abs[idx]), int(idx)))
    return [int(idx) for idx in rough]


def selected_rows(
    *,
    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float]]],
    module_meta: list[dict[str, Any]],
    model_alias: str,
    subset: str,
    top_ratio: float,
    min_neurons_per_module: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    consensus_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for meta in module_meta:
        key = str(meta["key"])
        dim = int(meta["dim"])
        k = top_count(dim, top_ratio, min_neurons_per_module)
        consensus = consensus_for_module(score_by_task, key)
        consensus_by_key[key] = consensus
        top_indices = sort_selected_indices(consensus, k)
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
                "top_ratio": top_ratio,
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
                "top_ratio": top_ratio,
                "module_dim": dim,
                "selected_neurons_in_module": len(top_indices),
                "score": float(consensus["score"][idx]),
                "signed_consensus_score": float(consensus["score"][idx]),
                "direction_sign": direction_sign,
                "direction": "tool_high" if direction_sign > 0 else "direct_high",
                "positive_consensus": float(consensus["positive_consensus"][idx]),
                "negative_consensus": float(consensus["negative_consensus"][idx]),
                "mean_abs_z": float(consensus["mean_abs_z"][idx]),
                "score_A": float(consensus["z_A"][idx]),
                "score_B": float(consensus["z_B"][idx]),
                "score_C": float(consensus["z_C"][idx]),
            }
            for task_type in TASK_TYPES:
                signed = score_by_task[task_type][key]["signed_effect"]
                if not isinstance(signed, torch.Tensor):
                    raise TypeError(f"Missing signed_effect tensor for {task_type}/{key}")
                row[f"signed_effect_{task_type}"] = float(signed[idx])
            rows.append(row)
    rows.sort(key=lambda item: (-float(item["score"]), -float(item["mean_abs_z"]), int(item["layer"]), str(item["module"]), int(item["index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
    return rows, layer_rows, consensus_by_key


def rows_by_task_direction(rows: list[dict[str, Any]], task_type: str) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["task_type"] = task_type
        item["task_score"] = row.get(f"score_{task_type}")
        item["task_signed_effect"] = row.get(f"signed_effect_{task_type}")
        copied.append(item)
    copied.sort(key=lambda item: (-abs(float(item.get("task_score") or 0.0)), int(item["layer"]), str(item["module"]), int(item["index"])))
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
        write_empty_plot(out_path, "TDA-CTD Shared Neurons")
        return
    dims = {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}
    counts = Counter((int(row["layer"]), str(row["module"])) for row in rows)
    matrix = []
    for layer in layers:
        matrix.append([counts.get((layer, module), 0) / max(dims.get((layer, module), 1), 1) for module in modules])
    fig, ax = plt.subplots(figsize=(4.8, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("TDA-CTD Shared Neurons")
    ax.set_xticks(range(len(modules)), modules, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN module")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    ax.set_xlabel("TDA-CTD rank")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{module}" for layer, module in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="signed ABC consensus score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_score(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    if not rows:
        write_empty_plot(out_path, title)
        return
    layers = sorted({int(row["layer"]) for row in rows})
    modules = [module for module in MODULE_ORDER if any(str(row["module"]) == module for row in rows)]
    by_group: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        by_group.setdefault((int(row["layer"]), str(row["module"])), []).append(float(row["score"]))
    matrix = []
    for layer in layers:
        matrix.append([sum(by_group.get((layer, module), [0.0])) / max(len(by_group.get((layer, module), [])), 1) for module in modules])
    fig, ax = plt.subplots(figsize=(4.8, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(modules)), modules, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN module")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean selected score")
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
) -> dict[str, Any]:
    return {
        "stage": "tda_05_signed_consensus_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activation_dir": str(activation_dir),
        "activation_manifest": activation_manifest,
        "neurons_root": str(neurons_root),
        "visualizations_root": str(viz_root),
        "modules": modules,
        "top_ratio": args.top_ratio,
        "min_neurons_per_module": args.min_neurons_per_module,
        "min_class_count": args.min_class_count,
        "epsilon": args.epsilon,
        "score_definition": (
            "For each task type c in A/B/C, z_c = zscore((mean(tool=1)-mean(tool=0))/"
            "sqrt(var(tool=1)+var(tool=0)+epsilon)) within each FFN module. "
            "shared_score=max(min(z_A,z_B,z_C), min(-z_A,-z_B,-z_C))."
        ),
        "neuron_identity": "(layer, module, index) over FFN output module elements",
    }


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"tda_ctd_density_heatmap_{subset}.png",
        viz_dir / f"tda_ctd_score_heatmap_{subset}.png",
        viz_dir / f"tda_ctd_layer_score_heatmap_{subset}.png",
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
        print(f"Skip existing ToolDecisionAnchors shared neurons: {out_dir}", flush=True)
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
        raise FileNotFoundError(f"Missing ToolDecisionAnchors train activations for {subset}: {act_dir}")

    activation_manifest = read_json(manifest_path)
    shared_dir = shared_root(neurons_root, args.model_alias)
    single_root = single_type_root(neurons_root, args.model_alias)
    out_dir = shared_dir / subset
    viz_dir = viz_root / args.model_alias / "shared_by_subset"
    params = expected_params(
        args,
        subset=subset,
        activation_dir=act_dir,
        activation_manifest=activation_manifest,
        modules=modules,
        neurons_root=neurons_root,
        viz_root=viz_root,
    )
    if should_skip(out_dir, single_root, viz_dir, subset, params, overwrite=args.overwrite, clean=args.clean):
        return read_json(out_dir / "summary.json")

    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations: dict[str, torch.Tensor] = payload["activations"]
    full_module_meta: list[dict[str, Any]] = payload["module_meta"]
    module_meta = selected_module_meta(full_module_meta, modules)
    meta_rows = read_jsonl(meta_path)

    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float]]] = {}
    class_summaries: dict[str, Any] = {}
    for task_type in TASK_TYPES:
        task_scores, class_summary = compute_task_scores(
            activations=activations,
            module_meta=module_meta,
            meta_rows=meta_rows,
            task_type=task_type,
            min_class_count=args.min_class_count,
            device=device,
            eps=args.epsilon,
            desc=f"{subset}/type {task_type} signed effect",
        )
        score_by_task[task_type] = task_scores
        class_summaries[task_type] = class_summary

    rows, layer_rows, consensus_by_key = selected_rows(
        score_by_task=score_by_task,
        module_meta=module_meta,
        model_alias=args.model_alias,
        subset=subset,
        top_ratio=args.top_ratio,
        min_neurons_per_module=args.min_neurons_per_module,
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
        write_json(
            type_dir / "summary.json",
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "task_type": task_type,
                "method": METHOD_NAME,
                "neuron_set": "TDA_TDN",
                "selected_neurons": len(type_rows),
                "class_summary": class_summaries[task_type],
            },
        )

    density_path = viz_dir / f"tda_ctd_density_heatmap_{subset}.png"
    score_path = viz_dir / f"tda_ctd_score_heatmap_{subset}.png"
    layer_score_path = viz_dir / f"tda_ctd_layer_score_heatmap_{subset}.png"
    plot_density(rows, module_meta, density_path)
    plot_score_heatmap(rows, score_path, args.heatmap_top_n, f"{subset} TDA-CTD signed ABC consensus")
    plot_layer_score(rows, layer_score_path, f"{subset} TDA-CTD mean selected score")

    torch.save(
        {
            "method": METHOD_NAME,
            "subset": subset,
            "modules": modules,
            "top_ratio": args.top_ratio,
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
        out_dir / "tda_scores.pt",
    )

    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_file": str(out_dir / CTD_FILENAME),
        "neuron_set": "TDA_CTD",
        "selected_neurons": len(rows),
        "modules": modules,
        "top_ratio": args.top_ratio,
        "class_summaries": class_summaries,
        "score_stats": {
            "min": min((float(row["score"]) for row in rows), default=0.0),
            "mean": sum(float(row["score"]) for row in rows) / max(len(rows), 1),
            "max": max((float(row["score"]) for row in rows), default=0.0),
        },
        "top_layers": Counter(int(row["layer"]) for row in rows).most_common(10),
        "top_modules": Counter(str(row["module"]) for row in rows).most_common(),
        "visualizations": {
            "density_heatmap": str(density_path),
            "score_heatmap": str(score_path),
            "layer_score_heatmap": str(layer_score_path),
        },
    }
    if not rows:
        summary["warning"] = "TDA_CTD is empty. Increase --top-ratio or check train label coverage."
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    write_json(single_root / subset / "manifest.json", {"params": params, "summary": summary})
    print(
        f"{subset}: TDA_CTD={len(rows)}, modules={','.join(modules)}, "
        f"score_mean={summary['score_stats']['mean']:.4f}, score_max={summary['score_stats']['max']:.4f}",
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
    print(f"ToolDecisionAnchors compute device: {device}", flush=True)
    print(f"ToolDecisionAnchors modules: {','.join(modules)}", flush=True)
    print(f"ToolDecisionAnchors subset order = {' -> '.join(subset_values(args.subset))}", flush=True)

    root_manifest: dict[str, Any] = {
        "stage": "tda_05_signed_consensus_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for subset in progress(subset_values(args.subset), desc=f"TDA-5 {args.model_alias}", unit="subset"):
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
                "neuron_set": "TDA_CTD",
                "selected_neurons": summary["selected_neurons"],
                "score_mean": summary["score_stats"]["mean"],
                "score_max": summary["score_stats"]["max"],
            }
        )

    model_root = shared_root(neurons_root, args.model_alias)
    ensure_dir(model_root)
    write_csv_rows(
        model_root / "shared_summary.csv",
        summary_rows,
        fieldnames=["model_alias", "subset", "method", "neuron_set", "selected_neurons", "score_mean", "score_max"],
    )
    write_json(model_root / "manifest.json", root_manifest)
    print(f"Wrote ToolDecisionAnchors shared manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
