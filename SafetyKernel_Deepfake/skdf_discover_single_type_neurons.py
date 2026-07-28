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


STAGE_VERSION = 2
METHOD_NAME = "SafetyKernel_Deepfake"
TDN_FILENAME = "SKD_TDN_neurons.jsonl"
MODULE_ORDER = ["gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SafetyKernel_Deepfake SKD-5: discover A/B/C tool-decision neurons with "
            "Safety Kernel FFN output coordinates and Deepfake paired-shift scoring."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="SKD-4 activation root; defaults to ../cross_task_tool_neurons_data/safety_kernel_deepfake/activations.",
    )
    parser.add_argument(
        "--neurons-dir",
        default=None,
        help="SKD neuron output root; defaults to ../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help="SKD visualization root; defaults to ../cross_task_tool_neurons_data/safety_kernel_deepfake/visualizations.",
    )
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.10,
        help="Deepfake-style per-module selected ratio. k_m=ceil(top_ratio * dim_m). Default matches deepfake-code.",
    )
    parser.add_argument("--min-neurons-per-module", type=int, default=1)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument(
        "--floor-ratio",
        type=float,
        default=0.05,
        help="Deepfake paired_shift_score floor = max(epsilon, median(valid std_delta) * floor_ratio).",
    )
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all deterministic label pairs.")
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
        "activations": data_root() / "safety_kernel_deepfake" / "activations",
        "neurons": data_root() / "safety_kernel_deepfake" / "neurons",
        "visualizations": data_root() / "safety_kernel_deepfake" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown SafetyKernel_Deepfake root kind: {kind}")
    return mapping[kind]


def resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


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
        raise ValueError("--min-neurons-per-module must be >= 1")
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


def paired_shift_score(mean_delta: torch.Tensor, std_delta: torch.Tensor, eps: float, floor_ratio: float) -> tuple[torch.Tensor, float]:
    valid = std_delta[torch.isfinite(std_delta) & (std_delta > eps)]
    floor = max(eps, float(torch.median(valid).item()) * floor_ratio) if valid.numel() else 1.0
    score = mean_delta.abs() / torch.sqrt(std_delta.square() + floor * floor)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score, floor


def compute_one_module(
    *,
    key: str,
    activations: dict[str, torch.Tensor],
    call_idx: torch.Tensor,
    direct_idx: torch.Tensor,
    device: torch.device,
    eps: float,
    floor_ratio: float,
) -> tuple[str, dict[str, torch.Tensor | float]]:
    if device.type == "cuda":
        with torch.cuda.device(device):
            call_x = activations[key].index_select(0, call_idx).to(device, non_blocking=True).float()
            direct_x = activations[key].index_select(0, direct_idx).to(device, non_blocking=True).float()
            delta = call_x - direct_x
            mean_delta = delta.mean(dim=0)
            std_delta = delta.std(dim=0, unbiased=True)
            score, floor = paired_shift_score(mean_delta, std_delta, eps, floor_ratio)
            out: dict[str, torch.Tensor | float] = {
                "paired_shift_score": score.detach().cpu(),
                "mean_delta": mean_delta.detach().cpu(),
                "std_delta": std_delta.detach().cpu(),
                "std_floor": float(floor),
            }
            del call_x, direct_x, delta, mean_delta, std_delta, score
            torch.cuda.empty_cache()
            return key, out

    call_x = activations[key].index_select(0, call_idx).to(device).float()
    direct_x = activations[key].index_select(0, direct_idx).to(device).float()
    delta = call_x - direct_x
    mean_delta = delta.mean(dim=0)
    std_delta = delta.std(dim=0, unbiased=True)
    score, floor = paired_shift_score(mean_delta, std_delta, eps, floor_ratio)
    return (
        key,
        {
            "paired_shift_score": score.detach().cpu(),
            "mean_delta": mean_delta.detach().cpu(),
            "std_delta": std_delta.detach().cpu(),
            "std_floor": float(floor),
        },
    )


def compute_scores_for_modules(
    *,
    activations: dict[str, torch.Tensor],
    module_meta: list[dict[str, Any]],
    call_indices: list[int],
    direct_indices: list[int],
    device: torch.device,
    eps: float,
    floor_ratio: float,
    desc: str,
) -> dict[str, dict[str, torch.Tensor | float]]:
    call_idx = torch.tensor(call_indices, dtype=torch.long)
    direct_idx = torch.tensor(direct_indices, dtype=torch.long)
    score_pack: dict[str, dict[str, torch.Tensor | float]] = {}
    for meta in progress(module_meta, desc=desc, unit="module"):
        key, pack = compute_one_module(
            key=str(meta["key"]),
            activations=activations,
            call_idx=call_idx,
            direct_idx=direct_idx,
            device=device,
            eps=eps,
            floor_ratio=floor_ratio,
        )
        score_pack[key] = pack
    return score_pack


def selected_rows_for_task_type(
    *,
    score_pack: dict[str, dict[str, torch.Tensor | float]],
    module_meta: list[dict[str, Any]],
    task_type: str,
    top_ratio: float,
    min_neurons_per_module: int,
    pair_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta_by_key = {str(meta["key"]): meta for meta in module_meta}
    for key, pack in score_pack.items():
        score = pack["paired_shift_score"]
        mean_delta = pack["mean_delta"]
        std_delta = pack["std_delta"]
        if not isinstance(score, torch.Tensor) or not isinstance(mean_delta, torch.Tensor) or not isinstance(std_delta, torch.Tensor):
            raise TypeError(f"Invalid score pack for module {key}")
        meta = meta_by_key[key]
        count = top_count(score.numel(), top_ratio, min_neurons_per_module)
        values, indices = torch.topk(score, count)
        for rank_in_module, (value, raw_index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            index = int(raw_index)
            rows.append(
                {
                    "layer": int(meta["layer"]),
                    "module": str(meta["module"]),
                    "module_key": key,
                    "index": index,
                    "score": float(value),
                    "paired_shift_score": float(value),
                    "mean_delta": float(mean_delta[index]),
                    "std_delta": float(std_delta[index]),
                    "std_floor": float(pack["std_floor"]),
                    "n_pairs": int(pair_count),
                    "rank_in_module": rank_in_module,
                    "top_ratio": float(top_ratio),
                    "selection_scope": f"task_type_{task_type}",
                    "score_source": "Deepfake paired_shift_score",
                }
            )
    rows.sort(key=lambda row: (-float(row["score"]), int(row["layer"]), str(row["module"]), int(row["index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def layer_summary_rows(
    *,
    score_pack: dict[str, dict[str, torch.Tensor | float]],
    module_meta: list[dict[str, Any]],
    top_ratio: float,
    min_neurons_per_module: int,
    pair_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for meta in module_meta:
        key = str(meta["key"])
        score = score_pack[key]["paired_shift_score"]
        if not isinstance(score, torch.Tensor):
            raise TypeError(f"Invalid paired_shift_score for {key}")
        count = top_count(score.numel(), top_ratio, min_neurons_per_module)
        top_values = torch.topk(score, count).values
        rows.append(
            {
                "layer": int(meta["layer"]),
                "module": str(meta["module"]),
                "module_key": key,
                "n_pairs": int(pair_count),
                "channels": int(score.numel()),
                "top_count": int(count),
                "d_shift_rms": float(torch.sqrt(torch.mean(score.square()))),
                "mean_channel_score": float(score.mean()),
                "best_channel_score": float(top_values[0]),
                "std_floor": float(score_pack[key]["std_floor"]),
            }
        )
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    names = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        if not names:
            return
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


def module_dims(module_meta: list[dict[str, Any]]) -> dict[tuple[int, str], int]:
    return {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}


def plot_density(rows_by_type: dict[str, list[dict[str, Any]]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    dims = module_dims(module_meta)
    layers = sorted({layer for layer, _module in dims})
    if not layers:
        write_empty_plot(out_path, "SafetyKernel_Deepfake TDN density")
        return
    fig, axes = plt.subplots(1, len(TASK_TYPES), figsize=(10, max(5, len(layers) * 0.22)), sharey=True)
    if len(TASK_TYPES) == 1:
        axes = [axes]
    for ax, task_type in zip(axes, TASK_TYPES):
        counts = Counter((int(row["layer"]), str(row["module"])) for row in rows_by_type.get(task_type, []))
        matrix = []
        for layer in layers:
            matrix.append([counts.get((layer, module), 0) / max(dims.get((layer, module), 1), 1) for module in MODULE_ORDER])
        im = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_title(f"Type {task_type}")
        ax.set_xticks(range(len(MODULE_ORDER)), MODULE_ORDER, rotation=30, ha="right")
        ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
        ax.set_xlabel("FFN module")
        if ax is axes[0]:
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
    matrix = torch.full((len(group_order), len(selected)), float("nan"), dtype=torch.float32)
    for x, row in enumerate(selected):
        matrix[y_by_group[(int(row["layer"]), str(row["module"]))], x] = float(row["score"])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap, vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel("Global SKD-TDN rank")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{module}" for layer, module in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="paired shift score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_score_heatmap(
    *,
    score_pack: dict[str, dict[str, torch.Tensor | float]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    top_ratio: float,
    min_neurons_per_module: int,
    title: str,
) -> None:
    module_order = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    ordered_meta = sorted(module_meta, key=lambda meta: (int(meta["layer"]), module_order.get(str(meta["module"]), 99), str(meta["key"])))
    row_values: list[torch.Tensor] = []
    row_labels: list[str] = []
    max_cols = 0
    for meta in ordered_meta:
        key = str(meta["key"])
        score = score_pack.get(key, {}).get("paired_shift_score")
        if not isinstance(score, torch.Tensor):
            continue
        count = top_count(score.numel(), top_ratio, min_neurons_per_module)
        values = torch.topk(score.detach().float().cpu(), count).values
        row_values.append(values)
        row_labels.append(f"L{int(meta['layer'])}.{meta['module']}")
        max_cols = max(max_cols, count)
    if not row_values or max_cols <= 0:
        write_empty_plot(out_path, title)
        return
    matrix = torch.full((len(row_values), max_cols), float("nan"), dtype=torch.float32)
    for row_idx, values in enumerate(row_values):
        matrix[row_idx, : values.numel()] = values
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig_width = max(10, min(42, max_cols * 0.018))
    fig_height = max(6, len(row_labels) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap, vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel(f"Selected neuron rank within top {top_ratio:g} of each layer/module")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, max_cols, max(1, max_cols // 10)))
    if ticks and ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="paired shift score")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_shift(layer_rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    if not layer_rows:
        write_empty_plot(out_path, title)
        return
    labels = [f"L{row['layer']}.{row['module']}" for row in layer_rows]
    x = list(range(1, len(labels) + 1))
    rms = [float(row["d_shift_rms"]) for row in layer_rows]
    best = [float(row["best_channel_score"]) for row in layer_rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.34 * len(labels)), 4.8), dpi=180)
    ax.plot(x, rms, marker="o", linewidth=2, label="D_shift RMS")
    ax.plot(x, best, marker="s", linewidth=1.7, label="best neuron score")
    ax.set_xticks(x, labels, rotation=50, ha="right")
    ax.set_ylabel("paired standardized activation shift")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_selected_counts(layer_rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    if not layer_rows:
        write_empty_plot(out_path, title)
        return
    labels = [f"L{row['layer']}.{row['module']}" for row in layer_rows]
    counts = [int(row["top_count"]) for row in layer_rows]
    x = list(range(1, len(labels) + 1))
    fig, ax = plt.subplots(figsize=(max(8, 0.34 * len(labels)), 4.8), dpi=180)
    ax.bar(x, counts, color="#1769aa")
    ax.set_xticks(x, labels, rotation=50, ha="right")
    ax.set_ylabel("selected neurons")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def class_balance_rows(meta_rows: list[dict[str, Any]], model_alias: str, subset: str) -> list[dict[str, Any]]:
    counts = Counter((str(row.get("task_type", "unknown")), int(row["tool_necessary"])) for row in meta_rows)
    rows = []
    for task_type in [*TASK_TYPES, "overall"]:
        for label in [0, 1]:
            if task_type == "overall":
                count = sum(1 for row in meta_rows if int(row["tool_necessary"]) == label)
            else:
                count = counts.get((task_type, label), 0)
            rows.append(
                {
                    "model_alias": model_alias,
                    "subset": subset,
                    "task_type": task_type,
                    "tool_necessary": label,
                    "count": int(count),
                }
            )
    return rows


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    neurons_root: Path,
    viz_root: Path,
) -> dict[str, Any]:
    return {
        "stage": "skdf_05_single_type_paired_shift_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activations_dir": str(activation_dir),
        "neurons_dir": str(neurons_root),
        "visualizations_dir": str(viz_root),
        "top_ratio": args.top_ratio,
        "min_neurons_per_module": args.min_neurons_per_module,
        "heatmap_top_n": args.heatmap_top_n,
        "epsilon": args.epsilon,
        "floor_ratio": args.floor_ratio,
        "min_pairs": args.min_pairs,
        "max_pairs": args.max_pairs,
        "activation_manifest_params": activation_manifest.get("params", {}),
        "activation_definition": "last_input_token_ffn_module_output",
        "score_definition": "paired_shift_score=abs(mean(tool_necessary_1_minus_0_delta))/sqrt(std(delta)^2+floor^2)",
        "selection": "A/B/C separately; deterministic one-to-one label pairs inside each task type; per-module TopRatio",
        "neuron_identity": "(layer, module, index) over FFN output modules gate_proj/up_proj/down_proj",
    }


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    paths = [viz_dir / f"skd_tdn_density_heatmap_{subset}.png"]
    for task_type in TASK_TYPES:
        paths.extend(
            [
                viz_dir / f"skd_paired_shift_heatmap_{subset}_{task_type}.png",
                viz_dir / f"skd_layer_topratio_shift_heatmap_{subset}_{task_type}.png",
                viz_dir / f"skd_layer_shift_{subset}_{task_type}.png",
                viz_dir / f"skd_selected_counts_{subset}_{task_type}.png",
            ]
        )
    return paths


def expected_outputs(out_root: Path, viz_dir: Path, subset: str) -> list[Path]:
    paths = [
        out_root / subset / "module_meta.json",
        out_root / subset / "class_balance.csv",
        out_root / subset / "summary.json",
        out_root / subset / "manifest.json",
        *expected_visualizations(viz_dir, subset),
    ]
    for task_type in TASK_TYPES:
        out_dir = out_root / subset / task_type
        paths.extend(
            [
                out_dir / TDN_FILENAME,
                out_dir / "deepfake_scores.pt",
                out_dir / "layer_summary.csv",
                out_dir / "top_neurons.csv",
                out_dir / "pair_meta.jsonl",
                out_dir / "layer_counts.csv",
                out_dir / "module_counts.csv",
                out_dir / "summary.json",
            ]
        )
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


def clean_outputs(out_root: Path, viz_dir: Path, subset: str) -> None:
    clean_directory(out_root / subset, data_root())
    remove_existing_files(expected_visualizations(viz_dir, subset))


def should_skip(
    out_root: Path,
    viz_dir: Path,
    subset: str,
    params: dict[str, Any],
    *,
    overwrite: bool,
    clean: bool,
) -> bool:
    if clean:
        clean_outputs(out_root, viz_dir, subset)
        return False
    if overwrite:
        return False
    expected = expected_outputs(out_root, viz_dir, subset)
    if not all(path.exists() for path in expected):
        return False
    manifest = read_json(out_root / subset / "manifest.json")
    if manifest.get("params") == params:
        print(f"Skip existing SafetyKernel_Deepfake single-type neurons: {out_root / subset}", flush=True)
        return True
    return False


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
        raise FileNotFoundError(f"Missing SafetyKernel_Deepfake train activations for {subset}: {act_dir}")

    activation_manifest = read_json(manifest_path)
    out_root = single_type_root(neurons_root, args.model_alias)
    viz_dir = viz_root / args.model_alias / "single_type_by_subset"
    params = expected_params(
        args,
        subset=subset,
        activation_dir=act_dir,
        activation_manifest=activation_manifest,
        neurons_root=neurons_root,
        viz_root=viz_root,
    )
    if should_skip(out_root, viz_dir, subset, params, overwrite=args.overwrite, clean=args.clean):
        return read_json(out_root / subset / "summary.json")

    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations: dict[str, torch.Tensor] = payload["activations"]
    module_meta: list[dict[str, Any]] = payload["module_meta"]
    meta_rows = read_jsonl(meta_path)
    rows_by_type: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "task_types": {},
        "visualizations": {},
    }

    ensure_dir(out_root / subset)
    write_json(out_root / subset / "module_meta.json", module_meta)
    class_rows = class_balance_rows(meta_rows, args.model_alias, subset)
    write_csv_rows(
        out_root / subset / "class_balance.csv",
        class_rows,
        fieldnames=["model_alias", "subset", "task_type", "tool_necessary", "count"],
    )

    for task_type in TASK_TYPES:
        call_indices, direct_indices, pair_rows = make_label_pairs(
            meta_rows,
            task_type,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
        )
        score_pack = compute_scores_for_modules(
            activations=activations,
            module_meta=module_meta,
            call_indices=call_indices,
            direct_indices=direct_indices,
            device=device,
            eps=args.epsilon,
            floor_ratio=args.floor_ratio,
            desc=f"{subset}/type {task_type} paired shift",
        )
        rows = selected_rows_for_task_type(
            score_pack=score_pack,
            module_meta=module_meta,
            task_type=task_type,
            top_ratio=args.top_ratio,
            min_neurons_per_module=args.min_neurons_per_module,
            pair_count=len(pair_rows),
        )
        layer_rows = layer_summary_rows(
            score_pack=score_pack,
            module_meta=module_meta,
            top_ratio=args.top_ratio,
            min_neurons_per_module=args.min_neurons_per_module,
            pair_count=len(pair_rows),
        )
        rows_by_type[task_type] = rows
        out_dir = out_root / subset / task_type
        ensure_dir(out_dir)
        write_jsonl(out_dir / TDN_FILENAME, rows)
        write_jsonl(out_dir / "pair_meta.jsonl", pair_rows)
        write_csv_rows(out_dir / "top_neurons.csv", rows)
        write_csv_rows(out_dir / "layer_summary.csv", layer_rows)
        write_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
        write_counts_csv(rows, out_dir / "module_counts.csv", "module")
        torch.save(
            {
                "method": METHOD_NAME,
                "task_type": task_type,
                "subset": subset,
                "top_ratio": args.top_ratio,
                "min_neurons_per_module": args.min_neurons_per_module,
                "n_pairs": len(pair_rows),
                "module_meta": module_meta,
                "scores": {key: {name: value.to(torch.float32) if isinstance(value, torch.Tensor) else value for name, value in pack.items()} for key, pack in score_pack.items()},
                "score_definition": "Deepfake paired_shift_score over deterministic tool_necessary=1 minus 0 pairs",
            },
            out_dir / "deepfake_scores.pt",
        )
        heatmap_path = viz_dir / f"skd_paired_shift_heatmap_{subset}_{task_type}.png"
        layer_top_path = viz_dir / f"skd_layer_topratio_shift_heatmap_{subset}_{task_type}.png"
        layer_shift_path = viz_dir / f"skd_layer_shift_{subset}_{task_type}.png"
        selected_counts_path = viz_dir / f"skd_selected_counts_{subset}_{task_type}.png"
        plot_score_heatmap(rows, heatmap_path, args.heatmap_top_n, f"{subset} Type {task_type}: SKD paired shift")
        plot_layer_top_score_heatmap(
            score_pack=score_pack,
            module_meta=module_meta,
            out_path=layer_top_path,
            top_ratio=args.top_ratio,
            min_neurons_per_module=args.min_neurons_per_module,
            title=f"{subset} Type {task_type}: top-ratio paired shift by layer/module",
        )
        plot_layer_shift(layer_rows, layer_shift_path, f"{subset} Type {task_type}: layer and top-neuron shifts")
        plot_selected_counts(layer_rows, selected_counts_path, f"{subset} Type {task_type}: selected neurons per layer/module")
        type_summary = {
            "task_type": task_type,
            "n_pairs": len(pair_rows),
            "n_tool_necessary": len(call_indices),
            "n_tool_unnecessary": len(direct_indices),
            "selected_neurons": len(rows),
            "top_ratio": args.top_ratio,
            "top_layers": Counter(int(row["layer"]) for row in rows).most_common(10),
            "top_modules": Counter(str(row["module"]) for row in rows).most_common(),
            "visualizations": {
                "paired_shift_heatmap": str(heatmap_path),
                "layer_topratio_shift_heatmap": str(layer_top_path),
                "layer_shift": str(layer_shift_path),
                "selected_counts": str(selected_counts_path),
            },
        }
        write_json(out_dir / "summary.json", type_summary)
        summary["task_types"][task_type] = type_summary
        print(
            f"{subset}/type {task_type}: SKD_TDN={len(rows)}, pairs={len(pair_rows)}, "
            f"top_ratio={args.top_ratio:g}",
            flush=True,
        )

    density_path = viz_dir / f"skd_tdn_density_heatmap_{subset}.png"
    plot_density(rows_by_type, module_meta, density_path)
    summary["visualizations"]["density_heatmap"] = str(density_path)
    write_json(out_root / subset / "summary.json", summary)
    write_json(out_root / subset / "manifest.json", {"params": params, "summary": summary})
    return summary


def main() -> None:
    args = parse_args()
    activation_root = resolve_root(args.activations_dir, "activations")
    neurons_root = resolve_root(args.neurons_dir, "neurons")
    viz_root = resolve_root(args.visualizations_dir, "visualizations")
    device = resolve_compute_device(args.device)
    print(f"SafetyKernel_Deepfake compute device: {device}", flush=True)
    print(f"SafetyKernel_Deepfake subset order = {' -> '.join(subset_values(args.subset))}", flush=True)

    root_manifest: dict[str, Any] = {
        "stage": "skdf_05_single_type_paired_shift_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for subset in progress(subset_values(args.subset), desc=f"SKD-5 {args.model_alias}", unit="subset"):
        subset_summary = run_subset(
            args,
            subset=subset,
            activation_root=activation_root,
            neurons_root=neurons_root,
            viz_root=viz_root,
            device=device,
        )
        root_manifest["subsets"][subset] = subset_summary
        for task_type, type_summary in subset_summary["task_types"].items():
            summary_rows.append(
                {
                    "model_alias": args.model_alias,
                    "subset": subset,
                    "task_type": task_type,
                    "method": METHOD_NAME,
                    "neuron_set": "SKD_TDN",
                    "selected_neurons": type_summary["selected_neurons"],
                    "n_pairs": type_summary["n_pairs"],
                }
            )

    model_root = single_type_root(neurons_root, args.model_alias)
    ensure_dir(model_root)
    write_csv_rows(
        model_root / "single_type_summary.csv",
        summary_rows,
        fieldnames=["model_alias", "subset", "task_type", "method", "neuron_set", "selected_neurons", "n_pairs"],
    )
    write_json(model_root / "manifest.json", root_manifest)
    print(f"Wrote SafetyKernel_Deepfake single-type manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
