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


METHOD_NAME = "ResidualDecisionAnchors"
STAGE_VERSION = 1
CTD_FILENAME = "RDA_CTD_neurons.jsonl"
TDN_FILENAME = "RDA_TDN_neurons.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDA-5: discover ABC shared residual-state decision anchors.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--activations-dir", default="../cross_task_tool_neurons_data/residual_decision_anchors/activations")
    parser.add_argument("--neurons-dir", default="../cross_task_tool_neurons_data/residual_decision_anchors/neurons")
    parser.add_argument("--visualizations-dir", default="../cross_task_tool_neurons_data/residual_decision_anchors/visualizations")
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--top-ratio", type=float, default=0.80)
    parser.add_argument("--min-neurons-per-layer", type=int, default=1)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


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
    return min(dim, max(int(math.ceil(dim * ratio)), int(minimum)))


def shared_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "shared_by_subset"


def single_type_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "single_type_by_subset"


def zscore(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.nan_to_num((tensor - tensor.mean()) / tensor.std(unbiased=False).clamp_min(eps), nan=0.0, posinf=0.0, neginf=0.0)


def task_indices(meta_rows: list[dict[str, Any]], task_type: str, min_class_count: int) -> tuple[list[int], list[int]]:
    call = [idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 1]
    direct = [idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 0]
    if len(call) < min_class_count or len(direct) < min_class_count:
        raise ValueError(f"{task_type}: insufficient class coverage: tool={len(call)}, direct={len(direct)}")
    return call, direct


def compute_one_layer(
    *,
    tensor: torch.Tensor,
    call_indices: list[int],
    direct_indices: list[int],
    device: torch.device,
    eps: float,
) -> dict[str, torch.Tensor | float]:
    call_idx = torch.tensor(call_indices, dtype=torch.long)
    direct_idx = torch.tensor(direct_indices, dtype=torch.long)
    call_x = tensor.index_select(0, call_idx).to(device, non_blocking=True).float()
    direct_x = tensor.index_select(0, direct_idx).to(device, non_blocking=True).float()
    signed = (call_x.mean(dim=0) - direct_x.mean(dim=0)) / torch.sqrt(call_x.var(dim=0, unbiased=False) + direct_x.var(dim=0, unbiased=False) + eps)
    signed = torch.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
    z = zscore(signed, eps)
    out = {
        "signed_effect": signed.detach().cpu(),
        "z_effect": z.detach().cpu(),
        "mean_signed_effect": float(signed.mean().detach().cpu().item()),
        "std_signed_effect": float(signed.std(unbiased=False).detach().cpu().item()),
    }
    del call_x, direct_x, signed, z
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def compute_scores(
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
    call, direct = task_indices(meta_rows, task_type, min_class_count)
    scores: dict[str, dict[str, torch.Tensor | float]] = {}
    for meta in progress(module_meta, desc=desc, unit="layer"):
        key = str(meta["key"])
        scores[key] = compute_one_layer(tensor=activations[key], call_indices=call, direct_indices=direct, device=device, eps=eps)
    return scores, {"task_type": task_type, "n_tool": len(call), "n_direct": len(direct)}


def consensus(score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float]]], key: str) -> dict[str, torch.Tensor]:
    z = []
    for task_type in TASK_TYPES:
        item = score_by_task[task_type][key]["z_effect"]
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"Missing z tensor for {task_type}/{key}")
        z.append(item)
    positive = torch.minimum(torch.minimum(z[0], z[1]), z[2])
    negative = torch.minimum(torch.minimum(-z[0], -z[1]), -z[2])
    shared = torch.maximum(positive, negative)
    direction = torch.where(positive >= negative, torch.ones_like(shared), -torch.ones_like(shared))
    mean_abs = torch.stack([item.abs() for item in z], dim=0).mean(dim=0)
    return {"score": shared, "direction_sign": direction, "mean_abs_z": mean_abs, "z_A": z[0], "z_B": z[1], "z_C": z[2]}


def select_rows(
    *,
    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float]]],
    module_meta: list[dict[str, Any]],
    model_alias: str,
    subset: str,
    top_ratio: float,
    min_neurons_per_layer: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for meta in module_meta:
        key = str(meta["key"])
        dim = int(meta["dim"])
        k = top_count(dim, top_ratio, min_neurons_per_layer)
        pack = consensus(score_by_task, key)
        idxs = torch.topk(pack["score"], k=k, largest=True, sorted=False).indices.tolist()
        idxs.sort(key=lambda idx: (-float(pack["score"][idx]), -float(pack["mean_abs_z"][idx]), int(idx)))
        selected_scores = [float(pack["score"][idx]) for idx in idxs]
        layer_rows.append(
            {
                "model_alias": model_alias,
                "subset": subset,
                "layer": int(meta["layer"]),
                "module": "residual_state",
                "module_key": key,
                "module_dim": dim,
                "selected_neurons": len(idxs),
                "top_ratio": top_ratio,
                "score_mean": sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
                "score_min": min(selected_scores) if selected_scores else 0.0,
                "score_max": max(selected_scores) if selected_scores else 0.0,
            }
        )
        for rank_in_layer, idx in enumerate(idxs, start=1):
            direction_sign = int(float(pack["direction_sign"][idx]))
            row = {
                "model_alias": model_alias,
                "subset": subset,
                "layer": int(meta["layer"]),
                "module": "residual_state",
                "module_key": key,
                "index": int(idx),
                "rank_in_layer": rank_in_layer,
                "score": float(pack["score"][idx]),
                "signed_consensus_score": float(pack["score"][idx]),
                "mean_abs_z": float(pack["mean_abs_z"][idx]),
                "direction_sign": direction_sign,
                "direction": "tool_high" if direction_sign > 0 else "direct_high",
                "score_A": float(pack["z_A"][idx]),
                "score_B": float(pack["z_B"][idx]),
                "score_C": float(pack["z_C"][idx]),
                "top_ratio": top_ratio,
                "module_dim": dim,
            }
            for task_type in TASK_TYPES:
                signed = score_by_task[task_type][key]["signed_effect"]
                if not isinstance(signed, torch.Tensor):
                    raise TypeError(f"Missing signed tensor for {task_type}/{key}")
                row[f"signed_effect_{task_type}"] = float(signed[idx])
            rows.append(row)
    rows.sort(key=lambda row: (-float(row["score"]), -float(row["mean_abs_z"]), int(row["layer"]), int(row["index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
    return rows, layer_rows


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


def plot_layer_counts(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    counts = Counter(int(row["layer"]) for row in rows)
    layers = sorted(counts)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(layer) for layer in layers], [counts[layer] for layer in layers], color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel("hidden layer")
    ax.set_ylabel("selected residual dims")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_heatmap(rows: list[dict[str, Any]], out_path: Path, top_n: int, title: str) -> None:
    selected = rows[: max(1, min(top_n, len(rows)))]
    ensure_dir(out_path.parent)
    if not selected:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No neurons", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return
    layers = sorted({int(row["layer"]) for row in selected})
    y = {layer: idx for idx, layer in enumerate(layers)}
    matrix = [[float("nan") for _ in selected] for _ in layers]
    for x, row in enumerate(selected):
        matrix[y[int(row["layer"])]][x] = float(row["score"])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(layers) * 0.18)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("RDA-CTD rank")
    ax.set_ylabel("hidden layer")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="signed ABC consensus score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def expected_params(args: argparse.Namespace, subset: str, activation_dir: Path, activation_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "rda_05_signed_consensus_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activation_dir": str(activation_dir),
        "activation_manifest": activation_manifest,
        "top_ratio": args.top_ratio,
        "min_neurons_per_layer": args.min_neurons_per_layer,
        "min_class_count": args.min_class_count,
        "epsilon": args.epsilon,
        "score_definition": "score=max(min(z_A,z_B,z_C), min(-z_A,-z_B,-z_C)) over residual_state dimensions",
    }


def should_skip(out_dir: Path, single_subset_root: Path, viz_dir: Path, subset: str, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        clean_directory(single_subset_root, data_root())
        for path in [viz_dir / f"rda_ctd_layer_counts_{subset}.png", viz_dir / f"rda_ctd_score_heatmap_{subset}.png"]:
            if path.exists():
                path.unlink()
        return False
    expected = [
        out_dir / CTD_FILENAME,
        out_dir / "summary.json",
        out_dir / "manifest.json",
        out_dir / "top_neurons.csv",
        single_subset_root / "module_meta.json",
        single_subset_root / "manifest.json",
    ]
    if overwrite or not all(path.exists() for path in expected):
        return False
    if read_json(out_dir / "manifest.json").get("params") == params:
        print(f"Skip existing ResidualDecisionAnchors shared neurons: {out_dir}", flush=True)
        return True
    return False


def run_subset(args: argparse.Namespace, subset: str, activation_root: Path, neurons_root: Path, viz_root: Path, device: torch.device) -> dict[str, Any]:
    act_dir = activation_root / args.model_alias / subset / "train"
    activation_path = act_dir / "activations.pt"
    meta_path = act_dir / "meta.jsonl"
    manifest_path = act_dir / "manifest.json"
    if not activation_path.exists() or not meta_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing RDA train activations for {subset}: {act_dir}")
    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations: dict[str, torch.Tensor] = payload["activations"]
    module_meta: list[dict[str, Any]] = payload["module_meta"]
    meta_rows = read_jsonl(meta_path)
    activation_manifest = read_json(manifest_path)
    shared_dir = shared_root(neurons_root, args.model_alias)
    single_root = single_type_root(neurons_root, args.model_alias)
    out_dir = shared_dir / subset
    single_subset_root = single_root / subset
    viz_dir = viz_root / args.model_alias / "shared_by_subset"
    params = expected_params(args, subset, act_dir, activation_manifest)
    if should_skip(out_dir, single_subset_root, viz_dir, subset, params, args.overwrite, args.clean):
        return read_json(out_dir / "summary.json")

    score_by_task: dict[str, dict[str, dict[str, torch.Tensor | float]]] = {}
    class_summaries = {}
    for task_type in TASK_TYPES:
        scores, class_summary = compute_scores(
            activations=activations,
            module_meta=module_meta,
            meta_rows=meta_rows,
            task_type=task_type,
            min_class_count=args.min_class_count,
            device=device,
            eps=args.epsilon,
            desc=f"{subset}/type {task_type} residual effect",
        )
        score_by_task[task_type] = scores
        class_summaries[task_type] = class_summary

    rows, layer_rows = select_rows(
        score_by_task=score_by_task,
        module_meta=module_meta,
        model_alias=args.model_alias,
        subset=subset,
        top_ratio=args.top_ratio,
        min_neurons_per_layer=args.min_neurons_per_layer,
    )
    ensure_dir(out_dir)
    ensure_dir(single_subset_root)
    write_jsonl(out_dir / CTD_FILENAME, rows)
    write_csv_rows(out_dir / "top_neurons.csv", rows)
    write_csv_rows(out_dir / "layer_summary.csv", layer_rows)
    write_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
    write_counts_csv(rows, out_dir / "module_counts.csv", "module")
    write_json(single_subset_root / "module_meta.json", module_meta)
    for task_type in TASK_TYPES:
        type_dir = single_subset_root / task_type
        ensure_dir(type_dir)
        type_rows = [dict(row, task_type=task_type, task_score=row.get(f"score_{task_type}")) for row in rows]
        write_jsonl(type_dir / TDN_FILENAME, type_rows)
        write_json(type_dir / "summary.json", {"model_alias": args.model_alias, "subset": subset, "task_type": task_type, "selected_neurons": len(type_rows)})
    count_path = viz_dir / f"rda_ctd_layer_counts_{subset}.png"
    score_path = viz_dir / f"rda_ctd_score_heatmap_{subset}.png"
    plot_layer_counts(rows, count_path, f"{subset} RDA-CTD selected residual dimensions")
    plot_score_heatmap(rows, score_path, args.heatmap_top_n, f"{subset} RDA-CTD signed ABC consensus")
    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_file": str(out_dir / CTD_FILENAME),
        "neuron_set": "RDA_CTD",
        "selected_neurons": len(rows),
        "top_ratio": args.top_ratio,
        "class_summaries": class_summaries,
        "score_stats": {
            "min": min((float(row["score"]) for row in rows), default=0.0),
            "mean": sum(float(row["score"]) for row in rows) / max(len(rows), 1),
            "max": max((float(row["score"]) for row in rows), default=0.0),
        },
        "top_layers": Counter(int(row["layer"]) for row in rows).most_common(10),
        "visualizations": {"layer_counts": str(count_path), "score_heatmap": str(score_path)},
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    write_json(single_subset_root / "manifest.json", {"params": params, "summary": summary})
    print(
        f"{subset}: RDA_CTD={len(rows)}, score_mean={summary['score_stats']['mean']:.4f}, "
        f"score_max={summary['score_stats']['max']:.4f}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    activation_root = resolve_path(args.activations_dir)
    neurons_root = resolve_path(args.neurons_dir)
    viz_root = resolve_path(args.visualizations_dir)
    device = resolve_compute_device(args.device)
    print(f"ResidualDecisionAnchors compute device: {device}", flush=True)
    root_manifest: dict[str, Any] = {"stage": "rda_05_signed_consensus_shared_neuron_discovery", "stage_version": STAGE_VERSION, "method": METHOD_NAME, "model_alias": args.model_alias, "subsets": {}}
    summary_rows = []
    for subset in progress(subset_values(args.subset), desc=f"RDA-5 {args.model_alias}", unit="subset"):
        summary = run_subset(args, subset, activation_root, neurons_root, viz_root, device)
        root_manifest["subsets"][subset] = summary
        summary_rows.append({"model_alias": args.model_alias, "subset": subset, "method": METHOD_NAME, "neuron_set": "RDA_CTD", "selected_neurons": summary["selected_neurons"], "score_mean": summary["score_stats"]["mean"], "score_max": summary["score_stats"]["max"]})
    model_root = shared_root(neurons_root, args.model_alias)
    ensure_dir(model_root)
    write_csv_rows(model_root / "shared_summary.csv", summary_rows)
    write_json(model_root / "manifest.json", root_manifest)
    print(f"Wrote ResidualDecisionAnchors shared manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
