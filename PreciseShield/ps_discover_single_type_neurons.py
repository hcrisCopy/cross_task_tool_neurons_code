from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ps_common import (
    INTERMEDIATE_MODULE,
    STAGE_VERSION,
    TASK_TYPES,
    clean_path,
    ps_resolve_root,
    read_json,
    read_jsonl,
    should_skip,
    subset_values,
    write_json,
    write_jsonl,
)
from cttn.paths import ensure_dir
from cttn.progress import progress


LAYER_TOP_SCORE_RATIO = 0.01
HARDWARE_ONLY_PARAM_KEYS = {"device"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PreciseShield stage 5: discover A/B/C tool-call neurons with saliency set difference."
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--visualizations-dir", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument(
        "--intervention-ratio",
        type=float,
        default=0.01,
        help="PreciseShield per-layer ratio p; k_l=floor(p*d_m).",
    )
    parser.add_argument("--min-neurons-per-layer", type=int, default=1)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for PreciseShield saliency tensor statistics: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def single_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "single_type_by_subset"


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "ps_05_single_type_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": "PreciseShield",
        "model_alias": args.model_alias,
        "subset": subset,
        "intervention_ratio": args.intervention_ratio,
        "min_neurons_per_layer": args.min_neurons_per_layer,
        "heatmap_top_n": args.heatmap_top_n,
        "epsilon": args.epsilon,
        "min_class_count": args.min_class_count,
        "device": args.device,
        "activation_dir": str(activation_dir),
        "activation_manifest_params": activation_manifest.get("params", {}),
        "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
        "score_definition": "I_D_i=||mean_activation_D_i * W_down_col_i||_2; S_D_i=I_D_i/(sum_k I_D_k+eps)",
        "selection": "per_type_per_layer TopK(S_call) minus TopK(S_direct)",
    }


def topk_count(dim: int, ratio: float, minimum: int) -> int:
    if ratio <= 0:
        raise ValueError("--intervention-ratio must be > 0")
    return min(dim, max(int(math.floor(ratio * dim)), int(minimum)))


def resolve_compute_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for PreciseShield PS-5, but torch.cuda.is_available() is false")
    return device


def compute_layer_saliency(
    activations: torch.Tensor,
    down_norm: torch.Tensor,
    indices: list[int],
    eps: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    idx = torch.tensor(indices, dtype=torch.long)
    if device.type == "cuda":
        with torch.cuda.device(device):
            x = activations.index_select(0, idx).to(device, non_blocking=True).float()
            norm = down_norm.to(device, non_blocking=True).float()
            mean_activation = x.mean(dim=0)
            # Eq. (5): I_i(D)=||a_bar_i(D) * w_down_i||_2 = |a_bar_i(D)| * ||w_down_i||_2.
            importance = mean_activation.abs() * norm
            saliency = importance / (importance.sum() + eps)
            result = {
                "mean_activation": mean_activation.detach().cpu(),
                "importance": importance.detach().cpu(),
                "saliency": saliency.detach().cpu(),
            }
            del x, norm, mean_activation, importance, saliency
            torch.cuda.empty_cache()
            return result

    x = activations.index_select(0, idx).to(device).float()
    norm = down_norm.to(device).float()
    mean_activation = x.mean(dim=0)
    # Eq. (5): I_i(D)=||a_bar_i(D) * w_down_i||_2 = |a_bar_i(D)| * ||w_down_i||_2.
    importance = mean_activation.abs() * norm
    saliency = importance / (importance.sum() + eps)
    return {
        "mean_activation": mean_activation.cpu(),
        "importance": importance.cpu(),
        "saliency": saliency.cpu(),
    }


def topk_cpu(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(values, k)


def selected_rows_for_type(
    *,
    task_type: str,
    payload: dict[str, Any],
    meta_rows: list[dict[str, Any]],
    intervention_ratio: float,
    min_neurons_per_layer: int,
    eps: float,
    min_class_count: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    activations: dict[str, torch.Tensor] = payload["activations"]
    down_norms: dict[str, torch.Tensor] = payload["down_weight_norms"]
    module_meta: list[dict[str, Any]] = payload["module_meta"]
    call_indices = [
        idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 1
    ]
    direct_indices = [
        idx for idx, row in enumerate(meta_rows) if row["task_type"] == task_type and int(row["tool_necessary"]) == 0
    ]
    if len(call_indices) < min_class_count or len(direct_indices) < min_class_count:
        raise ValueError(
            f"{payload['subset']}/type {task_type} needs at least {min_class_count} per class, "
            f"got call={len(call_indices)}, direct={len(direct_indices)}"
        )

    rows: list[dict[str, Any]] = []
    score_pack: dict[str, dict[str, torch.Tensor]] = {}
    for meta in progress(module_meta, desc=f"{payload['subset']}/type {task_type} PS saliency"):
        key = meta["key"]
        layer = int(meta["layer"])
        dim = int(meta["dim"])
        k = topk_count(dim, intervention_ratio, min_neurons_per_layer)
        call_pack = compute_layer_saliency(activations[key], down_norms[key], call_indices, eps, device)
        direct_pack = compute_layer_saliency(activations[key], down_norms[key], direct_indices, eps, device)
        call_vals, call_idxs = topk_cpu(call_pack["saliency"], k)
        _direct_vals, direct_idxs = topk_cpu(direct_pack["saliency"], k)
        direct_set = {int(i) for i in direct_idxs.tolist()}
        rank_by_index = {int(idx): rank for rank, idx in enumerate(call_idxs.tolist(), start=1)}
        for score, idx in zip(call_vals.tolist(), call_idxs.tolist()):
            index = int(idx)
            if index in direct_set:
                continue
            rows.append(
                {
                    "layer": layer,
                    "module": INTERMEDIATE_MODULE,
                    "module_key": key,
                    "index": index,
                    "score": float(score),
                    "call_saliency": float(call_pack["saliency"][index]),
                    "direct_saliency": float(direct_pack["saliency"][index]),
                    "call_importance": float(call_pack["importance"][index]),
                    "direct_importance": float(direct_pack["importance"][index]),
                    "call_mean_activation": float(call_pack["mean_activation"][index]),
                    "direct_mean_activation": float(direct_pack["mean_activation"][index]),
                    "down_weight_l2_norm": float(down_norms[key][index]),
                    "rank_in_layer_call": int(rank_by_index[index]),
                    "topk_per_layer": k,
                }
            )
        score_pack[key] = {
            "call_saliency": call_pack["saliency"],
            "direct_saliency": direct_pack["saliency"],
            "call_importance": call_pack["importance"],
            "direct_importance": direct_pack["importance"],
            "call_mean_activation": call_pack["mean_activation"],
            "direct_mean_activation": direct_pack["mean_activation"],
            "down_weight_l2_norm": down_norms[key].float().cpu(),
        }

    rows.sort(key=lambda row: (-float(row["score"]), int(row["layer"]), int(row["index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    summary = {
        "task_type": task_type,
        "n_call": len(call_indices),
        "n_direct": len(direct_indices),
        "selected_neurons": len(rows),
        "intervention_ratio": intervention_ratio,
        "top_layers": Counter(row["layer"] for row in rows).most_common(10),
    }
    return rows, summary, score_pack


def plot_density(rows_by_type: dict[str, list[dict[str, Any]]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    layers = [int(meta["layer"]) for meta in module_meta]
    dims = {int(meta["layer"]): int(meta["dim"]) for meta in module_meta}
    fig, axes = plt.subplots(1, 3, figsize=(9, max(4, len(layers) * 0.22)), sharey=True)
    for ax, task_type in zip(axes, TASK_TYPES):
        counts = Counter(int(row["layer"]) for row in rows_by_type.get(task_type, []))
        matrix = [[counts.get(layer, 0) / max(dims.get(layer, 1), 1)] for layer in layers]
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(f"Type {task_type}")
        ax.set_xticks([0], [INTERMEDIATE_MODULE], rotation=30, ha="right")
        ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
        if ax is axes[0]:
            ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_saliency_rows(rows: list[dict[str, Any]], out_path: Path, top_n: int, title: str) -> None:
    selected = rows[: max(1, min(top_n, len(rows)))]
    if not selected:
        return
    layers = sorted({int(row["layer"]) for row in selected})
    layer_to_y = {layer: idx for idx, layer in enumerate(layers)}
    matrix = torch.full((len(layers), len(selected)), float("nan"), dtype=torch.float32)
    for x, row in enumerate(selected):
        matrix[layer_to_y[int(row["layer"])], x] = float(row["score"])
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(layers) * 0.28)))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Global PS-TDN rank")
    ax.set_ylabel("Layer")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="call saliency")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_saliency_heatmap(
    *,
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    score_field: str,
    score_label: str,
    title: str,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    ordered_meta = sorted(module_meta, key=lambda meta: (int(meta["layer"]), str(meta["key"])))
    row_values: list[torch.Tensor] = []
    row_labels: list[str] = []
    max_cols = 0
    for meta in ordered_meta:
        key = meta["key"]
        scores = score_pack.get(key, {}).get(score_field)
        if scores is None:
            continue
        values = scores.detach().float().cpu()
        k = max(1, int(values.numel() * ratio))
        k = min(k, values.numel())
        top_values = torch.topk(values, k).values
        row_values.append(top_values)
        row_labels.append(f"L{int(meta['layer'])}.{INTERMEDIATE_MODULE}")
        max_cols = max(max_cols, k)
    if not row_values or max_cols <= 0:
        return

    matrix = torch.full((len(row_values), max_cols), float("nan"), dtype=torch.float32)
    for row_idx, values in enumerate(row_values):
        matrix[row_idx, : values.numel()] = values

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#f3f4f6")
    fig_width = max(10, min(42, max_cols * 0.018))
    fig_height = max(6, len(row_labels) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(f"Neuron rank within top {int(ratio * 100)}% of each layer")
    ax.set_ylabel("Layer / FFN neuron space")
    ticks = list(range(0, max_cols, max(1, max_cols // 10)))
    if ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label=score_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_counts_csv(rows: list[dict[str, Any]], path: Path, field: str) -> None:
    counts = Counter(row[field] for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field, "count"])
        for key, count in sorted(counts.items()):
            writer.writerow([key, count])


def expected_viz(viz_dir: Path, subset: str) -> list[Path]:
    return (
        [viz_dir / f"{subset}_ps_density_heatmap.png"]
        + [viz_dir / f"ps_tdn_saliency_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]
        + [viz_dir / f"ps_layer_top1pct_saliency_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]
    )


def expected_layer_top1pct_viz(viz_dir: Path, subset: str) -> list[Path]:
    return [viz_dir / f"ps_layer_top1pct_saliency_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]


def legacy_layer_top_viz(viz_dir: Path, subset: str) -> list[Path]:
    patterns = [
        "ps_layer_top3pct_saliency_heatmap_{subset}_{task_type}.png",
        "ps_layer_top10_saliency_heatmap_{subset}_{task_type}.png",
    ]
    return [
        viz_dir / pattern.format(subset=subset, task_type=task_type)
        for pattern in patterns
        for task_type in TASK_TYPES
    ]


def clean_legacy_layer_top_viz(viz_dir: Path, subset: str) -> None:
    for path in legacy_layer_top_viz(viz_dir, subset):
        if path.exists():
            path.unlink()


def clean_viz(viz_dir: Path, subset: str) -> None:
    for path in expected_viz(viz_dir, subset):
        if path.exists():
            path.unlink()
    clean_legacy_layer_top_viz(viz_dir, subset)


def params_match_or_hardware_compatible(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    if existing == expected:
        return True
    existing_core = {k: v for k, v in existing.items() if k not in HARDWARE_ONLY_PARAM_KEYS}
    expected_core = {k: v for k, v in expected.items() if k not in HARDWARE_ONLY_PARAM_KEYS}
    return existing_core == expected_core


def can_backfill_from_scores(subset_dir: Path, params: dict[str, Any]) -> bool:
    manifest_path = subset_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    if not params_match_or_hardware_compatible(manifest.get("params", {}), params):
        return False
    expected = [subset_dir / task_type / "PS_TDN_neurons.jsonl" for task_type in TASK_TYPES]
    expected.extend(subset_dir / task_type / "saliency_scores.pt" for task_type in TASK_TYPES)
    expected.append(subset_dir / "module_meta.json")
    return all(path.exists() for path in expected)


def backfill_layer_top1pct_visualizations(
    *,
    subset_dir: Path,
    viz_dir: Path,
    subset: str,
    params: dict[str, Any],
) -> bool:
    target_paths = expected_layer_top1pct_viz(viz_dir, subset)
    if all(path.exists() for path in target_paths):
        clean_legacy_layer_top_viz(viz_dir, subset)
        return False
    if not can_backfill_from_scores(subset_dir, params):
        return False

    clean_legacy_layer_top_viz(viz_dir, subset)
    layer_top1pct_paths: dict[str, str] = {}
    for task_type in TASK_TYPES:
        scores_payload = torch.load(subset_dir / task_type / "saliency_scores.pt", map_location="cpu")
        out_path = viz_dir / f"ps_layer_top1pct_saliency_heatmap_{subset}_{task_type}.png"
        plot_layer_top_saliency_heatmap(
            score_pack=scores_payload["scores"],
            module_meta=scores_payload["module_meta"],
            out_path=out_path,
            score_field="call_saliency",
            score_label="PreciseShield call saliency",
            title=f"{subset} Type {task_type}: top 1% PS saliency by layer",
        )
        layer_top1pct_paths[task_type] = str(out_path)

    summary_path = subset_dir / "summary.json"
    manifest_path = subset_dir / "manifest.json"
    summary = read_json(summary_path) if summary_path.exists() else {"subset": subset}
    visualizations = summary.setdefault("visualizations", {})
    visualizations.pop("layer_top3pct_saliency_heatmaps", None)
    visualizations.pop("layer_top10_saliency_heatmaps", None)
    visualizations["layer_top1pct_saliency_heatmaps"] = layer_top1pct_paths
    write_json(summary_path, summary)
    manifest = read_json(manifest_path)
    manifest["params"] = params
    manifest["summary"] = summary
    manifest.setdefault("visualizations", {})
    manifest["visualizations"].pop("layer_top3pct_saliency_heatmaps", None)
    manifest["visualizations"].pop("layer_top10_saliency_heatmaps", None)
    manifest["visualizations"].update(summary["visualizations"])
    write_json(manifest_path, manifest)
    print(f"Backfilled layer top-1% PreciseShield saliency visualizations: {subset_dir}")
    return True


def should_skip_hardware_compatible(
    subset_dir: Path,
    params: dict[str, Any],
    expected_files: list[Path],
    *,
    overwrite: bool,
    clean: bool,
) -> bool:
    if overwrite or clean:
        return False
    manifest_path = subset_dir / "manifest.json"
    files = [manifest_path, *expected_files]
    if not all(path.exists() for path in files):
        return False
    manifest = read_json(manifest_path)
    if params_match_or_hardware_compatible(manifest.get("params", {}), params):
        print(f"Skip existing output: {subset_dir}")
        return True
    return False


def main() -> None:
    args = parse_args()
    activation_root = ps_resolve_root(args.activations_dir, "activations")
    neurons_root = ps_resolve_root(args.neurons_dir, "neurons")
    viz_root = ps_resolve_root(args.visualizations_dir, "visualizations")
    model_single_root = single_root(neurons_root, args.model_alias)
    viz_dir = viz_root / args.model_alias / "single_type_by_subset"
    subsets = subset_values(args.subset)
    compute_device = resolve_compute_device(args.device)
    print(f"PreciseShield PS-5 compute device: {compute_device}")

    for subset in subsets:
        act_dir = activation_root / args.model_alias / subset / "train"
        activation_path = act_dir / "activations.pt"
        meta_path = act_dir / "meta.jsonl"
        manifest_path = act_dir / "manifest.json"
        if not activation_path.exists() or not meta_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Missing PreciseShield train activations for {subset}: {act_dir}")
        activation_manifest = read_json(manifest_path)
        params = expected_params(args, subset=subset, activation_dir=act_dir, activation_manifest=activation_manifest)
        subset_dir = model_single_root / subset
        expected_files = [
            subset_dir / task_type / "PS_TDN_neurons.jsonl" for task_type in TASK_TYPES
        ] + expected_viz(viz_dir, subset)
        if args.clean:
            clean_viz(viz_dir, subset)
        if should_skip_hardware_compatible(
            subset_dir,
            params,
            expected_files,
            overwrite=args.overwrite,
            clean=args.clean,
        ):
            continue
        if should_skip(
            subset_dir,
            params,
            expected_files,
            overwrite=args.overwrite,
            clean=args.clean,
        ):
            continue
        if (not args.overwrite) and (not args.clean) and backfill_layer_top1pct_visualizations(
            subset_dir=subset_dir,
            viz_dir=viz_dir,
            subset=subset,
            params=params,
        ):
            continue

        payload = torch.load(activation_path, map_location="cpu")
        meta_rows = read_jsonl(meta_path)
        module_meta = payload["module_meta"]
        rows_by_type: dict[str, list[dict[str, Any]]] = {}
        summary = {"subset": subset, "task_types": {}, "visualizations": {}}
        layer_top1pct_saliency_heatmaps: dict[str, str] = {}

        for task_type in TASK_TYPES:
            rows, type_summary, score_pack = selected_rows_for_type(
                task_type=task_type,
                payload=payload,
                meta_rows=meta_rows,
                intervention_ratio=args.intervention_ratio,
                min_neurons_per_layer=args.min_neurons_per_layer,
                eps=args.epsilon,
                min_class_count=args.min_class_count,
                device=compute_device,
            )
            rows_by_type[task_type] = rows
            out_dir = subset_dir / task_type
            ensure_dir(out_dir)
            write_jsonl(out_dir / "PS_TDN_neurons.jsonl", rows)
            torch.save(
                {
                    "task_type": task_type,
                    "subset": subset,
                    "module_meta": module_meta,
                    "scores": score_pack,
                    "score_definition": "PreciseShield saliency set difference",
                },
                out_dir / "saliency_scores.pt",
            )
            write_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
            write_json(out_dir / "summary.json", type_summary)
            heatmap_path = viz_dir / f"ps_tdn_saliency_heatmap_{subset}_{task_type}.png"
            plot_saliency_rows(rows, heatmap_path, args.heatmap_top_n, f"{subset} Type {task_type} PS-TDN")
            layer_top1pct_path = viz_dir / f"ps_layer_top1pct_saliency_heatmap_{subset}_{task_type}.png"
            plot_layer_top_saliency_heatmap(
                score_pack=score_pack,
                module_meta=module_meta,
                out_path=layer_top1pct_path,
                score_field="call_saliency",
                score_label="PreciseShield call saliency",
                title=f"{subset} Type {task_type}: top 1% PS saliency by layer",
            )
            type_summary["saliency_heatmap"] = str(heatmap_path)
            type_summary["layer_top1pct_saliency_heatmap"] = str(layer_top1pct_path)
            layer_top1pct_saliency_heatmaps[task_type] = str(layer_top1pct_path)
            summary["task_types"][task_type] = type_summary
            print(f"{subset}/type {task_type}: selected {len(rows)} PS-TDN neurons")

        density_path = viz_dir / f"{subset}_ps_density_heatmap.png"
        plot_density(rows_by_type, module_meta, density_path)
        summary["visualizations"]["density_heatmap"] = str(density_path)
        summary["visualizations"]["layer_top1pct_saliency_heatmaps"] = layer_top1pct_saliency_heatmaps
        write_json(subset_dir / "module_meta.json", module_meta)
        write_json(subset_dir / "summary.json", summary)
        write_json(subset_dir / "manifest.json", {"params": params, "summary": summary})
        print(f"Wrote PreciseShield single-type neurons: {subset_dir}")


if __name__ == "__main__":
    main()
