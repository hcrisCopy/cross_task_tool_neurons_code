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
METHOD_NAME = "PreciseShield_noABC"
INTERMEDIATE_MODULE = "ffn_intermediate"
NOABC_FILENAME = "PS_noABC_TDN_neurons.jsonl"
LAYER_TOP_SCORE_RATIO = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PreciseShield_noABC: discover global tool-decision FFN intermediate neurons "
            "with tool_necessary=1 vs tool_necessary=0, without A/B/C splitting."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="PreciseShield PS-4 activation root; defaults to ../cross_task_tool_neurons_data/precise_shield/activations.",
    )
    parser.add_argument(
        "--neurons-dir",
        default=None,
        help="PreciseShield_noABC neuron output root; defaults to ../cross_task_tool_neurons_data/precise_shield_noabc/neurons.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help="PreciseShield_noABC visualization root; defaults to ../cross_task_tool_neurons_data/precise_shield_noabc/visualizations.",
    )
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument(
        "--intervention-ratio",
        type=float,
        default=0.01,
        help="PreciseShield per-layer ratio p; k_l=max(floor(p*d_m), min_neurons_per_layer).",
    )
    parser.add_argument("--min-neurons-per-layer", type=int, default=1)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        help="Tensor statistics device: auto, cpu, cuda, or cuda:<index>. This is not a method parameter.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_root(kind: str) -> Path:
    mapping = {
        "activations": data_root() / "precise_shield" / "activations",
        "neurons": data_root() / "precise_shield_noabc" / "neurons",
        "visualizations": data_root() / "precise_shield_noabc" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown PreciseShield_noABC root kind: {kind}")
    return mapping[kind]


def resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def shared_root(neurons_root: Path, model_alias: str) -> Path:
    # ProbePrefill reads one method-wide neuron set from shared_by_subset.
    return neurons_root / model_alias / "shared_by_subset"


def topk_count(dim: int, ratio: float, minimum: int) -> int:
    if ratio <= 0:
        raise ValueError("--intervention-ratio must be > 0")
    if minimum <= 0:
        raise ValueError("--min-neurons-per-layer must be > 0")
    return min(dim, max(int(math.floor(float(ratio) * dim)), int(minimum)))


def resolve_compute_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    return device


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
        "stage": "psna_05_noabc_tool_decision_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activations_dir": str(activation_dir),
        "neurons_dir": str(neurons_root),
        "visualizations_dir": str(viz_root),
        "intervention_ratio": args.intervention_ratio,
        "min_neurons_per_layer": args.min_neurons_per_layer,
        "heatmap_top_n": args.heatmap_top_n,
        "epsilon": args.epsilon,
        "min_class_count": args.min_class_count,
        "activation_manifest_params": activation_manifest.get("params", {}),
        "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
        "score_definition": "I_D_i=|mean_D h_i|*||W_down[:,i]||_2; S_D_i=I_D_i/(sum_j I_D_j+epsilon)",
        "selection": "per_layer TopK(S_tool_necessary) minus TopK(S_tool_unnecessary), no A/B/C split",
        "neuron_identity": "(layer, index) over ffn_intermediate",
    }


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"ps_noabc_density_heatmap_{subset}.png",
        viz_dir / f"ps_noabc_saliency_heatmap_{subset}.png",
        viz_dir / f"ps_noabc_layer_top1pct_saliency_heatmap_{subset}.png",
    ]


def expected_outputs(out_dir: Path, viz_dir: Path, subset: str) -> list[Path]:
    return [
        out_dir / NOABC_FILENAME,
        out_dir / "saliency_scores.pt",
        out_dir / "layer_counts.csv",
        out_dir / "class_balance.csv",
        out_dir / "summary.json",
        out_dir / "manifest.json",
        *expected_visualizations(viz_dir, subset),
    ]


def remove_existing_files(paths: Iterable[Path]) -> None:
    root = data_root().resolve()
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Refusing to remove outside data root: {resolved}")
        path.unlink()


def clean_outputs(out_dir: Path, viz_dir: Path, subset: str) -> None:
    clean_directory(out_dir, data_root())
    remove_existing_files(expected_visualizations(viz_dir, subset))


def should_skip(
    out_dir: Path,
    viz_dir: Path,
    subset: str,
    params: dict[str, Any],
    *,
    overwrite: bool,
    clean: bool,
) -> bool:
    if clean:
        clean_outputs(out_dir, viz_dir, subset)
        return False
    if overwrite:
        return False
    expected = expected_outputs(out_dir, viz_dir, subset)
    if not all(path.exists() for path in expected):
        return False
    manifest = read_json(out_dir / "manifest.json")
    if manifest.get("params") == params:
        print(f"Skip existing PreciseShield_noABC neurons: {out_dir}", flush=True)
        return True
    return False


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
    importance = mean_activation.abs() * norm
    saliency = importance / (importance.sum() + eps)
    return {
        "mean_activation": mean_activation.cpu(),
        "importance": importance.cpu(),
        "saliency": saliency.cpu(),
    }


def label_indices(meta_rows: list[dict[str, Any]], min_class_count: int) -> tuple[list[int], list[int]]:
    call_indices = [idx for idx, row in enumerate(meta_rows) if int(row["tool_necessary"]) == 1]
    direct_indices = [idx for idx, row in enumerate(meta_rows) if int(row["tool_necessary"]) == 0]
    if len(call_indices) < min_class_count or len(direct_indices) < min_class_count:
        raise ValueError(
            f"Need at least {min_class_count} train examples per label, "
            f"got tool_necessary=1: {len(call_indices)}, tool_necessary=0: {len(direct_indices)}"
        )
    return call_indices, direct_indices


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


def discover_noabc_rows(
    *,
    payload: dict[str, Any],
    meta_rows: list[dict[str, Any]],
    intervention_ratio: float,
    min_neurons_per_layer: int,
    eps: float,
    min_class_count: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    activations: dict[str, torch.Tensor] = payload["activations"]
    down_norms: dict[str, torch.Tensor] = payload["down_weight_norms"]
    module_meta: list[dict[str, Any]] = payload["module_meta"]
    call_indices, direct_indices = label_indices(meta_rows, min_class_count)

    rows: list[dict[str, Any]] = []
    score_pack: dict[str, dict[str, torch.Tensor]] = {}
    for meta in progress(module_meta, desc=f"{payload['subset']} noABC PS saliency", unit="layer"):
        key = str(meta["key"])
        layer = int(meta["layer"])
        dim = int(meta["dim"])
        k = topk_count(dim, intervention_ratio, min_neurons_per_layer)
        call_pack = compute_layer_saliency(activations[key], down_norms[key], call_indices, eps, device)
        direct_pack = compute_layer_saliency(activations[key], down_norms[key], direct_indices, eps, device)
        call_vals, call_idxs = torch.topk(call_pack["saliency"], k)
        _direct_vals, direct_idxs = torch.topk(direct_pack["saliency"], k)
        direct_set = {int(index) for index in direct_idxs.tolist()}
        rank_by_index = {int(index): rank for rank, index in enumerate(call_idxs.tolist(), start=1)}
        for score, raw_index in zip(call_vals.tolist(), call_idxs.tolist()):
            index = int(raw_index)
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
                    "selection_scope": "all_task_types_noABC",
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
        "n_tool_necessary": len(call_indices),
        "n_tool_unnecessary": len(direct_indices),
        "selected_neurons": len(rows),
        "intervention_ratio": intervention_ratio,
        "top_layers": Counter(int(row["layer"]) for row in rows).most_common(10),
    }
    return rows, score_pack, summary


def write_counts_csv(rows: list[dict[str, Any]], path: Path, field: str) -> None:
    ensure_dir(path.parent)
    counts = Counter(row[field] for row in rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field, "count"])
        for key, count in sorted(counts.items()):
            writer.writerow([key, count])


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    names = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        if not names:
            return
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


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


def layer_dims_from_meta(module_meta: list[dict[str, Any]]) -> dict[int, int]:
    return {int(meta["layer"]): int(meta["dim"]) for meta in module_meta}


def plot_density(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    layer_dims = layer_dims_from_meta(module_meta)
    layers = sorted(layer_dims) if layer_dims else sorted({int(row["layer"]) for row in rows})
    if not layers:
        write_empty_plot(out_path, "PreciseShield_noABC Neurons")
        return
    counts = Counter(int(row["layer"]) for row in rows)
    matrix = [[counts.get(layer, 0) / max(layer_dims.get(layer, 1), 1)] for layer in layers]
    fig, ax = plt.subplots(figsize=(4.2, max(4, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("PreciseShield_noABC Neurons")
    ax.set_xticks([0], [INTERMEDIATE_MODULE], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN neuron space")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_saliency(rows: list[dict[str, Any]], out_path: Path, top_n: int) -> None:
    selected = rows[: max(1, min(top_n, len(rows)))]
    if not selected:
        write_empty_plot(out_path, "PreciseShield_noABC call saliency")
        return
    layers = sorted({int(row["layer"]) for row in selected})
    y_by_layer = {layer: idx for idx, layer in enumerate(layers)}
    matrix = torch.full((len(layers), len(selected)), float("nan"), dtype=torch.float32)
    for x, row in enumerate(selected):
        matrix[y_by_layer[int(row["layer"])], x] = float(row["score"])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(layers) * 0.28)))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title("PreciseShield_noABC call saliency")
    ax.set_xlabel("Global neuron rank")
    ax.set_ylabel("Layer")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="S(tool_necessary=1)")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_saliency_heatmap(
    *,
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    ordered_meta = sorted(module_meta, key=lambda meta: (int(meta["layer"]), str(meta["key"])))
    row_values: list[torch.Tensor] = []
    row_labels: list[str] = []
    max_cols = 0
    for meta in ordered_meta:
        key = str(meta["key"])
        scores = score_pack.get(key, {}).get("call_saliency")
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
        write_empty_plot(out_path, "PreciseShield_noABC top 1% saliency by layer")
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
    ax.set_title("PreciseShield_noABC: top 1% call saliency by layer")
    ax.set_xlabel(f"Neuron rank within top {int(ratio * 100)}% of each layer")
    ax.set_ylabel("Layer / FFN neuron space")
    ticks = list(range(0, max_cols, max(1, max_cols // 10)))
    if ticks and ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="S(tool_necessary=1)")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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
        raise FileNotFoundError(f"Missing PreciseShield train activations for {subset}: {act_dir}")

    activation_manifest = read_json(manifest_path)
    out_dir = shared_root(neurons_root, args.model_alias) / subset
    viz_dir = viz_root / args.model_alias / "shared_by_subset"
    params = expected_params(
        args,
        subset=subset,
        activation_dir=act_dir,
        activation_manifest=activation_manifest,
        neurons_root=neurons_root,
        viz_root=viz_root,
    )
    if should_skip(out_dir, viz_dir, subset, params, overwrite=args.overwrite, clean=args.clean):
        return read_json(out_dir / "summary.json")

    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    meta_rows = read_jsonl(meta_path)
    module_meta = payload["module_meta"]
    rows, score_pack, discovery_summary = discover_noabc_rows(
        payload=payload,
        meta_rows=meta_rows,
        intervention_ratio=args.intervention_ratio,
        min_neurons_per_layer=args.min_neurons_per_layer,
        eps=args.epsilon,
        min_class_count=args.min_class_count,
        device=device,
    )

    ensure_dir(out_dir)
    write_jsonl(out_dir / NOABC_FILENAME, rows)
    torch.save(
        {
            "method": METHOD_NAME,
            "subset": subset,
            "module_meta": module_meta,
            "scores": score_pack,
            "score_definition": "PreciseShield saliency set difference over all task types",
        },
        out_dir / "saliency_scores.pt",
    )
    write_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
    class_rows = class_balance_rows(meta_rows, args.model_alias, subset)
    write_csv_rows(
        out_dir / "class_balance.csv",
        class_rows,
        fieldnames=["model_alias", "subset", "task_type", "tool_necessary", "count"],
    )

    density_path = viz_dir / f"ps_noabc_density_heatmap_{subset}.png"
    saliency_path = viz_dir / f"ps_noabc_saliency_heatmap_{subset}.png"
    layer_top1pct_path = viz_dir / f"ps_noabc_layer_top1pct_saliency_heatmap_{subset}.png"
    plot_density(rows, module_meta, density_path)
    plot_saliency(rows, saliency_path, args.heatmap_top_n)
    plot_layer_top_saliency_heatmap(score_pack=score_pack, module_meta=module_meta, out_path=layer_top1pct_path)

    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_file": str(out_dir / NOABC_FILENAME),
        "neuron_set": "PS_noABC_TDN",
        "selected_neurons": len(rows),
        "module": INTERMEDIATE_MODULE,
        "class_balance": class_rows,
        "top_layers": discovery_summary["top_layers"],
        "n_tool_necessary": discovery_summary["n_tool_necessary"],
        "n_tool_unnecessary": discovery_summary["n_tool_unnecessary"],
        "visualizations": {
            "density_heatmap": str(density_path),
            "saliency_heatmap": str(saliency_path),
            "layer_top1pct_saliency_heatmap": str(layer_top1pct_path),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    print(
        f"{subset}: PS_noABC_TDN={len(rows)}, "
        f"tool_necessary=1/{discovery_summary['n_tool_necessary']}, "
        f"tool_necessary=0/{discovery_summary['n_tool_unnecessary']}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    activation_root = resolve_root(args.activations_dir, "activations")
    neurons_root = resolve_root(args.neurons_dir, "neurons")
    viz_root = resolve_root(args.visualizations_dir, "visualizations")
    device = resolve_compute_device(args.device)
    print(f"PreciseShield_noABC compute device: {device}", flush=True)
    print(f"PreciseShield_noABC subset order = {' -> '.join(subset_values(args.subset))}", flush=True)

    root_manifest: dict[str, Any] = {
        "stage": "psna_05_noabc_tool_decision_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for subset in progress(subset_values(args.subset), desc=f"PSNA-5 {args.model_alias}", unit="subset"):
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
                "neuron_set": "PS_noABC_TDN",
                "selected_neurons": summary.get("selected_neurons", 0),
                "n_tool_necessary": summary.get("n_tool_necessary", 0),
                "n_tool_unnecessary": summary.get("n_tool_unnecessary", 0),
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
            "n_tool_necessary",
            "n_tool_unnecessary",
        ],
    )
    write_json(model_root / "manifest.json", root_manifest)
    print(f"Wrote PreciseShield_noABC manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
