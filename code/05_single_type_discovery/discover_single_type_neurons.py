from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cttn.data import TASK_TYPES
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import progress


LAYER_TOP_SCORE_RATIO = 0.01
HARDWARE_ONLY_PARAM_KEYS = {"device", "devices"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: discover A/B/C single-type tool-decision neurons.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--visualizations-dir", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        help="Single device for SCAR tensor statistics when --devices is not set: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for module-parallel SCAR statistics, for example cuda:0,cuda:1,...,cuda:7.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_compute_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for Stage 5, but torch.cuda.is_available() is false")
    return device


def resolve_compute_devices(args: argparse.Namespace) -> list[torch.device]:
    if args.devices:
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
        if not devices:
            raise ValueError("--devices must contain at least one device")
        resolved = [resolve_compute_device(item) for item in devices]
        cuda_indices = [dev.index for dev in resolved if dev.type == "cuda"]
        if len(cuda_indices) != len(set(cuda_indices)):
            raise ValueError(f"--devices contains duplicate CUDA devices: {args.devices}")
        return resolved
    return [resolve_compute_device(args.device)]


def zscore(values: torch.Tensor, eps: float) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def compute_scar_for_module(x: torch.Tensor, labels: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    x = x.float()
    pos = x[labels == 1]
    neg = x[labels == 0]
    n1 = pos.shape[0]
    n0 = neg.shape[0]
    mu1 = pos.mean(dim=0)
    mu0 = neg.mean(dim=0)
    sigma1 = pos.std(dim=0, unbiased=True)
    sigma0 = neg.std(dim=0, unbiased=True)
    delta = mu1 - mu0
    pooled = torch.sqrt(((n1 - 1) * sigma1.pow(2) + (n0 - 1) * sigma0.pow(2)) / max(n1 + n0 - 2, 1)) + eps
    discriminability = zscore(delta / (pooled + eps), eps)
    responsiveness = zscore(delta, eps)
    scar = discriminability + responsiveness
    return {
        "scar": scar,
        "discriminability": discriminability,
        "responsiveness": responsiveness,
        "delta": delta,
        "mu1": mu1,
        "mu0": mu0,
    }


def compute_one_module(
    *,
    key: str,
    activations: dict[str, torch.Tensor],
    idx_tensor: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    eps: float,
) -> tuple[str, dict[str, torch.Tensor]]:
    if device.type == "cuda":
        with torch.cuda.device(device):
            x = activations[key].index_select(0, idx_tensor).to(device, non_blocking=True)
            scores = compute_scar_for_module(x, labels.to(device, non_blocking=True), eps)
            out = {name: tensor.detach().cpu() for name, tensor in scores.items()}
            del x
            torch.cuda.empty_cache()
            return key, out
    x = activations[key].index_select(0, idx_tensor).to(device)
    scores = compute_scar_for_module(x, labels.to(device), eps)
    return key, {name: tensor.detach().cpu() for name, tensor in scores.items()}


def compute_scar_for_modules(
    *,
    activations: dict[str, torch.Tensor],
    module_meta: list[dict[str, Any]],
    idx_tensor: torch.Tensor,
    labels: torch.Tensor,
    devices: list[torch.device],
    eps: float,
    desc: str,
) -> dict[str, dict[str, torch.Tensor]]:
    score_pack: dict[str, dict[str, torch.Tensor]] = {}
    if len(devices) == 1:
        device = devices[0]
        for meta in progress(module_meta, desc=desc):
            key, scores = compute_one_module(
                key=meta["key"],
                activations=activations,
                idx_tensor=idx_tensor,
                labels=labels,
                device=device,
                eps=eps,
            )
            score_pack[key] = scores
        return score_pack

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for pos, meta in enumerate(module_meta):
            futures.append(
                executor.submit(
                    compute_one_module,
                    key=meta["key"],
                    activations=activations,
                    idx_tensor=idx_tensor,
                    labels=labels,
                    device=devices[pos % len(devices)],
                    eps=eps,
                )
            )
        for future in progress(as_completed(futures), total=len(futures), desc=desc):
            key, scores = future.result()
            score_pack[key] = scores
    return score_pack


def topk_rows(
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    meta_by_key = {meta["key"]: meta for meta in module_meta}
    for key, scores in score_pack.items():
        scar = scores["scar"]
        kk = min(k, scar.numel())
        vals, idxs = torch.topk(scar, kk)
        meta = meta_by_key[key]
        for rank_in_module, (score, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
            i = int(idx)
            candidates.append(
                {
                    "layer": int(meta["layer"]),
                    "module": meta["module"],
                    "module_key": key,
                    "index": i,
                    "score": float(score),
                    "discriminability": float(scores["discriminability"][i]),
                    "responsiveness": float(scores["responsiveness"][i]),
                    "delta": float(scores["delta"][i]),
                    "mu1": float(scores["mu1"][i]),
                    "mu0": float(scores["mu0"][i]),
                    "rank_in_module": rank_in_module,
                }
            )
    candidates.sort(key=lambda row: (-row["score"], row["layer"], row["module"], row["index"]))
    rows = candidates[:k]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def plot_heatmap(rows_by_type: dict[str, list[dict[str, Any]]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    layers = sorted({int(meta["layer"]) for meta in module_meta})
    modules = ["gate_proj", "up_proj", "down_proj"]
    dims = {(int(meta["layer"]), meta["module"]): int(meta["dim"]) for meta in module_meta}
    fig, axes = plt.subplots(1, 3, figsize=(10, max(5, len(layers) * 0.22)), sharey=True)
    if len(TASK_TYPES) == 1:
        axes = [axes]
    for ax, task_type in zip(axes, TASK_TYPES):
        counts = Counter((row["layer"], row["module"]) for row in rows_by_type.get(task_type, []))
        matrix = []
        for layer in layers:
            row_vals = []
            for module in modules:
                dim = max(dims.get((layer, module), 1), 1)
                row_vals.append(counts.get((layer, module), 0) / dim)
            matrix.append(row_vals)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(f"Type {task_type}")
        ax.set_xticks(range(len(modules)), modules, rotation=30, ha="right")
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xlabel("FFN module")
        if ax is axes[0]:
            ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scar_heatmap(rows: list[dict[str, Any]], out_path: Path, top_n: int) -> None:
    selected = rows[: max(1, min(top_n, len(rows)))]
    if not selected:
        return
    group_order = sorted({(int(row["layer"]), str(row["module"])) for row in selected})
    group_to_y = {group: i for i, group in enumerate(group_order)}
    matrix = torch.full((len(group_order), len(selected)), float("nan"), dtype=torch.float32)
    for x, row in enumerate(selected):
        y = group_to_y[(int(row["layer"]), str(row["module"]))]
        matrix[y, x] = float(row["score"])

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(f"Top-{len(selected)} TDN-SCAR neurons")
    ax.set_xlabel("Global TDN rank")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{module}" for layer, module in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="SCAR")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_score_heatmap(
    *,
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    score_field: str,
    score_label: str,
    title: str,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    modules = ["gate_proj", "up_proj", "down_proj"]
    module_order = {name: idx for idx, name in enumerate(modules)}
    ordered_meta = sorted(
        module_meta,
        key=lambda meta: (int(meta["layer"]), module_order.get(str(meta["module"]), 99), str(meta["key"])),
    )
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
        row_labels.append(f"L{int(meta['layer'])}.{meta['module']}")
        max_cols = max(max_cols, k)
    if not row_values or max_cols <= 0:
        return

    matrix = torch.full((len(row_values), max_cols), float("nan"), dtype=torch.float32)
    for row_idx, values in enumerate(row_values):
        matrix[row_idx, : values.numel()] = values

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#f3f4f6")
    fig_width = max(10, min(42, max_cols * 0.018))
    fig_height = max(6, len(row_labels) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(f"Neuron rank within top {int(ratio * 100)}% of each layer/module")
    ax.set_ylabel("Layer / FFN module")
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


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return (
        [viz_dir / f"{subset}_heatmap.png"]
        + [viz_dir / f"tdn_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]
        + [viz_dir / f"layer_top1pct_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]
    )


def expected_layer_top1pct_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [viz_dir / f"layer_top1pct_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]


def legacy_layer_top_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    patterns = [
        "layer_top3pct_scar_heatmap_{subset}_{task_type}.png",
        "layer_top10_scar_heatmap_{subset}_{task_type}.png",
    ]
    return [
        viz_dir / pattern.format(subset=subset, task_type=task_type)
        for pattern in patterns
        for task_type in TASK_TYPES
    ]


def clean_legacy_layer_top_visualizations(viz_dir: Path, subset: str) -> None:
    for path in legacy_layer_top_visualizations(viz_dir, subset):
        if path.exists():
            path.unlink()


def clean_visualizations(viz_dir: Path, subset: str) -> None:
    for path in expected_visualizations(viz_dir, subset):
        if path.exists():
            path.unlink()
    clean_legacy_layer_top_visualizations(viz_dir, subset)


def expected_params(
    args: argparse.Namespace,
    *,
    activation_manifest: dict[str, Any],
    activation_path: Path,
    meta_path: Path,
    subset: str,
) -> dict[str, Any]:
    return {
        "stage": "05_single_type_discovery",
        "model_alias": args.model_alias,
        "subset": subset,
        "top_k": args.top_k,
        "heatmap_top_n": args.heatmap_top_n,
        "epsilon": args.epsilon,
        "min_class_count": args.min_class_count,
        "device": args.device,
        "devices": args.devices,
        "activation_path": str(activation_path),
        "meta_path": str(meta_path),
        "activation_manifest_params": activation_manifest.get("params", {}),
    }


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
    expected = [subset_dir / task_type / "TDN_neurons.jsonl" for task_type in TASK_TYPES]
    expected.extend(subset_dir / task_type / "scar_scores.pt" for task_type in TASK_TYPES)
    expected.append(subset_dir / "module_meta.json")
    return all(path.exists() for path in expected)


def backfill_layer_top1pct_visualizations(
    *,
    subset_dir: Path,
    viz_dir: Path,
    subset: str,
    params: dict[str, Any],
) -> bool:
    target_paths = expected_layer_top1pct_visualizations(viz_dir, subset)
    if all(path.exists() for path in target_paths):
        clean_legacy_layer_top_visualizations(viz_dir, subset)
        return False
    if not can_backfill_from_scores(subset_dir, params):
        return False

    clean_legacy_layer_top_visualizations(viz_dir, subset)
    layer_top1pct_paths: dict[str, str] = {}
    for task_type in TASK_TYPES:
        scores_payload = torch.load(subset_dir / task_type / "scar_scores.pt", map_location="cpu")
        out_path = viz_dir / f"layer_top1pct_scar_heatmap_{subset}_{task_type}.png"
        plot_layer_top_score_heatmap(
            score_pack=scores_payload["scores"],
            module_meta=scores_payload["module_meta"],
            out_path=out_path,
            score_field="scar",
            score_label="SCAR",
            title=f"{subset} Type {task_type}: top 1% SCAR by layer/module",
        )
        layer_top1pct_paths[task_type] = str(out_path)

    summary_path = subset_dir / "summary.json"
    manifest_path = subset_dir / "manifest.json"
    summary = read_json(summary_path) if summary_path.exists() else {"subset": subset}
    visualizations = summary.setdefault("visualizations", {})
    visualizations.pop("layer_top3pct_scar_heatmaps", None)
    visualizations.pop("layer_top10_scar_heatmaps", None)
    visualizations["layer_top1pct_scar_heatmaps"] = layer_top1pct_paths
    write_json(summary_path, summary)
    manifest = read_json(manifest_path)
    manifest["params"] = params
    manifest["summary"] = summary
    manifest.setdefault("visualizations", {})
    manifest["visualizations"].pop("layer_top3pct_scar_heatmaps", None)
    manifest["visualizations"].pop("layer_top10_scar_heatmaps", None)
    manifest["visualizations"].update(summary["visualizations"])
    write_json(manifest_path, manifest)
    print(f"Backfilled layer top-1% SCAR visualizations: {subset_dir}")
    return True


def should_skip(out_root: Path, viz_dir: Path, subset: str, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    subset_dir = out_root / subset
    if clean:
        clean_directory(subset_dir, data_root())
        clean_visualizations(viz_dir, subset)
        return False
    expected = [subset_dir / task_type / "TDN_neurons.jsonl" for task_type in TASK_TYPES]
    expected.extend(expected_visualizations(viz_dir, subset))
    manifest_path = subset_dir / "manifest.json"
    expected.append(manifest_path)
    if overwrite or not all(path.exists() for path in expected):
        return False
    manifest = read_json(manifest_path)
    if params_match_or_hardware_compatible(manifest.get("params", {}), params):
        print(f"Skip existing single-type neurons: {subset_dir}")
        return True
    return False


def save_counts_csv(rows: list[dict[str, Any]], path: Path, field: str) -> None:
    counts = Counter(row[field] for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field, "count"])
        for key, value in sorted(counts.items()):
            writer.writerow([key, value])


def main() -> None:
    args = parse_args()
    activation_root = resolve_path(args.activations_dir) if args.activations_dir else path_from_config("activations_dir")
    neurons_root = resolve_path(args.neurons_dir) if args.neurons_dir else path_from_config("neurons_dir")
    viz_root = resolve_path(args.visualizations_dir) if args.visualizations_dir else path_from_config("visualizations_dir")

    subsets = ["single_hop", "multi_hop"] if args.subset == "all" else [args.subset]
    model_out_root = neurons_root / args.model_alias / "single_type_by_subset"
    viz_dir = viz_root / args.model_alias / "single_type_by_subset"
    compute_devices = resolve_compute_devices(args)
    print(f"Stage 5 compute devices: {', '.join(str(device) for device in compute_devices)}")

    for subset in subsets:
        act_dir = activation_root / args.model_alias / subset / "train"
        activation_path = act_dir / "activations.pt"
        meta_path = act_dir / "meta.jsonl"
        activation_manifest_path = act_dir / "manifest.json"
        if not activation_path.exists() or not meta_path.exists() or not activation_manifest_path.exists():
            raise FileNotFoundError(f"Missing train activations for {subset}: {act_dir}")
        activation_manifest = read_json(activation_manifest_path)
        params = expected_params(
            args,
            activation_manifest=activation_manifest,
            activation_path=activation_path,
            meta_path=meta_path,
            subset=subset,
        )
        if should_skip(model_out_root, viz_dir, subset, params, args.overwrite, args.clean):
            continue
        if (not args.overwrite) and (not args.clean) and backfill_layer_top1pct_visualizations(
            subset_dir=model_out_root / subset,
            viz_dir=viz_dir,
            subset=subset,
            params=params,
        ):
            continue
        payload = torch.load(activation_path, map_location="cpu")
        activations: dict[str, torch.Tensor] = payload["activations"]
        module_meta = payload["module_meta"]
        meta_rows = read_jsonl(meta_path)

        rows_by_type: dict[str, list[dict[str, Any]]] = {}
        summary: dict[str, Any] = {"subset": subset, "task_types": {}}
        scar_heatmaps: dict[str, str] = {}
        layer_top1pct_scar_heatmaps: dict[str, str] = {}

        for task_type in TASK_TYPES:
            indices = [i for i, row in enumerate(meta_rows) if row["task_type"] == task_type]
            labels = torch.tensor([int(meta_rows[i]["tool_necessary"]) for i in indices], dtype=torch.long)
            n1 = int((labels == 1).sum())
            n0 = int((labels == 0).sum())
            if n1 < args.min_class_count or n0 < args.min_class_count:
                raise ValueError(
                    f"{subset}/type {task_type} needs at least {args.min_class_count} per class, got y1={n1}, y0={n0}"
                )
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            score_pack = compute_scar_for_modules(
                activations=activations,
                module_meta=module_meta,
                idx_tensor=idx_tensor,
                labels=labels,
                devices=compute_devices,
                eps=args.epsilon,
                desc=f"{subset}/type {task_type} SCAR",
            )
            rows = topk_rows(score_pack, module_meta, args.top_k)
            rows_by_type[task_type] = rows

            out_dir = model_out_root / subset / task_type
            ensure_dir(out_dir)
            write_jsonl(out_dir / "TDN_neurons.jsonl", rows)
            torch.save(
                {
                    "task_type": task_type,
                    "subset": subset,
                    "top_k": args.top_k,
                    "module_meta": module_meta,
                    "scores": {k: {name: tensor.to(torch.float32) for name, tensor in pack.items()} for k, pack in score_pack.items()},
                },
                out_dir / "scar_scores.pt",
            )
            save_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
            save_counts_csv(rows, out_dir / "module_counts.csv", "module")
            type_summary = {
                "n_total": len(indices),
                "n_tool_necessary_1": n1,
                "n_tool_necessary_0": n0,
                "top_k": len(rows),
                "top_layers": Counter(row["layer"] for row in rows).most_common(10),
                "top_modules": Counter(row["module"] for row in rows).most_common(),
            }
            summary["task_types"][task_type] = type_summary
            write_json(out_dir / "summary.json", type_summary)
            scar_heatmap_path = viz_dir / f"tdn_scar_heatmap_{subset}_{task_type}.png"
            plot_scar_heatmap(rows, scar_heatmap_path, args.heatmap_top_n)
            scar_heatmaps[task_type] = str(scar_heatmap_path)
            layer_top1pct_path = viz_dir / f"layer_top1pct_scar_heatmap_{subset}_{task_type}.png"
            plot_layer_top_score_heatmap(
                score_pack=score_pack,
                module_meta=module_meta,
                out_path=layer_top1pct_path,
                score_field="scar",
                score_label="SCAR",
                title=f"{subset} Type {task_type}: top 1% SCAR by layer/module",
            )
            layer_top1pct_scar_heatmaps[task_type] = str(layer_top1pct_path)
            print(f"{subset}/type {task_type}: wrote {len(rows)} neurons")

        heatmap_path = viz_dir / f"{subset}_heatmap.png"
        plot_heatmap(rows_by_type, module_meta, heatmap_path)
        write_json(model_out_root / subset / "module_meta.json", module_meta)
        summary["visualizations"] = {
            "density_heatmap": str(heatmap_path),
            "tdn_scar_heatmaps": scar_heatmaps,
            "layer_top1pct_scar_heatmaps": layer_top1pct_scar_heatmaps,
        }
        write_json(model_out_root / subset / "summary.json", summary)
        write_json(
            model_out_root / subset / "manifest.json",
            {
                "params": params,
                "summary": summary,
                "visualizations": summary["visualizations"],
            },
        )
        print(f"Wrote heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
