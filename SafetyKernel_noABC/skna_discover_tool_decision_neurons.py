from __future__ import annotations

import argparse
import csv
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
METHOD_NAME = "SafetyKernel_noABC"
NOABC_FILENAME = "SK_noABC_TDN_neurons.jsonl"
LAYER_TOP_SCORE_RATIO = 0.01
MODULE_ORDER = ["gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SafetyKernel_noABC: discover global tool-decision neurons with Safety Kernel "
            "SCAR scoring over tool_necessary=1 vs tool_necessary=0, without A/B/C splitting."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="SafetyKernel_noABC SKNA-4 activation root; defaults to ../cross_task_tool_neurons_data/safety_kernel_noabc/activations.",
    )
    parser.add_argument(
        "--neurons-dir",
        default=None,
        help="SafetyKernel_noABC neuron output root; defaults to ../cross_task_tool_neurons_data/safety_kernel_noabc/neurons.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help="SafetyKernel_noABC visualization root; defaults to ../cross_task_tool_neurons_data/safety_kernel_noabc/visualizations.",
    )
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
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
        "activations": data_root() / "safety_kernel_noabc" / "activations",
        "neurons": data_root() / "safety_kernel_noabc" / "neurons",
        "visualizations": data_root() / "safety_kernel_noabc" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown SafetyKernel_noABC root kind: {kind}")
    return mapping[kind]


def resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def shared_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "shared_by_subset"


def resolve_compute_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    return device


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
    labels: torch.Tensor,
    device: torch.device,
    eps: float,
) -> tuple[str, dict[str, torch.Tensor]]:
    if device.type == "cuda":
        with torch.cuda.device(device):
            x = activations[key].to(device, non_blocking=True)
            scores = compute_scar_for_module(x, labels.to(device, non_blocking=True), eps)
            out = {name: tensor.detach().cpu() for name, tensor in scores.items()}
            del x
            torch.cuda.empty_cache()
            return key, out

    x = activations[key].to(device)
    scores = compute_scar_for_module(x, labels.to(device), eps)
    return key, {name: tensor.detach().cpu() for name, tensor in scores.items()}


def compute_scar_for_modules(
    *,
    activations: dict[str, torch.Tensor],
    module_meta: list[dict[str, Any]],
    labels: torch.Tensor,
    device: torch.device,
    eps: float,
    desc: str,
) -> dict[str, dict[str, torch.Tensor]]:
    score_pack: dict[str, dict[str, torch.Tensor]] = {}
    for meta in progress(module_meta, desc=desc, unit="module"):
        key, scores = compute_one_module(
            key=str(meta["key"]),
            activations=activations,
            labels=labels,
            device=device,
            eps=eps,
        )
        score_pack[key] = scores
    return score_pack


def topk_rows(
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    if k <= 0:
        raise ValueError("--top-k must be positive")
    candidates: list[dict[str, Any]] = []
    meta_by_key = {str(meta["key"]): meta for meta in module_meta}
    for key, scores in score_pack.items():
        scar = scores["scar"]
        kk = min(k, scar.numel())
        vals, idxs = torch.topk(scar, kk)
        meta = meta_by_key[key]
        for rank_in_module, (score, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
            index = int(idx)
            candidates.append(
                {
                    "layer": int(meta["layer"]),
                    "module": str(meta["module"]),
                    "module_key": key,
                    "index": index,
                    "score": float(score),
                    "discriminability": float(scores["discriminability"][index]),
                    "responsiveness": float(scores["responsiveness"][index]),
                    "delta": float(scores["delta"][index]),
                    "mu1": float(scores["mu1"][index]),
                    "mu0": float(scores["mu0"][index]),
                    "rank_in_module": rank_in_module,
                    "selection_scope": "all_task_types_noABC",
                }
            )
    candidates.sort(key=lambda row: (-float(row["score"]), int(row["layer"]), str(row["module"]), int(row["index"])))
    rows = candidates[:k]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


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


def validate_labels(meta_rows: list[dict[str, Any]], min_class_count: int) -> dict[str, int]:
    n1 = sum(1 for row in meta_rows if int(row["tool_necessary"]) == 1)
    n0 = sum(1 for row in meta_rows if int(row["tool_necessary"]) == 0)
    if n1 < min_class_count or n0 < min_class_count:
        raise ValueError(
            f"Need at least {min_class_count} train examples per label, "
            f"got tool_necessary=1: {n1}, tool_necessary=0: {n0}"
        )
    return {"n_tool_necessary": n1, "n_tool_unnecessary": n0}


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


def module_dims(module_meta: list[dict[str, Any]]) -> dict[tuple[int, str], int]:
    return {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}


def module_groups(dims: dict[tuple[int, str], int], rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    groups = list(dims) if dims else [(int(row["layer"]), str(row["module"])) for row in rows]
    module_order = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    return sorted(set(groups), key=lambda item: (item[0], module_order.get(item[1], 99), item[1]))


def plot_density(rows: list[dict[str, Any]], module_meta: list[dict[str, Any]], out_path: Path) -> None:
    dims = module_dims(module_meta)
    groups = module_groups(dims, rows)
    if not groups:
        write_empty_plot(out_path, "SafetyKernel_noABC Neurons")
        return
    layers = sorted({layer for layer, _module in groups})
    counts = Counter((int(row["layer"]), str(row["module"])) for row in rows)
    matrix = []
    for layer in layers:
        vals = []
        for module in MODULE_ORDER:
            dim = max(dims.get((layer, module), 1), 1)
            vals.append(counts.get((layer, module), 0) / dim)
        matrix.append(vals)
    fig, ax = plt.subplots(figsize=(4.8, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("SafetyKernel_noABC Neurons")
    ax.set_xticks(range(len(MODULE_ORDER)), MODULE_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN module")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scar_heatmap(rows: list[dict[str, Any]], out_path: Path, top_n: int) -> None:
    selected = rows[: max(1, min(top_n, len(rows)))]
    if not selected:
        write_empty_plot(out_path, "SafetyKernel_noABC SCAR")
        return
    group_order = sorted({(int(row["layer"]), str(row["module"])) for row in selected})
    y_by_group = {group: idx for idx, group in enumerate(group_order)}
    matrix = torch.full((len(group_order), len(selected)), float("nan"), dtype=torch.float32)
    for x, row in enumerate(selected):
        y = y_by_group[(int(row["layer"]), str(row["module"]))]
        matrix[y, x] = float(row["score"])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(f"Top-{len(selected)} SafetyKernel_noABC SCAR neurons")
    ax.set_xlabel("Global neuron rank")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{module}" for layer, module in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="SCAR")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_scar_heatmap(
    *,
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    module_order = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    ordered_meta = sorted(
        module_meta,
        key=lambda meta: (int(meta["layer"]), module_order.get(str(meta["module"]), 99), str(meta["key"])),
    )
    row_values: list[torch.Tensor] = []
    row_labels: list[str] = []
    max_cols = 0
    for meta in ordered_meta:
        key = str(meta["key"])
        scores = score_pack.get(key, {}).get("scar")
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
        write_empty_plot(out_path, "SafetyKernel_noABC top 1% SCAR by layer/module")
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
    ax.set_title("SafetyKernel_noABC: top 1% SCAR by layer/module")
    ax.set_xlabel(f"Neuron rank within top {int(ratio * 100)}% of each layer/module")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, max_cols, max(1, max_cols // 10)))
    if ticks and ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="SCAR")
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
) -> dict[str, Any]:
    return {
        "stage": "skna_05_noabc_tool_decision_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "activations_dir": str(activation_dir),
        "neurons_dir": str(neurons_root),
        "visualizations_dir": str(viz_root),
        "top_k": args.top_k,
        "heatmap_top_n": args.heatmap_top_n,
        "epsilon": args.epsilon,
        "min_class_count": args.min_class_count,
        "activation_manifest_params": activation_manifest.get("params", {}),
        "activation_definition": "last_input_token_ffn_module_output",
        "score_definition": "SCAR=zscore((mu_tool_necessary-mu_tool_unnecessary)/pooled_std)+zscore(mu_tool_necessary-mu_tool_unnecessary)",
        "selection": "global TopK SCAR over train tool_necessary=1 vs tool_necessary=0, no A/B/C split",
        "neuron_identity": "(layer, module, index) over FFN output modules gate_proj/up_proj/down_proj",
    }


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"sk_noabc_density_heatmap_{subset}.png",
        viz_dir / f"sk_noabc_scar_heatmap_{subset}.png",
        viz_dir / f"sk_noabc_layer_top1pct_scar_heatmap_{subset}.png",
    ]


def expected_outputs(out_dir: Path, viz_dir: Path, subset: str) -> list[Path]:
    return [
        out_dir / NOABC_FILENAME,
        out_dir / "scar_scores.pt",
        out_dir / "module_meta.json",
        out_dir / "layer_counts.csv",
        out_dir / "module_counts.csv",
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
        print(f"Skip existing SafetyKernel_noABC neurons: {out_dir}", flush=True)
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
        raise FileNotFoundError(f"Missing SafetyKernel_noABC train activations for {subset}: {act_dir}")

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

    meta_rows = read_jsonl(meta_path)
    label_counts = validate_labels(meta_rows, args.min_class_count)
    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations: dict[str, torch.Tensor] = payload["activations"]
    module_meta: list[dict[str, Any]] = payload["module_meta"]
    labels = torch.tensor([int(row["tool_necessary"]) for row in meta_rows], dtype=torch.long)
    score_pack = compute_scar_for_modules(
        activations=activations,
        module_meta=module_meta,
        labels=labels,
        device=device,
        eps=args.epsilon,
        desc=f"{subset} noABC SCAR",
    )
    rows = topk_rows(score_pack, module_meta, args.top_k)

    ensure_dir(out_dir)
    write_jsonl(out_dir / NOABC_FILENAME, rows)
    torch.save(
        {
            "method": METHOD_NAME,
            "subset": subset,
            "top_k": args.top_k,
            "module_meta": module_meta,
            "scores": {k: {name: tensor.to(torch.float32) for name, tensor in pack.items()} for k, pack in score_pack.items()},
            "score_definition": "Safety Kernel SCAR over all task types, tool_necessary=1 vs 0",
        },
        out_dir / "scar_scores.pt",
    )
    write_json(out_dir / "module_meta.json", module_meta)
    write_counts_csv(rows, out_dir / "layer_counts.csv", "layer")
    write_counts_csv(rows, out_dir / "module_counts.csv", "module")
    class_rows = class_balance_rows(meta_rows, args.model_alias, subset)
    write_csv_rows(
        out_dir / "class_balance.csv",
        class_rows,
        fieldnames=["model_alias", "subset", "task_type", "tool_necessary", "count"],
    )

    density_path = viz_dir / f"sk_noabc_density_heatmap_{subset}.png"
    scar_path = viz_dir / f"sk_noabc_scar_heatmap_{subset}.png"
    layer_top1pct_path = viz_dir / f"sk_noabc_layer_top1pct_scar_heatmap_{subset}.png"
    plot_density(rows, module_meta, density_path)
    plot_scar_heatmap(rows, scar_path, args.heatmap_top_n)
    plot_layer_top_scar_heatmap(score_pack=score_pack, module_meta=module_meta, out_path=layer_top1pct_path)

    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_file": str(out_dir / NOABC_FILENAME),
        "neuron_set": "SK_noABC_TDN",
        "selected_neurons": len(rows),
        "top_k": args.top_k,
        "class_balance": class_rows,
        "top_layers": Counter(int(row["layer"]) for row in rows).most_common(10),
        "top_modules": Counter(str(row["module"]) for row in rows).most_common(),
        "n_tool_necessary": label_counts["n_tool_necessary"],
        "n_tool_unnecessary": label_counts["n_tool_unnecessary"],
        "visualizations": {
            "density_heatmap": str(density_path),
            "scar_heatmap": str(scar_path),
            "layer_top1pct_scar_heatmap": str(layer_top1pct_path),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    print(
        f"{subset}: SK_noABC_TDN={len(rows)}, "
        f"tool_necessary=1/{label_counts['n_tool_necessary']}, "
        f"tool_necessary=0/{label_counts['n_tool_unnecessary']}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    activation_root = resolve_root(args.activations_dir, "activations")
    neurons_root = resolve_root(args.neurons_dir, "neurons")
    viz_root = resolve_root(args.visualizations_dir, "visualizations")
    device = resolve_compute_device(args.device)
    print(f"SafetyKernel_noABC compute device: {device}", flush=True)
    print(f"SafetyKernel_noABC subset order = {' -> '.join(subset_values(args.subset))}", flush=True)

    root_manifest: dict[str, Any] = {
        "stage": "skna_05_noabc_tool_decision_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for subset in progress(subset_values(args.subset), desc=f"SKNA-5 {args.model_alias}", unit="subset"):
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
                "neuron_set": "SK_noABC_TDN",
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
    print(f"Wrote SafetyKernel_noABC manifest: {model_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
