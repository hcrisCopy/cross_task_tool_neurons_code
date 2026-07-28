from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

COMMON_DIR = Path(__file__).resolve().parents[1] / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cttn.data import SUBSETS, TASK_TYPES
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import progress


STAGE_VERSION = 1
METHOD_NAME = "SafetyKernel_Union"
UNION_FILENAME = "CTD_Union_neurons.jsonl"
LAYER_TOP_SCORE_RATIO = 0.01
MODULE_ORDER = ["gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafetyKernel_Union stage 6: discover CTD-Union neurons.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--input-neurons-dir", default=None, help="Stage 5 Safety Kernel TDN root.")
    parser.add_argument("--output-neurons-dir", default=None, help="SafetyKernel_Union neuron output root.")
    parser.add_argument("--visualizations-dir", default=None, help="SafetyKernel_Union visualization output root.")
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_union_root(kind: str) -> Path:
    if kind not in {"neurons", "visualizations"}:
        raise KeyError(f"Unknown SafetyKernel_Union root kind: {kind}")
    return data_root() / "safety_kernel_union" / kind


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def neuron_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["layer"]), str(row["module"]), int(row["index"])


def single_type_root(input_neurons_root: Path, model_alias: str) -> Path:
    return input_neurons_root / model_alias / "single_type_by_subset"


def shared_root(output_neurons_root: Path, model_alias: str) -> Path:
    return output_neurons_root / model_alias / "shared_by_subset"


def single_type_viz_root(viz_root: Path, model_alias: str) -> Path:
    return viz_root / model_alias / "single_type_by_subset"


def read_tdn(root: Path, subset: str, task_type: str) -> dict[tuple[int, str, int], dict[str, Any]]:
    path = root / subset / task_type / "TDN_neurons.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty Safety Kernel TDN file: {path}")
    return {neuron_key(row): row for row in rows}


def rows_for_keys(
    keys: Iterable[tuple[int, str, int]],
    by_type: dict[str, dict[tuple[int, str, int], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for union_rank, (layer, module, index) in enumerate(sorted(keys), start=1):
        row: dict[str, Any] = {
            "layer": int(layer),
            "module": str(module),
            "index": int(index),
            "union_rank": union_rank,
        }
        scores: list[float] = []
        ranks: list[int] = []
        source_task_types: list[str] = []
        for task_type in TASK_TYPES:
            src = by_type.get(task_type, {}).get((layer, module, index))
            if not src:
                continue
            source_task_types.append(task_type)
            row[f"score_{task_type}"] = src.get("score")
            row[f"rank_{task_type}"] = src.get("rank")
            if src.get("module_key") is not None:
                row[f"module_key_{task_type}"] = src.get("module_key")
            if src.get("score") is not None:
                scores.append(float(src["score"]))
            if src.get("rank") is not None:
                ranks.append(int(src["rank"]))
        row["source_task_types"] = source_task_types
        row["membership_count"] = len(source_task_types)
        if scores:
            row["score_min"] = min(scores)
            row["score_mean"] = sum(scores) / len(scores)
            row["score_max"] = max(scores)
        if ranks:
            row["rank_min"] = min(ranks)
            row["rank_mean"] = sum(ranks) / len(ranks)
            row["rank_max"] = max(ranks)
        rows.append(row)
    return rows


def load_module_dims(single_root: Path, subset: str) -> dict[tuple[int, str], int]:
    meta_path = single_root / subset / "module_meta.json"
    if not meta_path.exists():
        return {}
    module_meta = read_json(meta_path)
    return {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}


def load_module_meta(single_root: Path, subset: str) -> list[dict[str, Any]]:
    meta_path = single_root / subset / "module_meta.json"
    if not meta_path.exists():
        return []
    return read_json(meta_path)


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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def plot_density(rows: list[dict[str, Any]], module_dims: dict[tuple[int, str], int], out_path: Path) -> None:
    if module_dims:
        layers = sorted({layer for layer, _module in module_dims})
    else:
        layers = sorted({int(row["layer"]) for row in rows})
    if not layers:
        write_empty_plot(out_path, "CTD-Union Neurons")
        return
    modules = ["gate_proj", "up_proj", "down_proj"]
    counts = Counter((int(row["layer"]), str(row["module"])) for row in rows)
    matrix = []
    for layer in layers:
        vals = []
        for module in modules:
            dim = max(module_dims.get((layer, module), 1), 1)
            vals.append(counts.get((layer, module), 0) / dim)
        matrix.append(vals)
    fig, ax = plt.subplots(figsize=(4.8, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("CTD-Union Neurons")
    ax.set_xticks(range(len(modules)), modules, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xlabel("FFN module")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scar_heatmap(rows: list[dict[str, Any]], out_path: Path, score_field: str, top_n: int) -> None:
    scored_rows = [row for row in rows if row.get(score_field) is not None]
    scored_rows.sort(key=lambda row: float(row[score_field]), reverse=True)
    selected = scored_rows[: max(1, min(top_n, len(scored_rows)))]
    if not selected:
        write_empty_plot(out_path, f"CTD-Union SCAR ({score_field})")
        return
    group_order = sorted({(int(row["layer"]), str(row["module"])) for row in selected})
    group_to_y = {group: i for i, group in enumerate(group_order)}
    matrix = [[float("nan") for _ in selected] for _ in group_order]
    for x, row in enumerate(selected):
        y = group_to_y[(int(row["layer"]), str(row["module"]))]
        matrix[y][x] = float(row[score_field])

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(f"Top-{len(selected)} CTD-Union SCAR neurons ({score_field})")
    ax.set_xlabel("CTD-Union rank")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{module}" for layer, module in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label=score_field)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def module_groups(module_dims: dict[tuple[int, str], int], rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    if module_dims:
        groups = list(module_dims)
    else:
        groups = [(int(row["layer"]), str(row["module"])) for row in rows]
    module_order = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    return sorted(set(groups), key=lambda item: (item[0], module_order.get(item[1], 99), item[1]))


def plot_layer_top_union_score_heatmap(
    rows: list[dict[str, Any]],
    module_dims: dict[tuple[int, str], int],
    out_path: Path,
    *,
    score_field: str,
    score_label: str,
    title: str,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    groups = module_groups(module_dims, rows)
    row_values: list[list[float]] = []
    row_labels: list[str] = []
    max_cols = 0
    for layer, module in groups:
        candidates = [
            float(row[score_field])
            for row in rows
            if int(row["layer"]) == layer and str(row["module"]) == module and row.get(score_field) is not None
        ]
        dim = max(module_dims.get((layer, module), len(candidates)), 1)
        k = max(1, int(dim * ratio))
        candidates.sort(reverse=True)
        values = candidates[: min(k, len(candidates))]
        row_values.append(values)
        row_labels.append(f"L{layer}.{module}")
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
    fig_height = max(6, len(row_labels) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(f"Neuron rank within top {int(ratio * 100)}% of each layer/module")
    ax.set_ylabel("Layer / FFN module")
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


def plot_single_type_layer_top_scar_heatmap(
    *,
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    title: str,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    module_order = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    ordered_meta = sorted(
        module_meta,
        key=lambda meta: (int(meta["layer"]), module_order.get(str(meta["module"]), 99), str(meta.get("key", ""))),
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
        write_empty_plot(out_path, title)
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
    if ticks and ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="SCAR")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_single_type_layer_top1pct_visualizations(
    single_root: Path,
    single_viz_dir: Path,
    subset: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    module_meta = load_module_meta(single_root, subset)
    for task_type in TASK_TYPES:
        scores_path = single_root / subset / task_type / "scar_scores.pt"
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing Safety Kernel SCAR scores for SKU single-type plot: {scores_path}")
        payload = torch.load(scores_path, map_location="cpu")
        payload_meta = payload.get("module_meta") or module_meta
        score_pack = payload.get("scores")
        if not isinstance(score_pack, dict):
            raise ValueError(f"Invalid Safety Kernel SCAR score payload: {scores_path}")
        out_path = single_viz_dir / f"sku_layer_top1pct_scar_heatmap_{subset}_{task_type}.png"
        plot_single_type_layer_top_scar_heatmap(
            score_pack=score_pack,
            module_meta=payload_meta,
            out_path=out_path,
            title=f"{subset} SKU Type {task_type}: top 1% SCAR by layer/module",
        )
        paths[task_type] = str(out_path)
    return paths


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"ctd_union_density_heatmap_{subset}.png",
        viz_dir / f"ctd_union_scar_min_heatmap_{subset}.png",
        viz_dir / f"ctd_union_scar_mean_heatmap_{subset}.png",
        viz_dir / f"ctd_union_layer_top1pct_scar_heatmap_{subset}.png",
    ]


def expected_single_type_visualizations(single_viz_dir: Path, subset: str) -> list[Path]:
    return [single_viz_dir / f"sku_layer_top1pct_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]


def legacy_visualizations(viz_dir: Path, single_viz_dir: Path, subset: str) -> list[Path]:
    shared_names = [
        f"ctd_union_layer_top1pct_scar_min_heatmap_{subset}.png",
        f"ctd_union_layer_top1pct_scar_mean_heatmap_{subset}.png",
        *[f"ctd_union_layer_top1pct_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES],
        f"ctd_union_layer_top3pct_scar_min_heatmap_{subset}.png",
        f"ctd_union_layer_top3pct_scar_mean_heatmap_{subset}.png",
        f"ctd_union_layer_top10_scar_min_heatmap_{subset}.png",
        f"ctd_union_layer_top10_scar_mean_heatmap_{subset}.png",
    ]
    single_names = [
        f"sku_layer_top3pct_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES
    ] + [f"sku_layer_top10_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES]
    return [viz_dir / name for name in shared_names] + [single_viz_dir / name for name in single_names]


def clean_visualizations(viz_dir: Path, single_viz_dir: Path, subset: str) -> None:
    root = data_root().resolve()
    for path in [
        *expected_visualizations(viz_dir, subset),
        *expected_single_type_visualizations(single_viz_dir, subset),
        *legacy_visualizations(viz_dir, single_viz_dir, subset),
    ]:
        if path.exists():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Refusing to remove visualization outside data root: {resolved}")
            path.unlink()


def clean_legacy_visualizations(viz_dir: Path, single_viz_dir: Path, subset: str) -> None:
    root = data_root().resolve()
    for path in legacy_visualizations(viz_dir, single_viz_dir, subset):
        if path.exists():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Refusing to remove visualization outside data root: {resolved}")
            path.unlink()


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    input_neurons_root: Path,
    output_neurons_root: Path,
    viz_root: Path,
    single_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "safety_kernel_union_06_shared_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "input_neurons_dir": str(input_neurons_root),
        "output_neurons_dir": str(output_neurons_root),
        "visualizations_dir": str(viz_root),
        "heatmap_top_n": args.heatmap_top_n,
        "single_type_manifest_params": single_manifest.get("params", {}),
        "selection": "CTD_Union = TDN_A union TDN_B union TDN_C",
    }


def expected_outputs(out_dir: Path, viz_dir: Path, single_viz_dir: Path, subset: str) -> list[Path]:
    return [
        out_dir / UNION_FILENAME,
        out_dir / "pairwise_AB_neurons.jsonl",
        out_dir / "pairwise_AC_neurons.jsonl",
        out_dir / "pairwise_BC_neurons.jsonl",
        out_dir / "triple_intersection_neurons.jsonl",
        out_dir / "exclusive_A_neurons.jsonl",
        out_dir / "exclusive_B_neurons.jsonl",
        out_dir / "exclusive_C_neurons.jsonl",
        out_dir / "layer_counts.csv",
        out_dir / "module_counts.csv",
        out_dir / "share_rates.csv",
        out_dir / "summary.json",
        out_dir / "manifest.json",
        *expected_visualizations(viz_dir, subset),
        *expected_single_type_visualizations(single_viz_dir, subset),
    ]


def should_skip(
    out_dir: Path,
    viz_dir: Path,
    single_viz_dir: Path,
    subset: str,
    params: dict[str, Any],
    *,
    overwrite: bool,
    clean: bool,
) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        clean_visualizations(viz_dir, single_viz_dir, subset)
        return False
    expected = expected_outputs(out_dir, viz_dir, single_viz_dir, subset)
    if overwrite or not all(path.exists() for path in expected):
        return False
    manifest = read_json(out_dir / "manifest.json")
    if manifest.get("params") == params:
        print(f"Skip existing CTD-Union neurons: {out_dir}")
        return True
    return False


def backfill_visualizations(
    out_dir: Path,
    viz_dir: Path,
    single_viz_dir: Path,
    single_root: Path,
    subset: str,
    params: dict[str, Any],
    heatmap_top_n: int,
) -> bool:
    manifest_path = out_dir / "manifest.json"
    union_path = out_dir / UNION_FILENAME
    if not manifest_path.exists() or not union_path.exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") != params:
        return False

    clean_legacy_visualizations(viz_dir, single_viz_dir, subset)
    targets = [*expected_visualizations(viz_dir, subset), *expected_single_type_visualizations(single_viz_dir, subset)]
    if all(path.exists() for path in targets):
        return False

    union_rows = read_jsonl(union_path)
    module_dims = load_module_dims(single_root, subset)

    density_path = viz_dir / f"ctd_union_density_heatmap_{subset}.png"
    score_min_path = viz_dir / f"ctd_union_scar_min_heatmap_{subset}.png"
    score_mean_path = viz_dir / f"ctd_union_scar_mean_heatmap_{subset}.png"
    plot_density(union_rows, module_dims, density_path)
    plot_scar_heatmap(union_rows, score_min_path, "score_min", heatmap_top_n)
    plot_scar_heatmap(union_rows, score_mean_path, "score_mean", heatmap_top_n)
    layer_top1pct_path = viz_dir / f"ctd_union_layer_top1pct_scar_heatmap_{subset}.png"
    plot_layer_top_union_score_heatmap(
        union_rows,
        module_dims,
        layer_top1pct_path,
        score_field="score_max",
        score_label="Union SCAR = max(available score_A, score_B, score_C)",
        title=f"{subset} CTD-Union: top 1% union SCAR by layer/module",
    )
    single_top_paths = plot_single_type_layer_top1pct_visualizations(single_root, single_viz_dir, subset)

    summary = read_json(out_dir / "summary.json") if (out_dir / "summary.json").exists() else manifest.get("summary", {})
    visualizations = summary.setdefault("visualizations", {})
    visualizations.pop("layer_top1pct_score_min_heatmap", None)
    visualizations.pop("layer_top1pct_score_mean_heatmap", None)
    visualizations.pop("layer_top1pct_by_type_heatmaps", None)
    visualizations.update(
        {
            "density_heatmap": str(density_path),
            "score_min_heatmap": str(score_min_path),
            "score_mean_heatmap": str(score_mean_path),
            "layer_top1pct_union_score_heatmap": str(layer_top1pct_path),
            "single_type_layer_top1pct_scar_heatmaps": single_top_paths,
        }
    )
    manifest["summary"] = summary
    write_json(out_dir / "summary.json", summary)
    write_json(manifest_path, manifest)
    print(f"Backfilled SafetyKernel_Union top-1% visualizations: {out_dir}")
    return True


def source_membership_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["membership_count"]) for row in rows)
    return {str(key): int(counts.get(key, 0)) for key in [1, 2, 3]}


def main() -> None:
    args = parse_args()
    input_neurons_root = resolve_path(args.input_neurons_dir) if args.input_neurons_dir else path_from_config("neurons_dir")
    output_neurons_root = (
        resolve_path(args.output_neurons_dir) if args.output_neurons_dir else default_union_root("neurons")
    )
    viz_root = resolve_path(args.visualizations_dir) if args.visualizations_dir else default_union_root("visualizations")

    single_root = single_type_root(input_neurons_root, args.model_alias)
    shared_dir = shared_root(output_neurons_root, args.model_alias)
    viz_dir = viz_root / args.model_alias / "shared_by_subset"
    single_viz_dir = single_type_viz_root(viz_root, args.model_alias)
    root_manifest: dict[str, Any] = {
        "stage": "safety_kernel_union_06_shared_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    global_share_rows: list[dict[str, Any]] = []

    for subset in progress(subset_values(args.subset), desc=f"SKU-6 {args.model_alias}", unit="subset"):
        subset_out = shared_dir / subset
        single_manifest_path = single_root / subset / "manifest.json"
        if not single_manifest_path.exists():
            raise FileNotFoundError(f"Missing Safety Kernel single-type manifest: {single_manifest_path}")
        params = expected_params(
            args,
            subset=subset,
            input_neurons_root=input_neurons_root,
            output_neurons_root=output_neurons_root,
            viz_root=viz_root,
            single_manifest=read_json(single_manifest_path),
        )
        if (
            not args.overwrite
            and not args.clean
            and backfill_visualizations(
                subset_out,
                viz_dir,
                single_viz_dir,
                single_root,
                subset,
                params,
                args.heatmap_top_n,
            )
            and should_skip(
                subset_out,
                viz_dir,
                single_viz_dir,
                subset,
                params,
                overwrite=args.overwrite,
                clean=args.clean,
            )
        ):
            root_manifest["subsets"][subset] = read_json(subset_out / "summary.json")
            global_share_rows.extend(read_csv_rows(subset_out / "share_rates.csv"))
            continue
        if should_skip(
            subset_out,
            viz_dir,
            single_viz_dir,
            subset,
            params,
            overwrite=args.overwrite,
            clean=args.clean,
        ):
            root_manifest["subsets"][subset] = read_json(subset_out / "summary.json")
            global_share_rows.extend(read_csv_rows(subset_out / "share_rates.csv"))
            continue

        ensure_dir(subset_out)
        by_type = {task_type: read_tdn(single_root, subset, task_type) for task_type in TASK_TYPES}
        sets = {task_type: set(rows.keys()) for task_type, rows in by_type.items()}
        union_keys = set.union(*(sets[task_type] for task_type in TASK_TYPES))
        triple = set.intersection(*(sets[task_type] for task_type in TASK_TYPES))
        pairwise = {
            "AB": sets["A"] & sets["B"],
            "AC": sets["A"] & sets["C"],
            "BC": sets["B"] & sets["C"],
        }
        exclusive = {
            task_type: sets[task_type] - set.union(*(sets[other] for other in TASK_TYPES if other != task_type))
            for task_type in TASK_TYPES
        }

        union_rows = rows_for_keys(union_keys, by_type)
        write_jsonl(subset_out / UNION_FILENAME, union_rows)
        for name, keys in pairwise.items():
            write_jsonl(subset_out / f"pairwise_{name}_neurons.jsonl", rows_for_keys(keys, by_type))
        write_jsonl(subset_out / "triple_intersection_neurons.jsonl", rows_for_keys(triple, by_type))
        for task_type, keys in exclusive.items():
            write_jsonl(subset_out / f"exclusive_{task_type}_neurons.jsonl", rows_for_keys(keys, by_type))

        write_counts_csv(union_rows, subset_out / "layer_counts.csv", "layer")
        write_counts_csv(union_rows, subset_out / "module_counts.csv", "module")

        union_count = len(union_keys)
        share_rows = []
        for task_type in TASK_TYPES:
            tdn_count = len(sets[task_type])
            other_union = set.union(*(sets[other] for other in TASK_TYPES if other != task_type))
            shared_with_other = len(sets[task_type] & other_union)
            share_rows.append(
                {
                    "model_alias": args.model_alias,
                    "subset": subset,
                    "task_type": task_type,
                    "tdn_count": tdn_count,
                    "ctd_union_count": union_count,
                    "type_share_of_union": tdn_count / max(union_count, 1),
                    "overlap_with_other_types": shared_with_other / max(tdn_count, 1),
                }
            )
        write_csv_rows(subset_out / "share_rates.csv", share_rows)
        global_share_rows.extend(share_rows)

        module_dims = load_module_dims(single_root, subset)
        density_path = viz_dir / f"ctd_union_density_heatmap_{subset}.png"
        score_min_path = viz_dir / f"ctd_union_scar_min_heatmap_{subset}.png"
        score_mean_path = viz_dir / f"ctd_union_scar_mean_heatmap_{subset}.png"
        plot_density(union_rows, module_dims, density_path)
        plot_scar_heatmap(union_rows, score_min_path, "score_min", args.heatmap_top_n)
        plot_scar_heatmap(union_rows, score_mean_path, "score_mean", args.heatmap_top_n)
        layer_top1pct_path = viz_dir / f"ctd_union_layer_top1pct_scar_heatmap_{subset}.png"
        plot_layer_top_union_score_heatmap(
            union_rows,
            module_dims,
            layer_top1pct_path,
            score_field="score_max",
            score_label="Union SCAR = max(available score_A, score_B, score_C)",
            title=f"{subset} CTD-Union: top 1% union SCAR by layer/module",
        )
        single_top_paths = plot_single_type_layer_top1pct_visualizations(single_root, single_viz_dir, subset)

        summary = {
            "tdn_counts": {task_type: len(sets[task_type]) for task_type in TASK_TYPES},
            "ctd_union_count": union_count,
            "triple_intersection_count": len(triple),
            "pairwise_counts": {name: len(keys) for name, keys in pairwise.items()},
            "exclusive_counts": {task_type: len(keys) for task_type, keys in exclusive.items()},
            "membership_counts": source_membership_counts(union_rows),
            "share_rates": share_rows,
            "top_layers": Counter(row["layer"] for row in union_rows).most_common(10),
            "top_modules": Counter(row["module"] for row in union_rows).most_common(),
            "visualizations": {
                "density_heatmap": str(density_path),
                "score_min_heatmap": str(score_min_path),
                "score_mean_heatmap": str(score_mean_path),
                "layer_top1pct_union_score_heatmap": str(layer_top1pct_path),
                "single_type_layer_top1pct_scar_heatmaps": single_top_paths,
            },
        }
        write_json(subset_out / "summary.json", summary)
        write_json(subset_out / "manifest.json", {"params": params, "summary": summary})
        root_manifest["subsets"][subset] = summary
        print(
            f"{subset}: CTD_Union={union_count}, triple={len(triple)}, "
            f"pairwise={summary['pairwise_counts']}, membership={summary['membership_counts']}",
            flush=True,
        )

    ensure_dir(shared_dir)
    write_csv_rows(
        shared_dir / "shared_summary.csv",
        global_share_rows,
        fieldnames=[
            "model_alias",
            "subset",
            "task_type",
            "tdn_count",
            "ctd_union_count",
            "type_share_of_union",
            "overlap_with_other_types",
        ],
    )
    write_json(shared_dir / "manifest.json", root_manifest)
    print(f"Wrote SafetyKernel_Union shared manifest: {shared_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
