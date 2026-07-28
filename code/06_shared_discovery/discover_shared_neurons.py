from __future__ import annotations

import argparse
import csv
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

from cttn.data import TASK_TYPES
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path


LAYER_TOP_SCORE_RATIO = 0.01
MODULE_ORDER = ["gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: discover CTD shared neurons by A/B/C intersection.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--visualizations-dir", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def neuron_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["layer"]), str(row["module"]), int(row["index"])


def read_tdn(root: Path, subset: str, task_type: str) -> dict[tuple[int, str, int], dict[str, Any]]:
    path = root / subset / task_type / "TDN_neurons.jsonl"
    rows = read_jsonl(path)
    return {neuron_key(row): row for row in rows}


def rows_for_keys(
    keys: set[tuple[int, str, int]],
    by_type: dict[str, dict[tuple[int, str, int], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for layer, module, index in sorted(keys):
        row: dict[str, Any] = {"layer": layer, "module": module, "index": index}
        scores = []
        ranks = []
        for task_type in TASK_TYPES:
            src = by_type.get(task_type, {}).get((layer, module, index))
            if src:
                score = src.get("score")
                rank = src.get("rank")
                row[f"score_{task_type}"] = score
                row[f"rank_{task_type}"] = rank
                if score is not None:
                    scores.append(float(score))
                if rank is not None:
                    ranks.append(int(rank))
        if scores:
            row["score_min"] = min(scores)
            row["score_mean"] = sum(scores) / len(scores)
        if ranks:
            row["rank_max"] = max(ranks)
            row["rank_mean"] = sum(ranks) / len(ranks)
        rows.append(row)
    return rows


def write_counts_csv(rows: list[dict[str, Any]], path: Path, field: str) -> None:
    counts = Counter(row[field] for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field, "count"])
        for key, count in sorted(counts.items()):
            writer.writerow([key, count])


def write_empty_plot(out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title(title)
    ax.text(0.5, 0.5, "No neurons", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, Any]], module_dims: dict[tuple[int, str], int], out_path: Path) -> None:
    if module_dims:
        layers = sorted({layer for layer, _ in module_dims})
    else:
        layers = sorted({int(row["layer"]) for row in rows})
    if not layers:
        write_empty_plot(out_path, "CTD Shared Neurons")
        return
    modules = MODULE_ORDER
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
    ax.set_title("CTD Shared Neurons")
    ax.set_xticks(range(len(modules)), modules, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xlabel("FFN module")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ctd_scar_heatmap(rows: list[dict[str, Any]], out_path: Path, score_field: str, top_n: int) -> None:
    scored_rows = [row for row in rows if row.get(score_field) is not None]
    scored_rows.sort(key=lambda row: float(row[score_field]), reverse=True)
    selected = scored_rows[: max(1, min(top_n, len(scored_rows)))]
    if not selected:
        write_empty_plot(out_path, f"CTD-SCAR ({score_field})")
        return
    group_order = sorted({(int(row["layer"]), str(row["module"])) for row in selected})
    group_to_y = {group: i for i, group in enumerate(group_order)}
    import math

    matrix = [[math.nan for _ in selected] for _ in group_order]
    for x, row in enumerate(selected):
        y = group_to_y[(int(row["layer"]), str(row["module"]))]
        matrix[y][x] = float(row[score_field])

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(f"Top-{len(selected)} CTD-SCAR neurons ({score_field})")
    ax.set_xlabel("CTD rank")
    ax.set_ylabel("Layer / FFN module")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{module}" for layer, module in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label=score_field)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def module_groups(module_dims: dict[tuple[int, str], int], rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    if module_dims:
        groups = list(module_dims)
    else:
        groups = [(int(row["layer"]), str(row["module"])) for row in rows]
    module_order = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    return sorted(set(groups), key=lambda item: (item[0], module_order.get(item[1], 99), item[1]))


def plot_layer_top_shared_score_heatmap(
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_module_dims(single_root: Path, subset: str) -> dict[tuple[int, str], int]:
    meta_path = single_root / subset / "module_meta.json"
    if not meta_path.exists():
        return {}
    module_meta = read_json(meta_path)
    return {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"shared_neuron_heatmap_{subset}.png",
        viz_dir / f"ctd_scar_min_heatmap_{subset}.png",
        viz_dir / f"ctd_scar_mean_heatmap_{subset}.png",
        viz_dir / f"ctd_layer_top1pct_scar_heatmap_{subset}.png",
    ]


def legacy_layer_top_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"ctd_layer_top1pct_scar_min_heatmap_{subset}.png",
        viz_dir / f"ctd_layer_top1pct_scar_mean_heatmap_{subset}.png",
        *[viz_dir / f"ctd_layer_top1pct_scar_heatmap_{subset}_{task_type}.png" for task_type in TASK_TYPES],
        viz_dir / f"ctd_layer_top3pct_scar_min_heatmap_{subset}.png",
        viz_dir / f"ctd_layer_top3pct_scar_mean_heatmap_{subset}.png",
        viz_dir / f"ctd_layer_top10_scar_min_heatmap_{subset}.png",
        viz_dir / f"ctd_layer_top10_scar_mean_heatmap_{subset}.png",
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


def expected_params(args: argparse.Namespace, *, single_manifest: dict[str, Any], subset: str) -> dict[str, Any]:
    return {
        "stage": "06_shared_discovery",
        "model_alias": args.model_alias,
        "subset": subset,
        "heatmap_top_n": args.heatmap_top_n,
        "single_type_manifest_params": single_manifest.get("params", {}),
    }


def should_skip(out_dir: Path, viz_dir: Path, subset: str, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        clean_visualizations(viz_dir, subset)
        return False
    manifest_path = out_dir / "manifest.json"
    expected = [out_dir / "CTD_neurons.jsonl", manifest_path, *expected_visualizations(viz_dir, subset)]
    if overwrite or not all(path.exists() for path in expected):
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing shared neurons: {out_dir}")
        return True
    return False


def backfill_layer_top1pct_visualizations(
    out_dir: Path,
    viz_dir: Path,
    single_root: Path,
    subset: str,
    params: dict[str, Any],
) -> bool:
    manifest_path = out_dir / "manifest.json"
    ctd_path = out_dir / "CTD_neurons.jsonl"
    if not manifest_path.exists() or not ctd_path.exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") != params:
        return False
    clean_legacy_layer_top_visualizations(viz_dir, subset)
    target = viz_dir / f"ctd_layer_top1pct_scar_heatmap_{subset}.png"
    if target.exists():
        return False

    ctd_rows = read_jsonl(ctd_path)
    module_dims = load_module_dims(single_root, subset)
    plot_layer_top_shared_score_heatmap(
        ctd_rows,
        module_dims,
        target,
        score_field="score_min",
        score_label="Shared SCAR = min(score_A, score_B, score_C)",
        title=f"{subset} CTD: top 1% shared SCAR by layer/module",
    )

    summary = read_json(out_dir / "summary.json") if (out_dir / "summary.json").exists() else manifest.get("summary", {})
    scar_visualizations = summary.setdefault("scar_visualizations", {})
    scar_visualizations.pop("layer_top1pct_score_min", None)
    scar_visualizations.pop("layer_top1pct_score_mean", None)
    scar_visualizations.pop("layer_top1pct_by_type", None)
    scar_visualizations["layer_top1pct_shared_score"] = str(target)
    manifest["summary"] = summary
    write_json(out_dir / "summary.json", summary)
    write_json(manifest_path, manifest)
    print(f"Backfilled shared layer top-1% SCAR visualizations: {out_dir}")
    return True


def main() -> None:
    args = parse_args()
    neurons_root = resolve_path(args.neurons_dir) if args.neurons_dir else path_from_config("neurons_dir")
    viz_root = resolve_path(args.visualizations_dir) if args.visualizations_dir else path_from_config("visualizations_dir")
    single_root = neurons_root / args.model_alias / "single_type_by_subset"
    shared_root = neurons_root / args.model_alias / "shared_by_subset"
    viz_dir = viz_root / args.model_alias / "shared_by_subset"
    subsets = ["single_hop", "multi_hop"] if args.subset == "all" else [args.subset]

    global_rows = []
    manifest = {"stage": "06_shared_discovery", "model_alias": args.model_alias, "subsets": {}}

    for subset in subsets:
        out_dir = shared_root / subset
        single_manifest_path = single_root / subset / "manifest.json"
        if not single_manifest_path.exists():
            raise FileNotFoundError(f"Missing single-type manifest for {subset}: {single_manifest_path}")
        params = expected_params(args, single_manifest=read_json(single_manifest_path), subset=subset)
        if (
            not args.overwrite
            and not args.clean
            and backfill_layer_top1pct_visualizations(out_dir, viz_dir, single_root, subset, params)
            and should_skip(out_dir, viz_dir, subset, params, args.overwrite, args.clean)
        ):
            continue
        if should_skip(out_dir, viz_dir, subset, params, args.overwrite, args.clean):
            continue
        ensure_dir(out_dir)
        by_type = {task_type: read_tdn(single_root, subset, task_type) for task_type in TASK_TYPES}
        sets = {task_type: set(rows.keys()) for task_type, rows in by_type.items()}
        ctd = set.intersection(*(sets[task_type] for task_type in TASK_TYPES))
        pairwise = {
            "AB": sets["A"] & sets["B"],
            "AC": sets["A"] & sets["C"],
            "BC": sets["B"] & sets["C"],
        }

        ctd_rows = rows_for_keys(ctd, by_type)
        write_jsonl(out_dir / "CTD_neurons.jsonl", ctd_rows)
        for name, keys in pairwise.items():
            write_jsonl(out_dir / f"pairwise_{name}_neurons.jsonl", rows_for_keys(keys, by_type))

        write_counts_csv(ctd_rows, out_dir / "layer_counts.csv", "layer")
        write_counts_csv(ctd_rows, out_dir / "module_counts.csv", "module")

        share_rows = []
        for task_type in TASK_TYPES:
            denom = max(len(sets[task_type]), 1)
            share_rows.append(
                {
                    "model_alias": args.model_alias,
                    "subset": subset,
                    "task_type": task_type,
                    "tdn_count": len(sets[task_type]),
                    "ctd_count": len(ctd),
                    "share_rate": len(ctd) / denom,
                }
            )
        with (out_dir / "share_rates.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(share_rows[0].keys()))
            writer.writeheader()
            writer.writerows(share_rows)

        module_dims = load_module_dims(single_root, subset)
        heatmap_path = viz_dir / f"shared_neuron_heatmap_{subset}.png"
        plot_heatmap(ctd_rows, module_dims, heatmap_path)
        scar_min_heatmap_path = viz_dir / f"ctd_scar_min_heatmap_{subset}.png"
        scar_mean_heatmap_path = viz_dir / f"ctd_scar_mean_heatmap_{subset}.png"
        plot_ctd_scar_heatmap(ctd_rows, scar_min_heatmap_path, "score_min", args.heatmap_top_n)
        plot_ctd_scar_heatmap(ctd_rows, scar_mean_heatmap_path, "score_mean", args.heatmap_top_n)
        layer_top1pct_path = viz_dir / f"ctd_layer_top1pct_scar_heatmap_{subset}.png"
        plot_layer_top_shared_score_heatmap(
            ctd_rows,
            module_dims,
            layer_top1pct_path,
            score_field="score_min",
            score_label="Shared SCAR = min(score_A, score_B, score_C)",
            title=f"{subset} CTD: top 1% shared SCAR by layer/module",
        )

        summary = {
            "tdn_counts": {task_type: len(sets[task_type]) for task_type in TASK_TYPES},
            "ctd_count": len(ctd),
            "pairwise_counts": {name: len(keys) for name, keys in pairwise.items()},
            "share_rates": share_rows,
            "top_layers": Counter(row["layer"] for row in ctd_rows).most_common(10),
            "top_modules": Counter(row["module"] for row in ctd_rows).most_common(),
            "visualization": str(heatmap_path),
            "scar_visualizations": {
                "score_min": str(scar_min_heatmap_path),
                "score_mean": str(scar_mean_heatmap_path),
                "layer_top1pct_shared_score": str(layer_top1pct_path),
            },
        }
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
        manifest["subsets"][subset] = summary
        global_rows.extend(share_rows)
        print(f"{subset}: CTD={len(ctd)}, pairwise={summary['pairwise_counts']}")

    ensure_dir(shared_root)
    if global_rows:
        with (shared_root / "shared_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()))
            writer.writeheader()
            writer.writerows(global_rows)
    write_json(shared_root / "manifest.json", manifest)
    print(f"Wrote shared manifest: {shared_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
