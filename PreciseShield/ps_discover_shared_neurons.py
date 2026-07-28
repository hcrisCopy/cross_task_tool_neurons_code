from __future__ import annotations

import argparse
import csv
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
    ps_neuron_key,
    ps_resolve_root,
    read_json,
    read_jsonl,
    remove_files,
    rows_by_layer,
    should_skip,
    subset_values,
    write_json,
    write_jsonl,
)
from cttn.paths import ensure_dir


LAYER_TOP_SCORE_RATIO = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield stage 6: discover shared PS-CTD neurons.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--visualizations-dir", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def single_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "single_type_by_subset"


def shared_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "shared_by_subset"


def read_ps_tdn(root: Path, subset: str, task_type: str) -> dict[tuple[int, int], dict[str, Any]]:
    path = root / subset / task_type / "PS_TDN_neurons.jsonl"
    rows = read_jsonl(path)
    return {ps_neuron_key(row): row for row in rows}


def rows_for_keys(keys: set[tuple[int, int]], by_type: dict[str, dict[tuple[int, int], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, index in sorted(keys):
        row: dict[str, Any] = {"layer": int(layer), "module": INTERMEDIATE_MODULE, "index": int(index)}
        scores = []
        ranks = []
        for task_type in TASK_TYPES:
            src = by_type.get(task_type, {}).get((layer, index))
            if src:
                row[f"score_{task_type}"] = src.get("score")
                row[f"rank_{task_type}"] = src.get("rank")
                row[f"call_saliency_{task_type}"] = src.get("call_saliency")
                row[f"direct_saliency_{task_type}"] = src.get("direct_saliency")
                if src.get("score") is not None:
                    scores.append(float(src["score"]))
                if src.get("rank") is not None:
                    ranks.append(int(src["rank"]))
        if scores:
            row["score_min"] = min(scores)
            row["score_mean"] = sum(scores) / len(scores)
        if ranks:
            row["rank_max"] = max(ranks)
            row["rank_mean"] = sum(ranks) / len(ranks)
        rows.append(row)
    rows.sort(key=lambda row: (-float(row.get("score_min", 0.0)), int(row["layer"]), int(row["index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def private_rows(tdn: dict[tuple[int, int], dict[str, Any]], ctd: set[tuple[int, int]]) -> list[dict[str, Any]]:
    rows = []
    for key, row in tdn.items():
        if key not in ctd:
            out = dict(row)
            out["private_to_type"] = True
            rows.append(out)
    rows.sort(key=lambda row: int(row.get("rank", 10**9)))
    return rows


def load_module_dims(single_dir: Path, subset: str) -> dict[int, int]:
    meta_path = single_dir / subset / "module_meta.json"
    if not meta_path.exists():
        return {}
    meta = read_json(meta_path)
    return {int(row["layer"]): int(row["dim"]) for row in meta}


def expected_params(args: argparse.Namespace, *, subset: str, single_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "ps_06_shared_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": "PreciseShield",
        "model_alias": args.model_alias,
        "subset": subset,
        "heatmap_top_n": args.heatmap_top_n,
        "single_type_manifest_params": single_manifest.get("params", {}),
        "selection": "PS_CTD = PS_TDN_A intersection PS_TDN_B intersection PS_TDN_C",
    }


def expected_viz(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"ps_ctd_density_heatmap_{subset}.png",
        viz_dir / f"ps_ctd_saliency_min_heatmap_{subset}.png",
        viz_dir / f"ps_ctd_saliency_mean_heatmap_{subset}.png",
        viz_dir / f"ps_ctd_layer_top1pct_saliency_min_heatmap_{subset}.png",
        viz_dir / f"ps_ctd_layer_top1pct_saliency_mean_heatmap_{subset}.png",
    ]


def legacy_layer_top_viz(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"ps_ctd_layer_top3pct_saliency_min_heatmap_{subset}.png",
        viz_dir / f"ps_ctd_layer_top3pct_saliency_mean_heatmap_{subset}.png",
        viz_dir / f"ps_ctd_layer_top10_saliency_min_heatmap_{subset}.png",
        viz_dir / f"ps_ctd_layer_top10_saliency_mean_heatmap_{subset}.png",
    ]


def clean_legacy_layer_top_viz(viz_dir: Path, subset: str) -> None:
    remove_files(legacy_layer_top_viz(viz_dir, subset))


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


def plot_density(rows: list[dict[str, Any]], layer_dims: dict[int, int], out_path: Path) -> None:
    layers = sorted(layer_dims) if layer_dims else sorted({int(row["layer"]) for row in rows})
    if not layers:
        write_empty_plot(out_path, "PS-CTD Shared Neurons")
        return
    counts = Counter(int(row["layer"]) for row in rows)
    matrix = [[counts.get(layer, 0) / max(layer_dims.get(layer, 1), 1)] for layer in layers]
    fig, ax = plt.subplots(figsize=(4.2, max(4, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("PS-CTD Shared Neurons")
    ax.set_xticks([0], [INTERMEDIATE_MODULE], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN neuron space")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_saliency(rows: list[dict[str, Any]], out_path: Path, score_field: str, top_n: int) -> None:
    scored = [row for row in rows if row.get(score_field) is not None]
    scored.sort(key=lambda row: float(row[score_field]), reverse=True)
    selected = scored[: max(1, min(top_n, len(scored)))]
    if not selected:
        write_empty_plot(out_path, f"PS-CTD saliency ({score_field})")
        return
    layers = sorted({int(row["layer"]) for row in selected})
    y_by_layer = {layer: idx for idx, layer in enumerate(layers)}
    matrix = torch.full((len(layers), len(selected)), float("nan"), dtype=torch.float32)
    for x, row in enumerate(selected):
        matrix[y_by_layer[int(row["layer"])], x] = float(row[score_field])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(layers) * 0.28)))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(f"PS-CTD saliency ({score_field})")
    ax.set_xlabel("PS-CTD rank")
    ax.set_ylabel("Layer")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label=score_field)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_shared_saliency_heatmap(
    rows: list[dict[str, Any]],
    layer_dims: dict[int, int],
    out_path: Path,
    *,
    score_field: str,
    score_label: str,
    title: str,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    layers = sorted(layer_dims) if layer_dims else sorted({int(row["layer"]) for row in rows})
    row_values: list[list[float]] = []
    row_labels: list[str] = []
    max_cols = 0
    for layer in layers:
        candidates = [
            float(row[score_field])
            for row in rows
            if int(row["layer"]) == layer and row.get(score_field) is not None
        ]
        dim = max(layer_dims.get(layer, len(candidates)), 1)
        k = max(1, int(dim * ratio))
        candidates.sort(reverse=True)
        values = candidates[: min(k, len(candidates))]
        row_values.append(values)
        row_labels.append(f"L{layer}.{INTERMEDIATE_MODULE}")
        max_cols = max(max_cols, k, len(values))

    if not row_values or not any(row_values) or max_cols <= 0:
        write_empty_plot(out_path, title)
        return

    matrix = torch.full((len(row_values), max_cols), float("nan"), dtype=torch.float32)
    for row_idx, values in enumerate(row_values):
        if values:
            matrix[row_idx, : len(values)] = torch.tensor(values, dtype=torch.float32)

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
    if ticks and ticks[-1] != max_cols - 1:
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


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backfill_layer_top1pct_visualizations(
    subset_out: Path,
    viz_dir: Path,
    single_dir: Path,
    subset: str,
    params: dict[str, Any],
) -> bool:
    manifest_path = subset_out / "manifest.json"
    ctd_path = subset_out / "PS_CTD_neurons.jsonl"
    if not manifest_path.exists() or not ctd_path.exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") != params:
        return False

    clean_legacy_layer_top_viz(viz_dir, subset)
    targets = {
        "score_min": viz_dir / f"ps_ctd_layer_top1pct_saliency_min_heatmap_{subset}.png",
        "score_mean": viz_dir / f"ps_ctd_layer_top1pct_saliency_mean_heatmap_{subset}.png",
    }
    if all(path.exists() for path in targets.values()):
        return False

    ctd_rows = read_jsonl(ctd_path)
    layer_dims = load_module_dims(single_dir, subset)
    for score_field, out_path in targets.items():
        plot_layer_top_shared_saliency_heatmap(
            ctd_rows,
            layer_dims,
            out_path,
            score_field=score_field,
            score_label=f"PS-CTD call saliency {score_field}",
            title=f"{subset} PS-CTD: top 1% shared call saliency by layer ({score_field})",
        )

    summary = read_json(subset_out / "summary.json") if (subset_out / "summary.json").exists() else manifest.get("summary", {})
    visualizations = summary.setdefault("visualizations", {})
    visualizations["layer_top1pct_score_min_heatmap"] = str(targets["score_min"])
    visualizations["layer_top1pct_score_mean_heatmap"] = str(targets["score_mean"])
    manifest["summary"] = summary
    write_json(subset_out / "summary.json", summary)
    write_json(manifest_path, manifest)
    print(f"Backfilled PS-CTD layer top-1% saliency visualizations: {subset_out}")
    return True


def main() -> None:
    args = parse_args()
    neurons_root = ps_resolve_root(args.neurons_dir, "neurons")
    viz_root = ps_resolve_root(args.visualizations_dir, "visualizations")
    single_dir = single_root(neurons_root, args.model_alias)
    shared_dir = shared_root(neurons_root, args.model_alias)
    viz_dir = viz_root / args.model_alias / "shared_by_subset"
    subsets = subset_values(args.subset)
    root_manifest = {"stage": "ps_06_shared_neuron_discovery", "model_alias": args.model_alias, "subsets": {}}
    global_share_rows = []

    for subset in subsets:
        subset_out = shared_dir / subset
        single_manifest_path = single_dir / subset / "manifest.json"
        if not single_manifest_path.exists():
            raise FileNotFoundError(f"Missing PreciseShield single-type manifest: {single_manifest_path}")
        params = expected_params(args, subset=subset, single_manifest=read_json(single_manifest_path))
        if args.clean:
            remove_files(expected_viz(viz_dir, subset))
            clean_legacy_layer_top_viz(viz_dir, subset)
        if (
            not args.overwrite
            and not args.clean
            and backfill_layer_top1pct_visualizations(subset_out, viz_dir, single_dir, subset, params)
            and should_skip(
                subset_out,
                params,
                [subset_out / "PS_CTD_neurons.jsonl", *expected_viz(viz_dir, subset)],
                overwrite=args.overwrite,
                clean=args.clean,
            )
        ):
            continue
        if should_skip(
            subset_out,
            params,
            [subset_out / "PS_CTD_neurons.jsonl", *expected_viz(viz_dir, subset)],
            overwrite=args.overwrite,
            clean=args.clean,
        ):
            continue

        ensure_dir(subset_out)
        by_type = {task_type: read_ps_tdn(single_dir, subset, task_type) for task_type in TASK_TYPES}
        sets = {task_type: set(rows.keys()) for task_type, rows in by_type.items()}
        ctd = set.intersection(*(sets[task_type] for task_type in TASK_TYPES))
        pairwise = {"AB": sets["A"] & sets["B"], "AC": sets["A"] & sets["C"], "BC": sets["B"] & sets["C"]}
        ctd_rows = rows_for_keys(ctd, by_type)
        write_jsonl(subset_out / "PS_CTD_neurons.jsonl", ctd_rows)
        for name, keys in pairwise.items():
            write_jsonl(subset_out / f"pairwise_{name}_neurons.jsonl", rows_for_keys(keys, by_type))
        for task_type in TASK_TYPES:
            write_jsonl(subset_out / f"private_{task_type}_neurons.jsonl", private_rows(by_type[task_type], ctd))

        write_counts_csv(ctd_rows, subset_out / "layer_counts.csv", "layer")
        share_rows = []
        for task_type in TASK_TYPES:
            denom = max(len(sets[task_type]), 1)
            share_rows.append(
                {
                    "model_alias": args.model_alias,
                    "subset": subset,
                    "task_type": task_type,
                    "ps_tdn_count": len(sets[task_type]),
                    "ps_ctd_count": len(ctd),
                    "share_rate": len(ctd) / denom,
                }
            )
        write_csv_rows(subset_out / "share_rates.csv", share_rows)
        global_share_rows.extend(share_rows)

        layer_dims = load_module_dims(single_dir, subset)
        density_path = viz_dir / f"ps_ctd_density_heatmap_{subset}.png"
        min_path = viz_dir / f"ps_ctd_saliency_min_heatmap_{subset}.png"
        mean_path = viz_dir / f"ps_ctd_saliency_mean_heatmap_{subset}.png"
        layer_top1pct_min_path = viz_dir / f"ps_ctd_layer_top1pct_saliency_min_heatmap_{subset}.png"
        layer_top1pct_mean_path = viz_dir / f"ps_ctd_layer_top1pct_saliency_mean_heatmap_{subset}.png"
        plot_density(ctd_rows, layer_dims, density_path)
        plot_saliency(ctd_rows, min_path, "score_min", args.heatmap_top_n)
        plot_saliency(ctd_rows, mean_path, "score_mean", args.heatmap_top_n)
        plot_layer_top_shared_saliency_heatmap(
            ctd_rows,
            layer_dims,
            layer_top1pct_min_path,
            score_field="score_min",
            score_label="PS-CTD call saliency score_min",
            title=f"{subset} PS-CTD: top 1% shared call saliency by layer (score_min)",
        )
        plot_layer_top_shared_saliency_heatmap(
            ctd_rows,
            layer_dims,
            layer_top1pct_mean_path,
            score_field="score_mean",
            score_label="PS-CTD call saliency score_mean",
            title=f"{subset} PS-CTD: top 1% shared call saliency by layer (score_mean)",
        )

        summary = {
            "ps_tdn_counts": {task_type: len(sets[task_type]) for task_type in TASK_TYPES},
            "ps_ctd_count": len(ctd),
            "pairwise_counts": {name: len(keys) for name, keys in pairwise.items()},
            "private_counts": {task_type: len(sets[task_type] - ctd) for task_type in TASK_TYPES},
            "share_rates": share_rows,
            "top_layers": Counter(row["layer"] for row in ctd_rows).most_common(10),
            "visualizations": {
                "density_heatmap": str(density_path),
                "score_min_heatmap": str(min_path),
                "score_mean_heatmap": str(mean_path),
                "layer_top1pct_score_min_heatmap": str(layer_top1pct_min_path),
                "layer_top1pct_score_mean_heatmap": str(layer_top1pct_mean_path),
            },
        }
        write_json(subset_out / "summary.json", summary)
        write_json(subset_out / "manifest.json", {"params": params, "summary": summary})
        root_manifest["subsets"][subset] = summary
        print(f"{subset}: PS-CTD={len(ctd)}, pairwise={summary['pairwise_counts']}")

    ensure_dir(shared_dir)
    write_csv_rows(shared_dir / "shared_summary.csv", global_share_rows)
    write_json(shared_dir / "manifest.json", root_manifest)
    print(f"Wrote PreciseShield shared manifest: {shared_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
