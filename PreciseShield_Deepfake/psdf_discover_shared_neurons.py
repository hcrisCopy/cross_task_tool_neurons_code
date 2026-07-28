from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
PRECISE_DIR = REPO_ROOT / "PreciseShield"
for candidate in (COMMON_DIR, PRECISE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cttn.data import SUBSETS, TASK_TYPES
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, resolve_path
from cttn.progress import progress
from ps_common import INTERMEDIATE_MODULE


STAGE_VERSION = 1
METHOD_NAME = "PreciseShield_Deepfake"
TDN_FILENAME = "PSDF_TDN_neurons.jsonl"
CTD_FILENAME = "PSDF_CTD_neurons.jsonl"
LAYER_TOP_SCORE_RATIO = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield_Deepfake PSDF-6: discover shared PSDF-CTD neurons by A/B/C intersection.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--neurons-dir",
        default=None,
        help="PSDF neuron root; defaults to ../cross_task_tool_neurons_data/precise_shield_deepfake/neurons.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default=None,
        help="PSDF visualization root; defaults to ../cross_task_tool_neurons_data/precise_shield_deepfake/visualizations.",
    )
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_root(kind: str) -> Path:
    mapping = {
        "neurons": data_root() / "precise_shield_deepfake" / "neurons",
        "visualizations": data_root() / "precise_shield_deepfake" / "visualizations",
    }
    if kind not in mapping:
        raise KeyError(f"Unknown PreciseShield_Deepfake root kind: {kind}")
    return mapping[kind]


def resolve_root(value: str | None, kind: str) -> Path:
    return resolve_path(value) if value else default_root(kind)


def subset_values(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def single_type_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "single_type_by_subset"


def shared_root(neurons_root: Path, model_alias: str) -> Path:
    return neurons_root / model_alias / "shared_by_subset"


def neuron_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["layer"]), str(row.get("module", INTERMEDIATE_MODULE)), int(row["index"])


def read_tdn(root: Path, subset: str, task_type: str) -> dict[tuple[int, str, int], dict[str, Any]]:
    path = root / subset / task_type / TDN_FILENAME
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty PreciseShield_Deepfake TDN file: {path}")
    return {neuron_key(row): row for row in rows}


def rows_for_keys(
    keys: Iterable[tuple[int, str, int]],
    by_type: dict[str, dict[tuple[int, str, int], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, module, index in sorted(keys):
        row: dict[str, Any] = {"layer": int(layer), "module": str(module), "index": int(index)}
        scores: list[float] = []
        ranks: list[int] = []
        means: list[float] = []
        stds: list[float] = []
        for task_type in TASK_TYPES:
            src = by_type.get(task_type, {}).get((layer, module, index))
            if not src:
                continue
            row[f"score_{task_type}"] = src.get("score")
            row[f"rank_{task_type}"] = src.get("rank")
            row[f"mean_delta_{task_type}"] = src.get("mean_delta")
            row[f"std_delta_{task_type}"] = src.get("std_delta")
            row[f"n_pairs_{task_type}"] = src.get("n_pairs")
            if src.get("module_key") is not None:
                row[f"module_key_{task_type}"] = src.get("module_key")
            if src.get("score") is not None:
                scores.append(float(src["score"]))
            if src.get("rank") is not None:
                ranks.append(int(src["rank"]))
            if src.get("mean_delta") is not None:
                means.append(float(src["mean_delta"]))
            if src.get("std_delta") is not None:
                stds.append(float(src["std_delta"]))
        if scores:
            row["score_min"] = min(scores)
            row["score_mean"] = sum(scores) / len(scores)
            row["score_max"] = max(scores)
        if ranks:
            row["rank_max"] = max(ranks)
            row["rank_mean"] = sum(ranks) / len(ranks)
        if means:
            row["mean_delta_mean"] = sum(means) / len(means)
        if stds:
            row["std_delta_mean"] = sum(stds) / len(stds)
        rows.append(row)
    rows.sort(key=lambda item: (-float(item.get("score_min", 0.0)), int(item["layer"]), str(item["module"]), int(item["index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["shared_rank"] = rank
    return rows


def load_layer_dims(single_root: Path, subset: str) -> dict[int, int]:
    meta_path = single_root / subset / "module_meta.json"
    if not meta_path.exists():
        return {}
    return {int(meta["layer"]): int(meta["dim"]) for meta in read_json(meta_path)}


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


def layer_groups(layer_dims: dict[int, int], rows: list[dict[str, Any]]) -> list[int]:
    groups = list(layer_dims) if layer_dims else [int(row["layer"]) for row in rows]
    return sorted(set(groups))


def plot_density(rows: list[dict[str, Any]], layer_dims: dict[int, int], out_path: Path) -> None:
    layers = layer_groups(layer_dims, rows)
    if not layers:
        write_empty_plot(out_path, "PSDF-CTD Shared Neurons")
        return
    counts = Counter(int(row["layer"]) for row in rows)
    matrix = [[counts.get(layer, 0) / max(layer_dims.get(layer, 1), 1)] for layer in layers]
    fig, ax = plt.subplots(figsize=(4.2, max(5, len(layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("PSDF-CTD Shared Neurons")
    ax.set_xticks([0], [INTERMEDIATE_MODULE], rotation=30, ha="right")
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("FFN neuron space")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_heatmap(rows: list[dict[str, Any]], out_path: Path, score_field: str, top_n: int, title: str) -> None:
    scored_rows = [row for row in rows if row.get(score_field) is not None]
    scored_rows.sort(key=lambda row: float(row[score_field]), reverse=True)
    selected = scored_rows[: max(1, min(top_n, len(scored_rows)))]
    if not selected:
        write_empty_plot(out_path, title)
        return
    group_order = sorted({int(row["layer"]) for row in selected})
    y_by_group = {group: idx for idx, group in enumerate(group_order)}
    matrix = [[float("nan") for _ in selected] for _ in group_order]
    for x, row in enumerate(selected):
        matrix[y_by_group[int(row["layer"])]][x] = float(row[score_field])
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.035), max(4, len(group_order) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel("PSDF-CTD rank")
    ax.set_ylabel("Layer / FFN neuron space")
    ticks = list(range(0, len(selected), max(1, len(selected) // 10)))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(group_order)), [f"L{layer}.{INTERMEDIATE_MODULE}" for layer in group_order])
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label=score_field)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_top_shared_score_heatmap(
    rows: list[dict[str, Any]],
    layer_dims: dict[int, int],
    out_path: Path,
    *,
    score_field: str,
    score_label: str,
    title: str,
    ratio: float = LAYER_TOP_SCORE_RATIO,
) -> None:
    layers = layer_groups(layer_dims, rows)
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
    matrix = [[float("nan") for _ in range(max_cols)] for _ in row_values]
    for row_idx, values in enumerate(row_values):
        for col_idx, value in enumerate(values):
            matrix[row_idx][col_idx] = value
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    fig_width = max(10, min(42, max_cols * 0.018))
    fig_height = max(6, len(row_labels) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0)
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
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    neurons_root: Path,
    viz_root: Path,
    single_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "psdf_06_shared_paired_shift_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subset": subset,
        "neurons_dir": str(neurons_root),
        "visualizations_dir": str(viz_root),
        "heatmap_top_n": args.heatmap_top_n,
        "single_type_manifest_params": single_manifest.get("params", {}),
        "selection": "PSDF_CTD = PSDF_TDN_A intersection PSDF_TDN_B intersection PSDF_TDN_C",
        "neuron_identity": "(layer, module=ffn_intermediate, index)",
    }


def expected_visualizations(viz_dir: Path, subset: str) -> list[Path]:
    return [
        viz_dir / f"psdf_ctd_density_heatmap_{subset}.png",
        viz_dir / f"psdf_ctd_shift_min_heatmap_{subset}.png",
        viz_dir / f"psdf_ctd_shift_mean_heatmap_{subset}.png",
        viz_dir / f"psdf_ctd_layer_top1pct_shift_heatmap_{subset}.png",
    ]


def expected_outputs(out_dir: Path, viz_dir: Path, subset: str) -> list[Path]:
    return [
        out_dir / CTD_FILENAME,
        out_dir / "pairwise_AB_neurons.jsonl",
        out_dir / "pairwise_AC_neurons.jsonl",
        out_dir / "pairwise_BC_neurons.jsonl",
        out_dir / "layer_counts.csv",
        out_dir / "module_counts.csv",
        out_dir / "share_rates.csv",
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
    expected = expected_outputs(out_dir, viz_dir, subset)
    if overwrite or not all(path.exists() for path in expected):
        return False
    manifest = read_json(out_dir / "manifest.json")
    if manifest.get("params") == params:
        print(f"Skip existing PreciseShield_Deepfake shared neurons: {out_dir}", flush=True)
        return True
    return False


def run_subset(
    args: argparse.Namespace,
    *,
    subset: str,
    neurons_root: Path,
    viz_root: Path,
) -> dict[str, Any]:
    single_root = single_type_root(neurons_root, args.model_alias)
    shared_dir = shared_root(neurons_root, args.model_alias)
    out_dir = shared_dir / subset
    viz_dir = viz_root / args.model_alias / "shared_by_subset"
    single_manifest_path = single_root / subset / "manifest.json"
    if not single_manifest_path.exists():
        raise FileNotFoundError(f"Missing PSDF single-type manifest: {single_manifest_path}")
    params = expected_params(
        args,
        subset=subset,
        neurons_root=neurons_root,
        viz_root=viz_root,
        single_manifest=read_json(single_manifest_path),
    )
    if should_skip(out_dir, viz_dir, subset, params, overwrite=args.overwrite, clean=args.clean):
        return read_json(out_dir / "summary.json")

    ensure_dir(out_dir)
    by_type = {task_type: read_tdn(single_root, subset, task_type) for task_type in TASK_TYPES}
    sets = {task_type: set(rows) for task_type, rows in by_type.items()}
    ctd = set.intersection(*(sets[task_type] for task_type in TASK_TYPES))
    pairwise = {
        "AB": sets["A"] & sets["B"],
        "AC": sets["A"] & sets["C"],
        "BC": sets["B"] & sets["C"],
    }
    ctd_rows = rows_for_keys(ctd, by_type)
    write_jsonl(out_dir / CTD_FILENAME, ctd_rows)
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
                "psdf_tdn_count": len(sets[task_type]),
                "psdf_ctd_count": len(ctd),
                "share_rate": len(ctd) / denom,
            }
        )
    write_csv_rows(out_dir / "share_rates.csv", share_rows)

    layer_dims = load_layer_dims(single_root, subset)
    density_path = viz_dir / f"psdf_ctd_density_heatmap_{subset}.png"
    score_min_path = viz_dir / f"psdf_ctd_shift_min_heatmap_{subset}.png"
    score_mean_path = viz_dir / f"psdf_ctd_shift_mean_heatmap_{subset}.png"
    layer_top_path = viz_dir / f"psdf_ctd_layer_top1pct_shift_heatmap_{subset}.png"
    plot_density(ctd_rows, layer_dims, density_path)
    plot_score_heatmap(ctd_rows, score_min_path, "score_min", args.heatmap_top_n, f"{subset} PSDF-CTD shared paired shift (min)")
    plot_score_heatmap(ctd_rows, score_mean_path, "score_mean", args.heatmap_top_n, f"{subset} PSDF-CTD shared paired shift (mean)")
    plot_layer_top_shared_score_heatmap(
        ctd_rows,
        layer_dims,
        layer_top_path,
        score_field="score_min",
        score_label="Shared paired shift = min(score_A, score_B, score_C)",
        title=f"{subset} PSDF-CTD: top 1% shared paired shift by layer",
    )

    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "method": METHOD_NAME,
        "neuron_file": str(out_dir / CTD_FILENAME),
        "neuron_set": "PSDF_CTD",
        "psdf_tdn_counts": {task_type: len(sets[task_type]) for task_type in TASK_TYPES},
        "psdf_ctd_count": len(ctd),
        "pairwise_counts": {name: len(keys) for name, keys in pairwise.items()},
        "share_rates": share_rows,
        "top_layers": Counter(int(row["layer"]) for row in ctd_rows).most_common(10),
        "empty_shared_neurons": len(ctd_rows) == 0,
        "visualizations": {
            "density_heatmap": str(density_path),
            "score_min_heatmap": str(score_min_path),
            "score_mean_heatmap": str(score_mean_path),
            "layer_top1pct_shared_score_heatmap": str(layer_top_path),
        },
    }
    if not ctd_rows:
        summary["warning"] = (
            "PSDF_CTD is empty under the strict A/B/C intersection. Keep the method definition unchanged; "
            "rerun PSDF-5 with a larger --top-ratio, then rerun PSDF-6 before ProbePrefill."
        )
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
    print(f"{subset}: PSDF_CTD={len(ctd)}, pairwise={summary['pairwise_counts']}", flush=True)
    if not ctd_rows:
        print(
            f"WARNING {subset}: empty PSDF_CTD. Rerun PSDF-5 with a larger --top-ratio "
            "(deepfake-code default is 0.10), then rerun PSDF-6 before PP-1.",
            flush=True,
        )
    return summary


def main() -> None:
    args = parse_args()
    neurons_root = resolve_root(args.neurons_dir, "neurons")
    viz_root = resolve_root(args.visualizations_dir, "visualizations")
    print(f"PreciseShield_Deepfake subset order = {' -> '.join(subset_values(args.subset))}", flush=True)
    shared_dir = shared_root(neurons_root, args.model_alias)
    root_manifest: dict[str, Any] = {
        "stage": "psdf_06_shared_paired_shift_neuron_discovery",
        "stage_version": STAGE_VERSION,
        "method": METHOD_NAME,
        "model_alias": args.model_alias,
        "subsets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    global_share_rows: list[dict[str, Any]] = []
    for subset in progress(subset_values(args.subset), desc=f"PSDF-6 {args.model_alias}", unit="subset"):
        summary = run_subset(args, subset=subset, neurons_root=neurons_root, viz_root=viz_root)
        root_manifest["subsets"][subset] = summary
        summary_rows.append(
            {
                "model_alias": args.model_alias,
                "subset": subset,
                "method": METHOD_NAME,
                "neuron_set": "PSDF_CTD",
                "selected_neurons": summary["psdf_ctd_count"],
            }
        )
        global_share_rows.extend(read_csv_rows(shared_dir / subset / "share_rates.csv"))
    ensure_dir(shared_dir)
    write_csv_rows(
        shared_dir / "shared_summary.csv",
        summary_rows,
        fieldnames=["model_alias", "subset", "method", "neuron_set", "selected_neurons"],
    )
    write_csv_rows(
        shared_dir / "share_rates.csv",
        global_share_rows,
        fieldnames=["model_alias", "subset", "task_type", "psdf_tdn_count", "psdf_ctd_count", "share_rate"],
    )
    write_json(shared_dir / "manifest.json", root_manifest)
    print(f"Wrote PreciseShield_Deepfake shared manifest: {shared_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
