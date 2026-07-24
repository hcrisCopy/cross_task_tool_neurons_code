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
from cttn.io import read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: discover CTD shared neurons by A/B/C intersection.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--visualizations-dir", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
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
        for task_type in TASK_TYPES:
            src = by_type.get(task_type, {}).get((layer, module, index))
            if src:
                row[f"score_{task_type}"] = src.get("score")
                row[f"rank_{task_type}"] = src.get("rank")
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


def plot_heatmap(rows: list[dict[str, Any]], module_dims: dict[tuple[int, str], int], out_path: Path) -> None:
    if module_dims:
        layers = sorted({layer for layer, _ in module_dims})
    else:
        layers = sorted({int(row["layer"]) for row in rows})
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


def load_module_dims(single_root: Path, subset: str) -> dict[tuple[int, str], int]:
    meta_path = single_root / subset / "module_meta.json"
    if not meta_path.exists():
        return {}
    import json

    with meta_path.open("r", encoding="utf-8") as f:
        module_meta = json.load(f)
    return {(int(meta["layer"]), str(meta["module"])): int(meta["dim"]) for meta in module_meta}


def should_skip(out_dir: Path, overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    if (out_dir / "CTD_neurons.jsonl").exists() and not overwrite:
        print(f"Skip existing shared neurons: {out_dir}")
        return True
    return False


def main() -> None:
    args = parse_args()
    neurons_root = resolve_path(args.neurons_dir) if args.neurons_dir else path_from_config("neurons_dir")
    viz_root = resolve_path(args.visualizations_dir) if args.visualizations_dir else path_from_config("visualizations_dir")
    single_root = neurons_root / args.model_alias / "single_type_by_subset"
    shared_root = neurons_root / args.model_alias / "shared_by_subset"
    subsets = ["single_hop", "multi_hop"] if args.subset == "all" else [args.subset]

    global_rows = []
    manifest = {"stage": "06_shared_discovery", "model_alias": args.model_alias, "subsets": {}}

    for subset in subsets:
        out_dir = shared_root / subset
        if should_skip(out_dir, args.overwrite, args.clean):
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
        heatmap_path = viz_root / args.model_alias / "shared_by_subset" / f"shared_neuron_heatmap_{subset}.png"
        plot_heatmap(ctd_rows, module_dims, heatmap_path)

        summary = {
            "tdn_counts": {task_type: len(sets[task_type]) for task_type in TASK_TYPES},
            "ctd_count": len(ctd),
            "pairwise_counts": {name: len(keys) for name, keys in pairwise.items()},
            "share_rates": share_rows,
            "top_layers": Counter(row["layer"] for row in ctd_rows).most_common(10),
            "top_modules": Counter(row["module"] for row in ctd_rows).most_common(),
            "visualization": str(heatmap_path),
        }
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "manifest.json", {"params": vars(args), "summary": summary})
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
