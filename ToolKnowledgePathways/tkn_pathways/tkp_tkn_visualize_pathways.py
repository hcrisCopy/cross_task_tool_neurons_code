"""Read-only visualization for TKN-based ToolKnowledgePathways outputs.

This script mirrors the original top-level ToolKnowledgePathways visualizer
while reading the isolated tkn_pathways output names.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from cttn.io import read_jsonl, write_json
from cttn.paths import clean_directory, data_root, ensure_dir, resolve_path


NODE_FILENAME = "TKP_TKN_CTD_neurons.jsonl"
EDGE_FILENAME = "TKP_TKN_path_edges.jsonl"
TKN_FILENAME = "TKN_CTD_neurons.jsonl"
DIRECTIONS = ("tool_high", "direct_high")
DIRECTION_COLORS = {"tool_high": "#2563eb", "direct_high": "#f97316"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize selected TKN pathway nodes and edges.")
    parser.add_argument("--output-neurons-dir", default=None, help="TKP neuron root written by tkp_tkn_discover_pathways.py.")
    parser.add_argument("--tkn-neurons-dir", default=None, help="Original TKN neuron root for top-score comparison.")
    parser.add_argument("--visualizations-dir", default=None)
    parser.add_argument("--model-alias", default=None, help="Optional; inferred if the neuron root contains exactly one model directory.")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--top-score-ratio", type=float, default=0.01, help="Original TKN per-layer/per-direction ratio in the heatmap.")
    parser.add_argument("--edge-plot-limit", type=int, default=30000, help="Maximum edges drawn in the edge scatter plot.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def default_neurons_root() -> Path:
    return data_root() / "tool_knowledge_pathways" / "neurons"


def default_viz_root() -> Path:
    return data_root() / "tool_knowledge_pathways" / "visualizations"


def default_tkn_neurons_root() -> Path:
    return data_root() / "tool_knowledge_neurons" / "neurons"


def infer_model_alias(root: Path, value: str | None) -> str:
    if value:
        return value
    options = sorted(path.name for path in root.iterdir() if path.is_dir())
    if len(options) != 1:
        raise ValueError(f"Cannot infer model alias under {root}; pass --model-alias. Found: {options}")
    return options[0]


def subset_values(value: str) -> list[str]:
    return ["single_hop", "multi_hop"] if value == "all" else [value]


def node_edge_paths(neuron_root: Path, model_alias: str, subset: str) -> tuple[Path, Path]:
    root = neuron_root / model_alias / "shared_by_subset" / subset
    return root / NODE_FILENAME, root / EDGE_FILENAME


def original_tkn_path(tkn_root: Path, model_alias: str, subset: str) -> Path:
    return tkn_root / model_alias / "shared_by_subset" / subset / TKN_FILENAME


def score_value(row: dict[str, Any]) -> float:
    for key in ("causal_path_score", "tkp_pathway_score", "score", "tkn_shared_score"):
        try:
            return float(row.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def edge_strength(edge: dict[str, Any]) -> float:
    for key in ("causal_edge_score", "coactivation_score", "coactivation_lift"):
        try:
            return float(edge.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def direction_label(value: str) -> str:
    return "tool" if value == "tool_high" else "direct"


def plot_layer_counts(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    layers = sorted({int(row["layer"]) for row in rows})
    fig, ax = plt.subplots(figsize=(9, max(4, len(layers) * 0.22)))
    left = [0] * len(layers)
    for direction in DIRECTIONS:
        values = [sum(1 for row in rows if int(row["layer"]) == layer and row.get("direction") == direction) for layer in layers]
        ax.barh(
            [str(layer) for layer in layers],
            values,
            left=left,
            color=DIRECTION_COLORS[direction],
            label=direction,
        )
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlabel("Selected pathway neurons")
    ax.set_ylabel("Transformer layer")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scores(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(int(row["layer"]), str(row.get("direction", "tool_high")))].append(score_value(row))
    for (layer, direction), values in sorted(groups.items()):
        color = DIRECTION_COLORS.get(direction, "#0f766e")
        ax.scatter([layer] * len(values), values, color=color, alpha=0.32, s=12)
        ax.scatter([layer], [sum(values) / len(values)], color=color, marker="_", s=180, linewidths=2)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Pathway score")
    ax.set_title(title)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DIRECTION_COLORS[direction], markersize=7, label=direction)
        for direction in DIRECTIONS
    ]
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tkn_score_heatmap(
    pathway_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    *,
    top_score_ratio: float,
) -> None:
    selected = {
        (int(row["layer"]), int(row["index"]), str(row.get("direction", "")))
        for row in pathway_rows
    }
    original_by_group: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in original_rows:
        direction = str(row.get("direction", ""))
        if direction in DIRECTIONS:
            original_by_group[(int(row["layer"]), direction)].append(row)
    groups = sorted(original_by_group)
    top_rows_by_group: dict[tuple[int, str], list[dict[str, Any]]] = {}
    width = 1
    for group in groups:
        candidates = sorted(
            original_by_group[group],
            key=lambda row: -float(row.get("tkn_shared_score", row.get("score", 0.0))),
        )
        module_dim = int(candidates[0].get("module_dim", len(candidates)))
        count = max(1, int(math.ceil(module_dim * float(top_score_ratio))))
        top_rows_by_group[group] = candidates[:count]
        width = max(width, len(top_rows_by_group[group]))

    initial_matrix: list[list[float]] = []
    retained_matrix: list[list[float]] = []
    all_scores: list[float] = []
    visible_retained = 0
    for group in groups:
        top_rows = top_rows_by_group[group]
        initial_values = [float(row.get("tkn_shared_score", row.get("score", 0.0))) for row in top_rows]
        retained_values = []
        for row, score in zip(top_rows, initial_values):
            key = (int(row["layer"]), int(row["index"]), str(row.get("direction", "")))
            if key in selected:
                retained_values.append(score)
                visible_retained += 1
            else:
                retained_values.append(float("nan"))
        initial_matrix.append(initial_values + [float("nan")] * (width - len(initial_values)))
        retained_matrix.append(retained_values + [float("nan")] * (width - len(retained_values)))
        all_scores.extend(initial_values)

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f3f4f6")
    vmin, vmax = min(all_scores), max(all_scores)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(16, width * 0.22), max(5, len(groups) * 0.18)),
        sharey=True,
        layout="constrained",
    )
    image = axes[0].imshow(initial_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].imshow(retained_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    tick_step = max(1, width // 10)
    ticks = list(range(0, width, tick_step))
    labels = [str(value + 1) for value in ticks]
    ratio_label = f"{100 * float(top_score_ratio):.1f}%"
    for ax, panel_title in zip(
        axes,
        [
            f"Initial TKN top {ratio_label} by layer and direction",
            f"Retained pathway nodes within initial top {ratio_label} ({visible_retained}/{len(pathway_rows)} visible)",
        ],
    ):
        ax.set_title(panel_title)
        ax.set_xlabel(f"Neuron rank within top {ratio_label}")
        ax.set_xticks(ticks, labels, rotation=30, ha="right")
    axes[0].set_ylabel("Layer / direction")
    axes[0].set_yticks(
        range(len(groups)),
        [f"L{layer}.{direction_label(direction)}" for layer, direction in groups],
    )
    fig.suptitle(title)
    fig.colorbar(image, ax=axes, fraction=0.026, pad=0.02, label="TKN shared importance score")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_edges(rows: list[dict[str, Any]], edges: list[dict[str, Any]], out_path: Path, title: str, edge_limit: int) -> None:
    selected = {
        (int(row["layer"]), int(row["index"]), str(row.get("direction", "")))
        for row in rows
    }
    kept_edges = [
        edge for edge in edges
        if (
            int(edge["source_layer"]),
            int(edge["source_index"]),
            str(edge.get("direction", "")),
        ) in selected
        and (
            int(edge["target_layer"]),
            int(edge["target_index"]),
            str(edge.get("direction", "")),
        ) in selected
    ]
    kept_edges.sort(key=lambda edge: -edge_strength(edge))
    if edge_limit > 0:
        kept_edges = kept_edges[:edge_limit]

    fig, ax = plt.subplots(figsize=(12, 7))
    for edge in kept_edges:
        direction = str(edge.get("direction", "tool_high"))
        color = DIRECTION_COLORS.get(direction, "#60a5fa")
        strength = max(edge_strength(edge), 0.0)
        ax.plot(
            [int(edge["source_layer"]), int(edge["target_layer"])],
            [int(edge["source_index"]), int(edge["target_index"])],
            color=color,
            alpha=min(0.70, 0.08 + strength),
            linewidth=0.55,
        )
    for direction in DIRECTIONS:
        points = [row for row in rows if row.get("direction") == direction]
        if points:
            ax.scatter(
                [int(row["layer"]) for row in points],
                [int(row["index"]) for row in points],
                s=8,
                color=DIRECTION_COLORS[direction],
                alpha=0.55,
                label=direction,
            )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("FFN intermediate index")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_subset(
    neuron_root: Path,
    tkn_root: Path,
    viz_root: Path,
    model_alias: str,
    subset: str,
    *,
    top_score_ratio: float,
    edge_plot_limit: int,
    clean: bool,
) -> None:
    nodes_path, edges_path = node_edge_paths(neuron_root, model_alias, subset)
    rows = read_jsonl(nodes_path)
    if not rows:
        raise ValueError(f"No selected pathway nodes: {nodes_path}")
    edges = read_jsonl(edges_path) if edges_path.exists() else []
    original_rows = read_jsonl(original_tkn_path(tkn_root, model_alias, subset))
    out_dir = viz_root / model_alias / subset
    if clean and out_dir.exists():
        clean_directory(out_dir, viz_root)
    ensure_dir(out_dir)

    plot_layer_counts(rows, out_dir / "pathway_node_counts_by_layer.png", f"{subset}: selected pathway neurons")
    plot_scores(rows, out_dir / "pathway_scores_by_layer.png", f"{subset}: pathway scores")
    plot_tkn_score_heatmap(
        rows,
        original_rows,
        out_dir / "pathway_tkn_scores_heatmap.png",
        f"{subset}: TKN score comparison",
        top_score_ratio=top_score_ratio,
    )
    plot_edges(rows, edges, out_dir / "causal_pathway_edges.png", f"{subset}: directional FFN pathways", edge_plot_limit)
    summary = {
        "model_alias": model_alias,
        "subset": subset,
        "node_count": len(rows),
        "edge_count": len(edges),
        "visualized_edge_limit": edge_plot_limit,
        "top_score_ratio": top_score_ratio,
        "node_counts_by_layer": dict(sorted(Counter(int(row["layer"]) for row in rows).items())),
        "node_counts_by_direction": dict(sorted(Counter(str(row.get("direction", "")) for row in rows).items())),
        "files": {
            "pathway_node_counts_by_layer": str(out_dir / "pathway_node_counts_by_layer.png"),
            "pathway_scores_by_layer": str(out_dir / "pathway_scores_by_layer.png"),
            "pathway_tkn_scores_heatmap": str(out_dir / "pathway_tkn_scores_heatmap.png"),
            "causal_pathway_edges": str(out_dir / "causal_pathway_edges.png"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(f"{subset}: wrote pathway visualizations to {out_dir}")


def main() -> None:
    args = parse_args()
    neuron_root = resolve_path(args.output_neurons_dir) if args.output_neurons_dir else default_neurons_root()
    tkn_root = resolve_path(args.tkn_neurons_dir) if args.tkn_neurons_dir else default_tkn_neurons_root()
    viz_root = resolve_path(args.visualizations_dir) if args.visualizations_dir else default_viz_root()
    model_alias = infer_model_alias(neuron_root, args.model_alias)
    for subset in subset_values(args.subset):
        run_subset(
            neuron_root,
            tkn_root,
            viz_root,
            model_alias,
            subset,
            top_score_ratio=args.top_score_ratio,
            edge_plot_limit=args.edge_plot_limit,
            clean=args.clean,
        )


if __name__ == "__main__":
    main()
