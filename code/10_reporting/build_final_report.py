from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cttn.data import SUBSETS, TASK_TYPES
from cttn.eval_metrics import flatten_summary, write_csv
from cttn.io import read_json, write_json
from cttn.modeling import VALID_MODEL_ALIASES
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10: collect result tables and simple figures.")
    parser.add_argument("--model-alias", required=True, help="Model alias, comma list, or all.")
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--causal-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def relative_posix(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def selected_models(model_arg: str, roots: dict[str, Path]) -> list[str]:
    if model_arg == "all":
        found = set()
        for root in roots.values():
            if root.exists():
                found.update(path.name for path in root.iterdir() if path.is_dir())
        return [alias for alias in VALID_MODEL_ALIASES if alias in found]
    values = [item.strip() for item in model_arg.split(",") if item.strip()]
    bad = [item for item in values if item not in VALID_MODEL_ALIASES]
    if bad:
        raise ValueError(f"Unknown model aliases: {bad}")
    return values


def default_report_dir(outputs_root: Path, model_arg: str, model_aliases: list[str]) -> Path:
    if len(model_aliases) == 1:
        scope = model_aliases[0]
    elif model_arg == "all":
        scope = "all_models"
    else:
        scope = "__".join(model_aliases)
    return outputs_root / "final_report" / scope


def collect_source_manifest_params(
    model_aliases: list[str],
    *,
    labels_root: Path,
    neurons_root: Path,
    checkpoints_root: Path,
    outputs_root: Path,
    causal_root: Path,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for model_alias in model_aliases:
        model_sources: dict[str, Any] = {"labels": {}, "shared": {}, "training": {}, "evaluation": {}, "causal": {}}
        for subset in SUBSETS:
            model_sources["labels"][subset] = {}
            for split in ["train", "test"]:
                path = labels_root / model_alias / subset / split / "manifest.json"
                if path.exists():
                    model_sources["labels"][subset][split] = read_json(path).get("params", {})
            for key, root_path in [
                ("shared", neurons_root / model_alias / "shared_by_subset" / subset / "manifest.json"),
                ("training", checkpoints_root / model_alias / "ctd_masked_lora" / subset / "manifest.json"),
                ("evaluation", outputs_root / model_alias / "trained_evaluation" / subset / "manifest.json"),
                ("causal", causal_root / model_alias / subset / "manifest.json"),
            ]:
                if root_path.exists():
                    model_sources[key][subset] = read_json(root_path).get("params", {})
        sources[model_alias] = model_sources
    return sources


def should_skip(report_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(report_dir, data_root())
        return False
    manifest_path = report_dir / "manifest.json"
    if overwrite or not manifest_path.exists() or not (report_dir / "README_results.md").exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing final report: {report_dir}")
        return True
    return False


def collect_neuron_rows(model_aliases: list[str], neurons_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            shared_dir = neurons_root / model_alias / "shared_by_subset" / subset
            ctd_summary = read_json(shared_dir / "summary.json") if (shared_dir / "summary.json").exists() else {}
            for task_type in TASK_TYPES:
                single_summary_path = neurons_root / model_alias / "single_type_by_subset" / subset / task_type / "summary.json"
                single_summary = read_json(single_summary_path) if single_summary_path.exists() else {}
                share_rate = None
                for item in ctd_summary.get("share_rates", []):
                    if item.get("task_type") == task_type:
                        share_rate = item.get("share_rate")
                        break
                rows.append(
                    {
                        "model_alias": model_alias,
                        "subset": subset,
                        "task_type": task_type,
                        "tdn_count": single_summary.get("top_k", 0),
                        "ctd_count": ctd_summary.get("ctd_count", 0),
                        "share_rate": share_rate if share_rate is not None else 0.0,
                        "top_layers": ctd_summary.get("top_layers", []),
                        "top_modules": ctd_summary.get("top_modules", []),
                    }
                )
    return rows


def collect_training_rows(model_aliases: list[str], checkpoints_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            summary_path = checkpoints_root / model_alias / "ctd_masked_lora" / subset / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            mask = summary.get("mask_summary", {})
            rows.append(
                {
                    "model_alias": model_alias,
                    "subset": subset,
                    "method": "CTD-Masked-LoRA",
                    "train_rows": summary.get("train_rows"),
                    "training_examples": summary.get("training_examples"),
                    "trainable_features": summary.get("trainable_features"),
                    "skipped_trajectory_examples": summary.get("skipped_trajectory_examples"),
                    "skipped_tokenization_examples": summary.get("skipped_tokenization_examples"),
                    "update_steps": summary.get("update_steps"),
                    "last_loss": summary.get("last_loss"),
                    "ctd_neuron_count": mask.get("ctd_neuron_count"),
                    "trainable_lora_parameters": mask.get("trainable_lora_parameters"),
                }
            )
    return rows


def collect_evaluation_rows(model_aliases: list[str], outputs_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            summary_path = outputs_root / model_alias / "trained_evaluation" / subset / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            if "overall" in summary:
                rows.extend(
                    flatten_summary(
                        summary,
                        model_alias=model_alias,
                        subset=subset,
                        method="CTD-Masked-LoRA",
                    )
                )
            else:
                mean_std = summary.get("mean_std", {})
                overall = mean_std.get("overall", {})
                row = {"model_alias": model_alias, "subset": subset, "method": "CTD-Masked-LoRA", "group_kind": "overall", "group_name": "overall"}
                for key, stat in overall.items():
                    row[f"{key}_mean"] = stat.get("mean")
                    row[f"{key}_std"] = stat.get("std")
                rows.append(row)
    return rows


def collect_causal_rows(model_aliases: list[str], causal_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    cross_rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            subset_dir = causal_root / model_alias / subset
            summary_rows.extend(read_csv_rows(subset_dir / "summary_table.csv"))
            cross_rows.extend(read_csv_rows(subset_dir / "cross_type_summary.csv"))
    return summary_rows, cross_rows


def build_model_summary(
    model_aliases: list[str],
    neuron_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    causal_cross_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            neuron_subset = [row for row in neuron_rows if row["model_alias"] == model_alias and row["subset"] == subset]
            train = next((row for row in training_rows if row["model_alias"] == model_alias and row["subset"] == subset), {})
            eval_overall = next(
                (
                    row
                    for row in evaluation_rows
                    if row.get("model_alias") == model_alias
                    and row.get("subset") == subset
                    and row.get("group_kind") == "overall"
                ),
                {},
            )
            causal_ctd = next(
                (
                    row
                    for row in causal_cross_rows
                    if row.get("model_alias") == model_alias
                    and row.get("subset") == subset
                    and row.get("intervention") == "Mask-CTD"
                ),
                {},
            )
            ctd_count = max([parse_int(row.get("ctd_count")) for row in neuron_subset] or [0])
            out.append(
                {
                    "model_alias": model_alias,
                    "subset": subset,
                    "ctd_count": ctd_count,
                    "training_examples": train.get("training_examples", ""),
                    "trainable_lora_parameters": train.get("trainable_lora_parameters", ""),
                    "ctd_masked_lora_acc": eval_overall.get("final_accuracy", eval_overall.get("final_accuracy_mean", "")),
                    "ctd_masked_lora_avg_tc": eval_overall.get("avg_tool_calls", eval_overall.get("avg_tool_calls_mean", "")),
                    "ctd_masked_lora_tool_acc": eval_overall.get("decision_accuracy", eval_overall.get("decision_accuracy_mean", "")),
                    "mask_ctd_avg_delta_acc": causal_ctd.get("avg_delta_acc", ""),
                    "mask_ctd_avg_delta_tcr": causal_ctd.get("avg_delta_tcr", ""),
                }
            )
    return out


def plot_ctd_counts(rows: list[dict[str, Any]], path: Path) -> None:
    subset_rows = []
    seen = set()
    for row in rows:
        key = (row["model_alias"], row["subset"])
        if key in seen:
            continue
        seen.add(key)
        subset_rows.append(row)
    if not subset_rows:
        return
    labels = [f"{row['model_alias']}\n{row['subset']}" for row in subset_rows]
    values = [parse_int(row.get("ctd_count")) for row in subset_rows]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    ax.bar(labels, values, color="#3b82f6")
    ax.set_ylabel("CTD neurons")
    ax.set_title("Shared Tool-Decision Neurons")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_eval(rows: list[dict[str, Any]], path: Path) -> None:
    overall = [row for row in rows if row.get("group_kind") == "overall"]
    if not overall:
        return
    labels = [f"{row['model_alias']}\n{row['subset']}" for row in overall]
    acc = [parse_float(row.get("final_accuracy", row.get("final_accuracy_mean"))) for row in overall]
    avg_tc = [parse_float(row.get("avg_tool_calls", row.get("avg_tool_calls_mean"))) for row in overall]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.6), 4))
    axes[0].bar(labels, acc, color="#16a34a")
    axes[0].set_ylabel("Final accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, avg_tc, color="#f97316")
    axes[1].set_ylabel("Avg tool calls")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("CTD-Masked-LoRA Test Evaluation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_causal(rows: list[dict[str, Any]], path: Path) -> None:
    ctd = [row for row in rows if row.get("intervention") == "Mask-CTD"]
    if not ctd:
        return
    labels = [f"{row['model_alias']}\n{row['subset']}" for row in ctd]
    values = [parse_float(row.get("avg_delta_acc")) for row in ctd]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    ax.axhline(0, color="#111827", linewidth=1)
    ax.bar(labels, values, color="#9333ea")
    ax.set_ylabel("Avg delta accuracy vs Base")
    ax.set_title("Mask-CTD Causal Effect")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_results_md(
    path: Path,
    *,
    model_aliases: list[str],
    model_summary: list[dict[str, Any]],
    figures: list[Path],
) -> None:
    report_dir = path.parent
    lines = [
        "# Cross-Task Tool-Decision Neurons Results",
        "",
        "This report summarizes generated artifacts for the selected model aliases.",
        "When2Tool baseline values for Default, Sparse, Reason-then-Act, and Probe&Prefill should be cited from the paper tables; this pipeline adds CTD-Masked-LoRA and causal masking results.",
        "",
        f"Models: {', '.join(model_aliases)}",
        "",
        "## Model Summary",
        "",
        "| Model | Subset | CTD | Train Examples | Acc | AvgTC | ToolAcc | Mask-CTD DeltaAcc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        lines.append(
            "| {model_alias} | {subset} | {ctd_count} | {training_examples} | {acc} | {avg_tc} | {tool_acc} | {delta} |".format(
                model_alias=row["model_alias"],
                subset=row["subset"],
                ctd_count=row["ctd_count"],
                training_examples=row["training_examples"],
                acc=row["ctd_masked_lora_acc"],
                avg_tc=row["ctd_masked_lora_avg_tc"],
                tool_acc=row["ctd_masked_lora_tool_acc"],
                delta=row["mask_ctd_avg_delta_acc"],
            )
        )
    if figures:
        lines.extend(["", "## Figures", ""])
        for fig in figures:
            lines.append(f"- `{relative_posix(fig, report_dir)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    labels_root = resolve_path(args.labels_dir) if args.labels_dir else path_from_config("labels_dir")
    neurons_root = resolve_path(args.neurons_dir) if args.neurons_dir else path_from_config("neurons_dir")
    checkpoints_root = resolve_path(args.checkpoints_dir) if args.checkpoints_dir else path_from_config("checkpoints_dir")
    outputs_root = resolve_path(args.outputs_dir) if args.outputs_dir else path_from_config("outputs_dir")
    causal_root = resolve_path(args.causal_dir) if args.causal_dir else path_from_config("causal_validation_dir")
    roots = {
        "labels": labels_root,
        "neurons": neurons_root,
        "checkpoints": checkpoints_root,
        "outputs": outputs_root,
        "causal": causal_root,
    }
    model_aliases = selected_models(args.model_alias, roots)
    report_dir = resolve_path(args.report_dir) if args.report_dir else default_report_dir(outputs_root, args.model_alias, model_aliases)
    params = {
        "stage": "10_reporting",
        "model_aliases": model_aliases,
        "report_scope": report_dir.name,
        "source_manifest_params": collect_source_manifest_params(
            model_aliases,
            labels_root=labels_root,
            neurons_root=neurons_root,
            checkpoints_root=checkpoints_root,
            outputs_root=outputs_root,
            causal_root=causal_root,
        ),
    }
    if should_skip(report_dir, params, args.overwrite, args.clean):
        return
    ensure_dir(report_dir)
    figures_dir = ensure_dir(report_dir / "figures")

    neuron_rows = collect_neuron_rows(model_aliases, neurons_root)
    training_rows = collect_training_rows(model_aliases, checkpoints_root)
    evaluation_rows = collect_evaluation_rows(model_aliases, outputs_root)
    causal_rows, causal_cross_rows = collect_causal_rows(model_aliases, causal_root)
    model_summary = build_model_summary(model_aliases, neuron_rows, training_rows, evaluation_rows, causal_cross_rows)

    write_csv(report_dir / "neuron_discovery_summary.csv", neuron_rows)
    write_csv(report_dir / "training_run_summary.csv", training_rows)
    write_csv(report_dir / "training_comparison.csv", evaluation_rows)
    write_csv(report_dir / "causal_validation_summary.csv", causal_rows)
    write_csv(report_dir / "causal_cross_type_summary.csv", causal_cross_rows)
    write_csv(report_dir / "model_summary.csv", model_summary)

    figures = [
        figures_dir / "ctd_counts.png",
        figures_dir / "trained_evaluation.png",
        figures_dir / "mask_ctd_causal_effect.png",
    ]
    plot_ctd_counts(neuron_rows, figures[0])
    plot_eval(evaluation_rows, figures[1])
    plot_causal(causal_cross_rows, figures[2])
    existing_figures = [fig for fig in figures if fig.exists()]
    write_results_md(report_dir / "README_results.md", model_aliases=model_aliases, model_summary=model_summary, figures=existing_figures)
    write_json(
        report_dir / "manifest.json",
        {
            "params": params,
            "row_counts": {
                "neuron_discovery_summary": len(neuron_rows),
                "training_run_summary": len(training_rows),
                "training_comparison": len(evaluation_rows),
                "causal_validation_summary": len(causal_rows),
                "causal_cross_type_summary": len(causal_cross_rows),
                "model_summary": len(model_summary),
            },
            "figures": [relative_posix(fig, report_dir) for fig in existing_figures],
        },
    )
    print(f"Wrote final report: {report_dir}")


if __name__ == "__main__":
    main()
