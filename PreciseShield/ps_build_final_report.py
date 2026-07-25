from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(REPO_ROOT / "PreciseShield") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PreciseShield"))

from cttn.eval_metrics import write_csv
from cttn.io import read_json, write_json
from cttn.modeling import VALID_MODEL_ALIASES
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path


SUBSETS = ("single_hop", "multi_hop")
TASK_TYPES = ("A", "B", "C")


def precise_root() -> Path:
    return data_root() / "precise_shield"


def ps_resolve_root(value: str | None, kind: str) -> Path:
    mapping = {
        "neurons": "neurons",
        "checkpoints": "checkpoints",
        "outputs": "outputs",
        "causal": "causal_validation",
    }
    if value:
        return resolve_path(value)
    return precise_root() / mapping[kind]


def clean_path(path: Path) -> None:
    clean_directory(path, data_root())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield PS-11: collect result tables and figures.")
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


def should_skip(report_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_path(report_dir)
        return False
    manifest_path = report_dir / "manifest.json"
    if overwrite or not manifest_path.exists() or not (report_dir / "README_results.md").exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing PreciseShield final report: {report_dir}")
        return True
    return False


def source_manifest_params(
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
        model_sources: dict[str, Any] = {
            "labels": {},
            "ps_single_type": {},
            "ps_shared": {},
            "ps_training": {},
            "ps_trained_eval": {},
            "ps_base_eval": {},
            "ps_comparison": {},
            "ps_causal": {},
        }
        for subset in SUBSETS:
            model_sources["labels"][subset] = {}
            for split in ("train", "test"):
                path = labels_root / model_alias / subset / split / "manifest.json"
                if path.exists():
                    model_sources["labels"][subset][split] = read_json(path).get("params", {})
            paths = {
                "ps_single_type": neurons_root / model_alias / "single_type_by_subset" / subset / "manifest.json",
                "ps_shared": neurons_root / model_alias / "shared_by_subset" / subset / "manifest.json",
                "ps_training": checkpoints_root / model_alias / "ps_masked_lora" / subset / "manifest.json",
                "ps_trained_eval": outputs_root / model_alias / "trained_evaluation" / subset / "manifest.json",
                "ps_base_eval": outputs_root / model_alias / "base_evaluation" / subset / "manifest.json",
                "ps_comparison": outputs_root
                / model_alias
                / "trained_evaluation"
                / subset
                / "comparison_with_base_manifest.json",
                "ps_causal": causal_root / model_alias / subset / "manifest.json",
            }
            for key, path in paths.items():
                if path.exists():
                    model_sources[key][subset] = read_json(path).get("params", {})
        sources[model_alias] = model_sources
    return sources


def collect_neuron_rows(model_aliases: list[str], neurons_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            shared_dir = neurons_root / model_alias / "shared_by_subset" / subset
            shared_summary = read_json(shared_dir / "summary.json") if (shared_dir / "summary.json").exists() else {}
            ps_tdn_counts = shared_summary.get("ps_tdn_counts", {})
            private_counts = shared_summary.get("private_counts", {})
            pairwise_counts = shared_summary.get("pairwise_counts", {})
            share_rates = {
                row.get("task_type"): row.get("share_rate")
                for row in shared_summary.get("share_rates", [])
                if isinstance(row, dict)
            }
            for task_type in TASK_TYPES:
                single_summary_path = neurons_root / model_alias / "single_type_by_subset" / subset / task_type / "summary.json"
                single_summary = read_json(single_summary_path) if single_summary_path.exists() else {}
                rows.append(
                    {
                        "model_alias": model_alias,
                        "subset": subset,
                        "task_type": task_type,
                        "ps_tdn_count": ps_tdn_counts.get(task_type, single_summary.get("selected_neurons", "")),
                        "ps_ctd_count": shared_summary.get("ps_ctd_count", ""),
                        "private_count": private_counts.get(task_type, ""),
                        "share_rate": share_rates.get(task_type, ""),
                        "pairwise_AB": pairwise_counts.get("AB", ""),
                        "pairwise_AC": pairwise_counts.get("AC", ""),
                        "pairwise_BC": pairwise_counts.get("BC", ""),
                        "n_call": single_summary.get("n_call", ""),
                        "n_direct": single_summary.get("n_direct", ""),
                        "top_layers": shared_summary.get("top_layers", []),
                    }
                )
    return rows


def collect_training_rows(model_aliases: list[str], checkpoints_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            summary_path = checkpoints_root / model_alias / "ps_masked_lora" / subset / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            mask = summary.get("mask_summary", {})
            rows.append(
                {
                    "model_alias": model_alias,
                    "subset": subset,
                    "method": "PreciseShield-Masked-LoRA",
                    "train_rows": summary.get("train_rows"),
                    "training_examples": summary.get("training_examples"),
                    "trainable_features": summary.get("trainable_features"),
                    "skipped_trajectory_examples": summary.get("skipped_trajectory_examples"),
                    "skipped_tokenization_examples": summary.get("skipped_tokenization_examples"),
                    "update_steps": summary.get("update_steps"),
                    "last_loss": summary.get("last_loss"),
                    "ps_ctd_neuron_count": mask.get("ps_ctd_neuron_count"),
                    "trainable_lora_parameters": mask.get("trainable_lora_parameters"),
                    "target_module_count": mask.get("target_module_count"),
                }
            )
    return rows


def collect_summary_tables(model_aliases: list[str], root: Path, stage: str) -> list[dict[str, Any]]:
    rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            rows.extend(read_csv_rows(root / model_alias / stage / subset / "summary_table.csv"))
    return rows


def collect_comparison_rows(model_aliases: list[str], outputs_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            rows.extend(read_csv_rows(outputs_root / model_alias / "trained_evaluation" / subset / "comparison_with_base.csv"))
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


def row_lookup(rows: list[dict[str, Any]], model_alias: str, subset: str, **keys: str) -> dict[str, Any]:
    for row in rows:
        if row.get("model_alias") != model_alias or row.get("subset") != subset:
            continue
        if all(row.get(key) == value for key, value in keys.items()):
            return row
    return {}


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_model_summary(
    model_aliases: list[str],
    neuron_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    trained_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    causal_cross_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for model_alias in model_aliases:
        for subset in SUBSETS:
            subset_neurons = [row for row in neuron_rows if row["model_alias"] == model_alias and row["subset"] == subset]
            train = row_lookup(training_rows, model_alias, subset)
            trained_overall = row_lookup(trained_rows, model_alias, subset, group_kind="overall", group_name="overall")
            base_overall = row_lookup(base_rows, model_alias, subset, group_kind="overall", group_name="overall")
            comp = row_lookup(comparison_rows, model_alias, subset, group_kind="overall", group_name="overall")
            causal_ctd = row_lookup(causal_cross_rows, model_alias, subset, intervention="Mask-PS-CTD")
            if not causal_ctd:
                causal_ctd = row_lookup(causal_cross_rows, model_alias, subset, intervention="Mask-CTD")
            ctd_count = max([parse_int(row.get("ps_ctd_count")) for row in subset_neurons] or [0])
            out.append(
                {
                    "model_alias": model_alias,
                    "subset": subset,
                    "ps_ctd_count": ctd_count,
                    "training_examples": train.get("training_examples", ""),
                    "trainable_lora_parameters": train.get("trainable_lora_parameters", ""),
                    "base_acc": comp.get("base_final_accuracy", base_overall.get("final_accuracy", base_overall.get("final_accuracy_mean", ""))),
                    "base_avg_tool_calls": comp.get("base_avg_tool_calls", base_overall.get("avg_tool_calls", base_overall.get("avg_tool_calls_mean", ""))),
                    "ps_acc": comp.get("ctd_final_accuracy", trained_overall.get("final_accuracy", trained_overall.get("final_accuracy_mean", ""))),
                    "ps_avg_tool_calls": comp.get("ctd_avg_tool_calls", trained_overall.get("avg_tool_calls", trained_overall.get("avg_tool_calls_mean", ""))),
                    "ps_decision_accuracy": comp.get(
                        "ctd_decision_accuracy",
                        trained_overall.get("decision_accuracy", trained_overall.get("decision_accuracy_mean", "")),
                    ),
                    "delta_acc_pp": comp.get("delta_acc_pp", ""),
                    "delta_avg_tool_calls": comp.get("delta_avg_tool_calls", ""),
                    "tool_call_reduction_percent": comp.get("tool_call_reduction_percent", ""),
                    "mask_ps_ctd_avg_delta_acc_pp": causal_ctd.get("avg_delta_acc_pp", ""),
                    "mask_ps_ctd_avg_delta_tool_call_rate": causal_ctd.get("avg_delta_tool_call_rate", causal_ctd.get("avg_delta_tcr", "")),
                }
            )
    return out


def plot_bar(rows: list[dict[str, Any]], value_key: str, ylabel: str, title: str, path: Path) -> None:
    plot_rows = [row for row in rows if row.get(value_key) not in (None, "")]
    if not plot_rows:
        return
    plt = import_pyplot()
    labels = [f"{row['model_alias']}\n{row['subset']}" for row in plot_rows]
    values = [parse_float(row.get(value_key)) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.3), 4))
    ax.axhline(0, color="#111827", linewidth=1)
    ax.bar(labels, values, color="#2563eb")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_eval(rows: list[dict[str, Any]], path: Path) -> None:
    plot_rows = [row for row in rows if row.get("base_acc") not in (None, "") or row.get("ps_acc") not in (None, "")]
    if not plot_rows:
        return
    plt = import_pyplot()
    labels = [f"{row['model_alias']}\n{row['subset']}" for row in plot_rows]
    x = list(range(len(labels)))
    width = 0.38
    base_acc = [parse_float(row.get("base_acc")) for row in plot_rows]
    ps_acc = [parse_float(row.get("ps_acc")) for row in plot_rows]
    base_tc = [parse_float(row.get("base_avg_tool_calls")) for row in plot_rows]
    ps_tc = [parse_float(row.get("ps_avg_tool_calls")) for row in plot_rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(labels) * 1.5), 4))
    axes[0].bar([i - width / 2 for i in x], base_acc, width=width, color="#64748b", label="Base")
    axes[0].bar([i + width / 2 for i in x], ps_acc, width=width, color="#16a34a", label="PS")
    axes[1].bar([i - width / 2 for i in x], base_tc, width=width, color="#64748b", label="Base")
    axes[1].bar([i + width / 2 for i in x], ps_tc, width=width, color="#f97316", label="PS")
    axes[0].set_ylabel("Final accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Avg tool calls")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend()
    fig.suptitle("Base vs PreciseShield-Masked-LoRA")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def relative_posix(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_results_md(path: Path, model_aliases: list[str], model_summary: list[dict[str, Any]], figures: list[Path]) -> None:
    lines = [
        "# PreciseShield Results",
        "",
        "This report summarizes already generated PreciseShield artifacts. It does not rerun models.",
        "",
        f"Models: {', '.join(model_aliases)}",
        "",
        "| Model | Subset | PS-CTD | Train Examples | Base Acc | PS Acc | Delta Acc pp | PS AvgTC | TC Reduction % | Mask-PS-CTD DeltaAcc pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        lines.append(
            "| {model_alias} | {subset} | {ps_ctd_count} | {training_examples} | {base_acc} | {ps_acc} | {delta_acc_pp} | {ps_avg_tool_calls} | {tool_call_reduction_percent} | {mask_ps_ctd_avg_delta_acc_pp} |".format(
                **row
            )
        )
    if figures:
        lines.extend(["", "## Figures", ""])
        for figure in figures:
            lines.append(f"- `{relative_posix(figure, path.parent)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    labels_root = resolve_path(args.labels_dir) if args.labels_dir else path_from_config("labels_dir")
    neurons_root = ps_resolve_root(args.neurons_dir, "neurons")
    checkpoints_root = ps_resolve_root(args.checkpoints_dir, "checkpoints")
    outputs_root = ps_resolve_root(args.outputs_dir, "outputs")
    causal_root = ps_resolve_root(args.causal_dir, "causal")
    roots = {
        "neurons": neurons_root,
        "checkpoints": checkpoints_root,
        "outputs": outputs_root,
        "causal": causal_root,
    }
    model_aliases = selected_models(args.model_alias, roots)
    report_dir = resolve_path(args.report_dir) if args.report_dir else default_report_dir(outputs_root, args.model_alias, model_aliases)
    params = {
        "stage": "ps_11_reporting",
        "method": "PreciseShield",
        "model_aliases": model_aliases,
        "source_manifest_params": source_manifest_params(
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
    trained_rows = collect_summary_tables(model_aliases, outputs_root, "trained_evaluation")
    base_rows = collect_summary_tables(model_aliases, outputs_root, "base_evaluation")
    comparison_rows = collect_comparison_rows(model_aliases, outputs_root)
    causal_rows, causal_cross_rows = collect_causal_rows(model_aliases, causal_root)
    model_summary = build_model_summary(
        model_aliases,
        neuron_rows,
        training_rows,
        trained_rows,
        base_rows,
        comparison_rows,
        causal_cross_rows,
    )

    write_csv(report_dir / "neuron_discovery_summary.csv", neuron_rows)
    write_csv(report_dir / "training_run_summary.csv", training_rows)
    write_csv(report_dir / "base_evaluation_summary.csv", base_rows)
    write_csv(report_dir / "trained_evaluation_summary.csv", trained_rows)
    write_csv(report_dir / "training_comparison.csv", comparison_rows)
    write_csv(report_dir / "causal_validation_summary.csv", causal_rows)
    write_csv(report_dir / "causal_cross_type_summary.csv", causal_cross_rows)
    write_csv(report_dir / "model_summary.csv", model_summary)

    figures = [
        figures_dir / "ps_ctd_counts.png",
        figures_dir / "base_vs_precise_shield.png",
        figures_dir / "mask_ps_ctd_causal_effect.png",
    ]
    plot_bar(model_summary, "ps_ctd_count", "PS-CTD neurons", "Shared PreciseShield Neurons", figures[0])
    plot_eval(model_summary, figures[1])
    plot_bar(
        model_summary,
        "mask_ps_ctd_avg_delta_acc_pp",
        "Avg delta accuracy pp",
        "Mask-PS-CTD Causal Effect",
        figures[2],
    )
    existing_figures = [figure for figure in figures if figure.exists()]
    write_results_md(report_dir / "README_results.md", model_aliases, model_summary, existing_figures)
    write_json(
        report_dir / "manifest.json",
        {
            "params": params,
            "row_counts": {
                "neuron_discovery_summary": len(neuron_rows),
                "training_run_summary": len(training_rows),
                "base_evaluation_summary": len(base_rows),
                "trained_evaluation_summary": len(trained_rows),
                "training_comparison": len(comparison_rows),
                "causal_validation_summary": len(causal_rows),
                "causal_cross_type_summary": len(causal_cross_rows),
                "model_summary": len(model_summary),
            },
            "figures": [relative_posix(figure, report_dir) for figure in existing_figures],
        },
    )
    print(f"Wrote PreciseShield final report: {report_dir}")


if __name__ == "__main__":
    main()
