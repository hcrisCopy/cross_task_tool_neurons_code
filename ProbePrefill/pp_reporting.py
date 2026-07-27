from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable


MODEL_DISPLAY = {
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b-instruct": "Qwen3-4B-Inst.",
    "qwen3-14b": "Qwen3-14B",
    "qwen3-32b": "Qwen3-32B",
    "llama3.1-8b": "Llama-3.1-8B",
    "llama3.3-70b": "Llama-3.3-70B",
}


WHEN2TOOL_PROBE_REFERENCE = {
    "single_hop": {
        "qwen3-1.7b": {"auroc": 0.894, "accuracy": 0.847, "easy_auroc": 0.864, "medium_auroc": 0.831, "hard_auroc": 0.904},
        "qwen3-4b-instruct": {"auroc": 0.948, "accuracy": 0.877, "easy_auroc": 0.933, "medium_auroc": 0.906, "hard_auroc": 0.948},
        "llama3.1-8b": {"auroc": 0.927, "accuracy": 0.849, "easy_auroc": 0.892, "medium_auroc": 0.867, "hard_auroc": 0.884},
        "qwen3-14b": {"auroc": 0.957, "accuracy": 0.892, "easy_auroc": 0.955, "medium_auroc": 0.907, "hard_auroc": 0.941},
        "qwen3-32b": {"auroc": 0.952, "accuracy": 0.885, "easy_auroc": 0.951, "medium_auroc": 0.903, "hard_auroc": 0.939},
        "llama3.3-70b": {"auroc": 0.936, "accuracy": 0.872, "easy_auroc": 0.906, "medium_auroc": 0.849, "hard_auroc": 0.956},
    },
    "multi_hop": {
        "qwen3-1.7b": {"auroc": 0.839, "accuracy": 0.796},
        "qwen3-4b-instruct": {"auroc": 0.966, "accuracy": 0.947},
        "qwen3-14b": {"auroc": 0.906, "accuracy": 0.822},
        "qwen3-32b": {"auroc": 0.944, "accuracy": 0.873},
        "llama3.1-8b": {"auroc": 0.895, "accuracy": 0.829},
        "llama3.3-70b": {"auroc": 0.804, "accuracy": 0.729},
    },
}


_SINGLE_BASE = {
    "qwen3-1.7b": {"accuracy_pct": 88.2, "total_tool_calls": 2709},
    "qwen3-4b-instruct": {"accuracy_pct": 89.2, "total_tool_calls": 2118},
    "qwen3-14b": {"accuracy_pct": 93.7, "total_tool_calls": 2211},
    "qwen3-32b": {"accuracy_pct": 94.1, "total_tool_calls": 2404},
    "llama3.1-8b": {"accuracy_pct": 79.5, "total_tool_calls": 3708},
    "llama3.3-70b": {"accuracy_pct": 83.1, "total_tool_calls": 4377},
}


_SINGLE_SOFT = {
    "qwen3-1.7b": {0.1: (88.8, 2512), 0.3: (89.0, 2507), 0.5: (88.3, 2128), 0.7: (81.6, 1415), 0.9: (47.9, 293)},
    "qwen3-4b-instruct": {0.1: (91.1, 2216), 0.3: (90.4, 1707), 0.5: (88.5, 1309), 0.7: (84.8, 1026), 0.9: (74.7, 657)},
    "qwen3-14b": {0.1: (94.3, 2128), 0.3: (94.1, 1509), 0.5: (92.4, 1227), 0.7: (85.8, 907), 0.9: (66.0, 347)},
    "qwen3-32b": {0.1: (94.0, 1996), 0.3: (93.2, 1493), 0.5: (90.1, 1155), 0.7: (82.3, 896), 0.9: (71.5, 604)},
    "llama3.1-8b": {0.1: (69.2, 3027), 0.3: (68.9, 2770), 0.5: (69.7, 2381), 0.7: (66.5, 2146), 0.9: (61.7, 1753)},
    "llama3.3-70b": {0.1: (88.4, 2976), 0.3: (88.6, 2902), 0.5: (88.3, 2871), 0.7: (88.6, 2828), 0.9: (89.2, 2804)},
}


_SINGLE_HARD = {
    "qwen3-1.7b": {0.1: (85.3, 2765), 0.3: (85.7, 2723), 0.5: (85.7, 2275), 0.7: (74.3, 1506), 0.9: (35.9, 169)},
    "qwen3-4b-instruct": {0.1: (91.6, 2222), 0.3: (87.6, 1612), 0.5: (81.7, 1185), 0.7: (71.6, 824), 0.9: (49.0, 195)},
    "qwen3-14b": {0.1: (93.6, 2111), 0.3: (91.1, 1498), 0.5: (87.9, 1198), 0.7: (76.9, 841), 0.9: (53.4, 200)},
    "qwen3-32b": {0.1: (92.9, 2053), 0.3: (89.9, 1439), 0.5: (84.1, 1064), 0.7: (74.0, 741), 0.9: (56.8, 241)},
    "llama3.1-8b": {0.1: (79.9, 3561), 0.3: (77.9, 2969), 0.5: (69.6, 2105), 0.7: (55.5, 1363), 0.9: (33.7, 554)},
    "llama3.3-70b": {0.1: (79.0, 3609), 0.3: (67.8, 2427), 0.5: (56.0, 1830), 0.7: (46.7, 1258), 0.9: (33.9, 366)},
}


_MULTI_BASE = {
    "qwen3-1.7b": {"accuracy_pct": 21.2, "total_tool_calls": 1180},
    "qwen3-4b-instruct": {"accuracy_pct": 82.1, "total_tool_calls": 1719},
    "qwen3-14b": {"accuracy_pct": 87.5, "total_tool_calls": 1503},
    "qwen3-32b": {"accuracy_pct": 88.9, "total_tool_calls": 1634},
    "llama3.1-8b": {"accuracy_pct": 40.2, "total_tool_calls": 1005},
    "llama3.3-70b": {"accuracy_pct": 62.4, "total_tool_calls": 985},
}


_MULTI_PROBE = {
    "qwen3-1.7b": {0.1: (22.9, 1197), 0.3: (24.4, 1190), 0.5: (32.5, 1121), 0.7: (39.5, 627), 0.9: (59.2, 85)},
    "qwen3-4b-instruct": {0.1: (82.1, 1287), 0.3: (85.3, 437), 0.5: (83.2, 366), 0.7: (83.4, 354), 0.9: (79.6, 175)},
    "qwen3-14b": {0.1: (87.3, 1483), 0.3: (86.1, 996), 0.5: (82.9, 771), 0.7: (79.9, 701), 0.9: (79.3, 686)},
    "qwen3-32b": {0.1: (88.7, 1481), 0.3: (89.0, 727), 0.5: (86.1, 553), 0.7: (83.3, 493), 0.9: (74.6, 353)},
    "llama3.1-8b": {0.1: (60.1, 1361), 0.3: (59.0, 1291), 0.5: (57.2, 1186), 0.7: (55.6, 1156), 0.9: (54.3, 1047)},
    "llama3.3-70b": {0.1: (80.1, 1856), 0.3: (80.3, 1789), 0.5: (79.7, 1494), 0.7: (78.7, 1228), 0.9: (69.9, 1074)},
}


WHEN2TOOL_EVAL_REFERENCE = {
    "single_hop": {"n": 2250, "base_default": _SINGLE_BASE, "probe_prefill": {"soft": _SINGLE_SOFT, "hard": _SINGLE_HARD}},
    "multi_hop": {"n": 450, "base_default": _MULTI_BASE, "probe_prefill": {"paper_default": _MULTI_PROBE}},
}


METRIC_GLOSSARY = [
    ("AUROC", "Area under ROC for probe P(tool_necessary); threshold independent."),
    ("Accuracy", "Probe decision accuracy at the printed threshold, or final-answer accuracy for generation."),
    ("Final Accuracy", "correct / N using When2Tool item_final_eval on the final boxed answer."),
    ("Total Tool Calls", "sum(tool_calls_i) over evaluated tasks and runs after mean aggregation."),
    ("Avg Tool Calls", "Total Tool Calls / N."),
    ("Tool Call Rate", "sum(tool_calls_i) / sum(expected_steps_i); expected_steps=1 for single-hop, 3 for multi-hop unless task steps are present."),
    ("Total Token Cost", "generation_tokens + 0.2 * prefill_tokens, matching When2Tool utils.finalize_state."),
    ("DecisionAcc", "mean(1[actual_tool_call == tool_necessary])."),
    ("OverCall", "P(actual_tool_call=1 | tool_necessary=0)."),
    ("UnderCall", "P(actual_tool_call=0 | tool_necessary=1)."),
    ("ToolPrecision/Recall/F1", "Binary tool-call decision metrics with actual_tool_call as prediction and tool_necessary as label."),
    ("ValidToolCallRate", "tool_calls / (tool_calls + invalid_tool_attempts)."),
    ("ToolTrajectorySuccessRate", "final accuracy among examples that made at least one valid tool call."),
    ("DeltaAcc(pp)", "100 * (metric_PP.final_accuracy - metric_Base.final_accuracy), in percentage points."),
    ("DeltaAvgTC", "PP avg_tool_calls - Base avg_tool_calls. Negative means fewer calls than Base."),
    ("DeltaTCR", "PP tool_call_rate - Base tool_call_rate. Negative means lower call rate than Base."),
    ("DeltaTC%", "100 * (PP total_tool_calls - Base total_tool_calls) / Base total_tool_calls."),
    ("ToolCallReduction%", "-DeltaTC%. Positive means PP saved tool calls relative to Base."),
    ("Cost", "DeltaAcc(pp) / (-DeltaAvgTC), only meaningful when DeltaAvgTC < 0."),
]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_value(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        mean = value.get("mean")
        std = value.get("std")
        return (
            float(mean) if isinstance(mean, (int, float)) else None,
            float(std) if isinstance(std, (int, float)) else None,
        )
    if isinstance(value, (int, float)):
        return float(value), None
    return None, None


def section_metrics(summary: dict[str, Any], section: str = "overall", group: str | None = None) -> dict[str, Any]:
    source = summary.get("mean_std", summary)
    if section == "overall":
        raw = source.get("overall", {})
    else:
        raw = source.get(section, {}).get(str(group), {})
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        mean, std = _metric_value(value)
        if mean is not None:
            flat[key] = mean
        if std is not None:
            flat[f"{key}_std"] = std
    return flat


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _threshold_key(threshold: float) -> float:
    return round(float(threshold), 3)


def paper_probe_reference(model_alias: str, subset: str) -> dict[str, Any] | None:
    return WHEN2TOOL_PROBE_REFERENCE.get(subset, {}).get(model_alias)


def paper_eval_reference(model_alias: str, subset: str, threshold: float, prefill_mode: str) -> dict[str, Any] | None:
    block = WHEN2TOOL_EVAL_REFERENCE.get(subset)
    if not block:
        return None
    base = block["base_default"].get(model_alias)
    mode = prefill_mode if subset == "single_hop" else "paper_default"
    probe_table = block["probe_prefill"].get(mode, {}).get(model_alias, {})
    probe = probe_table.get(_threshold_key(threshold))
    if not base or not probe:
        return None
    acc_pct, total_tc = probe
    n = int(block["n"])
    delta_acc_pp = float(acc_pct) - float(base["accuracy_pct"])
    delta_avg_tc = (float(total_tc) - float(base["total_tool_calls"])) / n
    delta_tc_percent = 100.0 * (float(total_tc) - float(base["total_tool_calls"])) / max(float(base["total_tool_calls"]), 1.0)
    return {
        "source": "When2Tool paper hidden-state Probe&Prefill",
        "n": n,
        "prefill_mode": mode,
        "threshold": float(threshold),
        "base_accuracy_pct": float(base["accuracy_pct"]),
        "base_total_tool_calls": int(base["total_tool_calls"]),
        "probe_accuracy_pct": float(acc_pct),
        "probe_total_tool_calls": int(total_tc),
        "delta_acc_pp": delta_acc_pp,
        "delta_avg_tool_calls": delta_avg_tc,
        "tool_call_reduction_percent": -delta_tc_percent,
    }


def metric_glossary_markdown() -> str:
    lines = ["### Metric glossary", "", "| Metric | Meaning |", "|---|---|"]
    for name, meaning in METRIC_GLOSSARY:
        lines.append(f"| {name} | {meaning} |")
    return "\n".join(lines)


def classification_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "test"]:
        split_payload = results.get(split, {})
        for section, group_kind in [
            ("overall", "overall"),
            ("by_task_type", "task_type"),
            ("by_env", "env"),
            ("by_difficulty", "difficulty"),
        ]:
            if section == "overall":
                groups = {"overall": split_payload.get("overall", {})}
            else:
                groups = split_payload.get(section, {})
            for group_name, metrics in groups.items():
                row = {
                    "model_alias": results.get("model_alias"),
                    "subset": results.get("subset"),
                    "split": split,
                    "group_kind": group_kind,
                    "group_name": group_name,
                }
                for key, value in metrics.items():
                    if key != "confusion_matrix":
                        row[key] = value
                row["confusion_matrix"] = metrics.get("confusion_matrix")
                rows.append(row)
    return rows


def write_probe_training_report(
    *,
    out_dir: Path,
    results: dict[str, Any],
    train_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "probe_metrics_table.csv", classification_rows(results))
    _write_probe_plots(out_dir=out_dir, train_predictions=train_predictions, test_predictions=test_predictions)

    model_alias = str(results.get("model_alias", ""))
    subset = str(results.get("subset", ""))
    ref = paper_probe_reference(model_alias, subset)
    overall = results.get("test", {}).get("overall", {})
    lines = [
        f"# Probe report: {model_alias}/{subset}",
        "",
        "## Current CTD probe",
        "",
        f"- Feature set: CTD FFN last-token activations, dim={results.get('feature_dim')}",
        f"- Train/Test: {results.get('n_train')} / {results.get('n_test')}",
        f"- Logistic regression: StandardScaler + L2, reg={results.get('reg')} (C={results.get('C')}), threshold={results.get('threshold')}",
        f"- Test AUROC: {_fmt_float(overall.get('auroc'))}",
        f"- Test Accuracy: {_fmt_pct(overall.get('accuracy'))}",
        f"- Precision/Recall/F1: {_fmt_float(overall.get('precision'))} / {_fmt_float(overall.get('recall'))} / {_fmt_float(overall.get('f1'))}",
        f"- Confusion matrix [[TN, FP], [FN, TP]]: {overall.get('confusion_matrix')}",
        "",
        "## When2Tool paper reference",
        "",
    ]
    if ref:
        lines.extend(
            [
                "- Reference uses all-layer pre-generation hidden states, not CTD neuron activations.",
                f"- Paper AUROC / Acc: {_fmt_float(ref.get('auroc'))} / {_fmt_pct(ref.get('accuracy'))}",
            ]
        )
        if subset == "single_hop":
            lines.append(
                f"- Paper AUROC by difficulty: easy={_fmt_float(ref.get('easy_auroc'))}, medium={_fmt_float(ref.get('medium_auroc'))}, hard={_fmt_float(ref.get('hard_auroc'))}"
            )
    else:
        lines.append("- No exact paper reference stored for this model/subset.")
    lines.extend(["", metric_glossary_markdown(), ""])
    (out_dir / "probe_report.md").write_text("\n".join(lines), encoding="utf-8")
    print_probe_training_summary(results, ref)


def print_probe_training_summary(results: dict[str, Any], ref: dict[str, Any] | None = None) -> None:
    model_alias = results.get("model_alias")
    subset = results.get("subset")
    overall = results.get("test", {}).get("overall", {})
    print("\n=== Probe metrics (CTD neuron features) ===")
    print(f"Model/subset: {model_alias}/{subset}")
    print("Definition: label=1 means no-tool answer failed, so tool is necessary.")
    print(f"Feature dim: {results.get('feature_dim')} | reg={results.get('reg')} -> C={results.get('C')} | threshold={results.get('threshold')}")
    print(
        "Test: "
        f"AUROC={_fmt_float(overall.get('auroc'))} "
        f"Acc={_fmt_float(overall.get('accuracy'))} "
        f"Precision={_fmt_float(overall.get('precision'))} "
        f"Recall={_fmt_float(overall.get('recall'))} "
        f"F1={_fmt_float(overall.get('f1'))}"
    )
    if ref:
        print(
            "When2Tool paper hidden-state reference: "
            f"AUROC={_fmt_float(ref.get('auroc'))} Acc={_fmt_float(ref.get('accuracy'))}"
        )
    print("Saved: probe_report.md, probe_metrics_table.csv, probe_probability_hist.png, probe_roc_curve.png")


def _write_probe_plots(*, out_dir: Path, train_predictions: list[dict[str, Any]], test_predictions: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, roc_curve
    except Exception as exc:
        (out_dir / "plot_warning.txt").write_text(f"Plot skipped: {exc}\n", encoding="utf-8")
        return

    def arrays(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
        return [int(r["tool_necessary"]) for r in rows], [float(r["probe_probability"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, rows, color in [("train", train_predictions, "#4c78a8"), ("test", test_predictions, "#f58518")]:
        y, p = arrays(rows)
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{split} AUROC={auc(fpr, tpr):.3f}", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CTD Probe ROC")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "probe_roc_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split, rows in zip(axes, ["train", "test"], [train_predictions, test_predictions]):
        y, p = arrays(rows)
        neg = [prob for label, prob in zip(y, p) if label == 0]
        pos = [prob for label, prob in zip(y, p) if label == 1]
        ax.hist(neg, bins=20, alpha=0.7, label="tool_necessary=0", color="#4c78a8")
        ax.hist(pos, bins=20, alpha=0.7, label="tool_necessary=1", color="#f58518")
        ax.set_title(split)
        ax.set_xlabel("P(tool_necessary)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Count")
    axes[1].legend()
    fig.suptitle("Probe probability distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "probe_probability_hist.png", dpi=180)
    plt.close(fig)


def write_eval_case_report(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    model_alias: str,
    subset: str,
    method: str,
    threshold: float | None = None,
    temperature: float | None = None,
    prefill_mode: str | None = None,
    tool_format: str | None = None,
    prefill_stats: dict[str, Any] | None = None,
) -> None:
    overall = section_metrics(summary, "overall")
    ref = paper_eval_reference(model_alias, subset, threshold, prefill_mode or "") if threshold is not None else None
    lines = [
        f"# Evaluation report: {model_alias}/{subset}/{method}",
        "",
        "## Conditions",
        "",
        f"- Prompt mode: current",
        f"- Reasoning mode: no_reasoning",
        f"- Tool format: {tool_format or 'unknown'}",
        f"- Prefill mode: {prefill_mode or 'none'}",
        f"- Threshold tau: {threshold if threshold is not None else 'none'}",
        f"- Probe temperature T: {temperature if temperature is not None else 'none'}",
        "- Delta baseline: Base/Default with same model, same test ids, same prompt/tool/parser/generation settings, no prefill.",
        "",
        "## Current metrics",
        "",
        _markdown_metric_row(overall),
        "",
    ]
    if prefill_stats:
        total = int(prefill_stats.get("skip_tool", 0)) + int(prefill_stats.get("use_tool", 0))
        lines.extend(
            [
                "## Probe prefill decisions",
                "",
                f"- skip_tool: {prefill_stats.get('skip_tool', 0)}",
                f"- use_tool: {prefill_stats.get('use_tool', 0)}",
                f"- predicted tool-use rate: {_fmt_pct((prefill_stats.get('use_tool', 0) / total) if total else 0.0)}",
                "",
            ]
        )
    lines.extend(["## When2Tool paper reference", ""])
    if ref:
        lines.extend(
            [
                "- Reference uses hidden-state Probe&Prefill from the paper; current run uses CTD neuron features.",
                f"- Paper Base(Default): Acc={ref['base_accuracy_pct']:.1f}%, TotalTC={ref['base_total_tool_calls']}",
                f"- Paper Probe&Prefill tau={ref['threshold']}: Acc={ref['probe_accuracy_pct']:.1f}%, TotalTC={ref['probe_total_tool_calls']}, DeltaAcc={ref['delta_acc_pp']:.1f}pp, ToolCallReduction={ref['tool_call_reduction_percent']:.1f}%",
                "",
            ]
        )
    else:
        lines.extend(["- No exact paper reference for this model/subset/threshold/mode.", ""])
    lines.extend([metric_glossary_markdown(), ""])
    (out_dir / "metric_report.md").write_text("\n".join(lines), encoding="utf-8")
    print_eval_case_summary(
        summary=summary,
        model_alias=model_alias,
        subset=subset,
        method=method,
        threshold=threshold,
        temperature=temperature,
        prefill_mode=prefill_mode,
        ref=ref,
    )


def print_eval_case_summary(
    *,
    summary: dict[str, Any],
    model_alias: str,
    subset: str,
    method: str,
    threshold: float | None,
    temperature: float | None,
    prefill_mode: str | None,
    ref: dict[str, Any] | None,
) -> None:
    overall = section_metrics(summary, "overall")
    print(f"\n=== {method} metrics ===")
    print(f"Model/subset: {model_alias}/{subset}")
    print(f"tau={threshold} | T={temperature} | prefill_mode={prefill_mode} | deltas are PP - Base")
    print(
        "Overall: "
        f"Acc={_fmt_float(overall.get('final_accuracy'))} "
        f"TotalTC={_fmt_float(overall.get('total_tool_calls'), 2)} "
        f"AvgTC={_fmt_float(overall.get('avg_tool_calls'))} "
        f"TCR={_fmt_float(overall.get('tool_call_rate'))} "
        f"AvgTokenCost={_fmt_float(overall.get('avg_token_cost'))}"
    )
    print(
        "Decision: "
        f"DecisionAcc={_fmt_float(overall.get('decision_accuracy'))} "
        f"OverCall={_fmt_float(overall.get('over_call_rate'))} "
        f"UnderCall={_fmt_float(overall.get('under_call_rate'))} "
        f"ToolF1={_fmt_float(overall.get('tool_f1'))}"
    )
    if ref:
        print(
            "When2Tool paper reference: "
            f"PP Acc={ref['probe_accuracy_pct']:.1f}% TotalTC={ref['probe_total_tool_calls']} "
            f"vs Base Acc={ref['base_accuracy_pct']:.1f}% TotalTC={ref['base_total_tool_calls']}"
        )


def _markdown_metric_row(metrics: dict[str, Any]) -> str:
    headers = [
        "N",
        "FinalAcc",
        "TotalTC",
        "AvgTC",
        "TCR",
        "TotalTokenCost",
        "AvgTokenCost",
        "DecisionAcc",
        "OverCall",
        "UnderCall",
        "ToolF1",
    ]
    values = [
        _fmt_float(metrics.get("n"), 0),
        _fmt_float(metrics.get("final_accuracy")),
        _fmt_float(metrics.get("total_tool_calls"), 2),
        _fmt_float(metrics.get("avg_tool_calls")),
        _fmt_float(metrics.get("tool_call_rate")),
        _fmt_float(metrics.get("total_token_cost"), 2),
        _fmt_float(metrics.get("avg_token_cost")),
        _fmt_float(metrics.get("decision_accuracy")),
        _fmt_float(metrics.get("over_call_rate")),
        _fmt_float(metrics.get("under_call_rate")),
        _fmt_float(metrics.get("tool_f1")),
    ]
    return "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n| " + " | ".join(values) + " |"


def write_threshold_sweep_report(
    *,
    out_dir: Path,
    cases: list[tuple[float, Path]],
    model_alias: str,
    subset: str,
    prefill_mode: str,
    temperature: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for threshold, case_dir in cases:
        summary_path = case_dir / "summary.json"
        if not summary_path.exists():
            continue
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = section_metrics(summary, "overall")
        row = {
            "model_alias": model_alias,
            "subset": subset,
            "threshold": threshold,
            "temperature": temperature,
            "prefill_mode": prefill_mode,
        }
        for metric in [
            "n",
            "final_accuracy",
            "total_tool_calls",
            "avg_tool_calls",
            "tool_call_rate",
            "total_token_cost",
            "avg_token_cost",
            "decision_accuracy",
            "over_call_rate",
            "under_call_rate",
            "tool_precision",
            "tool_recall",
            "tool_f1",
        ]:
            row[metric] = overall.get(metric)
            if f"{metric}_std" in overall:
                row[f"{metric}_std"] = overall[f"{metric}_std"]
        ref = paper_eval_reference(model_alias, subset, threshold, prefill_mode)
        if ref:
            row["paper_probe_accuracy_pct"] = ref["probe_accuracy_pct"]
            row["paper_probe_total_tool_calls"] = ref["probe_total_tool_calls"]
            row["paper_delta_acc_pp"] = ref["delta_acc_pp"]
            row["paper_tool_call_reduction_percent"] = ref["tool_call_reduction_percent"]
        rows.append(row)
    rows.sort(key=lambda r: float(r["threshold"]))
    _write_csv(out_dir / "threshold_sweep_summary.csv", rows)
    _write_tradeoff_plot(out_dir / "threshold_tradeoff.png", rows, title=f"{model_alias}/{subset} threshold sweep")

    lines = [
        f"# Threshold sweep: {model_alias}/{subset}",
        "",
        f"- Temperature T: {temperature}",
        f"- Prefill mode: {prefill_mode}",
        "- Lower tau predicts tool-necessary more often; higher tau skips more tools.",
        "",
        "| tau | FinalAcc | TotalTC | AvgTC | TCR | DecisionAcc | OverCall | UnderCall | Paper Acc/TC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        paper = "NA"
        if row.get("paper_probe_accuracy_pct") is not None:
            paper = f"{float(row['paper_probe_accuracy_pct']):.1f}%/{int(row['paper_probe_total_tool_calls'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt_float(row.get("threshold"), 1),
                    _fmt_float(row.get("final_accuracy")),
                    _fmt_float(row.get("total_tool_calls"), 2),
                    _fmt_float(row.get("avg_tool_calls")),
                    _fmt_float(row.get("tool_call_rate")),
                    _fmt_float(row.get("decision_accuracy")),
                    _fmt_float(row.get("over_call_rate")),
                    _fmt_float(row.get("under_call_rate")),
                    paper,
                ]
            )
            + " |"
        )
    lines.extend(["", metric_glossary_markdown(), ""])
    (out_dir / "threshold_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")
    if rows:
        best = max(rows, key=lambda r: float(r.get("final_accuracy") or 0.0))
        print(
            f"Threshold sweep saved: {out_dir / 'threshold_sweep_summary.csv'} | "
            f"best Acc tau={best['threshold']} Acc={_fmt_float(best.get('final_accuracy'))} TotalTC={_fmt_float(best.get('total_tool_calls'), 2)}"
        )


def write_delta_sweep_report(
    *,
    out_dir: Path,
    comparisons: list[tuple[float, Path]],
    model_alias: str,
    subset: str,
    prefill_mode: str,
    temperature: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for threshold, csv_path in comparisons:
        if not csv_path.exists():
            continue
        overall = None
        for row in _read_csv(csv_path):
            if row.get("group_kind") == "overall" and row.get("group_name") == "overall":
                overall = row
                break
        if not overall:
            continue
        parsed: dict[str, Any] = {
            "model_alias": model_alias,
            "subset": subset,
            "threshold": threshold,
            "temperature": temperature,
            "prefill_mode": prefill_mode,
        }
        for key, value in overall.items():
            parsed[key] = _maybe_float(value)
        ref = paper_eval_reference(model_alias, subset, threshold, prefill_mode)
        if ref:
            parsed["paper_delta_acc_pp"] = ref["delta_acc_pp"]
            parsed["paper_tool_call_reduction_percent"] = ref["tool_call_reduction_percent"]
        rows.append(parsed)
    rows.sort(key=lambda r: float(r["threshold"]))
    _write_csv(out_dir / "delta_sweep_summary.csv", rows)
    _write_delta_plot(out_dir / "delta_tradeoff.png", rows, title=f"{model_alias}/{subset} delta vs Base")

    lines = [
        f"# Delta report: {model_alias}/{subset}",
        "",
        "- All deltas below are CTD-Probe&Prefill minus Base/Default on the same test ids.",
        "- Positive ToolCallReduction% means fewer total tool calls than Base.",
        "",
        "| tau | BaseAcc | PPAcc | DeltaAcc(pp) | BaseTC | PPTC | ToolCallReduction% | DeltaAvgTC | Cost | Paper DeltaAcc/Reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        paper = "NA"
        if row.get("paper_delta_acc_pp") is not None:
            paper = f"{float(row['paper_delta_acc_pp']):.1f}pp/{float(row['paper_tool_call_reduction_percent']):.1f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt_float(row.get("threshold"), 1),
                    _fmt_float(row.get("base_final_accuracy")),
                    _fmt_float(row.get("ctd_final_accuracy")),
                    _fmt_float(row.get("delta_acc_pp"), 2),
                    _fmt_float(row.get("base_total_tool_calls"), 2),
                    _fmt_float(row.get("ctd_total_tool_calls"), 2),
                    _fmt_float(row.get("tool_call_reduction_percent"), 2),
                    _fmt_float(row.get("delta_avg_tool_calls")),
                    _fmt_float(row.get("acc_cost_per_saved_call"), 2),
                    paper,
                ]
            )
            + " |"
        )
    lines.extend(["", metric_glossary_markdown(), ""])
    (out_dir / "delta_report.md").write_text("\n".join(lines), encoding="utf-8")
    if rows:
        best_reduction = max(rows, key=lambda r: float(r.get("tool_call_reduction_percent") or -1e9))
        print(
            f"Delta sweep saved: {out_dir / 'delta_sweep_summary.csv'} | "
            f"max reduction tau={best_reduction['threshold']} reduction={_fmt_float(best_reduction.get('tool_call_reduction_percent'), 2)}% "
            f"DeltaAcc={_fmt_float(best_reduction.get('delta_acc_pp'), 2)}pp"
        )


def _maybe_float(value: Any) -> Any:
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _write_tradeoff_plot(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        path.with_suffix(".warning.txt").write_text(f"Plot skipped: {exc}\n", encoding="utf-8")
        return
    if not rows:
        return
    x = [float(row.get("total_tool_calls") or 0.0) for row in rows]
    y = [float(row.get("final_accuracy") or 0.0) for row in rows]
    labels = [str(row.get("threshold")) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", color="#2f7ed8", linewidth=2)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(4, 5), fontsize=9)
    ax.set_xlabel("Total Tool Calls")
    ax.set_ylabel("Final Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if max(y) <= 1.0:
        ax.set_ylim(max(0.0, min(y) - 0.05), min(1.02, max(y) + 0.05))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_delta_plot(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        path.with_suffix(".warning.txt").write_text(f"Plot skipped: {exc}\n", encoding="utf-8")
        return
    if not rows:
        return
    x = [float(row.get("tool_call_reduction_percent") or 0.0) for row in rows]
    y = [float(row.get("delta_acc_pp") or 0.0) for row in rows]
    labels = [str(row.get("threshold")) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax.plot(x, y, marker="o", color="#1a9850", linewidth=2)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(4, 5), fontsize=9)
    ax.set_xlabel("ToolCallReduction% vs Base")
    ax.set_ylabel("DeltaAcc(pp) vs Base")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_probe_control_report(*, out_dir: Path, rows: list[dict[str, Any]], model_alias: str, subset: str) -> None:
    lines = [
        f"# Probe control report: {model_alias}/{subset}",
        "",
        "- Same train/test split and logistic probe settings are used for every feature set.",
        "- CTD should be compared against Random-CTD first; TDN and Private rows diagnose type-specific signal.",
        "",
        "| Feature set | Neurons | Test AUROC | Test Acc | Test Precision | Test Recall | Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("feature_set")),
                    _fmt_float(row.get("neuron_count"), 0),
                    _fmt_float(row.get("test_auroc")),
                    _fmt_float(row.get("test_accuracy")),
                    _fmt_float(row.get("test_precision")),
                    _fmt_float(row.get("test_recall")),
                    _fmt_float(row.get("test_f1")),
                ]
            )
            + " |"
        )
    lines.extend(["", metric_glossary_markdown(), ""])
    (out_dir / "probe_control_report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_probe_control_plot(out_dir / "probe_control_bars.png", rows, title=f"{model_alias}/{subset} probe controls")
    print(f"Probe control report saved: {out_dir / 'probe_control_report.md'}")


def _write_probe_control_plot(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        path.with_suffix(".warning.txt").write_text(f"Plot skipped: {exc}\n", encoding="utf-8")
        return
    if not rows:
        return
    labels = [str(row.get("feature_set")) for row in rows]
    aurocs = [float(row.get("test_auroc") or 0.0) for row in rows]
    accs = [float(row.get("test_accuracy") or 0.0) for row in rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    ax.bar(x - width / 2, aurocs, width, label="AUROC", color="#4c78a8")
    ax.bar(x + width / 2, accs, width, label="Accuracy", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_activation_mask_report(
    *,
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    model_alias: str,
    subset: str,
) -> None:
    lines = [
        f"# Activation mask report: {model_alias}/{subset}",
        "",
        "- Base uses no mask. Other interventions zero selected FFN output coordinates during generation.",
        "- Delta columns below are intervention minus Base, averaged across task types.",
        "",
        "| Intervention | Avg DeltaAcc | Var DeltaAcc | Avg DeltaTCR | Acc A | Acc B | Acc C |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cross_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("intervention")),
                    _fmt_float(row.get("avg_delta_acc")),
                    _fmt_float(row.get("var_acc")),
                    _fmt_float(row.get("avg_delta_tcr")),
                    _fmt_float(row.get("acc_A")),
                    _fmt_float(row.get("acc_B")),
                    _fmt_float(row.get("acc_C")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per task type",
            "",
            "| Task type | Intervention | Masked neurons | FinalAcc | TotalTC | AvgTC | TCR | DecisionAcc |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task_type")),
                    str(row.get("intervention")),
                    _fmt_float(row.get("masked_neurons"), 0),
                    _fmt_float(row.get("final_accuracy")),
                    _fmt_float(row.get("total_tool_calls"), 2),
                    _fmt_float(row.get("avg_tool_calls")),
                    _fmt_float(row.get("tool_call_rate")),
                    _fmt_float(row.get("decision_accuracy")),
                ]
            )
            + " |"
        )
    lines.extend(["", metric_glossary_markdown(), ""])
    (out_dir / "activation_mask_report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_activation_mask_plot(out_dir / "activation_mask_delta.png", cross_rows, title=f"{model_alias}/{subset} activation mask")
    print(f"Activation mask report saved: {out_dir / 'activation_mask_report.md'}")


def _write_activation_mask_plot(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        path.with_suffix(".warning.txt").write_text(f"Plot skipped: {exc}\n", encoding="utf-8")
        return
    if not rows:
        return
    labels = [str(row.get("intervention")) for row in rows]
    delta_acc = [float(row.get("avg_delta_acc") or 0.0) for row in rows]
    delta_tcr = [float(row.get("avg_delta_tcr") or 0.0) for row in rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.8))
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.bar(x - width / 2, delta_acc, width, label="Avg DeltaAcc", color="#4c78a8")
    ax.bar(x + width / 2, delta_tcr, width, label="Avg DeltaTCR", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
