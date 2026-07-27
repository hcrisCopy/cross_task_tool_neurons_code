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
    ("AUROC", "探针指标，ROC 曲线下面积，衡量 P(tool_necessary) 的排序能力；不依赖阈值。"),
    ("Accuracy", "探针阶段表示按当前阈值 tau 得到的二分类准确率；生成评测阶段表示最终答案准确率。"),
    ("Precision", "探针/工具决策中，预测需要工具的样本里真正需要工具的比例。"),
    ("Recall", "探针/工具决策中，真正需要工具的样本里被识别出来的比例。"),
    ("F1", "Precision 和 Recall 的调和平均。"),
    ("Final Accuracy", "When2Tool item_final_eval 判断的最终答案正确率：correct / N。"),
    ("Total Tool Calls", "所有样本的有效工具调用次数总和。"),
    ("Avg Tool Calls", "平均每题工具调用次数：Total Tool Calls / N。"),
    ("Tool Call Rate", "工具调用率：sum(tool_calls_i) / sum(expected_steps_i)；single-hop 通常 expected_steps=1，multi-hop 通常 expected_steps=3。"),
    ("Total Token Cost", "总代价：generation_tokens + 0.2 * prefill_tokens，严格沿用 When2Tool 代码。"),
    ("Avg Token Cost", "平均每题 token cost：Total Token Cost / N。"),
    ("DecisionAcc", "工具决策准确率：actual_tool_call 是否等于 tool_necessary。"),
    ("OverCall", "过调用率：tool_necessary=0 时仍调用工具的比例。"),
    ("UnderCall", "漏调用率：tool_necessary=1 时没有调用工具的比例。"),
    ("ToolPrecision/Recall/F1", "把 actual_tool_call 当作预测、tool_necessary 当作标签得到的工具决策指标。"),
    ("ValidToolCallRate", "有效工具调用比例：tool_calls / (tool_calls + invalid_tool_attempts)。"),
    ("ToolTrajectorySuccessRate", "发生过有效工具调用的样本里，最终答案正确的比例。"),
    ("DeltaAcc(pp)", "百分点差值：100 * (Acc_PP - Acc_Base)。"),
    ("DeltaAvgTC", "平均工具调用差：AvgTC_PP - AvgTC_Base；负数表示比 Base 少调用工具。"),
    ("DeltaTCR", "工具调用率差：TCR_PP - TCR_Base；负数表示比 Base 工具调用率低。"),
    ("DeltaTC%", "总工具调用相对变化：100 * (TC_PP - TC_Base) / TC_Base。"),
    ("ToolCallReduction%", "工具调用减少比例：-DeltaTC%；正数表示相对 Base 节省了工具调用。"),
    ("Cost", "每节省 1 次平均工具调用付出的准确率变化：DeltaAcc(pp) / (-DeltaAvgTC)，只有 DeltaAvgTC<0 时有意义。"),
]


WHEN2TOOL_COST_AVG_REFERENCE = {
    "easy": {"delta_acc_pp": -1.1, "delta_avg_tool_calls": -0.66, "cost": -1.6},
    "medium": {"delta_acc_pp": -3.4, "delta_avg_tool_calls": -0.54, "cost": -6.2},
    "hard": {"delta_acc_pp": -0.8, "delta_avg_tool_calls": -0.24, "cost": -3.4},
    "overall": {"delta_acc_pp": -1.7, "delta_avg_tool_calls": -0.48, "cost": -3.6},
}


WHEN2TOOL_COST_MODEL_REFERENCE = {
    "qwen3-1.7b": {"delta_acc_pp": 0.1, "delta_avg_tool_calls": -0.26, "cost": 0.3},
    "qwen3-4b-instruct": {"delta_acc_pp": -0.7, "delta_avg_tool_calls": -0.36, "cost": -1.9},
    "qwen3-14b": {"delta_acc_pp": -1.3, "delta_avg_tool_calls": -0.44, "cost": -2.9},
    "qwen3-32b": {"delta_acc_pp": -4.0, "delta_avg_tool_calls": -0.56, "cost": -7.2},
    "llama3.1-8b": {"delta_acc_pp": -9.8, "delta_avg_tool_calls": -0.59, "cost": -16.6},
    "llama3.3-70b": {"delta_acc_pp": 5.2, "delta_avg_tool_calls": -0.67, "cost": 7.8},
}


WHEN2TOOL_MULTI_SUMMARY_REFERENCE = {
    "qwen3-1.7b": {"default_acc_pct": 21.2, "default_tc": 1180, "probe_acc_pct": 59.2, "probe_tc": 85, "probe_delta_tc_pct": -93.0},
    "qwen3-4b-instruct": {"default_acc_pct": 82.1, "default_tc": 1719, "probe_acc_pct": 85.3, "probe_tc": 437, "probe_delta_tc_pct": -75.0},
    "qwen3-14b": {"default_acc_pct": 87.5, "default_tc": 1503, "probe_acc_pct": 86.2, "probe_tc": 996, "probe_delta_tc_pct": -34.0},
    "qwen3-32b": {"default_acc_pct": 88.9, "default_tc": 1634, "probe_acc_pct": 89.0, "probe_tc": 727, "probe_delta_tc_pct": -55.0},
    "llama3.1-8b": {"default_acc_pct": 40.2, "default_tc": 1005, "probe_acc_pct": 60.2, "probe_tc": 1361, "probe_delta_tc_pct": 35.0},
    "llama3.3-70b": {"default_acc_pct": 62.4, "default_tc": 985, "probe_acc_pct": 80.3, "probe_tc": 1789, "probe_delta_tc_pct": 82.0},
}


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


def _fmt_pp(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}f}pp"
    except (TypeError, ValueError):
        return str(value)


def _fmt_acc_pct(value: Any, digits: int = 4, pct_digits: int = 2) -> str:
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    if abs(number) <= 1.0:
        return f"{number:.{digits}f} ({100.0 * number:.{pct_digits}f}%)"
    return f"{number:.{digits}f}"


def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    sep = "-+-".join("-" * width for width in widths)
    print(header)
    print(sep)
    for row in rendered:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def _print_metric_glossary(scope: str) -> None:
    scope_keys = {
        "probe": {"AUROC", "Accuracy", "Precision", "Recall", "F1"},
        "eval": {
            "Final Accuracy",
            "Total Tool Calls",
            "Avg Tool Calls",
            "Tool Call Rate",
            "Total Token Cost",
            "Avg Token Cost",
            "DecisionAcc",
            "OverCall",
            "UnderCall",
            "ToolPrecision/Recall/F1",
            "ValidToolCallRate",
            "ToolTrajectorySuccessRate",
        },
        "delta": {"DeltaAcc(pp)", "DeltaAvgTC", "DeltaTCR", "DeltaTC%", "ToolCallReduction%", "Cost"},
    }
    wanted = scope_keys.get(scope, {name for name, _meaning in METRIC_GLOSSARY})
    print("\n关键指标解释（中文）:")
    for name, meaning in METRIC_GLOSSARY:
        if name in wanted:
            print(f"  - {name}: {meaning}")


def _display_name(model_alias: str) -> str:
    return MODEL_DISPLAY.get(model_alias, model_alias)


def paper_probe_table_name(subset: str) -> str:
    return "表3（single-hop hidden-state probe）" if subset == "single_hop" else "表10（multi-hop hidden-state probe）"


def paper_eval_table_name(subset: str) -> str:
    return "表8（single-hop 全设置 Acc/TC）" if subset == "single_hop" else "表11（multi-hop 全设置 Acc/TC）"


def paper_delta_table_name(subset: str) -> str:
    return "表4/表7（相对 Default 的 accuracy cost per saved call）" if subset == "single_hop" else "表9/表11（multi-hop 相对 Default 的工具调用变化）"


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


def paper_base_reference(model_alias: str, subset: str) -> dict[str, Any] | None:
    block = WHEN2TOOL_EVAL_REFERENCE.get(subset)
    if not block:
        return None
    base = block["base_default"].get(model_alias)
    if not base:
        return None
    n = int(block["n"])
    return {
        "n": n,
        "accuracy_pct": float(base["accuracy_pct"]),
        "total_tool_calls": int(base["total_tool_calls"]),
        "avg_tool_calls": float(base["total_tool_calls"]) / n,
    }


def paper_threshold_rows(model_alias: str, subset: str, prefill_mode: str) -> list[dict[str, Any]]:
    block = WHEN2TOOL_EVAL_REFERENCE.get(subset)
    if not block:
        return []
    base = block["base_default"].get(model_alias)
    if not base:
        return []
    mode = prefill_mode if subset == "single_hop" else "paper_default"
    probe_table = block["probe_prefill"].get(mode, {}).get(model_alias, {})
    if not probe_table:
        return []
    n = int(block["n"])
    base_tc = float(base["total_tool_calls"])
    base_acc = float(base["accuracy_pct"])
    rows = []
    for threshold in sorted(probe_table):
        acc_pct, total_tc = probe_table[threshold]
        delta_acc_pp = float(acc_pct) - base_acc
        delta_avg_tc = (float(total_tc) - base_tc) / n
        reduction = -100.0 * (float(total_tc) - base_tc) / max(base_tc, 1.0)
        cost = delta_acc_pp / (-delta_avg_tc) if delta_avg_tc < 0 else None
        rows.append(
            {
                "threshold": float(threshold),
                "prefill_mode": mode,
                "accuracy_pct": float(acc_pct),
                "total_tool_calls": int(total_tc),
                "avg_tool_calls": float(total_tc) / n,
                "delta_acc_pp": delta_acc_pp,
                "delta_avg_tool_calls": delta_avg_tc,
                "tool_call_reduction_percent": reduction,
                "cost": cost,
            }
        )
    return rows


def metric_glossary_markdown() -> str:
    lines = ["### 指标解释", "", "| Metric | 中文含义 |", "|---|---|"]
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
    print("\n=== PP-2 探针训练指标（CTD 神经元特征）===")
    print(f"当前条件: model={model_alias} ({_display_name(str(model_alias))}), subset={subset}, feature_dim={results.get('feature_dim')}")
    print(
        "训练设置: StandardScaler + L2 LogisticRegression, "
        f"reg={results.get('reg')} -> C={results.get('C')}, "
        f"max_iter={results.get('max_iter', 'unknown')}, 报告阈值 tau={results.get('threshold')}"
    )
    print("标签定义: tool_necessary=1 表示当前模型无工具答错，所以工具必要；tool_necessary=0 表示无工具答对。")
    print(f"论文阶段: When2Tool Section 4 Probing Analysis，对应 {paper_probe_table_name(str(subset))}；论文特征是 all-layer pre-generation hidden states，本项目当前特征是 CTD FFN activation。")
    table_rows = [
        [
            "本实验 CTD probe",
            _fmt_float(overall.get("auroc")),
            _fmt_acc_pct(overall.get("accuracy")),
            _fmt_float(overall.get("precision")),
            _fmt_float(overall.get("recall")),
            _fmt_float(overall.get("f1")),
        ]
    ]
    if ref:
        table_rows.append(
            [
                f"When2Tool {paper_probe_table_name(str(subset))}",
                _fmt_float(ref.get("auroc")),
                _fmt_acc_pct(ref.get("accuracy")),
                "NA",
                "NA",
                "NA",
            ]
        )
    _print_table(["来源", "AUROC", "Accuracy", "Precision", "Recall", "F1"], table_rows)
    if ref and str(subset) == "single_hop":
        _print_table(
            ["论文 AUROC 分难度", "easy", "medium", "hard"],
            [[paper_probe_table_name(str(subset)), _fmt_float(ref.get("easy_auroc")), _fmt_float(ref.get("medium_auroc")), _fmt_float(ref.get("hard_auroc"))]],
        )
    print("论文补充表: 表14说明 all-layer hidden-state probe 通常优于单层；表16说明 lambda=10000 是论文默认且整体稳定。")
    _print_metric_glossary("probe")
    print("输出文件: probe_report.md, probe_metrics_table.csv, probe_probability_hist.png, probe_roc_curve.png")


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
    comparison_thresholds: list[float] | None = None,
    comparison_temperature: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
        comparison_thresholds=comparison_thresholds,
        comparison_temperature=comparison_temperature,
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
    comparison_thresholds: list[float] | None = None,
    comparison_temperature: float | None = None,
) -> None:
    overall = section_metrics(summary, "overall")
    base_ref = paper_base_reference(model_alias, subset)
    print(f"\n=== 生成评测指标：{method} ===")
    print(f"当前条件: model={model_alias} ({_display_name(model_alias)}), subset={subset}, prompt_mode=current, reasoning_mode=no_reasoning")
    if threshold is None:
        cmp_tau = ",".join(f"{x:g}" for x in comparison_thresholds or [])
        print(
            "Base/Default 自身不使用 probe 阈值和 prefill；"
            f"本命令后续用于对比的 Probe&Prefill 阈值 tau=[{cmp_tau or '未传入'}], 温度 T={comparison_temperature if comparison_temperature is not None else '未传入'}。"
        )
    else:
        print(f"Probe&Prefill 设置: tau={threshold:g}, 温度 T={temperature}, prefill_mode={prefill_mode}。")
    print("delta 口径提醒: 后续 comparison/delta 一律是 CTD-Probe&Prefill - Base/Default。")
    _print_table(
        ["当前结果", "N", "FinalAcc", "TotalTC", "AvgTC", "TCR", "AvgTokenCost"],
        [
            [
                method,
                _fmt_float(overall.get("n"), 0),
                _fmt_acc_pct(overall.get("final_accuracy")),
                _fmt_float(overall.get("total_tool_calls"), 2),
                _fmt_float(overall.get("avg_tool_calls")),
                _fmt_float(overall.get("tool_call_rate")),
                _fmt_float(overall.get("avg_token_cost")),
            ]
        ],
    )
    print("神经元方案额外诊断指标（论文主表不直接报告，用于检查工具决策质量）:")
    _print_table(
        ["DecisionAcc", "OverCall", "UnderCall", "ToolPrecision", "ToolRecall", "ToolF1", "ValidToolCallRate", "TrajectorySuccess"],
        [
            [
                _fmt_float(overall.get("decision_accuracy")),
                _fmt_float(overall.get("over_call_rate")),
                _fmt_float(overall.get("under_call_rate")),
                _fmt_float(overall.get("tool_precision")),
                _fmt_float(overall.get("tool_recall")),
                _fmt_float(overall.get("tool_f1")),
                _fmt_float(overall.get("valid_tool_call_rate")),
                _fmt_float(overall.get("tool_trajectory_success_rate")),
            ]
        ],
    )
    print(f"论文阶段: When2Tool Section 5 Probe&Prefill，对应 {paper_eval_table_name(subset)}；single-hop 还对应表12（soft/hard, T=2.0）和表13（temperature scaling）。")
    if threshold is None and base_ref:
        _print_table(
            ["论文参考", "N", "tau", "T", "设置", "Acc", "TotalTC", "AvgTC"],
            [
                [
                    f"{paper_eval_table_name(subset)} Prompt-only Default",
                    base_ref["n"],
                    "无",
                    "无",
                    "no prefill",
                    f"{base_ref['accuracy_pct']:.1f}%",
                    base_ref["total_tool_calls"],
                    _fmt_float(base_ref["avg_tool_calls"]),
                ]
            ],
        )
    elif ref:
        _print_table(
            ["论文参考", "tau", "T", "prefill", "BaseAcc", "BaseTC", "PPAcc", "PPTC", "DeltaAcc", "ToolCallReduction"],
            [
                [
                    paper_eval_table_name(subset),
                    _fmt_float(ref["threshold"], 1),
                    "2.0",
                    ref["prefill_mode"],
                    f"{ref['base_accuracy_pct']:.1f}%",
                    ref["base_total_tool_calls"],
                    f"{ref['probe_accuracy_pct']:.1f}%",
                    ref["probe_total_tool_calls"],
                    _fmt_pp(ref["delta_acc_pp"], 1),
                    f"{ref['tool_call_reduction_percent']:.1f}%",
                ]
            ],
        )
    else:
        print("论文参考: 当前模型/subset/tau/prefill_mode 没有内置可直接对照的 When2Tool 表格行。")
    _print_metric_glossary("eval")


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
        fewest_calls = min(rows, key=lambda r: float(r.get("total_tool_calls") or 1e18))
        print(f"\n=== PP-3 阈值 sweep 指标：{model_alias}/{subset} ===")
        print(f"当前条件: prefill_mode={prefill_mode}, 温度 T={temperature}, tau 列表={[row['threshold'] for row in rows]}")
        print("论文阶段: When2Tool Section 5 Probe&Prefill，主表是 " + paper_eval_table_name(subset) + "；single-hop 的 hard prefill 对应表12，温度设置对照表13。")
        print("可直接对比论文的主指标:")
        _print_table(
            [
                "tau",
                "本实验 FinalAcc",
                "本实验 TotalTC",
                "本实验 AvgTC",
                "论文 PPAcc",
                "论文 PPTC",
                "论文 DeltaAcc",
                "论文 ToolCallReduction",
            ],
            [
                [
                    _fmt_float(row.get("threshold"), 1),
                    _fmt_acc_pct(row.get("final_accuracy")),
                    _fmt_float(row.get("total_tool_calls"), 2),
                    _fmt_float(row.get("avg_tool_calls")),
                    f"{float(row['paper_probe_accuracy_pct']):.1f}%" if row.get("paper_probe_accuracy_pct") is not None else "NA",
                    int(row["paper_probe_total_tool_calls"]) if row.get("paper_probe_total_tool_calls") is not None else "NA",
                    _fmt_pp(row.get("paper_delta_acc_pp"), 1),
                    f"{float(row['paper_tool_call_reduction_percent']):.1f}%" if row.get("paper_tool_call_reduction_percent") is not None else "NA",
                ]
                for row in rows
            ],
        )
        print("神经元方案额外诊断指标（不作为论文主表直接对比）:")
        _print_table(
            ["tau", "DecisionAcc", "OverCall", "UnderCall", "ToolPrecision", "ToolRecall", "ToolF1"],
            [
                [
                    _fmt_float(row.get("threshold"), 1),
                    _fmt_float(row.get("decision_accuracy")),
                    _fmt_float(row.get("over_call_rate")),
                    _fmt_float(row.get("under_call_rate")),
                    _fmt_float(row.get("tool_precision")),
                    _fmt_float(row.get("tool_recall")),
                    _fmt_float(row.get("tool_f1")),
                ]
                for row in rows
            ],
        )
        paper_rows = paper_threshold_rows(model_alias, subset, prefill_mode)
        base_ref = paper_base_reference(model_alias, subset)
        if base_ref:
            print("When2Tool 论文 Base/Default 参考:")
            _print_table(
                ["论文表", "N", "tau", "T", "Acc", "TotalTC", "AvgTC"],
                [[paper_eval_table_name(subset), base_ref["n"], "无", "无", f"{base_ref['accuracy_pct']:.1f}%", base_ref["total_tool_calls"], _fmt_float(base_ref["avg_tool_calls"])]],
            )
        if paper_rows:
            print("When2Tool 论文完整阈值参考（同一模型；论文特征为 hidden-state probe，本实验特征为 CTD 神经元）:")
            _print_table(
                ["论文表", "tau", "T", "prefill", "PPAcc", "PPTC", "AvgTC", "DeltaAcc", "ToolCallReduction"],
                [
                    [
                        paper_eval_table_name(subset),
                        _fmt_float(row["threshold"], 1),
                        "2.0",
                        row["prefill_mode"],
                        f"{row['accuracy_pct']:.1f}%",
                        row["total_tool_calls"],
                        _fmt_float(row["avg_tool_calls"]),
                        _fmt_pp(row["delta_acc_pp"], 1),
                        f"{row['tool_call_reduction_percent']:.1f}%",
                    ]
                    for row in paper_rows
                ],
            )
        print(
            f"当前 sweep 最优准确率: tau={best['threshold']} FinalAcc={_fmt_acc_pct(best.get('final_accuracy'))}, "
            f"TotalTC={_fmt_float(best.get('total_tool_calls'), 2)}；最少工具调用: tau={fewest_calls['threshold']} "
            f"TotalTC={_fmt_float(fewest_calls.get('total_tool_calls'), 2)}, FinalAcc={_fmt_acc_pct(fewest_calls.get('final_accuracy'))}。"
        )
        print(f"输出文件: {out_dir / 'threshold_sweep_summary.csv'}, {out_dir / 'threshold_sweep_report.md'}, {out_dir / 'threshold_tradeoff.png'}")
        _print_metric_glossary("eval")


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
        best_acc_delta = max(rows, key=lambda r: float(r.get("delta_acc_pp") or -1e18))
        print(f"\n=== PP-4 delta 指标：{model_alias}/{subset} ===")
        print(f"当前条件: prefill_mode={prefill_mode}, 温度 T={temperature}, tau 列表={[row['threshold'] for row in rows]}")
        print("delta 基准: 同模型、同 subset、同一批 test id、同一套 When2Tool prompt/tool/parser/generation 参数的 Base/Default；全部 delta = CTD-Probe&Prefill - Base/Default。")
        print("可直接对比 When2Tool 论文的主 delta 指标:")
        _print_table(
            [
                "tau",
                "BaseAcc",
                "PPAcc",
                "DeltaAcc",
                "BaseTC",
                "PPTC",
                "ToolCallReduction",
                "DeltaAvgTC",
                "Cost",
                "论文 DeltaAcc/Reduction",
            ],
            [
                [
                    _fmt_float(row.get("threshold"), 1),
                    _fmt_acc_pct(row.get("base_final_accuracy")),
                    _fmt_acc_pct(row.get("ctd_final_accuracy")),
                    _fmt_pp(row.get("delta_acc_pp"), 2),
                    _fmt_float(row.get("base_total_tool_calls"), 2),
                    _fmt_float(row.get("ctd_total_tool_calls"), 2),
                    f"{_fmt_float(row.get('tool_call_reduction_percent'), 2)}%",
                    _fmt_float(row.get("delta_avg_tool_calls")),
                    _fmt_float(row.get("acc_cost_per_saved_call"), 2),
                    (
                        f"{float(row['paper_delta_acc_pp']):.1f}pp/{float(row['paper_tool_call_reduction_percent']):.1f}%"
                        if row.get("paper_delta_acc_pp") is not None
                        else "NA"
                    ),
                ]
                for row in rows
            ],
        )
        print("神经元方案额外诊断 delta（论文主表不直接报告，用于定位工具决策变化）:")
        _print_table(
            ["tau", "DeltaDecisionAcc", "DeltaOverCall", "DeltaUnderCall", "DeltaToolPrecision", "DeltaToolRecall", "DeltaToolF1"],
            [
                [
                    _fmt_float(row.get("threshold"), 1),
                    _fmt_pp(row.get("delta_decision_accuracy_pp"), 2),
                    _fmt_pp(row.get("delta_over_call_rate_pp"), 2),
                    _fmt_pp(row.get("delta_under_call_rate_pp"), 2),
                    _fmt_pp(row.get("delta_tool_precision_pp"), 2),
                    _fmt_pp(row.get("delta_tool_recall_pp"), 2),
                    _fmt_pp(row.get("delta_tool_f1_pp"), 2),
                ]
                for row in rows
            ],
        )
        paper_rows = paper_threshold_rows(model_alias, subset, prefill_mode)
        if paper_rows:
            print(f"When2Tool 论文阈值 delta 参考（{paper_eval_table_name(subset)}，论文特征为 hidden-state probe）:")
            _print_table(
                ["tau", "T", "prefill", "PPAcc", "PPTC", "DeltaAcc", "DeltaAvgTC", "ToolCallReduction", "Cost"],
                [
                    [
                        _fmt_float(row["threshold"], 1),
                        "2.0",
                        row["prefill_mode"],
                        f"{row['accuracy_pct']:.1f}%",
                        row["total_tool_calls"],
                        _fmt_pp(row["delta_acc_pp"], 1),
                        _fmt_float(row["delta_avg_tool_calls"]),
                        f"{row['tool_call_reduction_percent']:.1f}%",
                        _fmt_float(row["cost"], 1),
                    ]
                    for row in paper_rows
                ],
            )
        if subset == "single_hop":
            print("When2Tool 表4参考（六个模型平均，按难度；tau=0.5 的 Probe&Prefill 相对 Default）:")
            _print_table(
                ["难度", "DeltaAcc", "DeltaAvgTC", "Cost"],
                [
                    [difficulty, _fmt_pp(ref["delta_acc_pp"], 1), _fmt_float(ref["delta_avg_tool_calls"], 2), _fmt_float(ref["cost"], 1)]
                    for difficulty, ref in WHEN2TOOL_COST_AVG_REFERENCE.items()
                ],
            )
            model_ref = WHEN2TOOL_COST_MODEL_REFERENCE.get(model_alias)
            if model_ref:
                print("When2Tool 表7参考（当前模型，single-hop，tau=0.5，相对 Default）:")
                _print_table(
                    ["模型", "DeltaAcc", "DeltaAvgTC", "Cost"],
                    [[_display_name(model_alias), _fmt_pp(model_ref["delta_acc_pp"], 1), _fmt_float(model_ref["delta_avg_tool_calls"], 2), _fmt_float(model_ref["cost"], 1)]],
                )
        else:
            multi_ref = WHEN2TOOL_MULTI_SUMMARY_REFERENCE.get(model_alias)
            if multi_ref:
                print("When2Tool 表9参考（当前模型，multi-hop，Best Probe&Prefill 相对 Default）:")
                _print_table(
                    ["模型", "DefaultAcc", "DefaultTC", "BestPPAcc", "BestPPTC", "DeltaTC"],
                    [
                        [
                            _display_name(model_alias),
                            f"{multi_ref['default_acc_pct']:.1f}%",
                            multi_ref["default_tc"],
                            f"{multi_ref['probe_acc_pct']:.1f}%",
                            multi_ref["probe_tc"],
                            f"{multi_ref['probe_delta_tc_pct']:.1f}%",
                        ]
                    ],
                )
        print(
            f"当前 sweep 最大工具节省: tau={best_reduction['threshold']} ToolCallReduction="
            f"{_fmt_float(best_reduction.get('tool_call_reduction_percent'), 2)}%, DeltaAcc={_fmt_pp(best_reduction.get('delta_acc_pp'), 2)}；"
            f"最大准确率增量: tau={best_acc_delta['threshold']} DeltaAcc={_fmt_pp(best_acc_delta.get('delta_acc_pp'), 2)}。"
        )
        print(f"输出文件: {out_dir / 'delta_sweep_summary.csv'}, {out_dir / 'delta_report.md'}, {out_dir / 'delta_tradeoff.png'}")
        _print_metric_glossary("delta")


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
