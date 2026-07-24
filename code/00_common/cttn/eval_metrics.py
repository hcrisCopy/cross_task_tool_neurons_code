from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def task_env_signature(item: dict[str, Any]) -> str:
    names = sorted({env.get("name", "") for env in item.get("environments", []) if env.get("name", "")})
    return "+".join(names) if names else str(item.get("env_name", "no_env"))


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _trace_invalid_tool_attempts(item: dict[str, Any]) -> int:
    count = 0
    for trace in item.get("trace", []) or []:
        raw = str(trace.get("model_raw_output", ""))
        parsed = trace.get("parsed_output", {}) or {}
        tool_result = trace.get("tool_result", None)
        looked_like_tool = "<tool_call>" in raw or ('"name"' in raw and "arguments" in raw)
        parse_failed = looked_like_tool and parsed.get("type") != "tool"
        execution_failed = isinstance(tool_result, dict) and tool_result.get("success") is False
        if parse_failed or execution_failed:
            count += 1
    return count


def build_per_task(outputs: list[dict[str, Any]], w2t_utils: Any, *, run_id: int = 0) -> list[dict[str, Any]]:
    rows = []
    for item in outputs:
        raw, boxed, cleaned, final_correct = w2t_utils.item_final_eval(item)
        tool_calls = int(w2t_utils.item_tool_calls(item))
        expected_steps = int(w2t_utils.item_expected_steps(item))
        y = int(item.get("tool_necessary", 0))
        called = int(tool_calls > 0)
        rows.append(
            {
                "run_id": run_id,
                "id": str(item.get("id")),
                "subset": item.get("subset", ""),
                "task_type": item.get("task_type", ""),
                "env_name": task_env_signature(item),
                "difficulty": item.get("difficulty", "unknown"),
                "tool_necessary": y,
                "actual_tool_call": called,
                "final_correct": int(bool(final_correct)),
                "model_answer_raw": raw,
                "model_boxed_content": boxed,
                "model_answer": cleaned,
                "ground_truth": (item.get("expected", {}) or {}).get("answer", ""),
                "tool_calls": tool_calls,
                "expected_steps": expected_steps,
                "tool_call_rate": _safe_div(tool_calls, expected_steps),
                "generation_tokens": int(item.get("generation_tokens", 0) or 0),
                "prefill_tokens": int(item.get("prefill_tokens", 0) or 0),
                "token_cost": float(item.get("token_cost", 0.0) or 0.0),
                "rounds": int(item.get("rounds", 0) or 0),
                "invalid_tool_attempts": _trace_invalid_tool_attempts(item),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    final_correct = sum(int(row["final_correct"]) for row in rows)
    tool_calls = sum(int(row["tool_calls"]) for row in rows)
    expected_steps = sum(int(row["expected_steps"]) for row in rows)
    token_cost = sum(float(row["token_cost"]) for row in rows)
    generation_tokens = sum(int(row["generation_tokens"]) for row in rows)
    prefill_tokens = sum(int(row["prefill_tokens"]) for row in rows)
    invalid_tool_attempts = sum(int(row["invalid_tool_attempts"]) for row in rows)

    tp = sum(1 for row in rows if int(row["tool_necessary"]) == 1 and int(row["actual_tool_call"]) == 1)
    tn = sum(1 for row in rows if int(row["tool_necessary"]) == 0 and int(row["actual_tool_call"]) == 0)
    fp = sum(1 for row in rows if int(row["tool_necessary"]) == 0 and int(row["actual_tool_call"]) == 1)
    fn = sum(1 for row in rows if int(row["tool_necessary"]) == 1 and int(row["actual_tool_call"]) == 0)
    y0 = tn + fp
    y1 = tp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    called_rows = [row for row in rows if int(row["actual_tool_call"]) == 1]

    return {
        "n": n,
        "final_correct": final_correct,
        "final_accuracy": _safe_div(final_correct, n),
        "total_tool_calls": tool_calls,
        "avg_tool_calls": _safe_div(tool_calls, n),
        "tool_call_rate": _safe_div(tool_calls, expected_steps),
        "total_token_cost": token_cost,
        "avg_token_cost": _safe_div(token_cost, n),
        "total_generation_tokens": generation_tokens,
        "total_prefill_tokens": prefill_tokens,
        "decision_accuracy": _safe_div(tp + tn, n),
        "over_call_rate": _safe_div(fp, y0),
        "under_call_rate": _safe_div(fn, y1),
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
        "tool_true_positive": tp,
        "tool_true_negative": tn,
        "tool_false_positive": fp,
        "tool_false_negative": fn,
        "invalid_tool_attempts": invalid_tool_attempts,
        "valid_tool_call_rate": _safe_div(tool_calls, tool_calls + invalid_tool_attempts),
        "tool_trajectory_success_rate": _safe_div(
            sum(int(row["final_correct"]) for row in called_rows),
            len(called_rows),
        ),
    }


def group_rows(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return dict(grouped)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": aggregate(rows),
        "by_task_type": {key: aggregate(value) for key, value in sorted(group_rows(rows, "task_type").items())},
        "by_env": {key: aggregate(value) for key, value in sorted(group_rows(rows, "env_name").items())},
        "by_difficulty": {key: aggregate(value) for key, value in sorted(group_rows(rows, "difficulty").items())},
        "by_tool_necessary": {
            key: aggregate(value) for key, value in sorted(group_rows(rows, "tool_necessary").items())
        },
    }


def flatten_summary(
    summary: dict[str, Any],
    *,
    model_alias: str,
    subset: str,
    method: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    base = {"model_alias": model_alias, "subset": subset, "method": method}
    if extra:
        base.update(extra)

    def add(group_kind: str, group_name: str, metrics: dict[str, Any]) -> None:
        row = dict(base)
        row.update({"group_kind": group_kind, "group_name": group_name})
        row.update(metrics)
        rows.append(row)

    add("overall", "overall", summary.get("overall", {}))
    for kind in ["by_task_type", "by_env", "by_difficulty", "by_tool_necessary"]:
        group_kind = kind.replace("by_", "")
        for name, metrics in summary.get(kind, {}).items():
            add(group_kind, str(name), metrics)
    return rows


def flatten_mean_std_summary(
    summary: dict[str, Any],
    *,
    model_alias: str,
    subset: str,
    method: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    base = {"model_alias": model_alias, "subset": subset, "method": method}
    if extra:
        base.update(extra)

    def add(group_kind: str, group_name: str, metrics: dict[str, Any]) -> None:
        row = dict(base)
        row.update({"group_kind": group_kind, "group_name": group_name})
        for metric_name, stat in metrics.items():
            if isinstance(stat, dict):
                if "mean" in stat:
                    row[f"{metric_name}_mean"] = stat.get("mean")
                if "std" in stat:
                    row[f"{metric_name}_std"] = stat.get("std")
        rows.append(row)

    add("overall", "overall", summary.get("overall", {}))
    for kind in ["by_task_type", "by_env", "by_difficulty", "by_tool_necessary"]:
        group_kind = kind.replace("by_", "")
        for name, metrics in summary.get(kind, {}).items():
            add(group_kind, str(name), metrics)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    extra = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra:
                extra.append(key)
    fieldnames.extend(extra)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_run_summaries(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_summaries:
        return {}

    def numeric_mean_std(values: list[float]) -> dict[str, float]:
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": mean, "std": math.sqrt(var)}

    result: dict[str, Any] = {"n_runs": len(run_summaries)}
    for section in ["overall", "by_task_type", "by_env", "by_difficulty", "by_tool_necessary"]:
        keys = set()
        if section == "overall":
            group_names = ["overall"]
        else:
            group_names = sorted({name for summary in run_summaries for name in summary.get(section, {})})
        for group_name in group_names:
            group_metrics = []
            for summary in run_summaries:
                metrics = summary.get(section, {}) if section == "overall" else summary.get(section, {}).get(group_name, {})
                group_metrics.append(metrics)
                keys.update(key for key, value in metrics.items() if isinstance(value, (int, float)))
            stats = {}
            for key in sorted(keys):
                values = [float(metrics[key]) for metrics in group_metrics if isinstance(metrics.get(key), (int, float))]
                if values:
                    stats[key] = numeric_mean_std(values)
            if section == "overall":
                result[section] = stats
            else:
                result.setdefault(section, {})[group_name] = stats
    return result
