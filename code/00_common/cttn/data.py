from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TASK_TYPE_BY_ENV = {
    "CalculatorEnv": "A",
    "StatisticsEnv": "A",
    "CountingEnv": "A",
    "MatrixEnv": "A",
    "PrimeEnv": "A",
    "RetrieverEnv": "B",
    "HistoricalYearEnv": "B",
    "GameRuleEnv": "B",
    "HashEnv": "B",
    "DecodingEnv": "B",
    "ListManipulationEnv": "C",
    "DateTimeEnv": "C",
    "CodeExecutorEnv": "C",
    "ScheduleEnv": "C",
    "RegexMatchEnv": "C",
}

TASK_TYPES = ("A", "B", "C")
SUBSETS = ("single_hop", "multi_hop")
SPLITS = ("train", "test")


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def task_type_for_env(env_name: str) -> str:
    try:
        return TASK_TYPE_BY_ENV[env_name]
    except KeyError as exc:
        raise ValueError(f"Unknown When2Tool env_name: {env_name}") from exc


def raw_parquet_path(raw_dataset_dir: Path, subset: str, split: str) -> Path:
    matches = sorted((raw_dataset_dir / subset).glob(f"{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet found for {subset}/{split} under {raw_dataset_dir}")
    return matches[0]


def load_raw_dataframe(raw_dataset_dir: Path, subset: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(raw_parquet_path(raw_dataset_dir, subset, split))


def row_to_task(row: dict[str, Any], subset: str, split: str) -> dict[str, Any]:
    env_name = str(row["env_name"])
    steps = parse_json_field(row.get("steps"), [])
    task = {
        "id": str(row["id"]),
        "difficulty": str(row.get("difficulty", "unknown")),
        "multi_step": bool(row.get("multi_step", subset == "multi_hop")),
        "instruction": str(row["instruction"]),
        "environments": [
            {
                "name": env_name,
                "tools": parse_json_field(row.get("tools"), []),
                "parameters": parse_json_field(row.get("parameters"), {}),
            }
        ],
        "expected": {"answer": str(row.get("answer", ""))},
        "tags": parse_json_field(row.get("tags"), []),
        "subset": subset,
        "split": split,
        "env_name": env_name,
        "task_type": task_type_for_env(env_name),
    }
    if steps:
        task["expected"]["steps"] = steps
    return task


def load_raw_tasks(raw_dataset_dir: Path, subset: str, split: str) -> list[dict[str, Any]]:
    df = load_raw_dataframe(raw_dataset_dir, subset, split)
    return [row_to_task(row, subset, split) for row in df.to_dict(orient="records")]


def _split_quota(total: int, keys: list[str]) -> dict[str, int]:
    if total <= 0 or not keys:
        return {key: 0 for key in keys}
    base = total // len(keys)
    rem = total % len(keys)
    return {key: base + (1 if i < rem else 0) for i, key in enumerate(keys)}


def balanced_sample_tasks(tasks: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(tasks):
        return list(tasks)
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_type[task["task_type"]].append(task)

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    type_keys = [key for key in TASK_TYPES if by_type.get(key)]
    for task_type, type_target in _split_quota(count, type_keys).items():
        type_tasks = by_type[task_type]
        by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in type_tasks:
            by_diff[str(task.get("difficulty", "unknown"))].append(task)
        diff_keys = sorted(by_diff)
        for difficulty, diff_target in _split_quota(type_target, diff_keys).items():
            pool = list(by_diff[difficulty])
            rng.shuffle(pool)
            for task in pool[:diff_target]:
                selected.append(task)
                used_ids.add(str(task["id"]))

    if len(selected) < count:
        leftovers = [task for task in tasks if str(task["id"]) not in used_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])

    selected = selected[:count]
    selected.sort(key=lambda x: str(x["id"]))
    return selected


def select_label_balanced(
    records: list[dict[str, Any]],
    count: int,
    seed: int,
    *,
    require_per_type_labels: bool = False,
) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(records):
        selected = list(records)
    else:
        rng = random.Random(seed)
        selected = []
        used: set[str] = set()
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_type[record["task_type"]].append(record)
        type_keys = [key for key in TASK_TYPES if by_type.get(key)]
        for task_type, type_target in _split_quota(count, type_keys).items():
            type_records = by_type[task_type]
            by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in type_records:
                by_label[int(record["tool_necessary"])].append(record)
            half_targets = _split_quota(type_target, ["0", "1"])
            for label_key in ["0", "1"]:
                label = int(label_key)
                pool = list(by_label.get(label, []))
                rng.shuffle(pool)
                for record in pool[: half_targets[label_key]]:
                    selected.append(record)
                    used.add(str(record["id"]))
            if len([r for r in selected if r["task_type"] == task_type]) < type_target:
                leftovers = [r for r in type_records if str(r["id"]) not in used]
                rng.shuffle(leftovers)
                need = type_target - len([r for r in selected if r["task_type"] == task_type])
                for record in leftovers[:need]:
                    selected.append(record)
                    used.add(str(record["id"]))
        if len(selected) < count:
            leftovers = [r for r in records if str(r["id"]) not in used]
            rng.shuffle(leftovers)
            selected.extend(leftovers[: count - len(selected)])
        selected = selected[:count]

    selected.sort(key=lambda x: str(x["id"]))
    validate_selected_records(selected, require_per_type_labels=require_per_type_labels)
    return selected


def validate_selected_records(records: list[dict[str, Any]], *, require_per_type_labels: bool) -> None:
    task_types = {record["task_type"] for record in records}
    missing_types = [task_type for task_type in TASK_TYPES if task_type not in task_types]
    if missing_types:
        raise ValueError(f"Selected records do not cover task types: {missing_types}")
    labels = {int(record["tool_necessary"]) for record in records}
    if labels != {0, 1}:
        raise ValueError(f"Selected records do not cover both labels 0/1: {sorted(labels)}")
    if require_per_type_labels:
        bad = {}
        for task_type in TASK_TYPES:
            type_labels = {int(r["tool_necessary"]) for r in records if r["task_type"] == task_type}
            if type_labels != {0, 1}:
                bad[task_type] = sorted(type_labels)
        if bad:
            raise ValueError(f"Per-task-type label coverage failed: {bad}")


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    return {
        "count": len(rows),
        "by_task_type": dict(Counter(row.get("task_type", "unknown") for row in rows)),
        "by_difficulty": dict(Counter(row.get("difficulty", "unknown") for row in rows)),
        "by_env": dict(Counter(row.get("env_name", "unknown") for row in rows)),
        "by_tool_necessary": dict(Counter(str(row.get("tool_necessary", "unknown")) for row in rows)),
    }
