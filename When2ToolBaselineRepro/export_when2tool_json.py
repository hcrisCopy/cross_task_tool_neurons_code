from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET_DIR = "../cross_task_tool_neurons_data/datasets/raw_when2tool"
DEFAULT_OUTPUT_DIR = "../cross_task_tool_neurons_data/when2tool_baseline_repro/data"

TASK_FILES = {
    ("single_hop", "train"): "tasks_v1_train.json",
    ("single_hop", "test"): "tasks_v1_test.json",
    ("multi_hop", "train"): "tasks_v1_multihop_train.json",
    ("multi_hop", "test"): "tasks_v1_multihop_test.json",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def clean_output_dir(output_dir: Path) -> None:
    repro_root = resolve_path("../cross_task_tool_neurons_data/when2tool_baseline_repro")
    output_dir = output_dir.resolve()
    if output_dir == repro_root or not is_relative_to(output_dir, repro_root):
        raise ValueError(f"Refusing to clean path outside reproduction root: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


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


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def parquet_path(raw_dataset_dir: Path, subset: str, split: str) -> Path:
    matches = sorted((raw_dataset_dir / subset).glob(f"{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet found for {subset}/{split} under {raw_dataset_dir}")
    return matches[0]


def row_to_official_task(row: dict[str, Any], subset: str) -> dict[str, Any]:
    env_name = str(row["env_name"])
    steps = parse_json_field(row.get("steps"), [])
    task = {
        "id": to_builtin(row["id"]),
        "difficulty": str(row["difficulty"]),
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
    }
    if steps:
        task["expected"]["steps"] = steps
    return task


def load_tasks(raw_dataset_dir: Path, subset: str, split: str) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path(raw_dataset_dir, subset, split))
    return [row_to_official_task(row, subset) for row in frame.to_dict(orient="records")]


def summarize_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    envs = [
        str(task.get("environments", [{}])[0].get("name", "unknown"))
        for task in tasks
    ]
    return {
        "count": len(tasks),
        "by_difficulty": dict(Counter(str(task.get("difficulty", "unknown")) for task in tasks)),
        "by_env": dict(Counter(envs)),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def manifest_params(raw_dataset_dir: Path, output_dir: Path) -> dict[str, str]:
    return {
        "raw_dataset_dir": rel(raw_dataset_dir),
        "output_dir": rel(output_dir),
        "format": "when2tool_official_json_v1",
    }


def existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_all(
    raw_dataset_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    clean: bool = False,
) -> dict[str, Any]:
    raw_dataset_dir = raw_dataset_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    expected_files = [output_dir / name for name in TASK_FILES.values()]
    params = manifest_params(raw_dataset_dir, output_dir)

    if clean:
        clean_output_dir(output_dir)

    old_manifest = existing_manifest(manifest_path)
    if (
        not overwrite
        and old_manifest
        and old_manifest.get("params") == params
        and all(path.exists() for path in expected_files)
    ):
        print(f"[skip] official JSON tasks already complete: {rel(output_dir)}")
        return old_manifest

    summary: dict[str, Any] = {
        "params": params,
        "files": {},
    }
    for (subset, split), filename in TASK_FILES.items():
        tasks = load_tasks(raw_dataset_dir, subset, split)
        out_path = output_dir / filename
        write_json(out_path, tasks)
        summary["files"][filename] = {
            "subset": subset,
            "split": split,
            "path": rel(out_path),
            **summarize_tasks(tasks),
        }
        print(
            f"[write] {filename}: {len(tasks)} tasks "
            f"({subset}/{split}) -> {rel(out_path)}"
        )

    write_json(manifest_path, summary)
    print(f"[write] manifest -> {rel(manifest_path)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cached When2Tool parquet files as official JSON task files.")
    parser.add_argument("--raw-dataset-dir", type=str, default=DEFAULT_RAW_DATASET_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Rewrite JSON files even if manifest matches.")
    parser.add_argument("--clean", action="store_true", help="Delete the isolated JSON output directory before export.")
    args = parser.parse_args()

    export_all(
        resolve_path(args.raw_dataset_dir),
        resolve_path(args.output_dir),
        overwrite=args.overwrite,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
