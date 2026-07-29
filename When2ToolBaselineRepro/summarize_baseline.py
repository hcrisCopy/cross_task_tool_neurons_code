from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPRO_ROOT = "../cross_task_tool_neurons_data/when2tool_baseline_repro"
MODE = "no_reasoning"


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


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def subset_dir(repro_root: Path, model_alias: str, subset: str) -> Path:
    suffix = "" if subset == "single_hop" else "_multihop"
    return repro_root / "probe_data" / f"{model_alias}{suffix}"


def label_summary(labels_path: Path) -> dict[str, Any] | None:
    data = read_json(labels_path)
    if not data:
        return None
    meta = data.get("task_meta", [])
    return {
        "n": len(meta),
        "labels": dict(Counter(str(int(item["tool_necessary"])) for item in meta)),
        "env": dict(Counter(str(item.get("env", "unknown")) for item in meta)),
        "difficulty": dict(Counter(str(item.get("difficulty", "unknown")) for item in meta)),
    }


def metric_summary(results_path: Path) -> dict[str, Any] | None:
    data = read_json(results_path)
    if not data:
        return None
    return {
        "path": rel(results_path),
        "best_layer": data.get("best_layer"),
        "acc": data.get("best_test_acc"),
        "auroc": data.get("best_test_auroc"),
        "train_acc": data.get("best_train_acc"),
        "all_layers": data.get("all_layers", False),
    }


def transfer_summary(path: Path) -> dict[str, Any] | None:
    data = read_json(path)
    if not data:
        return None
    return {
        "path": rel(path),
        "source_layer": data.get("source_layer"),
        "n_tasks": data.get("n_tasks"),
        "acc": round(float(data["accuracy"]), 4) if data.get("accuracy") is not None else None,
        "auroc": round(float(data["auroc"]), 4) if data.get("auroc") is not None else None,
    }


def print_json(title: str, value: Any) -> None:
    print(f"\n{title}")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize isolated upstream When2Tool reproduction outputs.")
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--repro-root", default=DEFAULT_REPRO_ROOT)
    args = parser.parse_args()

    repro_root = resolve_path(args.repro_root)
    single_dir = subset_dir(repro_root, args.model_alias, "single_hop")
    multi_dir = subset_dir(repro_root, args.model_alias, "multi_hop")
    transfer_dir = repro_root / "transfer" / f"{args.model_alias}_single_to_multihop"

    summary = {
        "repro_root": rel(repro_root),
        "model_alias": args.model_alias,
        "single_hop": {
            "train_labels": label_summary(single_dir / f"train_labels_{MODE}.json"),
            "test_labels": label_summary(single_dir / f"test_labels_{MODE}.json"),
            "probe": metric_summary(single_dir / f"probe_results_{MODE}.json"),
        },
        "multi_hop": {
            "train_labels": label_summary(multi_dir / f"train_labels_{MODE}.json"),
            "test_labels": label_summary(multi_dir / f"test_labels_{MODE}.json"),
            "probe": metric_summary(multi_dir / f"probe_results_{MODE}.json"),
        },
        "single_to_multi_transfer": transfer_summary(transfer_dir / f"transfer_{MODE}.json"),
    }

    print_json("When2Tool official baseline reproduction summary", summary)


if __name__ == "__main__":
    main()
