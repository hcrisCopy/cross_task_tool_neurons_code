from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from cttn.data import SPLITS, SUBSETS, load_raw_tasks, summarize_records
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: build model-specific modified datasets.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--raw-dataset-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expected_params(args: argparse.Namespace, raw_dir: Path, labels_root: Path) -> dict[str, Any]:
    label_manifests: dict[str, Any] = {}
    for subset in SUBSETS:
        label_manifests[subset] = {}
        for split in SPLITS:
            manifest_path = labels_root / args.model_alias / subset / split / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing label manifest: {manifest_path}")
            label_manifests[subset][split] = read_json(manifest_path).get("params", {})
    return {
        "stage": "03_dataset_building",
        "model_alias": args.model_alias,
        "raw_dataset_dir": str(raw_dir),
        "labels_dir": str(labels_root / args.model_alias),
        "label_manifest_params": label_manifests,
    }


def should_skip_model(model_out: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(model_out, data_root())
        return False
    expected = [model_out / subset / f"{split}.jsonl" for subset in SUBSETS for split in SPLITS]
    manifest_path = model_out / "manifest.json"
    if overwrite or not manifest_path.exists() or not all(path.exists() for path in expected):
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing modified dataset: {model_out}")
        return True
    return False


def attach_label(task: dict[str, Any], label: dict[str, Any], model_alias: str) -> dict[str, Any]:
    out = dict(task)
    out["model_alias"] = model_alias
    out["no_tool_correct"] = int(label["no_tool_correct"])
    out["tool_necessary"] = int(label["tool_necessary"])
    out["label_metadata"] = {
        "model_final_boxed": label.get("model_final_boxed", ""),
        "model_final_cleaned": label.get("model_final_cleaned", ""),
        "model_final_raw": label.get("model_final_raw", ""),
        "token_cost": label.get("token_cost", 0),
        "rounds": label.get("rounds", 0),
        "source": "hard_no_tool/no_reasoning",
    }
    return out


def main() -> None:
    args = parse_args()
    raw_dir = resolve_path(args.raw_dataset_dir) if args.raw_dataset_dir else path_from_config("raw_dataset_dir")
    labels_root = resolve_path(args.labels_dir) if args.labels_dir else path_from_config("labels_dir")
    output_root = resolve_path(args.output_dir) if args.output_dir else path_from_config("modified_dataset_dir")
    model_out = output_root / args.model_alias
    params = expected_params(args, raw_dir, labels_root)
    if should_skip_model(model_out, params, args.overwrite, args.clean):
        return
    ensure_dir(model_out)

    manifest = {
        "params": params,
        "subsets": {},
        "schema": {
            "task_type": "A/B/C task category derived from env_name",
            "tool_necessary": "1 iff this model failed hard_no_tool direct answering",
            "no_tool_correct": "1 iff hard_no_tool final boxed answer matched expected.answer",
            "label_metadata": "diagnostic output from stage 2",
        },
    }

    for subset in SUBSETS:
        manifest["subsets"][subset] = {}
        for split in SPLITS:
            out_path = model_out / subset / f"{split}.jsonl"
            raw_tasks = {str(task["id"]): task for task in load_raw_tasks(raw_dir, subset, split)}
            labels_path = labels_root / args.model_alias / subset / split / "labels.jsonl"
            if not labels_path.exists():
                raise FileNotFoundError(f"Missing labels: {labels_path}")
            labels = read_jsonl(labels_path)

            rows = []
            for label in labels:
                task_id = str(label["id"])
                if task_id not in raw_tasks:
                    raise KeyError(f"Label id not found in raw data: {task_id}")
                rows.append(attach_label(raw_tasks[task_id], label, args.model_alias))

            ensure_dir(out_path.parent)
            write_jsonl(out_path, rows)
            summary = summarize_records(rows)
            manifest["subsets"][subset][split] = {
                "path": str(out_path),
                "summary": summary,
            }
            write_json(out_path.parent / f"{split}_summary.json", summary)
            print(f"Wrote {len(rows)} rows: {out_path}")

    write_json(model_out / "manifest.json", manifest)
    print(f"Wrote manifest: {model_out / 'manifest.json'}")


if __name__ == "__main__":
    main()
