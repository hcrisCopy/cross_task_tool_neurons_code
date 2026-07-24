from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from cttn.data import (
    SPLITS,
    SUBSETS,
    balanced_sample_tasks,
    load_raw_tasks,
    select_label_balanced,
    summarize_records,
)
from cttn.io import write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.when2tool_bridge import load_model_module, load_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: generate model-specific tool_necessary labels.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--raw-dataset-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument("--require-per-type-labels", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--single-train-count", type=int, default=100)
    parser.add_argument("--single-test-count", type=int, default=30)
    parser.add_argument("--multi-train-count", type=int, default=40)
    parser.add_argument("--multi-test-count", type=int, default=30)

    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Hardware-dependent vLLM tensor parallel size. "
            "Use 1 on a single GPU; increase only when the model is sharded across multiple GPUs."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Aligned with When2Tool src/extract_features.py default.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Aligned with When2Tool src/extract_features.py default.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=12,
        help="Aligned with When2Tool src/extract_features.py default for label extraction.",
    )
    parser.add_argument("--record-mode", default="lite", choices=["lite", "full", "off"])
    parser.add_argument("--vllm-dtype", default="bfloat16")
    return parser.parse_args()


def target_count(args: argparse.Namespace, subset: str, split: str) -> int:
    if subset == "single_hop" and split == "train":
        return args.single_train_count
    if subset == "single_hop" and split == "test":
        return args.single_test_count
    if subset == "multi_hop" and split == "train":
        return args.multi_train_count
    if subset == "multi_hop" and split == "test":
        return args.multi_test_count
    raise ValueError(f"Unsupported subset/split: {subset}/{split}")


def output_dir_for(labels_root: Path, model_alias: str, subset: str, split: str) -> Path:
    return labels_root / model_alias / subset / split


def build_label_records(outputs: list[dict[str, Any]], task_by_id: dict[str, dict[str, Any]], w2t_utils: Any) -> list[dict[str, Any]]:
    records = []
    for item in outputs:
        task = task_by_id[str(item["id"])]
        raw, boxed, cleaned, correct = w2t_utils.item_final_eval(item)
        record = {
            "id": str(item["id"]),
            "model_alias": task.get("model_alias"),
            "subset": task["subset"],
            "split": task["split"],
            "env_name": task["env_name"],
            "task_type": task["task_type"],
            "difficulty": task.get("difficulty", "unknown"),
            "no_tool_correct": int(bool(correct)),
            "tool_necessary": int(not bool(correct)),
            "expected_answer": task.get("expected", {}).get("answer", ""),
            "model_final_boxed": boxed,
            "model_final_cleaned": cleaned,
            "model_final_raw": raw,
            "tool_calls": int(item.get("tool_calls", 0)),
            "rounds": int(item.get("rounds", 0)),
            "token_cost": item.get("token_cost", 0),
        }
        records.append(record)
    return records


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    manifest_path = out_dir / "manifest.json"
    labels_path = out_dir / "labels.jsonl"
    if clean:
        clean_directory(out_dir, data_root())
        return False
    if overwrite:
        return False
    if not manifest_path.exists() or not labels_path.exists():
        return False
    try:
        import json

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("params") == params:
            print(f"Skip existing labels: {labels_path}")
            return True
    except Exception:
        return False
    return False


def main() -> None:
    args = parse_args()
    raw_dir = resolve_path(args.raw_dataset_dir) if args.raw_dataset_dir else path_from_config("raw_dataset_dir")
    labels_root = resolve_path(args.labels_dir) if args.labels_dir else path_from_config("labels_dir")
    model_path = resolve_model_path(args.model_alias, args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)
    tool_format = infer_tool_format(args.model_alias, model_path)
    system_prompt = w2t_utils.get_system_prompt(tool_format)
    print(f"Model: {args.model_alias} -> {model_path}")
    print(f"Tool format: {tool_format}")

    model = w2t_model.AgentModel(
        model_path=str(model_path),
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        vllm_dtype=args.vllm_dtype,
        enable_thinking=False,
        system_prompt_override=system_prompt,
    )
    # Official evaluate_batched prefilters over-length prompts only when this
    # attribute exists on the wrapper object.
    model.max_model_len = args.max_model_len

    try:
        for subset in SUBSETS:
            for split in SPLITS:
                target = target_count(args, subset, split)
                params = {
                    "stage": "02_labeling",
                    "model_alias": args.model_alias,
                    "model_path": str(model_path),
                    "subset": subset,
                    "split": split,
                    "target_count": target,
                    "seed": args.seed,
                    "candidate_multiplier": args.candidate_multiplier,
                    "prompt_mode": "hard_no_tool",
                    "reasoning_mode": "no_reasoning",
                    "enable_thinking": False,
                    "tool_format": tool_format,
                    "backend": args.backend,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "max_rounds": args.max_rounds,
                    "max_new_tokens": args.max_new_tokens,
                    "max_model_len": args.max_model_len,
                    "vllm_dtype": args.vllm_dtype,
                    "record_mode": args.record_mode,
                    "require_per_type_labels": args.require_per_type_labels,
                }
                out_dir = output_dir_for(labels_root, args.model_alias, subset, split)
                if should_skip(out_dir, params, args.overwrite, args.clean):
                    continue
                ensure_dir(out_dir)

                all_tasks = load_raw_tasks(raw_dir, subset, split)
                candidate_count = min(len(all_tasks), max(target, int(math.ceil(target * args.candidate_multiplier))))
                candidates = balanced_sample_tasks(all_tasks, candidate_count, args.seed + len(subset) + len(split))
                for task in candidates:
                    task["model_alias"] = args.model_alias
                task_by_id = {str(task["id"]): task for task in candidates}

                print(f"\n{subset}/{split}: target={target}, candidates={len(candidates)}")
                outputs = w2t_utils.evaluate_batched(
                    candidates,
                    model,
                    max_rounds=args.max_rounds,
                    record_mode=args.record_mode,
                    prompt_mode="hard_no_tool",
                    require_reasoning=False,
                    tool_format=tool_format,
                )
                candidate_records = build_label_records(outputs, task_by_id, w2t_utils)
                selected_records = select_label_balanced(
                    candidate_records,
                    target,
                    args.seed,
                    require_per_type_labels=args.require_per_type_labels,
                )
                selected_ids = {record["id"] for record in selected_records}
                selected_outputs = [item for item in outputs if str(item["id"]) in selected_ids]

                write_json(out_dir / "candidate_outputs.json", outputs)
                write_jsonl(out_dir / "candidate_labels.jsonl", candidate_records)
                write_json(out_dir / "raw_outputs.json", selected_outputs)
                write_jsonl(out_dir / "labels.jsonl", selected_records)
                summary = {
                    "candidate_summary": summarize_records(candidate_records),
                    "selected_summary": summarize_records(selected_records),
                }
                write_json(out_dir / "summary.json", summary)
                write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
                print(f"Wrote labels: {out_dir / 'labels.jsonl'}")
                print(summary["selected_summary"])
    finally:
        del model


if __name__ == "__main__":
    main()
