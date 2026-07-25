from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(REPO_ROOT / "PreciseShield") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PreciseShield"))

import torch

from cttn.agent import HFGenerationAgent
from cttn.eval_metrics import (
    aggregate_run_summaries,
    build_per_task,
    build_summary,
    flatten_mean_std_summary,
    flatten_summary,
    write_csv,
)
from cttn.paths import ensure_dir, path_from_config
from cttn.progress import ProgressTracker, evaluate_batched_with_task_progress, progress
from cttn.seeds import seed_arg_kwargs
from ps_common import (
    STAGE_VERSION,
    dataset_manifest,
    infer_tool_format,
    load_model_module,
    load_ps_masked_lora_adapter,
    load_utils,
    ps_resolve_root,
    read_json,
    read_jsonl,
    resolve_model_path,
    resolve_path,
    select_records,
    stable_sha256,
    subset_values,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield stage 8: evaluate trained PS masked LoRA.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="balanced")
    parser.add_argument("--seed", **seed_arg_kwargs())
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    parser.add_argument("--progress-file", default=None)
    return parser.parse_args()


def output_dir(outputs_root: Path, model_alias: str, subset: str) -> Path:
    return outputs_root / model_alias / "trained_evaluation" / subset


def load_test_rows(model_dataset: Path, subset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(model_dataset / subset / "test.jsonl")
    return select_records(
        rows,
        args.max_test_samples,
        args.seed,
        strategy=args.sample_strategy,
        require_per_type_labels=True,
    )


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    model_path: Path,
    model_dataset: Path,
    rows: list[dict[str, Any]],
    adapter_config: dict[str, Any],
    training_manifest: dict[str, Any],
    tool_format: str,
) -> dict[str, Any]:
    return {
        "stage": "ps_08_trained_evaluation",
        "stage_version": STAGE_VERSION,
        "method": "PreciseShield-Masked-LoRA",
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "selected_rows": {"count": len(rows), "sha256": stable_sha256([row["id"] for row in rows])},
        "max_test_samples": args.max_test_samples,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "n_runs": args.n_runs,
        "batch_size": args.batch_size,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "record_mode": args.record_mode,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "dataset_manifest_params": dataset_manifest(model_dataset).get("params", {}),
        "adapter_config": adapter_config,
        "training_manifest_params": training_manifest.get("params", {}),
    }


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        from ps_common import clean_path

        clean_path(out_dir)
        return False
    manifest_path = out_dir / "manifest.json"
    if overwrite or not manifest_path.exists() or not (out_dir / "summary.json").exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing PreciseShield trained evaluation: {out_dir}")
        return True
    return False


def evaluate_subset(
    subset: str,
    *,
    args: argparse.Namespace,
    model_path: Path,
    model_dataset: Path,
    checkpoints_root: Path,
    out_dir: Path,
    w2t_utils: Any,
    w2t_model: Any,
) -> dict[str, Any] | None:
    adapter_dir = checkpoints_root / args.model_alias / "ps_masked_lora" / subset / "adapter"
    training_manifest_path = checkpoints_root / args.model_alias / "ps_masked_lora" / subset / "manifest.json"
    if not adapter_dir.exists() or not training_manifest_path.exists():
        raise FileNotFoundError(f"Missing PreciseShield adapter for {subset}: {adapter_dir}")
    rows = load_test_rows(model_dataset, subset, args)
    tool_format = infer_tool_format(args.model_alias, model_path)
    params = expected_params(
        args,
        subset=subset,
        model_path=model_path,
        model_dataset=model_dataset,
        rows=rows,
        adapter_config=read_json(adapter_dir / "adapter_config.json"),
        training_manifest=read_json(training_manifest_path),
        tool_format=tool_format,
    )
    if should_skip(out_dir, params, args.overwrite, args.clean):
        return None
    ensure_dir(out_dir)
    system_prompt = w2t_utils.get_system_prompt(tool_format)
    normalizer = w2t_model._normalize_generation_output
    agent = HFGenerationAgent(
        model_path=model_path,
        system_prompt=system_prompt,
        normalizer=normalizer,
        torch_dtype_name=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        enable_thinking=False,
    )
    try:
        adapter_loaded = load_ps_masked_lora_adapter(agent.model, adapter_dir)
        agent.model.eval()
        run_outputs: dict[str, list[dict[str, Any]]] = {}
        run_summaries: dict[str, dict[str, Any]] = {}
        all_per_task = []
        tracker = ProgressTracker(args.progress_file, total=len(rows) * args.n_runs) if args.progress_file else None
        for run_id in progress(range(args.n_runs), desc=f"{subset} PS-Masked-LoRA runs", unit="run"):
            print(f"{subset}: PreciseShield-Masked-LoRA evaluation run {run_id + 1}/{args.n_runs}")
            outputs = evaluate_batched_with_task_progress(
                w2t_utils,
                rows,
                agent,
                batch_size=args.batch_size,
                desc=f"{subset} PS run {run_id + 1}/{args.n_runs} tasks",
                tracker=tracker,
                max_rounds=args.max_rounds,
                record_mode=args.record_mode,
                prompt_mode="current",
                require_reasoning=False,
                tool_format=tool_format,
            )
            per_task = build_per_task(outputs, w2t_utils, run_id=run_id)
            summary = build_summary(per_task)
            run_outputs[f"run_{run_id}"] = outputs
            run_summaries[f"run_{run_id}"] = summary
            all_per_task.extend(per_task)

        outputs_payload: Any = run_outputs["run_0"] if args.n_runs == 1 else run_outputs
        summary_payload = (
            run_summaries["run_0"]
            if args.n_runs == 1
            else {"runs": run_summaries, "mean_std": aggregate_run_summaries(list(run_summaries.values()))}
        )
        write_json(out_dir / "outputs.json", outputs_payload)
        write_jsonl(out_dir / "per_task.jsonl", all_per_task)
        write_json(out_dir / "summary.json", summary_payload)
        flat_rows = (
            flatten_summary(summary_payload, model_alias=args.model_alias, subset=subset, method="PreciseShield-Masked-LoRA")
            if args.n_runs == 1
            else flatten_mean_std_summary(
                summary_payload["mean_std"],
                model_alias=args.model_alias,
                subset=subset,
                method="PreciseShield-Masked-LoRA",
            )
        )
        write_csv(out_dir / "summary_table.csv", flat_rows)
        write_json(
            out_dir / "manifest.json",
            {
                "params": params,
                "adapter_loaded": adapter_loaded,
                "summary": summary_payload.get("overall", summary_payload.get("mean_std", {}).get("overall", {})),
            },
        )
        print(f"Wrote PreciseShield trained evaluation: {out_dir}")
        return summary_payload
    finally:
        agent.close()
        gc.collect()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    checkpoints_root = ps_resolve_root(args.checkpoints_dir, "checkpoints")
    outputs_root = ps_resolve_root(args.outputs_dir, "outputs")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")
    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)
    root_manifest = {"stage": "ps_08_trained_evaluation", "model_alias": args.model_alias, "subsets": {}}
    for subset in subset_values(args.subset):
        summary = evaluate_subset(
            subset,
            args=args,
            model_path=model_path,
            model_dataset=model_dataset,
            checkpoints_root=checkpoints_root,
            out_dir=output_dir(outputs_root, args.model_alias, subset),
            w2t_utils=w2t_utils,
            w2t_model=w2t_model,
        )
        if summary is not None:
            root_manifest["subsets"][subset] = {
                "path": str(output_dir(outputs_root, args.model_alias, subset)),
                "overall": summary.get("overall", summary.get("mean_std", {}).get("overall", {})),
            }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest_path = outputs_root / args.model_alias / "trained_evaluation" / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote PreciseShield trained evaluation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
