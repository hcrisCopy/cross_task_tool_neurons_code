from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import torch

from cttn.agent import HFGenerationAgent
from cttn.data import SUBSETS
from cttn.eval_metrics import aggregate_run_summaries, build_per_task, build_summary, flatten_mean_std_summary, flatten_summary, write_csv
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.when2tool_bridge import load_model_module, load_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8: evaluate CTD-Masked LoRA adapters on test split.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    return parser.parse_args()


def subset_output_dir(outputs_root: Path, model_alias: str, subset: str) -> Path:
    return outputs_root / model_alias / "trained_evaluation" / subset


def expected_params(args: argparse.Namespace, subset: str) -> dict[str, Any]:
    return {
        "stage": "08_evaluation",
        "model_alias": args.model_alias,
        "subset": subset,
        "max_test_samples": args.max_test_samples,
        "n_runs": args.n_runs,
        "batch_size": args.batch_size,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "torch_dtype": args.torch_dtype,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "method": "CTD-Masked-LoRA",
    }


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    manifest_path = out_dir / "manifest.json"
    if overwrite or not manifest_path.exists() or not (out_dir / "summary.json").exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing trained evaluation: {out_dir}")
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
    params = expected_params(args, subset)
    if should_skip(out_dir, params, args.overwrite, args.clean):
        return None
    ensure_dir(out_dir)

    data = read_jsonl(model_dataset / subset / "test.jsonl")
    if args.max_test_samples > 0:
        data = data[: args.max_test_samples]
    adapter_dir = checkpoints_root / args.model_alias / "ctd_masked_lora" / subset / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter for {subset}: {adapter_dir}")

    tool_format = infer_tool_format(args.model_alias, model_path)
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
        adapter_dir=adapter_dir,
        batch_size=args.batch_size,
        enable_thinking=False,
    )
    try:
        run_outputs: dict[str, list[dict[str, Any]]] = {}
        run_summaries: dict[str, dict[str, Any]] = {}
        all_per_task = []
        for run_id in range(args.n_runs):
            print(f"{subset}: CTD-Masked-LoRA evaluation run {run_id + 1}/{args.n_runs}")
            outputs = w2t_utils.evaluate_batched(
                data,
                agent,
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
        summary_payload: dict[str, Any] = (
            run_summaries["run_0"]
            if args.n_runs == 1
            else {"runs": run_summaries, "mean_std": aggregate_run_summaries(list(run_summaries.values()))}
        )
        write_json(out_dir / "outputs.json", outputs_payload)
        write_jsonl(out_dir / "per_task.jsonl", all_per_task)
        write_json(out_dir / "summary.json", summary_payload)
        flat_rows = []
        if args.n_runs == 1:
            flat_rows = flatten_summary(
                summary_payload,
                model_alias=args.model_alias,
                subset=subset,
                method="CTD-Masked-LoRA",
            )
        else:
            flat_rows = flatten_mean_std_summary(
                summary_payload["mean_std"],
                model_alias=args.model_alias,
                subset=subset,
                method="CTD-Masked-LoRA",
            )
        write_csv(out_dir / "summary_table.csv", flat_rows)
        write_json(
            out_dir / "manifest.json",
            {
                "params": params,
                "summary": summary_payload.get("overall", summary_payload.get("mean_std", {}).get("overall", {})),
            },
        )
        print(f"Wrote trained evaluation: {out_dir}")
        return summary_payload
    finally:
        agent.close()
        gc.collect()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    checkpoints_root = resolve_path(args.checkpoints_dir) if args.checkpoints_dir else path_from_config("checkpoints_dir")
    outputs_root = resolve_path(args.outputs_dir) if args.outputs_dir else path_from_config("outputs_dir")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")

    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]
    root_manifest = {"stage": "08_evaluation", "model_alias": args.model_alias, "subsets": {}}

    for subset in subsets:
        out_dir = subset_output_dir(outputs_root, args.model_alias, subset)
        summary = evaluate_subset(
            subset,
            args=args,
            model_path=model_path,
            model_dataset=model_dataset,
            checkpoints_root=checkpoints_root,
            out_dir=out_dir,
            w2t_utils=w2t_utils,
            w2t_model=w2t_model,
        )
        if summary is not None:
            root_manifest["subsets"][subset] = {
                "path": str(out_dir),
                "overall": summary.get("overall", summary.get("mean_std", {}).get("overall", {})),
            }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = outputs_root / args.model_alias / "trained_evaluation" / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote trained evaluation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
