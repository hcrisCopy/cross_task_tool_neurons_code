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
from cttn.eval_metrics import (
    aggregate_run_summaries,
    build_comparison_with_base,
    build_per_task,
    build_summary,
    flatten_mean_std_summary,
    flatten_summary,
    write_csv,
)
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import ProgressTracker, evaluate_batched_with_task_progress, progress
from cttn.when2tool_bridge import load_model_module, load_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9: evaluate base/default model and compare with CTD-Masked-LoRA.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--max-test-samples", type=int, default=0)
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
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--progress-file", default=None)
    return parser.parse_args()


def base_output_dir(outputs_root: Path, model_alias: str, subset: str) -> Path:
    return outputs_root / model_alias / "base_evaluation" / subset


def trained_output_dir(outputs_root: Path, model_alias: str, subset: str) -> Path:
    return outputs_root / model_alias / "trained_evaluation" / subset


def expected_base_params(
    args: argparse.Namespace,
    subset: str,
    *,
    model_path: Path,
    dataset_manifest: dict[str, Any],
    tool_format: str,
) -> dict[str, Any]:
    return {
        "stage": "09_base_evaluation",
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "max_test_samples": args.max_test_samples,
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
        "method": "Base/Default",
        "tool_format": tool_format,
        "adapter": None,
        "activation_mask": None,
        "dataset_manifest_params": dataset_manifest.get("params", {}),
    }


def expected_comparison_params(
    *,
    model_alias: str,
    subset: str,
    base_manifest: dict[str, Any],
    trained_manifest: dict[str, Any],
    base_summary_path: Path,
    trained_summary_path: Path,
) -> dict[str, Any]:
    return {
        "stage": "09_base_vs_ctd_comparison",
        "model_alias": model_alias,
        "subset": subset,
        "base_manifest_params": base_manifest.get("params", {}),
        "trained_evaluation_manifest_params": trained_manifest.get("params", {}),
        "base_summary_path": str(base_summary_path),
        "trained_summary_path": str(trained_summary_path),
    }


def should_skip_base(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    manifest_path = out_dir / "manifest.json"
    if overwrite or not manifest_path.exists() or not (out_dir / "summary.json").exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing base evaluation: {out_dir}")
        return True
    return False


def clean_comparison(trained_dir: Path) -> None:
    for path in [trained_dir / "comparison_with_base.csv", trained_dir / "comparison_with_base_manifest.json"]:
        if path.exists():
            path.unlink()


def maybe_build_comparison(
    *,
    args: argparse.Namespace,
    outputs_root: Path,
    subset: str,
) -> dict[str, Any]:
    base_dir = base_output_dir(outputs_root, args.model_alias, subset)
    trained_dir = trained_output_dir(outputs_root, args.model_alias, subset)
    if args.clean:
        clean_comparison(trained_dir)
    base_manifest_path = base_dir / "manifest.json"
    trained_manifest_path = trained_dir / "manifest.json"
    base_summary_path = base_dir / "summary.json"
    trained_summary_path = trained_dir / "summary.json"
    if not base_manifest_path.exists() or not base_summary_path.exists():
        raise FileNotFoundError(f"Missing base evaluation outputs for {subset}: {base_dir}")
    if not trained_manifest_path.exists() or not trained_summary_path.exists():
        raise FileNotFoundError(
            f"Missing trained evaluation outputs for {subset}: {trained_dir}. Run CTD-Masked-LoRA evaluation before base comparison."
        )
    params = expected_comparison_params(
        model_alias=args.model_alias,
        subset=subset,
        base_manifest=read_json(base_manifest_path),
        trained_manifest=read_json(trained_manifest_path),
        base_summary_path=base_summary_path,
        trained_summary_path=trained_summary_path,
    )
    comparison_path = trained_dir / "comparison_with_base.csv"
    comparison_manifest_path = trained_dir / "comparison_with_base_manifest.json"
    if not args.overwrite and comparison_path.exists() and comparison_manifest_path.exists():
        manifest = read_json(comparison_manifest_path)
        if manifest.get("params") == params:
            print(f"Skip existing base comparison: {comparison_path}")
            return manifest

    rows = build_comparison_with_base(
        base_summary=read_json(base_summary_path),
        trained_summary=read_json(trained_summary_path),
        model_alias=args.model_alias,
        subset=subset,
    )
    write_csv(comparison_path, rows)
    manifest = {"params": params, "rows": len(rows), "path": str(comparison_path)}
    write_json(comparison_manifest_path, manifest)
    print(f"Wrote base comparison: {comparison_path}")
    return manifest


def evaluate_subset(
    subset: str,
    *,
    args: argparse.Namespace,
    model_path: Path,
    model_dataset: Path,
    out_dir: Path,
    w2t_utils: Any,
    w2t_model: Any,
) -> dict[str, Any]:
    tool_format = infer_tool_format(args.model_alias, model_path)
    params = expected_base_params(
        args,
        subset,
        model_path=model_path,
        dataset_manifest=read_json(model_dataset / "manifest.json") if (model_dataset / "manifest.json").exists() else {},
        tool_format=tool_format,
    )
    if should_skip_base(out_dir, params, args.overwrite, args.clean):
        return read_json(out_dir / "summary.json")
    ensure_dir(out_dir)

    data = read_jsonl(model_dataset / subset / "test.jsonl")
    if args.max_test_samples > 0:
        data = data[: args.max_test_samples]
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
        adapter_dir=None,
        batch_size=args.batch_size,
        enable_thinking=False,
    )
    try:
        run_outputs: dict[str, list[dict[str, Any]]] = {}
        run_summaries: dict[str, dict[str, Any]] = {}
        all_per_task = []
        tracker = ProgressTracker(args.progress_file, total=len(data) * args.n_runs) if args.progress_file else None
        for run_id in progress(range(args.n_runs), desc=f"{subset} Base/Default runs", unit="run"):
            print(f"{subset}: Base/Default evaluation run {run_id + 1}/{args.n_runs}")
            outputs = evaluate_batched_with_task_progress(
                w2t_utils,
                data,
                agent,
                batch_size=args.batch_size,
                desc=f"{subset} run {run_id + 1}/{args.n_runs} tasks",
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
        summary_payload: dict[str, Any] = (
            run_summaries["run_0"]
            if args.n_runs == 1
            else {"runs": run_summaries, "mean_std": aggregate_run_summaries(list(run_summaries.values()))}
        )
        write_json(out_dir / "outputs.json", outputs_payload)
        write_jsonl(out_dir / "per_task.jsonl", all_per_task)
        write_json(out_dir / "summary.json", summary_payload)
        flat_rows = (
            flatten_summary(summary_payload, model_alias=args.model_alias, subset=subset, method="Base/Default")
            if args.n_runs == 1
            else flatten_mean_std_summary(
                summary_payload["mean_std"],
                model_alias=args.model_alias,
                subset=subset,
                method="Base/Default",
            )
        )
        write_csv(out_dir / "summary_table.csv", flat_rows)
        write_json(
            out_dir / "manifest.json",
            {
                "params": params,
                "summary": summary_payload.get("overall", summary_payload.get("mean_std", {}).get("overall", {})),
            },
        )
        overall = summary_payload.get("overall", summary_payload.get("mean_std", {}).get("overall", {}))
        print(
            f"{subset}: Base/Default metrics "
            f"n={overall.get('n')}, "
            f"Acc={overall.get('final_accuracy')}, "
            f"AvgTC={overall.get('avg_tool_calls')}, "
            f"ToolAcc={overall.get('decision_accuracy')}"
        )
        print(f"Wrote base evaluation: {out_dir}")
        return summary_payload
    finally:
        agent.close()
        gc.collect()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    outputs_root = resolve_path(args.outputs_dir) if args.outputs_dir else path_from_config("outputs_dir")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")

    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]
    root_manifest = {"stage": "09_base_evaluation", "model_alias": args.model_alias, "subsets": {}}

    for subset in subsets:
        out_dir = base_output_dir(outputs_root, args.model_alias, subset)
        summary = evaluate_subset(
            subset,
            args=args,
            model_path=model_path,
            model_dataset=model_dataset,
            out_dir=out_dir,
            w2t_utils=w2t_utils,
            w2t_model=w2t_model,
        )
        comparison_manifest = None if args.skip_comparison else maybe_build_comparison(args=args, outputs_root=outputs_root, subset=subset)
        root_manifest["subsets"][subset] = {
            "path": str(out_dir),
            "overall": summary.get("overall", summary.get("mean_std", {}).get("overall", {})),
            "comparison": comparison_manifest,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = outputs_root / args.model_alias / "base_evaluation" / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote base evaluation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
