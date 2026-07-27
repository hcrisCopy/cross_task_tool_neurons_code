from __future__ import annotations

import argparse
import gc
import os
import random
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["WHEN2TOOL_QUIET_PROGRESS"] = "1"

from pathlib import Path
from typing import Any

import torch

from pp_common import (
    PP_STAGE_VERSION,
    compare_summaries_to_base,
    is_dp_parent,
    default_prefill_mode,
    infer_tool_format,
    load_model_module,
    load_utils,
    path_from_config,
    make_dp_run_root,
    parse_gpus,
    parse_thresholds,
    print_subset_plan,
    prepare_feature_meta_shard,
    pp_subdir,
    probe_prefill_root,
    read_json,
    read_jsonl,
    remove_files,
    resolve_model_path,
    resolve_path,
    run_data_parallel_workers,
    set_single_process_cuda_visible,
    shard_indices,
    should_skip,
    sort_records_by_task_ids,
    stable_sha256,
    validate_records_cover_task_ids,
    write_json,
    write_jsonl,
)
from pp_reporting import write_delta_sweep_report, write_eval_case_report

from cttn.eval_metrics import (
    aggregate_run_summaries,
    build_per_task,
    build_summary,
    flatten_mean_std_summary,
    flatten_summary,
    write_csv,
)
from cttn.progress import evaluate_batched_with_task_progress
from cttn.seeds import seed_arg_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProbePrefill stage 4: evaluate Base/Default and compute deltas.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--thresholds", default="0.5")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--prefill-mode", choices=["auto", "soft", "hard"], default="auto")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Tasks per When2Tool evaluate_batched call; also controls progress granularity.")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids. More than one GPU runs data-parallel workers inside this script.")
    parser.add_argument("--_worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--_num-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--_skip-delta", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", **seed_arg_kwargs())
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_meta(root: Path, model_alias: str, subset: str) -> list[dict[str, Any]]:
    path = pp_subdir(root, "features") / model_alias / subset / "test_meta.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing ProbePrefill test metadata: {path}")
    return read_jsonl(path)


def task_order_from_meta(model_dataset: Path, subset: str, meta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(model_dataset / subset / "test.jsonl")
    by_id = {str(row["id"]): row for row in rows}
    out = []
    missing = []
    for meta in meta_rows:
        task = by_id.get(str(meta["id"]))
        if task is None:
            missing.append(str(meta["id"]))
        else:
            out.append(task)
    if missing:
        raise KeyError(f"Feature metadata ids missing from dataset: {missing[:5]}")
    return out


def tag_for(threshold: float, temperature: float, prefill_mode: str) -> str:
    return f"t{threshold:g}_temp{temperature:g}_{prefill_mode}"


def base_dir(root: Path, model_alias: str, subset: str) -> Path:
    return pp_subdir(root, "outputs") / model_alias / "base_evaluation" / subset


def probe_eval_dir(root: Path, model_alias: str, subset: str, tag: str) -> Path:
    return pp_subdir(root, "outputs") / model_alias / "probe_prefill" / subset / tag


def expected_base_params(
    args: argparse.Namespace,
    *,
    subset: str,
    model_path: Path,
    tool_format: str,
    feature_meta_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": "pp_04_base_evaluation",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "backend": args.backend,
        "n_runs": args.n_runs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "vllm_dtype": args.vllm_dtype,
        "record_mode": args.record_mode,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "prefill": None,
        "tool_format": tool_format,
        "selected_test_ids_sha256": stable_sha256([row["id"] for row in feature_meta_rows]),
    }


def make_agent(args: argparse.Namespace, *, model_path: Path, w2t_model: Any) -> Any:
    kwargs = {
        "model_path": str(model_path),
        "backend": args.backend,
        "max_new_tokens": args.max_new_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "vllm_dtype": args.vllm_dtype,
        "enable_thinking": False,
    }
    if args.backend == "hf":
        kwargs.pop("tensor_parallel_size", None)
        kwargs.pop("vllm_dtype", None)
    return w2t_model.AgentModel(**kwargs)


def write_base_outputs(
    args: argparse.Namespace,
    *,
    subset: str,
    out_dir: Path,
    run_outputs: dict[str, list[dict[str, Any]]],
    all_per_task: list[dict[str, Any]],
    params: dict[str, Any],
    tool_format: str,
    comparison_thresholds: list[float],
    comparison_temperature: float,
) -> dict[str, Any]:
    run_summaries: dict[str, dict[str, Any]] = {}
    for run_id in range(args.n_runs):
        rows = [row for row in all_per_task if int(row.get("run_id", 0)) == run_id]
        run_summaries[f"run_{run_id}"] = build_summary(rows)

    outputs_payload: Any = run_outputs["run_0"] if args.n_runs == 1 else run_outputs
    summary_payload: dict[str, Any] = (
        run_summaries["run_0"]
        if args.n_runs == 1
        else {"runs": run_summaries, "mean_std": aggregate_run_summaries(list(run_summaries.values()))}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "outputs.json", outputs_payload)
    write_jsonl(out_dir / "per_task.jsonl", all_per_task)
    write_json(out_dir / "summary.json", summary_payload)
    flat_rows = (
        flatten_summary(summary_payload, model_alias=args.model_alias, subset=subset, method="Base/Default")
        if args.n_runs == 1
        else flatten_mean_std_summary(summary_payload["mean_std"], model_alias=args.model_alias, subset=subset, method="Base/Default")
    )
    write_csv(out_dir / "summary_table.csv", flat_rows)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary_payload.get("overall", summary_payload.get("mean_std", {}).get("overall", {}))})
    write_eval_case_report(
        out_dir=out_dir,
        summary=summary_payload,
        model_alias=args.model_alias,
        subset=subset,
        method="Base/Default",
        threshold=None,
        temperature=None,
        prefill_mode="none",
        tool_format=params.get("tool_format"),
        prefill_stats=None,
        comparison_thresholds=comparison_thresholds,
        comparison_temperature=comparison_temperature,
    )
    return summary_payload


def evaluate_base(
    args: argparse.Namespace,
    *,
    subset: str,
    root: Path,
    model_path: Path,
    model_dataset: Path,
    w2t_utils: Any,
    agent: Any,
    tool_format: str,
    thresholds: list[float],
) -> dict[str, Any]:
    meta_rows = feature_meta(root, args.model_alias, subset)
    tasks = task_order_from_meta(model_dataset, subset, meta_rows)
    out_dir = base_dir(root, args.model_alias, subset)
    params = expected_base_params(
        args,
        subset=subset,
        model_path=model_path,
        tool_format=tool_format,
        feature_meta_rows=meta_rows,
    )
    if args._worker_index >= 0:
        params["data_parallel_worker"] = {"worker_index": args._worker_index, "num_workers": args._num_workers}
    expected = [out_dir / "summary.json", out_dir / "per_task.jsonl", out_dir / "outputs.json"]
    if should_skip(out_dir, params, expected, overwrite=args.overwrite, clean=args.clean, allowed_root=pp_subdir(root, "outputs")):
        summary = read_json(out_dir / "summary.json")
        write_eval_case_report(
            out_dir=out_dir,
            summary=summary,
            model_alias=args.model_alias,
            subset=subset,
            method="Base/Default",
            threshold=None,
            temperature=None,
            prefill_mode="none",
            tool_format=tool_format,
            prefill_stats=None,
            comparison_thresholds=thresholds,
            comparison_temperature=args.temperature,
        )
        return summary

    run_outputs: dict[str, list[dict[str, Any]]] = {}
    all_per_task: list[dict[str, Any]] = []
    for run_id in range(args.n_runs):
        print(f"{subset}: Base/Default evaluation run {run_id + 1}/{args.n_runs}")
        outputs = evaluate_batched_with_task_progress(
            w2t_utils,
            tasks,
            agent,
            batch_size=args.batch_size,
            desc=f"{subset}/base/run{run_id + 1}",
            max_rounds=args.max_rounds,
            record_mode=args.record_mode,
            prompt_mode="current",
            require_reasoning=False,
            prefills=None,
            tool_format=tool_format,
        )
        per_task = build_per_task(outputs, w2t_utils, run_id=run_id)
        run_outputs[f"run_{run_id}"] = outputs
        all_per_task.extend(per_task)

    summary_payload = write_base_outputs(
        args,
        subset=subset,
        out_dir=out_dir,
        run_outputs=run_outputs,
        all_per_task=all_per_task,
        params=params,
        tool_format=tool_format,
        comparison_thresholds=thresholds,
        comparison_temperature=args.temperature,
    )
    print(f"Wrote Base/Default evaluation: {out_dir}")
    return summary_payload


def merge_base_shards(
    *,
    args: argparse.Namespace,
    subset: str,
    root: Path,
    worker_roots: list[Path],
    model_path: Path,
    tool_format: str,
    thresholds: list[float],
) -> dict[str, Any]:
    meta_rows = feature_meta(root, args.model_alias, subset)
    task_ids = [str(row["id"]) for row in meta_rows]
    params = expected_base_params(
        args,
        subset=subset,
        model_path=model_path,
        tool_format=tool_format,
        feature_meta_rows=meta_rows,
    )
    params["data_parallel"] = {"num_workers": len(worker_roots), "gpus": parse_gpus(args.gpus)}
    run_outputs: dict[str, list[dict[str, Any]]] = {}
    all_per_task: list[dict[str, Any]] = []
    for run_id in range(args.n_runs):
        merged_outputs: list[dict[str, Any]] = []
        merged_per_task: list[dict[str, Any]] = []
        for worker_root in worker_roots:
            shard_dir = base_dir(worker_root, args.model_alias, subset)
            payload = read_json(shard_dir / "outputs.json")
            outputs = payload if args.n_runs == 1 else payload[f"run_{run_id}"]
            per_task = [row for row in read_jsonl(shard_dir / "per_task.jsonl") if int(row.get("run_id", 0)) == run_id]
            merged_outputs.extend(outputs)
            merged_per_task.extend(per_task)
        sorted_outputs = sort_records_by_task_ids(merged_outputs, task_ids)
        sorted_per_task = sort_records_by_task_ids(merged_per_task, task_ids)
        validate_records_cover_task_ids(sorted_outputs, task_ids, label=f"PP-4 {args.model_alias}/{subset}/base/run{run_id} outputs")
        validate_records_cover_task_ids(sorted_per_task, task_ids, label=f"PP-4 {args.model_alias}/{subset}/base/run{run_id} per_task")
        run_outputs[f"run_{run_id}"] = sorted_outputs
        all_per_task.extend(sorted_per_task)
    return write_base_outputs(
        args,
        subset=subset,
        out_dir=base_dir(root, args.model_alias, subset),
        run_outputs=run_outputs,
        all_per_task=all_per_task,
        params=params,
        tool_format=tool_format,
        comparison_thresholds=thresholds,
        comparison_temperature=args.temperature,
    )


def run_data_parallel(
    args: argparse.Namespace,
    *,
    root: Path,
    model_path: Path,
    tool_format: str,
    thresholds: list[float],
    prefill_mode: str,
) -> dict[str, Any]:
    gpus = parse_gpus(args.gpus)
    root_manifest: dict[str, Any] = {
        "stage": "pp_04_base_evaluation_and_delta",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "tool_format": tool_format,
        "prefill_mode": prefill_mode,
        "data_parallel": {"num_workers": len(gpus), "gpus": gpus},
        "subsets": {},
    }
    for subset in print_subset_plan(args.subset, stage="PP-4", model_alias=args.model_alias):
        meta_rows = feature_meta(root, args.model_alias, subset)
        base_out = base_dir(root, args.model_alias, subset)
        params = expected_base_params(
            args,
            subset=subset,
            model_path=model_path,
            tool_format=tool_format,
            feature_meta_rows=meta_rows,
        )
        params["data_parallel"] = {"num_workers": len(gpus), "gpus": gpus}
        expected = [base_out / "summary.json", base_out / "per_task.jsonl", base_out / "outputs.json"]
        if should_skip(base_out, params, expected, overwrite=args.overwrite, clean=args.clean, allowed_root=pp_subdir(root, "outputs")):
            summary = read_json(base_out / "summary.json")
            write_eval_case_report(
                out_dir=base_out,
                summary=summary,
                model_alias=args.model_alias,
                subset=subset,
                method="Base/Default",
                threshold=None,
                temperature=None,
                prefill_mode="none",
                tool_format=tool_format,
                prefill_stats=None,
                comparison_thresholds=thresholds,
                comparison_temperature=args.temperature,
            )
            print(f"PP-4 {args.model_alias}/{subset}: Base already complete; skip worker launch.")
        else:
            shard_sets = shard_indices(len(meta_rows), len(gpus))
            run_root = make_dp_run_root(root, stage="pp_04", model_alias=args.model_alias, subset=subset)
            worker_roots: list[Path] = []
            for index, indices in enumerate(shard_sets):
                worker_root = run_root / f"shard_{index:02d}"
                prepare_feature_meta_shard(root, worker_root, model_alias=args.model_alias, subset=subset, indices=indices, split="test")
                worker_roots.append(worker_root)
            run_data_parallel_workers(
                script_path=Path(__file__).resolve(),
                args=args,
                gpus=gpus,
                subset=subset,
                worker_roots=worker_roots,
                total_progress=len(meta_rows) * args.n_runs,
                desc=f"PP-4 {args.model_alias}/{subset}",
                shard_sizes=[len(indices) for indices in shard_sets],
                extra_cli=["--_skip-delta"],
            )
            summary = merge_base_shards(
                args=args,
                subset=subset,
                root=root,
                worker_roots=worker_roots,
                model_path=model_path,
                tool_format=tool_format,
                thresholds=thresholds,
            )
        comparisons = build_delta_tables(args, root=root, subset=subset, thresholds=thresholds, prefill_mode=prefill_mode)
        root_manifest["subsets"][subset] = {
            "base_overall": summary.get("overall", summary.get("mean_std", {}).get("overall", {})),
            "comparisons": comparisons,
        }
    return root_manifest

def build_delta_tables(
    args: argparse.Namespace,
    *,
    root: Path,
    subset: str,
    thresholds: list[float],
    prefill_mode: str,
) -> dict[str, Any]:
    out = {}
    outputs_root = pp_subdir(root, "outputs")
    base_out = base_dir(root, args.model_alias, subset)
    if not (base_out / "summary.json").exists():
        raise FileNotFoundError(f"Missing Base summary: {base_out / 'summary.json'}")
    for threshold in thresholds:
        tag = tag_for(threshold, args.temperature, prefill_mode)
        probe_out = probe_eval_dir(root, args.model_alias, subset, tag)
        if args.clean:
            remove_files(
                [probe_out / "comparison_with_base.csv", probe_out / "comparison_with_base_manifest.json"],
                allowed_root=outputs_root,
            )
        if not (probe_out / "summary.json").exists() or not (probe_out / "manifest.json").exists():
            raise FileNotFoundError(f"Missing Probe&Prefill evaluation for {subset}/{tag}: {probe_out}")
        params = {
            "stage": "pp_04_base_vs_probe_prefill_delta",
            "stage_version": PP_STAGE_VERSION,
            "model_alias": args.model_alias,
            "subset": subset,
            "tag": tag,
            "base_manifest_params": read_json(base_out / "manifest.json").get("params", {}),
            "probe_prefill_manifest_params": read_json(probe_out / "manifest.json").get("params", {}),
        }
        out[tag] = compare_summaries_to_base(
            base_summary_path=base_out / "summary.json",
            probe_summary_path=probe_out / "summary.json",
            out_csv=probe_out / "comparison_with_base.csv",
            out_manifest=probe_out / "comparison_with_base_manifest.json",
            model_alias=args.model_alias,
            subset=subset,
            method="CTD-Probe&Prefill",
            params=params,
            overwrite=args.overwrite,
        )
    write_delta_sweep_report(
        out_dir=pp_subdir(root, "outputs") / args.model_alias / "probe_prefill" / subset,
        comparisons=[
            (
                threshold,
                probe_eval_dir(root, args.model_alias, subset, tag_for(threshold, args.temperature, prefill_mode))
                / "comparison_with_base.csv",
            )
            for threshold in thresholds
        ],
        model_alias=args.model_alias,
        subset=subset,
        prefill_mode=prefill_mode,
        temperature=args.temperature,
    )
    return out


def main() -> None:
    args = parse_args()
    if args.n_runs < 1:
        raise ValueError("--n-runs must be >= 1")
    set_single_process_cuda_visible(args.gpus)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = probe_prefill_root(args.output_root)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset: {model_dataset}")
    model_path = resolve_model_path(args.model_alias, args.model_path)
    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)
    tool_format = infer_tool_format(args.model_alias, model_path)
    prefill_mode = default_prefill_mode(args.model_alias, tool_format) if args.prefill_mode == "auto" else args.prefill_mode
    thresholds = parse_thresholds(args.thresholds)
    if is_dp_parent(args):
        root_manifest = run_data_parallel(
            args,
            root=root,
            model_path=model_path,
            tool_format=tool_format,
            thresholds=thresholds,
            prefill_mode=prefill_mode,
        )
    else:
        agent = make_agent(args, model_path=model_path, w2t_model=w2t_model)
        try:
            root_manifest = {
                "stage": "pp_04_base_evaluation_and_delta",
                "stage_version": PP_STAGE_VERSION,
                "model_alias": args.model_alias,
                "tool_format": tool_format,
                "prefill_mode": prefill_mode,
                "subsets": {},
            }
            for subset in print_subset_plan(args.subset, stage="PP-4", model_alias=args.model_alias):
                summary = evaluate_base(
                    args,
                    subset=subset,
                    root=root,
                    model_path=model_path,
                    model_dataset=model_dataset,
                    w2t_utils=w2t_utils,
                    agent=agent,
                    tool_format=tool_format,
                    thresholds=thresholds,
                )
                comparisons = {} if args._skip_delta else build_delta_tables(args, root=root, subset=subset, thresholds=thresholds, prefill_mode=prefill_mode)
                root_manifest["subsets"][subset] = {
                    "base_overall": summary.get("overall", summary.get("mean_std", {}).get("overall", {})),
                    "comparisons": comparisons,
                }
        finally:
            del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest_path = pp_subdir(root, "outputs") / args.model_alias / "base_evaluation" / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote Base/delta manifest: {manifest_path}")


if __name__ == "__main__":
    main()
