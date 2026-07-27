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

import numpy as np
import torch

from pp_common import (
    PP_METHOD,
    PP_STAGE_VERSION,
    compute_prefills,
    default_prefill_mode,
    flatten_probe_predictions,
    infer_tool_format,
    copy_probe_artifacts,
    is_dp_parent,
    load_model_module,
    load_utils,
    path_from_config,
    make_dp_run_root,
    parse_gpus,
    parse_thresholds,
    print_subset_plan,
    prepare_feature_meta_shard,
    prepare_feature_tensor_shard,
    pp_subdir,
    probe_prefill_root,
    read_json,
    read_jsonl,
    resolve_model_path,
    resolve_path,
    run_data_parallel_workers,
    set_single_process_cuda_visible,
    shard_indices,
    should_skip,
    sigmoid_temperature,
    sort_records_by_task_ids,
    validate_records_cover_task_ids,
    write_json,
    write_jsonl,
)
from pp_reporting import write_eval_case_report, write_threshold_sweep_report

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
    parser = argparse.ArgumentParser(description="ProbePrefill stage 3: run When2Tool Probe&Prefill with CTD probe.")
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
    parser.add_argument("--seed", **seed_arg_kwargs())
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_dir(root: Path, model_alias: str, subset: str) -> Path:
    return pp_subdir(root, "features") / model_alias / subset


def probe_dir(root: Path, model_alias: str, subset: str) -> Path:
    return pp_subdir(root, "probes") / model_alias / subset


def output_dir(root: Path, model_alias: str, subset: str, tag: str) -> Path:
    return pp_subdir(root, "outputs") / model_alias / "probe_prefill" / subset / tag


def load_probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trained CTD probe: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_test_features(root: Path, model_alias: str, subset: str) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    base = feature_dir(root, model_alias, subset)
    payload_path = base / "test_features.pt"
    meta_path = base / "test_meta.jsonl"
    summary_path = base / "test_summary.json"
    if not payload_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing ProbePrefill test features: {base}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    return payload["features"].float(), read_jsonl(meta_path), read_json(summary_path)


def probe_probabilities(probe: dict[str, Any], features: torch.Tensor, temperature: float) -> np.ndarray:
    coef = probe["coef"].float().numpy()
    intercept = float(probe["intercept"])
    mean = probe["scaler_mean"].float().numpy()
    scale = probe["scaler_scale"].float().numpy()
    scale = np.where(scale == 0.0, 1.0, scale)
    x = (features.numpy() - mean) / scale
    logits = x @ coef + intercept
    return sigmoid_temperature(logits, temperature)


def task_order_from_meta(model_dataset: Path, subset: str, meta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(model_dataset / subset / "test.jsonl")
    by_id = {str(row["id"]): row for row in rows}
    ordered = []
    missing = []
    for meta in meta_rows:
        task = by_id.get(str(meta["id"]))
        if task is None:
            missing.append(str(meta["id"]))
        else:
            ordered.append(task)
    if missing:
        raise KeyError(f"Test feature metadata contains ids not found in dataset: {missing[:5]}")
    return ordered


def tag_for(threshold: float, temperature: float, prefill_mode: str) -> str:
    return f"t{threshold:g}_temp{temperature:g}_{prefill_mode}"


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    threshold: float,
    prefill_mode: str,
    tool_format: str,
    model_path: Path,
    probe_manifest: dict[str, Any],
    feature_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "pp_03_probe_prefill_evaluation",
        "stage_version": PP_STAGE_VERSION,
        "method": PP_METHOD,
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "threshold": threshold,
        "temperature": args.temperature,
        "prefill_mode": prefill_mode,
        "backend": args.backend,
        "n_runs": args.n_runs,
        "batch_size": args.batch_size,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "vllm_dtype": args.vllm_dtype,
        "record_mode": args.record_mode,
        "seed": args.seed,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "probe_manifest_params": probe_manifest.get("params", {}),
        "feature_summary": feature_summary,
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


def write_case_outputs(
    args: argparse.Namespace,
    *,
    subset: str,
    threshold: float,
    prefill_mode: str,
    out_dir: Path,
    features: torch.Tensor,
    meta_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    prefills: dict[str, str],
    prefill_stats: dict[str, Any],
    run_outputs: dict[str, list[dict[str, Any]]],
    all_per_task: list[dict[str, Any]],
    params: dict[str, Any],
    tool_format: str,
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
    summary_payload["config"] = {
        "threshold": threshold,
        "temperature": args.temperature,
        "prefill_mode": prefill_mode,
        "prefill_stats": prefill_stats,
        "probe_feature_dim": int(features.shape[1]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "probe_predictions.jsonl", flatten_probe_predictions(meta_rows, probabilities, threshold=threshold))
    write_json(out_dir / "prefills.json", prefills)
    write_json(out_dir / "prefill_stats.json", prefill_stats)
    write_json(out_dir / "outputs.json", outputs_payload)
    write_jsonl(out_dir / "per_task.jsonl", all_per_task)
    write_json(out_dir / "summary.json", summary_payload)
    flat_rows = (
        flatten_summary(
            summary_payload,
            model_alias=args.model_alias,
            subset=subset,
            method=PP_METHOD,
            extra={"threshold": threshold, "temperature": args.temperature, "prefill_mode": prefill_mode},
        )
        if args.n_runs == 1
        else flatten_mean_std_summary(
            summary_payload["mean_std"],
            model_alias=args.model_alias,
            subset=subset,
            method=PP_METHOD,
            extra={"threshold": threshold, "temperature": args.temperature, "prefill_mode": prefill_mode},
        )
    )
    write_csv(out_dir / "summary_table.csv", flat_rows)
    write_json(out_dir / "manifest.json", {"params": params, "summary": summary_payload.get("overall", summary_payload.get("mean_std", {}).get("overall", {}))})
    write_eval_case_report(
        out_dir=out_dir,
        summary=summary_payload,
        model_alias=args.model_alias,
        subset=subset,
        method=PP_METHOD,
        threshold=threshold,
        temperature=args.temperature,
        prefill_mode=prefill_mode,
        tool_format=tool_format,
        prefill_stats=prefill_stats,
    )
    return summary_payload


def evaluate_case(
    args: argparse.Namespace,
    *,
    subset: str,
    threshold: float,
    prefill_mode: str,
    root: Path,
    model_path: Path,
    model_dataset: Path,
    w2t_utils: Any,
    agent: Any,
    tool_format: str,
) -> dict[str, Any]:
    features, meta_rows, feature_summary = load_test_features(root, args.model_alias, subset)
    probe_path = probe_dir(root, args.model_alias, subset) / "probe_no_reasoning.pt"
    probe_manifest_path = probe_dir(root, args.model_alias, subset) / "manifest.json"
    probe = load_probe(probe_path)
    probabilities = probe_probabilities(probe, features, args.temperature)
    prefills, prefill_stats = compute_prefills(
        task_ids=[str(row["id"]) for row in meta_rows],
        probabilities=probabilities,
        threshold=threshold,
        prefill_mode=prefill_mode,
        tool_format=tool_format,
    )
    tasks = task_order_from_meta(model_dataset, subset, meta_rows)
    tag = tag_for(threshold, args.temperature, prefill_mode)
    out_dir = output_dir(root, args.model_alias, subset, tag)
    params = expected_params(
        args,
        subset=subset,
        threshold=threshold,
        prefill_mode=prefill_mode,
        tool_format=tool_format,
        model_path=model_path,
        probe_manifest=read_json(probe_manifest_path) if probe_manifest_path.exists() else {},
        feature_summary=feature_summary,
    )
    if args._worker_index >= 0:
        params["data_parallel_worker"] = {"worker_index": args._worker_index, "num_workers": args._num_workers}
    expected = [out_dir / "summary.json", out_dir / "per_task.jsonl", out_dir / "outputs.json"]
    if should_skip(out_dir, params, expected, overwrite=args.overwrite, clean=args.clean, allowed_root=pp_subdir(root, "outputs")):
        summary = read_json(out_dir / "summary.json")
        prefill_stats_existing = read_json(out_dir / "prefill_stats.json") if (out_dir / "prefill_stats.json").exists() else prefill_stats
        write_eval_case_report(
            out_dir=out_dir,
            summary=summary,
            model_alias=args.model_alias,
            subset=subset,
            method=PP_METHOD,
            threshold=threshold,
            temperature=args.temperature,
            prefill_mode=prefill_mode,
            tool_format=tool_format,
            prefill_stats=prefill_stats_existing,
        )
        return summary

    run_outputs: dict[str, list[dict[str, Any]]] = {}
    all_per_task: list[dict[str, Any]] = []
    for run_id in range(args.n_runs):
        print(f"{subset}/{tag}: Probe&Prefill evaluation run {run_id + 1}/{args.n_runs}")
        outputs = evaluate_batched_with_task_progress(
            w2t_utils,
            tasks,
            agent,
            batch_size=args.batch_size,
            desc=f"{subset}/{tag}/run{run_id + 1}",
            max_rounds=args.max_rounds,
            record_mode=args.record_mode,
            prompt_mode="current",
            require_reasoning=False,
            prefills=prefills,
            tool_format=tool_format,
        )
        per_task = build_per_task(outputs, w2t_utils, run_id=run_id)
        run_outputs[f"run_{run_id}"] = outputs
        all_per_task.extend(per_task)

    summary_payload = write_case_outputs(
        args,
        subset=subset,
        threshold=threshold,
        prefill_mode=prefill_mode,
        out_dir=out_dir,
        features=features,
        meta_rows=meta_rows,
        probabilities=probabilities,
        prefills=prefills,
        prefill_stats=prefill_stats,
        run_outputs=run_outputs,
        all_per_task=all_per_task,
        params=params,
        tool_format=tool_format,
    )
    print(f"Wrote Probe&Prefill evaluation: {out_dir}")
    return summary_payload


def merge_shard_outputs(
    *,
    args: argparse.Namespace,
    subset: str,
    threshold: float,
    prefill_mode: str,
    root: Path,
    worker_roots: list[Path],
    model_path: Path,
    tool_format: str,
) -> dict[str, Any]:
    features, meta_rows, feature_summary = load_test_features(root, args.model_alias, subset)
    probe_path = probe_dir(root, args.model_alias, subset) / "probe_no_reasoning.pt"
    probe_manifest_path = probe_dir(root, args.model_alias, subset) / "manifest.json"
    probabilities = probe_probabilities(load_probe(probe_path), features, args.temperature)
    prefills, prefill_stats = compute_prefills(
        task_ids=[str(row["id"]) for row in meta_rows],
        probabilities=probabilities,
        threshold=threshold,
        prefill_mode=prefill_mode,
        tool_format=tool_format,
    )
    params = expected_params(
        args,
        subset=subset,
        threshold=threshold,
        prefill_mode=prefill_mode,
        tool_format=tool_format,
        model_path=model_path,
        probe_manifest=read_json(probe_manifest_path) if probe_manifest_path.exists() else {},
        feature_summary=feature_summary,
    )
    params["data_parallel"] = {"num_workers": len(worker_roots), "gpus": parse_gpus(args.gpus)}
    tag = tag_for(threshold, args.temperature, prefill_mode)
    task_ids = [str(row["id"]) for row in meta_rows]
    run_outputs: dict[str, list[dict[str, Any]]] = {}
    all_per_task: list[dict[str, Any]] = []
    for run_id in range(args.n_runs):
        merged_outputs: list[dict[str, Any]] = []
        merged_per_task: list[dict[str, Any]] = []
        for worker_root in worker_roots:
            shard_dir = output_dir(worker_root, args.model_alias, subset, tag)
            payload = read_json(shard_dir / "outputs.json")
            outputs = payload if args.n_runs == 1 else payload[f"run_{run_id}"]
            per_task = [row for row in read_jsonl(shard_dir / "per_task.jsonl") if int(row.get("run_id", 0)) == run_id]
            merged_outputs.extend(outputs)
            merged_per_task.extend(per_task)
        sorted_outputs = sort_records_by_task_ids(merged_outputs, task_ids)
        sorted_per_task = sort_records_by_task_ids(merged_per_task, task_ids)
        validate_records_cover_task_ids(sorted_outputs, task_ids, label=f"PP-3 {args.model_alias}/{subset}/{tag}/run{run_id} outputs")
        validate_records_cover_task_ids(sorted_per_task, task_ids, label=f"PP-3 {args.model_alias}/{subset}/{tag}/run{run_id} per_task")
        run_outputs[f"run_{run_id}"] = sorted_outputs
        all_per_task.extend(sorted_per_task)
    return write_case_outputs(
        args,
        subset=subset,
        threshold=threshold,
        prefill_mode=prefill_mode,
        out_dir=output_dir(root, args.model_alias, subset, tag),
        features=features,
        meta_rows=meta_rows,
        probabilities=probabilities,
        prefills=prefills,
        prefill_stats=prefill_stats,
        run_outputs=run_outputs,
        all_per_task=all_per_task,
        params=params,
        tool_format=tool_format,
    )


def run_data_parallel(
    args: argparse.Namespace,
    *,
    root: Path,
    model_path: Path,
    model_dataset: Path,
    tool_format: str,
    prefill_mode: str,
    thresholds: list[float],
) -> dict[str, Any]:
    gpus = parse_gpus(args.gpus)
    root_manifest: dict[str, Any] = {
        "stage": "pp_03_probe_prefill_evaluation",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "tool_format": tool_format,
        "prefill_mode": prefill_mode,
        "data_parallel": {"num_workers": len(gpus), "gpus": gpus},
        "subsets": {},
    }
    for subset in print_subset_plan(args.subset, stage="PP-3", model_alias=args.model_alias):
        _features, meta_rows, feature_summary = load_test_features(root, args.model_alias, subset)
        probe_manifest_path = probe_dir(root, args.model_alias, subset) / "manifest.json"
        probe_manifest = read_json(probe_manifest_path) if probe_manifest_path.exists() else {}
        pending_thresholds: list[float] = []
        for threshold in thresholds:
            params = expected_params(
                args,
                subset=subset,
                threshold=threshold,
                prefill_mode=prefill_mode,
                tool_format=tool_format,
                model_path=model_path,
                probe_manifest=probe_manifest,
                feature_summary=feature_summary,
            )
            params["data_parallel"] = {"num_workers": len(gpus), "gpus": gpus}
            tag = tag_for(threshold, args.temperature, prefill_mode)
            final_dir = output_dir(root, args.model_alias, subset, tag)
            expected = [final_dir / "summary.json", final_dir / "per_task.jsonl", final_dir / "outputs.json"]
            if not should_skip(final_dir, params, expected, overwrite=args.overwrite, clean=args.clean, allowed_root=pp_subdir(root, "outputs")):
                pending_thresholds.append(threshold)

        if pending_thresholds:
            shard_sets = shard_indices(len(meta_rows), len(gpus))
            run_root = make_dp_run_root(root, stage="pp_03", model_alias=args.model_alias, subset=subset)
            worker_roots: list[Path] = []
            for index, indices in enumerate(shard_sets):
                worker_root = run_root / f"shard_{index:02d}"
                prepare_feature_meta_shard(root, worker_root, model_alias=args.model_alias, subset=subset, indices=indices, split="test")
                prepare_feature_tensor_shard(root, worker_root, model_alias=args.model_alias, subset=subset, indices=indices, split="test")
                copy_probe_artifacts(root, worker_root, model_alias=args.model_alias, subset=subset)
                worker_roots.append(worker_root)
            worker_args = argparse.Namespace(**vars(args))
            worker_args.thresholds = ",".join(f"{threshold:g}" for threshold in pending_thresholds)
            total = len(meta_rows) * args.n_runs * len(pending_thresholds)
            run_data_parallel_workers(
                script_path=Path(__file__).resolve(),
                args=worker_args,
                gpus=gpus,
                subset=subset,
                worker_roots=worker_roots,
                total_progress=total,
                desc=f"PP-3 {args.model_alias}/{subset}",
                shard_sizes=[len(indices) for indices in shard_sets],
            )
        else:
            print(f"PP-3 {args.model_alias}/{subset}: all requested thresholds already complete; skip worker launch.")
        root_manifest["subsets"][subset] = {}
        for threshold in pending_thresholds:
            summary = merge_shard_outputs(
                args=args,
                subset=subset,
                threshold=threshold,
                prefill_mode=prefill_mode,
                root=root,
                worker_roots=worker_roots,
                model_path=model_path,
                tool_format=tool_format,
            )
            root_manifest["subsets"][subset][tag_for(threshold, args.temperature, prefill_mode)] = summary.get(
                "overall", summary.get("mean_std", {}).get("overall", {})
            )
        for threshold in thresholds:
            summary_path = output_dir(root, args.model_alias, subset, tag_for(threshold, args.temperature, prefill_mode)) / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing merged Probe&Prefill summary after data-parallel run: {summary_path}")
            summary = read_json(summary_path)
            root_manifest["subsets"][subset][tag_for(threshold, args.temperature, prefill_mode)] = summary.get(
                "overall", summary.get("mean_std", {}).get("overall", {})
            )
        subset_out_dir = pp_subdir(root, "outputs") / args.model_alias / "probe_prefill" / subset
        write_threshold_sweep_report(
            out_dir=subset_out_dir,
            cases=[(threshold, output_dir(root, args.model_alias, subset, tag_for(threshold, args.temperature, prefill_mode))) for threshold in thresholds],
            model_alias=args.model_alias,
            subset=subset,
            prefill_mode=prefill_mode,
            temperature=args.temperature,
        )
    return root_manifest

def main() -> None:
    args = parse_args()
    if args.n_runs < 1:
        raise ValueError("--n-runs must be >= 1")
    # Keep task sharding, tool-call identifiers, and backend-independent
    # stochastic operations reproducible; generation settings remain official.
    set_single_process_cuda_visible(args.gpus)
    random.seed(args.seed)
    np.random.seed(args.seed)
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
            model_dataset=model_dataset,
            tool_format=tool_format,
            prefill_mode=prefill_mode,
            thresholds=thresholds,
        )
    else:
        agent = make_agent(args, model_path=model_path, w2t_model=w2t_model)
        try:
            root_manifest = {
                "stage": "pp_03_probe_prefill_evaluation",
                "stage_version": PP_STAGE_VERSION,
                "model_alias": args.model_alias,
                "tool_format": tool_format,
                "prefill_mode": prefill_mode,
                "subsets": {},
            }
            for subset in print_subset_plan(args.subset, stage="PP-3", model_alias=args.model_alias):
                root_manifest["subsets"][subset] = {}
                for threshold in thresholds:
                    summary = evaluate_case(
                        args,
                        subset=subset,
                        threshold=threshold,
                        prefill_mode=prefill_mode,
                        root=root,
                        model_path=model_path,
                        model_dataset=model_dataset,
                        w2t_utils=w2t_utils,
                        agent=agent,
                        tool_format=tool_format,
                    )
                    root_manifest["subsets"][subset][tag_for(threshold, args.temperature, prefill_mode)] = summary.get(
                        "overall", summary.get("mean_std", {}).get("overall", {})
                    )
                subset_out_dir = pp_subdir(root, "outputs") / args.model_alias / "probe_prefill" / subset
                write_threshold_sweep_report(
                    out_dir=subset_out_dir,
                    cases=[(threshold, output_dir(root, args.model_alias, subset, tag_for(threshold, args.temperature, prefill_mode))) for threshold in thresholds],
                    model_alias=args.model_alias,
                    subset=subset,
                    prefill_mode=prefill_mode,
                    temperature=args.temperature,
                )
        finally:
            del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest_path = pp_subdir(root, "outputs") / args.model_alias / "probe_prefill" / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote Probe&Prefill evaluation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
