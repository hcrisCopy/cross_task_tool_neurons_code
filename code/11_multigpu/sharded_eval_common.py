from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc or "progress"
            self.count = 0

        def __iter__(self):
            for item in self.iterable or []:
                yield item

        def __enter__(self):
            print(f"{self.desc}: 0/{self.total}")
            return self

        def __exit__(self, exc_type, exc, tb):
            print(f"{self.desc}: {self.count}/{self.total}")

        def update(self, value=1):
            self.count += value
            print(f"{self.desc}: {self.count}/{self.total}")

        def set_postfix_str(self, text):
            print(f"{self.desc}: {text}")

from cttn.eval_metrics import (
    aggregate_run_summaries,
    build_comparison_with_base,
    build_summary,
    flatten_mean_std_summary,
    flatten_summary,
    write_csv,
)
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path


SUBSETS = ("single_hop", "multi_hop")
TASK_TYPES = ("A", "B", "C")


def selected_subsets(value: str) -> list[str]:
    return list(SUBSETS) if value == "all" else [value]


def split_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def resolve_dataset_root(value: str | None) -> Path:
    return resolve_path(value) if value else path_from_config("modified_dataset_dir")


def resolve_outputs_root(value: str | None) -> Path:
    return resolve_path(value) if value else path_from_config("outputs_dir")


def resolve_checkpoints_root(value: str | None) -> Path:
    return resolve_path(value) if value else path_from_config("checkpoints_dir")


def resolve_neurons_root(value: str | None) -> Path:
    return resolve_path(value) if value else path_from_config("neurons_dir")


def resolve_causal_root(value: str | None) -> Path:
    return resolve_path(value) if value else path_from_config("causal_validation_dir")


def clean_if_exists(path: Path) -> None:
    if path.exists():
        clean_directory(path, data_root())


def ids_in_order(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(row.get("id")): idx for idx, row in enumerate(rows)}


def sort_outputs(rows: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: order.get(str(row.get("id")), 10**12))


def sort_per_task(rows: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get("run_id", 0)), order.get(str(row.get("id")), 10**12)))


def load_subset_test_rows(model_dataset: Path, subset: str, max_test_samples: int) -> list[dict[str, Any]]:
    rows = read_jsonl(model_dataset / subset / "test.jsonl")
    if max_test_samples > 0:
        rows = rows[:max_test_samples]
    return rows


def split_rows(rows: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    return [rows[idx::num_shards] for idx in range(num_shards)]


def write_shard_dataset(
    *,
    shard_dataset_root: Path,
    model_alias: str,
    subset: str,
    rows: list[dict[str, Any]],
    model_manifest: dict[str, Any],
) -> None:
    model_root = ensure_dir(shard_dataset_root / model_alias)
    ensure_dir(model_root / subset)
    write_json(model_root / "manifest.json", model_manifest)
    write_jsonl(model_root / subset / "test.jsonl", rows)


def tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def read_progress_file(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    done = int(payload.get("done", 0) or 0)
    total = int(payload.get("total", 0) or 0)
    return done, total


def run_process_group(jobs: list[dict[str, Any]], *, desc: str, repo_root: Path) -> None:
    if not jobs:
        print(f"{desc}: no non-empty shards to run.")
        return
    use_task_progress = all("progress_path" in job and "progress_total" in job for job in jobs)
    progress_total = (
        sum(int(job.get("progress_total", 0) or 0) for job in jobs)
        if use_task_progress
        else len(jobs)
    )
    progress_unit = "task" if use_task_progress else "shard"
    processes = []
    log_files = []
    for job in jobs:
        log_path = Path(job["log_path"])
        ensure_dir(log_path.parent)
        log_file = log_path.open("w", encoding="utf-8", errors="replace")
        log_file.write("+ " + " ".join(str(item) for item in job["cmd"]) + "\n")
        log_file.write(f"CUDA_VISIBLE_DEVICES={job['gpu']}\n\n")
        log_file.flush()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            job["cmd"],
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((job, proc))
        log_files.append(log_file)

    completed: set[int] = set()
    last_done_by_job = {idx: 0 for idx in range(len(processes))}
    try:
        with tqdm(total=progress_total, desc=desc, dynamic_ncols=True, unit=progress_unit) as bar:
            while len(completed) < len(processes):
                if use_task_progress:
                    for idx, (job, _proc) in enumerate(processes):
                        progress = read_progress_file(Path(job["progress_path"]))
                        if progress is None:
                            continue
                        done, total = progress
                        done = min(done, total)
                        if done > last_done_by_job[idx]:
                            bar.update(done - last_done_by_job[idx])
                            last_done_by_job[idx] = done
                for idx, (job, proc) in enumerate(processes):
                    if idx in completed:
                        continue
                    ret = proc.poll()
                    if ret is None:
                        continue
                    completed.add(idx)
                    log_files[idx].close()
                    if use_task_progress:
                        progress = read_progress_file(Path(job["progress_path"]))
                        done = progress[0] if progress else 0
                        total = int(job.get("progress_total", 0) or 0)
                        done = max(done, last_done_by_job[idx])
                        if ret == 0 and done < total:
                            bar.update(total - done)
                            last_done_by_job[idx] = total
                    else:
                        bar.update(1)
                    bar.set_postfix_str(f"last=shard{job['shard_index']} rc={ret}")
                    if ret != 0:
                        for other_idx, (_other_job, other_proc) in enumerate(processes):
                            if other_idx not in completed and other_proc.poll() is None:
                                other_proc.terminate()
                        for other_idx, (_other_job, other_proc) in enumerate(processes):
                            if other_idx not in completed and other_proc.poll() is None:
                                try:
                                    other_proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    other_proc.kill()
                                    other_proc.wait()
                        print(f"\nShard {job['shard_index']} failed. Log tail:\n{tail_text(Path(job['log_path']))}")
                        raise subprocess.CalledProcessError(ret, job["cmd"])
                time.sleep(1.0)
    finally:
        for log_file in log_files:
            if not log_file.closed:
                log_file.close()


def eval_overall(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("overall", summary.get("mean_std", {}).get("overall", {}))


def format_metric(value: Any) -> str:
    if isinstance(value, dict) and "mean" in value:
        mean = float(value.get("mean", 0.0))
        std = float(value.get("std", 0.0))
        return f"{mean:.4f}+/-{std:.4f}"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def metric_delta(metrics: dict[str, Any], base_metrics: dict[str, Any], name: str) -> float | None:
    if name not in metrics or name not in base_metrics:
        return None
    try:
        return float(metrics[name]) - float(base_metrics[name])
    except (TypeError, ValueError):
        return None


def add_tradeoff_deltas(row: dict[str, Any], metrics: dict[str, Any], base_metrics: dict[str, Any]) -> dict[str, Any]:
    eps = 1.0e-12
    delta_acc = metric_delta(metrics, base_metrics, "final_accuracy")
    delta_avg_tc = metric_delta(metrics, base_metrics, "avg_tool_calls")
    delta_tcr = metric_delta(metrics, base_metrics, "tool_call_rate")
    delta_total_tc = metric_delta(metrics, base_metrics, "total_tool_calls")

    if delta_acc is not None:
        row["delta_acc_pp"] = 100.0 * delta_acc
    if delta_avg_tc is not None:
        row["delta_avg_tool_calls"] = delta_avg_tc
        row["delta_acc_per_delta_avg_tool_call"] = (
            (100.0 * delta_acc) / delta_avg_tc
            if delta_acc is not None and abs(delta_avg_tc) > eps
            else ""
        )
        row["acc_cost_per_saved_call"] = (
            (100.0 * delta_acc) / (-delta_avg_tc)
            if delta_acc is not None and delta_avg_tc < 0
            else ""
        )
    if delta_tcr is not None:
        row["delta_tool_call_rate"] = delta_tcr
    if delta_total_tc is not None:
        base_total_tc = float(base_metrics.get("total_tool_calls", 0.0) or 0.0)
        if abs(base_total_tc) > eps:
            delta_pct = 100.0 * delta_total_tc / (base_total_tc + eps)
            row["delta_total_tool_calls_percent"] = delta_pct
            row["tool_call_reduction_percent"] = -delta_pct
        else:
            row["delta_total_tool_calls_percent"] = ""
            row["tool_call_reduction_percent"] = ""
    return row


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def print_eval_metrics(stage: str, subset: str, summary: dict[str, Any]) -> None:
    overall = eval_overall(summary)
    parts = [
        f"n={format_metric(overall.get('n', ''))}",
        f"Acc={format_metric(overall.get('final_accuracy', ''))}",
        f"TotalTC={format_metric(overall.get('total_tool_calls', ''))}",
        f"AvgTC={format_metric(overall.get('avg_tool_calls', ''))}",
        f"TCR={format_metric(overall.get('tool_call_rate', ''))}",
        f"ToolAcc={format_metric(overall.get('decision_accuracy', ''))}",
        f"OverCall={format_metric(overall.get('over_call_rate', ''))}",
        f"UnderCall={format_metric(overall.get('under_call_rate', ''))}",
        f"ValidToolRate={format_metric(overall.get('valid_tool_call_rate', ''))}",
    ]
    print(f"[{stage}] {subset} metrics: " + ", ".join(parts))


def write_eval_outputs(
    *,
    out_dir: Path,
    shard_dirs: list[Path],
    data_rows: list[dict[str, Any]],
    n_runs: int,
    model_alias: str,
    subset: str,
    method: str,
    params: dict[str, Any],
    shard_count: int,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    order = ids_in_order(data_rows)
    all_per_task: list[dict[str, Any]] = []
    if n_runs == 1:
        outputs: list[dict[str, Any]] = []
        for shard_dir in shard_dirs:
            payload = read_json(shard_dir / "outputs.json")
            outputs.extend(payload.get("run_0", []) if isinstance(payload, dict) else payload)
            all_per_task.extend(read_jsonl(shard_dir / "per_task.jsonl"))
        outputs = sort_outputs(outputs, order)
        all_per_task = sort_per_task(all_per_task, order)
        summary_payload = build_summary(all_per_task)
        outputs_payload: Any = outputs
        flat_rows = flatten_summary(summary_payload, model_alias=model_alias, subset=subset, method=method)
    else:
        outputs_by_run: dict[str, list[dict[str, Any]]] = {f"run_{idx}": [] for idx in range(n_runs)}
        for shard_dir in shard_dirs:
            payload = read_json(shard_dir / "outputs.json")
            for run_id in range(n_runs):
                outputs_by_run[f"run_{run_id}"].extend(payload.get(f"run_{run_id}", []))
            all_per_task.extend(read_jsonl(shard_dir / "per_task.jsonl"))
        for run_id in range(n_runs):
            key = f"run_{run_id}"
            outputs_by_run[key] = sort_outputs(outputs_by_run[key], order)
        all_per_task = sort_per_task(all_per_task, order)
        run_summaries = {
            f"run_{run_id}": build_summary([row for row in all_per_task if int(row.get("run_id", 0)) == run_id])
            for run_id in range(n_runs)
        }
        summary_payload = {
            "runs": run_summaries,
            "mean_std": aggregate_run_summaries(list(run_summaries.values())),
        }
        outputs_payload = outputs_by_run
        flat_rows = flatten_mean_std_summary(
            summary_payload["mean_std"],
            model_alias=model_alias,
            subset=subset,
            method=method,
        )

    write_json(out_dir / "outputs.json", outputs_payload)
    write_jsonl(out_dir / "per_task.jsonl", all_per_task)
    write_json(out_dir / "summary.json", summary_payload)
    write_csv(out_dir / "summary_table.csv", flat_rows)
    write_json(
        out_dir / "manifest.json",
        {
            "params": params,
            "summary": eval_overall(summary_payload),
            "data_parallel": {
                "shard_count": shard_count,
                "merged_task_rows": len(data_rows),
                "per_task_rows": len(all_per_task),
            },
        },
    )
    return summary_payload


def write_root_manifest(path: Path, *, stage: str, model_alias: str, subsets: dict[str, Any]) -> None:
    write_json(path, {"stage": stage, "model_alias": model_alias, "subsets": subsets})
    print(f"Wrote manifest: {path}")


def expected_trained_params(
    args: argparse.Namespace,
    subset: str,
    *,
    model_path: Path,
    model_dataset: Path,
    checkpoints_root: Path,
    tool_format: str,
) -> dict[str, Any]:
    adapter_dir = checkpoints_root / args.model_alias / "ctd_masked_lora" / subset / "adapter"
    training_manifest_path = checkpoints_root / args.model_alias / "ctd_masked_lora" / subset / "manifest.json"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter for {subset}: {adapter_dir}")
    if not training_manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest for {subset}: {training_manifest_path}")
    return {
        "stage": "08_evaluation",
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
        "method": "CTD-Masked-LoRA",
        "tool_format": tool_format,
        "dataset_manifest_params": read_json(model_dataset / "manifest.json").get("params", {})
        if (model_dataset / "manifest.json").exists()
        else {},
        "adapter_config": read_json(adapter_dir / "adapter_config.json"),
        "training_manifest_params": read_json(training_manifest_path).get("params", {}),
        "data_parallel": {"strategy": "test_interleaved_shards", "workers": len(split_gpus(args.gpus))},
    }


def expected_base_params(
    args: argparse.Namespace,
    subset: str,
    *,
    model_path: Path,
    model_dataset: Path,
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
        "dataset_manifest_params": read_json(model_dataset / "manifest.json").get("params", {})
        if (model_dataset / "manifest.json").exists()
        else {},
        "data_parallel": {"strategy": "test_interleaved_shards", "workers": len(split_gpus(args.gpus))},
    }


def params_match(manifest_path: Path, params: dict[str, Any], required: list[Path]) -> bool:
    if not manifest_path.exists() or not all(path.exists() for path in required):
        return False
    return read_json(manifest_path).get("params") == params


def prepare_shards(
    *,
    work_dir: Path,
    model_alias: str,
    subset: str,
    data_rows: list[dict[str, Any]],
    model_manifest: dict[str, Any],
    gpus: list[str],
) -> list[dict[str, Any]]:
    clean_if_exists(work_dir)
    ensure_dir(work_dir)
    shards = split_rows(data_rows, len(gpus))
    shard_infos = []
    for shard_index, (gpu, rows) in enumerate(zip(gpus, shards)):
        if not rows:
            continue
        shard_dataset_root = work_dir / "datasets" / f"shard_{shard_index:02d}"
        write_shard_dataset(
            shard_dataset_root=shard_dataset_root,
            model_alias=model_alias,
            subset=subset,
            rows=rows,
            model_manifest=model_manifest,
        )
        shard_infos.append(
            {
                "shard_index": shard_index,
                "gpu": gpu,
                "rows": rows,
                "dataset_root": shard_dataset_root,
                "output_root": work_dir / "outputs" / f"shard_{shard_index:02d}",
                "log_path": work_dir / "logs" / f"shard_{shard_index:02d}.log",
                "progress_path": work_dir / "progress" / f"shard_{shard_index:02d}.json",
            }
        )
    return shard_infos


def append_common_eval_args(cmd: list[str], args: argparse.Namespace, subset: str) -> None:
    cmd.extend(
        [
            "--model-alias",
            args.model_alias,
            "--when2tool-repo",
            args.when2tool_repo,
            "--subset",
            subset,
            "--max-test-samples",
            "0",
            "--n-runs",
            str(args.n_runs),
            "--batch-size",
            str(args.batch_size),
            "--max-rounds",
            str(args.max_rounds),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--torch-dtype",
            args.torch_dtype,
            "--device-map",
            args.device_map,
            "--record-mode",
            args.record_mode,
            "--overwrite",
        ]
    )
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])


def run_trained_subset_dp(args: argparse.Namespace, *, script_path: Path, repo_root: Path, subset: str) -> dict[str, Any]:
    gpus = split_gpus(args.gpus)
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_dataset_root(args.dataset_dir)
    checkpoints_root = resolve_checkpoints_root(args.checkpoints_dir)
    outputs_root = resolve_outputs_root(args.outputs_dir)
    model_dataset = dataset_root / args.model_alias
    tool_format = infer_tool_format(args.model_alias, model_path)
    params = expected_trained_params(
        args,
        subset,
        model_path=model_path,
        model_dataset=model_dataset,
        checkpoints_root=checkpoints_root,
        tool_format=tool_format,
    )
    final_dir = outputs_root / args.model_alias / "trained_evaluation" / subset
    if args.clean:
        clean_if_exists(final_dir)
    if not args.overwrite and params_match(
        final_dir / "manifest.json",
        params,
        [
            final_dir / "outputs.json",
            final_dir / "per_task.jsonl",
            final_dir / "summary.json",
            final_dir / "summary_table.csv",
        ],
    ):
        summary = read_json(final_dir / "summary.json")
        print(f"Skip existing trained evaluation: {final_dir}")
        print_eval_metrics("Stage 8", subset, summary)
        return summary
    clean_if_exists(final_dir)

    data_rows = load_subset_test_rows(model_dataset, subset, args.max_test_samples)
    model_manifest = read_json(model_dataset / "manifest.json") if (model_dataset / "manifest.json").exists() else {}
    work_dir = final_dir / "_shards"
    shard_infos = prepare_shards(
        work_dir=work_dir,
        model_alias=args.model_alias,
        subset=subset,
        data_rows=data_rows,
        model_manifest=model_manifest,
        gpus=gpus,
    )
    jobs = []
    for info in shard_infos:
        cmd = [sys.executable, str(script_path)]
        append_common_eval_args(cmd, args, subset)
        cmd.extend(
            [
                "--dataset-dir",
                str(info["dataset_root"]),
                "--checkpoints-dir",
                str(checkpoints_root),
                "--outputs-dir",
                str(info["output_root"]),
                "--progress-file",
                str(info["progress_path"]),
            ]
        )
        jobs.append({**info, "cmd": cmd, "progress_total": len(info["rows"]) * args.n_runs})
    print(f"[Stage 8] {subset}: {len(data_rows)} test rows -> {len(jobs)} GPU shards")
    run_process_group(jobs, desc=f"Stage 8 {subset}", repo_root=repo_root)
    shard_dirs = [info["output_root"] / args.model_alias / "trained_evaluation" / subset for info in shard_infos]
    summary = write_eval_outputs(
        out_dir=final_dir,
        shard_dirs=shard_dirs,
        data_rows=data_rows,
        n_runs=args.n_runs,
        model_alias=args.model_alias,
        subset=subset,
        method="CTD-Masked-LoRA",
        params=params,
        shard_count=len(jobs),
    )
    if not args.keep_shards:
        clean_if_exists(work_dir)
    print_eval_metrics("Stage 8", subset, summary)
    return summary


def run_base_subset_dp(args: argparse.Namespace, *, script_path: Path, repo_root: Path, subset: str) -> dict[str, Any]:
    gpus = split_gpus(args.gpus)
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_dataset_root(args.dataset_dir)
    outputs_root = resolve_outputs_root(args.outputs_dir)
    model_dataset = dataset_root / args.model_alias
    tool_format = infer_tool_format(args.model_alias, model_path)
    params = expected_base_params(args, subset, model_path=model_path, model_dataset=model_dataset, tool_format=tool_format)
    final_dir = outputs_root / args.model_alias / "base_evaluation" / subset
    if args.clean:
        clean_if_exists(final_dir)
        clean_comparison_files(outputs_root, args.model_alias, subset)
    if not args.overwrite and params_match(
        final_dir / "manifest.json",
        params,
        [
            final_dir / "outputs.json",
            final_dir / "per_task.jsonl",
            final_dir / "summary.json",
            final_dir / "summary_table.csv",
        ],
    ):
        summary = read_json(final_dir / "summary.json")
        print(f"Skip existing base evaluation: {final_dir}")
    else:
        clean_if_exists(final_dir)
        data_rows = load_subset_test_rows(model_dataset, subset, args.max_test_samples)
        model_manifest = read_json(model_dataset / "manifest.json") if (model_dataset / "manifest.json").exists() else {}
        work_dir = final_dir / "_shards"
        shard_infos = prepare_shards(
            work_dir=work_dir,
            model_alias=args.model_alias,
            subset=subset,
            data_rows=data_rows,
            model_manifest=model_manifest,
            gpus=gpus,
        )
        jobs = []
        for info in shard_infos:
            cmd = [sys.executable, str(script_path)]
            append_common_eval_args(cmd, args, subset)
            cmd.extend(
                [
                    "--dataset-dir",
                    str(info["dataset_root"]),
                    "--outputs-dir",
                    str(info["output_root"]),
                    "--skip-comparison",
                    "--progress-file",
                    str(info["progress_path"]),
                ]
            )
            jobs.append({**info, "cmd": cmd, "progress_total": len(info["rows"]) * args.n_runs})
        print(f"[Stage 9] {subset}: {len(data_rows)} test rows -> {len(jobs)} GPU shards")
        run_process_group(jobs, desc=f"Stage 9 {subset}", repo_root=repo_root)
        shard_dirs = [info["output_root"] / args.model_alias / "base_evaluation" / subset for info in shard_infos]
        summary = write_eval_outputs(
            out_dir=final_dir,
            shard_dirs=shard_dirs,
            data_rows=data_rows,
            n_runs=args.n_runs,
            model_alias=args.model_alias,
            subset=subset,
            method="Base/Default",
            params=params,
            shard_count=len(jobs),
        )
        if not args.keep_shards:
            clean_if_exists(work_dir)
    print_eval_metrics("Stage 9", subset, summary)
    comparison = write_base_comparison(args=args, outputs_root=outputs_root, subset=subset)
    print_comparison_metrics(subset, comparison)
    return summary


def clean_comparison_files(outputs_root: Path, model_alias: str, subset: str) -> None:
    trained_dir = outputs_root / model_alias / "trained_evaluation" / subset
    for path in [trained_dir / "comparison_with_base.csv", trained_dir / "comparison_with_base_manifest.json"]:
        if path.exists():
            path.unlink()


def write_base_comparison(*, args: argparse.Namespace, outputs_root: Path, subset: str) -> dict[str, Any]:
    base_dir = outputs_root / args.model_alias / "base_evaluation" / subset
    trained_dir = outputs_root / args.model_alias / "trained_evaluation" / subset
    base_manifest_path = base_dir / "manifest.json"
    trained_manifest_path = trained_dir / "manifest.json"
    base_summary_path = base_dir / "summary.json"
    trained_summary_path = trained_dir / "summary.json"
    if not base_manifest_path.exists() or not base_summary_path.exists():
        raise FileNotFoundError(f"Missing base evaluation outputs for {subset}: {base_dir}")
    if not trained_manifest_path.exists() or not trained_summary_path.exists():
        raise FileNotFoundError(
            f"Missing trained evaluation outputs for {subset}: {trained_dir}. Run Stage 8 before Stage 9."
        )
    params = {
        "stage": "09_base_vs_ctd_comparison",
        "model_alias": args.model_alias,
        "subset": subset,
        "base_manifest_params": read_json(base_manifest_path).get("params", {}),
        "trained_evaluation_manifest_params": read_json(trained_manifest_path).get("params", {}),
        "base_summary_path": str(base_summary_path),
        "trained_summary_path": str(trained_summary_path),
    }
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
    manifest = {"params": params, "rows": len(rows), "path": str(comparison_path), "overall": find_overall_row(rows)}
    write_json(comparison_manifest_path, manifest)
    print(f"Wrote base comparison: {comparison_path}")
    return manifest


def find_overall_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if row.get("group_kind") == "overall"), {})


def print_comparison_metrics(subset: str, manifest: dict[str, Any]) -> None:
    row = manifest.get("overall", {})
    if not row and "path" in manifest:
        rows = []
        path = Path(str(manifest["path"]))
        if path.exists():
            import csv

            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        row = find_overall_row(rows)
    if row:
        print(
            "[Stage 9] "
            f"{subset} delta: delta_acc_pp={row.get('delta_acc_pp', '')}, "
            f"delta_total_tool_calls_percent={row.get('delta_total_tool_calls_percent', '')}, "
            f"tool_call_reduction_percent={row.get('tool_call_reduction_percent', '')}, "
            f"delta_avg_tool_calls={row.get('delta_avg_tool_calls', '')}, "
            f"delta_tool_call_rate={row.get('delta_tool_call_rate', '')}"
        )


def parse_interventions(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def pure_rows_to_keys(rows: list[dict[str, Any]]) -> set[tuple[int, str, int]]:
    return {(int(row["layer"]), str(row["module"]), int(row["index"])) for row in rows}


def private_rows(tdn_rows: list[dict[str, Any]], ctd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctd_keys = pure_rows_to_keys(ctd_rows)
    return [row for row in tdn_rows if (int(row["layer"]), str(row["module"]), int(row["index"])) not in ctd_keys]


def intervention_rows(
    name: str,
    *,
    tdn_rows: list[dict[str, Any]],
    ctd_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if name == "Base":
        return []
    if name == "Mask-Random":
        return random_rows
    if name == "Mask-TDN_c":
        return tdn_rows
    if name == "Mask-CTD":
        return ctd_rows
    if name == "Mask-Private_c":
        return private_rows(tdn_rows, ctd_rows)
    raise ValueError(f"Unknown intervention: {name}")


def expected_causal_params(
    args: argparse.Namespace,
    subset: str,
    *,
    model_path: Path,
    model_dataset: Path,
    neurons_root: Path,
    tool_format: str,
) -> dict[str, Any]:
    single_manifest_path = neurons_root / args.model_alias / "single_type_by_subset" / subset / "manifest.json"
    shared_manifest_path = neurons_root / args.model_alias / "shared_by_subset" / subset / "manifest.json"
    if not single_manifest_path.exists():
        raise FileNotFoundError(f"Missing single-type manifest for {subset}: {single_manifest_path}")
    if not shared_manifest_path.exists():
        raise FileNotFoundError(f"Missing shared neuron manifest for {subset}: {shared_manifest_path}")
    return {
        "stage": "10_causal_validation",
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "max_test_samples": args.max_test_samples,
        "interventions": parse_interventions(args.interventions),
        "batch_size": args.batch_size,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "record_mode": args.record_mode,
        "seed": args.seed,
        "random_mask_seed": args.seed,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "dataset_manifest_params": read_json(model_dataset / "manifest.json").get("params", {})
        if (model_dataset / "manifest.json").exists()
        else {},
        "single_type_manifest_params": read_json(single_manifest_path).get("params", {}),
        "shared_neuron_manifest_params": read_json(shared_manifest_path).get("params", {}),
        "data_parallel": {"strategy": "test_interleaved_shards", "workers": len(split_gpus(args.gpus))},
    }


def run_causal_subset_dp(args: argparse.Namespace, *, script_path: Path, repo_root: Path, subset: str) -> dict[str, Any]:
    gpus = split_gpus(args.gpus)
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_dataset_root(args.dataset_dir)
    neurons_root = resolve_neurons_root(args.neurons_dir)
    causal_root = resolve_causal_root(args.causal_dir)
    model_dataset = dataset_root / args.model_alias
    tool_format = infer_tool_format(args.model_alias, model_path)
    params = expected_causal_params(
        args,
        subset,
        model_path=model_path,
        model_dataset=model_dataset,
        neurons_root=neurons_root,
        tool_format=tool_format,
    )
    final_dir = causal_root / args.model_alias / subset
    if args.clean:
        clean_if_exists(final_dir)
    if not args.overwrite and params_match(
        final_dir / "manifest.json",
        params,
        [
            final_dir / "summary_table.csv",
            final_dir / "cross_type_summary.csv",
        ],
    ):
        manifest = read_json(final_dir / "manifest.json")
        print(f"Skip existing causal validation: {final_dir}")
        print_causal_metrics(subset, read_csv_dicts(final_dir / "cross_type_summary.csv"))
        return manifest
    clean_if_exists(final_dir)

    data_rows = load_subset_test_rows(model_dataset, subset, args.max_test_samples)
    model_manifest = read_json(model_dataset / "manifest.json") if (model_dataset / "manifest.json").exists() else {}
    work_dir = final_dir / "_shards"
    shard_infos = prepare_shards(
        work_dir=work_dir,
        model_alias=args.model_alias,
        subset=subset,
        data_rows=data_rows,
        model_manifest=model_manifest,
        gpus=gpus,
    )
    jobs = []
    for info in shard_infos:
        cmd = [
            sys.executable,
            str(script_path),
            "--model-alias",
            args.model_alias,
            "--when2tool-repo",
            args.when2tool_repo,
            "--subset",
            subset,
            "--max-test-samples",
            "0",
            "--interventions",
            args.interventions,
            "--batch-size",
            str(args.batch_size),
            "--max-rounds",
            str(args.max_rounds),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--torch-dtype",
            args.torch_dtype,
            "--device-map",
            args.device_map,
            "--record-mode",
            args.record_mode,
            "--seed",
            str(args.seed),
            "--dataset-dir",
            str(info["dataset_root"]),
            "--neurons-dir",
            str(neurons_root),
            "--causal-dir",
            str(info["output_root"]),
            "--progress-file",
            str(info["progress_path"]),
            "--overwrite",
        ]
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        shard_progress_total = sum(
            len([row for row in info["rows"] if row.get("task_type") == task_type])
            for task_type in TASK_TYPES
        ) * len(parse_interventions(args.interventions))
        jobs.append({**info, "cmd": cmd, "progress_total": shard_progress_total})
    print(f"[Stage 10] {subset}: {len(data_rows)} test rows -> {len(jobs)} GPU shards")
    run_process_group(jobs, desc=f"Stage 10 {subset}", repo_root=repo_root)
    shard_dirs = [info["output_root"] / args.model_alias / subset for info in shard_infos]
    manifest = write_causal_outputs(
        out_dir=final_dir,
        shard_dirs=shard_dirs,
        data_rows=data_rows,
        model_alias=args.model_alias,
        subset=subset,
        neurons_root=neurons_root,
        interventions=parse_interventions(args.interventions),
        params=params,
        shard_count=len(jobs),
    )
    if not args.keep_shards:
        clean_if_exists(work_dir)
    print_causal_metrics(subset, read_csv_dicts(final_dir / "cross_type_summary.csv"))
    return manifest


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_causal_outputs(
    *,
    out_dir: Path,
    shard_dirs: list[Path],
    data_rows: list[dict[str, Any]],
    model_alias: str,
    subset: str,
    neurons_root: Path,
    interventions: list[str],
    params: dict[str, Any],
    shard_count: int,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    order = ids_in_order(data_rows)
    shared_dir = neurons_root / model_alias / "shared_by_subset" / subset
    single_dir = neurons_root / model_alias / "single_type_by_subset" / subset
    ctd_rows = read_jsonl(shared_dir / "CTD_neurons.jsonl")
    random_rows = first_random_rows(shard_dirs)
    write_jsonl(out_dir / "random_mask_neurons.jsonl", random_rows)

    summary_rows: list[dict[str, Any]] = []
    cross_by_intervention: dict[str, dict[str, Any]] = {}
    for task_type in TASK_TYPES:
        task_rows = [row for row in data_rows if row.get("task_type") == task_type]
        if not task_rows:
            continue
        tdn_rows = read_jsonl(single_dir / task_type / "TDN_neurons.jsonl")
        base_metrics = None
        type_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for intervention in interventions:
            outputs: list[dict[str, Any]] = []
            per_task: list[dict[str, Any]] = []
            for shard_dir in shard_dirs:
                case_dir = shard_dir / task_type / intervention
                if not case_dir.exists():
                    continue
                outputs.extend(read_json(case_dir / "outputs.json"))
                per_task.extend(read_jsonl(case_dir / "per_task.jsonl"))
            outputs = sort_outputs(outputs, order)
            per_task = sort_per_task(per_task, order)
            summary = build_summary(per_task)
            case_out = ensure_dir(out_dir / task_type / intervention)
            write_json(case_out / "outputs.json", outputs)
            write_jsonl(case_out / "per_task.jsonl", per_task)
            write_json(case_out / "summary.json", summary)

            mask_rows = intervention_rows(intervention, tdn_rows=tdn_rows, ctd_rows=ctd_rows, random_rows=random_rows)
            metrics = summary["overall"]
            if intervention == "Base":
                base_metrics = metrics
            row = {
                "model_alias": model_alias,
                "subset": subset,
                "task_type": task_type,
                "intervention": intervention,
                "masked_neurons": len(mask_rows),
            }
            row.update(metrics)
            type_rows.append((intervention, row, metrics))
        if base_metrics is not None:
            for intervention, row, metrics in type_rows:
                add_tradeoff_deltas(row, metrics, base_metrics)
                cross = cross_by_intervention.setdefault(
                    intervention,
                    {
                        "delta_acc": [],
                        "delta_tcr": [],
                        "delta_rows": [],
                        "task_type_metrics": {},
                        "task_type_delta_rows": {},
                    },
                )
                cross["delta_acc"].append(float(metrics["final_accuracy"]) - float(base_metrics["final_accuracy"]))
                cross["delta_tcr"].append(float(metrics["tool_call_rate"]) - float(base_metrics["tool_call_rate"]))
                cross["delta_rows"].append(row)
                cross["task_type_metrics"][task_type] = metrics
                cross["task_type_delta_rows"][task_type] = row
        summary_rows.extend(row for _intervention, row, _metrics in type_rows)

    cross_rows = []
    for intervention, payload in sorted(cross_by_intervention.items()):
        deltas = payload["delta_acc"]
        delta_tcr = payload["delta_tcr"]
        delta_rows = payload["delta_rows"]
        metrics_by_type = payload["task_type_metrics"]
        delta_rows_by_type = payload["task_type_delta_rows"]
        delta_avg_tc_values = numeric_values(delta_rows, "delta_avg_tool_calls")
        delta_total_tc_pct_values = numeric_values(delta_rows, "delta_total_tool_calls_percent")
        tool_reduction_values = numeric_values(delta_rows, "tool_call_reduction_percent")
        acc_per_delta_tc_values = numeric_values(delta_rows, "delta_acc_per_delta_avg_tool_call")
        acc_cost_values = numeric_values(delta_rows, "acc_cost_per_saved_call")
        row = {
            "model_alias": model_alias,
            "subset": subset,
            "intervention": intervention,
            "avg_delta_acc": sum(deltas) / len(deltas) if deltas else 0.0,
            "avg_delta_acc_pp": 100.0 * (sum(deltas) / len(deltas)) if deltas else 0.0,
            "var_acc": statistics.pvariance(deltas) if len(deltas) > 1 else 0.0,
            "avg_delta_tcr": sum(delta_tcr) / len(delta_tcr) if delta_tcr else 0.0,
            "avg_delta_tool_call_rate": sum(delta_tcr) / len(delta_tcr) if delta_tcr else 0.0,
            "avg_delta_avg_tool_calls": sum(delta_avg_tc_values) / len(delta_avg_tc_values) if delta_avg_tc_values else 0.0,
            "avg_delta_total_tool_calls_percent": (
                sum(delta_total_tc_pct_values) / len(delta_total_tc_pct_values)
                if delta_total_tc_pct_values
                else ""
            ),
            "avg_tool_call_reduction_percent": (
                sum(tool_reduction_values) / len(tool_reduction_values) if tool_reduction_values else ""
            ),
            "avg_delta_acc_per_delta_avg_tool_call": (
                sum(acc_per_delta_tc_values) / len(acc_per_delta_tc_values) if acc_per_delta_tc_values else ""
            ),
            "avg_acc_cost_per_saved_call": sum(acc_cost_values) / len(acc_cost_values) if acc_cost_values else "",
        }
        for task_type in TASK_TYPES:
            metrics = metrics_by_type.get(task_type, {})
            delta_row = delta_rows_by_type.get(task_type, {})
            row[f"acc_{task_type}"] = metrics.get("final_accuracy")
            row[f"tool_acc_{task_type}"] = metrics.get("decision_accuracy")
            row[f"tcr_{task_type}"] = metrics.get("tool_call_rate")
            row[f"delta_acc_pp_{task_type}"] = delta_row.get("delta_acc_pp")
            row[f"delta_avg_tool_calls_{task_type}"] = delta_row.get("delta_avg_tool_calls")
            row[f"tool_call_reduction_percent_{task_type}"] = delta_row.get("tool_call_reduction_percent")
            row[f"delta_acc_per_delta_avg_tool_call_{task_type}"] = delta_row.get("delta_acc_per_delta_avg_tool_call")
            row[f"acc_cost_per_saved_call_{task_type}"] = delta_row.get("acc_cost_per_saved_call")
        cross_rows.append(row)

    write_csv(out_dir / "summary_table.csv", summary_rows)
    write_csv(out_dir / "cross_type_summary.csv", cross_rows)
    manifest = {
        "params": params,
        "ctd_neuron_count": len(ctd_rows),
        "random_neuron_count": len(random_rows),
        "summary_rows": len(summary_rows),
        "cross_rows": len(cross_rows),
        "data_parallel": {"shard_count": shard_count, "merged_task_rows": len(data_rows)},
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def first_random_rows(shard_dirs: list[Path]) -> list[dict[str, Any]]:
    random_rows: list[dict[str, Any]] | None = None
    for shard_dir in shard_dirs:
        path = shard_dir / "random_mask_neurons.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if random_rows is None:
            random_rows = rows
        elif rows != random_rows:
            raise ValueError(f"Random mask mismatch across shards: {path}")
    return random_rows or []


def print_causal_metrics(subset: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"[Stage 10] {subset}: no causal summary rows")
        return
    ctd = next((row for row in rows if row.get("intervention") == "Mask-CTD"), rows[0])
    print(
        "[Stage 10] "
        f"{subset} causal: intervention={ctd.get('intervention')}, "
        f"avg_delta_acc_pp={ctd.get('avg_delta_acc_pp', ctd.get('avg_delta_acc', ''))}, "
        f"avg_tool_call_reduction_percent={ctd.get('avg_tool_call_reduction_percent', '')}, "
        f"avg_delta_avg_tool_calls={ctd.get('avg_delta_avg_tool_calls', '')}, "
        f"avg_delta_tool_call_rate={ctd.get('avg_delta_tool_call_rate', ctd.get('avg_delta_tcr', ''))}, "
        f"var_acc={ctd.get('var_acc', '')}"
    )
