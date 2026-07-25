from __future__ import annotations

import argparse
import gc
import sys
import statistics
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
from cttn.data import TASK_TYPES
from cttn.eval_metrics import build_per_task, build_summary, write_csv
from cttn.paths import ensure_dir, path_from_config
from cttn.progress import ProgressTracker, evaluate_batched_with_task_progress, progress
from ps_common import (
    STAGE_VERSION,
    dataset_manifest,
    infer_tool_format,
    load_model_module,
    load_utils,
    ps_activation_mask,
    ps_neuron_key,
    ps_resolve_root,
    read_json,
    read_jsonl,
    resolve_model_path,
    resolve_path,
    sample_random_like_intermediate,
    select_records,
    stable_sha256,
    subset_values,
    write_json,
    write_jsonl,
)


INTERVENTIONS = ("Base", "Mask-Random", "Mask-PS-TDN_c", "Mask-PS-CTD", "Mask-PS-Private_c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield stage 10: causal validation with FFN h masks.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--causal-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="balanced")
    parser.add_argument("--interventions", default=",".join(INTERVENTIONS))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    parser.add_argument("--seed", type=int, choices=[2026], default=2026)
    parser.add_argument("--progress-file", default=None)
    return parser.parse_args()


def parse_interventions(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    alias = {
        "Mask-TDN_c": "Mask-PS-TDN_c",
        "Mask-CTD": "Mask-PS-CTD",
        "Mask-Private_c": "Mask-PS-Private_c",
    }
    values = [alias.get(item, item) for item in values]
    bad = [item for item in values if item not in INTERVENTIONS]
    if bad:
        raise ValueError(f"Unknown interventions: {bad}. Valid: {INTERVENTIONS}")
    return values


def subset_output_dir(causal_root: Path, model_alias: str, subset: str) -> Path:
    return causal_root / model_alias / subset


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
    tool_format: str,
    single_manifest: dict[str, Any],
    shared_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "ps_10_causal_validation",
        "stage_version": STAGE_VERSION,
        "method": "PreciseShield",
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "selected_rows": {"count": len(rows), "sha256": stable_sha256([row["id"] for row in rows])},
        "max_test_samples": args.max_test_samples,
        "sample_strategy": args.sample_strategy,
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
        "activation_mask": "zero FFN intermediate h at down_proj input for all token positions",
        "dataset_manifest_params": dataset_manifest(model_dataset).get("params", {}),
        "single_type_manifest_params": single_manifest.get("params", {}),
        "shared_neuron_manifest_params": shared_manifest.get("params", {}),
    }


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        from ps_common import clean_path

        clean_path(out_dir)
        return False
    manifest_path = out_dir / "manifest.json"
    if overwrite or not manifest_path.exists() or not (out_dir / "summary_table.csv").exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing PreciseShield causal validation: {out_dir}")
        return True
    return False


def private_rows(tdn_rows: list[dict[str, Any]], ctd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctd_keys = {ps_neuron_key(row) for row in ctd_rows}
    return [row for row in tdn_rows if ps_neuron_key(row) not in ctd_keys]


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
    if name == "Mask-PS-TDN_c":
        return tdn_rows
    if name == "Mask-PS-CTD":
        return ctd_rows
    if name == "Mask-PS-Private_c":
        return private_rows(tdn_rows, ctd_rows)
    raise ValueError(name)


def evaluate_intervention(
    *,
    agent: HFGenerationAgent,
    tasks: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    w2t_utils: Any,
    tool_format: str,
    args: argparse.Namespace,
    tracker: ProgressTracker | None = None,
    desc: str = "PS causal tasks",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with ps_activation_mask(agent.model, mask_rows):
        outputs = evaluate_batched_with_task_progress(
            w2t_utils,
            tasks,
            agent,
            batch_size=args.batch_size,
            desc=desc,
            tracker=tracker,
            max_rounds=args.max_rounds,
            record_mode=args.record_mode,
            prompt_mode="current",
            require_reasoning=False,
            tool_format=tool_format,
        )
    per_task = build_per_task(outputs, w2t_utils, run_id=0)
    return outputs, per_task, build_summary(per_task)


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


def run_subset(
    subset: str,
    *,
    args: argparse.Namespace,
    agent: HFGenerationAgent,
    model_path: Path,
    model_dataset: Path,
    neurons_root: Path,
    out_dir: Path,
    w2t_utils: Any,
    tool_format: str,
) -> dict[str, Any] | None:
    single_dir = neurons_root / args.model_alias / "single_type_by_subset" / subset
    shared_dir = neurons_root / args.model_alias / "shared_by_subset" / subset
    single_manifest_path = single_dir / "manifest.json"
    shared_manifest_path = shared_dir / "manifest.json"
    if not single_manifest_path.exists() or not shared_manifest_path.exists():
        raise FileNotFoundError(f"Missing PreciseShield neuron manifests for {subset}")
    rows = load_test_rows(model_dataset, subset, args)
    params = expected_params(
        args,
        subset=subset,
        model_path=model_path,
        model_dataset=model_dataset,
        rows=rows,
        tool_format=tool_format,
        single_manifest=read_json(single_manifest_path),
        shared_manifest=read_json(shared_manifest_path),
    )
    if should_skip(out_dir, params, args.overwrite, args.clean):
        return None
    ensure_dir(out_dir)

    ctd_rows = read_jsonl(shared_dir / "PS_CTD_neurons.jsonl")
    random_rows = sample_random_like_intermediate(
        ctd_rows,
        agent.model,
        seed=args.seed,
        exclude_rows=ctd_rows,
    )
    write_jsonl(out_dir / "random_mask_neurons.jsonl", random_rows)
    interventions = parse_interventions(args.interventions)
    summary_rows = []
    cross_by_intervention: dict[str, dict[str, Any]] = {}
    progress_total = sum(
        len([row for row in rows if row.get("task_type") == task_type]) * len(interventions)
        for task_type in TASK_TYPES
    )
    tracker = ProgressTracker(args.progress_file, total=progress_total) if args.progress_file else None

    for task_type in progress(TASK_TYPES, desc=f"{subset} PS causal task types", unit="type"):
        task_rows = [row for row in rows if row.get("task_type") == task_type]
        if not task_rows:
            continue
        tdn_rows = read_jsonl(single_dir / task_type / "PS_TDN_neurons.jsonl")
        base_metrics = None
        type_metrics: dict[str, dict[str, Any]] = {}
        type_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for intervention in interventions:
            mask_rows = intervention_rows(
                intervention,
                tdn_rows=tdn_rows,
                ctd_rows=ctd_rows,
                random_rows=random_rows,
            )
            case_dir = out_dir / task_type / intervention
            ensure_dir(case_dir)
            print(f"{subset}/{task_type}/{intervention}: tasks={len(task_rows)}, masked_neurons={len(mask_rows)}")
            outputs, per_task, summary = evaluate_intervention(
                agent=agent,
                tasks=task_rows,
                mask_rows=mask_rows,
                w2t_utils=w2t_utils,
                tool_format=tool_format,
                args=args,
                tracker=tracker,
                desc=f"{subset}/{task_type}/{intervention} tasks",
            )
            write_json(case_dir / "outputs.json", outputs)
            write_jsonl(case_dir / "per_task.jsonl", per_task)
            write_json(case_dir / "summary.json", summary)
            metrics = summary["overall"]
            type_metrics[intervention] = metrics
            if intervention == "Base":
                base_metrics = metrics
            row = {
                "model_alias": args.model_alias,
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
            "model_alias": args.model_alias,
            "subset": subset,
            "intervention": intervention,
            "avg_delta_acc": sum(deltas) / len(deltas) if deltas else 0.0,
            "avg_delta_acc_pp": 100.0 * (sum(deltas) / len(deltas)) if deltas else 0.0,
            "var_acc": statistics.pvariance(deltas) if len(deltas) > 1 else 0.0,
            "avg_delta_tcr": sum(delta_tcr) / len(delta_tcr) if delta_tcr else 0.0,
            "avg_delta_tool_call_rate": sum(delta_tcr) / len(delta_tcr) if delta_tcr else 0.0,
            "avg_delta_avg_tool_calls": (
                sum(delta_avg_tc_values) / len(delta_avg_tc_values)
                if delta_avg_tc_values
                else 0.0
            ),
            "avg_delta_total_tool_calls_percent": (
                sum(delta_total_tc_pct_values) / len(delta_total_tc_pct_values)
                if delta_total_tc_pct_values
                else ""
            ),
            "avg_tool_call_reduction_percent": (
                sum(tool_reduction_values) / len(tool_reduction_values)
                if tool_reduction_values
                else ""
            ),
            "avg_delta_acc_per_delta_avg_tool_call": (
                sum(acc_per_delta_tc_values) / len(acc_per_delta_tc_values)
                if acc_per_delta_tc_values
                else ""
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
            row[f"delta_acc_per_delta_avg_tool_call_{task_type}"] = delta_row.get(
                "delta_acc_per_delta_avg_tool_call"
            )
            row[f"acc_cost_per_saved_call_{task_type}"] = delta_row.get("acc_cost_per_saved_call")
        cross_rows.append(row)

    write_csv(out_dir / "summary_table.csv", summary_rows)
    write_csv(out_dir / "cross_type_summary.csv", cross_rows)
    manifest = {
        "params": params,
        "ps_ctd_neuron_count": len(ctd_rows),
        "random_neuron_count": len(random_rows),
        "summary_rows": len(summary_rows),
        "cross_rows": len(cross_rows),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote PreciseShield causal validation: {out_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    neurons_root = ps_resolve_root(args.neurons_dir, "neurons")
    causal_root = ps_resolve_root(args.causal_dir, "causal")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")
    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)
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
        batch_size=args.batch_size,
        enable_thinking=False,
    )
    root_manifest = {"stage": "ps_10_causal_validation", "model_alias": args.model_alias, "subsets": {}}
    try:
        for subset in subset_values(args.subset):
            summary = run_subset(
                subset,
                args=args,
                agent=agent,
                model_path=model_path,
                model_dataset=model_dataset,
                neurons_root=neurons_root,
                out_dir=subset_output_dir(causal_root, args.model_alias, subset),
                w2t_utils=w2t_utils,
                tool_format=tool_format,
            )
            if summary is not None:
                root_manifest["subsets"][subset] = summary
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        agent.close()
        gc.collect()
    manifest_path = causal_root / args.model_alias / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote PreciseShield causal manifest: {manifest_path}")


if __name__ == "__main__":
    main()
