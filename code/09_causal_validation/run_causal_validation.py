from __future__ import annotations

import argparse
import gc
import statistics
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import torch

from cttn.agent import HFGenerationAgent
from cttn.data import SUBSETS, TASK_TYPES
from cttn.eval_metrics import build_per_task, build_summary, write_csv
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.lora import rows_to_keys, sample_random_like
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import ProgressTracker, evaluate_batched_with_task_progress, progress
from cttn.when2tool_bridge import load_model_module, load_utils


STAGE_NAME = "10_causal_validation"
INTERVENTIONS = ("Base", "Mask-Random", "Mask-TDN_c", "Mask-CTD", "Mask-Private_c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10: causal validation with FFN activation masks.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--causal-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--max-test-samples", type=int, default=0)
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
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--progress-file", default=None)
    return parser.parse_args()


def parse_interventions(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    bad = [item for item in values if item not in INTERVENTIONS]
    if bad:
        raise ValueError(f"Unknown interventions: {bad}. Valid: {INTERVENTIONS}")
    return values


def subset_output_dir(causal_root: Path, model_alias: str, subset: str) -> Path:
    return causal_root / model_alias / subset


def expected_params(
    args: argparse.Namespace,
    subset: str,
    *,
    model_path: Path,
    tool_format: str,
    dataset_manifest: dict[str, Any],
    single_manifest: dict[str, Any],
    shared_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
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
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "dataset_manifest_params": dataset_manifest.get("params", {}),
        "single_type_manifest_params": single_manifest.get("params", {}),
        "shared_neuron_manifest_params": shared_manifest.get("params", {}),
    }


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    manifest_path = out_dir / "manifest.json"
    if overwrite or not manifest_path.exists() or not (out_dir / "summary_table.csv").exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing causal validation: {out_dir}")
        return True
    return False


def private_rows(tdn_rows: list[dict[str, Any]], ctd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctd_keys = rows_to_keys(ctd_rows)
    return [row for row in tdn_rows if (int(row["layer"]), str(row["module"]), int(row["index"])) not in ctd_keys]


def intervention_rows(
    name: str,
    *,
    task_type: str,
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
    raise ValueError(name)


def evaluate_intervention(
    *,
    agent: HFGenerationAgent,
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    w2t_utils: Any,
    tool_format: str,
    args: argparse.Namespace,
    tracker: ProgressTracker | None = None,
    desc: str = "causal tasks",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with agent.activation_mask(rows):
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
    return outputs, build_summary(per_task)


def run_subset(
    subset: str,
    *,
    args: argparse.Namespace,
    agent: HFGenerationAgent,
    model_dataset: Path,
    neurons_root: Path,
    out_dir: Path,
    w2t_utils: Any,
    tool_format: str,
    model_path: Path,
) -> dict[str, Any] | None:
    shared_dir = neurons_root / args.model_alias / "shared_by_subset" / subset
    single_dir = neurons_root / args.model_alias / "single_type_by_subset" / subset
    shared_manifest_path = shared_dir / "manifest.json"
    single_manifest_path = single_dir / "manifest.json"
    if not shared_manifest_path.exists():
        raise FileNotFoundError(f"Missing shared neuron manifest for {subset}: {shared_manifest_path}")
    if not single_manifest_path.exists():
        raise FileNotFoundError(f"Missing single-type manifest for {subset}: {single_manifest_path}")
    params = expected_params(
        args,
        subset,
        model_path=model_path,
        tool_format=tool_format,
        dataset_manifest=read_json(model_dataset / "manifest.json") if (model_dataset / "manifest.json").exists() else {},
        single_manifest=read_json(single_manifest_path),
        shared_manifest=read_json(shared_manifest_path),
    )
    if should_skip(out_dir, params, args.overwrite, args.clean):
        return None
    ensure_dir(out_dir)

    data = read_jsonl(model_dataset / subset / "test.jsonl")
    if args.max_test_samples > 0:
        data = data[: args.max_test_samples]

    ctd_rows = read_jsonl(shared_dir / "CTD_neurons.jsonl")
    random_rows = sample_random_like(ctd_rows, agent.model, seed=args.seed + len(subset), exclude_rows=ctd_rows)
    write_jsonl(out_dir / "random_mask_neurons.jsonl", random_rows)

    interventions = parse_interventions(args.interventions)
    summary_rows = []
    cross_by_intervention: dict[str, dict[str, Any]] = {}
    progress_total = sum(
        len([row for row in data if row.get("task_type") == task_type]) * len(interventions)
        for task_type in TASK_TYPES
    )
    tracker = ProgressTracker(args.progress_file, total=progress_total) if args.progress_file else None

    for task_type in progress(TASK_TYPES, desc=f"{subset} causal task types", unit="type"):
        task_rows = [row for row in data if row.get("task_type") == task_type]
        if not task_rows:
            continue
        tdn_rows = read_jsonl(single_dir / task_type / "TDN_neurons.jsonl")
        base_metrics = None
        type_metrics: dict[str, dict[str, Any]] = {}
        for intervention in progress(interventions, desc=f"{subset}/{task_type} interventions", unit="case", leave=False):
            mask_rows = intervention_rows(
                intervention,
                task_type=task_type,
                tdn_rows=tdn_rows,
                ctd_rows=ctd_rows,
                random_rows=random_rows,
            )
            case_dir = out_dir / task_type / intervention
            ensure_dir(case_dir)
            print(f"{subset}/{task_type}/{intervention}: tasks={len(task_rows)}, masked_neurons={len(mask_rows)}")
            outputs, summary = evaluate_intervention(
                agent=agent,
                tasks=task_rows,
                rows=mask_rows,
                w2t_utils=w2t_utils,
                tool_format=tool_format,
                args=args,
                tracker=tracker,
                desc=f"{subset}/{task_type}/{intervention} tasks",
            )
            per_task = build_per_task(outputs, w2t_utils)
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
            summary_rows.append(row)

        if base_metrics is not None:
            for intervention, metrics in type_metrics.items():
                cross = cross_by_intervention.setdefault(
                    intervention,
                    {"delta_acc": [], "delta_tcr": [], "task_type_metrics": {}},
                )
                cross["delta_acc"].append(float(metrics["final_accuracy"]) - float(base_metrics["final_accuracy"]))
                cross["delta_tcr"].append(float(metrics["tool_call_rate"]) - float(base_metrics["tool_call_rate"]))
                cross["task_type_metrics"][task_type] = metrics

    cross_rows = []
    for intervention, payload in sorted(cross_by_intervention.items()):
        deltas = payload["delta_acc"]
        delta_tcr = payload["delta_tcr"]
        metrics_by_type = payload["task_type_metrics"]
        row = {
            "model_alias": args.model_alias,
            "subset": subset,
            "intervention": intervention,
            "avg_delta_acc": sum(deltas) / len(deltas) if deltas else 0.0,
            "var_acc": statistics.pvariance(deltas) if len(deltas) > 1 else 0.0,
            "avg_delta_tcr": sum(delta_tcr) / len(delta_tcr) if delta_tcr else 0.0,
        }
        for task_type in TASK_TYPES:
            metrics = metrics_by_type.get(task_type, {})
            row[f"acc_{task_type}"] = metrics.get("final_accuracy")
            row[f"tool_acc_{task_type}"] = metrics.get("decision_accuracy")
            row[f"tcr_{task_type}"] = metrics.get("tool_call_rate")
        cross_rows.append(row)

    write_csv(out_dir / "summary_table.csv", summary_rows)
    write_csv(out_dir / "cross_type_summary.csv", cross_rows)
    manifest = {
        "params": params,
        "ctd_neuron_count": len(ctd_rows),
        "random_neuron_count": len(random_rows),
        "summary_rows": len(summary_rows),
        "cross_rows": len(cross_rows),
    }
    write_json(out_dir / "manifest.json", manifest)
    mask_ctd = next((row for row in cross_rows if row.get("intervention") == "Mask-CTD"), None)
    if mask_ctd:
        print(
            f"{subset}: Mask-CTD causal metrics "
            f"avg_delta_acc={mask_ctd.get('avg_delta_acc')}, "
            f"avg_delta_tcr={mask_ctd.get('avg_delta_tcr')}, "
            f"var_acc={mask_ctd.get('var_acc')}"
        )
    print(f"Wrote causal validation: {out_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    neurons_root = resolve_path(args.neurons_dir) if args.neurons_dir else path_from_config("neurons_dir")
    causal_root = resolve_path(args.causal_dir) if args.causal_dir else path_from_config("causal_validation_dir")
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
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]
    root_manifest = {"stage": STAGE_NAME, "model_alias": args.model_alias, "subsets": {}}
    try:
        for subset in subsets:
            out_dir = subset_output_dir(causal_root, args.model_alias, subset)
            summary = run_subset(
                subset,
                args=args,
                agent=agent,
                model_dataset=model_dataset,
                neurons_root=neurons_root,
                out_dir=out_dir,
                w2t_utils=w2t_utils,
                tool_format=tool_format,
                model_path=model_path,
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
    print(f"Wrote causal manifest: {manifest_path}")


if __name__ == "__main__":
    main()
