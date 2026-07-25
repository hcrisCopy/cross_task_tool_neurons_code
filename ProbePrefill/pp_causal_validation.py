from __future__ import annotations

import argparse
import gc
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pp_common import (
    PP_STAGE_VERSION,
    features_from_rows,
    grouped_classification_metrics,
    infer_tool_format,
    load_ctd_rows,
    load_model_module,
    load_stage_activations,
    load_tdn_rows,
    load_utils,
    path_from_config,
    pp_subdir,
    private_rows,
    probe_prefill_root,
    read_json,
    read_jsonl,
    remove_files,
    resolve_model_path,
    resolve_path,
    sample_random_like_rows,
    should_skip,
    stable_sha256,
    subset_values,
    write_csv,
    write_json,
    write_jsonl,
)

from cttn.agent import HFGenerationAgent
from cttn.data import TASK_TYPES
from cttn.eval_metrics import build_per_task, build_summary
from cttn.lora import sample_random_like
from cttn.seeds import derive_allowed_seed, seed_arg_kwargs


INTERVENTIONS = ("Base", "Mask-Random", "Mask-TDN_c", "Mask-CTD", "Mask-Private_c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProbePrefill stage 5: causal validation for CTD probe features and CTD neurons.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--reg", type=float, default=10000.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interventions", default=",".join(INTERVENTIONS))
    parser.add_argument("--skip-probe-controls", action="store_true")
    parser.add_argument("--skip-activation-mask", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    parser.add_argument("--seed", **seed_arg_kwargs())
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_interventions(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    bad = [item for item in values if item not in INTERVENTIONS]
    if bad:
        raise ValueError(f"Unknown interventions: {bad}. Valid: {INTERVENTIONS}")
    return values


def causal_dir(root: Path, model_alias: str, subset: str) -> Path:
    return pp_subdir(root, "causal") / model_alias / subset


def feature_meta(root: Path, model_alias: str, subset: str, split: str) -> list[dict[str, Any]]:
    path = pp_subdir(root, "features") / model_alias / subset / f"{split}_meta.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing ProbePrefill feature metadata: {path}")
    return read_jsonl(path)


def indices_for_feature_meta(stage_meta: list[dict[str, Any]], feature_meta_rows: list[dict[str, Any]]) -> list[int]:
    by_id = {str(row["id"]): idx for idx, row in enumerate(stage_meta)}
    indices = []
    missing = []
    for row in feature_meta_rows:
        idx = by_id.get(str(row["id"]))
        if idx is None:
            missing.append(str(row["id"]))
        else:
            indices.append(idx)
    if missing:
        raise KeyError(f"Feature metadata ids missing from stage activations: {missing[:5]}")
    return indices


def tasks_for_feature_meta(model_dataset: Path, subset: str, feature_rows: list[dict[str, Any]], task_type: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(model_dataset / subset / "test.jsonl")
    by_id = {str(row["id"]): row for row in rows}
    tasks = []
    for meta in feature_rows:
        task = by_id.get(str(meta["id"]))
        if task is not None and (task_type is None or task.get("task_type") == task_type):
            tasks.append(task)
    return tasks


def fit_and_eval_probe(
    *,
    X_train: torch.Tensor,
    y_train: np.ndarray,
    meta_train: list[dict[str, Any]],
    X_test: torch.Tensor,
    y_test: np.ndarray,
    meta_test: list[dict[str, Any]],
    reg: float,
    max_iter: int,
    threshold: float,
) -> dict[str, Any]:
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Probe control training labels contain only one class")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train.numpy())
    x_test = scaler.transform(X_test.numpy())
    clf = LogisticRegression(C=1.0 / reg, solver="lbfgs", max_iter=max_iter, random_state=42)
    clf.fit(x_train, y_train)
    train_prob = clf.predict_proba(x_train)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]
    return {
        "train": grouped_classification_metrics(y_train, train_prob, meta_train, threshold=threshold),
        "test": grouped_classification_metrics(y_test, test_prob, meta_test, threshold=threshold),
    }


def control_feature_sets(
    *,
    neurons_dir: Path,
    model_alias: str,
    subset: str,
    module_meta: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    ctd = load_ctd_rows(neurons_dir, model_alias, subset)
    result = {
        "CTD": ctd,
        "Random-CTD": sample_random_like_rows(ctd, module_meta, seed=seed, exclude_rows=ctd),
    }
    for task_type in TASK_TYPES:
        tdn = load_tdn_rows(neurons_dir, model_alias, subset, task_type)
        result[f"TDN_{task_type}"] = tdn
        result[f"Private_{task_type}"] = private_rows(tdn, ctd)
    return result


def run_probe_controls(
    args: argparse.Namespace,
    *,
    subset: str,
    root: Path,
    activations_dir: Path,
    neurons_dir: Path,
) -> dict[str, Any]:
    out_dir = causal_dir(root, args.model_alias, subset)
    params = {
        "stage": "pp_05_probe_feature_controls",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "subset": subset,
        "reg": args.reg,
        "C": 1.0 / args.reg,
        "max_iter": args.max_iter,
        "threshold": args.threshold,
        "seed": args.seed,
        "random_control_seed": derive_allowed_seed(args.seed, subset, "probe_controls"),
        "activations_dir": str(activations_dir),
        "neurons_dir": str(neurons_dir),
    }
    expected = [out_dir / "probe_control_summary.csv", out_dir / "probe_control_results.json"]
    if should_skip(out_dir / "probe_controls", params, expected, overwrite=args.overwrite, clean=args.clean, allowed_root=pp_subdir(root, "causal")):
        return read_json(out_dir / "probe_control_results.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_payload, stage_train_meta, _train_manifest = load_stage_activations(activations_dir, args.model_alias, subset, "train")
    test_payload, stage_test_meta, _test_manifest = load_stage_activations(activations_dir, args.model_alias, subset, "test")
    train_feature_meta = feature_meta(root, args.model_alias, subset, "train")
    test_feature_meta = feature_meta(root, args.model_alias, subset, "test")
    train_indices = indices_for_feature_meta(stage_train_meta, train_feature_meta)
    test_indices = indices_for_feature_meta(stage_test_meta, test_feature_meta)
    sets = control_feature_sets(
        neurons_dir=neurons_dir,
        model_alias=args.model_alias,
        subset=subset,
        module_meta=train_payload["module_meta"],
        seed=derive_allowed_seed(args.seed, subset, "probe_controls"),
    )

    summary_rows = []
    results: dict[str, Any] = {"params": params, "sets": {}}
    for name, rows in sets.items():
        if not rows:
            continue
        print(f"{subset}/probe-control/{name}: neurons={len(rows)}")
        X_train, y_train, meta_train, ordered_rows = features_from_rows(train_payload, stage_train_meta, rows, train_indices)
        X_test, y_test, meta_test, _ = features_from_rows(test_payload, stage_test_meta, ordered_rows, test_indices)
        metrics = fit_and_eval_probe(
            X_train=X_train,
            y_train=y_train,
            meta_train=meta_train,
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test,
            reg=args.reg,
            max_iter=args.max_iter,
            threshold=args.threshold,
        )
        result = {
            "feature_set": name,
            "neuron_count": len(ordered_rows),
            "neuron_sha256": stable_sha256(
                [{"layer": row["layer"], "module": row["module"], "index": row["index"]} for row in ordered_rows]
            ),
            "metrics": metrics,
        }
        results["sets"][name] = result
        row = {
            "model_alias": args.model_alias,
            "subset": subset,
            "feature_set": name,
            "neuron_count": len(ordered_rows),
            "train_n": int(X_train.shape[0]),
            "test_n": int(X_test.shape[0]),
        }
        row.update({f"test_{key}": value for key, value in metrics["test"]["overall"].items() if key != "confusion_matrix"})
        summary_rows.append(row)
    write_json(out_dir / "probe_control_results.json", results)
    write_csv(out_dir / "probe_control_summary.csv", summary_rows)
    write_json(out_dir / "probe_controls" / "manifest.json", {"params": params, "rows": len(summary_rows)})
    print(f"Wrote probe feature controls: {out_dir / 'probe_control_summary.csv'}")
    return results


def intervention_rows(name: str, *, task_type: str, tdn_rows: list[dict[str, Any]], ctd_rows: list[dict[str, Any]], random_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def evaluate_mask_case(
    *,
    agent: HFGenerationAgent,
    tasks: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    w2t_utils: Any,
    tool_format: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with agent.activation_mask(mask_rows):
        outputs = w2t_utils.evaluate_batched(
            tasks,
            agent,
            max_rounds=args.max_rounds,
            record_mode=args.record_mode,
            prompt_mode="current",
            require_reasoning=False,
            tool_format=tool_format,
        )
    per_task = build_per_task(outputs, w2t_utils, run_id=0)
    return outputs, per_task, build_summary(per_task)


def run_activation_mask_validation(
    args: argparse.Namespace,
    *,
    subset: str,
    root: Path,
    model_path: Path,
    model_dataset: Path,
    neurons_dir: Path,
    w2t_utils: Any,
    w2t_model: Any,
    tool_format: str,
) -> dict[str, Any]:
    out_dir = causal_dir(root, args.model_alias, subset)
    causal_root = pp_subdir(root, "causal")
    ctd_rows = load_ctd_rows(neurons_dir, args.model_alias, subset)
    single_manifest = read_json(neurons_dir / args.model_alias / "single_type_by_subset" / subset / "manifest.json")
    shared_manifest = read_json(neurons_dir / args.model_alias / "shared_by_subset" / subset / "manifest.json")
    feature_rows = feature_meta(root, args.model_alias, subset, "test")
    params = {
        "stage": "pp_05_activation_mask_validation",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "interventions": parse_interventions(args.interventions),
        "batch_size": args.batch_size,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "record_mode": args.record_mode,
        "seed": args.seed,
        "random_mask_seed": derive_allowed_seed(args.seed, subset, "activation_mask"),
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "selected_test_ids_sha256": stable_sha256([row["id"] for row in feature_rows]),
        "single_type_manifest_params": single_manifest.get("params", {}),
        "shared_neuron_manifest_params": shared_manifest.get("params", {}),
    }
    expected = [out_dir / "summary_table.csv", out_dir / "cross_type_summary.csv"]
    if should_skip(out_dir / "activation_mask", params, expected, overwrite=args.overwrite, clean=args.clean, allowed_root=pp_subdir(root, "causal")):
        return read_json(out_dir / "activation_mask" / "manifest.json")

    if args.clean:
        remove_files([out_dir / task_type for task_type in TASK_TYPES], allowed_root=causal_root)
        remove_files(
            [out_dir / "summary_table.csv", out_dir / "cross_type_summary.csv", out_dir / "random_mask_neurons.jsonl"],
            allowed_root=causal_root,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
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
        random_seed = derive_allowed_seed(args.seed, subset, "activation_mask")
        random_rows = sample_random_like(ctd_rows, agent.model, seed=random_seed, exclude_rows=ctd_rows)
        write_jsonl(out_dir / "random_mask_neurons.jsonl", random_rows)
        summary_rows = []
        cross_by_intervention: dict[str, dict[str, Any]] = {}
        interventions = parse_interventions(args.interventions)
        for task_type in TASK_TYPES:
            tasks = tasks_for_feature_meta(model_dataset, subset, feature_rows, task_type=task_type)
            if not tasks:
                continue
            tdn_rows = load_tdn_rows(neurons_dir, args.model_alias, subset, task_type)
            base_metrics = None
            type_metrics: dict[str, dict[str, Any]] = {}
            for intervention in interventions:
                mask_rows = intervention_rows(
                    intervention,
                    task_type=task_type,
                    tdn_rows=tdn_rows,
                    ctd_rows=ctd_rows,
                    random_rows=random_rows,
                )
                case_dir = out_dir / task_type / intervention
                case_dir.mkdir(parents=True, exist_ok=True)
                print(f"{subset}/{task_type}/{intervention}: tasks={len(tasks)}, masked_neurons={len(mask_rows)}")
                outputs, per_task, summary = evaluate_mask_case(
                    agent=agent,
                    tasks=tasks,
                    mask_rows=mask_rows,
                    w2t_utils=w2t_utils,
                    tool_format=tool_format,
                    args=args,
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
                summary_rows.append(row)

            if base_metrics is not None:
                for intervention, metrics in type_metrics.items():
                    cross = cross_by_intervention.setdefault(intervention, {"delta_acc": [], "delta_tcr": [], "task_type_metrics": {}})
                    cross["delta_acc"].append(float(metrics["final_accuracy"]) - float(base_metrics["final_accuracy"]))
                    cross["delta_tcr"].append(float(metrics["tool_call_rate"]) - float(base_metrics["tool_call_rate"]))
                    cross["task_type_metrics"][task_type] = metrics

        cross_rows = []
        for intervention, payload in sorted(cross_by_intervention.items()):
            deltas = payload["delta_acc"]
            delta_tcr = payload["delta_tcr"]
            row = {
                "model_alias": args.model_alias,
                "subset": subset,
                "intervention": intervention,
                "avg_delta_acc": sum(deltas) / len(deltas) if deltas else 0.0,
                "var_acc": statistics.pvariance(deltas) if len(deltas) > 1 else 0.0,
                "avg_delta_tcr": sum(delta_tcr) / len(delta_tcr) if delta_tcr else 0.0,
            }
            for task_type in TASK_TYPES:
                metrics = payload["task_type_metrics"].get(task_type, {})
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
        write_json(out_dir / "activation_mask" / "manifest.json", manifest)
        print(f"Wrote activation-mask causal validation: {out_dir}")
        return manifest
    finally:
        agent.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.reg <= 0:
        raise ValueError("--reg must be positive")
    root = probe_prefill_root(args.output_root)
    activations_dir = resolve_path(args.activations_dir) if args.activations_dir else path_from_config("activations_dir")
    neurons_dir = resolve_path(args.neurons_dir) if args.neurons_dir else path_from_config("neurons_dir")
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset: {model_dataset}")
    model_path = resolve_model_path(args.model_alias, args.model_path)
    tool_format = infer_tool_format(args.model_alias, model_path)
    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)

    root_manifest: dict[str, Any] = {
        "stage": "pp_05_causal_validation",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "tool_format": tool_format,
        "subsets": {},
    }
    for subset in subset_values(args.subset):
        subset_manifest: dict[str, Any] = {}
        if not args.skip_probe_controls:
            subset_manifest["probe_controls"] = run_probe_controls(
                args,
                subset=subset,
                root=root,
                activations_dir=activations_dir,
                neurons_dir=neurons_dir,
            )
        if not args.skip_activation_mask:
            subset_manifest["activation_mask"] = run_activation_mask_validation(
                args,
                subset=subset,
                root=root,
                model_path=model_path,
                model_dataset=model_dataset,
                neurons_dir=neurons_dir,
                w2t_utils=w2t_utils,
                w2t_model=w2t_model,
                tool_format=tool_format,
            )
        root_manifest["subsets"][subset] = subset_manifest

    manifest_path = pp_subdir(root, "causal") / args.model_alias / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote ProbePrefill causal manifest: {manifest_path}")


if __name__ == "__main__":
    main()
