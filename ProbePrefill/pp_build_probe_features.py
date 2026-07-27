from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from pp_common import (
    PP_STAGE_VERSION,
    PROBE_METHOD_SAFETY_KERNEL,
    default_method_activations_dir,
    default_method_neurons_dir,
    features_from_rows,
    load_shared_neuron_rows,
    load_stage_activations,
    method_feature_definition,
    method_feature_description,
    method_feature_set,
    method_label,
    method_neuron_identity,
    normalize_probe_method,
    prepare_probe_method_root,
    print_subset_plan,
    probe_method_choices,
    pp_subdir,
    probe_prefill_root,
    read_json,
    resolve_path,
    select_meta_indices,
    should_skip,
    stable_sha256,
    summarize_labels,
    write_json,
    write_jsonl,
)
from cttn.seeds import derive_allowed_seed, seed_arg_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProbePrefill stage 1: build shared-neuron activation features.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--probe-method", choices=probe_method_choices(), default=PROBE_METHOD_SAFETY_KERNEL)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="balanced")
    parser.add_argument("--require-per-type-labels", action="store_true")
    parser.add_argument("--seed", **seed_arg_kwargs())
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_max_samples(args: argparse.Namespace, split: str) -> int:
    return args.max_train_samples if split == "train" else args.max_test_samples


def output_dir(features_root: Path, model_alias: str, subset: str) -> Path:
    return features_root / model_alias / subset


def feature_params(
    args: argparse.Namespace,
    *,
    subset: str,
    split: str,
    ctd_rows: list[dict[str, Any]],
    activation_manifest: dict[str, Any],
    selected_meta: list[dict[str, Any]],
    activations_dir: Path,
    neurons_dir: Path,
) -> dict[str, Any]:
    legacy_params = {
        "stage": "pp_01_build_ctd_probe_features",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "subset": subset,
        "split": split,
        "activations_dir": str(activations_dir),
        "neurons_dir": str(neurons_dir),
        "ctd_neuron_count": len(ctd_rows),
        "ctd_neuron_sha256": stable_sha256(
            [
                {"layer": row["layer"], "module": row["module"], "index": row["index"]}
                for row in ctd_rows
            ]
        ),
        "activation_manifest_params": activation_manifest.get("params", {}),
        "max_samples": split_max_samples(args, split),
        "sample_strategy": args.sample_strategy,
        "require_per_type_labels": bool(args.require_per_type_labels),
        "seed": args.seed,
        "selected_ids_sha256": stable_sha256([row["id"] for row in selected_meta]),
        "feature_definition": "stage4 last-input-token FFN output activation restricted to stage6 CTD neurons",
    }
    if args.probe_method == PROBE_METHOD_SAFETY_KERNEL:
        return legacy_params
    return {
        **legacy_params,
        "stage": "pp_01_build_probe_features",
        "probe_method": args.probe_method,
        "probe_method_label": method_label(args.probe_method),
        "feature_set": method_feature_set(args.probe_method),
        "feature_description": method_feature_description(args.probe_method),
        "shared_neuron_count": len(ctd_rows),
        "shared_neuron_sha256": stable_sha256([method_neuron_identity(row) for row in ctd_rows]),
        "feature_definition": method_feature_definition(args.probe_method),
    }


def build_split(
    args: argparse.Namespace,
    *,
    subset: str,
    split: str,
    features_root: Path,
    activations_dir: Path,
    neurons_dir: Path,
    ctd_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload, meta_rows, activation_manifest = load_stage_activations(activations_dir, args.model_alias, subset, split)
    indices = select_meta_indices(
        meta_rows,
        split_max_samples(args, split),
        derive_allowed_seed(args.seed, subset, split, "probe_features"),
        strategy=args.sample_strategy,
        require_per_type_labels=args.require_per_type_labels,
    )
    features, labels, selected_meta, ordered_rows = features_from_rows(payload, meta_rows, ctd_rows, indices)
    out_dir = output_dir(features_root, args.model_alias, subset)
    params = feature_params(
        args,
        subset=subset,
        split=split,
        ctd_rows=ordered_rows,
        activation_manifest=activation_manifest,
        selected_meta=selected_meta,
        activations_dir=activations_dir,
        neurons_dir=neurons_dir,
    )
    expected = [out_dir / f"{split}_features.pt", out_dir / f"{split}_meta.jsonl", out_dir / f"{split}_summary.json"]
    if should_skip(
        out_dir / split,
        params,
        expected,
        overwrite=args.overwrite,
        clean=args.clean,
        allowed_root=features_root,
    ):
        return read_json(out_dir / f"{split}_summary.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": torch.tensor(labels, dtype=torch.long),
            "neuron_rows": ordered_rows,
            "feature_shape": list(features.shape),
        },
        out_dir / f"{split}_features.pt",
    )
    write_jsonl(out_dir / f"{split}_meta.jsonl", selected_meta)
    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "split": split,
        "feature_shape": list(features.shape),
        "label_summary": summarize_labels(selected_meta),
        "ctd_neuron_count": len(ordered_rows),
    }
    if args.probe_method != PROBE_METHOD_SAFETY_KERNEL:
        summary.update(
            {
                "probe_method": args.probe_method,
                "feature_set": method_feature_set(args.probe_method),
                "shared_neuron_count": len(ordered_rows),
            }
        )
    write_json(out_dir / f"{split}_summary.json", summary)
    write_json(out_dir / split / "manifest.json", {"params": params, "summary": summary})
    print(f"Wrote {subset}/{split} {method_feature_set(args.probe_method)} probe features: {out_dir / f'{split}_features.pt'}")
    return summary


def main() -> None:
    args = parse_args()
    args.probe_method = normalize_probe_method(args.probe_method)
    activations_dir = resolve_path(args.activations_dir) if args.activations_dir else default_method_activations_dir(args.probe_method)
    neurons_dir = resolve_path(args.neurons_dir) if args.neurons_dir else default_method_neurons_dir(args.probe_method)
    root = prepare_probe_method_root(probe_prefill_root(args.output_root), args.probe_method)
    features_root = pp_subdir(root, "features")
    subsets = print_subset_plan(args.subset, stage="PP-1", model_alias=args.model_alias)
    root_manifest: dict[str, Any] = {
        "stage": "pp_01_build_ctd_probe_features" if args.probe_method == PROBE_METHOD_SAFETY_KERNEL else "pp_01_build_probe_features",
        "stage_version": PP_STAGE_VERSION,
        "probe_method": args.probe_method,
        "feature_set": method_feature_set(args.probe_method),
        "model_alias": args.model_alias,
        "subsets": {},
    }

    for subset in subsets:
        ctd_rows = load_shared_neuron_rows(neurons_dir, args.model_alias, subset, args.probe_method)
        subset_summary = {}
        for split in ["train", "test"]:
            subset_summary[split] = build_split(
                args,
                subset=subset,
                split=split,
                features_root=features_root,
                activations_dir=activations_dir,
                neurons_dir=neurons_dir,
                ctd_rows=ctd_rows,
            )
        subset_summary["ctd_neuron_count"] = len(ctd_rows)
        subset_summary["feature_set"] = method_feature_set(args.probe_method)
        root_manifest["subsets"][subset] = subset_summary

    manifest_path = features_root / args.model_alias / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote ProbePrefill feature manifest: {manifest_path}")


if __name__ == "__main__":
    main()
