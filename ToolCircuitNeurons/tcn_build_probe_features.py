from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PREFILL_DIR = REPO_ROOT / "ProbePrefill"
COMMON_DIR = REPO_ROOT / "code" / "00_common"
for path in (PROBE_PREFILL_DIR, COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cttn.progress import progress
from pp_common import (
    PP_STAGE_VERSION,
    PROBE_METHOD_TOOL_CIRCUIT_NEURONS,
    PROBE_METHOD_TOOL_DECISION_ANCHORS,
    PROBE_METHOD_TOOL_KNOWLEDGE_NEURONS,
    PROBE_METHOD_TOOL_ROUTING_NEURONS,
    clean_path,
    method_feature_definition,
    method_feature_description,
    method_feature_set,
    method_label,
    normalize_probe_method,
    pp_subdir,
    prepare_probe_method_root,
    print_subset_plan,
    probe_method_root,
    probe_prefill_root,
    read_json,
    read_jsonl,
    should_skip,
    stable_sha256,
    summarize_labels,
    task_id,
    write_json,
    write_jsonl,
)


DEFAULT_SOURCE_METHODS = (
    PROBE_METHOD_TOOL_DECISION_ANCHORS,
    PROBE_METHOD_TOOL_KNOWLEDGE_NEURONS,
    PROBE_METHOD_TOOL_ROUTING_NEURONS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ToolCircuitNeurons stage 1: fuse already-built ProbePrefill features from "
            "complementary shared-neuron spaces."
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--source-root", default=None, help="ProbePrefill root containing source method feature folders.")
    parser.add_argument("--output-root", default=None, help="ProbePrefill root for ToolCircuitNeurons outputs.")
    parser.add_argument(
        "--source-methods",
        default=",".join(DEFAULT_SOURCE_METHODS),
        help="Comma-separated source probe methods to concatenate, in feature order.",
    )
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_source_methods(text: str) -> list[str]:
    methods = [normalize_probe_method(item.strip()) for item in text.split(",") if item.strip()]
    if not methods:
        raise ValueError("--source-methods must contain at least one source method")
    if PROBE_METHOD_TOOL_CIRCUIT_NEURONS in methods:
        raise ValueError("ToolCircuitNeurons cannot use itself as a source method")
    seen: set[str] = set()
    unique: list[str] = []
    for method in methods:
        if method in seen:
            continue
        seen.add(method)
        unique.append(method)
    return unique


def output_dir(features_root: Path, model_alias: str, subset: str) -> Path:
    return features_root / model_alias / subset


def source_feature_dir(source_root: Path, source_method: str, model_alias: str, subset: str) -> Path:
    method_root = probe_method_root(source_root, source_method)
    return pp_subdir(method_root, "features") / model_alias / subset


def load_source_split(
    *,
    source_root: Path,
    source_method: str,
    model_alias: str,
    subset: str,
    split: str,
) -> dict[str, Any]:
    base = source_feature_dir(source_root, source_method, model_alias, subset)
    payload_path = base / f"{split}_features.pt"
    meta_path = base / f"{split}_meta.jsonl"
    summary_path = base / f"{split}_summary.json"
    if not payload_path.exists() or not meta_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Missing source ProbePrefill features for {method_label(source_method)} {subset}/{split}: {base}"
        )
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    features = payload.get("features")
    labels = payload.get("labels")
    neuron_rows = payload.get("neuron_rows")
    if features is None or labels is None or neuron_rows is None:
        raise KeyError(f"Source payload is missing features/labels/neuron_rows: {payload_path}")
    features = features.float().cpu().contiguous()
    labels = labels.cpu().long().contiguous()
    if features.ndim != 2:
        raise ValueError(f"Source features must be a rank-2 tensor: {payload_path}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(f"Source labels do not match feature rows: {payload_path}")
    if len(neuron_rows) != features.shape[1]:
        raise ValueError(
            f"Source neuron row count does not match feature dim for {source_method} {subset}/{split}: "
            f"rows={len(neuron_rows)} dim={features.shape[1]}"
        )
    meta_rows = read_jsonl(meta_path)
    if len(meta_rows) != features.shape[0]:
        raise ValueError(
            f"Source metadata rows do not match feature rows for {source_method} {subset}/{split}: "
            f"meta={len(meta_rows)} features={features.shape[0]}"
        )
    return {
        "method": source_method,
        "feature_set": method_feature_set(source_method),
        "payload_path": payload_path,
        "meta_path": meta_path,
        "summary_path": summary_path,
        "payload": payload,
        "features": features,
        "labels": labels,
        "neuron_rows": list(neuron_rows),
        "meta_rows": meta_rows,
        "summary": read_json(summary_path),
    }


def validate_aligned_sources(source_splits: list[dict[str, Any]], *, subset: str, split: str) -> tuple[list[str], torch.Tensor, list[dict[str, Any]]]:
    base = source_splits[0]
    ids = [task_id(row) for row in base["meta_rows"]]
    labels = base["labels"]
    meta_rows = base["meta_rows"]
    for item in source_splits[1:]:
        item_ids = [task_id(row) for row in item["meta_rows"]]
        if item_ids != ids:
            raise ValueError(
                f"Source feature ids are not exactly aligned for {subset}/{split}: "
                f"{base['method']} vs {item['method']}"
            )
        if not torch.equal(item["labels"], labels):
            raise ValueError(
                f"Source labels are not exactly aligned for {subset}/{split}: "
                f"{base['method']} vs {item['method']}"
            )
    return ids, labels, meta_rows


def fused_neuron_rows(source_splits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for item in source_splits:
        source_method = item["method"]
        source_feature_set = item["feature_set"]
        for local_index, raw_row in enumerate(item["neuron_rows"]):
            row = dict(raw_row)
            row["source_method"] = source_method
            row["source_label"] = method_label(source_method)
            row["source_feature_set"] = source_feature_set
            row["source_feature_index"] = int(local_index)
            row["fused_feature_index"] = int(offset + local_index)
            rows.append(row)
        offset += len(item["neuron_rows"])
    return rows


def source_digest(source_splits: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "method": item["method"],
            "feature_set": item["feature_set"],
            "feature_shape": list(item["features"].shape),
            "summary": item["summary"],
            "payload_path": str(item["payload_path"]),
            "ids_sha256": stable_sha256(ids),
        }
        for item in source_splits
    ]


def expected_params(
    args: argparse.Namespace,
    *,
    subset: str,
    split: str,
    source_methods: list[str],
    source_splits: list[dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    return {
        "stage": "tcn_01_build_probe_features",
        "stage_version": PP_STAGE_VERSION,
        "model_alias": args.model_alias,
        "subset": subset,
        "split": split,
        "probe_method": PROBE_METHOD_TOOL_CIRCUIT_NEURONS,
        "probe_method_label": method_label(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "feature_set": method_feature_set(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "feature_description": method_feature_description(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "feature_definition": method_feature_definition(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "source_methods": source_methods,
        "source_feature_sets": [method_feature_set(method) for method in source_methods],
        "source_digest": source_digest(source_splits, ids),
        "aligned_ids_sha256": stable_sha256(ids),
        "source_methods_sha256": stable_sha256(source_methods),
    }


def build_split(
    args: argparse.Namespace,
    *,
    subset: str,
    split: str,
    source_root: Path,
    features_root: Path,
    source_methods: list[str],
) -> dict[str, Any]:
    source_splits = [
        load_source_split(
            source_root=source_root,
            source_method=method,
            model_alias=args.model_alias,
            subset=subset,
            split=split,
        )
        for method in source_methods
    ]
    ids, labels, meta_rows = validate_aligned_sources(source_splits, subset=subset, split=split)
    out_dir = output_dir(features_root, args.model_alias, subset)
    params = expected_params(
        args,
        subset=subset,
        split=split,
        source_methods=source_methods,
        source_splits=source_splits,
        ids=ids,
    )
    expected = [out_dir / f"{split}_features.pt", out_dir / f"{split}_meta.jsonl", out_dir / f"{split}_summary.json"]
    if should_skip(
        out_dir / split,
        params,
        expected,
        overwrite=args.overwrite,
        clean=False,
        allowed_root=features_root,
    ):
        return read_json(out_dir / f"{split}_summary.json")

    features = torch.cat([item["features"] for item in source_splits], dim=1).contiguous()
    neuron_rows = fused_neuron_rows(source_splits)
    if len(neuron_rows) != features.shape[1]:
        raise RuntimeError("Internal error: fused neuron row count does not match feature dimension")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "neuron_rows": neuron_rows,
            "feature_shape": list(features.shape),
            "source_methods": source_methods,
            "source_feature_sets": [method_feature_set(method) for method in source_methods],
        },
        out_dir / f"{split}_features.pt",
    )
    write_jsonl(out_dir / f"{split}_meta.jsonl", meta_rows)
    source_dims = {item["method"]: int(item["features"].shape[1]) for item in source_splits}
    summary = {
        "model_alias": args.model_alias,
        "subset": subset,
        "split": split,
        "probe_method": PROBE_METHOD_TOOL_CIRCUIT_NEURONS,
        "feature_set": method_feature_set(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "feature_description": method_feature_description(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "feature_shape": list(features.shape),
        "feature_dim": int(features.shape[1]),
        "source_methods": source_methods,
        "source_feature_sets": [method_feature_set(method) for method in source_methods],
        "source_feature_dims": source_dims,
        "label_summary": summarize_labels(meta_rows),
        "aligned_ids_sha256": stable_sha256(ids),
    }
    write_json(out_dir / f"{split}_summary.json", summary)
    write_json(out_dir / split / "manifest.json", {"params": params, "summary": summary})
    print(
        f"Wrote {subset}/{split} {method_feature_set(PROBE_METHOD_TOOL_CIRCUIT_NEURONS)} "
        f"features: dim={features.shape[1]} rows={features.shape[0]}"
    )
    return summary


def main() -> None:
    args = parse_args()
    source_methods = parse_source_methods(args.source_methods)
    source_root = probe_prefill_root(args.source_root)
    root = prepare_probe_method_root(probe_prefill_root(args.output_root), PROBE_METHOD_TOOL_CIRCUIT_NEURONS)
    features_root = pp_subdir(root, "features")
    subsets = print_subset_plan(args.subset, stage="TCN-1", model_alias=args.model_alias)

    if args.clean:
        for subset in subsets:
            out_dir = output_dir(features_root, args.model_alias, subset)
            clean_path(out_dir, allowed_root=features_root)

    root_manifest: dict[str, Any] = {
        "stage": "tcn_01_build_probe_features",
        "stage_version": PP_STAGE_VERSION,
        "probe_method": PROBE_METHOD_TOOL_CIRCUIT_NEURONS,
        "feature_set": method_feature_set(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "feature_definition": method_feature_definition(PROBE_METHOD_TOOL_CIRCUIT_NEURONS),
        "model_alias": args.model_alias,
        "source_methods": source_methods,
        "source_feature_sets": [method_feature_set(method) for method in source_methods],
        "subsets": {},
    }
    for subset in progress(subsets, desc="TCN subsets", unit="subset"):
        subset_summary = {}
        for split in progress(["train", "test"], desc=f"TCN {subset}", unit="split", leave=False):
            subset_summary[split] = build_split(
                args,
                subset=subset,
                split=split,
                source_root=source_root,
                features_root=features_root,
                source_methods=source_methods,
            )
        root_manifest["subsets"][subset] = subset_summary

    manifest_path = features_root / args.model_alias / "manifest.json"
    write_json(manifest_path, root_manifest)
    print(f"Wrote ToolCircuitNeurons feature manifest: {manifest_path}")


if __name__ == "__main__":
    main()
