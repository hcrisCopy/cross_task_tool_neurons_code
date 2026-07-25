from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ps_common import (
    STAGE_VERSION,
    build_current_prompt,
    dataset_manifest,
    discover_ffn_intermediate_layers,
    down_weight_norms,
    infer_tool_format,
    load_utils,
    module_meta_from_layers,
    ps_resolve_root,
    read_json,
    read_jsonl,
    resolve_model_path,
    resolve_path,
    select_records,
    should_skip,
    split_values,
    stable_sha256,
    subset_values,
    torch_dtype,
    write_json,
    write_jsonl,
)
from cttn.paths import ensure_dir, path_from_config
from cttn.progress import progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PreciseShield stage 4: extract last-token FFN intermediate activations h."
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--save-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="balanced")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--num-data-shards", type=int, default=1)
    parser.add_argument("--data-shard-index", type=int, default=0)
    parser.add_argument("--merge-data-shards", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_dir(root: Path, model_alias: str, subset: str, split: str) -> Path:
    return root / model_alias / subset / split


def shard_output_dir(root: Path, model_alias: str, subset: str, split: str, shard_index: int) -> Path:
    return root / model_alias / "_activation_shards" / subset / split / f"shard_{shard_index:03d}"


def shard_bounds(n_rows: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    if num_shards < 1:
        raise ValueError("--num-data-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--data-shard-index must be in [0, {num_shards - 1}], got {shard_index}")
    start = (n_rows * shard_index) // num_shards
    end = (n_rows * (shard_index + 1)) // num_shards
    return start, end


def expected_params(
    args: argparse.Namespace,
    *,
    model_path: Path,
    model_dataset: Path,
    data_path: Path,
    rows: list[dict[str, Any]],
    subset: str,
    split: str,
    tool_format: str,
    shard_index: int | None = None,
) -> dict[str, Any]:
    params = {
        "stage": "ps_04_intermediate_activation_extraction",
        "stage_version": STAGE_VERSION,
        "method": "PreciseShield",
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "dataset_path": str(data_path),
        "dataset_manifest_params": dataset_manifest(model_dataset).get("params", {}),
        "selected_rows": {"count": len(rows), "sha256": stable_sha256([row.get("id") for row in rows])},
        "subset": subset,
        "split": split,
        "batch_size": args.batch_size,
        "torch_dtype": args.torch_dtype,
        "save_dtype": args.save_dtype,
        "device_map": args.device_map,
        "max_samples": args.max_samples,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "prompt_builder": "when2tool_init_state",
        "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
        "num_data_shards": args.num_data_shards,
    }
    if shard_index is not None:
        params["data_shard_index"] = shard_index
    return params


def merge_shards(
    *,
    final_dir: Path,
    activation_root: Path,
    model_alias: str,
    subset: str,
    split: str,
    num_shards: int,
    params: dict[str, Any],
) -> None:
    shard_dirs = [shard_output_dir(activation_root, model_alias, subset, split, idx) for idx in range(num_shards)]
    missing = [path for path in shard_dirs if not (path / "activations.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing activation shard outputs: {missing[:3]}")

    payloads = [torch.load(path / "activations.pt", map_location="cpu") for path in shard_dirs]
    module_meta = payloads[0]["module_meta"]
    down_norms = payloads[0]["down_weight_norms"]
    for payload in payloads[1:]:
        if payload["module_meta"] != module_meta:
            raise ValueError("Activation shard module metadata do not match")
        if payload["down_weight_norms"].keys() != down_norms.keys():
            raise ValueError("Activation shard down_weight_norm keys do not match")

    tensors = {
        meta["key"]: torch.cat([payload["activations"][meta["key"]] for payload in payloads], dim=0).contiguous()
        for meta in module_meta
    }
    meta_rows: list[dict[str, Any]] = []
    for path in shard_dirs:
        meta_rows.extend(read_jsonl(path / "meta.jsonl"))
    meta_rows.sort(key=lambda row: int(row["row_index"]))
    for expected_index, row in enumerate(meta_rows):
        if int(row["row_index"]) != expected_index:
            raise ValueError(f"Shard merge row order gap at {expected_index}: got {row['row_index']}")
    for row in meta_rows:
        row.pop("row_index", None)

    ensure_dir(final_dir)
    torch.save(
        {
            "model_alias": model_alias,
            "subset": subset,
            "split": split,
            "module_meta": module_meta,
            "down_weight_norms": down_norms,
            "activations": tensors,
            "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
        },
        final_dir / "activations.pt",
    )
    write_jsonl(final_dir / "meta.jsonl", meta_rows)
    summary = {
        "count": len(meta_rows),
        "layer_count": len(module_meta),
        "save_dtype": params["save_dtype"],
        "merged_data_shards": num_shards,
    }
    write_json(final_dir / "summary.json", summary)
    write_json(final_dir / "manifest.json", {"params": params, "summary": summary})
    print(f"Merged PreciseShield activation shards: {final_dir / 'activations.pt'}")


def load_selected_rows(data_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(data_path)
    return select_records(
        rows,
        args.max_samples,
        args.seed,
        strategy=args.sample_strategy,
        require_per_type_labels=True,
    )


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    activation_root = ps_resolve_root(args.activations_dir, "activations")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")

    tool_format = infer_tool_format(args.model_alias, model_path)
    subsets = subset_values(args.subset)
    splits = split_values(args.split)

    if args.merge_data_shards:
        for subset in subsets:
            for split in splits:
                data_path = model_dataset / subset / f"{split}.jsonl"
                rows = load_selected_rows(data_path, args)
                params = expected_params(
                    args,
                    model_path=model_path,
                    model_dataset=model_dataset,
                    data_path=data_path,
                    rows=rows,
                    subset=subset,
                    split=split,
                    tool_format=tool_format,
                )
                final_dir = output_dir(activation_root, args.model_alias, subset, split)
                if should_skip(
                    final_dir,
                    params,
                    [final_dir / "activations.pt", final_dir / "meta.jsonl", final_dir / "summary.json"],
                    overwrite=args.overwrite,
                    clean=args.clean,
                ):
                    continue
                merge_shards(
                    final_dir=final_dir,
                    activation_root=activation_root,
                    model_alias=args.model_alias,
                    subset=subset,
                    split=split,
                    num_shards=args.num_data_shards,
                    params=params,
                )
        return

    w2t_utils = load_utils(args.when2tool_repo)
    system_prompt = w2t_utils.get_system_prompt(tool_format)
    print(f"Model: {args.model_alias} -> {model_path}")
    print(f"Tool format: {tool_format}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype(args.torch_dtype),
        device_map=args.device_map,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers = discover_ffn_intermediate_layers(model)
    module_meta = module_meta_from_layers(layers)
    down_norms = down_weight_norms(layers)
    print(f"Hooking {len(layers)} FFN intermediate tensors")
    save_dtype = torch_dtype(args.save_dtype)

    for subset in subsets:
        for split in splits:
            data_path = model_dataset / subset / f"{split}.jsonl"
            rows = load_selected_rows(data_path, args)
            global_offset = 0
            out_dir = (
                shard_output_dir(activation_root, args.model_alias, subset, split, args.data_shard_index)
                if args.num_data_shards > 1
                else output_dir(activation_root, args.model_alias, subset, split)
            )
            if args.num_data_shards > 1:
                start, end = shard_bounds(len(rows), args.num_data_shards, args.data_shard_index)
                rows = rows[start:end]
                global_offset = start
                print(f"{subset}/{split}: data shard {args.data_shard_index + 1}/{args.num_data_shards}, rows [{start}, {end})")

            params = expected_params(
                args,
                model_path=model_path,
                model_dataset=model_dataset,
                data_path=data_path,
                rows=rows,
                subset=subset,
                split=split,
                tool_format=tool_format,
                shard_index=args.data_shard_index if args.num_data_shards > 1 else None,
            )
            if should_skip(
                out_dir,
                params,
                [out_dir / "activations.pt", out_dir / "meta.jsonl", out_dir / "summary.json"],
                overwrite=args.overwrite,
                clean=args.clean,
            ):
                continue
            ensure_dir(out_dir)

            prompts = [
                build_current_prompt(
                    task,
                    tokenizer=tokenizer,
                    w2t_utils=w2t_utils,
                    system_prompt=system_prompt,
                    tool_format=tool_format,
                )
                for task in rows
            ]
            accum: dict[str, list[torch.Tensor]] = {meta["key"]: [] for meta in module_meta}
            meta_rows: list[dict[str, Any]] = []
            captures: dict[str, torch.Tensor] = {}
            last_indices: torch.Tensor | None = None
            handles = []

            def make_hook(key: str):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    nonlocal captures, last_indices
                    if last_indices is None:
                        raise RuntimeError("last_indices was not set before forward pass")
                    hidden = inputs[0]
                    idx = last_indices.to(hidden.device)
                    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                    captures[key] = hidden[batch_idx, idx, :].detach().to(device="cpu", dtype=save_dtype)

                return hook

            for layer in layers:
                handles.append(layer.down.register_forward_pre_hook(make_hook(layer.key)))

            try:
                for start in progress(range(0, len(prompts), args.batch_size), desc=f"{subset}/{split}"):
                    batch_prompts = prompts[start : start + args.batch_size]
                    batch_rows = rows[start : start + args.batch_size]
                    encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)
                    last_indices = encoded["attention_mask"].sum(dim=1) - 1
                    first_device = next(model.parameters()).device
                    encoded = {key: value.to(first_device) for key, value in encoded.items()}
                    captures = {}
                    with torch.inference_mode():
                        _ = model(**encoded, use_cache=False)
                    missing = [meta["key"] for meta in module_meta if meta["key"] not in captures]
                    if missing:
                        raise RuntimeError(f"Missing hook captures for layers: {missing[:5]}")
                    for meta in module_meta:
                        accum[meta["key"]].append(captures[meta["key"]])
                    for local_offset, task in enumerate(batch_rows, start=start):
                        meta_rows.append(
                            {
                                "row_index": global_offset + local_offset,
                                "id": str(task["id"]),
                                "subset": subset,
                                "split": split,
                                "env_name": task["env_name"],
                                "task_type": task["task_type"],
                                "difficulty": task.get("difficulty", "unknown"),
                                "tool_necessary": int(task["tool_necessary"]),
                                "no_tool_correct": int(task["no_tool_correct"]),
                            }
                        )
                    del encoded
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                for handle in handles:
                    handle.remove()

            tensors = {key: torch.cat(parts, dim=0).contiguous() for key, parts in accum.items()}
            torch.save(
                {
                    "model_alias": args.model_alias,
                    "model_path": str(model_path),
                    "subset": subset,
                    "split": split,
                    "tool_format": tool_format,
                    "module_meta": module_meta,
                    "down_weight_norms": down_norms,
                    "activations": tensors,
                    "activation_definition": "last_input_token_ffn_intermediate_h_before_down_proj",
                },
                out_dir / "activations.pt",
            )
            write_jsonl(out_dir / "meta.jsonl", meta_rows)
            summary = {
                "count": len(meta_rows),
                "layer_count": len(module_meta),
                "save_dtype": args.save_dtype,
                "batch_size": args.batch_size,
                "selected_row_ids_sha256": stable_sha256([row["id"] for row in rows]),
            }
            write_json(out_dir / "summary.json", summary)
            write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
            print(f"Wrote PreciseShield activations: {out_dir / 'activations.pt'}")

    del model


if __name__ == "__main__":
    main()

