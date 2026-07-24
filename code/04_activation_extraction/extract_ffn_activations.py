from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import find_ffn_target_modules, infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import progress
from cttn.when2tool_bridge import load_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: extract last-token FFN module activations.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model forward dtype; bfloat16 matches the remote vLLM/HF setup.",
    )
    parser.add_argument(
        "--save-dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Activation storage dtype. Default float32 follows When2Tool hidden-state extraction.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-data-shards", type=int, default=1)
    parser.add_argument("--data-shard-index", type=int, default=0)
    parser.add_argument(
        "--merge-data-shards",
        action="store_true",
        help="Merge precomputed data shards into the standard Stage 4 output files without loading the model.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt_text(task: dict[str, Any], tokenizer: Any, w2t_utils: Any, system_prompt: str, tool_format: str) -> str:
    state = w2t_utils.init_state(
        task,
        system_prompt,
        record_mode="lite",
        prompt_mode="current",
        require_reasoning=False,
        tool_format=tool_format,
        tokenizer=tokenizer,
    )
    return apply_chat_template(tokenizer, state["messages"], state["tools"])


def output_dir(root: Path, model_alias: str, subset: str, split: str) -> Path:
    return root / model_alias / subset / split


def dataset_manifest_params(data_path: Path) -> dict[str, Any]:
    manifest_path = data_path.parents[1] / "manifest.json"
    if not manifest_path.exists():
        return {}
    return read_json(manifest_path).get("params", {})


def shard_output_dir(root: Path, model_alias: str, subset: str, split: str, shard_index: int) -> Path:
    return root / model_alias / "_activation_shards" / subset / split / f"shard_{shard_index:03d}"


def expected_params(
    args: argparse.Namespace,
    *,
    model_path: Path,
    data_path: Path,
    subset: str,
    split: str,
    tool_format: str,
    shard_index: int | None = None,
) -> dict[str, Any]:
    params = {
        "stage": "04_activation_extraction",
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "dataset_path": str(data_path),
        "dataset_manifest_params": dataset_manifest_params(data_path),
        "subset": subset,
        "split": split,
        "batch_size": args.batch_size,
        "torch_dtype": args.torch_dtype,
        "save_dtype": args.save_dtype,
        "device_map": args.device_map,
        "max_samples": args.max_samples,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "prompt_builder": "when2tool_init_state",
        "num_data_shards": args.num_data_shards,
    }
    if shard_index is not None:
        params["data_shard_index"] = shard_index
    return params


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    manifest_path = out_dir / "manifest.json"
    expected_files = [out_dir / "activations.pt", out_dir / "meta.jsonl", out_dir / "summary.json", manifest_path]
    if overwrite or not all(path.exists() for path in expected_files):
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing activations: {out_dir}")
        return True
    return False


def shard_bounds(n_rows: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    if num_shards < 1:
        raise ValueError("--num-data-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--data-shard-index must be in [0, {num_shards - 1}], got {shard_index}")
    start = (n_rows * shard_index) // num_shards
    end = (n_rows * (shard_index + 1)) // num_shards
    return start, end


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
    module_keys = [meta["key"] for meta in module_meta]
    for payload in payloads[1:]:
        if payload["module_meta"] != module_meta:
            raise ValueError("Activation shard module metadata do not match")

    tensors = {
        key: torch.cat([payload["activations"][key] for payload in payloads], dim=0).contiguous()
        for key in module_keys
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
            "model_path": params["model_path"],
            "subset": subset,
            "split": split,
            "tool_format": params["tool_format"],
            "module_meta": module_meta,
            "activations": tensors,
        },
        final_dir / "activations.pt",
    )
    write_jsonl(final_dir / "meta.jsonl", meta_rows)
    summary = {
        "count": len(meta_rows),
        "module_count": len(module_meta),
        "save_dtype": params["save_dtype"],
        "batch_size": params["batch_size"],
        "merged_data_shards": num_shards,
    }
    write_json(final_dir / "summary.json", summary)
    write_json(final_dir / "manifest.json", {"params": params, "summary": summary})
    print(f"Merged activation shards: {final_dir / 'activations.pt'}")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    activation_root = resolve_path(args.activations_dir) if args.activations_dir else path_from_config("activations_dir")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")

    if args.merge_data_shards:
        tool_format = infer_tool_format(args.model_alias, model_path)
        subsets = ["single_hop", "multi_hop"] if args.subset == "all" else [args.subset]
        splits = ["train", "test"] if args.split == "all" else [args.split]
        for subset in subsets:
            for split in splits:
                data_path = model_dataset / subset / f"{split}.jsonl"
                final_dir = output_dir(activation_root, args.model_alias, subset, split)
                params = expected_params(
                    args,
                    model_path=model_path,
                    data_path=data_path,
                    subset=subset,
                    split=split,
                    tool_format=tool_format,
                )
                if should_skip(final_dir, params, args.overwrite, args.clean):
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
    tool_format = infer_tool_format(args.model_alias, model_path)
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

    target_modules = find_ffn_target_modules(model)
    module_meta = [
        {"key": key, "layer": layer, "module": module_name, "dim": int(module.out_features)}
        for key, layer, module_name, module in target_modules
    ]
    print(f"Hooking {len(target_modules)} FFN modules")

    subsets = ["single_hop", "multi_hop"] if args.subset == "all" else [args.subset]
    splits = ["train", "test"] if args.split == "all" else [args.split]
    save_dtype = torch_dtype(args.save_dtype)

    for subset in subsets:
        for split in splits:
            data_path = model_dataset / subset / f"{split}.jsonl"
            out_dir = (
                shard_output_dir(activation_root, args.model_alias, subset, split, args.data_shard_index)
                if args.num_data_shards > 1
                else output_dir(activation_root, args.model_alias, subset, split)
            )
            params = expected_params(
                args,
                model_path=model_path,
                data_path=data_path,
                subset=subset,
                split=split,
                tool_format=tool_format,
                shard_index=args.data_shard_index if args.num_data_shards > 1 else None,
            )
            if should_skip(out_dir, params, args.overwrite, args.clean):
                continue
            rows = read_jsonl(data_path)
            if args.max_samples > 0:
                rows = rows[: args.max_samples]
            global_offset = 0
            if args.num_data_shards > 1:
                start, end = shard_bounds(len(rows), args.num_data_shards, args.data_shard_index)
                rows = rows[start:end]
                global_offset = start
                print(
                    f"{subset}/{split}: data shard {args.data_shard_index + 1}/{args.num_data_shards}, "
                    f"rows [{start}, {end})"
                )
            ensure_dir(out_dir)

            prompts = [build_prompt_text(task, tokenizer, w2t_utils, system_prompt, tool_format) for task in rows]
            accum: dict[str, list[torch.Tensor]] = {meta["key"]: [] for meta in module_meta}
            meta_rows: list[dict[str, Any]] = []
            handles = []
            captures: dict[str, torch.Tensor] = {}
            last_indices: torch.Tensor | None = None

            def make_hook(key: str):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    nonlocal captures, last_indices
                    out = output[0] if isinstance(output, tuple) else output
                    if last_indices is None:
                        raise RuntimeError("last_indices was not set before forward pass")
                    idx = last_indices.to(out.device)
                    batch_idx = torch.arange(out.shape[0], device=out.device)
                    captures[key] = out[batch_idx, idx, :].detach().to(device="cpu", dtype=save_dtype)

                return hook

            for key, _layer, _module_name, module in target_modules:
                handles.append(module.register_forward_hook(make_hook(key)))

            try:
                for start in progress(range(0, len(prompts), args.batch_size), desc=f"{subset}/{split}"):
                    batch_prompts = prompts[start : start + args.batch_size]
                    batch_rows = rows[start : start + args.batch_size]
                    encoded = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                    )
                    last_indices = encoded["attention_mask"].sum(dim=1) - 1
                    first_device = next(model.parameters()).device
                    encoded = {k: v.to(first_device) for k, v in encoded.items()}
                    captures = {}
                    with torch.inference_mode():
                        _ = model(**encoded, use_cache=False)
                    missing = [meta["key"] for meta in module_meta if meta["key"] not in captures]
                    if missing:
                        raise RuntimeError(f"Missing hook captures for modules: {missing[:5]}")
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
                    "activations": tensors,
                },
                out_dir / "activations.pt",
            )
            write_jsonl(out_dir / "meta.jsonl", meta_rows)
            summary = {
                "count": len(meta_rows),
                "module_count": len(module_meta),
                "save_dtype": args.save_dtype,
                "batch_size": args.batch_size,
            }
            write_json(out_dir / "summary.json", summary)
            write_json(
                out_dir / "manifest.json",
                {
                    "params": params,
                    "summary": summary,
                },
            )
            print(f"Wrote activations: {out_dir / 'activations.pt'}")

    del model


if __name__ == "__main__":
    main()
