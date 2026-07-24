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

from cttn.io import read_jsonl, write_json, write_jsonl
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
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--save-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
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


def build_prompt_text(task: dict[str, Any], tokenizer: Any, w2t_utils: Any, system_prompt: str) -> str:
    user_content = w2t_utils.build_user_message(task["instruction"], "current", require_reasoning=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    tools_schema = w2t_utils.build_tools_schema(task)
    return apply_chat_template(tokenizer, messages, tools_schema)


def output_dir(root: Path, model_alias: str, subset: str, split: str) -> Path:
    return root / model_alias / subset / split


def should_skip(out_dir: Path, overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    if (out_dir / "activations.pt").exists() and (out_dir / "meta.jsonl").exists() and not overwrite:
        print(f"Skip existing activations: {out_dir}")
        return True
    return False


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    activation_root = resolve_path(args.activations_dir) if args.activations_dir else path_from_config("activations_dir")
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")

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
            out_dir = output_dir(activation_root, args.model_alias, subset, split)
            if should_skip(out_dir, args.overwrite, args.clean):
                continue
            rows = read_jsonl(data_path)
            if args.max_samples > 0:
                rows = rows[: args.max_samples]
            ensure_dir(out_dir)

            prompts = [build_prompt_text(task, tokenizer, w2t_utils, system_prompt) for task in rows]
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
                    for task in batch_rows:
                        meta_rows.append(
                            {
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
                    "stage": "04_activation_extraction",
                    "model_alias": args.model_alias,
                    "subset": subset,
                    "split": split,
                    "params": vars(args),
                    "summary": summary,
                },
            )
            print(f"Wrote activations: {out_dir / 'activations.pt'}")

    del model


if __name__ == "__main__":
    main()
