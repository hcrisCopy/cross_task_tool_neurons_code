from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "code" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import progress
from cttn.when2tool_bridge import load_utils


METHOD_NAME = "ResidualDecisionAnchors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDA-4: extract all-layer last-token residual hidden activations.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--activations-dir", default="../cross_task_tool_neurons_data/residual_decision_anchors/activations")
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--save-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def selected(value: str, choices: tuple[str, ...]) -> list[str]:
    return list(choices) if value == "all" else [value]


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


def expected_params(
    args: argparse.Namespace,
    *,
    model_path: Path,
    data_path: Path,
    subset: str,
    split: str,
    tool_format: str,
) -> dict[str, Any]:
    return {
        "stage": "rda_04_hidden_activation_extraction",
        "method": METHOD_NAME,
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
        "neuron_identity": "(layer, residual_state, index) over outputs.hidden_states[layer][last_token]",
    }


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    manifest_path = out_dir / "manifest.json"
    expected = [out_dir / "activations.pt", out_dir / "meta.jsonl", out_dir / "summary.json", manifest_path]
    if overwrite or not all(path.exists() for path in expected):
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing RDA activations: {out_dir}", flush=True)
        return True
    return False


@torch.inference_mode()
def extract_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    module_meta: list[dict[str, Any]],
    save_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    last_indices = encoded["attention_mask"].sum(dim=1) - 1
    first_device = next(model.parameters()).device
    encoded = {key: value.to(first_device) for key, value in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    batch_idx_by_device: dict[torch.device, torch.Tensor] = {}
    out: dict[str, torch.Tensor] = {}
    for meta, hidden in zip(module_meta, outputs.hidden_states):
        idx = last_indices.to(hidden.device)
        if hidden.device not in batch_idx_by_device:
            batch_idx_by_device[hidden.device] = torch.arange(hidden.shape[0], device=hidden.device)
        batch_idx = batch_idx_by_device[hidden.device]
        out[str(meta["key"])] = hidden[batch_idx, idx, :].detach().to(device="cpu", dtype=save_dtype)
    del encoded, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    activation_root = resolve_path(args.activations_dir)
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")

    w2t_utils = load_utils(args.when2tool_repo)
    tool_format = infer_tool_format(args.model_alias, model_path)
    system_prompt = w2t_utils.get_system_prompt(tool_format)
    print(f"Model: {args.model_alias} -> {model_path}", flush=True)
    print(f"Tool format: {tool_format}", flush=True)

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
    ).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    hidden_size = int(model.config.hidden_size)
    num_layers = int(model.config.num_hidden_layers) + 1
    module_meta = [
        {"key": f"residual_state.{layer:02d}", "layer": layer, "module": "residual_state", "dim": hidden_size}
        for layer in range(num_layers)
    ]
    print(f"Extracting {num_layers} residual-state layers; hidden_size={hidden_size}", flush=True)

    save_dtype = torch_dtype(args.save_dtype)
    for subset in selected(args.subset, ("single_hop", "multi_hop")):
        for split in selected(args.split, ("train", "test")):
            data_path = model_dataset / subset / f"{split}.jsonl"
            out_dir = output_dir(activation_root, args.model_alias, subset, split)
            params = expected_params(args, model_path=model_path, data_path=data_path, subset=subset, split=split, tool_format=tool_format)
            if should_skip(out_dir, params, args.overwrite, args.clean):
                continue
            rows = read_jsonl(data_path)
            if args.max_samples > 0:
                rows = rows[: args.max_samples]
            ensure_dir(out_dir)
            prompts = [build_prompt_text(task, tokenizer, w2t_utils, system_prompt, tool_format) for task in rows]
            accum: dict[str, list[torch.Tensor]] = {str(meta["key"]): [] for meta in module_meta}
            meta_rows: list[dict[str, Any]] = []
            for start in progress(range(0, len(prompts), args.batch_size), desc=f"RDA-4 {subset}/{split}", unit="batch"):
                batch_prompts = prompts[start : start + args.batch_size]
                batch_rows = rows[start : start + args.batch_size]
                captures = extract_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    module_meta=module_meta,
                    save_dtype=save_dtype,
                )
                for meta in module_meta:
                    accum[str(meta["key"])].append(captures[str(meta["key"])])
                for local_offset, task in enumerate(batch_rows, start=start):
                    meta_rows.append(
                        {
                            "row_index": local_offset,
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
            summary = {"count": len(meta_rows), "module_count": len(module_meta), "save_dtype": args.save_dtype, "batch_size": args.batch_size}
            write_json(out_dir / "summary.json", summary)
            write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
            print(f"Wrote RDA activations: {out_dir / 'activations.pt'}", flush=True)


if __name__ == "__main__":
    main()
