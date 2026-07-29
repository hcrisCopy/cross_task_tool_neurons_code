from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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


STAGE_VERSION = 1
METHOD_NAME = "ToolRoutingNeurons"
DEFAULT_ACTIVATIONS_DIR = "../cross_task_tool_neurons_data/tool_routing_neurons/activations"
ATTN_MODULES = ("attn_q", "attn_k", "attn_v", "attn_o_in")


@dataclass(frozen=True)
class AttentionTarget:
    key: str
    layer: int
    module: str
    projection_name: str
    hook_kind: str
    module_obj: Any
    dim: int
    weight_norm: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRN-4: extract last-input-token attention routing neuron activations "
            "from Q/K/V projection outputs and O-projection input."
        )
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
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. With --gpus 0 this is restricted to one visible GPU.",
    )
    parser.add_argument("--gpus", default=None, help="Single GPU id to expose, for example 0. Multiple GPUs are rejected.")
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


def set_visible_gpu(gpus: str | None) -> None:
    if not gpus:
        return
    parsed = [item.strip() for item in gpus.split(",") if item.strip()]
    if len(parsed) != 1:
        raise ValueError("TRN-4 is a single-process extractor. Pass exactly one GPU id, e.g. --gpus 0.")
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = parsed[0]
    print(f"TRN-4 CUDA_VISIBLE_DEVICES={parsed[0]}", flush=True)


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


def subset_values(value: str) -> list[str]:
    return ["single_hop", "multi_hop"] if value == "all" else [value]


def split_values(value: str) -> list[str]:
    return ["train", "test"] if value == "all" else [value]


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
    module_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": "trn_04_attention_routing_activation_extraction",
        "stage_version": STAGE_VERSION,
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
        "gpus": args.gpus,
        "max_samples": args.max_samples,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "prompt_builder": "when2tool_init_state",
        "activation_definition": "last_input_token_attention_routing_coordinates",
        "neuron_definition": "rows of q/k/v projection outputs and columns of o projection input",
        "modules": [
            {
                "layer": meta["layer"],
                "module": meta["module"],
                "dim": meta["dim"],
                "projection_name": meta["projection_name"],
            }
            for meta in module_meta
        ],
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
        print(f"Skip existing TRN activations: {out_dir}", flush=True)
        return True
    return False


def parse_layer_index(name: str) -> int | None:
    patterns = [
        r"(?:^|\.)layers\.(\d+)\.",
        r"(?:^|\.)h\.(\d+)\.",
        r"(?:^|\.)blocks\.(\d+)\.",
        r"(?:^|\.)decoder\.layers\.(\d+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def projection_weight_norm(module: Any, module_name: str) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(f"Projection has no weight tensor: {module_name}")
    matrix = weight.detach().float().cpu()
    if module_name == "attn_o_in":
        return torch.linalg.vector_norm(matrix, ord=2, dim=0)
    return torch.linalg.vector_norm(matrix, ord=2, dim=1)


def discover_attention_targets(model: Any) -> list[AttentionTarget]:
    suffix_to_module = {
        "q_proj": "attn_q",
        "k_proj": "attn_k",
        "v_proj": "attn_v",
        "o_proj": "attn_o_in",
    }
    targets: list[AttentionTarget] = []
    for name, module in model.named_modules():
        short = name.rsplit(".", 1)[-1]
        if short not in suffix_to_module:
            continue
        layer = parse_layer_index(name)
        if layer is None:
            continue
        module_name = suffix_to_module[short]
        if short == "o_proj":
            dim = int(getattr(module, "in_features"))
            hook_kind = "pre"
        else:
            dim = int(getattr(module, "out_features"))
            hook_kind = "post"
        key = f"layer_{layer:03d}.{module_name}"
        targets.append(
            AttentionTarget(
                key=key,
                layer=layer,
                module=module_name,
                projection_name=name,
                hook_kind=hook_kind,
                module_obj=module,
                dim=dim,
                weight_norm=projection_weight_norm(module, module_name),
            )
        )
    module_rank = {name: index for index, name in enumerate(ATTN_MODULES)}
    targets.sort(key=lambda item: (item.layer, module_rank[item.module]))
    seen = set()
    deduped: list[AttentionTarget] = []
    for target in targets:
        if target.key in seen:
            raise ValueError(f"Duplicate attention target key: {target.key}")
        seen.add(target.key)
        deduped.append(target)
    if not deduped:
        raise ValueError("No attention projection targets found. Expected q_proj/k_proj/v_proj/o_proj modules.")
    return deduped


def module_meta_from_targets(targets: list[AttentionTarget]) -> list[dict[str, Any]]:
    return [
        {
            "key": target.key,
            "layer": target.layer,
            "module": target.module,
            "dim": target.dim,
            "projection_name": target.projection_name,
            "hook_kind": target.hook_kind,
            "activation_aggregation": "last_input_token",
            "neuron_definition": (
                "q/k/v projection output row coordinate"
                if target.module != "attn_o_in"
                else "o projection input coordinate, corresponding to an o_proj column"
            ),
        }
        for target in targets
    ]


def target_norms(targets: list[AttentionTarget]) -> dict[str, torch.Tensor]:
    return {target.key: target.weight_norm.contiguous() for target in targets}


def main() -> None:
    args = parse_args()
    set_visible_gpu(args.gpus)
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    activation_root = resolve_path(args.activations_dir) if args.activations_dir else resolve_path(DEFAULT_ACTIVATIONS_DIR)
    model_dataset = dataset_root / args.model_alias
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")

    w2t_utils = load_utils(args.when2tool_repo)
    tool_format = infer_tool_format(args.model_alias, model_path)
    system_prompt = w2t_utils.get_system_prompt(tool_format)
    print(f"TRN-4 model: {args.model_alias} -> {model_path}", flush=True)
    print(f"TRN-4 tool format: {tool_format}", flush=True)

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

    targets = discover_attention_targets(model)
    module_meta = module_meta_from_targets(targets)
    norms = target_norms(targets)
    print(f"TRN-4 hooking {len(targets)} attention routing modules", flush=True)
    save_dtype = torch_dtype(args.save_dtype)

    for subset in subset_values(args.subset):
        for split in split_values(args.split):
            data_path = model_dataset / subset / f"{split}.jsonl"
            out_dir = output_dir(activation_root, args.model_alias, subset, split)
            params = expected_params(
                args,
                model_path=model_path,
                data_path=data_path,
                subset=subset,
                split=split,
                tool_format=tool_format,
                module_meta=module_meta,
            )
            if should_skip(out_dir, params, args.overwrite, args.clean):
                continue
            rows = read_jsonl(data_path)
            if args.max_samples > 0:
                rows = rows[: args.max_samples]
            ensure_dir(out_dir)
            prompts = [build_prompt_text(task, tokenizer, w2t_utils, system_prompt, tool_format) for task in rows]
            accum: dict[str, list[torch.Tensor]] = {target.key: [] for target in targets}
            meta_rows: list[dict[str, Any]] = []
            captures: dict[str, torch.Tensor] = {}
            last_indices: torch.Tensor | None = None
            handles = []

            def capture_last_token(key: str, tensor: torch.Tensor) -> None:
                nonlocal captures, last_indices
                if last_indices is None:
                    raise RuntimeError("last_indices was not set before forward pass")
                idx = last_indices.to(tensor.device)
                batch_idx = torch.arange(tensor.shape[0], device=tensor.device)
                captures[key] = tensor[batch_idx, idx, :].detach().to(device="cpu", dtype=save_dtype)

            def make_post_hook(key: str):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    out = output[0] if isinstance(output, tuple) else output
                    capture_last_token(key, out)

                return hook

            def make_pre_hook(key: str):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    capture_last_token(key, inputs[0])

                return hook

            for target in targets:
                if target.hook_kind == "pre":
                    handles.append(target.module_obj.register_forward_pre_hook(make_pre_hook(target.key)))
                else:
                    handles.append(target.module_obj.register_forward_hook(make_post_hook(target.key)))

            try:
                for start in progress(range(0, len(prompts), args.batch_size), desc=f"TRN-4 {subset}/{split}", unit="batch"):
                    batch_prompts = prompts[start : start + args.batch_size]
                    batch_rows = rows[start : start + args.batch_size]
                    encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)
                    last_indices = encoded["attention_mask"].sum(dim=1) - 1
                    first_device = next(model.parameters()).device
                    encoded = {key: value.to(first_device) for key, value in encoded.items()}
                    captures = {}
                    with torch.inference_mode():
                        _ = model(**encoded, use_cache=False)
                    missing = [target.key for target in targets if target.key not in captures]
                    if missing:
                        raise RuntimeError(f"Missing TRN hook captures: {missing[:5]}")
                    for target in targets:
                        accum[target.key].append(captures[target.key])
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
                    "projection_weight_norms": norms,
                    "activations": tensors,
                    "activation_definition": "last_input_token_attention_routing_coordinates",
                },
                out_dir / "activations.pt",
            )
            write_jsonl(out_dir / "meta.jsonl", meta_rows)
            summary = {
                "count": len(meta_rows),
                "module_count": len(module_meta),
                "save_dtype": args.save_dtype,
                "batch_size": args.batch_size,
                "activation_definition": "last_input_token_attention_routing_coordinates",
            }
            write_json(out_dir / "summary.json", summary)
            write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
            print(f"Wrote TRN activations: {out_dir / 'activations.pt'}", flush=True)

    del model


if __name__ == "__main__":
    main()
