from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import torch
from torch.nn import functional as F

from cttn.agent import HFGenerationAgent, apply_chat_template
from cttn.data import SUBSETS
from cttn.io import read_json, read_jsonl, write_json, write_jsonl
from cttn.lora import (
    apply_masked_lora,
    mark_only_lora_trainable,
    module_masks_for_config,
    save_masked_lora_adapter,
    trainable_lora_parameters,
)
from cttn.modeling import infer_tool_format, resolve_model_path
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path
from cttn.progress import progress
from cttn.when2tool_bridge import load_model_module, load_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7: train CTD-Masked LoRA adapters.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--when2tool-repo", default=None)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--trajectory-attempts", type=int, default=2)
    parser.add_argument("--trajectory-batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--record-mode", choices=["full", "lite"], default="full")
    return parser.parse_args()


def subset_output_dir(checkpoints_root: Path, model_alias: str, subset: str) -> Path:
    return checkpoints_root / model_alias / "ctd_masked_lora" / subset


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_trajectory_params(
    args: argparse.Namespace,
    subset: str,
    *,
    model_path: Path,
    rows: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
    system_prompt: str,
    tool_format: str,
) -> dict[str, Any]:
    return {
        "stage": "07_training_trajectories",
        "trajectory_builder_version": 1,
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "max_train_samples": args.max_train_samples,
        "trajectory_attempts": args.trajectory_attempts,
        "trajectory_batch_size": args.trajectory_batch_size,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "record_mode": args.record_mode,
        "direct_record_mode": "full",
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "system_prompt_sha256": stable_sha256(system_prompt),
        "train_rows": {
            "count": len(rows),
            "sha256": stable_sha256(rows),
        },
        "dataset_manifest_params": dataset_manifest.get("params", {}),
    }


def expected_params(
    args: argparse.Namespace,
    subset: str,
    *,
    model_path: Path,
    dataset_manifest: dict[str, Any],
    shared_manifest: dict[str, Any],
    tool_format: str,
    trajectory_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "07_training",
        "model_alias": args.model_alias,
        "model_path": str(model_path),
        "subset": subset,
        "max_train_samples": args.max_train_samples,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "max_seq_length": args.max_seq_length,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "gradient_checkpointing": args.gradient_checkpointing,
        "trajectory_attempts": args.trajectory_attempts,
        "trajectory_batch_size": args.trajectory_batch_size,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "record_mode": args.record_mode,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "enable_thinking": False,
        "tool_format": tool_format,
        "trajectory_params": trajectory_params,
        "dataset_manifest_params": dataset_manifest.get("params", {}),
        "shared_neuron_manifest_params": shared_manifest.get("params", {}),
    }


def should_skip(out_dir: Path, params: dict[str, Any], overwrite: bool, clean: bool) -> bool:
    if clean:
        clean_directory(out_dir, data_root())
        return False
    manifest_path = out_dir / "manifest.json"
    adapter_path = out_dir / "adapter" / "adapter_model.pt"
    if overwrite or not manifest_path.exists() or not adapter_path.exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("params") == params:
        print(f"Skip existing CTD-Masked LoRA adapter: {out_dir}")
        return True
    return False


def direct_answer_text(task: dict[str, Any]) -> str:
    meta = task.get("label_metadata", {}) or {}
    raw = str(meta.get("model_final_raw", "") or "").strip()
    if "\\boxed{" in raw:
        return raw
    boxed = str(meta.get("model_final_boxed", "") or meta.get("model_final_cleaned", "") or "").strip()
    if boxed:
        return f"\\boxed{{{boxed}}}"
    return f"\\boxed{{{(task.get('expected', {}) or {}).get('answer', '')}}}"


def no_failed_tool_execution(item: dict[str, Any]) -> bool:
    for trace in item.get("trace", []) or []:
        tool_result = trace.get("tool_result", None)
        if isinstance(tool_result, dict) and tool_result.get("success") is False:
            return False
    return True


def make_direct_example(
    task: dict[str, Any],
    *,
    w2t_utils: Any,
    system_prompt: str,
    tool_format: str,
    tokenizer: Any,
) -> dict[str, Any]:
    state = w2t_utils.init_state(
        task,
        system_prompt,
        record_mode="full",
        prompt_mode="current",
        require_reasoning=False,
        tool_format=tool_format,
        tokenizer=tokenizer,
    )
    messages = list(state["messages"])
    messages.append({"role": "assistant", "content": direct_answer_text(task)})
    return {
        "id": str(task["id"]),
        "subset": task["subset"],
        "task_type": task["task_type"],
        "env_name": task["env_name"],
        "difficulty": task.get("difficulty", "unknown"),
        "tool_necessary": int(task["tool_necessary"]),
        "trajectory_source": "hard_no_tool_direct_answer",
        "messages": messages,
        "tools": state["tools"],
    }


def make_tool_example(
    task: dict[str, Any],
    item: dict[str, Any],
    *,
    w2t_utils: Any,
    system_prompt: str,
    tool_format: str,
    tokenizer: Any,
) -> dict[str, Any]:
    state = w2t_utils.init_state(
        task,
        system_prompt,
        record_mode="full",
        prompt_mode="current",
        require_reasoning=False,
        tool_format=tool_format,
        tokenizer=tokenizer,
    )
    return {
        "id": str(task["id"]),
        "subset": task["subset"],
        "task_type": task["task_type"],
        "env_name": task["env_name"],
        "difficulty": task.get("difficulty", "unknown"),
        "tool_necessary": int(task["tool_necessary"]),
        "trajectory_source": "current_no_reasoning_tool_trajectory",
        "messages": item.get("output", []),
        "tools": state["tools"],
        "tool_calls": int(w2t_utils.item_tool_calls(item)),
    }


def trajectory_success(item: dict[str, Any], w2t_utils: Any) -> bool:
    _raw, _boxed, _cleaned, correct = w2t_utils.item_final_eval(item)
    return bool(correct) and int(w2t_utils.item_tool_calls(item)) > 0 and no_failed_tool_execution(item)


def build_or_load_training_examples(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    agent: HFGenerationAgent,
    w2t_utils: Any,
    system_prompt: str,
    tool_format: str,
    trajectory_params: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples_path = out_dir / "training_examples.jsonl"
    skipped_path = out_dir / "skipped_examples.jsonl"
    manifest_path = out_dir / "trajectory_manifest.json"
    if examples_path.exists() and skipped_path.exists() and manifest_path.exists() and not args.overwrite:
        manifest = read_json(manifest_path)
        if manifest.get("params") == trajectory_params:
            print(f"Reuse existing training trajectories: {examples_path}")
            return read_jsonl(examples_path), read_jsonl(skipped_path)
        print(f"Rebuild training trajectories because manifest changed: {manifest_path}")
    elif examples_path.exists() and skipped_path.exists() and not args.overwrite:
        print(f"Rebuild training trajectories because manifest is missing: {manifest_path}")

    examples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    y1_tasks = []
    for task in rows:
        if int(task["tool_necessary"]) == 0:
            examples.append(
                make_direct_example(
                    task,
                    w2t_utils=w2t_utils,
                    system_prompt=system_prompt,
                    tool_format=tool_format,
                    tokenizer=agent.tokenizer,
                )
            )
        else:
            y1_tasks.append(task)

    pending = {str(task["id"]): task for task in y1_tasks}
    for attempt in range(1, max(1, args.trajectory_attempts) + 1):
        if not pending:
            break
        batch = list(pending.values())
        print(f"Generating tool trajectories: attempt {attempt}, pending={len(batch)}")
        outputs = w2t_utils.evaluate_batched(
            batch,
            agent,
            max_rounds=args.max_rounds,
            record_mode=args.record_mode,
            prompt_mode="current",
            require_reasoning=False,
            tool_format=tool_format,
        )
        for item in outputs:
            task_id = str(item["id"])
            task = pending.get(task_id)
            if task is None:
                continue
            if trajectory_success(item, w2t_utils) and item.get("output"):
                examples.append(
                    make_tool_example(
                        task,
                        item,
                        w2t_utils=w2t_utils,
                        system_prompt=system_prompt,
                        tool_format=tool_format,
                        tokenizer=agent.tokenizer,
                    )
                )
                pending.pop(task_id, None)
            elif attempt == args.trajectory_attempts:
                _raw, boxed, cleaned, correct = w2t_utils.item_final_eval(item)
                skipped.append(
                    {
                        "id": task_id,
                        "reason": "tool_trajectory_failed",
                        "tool_calls": int(w2t_utils.item_tool_calls(item)),
                        "final_correct": int(bool(correct)),
                        "model_boxed_content": boxed,
                        "model_answer": cleaned,
                    }
                )
    examples.sort(key=lambda row: str(row["id"]))
    skipped.sort(key=lambda row: str(row["id"]))
    write_jsonl(examples_path, examples)
    write_jsonl(skipped_path, skipped)
    trajectory_summary = {
        "total_train_rows": len(rows),
        "training_examples": len(examples),
        "skipped_examples": len(skipped),
        "by_tool_necessary": {
            "0": sum(1 for row in examples if int(row["tool_necessary"]) == 0),
            "1": sum(1 for row in examples if int(row["tool_necessary"]) == 1),
        },
    }
    write_json(out_dir / "trajectory_summary.json", trajectory_summary)
    write_json(out_dir / "trajectory_manifest.json", {"params": trajectory_params, "summary": trajectory_summary})
    return examples, skipped


def find_subsequence(haystack: list[int], needle: list[int], start: int) -> tuple[int, int] | None:
    if not needle:
        return None
    limit = len(haystack) - len(needle) + 1
    for pos in range(max(0, start), max(0, limit)):
        if haystack[pos : pos + len(needle)] == needle:
            return pos, pos + len(needle)
    return None


def supports_assistant_tokens_mask(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        text = "\n".join(str(value) for value in template.values())
    else:
        text = str(template or "")
    return "{% generation" in text


def tokenize_with_assistant_mask(tokenizer: Any, example: dict[str, Any]) -> tuple[list[int], list[int]]:
    messages = example["messages"]
    tools = example.get("tools", [])
    if supports_assistant_tokens_mask(tokenizer):
        try:
            encoded = apply_chat_template(
                tokenizer,
                messages,
                tools,
                enable_thinking=False,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            input_ids = list(encoded["input_ids"])
            mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
            if mask is not None and sum(mask) > 0:
                labels = [token if int(is_assistant) else -100 for token, is_assistant in zip(input_ids, mask)]
                return input_ids, labels
        except Exception:
            pass

    text = apply_chat_template(
        tokenizer,
        messages,
        tools,
        enable_thinking=False,
        add_generation_prompt=False,
        tokenize=False,
    )
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(input_ids)
    cursor = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content_ids = tokenizer(str(message.get("content", "")), add_special_tokens=False)["input_ids"]
        match = find_subsequence(input_ids, content_ids, cursor)
        if match is None:
            continue
        start, end = match
        for idx in range(start, end):
            labels[idx] = input_ids[idx]
        cursor = end
    return input_ids, labels


def build_features(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    *,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = []
    skipped = []
    for example in examples:
        input_ids, labels = tokenize_with_assistant_mask(tokenizer, example)
        supervised = sum(1 for label in labels if label != -100)
        if len(input_ids) > max_seq_length:
            skipped.append({"id": example["id"], "reason": "sequence_too_long", "length": len(input_ids)})
            continue
        if supervised == 0:
            skipped.append({"id": example["id"], "reason": "no_assistant_tokens", "length": len(input_ids)})
            continue
        features.append(
            {
                "id": example["id"],
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
                "assistant_tokens": supervised,
            }
        )
    return features, skipped


def collate(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in features)
    input_ids = []
    attention_mask = []
    labels = []
    for item in features:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def batch_loss(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)
    valid = (labels != -100).float()
    per_sample = token_loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    active = valid.sum(dim=1) > 0
    return per_sample[active].mean()


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remain = max(total_steps - step, 0)
        denom = max(total_steps - warmup_steps, 1)
        return remain / denom

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_subset(
    subset: str,
    *,
    args: argparse.Namespace,
    model_path: Path,
    model_dataset: Path,
    shared_root: Path,
    out_dir: Path,
    w2t_utils: Any,
    w2t_model: Any,
) -> None:
    tool_format = infer_tool_format(args.model_alias, model_path)
    shared_manifest_path = shared_root / subset / "manifest.json"
    if not shared_manifest_path.exists():
        raise FileNotFoundError(f"Missing shared neuron manifest for {subset}: {shared_manifest_path}")
    dataset_manifest = read_json(model_dataset / "manifest.json") if (model_dataset / "manifest.json").exists() else {}
    rows = read_jsonl(model_dataset / subset / "train.jsonl")
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]
    system_prompt = w2t_utils.get_system_prompt(tool_format)
    trajectory_params = expected_trajectory_params(
        args,
        subset,
        model_path=model_path,
        rows=rows,
        dataset_manifest=dataset_manifest,
        system_prompt=system_prompt,
        tool_format=tool_format,
    )
    params = expected_params(
        args,
        subset,
        model_path=model_path,
        dataset_manifest=dataset_manifest,
        shared_manifest=read_json(shared_manifest_path),
        tool_format=tool_format,
        trajectory_params=trajectory_params,
    )
    if should_skip(out_dir, params, args.overwrite, args.clean):
        return
    ensure_dir(out_dir)

    ctd_path = shared_root / subset / "CTD_neurons.jsonl"
    ctd_rows = read_jsonl(ctd_path)
    if not ctd_rows:
        raise ValueError(f"No CTD neurons found for {subset}: {ctd_path}")

    normalizer = w2t_model._normalize_generation_output

    agent = HFGenerationAgent(
        model_path=model_path,
        system_prompt=system_prompt,
        normalizer=normalizer,
        torch_dtype_name=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        batch_size=args.trajectory_batch_size,
        enable_thinking=False,
    )
    try:
        examples, skipped_trajectory = build_or_load_training_examples(
            out_dir,
            rows,
            agent=agent,
            w2t_utils=w2t_utils,
            system_prompt=system_prompt,
            tool_format=tool_format,
            trajectory_params=trajectory_params,
            args=args,
        )
        features, skipped_features = build_features(
            agent.tokenizer,
            examples,
            max_seq_length=args.max_seq_length,
        )
        if skipped_features:
            write_jsonl(out_dir / "skipped_tokenization_examples.jsonl", skipped_features)
        if not features:
            raise ValueError(f"No trainable features left for {subset}")

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        agent.model.train()
        if hasattr(agent.model.config, "use_cache"):
            agent.model.config.use_cache = False
        if args.gradient_checkpointing and hasattr(agent.model, "gradient_checkpointing_enable"):
            agent.model.gradient_checkpointing_enable()
            if hasattr(agent.model, "enable_input_require_grads"):
                agent.model.enable_input_require_grads()

        mask_summary = apply_masked_lora(
            agent.model,
            ctd_rows,
            rank=args.rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        mark_only_lora_trainable(agent.model)
        trainable = sum(param.numel() for param in trainable_lora_parameters(agent.model))
        mask_summary["trainable_lora_parameters"] = int(trainable)
        mask_summary["ctd_neuron_count"] = len(ctd_rows)
        write_json(out_dir / "mask_summary.json", mask_summary)

        optimizer = torch.optim.AdamW(list(trainable_lora_parameters(agent.model)), lr=args.learning_rate)
        steps_per_epoch = math.ceil(len(features) / args.per_device_batch_size)
        update_steps = math.ceil((steps_per_epoch * args.epochs) / args.gradient_accumulation_steps)
        scheduler = make_scheduler(optimizer, max(update_steps, 1), args.warmup_ratio)

        log_rows = []
        global_micro_step = 0
        global_update_step = 0
        optimizer.zero_grad(set_to_none=True)
        pad_id = agent.tokenizer.pad_token_id or agent.tokenizer.eos_token_id
        first_device = agent.first_device()

        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(features)).tolist()
            iterator = range(0, len(order), args.per_device_batch_size)
            for start in progress(iterator, desc=f"{subset} epoch {epoch}/{args.epochs}"):
                batch_items = [features[idx] for idx in order[start : start + args.per_device_batch_size]]
                batch = collate(batch_items, pad_id)
                batch = {key: value.to(first_device) for key, value in batch.items()}
                loss = batch_loss(agent.model, batch)
                (loss / args.gradient_accumulation_steps).backward()
                global_micro_step += 1
                do_update = (
                    global_micro_step % args.gradient_accumulation_steps == 0
                    or (epoch == args.epochs and start + args.per_device_batch_size >= len(order))
                )
                if do_update:
                    torch.nn.utils.clip_grad_norm_(list(trainable_lora_parameters(agent.model)), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_update_step += 1
                    log_rows.append(
                        {
                            "epoch": epoch,
                            "micro_step": global_micro_step,
                            "update_step": global_update_step,
                            "loss": float(loss.detach().cpu()),
                            "lr": float(scheduler.get_last_lr()[0]),
                        }
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        with (out_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "micro_step", "update_step", "loss", "lr"])
            writer.writeheader()
            writer.writerows(log_rows)

        adapter_dir = out_dir / "adapter"
        adapter_config = {
            "model_alias": args.model_alias,
            "subset": subset,
            "rank": args.rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "module_masks": module_masks_for_config(agent.model, ctd_rows),
            "mask_summary": mask_summary,
        }
        save_masked_lora_adapter(adapter_dir, agent.model, adapter_config)

        summary = {
            "train_rows": len(rows),
            "training_examples": len(examples),
            "trainable_features": len(features),
            "skipped_trajectory_examples": len(skipped_trajectory),
            "skipped_tokenization_examples": len(skipped_features),
            "update_steps": global_update_step,
            "last_loss": log_rows[-1]["loss"] if log_rows else None,
            "mask_summary": mask_summary,
        }
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "manifest.json", {"params": params, "summary": summary})
        print(f"Wrote CTD-Masked LoRA adapter: {adapter_dir}")
    finally:
        agent.close()
        gc.collect()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_alias, args.model_path)
    dataset_root = resolve_path(args.dataset_dir) if args.dataset_dir else path_from_config("modified_dataset_dir")
    neurons_root = resolve_path(args.neurons_dir) if args.neurons_dir else path_from_config("neurons_dir")
    checkpoints_root = resolve_path(args.checkpoints_dir) if args.checkpoints_dir else path_from_config("checkpoints_dir")
    model_dataset = dataset_root / args.model_alias
    shared_root = neurons_root / args.model_alias / "shared_by_subset"
    if not model_dataset.exists():
        raise FileNotFoundError(f"Missing modified dataset dir: {model_dataset}")
    if not shared_root.exists():
        raise FileNotFoundError(f"Missing shared neuron dir: {shared_root}")

    w2t_utils = load_utils(args.when2tool_repo)
    w2t_model = load_model_module(args.when2tool_repo)
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]

    for subset in subsets:
        out_dir = subset_output_dir(checkpoints_root, args.model_alias, subset)
        train_subset(
            subset,
            args=args,
            model_path=model_path,
            model_dataset=model_dataset,
            shared_root=shared_root,
            out_dir=out_dir,
            w2t_utils=w2t_utils,
            w2t_model=w2t_model,
        )


if __name__ == "__main__":
    main()
