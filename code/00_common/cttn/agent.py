from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .lora import activation_mask, load_masked_lora_adapter


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    *,
    enable_thinking: bool | None = False,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs: Any,
) -> Any:
    call_kwargs = dict(kwargs)
    call_kwargs.update({"tokenize": tokenize, "add_generation_prompt": add_generation_prompt})
    if tools:
        call_kwargs["tools"] = tools
    if enable_thinking is not None:
        call_kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **call_kwargs)
    except TypeError:
        call_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **call_kwargs)


class HFGenerationAgent:
    def __init__(
        self,
        *,
        model_path: str | Path,
        system_prompt: str,
        normalizer: Callable[[str], dict[str, Any]],
        torch_dtype_name: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 2048,
        max_model_len: int = 32768,
        adapter_dir: str | Path | None = None,
        enable_thinking: bool | None = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        batch_size: int = 1,
    ) -> None:
        self.model_path = str(model_path)
        self.system_prompt = system_prompt
        self.normalizer = normalizer
        self.max_new_tokens = int(max_new_tokens)
        self.max_model_len = int(max_model_len)
        self.enable_thinking = enable_thinking
        self.batch_size = max(1, int(batch_size))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype(torch_dtype_name),
            device_map=device_map,
        )
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        if temperature is not None:
            self.generation_config.temperature = float(temperature)
        if top_p is not None:
            self.generation_config.top_p = float(top_p)
        if top_k is not None:
            self.generation_config.top_k = int(top_k)

        self.adapter_config: dict[str, Any] | None = None
        if adapter_dir is not None:
            self.adapter_config = load_masked_lora_adapter(self.model, Path(adapter_dir))
            self.model.eval()

    def first_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def render_prompt(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        return apply_chat_template(
            self.tokenizer,
            messages,
            tools,
            enable_thinking=self.enable_thinking,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        tools_batch: list[list[dict[str, Any]]],
        prefills: list[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        prompts = [self.render_prompt(messages, tools) for messages, tools in zip(messages_batch, tools_batch)]
        if prefills:
            prompts = [prompt + (prefill or "") for prompt, prefill in zip(prompts, prefills)]

        outs: list[dict[str, Any]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_width = encoded["input_ids"].shape[1]
            encoded = {key: value.to(self.first_device()) for key, value in encoded.items()}

            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    generation_config=self.generation_config,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decoded = self.tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
            for offset, text in enumerate(decoded):
                idx = start + offset
                prefill = (prefills[idx] or "") if prefills else ""
                full_text = prefill + text
                out = self.normalizer(full_text)
                out["prompt_text"] = prompts[idx]
                out["finish_reason"] = "length_or_eos"
                outs.append(out)
        return outs

    def activation_mask(self, neuron_rows: list[dict[str, Any]]):
        return activation_mask(self.model, neuron_rows)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
