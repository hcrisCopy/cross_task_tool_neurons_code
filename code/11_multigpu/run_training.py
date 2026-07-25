from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE7_SCRIPT = REPO_ROOT / "code" / "07_training" / "train_ctd_masked_lora.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7 eight-GPU launcher for CTD-Masked LoRA training.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-train-samples", type=int, default=0)
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
    parser.add_argument("--seed", type=int, choices=[2026, 42, 123456], default=2026)
    parser.add_argument("--trajectory-attempts", type=int, default=2)
    parser.add_argument("--trajectory-batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite"], default="full")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def split_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def build_stage7_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(STAGE7_SCRIPT),
        "--model-alias",
        args.model_alias,
        "--when2tool-repo",
        args.when2tool_repo,
        "--subset",
        args.subset,
        "--max-train-samples",
        str(args.max_train_samples),
        "--rank",
        str(args.rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--epochs",
        str(args.epochs),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--max-seq-length",
        str(args.max_seq_length),
        "--seed",
        str(args.seed),
        "--trajectory-attempts",
        str(args.trajectory_attempts),
        "--trajectory-batch-size",
        str(args.trajectory_batch_size),
        "--max-rounds",
        str(args.max_rounds),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--torch-dtype",
        args.torch_dtype,
        "--device-map",
        args.device_map,
        "--record-mode",
        args.record_mode,
    ]
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.dataset_dir:
        cmd.extend(["--dataset-dir", args.dataset_dir])
    if args.neurons_dir:
        cmd.extend(["--neurons-dir", args.neurons_dir])
    if args.checkpoints_dir:
        cmd.extend(["--checkpoints-dir", args.checkpoints_dir])
    if args.clean:
        cmd.append("--clean")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_gradient_checkpointing:
        cmd.append("--no-gradient-checkpointing")
    return cmd


def main() -> None:
    args = parse_args()
    gpus = split_gpus(args.gpus)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    cmd = build_stage7_cmd(args)
    print("+", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
