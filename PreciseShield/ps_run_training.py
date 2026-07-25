from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = REPO_ROOT / "PreciseShield" / "ps_train_masked_lora.py"
MODEL_PARALLEL_ALIASES = {"qwen3-32b", "llama3.3-70b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield PS-7 eight-GPU launcher.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--parallel-mode", choices=["auto", "model", "subset"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="balanced")
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


def gpu_groups(gpus: list[str], n_groups: int) -> list[list[str]]:
    group_size = max(1, (len(gpus) + n_groups - 1) // n_groups)
    return [gpus[start : start + group_size] for start in range(0, len(gpus), group_size)][:n_groups]


def selected_subsets(value: str) -> list[str]:
    return ["single_hop", "multi_hop"] if value == "all" else [value]


def build_cmd(args: argparse.Namespace, subset: str) -> list[str]:
    cmd = [
        sys.executable,
        str(STAGE_SCRIPT),
        "--model-alias",
        args.model_alias,
        "--when2tool-repo",
        args.when2tool_repo,
        "--subset",
        subset,
        "--max-train-samples",
        str(args.max_train_samples),
        "--sample-strategy",
        args.sample_strategy,
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


def run_checked(cmd: list[str], cuda_visible_devices: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    print("+", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={cuda_visible_devices})")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def run_subset_parallel(args: argparse.Namespace, gpus: list[str]) -> None:
    subsets = selected_subsets(args.subset)
    groups = gpu_groups(gpus, len(subsets))
    workers: list[tuple[str, subprocess.Popen[bytes]]] = []
    for subset, group in zip(subsets, groups):
        cmd = build_cmd(args, subset)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(group)
        print("+", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
        workers.append((subset, subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)))
    failed = []
    for subset, proc in workers:
        code = proc.wait()
        if code != 0:
            failed.append((subset, code))
    if failed:
        raise RuntimeError(f"PreciseShield training workers failed: {failed}")


def main() -> None:
    args = parse_args()
    gpus = split_gpus(args.gpus)
    mode = args.parallel_mode
    if mode == "auto":
        mode = "model" if args.model_alias in MODEL_PARALLEL_ALIASES or args.subset != "all" else "subset"
    print(f"PreciseShield PS-7 mode: {mode}; GPUs={gpus}")
    if mode == "subset" and args.subset == "all":
        run_subset_parallel(args, gpus)
        return
    run_checked(build_cmd(args, args.subset), ",".join(gpus))


if __name__ == "__main__":
    main()
