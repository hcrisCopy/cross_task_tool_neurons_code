from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = REPO_ROOT / "PreciseShield" / "ps_extract_intermediate_activations.py"
MODEL_PARALLEL_ALIASES = {"qwen3-32b", "llama3.3-70b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield PS-4 eight-GPU activation launcher.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--parallel-mode", choices=["auto", "data", "model"], default="auto")
    parser.add_argument("--num-data-shards", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--save-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="balanced")
    parser.add_argument("--seed", type=int, choices=[2026, 42, 123456], default=2026)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("--gpus must contain at least one GPU id")
    return items


def selected(values: tuple[str, ...], value: str) -> list[str]:
    return list(values) if value == "all" else [value]


def base_cmd(args: argparse.Namespace, subset: str, split: str) -> list[str]:
    cmd = [
        sys.executable,
        str(STAGE_SCRIPT),
        "--model-alias",
        args.model_alias,
        "--when2tool-repo",
        args.when2tool_repo,
        "--subset",
        subset,
        "--split",
        split,
        "--batch-size",
        str(args.batch_size),
        "--torch-dtype",
        args.torch_dtype,
        "--save-dtype",
        args.save_dtype,
        "--max-samples",
        str(args.max_samples),
        "--sample-strategy",
        args.sample_strategy,
        "--seed",
        str(args.seed),
        "--device-map",
        "auto",
    ]
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.dataset_dir:
        cmd.extend(["--dataset-dir", args.dataset_dir])
    if args.activations_dir:
        cmd.extend(["--activations-dir", args.activations_dir])
    if args.clean:
        cmd.append("--clean")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_checked(cmd: list[str], *, cuda_visible_devices: str | None = None) -> None:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    suffix = f" (CUDA_VISIBLE_DEVICES={cuda_visible_devices})" if cuda_visible_devices else ""
    print("+", " ".join(cmd) + suffix)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def run_data_parallel(args: argparse.Namespace, gpus: list[str], subset: str, split: str) -> None:
    num_shards = args.num_data_shards or len(gpus)
    if num_shards > len(gpus):
        raise ValueError("--num-data-shards cannot exceed the number of GPUs in data mode")
    if num_shards == 1:
        run_checked(base_cmd(args, subset, split), cuda_visible_devices=gpus[0])
        return
    workers: list[tuple[int, subprocess.Popen[bytes]]] = []
    for shard_index in range(num_shards):
        cmd = base_cmd(args, subset, split)
        cmd.extend(["--num-data-shards", str(num_shards), "--data-shard-index", str(shard_index)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[shard_index]
        print("+", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={gpus[shard_index]})")
        workers.append((shard_index, subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)))

    failed = []
    for shard_index, proc in workers:
        code = proc.wait()
        if code != 0:
            failed.append((shard_index, code))
    if failed:
        raise RuntimeError(f"PreciseShield activation shard workers failed: {failed}")

    merge_cmd = base_cmd(args, subset, split)
    merge_cmd.extend(["--num-data-shards", str(num_shards), "--merge-data-shards"])
    run_checked(merge_cmd)


def main() -> None:
    args = parse_args()
    gpus = split_csv(args.gpus)
    mode = args.parallel_mode
    if mode == "auto":
        mode = "model" if args.model_alias in MODEL_PARALLEL_ALIASES else "data"
    print(f"PreciseShield PS-4 mode: {mode}; GPUs={gpus}")

    for subset in selected(("single_hop", "multi_hop"), args.subset):
        for split in selected(("train", "test"), args.split):
            if mode == "model":
                run_checked(base_cmd(args, subset, split), cuda_visible_devices=",".join(gpus))
            else:
                run_data_parallel(args, gpus, subset, split)


if __name__ == "__main__":
    main()
