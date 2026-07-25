from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE9_SCRIPT = REPO_ROOT / "code" / "09_causal_validation" / "run_causal_validation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10 eight-GPU launcher for causal validation.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--causal-dir", default=None)
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--interventions", default="Base,Mask-Random,Mask-TDN_c,Mask-CTD,Mask-Private_c")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def build_stage9_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(STAGE9_SCRIPT),
        "--model-alias",
        args.model_alias,
        "--when2tool-repo",
        args.when2tool_repo,
        "--subset",
        args.subset,
        "--max-test-samples",
        str(args.max_test_samples),
        "--interventions",
        args.interventions,
        "--batch-size",
        str(args.batch_size),
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
        "--seed",
        str(args.seed),
    ]
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.dataset_dir:
        cmd.extend(["--dataset-dir", args.dataset_dir])
    if args.neurons_dir:
        cmd.extend(["--neurons-dir", args.neurons_dir])
    if args.causal_dir:
        cmd.extend(["--causal-dir", args.causal_dir])
    if args.clean:
        cmd.append("--clean")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def main() -> None:
    args = parse_args()
    gpus = split_gpus(args.gpus)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    cmd = build_stage9_cmd(args)
    print("+", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
