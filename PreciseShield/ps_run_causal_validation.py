from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = REPO_ROOT / "PreciseShield" / "ps_causal_validation.py"
MODEL_PARALLEL_ALIASES = {"qwen3-32b", "llama3.3-70b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield PS-10 eight-GPU launcher.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--causal-dir", default=None)
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--parallel-mode", choices=["auto", "model", "subset"], default="auto")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="balanced")
    parser.add_argument("--interventions", default="Base,Mask-Random,Mask-PS-TDN_c,Mask-PS-CTD,Mask-PS-Private_c")
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


def selected_subsets(value: str) -> list[str]:
    return ["single_hop", "multi_hop"] if value == "all" else [value]


def gpu_groups(gpus: list[str], n_groups: int) -> list[list[str]]:
    group_size = max(1, (len(gpus) + n_groups - 1) // n_groups)
    return [gpus[start : start + group_size] for start in range(0, len(gpus), group_size)][:n_groups]


def resolve_causal_root(value: str | None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return (REPO_ROOT.parent / "cross_task_tool_neurons_data" / "precise_shield" / "causal_validation").resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
        "--max-test-samples",
        str(args.max_test_samples),
        "--sample-strategy",
        args.sample_strategy,
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
        raise RuntimeError(f"PreciseShield causal-validation workers failed: {failed}")


def merge_root_manifest(args: argparse.Namespace) -> None:
    root = resolve_causal_root(args.causal_dir) / args.model_alias
    payload: dict[str, Any] = {"stage": "ps_10_causal_validation", "model_alias": args.model_alias, "subsets": {}}
    for subset in selected_subsets(args.subset):
        manifest_path = root / subset / "manifest.json"
        if manifest_path.exists():
            payload["subsets"][subset] = read_json(manifest_path)
    write_json(root / "manifest.json", payload)


def main() -> None:
    args = parse_args()
    gpus = split_gpus(args.gpus)
    mode = args.parallel_mode
    if mode == "auto":
        mode = "model" if args.model_alias in MODEL_PARALLEL_ALIASES or args.subset != "all" else "subset"
    print(f"PreciseShield PS-10 mode: {mode}; GPUs={gpus}")
    if mode == "subset" and args.subset == "all":
        run_subset_parallel(args, gpus)
        merge_root_manifest(args)
        return
    run_checked(build_cmd(args, args.subset), ",".join(gpus))


if __name__ == "__main__":
    main()
