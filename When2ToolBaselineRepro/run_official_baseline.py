from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml

from export_when2tool_json import DEFAULT_RAW_DATASET_DIR, export_all, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPRO_ROOT = "../cross_task_tool_neurons_data/when2tool_baseline_repro"
MODE = "no_reasoning"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def rel_arg(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def clean_path(path: Path, repro_root: Path) -> None:
    path = path.resolve()
    repro_root = repro_root.resolve()
    if path == repro_root or not is_relative_to(path, repro_root):
        raise ValueError(f"Refusing to clean path outside reproduction root: {path}")
    if path.exists():
        shutil.rmtree(path)
        print(f"[clean] {rel(path)}")


def load_model_path(model_alias: str, explicit_model_path: str | None) -> str:
    if explicit_model_path:
        return explicit_model_path
    config_path = REPO_ROOT / "configs" / "models.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    model_cfg = (config.get("models") or {}).get(model_alias)
    if not model_cfg:
        raise KeyError(f"Unknown model alias in configs/models.yaml: {model_alias}")
    return str(model_cfg["local_path"])


def official_data_file(data_dir: Path, subset: str, split: str) -> Path:
    if subset == "single_hop":
        return data_dir / ("tasks_v1_train.json" if split == "train" else "tasks_v1_test.json")
    if subset == "multi_hop":
        return data_dir / ("tasks_v1_multihop_train.json" if split == "train" else "tasks_v1_multihop_test.json")
    raise ValueError(f"Unsupported subset: {subset}")


def subset_output_dir(repro_root: Path, model_alias: str, subset: str) -> Path:
    suffix = "" if subset == "single_hop" else "_multihop"
    return repro_root / "probe_data" / f"{model_alias}{suffix}"


def expected_extract_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / f"train_hidden_{MODE}.pt",
        output_dir / f"test_hidden_{MODE}.pt",
        output_dir / f"train_labels_{MODE}.json",
        output_dir / f"test_labels_{MODE}.json",
        output_dir / "train_no_tool_outputs.json",
        output_dir / "test_no_tool_outputs.json",
    ]


def expected_train_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / f"probe_{MODE}.pt",
        output_dir / f"probe_results_{MODE}.json",
    ]


def expected_transfer_files(output_dir: Path) -> list[Path]:
    return [output_dir / f"transfer_{MODE}.json"]


def complete(paths: Iterable[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def run_command(cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print(f"[run] {printable}", flush=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def run_extract(args: argparse.Namespace, repro_root: Path, data_dir: Path, subset: str, model_path: str) -> None:
    output_dir = subset_output_dir(repro_root, args.model_alias, subset)
    if args.clean:
        clean_path(output_dir, repro_root)
    if not args.overwrite and complete(expected_extract_files(output_dir)):
        print(f"[skip] official extract already complete: {rel(output_dir)}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = official_data_file(data_dir, subset, "train")
    test_path = official_data_file(data_dir, subset, "test")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Official JSON tasks are missing. Run --stage data first.")

    cmd = [
        sys.executable,
        "third_party/when2tool/src/extract_features.py",
        "--model_path",
        model_path,
        "--output_dir",
        rel_arg(output_dir),
        "--data_path",
        rel_arg(train_path),
        "--data_path_test",
        rel_arg(test_path),
        "--phase",
        "all",
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--max_model_len",
        str(args.max_model_len),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--max_rounds",
        str(args.max_rounds),
    ]
    run_command(cmd)


def run_train(args: argparse.Namespace, repro_root: Path, subset: str) -> None:
    output_dir = subset_output_dir(repro_root, args.model_alias, subset)
    if args.clean_train_outputs:
        for path in expected_train_files(output_dir):
            if path.exists():
                path.unlink()
                print(f"[clean] {rel(path)}")
    if not complete(expected_extract_files(output_dir)):
        raise FileNotFoundError(f"Official extraction outputs are incomplete: {rel(output_dir)}")
    if not args.overwrite and complete(expected_train_files(output_dir)):
        print(f"[skip] official train already complete: {rel(output_dir)}")
        return

    cmd = [
        sys.executable,
        "third_party/when2tool/src/train_probe.py",
        "--data_dir",
        rel_arg(output_dir),
        "--mode",
        MODE,
        "--reg",
        str(args.reg),
        "--all_layers",
    ]
    run_command(cmd)


def run_transfer(args: argparse.Namespace, repro_root: Path) -> None:
    single_dir = subset_output_dir(repro_root, args.model_alias, "single_hop")
    multi_dir = subset_output_dir(repro_root, args.model_alias, "multi_hop")
    output_dir = repro_root / "transfer" / f"{args.model_alias}_single_to_multihop"
    if args.clean:
        clean_path(output_dir, repro_root)
    if not complete(expected_train_files(single_dir)):
        raise FileNotFoundError(f"Single-hop official probe is missing: {rel(single_dir)}")
    if not complete(expected_extract_files(multi_dir)):
        raise FileNotFoundError(f"Multi-hop official extraction outputs are incomplete: {rel(multi_dir)}")
    if not args.overwrite and complete(expected_transfer_files(output_dir)):
        print(f"[skip] official transfer already complete: {rel(output_dir)}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "third_party/when2tool/src/eval_probe_transfer.py",
        "--source_probe",
        rel_arg(single_dir / f"probe_{MODE}.pt"),
        "--target_hidden_dir",
        rel_arg(multi_dir),
        "--mode",
        MODE,
        "--output_dir",
        rel_arg(output_dir),
    ]
    run_command(cmd)


def selected_subsets(value: str) -> list[str]:
    if value == "both":
        return ["single_hop", "multi_hop"]
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the upstream When2Tool baseline in an isolated output root.")
    parser.add_argument("--stage", choices=["data", "extract", "train", "transfer", "all"], required=True)
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "both"], default="both")
    parser.add_argument("--model-alias", default="qwen3-4b-instruct")
    parser.add_argument("--model-path", default=None, help="Override configs/models.yaml local_path.")
    parser.add_argument("--raw-dataset-dir", default=DEFAULT_RAW_DATASET_DIR)
    parser.add_argument("--repro-root", default=DEFAULT_REPRO_ROOT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--reg", type=float, default=10000.0)
    parser.add_argument("--overwrite", action="store_true", help="Rerun stages even when expected outputs exist.")
    parser.add_argument("--clean", action="store_true", help="Clean only the relevant isolated output directories first.")
    parser.add_argument(
        "--clean-train-outputs",
        action="store_true",
        help="Remove only official probe/result files before training.",
    )
    args = parser.parse_args()

    repro_root = resolve_path(args.repro_root)
    data_dir = repro_root / "data"
    raw_dataset_dir = resolve_path(args.raw_dataset_dir)
    model_path = load_model_path(args.model_alias, args.model_path)

    repro_root.mkdir(parents=True, exist_ok=True)
    stages = ["data", "extract", "train", "transfer"] if args.stage == "all" else [args.stage]

    if "data" in stages:
        export_all(raw_dataset_dir, data_dir, overwrite=args.overwrite, clean=args.clean and args.stage == "data")

    if "extract" in stages:
        for subset in selected_subsets(args.subset):
            run_extract(args, repro_root, data_dir, subset, model_path)

    if "train" in stages:
        for subset in selected_subsets(args.subset):
            run_train(args, repro_root, subset)

    if "transfer" in stages:
        run_transfer(args, repro_root)

    run_manifest = {
        "model_alias": args.model_alias,
        "model_path": model_path,
        "repro_root": rel(repro_root),
        "stage": args.stage,
        "subset": args.subset,
        "mode": MODE,
        "reg": args.reg,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "max_rounds": args.max_rounds,
    }
    with (repro_root / f"run_{args.model_alias}.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] run manifest -> {rel(repro_root / f'run_{args.model_alias}.json')}")


if __name__ == "__main__":
    main()
