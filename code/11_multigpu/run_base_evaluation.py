from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_EVAL_SCRIPT = REPO_ROOT / "code" / "08_evaluation" / "evaluate_base_model.py"


def selected_subsets(value: str) -> list[str]:
    return ["single_hop", "multi_hop"] if value == "all" else [value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9 data-parallel launcher for base/default evaluation.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    parser.add_argument("--keep-shards", action="store_true", help="Keep shard inputs, outputs, and logs after merge.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sharded_eval_common import resolve_outputs_root, run_base_subset_dp, write_root_manifest

    subsets = selected_subsets(args.subset)
    outputs_root = resolve_outputs_root(args.outputs_dir)
    root_subsets = {}
    print(f"[Stage 9] {args.model_alias}: running subsets sequentially: {', '.join(subsets)}")
    for subset in subsets:
        summary = run_base_subset_dp(args, script_path=BASE_EVAL_SCRIPT, repo_root=REPO_ROOT, subset=subset)
        root_subsets[subset] = {
            "path": str(outputs_root / args.model_alias / "base_evaluation" / subset),
            "overall": summary.get("overall", summary.get("mean_std", {}).get("overall", {})),
        }
    write_root_manifest(
        outputs_root / args.model_alias / "base_evaluation" / "manifest.json",
        stage="09_base_evaluation",
        model_alias=args.model_alias,
        subsets=root_subsets,
    )
    print(f"[Stage 9] {args.model_alias}: finished {len(subsets)} subset(s).")


if __name__ == "__main__":
    main()
