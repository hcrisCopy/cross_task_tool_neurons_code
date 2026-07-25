from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = REPO_ROOT / "PreciseShield" / "ps_causal_validation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PreciseShield PS-10 data-parallel launcher.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--neurons-dir", default=None)
    parser.add_argument("--causal-dir", default=None)
    parser.add_argument("--when2tool-repo", default="third_party/when2tool")
    parser.add_argument("--subset", choices=["single_hop", "multi_hop", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--parallel-mode", choices=["auto", "model", "subset", "data"], default="data", help=argparse.SUPPRESS)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["balanced", "first"], default="first")
    parser.add_argument("--interventions", default="Base,Mask-Random,Mask-PS-TDN_c,Mask-PS-CTD,Mask-PS-Private_c")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--record-mode", choices=["full", "lite", "off"], default="lite")
    parser.add_argument("--seed", type=int, choices=[2026], default=2026)
    parser.add_argument("--keep-shards", action="store_true", help="Keep shard inputs, outputs, logs, and progress files.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ps_sharded_eval_common import resolve_causal_root, run_ps_causal_subset_dp, selected_subsets, write_root_manifest

    subsets = selected_subsets(args.subset)
    causal_root = resolve_causal_root(args.causal_dir)
    root_subsets = {}
    print(f"[PS-10] {args.model_alias}: running subsets sequentially: {', '.join(subsets)}")
    for subset in subsets:
        manifest = run_ps_causal_subset_dp(args, script_path=STAGE_SCRIPT, repo_root=REPO_ROOT, subset=subset)
        root_subsets[subset] = manifest
    write_root_manifest(
        causal_root / args.model_alias / "manifest.json",
        stage="ps_10_causal_validation",
        model_alias=args.model_alias,
        subsets=root_subsets,
    )
    print(f"[PS-10] {args.model_alias}: finished {len(subsets)} subset(s).")


if __name__ == "__main__":
    main()
