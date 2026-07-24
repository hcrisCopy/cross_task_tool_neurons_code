from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from cttn.data import SUBSETS, SPLITS, load_raw_tasks, summarize_records
from cttn.io import write_json
from cttn.paths import clean_directory, data_root, ensure_dir, path_from_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw When2Tool parquet files and write a manifest.")
    parser.add_argument("--raw-dataset-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = resolve_path(args.raw_dataset_dir) if args.raw_dataset_dir else path_from_config("raw_dataset_dir")
    output_path = resolve_path(args.output_path) if args.output_path else raw_dir / "manifest.json"

    if output_path.exists() and not args.overwrite and not args.clean:
        print(f"Skip existing manifest: {output_path}")
        return
    if args.clean and output_path.exists():
        output_path.unlink()

    manifest = {
        "raw_dataset_dir": str(raw_dir),
        "subsets": {},
        "env_to_task_type": {},
    }
    for subset in SUBSETS:
        manifest["subsets"][subset] = {}
        for split in SPLITS:
            tasks = load_raw_tasks(raw_dir, subset, split)
            summary = summarize_records(tasks)
            summary["by_multi_step"] = dict(Counter(str(task.get("multi_step")) for task in tasks))
            manifest["subsets"][subset][split] = summary
            print(f"{subset}/{split}: {summary}")
            for task in tasks:
                manifest["env_to_task_type"][task["env_name"]] = task["task_type"]

    ensure_dir(output_path.parent)
    write_json(output_path, manifest)
    print(f"Wrote manifest: {output_path}")


if __name__ == "__main__":
    main()
