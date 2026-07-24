from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import sys

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from cttn.paths import repo_root, resolve_path


DEFAULT_URL = "https://github.com/Trustworthy-ML-Lab/when2tool.git"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_with_network_turbo(command: str, network_turbo: str) -> None:
    wrapped = f"source {network_turbo} && {command}"
    print("+ bash -lc", wrapped)
    subprocess.run(["bash", "-lc", wrapped], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the official When2Tool repo outside this git repo.")
    parser.add_argument("--repo-dir", default="../when2tool_repo")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--network-turbo", default="", help="Optional path like /etc/network_turbo on the remote server.")
    parser.add_argument("--pull", action="store_true", help="Pull if repo already exists.")
    args = parser.parse_args()

    target = resolve_path(args.repo_dir, base=repo_root())
    target.parent.mkdir(parents=True, exist_ok=True)

    if (target / ".git").exists():
        if args.pull:
            if args.network_turbo:
                run_with_network_turbo(f"git -C {target} pull --ff-only", args.network_turbo)
            else:
                run(["git", "-C", str(target), "pull", "--ff-only"])
        else:
            print(f"When2Tool repo already exists: {target}")
    else:
        if args.network_turbo:
            run_with_network_turbo(f"git clone --depth 1 {args.url} {target}", args.network_turbo)
        else:
            run(["git", "clone", "--depth", "1", args.url, str(target)])

    if not (target / "src" / "utils.py").exists():
        raise FileNotFoundError(f"Fetched repo does not look like When2Tool: {target}")
    print(f"When2Tool repo ready: {target}")


if __name__ == "__main__":
    main()
