from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE4_LAUNCHER = REPO_ROOT / "code" / "11_multigpu" / "run_activation_extraction.py"
DEFAULT_ACTIVATIONS_DIR = "../cross_task_tool_neurons_data/tool_decision_anchors/activations"


def has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(f"{option}=") for item in argv)


def main() -> None:
    argv = list(sys.argv[1:])
    cmd = [sys.executable, str(STAGE4_LAUNCHER), *argv]
    if not has_option(argv, "--activations-dir"):
        cmd.extend(["--activations-dir", DEFAULT_ACTIVATIONS_DIR])
    if not has_option(argv, "--when2tool-repo"):
        cmd.extend(["--when2tool-repo", "third_party/when2tool"])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("ToolDecisionAnchors TDA-4 delegates to the shared Safety Kernel FFN activation extractor.")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
