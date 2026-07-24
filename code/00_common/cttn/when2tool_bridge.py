from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from .paths import when2tool_repo_path


def add_when2tool_repo(repo_dir: str | Path | None = None) -> Path:
    repo = Path(repo_dir) if repo_dir else when2tool_repo_path()
    repo = repo.resolve()
    if not (repo / "src" / "utils.py").exists() or not (repo / "envs").exists():
        raise FileNotFoundError(
            "When2Tool official repo was not found. Run:\n"
            "python code/00_common/sync_when2tool_repo.py --repo-dir ../when2tool_repo"
        )
    for path in [repo / "src", repo]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo


def import_when2tool_module(module_name: str, repo_dir: str | Path | None = None) -> ModuleType:
    add_when2tool_repo(repo_dir)
    return importlib.import_module(module_name)


def load_utils(repo_dir: str | Path | None = None) -> ModuleType:
    return import_when2tool_module("utils", repo_dir)


def load_model_module(repo_dir: str | Path | None = None) -> ModuleType:
    return import_when2tool_module("model", repo_dir)
