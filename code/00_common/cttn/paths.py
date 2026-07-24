from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return repo_root() / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_paths_config() -> dict[str, Any]:
    return load_yaml(config_dir() / "paths.yaml")


def load_models_config() -> dict[str, Any]:
    return load_yaml(config_dir() / "models.yaml")


def load_experiment_config() -> dict[str, Any]:
    return load_yaml(config_dir() / "experiment.yaml")


def load_stage_defaults() -> dict[str, Any]:
    return load_yaml(config_dir() / "stage_defaults.yaml")


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return ((base or repo_root()) / p).resolve()


def path_from_config(key: str) -> Path:
    cfg = load_paths_config()
    if key not in cfg:
        raise KeyError(f"Missing path key in configs/paths.yaml: {key}")
    return resolve_path(cfg[key])


def data_root() -> Path:
    return path_from_config("data_root")


def when2tool_repo_path() -> Path:
    return path_from_config("when2tool_repo")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def clean_directory(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if path == allowed_root or not is_relative_to(path, allowed_root):
        raise ValueError(f"Refusing to clean path outside allowed root: {path}")
    if path.exists():
        shutil.rmtree(path)


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_matches(path: Path, expected_params: dict[str, Any]) -> bool:
    manifest = read_manifest(path)
    if not manifest:
        return False
    return manifest.get("params") == expected_params
