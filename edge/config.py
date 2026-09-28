"""Loads edge/config.yaml. Relative paths in the config are resolved against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


class Section(dict):
    """A config mapping that also allows attribute access (cfg.models.v2.imgsz)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Section({key: _wrap(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_wrap(value) for value in obj]
    return obj


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(*paths: str | Path) -> Section:
    """Loads edge/config.yaml, then deep-merges each overlay in order (e.g. edge/config.pi5.yaml)."""
    merged: dict = {}
    for path in (DEFAULT_CONFIG, *paths):
        if Path(path).resolve() == DEFAULT_CONFIG and merged:
            continue
        with open(path, encoding="utf-8") as fh:
            _merge(merged, yaml.safe_load(fh) or {})
    return _wrap(merged)


def repo_path(path: str | Path | None) -> Path | None:
    """Resolves a config path: absolute paths are kept, relative ones are taken from the repo root."""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
