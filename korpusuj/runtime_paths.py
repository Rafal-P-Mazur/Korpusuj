# -*- coding: utf-8 -*-
"""Runtime path contracts shared by source and frozen Korpusuj.

Packaged resources are read-only. Configuration, logs and application-owned
working files are stored outside Program Files. Corpus/project artifacts and
model policy are intentionally outside this module's responsibility.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "Korpusuj"


def resource_root() -> Path:
    """Return the read-only source/PyInstaller resource root."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parents[1]


def executable_root() -> Path:
    """Return the executable directory in frozen mode or the project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def portable_mode_enabled() -> bool:
    """Return whether frozen Korpusuj uses portable model storage."""
    return bool(
        getattr(sys, "frozen", False)
        and (executable_root() / "portable.txt").is_file()
    )


def _required_environment_path(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else fallback


def roaming_data_root() -> Path:
    """Return per-user roaming data root for small persistent preferences."""
    fallback = Path.home() / "AppData" / "Roaming"
    return _required_environment_path("APPDATA", fallback) / APP_NAME


def local_data_root() -> Path:
    """Return per-user local data root for logs and application work files."""
    fallback = Path.home() / "AppData" / "Local"
    return _required_environment_path("LOCALAPPDATA", fallback) / APP_NAME


def writable_temp_root() -> Path:
    """Return/create application-owned temp below LOCALAPPDATA, never %TEMP%."""
    path = local_data_root() / "temp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def gui_log_dir() -> Path:
    """Return/create the per-user GUI log directory."""
    path = local_data_root() / "logs" / "gui"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_user_config(source_default: Path | None = None) -> Path:
    """Return the writable config path, copying the old default once if present."""
    target = roaming_data_root() / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and source_default is not None:
        source = Path(source_default)
        if source.is_file():
            shutil.copy2(source, target)
    return target


def _configured_models_dir() -> str | None:
    """Read an optional installed-mode model directory from user config."""
    config_path = roaming_data_root() / "config.json"
    if not config_path.is_file():
        return None
    try:
        import json
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    value = data.get("models_dir") if isinstance(data, dict) else None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def flashcards_root() -> Path:
    """Return/create flashcard storage for portable or installed execution."""
    if portable_mode_enabled():
        path = executable_root() / "fiszki"
    else:
        path = roaming_data_root() / "fiszki"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_root(configured_models_dir: str | Path | None = None) -> Path:
    """Return/create the effective model root for source, portable, or installed mode."""
    if not getattr(sys, "frozen", False):
        path = Path(__file__).resolve().parents[1] / "models"
    elif portable_mode_enabled():
        path = executable_root() / "models"
    else:
        configured = configured_models_dir
        if configured is None or not str(configured).strip():
            configured = _configured_models_dir()
        path = (Path(str(configured)).expanduser()
                if configured is not None and str(configured).strip()
                else local_data_root() / "models")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

PATCH_182N5_MODEL_ROOTS_AND_ML_CACHES_APPLIED = True


def configure_ml_cache_environment(configured_models_dir: str | Path | None = None) -> Path:
    """Route all process-local ML caches below the effective models root."""
    target = models_root(configured_models_dir)
    target.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "SENTENCE_TRANSFORMERS_HOME": target / "sentence-transformers",
        "TORCH_HOME": target / "torch",
    }
    for variable, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    return target
