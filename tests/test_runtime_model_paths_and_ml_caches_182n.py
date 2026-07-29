from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

import korpusuj.runtime_paths as runtime_paths


CACHE_RELATIVE = {
    "HF_HOME": Path(".huggingface"),
    "HF_HUB_CACHE": Path(".huggingface/hub"),
    "HUGGINGFACE_HUB_CACHE": Path(".huggingface/hub"),
    "HF_XET_CACHE": Path(".huggingface/xet"),
    "TRANSFORMERS_CACHE": Path(".huggingface/transformers"),
    "SENTENCE_TRANSFORMERS_HOME": Path("sentence-transformers"),
    "TORCH_HOME": Path("torch"),
}


def _clear_cache_environment(monkeypatch):
    for variable in CACHE_RELATIVE:
        monkeypatch.delenv(variable, raising=False)


def test_source_mode_ignores_installed_models_configuration(monkeypatch, tmp_path):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(runtime_paths, "__file__", str(tmp_path / "project" / "korpusuj" / "runtime_paths.py"))
    monkeypatch.setattr(runtime_paths, "_configured_models_dir", lambda: str(tmp_path / "installed-models"))
    assert runtime_paths.models_root() == (tmp_path / "project" / "models").resolve()


def test_frozen_portable_uses_models_beside_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_paths, "portable_mode_enabled", lambda: True)
    monkeypatch.setattr(runtime_paths, "executable_root", lambda: tmp_path / "portable")
    assert runtime_paths.models_root() == (tmp_path / "portable" / "models").resolve()


def test_frozen_installed_uses_configured_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_paths, "portable_mode_enabled", lambda: False)
    monkeypatch.setattr(runtime_paths, "_configured_models_dir", lambda: str(tmp_path / "chosen-models"))
    assert runtime_paths.models_root() == (tmp_path / "chosen-models").resolve()


def test_frozen_installed_fallback_uses_localappdata_models(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_paths, "portable_mode_enabled", lambda: False)
    monkeypatch.setattr(runtime_paths, "_configured_models_dir", lambda: None)
    monkeypatch.setattr(runtime_paths, "local_data_root", lambda: tmp_path / "local" / "Korpusuj")
    assert runtime_paths.models_root() == (tmp_path / "local" / "Korpusuj" / "models").resolve()

def test_huggingface_uses_library_default_cache_while_other_ml_caches_remain_routed():
    runtime_source = (PROJECT_ROOT / "korpusuj" / "runtime_paths.py").read_text(
        encoding="utf-8-sig"
    )
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        assert f'"{key}"' not in runtime_source
        assert f"'{key}'" not in runtime_source
    assert '"SENTENCE_TRANSFORMERS_HOME": target / "sentence-transformers"' in runtime_source
    assert '"TORCH_HOME": target / "torch"' in runtime_source

def test_stanza_initialization_failure_logs_full_runtime_traceback():
    source = (PROJECT_ROOT / "korpusuj" / "corpus" / "creator_nlp.py").read_text(
        encoding="utf-8-sig"
    )
    assert "STANZA_INITIALIZATION_FAILURE_182Y" in source
    assert "logging.getLogger(__name__).exception(" in source
    assert 'bool(getattr(sys, "frozen", False))' in source
    assert 'getattr(sys, "_MEIPASS", None)' in source
    assert 'os.environ.get("HF_HOME")' in source
    assert 'os.environ.get("HF_HUB_CACHE")' in source
    assert '_package_version_182y("stanza")' in source
    assert '_package_version_182y("transformers")' in source
    assert '_package_version_182y("peft")' in source
    assert '_package_version_182y("torch")' in source
    assert 'reporter.error("Nie udało się załadować Stanza", exc)' in source

