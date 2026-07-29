# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path


def _creator_source():
    path = Path("korpusuj/corpus/creator.py")
    return path, path.read_text(encoding="utf-8")


def _nlp_source():
    path = Path("korpusuj/corpus/creator_nlp.py")
    return path, path.read_text(encoding="utf-8")


def test_source_mode_application_root_points_to_project_root(monkeypatch):
    from korpusuj.corpus import creator
    monkeypatch.delattr(sys, "frozen", raising=False)
    expected = Path(creator.__file__).resolve().parents[2]
    assert creator.get_application_root() == expected
    assert creator.MODELS_DIR == expected / "models"


def test_frozen_mode_application_root_points_to_executable_directory(monkeypatch, tmp_path):
    from korpusuj.corpus import creator
    fake_executable = tmp_path / "Korpusuj.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_executable))
    assert creator.get_application_root() == tmp_path.resolve()


def test_models_directory_is_not_package_local():
    from korpusuj.corpus import creator
    package_local = Path(creator.__file__).resolve().parent / "models"
    assert creator.MODELS_DIR != package_local
    assert creator.MODELS_DIR.name == "models"


def test_creator_wrappers_pass_models_dir_not_base_dir():
    path, text = _creator_source()
    tree = ast.parse(text, filename=str(path))
    targets = {"initialize_stanza", "initialize_spacy"}
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            seen.add(node.name)
            calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)]
            keywords = {kw.arg for call in calls for kw in call.keywords if kw.arg}
            assert "models_dir" in keywords
            assert "base_dir" not in keywords
    assert seen == targets


def test_stateful_initializers_accept_models_dir():
    from korpusuj.corpus.creator_nlp import initialize_spacy, initialize_stanza
    for function in (initialize_stanza, initialize_spacy):
        parameters = inspect.signature(function).parameters
        assert "models_dir" in parameters
        assert "base_dir" not in parameters


def test_creator_model_code_does_not_write_to_meipass():
    _, creator_text = _creator_source()
    _, nlp_text = _nlp_source()
    assert "sys._MEIPASS" not in creator_text
    assert "sys._MEIPASS" not in nlp_text


def test_no_package_relative_model_join_remains():
    _, text = _nlp_source()
    compact = "".join(text.split())
    assert 'os.path.join(base_dir,"models","stanza")' not in compact
    assert 'os.path.join(base_dir,"models","spacy")' not in compact
    assert 'os.path.join(models_dir,"stanza")' in compact
    assert 'os.path.join(models_dir,"spacy")' in compact
