# -*- coding: utf-8 -*-
"""Production-negative regression: experimental SRL must not return."""
from __future__ import annotations

import ast
from pathlib import Path


PRODUCTION_FILES = [
    Path("korpusuj/corpus/creator.py"),
    Path("korpusuj/corpus/creator_nlp.py"),
]
FORBIDDEN_TEXT = [
    "roberta-srl-model-v9d-dual-core",
    "AutoModelForTokenClassification",
    "AutoTokenizer",
    "RobertaTokenizerFast",
    "SRL_MODEL_PATH",
    "_initialize_srl_stateful",
    "initialize_srl",
    "srl_tokenizer",
    "srl_model",
    "srl_id2label",
    "srl_device",
    "srl_available",
    "build_srl_frames",
    '"srl"',
    '"srls"',
    '"srl_frames"',
]


def test_creator_production_modules_have_no_experimental_srl():
    for path in PRODUCTION_FILES:
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        found = [marker for marker in FORBIDDEN_TEXT if marker in text]
        assert not found, f"{path} still contains experimental SRL markers: {found}"


def test_creator_production_modules_do_not_import_transformers():
    for path in PRODUCTION_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.split(".")[0] == "transformers" for name in imports), path
