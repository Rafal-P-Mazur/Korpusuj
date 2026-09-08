# -*- coding: utf-8 -*-
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from korpusuj.corpus.lemma_corrections import (
    LemmaCorrectionsError,
    apply_lemma_corrections,
    lemma_corrections_identity,
    load_lemma_corrections,
)


def _write_config(path, rules=None):
    payload = {
        "schema_version": 1,
        "name": "test corrections",
        "rules": rules or [{
            "orth": "Ukraina", "lemma": "Ukrain", "upos": "PROPN",
            "replacement": "Ukraina", "reason": "test",
        }],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_apply_and_count_exact_rule(tmp_path):
    config = load_lemma_corrections(str(_write_config(tmp_path / "rules.json")))
    tokens = [
        {"token": "Ukraina", "lemma": "Ukrain", "upos": "PROPN"},
        {"token": "Ukraina", "lemma": "Ukrain", "upos": "NOUN"},
        {"token": "Ukrain", "lemma": "Ukrain", "upos": "PROPN"},
    ]
    apply_lemma_corrections(tokens, config)
    assert [x["lemma"] for x in tokens] == ["Ukraina", "Ukrain", "Ukrain"]
    assert sum(config.counts.values()) == 1
    assert lemma_corrections_identity(config)["config_sha256"]


def test_no_path_is_disabled_and_noop():
    config = load_lemma_corrections(None)
    tokens = [{"token": "Ukraina", "lemma": "Ukrain", "upos": "PROPN"}]
    apply_lemma_corrections(tokens, config)
    assert tokens[0]["lemma"] == "Ukrain"
    assert lemma_corrections_identity(config)["enabled"] is False


def test_duplicate_rule_is_rejected(tmp_path):
    rule = {"orth": "x", "lemma": "y", "upos": "NOUN", "replacement": "z"}
    path = _write_config(tmp_path / "rules.json", [rule, dict(rule)])
    with pytest.raises(LemmaCorrectionsError, match="duplicate"):
        load_lemma_corrections(str(path))


def test_cli_exposes_lemma_corrections():
    from korpusuj.corpus.creator_cli import build_arg_parser
    args = build_arg_parser().parse_args([
        "--input", "a.txt", "--output", "out.parquet",
        "--lemma-corrections", "rules.json",
    ])
    assert args.lemma_corrections == "rules.json"


def test_resume_identity_rejects_added_or_changed_config(tmp_path, monkeypatch):
    import korpusuj.corpus.creator_orchestration as orchestration
    path = _write_config(tmp_path / "rules.json")
    config = load_lemma_corrections(str(path))
    monkeypatch.setattr(orchestration, "_active_lemma_corrections", config)

    table = pa.table({"x": [1]})
    meta = {"lemma_corrections": {"enabled": False, "schema_version": None, "name": None, "config_sha256": None}}
    table = table.replace_schema_metadata({b"korpus_meta": json.dumps(meta).encode("utf-8")})
    part = tmp_path / "part.parquet"
    pq.write_table(table, part)
    with pytest.raises(ValueError, match="lemma_corrections"):
        orchestration._validate_resume_lemma_corrections(str(part))
