# -*- coding: utf-8 -*-
"""Tests for production creator protocol/state scaffolding (track 172)."""
from __future__ import annotations

from korpusuj.corpus.creator_core import CreatorRunOptions, NullProgressReporter, ProgressReporter
from korpusuj.corpus.creator_nlp import CreatorModelState


def test_null_progress_reporter_satisfies_runtime_protocol_and_is_noop():
    reporter = NullProgressReporter()
    assert isinstance(reporter, ProgressReporter)
    assert reporter.status("status") is None
    assert reporter.current(0.25) is None
    assert reporter.total(0.5) is None
    assert reporter.size_info("10 MB") is None
    assert reporter.warning("warning") is None
    assert reporter.error("error", ValueError("x")) is None
    assert reporter.tick() is None


def test_creator_run_options_defaults_and_normalization():
    options = CreatorRunOptions(input_files=["a.txt", "b.docx"], output_parquet_file="out.parquet")
    assert options.input_files == ["a.txt", "b.docx"]
    assert options.output_parquet_file == "out.parquet"
    assert options.metadata_path is None
    assert options.model_name == "stanza"
    assert options.excel_mappings is None
    assert options.resume_mode is False
    assert options.processed_set is None


def test_creator_run_options_preserves_explicit_values():
    options = CreatorRunOptions(
        input_files=["a.txt"], output_parquet_file="out.parquet",
        metadata_path="meta.xlsx", model_name="spacy",
        excel_mappings={"Treść": "body"}, resume_mode=True,
        processed_set={"done.txt"},
    )
    assert options.metadata_path == "meta.xlsx"
    assert options.model_name == "spacy"
    assert options.excel_mappings == {"Treść": "body"}
    assert options.resume_mode is True
    assert options.processed_set == {"done.txt"}


def test_creator_model_state_defaults_are_empty():
    state = CreatorModelState()
    assert state.nlp_stanza is None
    assert state.nlp_spacy is None


def test_creator_model_state_clear_all():
    state = CreatorModelState(nlp_stanza=object(), nlp_spacy=object())
    state.clear_all()
    assert state == CreatorModelState()
