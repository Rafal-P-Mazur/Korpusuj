# -*- coding: utf-8 -*-
"""Lightweight tests for stateful creator text processors (track 172g4)."""
from __future__ import annotations

from unittest.mock import patch

from korpusuj.corpus.creator_core import NullProgressReporter
from korpusuj.corpus.creator_nlp import (
    CreatorModelState,
    process_single_text,
    process_single_text_spacy,
)


class DummyApp:
    def after(self, delay, callback):
        callback()

    def update_idletasks(self):
        return None


class DummyLabel:
    def configure(self, **kwargs):
        return None


class DummyProgress:
    def set(self, value):
        return None


def test_stateful_processors_return_none_for_empty_text_without_models():
    state = CreatorModelState()
    reporter = NullProgressReporter()
    assert process_single_text("   ", "empty.txt", state, reporter) is None
    assert process_single_text_spacy("\n\t", "empty.txt", state, reporter) is None


def test_package_creator_stanza_wrapper_delegates_and_preserves_signature():
    from korpusuj.corpus import creator
    sentinel = [{"token": "Ala"}]
    with patch.object(creator, "_process_single_text_stateful", return_value=sentinel) as mocked:
        result = creator.process_single_text(
            "Ala", "a.txt", DummyLabel(), DummyProgress(), DummyApp()
        )
    assert result is sentinel
    assert mocked.call_count == 1
    args = mocked.call_args.args
    assert args[0:2] == ("Ala", "a.txt")
    assert args[2] is creator._creator_model_state


def test_package_creator_spacy_wrapper_delegates_and_preserves_signature():
    from korpusuj.corpus import creator
    sentinel = [{"token": "Kot"}]
    with patch.object(creator, "_process_single_text_spacy_stateful", return_value=sentinel) as mocked:
        result = creator.process_single_text_spacy(
            "Kot", "b.txt", DummyLabel(), DummyProgress(), DummyApp()
        )
    assert result is sentinel
    assert mocked.call_count == 1
    args = mocked.call_args.args
    assert args[0:2] == ("Kot", "b.txt")
    assert args[2] is creator._creator_model_state


def test_creator_nlp_processors_do_not_reference_legacy_model_globals():
    import inspect
    import korpusuj.corpus.creator_nlp as creator_nlp
    stanza_source = inspect.getsource(creator_nlp.process_single_text)
    spacy_source = inspect.getsource(creator_nlp.process_single_text_spacy)
    assert "nlp_stanza" not in stanza_source.replace("state.nlp_stanza", "")
    assert "nlp_spacy" not in spacy_source.replace("state.nlp_spacy", "")
    assert "status_label" not in stanza_source + spacy_source
    assert "progress_bar" not in stanza_source + spacy_source
    assert "update_idletasks" not in stanza_source + spacy_source
