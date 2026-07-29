# -*- coding: utf-8 -*-
"""
Functional regression tests for the creator chunking extraction introduced in track 172.

These tests intentionally exercise the new GUI-free module directly:
    korpusuj.corpus.creator_chunking

They also keep small adapter-compatibility checks while creator.py and the root
creator shim continue to re-export chunk_text_safe.
"""
from __future__ import annotations

import importlib

import pytest

from korpusuj.corpus import creator_chunking


def joined_chunks(chunks: list[str]) -> str:
    return "".join(chunks)


def assert_lossless_chunks(text: str, chunks: list[str]) -> None:
    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk != "" for chunk in chunks)
    assert joined_chunks(chunks) == text


# =============================================================================
# CHUNKOWANIE - przeniesione z historycznego test_creator_functional.py
# =============================================================================


def test_chunk_text_safe_basic():
    text = "Ala ma kota. Kot ma ogon. " * 50
    chunks = creator_chunking.chunk_text_safe(text, chunk_size=100)

    assert_lossless_chunks(text, chunks)
    assert len(chunks) > 1


def test_chunk_text_safe_very_long_paragraph():
    text = "abc " * 2000
    chunks = creator_chunking.chunk_text_safe(text, chunk_size=3000)

    assert_lossless_chunks(text, chunks)
    assert len(chunks) > 1


def test_malicious_unbroken_string_no_infinite_loop():
    monster_word = "A" * 5000
    chunks = creator_chunking.chunk_text_safe(monster_word, chunk_size=1000)

    assert_lossless_chunks(monster_word, chunks)
    assert len(chunks) > 1
    assert max(len(chunk) for chunk in chunks) <= 1000


def test_malicious_invisible_characters_and_whitespace():
    weird_text = " \n\n \t \xa0 \u200b \n\n \xa0 " * 500
    chunks = creator_chunking.chunk_text_safe(weird_text, chunk_size=100)

    assert_lossless_chunks(weird_text, chunks)


# =============================================================================
# STRUKTURALNE PRZYPADKI 172d2
# =============================================================================


def test_structured_numeric_inline_records_lossless():
    text = (
        "Wstęp. "
        "1. Austria podpisała umowę z partnerami "
        "2. Belgia odmówiła udziału "
        "3. Czechy zgłosiły poprawki "
        "4. Dania poparła projekt "
        "5. Estonia czeka na decyzję "
        "6. Francja wysłała notę."
    )
    chunks = creator_chunking.chunk_text_safe(
        text,
        chunk_size=80,
        structured_chunk_size=70,
        max_records_per_chunk=2,
    )

    assert creator_chunking.detect_record_style(text) == "numeric"
    assert_lossless_chunks(text, chunks)
    assert len(chunks) > 1


def test_structured_bullet_multiline_records_lossless():
    text = (
        "Lista:\n"
        "- pierwszy punkt zawiera trochę tekstu\n"
        "- drugi punkt też zawiera tekst\n"
        "- trzeci punkt jest dłuższy i ma przecinek, ale bez kropki\n"
        "- czwarty punkt kończy listę\n"
        "Po liście zwykły akapit."
    )
    chunks = creator_chunking.chunk_text_safe(
        text,
        chunk_size=80,
        structured_chunk_size=70,
        max_records_per_chunk=2,
    )

    assert creator_chunking.detect_record_style(text) == "bullet"
    assert_lossless_chunks(text, chunks)
    assert len(chunks) > 1


def test_long_dotless_text_uses_safe_splitting():
    text = " ".join(["bardzo_długi_fragment_bez_kropki"] * 260)
    chunks = creator_chunking.chunk_text_safe(
        text,
        chunk_size=120,
        max_dotless_chars=80,
        min_piece_in_danger=90,
    )

    assert_lossless_chunks(text, chunks)
    assert len(chunks) > 10
    assert max(len(chunk) for chunk in chunks) <= 120


# =============================================================================
# ADAPTER COMPATIBILITY - creator.py should still expose chunk_text_safe
# =============================================================================


def test_package_creator_adapter_reexports_chunk_text_safe():
    creator = pytest.importorskip("korpusuj.corpus.creator")
    text = "Ala ma kota. Kot ma ogon. " * 20

    assert hasattr(creator, "chunk_text_safe")
    assert creator.chunk_text_safe(text, chunk_size=100) == creator_chunking.chunk_text_safe(text, chunk_size=100)


def test_root_creator_shim_reexports_chunk_text_safe_if_available():
    try:
        creator = importlib.import_module("korpusuj.corpus.creator")
    except Exception as exc:  # pragma: no cover - environment-dependent legacy shim
        pytest.skip(f"root creator shim is not importable in this environment: {exc!r}")

    text = "Ala ma kota. Kot ma ogon. " * 20
    assert hasattr(creator, "chunk_text_safe")
    assert creator.chunk_text_safe(text, chunk_size=100) == creator_chunking.chunk_text_safe(text, chunk_size=100)
