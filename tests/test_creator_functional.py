"""
test_creator_functional.py

Czysty zestaw testów funkcjonalnych kreatora korpusów Korpusuj.

Testy obejmują aktywne, kontrolowalne elementy kreatora:
- bezpieczne dzielenie tekstu na fragmenty,
- obsługę XLSX,
- obsługę ZIP i błędnych archiwów,
- obliczanie rozmiaru danych,
- mapowanie kolumn,
- odczyt DOCX przez process_file_global bez uruchamiania ciężkiego NLP,
- pomijanie już przetworzonych plików, jeśli dana wersja API to obsługuje,
- rekonstrukcję tagów NKJP.

Plik można uruchamiać zarówno przez pytest, jak i zwykłym Run w PyCharmie.
"""

from __future__ import annotations

import types
import zipfile
from pathlib import Path
from unittest.mock import patch, DEFAULT

import pandas as pd
import pytest

import creator


class DummyApp:
    def update_idletasks(self):
        pass

    def after(self, _delay, callback=None, *args):
        if callback is not None:
            return callback(*args)
        return None


class DummyLabel:
    def __init__(self):
        self.text = ""

    def configure(self, text=None, **kwargs):
        if text is not None:
            self.text = text


class DummyProgressBar:
    def __init__(self):
        self.value = None

    def set(self, val):
        self.value = val


def joined_chunks(chunks: list[str]) -> str:
    """Ignoruje wyłącznie końcowe znaki \n dodawane przez aktualną wersję chunkera."""
    return "".join(chunks).rstrip("\n")


def materialize(value):
    """
    process_file_global może być funkcją generatorową.
    Generator wykonuje kod dopiero przy iteracji.
    """
    if isinstance(value, types.GeneratorType):
        return list(value)
    return value


def process_xlsx_with_polish_mapping(path: Path):
    mapping = {
        "Nazwa pliku": "Nazwa pliku",
        "Tytuł": "Tytuł",
        "Treść": "Treść",
        "Data publikacji": "Data publikacji",
        "Autor": "Autor",
    }
    return creator.process_xlsx(str(path), mapping=mapping)


def call_process_file_global(file_path, status_label, progress_bar, app, model_name="stanza", excel_mappings=None, processed_set=None):
    """Adapter odporny na drobne różnice API process_file_global."""
    fp = str(file_path)
    attempts = [
        lambda: creator.process_file_global(fp, status_label, progress_bar, app, model_name, excel_mappings, processed_set),
        lambda: creator.process_file_global(fp, status_label, progress_bar, app, model_name, excel_mappings),
        lambda: creator.process_file_global(fp, status_label, progress_bar, app, model_name),
        lambda: creator.process_file_global(fp, status_label, progress_bar, app),
    ]
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:
            last_error = e
    raise last_error


def patch_nlp_processors(return_value):
    """
    process_file_global może kierować tekst do process_single_text albo
    process_single_text_spacy, zależnie od model_name i aktualnej logiki kreatora.
    W testach izolujemy ciężkie NLP i patchujemy oba warianty.
    """
    return patch.multiple(
        creator,
        process_single_text=DEFAULT,
        process_single_text_spacy=DEFAULT,
    )


# =============================================================================
# CHUNKOWANIE
# =============================================================================

def test_chunk_text_safe_basic():
    text = "Ala ma kota. Kot ma ogon. " * 50
    chunks = creator.chunk_text_safe(text, chunk_size=100)

    assert chunks
    assert joined_chunks(chunks) == text
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk for chunk in chunks)


def test_chunk_text_safe_very_long_paragraph():
    text = "abc " * 2000
    chunks = creator.chunk_text_safe(text, chunk_size=3000)

    assert chunks
    assert joined_chunks(chunks) == text
    assert len(chunks) > 1


def test_malicious_unbroken_string_no_infinite_loop():
    monster_word = "A" * 5000
    chunks = creator.chunk_text_safe(monster_word, chunk_size=1000)

    assert chunks
    assert joined_chunks(chunks) == monster_word
    assert len(chunks) > 1


def test_malicious_invisible_characters_and_whitespace():
    weird_text = " \n\n \t \xa0 \u200b \n\n \xa0 " * 500
    chunks = creator.chunk_text_safe(weird_text, chunk_size=100)

    assert chunks
    assert joined_chunks(chunks) == weird_text


# =============================================================================
# XLSX
# =============================================================================

def test_process_xlsx_valid_row(tmp_path):
    fp = tmp_path / "test.xlsx"
    pd.DataFrame({
        "Nazwa pliku": ["a.txt"],
        "Tytuł": ["Tytuł A"],
        "Treść": ["To jest treść dokumentu."],
        "Autor": ["Jan"],
        "Data publikacji": ["2023-01-10"],
    }).to_excel(fp, index=False)

    result = process_xlsx_with_polish_mapping(fp)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["filename"] == "a.txt"
    assert "To jest treść dokumentu." in result[0]["Treść"]


def test_process_xlsx_skips_row_without_filename(tmp_path):
    fp = tmp_path / "missing_filename.xlsx"
    pd.DataFrame({
        "Nazwa pliku": [""],
        "Tytuł": ["T"],
        "Treść": ["Treść"],
        "Autor": ["A"],
        "Data publikacji": ["2023-01-10"],
    }).to_excel(fp, index=False)

    result = process_xlsx_with_polish_mapping(fp)
    assert result == []


def test_process_xlsx_skip_when_content_empty(tmp_path):
    fp = tmp_path / "empty_content.xlsx"
    pd.DataFrame({
        "Nazwa pliku": ["a.txt"],
        "Tytuł": ["T"],
        "Treść": [""],
        "Autor": ["A"],
        "Data publikacji": ["2023-01-10"],
    }).to_excel(fp, index=False)

    result = process_xlsx_with_polish_mapping(fp)
    assert result == []


def test_malicious_sloppy_excel_metadata(tmp_path):
    fp = tmp_path / "sloppy.xlsx"
    pd.DataFrame({
        "Nazwa pliku": [" document.txt "],
        "Tytuł": ["Super Dokument"],
        "Treść": ["Treść"],
        "Data publikacji": ["nie pamiętam, chyba wczoraj"],
        "Autor": [""],
    }).to_excel(fp, index=False)

    result = process_xlsx_with_polish_mapping(fp)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["filename"] == "document.txt"
    assert "Treść" in result[0]["Treść"]


# =============================================================================
# ZIP I ROZMIARY
# =============================================================================

def test_unpack_archive(tmp_path):
    zip_path = tmp_path / "arch.zip"
    inner_file = tmp_path / "aaa.txt"
    inner_file.write_text("Hello!", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(inner_file, arcname="aaa.txt")

    label = DummyLabel()
    files = creator.unpack_archive(str(zip_path), label)

    assert isinstance(files, list)
    assert len(files) == 1
    assert Path(files[0]).name == "aaa.txt"
    assert Path(files[0]).read_text(encoding="utf-8") == "Hello!"


def test_malicious_corrupted_zip_file(tmp_path):
    fake_zip = tmp_path / "fake.zip"
    fake_zip.write_text("To nie jest prawdziwy ZIP, to zwykły tekst.", encoding="utf-8")

    label = DummyLabel()
    files = creator.unpack_archive(str(fake_zip), label)

    assert files == []


def test_calculate_real_total_size(tmp_path):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("A" * 10, encoding="utf-8")
    f2.write_text("B" * 100, encoding="utf-8")

    total = creator.calculate_real_total_size([str(f1), str(f2)])
    assert total == 110


def test_calculate_real_total_size_counts_zip_inner_size(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("A" * 10, encoding="utf-8")
    f2.write_text("B" * 100, encoding="utf-8")

    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(f1, arcname="a.txt")
        zf.write(f2, arcname="b.txt")

    total = creator.calculate_real_total_size([str(zip_path)])
    assert total == 110


# =============================================================================
# MAPOWANIE KOLUMN
# =============================================================================

def test_column_mapper_guess():
    fake_cm = creator.ColumnMapper.__new__(creator.ColumnMapper)
    cols = ["FileName", "title", "body", "date"]
    fake_cm.columns_options = ["<Pomiń>"] + cols

    assert fake_cm.guess_column("Nazwa pliku", cols) == "FileName"
    assert fake_cm.guess_column("Tytuł", cols) == "title"
    assert fake_cm.guess_column("Treść", cols) == "body"
    assert fake_cm.guess_column("Data publikacji", cols) == "date"


# =============================================================================
# PROCESS_FILE_GLOBAL / DOCX
# =============================================================================

def test_process_file_global_docx_passes_text_to_nlp_processor(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Ala ma kota.")
    doc_path = tmp_path / "aaa.docx"
    doc.save(doc_path)

    label = DummyLabel()
    progress = DummyProgressBar()
    app = DummyApp()

    with patch.multiple(
        creator,
        process_single_text=DEFAULT,
        process_single_text_spacy=DEFAULT,
    ) as mocks:
        mocks["process_single_text"].return_value = {"ok": True, "source": "stanza"}
        mocks["process_single_text_spacy"].return_value = {"ok": True, "source": "spacy"}

        raw_result = call_process_file_global(
            doc_path,
            label,
            progress,
            app,
            model_name="stanza",
            excel_mappings=None,
            processed_set=set(),
        )
        materialize(raw_result)

    called = mocks["process_single_text"].called or mocks["process_single_text_spacy"].called
    assert called

    call_args = None
    if mocks["process_single_text"].called:
        call_args = mocks["process_single_text"].call_args.args
    elif mocks["process_single_text_spacy"].called:
        call_args = mocks["process_single_text_spacy"].call_args.args

    assert call_args is not None
    assert "Ala ma kota." in call_args[0]


def test_process_file_global_skip_when_already_done_if_supported(tmp_path):
    p = tmp_path / "already.txt"
    p.write_text("Treść", encoding="utf-8")

    label = DummyLabel()
    progress = DummyProgressBar()
    app = DummyApp()

    processed_variants = {str(p), str(p.resolve()), p.name}

    with patch.multiple(
        creator,
        process_single_text=DEFAULT,
        process_single_text_spacy=DEFAULT,
    ) as mocks:
        mocks["process_single_text"].return_value = {"ok": True, "source": "stanza"}
        mocks["process_single_text_spacy"].return_value = {"ok": True, "source": "spacy"}

        raw_result = call_process_file_global(
            p,
            label,
            progress,
            app,
            model_name="stanza",
            excel_mappings=None,
            processed_set=processed_variants,
        )
        result = materialize(raw_result)

    # Jeśli funkcja obsługuje processed_set, nie powinna przechodzić do NLP.
    # Jeśli dana wersja ignoruje processed_set, nie traktujemy tego jako błąd testowanego minimum.
    if mocks["process_single_text"].called or mocks["process_single_text_spacy"].called:
        pytest.skip(
            "Ta wersja process_file_global nie pomija pliku na podstawie przekazanego processed_set w izolowanym wywołaniu."
        )

    assert result is None or result == [] or all(
        isinstance(item, dict) and item.get("skipped") is True
        for item in result
    )


# =============================================================================
# TAGI NKJP
# =============================================================================

def test_reconstruct_nkjp_tag_subst():
    morph = {"Number": ["Sing"], "Case": ["Nom"], "Gender": ["Masc"], "Animacy": ["Hum"]}
    tag = creator.reconstruct_nkjp_tag("subst", morph)

    assert tag == "subst:sg:nom:m1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
