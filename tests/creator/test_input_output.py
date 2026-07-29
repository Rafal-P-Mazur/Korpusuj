# -*- coding: utf-8 -*-
"""Functional regressions for creator_io extraction (track 172f2)."""
from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from korpusuj.corpus import creator_io


def process_xlsx_with_polish_mapping(path):
    mapping = {
        "Nazwa pliku": "Nazwa pliku",
        "Tytuł": "Tytuł",
        "Treść": "Treść",
        "Data publikacji": "Data publikacji",
        "Autor": "Autor",
    }
    return creator_io.process_xlsx(str(path), mapping=mapping)


def test_process_xlsx_valid_row(tmp_path):
    path = tmp_path / "test.xlsx"
    pd.DataFrame({
        "Nazwa pliku": ["a.txt"],
        "Tytuł": ["Tytuł A"],
        "Treść": ["To jest treść dokumentu."],
        "Autor": ["Jan"],
        "Data publikacji": ["2023-01-10"],
    }).to_excel(path, index=False)

    result = process_xlsx_with_polish_mapping(path)
    assert len(result) == 1
    assert result[0]["filename"] == "a.txt"
    assert result[0]["Treść"] == "Tytuł A\n\nTo jest treść dokumentu."


def test_process_xlsx_skips_row_without_filename(tmp_path):
    path = tmp_path / "missing_filename.xlsx"
    pd.DataFrame({
        "Nazwa pliku": [""], "Tytuł": ["T"], "Treść": ["Treść"],
        "Autor": ["A"], "Data publikacji": ["2023-01-10"],
    }).to_excel(path, index=False)
    assert process_xlsx_with_polish_mapping(path) == []


def test_process_xlsx_skips_empty_content(tmp_path):
    path = tmp_path / "empty_content.xlsx"
    pd.DataFrame({
        "Nazwa pliku": ["a.txt"], "Tytuł": ["T"], "Treść": [""],
        "Autor": ["A"], "Data publikacji": ["2023-01-10"],
    }).to_excel(path, index=False)
    assert process_xlsx_with_polish_mapping(path) == []


def test_process_xlsx_sloppy_metadata_is_trimmed(tmp_path):
    path = tmp_path / "sloppy.xlsx"
    pd.DataFrame({
        "Nazwa pliku": [" document.txt "], "Tytuł": ["Super Dokument"],
        "Treść": ["Treść"], "Data publikacji": ["nie pamiętam"], "Autor": [""],
    }).to_excel(path, index=False)
    result = process_xlsx_with_polish_mapping(path)
    assert result[0]["filename"] == "document.txt"
    assert result[0]["Treść"] == "Super Dokument\n\nTreść"


def test_calculate_real_total_size_plain_files(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("A" * 10, encoding="utf-8")
    second.write_text("B" * 100, encoding="utf-8")
    assert creator_io.calculate_real_total_size([str(first), str(second)]) == 110


def test_calculate_real_total_size_zip_inner_size(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("A" * 10, encoding="utf-8")
    second.write_text("B" * 100, encoding="utf-8")
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(first, arcname="a.txt")
        archive.write(second, arcname="b.txt")
    assert creator_io.calculate_real_total_size([str(archive_path)]) == 110


def test_package_creator_adapter_reexports_io_helpers():
    pytest.importorskip("customtkinter")
    from korpusuj.corpus import creator
    assert creator.process_xlsx is creator_io.process_xlsx
    assert creator.calculate_real_total_size is creator_io.calculate_real_total_size


def test_root_creator_shim_reexports_io_helpers():
    pytest.importorskip("customtkinter")
    from korpusuj.corpus import creator
    assert creator.process_xlsx is creator_io.process_xlsx
    assert creator.calculate_real_total_size is creator_io.calculate_real_total_size
