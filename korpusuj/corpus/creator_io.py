# -*- coding: utf-8 -*-
"""GUI-free near-pure IO helpers used by the corpus creator."""
from __future__ import annotations

import logging
import os
import zipfile
from typing import Any

import pandas as pd


def calculate_real_total_size(file_paths):
    total_size = 0
    for path in file_paths:
        try:
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(path, "r") as archive:
                    for info in archive.infolist():
                        if not info.is_dir():
                            total_size += info.file_size
            else:
                total_size += os.path.getsize(path)
        except Exception as exc:
            logging.warning("Błąd obliczania rozmiaru dla %s: %s", path, exc)
            total_size += os.path.getsize(path)
    return total_size


def process_xlsx(file_path, mapping=None):
    try:
        df = pd.read_excel(file_path)
        data = []

        col_filename = "Nazwa pliku"
        col_title = "Tytuł"
        col_content = "Treść"
        col_date = "Data publikacji"
        col_author = "Autor"

        if mapping:
            col_filename = mapping.get("Nazwa pliku", col_filename)
            col_title = mapping.get("Tytuł", col_title)
            col_content = mapping.get("Treść", col_content)
            col_date = mapping.get("Data publikacji", col_date)
            col_author = mapping.get("Autor", col_author)

        def get_val(row, col_name):
            if col_name == "<Pomiń>" or col_name not in df.columns:
                return ""
            value = row[col_name]
            return str(value).strip() if pd.notna(value) else ""

        for _, row in df.iterrows():
            virt_filename = get_val(row, col_filename)
            if not virt_filename:
                continue

            title = get_val(row, col_title)
            content = get_val(row, col_content)
            if not content:
                continue
            if title and not content.startswith(title):
                content = f"{title}\n\n{content}".strip()

            data.append({
                "filename": virt_filename,
                "Tytuł": title,
                "Treść": content,
                "Data publikacji": get_val(row, col_date),
                "Autor": get_val(row, col_author),
            })
        return data
    except Exception as exc:
        logging.warning("Błąd Excel %s: %s", file_path, exc)
        return []


__all__ = ["calculate_real_total_size", "process_xlsx"]
