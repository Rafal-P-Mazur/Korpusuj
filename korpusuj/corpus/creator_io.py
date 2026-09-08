# -*- coding: utf-8 -*-
"""GUI-free near-pure IO helpers used by the corpus creator."""
from __future__ import annotations

import logging
import os
import shutil
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



# KORPUSUJ_PATCH_189B_SAFE_ZIP_EXTRACTION
class UnsafeZipEntryError(ValueError):
    """A ZIP member would escape or subvert the selected extraction root."""


def _validated_zip_members(archive, destination):
    """Validate the complete manifest and return safe member/target pairs."""
    from pathlib import Path, PurePosixPath, PureWindowsPath
    import stat

    root = Path(destination).resolve()
    validated = []
    seen_targets = set()

    for member in archive.infolist():
        raw_name = str(member.filename or "")
        if not raw_name or "\x00" in raw_name:
            raise UnsafeZipEntryError("Archiwum ZIP zawiera pustą lub niepoprawną nazwę wpisu.")

        # ZIP formally uses '/', but backslashes are path separators on Windows.
        normalized_name = raw_name.replace("\\", "/")
        posix_path = PurePosixPath(normalized_name)
        windows_path = PureWindowsPath(raw_name)

        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or normalized_name.startswith("//")
            or any(part == ".." for part in posix_path.parts)
        ):
            raise UnsafeZipEntryError(f"Niebezpieczna ścieżka w archiwum ZIP: {raw_name!r}")

        clean_parts = [part for part in posix_path.parts if part not in ("", ".")]
        if not clean_parts:
            raise UnsafeZipEntryError(f"Niepoprawna ścieżka w archiwum ZIP: {raw_name!r}")

        unix_mode = (int(member.external_attr) >> 16) & 0xFFFF
        if stat.S_ISLNK(unix_mode):
            raise UnsafeZipEntryError(f"Dowiązania symboliczne w ZIP nie są obsługiwane: {raw_name!r}")

        target = root.joinpath(*clean_parts).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise UnsafeZipEntryError(f"Wpis ZIP wychodzi poza katalog rozpakowania: {raw_name!r}") from exc

        target_key = str(target).casefold()
        if target_key in seen_targets:
            raise UnsafeZipEntryError(f"Powielona ścieżka docelowa w ZIP: {raw_name!r}")
        seen_targets.add(target_key)
        validated.append((member, target))

    return validated


def safe_extract_zip(archive_path, destination):
    """Safely extract one ZIP after validating its complete manifest.

    Returns absolute paths of extracted regular files in archive order. No ZIP
    member is written when any manifest entry is unsafe.
    """
    from pathlib import Path

    archive_path = Path(archive_path)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        validated = _validated_zip_members(archive, destination)
        for member, target in validated:
            if member.is_dir() or str(member.filename).endswith(("/", "\\")):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source_stream, target.open("wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
            extracted_files.append(str(target))
    return extracted_files
# END KORPUSUJ_PATCH_189B_SAFE_ZIP_EXTRACTION


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


__all__ = ["UnsafeZipEntryError", "calculate_real_total_size", "process_xlsx", "safe_extract_zip"]
