# -*- coding: utf-8 -*-
from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from korpusuj.corpus.creator_io import UnsafeZipEntryError, safe_extract_zip


def _write_zip(path: Path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in entries:
            if isinstance(value, zipfile.ZipInfo):
                archive.writestr(value, b"target")
            else:
                archive.writestr(name, value)


def test_safe_extract_zip_extracts_nested_regular_files(tmp_path):
    archive = tmp_path / "ok.zip"
    _write_zip(archive, [("a.txt", "A"), ("nested/b.txt", "B")])
    destination = tmp_path / "out"

    extracted = safe_extract_zip(archive, destination)

    assert [Path(item).relative_to(destination).as_posix() for item in extracted] == ["a.txt", "nested/b.txt"]
    assert (destination / "a.txt").read_text(encoding="utf-8") == "A"
    assert (destination / "nested" / "b.txt").read_text(encoding="utf-8") == "B"


@pytest.mark.parametrize("member", [
    "../escape.txt",
    "nested/../../escape.txt",
    "/absolute.txt",
    r"C:\\escape.txt",
    r"\\server\\share\\escape.txt",
    r"nested\\..\\escape.txt",
])
def test_safe_extract_zip_rejects_escaping_paths_before_writing(tmp_path, member):
    archive = tmp_path / "unsafe.zip"
    _write_zip(archive, [("safe.txt", "SAFE"), (member, "BAD")])
    destination = tmp_path / "out"

    with pytest.raises(UnsafeZipEntryError):
        safe_extract_zip(archive, destination)

    assert not (destination / "safe.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_zip_rejects_symlink_before_writing(tmp_path):
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    _write_zip(archive, [("safe.txt", "SAFE"), ("link", link)])
    destination = tmp_path / "out"

    with pytest.raises(UnsafeZipEntryError):
        safe_extract_zip(archive, destination)

    assert not (destination / "safe.txt").exists()


def test_safe_extract_zip_rejects_duplicate_normalized_target(tmp_path):
    archive = tmp_path / "duplicate.zip"
    _write_zip(archive, [("folder/file.txt", "A"), (r"folder\\file.txt", "B")])

    with pytest.raises(UnsafeZipEntryError):
        safe_extract_zip(archive, tmp_path / "out")
