# -*- coding: utf-8 -*-
"""Inspect search-index freshness, compatibility and source identity."""
from __future__ import annotations

import os
import sqlite3
from typing import Any

from korpusuj.index.sqlite_index import SEARCH_INDEX_VERSION, search_sidecar_path

INDEX_STATUSES = ("missing", "fresh", "stale", "incompatible", "corrupt")
_REQUIRED_META_KEYS = ("source_parquet_path", "source_parquet_mtime", "engine_index_version", "indexed_attrs")


def _canonical_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def inspect_search_index(parquet_path, index_path=None, indexed_attrs=None, *, check_integrity=False) -> dict[str, Any]:
    """Inspect a derived .search sidecar without creating or modifying it."""
    parquet = os.fspath(parquet_path)
    index = os.fspath(index_path) if index_path is not None else search_sidecar_path(parquet)
    expected_attrs = tuple(indexed_attrs or ())
    result = {
        "status": "missing", "parquet_path": os.path.abspath(parquet),
        "index_path": os.path.abspath(index), "expected_index_version": SEARCH_INDEX_VERSION,
        "expected_indexed_attrs": list(expected_attrs), "reasons": [], "meta": {},
        "integrity_check": None,
    }
    if not os.path.exists(index):
        result["reasons"].append("index_missing")
        return result
    if not os.path.exists(parquet):
        result["status"] = "stale"; result["reasons"].append("source_parquet_missing")
        return result
    con = None
    try:
        con = sqlite3.connect(index)
        meta = {str(k): str(v) for k, v in con.execute("SELECT key, value FROM meta").fetchall()}
        result["meta"] = meta
        missing = [key for key in _REQUIRED_META_KEYS if not meta.get(key)]
        if missing:
            result["status"] = "incompatible"; result["reasons"].append("missing_meta:" + ",".join(missing))
            return result
        if meta["engine_index_version"] != SEARCH_INDEX_VERSION:
            result["status"] = "incompatible"; result["reasons"].append("index_version_mismatch")
            return result
        if expected_attrs and meta["indexed_attrs"] != ",".join(expected_attrs):
            result["status"] = "stale"; result["reasons"].append("indexed_attrs_mismatch")
            return result
        if _canonical_path(meta["source_parquet_path"]) != _canonical_path(parquet):
            result["status"] = "stale"; result["reasons"].append("source_parquet_path_mismatch")
            return result
        try:
            stored_mtime = float(meta["source_parquet_mtime"])
        except (TypeError, ValueError):
            result["status"] = "incompatible"; result["reasons"].append("invalid_source_parquet_mtime")
            return result
        if stored_mtime != os.path.getmtime(parquet):
            result["status"] = "stale"; result["reasons"].append("source_parquet_mtime_mismatch")
            return result
        if meta.get("source_parquet_size"):
            try:
                stored_size = int(meta["source_parquet_size"])
            except (TypeError, ValueError):
                result["status"] = "incompatible"; result["reasons"].append("invalid_source_parquet_size")
                return result
            if stored_size != os.path.getsize(parquet):
                result["status"] = "stale"; result["reasons"].append("source_parquet_size_mismatch")
                return result
        if check_integrity:
            row = con.execute("PRAGMA integrity_check").fetchone()
            integrity = str(row[0]) if row else "missing-result"
            result["integrity_check"] = integrity
            if integrity.lower() != "ok":
                result["status"] = "corrupt"; result["reasons"].append("integrity_check_failed")
                return result
        result["status"] = "fresh"
        return result
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
        result["status"] = "corrupt"; result["reasons"].append("sqlite_error"); result["error"] = str(exc)
        return result
    except OSError as exc:
        result["status"] = "corrupt"; result["reasons"].append("index_io_error"); result["error"] = str(exc)
        return result
    finally:
        if con is not None:
            con.close()
