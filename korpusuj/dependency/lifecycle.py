# -*- coding: utf-8 -*-
"""Inspect, build and atomically publish dependency and search index artifacts.

Parquet remains canonical. The .search and .dep_cache files are staged,
validated and published as one derived artifact set.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any

import pyarrow.parquet as pq

from korpusuj.dependency.disk_cache import DependencyMapDiskCache
from korpusuj.dependency.maps import build_dependency_maps
from korpusuj.dependency.policy import DEPENDENCY_DISK_CACHE_VERSION
from korpusuj.index.builder import SearchIndexBuilder
from korpusuj.index.status import inspect_search_index

_REQUIRED_DEP_COLUMNS = ("tokens", "word_ids", "head_ids")
_STATUS_PRIORITY = {"fresh": 0, "missing": 1, "stale": 2, "incompatible": 3, "corrupt": 4}


def dependency_cache_path(parquet_path: str | os.PathLike[str]) -> str:
    return str(Path(parquet_path).with_suffix(".dep_cache"))


def _canonical(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _dependency_columns(parquet_path: str | os.PathLike[str]) -> tuple[str, ...] | None:
    pf = pq.ParquetFile(os.fspath(parquet_path))
    available = set(getattr(pf, "schema_arrow", pf.schema).names)
    if any(name not in available for name in _REQUIRED_DEP_COLUMNS):
        return None
    sentence = "sentence_ids" if "sentence_ids" in available else ("sentence_id" if "sentence_id" in available else None)
    if sentence is None:
        return None
    return ("tokens", sentence, "word_ids", "head_ids")


def inspect_dependency_cache(
    parquet_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str] | None = None,
    *,
    check_integrity: bool = False,
) -> dict[str, Any]:
    """Inspect dependency-cache identity, completeness and optional SQLite integrity."""
    parquet = os.fspath(parquet_path)
    cache = os.fspath(cache_path) if cache_path is not None else dependency_cache_path(parquet)
    result: dict[str, Any] = {
        "status": "missing", "parquet_path": os.path.abspath(parquet),
        "cache_path": os.path.abspath(cache), "reasons": [], "meta": {},
        "row_count": None, "integrity_check": None,
        "expected_version": str(DEPENDENCY_DISK_CACHE_VERSION),
    }
    if not os.path.exists(cache):
        result["reasons"].append("dependency_cache_missing")
        return result
    if not os.path.exists(parquet):
        result["status"] = "stale"; result["reasons"].append("source_parquet_missing")
        return result
    try:
        columns = _dependency_columns(parquet)
    except Exception as exc:
        result["status"] = "corrupt"; result["reasons"].append("source_parquet_error"); result["error"] = str(exc)
        return result
    if columns is None:
        result["status"] = "incompatible"; result["reasons"].append("missing_dependency_columns")
        return result
    con = None
    try:
        con = sqlite3.connect(cache)
        meta = {str(k): str(v) for k, v in con.execute("SELECT key, value FROM meta").fetchall()}
        result["meta"] = meta
        required = ("dependency_cache_version", "source_parquet_path", "source_parquet_mtime", "source_parquet_size", "total_docs", "completed")
        missing = [key for key in required if not meta.get(key)]
        if missing:
            result["status"] = "incompatible"; result["reasons"].append("missing_meta:" + ",".join(missing)); return result
        if meta["dependency_cache_version"] != str(DEPENDENCY_DISK_CACHE_VERSION):
            result["status"] = "incompatible"; result["reasons"].append("dependency_cache_version_mismatch"); return result
        if meta["completed"] != "1":
            result["status"] = "stale"; result["reasons"].append("build_incomplete"); return result
        if _canonical(meta["source_parquet_path"]) != _canonical(parquet):
            result["status"] = "stale"; result["reasons"].append("source_parquet_path_mismatch"); return result
        if meta["source_parquet_mtime"] != str(os.path.getmtime(parquet)):
            result["status"] = "stale"; result["reasons"].append("source_parquet_mtime_mismatch"); return result
        if meta["source_parquet_size"] != str(os.path.getsize(parquet)):
            result["status"] = "stale"; result["reasons"].append("source_parquet_size_mismatch"); return result
        expected_docs = int(pq.ParquetFile(parquet).metadata.num_rows or 0)
        stored_docs = int(meta["total_docs"])
        row = con.execute("SELECT COUNT(*) FROM dependency_maps WHERE version=?", (str(DEPENDENCY_DISK_CACHE_VERSION),)).fetchone()
        rows = int(row[0] or 0) if row else 0
        result["row_count"] = rows
        if stored_docs != expected_docs or rows != expected_docs:
            result["status"] = "stale"; result["reasons"].append("dependency_map_count_mismatch"); return result
        if check_integrity:
            row = con.execute("PRAGMA integrity_check").fetchone()
            integrity = str(row[0]) if row else "missing-result"
            result["integrity_check"] = integrity
            if integrity.lower() != "ok":
                result["status"] = "corrupt"; result["reasons"].append("integrity_check_failed"); return result
        result["status"] = "fresh"
        return result
    except (sqlite3.DatabaseError, sqlite3.OperationalError, ValueError, TypeError) as exc:
        result["status"] = "corrupt"; result["reasons"].append("dependency_cache_error"); result["error"] = str(exc)
        return result
    finally:
        if con is not None:
            con.close()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _remove_sqlite_artifacts(path: str | os.PathLike[str]) -> None:
    base = os.fspath(path)
    for candidate in (base, base + "-wal", base + "-shm"):
        try:
            os.remove(candidate)
        except FileNotFoundError:
            pass


def build_dependency_cache_atomic(
    parquet_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str] | None = None,
    *,
    batch_docs: int = 5000,
    progress_callback=None,
    publish: bool = True,
) -> str:
    """Build and validate a dependency cache before optionally publishing it atomically."""
    parquet = os.path.abspath(os.fspath(parquet_path))
    target = os.path.abspath(os.fspath(cache_path) if cache_path is not None else dependency_cache_path(parquet))
    columns = _dependency_columns(parquet)
    if columns is None:
        raise RuntimeError("Brak wymaganych kolumn dependency: tokens, word_ids, head_ids i sentence_ids/sentence_id")
    stage = target + ".tmp"
    _remove_sqlite_artifacts(stage)
    pf = pq.ParquetFile(parquet)
    total_docs = int(pf.metadata.num_rows or 0)
    cache = DependencyMapDiskCache(parquet, cache_path=stage)
    seen = 0
    built = 0
    try:
        cache.mark_rebuild_started(total_docs=total_docs)
        for batch in pf.iter_batches(batch_size=max(1, int(batch_docs or 5000)), columns=list(columns)):
            frame = batch.to_pandas()
            for row in frame.itertuples(index=False):
                tokens = _as_list(getattr(row, "tokens", None))
                sentence_ids = _as_list(getattr(row, columns[1], None))
                word_ids = _as_list(getattr(row, "word_ids", None))
                head_ids = _as_list(getattr(row, "head_ids", None))
                if not tokens or not (len(tokens) == len(sentence_ids) == len(word_ids) == len(head_ids)):
                    raise RuntimeError(f"Nieprawidłowe długości dependency: doc_id={seen}, tokens={len(tokens)}, sentence_ids={len(sentence_ids)}, word_ids={len(word_ids)}, head_ids={len(head_ids)}")
                maps = build_dependency_maps(sentence_ids, word_ids, head_ids)
                if not cache.put(seen, maps, commit=False):
                    raise RuntimeError(f"Nie udało się zapisać mapy dependency: doc_id={seen}")
                seen += 1; built += 1
                if seen % 1000 == 0:
                    cache.commit()
                    if progress_callback is not None:
                        progress_callback(f"Budowanie .dep_cache: {seen} / {total_docs}")
            del frame
        cache.commit()
        if seen != total_docs or built != total_docs or cache.row_count() != total_docs:
            raise RuntimeError(f"Niepełny .dep_cache: expected={total_docs}, seen={seen}, built={built}, rows={cache.row_count()}")
        cache.mark_complete(total_docs=total_docs)
        integrity = str(cache.con.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.lower() != "ok":
            raise RuntimeError(f"PRAGMA integrity_check: {integrity}")
        cache.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cache.con.execute("PRAGMA journal_mode=DELETE")
        cache.commit()
    except Exception:
        cache.close(); _remove_sqlite_artifacts(stage); raise
    else:
        cache.close()
    inspection = inspect_dependency_cache(parquet, stage, check_integrity=True)
    if inspection["status"] != "fresh":
        _remove_sqlite_artifacts(stage)
        raise RuntimeError("Walidacja .dep_cache.tmp nie powiodła się: " + json_safe(inspection))
    if publish:
        os.replace(stage, target)
    if progress_callback is not None:
        progress_callback(f"Cache map zależności gotowy: {target}")
    return target if publish else stage


def json_safe(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def inspect_index_artifacts(parquet_path, search_path=None, indexed_attrs=None, *, check_integrity=True) -> dict[str, Any]:
    """Inspect the combined .search and .dep_cache artifact set for a Parquet corpus."""
    parquet = os.path.abspath(os.fspath(parquet_path))
    search = os.path.abspath(os.fspath(search_path) if search_path is not None else str(Path(parquet).with_suffix(".search")))
    dep = dependency_cache_path(parquet)
    search_result = inspect_search_index(parquet, search, indexed_attrs, check_integrity=check_integrity)
    dep_result = inspect_dependency_cache(parquet, dep, check_integrity=check_integrity)
    statuses = (str(search_result["status"]), str(dep_result["status"]))
    overall = max(statuses, key=lambda status: _STATUS_PRIORITY.get(status, 4))
    return {"status": overall, "search": search_result, "dep_cache": dep_result,
            "parquet_path": parquet, "search_path": search, "dep_cache_path": dep}


def _cleanup_stage(path: str) -> None:
    _remove_sqlite_artifacts(path)
    _remove_sqlite_artifacts(path + ".tmp")


def _publish_pair(search_stage: str, search_target: str, dep_stage: str, dep_target: str) -> None:
    search_backup = search_target + ".publish_backup"
    dep_backup = dep_target + ".publish_backup"
    for path in (search_backup, dep_backup):
        _remove_sqlite_artifacts(path)
    backed_search = False; backed_dep = False; published_search = False; published_dep = False
    try:
        if os.path.exists(search_target): os.replace(search_target, search_backup); backed_search = True
        if os.path.exists(dep_target): os.replace(dep_target, dep_backup); backed_dep = True
        os.replace(search_stage, search_target); published_search = True
        os.replace(dep_stage, dep_target); published_dep = True
    except Exception:
        if published_search: _remove_sqlite_artifacts(search_target)
        if published_dep: _remove_sqlite_artifacts(dep_target)
        if backed_search and os.path.exists(search_backup): os.replace(search_backup, search_target)
        if backed_dep and os.path.exists(dep_backup): os.replace(dep_backup, dep_target)
        raise
    else:
        _remove_sqlite_artifacts(search_backup); _remove_sqlite_artifacts(dep_backup)


def build_index_artifacts_atomic(parquet_path, search_path=None, indexed_attrs=None, *, batch_docs=5000, progress_callback=None) -> dict[str, Any]:
    """Build, validate and publish the .search and .dep_cache artifacts as one set."""
    parquet = os.path.abspath(os.fspath(parquet_path))
    search = os.path.abspath(os.fspath(search_path) if search_path is not None else str(Path(parquet).with_suffix(".search")))
    dep = dependency_cache_path(parquet)
    search_stage = search + ".stage"
    dep_stage = dep + ".stage"
    _cleanup_stage(search_stage); _cleanup_stage(dep_stage)
    try:
        SearchIndexBuilder().build_from_parquet(parquet, index_path=search_stage, indexed_attrs=indexed_attrs, progress_callback=progress_callback)
        dep_stage = build_dependency_cache_atomic(
            parquet, dep_stage, batch_docs=batch_docs,
            progress_callback=progress_callback, publish=False,
        )
        search_check = inspect_search_index(parquet, search_stage, indexed_attrs, check_integrity=True)
        dep_check = inspect_dependency_cache(parquet, dep_stage, check_integrity=True)
        if search_check["status"] != "fresh" or dep_check["status"] != "fresh":
            raise RuntimeError("Walidacja staged artifacts nie powiodła się")
        _publish_pair(search_stage, search, dep_stage, dep)
    except Exception:
        _cleanup_stage(search_stage); _cleanup_stage(dep_stage); raise
    result = inspect_index_artifacts(parquet, search, indexed_attrs, check_integrity=True)
    if result["status"] != "fresh":
        raise RuntimeError("Opublikowany zestaw artefaktów nie jest fresh")
    return result
