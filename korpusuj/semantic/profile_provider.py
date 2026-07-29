from __future__ import annotations

# SQLite docs-array/profile_docs provider for korpusuj.semantic.word_profile.compute_word_profile.
# 036J: prefer optional profile_docs(doc_id, profile_json) cache; fallback to docs arrays.

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
import json
import logging
import sqlite3
import time
import zlib


# KORPUSUJ_PATCH_145C3A_LOGGING_GATES_IMPORT
try:
    from korpusuj.search.diagnostics import (
        korpusuj_diagnostics_enabled_145c1,
        korpusuj_verbose_diagnostics_enabled_145c1,
    )
except Exception:
    def korpusuj_diagnostics_enabled_145c1(config_obj=None):
        return False
    def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
        return False
# END KORPUSUJ_PATCH_145C3A_LOGGING_GATES_IMPORT

PROFILE_DOC_COLUMNS = ("lemmas", "sentence_ids", "deprels", "postags", "upostags", "full_postags")
REQUIRED_DOC_COLUMNS = set(PROFILE_DOC_COLUMNS) | {"doc_id"}
DEFAULT_DOC_LOAD_CHUNK_SIZE = 500


def _decode_json_zlib(value, default=None):
    if value is None:
        return default
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        raw = bytes(value)
        try:
            return json.loads(zlib.decompress(raw).decode("utf-8"))
        except Exception:
            pass
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if hasattr(value, "tolist"):
            return value.tolist()
    except Exception:
        pass
    return []


def _chunks(seq: list[int], size: int):
    size = max(1, int(size or DEFAULT_DOC_LOAD_CHUNK_SIZE))
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


@dataclass
class ProfileRowData:
    lemmas: list
    word_ids: list
    head_ids: list
    deprels: list
    upostags: list
    postags: list
    full_postags: list
    sentence_ids: list
    feats: list


class _ILoc:
    def __init__(self, provider: "ProfileDocProvider"):
        self.provider = provider

    def __getitem__(self, idx: int) -> ProfileRowData:
        return self.provider.get_row(int(idx))


class ProfileDocProvider:
    def __init__(
        self,
        corpus_path: str | Path,
        search_path: str | Path | None = None,
        doc_ids: Iterable[int] | None = None,
        *,
        chunk_size: int = DEFAULT_DOC_LOAD_CHUNK_SIZE,
    ):
        self.corpus_path = Path(corpus_path)
        self.search_path = Path(search_path) if search_path else self.corpus_path.with_suffix(".search")
        self.chunk_size = max(1, int(chunk_size or DEFAULT_DOC_LOAD_CHUNK_SIZE))
        self.timings: dict[str, float] = {}
        self.stats: dict[str, int] = {
            "requested_docs": 0,
            "unique_docs": 0,
            "loaded_docs": 0,
            "missing_docs": 0,
            "doc_chunks": 0,
            "dep_maps_loaded": 0,
            "cache_hits": 0,
            "lazy_loads": 0,
            "profile_blob_hits": 0,
            "docs_array_hits": 0,
        }
        self._closed = False
        self._rows: dict[int, ProfileRowData] = {}

        t0 = time.perf_counter()
        self.con = sqlite3.connect(str(self.search_path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.timings["connect_search"] = time.perf_counter() - t0

        self._has_profile_docs = self._detect_profile_docs()
        self.columns = ["lemmas", "word_ids", "head_ids", "deprels", "upostags", "postags", "full_postags", "sentence_ids", "feats"]
        self.iloc = _ILoc(self)
        self._len = self._read_len()

        t1 = time.perf_counter()
        from korpusuj.dependency.disk_cache import DependencyMapDiskCache
        self.dep_cache = DependencyMapDiskCache(self.corpus_path)
        self.timings["open_dep_cache"] = time.perf_counter() - t1

        if doc_ids:
            self.preload(doc_ids)

    def _detect_profile_docs(self) -> bool:
        try:
            row = self.con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='profile_docs'").fetchone()
            if not row:
                return False
            cols = {r[1] for r in self.con.execute("PRAGMA table_info(profile_docs)").fetchall()}
            return {"doc_id", "profile_json"}.issubset(cols)
        except Exception:
            return False

    def __len__(self) -> int:
        return self._len

    def _read_len(self) -> int:
        try:
            row = self.con.execute("SELECT value FROM meta WHERE key='total_docs'").fetchone()
            if row is not None:
                return int(row[0])
        except Exception:
            pass
        try:
            return int(self.con.execute("SELECT COUNT(*) FROM docs").fetchone()[0] or 0)
        except Exception:
            return 0

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.dep_cache.close()
        except Exception:
            pass
        try:
            self.con.close()
        except Exception:
            pass

    def diagnostics(self) -> dict[str, Any]:
        return {
            "corpus": str(self.corpus_path),
            "search": str(self.search_path),
            "df_len": self._len,
            "has_profile_docs": self._has_profile_docs,
            "stats": dict(self.stats),
            "timings": {k: round(float(v), 6) for k, v in self.timings.items()},
        }

    def preload(self, doc_ids: Iterable[int]):
        requested = [int(x) for x in (doc_ids or []) if x is not None]
        ids = sorted(set(requested))
        ids = [x for x in ids if x not in self._rows]
        self.stats["requested_docs"] += len(requested)
        self.stats["unique_docs"] += len(ids)
        if not ids:
            return

        t_total = time.perf_counter()
        t_dep = time.perf_counter()
        dep_many = self.dep_cache.get_many(ids)
        self.stats["dep_maps_loaded"] += len(dep_many)
        self.timings["preload_dep_cache"] = self.timings.get("preload_dep_cache", 0.0) + (time.perf_counter() - t_dep)

        loaded_ids: set[int] = set()

        if self._has_profile_docs:
            t_blob = time.perf_counter()
            for chunk in _chunks(ids, self.chunk_size):
                self.stats["doc_chunks"] += 1
                placeholders = ",".join(["?"] * len(chunk))
                sql = f"SELECT doc_id, profile_json FROM profile_docs WHERE doc_id IN ({placeholders})"
                for row in self.con.execute(sql, chunk).fetchall():
                    doc_id = int(row["doc_id"])
                    payload = _decode_json_zlib(row["profile_json"], {}) or {}
                    loaded_ids.add(doc_id)
                    self._rows[doc_id] = self._build_row(doc_id, payload, dep_many.get(doc_id))
                    self.stats["profile_blob_hits"] += 1
            self.timings["preload_profile_blobs"] = self.timings.get("preload_profile_blobs", 0.0) + (time.perf_counter() - t_blob)

        remaining = [x for x in ids if x not in loaded_ids]
        if remaining:
            t_docs = time.perf_counter()
            for chunk in _chunks(remaining, self.chunk_size):
                if not self._has_profile_docs:
                    self.stats["doc_chunks"] += 1
                placeholders = ",".join(["?"] * len(chunk))
                sql = f"SELECT doc_id, lemmas, sentence_ids, deprels, postags, upostags, full_postags FROM docs WHERE doc_id IN ({placeholders})"
                for row in self.con.execute(sql, chunk).fetchall():
                    doc_id = int(row["doc_id"])
                    loaded_ids.add(doc_id)
                    rec = {c: _as_list(_decode_json_zlib(row[c], [])) for c in PROFILE_DOC_COLUMNS}
                    self._rows[doc_id] = self._build_row(doc_id, rec, dep_many.get(doc_id))
                    self.stats["docs_array_hits"] += 1
            self.timings["preload_docs_arrays"] = self.timings.get("preload_docs_arrays", 0.0) + (time.perf_counter() - t_docs)

        missing = len(set(ids) - loaded_ids)
        self.stats["loaded_docs"] += len(loaded_ids)
        self.stats["missing_docs"] += missing
        self.timings["preload_total"] = self.timings.get("preload_total", 0.0) + (time.perf_counter() - t_total)

    def _build_row(self, doc_id: int, rec: dict[str, list], dep_value=None) -> ProfileRowData:
        lemmas = _as_list(rec.get("lemmas"))
        n = len(lemmas)
        sentence_ids = _as_list(rec.get("sentence_ids"))
        if len(sentence_ids) != n:
            sentence_ids = [0] * n

        def norm_str_array(name: str) -> list[str]:
            arr = [str(x) for x in _as_list(rec.get(name))]
            return arr if len(arr) == n else [""] * n

        if dep_value is None:
            try:
                dep_value = self.dep_cache.get(int(doc_id))
            except Exception:
                dep_value = None
        parent_idx = list(dep_value[0]) if dep_value is not None else [-1] * n
        head_ids = list(parent_idx[:n]) + [-1] * max(0, n - len(parent_idx))
        word_ids = list(range(n))
        return ProfileRowData(
            lemmas=lemmas,
            word_ids=word_ids,
            head_ids=head_ids,
            deprels=norm_str_array("deprels"),
            upostags=norm_str_array("upostags"),
            postags=norm_str_array("postags"),
            full_postags=norm_str_array("full_postags"),
            sentence_ids=sentence_ids,
            feats=[""] * n,
        )

    def get_row(self, doc_id: int) -> ProfileRowData:
        doc_id = int(doc_id)
        cached = self._rows.get(doc_id)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        self.stats["lazy_loads"] += 1
        self.preload([doc_id])
        cached = self._rows.get(doc_id)
        if cached is not None:
            return cached
        empty = ProfileRowData([], [], [], [], [], [], [], [], [])
        self._rows[doc_id] = empty
        return empty


def _doc_ids_from_results(results: Iterable[Any]) -> list[int]:
    out = []
    for res in results or []:
        try:
            out.append(int(res[11]))
        except Exception:
            pass
    return out


def _resolve_paths(df) -> tuple[Path | None, Path | None]:
    corpus_path = getattr(df, "parquet_path", None) or getattr(df, "corpus_path", None) or getattr(df, "path", None)
    search_path = getattr(df, "search_path", None)
    corpus = Path(corpus_path) if corpus_path else None
    search = Path(search_path) if search_path else (corpus.with_suffix(".search") if corpus else None)
    return corpus, search


def profile_provider_status(df) -> tuple[bool, str, Path | None, Path | None]:
    corpus, search = _resolve_paths(df)
    if corpus is None:
        return False, "missing_corpus_path", corpus, search
    if search is None or not search.exists():
        return False, "missing_search_sidecar", corpus, search
    if not corpus.with_suffix(".dep_cache").exists():
        return False, "missing_dep_cache", corpus, search
    try:
        con = sqlite3.connect(str(search))
        try:
            pd_row = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='profile_docs'").fetchone()
            if pd_row:
                pd_cols = {r[1] for r in con.execute("PRAGMA table_info(profile_docs)").fetchall()}
                if {"doc_id", "profile_json"}.issubset(pd_cols):
                    return True, "ok_profile_docs", corpus, search
            cols = {r[1] for r in con.execute("PRAGMA table_info(docs)").fetchall()}
        finally:
            con.close()
        missing = sorted(REQUIRED_DOC_COLUMNS - cols)
        if missing:
            return False, "missing_docs_columns:" + ",".join(missing), corpus, search
        return True, "ok_docs_arrays", corpus, search
    except Exception as exc:
        return False, "schema_error:" + repr(exc), corpus, search


def can_use_profile_provider(df) -> bool:
    ok, _reason, _corpus, _search = profile_provider_status(df)
    return ok


def build_profile_provider_for_results(df, results: Iterable[Any]) -> ProfileDocProvider | None:
    ok, reason, corpus, search = profile_provider_status(df)
    if not ok:
        logging.info("[APP semantic.profile.unavailable] reason=%s", reason)
        return None
    doc_ids = _doc_ids_from_results(results)
    t0 = time.perf_counter()
    provider = ProfileDocProvider(corpus, search, doc_ids=doc_ids)
    elapsed = time.perf_counter() - t0
    diag = provider.diagnostics()
    if korpusuj_verbose_diagnostics_enabled_145c1():
        logging.info(
            "[DIAG perf.semantic.profile] event='enabled' mode=%s docs=%s requested_hits=%s elapsed=%.6fs stats=%r timings=%r search=%r corpus=%r",
            reason, len(set(doc_ids)), len(doc_ids), elapsed, diag.get("stats"), diag.get("timings"), str(search), str(corpus)
        )
    return provider
