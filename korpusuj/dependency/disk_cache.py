# -*- coding: utf-8 -*-
from __future__ import annotations
import logging, pickle, sqlite3, threading
from datetime import datetime
from pathlib import Path
from korpusuj.dependency.policy import DEPENDENCY_DISK_CACHE_VERSION, DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE
from korpusuj.dependency.maps import LazyChildrenLookup, _encode_parent_idx_int32, _decode_parent_idx_int32


# KORPUSUJ_PATCH_145C5A_LOGGING_GATES_IMPORT
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
# END KORPUSUJ_PATCH_145C5A_LOGGING_GATES_IMPORT

def _dependency_cache_source_signature(corpus_path):
    try:
        p = Path(corpus_path)
        st = p.stat()
        return {
            "source_parquet_path": str(p.resolve()),
            "source_parquet_mtime": str(st.st_mtime),
            "source_parquet_size": str(st.st_size),
        }
    except Exception:
        return {
            "source_parquet_path": str(corpus_path or ""),
            "source_parquet_mtime": "",
            "source_parquet_size": "",
        }

def _dependency_cache_path_for_corpus_path(corpus_path):
    return str(Path(corpus_path).with_suffix(".dep_cache"))

class DependencyMapDiskCache:
    """Persistent dependency cache 3f: doc_id -> compact parent_idx int32 BLOB."""

    def __init__(self, corpus_path, version=DEPENDENCY_DISK_CACHE_VERSION, cache_path=None):
        self.corpus_path = str(corpus_path)
        self.cache_path = str(cache_path) if cache_path is not None else _dependency_cache_path_for_corpus_path(corpus_path)
        self.version = str(version)
        self.lock = threading.RLock()
        self.con = sqlite3.connect(self.cache_path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("PRAGMA temp_store=MEMORY")
        self._create_schema()

    def _create_schema(self):
        with self.lock:
            self.con.executescript("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dependency_maps (
                    doc_id INTEGER NOT NULL,
                    version TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (doc_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_dependency_maps_version ON dependency_maps(version);
            """)
            self.con.commit()

    def close(self):
        try:
            with self.lock:
                self.con.close()
        except Exception:
            pass

    def meta(self):
        try:
            with self.lock:
                return {r[0]: r[1] for r in self.con.execute("SELECT key, value FROM meta")}
        except Exception:
            return {}

    def set_meta(self, values):
        rows = [(str(k), str(v)) for k, v in (values or {}).items()]
        if rows:
            with self.lock:
                self.con.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", rows)
                self.con.commit()

    def row_count(self):
        try:
            with self.lock:
                row = self.con.execute(
                    "SELECT COUNT(*) FROM dependency_maps WHERE version=?",
                    (self.version,)
                ).fetchone()
            return int(row[0] or 0)
        except Exception:
            return 0

    def payload_bytes_for_doc_ids(self, doc_ids):
        try:
            ids = sorted({int(x) for x in doc_ids})
        except Exception:
            ids = []
        if not ids:
            return 0
        total = 0
        try:
            with self.lock:
                for i in range(0, len(ids), DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE):
                    batch = ids[i:i + DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE]
                    placeholders = ",".join(["?"] * len(batch))
                    sql = f"SELECT SUM(LENGTH(payload)) FROM dependency_maps WHERE version=? AND doc_id IN ({placeholders})"
                    row = self.con.execute(sql, [self.version] + batch).fetchone()
                    total += int(row[0] or 0)
        except Exception:
            return 0
        return total

    def is_fresh_and_complete(self, total_docs=None):
        meta = self.meta()
        sig = _dependency_cache_source_signature(self.corpus_path)
        if meta.get("dependency_cache_version") != self.version:
            return False
        if meta.get("source_parquet_mtime") != sig.get("source_parquet_mtime"):
            return False
        if meta.get("source_parquet_size") != sig.get("source_parquet_size"):
            return False
        if meta.get("completed") != "1":
            return False
        if total_docs is not None:
            try:
                expected = int(total_docs)
                if expected > 0:
                    if int(meta.get("total_docs", -1)) != expected:
                        return False
                    if self.row_count() < expected:
                        return False
            except Exception:
                return False
        return True

    def mark_rebuild_started(self, total_docs=None):
        sig = _dependency_cache_source_signature(self.corpus_path)
        try:
            with self.lock:
                self.con.execute("DELETE FROM dependency_maps WHERE version=?", (self.version,))
                self.con.commit()
        except Exception:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] event='clear_before_rebuild_failed' path=%s", self.cache_path, exc_info=True)
        self.set_meta({
            "dependency_cache_version": self.version,
            "source_parquet_path": sig.get("source_parquet_path", ""),
            "source_parquet_mtime": sig.get("source_parquet_mtime", ""),
            "source_parquet_size": sig.get("source_parquet_size", ""),
            "total_docs": int(total_docs or 0),
            "completed": "0",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        })

    def mark_complete(self, total_docs=None):
        self.set_meta({
            "total_docs": int(total_docs or 0),
            "completed": "1",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })

    def _decode_payload(self, payload):
        parent_idx = _decode_parent_idx_int32(payload)
        if parent_idx is not None:
            return (parent_idx, LazyChildrenLookup(parent_idx))
        try:
            raw = pickle.loads(payload)
            if isinstance(raw, tuple) and len(raw) == 2:
                parent_idx, children_lookup = raw
            elif isinstance(raw, list) and len(raw) == 2:
                parent_idx, children_lookup = raw
            elif isinstance(raw, dict):
                parent_idx = raw.get("parent_idx")
                children_lookup = raw.get("children_lookup")
            else:
                return None
            if not isinstance(children_lookup, list):
                children_lookup = LazyChildrenLookup(parent_idx or [])
            if not hasattr(parent_idx, "__len__"):
                return None
            return (parent_idx, children_lookup)
        except Exception:
            return None

    def get(self, doc_id):
        try:
            with self.lock:
                row = self.con.execute(
                    "SELECT payload FROM dependency_maps WHERE doc_id=? AND version=?",
                    (int(doc_id), self.version)
                ).fetchone()
                if row is None and self.version != globals().get("DEPENDENCY_LEGACY_DISK_CACHE_VERSION"):
                    row = self.con.execute(
                        "SELECT payload FROM dependency_maps WHERE doc_id=? AND version=?",
                        (int(doc_id), globals().get("DEPENDENCY_LEGACY_DISK_CACHE_VERSION"))
                    ).fetchone()
            if row is None:
                return None
            return self._decode_payload(row[0])
        except Exception:
            return None

    def get_many(self, doc_ids, batch_size=DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE):
        """Batchowy odczyt map dependency dla listy doc_id; zwraca {doc_id: (parent_idx, children_lookup)}."""
        out = {}
        try:
            ids = sorted({int(x) for x in doc_ids})
        except Exception:
            ids = []
        if not ids:
            return out
        batch_size = max(1, int(batch_size or DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE))
        try:
            with self.lock:
                for i in range(0, len(ids), batch_size):
                    batch = ids[i:i + batch_size]
                    placeholders = ",".join(["?"] * len(batch))
                    sql = f"SELECT doc_id, payload FROM dependency_maps WHERE version=? AND doc_id IN ({placeholders})"
                    rows = self.con.execute(sql, [self.version] + batch).fetchall()
                    for doc_id, payload in rows:
                        dep_maps = self._decode_payload(payload)
                        if dep_maps is not None:
                            out[int(doc_id)] = dep_maps
        except Exception as e:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] path=%s ids=%s reason=%r", self.cache_path, len(ids), e)
        return out

    def get_all(self, batch_size=DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE):
        """Wczytuje wszystkie mapy z .dep_cache. Używane tylko w trybie 'Maksymalna wydajność'."""
        out = {}
        try:
            with self.lock:
                rows = self.con.execute(
                    "SELECT doc_id, payload FROM dependency_maps WHERE version=? ORDER BY doc_id",
                    (self.version,)
                ).fetchall()
            for doc_id, payload in rows:
                dep_maps = self._decode_payload(payload)
                if dep_maps is not None:
                    out[int(doc_id)] = dep_maps
        except Exception as e:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] path=%s reason=%r", self.cache_path, e)
        return out

    def put(self, doc_id, dep_maps, commit=True):
        try:
            parent_idx, _children_lookup = dep_maps
            payload = _encode_parent_idx_int32(parent_idx)
            if payload is None:
                return False
            with self.lock:
                self.con.execute(
                    """
                    INSERT OR REPLACE INTO dependency_maps(doc_id, version, payload, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (int(doc_id), self.version, payload, datetime.now().isoformat(timespec="seconds"))
                )
                if commit:
                    self.con.commit()
            return True
        except Exception:
            return False

    def commit(self):
        try:
            with self.lock:
                self.con.commit()
        except Exception:
            pass
