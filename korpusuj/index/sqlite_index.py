# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import zlib
import re
import os, sys, sqlite3, zlib
from pathlib import Path
from korpusuj.index.lru import LRUCache
from korpusuj.index.postings import PostingList
from korpusuj.utils.serialization import _json_zlib_loads


# KORPUSUJ_PATCH_145C7I_MARKER_GATE_SHORT_CIRCUIT_IMPORT
try:
    from korpusuj.search.diagnostics import korpusuj_diagnostics_enabled_145c1, korpusuj_verbose_diagnostics_enabled_145c1
except Exception:
    def korpusuj_diagnostics_enabled_145c1(config_obj=None):
        return False
    def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
        return False
# END KORPUSUJ_PATCH_145C7I_MARKER_GATE_SHORT_CIRCUIT_IMPORT


# KORPUSUJ_PATCH_145C7F_REGEX_SQLITE_MARKER_TAXONOMY_IMPORT
try:
    from korpusuj.search.diagnostics import korpusuj_diagnostics_enabled_145c1
except Exception:
    def korpusuj_diagnostics_enabled_145c1(config_obj=None):
        return False
# END KORPUSUJ_PATCH_145C7F_REGEX_SQLITE_MARKER_TAXONOMY_IMPORT
SEARCH_INDEX_VERSION="1.8.4-coref-mentions-sidecar"
INDEX_PROFILES={"compact":("base","orth"),"full":("base","orth","pos","upos","deprel","ner")}
DEFAULT_INDEXED_ATTRS=INDEX_PROFILES["full"]
LINGUISTIC_COLUMNS_EXCLUDED_FROM_METADATA={"Data publikacji","Tytuł","Autor","tags","Treść","token_counts","tokens","lemmas","deprels","deprel","postags","pos","full_postags","word_ids","sentence_ids","head_ids","start_ids","start_id","end_ids","end_id","ners","ner","upostags","upos","corefs","coref_mentions","srl","srls","srl_frames"}
def _engine_config():
    for modname in ("engine","__main__"):
        mod=sys.modules.get(modname); cfg=getattr(mod,"config",None) if mod is not None else None
        if isinstance(cfg,dict): return cfg
    return {}
def get_search_indexed_attrs(profile=None):
    if profile is None:
        cfg=_engine_config(); profile=os.environ.get("KORPUSUJ_INDEX_PROFILE") or cfg.get("index_profile","full")
    profile=str(profile or "full").strip().lower()
    if profile in INDEX_PROFILES: return tuple(INDEX_PROFILES[profile])
    allowed=set(INDEX_PROFILES["full"]); attrs=tuple(a.strip() for a in profile.split(",") if a.strip() in allowed)
    return attrs or DEFAULT_INDEXED_ATTRS
def search_sidecar_path(parquet_path): return str(Path(parquet_path).with_suffix(".search"))
def _search_sidecar_path(parquet_path): return search_sidecar_path(parquet_path)
def _engine_dependency_mode():
    for modname in ("engine","__main__"):
        mod=sys.modules.get(modname); fn=getattr(mod,"_get_dependency_cache_ram_mode",None) if mod is not None else None
        if callable(fn):
            try: return "all" if str(fn()).strip().lower()=="all" else "none"
            except Exception: pass
    return "all" if str(_engine_config().get("dependency_cache_ram_mode","none")).strip().lower()=="all" else "none"
def _get_search_index_cache_sizes(): return (1024,4096) if _engine_dependency_mode()=="all" else (32,64)
def _get_lazy_term_index_cache_sizes(): return (1024,32) if _engine_dependency_mode()=="all" else (64,32)

class SearchIndex:
    def __init__(self, index_path, posting_cache_size=None, doc_cache_size=None):
        self.index_path = str(index_path)
        self.con = sqlite3.connect(self.index_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        # 3e: rozmiary cache SQLite zależą od trybu RAM.
        # Oszczędny trzyma tylko mały LRU; Zrównoważony zachowuje dotychczasowy komfort;
        # Maksymalna wydajność celowo buforuje więcej dokumentów/postingów.
        if posting_cache_size is None or doc_cache_size is None:
            try:
                dyn_posting_cache_size, dyn_doc_cache_size = _get_search_index_cache_sizes()
            except Exception:
                dyn_posting_cache_size, dyn_doc_cache_size = 128, 1000
            if posting_cache_size is None:
                posting_cache_size = dyn_posting_cache_size
            if doc_cache_size is None:
                doc_cache_size = dyn_doc_cache_size
        self.posting_cache = LRUCache(max(0, int(posting_cache_size or 0)))
        self.doc_cache = LRUCache(max(0, int(doc_cache_size or 0)))
        self._meta = None
    def close(self):
        try: self.con.close()
        except Exception: pass
    def meta(self):
        if self._meta is None:
            self._meta = {r["key"]: r["value"] for r in self.con.execute("SELECT key, value FROM meta")}
        return self._meta
    @property
    def total_docs(self): return int(self.meta().get("total_docs", 0) or 0)
    @property
    def total_tokens(self): return int(self.meta().get("total_tokens", 0) or 0)
    def get_term_info(self, attr, value):
        row = self.con.execute("SELECT df, cf FROM terms WHERE attr=? AND value=?", (attr, str(value))).fetchone()
        return {"df": int(row["df"]), "cf": int(row["cf"])} if row else {"df": 0, "cf": 0}
    def get_postings(self, attr, value):
        key = (attr, str(value)); cached = self.posting_cache.get(key)
        if cached is not None: return cached
        row = self.con.execute("SELECT postings FROM terms WHERE attr=? AND value=?", key).fetchone()
        return self.posting_cache.put(key, PostingList.decode(row["postings"]) if row else {})
    def get_doc_ids_for_term(self, attr, value): return set(self.get_postings(attr, value).keys())
    def get_doc(self, doc_id):
        doc_id = int(doc_id); cached = self.doc_cache.get(doc_id)
        if cached is not None: return cached
        row = self.con.execute("SELECT * FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        if not row: return None
        try: text = zlib.decompress(row["text"]).decode("utf-8") if row["text"] else ""
        except Exception: text = ""
        doc = {"doc_id": doc_id, "metadata": _json_zlib_loads(row["metadata_json"], {}) or {}, "text": text,
               "tokens": _json_zlib_loads(row["tokens"], []) or [], "lemmas": _json_zlib_loads(row["lemmas"], []) or [],
               "start_ids": _json_zlib_loads(row["start_ids"], []) or [], "end_ids": _json_zlib_loads(row["end_ids"], []) or [],
               "sentence_ids": _json_zlib_loads(row["sentence_ids"], []) or []}
        # KORPUSUJ_MIGRATION_036E_PROFILE_DOC_ARRAYS
        try:
            _row_keys = set(row.keys())
        except Exception:
            _row_keys = set()
        for _col in ("deprels", "postags", "upostags", "full_postags", "corefs", "coref_mentions"):
            if _col in _row_keys:
                doc[_col] = _json_zlib_loads(row[_col], []) or []
            else:
                doc[_col] = []
        # END KORPUSUJ_MIGRATION_036E_PROFILE_DOC_ARRAYS
        return self.doc_cache.put(doc_id, doc)





    # KORPUSUJ_MIGRATION_036L4F4_NARROW_FULL_POSTAGS_LOADER
    def get_full_postags_036l4f4(self, doc_id):
        """Return docs.full_postags for one document using a narrow SQLite read."""
        row = self.con.execute("SELECT full_postags FROM docs WHERE doc_id=?", (int(doc_id),)).fetchone()
        if not row:
            return []
        try:
            val = row["full_postags"]
        except Exception:
            val = row[0]
        if val is None:
            return []
        if isinstance(val, memoryview):
            val = val.tobytes()
        if isinstance(val, bytes):
            raw = bytes(val)
            try:
                return json.loads(zlib.decompress(raw).decode("utf-8")) or []
            except Exception:
                pass
            try:
                return json.loads(raw.decode("utf-8")) or []
            except Exception:
                return []
        if isinstance(val, str):
            try:
                return json.loads(val) or []
            except Exception:
                return []
        if isinstance(val, list):
            return val
        return []
    def get_corefs_138i3(self, doc_id):
        """Return docs.corefs for one document using a narrow SQLite read."""
        try: row = self.con.execute("SELECT corefs FROM docs WHERE doc_id=?", (int(doc_id),)).fetchone()
        except Exception: return []
        if not row: return []
        try: val = row["corefs"]
        except Exception: val = row[0]
        if val is None: return []
        if isinstance(val, memoryview): val = val.tobytes()
        if isinstance(val, bytes):
            raw = bytes(val)
            try: return json.loads(zlib.decompress(raw).decode("utf-8")) or []
            except Exception: pass
            try: return json.loads(raw.decode("utf-8")) or []
            except Exception: return []
        if isinstance(val, str):
            try: return json.loads(val) or []
            except Exception: return []
        if isinstance(val, list): return val
        return []


    # KORPUSUJ_PATCH_174J2_COREF_MENTIONS_GETTER
    def get_coref_mentions(self, doc_id):
        """Return canonical mention records; old sidecars safely return []."""
        try:
            row = self.con.execute(
                "SELECT coref_mentions FROM docs WHERE doc_id=?", (int(doc_id),)
            ).fetchone()
        except Exception:
            return []
        if not row:
            return []
        try:
            value = row["coref_mentions"]
        except Exception:
            value = row[0]
        if value is None:
            return []
        try:
            return _json_zlib_loads(value, []) or []
        except Exception:
            return []
    # END KORPUSUJ_PATCH_174J2_COREF_MENTIONS_GETTER
    # END KORPUSUJ_MIGRATION_036L4F4_NARROW_FULL_POSTAGS_LOADER

    # KORPUSUJ_MIGRATION_036L4G42E_METADATA_RANGE_OP_PRECEDENCE
    def filter_docs_by_metadata(self, filters):
        if not filters:
            return None
        current = None
        for key, op, value, match_type in filters:
            docs = set()
            value_s = str(value)
            if op in ("<", "<=", ">", ">="):
                # KORPUSUJ_MIGRATION_036L4G42G_METADATA_RANGE_DATE_THRESHOLD_NORMALIZATION
                # Normalize date-like thresholds before lexical range comparison.
                # This makes 2020-01.01 behave like ISO 2020-01-01.
                try:
                    _m_date_threshold_036l4g42g = re.fullmatch(r"(\d{4})[-./](\d{2})[-./](\d{2})", value_s.strip())
                    if _m_date_threshold_036l4g42g:
                        value_s = "-".join(_m_date_threshold_036l4g42g.groups())
                except Exception:
                    pass
                # KORPUSUJ_MIGRATION_036L4G42F_METADATA_RANGE_WILDCARD_THRESHOLDS
                wildcard_prefix = None
                try:
                    if value_s.endswith(".*") and len(value_s) > 2:
                        wildcard_prefix = value_s[:-2]
                except Exception:
                    wildcard_prefix = None
                for doc_id, raw in self.con.execute("SELECT doc_id, value FROM doc_meta WHERE key=?", (key,)).fetchall():
                    raw_s = str(raw)
                    if wildcard_prefix is not None:
                        in_bucket = raw_s.startswith(wildcard_prefix)
                        if op == ">=":
                            ok = in_bucket or raw_s > wildcard_prefix
                        elif op == ">":
                            ok = (not in_bucket) and raw_s > wildcard_prefix
                        elif op == "<=":
                            ok = in_bucket or raw_s < wildcard_prefix
                        else:
                            ok = (not in_bucket) and raw_s < wildcard_prefix
                    elif op == "<":
                        ok = raw_s < value_s
                    elif op == "<=":
                        ok = raw_s <= value_s
                    elif op == ">":
                        ok = raw_s > value_s
                    else:
                        ok = raw_s >= value_s
                    if ok:
                        docs.add(int(doc_id))
            elif match_type == "exact" and op in ("=", "!="):
                docs = {int(r[0]) for r in self.con.execute("SELECT doc_id FROM doc_meta WHERE key=? AND value=?", (key, value_s)).fetchall()}
                if op == "!=":
                    docs = set(range(self.total_docs)) - docs
            else:
                for doc_id, raw in self.con.execute("SELECT doc_id, value FROM doc_meta WHERE key=?", (key,)).fetchall():
                    raw_s = str(raw)
                    ok = False
                    if match_type in ("regex", "regex_search"):
                        try:
                            ok = re.search(value_s, raw_s, re.IGNORECASE) is not None
                        except re.error:
                            ok = False
                    else:
                        if op == "=":
                            ok = raw_s == value_s
                        elif op == "!=":
                            ok = raw_s != value_s
                    if ok:
                        docs.add(int(doc_id))
            current = docs if current is None else current & docs
        return current


    # KORPUSUJ_MIGRATION_036L4G8_BATCH_GET_DOCS
    def get_docs_many_036l4g8(self, doc_ids, chunk_size=800):
        """Batch-load docs by doc_id with the same decoding contract as get_doc(...)."""
        ids = []
        seen = set()
        for x in doc_ids or []:
            try:
                i = int(x)
            except Exception:
                continue
            if i not in seen:
                seen.add(i)
                ids.append(i)
        if not ids:
            return {}

        out = {}
        missing = []
        for doc_id in ids:
            cached = self.doc_cache.get(doc_id)
            if cached is not None:
                out[doc_id] = cached
            else:
                missing.append(doc_id)
        if not missing:
            return out

        try:
            chunk_size = max(1, int(chunk_size or 800))
        except Exception:
            chunk_size = 800
        for start in range(0, len(missing), chunk_size):
            chunk = missing[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT * FROM docs WHERE doc_id IN ({placeholders})"
            for row in self.con.execute(query, chunk).fetchall():
                try:
                    doc_id = int(row["doc_id"])
                except Exception:
                    continue
                try:
                    text = zlib.decompress(row["text"]).decode("utf-8") if row["text"] else ""
                except Exception:
                    text = ""
                doc = {
                    "doc_id": doc_id,
                    "metadata": _json_zlib_loads(row["metadata_json"], {}) or {},
                    "text": text,
                    "tokens": _json_zlib_loads(row["tokens"], []) or [],
                    "lemmas": _json_zlib_loads(row["lemmas"], []) or [],
                    "start_ids": _json_zlib_loads(row["start_ids"], []) or [],
                    "end_ids": _json_zlib_loads(row["end_ids"], []) or [],
                    "sentence_ids": _json_zlib_loads(row["sentence_ids"], []) or [],
                }
                try:
                    _row_keys = set(row.keys())
                except Exception:
                    _row_keys = set()
                for _col in ("deprels", "postags", "upostags", "full_postags", "corefs", "coref_mentions"):
                    doc[_col] = (_json_zlib_loads(row[_col], []) or []) if _col in _row_keys else []
                out[doc_id] = self.doc_cache.put(doc_id, doc)
        return out


    def get_docs_many_for_result_table(self, doc_ids, chunk_size=800):
        """Batch-load the minimal document data needed for concordance result-table rows.
        
        The loader omits full document text and heavy linguistic arrays. SearchCursor._result can build table contexts from tokens while full text remains lazy through full_text_ref. The returned mapping provides doc_id, metadata, an empty text value, tokens, lemmas, start_ids and end_ids. Metadata is decoded fully for row details and additional metadata fields.
        """
        ids = []
        seen = set()
        for x in doc_ids or []:
            try:
                i = int(x)
            except Exception:
                continue
            if i not in seen:
                seen.add(i)
                ids.append(i)
        out = {}
        if not ids:
            return out
        try:
            chunk_size = max(1, int(chunk_size or 800))
        except Exception:
            chunk_size = 800
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT doc_id, metadata_json, tokens, lemmas, start_ids, end_ids "
                f"FROM docs WHERE doc_id IN ({placeholders})"
            )
            for row in self.con.execute(query, chunk).fetchall():
                try:
                    doc_id = int(row["doc_id"])
                except Exception:
                    continue
                toks = _json_zlib_loads(row["tokens"], []) or []
                doc = {
                    "doc_id": doc_id,
                    "metadata": _json_zlib_loads(row["metadata_json"], {}) or {},
                    "text": "",
                    "tokens": toks,
                    "lemmas": _json_zlib_loads(row["lemmas"], []) or toks,
                    "start_ids": _json_zlib_loads(row["start_ids"], []) or [],
                    "end_ids": _json_zlib_loads(row["end_ids"], []) or [],
                    "_partial_result_table_doc": True,
                }
                out[doc_id] = doc
        return out

    def get_docs_many(self, doc_ids, chunk_size=800):
        return self.get_docs_many_036l4g8(doc_ids, chunk_size=chunk_size)



# KORPUSUJ_MIGRATION_036L4G37C_REGEX_SQLITE_CONFIG_ROUTE
# Config-driven regex support over the .search SQLite terms vocabulary.
import re as _re_036l4g37c
import json as _json_036l4g37c
import logging as _logging_036l4g37c

_REGEX_SQLITE_META_CHARS_036L4G37C = set('.^$*+?[]|\\()')
_REGEX_SQLITE_INDEXED_ATTRS_036L4G37C = {"base", "orth", "pos", "upos", "deprel", "ner"}
_REGEX_SQLITE_DEFAULTS_036L4G37C = {
    "regex_sqlite_route": True,
    "regex_sqlite_enabled": True,
    "regex_sqlite_debug": False,
    "regex_sqlite_max_terms": 5000,
    "regex_sqlite_max_cf": 2000000,
    "regex_sqlite_match_mode": "fullmatch",
}

class RegexSQLiteTooBroadError(RuntimeError):
    pass

class RegexSQLiteCompileError(RuntimeError):
    pass

def _regex_sqlite_config_036l4g37c():
    cfg = {}
    try:
        base_cfg = globals().get("_engine_config")
        if callable(base_cfg):
            got = base_cfg()
            if isinstance(got, dict):
                cfg.update(got)
    except Exception:
        pass
    if not cfg:
        try:
            p = Path(__file__).resolve()
            root = p.parents[2]
            config_path = root / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    got = _json_036l4g37c.load(f)
                if isinstance(got, dict):
                    cfg.update(got)
        except Exception:
            pass
    out = dict(_REGEX_SQLITE_DEFAULTS_036L4G37C)
    out.update(cfg or {})
    return out

def _regex_sqlite_bool_036l4g37c(name, default=None):
    cfg = _regex_sqlite_config_036l4g37c()
    if default is None:
        default = _REGEX_SQLITE_DEFAULTS_036L4G37C.get(name, False)
    val = cfg.get(name, default)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "tak", "on"}
    return bool(val)

def _regex_sqlite_int_036l4g37c(name, default=None):
    cfg = _regex_sqlite_config_036l4g37c()
    if default is None:
        default = _REGEX_SQLITE_DEFAULTS_036L4G37C.get(name, 0)
    try:
        return int(cfg.get(name, default))
    except Exception:
        return int(default)

def _regex_sqlite_enabled_036l4g37c():
    return _regex_sqlite_bool_036l4g37c("regex_sqlite_enabled", True)

def _regex_sqlite_debug_036l4g37c():
    return _regex_sqlite_bool_036l4g37c("regex_sqlite_debug", False)

def _regex_sqlite_log_036l4g37c(event, **data):
    if not _regex_sqlite_debug_036l4g37c():
        return
    try:
        if korpusuj_diagnostics_enabled_145c1():
            _logging_036l4g37c.info("[DIAG regex.sqlite.route] event=%r data=%r", event, data)
    except Exception:
        pass

def _regex_sqlite_strip_legacy_tilde_036l4g37c(pattern):
    pattern = str(pattern or "")
    if pattern.startswith("~") and len(pattern) > 1:
        return pattern[1:]
    return pattern

def _regex_sqlite_is_regex_value_036l4g37c(value):
    try:
        value = str(value or "")
        if value.startswith("~") and len(value) > 1:
            return True
        return any(ch in _REGEX_SQLITE_META_CHARS_036L4G37C for ch in value)
    except Exception:
        return False

def _regex_sqlite_is_supported_attr_036l4g37c(attr):
    try:
        return str(attr or "").strip().lower() in _REGEX_SQLITE_INDEXED_ATTRS_036L4G37C
    except Exception:
        return False

def _regex_sqlite_match_mode_036l4g37c():
    cfg = _regex_sqlite_config_036l4g37c()
    mode = str(cfg.get("regex_sqlite_match_mode", "fullmatch") or "fullmatch").strip().lower()
    return mode if mode in {"search", "fullmatch", "match"} else "search"

def _regex_sqlite_compile_036l4g37c(pattern):
    pattern = _regex_sqlite_strip_legacy_tilde_036l4g37c(pattern)
    try:
        return _re_036l4g37c.compile(pattern)
    except Exception as e:
        raise RegexSQLiteCompileError(str(e))

def _regex_sqlite_matches_036l4g37c(rx, value):
    # 036L4G37I: ordinary token regex must be full-token match.
    # Legacy contains semantics are handled separately for patterns starting with ~
    # inside _regex_sqlite_find_terms_036l4g37c().
    value = str(value or "")
    return rx.fullmatch(value) is not None

def _regex_sqlite_find_terms_036l4g37c(self, attr, pattern):
    attr = str(attr)
    raw_pattern_036l4g37g = str(pattern or "")
    tilde_contains_036l4g37g = raw_pattern_036l4g37g.startswith("~") and len(raw_pattern_036l4g37g) > 1
    pattern = _regex_sqlite_strip_legacy_tilde_036l4g37c(pattern)
    if not _regex_sqlite_is_supported_attr_036l4g37c(attr):
        raise RegexSQLiteCompileError(f"unsupported regex attr for sqlite route: {attr!r}")
    rx = _regex_sqlite_compile_036l4g37c(pattern)
    max_terms = _regex_sqlite_int_036l4g37c("regex_sqlite_max_terms", 5000)
    max_cf = _regex_sqlite_int_036l4g37c("regex_sqlite_max_cf", 2000000)
    matched = []
    total_cf = 0
    scanned = 0
    for row in self.con.execute("SELECT value, df, cf FROM terms WHERE attr=?", (attr,)):
        scanned += 1
        value = str(row["value"])
        if (rx.search(value) is not None if tilde_contains_036l4g37g else _regex_sqlite_matches_036l4g37c(rx, value)):
            df = int(row["df"] or 0)
            cf = int(row["cf"] or 0)
            matched.append((value, df, cf))
            total_cf += cf
            if len(matched) > max_terms:
                raise RegexSQLiteTooBroadError(f"regex matched too many terms for {attr}: {len(matched)} > {max_terms}")
            if total_cf > max_cf:
                raise RegexSQLiteTooBroadError(f"regex aggregate cf too large for {attr}: {total_cf} > {max_cf}")
    _regex_sqlite_log_036l4g37c("terms_ready", attr=attr, pattern=pattern, scanned=scanned, matched=len(matched), total_cf=total_cf)
    return matched

def _regex_sqlite_merge_postings_036l4g37c(posting_lists):
    merged = {}
    for postings in posting_lists:
        try:
            items = postings.items()
        except Exception:
            continue
        for doc_id, positions in items:
            try:
                doc_id_i = int(doc_id)
            except Exception:
                doc_id_i = doc_id
            bucket = merged.setdefault(doc_id_i, [])
            try:
                bucket.extend(list(positions))
            except Exception:
                try:
                    bucket.append(int(positions))
                except Exception:
                    pass
    for doc_id, positions in list(merged.items()):
        try:
            merged[doc_id] = sorted(set(int(p) for p in positions))
        except Exception:
            merged[doc_id] = sorted(set(positions))
    return merged

def _regex_sqlite_get_postings_036l4g37c(self, attr, pattern):
    terms = _regex_sqlite_find_terms_036l4g37c(self, attr, pattern)
    posting_lists = [self._get_postings_exact_036l4g37c(attr, value) for value, _df, _cf in terms]
    merged = _regex_sqlite_merge_postings_036l4g37c(posting_lists)
    _regex_sqlite_log_036l4g37c("postings_ready", attr=str(attr), pattern=str(pattern), terms=len(terms), docs=len(merged))
    return merged

def _regex_sqlite_get_term_info_036l4g37c(self, attr, pattern):
    terms = _regex_sqlite_find_terms_036l4g37c(self, attr, pattern)
    doc_ids = set()
    cf_total = 0
    for value, _df, cf in terms:
        cf_total += int(cf or 0)
        try:
            doc_ids.update(self._get_postings_exact_036l4g37c(attr, value).keys())
        except Exception:
            pass
    return {"df": len(doc_ids), "cf": int(cf_total)}

def _install_regex_sqlite_index_route_036l4g37c():
    cls = globals().get("SearchIndex")
    if cls is None or getattr(cls, "_regex_sqlite_036l4g37c_installed", False):
        return
    cls._get_postings_exact_036l4g37c = cls.get_postings
    cls._get_term_info_exact_036l4g37c = cls.get_term_info

    def get_postings(self, attr, value):
        if (_regex_sqlite_enabled_036l4g37c()
            and _regex_sqlite_is_supported_attr_036l4g37c(attr)
            and _regex_sqlite_is_regex_value_036l4g37c(value)):
            key = ("__regex_036l4g37c__", str(attr), str(value), _regex_sqlite_match_mode_036l4g37c())
            cached = self.posting_cache.get(key)
            if cached is not None:
                return cached
            return self.posting_cache.put(key, _regex_sqlite_get_postings_036l4g37c(self, attr, value))
        return self._get_postings_exact_036l4g37c(attr, value)

    def get_term_info(self, attr, value):
        if (_regex_sqlite_enabled_036l4g37c()
            and _regex_sqlite_is_supported_attr_036l4g37c(attr)
            and _regex_sqlite_is_regex_value_036l4g37c(value)):
            return _regex_sqlite_get_term_info_036l4g37c(self, attr, value)
        return self._get_term_info_exact_036l4g37c(attr, value)

    cls.get_postings = get_postings
    cls.get_term_info = get_term_info
    cls.find_terms_regex_036l4g37c = _regex_sqlite_find_terms_036l4g37c
    cls.get_postings_regex_036l4g37c = _regex_sqlite_get_postings_036l4g37c
    cls._regex_sqlite_036l4g37c_installed = True

_install_regex_sqlite_index_route_036l4g37c()
# KORPUSUJ_MIGRATION_036L4G37D_REGEX_SQLITE_POSTINGLIST_DICT_FIX
# KORPUSUJ_MIGRATION_036L4G37G_TILDE_CONTAINS_FIX
# KORPUSUJ_MIGRATION_036L4G37I_FORCE_FULLMATCH_NON_TILDE_REGEX
# END KORPUSUJ_MIGRATION_036L4G37C_REGEX_SQLITE_CONFIG_ROUTE

class LazyTermIndex:
# KORPUSUJ_MIGRATION_036L4G37H3_LAZYTERMINDEX_EXACT_GET_FIX
    def __init__(self, index_path, attr): self.index_path = index_path; self.attr = attr; self._idx = None
    def _open(self):
        if self._idx is None:
            try:
                posting_cache_size, doc_cache_size = _get_lazy_term_index_cache_sizes()
            except Exception:
                posting_cache_size, doc_cache_size = 64, 32
            self._idx = SearchIndex(self.index_path, posting_cache_size, doc_cache_size)
        return self._idx
    def get(self, value, default=None):
        idx = self._open()
        try:
            exact_postings = getattr(idx, "_get_postings_exact_036l4g37c", None)
            if callable(exact_postings):
                docs = set(exact_postings(self.attr, value).keys())
            else:
                docs = idx.get_doc_ids_for_term(self.attr, value)
        except Exception:
            docs = set()
        return docs if docs else (default if default is not None else set())
    def __contains__(self, value):
        idx = self._open()
        try:
            exact_info = getattr(idx, "_get_term_info_exact_036l4g37c", None)
            if callable(exact_info):
                return exact_info(self.attr, value)["df"] > 0
            return idx.get_term_info(self.attr, value)["df"] > 0
        except Exception:
            return False
# KORPUSUJ_MIGRATION_036L4G9_NARROW_RESULT_DOCS_LOADER
def _install_searchindex_narrow_result_docs_loader_036l4g9():
    """Attach a narrow batch loader for SearchCursor result materialization.

    Unlike get_doc(...), this loader decodes only fields used by SearchCursor._result:
    metadata, text, tokens, lemmas, start_ids, end_ids. It intentionally does not
    populate SearchIndex.doc_cache, because that cache is shared with full get_doc
    callers that may expect sentence_ids/deprels/postags/full_postags.
    """
    import time as _time_036l4g9
    import logging as _logging_036l4g9

    cls = globals().get("SearchIndex")
    if cls is None or getattr(cls, "_narrow_result_docs_loader_036l4g9_installed", False):
        return

    def get_docs_many_for_results_036l4g9(self, doc_ids, chunk_size=800):
        t0 = _time_036l4g9.perf_counter()
        ids = []
        seen = set()
        for x in doc_ids or []:
            try:
                i = int(x)
            except Exception:
                continue
            if i not in seen:
                seen.add(i)
                ids.append(i)
        out = {}
        if not ids:
            return out
        try:
            chunk_size = max(1, int(chunk_size or 800))
        except Exception:
            chunk_size = 800
        rows_seen = 0
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT doc_id, text, metadata_json, tokens, lemmas, start_ids, end_ids "
                f"FROM docs WHERE doc_id IN ({placeholders})"
            )
            for row in self.con.execute(query, chunk).fetchall():
                rows_seen += 1
                try:
                    doc_id = int(row["doc_id"])
                except Exception:
                    continue
                try:
                    text = zlib.decompress(row["text"]).decode("utf-8") if row["text"] else ""
                except Exception:
                    text = ""
                toks = _json_zlib_loads(row["tokens"], []) or []
                doc = {
                    "doc_id": doc_id,
                    "metadata": _json_zlib_loads(row["metadata_json"], {}) or {},
                    "text": text,
                    "tokens": toks,
                    "lemmas": _json_zlib_loads(row["lemmas"], []) or toks,
                    "start_ids": _json_zlib_loads(row["start_ids"], []) or [],
                    "end_ids": _json_zlib_loads(row["end_ids"], []) or [],
                }
                out[doc_id] = doc
        try:
            korpusuj_verbose_diagnostics_enabled_145c1() and _logging_036l4g9.info(
                "[DIAG perf.search.narrow_docs] requested=%s loaded=%s elapsed=%.6fs chunk_size=%s",
                len(ids), rows_seen, _time_036l4g9.perf_counter() - t0, chunk_size
            )
        except Exception:
            pass
        return out

    setattr(cls, "get_docs_many_for_results_036l4g9", get_docs_many_for_results_036l4g9)
    cls._narrow_result_docs_loader_036l4g9_installed = True


_install_searchindex_narrow_result_docs_loader_036l4g9()
# END KORPUSUJ_MIGRATION_036L4G9_NARROW_RESULT_DOCS_LOADER


# KORPUSUJ_MIGRATION_PATCH_96_REGEX_SQLITE_NO_TERM_LIMIT
# Remove regex term/cf caps for SQLite regex expansion. The regex route may still
# fail on invalid regex/unsupported attrs, but it must not raise TooBroad solely
# because many vocabulary terms or many occurrences match a valid pattern.
def _install_regex_sqlite_no_term_limit_096():
    import logging as _logging_096

    cls = globals().get("SearchIndex")
    if cls is None or getattr(cls, "_regex_sqlite_no_term_limit_096_installed", False):
        return


    def _find_terms_no_limit_096(self, attr, pattern):
        attr = str(attr)
        raw_pattern = str(pattern or "")
        tilde_contains = raw_pattern.startswith("~") and len(raw_pattern) > 1
        pattern_stripped = _regex_sqlite_strip_legacy_tilde_036l4g37c(pattern)
        if not _regex_sqlite_is_supported_attr_036l4g37c(attr):
            raise RegexSQLiteCompileError(f"unsupported regex attr for sqlite route: {attr!r}")
        rx = _regex_sqlite_compile_036l4g37c(pattern_stripped)
        matched = []
        total_cf = 0
        scanned = 0
        for row in self.con.execute("SELECT value, df, cf FROM terms WHERE attr=?", (attr,)):
            scanned += 1
            value = str(row["value"])
            ok = (rx.search(value) is not None) if tilde_contains else _regex_sqlite_matches_036l4g37c(rx, value)
            if not ok:
                continue
            df = int(row["df"] or 0)
            cf = int(row["cf"] or 0)
            matched.append((value, df, cf))
            total_cf += cf
        try:
            _regex_sqlite_log_036l4g37c(
                "terms_ready_no_limit_096",
                attr=attr,
                pattern=pattern_stripped,
                scanned=scanned,
                matched=len(matched),
                total_cf=total_cf,
            )
        except Exception:
            pass
        return matched

    def _merge_postings_no_limit_096(posting_lists):
        try:
            return _regex_sqlite_merge_postings_036l4g37c(posting_lists)
        except Exception:
            merged = {}
            for postings in posting_lists:
                try:
                    items = postings.items()
                except Exception:
                    continue
                for doc_id, positions in items:
                    try:
                        doc_id_i = int(doc_id)
                    except Exception:
                        doc_id_i = doc_id
                    bucket = merged.setdefault(doc_id_i, [])
                    try:
                        bucket.extend(list(positions))
                    except Exception:
                        try:
                            bucket.append(int(positions))
                        except Exception:
                            pass
            for doc_id, positions in list(merged.items()):
                try:
                    merged[doc_id] = sorted(set(int(p) for p in positions))
                except Exception:
                    merged[doc_id] = sorted(set(positions))
            return merged

    def _get_postings_regex_no_limit_096(self, attr, pattern):
        terms = _find_terms_no_limit_096(self, attr, pattern)
        exact = getattr(self, "_get_postings_exact_036l4g37c", None)
        if not callable(exact):
            exact = cls.__dict__.get("get_postings")
        posting_lists = [exact(self, attr, value) if getattr(exact, "__self__", None) is None else exact(attr, value) for value, _df, _cf in terms]
        merged = _merge_postings_no_limit_096(posting_lists)
        try:
            _regex_sqlite_log_036l4g37c("postings_ready_no_limit_096", attr=str(attr), pattern=str(pattern), terms=len(terms), docs=len(merged))
        except Exception:
            pass
        return merged

    def _get_term_info_regex_no_limit_096(self, attr, pattern):
        terms = _find_terms_no_limit_096(self, attr, pattern)
        doc_ids = set()
        cf_total = 0
        exact = getattr(self, "_get_postings_exact_036l4g37c", None)
        if not callable(exact):
            exact = cls.__dict__.get("get_postings")
        for value, _df, cf in terms:
            cf_total += int(cf or 0)
            try:
                postings = exact(self, attr, value) if getattr(exact, "__self__", None) is None else exact(attr, value)
                doc_ids.update(postings.keys())
            except Exception:
                pass
        return {"df": len(doc_ids), "cf": int(cf_total)}

    def _get_postings_route_no_limit_096(self, attr, value):
        if _regex_sqlite_enabled_036l4g37c() and _regex_sqlite_is_supported_attr_036l4g37c(attr) and _regex_sqlite_is_regex_value_036l4g37c(value):
            return _get_postings_regex_no_limit_096(self, attr, value)
        exact = getattr(self, "_get_postings_exact_036l4g37c", None)
        if callable(exact):
            return exact(attr, value)
        return cls.__dict__["get_postings"](self, attr, value)

    def _get_term_info_route_no_limit_096(self, attr, value):
        if _regex_sqlite_enabled_036l4g37c() and _regex_sqlite_is_supported_attr_036l4g37c(attr) and _regex_sqlite_is_regex_value_036l4g37c(value):
            return _get_term_info_regex_no_limit_096(self, attr, value)
        exact = getattr(self, "_get_term_info_exact_036l4g37c", None)
        if callable(exact):
            return exact(attr, value)
        return cls.__dict__["get_term_info"](self, attr, value)

    # Replace global helper used by existing regex functions and replace public
    # regex APIs so direct calls and routed get_postings/get_term_info agree.
    globals()["_regex_sqlite_find_terms_036l4g37c"] = _find_terms_no_limit_096
    globals()["_regex_sqlite_get_postings_036l4g37c"] = _get_postings_regex_no_limit_096
    globals()["_regex_sqlite_get_term_info_036l4g37c"] = _get_term_info_regex_no_limit_096

    cls.find_terms_regex_036l4g37c = _find_terms_no_limit_096
    cls.get_postings_regex_036l4g37c = _get_postings_regex_no_limit_096
    cls.get_postings = _get_postings_route_no_limit_096
    cls.get_term_info = _get_term_info_route_no_limit_096
    cls._regex_sqlite_no_term_limit_096_installed = True

    try:
        if korpusuj_diagnostics_enabled_145c1():
            _logging_096.info("[DIAG regex.sqlite.install] installed no term/cf limit regex route")
    except Exception:
        pass

_install_regex_sqlite_no_term_limit_096()
# END KORPUSUJ_MIGRATION_PATCH_96_REGEX_SQLITE_NO_TERM_LIMIT

