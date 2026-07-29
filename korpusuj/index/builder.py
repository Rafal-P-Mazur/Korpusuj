# -*- coding: utf-8 -*-
"""Build the derived SQLite search index from a canonical Parquet corpus."""
from __future__ import annotations
import json, os, re, sqlite3, zlib, logging
from collections import defaultdict
import pandas as pd
import pyarrow.parquet as pq
from korpusuj.index.postings import PostingList
from korpusuj.index.sqlite_index import SEARCH_INDEX_VERSION,LINGUISTIC_COLUMNS_EXCLUDED_FROM_METADATA,get_search_indexed_attrs,search_sidecar_path
from korpusuj.index.status import inspect_search_index
from korpusuj.utils.serialization import _json_zlib_dumps,_as_plain_list,_safe_scalar
from datetime import datetime

class SearchIndexBuilder:
    """Stabilny builder indeksu: profil atrybutów, bezpieczne ograniczenie kolumn, większe batche.

    HOTFIX 1.8.2: wraca sprawdzony format postingów JSON+zlib, bo wersja 1.8.0
    z binarnymi postingami mogła budować pusty/niezgodny indeks w części środowisk.
    """
    @staticmethod
    def is_fresh(parquet_path, index_path=None, indexed_attrs=None):
        indexed_attrs = tuple(indexed_attrs or get_search_indexed_attrs())
        return inspect_search_index(
            parquet_path, index_path=index_path, indexed_attrs=indexed_attrs, check_integrity=False
        )["status"] == "fresh"

    def _select_parquet_columns(self, pf, indexed_attrs):
        """Wybiera tylko potrzebne kolumny; jeśli schemat wygląda nietypowo, wraca do odczytu wszystkich."""
        try:
            available = list(getattr(pf, "schema_arrow", pf.schema).names)
        except Exception:
            try:
                available = list(pf.schema.names)
            except Exception:
                return None
        available_set = set(available)
        # Bez tokens/lemmas nie ma sensu ograniczać kolumn — czytamy pełny plik dla bezpieczeństwa.
        if "tokens" not in available_set and "lemmas" not in available_set:
            return None
        ordered = []
        def add(col):
            if col in available_set and col not in ordered:
                ordered.append(col)
        for col in (
            "tokens", "lemmas", "Treść", "Data publikacji", "Tytuł", "Autor",
            "start_ids", "start_id", "end_ids", "end_id", "sentence_ids", "sentence_id",
            "deprels", "deprel", "postags", "pos", "upostags", "upos", "full_postags", "corefs", "coref", "coref_mentions",
        ):
            add(col)
        for col in available:
            if col not in LINGUISTIC_COLUMNS_EXCLUDED_FROM_METADATA:
                add(col)
        attr_cols = {"pos": ("pos", "postags"), "upos": ("upos", "upostags"), "deprel": ("deprel", "deprels"), "ner": ("ner", "ners")}
        for attr in indexed_attrs:
            for col in attr_cols.get(attr, ()):
                add(col)
        return ordered or None

    def build_from_parquet(self, parquet_path, index_path=None, batch_docs=5000, indexed_attrs=None, progress_callback=None):
        """Build a search index from Parquet batches and report progress when requested."""
        parquet_path = str(parquet_path)
        index_path = index_path or search_sidecar_path(parquet_path)
        indexed_attrs = tuple(indexed_attrs or get_search_indexed_attrs())
        batch_docs = max(1, int(batch_docs or 5000))
        tmp_path = index_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        con = sqlite3.connect(tmp_path)
        try:
            self._create_schema(con)
            pf = pq.ParquetFile(parquet_path)
            read_columns = self._select_parquet_columns(pf, indexed_attrs)
            total_expected_docs = int(getattr(pf.metadata, "num_rows", 0) or 0)
            total_docs = 0
            total_tokens = 0
            monthly_counts = defaultdict(int)
            doc_rows, meta_rows, stage_rows, stats_rows = [], [], [], []
            attr_cols = {"pos": ("pos", "postags"), "upos": ("upos", "upostags"), "deprel": ("deprel", "deprels"), "ner": ("ner", "ners")}
            for batch_no, record_batch in enumerate(pf.iter_batches(batch_size=batch_docs, columns=read_columns)):
                df_part = record_batch.to_pandas()
                for row in df_part.to_dict("records"):
                    doc_id = total_docs
                    total_docs += 1
                    tokens = [str(x) for x in _as_plain_list(row.get("tokens", []))]
                    lemmas = [str(x) for x in _as_plain_list(row.get("lemmas", []))]
                    if len(lemmas) != len(tokens):
                        lemmas = tokens[:]
                    start_ids = _as_plain_list(row.get("start_ids", row.get("start_id", [])))
                    end_ids = _as_plain_list(row.get("end_ids", row.get("end_id", [])))
                    sentence_ids = _as_plain_list(row.get("sentence_ids", row.get("sentence_id", [])))
                    # KORPUSUJ_MIGRATION_036E_PROFILE_DOC_ARRAYS
                    def _profile_array_036e(*names):
                        value = []
                        for _name in names:
                            value = _as_plain_list(row.get(_name, []))
                            if value:
                                break
                        value = [str(x) for x in value]
                        if len(value) != len(tokens):
                            value = [""] * len(tokens)
                        return value
                    deprels = _profile_array_036e("deprels", "deprel")
                    postags = _profile_array_036e("postags", "pos")
                    upostags = _profile_array_036e("upostags", "upos")
                    full_postags = _profile_array_036e("full_postags")
                    # --- COREFS_NESTED_TOKEN_ARRAY_STORAGE_174K2 ---
                    # ``corefs`` is token-aligned, but every token value is itself a
                    # list of zero or more Head-/Part- labels.  The generic profile
                    # helper is scalar-oriented and stringifies nested arrays.
                    _raw_corefs_174k2 = row.get("corefs", [])
                    if _raw_corefs_174k2 is None:
                        _raw_corefs_174k2 = []
                    elif hasattr(_raw_corefs_174k2, "tolist"):
                        _raw_corefs_174k2 = _raw_corefs_174k2.tolist()
                    elif isinstance(_raw_corefs_174k2, tuple):
                        _raw_corefs_174k2 = list(_raw_corefs_174k2)
                    elif not isinstance(_raw_corefs_174k2, list):
                        raise RuntimeError(
                            "174k2 invalid document corefs container: "
                            f"doc_id={doc_id}, type={type(_raw_corefs_174k2).__name__}"
                        )

                    if len(_raw_corefs_174k2) != len(tokens):
                        raise RuntimeError(
                            "174k2 token/corefs length mismatch: "
                            f"doc_id={doc_id}, tokens={len(tokens)}, "
                            f"corefs={len(_raw_corefs_174k2)}"
                        )

                    corefs = []
                    for _coref_pos_174k2, _raw_labels_174k2 in enumerate(_raw_corefs_174k2):
                        if _raw_labels_174k2 is None:
                            _labels_174k2 = []
                        elif hasattr(_raw_labels_174k2, "tolist"):
                            _labels_174k2 = _raw_labels_174k2.tolist()
                        elif isinstance(_raw_labels_174k2, tuple):
                            _labels_174k2 = list(_raw_labels_174k2)
                        elif isinstance(_raw_labels_174k2, list):
                            _labels_174k2 = list(_raw_labels_174k2)
                        elif isinstance(_raw_labels_174k2, str):
                            # Accept only a genuinely scalar historical label.  Do
                            # not parse stringified lists and never use eval.
                            if _raw_labels_174k2 == "":
                                _labels_174k2 = []
                            elif _raw_labels_174k2.startswith(("Head-", "Part-")):
                                _labels_174k2 = [_raw_labels_174k2]
                            else:
                                raise RuntimeError(
                                    "174k2 stringified or unsupported corefs token value: "
                                    f"doc_id={doc_id}, token={_coref_pos_174k2}, "
                                    f"value={_raw_labels_174k2!r}"
                                )
                        else:
                            raise RuntimeError(
                                "174k2 invalid corefs token value: "
                                f"doc_id={doc_id}, token={_coref_pos_174k2}, "
                                f"type={type(_raw_labels_174k2).__name__}"
                            )

                        if not isinstance(_labels_174k2, list):
                            raise RuntimeError(
                                "174k2 nested corefs conversion did not produce list: "
                                f"doc_id={doc_id}, token={_coref_pos_174k2}, "
                                f"type={type(_labels_174k2).__name__}"
                            )

                        _normalized_labels_174k2 = []
                        for _label_174k2 in _labels_174k2:
                            if not isinstance(_label_174k2, str):
                                raise RuntimeError(
                                    "174k2 non-string coreference label: "
                                    f"doc_id={doc_id}, token={_coref_pos_174k2}, "
                                    f"type={type(_label_174k2).__name__}, "
                                    f"value={_label_174k2!r}"
                                )
                            if not _label_174k2.startswith(("Head-", "Part-")):
                                raise RuntimeError(
                                    "174k2 unsupported coreference label: "
                                    f"doc_id={doc_id}, token={_coref_pos_174k2}, "
                                    f"value={_label_174k2!r}"
                                )
                            _normalized_labels_174k2.append(_label_174k2)
                        corefs.append(_normalized_labels_174k2)
                    # --- END COREFS_NESTED_TOKEN_ARRAY_STORAGE_174K2 ---
                    # KORPUSUJ_PATCH_174J2B_COREF_MENTIONS_DOCUMENT_ARRAY_STORAGE
                    # Unlike corefs/deprels/POS arrays, coref_mentions is a
                    # document-level list of structs and must never be padded
                    # or truncated to token_count.
                    # KORPUSUJ_PATCH_174J2E_COREF_MENTIONS_LOCAL_TOKEN_COUNT
                    # ``tokens`` has already been materialized at this point,
                    # while the original ``token_count`` assignment occurs
                    # later in the builder. Keep this validation block local.
                    _token_count_174j2e = len(tokens)
                    # END KORPUSUJ_PATCH_174J2E_COREF_MENTIONS_LOCAL_TOKEN_COUNT
                    _raw_coref_mentions_174j2b = row.get("coref_mentions", [])
                    if _raw_coref_mentions_174j2b is None:
                        _raw_coref_mentions_174j2b = []
                    if hasattr(_raw_coref_mentions_174j2b, "as_py"):
                        _raw_coref_mentions_174j2b = _raw_coref_mentions_174j2b.as_py()
                    elif hasattr(_raw_coref_mentions_174j2b, "tolist"):
                        _raw_coref_mentions_174j2b = _raw_coref_mentions_174j2b.tolist()
                    if isinstance(_raw_coref_mentions_174j2b, tuple):
                        _raw_coref_mentions_174j2b = list(_raw_coref_mentions_174j2b)
                    if not isinstance(_raw_coref_mentions_174j2b, list):
                        raise TypeError(
                            "coref_mentions must be a document-level list, got "
                            + type(_raw_coref_mentions_174j2b).__name__
                        )
                    coref_mentions = []
                    for _mention_174j2b in _raw_coref_mentions_174j2b:
                        if hasattr(_mention_174j2b, "as_py"):
                            _mention_174j2b = _mention_174j2b.as_py()
                        if not isinstance(_mention_174j2b, dict):
                            try:
                                _mention_174j2b = dict(_mention_174j2b)
                            except Exception as _exc_174j2b:
                                raise TypeError(
                                    "coref_mentions element must be a mapping, got "
                                    + type(_mention_174j2b).__name__
                                ) from _exc_174j2b
                        _missing_174j2b = [
                            _field_174j2b
                            for _field_174j2b in ("cluster_id", "mention_id", "start", "end", "head")
                            if _field_174j2b not in _mention_174j2b
                        ]
                        if _missing_174j2b:
                            raise ValueError(
                                "coref_mentions element lacks fields: "
                                + ", ".join(_missing_174j2b)
                            )
                        _canonical_mention_174j2b = {
                            "cluster_id": str(_mention_174j2b["cluster_id"]),
                            "mention_id": int(_mention_174j2b["mention_id"]),
                            "start": int(_mention_174j2b["start"]),
                            "end": int(_mention_174j2b["end"]),
                            "head": int(_mention_174j2b["head"]),
                        }
                        if not (
                            0 <= _canonical_mention_174j2b["start"]
                            < _canonical_mention_174j2b["end"]
                            <= _token_count_174j2e
                        ):
                            raise ValueError(
                                "coref_mentions span outside document token coordinates: "
                                + repr(_canonical_mention_174j2b)
                            )
                        if not (
                            _canonical_mention_174j2b["start"]
                            <= _canonical_mention_174j2b["head"]
                            < _canonical_mention_174j2b["end"]
                        ):
                            raise ValueError(
                                "coref_mentions head outside mention span: "
                                + repr(_canonical_mention_174j2b)
                            )
                        coref_mentions.append(_canonical_mention_174j2b)
                    # END KORPUSUJ_PATCH_174J2B_COREF_MENTIONS_DOCUMENT_ARRAY_STORAGE
                    # END KORPUSUJ_MIGRATION_036E_PROFILE_DOC_ARRAYS
                    text = str(row.get("Treść", "") or "")
                    metadata = {}
                    for col, value in row.items():
                        if col not in LINGUISTIC_COLUMNS_EXCLUDED_FROM_METADATA:
                            metadata[col] = _safe_scalar(value)
                    for col in ("Data publikacji", "Tytuł", "Autor"):
                        if col in row:
                            metadata[col] = _safe_scalar(row.get(col, ""))
                    metadata["doc_id"] = doc_id
                    token_count = len(tokens)
                    total_tokens += token_count
                    date_val = str(metadata.get("Data publikacji", ""))
                    author_val = str(metadata.get("Autor", ""))
                    title_val = str(metadata.get("Tytuł", ""))
                    doc_rows.append((doc_id, _json_zlib_dumps(metadata, level=1), zlib.compress(text.encode("utf-8"), level=1),
                                     _json_zlib_dumps(tokens, level=1), _json_zlib_dumps(lemmas, level=1), _json_zlib_dumps(start_ids, level=1),
                                     _json_zlib_dumps(end_ids, level=1), _json_zlib_dumps(sentence_ids, level=1), _json_zlib_dumps(deprels, level=1), _json_zlib_dumps(postags, level=1), _json_zlib_dumps(upostags, level=1), _json_zlib_dumps(full_postags, level=1), _json_zlib_dumps(corefs, level=1), _json_zlib_dumps(coref_mentions, level=1)))
                    stats_rows.append((doc_id, token_count, date_val, author_val, title_val))
                    for key in ("Data publikacji", "Autor", "Tytuł"):
                        if metadata.get(key) not in (None, ""):
                            meta_rows.append((doc_id, key, str(metadata.get(key))))
                    for key, value in metadata.items():
                        if key not in ("Data publikacji", "Autor", "Tytuł", "doc_id") and isinstance(value, (str, int, float, bool)) and str(value) != "":
                            meta_rows.append((doc_id, "metadane:" + key, str(value)))
                    m = re.search(r"(\d{4})[-./](\d{1,2})", date_val)
                    if m:
                        monthly_counts[f"{m.group(1)}-{int(m.group(2)):02d}"] += token_count
                    doc_pos = defaultdict(list)
                    if "orth" in indexed_attrs:
                        for pos, val in enumerate(tokens):
                            if val:
                                doc_pos[("orth", val)].append(pos)
                    if "base" in indexed_attrs:
                        for pos, val in enumerate(lemmas):
                            if val:
                                doc_pos[("base", val)].append(pos)
                    for attr, cols in attr_cols.items():
                        if attr not in indexed_attrs:
                            continue
                        vals = None
                        for col in cols:
                            if col in row:
                                cand = [str(x) for x in _as_plain_list(row.get(col, []))]
                                if len(cand) == len(tokens):
                                    vals = cand
                                    break
                        if vals is not None and len(vals) > 0:
                            for pos, val in enumerate(vals):
                                if val:
                                    doc_pos[(attr, val)].append(pos)
                    for (attr, value), positions in doc_pos.items():
                        stage_rows.append((attr, value, doc_id, json.dumps(sorted(set(positions))).encode("utf-8")))
                    if len(doc_rows) >= batch_docs:
                        self._flush_docs_and_stage(con, doc_rows, meta_rows, stage_rows, stats_rows)
                        doc_rows.clear(); meta_rows.clear(); stage_rows.clear(); stats_rows.clear()
                        if progress_callback:
                            if total_expected_docs:
                                percent = (100.0 * total_docs / max(1, total_expected_docs))
                                progress_callback(f"Etap 1/2: {total_docs:,} / {total_expected_docs:,} dokumentów ({percent:.1f}%) — przygotowywanie indeksu".replace(",", " "))
                            else:
                                progress_callback(f"Etap 1/2: zindeksowano {total_docs:,} dokumentów — przygotowywanie indeksu".replace(",", " "))
                del df_part
            if doc_rows or meta_rows or stage_rows:
                self._flush_docs_and_stage(con, doc_rows, meta_rows, stage_rows, stats_rows)
            if progress_callback:
                progress_callback("Etap 2/2: scalanie postingów...")
            self._finalize_terms(con, progress_callback=progress_callback)
            meta = {
                "index_version": SEARCH_INDEX_VERSION,
                "source_parquet_path": os.path.abspath(parquet_path),
                "source_parquet_mtime": str(os.path.getmtime(parquet_path)),
                "source_parquet_size": str(os.path.getsize(parquet_path)),
                "indexed_attrs": ",".join(indexed_attrs),
                "total_docs": str(total_docs), "total_tokens": str(total_tokens),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "creator_version": "unknown", "engine_index_version": SEARCH_INDEX_VERSION,
                "monthly_token_counts": json.dumps(dict(monthly_counts), ensure_ascii=False),
                "postings_format": "json-zlib-stable",
            }
            con.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", list(meta.items()))
            con.execute("DROP TABLE IF EXISTS postings_stage")
            con.commit()
            if progress_callback:
                progress_callback("Kompaktowanie pliku indeksu...")
            con.execute("VACUUM")
            con.commit()
        except Exception:
            con.close()
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                logging.exception("Nie udało się usunąć niekompletnego indeksu tymczasowego: %s", tmp_path)
            raise
        else:
            con.close()
        try:
            os.replace(tmp_path, index_path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                logging.exception("Nie udało się usunąć nieopublikowanego indeksu tymczasowego: %s", tmp_path)
            raise
        return index_path

    def _create_schema(self, con):
        con.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-200000;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE terms (attr TEXT NOT NULL, value TEXT NOT NULL, df INTEGER NOT NULL, cf INTEGER NOT NULL, postings BLOB NOT NULL, PRIMARY KEY (attr, value));
        CREATE TABLE docs (doc_id INTEGER PRIMARY KEY, metadata_json BLOB, text BLOB, tokens BLOB, lemmas BLOB, start_ids BLOB, end_ids BLOB, sentence_ids BLOB, deprels BLOB, postags BLOB, upostags BLOB, full_postags BLOB, corefs BLOB, coref_mentions BLOB);
        CREATE TABLE doc_stats (doc_id INTEGER PRIMARY KEY, token_count INTEGER, date TEXT, author TEXT, title TEXT);
        CREATE TABLE doc_meta (doc_id INTEGER NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL);
        CREATE TABLE postings_stage (attr TEXT NOT NULL, value TEXT NOT NULL, doc_id INTEGER NOT NULL, positions BLOB NOT NULL);
        """)
        con.commit()

    def _flush_docs_and_stage(self, con, doc_rows, meta_rows, stage_rows, stats_rows=None):
        if doc_rows:
            con.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", doc_rows)
            if stats_rows is None:
                stats_rows = []
                for doc_id, meta_blob, _text_blob, tokens_blob, *_ in doc_rows:
                    meta = _json_zlib_loads(meta_blob, {}) or {}
                    toks = _json_zlib_loads(tokens_blob, []) or []
                    stats_rows.append((doc_id, len(toks), str(meta.get("Data publikacji", "")), str(meta.get("Autor", "")), str(meta.get("Tytuł", ""))))
            if stats_rows:
                con.executemany("INSERT OR REPLACE INTO doc_stats VALUES (?, ?, ?, ?, ?)", stats_rows)
        if meta_rows:
            con.executemany("INSERT INTO doc_meta VALUES (?, ?, ?)", meta_rows)
        if stage_rows:
            con.executemany("INSERT INTO postings_stage(attr, value, doc_id, positions) VALUES (?, ?, ?, ?)", stage_rows)
        con.commit()

    def _finalize_terms(self, con, progress_callback=None):
        cur = con.cursor()
        cur.execute("CREATE INDEX idx_postings_stage_attr_value_doc ON postings_stage(attr, value, doc_id)")
        con.commit()
        rows = cur.execute("SELECT attr, value, doc_id, positions FROM postings_stage ORDER BY attr, value, doc_id")
        current_key = None
        postings = {}
        finalized = 0
        def flush_term(key, postings_map):
            if key is None or not postings_map:
                return
            attr, value = key
            df = len(postings_map)
            cf = sum(len(v) for v in postings_map.values())
            con.execute("INSERT OR REPLACE INTO terms(attr, value, df, cf, postings) VALUES (?, ?, ?, ?, ?)",
                        (attr, value, df, cf, PostingList.encode(postings_map)))
        for attr, value, doc_id, pos_blob in rows:
            key = (attr, value)
            if current_key is not None and key != current_key:
                flush_term(current_key, postings)
                finalized += 1
                if finalized % 5000 == 0:
                    con.commit()
                    if progress_callback:
                        progress_callback(f"Etap 2/2: scalono {finalized:,} terminów...".replace(",", " "))
                postings = {}
            current_key = key
            try:
                positions = json.loads(pos_blob.decode("utf-8") if isinstance(pos_blob, (bytes, bytearray)) else pos_blob)
            except Exception:
                positions = []
            postings[int(doc_id)] = [int(p) for p in positions]
        flush_term(current_key, postings)
        con.commit()
        con.executescript("CREATE INDEX idx_doc_meta_key_value_doc ON doc_meta(key, value, doc_id); CREATE INDEX idx_doc_meta_doc ON doc_meta(doc_id);")
        con.commit()

# KORPUSUJ_PATCH_174J2_COREF_MENTIONS_SIDECAR_STORAGE
# docs.coref_mentions uses the existing JSON+zlib document-array codec.
# END KORPUSUJ_PATCH_174J2_COREF_MENTIONS_SIDECAR_STORAGE
