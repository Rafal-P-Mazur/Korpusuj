# -*- coding: utf-8 -*-
"""Shared backend objects used to connect loaded corpora with indexed search execution."""
from __future__ import annotations

from pathlib import Path
from korpusuj.index.sqlite_index import SearchIndex
from korpusuj.dependency.disk_cache import DependencyMapDiskCache

# KORPUSUJ_PATCH_188_SQLITE_RESULT_ANALYTICS
class DocumentRow(dict):
    """Mapping returned by SQLite with legacy attribute-style row access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class LazyCorpus:
    """SQLite-backed corpus facade with an explicit Parquet compatibility fallback."""
    def __init__(self, parquet_path, search_path, columns=None, total_docs=0, meta=None):
        self.parquet_path = str(parquet_path)
        self.search_path = str(search_path)
        self.columns = list(columns or [])
        self.total_docs = int(total_docs or 0)
        self.meta = meta or {}
        self._search_index = None
        self._dependency_cache = None

    def __len__(self):
        return self.total_docs

    @property
    def index(self):
        return range(self.total_docs)

    def _index(self):
        if self._search_index is None:
            self._search_index = SearchIndex(self.search_path)
            if not self.total_docs:
                self.total_docs = self._search_index.total_docs
        return self._search_index

    def _dep(self):
        if self._dependency_cache is None:
            dep_path = Path(self.parquet_path).with_suffix('.dep_cache')
            if dep_path.is_file():
                self._dependency_cache = DependencyMapDiskCache(self.parquet_path, cache_path=dep_path)
        return self._dependency_cache

    @staticmethod
    def _dependency_arrays(doc, dep_maps):
        if not dep_maps:
            return [], []
        parents = list(dep_maps[0] or [])
        sentence_ids = list(doc.get('sentence_ids') or [])
        local_ids = {}
        counters = {}
        word_ids = []
        for pos, sentence_id in enumerate(sentence_ids):
            counters[sentence_id] = counters.get(sentence_id, 0) + 1
            local_ids[pos] = counters[sentence_id]
            word_ids.append(local_ids[pos])
        head_ids = []
        for parent in parents[:len(sentence_ids)]:
            try:
                parent = int(parent)
            except Exception:
                parent = -1
            head_ids.append(0 if parent < 0 else int(local_ids.get(parent, 0)))
        if len(head_ids) < len(sentence_ids):
            head_ids.extend([0] * (len(sentence_ids) - len(head_ids)))
        return word_ids, head_ids

    def _enrich(self, doc, doc_id, dep_maps=None, dependency_loaded=False):
        if doc is None:
            return None
        row = DocumentRow(doc)
        # KORPUSUJ_PATCH_188A_GUI_GRAPH_METADATA_FLATTEN
        # SearchIndex stores user-facing fields inside metadata_json, while the
        # long-standing GUI row contract expects them at row top level.
        metadata = row.get('metadata')
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                row.setdefault(str(key), value)
        row.setdefault('Treść', row.get('text', ''))
        # Common aliases keep strict display-metadata matching independent of
        # whether a sidecar uses Polish canonical names or normalized names.
        aliases = (
            ('Tytuł', 'title'),
            ('Data publikacji', 'publication_date'),
            ('Autor', 'author'),
        )
        for canonical, normalized in aliases:
            if canonical not in row and normalized in row:
                row[canonical] = row.get(normalized)
            if normalized not in row and canonical in row:
                row[normalized] = row.get(canonical)
        # END KORPUSUJ_PATCH_188A_GUI_GRAPH_METADATA_FLATTEN
        if not dependency_loaded:
            dep = self._dep()
            dep_maps = dep.get(int(doc_id)) if dep is not None else None
        word_ids, head_ids = self._dependency_arrays(row, dep_maps)
        row['word_ids'] = word_ids
        row['head_ids'] = head_ids
        return row

    def get_doc(self, doc_id):
        return self._enrich(self._index().get_doc(int(doc_id)), int(doc_id))

    def get_ner_labels(self, doc_id, token_count):
        """Load NER labels only for the currently displayed GUI document."""
        return self._index().get_ner_labels(int(doc_id), int(token_count))

    def get_docs_many(self, doc_ids, chunk_size=800):
        # KORPUSUJ_PATCH_189C_COLLOCATION_SQLITE_BATCHING
        idx = self._index()
        loader = getattr(idx, 'get_docs_many', None) or idx.get_docs_many_036l4g8
        docs = loader(doc_ids, chunk_size=chunk_size)
        dep = self._dep()
        dep_maps = dep.get_many(docs.keys(), batch_size=chunk_size) if dep is not None else {}
        return {
            int(doc_id): self._enrich(
                doc,
                int(doc_id),
                dep_maps=dep_maps.get(int(doc_id)),
                dependency_loaded=True,
            )
            for doc_id, doc in docs.items()
        }

    def get_background_frequencies(self, attr, values, ignore_case=False, chunk_size=800):
        """Return corpus frequencies only for requested collocation candidates."""
        idx = self._index()
        attr = str(attr)
        candidates = [str(value) for value in dict.fromkeys(values or [])]
        if ignore_case:
            candidates = list(dict.fromkeys(value.lower() for value in candidates))
        if not candidates:
            return {}
        chunk_size = max(1, int(chunk_size or 800))
        out = {}
        if ignore_case:
            idx.con.create_function("KORPUS_LOWER", 1, lambda value: str(value or '').lower())
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            if ignore_case:
                sql = (
                    "SELECT KORPUS_LOWER(value) AS normalized, SUM(cf) "
                    "FROM terms WHERE attr=? AND KORPUS_LOWER(value) IN ("
                    + placeholders + ") GROUP BY normalized"
                )
            else:
                sql = "SELECT value, cf FROM terms WHERE attr=? AND value IN (" + placeholders + ")"
            for value, frequency in idx.con.execute(sql, [attr] + chunk):
                out[str(value)] = int(frequency or 0)
        return out

    def background_index(self):
        idx = self._index()
        return {
            'frequency_provider': self,
            'total_tokens': max(1, idx.total_tokens),
        }
        # END KORPUSUJ_PATCH_189C_COLLOCATION_SQLITE_BATCHING

    def materialize(self):
        """Explicit compatibility fallback to the canonical Parquet corpus.

        This method is intentionally explicit. LazyCorpus has no broad
        __getattr__, so ordinary SQLite document access cannot trigger it.
        """
        if getattr(self, '_df', None) is None:
            import logging
            import pandas as pd
            logging.warning(
                "Jawny fallback wyszukiwania do kanonicznego Parquet: %s",
                self.parquet_path,
            )
            self._df = pd.read_parquet(self.parquet_path)
        return self._df

    def close(self):
        if self._dependency_cache is not None:
            self._dependency_cache.close(); self._dependency_cache = None
        if self._search_index is not None:
            self._search_index.close(); self._search_index = None
# END KORPUSUJ_PATCH_188_SQLITE_RESULT_ANALYTICS
