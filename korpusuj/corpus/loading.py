"""Corpus loading helpers for Korpusuj.

This module prepares lazy corpus bundles from Parquet files and .search sidecars.
It intentionally does not know about tkinter/customtkinter, messagebox, app.after,
file dialogs, engine globals, or dependency warmup orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pyarrow.parquet as pq

from korpusuj.index.builder import SearchIndexBuilder
from korpusuj.index.sqlite_index import SearchIndex, LazyTermIndex
from korpusuj.search.backend import LazyCorpus


@dataclass(frozen=True)
class LoadedCorpusBundle:
    """Hold the canonical corpus, derived index and metadata prepared for callers."""
    name: str
    parquet_path: str
    search_path: str
    columns: list[str]
    total_docs: int
    total_tokens: int
    monthly_token_counts: dict
    korpus_meta: dict
    search_meta: dict
    dataframe: object
    inverted_index: dict


def search_sidecar_path(parquet_path) -> str:
    """Return .search sidecar path for a Parquet corpus path."""
    return str(Path(parquet_path).with_suffix(".search"))


def read_korpus_meta_from_parquet_schema(parquet_path) -> tuple[list[str], dict]:
    """Read Parquet columns and optional korpus_meta JSON metadata."""
    schema = pq.read_schema(parquet_path)
    columns = list(schema.names)
    metadata = schema.metadata or {}
    if b"korpus_meta" not in metadata:
        return columns, {}
    try:
        return columns, json.loads(metadata[b"korpus_meta"].decode("utf-8"))
    except Exception:
        return columns, {}


def ensure_search_index_for_parquet(
    parquet_path,
    search_path,
    indexed_attrs,
    batch_docs=5000,
    progress_callback=None,
    builder=None,
) -> dict:
    """Ensure .search sidecar is fresh and return SearchIndex metadata.

    indexed_attrs and batch_docs are passed in from engine.py so this module does not
    duplicate profile/config resolution logic.
    """
    parquet_path = str(parquet_path)
    search_path = str(search_path)
    indexed_attrs = tuple(indexed_attrs)
    builder = builder or SearchIndexBuilder()

    if not SearchIndexBuilder.is_fresh(parquet_path, search_path, indexed_attrs=indexed_attrs):
        if progress_callback:
            progress_callback("Przygotowywanie indeksu wyszukiwania...")
        builder.build_from_parquet(
            parquet_path,
            search_path,
            batch_docs=int(batch_docs or 5000),
            indexed_attrs=indexed_attrs,
            progress_callback=progress_callback,
        )

    idx = SearchIndex(search_path)
    try:
        return dict(idx.meta() or {})
    finally:
        try:
            idx.close()
        except Exception:
            pass


def parse_monthly_counts_from_meta(search_meta: dict, korpus_meta: dict) -> dict:
    """Prefer monthly counts from .search metadata, fallback to Parquet korpus_meta."""
    try:
        raw = search_meta.get("monthly_token_counts", "{}")
        if isinstance(raw, str):
            parsed = json.loads(raw or "{}")
        elif isinstance(raw, dict):
            parsed = raw
        else:
            parsed = {}
        if parsed:
            return parsed
    except Exception:
        pass
    try:
        return korpus_meta.get("monthly_token_counts", {}) or {}
    except Exception:
        return {}


def build_lazy_corpus_bundle(
    name,
    parquet_path,
    search_path,
    columns,
    total_docs,
    total_tokens,
    monthly_counts,
    korpus_meta=None,
    search_meta=None,
) -> LoadedCorpusBundle:
    """Build LazyCorpus and inverted_index dict without mutating engine globals."""
    korpus_meta = dict(korpus_meta or {})
    search_meta = dict(search_meta or {})
    monthly_counts = monthly_counts or {}

    dataframe = LazyCorpus(
        str(parquet_path),
        str(search_path),
        list(columns),
        int(total_docs or 0),
        {"total_tokens": int(total_tokens or 0), "monthly_token_counts": monthly_counts},
    )
    inverted_index = {
        "base": LazyTermIndex(str(search_path), "base"),
        "orth": LazyTermIndex(str(search_path), "orth"),
        "base_tf": korpus_meta.get("base_tf", {}),
        "orth_tf": korpus_meta.get("orth_tf", {}),
        "total_tokens": int(total_tokens or 0),
        "monthly_token_counts": monthly_counts,
    }

    return LoadedCorpusBundle(
        name=str(name),
        parquet_path=str(parquet_path),
        search_path=str(search_path),
        columns=list(columns),
        total_docs=int(total_docs or 0),
        total_tokens=int(total_tokens or 0),
        monthly_token_counts=monthly_counts,
        korpus_meta=korpus_meta,
        search_meta=search_meta,
        dataframe=dataframe,
        inverted_index=inverted_index,
    )


def prepare_loaded_corpus_bundle(
    name,
    parquet_path,
    indexed_attrs,
    batch_docs=5000,
    progress_callback=None,
) -> LoadedCorpusBundle:
    """Prepare one corpus bundle from Parquet + .search sidecar.

    This helper intentionally receives indexed_attrs and batch_docs from engine.py.
    That preserves the existing config/profile behavior of get_search_indexed_attrs().
    """
    parquet_path = str(parquet_path)
    columns, korpus_meta = read_korpus_meta_from_parquet_schema(parquet_path)
    search_path = search_sidecar_path(parquet_path)

    search_meta = ensure_search_index_for_parquet(
        parquet_path=parquet_path,
        search_path=search_path,
        indexed_attrs=indexed_attrs,
        batch_docs=batch_docs,
        progress_callback=progress_callback,
    )

    try:
        pf = pq.ParquetFile(parquet_path)
        total_docs = int(getattr(pf.metadata, "num_rows", 0) or search_meta.get("total_docs", 0) or 0)
    except Exception:
        total_docs = int(search_meta.get("total_docs", 0) or 0)

    total_tokens = int(search_meta.get("total_tokens", 0) or korpus_meta.get("total_tokens", 0) or 0)
    monthly_counts = parse_monthly_counts_from_meta(search_meta, korpus_meta)

    return build_lazy_corpus_bundle(
        name=name,
        parquet_path=parquet_path,
        search_path=search_path,
        columns=columns,
        total_docs=total_docs,
        total_tokens=total_tokens,
        monthly_counts=monthly_counts,
        korpus_meta=korpus_meta,
        search_meta=search_meta,
    )
