# -*- coding: utf-8 -*-
"""Build and run GUI-independent search contexts over canonical Parquet corpora.

The module wires LazyCorpus, SearchIndex, executors, materialization, sorting
and legacy fallback adapters without importing the desktop interface.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from korpusuj.search.backend import LazyCorpus
from korpusuj.search.cursor import SearchCursor
from korpusuj.search.cursor_runtime import (
    configure_full_context_size_provider,
    configure_search_cursor_runtime,
)
from korpusuj.search.executor import CorpusSearchExecutor, configure_search_executor
from korpusuj.index.sqlite_index import SearchIndex
from korpusuj.search.headless import SearchBackendContext

__all__ = [
    "configure_non_gui_search_cursor_runtime",
    "build_lazy_corpus_for_headless",
    "build_corpus_search_executor_for_headless",
    "build_non_gui_find_lemma_context_adapter",
    "build_headless_context_from_parquet",
]


def _headless_runner_corpus_name_from_path_036l4g45d(path: Any) -> str:
    try:
        return Path(str(path)).stem
    except Exception:
        return str(path or "")


def configure_non_gui_search_cursor_runtime(
    *,
    full_context_size: int = 250,
    candidate_max_docs: int = 3000,
    candidate_stream_batch_docs: int = 256,
    dependency_maps_cache: Any = None,
) -> Any:
    """Configure SearchCursor runtime with safe non-GUI defaults.

    The GUI configures this runtime from ``engine.py``. A headless/CLI caller
    needs a minimal equivalent that does not touch GUI globals. This default is
    intentionally conservative: RAM dependency cache mode is ``"none"`` and
    preload callbacks are no-ops.

    Returns the dependency maps cache object used by the runtime, so callers can
    inspect or reuse it if needed.
    """
    if dependency_maps_cache is None:
        dependency_maps_cache = {}

    def get_dependency_cache_ram_mode() -> str:
        return "none"

    def dependency_ram_cache_size_for_corpus(corpus_name: Any = None) -> int:
        try:
            if corpus_name is None:
                return len(dependency_maps_cache)
            return sum(
                1
                for key in dependency_maps_cache
                if isinstance(key, tuple) and key and key[0] == corpus_name
            )
        except Exception:
            return 0

    def put_dependency_ram_cache(cache_key: Any, dep_maps: Any) -> None:
        try:
            dependency_maps_cache[cache_key] = dep_maps
        except Exception:
            pass

    def preload_dependency_maps_for_candidates(*args: Any, **kwargs: Any) -> int:
        return 0

    configure_full_context_size_provider(lambda: int(full_context_size))
    configure_search_cursor_runtime(
        dependency_cache_corpus_name_from_path=_headless_runner_corpus_name_from_path_036l4g45d,
        get_dependency_cache_ram_mode=get_dependency_cache_ram_mode,
        dependency_ram_cache_size_for_corpus=dependency_ram_cache_size_for_corpus,
        put_dependency_ram_cache=put_dependency_ram_cache,
        preload_dependency_maps_for_candidates=preload_dependency_maps_for_candidates,
        dependency_maps_cache=dependency_maps_cache,
        candidate_max_docs=int(candidate_max_docs),
        candidate_stream_batch_docs=int(candidate_stream_batch_docs),
        full_context_size=int(full_context_size),
    )
    return dependency_maps_cache


def _headless_runner_parquet_columns_036l4g45d(parquet_path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return list(pq.ParquetFile(str(parquet_path)).schema_arrow.names)
    except Exception:
        return []


def _headless_runner_parquet_rows_036l4g45d(parquet_path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(str(parquet_path))
        return int(sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)))
    except Exception:
        return None


def build_lazy_corpus_for_headless(
    parquet_path: str | Path,
    *,
    search_path: str | Path | None = None,
    corpus_name: str | None = None,
    columns: list[str] | None = None,
    total_docs: int | None = None,
    meta: dict[str, Any] | None = None,
) -> LazyCorpus:
    """Build a LazyCorpus for non-GUI headless execution."""
    parquet = Path(parquet_path)
    search = Path(search_path) if search_path is not None else parquet.with_suffix(".search")
    if columns is None:
        columns = _headless_runner_parquet_columns_036l4g45d(parquet)
    if total_docs is None:
        total_docs = _headless_runner_parquet_rows_036l4g45d(parquet)
    corpus_meta = dict(meta or {})
    corpus_meta.setdefault("source", "headless_runner_036L4G45D")
    corpus_meta.setdefault("corpus_name", corpus_name or parquet.stem)
    return LazyCorpus(
        parquet_path=str(parquet),
        search_path=str(search),
        columns=columns,
        total_docs=total_docs,
        meta=corpus_meta,
    )


def build_corpus_search_executor_for_headless(lazy_corpus: LazyCorpus) -> CorpusSearchExecutor:
    """Wire SearchExecutor dependencies and return a CorpusSearchExecutor."""
    configure_search_executor(search_cursor_cls=SearchCursor, search_index_cls=SearchIndex)
    return CorpusSearchExecutor(lazy_corpus=lazy_corpus)


def _headless_runner_materialize_limited_036l4g45d(
    results: Any,
    limit: int | None,
    offset: int = 0,
) -> Any:
    """Materialize one page without limiting discovery or final hit counting."""
    offset_i = max(0, int(offset or 0))
    limit_i = None if limit is None else int(limit)
    if limit_i is not None and limit_i < 0:
        limit_i = None

    get_range = getattr(results, "get_range", None)
    if callable(get_range):
        if limit_i is not None:
            return list(get_range(offset_i, offset_i + limit_i))
        # Exact counting has already exhausted SearchCursor at this boundary.
        # Slice its complete coordinate/result sequence without an arbitrary cap.
        ensure_all = getattr(results, "_ensure_all", None)
        if callable(ensure_all):
            ensure_all()
        hits = getattr(results, "_hits", None)
        result_at = getattr(results, "_result", None)
        if hits is not None and callable(result_at):
            return [result_at(i) for i in range(offset_i, len(hits))]

    if isinstance(results, (list, tuple)):
        end = None if limit_i is None else offset_i + limit_i
        return list(results[offset_i:end])
    try:
        iterator = itertools.islice(iter(results), offset_i, None)
        if limit_i is None:
            return list(iterator)
        return list(itertools.islice(iterator, limit_i))
    except Exception:
        return results




class MaterializedSearchResults(list):
    """List-compatible page of results that preserves the full hit count."""

    def __init__(
        self,
        rows: Any,
        total_hits: int | None = None,
        total_hits_source: str | None = None,
        total_hits_counting_strategy: str | None = None,
    ) -> None:
        super().__init__(rows or [])
        self.total_hits = total_hits
        self.total_hits_source = total_hits_source
        self.total_hits_counting_strategy = total_hits_counting_strategy

    def count_hits(self) -> int | None:
        """Return the exact number of materialized final hits."""
        return self.total_hits


def _count_search_results_before_materialization(results: Any) -> tuple[int | None, str | None, str | None]:
    """Return final total_hits from cursor-like results before page materialization.

    SENTENCE_OPERATOR_TOTAL_HITS_SOURCE_167K

    For SearchCursor plans carrying plan["sentence_operator"], the normal fast
    count/estimate path is not valid because the sentence operator is applied as
    a cursor-level post-filter. In that case, exhaust the already-filtered
    SearchCursor iterator via _ensure_until(...) and use its filtered _count_cache.

    Non-<s> queries keep the existing count_final_searchcursor_hits path.
    """
    try:
        plan = getattr(results, "plan", None)
        has_sentence_operator = isinstance(plan, dict) and isinstance(plan.get("sentence_operator"), dict)
    except Exception:
        has_sentence_operator = False

    if has_sentence_operator:
        try:
            ensure = getattr(results, "_ensure_until", None)
            if callable(ensure):
                ensure(10 ** 12)
                cache = getattr(results, "_count_cache", None)
                if cache is not None:
                    return int(cache), "final_count", "sentence_operator_exact_iter"
                hits = getattr(results, "_hits", None)
                if hits is not None:
                    return int(len(hits)), "final_count", "sentence_operator_exact_iter"
        except Exception:
            pass
        try:
            # Last-resort exact count. This is not expected for SearchCursor, but
            # keeps the function safe for cursor-like objects.
            total = sum(1 for _ in results)
            return int(total), "final_count", "sentence_operator_exact_iter"
        except Exception:
            pass

    try:
        from korpusuj.search.result_materialization import count_final_searchcursor_hits
        payload = count_final_searchcursor_hits(results)
        total = payload.get("total_hits") if isinstance(payload, dict) else None
        if total is not None:
            source = payload.get("source", "final_count") if isinstance(payload, dict) else "final_count"
            strategy = payload.get("strategy") if isinstance(payload, dict) else None
            return int(total), str(source), (str(strategy) if strategy else None)
    except Exception:
        pass
    return None, None, None

def _with_preserved_total_hits(
    rows: Any,
    total_hits: int | None,
    total_hits_source: str | None,
    total_hits_counting_strategy: str | None = None,
) -> Any:
    """Wrap materialized rows only when a final total count is known."""
    if total_hits is None:
        return rows
    try:
        return MaterializedSearchResults(
            rows,
            total_hits=total_hits,
            total_hits_source=total_hits_source,
            total_hits_counting_strategy=total_hits_counting_strategy,
        )
    except Exception:
        return rows

def _shared_first_real_token_for_sort(text: Any) -> str:
    """Return the first non-empty punctuation-stripped token, lower-cased."""
    import string
    if not text:
        return ""
    for token in str(text).split():
        cleaned = token.strip(string.punctuation).lower()
        if cleaned:
            return cleaned
    return ""


def _shared_last_real_token_for_sort(text: Any) -> str:
    """Return the last non-empty punctuation-stripped token, lower-cased."""
    import string
    if not text:
        return ""
    for token in reversed(str(text).split()):
        cleaned = token.strip(string.punctuation).lower()
        if cleaned:
            return cleaned
    return ""


def _shared_materialize_complete_results_for_sort(results: Any) -> list[Any]:
    """Materialize every final row without imposing an internal result cap."""
    hits = getattr(results, "_hits", None)
    result_at = getattr(results, "_result", None)
    if hits is not None and callable(result_at):
        ensure_all = getattr(results, "_ensure_all", None)
        if callable(ensure_all):
            ensure_all()
        return [result_at(index) for index in range(len(hits))]
    if isinstance(results, list):
        return list(results)
    if isinstance(results, tuple):
        return list(results)
    return list(results)


def _shared_sort_complete_results_before_paging(
    results: Any,
    sort_option: Any,
) -> tuple[Any, bool]:
    """Apply the engine.py GUI sort contract to the complete final result set.

    Returns ``(results_or_sorted_list, applied)``. Unknown/empty choices preserve
    the original cursor/list so natural-order paging remains lazy and unchanged.
    Python's stable list.sort preserves the original positional order for equal
    keys, matching the GUI contract.
    """
    from collections import Counter

    choice = str(sort_option or "").strip()
    supported = {
        "Data publikacji", "Tytuł", "Autor", "Alfabetycznie",
        "Prawy kontekst", "Lewy kontekst",
        "Frekwencja base", "Frekwencja orth",
    }
    if choice not in supported:
        return results, False

    rows = _shared_materialize_complete_results_for_sort(results)
    if choice == "Data publikacji":
        rows.sort(key=lambda row: str(row[0]) if row[0] else "")
    elif choice == "Tytuł":
        rows.sort(key=lambda row: str(row[6]) if row[6] else "")
    elif choice == "Autor":
        rows.sort(key=lambda row: str(row[7]) if row[7] else "")
    elif choice == "Alfabetycznie":
        rows.sort(key=lambda row: str(row[3]).lower() if row[3] else "")
    elif choice == "Prawy kontekst":
        rows.sort(key=lambda row: _shared_first_real_token_for_sort(row[10]))
    elif choice == "Lewy kontekst":
        rows.sort(key=lambda row: _shared_last_real_token_for_sort(row[9]))
    elif choice == "Frekwencja base":
        base_counter = Counter(str(row[4]) for row in rows)
        rows.sort(key=lambda row: (
            -base_counter[str(row[4])],
            str(row[4]).lower(),
            str(row[3]).lower(),
        ))
    elif choice == "Frekwencja orth":
        orth_counter = Counter(str(row[3]) for row in rows)
        rows.sort(key=lambda row: (
            -orth_counter[str(row[3])],
            str(row[3]).lower(),
            str(row[4]).lower(),
        ))
    return rows, True

def build_non_gui_find_lemma_context_adapter(
    executor: CorpusSearchExecutor,
    *,
    limit: int | None = None,
    normalize_hits: bool = False,
    normalize_fn: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Build the request-aware non-GUI search adapter."""
    if normalize_hits and normalize_fn is None:
        raise ValueError("normalize_fn is required when normalize_hits=True")

    def adapter(
        query: str,
        df: Any,
        corpus_name: str,
        left_context_size: int = 10,
        right_context_size: int = 10,
        warnings_list: list[str] | None = None,
        search_request: Any = None,
    ) -> Any:
        cursor_or_results = executor.search(
            query,
            left_context_size=int(left_context_size),
            right_context_size=int(right_context_size),
        )
        total_hits, total_hits_source, total_hits_counting_strategy = (
            _count_search_results_before_materialization(cursor_or_results)
        )
        effective_limit = (
            getattr(search_request, "limit", None)
            if search_request is not None
            else limit
        )
        effective_offset = (
            int(getattr(search_request, "offset", 0) or 0)
            if search_request is not None
            else 0
        )
        effective_sort_option = (
            getattr(search_request, "sort_option", None)
            if search_request is not None
            else None
        )

        sortable_results, sort_applied = _shared_sort_complete_results_before_paging(
            cursor_or_results,
            effective_sort_option,
        )
        rows = _headless_runner_materialize_limited_036l4g45d(
            sortable_results,
            effective_limit,
            effective_offset,
        )
        rows = _with_preserved_total_hits(
            rows,
            total_hits,
            total_hits_source,
            total_hits_counting_strategy,
        )
        try:
            rows.sort_option = effective_sort_option
            rows.sort_applied = bool(sort_applied)
            rows.sort_scope = "all_final_hits_before_paging" if sort_applied else "natural_order"
        except Exception:
            pass
        if normalize_hits:
            normalized_rows = normalize_fn(rows)  # type: ignore[misc]
            normalized_rows = _with_preserved_total_hits(
                normalized_rows,
                total_hits,
                total_hits_source,
                total_hits_counting_strategy,
            )
            try:
                normalized_rows.sort_option = effective_sort_option
                normalized_rows.sort_applied = bool(sort_applied)
                normalized_rows.sort_scope = "all_final_hits_before_paging" if sort_applied else "natural_order"
            except Exception:
                pass
            return normalized_rows
        return rows

    adapter.__name__ = "non_gui_find_lemma_context_adapter"
    adapter._supports_search_request = True
    return adapter


def build_headless_context_from_parquet(
    parquet_path: str | Path,
    *,
    corpus_name: str | None = None,
    search_path: str | Path | None = None,
    limit: int | None = None,
    normalize_hits: bool = False,
    normalize_fn: Callable[..., Any] | None = None,
    full_context_size: int = 250,
    candidate_max_docs: int = 3000,
    candidate_stream_batch_docs: int = 256,
    dependency_maps_cache: Any = None,
    config: dict[str, Any] | None = None,
) -> SearchBackendContext:
    """Build a SearchBackendContext backed by a Parquet corpus and its .search sidecar.
    
    This is the main convenience entry point for headless and CLI search. It configures the non-GUI SearchCursor runtime, builds LazyCorpus and the executor, and supplies the request-aware search adapter to the context.
    """
    parquet = Path(parquet_path)
    search = Path(search_path) if search_path is not None else parquet.with_suffix(".search")
    name = str(corpus_name or parquet.stem)

    configure_non_gui_search_cursor_runtime(
        full_context_size=full_context_size,
        candidate_max_docs=candidate_max_docs,
        candidate_stream_batch_docs=candidate_stream_batch_docs,
        dependency_maps_cache=dependency_maps_cache,
    )
    lazy = build_lazy_corpus_for_headless(
        parquet,
        search_path=search,
        corpus_name=name,
        meta={"source": "headless_context_from_parquet_036L4G45D", "corpus_name": name},
    )
    executor = build_corpus_search_executor_for_headless(lazy)
    adapter = build_non_gui_find_lemma_context_adapter(
        executor,
        limit=limit,
        normalize_hits=normalize_hits,
        normalize_fn=normalize_fn,
    )

    cfg = dict(config or {})
    cfg.setdefault("source", "headless_runner_036L4G45D")
    cfg.setdefault("normalize_hits", bool(normalize_hits))
    cfg.setdefault("limit", limit)

    return SearchBackendContext(
        dataframes={name: lazy},
        corpora={name: lazy},
        current_corpus_path=str(parquet),
        config=cfg,
        corpus_name=name,
        parquet_path=str(parquet),
        search_path=str(search),
        dep_cache_path=str(parquet.with_suffix(".dep_cache")),
        has_parquet=parquet.exists(),
        has_search_index=search.exists(),
        has_dep_cache=parquet.with_suffix(".dep_cache").exists(),
        df_type=type(lazy).__name__,
        search_df_type=type(lazy).__name__,
        df_is_lazy=True,
        search_df_is_lazy=True,
        stats_rows=getattr(lazy, "total_docs", None),
        search_rows=getattr(lazy, "total_docs", None),
        indexed_attrs=("base", "orth", "pos", "upos", "deprel", "ner"),
        metadata_columns=[],
        metadata_column_count=0,
        config_snapshot=cfg,
        find_lemma_context_adapter=adapter,
    )

# END KORPUSUJ_MIGRATION_036L4G45D_NON_GUI_HEADLESS_RUNNER_SCAFFOLD
