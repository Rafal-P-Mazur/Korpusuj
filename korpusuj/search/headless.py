# -*- coding: utf-8 -*-
"""GUI-independent request, result and backend contracts for corpus search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "SearchRequest",
    "SearchBackendContext",
    "SearchResultBundle",
    "SearchHit",
    "SearchMessage",
    "HeadlessSearchNotConfiguredError",
    "validate_search_request",
    "run_search_headless",
    "normalize_search_result_to_hit",
    "normalize_search_results_to_hits",
]


class HeadlessSearchNotConfiguredError(RuntimeError):
    """Raised when headless search cannot run because no backend adapter was supplied."""


@dataclass(slots=True)
class SearchMessage:
    """A GUI-independent message returned by headless search."""

    level: str
    text: str
    code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchRequest:
    """GUI-independent search request."""

    query: str
    corpus_name: str
    left_context: int = 10
    right_context: int = 10
    sort_option: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    selected_sense: str | None = None
    limit: int | None = 100
    offset: int = 0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchBackendContext:
    """Bundle the corpus, index and adapter objects required by headless search."""
    # KORPUSUJ_MIGRATION_036L4G39B2_CONTEXT_DATAFRAMES_COMPAT
    dataframes: Mapping[str, Any] = field(default_factory=dict)
    corpora: Mapping[str, Any] | None = None
    current_corpus_path: str | None = None
    config: Mapping[str, Any] | None = None
    corpus_name: str = ""
    parquet_path: str | None = None
    search_path: str | None = None
    dep_cache_path: str | None = None
    has_parquet: bool = False
    has_search_index: bool = False
    has_dep_cache: bool = False
    df_type: str | None = None
    search_df_type: str | None = None
    df_is_lazy: bool = False
    search_df_is_lazy: bool = False
    stats_rows: int | None = None
    search_rows: int | None = None
    indexed_attrs: tuple[str, ...] = ()
    metadata_columns: tuple[str, ...] = ()
    metadata_column_count: int = 0
    config_snapshot: dict = field(default_factory=dict)
    find_lemma_context_adapter: object | None = None
@dataclass(slots=True)
class SearchHit:
    """JSON-serializable representation of a single concordance hit returned by headless search."""

    doc_id: int | None = None
    start: int | None = None
    end: int | None = None
    match_text: str = ""
    left_context: str = ""
    right_context: str = ""
    extended_left: str = ""
    extended_match: str = ""
    extended_right: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any | None = None


@dataclass(slots=True)
class SearchResultBundle:
    """GUI-independent search result envelope."""

    request: SearchRequest
    results: Any = field(default_factory=list)
    total_hits: int | None = None
    warnings: list[str] = field(default_factory=list)
    messages: list[SearchMessage] = field(default_factory=list)
    statistics_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    offset: int = 0
    has_more: bool | None = None


def validate_search_request(request: SearchRequest) -> None:
    """Validate the basic shape of a headless request.

    This intentionally does not perform full CQL validation yet. Full query
    validation can later be migrated from engine.validate_query_for_ui.
    """

    if not isinstance(request, SearchRequest):
        raise TypeError("request must be a SearchRequest")
    if not str(request.query or "").strip():
        raise ValueError("SearchRequest.query cannot be empty")
    if not str(request.corpus_name or "").strip():
        raise ValueError("SearchRequest.corpus_name cannot be empty")
    if int(request.left_context) < 0 or int(request.right_context) < 0:
        raise ValueError("SearchRequest context sizes must be non-negative")
    if request.limit is not None and int(request.limit) <= 0:
        raise ValueError("SearchRequest.limit must be positive or None")
    if int(request.offset) < 0:
        raise ValueError("SearchRequest.offset must be non-negative")


def _resolve_corpus_object(request: SearchRequest, context: SearchBackendContext) -> Any:
    if request.corpus_name in context.dataframes:
        return context.dataframes[request.corpus_name]
    if context.corpora and request.corpus_name in context.corpora:
        return context.corpora[request.corpus_name]
    raise KeyError(f"Corpus not found in SearchBackendContext: {request.corpus_name!r}")


def _count_total_hits(results: Any) -> int | None:
    try:
        if hasattr(results, "count_hits"):
            return int(results.count_hits())
    except Exception:
        pass
    try:
        return len(results)  # type: ignore[arg-type]
    except Exception:
        return None


# KORPUSUJ_MIGRATION_036L4G39C_NORMALIZE_RESULTS_TO_HITS

def _headless_safe_int_036l4g39c(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _headless_safe_str_036l4g39c(value: Any) -> str:
    try:
        if value is None:
            return ""
        return str(value)
    except Exception:
        return ""


def _headless_context_part_036l4g39c(value: Any, index: int) -> str:
    try:
        if isinstance(value, (list, tuple)) and len(value) > index:
            return _headless_safe_str_036l4g39c(value[index])
    except Exception:
        pass
    return ""


def normalize_search_result_to_hit(row: Any) -> SearchHit:
    """Normalize a shared-search result row to the public SearchHit representation."""

    if isinstance(row, SearchHit):
        return row

    if isinstance(row, dict):
        metadata = dict(row.get("metadata") or {})
        extended = row.get("extended_context") or row.get("extended") or {}
        if isinstance(extended, dict):
            extended_left = extended.get("left", "")
            extended_match = extended.get("match", row.get("match_text", row.get("match", "")))
            extended_right = extended.get("right", "")
        elif isinstance(extended, (list, tuple)):
            extended_left = _headless_context_part_036l4g39c(extended, 0)
            extended_match = _headless_context_part_036l4g39c(extended, 1)
            extended_right = _headless_context_part_036l4g39c(extended, 2)
        else:
            extended_left = extended_match = extended_right = ""
        return SearchHit(
            doc_id=_headless_safe_int_036l4g39c(row.get("doc_id", row.get("doc"))),
            start=_headless_safe_int_036l4g39c(row.get("start", row.get("token_start"))),
            end=_headless_safe_int_036l4g39c(row.get("end", row.get("token_end"))),
            match_text=_headless_safe_str_036l4g39c(row.get("match_text", row.get("match", ""))),
            left_context=_headless_safe_str_036l4g39c(row.get("left_context", "")),
            right_context=_headless_safe_str_036l4g39c(row.get("right_context", "")),
            extended_left=_headless_safe_str_036l4g39c(extended_left),
            extended_match=_headless_safe_str_036l4g39c(extended_match),
            extended_right=_headless_safe_str_036l4g39c(extended_right),
            metadata=metadata,
            raw=row.get("raw", row),
        )

    if isinstance(row, (list, tuple)) and len(row) >= 14:
        publication_date = row[0]
        compact_context = row[1]
        extended_context = row[2]
        matched_text = row[3]
        matched_lemmas = row[4]
        month_key = row[5]
        title = row[6]
        author = row[7]
        additional_metadata = row[8]
        left_context = row[9]
        right_context = row[10]
        doc_id = row[11]
        start = row[12]
        end = row[13]

        compact_left = _headless_context_part_036l4g39c(compact_context, 0)
        compact_match = _headless_context_part_036l4g39c(compact_context, 1)
        compact_right = _headless_context_part_036l4g39c(compact_context, 2)
        full_left = _headless_context_part_036l4g39c(extended_context, 0)
        full_match = _headless_context_part_036l4g39c(extended_context, 1)
        full_right = _headless_context_part_036l4g39c(extended_context, 2)

        metadata: dict[str, Any] = {
            "Data publikacji": publication_date,
            "Tytuł": title,
            "Autor": author,
            "month_key": month_key,
            "matched_lemmas": matched_lemmas,
        }
        if isinstance(additional_metadata, dict):
            metadata.update(additional_metadata)

        return SearchHit(
            doc_id=_headless_safe_int_036l4g39c(doc_id),
            start=_headless_safe_int_036l4g39c(start),
            end=_headless_safe_int_036l4g39c(end),
            match_text=_headless_safe_str_036l4g39c(matched_text or compact_match),
            left_context=_headless_safe_str_036l4g39c(left_context or compact_left),
            right_context=_headless_safe_str_036l4g39c(right_context or compact_right),
            extended_left=_headless_safe_str_036l4g39c(full_left),
            extended_match=_headless_safe_str_036l4g39c(full_match or matched_text or compact_match),
            extended_right=_headless_safe_str_036l4g39c(full_right),
            metadata=metadata,
            raw=row,
        )

    return SearchHit(match_text=_headless_safe_str_036l4g39c(row), raw=row)


def normalize_search_results_to_hits(results: Any, *, limit: int | None = None, offset: int = 0) -> list[SearchHit]:
    """Normalize an iterable of current result rows to a list of SearchHit.

    This helper applies offset/limit while normalizing. It deliberately performs
    no GUI work and does not force any specific backend. If a SearchCursor is
    passed, iterating it may materialize rows, so callers should use this helper
    only at explicit materialization boundaries.
    """

    if results is None:
        return []
    offset_i = int(offset or 0)
    if offset_i < 0:
        raise ValueError("offset must be non-negative")
    limit_i = None if limit is None else int(limit)
    if limit_i is not None and limit_i <= 0:
        raise ValueError("limit must be positive or None")

    out: list[SearchHit] = []
    for idx, row in enumerate(results):
        if idx < offset_i:
            continue
        if limit_i is not None and len(out) >= limit_i:
            break
        out.append(normalize_search_result_to_hit(row))
    return out

# END KORPUSUJ_MIGRATION_036L4G39C_NORMALIZE_RESULTS_TO_HITS



def _infer_has_more(results: Any, total_hits: int | None, request: SearchRequest) -> bool | None:
    """Infer has_more for already materialized page results when total_hits is known."""
    if total_hits is None:
        return None
    try:
        returned = len(results or [])
    except Exception:
        return None
    try:
        offset = int(getattr(request, "offset", 0) or 0)
    except Exception:
        offset = 0
    try:
        return (offset + int(returned)) < int(total_hits)
    except Exception:
        return None


def _search_results_metadata(results: Any) -> dict[str, Any]:
    """Return optional result-container metadata for the headless bundle."""
    metadata: dict[str, Any] = {}
    try:
        total_hits_source = getattr(results, "total_hits_source", None)
        if total_hits_source:
            metadata["total_hits_source"] = str(total_hits_source)
    except Exception:
        pass
    try:
        total_hits_counting_strategy = getattr(results, "total_hits_counting_strategy", None)
        if total_hits_counting_strategy:
            metadata["total_hits_counting_strategy"] = str(total_hits_counting_strategy)
    except Exception:
        pass
    return metadata

def run_search_headless(request: SearchRequest, context: SearchBackendContext) -> SearchResultBundle:
    """Run a GUI-independent search using an injected backend adapter."""
    validate_search_request(request)
    corpus_obj = _resolve_corpus_object(request, context)
    adapter = context.find_lemma_context_adapter
    if adapter is None:
        raise HeadlessSearchNotConfiguredError(
            "Headless search adapter is not configured."
        )

    warnings_list: list[str] = []
    adapter_kwargs = {"warnings_list": warnings_list}
    if bool(getattr(adapter, "_supports_search_request", False)):
        adapter_kwargs["search_request"] = request
    results = adapter(
        request.query,
        corpus_obj,
        request.corpus_name,
        int(request.left_context),
        int(request.right_context),
        **adapter_kwargs,
    )
    total_hits = _count_total_hits(results)
    metadata = {
        "adapter": getattr(adapter, "__name__", type(adapter).__name__),
        "scaffold": "036L4G28",
        "pagination": "request_aware_exact",
    }
    metadata.update(_search_results_metadata(results))
    return SearchResultBundle(
        request=request,
        results=results,
        total_hits=total_hits,
        warnings=warnings_list,
        metadata=metadata,
        limit=request.limit,
        offset=int(request.offset),
        has_more=_infer_has_more(results, total_hits, request),
    )
