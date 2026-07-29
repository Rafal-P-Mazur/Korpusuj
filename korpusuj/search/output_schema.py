"""JSON-safe output schema helpers for Korpusuj search results.

The module maps the shared search-result contract to plain Python structures used by CLI, batch processing and exports. Public schema versions remain explicit, and raw internal result objects are excluded by default.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from korpusuj.search.headless import (
    SearchHit,
    SearchMessage,
    SearchRequest,
    SearchResultBundle,
    normalize_search_result_to_hit,
)

SEARCH_OUTPUT_SCHEMA_VERSION = "search-result-v1"

_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation of ``value``."""
    if isinstance(value, _JSON_SCALAR_TYPES):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        try:
            return _json_safe(asdict(value))
        except Exception:
            return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            try:
                safe_key = str(key)
            except Exception:
                safe_key = repr(key)
            out[safe_key] = _json_safe(item)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    try:
        return str(value)
    except Exception:
        return repr(value)


def search_message_to_jsonable(message: SearchMessage | Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert a SearchMessage-like object to a JSON-safe dict."""
    if isinstance(message, SearchMessage):
        return {
            "level": message.level,
            "text": message.text,
            "code": message.code,
            "details": _json_safe(message.details),
        }
    if isinstance(message, Mapping):
        return {
            "level": _json_safe(message.get("level", "")),
            "text": _json_safe(message.get("text", "")),
            "code": _json_safe(message.get("code")),
            "details": _json_safe(message.get("details", {})),
        }
    return {"level": "info", "text": _json_safe(message), "code": None, "details": {}}


def search_request_to_jsonable(request: SearchRequest | Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert a SearchRequest-like object to a JSON-safe dict."""
    if isinstance(request, SearchRequest):
        return {
            "query": request.query,
            "corpus": request.corpus_name,
            "left_context": request.left_context,
            "right_context": request.right_context,
            "sort_option": request.sort_option,
            "date_from": request.date_from,
            "date_to": request.date_to,
            "selected_sense": request.selected_sense,
            "limit": request.limit,
            "offset": request.offset,
            "options": _json_safe(request.options),
        }
    if isinstance(request, Mapping):
        return {
            "query": _json_safe(request.get("query", "")),
            "corpus": _json_safe(request.get("corpus_name", request.get("corpus", ""))),
            "left_context": _json_safe(request.get("left_context")),
            "right_context": _json_safe(request.get("right_context")),
            "sort_option": _json_safe(request.get("sort_option")),
            "date_from": _json_safe(request.get("date_from")),
            "date_to": _json_safe(request.get("date_to")),
            "selected_sense": _json_safe(request.get("selected_sense")),
            "limit": _json_safe(request.get("limit")),
            "offset": _json_safe(request.get("offset", 0)),
            "options": _json_safe(request.get("options", {})),
        }
    return {
        "query": _json_safe(getattr(request, "query", "")),
        "corpus": _json_safe(getattr(request, "corpus_name", getattr(request, "corpus", ""))),
        "left_context": _json_safe(getattr(request, "left_context", None)),
        "right_context": _json_safe(getattr(request, "right_context", None)),
        "sort_option": _json_safe(getattr(request, "sort_option", None)),
        "date_from": _json_safe(getattr(request, "date_from", None)),
        "date_to": _json_safe(getattr(request, "date_to", None)),
        "selected_sense": _json_safe(getattr(request, "selected_sense", None)),
        "limit": _json_safe(getattr(request, "limit", None)),
        "offset": _json_safe(getattr(request, "offset", 0)),
        "options": _json_safe(getattr(request, "options", {})),
    }


def search_hit_to_jsonable(hit: SearchHit) -> dict[str, Any]:
    """Convert a normalized SearchHit to a JSON-safe dict."""
    return {
        "doc_id": hit.doc_id,
        "start_idx": hit.start,
        "end_idx": hit.end,
        "match_text": hit.match_text,
        "left_context": hit.left_context,
        "right_context": hit.right_context,
        "extended_left": hit.extended_left,
        "extended_match": hit.extended_match,
        "extended_right": hit.extended_right,
        "metadata": _json_safe(hit.metadata),
        "raw_available": hit.raw is not None,
    }


def search_result_to_jsonable(row: SearchHit | Any) -> dict[str, Any]:
    """Convert one SearchHit or raw legacy result row to a JSON-safe dict."""
    hit = row if isinstance(row, SearchHit) else normalize_search_result_to_hit(row)
    return search_hit_to_jsonable(hit)


def search_results_to_jsonable(results: Iterable[Any] | Any) -> list[dict[str, Any]]:
    """Convert an iterable of result rows to JSON-safe result dictionaries."""
    if results is None:
        return []
    try:
        iterator = iter(results)
    except TypeError:
        return [search_result_to_jsonable(results)]
    return [search_result_to_jsonable(row) for row in iterator]


def search_bundle_to_jsonable(bundle: SearchResultBundle | Any) -> dict[str, Any]:
    """Convert a SearchResultBundle-like object to public schema v1 dict."""
    request = getattr(bundle, "request", None)
    results = search_results_to_jsonable(getattr(bundle, "results", []))
    request_json = search_request_to_jsonable(request) if request is not None else search_request_to_jsonable({})
    warnings = getattr(bundle, "warnings", []) or []
    messages = getattr(bundle, "messages", []) or []
    return {
        "schema_version": SEARCH_OUTPUT_SCHEMA_VERSION,
        "request": request_json,
        "query": request_json.get("query", ""),
        "corpus": request_json.get("corpus", ""),
        "total_hits": _json_safe(getattr(bundle, "total_hits", None)),
        "returned_hits": len(results),
        "has_more": _json_safe(getattr(bundle, "has_more", None)),
        "limit": _json_safe(getattr(bundle, "limit", request_json.get("limit"))),
        "offset": _json_safe(getattr(bundle, "offset", request_json.get("offset", 0))),
        "warnings": [_json_safe(item) for item in warnings],
        "messages": [search_message_to_jsonable(item) for item in messages],
        "metadata": _json_safe(getattr(bundle, "metadata", {})),
        "statistics_payload_available": getattr(bundle, "statistics_payload", None) is not None,
        "results": results,
    }


__all__ = [
    "SEARCH_OUTPUT_SCHEMA_VERSION",
    "search_message_to_jsonable",
    "search_request_to_jsonable",
    "search_hit_to_jsonable",
    "search_result_to_jsonable",
    "search_results_to_jsonable",
    "search_bundle_to_jsonable",
]
