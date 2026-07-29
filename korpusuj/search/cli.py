# KORPUSUJ_PATCH_157B2_CLI_COLLOCATION_SENTENCE_BOUND_BOOL_FLAG
# -*- coding: utf-8 -*-
"""Command-line interface for corpus search, analytics and export.

The module adapts the shared headless search service to JSON, JSONL, text,
XLSX and CSV output while keeping progress and diagnostics on stderr.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from korpusuj.search.search_service import (
    SearchHit,
    SearchRequest,
    build_search_service_context_from_parquet,
    normalize_search_results_to_hits,
    run_search_service,
)

__all__ = ["main"]


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and nested values to JSON-serializable shapes."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        pass
    if dataclasses.is_dataclass(value):
        try:
            return dataclasses.asdict(value)
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _strip_raw_fields(value: Any) -> Any:
    """Remove ``raw`` fields from normalized CLI output unless requested."""
    if isinstance(value, dict):
        return {
            str(k): _strip_raw_fields(v)
            for k, v in value.items()
            if k != "raw"
        }
    if isinstance(value, list):
        return [_strip_raw_fields(v) for v in value]
    if isinstance(value, tuple):
        return [_strip_raw_fields(v) for v in value]
    return value


def _truncate_str(value: Any, max_chars: int | None) -> Any:
    if max_chars is None:
        return value
    try:
        n = int(max_chars)
    except Exception:
        return value
    if n < 0:
        return value
    if isinstance(value, str) and len(value) > n:
        return value[:n] + "…"
    return value


def _apply_result_output_controls(
    row: Any,
    *,
    include_raw: bool,
    include_extended_context: bool,
    max_context_chars: int | None,
    fields: list[str] | None,
) -> Any:
    """Apply output-only controls to one JSON-able result row."""
    row = _jsonable(row)
    if not include_raw:
        row = _strip_raw_fields(row)
    if not isinstance(row, dict):
        return row

    if not include_extended_context:
        for key in ("extended_left", "extended_match", "extended_right"):
            row.pop(key, None)

    if max_context_chars is not None:
        for key in (
            "left_context",
            "right_context",
            "extended_left",
            "extended_match",
            "extended_right",
        ):
            if key in row:
                row[key] = _truncate_str(row[key], max_context_chars)

    if fields:
        wanted = set(fields)
        row = {k: v for k, v in row.items() if k in wanted}
    return row


def _parse_fields(value: str | None) -> list[str] | None:
    if not value:
        return None
    fields = [part.strip() for part in value.split(",") if part.strip()]
    return fields or None


def _read_query_file(path: str | Path) -> str:
    """Read one CQL query from UTF-8/UTF-8-BOM file and normalize file escapes."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    query = text.strip().lstrip("﻿")
    # Query files are not parsed by the shell, so shell-style escapes can remain
    # literal. Convert common escaped quotes to regular CQL quotes.
    query = query.replace('\\"', '"').replace("\\'", "'")
    return query


def _read_query_list(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    queries: list[str] = []
    for line in text.splitlines():
        query = line.strip().lstrip("\ufeff")
        if not query or query.startswith("#"):
            continue
        query = query.replace('\\"', '"').replace("\\'", "'")
        queries.append(query)
    return queries


def _build_cli_search_request(args: Any, query: str, corpus_name: str) -> SearchRequest:
    return SearchRequest(
        query=query,
        corpus_name=corpus_name,
        left_context=args.left_context,
        right_context=args.right_context,
        sort_option="Alfabetycznie",
        limit=args.limit,
        offset=args.offset,
        options={},
    )


def _run_cli_search(args: Any, context: Any, query: str, corpus_name: str) -> Any:
    request = _build_cli_search_request(args, query, corpus_name)
    return run_search_service(request, context)


def _batch_error_record(query: str, query_index: int, exc: BaseException, corpus_name: str) -> dict[str, Any]:
    message = str(exc) or type(exc).__name__
    return {
        "schema_version": "search-result-v1",
        "batch": True,
        "record_type": "query_error",
        "query_index": query_index,
        "query": query,
        "corpus": corpus_name,
        "error": {"type": type(exc).__name__, "message": message},
        "warnings": [],
        "messages": [{"level": "error", "text": message, "code": "query_error", "details": {}}],
        "metadata": {"cli_json_schema_v1": True},
    }


def _batch_query_data(args: Any, context: Any, query: str, corpus_name: str, query_index: int) -> dict[str, Any]:
    bundle = _run_cli_search(args, context, query, corpus_name)
    data = _schema_bundle_to_jsonable_for_cli(
        bundle,
        include_extended_context=not bool(getattr(args, "no_extended_context", False)),
        max_context_chars=args.max_context_chars,
        fields=_parse_fields(args.fields),
    )
    data = dict(data)
    data["batch"] = True
    data["query_index"] = query_index
    data["ok"] = True
    return data


def _batch_summary_record(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": data.get("schema_version"),
        "batch": True,
        "record_type": "query_summary",
        "query_index": data.get("query_index"),
        "query": data.get("query"),
        "corpus": data.get("corpus"),
        "request": data.get("request"),
        "total_hits": data.get("total_hits"),
        "returned_hits": data.get("returned_hits"),
        "has_more": data.get("has_more"),
        "limit": data.get("limit"),
        "offset": data.get("offset"),
        "warnings": data.get("warnings", []),
        "messages": data.get("messages", []),
        "metadata": data.get("metadata", {}),
        "results": [],
    }


def _print_batch_schema_jsonl(args: Any, context: Any, corpus_name: str, queries: list[str]) -> str:
    lines: list[str] = []
    for query_index, query in enumerate(queries, start=1):
        try:
            data = _batch_query_data(args, context, query, corpus_name, query_index)
        except Exception as exc:
            if not bool(getattr(args, "continue_on_error", False)):
                raise
            lines.append(json.dumps(_batch_error_record(query, query_index, exc, corpus_name), ensure_ascii=False, separators=(",", ":")))
            continue
        base = {
            "schema_version": data.get("schema_version"),
            "batch": True,
            "record_type": "result",
            "query_index": query_index,
            "query": data.get("query"),
            "corpus": data.get("corpus"),
            "request": data.get("request"),
            "total_hits": data.get("total_hits"),
            "returned_hits": data.get("returned_hits"),
            "has_more": data.get("has_more"),
            "limit": data.get("limit"),
            "offset": data.get("offset"),
            "warnings": data.get("warnings", []),
            "messages": data.get("messages", []),
            "metadata": data.get("metadata", {}),
        }
        results = data.get("results") or []
        if not results:
            lines.append(json.dumps(_batch_summary_record(data), ensure_ascii=False, separators=(",", ":")))
            continue
        for result_index, result in enumerate(results, start=1):
            row = dict(base)
            row["result_index"] = result_index
            row["result"] = result
            lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def _print_batch_schema_json(args: Any, context: Any, corpus_name: str, queries: list[str]) -> str:
    query_results: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries, start=1):
        try:
            query_results.append(_batch_query_data(args, context, query, corpus_name, query_index))
        except Exception as exc:
            if not bool(getattr(args, "continue_on_error", False)):
                raise
            err = _batch_error_record(query, query_index, exc, corpus_name)
            err["ok"] = False
            query_results.append(err)
    envelope = {
        "schema_version": "search-result-v1",
        "batch": True,
        "corpus": corpus_name,
        "query_count": len(queries),
        "limit": args.limit,
        "offset": args.offset,
        "warnings": [],
        "messages": [],
        "metadata": {"cli_json_schema_v1": True, "batch_query_list": True},
        "query_results": query_results,
    }
    if args.pretty:
        return json.dumps(envelope, ensure_ascii=False, indent=2)
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def _print_batch_text(args: Any, context: Any, corpus_name: str, queries: list[str]) -> str:
    sections: list[str] = []
    for query_index, query in enumerate(queries, start=1):
        try:
            bundle = _run_cli_search(args, context, query, corpus_name)
        except Exception as exc:
            if not bool(getattr(args, "continue_on_error", False)):
                raise
            sections.append(f"Query {query_index}: {query}\nERROR: {type(exc).__name__}: {exc}")
            continue
        rendered = _print_text(bundle, max_context_chars=args.max_context_chars, fields=_parse_fields(args.fields))
        sections.append(f"Query {query_index}: {query}\n{rendered}")
    return "\n\n".join(sections)


def _run_query_list_cli(args: Any) -> int:
    if _cli_output_is_table_format(getattr(args, "format", None)):
        print("--query-list with --format xlsx/csv is not supported in this version; use json/jsonl/text or run single-query exports.", file=sys.stderr)
        return 2
    queries = _read_query_list(args.query_list)
    if not queries:
        raise ValueError("--query-list did not contain any non-empty CQL queries")
    corpus_name = str(args.corpus_name or Path(args.corpus_path).stem)
    context = build_search_service_context_from_parquet(
        args.corpus_path,
        corpus_name=corpus_name,
        limit=args.limit,
        normalize_hits=not args.raw,
        normalize_fn=normalize_search_results_to_hits,
        full_context_size=args.full_context_size,
        candidate_max_docs=args.candidate_max_docs,
        candidate_stream_batch_docs=args.candidate_stream_batch_docs,
        config={},
    )
    if args.format == "json":
        text = _print_batch_schema_json(args, context, corpus_name, queries)
    elif args.format == "jsonl":
        text = _print_batch_schema_jsonl(args, context, corpus_name, queries)
    else:
        text = _print_batch_text(args, context, corpus_name, queries)
    _write_output(text, args.output)
    return 0



# KORPUSUJ_151D_CLI_EXTENDED_CONTEXT_RESOLUTION
# Normal CLI path for resolving lazy fulltext refs when extended context is requested.
# This is intentionally not a monkeypatch. Existing 137d/137e sanitizers remain as
# a safety net, but CLI should first try to turn lazy refs into real extended text.
def _resolve_cli_extended_context_row(row):
    """Return one row with lazy extended/fulltext context resolved when possible.

    Uses backend/cursor helpers only. Never imports engine.py and never exposes
    SearchIndex/lazy reference objects. If resolution fails, returns the original
    row so the existing JSON sanitizers can still omit unsafe lazy values.
    """
    try:
        from korpusuj.search.cursor import (
            is_lazy_fulltext_ref_111,
            resolve_lazy_fulltext_ref_111,
            resolve_result_row_fulltext_111,
        )
    except Exception:
        return row

    try:
        from korpusuj.search.headless import normalize_search_result_to_hit as _normalize_hit_151d
    except Exception:
        _normalize_hit_151d = globals().get("normalize_search_result_to_hit")

    _SearchHit_151d = globals().get("SearchHit")

    def _is_resolved_parts_151d(value):
        try:
            return isinstance(value, (list, tuple)) and len(value) >= 3 and not is_lazy_fulltext_ref_111(value)
        except Exception:
            return False

    def _safe_part_151d(parts, idx):
        try:
            value = parts[idx]
        except Exception:
            return ""
        try:
            return "" if value is None else str(value)
        except Exception:
            return repr(value)

    try:
        if _SearchHit_151d is not None and isinstance(row, _SearchHit_151d):
            raw = getattr(row, "raw", None)
            if isinstance(raw, (tuple, list)) and len(raw) > 2:
                try:
                    resolved_raw = resolve_result_row_fulltext_111(raw)
                except Exception:
                    resolved_raw = raw
                if resolved_raw is not raw and callable(_normalize_hit_151d):
                    try:
                        resolved_hit = _normalize_hit_151d(resolved_raw)
                        metadata = getattr(resolved_hit, "metadata", None) or getattr(row, "metadata", {})
                        return _SearchHit_151d(
                            doc_id=getattr(resolved_hit, "doc_id", getattr(row, "doc_id", None)),
                            start=getattr(resolved_hit, "start", getattr(row, "start", None)),
                            end=getattr(resolved_hit, "end", getattr(row, "end", None)),
                            match_text=getattr(resolved_hit, "match_text", getattr(row, "match_text", "")),
                            left_context=getattr(resolved_hit, "left_context", getattr(row, "left_context", "")),
                            right_context=getattr(resolved_hit, "right_context", getattr(row, "right_context", "")),
                            extended_left=getattr(resolved_hit, "extended_left", getattr(row, "extended_left", "")),
                            extended_match=getattr(resolved_hit, "extended_match", getattr(row, "extended_match", "")),
                            extended_right=getattr(resolved_hit, "extended_right", getattr(row, "extended_right", "")),
                            metadata=metadata or {},
                            raw=resolved_raw,
                        )
                    except Exception:
                        pass

            for attr in ("extended_left", "extended_match", "extended_right"):
                try:
                    value = getattr(row, attr)
                except Exception:
                    continue
                try:
                    if is_lazy_fulltext_ref_111(value):
                        parts = resolve_lazy_fulltext_ref_111(value, None)
                        if _is_resolved_parts_151d(parts):
                            return _SearchHit_151d(
                                doc_id=getattr(row, "doc_id", None),
                                start=getattr(row, "start", None),
                                end=getattr(row, "end", None),
                                match_text=getattr(row, "match_text", ""),
                                left_context=getattr(row, "left_context", ""),
                                right_context=getattr(row, "right_context", ""),
                                extended_left=_safe_part_151d(parts, 0),
                                extended_match=_safe_part_151d(parts, 1),
                                extended_right=_safe_part_151d(parts, 2),
                                metadata=getattr(row, "metadata", {}) or {},
                                raw=getattr(row, "raw", None),
                            )
                except Exception:
                    continue
            return row

        if isinstance(row, (tuple, list)):
            try:
                return resolve_result_row_fulltext_111(row)
            except Exception:
                return row

        if isinstance(row, dict):
            out = dict(row)
            for key in ("extended_left", "extended_match", "extended_right", "raw"):
                value = out.get(key)
                try:
                    if is_lazy_fulltext_ref_111(value):
                        parts = resolve_lazy_fulltext_ref_111(value, row.get("context"))
                        if _is_resolved_parts_151d(parts):
                            out["extended_left"] = _safe_part_151d(parts, 0)
                            out["extended_match"] = _safe_part_151d(parts, 1)
                            out["extended_right"] = _safe_part_151d(parts, 2)
                            break
                except Exception:
                    continue
            return out
    except Exception:
        return row

    return row


def _resolve_cli_extended_context_rows(rows):
    """Resolve lazy extended/fulltext context for the already materialized rows."""
    if rows is None:
        return []
    try:
        return [_resolve_cli_extended_context_row(row) for row in rows]
    except Exception:
        try:
            return list(rows)
        except Exception:
            return rows
# END KORPUSUJ_151D_CLI_EXTENDED_CONTEXT_RESOLUTION

def _bundle_to_dict(
    bundle: Any,
    *,
    include_request: bool = False,
    include_raw: bool = False,
    include_extended_context: bool = True,
    max_context_chars: int | None = None,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    results = getattr(bundle, "results", []) or []
    if include_extended_context:
        results = _resolve_cli_extended_context_rows(results)
    data: dict[str, Any] = {
        "total_hits": getattr(bundle, "total_hits", None),
        "limit": getattr(bundle, "limit", None),
        "offset": getattr(bundle, "offset", None),
        "has_more": getattr(bundle, "has_more", None),
        "warnings": _jsonable(getattr(bundle, "warnings", []) or []),
        "messages": _jsonable(getattr(bundle, "messages", []) or []),
        "metadata": _jsonable(getattr(bundle, "metadata", {}) or {}),
        "results": [
            _apply_result_output_controls(
                row,
                include_raw=include_raw,
                include_extended_context=include_extended_context,
                max_context_chars=max_context_chars,
                fields=fields,
            )
            for row in results
        ],
    }
    if include_request:
        data["request"] = _jsonable(getattr(bundle, "request", None))
    return data


def _result_text_line(row: Any, idx: int) -> str:
    if isinstance(row, SearchHit):
        left = row.left_context or ""
        right = row.right_context or ""
        meta = row.metadata or {}
        date = meta.get("Data publikacji", "")
        author = meta.get("Autor", "")
        return f"{idx}. {date} {author} | {left}[{row.match_text}]{right}"
    if isinstance(row, dict):
        match = row.get("match_text") or row.get("match") or ""
        left = row.get("left_context", "")
        right = row.get("right_context", "")
        return f"{idx}. {left}[{match}]{right}"
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        date = row[0]
        compact = row[1]
        if isinstance(compact, (list, tuple)) and len(compact) >= 3:
            return f"{idx}. {date} | {compact[0]}[{compact[1]}]{compact[2]}"
    return f"{idx}. {row!r}"


def _print_json(
    bundle: Any,
    *,
    include_request: bool,
    pretty: bool,
    include_raw: bool,
    include_extended_context: bool,
    max_context_chars: int | None,
    fields: list[str] | None,
) -> str:
    data = _bundle_to_dict(
        bundle,
        include_request=include_request,
        include_raw=include_raw,
        include_extended_context=include_extended_context,
        max_context_chars=max_context_chars,
        fields=fields,
    )
    if pretty:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _print_jsonl(
    bundle: Any,
    *,
    include_raw: bool,
    include_extended_context: bool,
    max_context_chars: int | None,
    fields: list[str] | None,
) -> str:
    lines: list[str] = []
    results = getattr(bundle, "results", []) or []
    if include_extended_context:
        results = _resolve_cli_extended_context_rows(results)
    for row in results:
        row_data = _apply_result_output_controls(
            row,
            include_raw=include_raw,
            include_extended_context=include_extended_context,
            max_context_chars=max_context_chars,
            fields=fields,
        )
        lines.append(json.dumps(row_data, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)



# KORPUSUJ_151E_CLI_JSON_JSONL_SCHEMA_V1
# CLI JSON/JSONL now use the explicit schema-v1 mapper. This is not a monkeypatch:
# main() calls these helpers directly for --format json/jsonl. The text format remains
# human-readable/debug output.
def _drop_schema_result_extended_context(value):
    """Remove result extended context keys recursively from schema-v1 output.

    Keep diagnostic/schema metadata such as ``extended_context_included`` and
    ``extended_context_resolution_attempted``. Only the actual result payload
    fields are removed when --no-extended-context is used.
    """
    drop_keys = {"extended_left", "extended_match", "extended_right"}
    if isinstance(value, dict):
        return {
            str(k): _drop_schema_result_extended_context(v)
            for k, v in value.items()
            if str(k) not in drop_keys
        }
    if isinstance(value, list):
        return [_drop_schema_result_extended_context(v) for v in value]
    if isinstance(value, tuple):
        return [_drop_schema_result_extended_context(v) for v in value]
    return value


def _truncate_schema_context(value, max_chars):
    """Truncate schema-v1 string values in context/match fields."""
    if max_chars is None:
        return value
    try:
        n = int(max_chars)
    except Exception:
        return value
    if n < 0:
        return value

    context_keys = {
        "match_text",
        "left_context",
        "right_context",
        "extended_left",
        "extended_match",
        "extended_right",
    }

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if str(key) in context_keys and isinstance(item, str) and len(item) > n:
                out[str(key)] = item[:n] + "…"
            else:
                out[str(key)] = _truncate_schema_context(item, n)
        return out
    if isinstance(value, list):
        return [_truncate_schema_context(item, n) for item in value]
    if isinstance(value, tuple):
        return [_truncate_schema_context(item, n) for item in value]
    return value


def _filter_schema_result_fields(data, fields):
    """Apply --fields to schema-v1 result objects only; keep envelope intact."""
    if not fields:
        return data
    try:
        wanted = {str(field) for field in fields}
    except Exception:
        return data
    try:
        results = data.get("results") or []
    except Exception:
        return data
    filtered = []
    for result in results:
        if isinstance(result, dict):
            filtered.append({key: value for key, value in result.items() if str(key) in wanted})
        else:
            filtered.append(result)
    try:
        data = dict(data)
        data["results"] = filtered
        data["returned_hits"] = len(filtered)
    except Exception:
        pass
    return data


def _bundle_with_cli_schema_results(bundle, results):
    """Return bundle-like object with replacement results without mutating when possible."""
    try:
        import dataclasses as _dataclasses_151e
        if _dataclasses_151e.is_dataclass(bundle):
            return _dataclasses_151e.replace(bundle, results=results)
    except Exception:
        pass
    try:
        import copy as _copy_151e
        new_bundle = _copy_151e.copy(bundle)
        try:
            setattr(new_bundle, "results", results)
        except Exception:
            return bundle
        return new_bundle
    except Exception:
        return bundle




# KORPUSUJ_PATCH_156B_CLI_COLLOCATIONS_ANALYTICS_MVP
_CLI_ANALYTICS_CONFIG = None


def _parse_cli_collocate_filter_group(text):
    if text is None:
        return {}
    raw = str(text).strip()
    if not raw:
        return {}
    allowed = {"upos", "pos", "tag"}
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --collocate-filter segment {part!r}; expected key=value")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in allowed:
            raise ValueError(f"Invalid --collocate-filter key {key!r}; allowed: upos,pos,tag")
        if not value:
            raise ValueError(f"Empty value in --collocate-filter for key {key!r}")
        out[key] = value
    return out


def _parse_cli_collocate_filter_groups(values):
    return [_parse_cli_collocate_filter_group(v) for v in (values or []) if str(v or "").strip()]


def _normalise_cli_collocate_concordance_values(values):
    """Return explicit selected-collocate labels from CLI args."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out = []
    try:
        iterable = list(values or [])
    except Exception:
        iterable = [values]
    for value in iterable:
        text = str(value or "").strip()
        if text:
            out.append(text)
    return out


def _normalise_cli_colloc_mode(value):
    return {"linear": "Liniowe", "syntactic": "Składniowe"}.get(str(value or "linear").lower(), value)


def _normalise_cli_colloc_form(value):
    return {"base": "Lemat (base)", "orth": "Token (orth)"}.get(str(value or "base").lower(), value)


def _normalise_cli_colloc_sort(value):
    value = str(value or "log-dice").lower()
    return {"log-likelihood": "Log-Likelihood", "mi": "MI Score", "mi-score": "MI Score", "t-score": "T-score", "log-dice": "Log-Dice"}.get(value, value)

# KORPUSUJ_PATCH_163B_CLI_SYNTACTIC_COLLOCATION_DIR_DEPREL_FLAGS_NORMALIZERS

def _normalise_cli_colloc_syn_dir(value):
    raw = str(value or "dependent").strip().lower()
    mapping = {
        "dependent": "Podrzędnik",
        "dependents": "Podrzędnik",
        "podrzędnik": "Podrzędnik",
        "podrzednik": "Podrzędnik",
        "head": "Nadrzędnik",
        "heads": "Nadrzędnik",
        "governor": "Nadrzędnik",
        "parent": "Nadrzędnik",
        "nadrzędnik": "Nadrzędnik",
        "nadrzednik": "Nadrzędnik",
        "both": "Wszystkie",
        "all": "Wszystkie",
        "wszystkie": "Wszystkie",
    }
    return mapping.get(raw, str(value or "dependent"))


def _public_cli_colloc_syn_dir(value):
    raw = str(value or "Podrzędnik").strip().lower()
    if raw in {"podrzędnik", "podrzednik", "dependent", "dependents"}:
        return "dependent"
    if raw in {"nadrzędnik", "nadrzednik", "head", "heads", "governor", "parent"}:
        return "head"
    if raw in {"wszystkie", "both", "all"}:
        return "both"
    return str(value or "dependent")


def _normalise_cli_colloc_deprel(value):
    raw = str(value or "Wszystkie").strip()
    if not raw:
        return "Wszystkie"
    if raw.lower() in {"all", "both", "any", "wszystkie", "*"}:
        return "Wszystkie"
    return raw


def _public_cli_colloc_deprel(value):
    raw = str(value or "Wszystkie").strip()
    return "all" if raw.lower() in {"", "wszystkie", "all", "any", "*"} else raw
# END KORPUSUJ_PATCH_163B_CLI_SYNTACTIC_COLLOCATION_DIR_DEPREL_FLAGS_NORMALIZERS


def _set_cli_analytics_config_from_args(args):
    global _CLI_ANALYTICS_CONFIG
    analytics = str(getattr(args, "analytics", "none") or "none").lower()
    include_analytics_payload_requested = (analytics == "collocations")
    requested_concordance_of = str(getattr(args, "concordance_of", "query") or "query").lower()
    collocate_concordance_values = _normalise_cli_collocate_concordance_values(getattr(args, "collocate_concordance", None))
    collocation_computation_reasons = []
    if include_analytics_payload_requested:
        collocation_computation_reasons.append("analytics_payload")
    if requested_concordance_of == "collocates":
        collocation_computation_reasons.append("concordance_of_collocates")
    if collocate_concordance_values:
        collocation_computation_reasons.append("collocate_concordance")
    should_compute_collocations = bool(collocation_computation_reasons)
    if not should_compute_collocations:
        _CLI_ANALYTICS_CONFIG = None
        return
    if analytics in {"", "none"}:
        # Internal collocation computation is needed for result mode, but analytics payload was not requested.
        analytics = "collocations"
    colloc_limit_value = getattr(args, "colloc_limit", None)
    if colloc_limit_value is not None and int(colloc_limit_value) < 0:
        raise ValueError("--colloc-limit must be non-negative")
    _CLI_ANALYTICS_CONFIG = {
        "analytics": analytics,
        "analytics_only": bool(getattr(args, "analytics_only", False)),
        "concordance_of": getattr(args, "concordance_of", "query"),
        "analytics_scope": str(getattr(args, "analytics_scope", "all-matches") or "all-matches"),
        "corpus_path": str(getattr(args, "corpus_path", "") or ""),
        "corpus_name": str(getattr(args, "corpus_name", "") or ""),
        "full_context_size": int(getattr(args, "full_context_size", 250) or 250),
        "candidate_max_docs": int(getattr(args, "candidate_max_docs", 3000) or 3000),
        "candidate_stream_batch_docs": int(getattr(args, "candidate_stream_batch_docs", 256) or 256),
        "colloc_mode": _normalise_cli_colloc_mode(getattr(args, "colloc_mode", "linear")),
        "colloc_form": _normalise_cli_colloc_form(getattr(args, "colloc_form", "base")),
        "colloc_left_span": int(getattr(args, "colloc_left_span", 5) or 5),
        "colloc_right_span": int(getattr(args, "colloc_right_span", 5) or 5),
        "colloc_min_freq": int(getattr(args, "colloc_min_freq", 1) or 1),
        "colloc_min_range": int(getattr(args, "colloc_min_range", 1) or 1),
        "colloc_sort": _normalise_cli_colloc_sort(getattr(args, "colloc_sort", "log-dice")),
        "colloc_syn_dir": _normalise_cli_colloc_syn_dir(getattr(args, "colloc_syn_dir", "dependent")),
        "colloc_deprel": _normalise_cli_colloc_deprel(getattr(args, "colloc_deprel", "Wszystkie")),
        "colloc_limit": getattr(args, "colloc_limit", None),
        "result_limit": getattr(args, "limit", None),
        "offset": getattr(args, "offset", 0),
        "left_context": getattr(args, "left_context", 10),
        "right_context": getattr(args, "right_context", 10),
        "collocate_filter_groups": _parse_cli_collocate_filter_groups(getattr(args, "collocate_filter", None)),
        "collocate_concordance": _normalise_cli_collocate_concordance_values(getattr(args, "collocate_concordance", None)),
        "colloc_sentence_bound": getattr(args, "colloc_sentence_bound", "true"),
    }
    _CLI_ANALYTICS_CONFIG.update({
        "include_analytics_payload": include_analytics_payload_requested,
        "needs_collocation_computation": should_compute_collocations,
        "collocation_computation_reason": list(collocation_computation_reasons),
        "analytics_payload_included": include_analytics_payload_requested,
        "collocate_concordance": collocate_concordance_values,
    })


def _collocate_filter_public_shape(config):
    return {"groups": list(config.get("collocate_filter_groups") or []), "group_semantics": "OR", "within_group_semantics": "AND", "tag_match": "tail_prefix_ordered"}


def _collocations_public_parameters(config):
    return {
        "mode": "syntactic" if config.get("colloc_mode") == "Składniowe" else "linear",
        "form": "orth" if config.get("colloc_form") == "Token (orth)" else "base",
        "left_span": int(config.get("colloc_left_span", 5) or 5),
        "right_span": int(config.get("colloc_right_span", 5) or 5),
        "sentence_bound": (str(config.get("colloc_sentence_bound", "true") or "true").lower() in {"1", "true", "yes", "y", "on"}),
        "min_frequency": int(config.get("colloc_min_freq", 1) or 1),
        "min_range": int(config.get("colloc_min_range", 1) or 1),
        "sort": str(config.get("colloc_sort", "Log-Dice")),
        "limit": (None if config.get("colloc_limit") is None else int(config.get("colloc_limit"))),
        "syntactic_direction": _public_cli_colloc_syn_dir(config.get("colloc_syn_dir", "Podrzędnik")),
        "deprel": _public_cli_colloc_deprel(config.get("colloc_deprel", "Wszystkie")),
    }


def _analytics_unavailable(reason, *, config):
    return {
        "requested": ["collocations"],
        "included": [],
        "unavailable_reasons": {"collocations": reason or "unavailable"},
        "collocations": {
            "available": False,
            "kind": "collocation_table",
            "source": {"scope": config.get("analytics_scope", "all-matches")},
            "parameters": _collocations_public_parameters(config),
            "collocate_filter": _collocate_filter_public_shape(config),
            "rows": [],
            "warnings": [reason] if reason else [],
        },
    }


def _public_collocation_rows(table, limit=None):
    rows = []
    for row in getattr(table, "rows", []) or []:
        rows.append({
            "rank": int(getattr(row, "rank", 0) or 0),
            "collocate": str(getattr(row, "colloc", "")),
            "cooccurrences": int(getattr(row, "fnc", 0) or 0),
            "collocate_frequency": int(getattr(row, "fc", 0) or 0),
            "log_likelihood": float(getattr(row, "ll", 0.0) or 0.0),
            "mi": float(getattr(row, "mi", 0.0) or 0.0),
            "t_score": float(getattr(row, "t", 0.0) or 0.0),
            "log_dice": float(getattr(row, "log_dice", 0.0) or 0.0),
        })
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            limit = 0
        return rows[:limit]
    return rows


def _schema_results_to_collocation_result_dicts(schema_results):
    out = []
    for r in schema_results or []:
        if not isinstance(r, dict):
            continue
        doc_id = r.get("doc_id")
        start_idx = r.get("start_idx")
        end_idx = r.get("end_idx")
        if doc_id is None or start_idx is None or end_idx is None:
            continue
        out.append({"row_idx": doc_id, "doc_id": doc_id, "start_idx": start_idx, "end_idx": end_idx})
    return out


def _cli_colloc_plain_list(value):
    """Convert list/tuple/numpy/pandas array-like values without boolean-testing them."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            return [converted]
    except Exception:
        pass
    try:
        return list(value)
    except Exception:
        return []


def _build_minimal_inverted_index_from_dataframe(df):
    from collections import Counter
    base_tf = Counter(); orth_tf = Counter(); total_tokens = 0
    try:
        iterator = df.itertuples(index=False)
    except Exception:
        iterator = []
    for row in iterator:
        lemmas = _cli_colloc_plain_list(getattr(row, "lemmas", None))
        tokens = _cli_colloc_plain_list(getattr(row, "tokens", None))
        total_tokens += max(len(lemmas), len(tokens))
        for x in lemmas: base_tf[str(x)] += 1
        for x in tokens: orth_tf[str(x)] += 1
    return {"base_tf": dict(base_tf), "orth_tf": dict(orth_tf), "total_tokens": int(total_tokens or 1)}




def _schema_or_hit_to_collocation_result_dict(item):
    """Return {row_idx, doc_id, start_idx, end_idx} for schema dict/SearchHit/legacy tuple."""
    if item is None:
        return None
    if isinstance(item, dict):
        doc_id = item.get("doc_id", item.get("row_idx"))
        start_idx = item.get("start_idx", item.get("start"))
        end_idx = item.get("end_idx", item.get("end"))
        if doc_id is None or start_idx is None or end_idx is None:
            return None
        return {"row_idx": doc_id, "doc_id": doc_id, "start_idx": start_idx, "end_idx": end_idx}
    if isinstance(item, (list, tuple)) and len(item) >= 14:
        return {"row_idx": item[11], "doc_id": item[11], "start_idx": item[12], "end_idx": item[13]}
    doc_id = getattr(item, "doc_id", None)
    start_idx = getattr(item, "start_idx", getattr(item, "start", None))
    end_idx = getattr(item, "end_idx", getattr(item, "end", None))
    if doc_id is None or start_idx is None or end_idx is None:
        raw = getattr(item, "raw", None)
        if raw is not None and raw is not item:
            return _schema_or_hit_to_collocation_result_dict(raw)
        return None
    return {"row_idx": doc_id, "doc_id": doc_id, "start_idx": start_idx, "end_idx": end_idx}


def _bundle_results_to_collocation_result_dicts(bundle):
    rows = []
    for item in list(getattr(bundle, "results", []) or []):
        converted = _schema_or_hit_to_collocation_result_dict(item)
        if converted is not None:
            rows.append(converted)
    return rows


def _build_cli_analytics_context(config, corpus_name):
    corpus_path = config.get("corpus_path")
    if not corpus_path:
        raise ValueError("missing_corpus_path")
    path_obj = Path(corpus_path)
    attempts = [
        lambda: build_search_service_context_from_parquet(
            path_obj,
            corpus_name=corpus_name,
            full_context_size=int(config.get("full_context_size", 250) or 250),
            candidate_max_docs=int(config.get("candidate_max_docs", 3000) or 3000),
            candidate_stream_batch_docs=int(config.get("candidate_stream_batch_docs", 256) or 256),
        ),
        lambda: build_search_service_context_from_parquet(path_obj, corpus_name=corpus_name),
        lambda: build_search_service_context_from_parquet(path_obj),
        lambda: build_search_service_context_from_parquet(str(path_obj), corpus_name=corpus_name),
        lambda: build_search_service_context_from_parquet(str(path_obj)),
    ]
    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("could_not_build_cli_analytics_context")


def _run_cli_all_matches_analytics_search(data, bundle, config):
    """Run a complete analytics search independent of the display page."""
    original_request = getattr(bundle, "request", None)
    corpus_name = (
        config.get("corpus_name")
        or data.get("corpus")
        or getattr(original_request, "corpus_name", None)
        or Path(str(config.get("corpus_path") or "corpus")).stem
    )
    request_options = dict(getattr(original_request, "options", {}) or {})
    request_options["client"] = "cli"
    request_options["analytics_internal_run"] = True
    request_options["analytics_scope"] = "all_matches"
    analytics_request = SearchRequest(
        query=data.get("query") or getattr(original_request, "query", ""),
        corpus_name=corpus_name,
        left_context=int(getattr(original_request, "left_context", 10) or 10),
        right_context=int(getattr(original_request, "right_context", 10) or 10),
        sort_option=getattr(original_request, "sort_option", None),
        date_from=getattr(original_request, "date_from", None),
        date_to=getattr(original_request, "date_to", None),
        selected_sense=getattr(original_request, "selected_sense", None),
        limit=None,
        offset=0,
        options=request_options,
    )
    context = _build_cli_analytics_context(config, corpus_name)
    analytics_bundle = run_search_service(analytics_request, context)
    analyzed_count = len(list(getattr(analytics_bundle, "results", []) or []))
    total_hits = getattr(analytics_bundle, "total_hits", analyzed_count)
    try:
        total_hits = int(total_hits)
    except Exception:
        total_hits = analyzed_count
    if analyzed_count != total_hits:
        raise RuntimeError(
            "all-matches analytics was incomplete: "
            f"materialized={analyzed_count}, total_hits={total_hits}"
        )
    return analytics_bundle, {
        "scope": "all-matches",
        "match_count": analyzed_count,
        "total_hits": total_hits,
        "returned_results_count": len(list((data or {}).get("results") or [])),
        "analysis_complete": True,
        "analysis_cap": None,
        "partial_reason": None,
    }



def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if hasattr(value, "tolist"):
            return list(value.tolist())
    except Exception:
        pass
    return [value]


def _row_by_index(df, row_idx):
    try:
        return df.iloc[int(row_idx)]
    except Exception:
        try:
            return df.loc[int(row_idx)]
        except Exception:
            return None


def _row_value(row_data, key, default=None):
    if row_data is None:
        return default
    try:
        if isinstance(row_data, dict):
            return row_data.get(key, default)
    except Exception:
        pass
    try:
        return row_data[key]
    except Exception:
        return getattr(row_data, key, default)


def _join_tokens(tokens, start_idx, end_idx):
    try:
        start = max(0, int(start_idx)); end = max(start, int(end_idx))
        parts = [str(x) for x in list(tokens or [])[start:end] if x is not None]
        return " ".join(parts)
    except Exception:
        return ""


def _collocate_occurrence_to_schema_result(occ, df, *, left_context=10, right_context=10):
    row_idx = int(getattr(occ, "source_row_idx"))
    colloc_start = int(getattr(occ, "collocate_idx"))
    colloc_end = int(getattr(occ, "collocate_end_idx", colloc_start + 1))
    row_data = _row_by_index(df, row_idx)
    tokens = _as_list(_row_value(row_data, "tokens", []))
    left_start = max(0, colloc_start - int(left_context or 0))
    right_end = min(len(tokens), colloc_end + int(right_context or 0)) if tokens else colloc_end
    match_text = getattr(occ, "collocate_form", None) or getattr(occ, "collocate", "")
    metadata = {
        "concordance_kind": "collocate",
        "collocate": getattr(occ, "collocate", None),
        "collocate_rank": getattr(occ, "collocate_rank", None),
        "source_match_start_idx": getattr(occ, "source_match_start_idx", None),
        "source_match_end_idx": getattr(occ, "source_match_end_idx", None),
        "source_match_text": getattr(occ, "source_match_text", None),
        "collocate_idx": colloc_start,
        "collocate_end_idx": colloc_end,
        "direction": getattr(occ, "direction", None),
        "distance": getattr(occ, "distance", None),
        "deprel": getattr(occ, "deprel", None),
    }
    return {
        "doc_id": row_idx,
        "start_idx": colloc_start,
        "end_idx": colloc_end,
        "match_text": str(match_text or ""),
        "left_context": _join_tokens(tokens, left_start, colloc_start),
        "right_context": _join_tokens(tokens, colloc_end, right_end),
        "metadata": metadata,
        "raw_available": True,
    }
def _compute_cli_collocations_analytics(data, bundle):
    config = dict(_CLI_ANALYTICS_CONFIG or {})
    analytics_kind = str(config.get("analytics", "none") or "none").lower()
    if analytics_kind not in {"collocations", "all"}:
        return None
    scope = str(config.get("analytics_scope", "all-matches") or "all-matches")
    source_info = None
    if scope == "returned-results":
        schema_results = list((data or {}).get("results") or [])
        colloc_results = _schema_results_to_collocation_result_dicts(schema_results)
        source_info = {
            "scope": "returned_results",
            "match_query": data.get("query"),
            "match_count": len(colloc_results),
            "total_hits": data.get("total_hits"),
            "returned_results_count": len(schema_results),
            "analysis_complete": None,
            "analysis_cap": None,
            "partial_reason": None,
        }
    elif scope == "all-matches":
        try:
            analytics_bundle, source_info = _run_cli_all_matches_analytics_search(data, bundle, config)
            colloc_results = _bundle_results_to_collocation_result_dicts(analytics_bundle)
            source_info["match_query"] = data.get("query")
            source_info["match_count"] = len(colloc_results)
        except Exception as exc:
            return _analytics_unavailable(f"collocations_all_matches_runtime_error: {type(exc).__name__}: {exc}", config=config)
    else:
        return _analytics_unavailable(f"unsupported_analytics_scope: {scope}", config=config)
    if not colloc_results:
        return _analytics_unavailable("no_results_with_doc_and_token_offsets_for_collocations", config=config)
    try:
        import pandas as pd
        from korpusuj.search.collocations import CollocationOptions, CollocateFilterGroup, compute_collocations
        df = pd.read_parquet(config.get("corpus_path"))
        filter_groups = [CollocateFilterGroup(**g) for g in (config.get("collocate_filter_groups") or [])]
        options = CollocationOptions(
            mode=config.get("colloc_mode", "Liniowe"), form_mode=config.get("colloc_form", "Lemat (base)"),
            sort_mode=config.get("colloc_sort", "Log-Dice"), min_freq=int(config.get("colloc_min_freq", 1) or 1),
            min_range=int(config.get("colloc_min_range", 1) or 1), l_span=int(config.get("colloc_left_span", 5) or 5),
            r_span=int(config.get("colloc_right_span", 5) or 5),
            use_sentence_bound=(str(config.get("colloc_sentence_bound", "true") or "true").lower() in {"1", "true", "yes", "y", "on"}),
            collocate_filter_groups=filter_groups,
            syn_dir=config.get("colloc_syn_dir", "Podrzędnik"),
            deprel_filter=config.get("colloc_deprel", "Wszystkie"),
        )
        table = compute_collocations(colloc_results, df, _build_minimal_inverted_index_from_dataframe(df), options, feat_mapping={})
        warnings = []

        _collocate_concordance_private = {}
        if str(config.get("concordance_of", "query") or "query").lower() == "collocates":
            if scope != "all-matches":
                warnings.append("concordance_of_collocates_requires_all_matches_scope")
            else:
                try:
                    from korpusuj.search.collocations import collect_collocate_occurrences
                    selected_collocation_rows = list(getattr(table, "rows", []) or [])
                    collocation_row_limit = config.get("colloc_limit")
                    if collocation_row_limit is not None:
                        selected_collocation_rows = selected_collocation_rows[: int(collocation_row_limit)]
                    selected_collocate_labels = list(config.get("collocate_concordance") or [])
                    if selected_collocate_labels:
                        selected_collocates_for_occurrences = [{"collocate": label, "rank": None} for label in selected_collocate_labels]
                        collocate_concordance_occurrence_selection_mode = "explicit"
                    else:
                        selected_collocates_for_occurrences = public_rows if "public_rows" in locals() else []
                        collocate_concordance_occurrence_selection_mode = "top_collocates"
                    selected_collocates=selected_collocates_for_occurrences,
                    all_collocate_occurrences = list(getattr(collocate_occurrence_table, "rows", []) or [])
                    all_collocate_occurrences.sort(
                        key=lambda occ: (
                            10**9 if getattr(occ, "collocate_rank", None) is None else int(getattr(occ, "collocate_rank")),
                            int(getattr(occ, "source_row_idx", 0)),
                            int(getattr(occ, "source_match_start_idx", 0)),
                            int(getattr(occ, "collocate_idx", 0)),
                        )
                    )
                    collocate_occurrence_offset = int(config.get("offset", 0) or 0)
                    collocate_occurrence_result_limit = config.get("result_limit", config.get("limit"))
                    if collocate_occurrence_result_limit is None:
                        visible_collocate_occurrences = all_collocate_occurrences[collocate_occurrence_offset:]
                    else:
                        visible_collocate_occurrences = all_collocate_occurrences[collocate_occurrence_offset: collocate_occurrence_offset + int(collocate_occurrence_result_limit)]
                    collocate_left_context = int(config.get("left_context", 10) or 10)
                    collocate_right_context = int(config.get("right_context", 10) or 10)
                    collocate_occurrence_results = [_collocate_occurrence_to_schema_result(occ, df, left_context=collocate_left_context, right_context=collocate_right_context) for occ in visible_collocate_occurrences]
                    _collocate_concordance_private = {
                        "results": collocate_occurrence_results,
                        "total_hits": len(all_collocate_occurrences),
                        "returned_hits": len(collocate_occurrence_results),
                        "offset": collocate_occurrence_offset,
                        "limit": collocate_occurrence_result_limit,
                        "has_more": (collocate_occurrence_offset + len(collocate_occurrence_results)) < len(all_collocate_occurrences),
                        "selected_collocates": list(getattr(collocate_occurrence_table, "selected_collocates", []) or []),
                    }
                except Exception as exc:
                    warnings.append(f"concordance_of_collocates_runtime_error: {type(exc).__name__}: {exc}")
        if source_info.get("partial_reason"):
            warnings.append(str(source_info.get("partial_reason")))
        _analytics_payload = {
            "requested": ["collocations"], "included": ["collocations"], "unavailable_reasons": {},
            "collocations": {
                "available": True, "kind": "collocation_table",
                "source": source_info,
                "parameters": _collocations_public_parameters(config),
                "collocate_filter": _collocate_filter_public_shape(config),
                "rows": _public_collocation_rows(table, config.get("colloc_limit")), "warnings": warnings,
            },
        }
        if _collocate_concordance_private:
            _analytics_payload["_collocate_concordance_results"] = _collocate_concordance_private.get("results", [])
            _analytics_payload["_collocate_concordance_metadata"] = {k: v for k, v in _collocate_concordance_private.items() if k != "results"}
        return _analytics_payload
    except Exception as exc:
        return _analytics_unavailable(f"collocations_runtime_error: {type(exc).__name__}: {exc}", config=config)




def _build_selected_collocate_concordance_results(data, bundle, analytics, config):
    """Build explicit selected-collocate occurrence rows if 163d main path did not.

    This is a narrow corrective fallback used only when:
      - --concordance-of collocates
      - --collocate-concordance contains explicit labels
      - _compute_cli_collocations_analytics(...) did not export private occurrence rows.
    """
    selected_labels = list((config or {}).get("collocate_concordance") or [])
    if not selected_labels:
        return None, {}

    scope = str((config or {}).get("analytics_scope", "all-matches") or "all-matches")
    source_info = {}
    if scope == "returned-results":
        schema_results = list((data or {}).get("results") or [])
        colloc_results = _schema_results_to_collocation_result_dicts(schema_results)
        source_info = {
            "scope": "returned_results",
            "match_query": (data or {}).get("query"),
            "match_count": len(colloc_results),
            "total_hits": (data or {}).get("total_hits"),
            "returned_results_count": len(schema_results),
            "analysis_complete": None,
            "analysis_cap": None,
            "partial_reason": None,
        }
    elif scope == "all-matches":
        analytics_bundle, source_info = _run_cli_all_matches_analytics_search(data, bundle, config)
        colloc_results = _bundle_results_to_collocation_result_dicts(analytics_bundle)
        try:
            source_info["match_query"] = (data or {}).get("query")
            source_info["match_count"] = len(colloc_results)
        except Exception:
            pass
    else:
        return None, {"error": f"unsupported_analytics_scope: {scope}"}

    if not colloc_results:
        return [], {
            "selected_collocates": [{"collocate": label, "rank": None} for label in selected_labels],
            "collocate_concordance_selection_mode": "explicit",
            "collocate_concordance_selector": selected_labels,
            "total_hits": 0,
            "returned_hits": 0,
            "offset": int((config or {}).get("offset", 0) or 0),
            "limit": (config or {}).get("result_limit"),
            "has_more": False,
            "source": source_info,
        }

    import pandas as _selected_collocates_pd
    from korpusuj.search.collocations import CollocationOptions, CollocateFilterGroup, collect_collocate_occurrences

    df = _selected_collocates_pd.read_parquet((config or {}).get("corpus_path"))
    filter_groups = [CollocateFilterGroup(**g) for g in ((config or {}).get("collocate_filter_groups") or [])]
    options = CollocationOptions(
        mode=(config or {}).get("colloc_mode", "Liniowe"),
        form_mode=(config or {}).get("colloc_form", "Lemat (base)"),
        sort_mode=(config or {}).get("colloc_sort", "Log-Dice"),
        min_freq=int((config or {}).get("colloc_min_freq", 1) or 1),
        min_range=int((config or {}).get("colloc_min_range", 1) or 1),
        l_span=int((config or {}).get("colloc_left_span", 5) or 5),
        r_span=int((config or {}).get("colloc_right_span", 5) or 5),
        use_sentence_bound=(str((config or {}).get("colloc_sentence_bound", "true") or "true").lower() in {"1", "true", "yes", "y", "on"}),
        collocate_filter_groups=filter_groups,
        syn_dir=(config or {}).get("colloc_syn_dir", "Podrzędnik"),
        deprel_filter=(config or {}).get("colloc_deprel", "Wszystkie"),
    )

    # Reuse rank from analytics.collocations.rows when available; explicit labels still work if absent.
    rank_by_label = {}
    try:
        for row in ((((analytics or {}).get("collocations") or {}).get("rows") or [])):
            if isinstance(row, dict):
                label = str(row.get("collocate", row.get("colloc", "")))
                if label:
                    rank_by_label[label] = row.get("rank")
    except Exception:
        rank_by_label = {}

    selected_collocates = [{"collocate": label, "rank": rank_by_label.get(label)} for label in selected_labels]
    occ_table = collect_collocate_occurrences(colloc_results, df, options, selected_collocates=selected_collocates, feat_mapping={})
    occs = list(getattr(occ_table, "rows", []) or [])

    converted = [
        _collocate_occurrence_to_schema_result(
            occ,
            df,
            left_context=int((config or {}).get("left_context", 10) or 10),
            right_context=int((config or {}).get("right_context", 10) or 10),
        )
        for occ in occs
    ]

    try:
        offset = max(0, int((config or {}).get("offset", 0) or 0))
    except Exception:
        offset = 0
    limit_raw = (config or {}).get("result_limit", None)
    try:
        limit = None if limit_raw is None else max(0, int(limit_raw))
    except Exception:
        limit = None
    total = len(converted)
    if limit is None:
        page = converted[offset:]
    else:
        page = converted[offset:offset + limit]
    meta = {
        "selected_collocates": selected_collocates,
        "collocate_concordance_selection_mode": "explicit",
        "collocate_concordance_selector": selected_labels,
        "total_hits": total,
        "returned_hits": len(page),
        "offset": offset,
        "limit": limit,
        "has_more": bool((offset + len(page)) < total),
        "source": source_info,
    }
    return page, meta



def _build_default_collocate_concordance_results(data, bundle, analytics, config):
    """Build default collocate occurrence rows for --concordance-of collocates.

    Used when no explicit --collocate-concordance selector is present and the
    main analytics path did not export private collocate occurrence rows.
    Selection comes from the current public analytics.collocations.rows, so
    --colloc-limit and --colloc-sort define the selected top/ranked collocates.
    """
    public_rows = list(((((analytics or {}).get("collocations") or {}).get("rows")) or []))
    selected_collocates = []
    for row in public_rows:
        if not isinstance(row, dict):
            continue
        label = row.get("collocate", row.get("colloc"))
        if label is None:
            continue
        selected_collocates.append({"collocate": str(label), "rank": row.get("rank")})
    if not selected_collocates:
        return [], {
            "selected_collocates": [],
            "collocate_concordance_selection_mode": "top_collocates",
            "collocate_concordance_selector": [],
            "total_hits": 0,
            "returned_hits": 0,
            "offset": int((config or {}).get("offset", 0) or 0),
            "limit": (config or {}).get("result_limit"),
            "has_more": False,
            "source": {"reason": "no_public_collocation_rows"},
        }

    scope = str((config or {}).get("analytics_scope", "all-matches") or "all-matches")
    source_info = {}
    if scope == "returned-results":
        schema_results = list((data or {}).get("results") or [])
        colloc_results = _schema_results_to_collocation_result_dicts(schema_results)
        source_info = {
            "scope": "returned_results",
            "match_query": (data or {}).get("query"),
            "match_count": len(colloc_results),
            "total_hits": (data or {}).get("total_hits"),
            "returned_results_count": len(schema_results),
            "analysis_complete": None,
            "analysis_cap": None,
            "partial_reason": None,
        }
    elif scope == "all-matches":
        analytics_bundle, source_info = _run_cli_all_matches_analytics_search(data, bundle, config)
        colloc_results = _bundle_results_to_collocation_result_dicts(analytics_bundle)
        try:
            source_info["match_query"] = (data or {}).get("query")
            source_info["match_count"] = len(colloc_results)
        except Exception:
            pass
    else:
        return None, {"error": f"unsupported_analytics_scope: {scope}"}

    import pandas as _default_collocates_pd
    from korpusuj.search.collocations import CollocationOptions, CollocateFilterGroup, collect_collocate_occurrences

    df = _default_collocates_pd.read_parquet((config or {}).get("corpus_path"))
    filter_groups = [CollocateFilterGroup(**g) for g in ((config or {}).get("collocate_filter_groups") or [])]
    options = CollocationOptions(
        mode=(config or {}).get("colloc_mode", "Liniowe"),
        form_mode=(config or {}).get("colloc_form", "Lemat (base)"),
        sort_mode=(config or {}).get("colloc_sort", "Log-Dice"),
        min_freq=int((config or {}).get("colloc_min_freq", 1) or 1),
        min_range=int((config or {}).get("colloc_min_range", 1) or 1),
        l_span=int((config or {}).get("colloc_left_span", 5) or 5),
        r_span=int((config or {}).get("colloc_right_span", 5) or 5),
        use_sentence_bound=(str((config or {}).get("colloc_sentence_bound", "true") or "true").lower() in {"1", "true", "yes", "y", "on"}),
        collocate_filter_groups=filter_groups,
        syn_dir=(config or {}).get("colloc_syn_dir", "Podrzędnik"),
        deprel_filter=(config or {}).get("colloc_deprel", "Wszystkie"),
    )

    occ_table = collect_collocate_occurrences(colloc_results, df, options, selected_collocates=selected_collocates, feat_mapping={})
    occs = list(getattr(occ_table, "rows", []) or [])
    converted = [
        _collocate_occurrence_to_schema_result(
            occ,
            df,
            left_context=int((config or {}).get("left_context", 10) or 10),
            right_context=int((config or {}).get("right_context", 10) or 10),
        )
        for occ in occs
    ]

    try:
        offset = max(0, int((config or {}).get("offset", 0) or 0))
    except Exception:
        offset = 0
    raw_limit = (config or {}).get("result_limit", None)
    try:
        limit = None if raw_limit is None else max(0, int(raw_limit))
    except Exception:
        limit = None
    total = len(converted)
    page = converted[offset:] if limit is None else converted[offset:offset + limit]
    meta = {
        "selected_collocates": selected_collocates,
        "collocate_concordance_selection_mode": "top_collocates",
        "collocate_concordance_selector": [x.get("collocate") for x in selected_collocates],
        "total_hits": total,
        "returned_hits": len(page),
        "offset": offset,
        "limit": limit,
        "has_more": bool((offset + len(page)) < total),
        "source": source_info,
    }
    return page, meta


_CLI_PROFILE_CONFIG = {}


def _build_cli_profile_config(args):
    try:
        mode = getattr(args, "profile", None)
    except Exception:
        mode = None
    return {
        "profile": mode,
        "profile_only": bool(getattr(args, "profile_only", False)),
        "profile_target_token": getattr(args, "profile_target_token", None),
        "profile_target_lemma": getattr(args, "profile_target_lemma", None),
        "profile_sort": getattr(args, "profile_sort", "log-dice"),
        "profile_min_freq": int(getattr(args, "profile_min_freq", 2) or 2),
        "profile_max_rows_per_relation": getattr(args, "profile_max_rows_per_relation", None),
        "profile_layout": getattr(args, "profile_layout", "grouped"),
        "profile_example_refs": int(getattr(args, "profile_example_refs", 0) or 0),
        "profile_examples": int(getattr(args, "profile_examples", 0) or 0),
        "profile_example_context": int(getattr(args, "profile_example_context", 6) or 6),
        "profile_expand_mwe": getattr(args, "profile_expand_mwe", "false"),
        "format": getattr(args, "format", "json"),
        "query": getattr(args, "query", None),
        "corpus_path": getattr(args, "corpus_path", None),
    }


def _cli_profile_fail(message):
    raise SystemExit(str(message))


def _cli_profile_bool(value, default=False):
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "tak", "on"}:
        return True
    if text in {"0", "false", "no", "n", "nie", "off"}:
        return False
    return bool(default)


def _cli_profile_to_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _cli_profile_row_get(row, name, default=None):
    try:
        if hasattr(row, name):
            return getattr(row, name)
    except Exception:
        pass
    try:
        return row[name]
    except Exception:
        return default


def _cli_profile_normalize_lemma(value, ignore_case=True):
    text = str(value)
    return text.lower() if ignore_case else text


def _cli_profile_extract_token_segments(query):
    import re as _re_profile
    q = str(query or "")
    segments = []
    for match in _re_profile.finditer(r"\[([^\[\]]*)\]", q):
        part = match.group(1)
        conds = []
        for cm in _re_profile.finditer(r"\b(base|orth|pos|upos)\s*(=|!=)\s*\"([^\"]*)\"", part):
            conds.append({"attr": cm.group(1), "op": cm.group(2), "value": cm.group(3)})
        segments.append({"raw": part, "conditions": conds})
    leftover = _re_profile.sub(r"\[[^\[\]]*\]", "", q).strip()
    return segments, leftover


def _cli_profile_resolve_query(config):
    query = str(config.get("query") or "")
    segments, leftover = _cli_profile_extract_token_segments(query)
    if leftover:
        _cli_profile_fail("Collocational profile v1 supports only simple token-segment queries; unsupported text outside [] was found.")
    if not segments:
        _cli_profile_fail("Collocational profile requires a token query such as [base=\"wojna\"].")
    target_token = config.get("profile_target_token")
    if len(segments) == 1 and target_token is None:
        target_token = 1
    if len(segments) > 1 and target_token is None:
        _cli_profile_fail("Collocational profile for multi-token queries requires --profile-target-token N, matching the GUI Token N selector.")
    try:
        target_token = int(target_token)
    except Exception:
        _cli_profile_fail("--profile-target-token must be a positive integer.")
    if target_token < 1 or target_token > len(segments):
        _cli_profile_fail(f"--profile-target-token {target_token} is outside query token range 1..{len(segments)}.")
    parsed = []
    for seg in segments:
        info = {"base": None, "upos": None, "pos": None, "orth": None, "unsupported": []}
        for cond in seg["conditions"]:
            if cond["op"] != "=":
                info["unsupported"].append(cond)
                continue
            attr = cond["attr"]
            if attr in info and info[attr] is None:
                info[attr] = cond["value"]
            else:
                info["unsupported"].append(cond)
        if info["unsupported"]:
            _cli_profile_fail("Collocational profile v1 supports only simple exact base/orth/pos/upos constraints in token segments.")
        if not info.get("base"):
            _cli_profile_fail("Collocational profile v1 requires base=\"...\" in each query token segment.")
        parsed.append(info)
    target = parsed[target_token - 1]
    return {
        "query": query,
        "segments": parsed,
        "target_token": target_token,
        "target_lemma": str(config.get("profile_target_lemma") or target.get("base") or ""),
        "target_upos_constraint": target.get("upos"),
        "target_pos_constraint": target.get("pos"),
    }


def _cli_profile_make_minimal_result(row_idx, token_idx, matched_lemma=""):
    items = [None] * 13
    items[4] = matched_lemma
    items[11] = int(row_idx)
    items[12] = int(token_idx)
    return tuple(items)


def _cli_profile_build_token_frequency_dict(df, ignore_case=True):
    from collections import Counter
    freq = Counter()
    total = 0
    for i in range(len(df)):
        try:
            row = df.iloc[i]
        except Exception:
            continue
        lemmas = _cli_profile_to_list(_cli_profile_row_get(row, "lemmas", []))
        for lemma in lemmas:
            key = _cli_profile_normalize_lemma(lemma, ignore_case=ignore_case)
            if key:
                freq[key] += 1
                total += 1
    return dict(freq), int(total)


def _cli_profile_sequence_match(lemmas, start, segments, ignore_case=True):
    for offset, seg in enumerate(segments):
        idx = start + offset
        if idx >= len(lemmas):
            return False
        if _cli_profile_normalize_lemma(lemmas[idx], ignore_case=ignore_case) != _cli_profile_normalize_lemma(seg.get("base"), ignore_case=ignore_case):
            return False
    return True


def _cli_profile_build_hits(df, resolved, ignore_case=True):
    from collections import Counter
    segments = resolved["segments"]
    target_offset = int(resolved["target_token"]) - 1
    upos_counter = Counter()
    pos_counter = Counter()
    hits = []
    for row_idx in range(len(df)):
        try:
            row = df.iloc[row_idx]
        except Exception:
            continue
        lemmas = _cli_profile_to_list(_cli_profile_row_get(row, "lemmas", []))
        upos = _cli_profile_to_list(_cli_profile_row_get(row, "upostags", []))
        pos = _cli_profile_to_list(_cli_profile_row_get(row, "postags", []))
        max_start = len(lemmas) - len(segments)
        if max_start < 0:
            continue
        for start in range(max_start + 1):
            if not _cli_profile_sequence_match(lemmas, start, segments, ignore_case=ignore_case):
                continue
            ok = True
            for offset, seg in enumerate(segments):
                tok_idx = start + offset
                tok_upos = str(upos[tok_idx]).upper() if tok_idx < len(upos) else ""
                tok_pos = str(pos[tok_idx]) if tok_idx < len(pos) else ""
                if seg.get("upos") and tok_upos != str(seg.get("upos")).upper():
                    ok = False
                    break
                if seg.get("pos") and tok_pos.lower() != str(seg.get("pos")).lower():
                    ok = False
                    break
            if not ok:
                continue
            target_idx = start + target_offset
            tok_upos = str(upos[target_idx]).upper() if target_idx < len(upos) else ""
            tok_pos = str(pos[target_idx]) if target_idx < len(pos) else ""
            matched_lemma = " ".join(str(lemmas[start + j]) for j in range(len(segments)))
            upos_counter[tok_upos] += 1
            pos_counter[tok_pos] += 1
            hits.append(_cli_profile_make_minimal_result(row_idx, target_idx, matched_lemma=matched_lemma))
    return hits, dict(upos_counter), dict(pos_counter)


def _cli_profile_example_ref_to_public(ref):
    try:
        row_idx, target_idx, collocate_idx = ref
        return {"doc_id": int(row_idx), "target_idx": int(target_idx), "collocate_idx": int(collocate_idx)}
    except Exception:
        return {"raw": repr(ref)}


def _cli_profile_reconstruct_example(df, ref, context_tokens=6):
    out = _cli_profile_example_ref_to_public(ref)
    try:
        row_idx = int(out["doc_id"])
        target_idx = int(out["target_idx"])
        collocate_idx = int(out["collocate_idx"])
        row = df.iloc[row_idx]
        tokens = _cli_profile_to_list(_cli_profile_row_get(row, "tokens", []))
        lemmas = _cli_profile_to_list(_cli_profile_row_get(row, "lemmas", []))
        upos = _cli_profile_to_list(_cli_profile_row_get(row, "upostags", []))
        display_tokens = tokens if tokens else lemmas
        n = len(display_tokens)
        left_start = max(0, min(target_idx, collocate_idx) - int(context_tokens))
        right_end = min(n, max(target_idx, collocate_idx) + int(context_tokens) + 1)
        out.update({
            "target_text": str(display_tokens[target_idx]) if 0 <= target_idx < len(display_tokens) else "",
            "target_lemma": str(lemmas[target_idx]) if 0 <= target_idx < len(lemmas) else "",
            "target_upos": str(upos[target_idx]) if 0 <= target_idx < len(upos) else "",
            "collocate_text": str(display_tokens[collocate_idx]) if 0 <= collocate_idx < len(display_tokens) else "",
            "collocate_lemma": str(lemmas[collocate_idx]) if 0 <= collocate_idx < len(lemmas) else "",
            "collocate_upos": str(upos[collocate_idx]) if 0 <= collocate_idx < len(upos) else "",
            "left_context": " ".join(str(x) for x in display_tokens[left_start:min(target_idx, collocate_idx)]),
            "span_text": " ".join(str(x) for x in display_tokens[min(target_idx, collocate_idx):max(target_idx, collocate_idx)+1]),
            "right_context": " ".join(str(x) for x in display_tokens[max(target_idx, collocate_idx)+1:right_end]),
        })
    except Exception as exc:
        out["reconstruction_error"] = repr(exc)
    return out


def _cli_profile_sort_rows(rows, sort_mode):
    metric_map = {
        "log-dice": "log_dice",
        "log-likelihood": "ll_score",
        "mi": "mi_score",
        "t-score": "t_score",
        "frequency": "cooc_freq",
    }
    attr = metric_map.get(str(sort_mode or "log-dice"), "log_dice")
    return sorted(list(rows or []), key=lambda r: (getattr(r, attr, 0) or 0, getattr(r, "cooc_freq", 0) or 0, getattr(r, "doc_freq", 0) or 0), reverse=True)


def _cli_profile_row_to_public(row, df=None, example_refs_n=0, examples_n=0, example_context=6):
    out = {
        "collocate": getattr(row, "collocate", ""),
        "display_collocate": getattr(row, "display_collocate", "") or getattr(row, "collocate", ""),
        "collocate_upos": getattr(row, "collocate_upos", ""),
        "counts": {
            "cooc_freq": getattr(row, "cooc_freq", 0),
            "doc_freq": getattr(row, "doc_freq", 0),
            "global_freq": getattr(row, "global_freq", 0),
        },
        "scores": {
            "log_likelihood": getattr(row, "ll_score", 0),
            "mi": getattr(row, "mi_score", 0),
            "t_score": getattr(row, "t_score", 0),
            "log_dice": getattr(row, "log_dice", 0),
        },
    }
    refs = list(getattr(row, "example_refs", []) or [])
    if int(example_refs_n or 0) > 0:
        out["example_refs"] = [_cli_profile_example_ref_to_public(ref) for ref in refs[:int(example_refs_n)]]
    if int(examples_n or 0) > 0 and df is not None:
        out["examples"] = [_cli_profile_reconstruct_example(df, ref, context_tokens=example_context) for ref in refs[:int(examples_n)]]
    return out



def _cli_profile_relation_group(name):
    n = str(name or "").lower()
    if any(x in n for x in ["modyfikowane", "czynności, których"]):
        return "7. Węzły nadrzędne (Co określa?)"
    if "się" in n:
        return "8. Zwrotność (się)"
    if any(x in n for x in ["wielowyrazowe", "złożenia", "człon", "flat", "fixed", "compound", "apozycj"]):
        return "6. Konstrukcje złożone i nazwy"
    if any(x in n for x in ["porównan", "punkt odniesienia"]):
        return "4. Porównania"
    if any(x in n for x in ["zdaniow", "dołączenia", "paratak", "szereg", "współrzędne", "przydawkow"]):
        return "5. Związki zdaniowe i szeregi"
    if "podmiot" in n:
        return "1. Podmioty"
    if any(x in n for x in ["argument", "dopełnien", "orzecznik"]):
        return "2. Argumenty (frazy wymagane)"
    if any(x in n for x in ["modyfikator", "okolicznik", "określnik", "przysłówek", "zaim", "przyimkow", "intensyfikator", "operator", "agens"]):
        return "3. Modyfikatory (frazy niewymagane)"
    return "9. Pozostałe"

def _cli_profile_flat_row_to_public(row, df=None, example_refs_n=0, examples_n=0, example_context=6):
    public = _cli_profile_row_to_public(row, df=df, example_refs_n=example_refs_n, examples_n=examples_n, example_context=example_context)
    counts = public.pop("counts", {})
    scores = public.pop("scores", {})
    relation = getattr(row, "relation", "")
    return {
        "group": _cli_profile_relation_group(relation),
        "relation": relation,
        **public,
        "cooc_freq": counts.get("cooc_freq"),
        "doc_freq": counts.get("doc_freq"),
        "global_freq": counts.get("global_freq"),
        "log_likelihood": scores.get("log_likelihood"),
        "mi": scores.get("mi"),
        "t_score": scores.get("t_score"),
        "log_dice": scores.get("log_dice"),
    }

def _cli_profile_serialize(profile_dict, df, resolved, hits_count, upos_distribution, pos_distribution, config):
    sort_mode = str(config.get("profile_sort") or "log-dice")
    layout = str(config.get("profile_layout") or "tree")
    if layout not in {"tree", "flat"}:
        _cli_profile_fail("--profile-layout must be one of: tree, flat")
    max_rows = config.get("profile_max_rows_per_relation")
    try:
        max_rows = int(max_rows) if max_rows is not None else None
    except Exception:
        max_rows = None
    example_refs_n = int(config.get("profile_example_refs") or 0)
    examples_n = int(config.get("profile_examples") or 0)
    example_context = int(config.get("profile_example_context") or 6)

    groups_by_name = {}
    flat_rows = []
    serialized_relation_count = 0
    serialized_row_count = 0

    for rel_name in sorted((profile_dict or {}).keys()):
        raw_rows = _cli_profile_sort_rows(profile_dict.get(rel_name) or [], sort_mode)
        rows = raw_rows[:max_rows] if max_rows is not None else raw_rows
        group_name = _cli_profile_relation_group(rel_name)
        serialized_relation_count += 1
        serialized_row_count += len(rows)

        if layout == "tree":
            public_rows = [_cli_profile_row_to_public(r, df=df, example_refs_n=example_refs_n, examples_n=examples_n, example_context=example_context) for r in rows]
            group_obj = groups_by_name.setdefault(group_name, {"group": group_name, "relation_count": 0, "row_count": 0, "relations": []})
            group_obj["relations"].append({"relation": rel_name, "row_count": len(raw_rows), "rows": public_rows})
            group_obj["relation_count"] += 1
            group_obj["row_count"] += len(public_rows)
        elif layout == "flat":
            flat_rows.extend([_cli_profile_flat_row_to_public(r, df=df, example_refs_n=example_refs_n, examples_n=examples_n, example_context=example_context) for r in rows])

    payload = {
        "target": {
            "lemma": resolved.get("target_lemma"),
            "target_token": resolved.get("target_token"),
            "query": resolved.get("query"),
            "upos_distribution": upos_distribution,
            "pos_distribution": pos_distribution,
            "explicit_constraints": {"upos": resolved.get("target_upos_constraint"), "pos": resolved.get("target_pos_constraint")},
        },
        "parameters": {
            "sort": sort_mode,
            "layout": layout,
            "min_freq": int(config.get("profile_min_freq") or 2),
            "max_rows_per_relation": max_rows,
            "example_refs": example_refs_n,
            "examples": examples_n,
            "example_context": example_context,
            "expand_mwe": _cli_profile_bool(config.get("profile_expand_mwe"), default=False),
        },
        "summary": {
            "relation_count": serialized_relation_count,
            "row_count": serialized_row_count,
            "hits_used": hits_count,
        },
    }
    if layout == "tree":
        payload["groups"] = [groups_by_name[name] for name in sorted(groups_by_name.keys())]
    elif layout == "flat":
        payload["flat_rows"] = flat_rows
    return payload



_CLI_SEARCH_STATISTICS_ARGS = None


def _set_cli_search_statistics_args(args):
    """Store parsed CLI args for schema-output statistics attachment."""
    global _CLI_SEARCH_STATISTICS_ARGS
    _CLI_SEARCH_STATISTICS_ARGS = args
    return args

def _cli_statistics_as_public_value(value):
    """Convert statistics helper outputs to JSON-safe public values."""
    try:
        import dataclasses
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _cli_statistics_as_public_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_cli_statistics_as_public_value(v) for v in value]
    if isinstance(value, list):
        return [_cli_statistics_as_public_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_cli_statistics_as_public_value(v) for v in value)
    return value


def _cli_statistics_as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
    except Exception:
        pass
    return []




def _cli_statistics_row_to_object(row, columns):
    """Convert one compact statistics row to a self-describing JSON object."""
    if isinstance(row, dict):
        return _cli_statistics_as_public_value(row)
    values = _cli_statistics_as_list(row)
    if not values:
        return _cli_statistics_as_public_value(row)
    out = {}
    for index, column in enumerate(columns):
        out[str(column)] = _cli_statistics_as_public_value(values[index]) if index < len(values) else None
    if len(values) > len(columns):
        out["extra_values"] = _cli_statistics_as_public_value(values[len(columns):])
    return out


def _cli_statistics_rows_to_objects(rows, columns):
    """Convert compact statistics table rows to list-of-object public rows."""
    return [_cli_statistics_row_to_object(row, columns) for row in _cli_statistics_as_list(rows)]


def _cli_statistics_ranked_match_rows_to_public(rows):
    return _cli_statistics_rows_to_objects(
        rows,
        ["rank", "match", "frequency", "pmw", "document_frequency", "tfidf"],
    )


def _cli_statistics_lemma_frequency_rows_to_public(rows):
    return _cli_statistics_rows_to_objects(rows, ["lemma", "frequency"])


def _cli_statistics_lemma_pmw_rows_to_public(rows):
    return _cli_statistics_rows_to_objects(rows, ["lemma", "pmw"])


def _cli_statistics_lemma_tfidf_rows_to_public(rows):
    return _cli_statistics_rows_to_objects(rows, ["lemma", "tfidf"])


def _cli_statistics_monthly_match_rows_to_public(rows):
    return _cli_statistics_rows_to_objects(
        rows,
        ["year", "month", "match", "frequency", "pmw", "tfidf", "z_score"],
    )

def _cli_statistics_month_from_value(value):
    import re
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"(\d{4})[-/.](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    match = re.search(r"\b(\d{4})(\d{2})\b", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    return None


def _cli_statistics_infer_month_from_row(row):
    for col in ["month_key", "month", "date", "publication_date", "data", "Data publikacji", "rok_miesiac", "year_month"]:
        try:
            if col in row.index:
                month = _cli_statistics_month_from_value(row[col])
                if month:
                    return month
        except Exception:
            pass
    return None


def _cli_statistics_infer_corpus_totals(df):
    from collections import defaultdict
    info = {"total_docs": int(len(df)) if df is not None else 0, "total_token_count": None, "monthly_total_count": 0, "monthly_total_method": None}
    if df is None:
        return {}, info
    token_col = "tokens" if "tokens" in df.columns else "lemmas" if "lemmas" in df.columns else None
    if not token_col:
        return {}, info
    total_token_count = int(sum(len(_cli_statistics_as_list(value)) for value in df[token_col]))
    monthly_totals = defaultdict(int)
    for _, row in df.iterrows():
        month = _cli_statistics_infer_month_from_row(row)
        if month:
            monthly_totals[month] += len(_cli_statistics_as_list(row[token_col]))
    monthly_totals = dict(sorted(monthly_totals.items()))
    info.update({
        "total_token_count": total_token_count,
        "monthly_total_count": len(monthly_totals),
        "monthly_total_method": f"infer_from_{token_col}_and_date_columns" if monthly_totals else None,
    })
    return monthly_totals, info


def _cli_statistics_make_df_callback(exact_orth_df, exact_lemma_df):
    def df_for_matched_key(key, mode="lemma", table_override=None):
        table = table_override if table_override is not None else (exact_orth_df if str(mode).lower() in {"orth", "token", "text"} else exact_lemma_df)
        if key in table:
            return len(table.get(key) or set())
        text_key = str(key)
        if text_key in table:
            return len(table.get(text_key) or set())
        return 0
    return df_for_matched_key


def _cli_statistics_calc_z_score(value, mean, std):
    try:
        if not std:
            return 0.0
        return (float(value) - float(mean)) / float(std)
    except Exception:
        return 0.0


def _cli_statistics_resolve_corpus_path(args):
    path = getattr(args, "corpus_path", None)
    if path:
        return path
    return None


def _build_cli_search_statistics_payload(data, args):
    """Build the statistics.search payload from current schema-v1 results.

    Statistics v1 uses scope=returned-results and unit=match. Multi-token query
    matches are counted as whole spans because match_text/matched_lemmas are used
    as aggregation keys by collect_search_frequency_inputs(...).
    """
    from korpusuj.search.statistics import (
        build_global_frequency_tables,
        build_monthly_frequency_tables,
        collect_search_frequency_inputs,
    )
    import pandas as pd

    results = list(data.get("results") or [])
    frequency_inputs = collect_search_frequency_inputs(results)
    corpus_path = _cli_statistics_resolve_corpus_path(args)
    if not corpus_path:
        raise RuntimeError("statistics search requires --corpus-path in v1")
    df = pd.read_parquet(corpus_path)
    monthly_totals, corpus_totals = _cli_statistics_infer_corpus_totals(df)
    total_token_count = corpus_totals.get("total_token_count") or 0
    total_docs = corpus_totals.get("total_docs") or 0
    callback = _cli_statistics_make_df_callback(frequency_inputs.exact_orth_df, frequency_inputs.exact_lemma_df)

    global_tables = build_global_frequency_tables(
        unique_matched_tokens=frequency_inputs.unique_matched_tokens,
        monthly_lemma_freq=frequency_inputs.monthly_lemma_freq,
        exact_orth_df=frequency_inputs.exact_orth_df,
        exact_lemma_df=frequency_inputs.exact_lemma_df,
        total_token_count=total_token_count,
        total_docs=total_docs,
        df_for_matched_key=callback,
    )
    monthly_tables = build_monthly_frequency_tables(
        monthly_lemma_freq=frequency_inputs.monthly_lemma_freq,
        unique_lemmas=frequency_inputs.unique_lemmas,
        true_monthly_totals=monthly_totals,
        total_docs=total_docs,
        exact_lemma_df=frequency_inputs.exact_lemma_df,
        df_for_matched_key=callback,
        calc_z_score_func=_cli_statistics_calc_z_score,
    )

    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    statistics_scope = _cli_statistics_scope_from_args(args)
    summary = {
        "query": data.get("query") or request.get("query"),
        "scope": statistics_scope,
        "unit": "match",
        "multi_token_matches_supported": True,
        "hits_used": len(results),
        "total_hits": data.get("total_hits"),
        "returned_hits_used": len(results),
        "corpus_total_docs": total_docs,
        "corpus_total_tokens": total_token_count,
        "monthly_total_months": len(monthly_totals),
        "monthly_total_method": corpus_totals.get("monthly_total_method"),
    }
    return {
        "summary": summary,
        "global": {
            "fq_data_token": _cli_statistics_ranked_match_rows_to_public(global_tables.fq_data_token),
            "fq_data": _cli_statistics_ranked_match_rows_to_public(global_tables.fq_data),
            "lemma_total_freq": _cli_statistics_lemma_frequency_rows_to_public(global_tables.s_lemma_total_freq),
            "lemma_global_pmw": _cli_statistics_lemma_pmw_rows_to_public(global_tables.s_lemma_global_pmw),
            "lemma_global_tfidf": _cli_statistics_lemma_tfidf_rows_to_public(global_tables.s_lemma_global_tfidf),
        },
        "monthly": {
            "fq_data_month": _cli_statistics_monthly_match_rows_to_public(monthly_tables.fq_data_month),
            "monthly_freq": _cli_statistics_as_public_value(monthly_tables.monthly_freq_for_use),
            "monthly_tfidf": _cli_statistics_as_public_value(monthly_tables.monthly_tfidf_for_use),
            "monthly_zscore": _cli_statistics_as_public_value(monthly_tables.monthly_zscore_for_use),
        },
    }




def _cli_statistics_scope_from_args(args):
    """Return the public statistics analytics scope selected by CLI args."""
    raw = str(getattr(args, "analytics_scope", "all-matches") or "all-matches").strip().lower()
    raw = raw.replace("_", "-")
    if raw not in {"all-matches", "returned-results"}:
        raise SystemExit("--analytics statistics supports --analytics-scope all-matches or returned-results.")
    return raw


def _build_cli_all_matches_statistics_schema_data(data, bundle, args):
    """Materialize all query matches and convert them to schema-v1 data for statistics."""
    if bundle is None:
        raise RuntimeError("statistics all-matches requires the current search bundle")
    original_request = getattr(bundle, "request", None)
    corpus_name = (
        getattr(args, "corpus_name", None)
        or data.get("corpus")
        or getattr(original_request, "corpus_name", None)
        or Path(str(getattr(args, "corpus_path", None) or "corpus")).stem
    )
    config = {
        "analytics_scope": "all-matches",
        "corpus_path": str(getattr(args, "corpus_path", "") or ""),
        "corpus_name": str(corpus_name or ""),
        "full_context_size": int(getattr(args, "full_context_size", 250) or 250),
        "candidate_max_docs": int(getattr(args, "candidate_max_docs", 3000) or 3000),
        "candidate_stream_batch_docs": int(getattr(args, "candidate_stream_batch_docs", 256) or 256),
    }
    analytics_bundle, source_info = _run_cli_all_matches_analytics_search(data, bundle, config)
    from korpusuj.search.output_schema import search_bundle_to_jsonable
    all_match_data = search_bundle_to_jsonable(analytics_bundle)
    if data.get("query") and not all_match_data.get("query"):
        all_match_data["query"] = data.get("query")
    if data.get("corpus") and not all_match_data.get("corpus"):
        all_match_data["corpus"] = data.get("corpus")
    if data.get("total_hits") is not None:
        all_match_data["total_hits"] = data.get("total_hits")
    return all_match_data, source_info

def _attach_cli_search_statistics_to_schema_data(data, args, bundle=None):
    analytics_kind = str(getattr(args, "analytics", "none") or "none").lower()
    if analytics_kind != "statistics":
        return data
    if str(getattr(args, "format", "json") or "json") != "json":
        raise SystemExit("--analytics statistics currently supports --format json only; JSONL statistics payload is not supported in v1.")
    if bool(getattr(args, "profile_only", False)):
        raise SystemExit("--profile-only cannot be combined with --analytics statistics; use --analytics-only for analytics-only output.")

    scope = _cli_statistics_scope_from_args(args)
    display_results_count = len(list((data or {}).get("results") or []))
    display_limit = getattr(args, "limit", data.get("limit") if isinstance(data, dict) else None)

    if scope == "returned-results":
        statistics_data = data
        source_info = {
            "scope": "returned-results",
            "match_count": display_results_count,
            "total_hits": data.get("total_hits") if isinstance(data, dict) else None,
            "returned_results_count": display_results_count,
            "analysis_complete": None,
            "analysis_cap": None,
            "partial_reason": None,
        }
    else:
        statistics_data, source_info = _build_cli_all_matches_statistics_schema_data(data, bundle, args)

    payload = _build_cli_search_statistics_payload(statistics_data, args)
    summary = payload.setdefault("summary", {})
    summary["scope"] = scope
    summary["unit"] = summary.get("unit") or "match"
    summary["total_hits"] = data.get("total_hits")
    summary["displayed_results"] = display_results_count
    summary["display_limit"] = display_limit
    summary["statistics_results_materialized"] = len(list((statistics_data or {}).get("results") or []))
    if scope == "returned-results":
        summary["returned_hits_used"] = display_results_count
    else:
        summary["all_match_materialization_complete"] = source_info.get("analysis_complete")
        summary["all_match_materialization_cap"] = source_info.get("analysis_cap")
        summary["all_match_materialization_partial_reason"] = source_info.get("partial_reason")
        summary["returned_results_used_for_display"] = display_results_count

    analytics = data.setdefault("analytics", {})
    requested = list(analytics.get("requested") or [])
    included = list(analytics.get("included") or [])
    if "statistics" not in requested:
        requested.append("statistics")
    if "statistics" not in included:
        included.append("statistics")
    analytics["requested"] = requested
    analytics["included"] = included
    analytics["statistics"] = payload

    metadata = data.setdefault("metadata", {})
    metadata["analytics_requested"] = requested
    metadata["analytics_included"] = included
    metadata["analytics_scope"] = scope
    metadata["analytics_unit"] = payload.get("summary", {}).get("unit")
    metadata["analytics_payload_included"] = True
    data["statistics_payload_available"] = True

    if bool(getattr(args, "analytics_only", False)):
        data["results_included"] = False
        data["results"] = []
        data["returned_hits"] = 0
        data["has_more"] = False
        metadata["analytics_only"] = True
    return data

def _attach_cli_collocational_profile_to_schema_data(data, bundle=None):
    config = globals().get("_CLI_PROFILE_CONFIG", {}) or {}
    if not config.get("profile"):
        return data
    if str(config.get("profile")) != "collocational":
        _cli_profile_fail("Only --profile collocational is supported in this version.")
    if str(config.get("format") or "json").lower() == "jsonl":
        _cli_profile_fail("--profile collocational currently supports --format json only; JSONL profile payload is not supported in v1.")
    corpus_path = config.get("corpus_path")
    if not corpus_path:
        _cli_profile_fail("--profile collocational currently requires --corpus-path.")
    try:
        import pandas as _pd_profile
        from korpusuj.semantic.word_profile import compute_word_profile as _compute_word_profile
    except Exception as exc:
        _cli_profile_fail(f"Could not import collocational profile runtime: {exc}")
    resolved = _cli_profile_resolve_query(config)
    ignore_case = True
    df = _pd_profile.read_parquet(corpus_path, engine="pyarrow")
    token_freq_dict, total_tokens = _cli_profile_build_token_frequency_dict(df, ignore_case=ignore_case)
    hits, upos_distribution, pos_distribution = _cli_profile_build_hits(df, resolved, ignore_case=ignore_case)
    if not hits:
        _cli_profile_fail("Collocational profile query produced no target hits.")
    min_freq = int(config.get("profile_min_freq") or 2)
    max_rows = config.get("profile_max_rows_per_relation")
    try:
        max_rows = int(max_rows) if max_rows is not None else None
    except Exception:
        max_rows = None
    keep_examples = max(int(config.get("profile_example_refs") or 0), int(config.get("profile_examples") or 0), 5)
    profile_dict = _compute_word_profile(
        results=hits,
        df=df,
        token_freq_dict=token_freq_dict,
        target_lemma=_cli_profile_normalize_lemma(resolved.get("target_lemma"), ignore_case=ignore_case),
        total_tokens=total_tokens,
        min_freq=min_freq,
        max_rows_per_relation=max_rows,
        keep_examples=keep_examples,
        ignore_case=ignore_case,
        expand_mwe=_cli_profile_bool(config.get("profile_expand_mwe"), default=False),
    )
    payload = _cli_profile_serialize(profile_dict, df, resolved, len(hits), upos_distribution, pos_distribution, config)
    if not isinstance(data, dict):
        return data
    profile_root = data.setdefault("profile", {})
    profile_root["collocational"] = payload
    metadata = data.setdefault("metadata", {})
    metadata["collocational_profile_included"] = True
    metadata["collocational_profile_hits_used"] = len(hits)
    if bool(config.get("profile_only")):
        data["results_included"] = False
        data["results"] = []
        data["returned_hits"] = 0
        data["has_more"] = False
        metadata["profile_only"] = True
    return data


def _attach_cli_analytics_to_schema_data(data, bundle):
    if not _CLI_ANALYTICS_CONFIG:
        return _attach_cli_collocational_profile_to_schema_data(data, bundle)
    analytics = _compute_cli_collocations_analytics(data, bundle)
    if analytics is None:
        return _attach_cli_collocational_profile_to_schema_data(data, bundle)
    private_collocate_results = None
    private_collocate_meta = {}
    if isinstance(analytics, dict):
        private_collocate_results = analytics.pop("_collocate_concordance_results", None)
        private_collocate_meta = analytics.pop("_collocate_concordance_metadata", {}) or {}
        if (private_collocate_results is None
            and str((_CLI_ANALYTICS_CONFIG or {}).get("concordance_of", "query") or "query").lower() == "collocates"
            and list((_CLI_ANALYTICS_CONFIG or {}).get("collocate_concordance") or [])):
            try:
                private_collocate_results, private_collocate_meta = _build_selected_collocate_concordance_results(data, bundle, analytics, dict(_CLI_ANALYTICS_CONFIG or {}))
            except Exception as _selected_collocate_concordance_exc:
                try:
                    analytics.setdefault("warnings", []).append(f"selected_collocate_concordance_runtime_error: {type(_selected_collocate_concordance_exc).__name__}: {_selected_collocate_concordance_exc}")
                except Exception:
                    pass
    if (private_collocate_results is None
        and str((_CLI_ANALYTICS_CONFIG or {}).get("concordance_of", "query") or "query").lower() == "collocates"
        and not list((_CLI_ANALYTICS_CONFIG or {}).get("collocate_concordance") or [])):
        try:
            private_collocate_results, private_collocate_meta = _build_default_collocate_concordance_results(data, bundle, analytics, dict(_CLI_ANALYTICS_CONFIG or {}))
        except Exception as _default_collocate_concordance_exc:
            try:
                analytics.setdefault("warnings", []).append(f"default_collocate_concordance_runtime_error: {type(_default_collocate_concordance_exc).__name__}: {_default_collocate_concordance_exc}")
            except Exception:
                pass
    data["analytics"] = analytics
    try:
        if _CLI_ANALYTICS_CONFIG and not bool((_CLI_ANALYTICS_CONFIG or {}).get("include_analytics_payload", True)):
            data.pop("analytics", None)
            metadata["analytics_requested"] = []
            metadata["analytics_included"] = []
            metadata["analytics_payload_included"] = False
    except Exception:
        pass
    metadata = data.setdefault("metadata", {})
    metadata["analytics_requested"] = analytics.get("requested", [])
    metadata["analytics_included"] = analytics.get("included", [])
    metadata["analytics_unavailable_reasons"] = analytics.get("unavailable_reasons", {})
    metadata["concordance_of"] = str(_CLI_ANALYTICS_CONFIG.get("concordance_of", "query") or "query")
    if (str(_CLI_ANALYTICS_CONFIG.get("concordance_of", "query") or "query").lower() == "collocates" and private_collocate_results is not None and not _CLI_ANALYTICS_CONFIG.get("analytics_only")):
        metadata["collocate_concordances_enabled"] = True
        metadata["collocate_concordance_source"] = "all_matches"
        metadata["collocate_concordance_sort"] = "collocate_rank_then_source_position"
        metadata["collocate_concordance_selected_count"] = len(private_collocate_meta.get("selected_collocates") or [])
        try:
            analytics_config_snapshot = dict(_CLI_ANALYTICS_CONFIG or {})
            if analytics_config_snapshot:
                metadata["collocation_computation_used"] = bool(analytics_config_snapshot.get("needs_collocation_computation", False))
                metadata["collocation_computation_reason"] = list(analytics_config_snapshot.get("collocation_computation_reason") or [])
                metadata["analytics_payload_included"] = bool(analytics_config_snapshot.get("include_analytics_payload", True))
        except Exception:
            pass
        collocate_concordance_selector_values = list((_CLI_ANALYTICS_CONFIG or {}).get("collocate_concordance") or [])
        if collocate_concordance_selector_values:
            metadata["collocate_concordance_selection_mode"] = "explicit"
            metadata["collocate_concordance_selector"] = collocate_concordance_selector_values[0] if len(collocate_concordance_selector_values) == 1 else collocate_concordance_selector_values
            metadata["collocate_concordance_selected_collocates"] = collocate_concordance_selector_values
        else:
            metadata["collocate_concordance_selection_mode"] = private_collocate_meta.get("collocate_concordance_selection_mode", "top_collocates")
        metadata["collocate_concordance_occurrence_count_before_limit"] = private_collocate_meta.get("total_hits")
        metadata["query_concordance_total_hits"] = data.get("total_hits")
        data["results"] = list(private_collocate_results or [])
        data["returned_hits"] = int(private_collocate_meta.get("returned_hits", len(data["results"])))
        data["total_hits"] = int(private_collocate_meta.get("total_hits", len(data["results"])))
        data["offset"] = int(private_collocate_meta.get("offset", data.get("offset", 0)) or 0)
        data["limit"] = private_collocate_meta.get("limit", data.get("limit"))
        data["has_more"] = bool(private_collocate_meta.get("has_more", False))
    if _CLI_ANALYTICS_CONFIG.get("analytics_only"):
        data["results_included"] = False
        data["results"] = []
        data["returned_hits"] = 0
    return _attach_cli_collocational_profile_to_schema_data(data, bundle)


def _schema_bundle_to_jsonable_for_cli(
    bundle,
    *,
    include_extended_context: bool,
    max_context_chars=None,
    fields=None,
):
    """Build schema-v1 dict for CLI JSON/JSONL output."""
    from korpusuj.search.output_schema import search_bundle_to_jsonable

    bundle_for_schema = bundle
    if include_extended_context:
        try:
            results = getattr(bundle, "results", []) or []
            results = _resolve_cli_extended_context_rows(results)
            bundle_for_schema = _bundle_with_cli_schema_results(bundle, results)
        except Exception:
            bundle_for_schema = bundle

    data = search_bundle_to_jsonable(bundle_for_schema)
    # KORPUSUJ_PATCH_156I_LIMIT_NONE_SCHEMA_NORMALIZATION
    try:
        request_obj = getattr(bundle_for_schema, "request", getattr(bundle, "request", None))
        if getattr(request_obj, "limit", None) is None:
            data["limit"] = None
            data["has_more"] = False
    except Exception:
        pass

    try:
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            data["metadata"] = metadata
        metadata["cli_json_schema_v1"] = True
        metadata["extended_context_included"] = bool(include_extended_context)
        metadata["extended_context_resolution_attempted"] = bool(include_extended_context)
    except Exception:
        pass

    if not include_extended_context:
        data = _drop_schema_result_extended_context(data)
    data = _truncate_schema_context(data, max_context_chars)
    data = _filter_schema_result_fields(data, fields)
    _attach_cli_analytics_to_schema_data(data, bundle)
    data = _attach_cli_search_statistics_to_schema_data(data, _CLI_SEARCH_STATISTICS_ARGS, bundle)
    return data


def _print_schema_json(
    bundle,
    *,
    pretty: bool,
    include_extended_context: bool,
    max_context_chars=None,
    fields=None,
) -> str:
    """Return one schema-v1 JSON envelope for --format json."""
    data = _schema_bundle_to_jsonable_for_cli(
        bundle,
        include_extended_context=include_extended_context,
        max_context_chars=max_context_chars,
        fields=fields,
    )
    if pretty:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _print_schema_jsonl(
    bundle,
    *,
    include_extended_context: bool,
    max_context_chars=None,
    fields=None,
) -> str:
    """Return schema-v1 JSONL for --format jsonl, one line per result."""
    data = _schema_bundle_to_jsonable_for_cli(
        bundle,
        include_extended_context=include_extended_context,
        max_context_chars=max_context_chars,
        fields=fields,
    )
    base = {
        "schema_version": data.get("schema_version"),
        "query": data.get("query"),
        "corpus": data.get("corpus"),
        "request": data.get("request"),
        "total_hits": data.get("total_hits"),
        "returned_hits": data.get("returned_hits"),
        "has_more": data.get("has_more"),
        "limit": data.get("limit"),
        "offset": data.get("offset"),
        "warnings": data.get("warnings", []),
        "messages": data.get("messages", []),
        "metadata": data.get("metadata", {}),
    }
    lines = []
    for idx, result in enumerate(data.get("results") or [], start=1):
        row = dict(base)
        row["result_index"] = idx
        row["result"] = result
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)
# END KORPUSUJ_151E_CLI_JSON_JSONL_SCHEMA_V1



def _cli_text_collapse_whitespace(value):
    """Normalize text for one-line terminal display without changing JSON data."""
    import re
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def _cli_text_tail(value, limit):
    text = _cli_text_collapse_whitespace(value)
    if limit is None or len(text) <= int(limit):
        return text
    return "..." + text[-int(limit):].lstrip()


def _cli_text_head(value, limit):
    text = _cli_text_collapse_whitespace(value)
    if limit is None or len(text) <= int(limit):
        return text
    return text[:int(limit)].rstrip() + "..."


def _cli_text_metadata_prefix(result, idx):
    metadata = result.get("metadata") if isinstance(result, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    date = _cli_text_collapse_whitespace(metadata.get("Data publikacji", ""))
    author = _cli_text_collapse_whitespace(metadata.get("Autor", ""))
    if author:
        return f"{idx}. {date} {author} |"
    if date:
        return f"{idx}. {date} |"
    return f"{idx}. |"


def _cli_text_result_line_from_schema(result, idx, *, context_chars=120):
    """Render one schema-v1 result row for human-readable CLI text output.

    Prefer extended_* fields because they are resolved from the fuller text context
    and preserve punctuation/spacing better than token-joined left/right context.
    Fall back to left_context/match_text/right_context when extended context is not
    available or was intentionally disabled.
    """
    if not isinstance(result, dict):
        return f"{idx}. {result!r}"
    has_extended = any(result.get(key) for key in ("extended_left", "extended_match", "extended_right"))
    if has_extended:
        left = _cli_text_tail(result.get("extended_left", ""), context_chars)
        match = _cli_text_collapse_whitespace(result.get("extended_match") or result.get("match_text") or result.get("match") or "")
        right = _cli_text_head(result.get("extended_right", ""), context_chars)
    else:
        left = _cli_text_tail(result.get("left_context", ""), context_chars)
        match = _cli_text_collapse_whitespace(result.get("match_text") or result.get("match") or "")
        right = _cli_text_head(result.get("right_context", ""), context_chars)
    prefix = _cli_text_metadata_prefix(result, idx)
    parts = []
    if left:
        parts.append(left)
    parts.append(f"[{match}]")
    if right:
        parts.append(right)
    return f"{prefix} {' '.join(parts)}"


def _cli_text_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _cli_text_table(rows, columns, *, max_rows=10):
    rows = [row for row in list(rows or []) if isinstance(row, dict)][: int(max_rows)]
    if not rows:
        return []
    widths = []
    for key, label in columns:
        width = len(str(label))
        for row in rows:
            width = max(width, len(_cli_text_value(row.get(key))))
        widths.append(min(width, 40))
    header = "  ".join(str(label).ljust(widths[i]) for i, (_, label) in enumerate(columns))
    sep = "  ".join("-" * widths[i] for i in range(len(columns)))
    out = [header, sep]
    for row in rows:
        cells = []
        for i, (key, _label) in enumerate(columns):
            cell = _cli_text_value(row.get(key))
            if len(cell) > widths[i]:
                cell = cell[: max(0, widths[i] - 3)] + "..."
            cells.append(cell.ljust(widths[i]))
        out.append("  ".join(cells))
    return out


def _cli_text_statistics_lines(data, *, max_rows=10):
    analytics = data.get("analytics") if isinstance(data, dict) else None
    statistics = (analytics or {}).get("statistics") if isinstance(analytics, dict) else None
    if not isinstance(statistics, dict):
        return []
    summary = statistics.get("summary") if isinstance(statistics.get("summary"), dict) else {}
    lines = ["", "STATISTICS"]
    for key, label in [
        ("scope", "scope"),
        ("unit", "unit"),
        ("hits_used", "hits_used"),
        ("total_hits", "total_hits"),
        ("displayed_results", "displayed_results"),
        ("display_limit", "display_limit"),
        ("statistics_results_materialized", "statistics_results_materialized"),
    ]:
        if key in summary:
            lines.append(f"{label}: {summary.get(key)}")
    global_tables = statistics.get("global") if isinstance(statistics.get("global"), dict) else {}
    fq_data_token = global_tables.get("fq_data_token") if isinstance(global_tables, dict) else None
    if isinstance(fq_data_token, list) and fq_data_token:
        lines.extend(["", f"TOP MATCH FORMS (first {min(int(max_rows), len(fq_data_token))})"])
        lines.extend(
            _cli_text_table(
                fq_data_token,
                [
                    ("rank", "rank"),
                    ("match", "match"),
                    ("frequency", "freq"),
                    ("pmw", "pmw"),
                    ("document_frequency", "docs"),
                    ("tfidf", "tfidf"),
                ],
                max_rows=max_rows,
            )
        )
    fq_data = global_tables.get("fq_data") if isinstance(global_tables, dict) else None
    if isinstance(fq_data, list) and fq_data:
        lines.extend(["", f"TOP MATCH LEMMAS/SPANS (first {min(int(max_rows), len(fq_data))})"])
        lines.extend(
            _cli_text_table(
                fq_data,
                [
                    ("rank", "rank"),
                    ("match", "match"),
                    ("frequency", "freq"),
                    ("pmw", "pmw"),
                    ("document_frequency", "docs"),
                    ("tfidf", "tfidf"),
                ],
                max_rows=max_rows,
            )
        )
    return lines


def _schema_data_for_cli_text(bundle, *, include_extended_context, max_context_chars=None, fields=None):
    """Build schema-v1 data for text rendering.

    statistics v1 is internally JSON-shaped. For text rendering, temporarily ask the
    existing statistics attachment path to build the same payload as JSON, then
    render a human-readable subset. This preserves JSONL/statistics rejection and
    does not change machine-readable schemas.
    """
    global _CLI_SEARCH_STATISTICS_ARGS
    original_args = globals().get("_CLI_SEARCH_STATISTICS_ARGS", None)
    restore_args = original_args
    if original_args is not None and str(getattr(original_args, "analytics", "none") or "none").lower() == "statistics":
        try:
            from types import SimpleNamespace
            try:
                args_dict = dict(vars(original_args))
            except TypeError:
                args_dict = dict(getattr(original_args, "__dict__", {}) or {})
            args_dict["format"] = "json"
            _CLI_SEARCH_STATISTICS_ARGS = SimpleNamespace(**args_dict)
        except Exception:
            pass
    try:
        return _schema_bundle_to_jsonable_for_cli(
            bundle,
            include_extended_context=include_extended_context,
            max_context_chars=max_context_chars,
            fields=fields,
        )
    finally:
        _CLI_SEARCH_STATISTICS_ARGS = restore_args




# BEGIN 166E CLI TEXT RESULTS/COLLOCATIONS/PROFILE RENDERING HELPERS

def _cli_text_truncate_middle(value, max_len):
    text = _cli_text_collapse_whitespace(value)
    if max_len is None or len(text) <= int(max_len):
        return text
    max_len = int(max_len)
    if max_len <= 6:
        return text[:max_len]
    left = max(1, (max_len - 3) // 2)
    right = max(1, max_len - 3 - left)
    return text[:left].rstrip() + "..." + text[-right:].lstrip()


def _cli_text_table_widths(rows, columns, *, max_rows=None):
    rows = [row for row in list(rows or []) if isinstance(row, dict)]
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    if not rows:
        return []
    widths = []
    for key, label, max_width in columns:
        width = len(str(label))
        for row in rows:
            width = max(width, len(_cli_text_value(row.get(key))))
        if max_width is not None:
            width = min(width, int(max_width))
        widths.append(width)

    def _align_cell(key, value, width, *, is_header=False):
        text = str(value)
        if key == "left_context":
            return text.rjust(width)
        return text.ljust(width)

    header = "  ".join(
        _align_cell(key, label, widths[i], is_header=True)
        for i, (key, label, _max_width) in enumerate(columns)
    )
    sep = "  ".join("-" * widths[i] for i in range(len(columns)))
    out = [header, sep]
    for row in rows:
        cells = []
        for i, (key, _label, _max_width) in enumerate(columns):
            cell = _cli_text_value(row.get(key))
            if len(cell) > widths[i]:
                if key == "left_context":
                    cell = "..." + cell[-max(0, widths[i] - 3):].lstrip()
                else:
                    cell = cell[: max(0, widths[i] - 3)].rstrip() + "..."
            cells.append(_align_cell(key, cell, widths[i]))
        out.append("  ".join(cells))
    return out


def _cli_text_tail_total(value, max_len):
    """Keep the right edge of text, including ellipsis inside max_len."""
    text = _cli_text_collapse_whitespace(value)
    if max_len is None:
        return text
    max_len = int(max_len)
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return "." * max_len
    return "..." + text[-(max_len - 3):].lstrip()


def _cli_text_head_total(value, max_len):
    """Keep the left edge of text, including ellipsis inside max_len."""
    text = _cli_text_collapse_whitespace(value)
    if max_len is None:
        return text
    max_len = int(max_len)
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return "." * max_len
    return text[: max_len - 3].rstrip() + "..."



def _cli_text_display_context(value):
    """Display-only cleanup for token-window context.

    The underlying contract remains --left-context/--right-context token windows.
    This helper only removes the most visible token-join artifacts in terminal
    text output; it does not affect JSON/JSONL or analytics payloads.
    """
    import re
    text = _cli_text_collapse_whitespace(value)
    if not text:
        return text
    # Remove spaces before punctuation and closing brackets/quotes.
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"\s+([”»])", r"\1", text)
    # Remove spaces after opening brackets/quotes.
    text = re.sub(r"([\(\[\{„«])\s+", r"\1", text)
    # Polish/typographic dash in tokenized text is often separated correctly;
    # keep surrounding spaces there.
    # Common abbreviation cleanup: r . -> r.; proc . -> proc.
    text = re.sub(r"\b(r|proc|tys|mln|mld)\s+\.", r"\1.", text, flags=re.IGNORECASE)
    return text.strip()

def _cli_text_result_components(result, idx, *, context_chars=None):
    """Return KWIC components for a text result row.

    RESULTS must respect --left-context and --right-context: use the schema
    left_context/right_context token windows, not full extended_* context. The
    extended_* fields can be huge and are intended for full/extended context
    consumers, not for the compact terminal Results table.

    No display truncation is applied by default. If context_chars is provided
    (from --max-context-chars), truncation is directional:
    - left_context keeps the words closest to the match, truncating from left;
    - right_context keeps the words closest to the match, truncating from right.
    """
    left_col_width = None if context_chars is None else int(context_chars)
    right_col_width = None if context_chars is None else int(context_chars)
    if not isinstance(result, dict):
        return {
            "#": idx,
            "date": "",
            "author": "",
            "left_context": "",
            "match": "",
            "right_context": repr(result),
        }
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    date = _cli_text_collapse_whitespace(metadata.get("Data publikacji", ""))
    author = _cli_text_collapse_whitespace(metadata.get("Autor", ""))

    left_source = result.get("left_context", "")
    match = _cli_text_display_context(result.get("match_text") or result.get("match") or "")
    right_source = result.get("right_context", "")

    if left_col_width is None:
        left = _cli_text_display_context(left_source)
    else:
        left = _cli_text_tail_total(_cli_text_display_context(left_source), left_col_width)
    if right_col_width is None:
        right = _cli_text_display_context(right_source)
    else:
        right = _cli_text_head_total(_cli_text_display_context(right_source), right_col_width)
    return {
        "#": idx,
        "date": date,
        "author": author,
        "left_context": left,
        "match": match,
        "right_context": right,
    }


def _cli_text_results_title(data):
    metadata = data.get("metadata") if isinstance(data, dict) and isinstance(data.get("metadata"), dict) else {}
    concordance_of = str(metadata.get("concordance_of", "query") or "query").lower()
    if concordance_of != "collocates":
        return "RESULTS"

    selection_mode = str(metadata.get("collocate_concordance_selection_mode", "") or "").lower()
    selector = metadata.get("collocate_concordance_selector")
    selected_collocates = metadata.get("collocate_concordance_selected_collocates")

    selected = None
    if selected_collocates:
        selected = selected_collocates
    elif selector:
        selected = selector

    # collocate_concordance_source is a provenance field such as "all_matches";
    # it is not the selected collocate label and should not be rendered as one.
    if selection_mode == "explicit" and selected:
        if isinstance(selected, (list, tuple)):
            selected_text = ", ".join(_cli_text_collapse_whitespace(x) for x in selected)
        else:
            selected_text = _cli_text_collapse_whitespace(selected)
        return f"RESULTS - COLLOCATE OCCURRENCES (selected: {selected_text})"

    if selection_mode == "top_collocates":
        return "RESULTS - COLLOCATE OCCURRENCES (top collocates)"

    return "RESULTS - COLLOCATE OCCURRENCES"


def _cli_text_results_lines(data, *, context_chars=None):
    results = list(data.get("results") or []) if isinstance(data, dict) else []
    if not results:
        return []
    rows = [_cli_text_result_components(row, idx, context_chars=context_chars) for idx, row in enumerate(results, start=1)]
    context_width = None if context_chars is None else int(context_chars)
    lines = ["", _cli_text_results_title(data)]
    lines.extend(
        _cli_text_table_widths(
            rows,
            [
                ("#", "#", 4),
                ("date", "date", 10),
                ("author", "author", 22),
                ("left_context", "left_context", context_width),
                ("match", "match", 24),
                ("right_context", "right_context", context_width),
            ],
        )
    )
    return lines


def _cli_text_collocations_lines(data, *, max_rows=10):
    analytics = data.get("analytics") if isinstance(data, dict) else None
    collocations = (analytics or {}).get("collocations") if isinstance(analytics, dict) else None
    if not isinstance(collocations, dict):
        return []
    rows = collocations.get("rows") if isinstance(collocations.get("rows"), list) else []
    source = collocations.get("source") if isinstance(collocations.get("source"), dict) else {}
    parameters = collocations.get("parameters") if isinstance(collocations.get("parameters"), dict) else {}
    lines = ["", "COLLOCATIONS"]
    for key, label in [
        ("scope", "scope"),
        ("match_count", "match_count"),
        ("total_hits", "total_hits"),
        ("analysis_complete", "analysis_complete"),
    ]:
        if key in source:
            lines.append(f"{label}: {source.get(key)}")
    param_bits = []
    for key in ["mode", "form", "sort", "limit", "left_span", "right_span", "sentence_bound", "syntactic_direction", "deprel"]:
        if key in parameters:
            param_bits.append(f"{key}={parameters.get(key)}")
    if param_bits:
        lines.append("parameters: " + ", ".join(param_bits))
    if rows:
        lines.extend(["", f"TOP COLLOCATIONS (first {min(int(max_rows), len(rows))})"])
        lines.extend(
            _cli_text_table_widths(
                rows,
                [
                    ("rank", "rank", 4),
                    ("collocate", "collocate", 24),
                    ("cooccurrences", "cooc", 8),
                    ("collocate_frequency", "colloc_freq", 12),
                    ("log_likelihood", "ll", 10),
                    ("mi", "mi", 8),
                    ("t_score", "t", 8),
                    ("log_dice", "logdice", 8),
                ],
                max_rows=max_rows,
            )
        )
    else:
        lines.append("rows: 0")
    warnings = collocations.get("warnings") if isinstance(collocations.get("warnings"), list) else []
    if warnings:
        lines.append("warnings: " + "; ".join(_cli_text_collapse_whitespace(w) for w in warnings[:5]))
    return lines


def _cli_text_profile_row_value(row, names):
    """Fetch a profile row value from flat or nested tree-shaped profile rows.

    Tree profile rows keep numeric values under counts.* and scores.*; flat rows
    expose the same values at top level. Names may therefore be plain keys or
    dotted paths, e.g. counts.cooc_freq or scores.log_dice.
    """
    if not isinstance(row, dict):
        return ""
    for name in names:
        if not isinstance(name, str):
            continue
        if "." in name:
            node = row
            ok = True
            for part in name.split("."):
                if isinstance(node, dict) and part in node:
                    node = node.get(part)
                else:
                    ok = False
                    break
            if ok and node is not None:
                return node
        elif name in row and row.get(name) is not None:
            return row.get(name)
    return ""


def _cli_text_profile_rows_for_table(rows, *, include_group_relation=False, max_rows=10):
    out = []
    for row in list(rows or [])[: int(max_rows)]:
        if not isinstance(row, dict):
            continue
        item = {
            "group": _cli_text_profile_row_value(row, ["group", "group_name"]),
            "relation": _cli_text_profile_row_value(row, ["relation", "relation_label", "deprel"]),
            "collocate": _cli_text_profile_row_value(row, ["display_collocate", "collocate", "lemma", "text"]),
            "upos": _cli_text_profile_row_value(row, ["collocate_upos", "upos"]),
            "cooc_freq": _cli_text_profile_row_value(row, ["cooc_freq", "counts.cooc_freq", "cooccurrences", "frequency", "freq"]),
            "doc_freq": _cli_text_profile_row_value(row, ["doc_freq", "counts.doc_freq", "document_frequency", "docs"]),
            "log_dice": _cli_text_profile_row_value(row, ["log_dice", "scores.log_dice", "logdice"]),
        }
        if not include_group_relation:
            item.pop("group", None)
            item.pop("relation", None)
        out.append(item)
    return out


def _cli_text_profile_summary_lines(profile):
    lines = []
    summary = profile.get("summary") if isinstance(profile.get("summary"), dict) else {}
    target = profile.get("target") if isinstance(profile.get("target"), dict) else {}
    for label, source in [("target", target), ("summary", summary)]:
        if not isinstance(source, dict) or not source:
            continue
        bits = []
        for key in ["lemma", "token", "target", "target_lemma", "target_token", "hits_used", "relation_count", "row_count", "layout"]:
            if key in source:
                bits.append(f"{key}={source.get(key)}")
        if bits:
            lines.append(f"{label}: " + ", ".join(bits))
    if not lines:
        for key in ["layout", "target", "target_lemma", "hits_used", "relation_count", "row_count"]:
            if key in profile:
                lines.append(f"{key}: {profile.get(key)}")
    return lines


def _cli_text_profile_group_heading(group_name, group_index):
    """Return a readable top-level profile group heading.

    Some profile group labels already contain a numeric prefix, e.g.
    "3. Modyfikatory...". Do not prefix those again, otherwise text output
    becomes "3. 3. Modyfikatory...".
    """
    import re
    label = _cli_text_collapse_whitespace(group_name or "")
    if not label:
        return f"{group_index}. Group {group_index}"
    if re.match(r"^\d+\.\s+", label):
        return label
    return f"{group_index}. {label}"

def _cli_text_profile_lines(data, *, max_rows_per_relation=5, max_flat_rows=20):
    profile_root = data.get("profile") if isinstance(data, dict) else None
    profile = (profile_root or {}).get("collocational") if isinstance(profile_root, dict) else None
    if not isinstance(profile, dict) or not profile:
        return []
    lines = ["", "COLLOCATIONAL PROFILE"]
    lines.extend(_cli_text_profile_summary_lines(profile))

    groups = profile.get("groups") if isinstance(profile.get("groups"), list) else None
    flat_rows = profile.get("flat_rows") if isinstance(profile.get("flat_rows"), list) else None

    if groups:
        lines.append("")
        for group_index, group in enumerate(groups, start=1):
            if not isinstance(group, dict):
                continue
            group_name = group.get("group") or group.get("name") or f"Group {group_index}"
            lines.append("")
            lines.append(_cli_text_profile_group_heading(group_name, group_index))
            relations = group.get("relations") if isinstance(group.get("relations"), list) else []
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                relation_name = _cli_text_collapse_whitespace(relation.get("relation") or relation.get("name") or relation.get("label") or "relation")
                rows = relation.get("rows") if isinstance(relation.get("rows"), list) else []
                if not rows:
                    continue
                lines.append("")
                lines.append(f"  {relation_name}")
                table_rows = _cli_text_profile_rows_for_table(rows, include_group_relation=False, max_rows=max_rows_per_relation)
                table = _cli_text_table_widths(
                    table_rows,
                    [
                        ("collocate", "collocate", 24),
                        ("upos", "upos", 8),
                        ("cooc_freq", "cooc", 8),
                        ("doc_freq", "docs", 8),
                        ("log_dice", "logdice", 8),
                    ],
                )
                lines.extend("  " + row for row in table)
    elif flat_rows:
        lines.extend(["", f"PROFILE FLAT ROWS (first {min(int(max_flat_rows), len(flat_rows))})"])
        table_rows = _cli_text_profile_rows_for_table(flat_rows, include_group_relation=True, max_rows=max_flat_rows)
        lines.extend(
            _cli_text_table_widths(
                table_rows,
                [
                    ("group", "group", 28),
                    ("relation", "relation", 28),
                    ("collocate", "collocate", 22),
                    ("upos", "upos", 8),
                    ("cooc_freq", "cooc", 8),
                    ("doc_freq", "docs", 8),
                    ("log_dice", "logdice", 8),
                ],
                max_rows=max_flat_rows,
            )
        )
    else:
        lines.append("rows: 0")
    return lines


def _cli_text_has_non_result_sections(data):
    if not isinstance(data, dict):
        return False
    analytics = data.get("analytics") if isinstance(data.get("analytics"), dict) else {}
    profile = data.get("profile") if isinstance(data.get("profile"), dict) else {}
    statistics = (analytics.get("statistics") if isinstance(analytics.get("statistics"), dict) else None)
    collocations = (analytics.get("collocations") if isinstance(analytics.get("collocations"), dict) else None)
    collocational = profile.get("collocational") if isinstance(profile.get("collocational"), dict) else None
    return bool(statistics or collocations or collocational)
# END 166E CLI TEXT RESULTS/COLLOCATIONS/PROFILE RENDERING HELPERS

def _print_schema_text(
    bundle,
    *,
    include_extended_context: bool,
    max_context_chars=None,
    fields=None,
) -> str:
    """Return human-readable schema-v1 based text output.

    Text is intentionally a human-readable view, not a complete machine-readable
    export. JSON remains the complete API surface. This renderer presents:
    - query/collocate occurrence RESULTS as a KWIC-like table;
    - analytics.statistics when present;
    - analytics.collocations when present;
    - profile.collocational when present.
    """
    data = _schema_data_for_cli_text(
        bundle,
        include_extended_context=include_extended_context,
        max_context_chars=max_context_chars,
        fields=fields,
    )
    results = list(data.get("results") or []) if isinstance(data, dict) else []
    total_hits = data.get("total_hits") if isinstance(data, dict) else getattr(bundle, "total_hits", None)
    returned_hits = data.get("returned_hits") if isinstance(data, dict) else len(results)
    lines = [f"Total hits: {total_hits}", f"Results: {returned_hits}"]
    context_chars = None if max_context_chars is None else max(20, int(max_context_chars))

    if results:
        lines.extend(_cli_text_results_lines(data, context_chars=context_chars))
    elif not _cli_text_has_non_result_sections(data):
        lines.extend(["", _cli_text_results_title(data), "(no results)"])

    lines.extend(_cli_text_statistics_lines(data, max_rows=10))
    lines.extend(_cli_text_collocations_lines(data, max_rows=10))
    lines.extend(_cli_text_profile_lines(data, max_rows_per_relation=5, max_flat_rows=20))
    return "\n".join(lines)


def _print_text(
    bundle: Any,
    *,
    max_context_chars: int | None,
    fields: list[str] | None,
) -> str:
    results = getattr(bundle, "results", []) or []
    lines = [f"Total hits: {getattr(bundle, 'total_hits', None)}", f"Results: {len(results)}"]
    for idx, row in enumerate(results, start=1):
        if max_context_chars is not None and isinstance(row, SearchHit):
            row = SearchHit(
                doc_id=row.doc_id,
                start=row.start,
                end=row.end,
                match_text=row.match_text,
                left_context=_truncate_str(row.left_context, max_context_chars),
                right_context=_truncate_str(row.right_context, max_context_chars),
                extended_left=_truncate_str(row.extended_left, max_context_chars),
                extended_match=_truncate_str(row.extended_match, max_context_chars),
                extended_right=_truncate_str(row.extended_right, max_context_chars),
                metadata=row.metadata,
                raw=row.raw,
            )
        lines.append(_result_text_line(row, idx))
    return "\n".join(lines)



def _cli_output_is_table_format(format_name):
    return str(format_name or "").lower() in {"xlsx", "csv"}


def _cli_output_require_table_output(args, parser):
    if not _cli_output_is_table_format(getattr(args, "format", None)):
        return
    if not getattr(args, "output", None):
        parser.error("--format xlsx/csv requires --output FILE")
    if getattr(args, "query_list", None):
        parser.error("--query-list with --format xlsx/csv is not supported in this version; use json/jsonl/text or run single-query exports.")


def _cli_output_table_value(row, *names, default=""):
    if row is None:
        return default
    for name in names:
        if isinstance(row, dict):
            if name in row:
                return row.get(name)
            parts = str(name).split(".")
            value = row
            ok = True
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value.get(part)
                else:
                    ok = False
                    break
            if ok:
                return value
        try:
            if hasattr(row, name):
                return getattr(row, name)
        except Exception:
            pass
    return default


def _cli_output_schema_result_to_export_row(result, index=0):
    metadata = _cli_output_table_value(result, "metadata", "additional_metadata", default={}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    hidden_metadata_keys = {
        "Data publikacji", "publication_date", "date",
        "Autor", "author",
        "Tytuł", "title",
        "month_key", "matched_lemmas", "lemma", "lemmas",
        "left_context", "right_context", "match_text", "match", "matched_text",
        "row_index", "row_idx", "doc_id", "start_idx", "start", "end_idx", "end",
    }
    additional_metadata = {str(k): v for k, v in dict(metadata).items() if str(k) not in hidden_metadata_keys}
    publication_date = (
        _cli_output_table_value(result, "publication_date", "date", default="")
        or metadata.get("Data publikacji", "")
        or metadata.get("publication_date", "")
    )
    title = (
        _cli_output_table_value(result, "title", "Tytuł", default="")
        or metadata.get("Tytuł", "")
        or metadata.get("title", "")
    )
    author = (
        _cli_output_table_value(result, "author", "Autor", default="")
        or metadata.get("Autor", "")
        or metadata.get("author", "")
    )
    compact_context = _cli_output_table_value(result, "context", default="")
    if isinstance(compact_context, (list, tuple)) and len(compact_context) >= 3:
        left_context = compact_context[0]
        match_text = compact_context[1]
        right_context = compact_context[2]
    else:
        left_context = _cli_output_table_value(result, "left_context", "left", default="")
        match_text = _cli_output_table_value(result, "match_text", "match", "matched_text", default="")
        right_context = _cli_output_table_value(result, "right_context", "right", default="")
    matched_lemmas = _cli_output_table_value(result, "matched_lemmas", "lemma", "lemmas", default="")
    if isinstance(matched_lemmas, (list, tuple)):
        matched_lemmas = " ".join(str(x) for x in matched_lemmas)
    row_index = _cli_output_table_value(result, "row_index", "row_idx", "doc_id", default=index)
    start_idx = _cli_output_table_value(result, "start_idx", "start", default="")
    end_idx = _cli_output_table_value(result, "end_idx", "end", default="")
    month_key = _cli_output_table_value(result, "month_key", default="") or metadata.get("month_key", "")
    full_text = _cli_output_table_value(result, "full_text_with_markers", "full_text", default="")
    return [
        publication_date,
        compact_context,
        full_text,
        match_text,
        matched_lemmas,
        month_key,
        title,
        author,
        additional_metadata,
        left_context,
        right_context,
        row_index,
        start_idx,
        end_idx,
    ]


def _cli_output_results_export_df(data, build_search_results_export_df):
    rows = list((data or {}).get("results") or [])
    export_rows = [_cli_output_schema_result_to_export_row(row, index=i) for i, row in enumerate(rows)]
    return build_search_results_export_df(export_rows)


def _cli_output_rows_to_table(rows, headers, mapping, build_table_export_df):
    table_rows = []
    for row in list(rows or []):
        if isinstance(row, dict):
            table_rows.append([_cli_output_table_value(row, *names, default="") for names in mapping])
        elif isinstance(row, (list, tuple)):
            table_rows.append(list(row))
        else:
            table_rows.append([_cli_output_table_value(row, *names, default="") for names in mapping])
    return build_table_export_df(table_rows, headers)


def _cli_output_add_statistics_sheets(data, sheets, helpers):
    analytics = (data or {}).get("analytics") if isinstance(data, dict) else None
    statistics = (analytics or {}).get("statistics") if isinstance(analytics, dict) else None
    if not isinstance(statistics, dict):
        return
    global_tables = statistics.get("global") if isinstance(statistics.get("global"), dict) else {}
    monthly_tables = statistics.get("monthly") if isinstance(statistics.get("monthly"), dict) else {}
    ranked_mapping = [
        ("rank", "Nr"),
        ("match", "lemma", "Forma podstawowa (base)"),
        ("frequency", "Liczba wystąpień"),
        ("pmw", "Częstość względna"),
        ("document_frequency", "Rozproszenie (DF)"),
        ("tfidf", "Ogólne TF-IDF"),
    ]
    month_mapping = [
        ("year", "Rok"),
        ("month", "Miesiąc"),
        ("match", "lemma", "Forma podstawowa"),
        ("frequency", "Liczba wystąpień"),
        ("pmw", "Częstość względna"),
        ("tfidf", "TF-IDF"),
        ("z_score", "zscore", "Z-score"),
    ]
    candidates = [
        ("Statistics_Lemma", global_tables.get("fq_data"), helpers["LEMMA_FREQUENCY_HEADERS"], ranked_mapping),
        ("Statistics_Token", global_tables.get("fq_data_token"), helpers["TOKEN_FREQUENCY_HEADERS"], ranked_mapping),
        ("Statistics_Monthly", monthly_tables.get("fq_data_month") or statistics.get("monthly_rows"), helpers["MONTH_FREQUENCY_HEADERS"], month_mapping),
    ]
    for name, rows, headers, mapping in candidates:
        if rows:
            sheets.append((name, _cli_output_rows_to_table(rows, headers, mapping, helpers["build_table_export_df"])))


def _cli_output_add_collocation_sheet(data, sheets, helpers):
    analytics = (data or {}).get("analytics") if isinstance(data, dict) else None
    collocations = (analytics or {}).get("collocations") if isinstance(analytics, dict) else None
    if not isinstance(collocations, dict):
        return
    rows = collocations.get("rows") or []
    if not rows:
        return
    mapping = [
        ("rank", "Nr"),
        ("collocate", "Kolokat"),
        ("cooccurrences", "fnc", "f(nc)"),
        ("collocate_frequency", "fc", "f(c)"),
        ("log_likelihood", "ll", "Log-Likelihood"),
        ("mi", "mi_score", "MI Score"),
        ("t_score", "t", "T-score"),
        ("log_dice", "Log-Dice"),
    ]
    sheets.append(("Collocations", _cli_output_rows_to_table(rows, helpers["COLLOCATION_HEADERS"], mapping, helpers["build_table_export_df"])))


def _cli_output_profile_rows(profile_payload):
    if not isinstance(profile_payload, dict):
        return []
    flat_rows = profile_payload.get("flat_rows")
    if isinstance(flat_rows, list) and flat_rows:
        return flat_rows
    rows = profile_payload.get("rows")
    if isinstance(rows, list) and rows:
        return rows
    groups = profile_payload.get("groups")
    out = []
    if isinstance(groups, list):
        for group in groups:
            group_name = ""
            if isinstance(group, dict):
                group_name = str(group.get("group") or group.get("name") or group.get("label") or "")
            relations = group.get("relations") if isinstance(group, dict) else None
            if isinstance(relations, list):
                for relation in relations:
                    relation_name = ""
                    if isinstance(relation, dict):
                        relation_name = str(
                            relation.get("relation")
                            or relation.get("name")
                            or relation.get("label")
                            or relation.get("relation_label")
                            or ""
                        )
                    rel_rows = relation.get("rows") if isinstance(relation, dict) else None
                    if isinstance(rel_rows, list):
                        for row in rel_rows:
                            if isinstance(row, dict):
                                item = dict(row)
                                item.setdefault("relation", relation_name)
                                item.setdefault("group", group_name)
                                out.append(item)
                            else:
                                out.append(row)
    return out


def _cli_output_profile_row_object(row):
    try:
        from types import SimpleNamespace
    except Exception:
        SimpleNamespace = None
    counts = row.get("counts") if isinstance(row, dict) and isinstance(row.get("counts"), dict) else {}
    scores = row.get("scores") if isinstance(row, dict) and isinstance(row.get("scores"), dict) else {}
    values = {
        "collocate": _cli_output_table_value(row, "collocate", "display_collocate", default=""),
        "collocate_upos": _cli_output_table_value(row, "collocate_upos", "upos", default=""),
        "relation": _cli_output_table_value(row, "relation", "relation_label", "deprel", default=""),
        "cooc_freq": _cli_output_table_value(row, "cooc_freq", "cooccurrences", "frequency", default=counts.get("cooc_freq", 0)),
        "doc_freq": _cli_output_table_value(row, "doc_freq", "document_frequency", "docs", default=counts.get("doc_freq", 0)),
        "global_freq": _cli_output_table_value(row, "global_freq", default=counts.get("global_freq", 0)),
        "ll_score": _cli_output_table_value(row, "ll_score", "log_likelihood", default=scores.get("log_likelihood", 0)),
        "mi_score": _cli_output_table_value(row, "mi_score", "mi", default=scores.get("mi", 0)),
        "t_score": _cli_output_table_value(row, "t_score", default=scores.get("t_score", 0)),
        "log_dice": _cli_output_table_value(row, "log_dice", default=scores.get("log_dice", 0)),
    }
    if SimpleNamespace is not None:
        return SimpleNamespace(**values)
    return type("ProfileExportRow", (), values)()


def _cli_output_add_profile_sheet(data, sheets, helpers):
    profile_root = (data or {}).get("profile") if isinstance(data, dict) else None
    profile_payload = (profile_root or {}).get("collocational") if isinstance(profile_root, dict) else None
    rows = _cli_output_profile_rows(profile_payload)
    if not rows:
        return
    relation = "Profile"
    objects = [_cli_output_profile_row_object(row) for row in rows]
    sheets.append((relation, helpers["build_profile_export_df"]({relation: objects})))


def _cli_output_build_export_sheets(data):
    from korpusuj.export.excel import (
        build_search_results_export_df,
        build_table_export_df,
        build_profile_export_df,
        LEMMA_FREQUENCY_HEADERS,
        TOKEN_FREQUENCY_HEADERS,
        MONTH_FREQUENCY_HEADERS,
        COLLOCATION_HEADERS,
        PROFILE_HEADERS,
    )
    helpers = {
        "build_search_results_export_df": build_search_results_export_df,
        "build_table_export_df": build_table_export_df,
        "build_profile_export_df": build_profile_export_df,
        "LEMMA_FREQUENCY_HEADERS": LEMMA_FREQUENCY_HEADERS,
        "TOKEN_FREQUENCY_HEADERS": TOKEN_FREQUENCY_HEADERS,
        "MONTH_FREQUENCY_HEADERS": MONTH_FREQUENCY_HEADERS,
        "COLLOCATION_HEADERS": COLLOCATION_HEADERS,
        "PROFILE_HEADERS": PROFILE_HEADERS,
    }
    sheets = []
    if (data or {}).get("results"):
        sheets.append(("Results", _cli_output_results_export_df(data, build_search_results_export_df)))
    _cli_output_add_statistics_sheets(data, sheets, helpers)
    _cli_output_add_collocation_sheet(data, sheets, helpers)
    _cli_output_add_profile_sheet(data, sheets, helpers)
    return [(name, df) for name, df in sheets if df is not None]



def _cli_output_schema_data_for_table_format(bundle, args, *, include_extended_context, max_context_chars=None, fields=None):
    """Build schema-v1 data for xlsx/csv while preserving JSON-only analytics internals.

    Some analytics builders intentionally validate the normal textual CLI format
    and currently accept only json. XLSX/CSV are table projections of the same
    schema data, so build the schema under the json contract, then restore args.
    """
    original_format = getattr(args, "format", None)
    profile_config = globals().get("_CLI_PROFILE_CONFIG", None)
    original_profile_format = None
    try:
        try:
            setattr(args, "format", "json")
        except Exception:
            pass
        if isinstance(profile_config, dict):
            original_profile_format = profile_config.get("format")
            profile_config["format"] = "json"
        return _schema_bundle_to_jsonable_for_cli(
            bundle,
            include_extended_context=include_extended_context,
            max_context_chars=max_context_chars,
            fields=fields,
        )
    finally:
        try:
            setattr(args, "format", original_format)
        except Exception:
            pass
        if isinstance(profile_config, dict):
            if original_profile_format is None:
                profile_config.pop("format", None)
            else:
                profile_config["format"] = original_profile_format

def _cli_output_write_table_format(data, output_path, format_name):
    if not output_path:
        raise SystemExit("--format xlsx/csv requires --output FILE")
    from korpusuj.export.excel import write_excel_workbook, write_csv_export
    sheets = _cli_output_build_export_sheets(data)
    if not sheets:
        raise SystemExit("No exportable tables were produced for --format xlsx/csv.")
    fmt = str(format_name or "").lower()
    if fmt == "xlsx":
        write_excel_workbook(output_path, sheets)
        return
    if fmt == "csv":
        preferred = ["Results", "Statistics_Lemma", "Statistics_Token", "Statistics_Monthly", "Collocations", "Profile"]
        by_name = {name: df for name, df in sheets}
        for name in preferred:
            if name in by_name:
                write_csv_export(output_path, by_name[name])
                return
        write_csv_export(output_path, sheets[0][1])
        return
    raise SystemExit(f"Unsupported tabular output format: {format_name}")


def _cli_subcorpus_requested(args):
    return bool(getattr(args, "subcorpus_output", None))


def _cli_subcorpus_selector_values(args):
    return {
        "query": getattr(args, "subcorpus_query", None),
        "author": getattr(args, "subcorpus_author", None),
        "title": getattr(args, "subcorpus_title", None),
        "date_from": getattr(args, "subcorpus_date_from", None),
        "date_to": getattr(args, "subcorpus_date_to", None),
    }


def _cli_subcorpus_has_selector(args):
    selectors = _cli_subcorpus_selector_values(args)
    return any(str(v or "").strip() for v in selectors.values())


def _cli_subcorpus_has_metadata_selector(args):
    selectors = _cli_subcorpus_selector_values(args)
    return any(str(selectors.get(k) or "").strip() for k in ("author", "title", "date_from", "date_to"))


def _cli_subcorpus_validate_args(args, parser):
    subcorpus_selector_used = any(
        str(getattr(args, name, "") or "").strip()
        for name in (
            "subcorpus_query",
            "subcorpus_author",
            "subcorpus_title",
            "subcorpus_date_from",
            "subcorpus_date_to",
        )
    )
    if subcorpus_selector_used and not getattr(args, "subcorpus_output", None):
        parser.error("subcorpus selectors require --subcorpus-output FILE.parquet")

    if not _cli_subcorpus_requested(args):
        if not (getattr(args, "query", None) or getattr(args, "query_file", None) or getattr(args, "query_list", None)):
            parser.error("one of --query, --query-file or --query-list is required unless --subcorpus-output is used")
        return

    # More-specific conflicts must be reported before the generic missing-selector
    # error, otherwise users do not learn that main --query is intentionally not
    # reused for subcorpus export.
    if getattr(args, "query_list", None):
        parser.error("--query-list with --subcorpus-output is not supported in this version")
    if getattr(args, "query", None) or getattr(args, "query_file", None):
        parser.error("--subcorpus-output does not implicitly use --query/--query-file; use --subcorpus-query for query-based subcorpus export")

    if not _cli_subcorpus_has_selector(args):
        parser.error("--subcorpus-output requires at least one explicit selector: --subcorpus-query or metadata filters")

    if getattr(args, "output", None):
        parser.error("--subcorpus-output is a standalone Parquet artifact; do not combine it with --output in v1")
    if str(getattr(args, "format", "json") or "json") != "json":
        parser.error("--subcorpus-output is not a --format output mode; keep --format json or omit --format")
    if str(getattr(args, "analytics", "none") or "none") != "none" or bool(getattr(args, "analytics_only", False)):
        parser.error("--subcorpus-output cannot be combined with --analytics in v1")
    if getattr(args, "profile", None) or bool(getattr(args, "profile_only", False)):
        parser.error("--subcorpus-output cannot be combined with --profile in v1")




def _cli_subcorpus_load_dataframe(corpus_path):
    import pandas as pd
    return pd.read_parquet(corpus_path, engine="pyarrow")


def _cli_subcorpus_run_query(args, corpus_name):
    context = build_search_service_context_from_parquet(
        args.corpus_path,
        corpus_name=corpus_name,
        limit=None,
        normalize_hits=False,
        normalize_fn=None,
        full_context_size=args.full_context_size,
        candidate_max_docs=args.candidate_max_docs,
        candidate_stream_batch_docs=args.candidate_stream_batch_docs,
        config={"source": "cli_subcorpus_export", "normalized": False},
    )
    request = SearchRequest(
        query=str(getattr(args, "subcorpus_query", "") or ""),
        corpus_name=corpus_name,
        left_context=int(getattr(args, "left_context", 10) or 10),
        right_context=int(getattr(args, "right_context", 10) or 10),
        sort_option=None,
        limit=None,
        offset=0,
        options={"client": "cli", "stage": "subcorpus_export", "normalized": False},
    )
    return run_search_service(request, context)


def _cli_subcorpus_result_rows(bundle):
    try:
        return list(getattr(bundle, "results", []) or [])
    except Exception:
        return []


def _cli_subcorpus_write_status(status, *, pretty=False):
    text = json.dumps(status, ensure_ascii=False, indent=2 if pretty else None)
    print(text)


def _run_cli_subcorpus_export(args, parser) -> int:
    try:
        from korpusuj.export.subcorpus import (
            select_rows_from_search_results,
            filter_dataframe_by_metadata,
            export_dataframe_to_subcorpus_parquet,
        )
        corpus_path = Path(args.corpus_path)
        corpus_name = args.corpus_name or corpus_path.stem
        df = _cli_subcorpus_load_dataframe(corpus_path)
        selected_df = df
        query_info = None

        if str(getattr(args, "subcorpus_query", "") or "").strip():
            bundle = _cli_subcorpus_run_query(args, corpus_name)
            rows = _cli_subcorpus_result_rows(bundle)
            selected_df = select_rows_from_search_results(df, rows)
            query_info = {
                "query": getattr(args, "subcorpus_query", None),
                "total_hits": getattr(bundle, "total_hits", None),
                "result_rows_used": len(rows),
            }

        if _cli_subcorpus_has_metadata_selector(args):
            selected_df = filter_dataframe_by_metadata(
                selected_df,
                date_from=getattr(args, "subcorpus_date_from", None),
                date_to=getattr(args, "subcorpus_date_to", None),
                author=getattr(args, "subcorpus_author", None),
                title=getattr(args, "subcorpus_title", None),
            )

        try:
            row_count = int(len(selected_df))
        except Exception:
            row_count = 0
        if row_count <= 0:
            print("subcorpus export produced no rows; no file was written", file=sys.stderr)
            return 2

        metadata = export_dataframe_to_subcorpus_parquet(selected_df, getattr(args, "subcorpus_output"))
        status = {
            "ok": True,
            "schema_version": "subcorpus-export-v1",
            "corpus": corpus_name,
            "corpus_path": str(corpus_path),
            "subcorpus_output": str(getattr(args, "subcorpus_output")),
            "rows": row_count,
            "selectors": _cli_subcorpus_selector_values(args),
            "query": query_info,
            "metadata_summary": {
                "total_tokens": metadata.get("total_tokens") if isinstance(metadata, dict) else None,
                "base_tf_size": len(metadata.get("base_tf", {})) if isinstance(metadata, dict) else None,
                "orth_tf_size": len(metadata.get("orth_tf", {})) if isinstance(metadata, dict) else None,
                "monthly_years": len(metadata.get("monthly_token_counts", {})) if isinstance(metadata, dict) else None,
            },
        }
        _cli_subcorpus_write_status(status, pretty=bool(getattr(args, "pretty", False)))
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        err = {
            "ok": False,
            "schema_version": "subcorpus-export-v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "subcorpus_output": str(getattr(args, "subcorpus_output", "") or ""),
        }
        print(json.dumps(err, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

def _write_output(text: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8", newline="")
    else:
        print(text)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the public argument parser for search, analytics and export."""
    parser = argparse.ArgumentParser(
        prog="python -m korpusuj.search.cli",
        description="Minimal Korpusuj headless search CLI, 036L4G47.",
        allow_abbrev=False,
    )
    parser.add_argument("--corpus-path", required=True, help="Path to corpus .parquet file")
    parser.add_argument("--corpus-name", default=None, help="Optional corpus name; default is parquet stem")
    query_group = parser.add_mutually_exclusive_group(required=False)
    query_group.add_argument("--query", help='CQL query, e.g. [base="wojna"]')
    query_group.add_argument("--query-file", help="Path to UTF-8 text file containing one CQL query")
    query_group.add_argument("--query-list", help="Path to UTF-8 text file containing one CQL query per non-empty line")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of concordance results to return; default: all")
    parser.add_argument("--offset", type=int, default=0, help="Request offset; currently passed through to SearchRequest")
    parser.add_argument("--left-context", type=int, default=10, help="Number of tokens shown to the left of each concordance match; default: 10")
    parser.add_argument("--right-context", type=int, default=10, help="Number of tokens shown to the right of each concordance match; default: 10")
    parser.add_argument("--full-context-size", type=int, default=250, help="Internal full-context window size for resolving extended/raw context; default: 250")
    parser.add_argument("--candidate-max-docs", type=int, default=3000, help="Dependency candidate preload budget in documents; performance/cache knob, not a result limit; default: 3000")
    parser.add_argument("--candidate-stream-batch-docs", type=int, default=256, help="Dependency candidate streaming batch size in documents; performance knob, not a result limit; default: 256")
    parser.add_argument("--format", choices=["json", "jsonl", "text", "xlsx", "csv"], default="json", help="Output format: json envelope, jsonl one record per result, text, xlsx workbook, or csv table; xlsx/csv require --output; default: json")
    parser.add_argument("--analytics", choices=["none", "collocations", "statistics"], default="none", help="Include analytics payload; supports collocations and statistics")
    parser.add_argument("--analytics-only", action="store_true", help="Omit concordance result rows and return analytics payload only")
    parser.add_argument("--analytics-scope", choices=["all-matches", "returned-results"], default="all-matches", help="Scope used to compute analytics; default all-matches")
    parser.add_argument("--concordance-of", choices=["query", "collocates"], default="query", help="Top-level concordance rows: query matches or collocate occurrences; default: query")
    parser.add_argument("--collocate-concordance", action="append", default=[], help="Select one collocate label for --concordance-of collocates occurrence rows; repeatable; label space follows --colloc-form")
    parser.add_argument("--collocate-filter", action="append", default=[], help="Repeatable collocate candidate filter, e.g. upos=NOUN,pos=subst,tag=sg:nom; filters collocates only")
    parser.add_argument("--colloc-mode", choices=["linear", "syntactic"], default="linear", help="Collocation mode: linear token window or syntactic dependency-based collocates; default: linear")
    parser.add_argument("--colloc-syn-dir", choices=["dependent", "head", "both"], default="dependent", help="Syntactic collocation direction: dependent=Podrzędnik, head=Nadrzędnik, both=all directions; default: dependent")
    parser.add_argument("--colloc-deprel", default="Wszystkie", help="Dependency relation filter for syntactic collocations, e.g. amod; default: all/Wszystkie")
    parser.add_argument("--colloc-form", choices=["base", "orth"], default="base", help="Form used for collocate labels and background frequencies: base lemma or orth surface form; default: base")
    parser.add_argument("--colloc-left-span", type=int, default=5, help="Left window size in tokens for linear collocations; default: 5")
    parser.add_argument("--colloc-right-span", type=int, default=5, help="Right window size in tokens for linear collocations; default: 5")
    parser.add_argument("--colloc-sentence-bound", choices=["true", "false"], default="true", help="Limit collocate candidate windows to the sentence of the source match; default: true")
    parser.add_argument("--colloc-min-freq", type=int, default=1, help="Minimum co-occurrence frequency required for a collocate row; default: 1")
    parser.add_argument("--colloc-min-range", type=int, default=1, help="Minimum number of distinct documents/results in which a collocate must occur; default: 1")
    parser.add_argument("--colloc-sort", choices=["log-likelihood", "mi", "t-score", "log-dice"], default="log-dice")
    parser.add_argument("--colloc-limit", type=int, default=None, help="Maximum rows in analytics.collocations.rows after ranking; does not affect concordance results; default: all rows")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--normalized", action="store_true", default=True, help="Return SearchHit-normalized results; default")
    mode.add_argument("--raw", action="store_true", help="Return raw SearchCursor/legacy-shaped rows")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--progress", choices=["auto", "off", "on"], default="auto", help="Show CLI progress spinner on stderr: auto=TTY only, off=never, on=force; default: auto")

    parser.add_argument("--profile", choices=["collocational"], default=None, help="Add a collocational profile / profil kolokacyjny payload to JSON output.")

    parser.add_argument("--profile-only", action="store_true", help="Return collocational profile payload without top-level concordance results.")
    parser.add_argument("--profile-target-token", type=int, default=None, help="1-based target token index for collocational profile; required for multi-token queries.")
    parser.add_argument("--profile-target-lemma", default=None, help="Optional target lemma override for collocational profile; normally inferred from base=... in the target token.")
    parser.add_argument("--profile-sort", choices=["log-dice", "log-likelihood", "mi", "t-score", "frequency"], default="log-dice", help="Sort profile rows within each relation by selected measure.")
    parser.add_argument("--profile-min-freq", type=int, default=2, help="Minimum co-occurrence frequency for profile rows; default: 2.")
    parser.add_argument("--profile-max-rows-per-relation", type=int, default=None, help="Limit profile rows per relation; default: unlimited.")
    parser.add_argument("--profile-layout", choices=["tree", "flat"], default="tree", help="Profile payload layout; tree matches the GUI group -> relation -> rows structure; flat is table-like.")
    parser.add_argument("--profile-example-refs", type=int, default=0, help="Include up to N technical example_refs per profile row; default: 0.")
    parser.add_argument("--profile-examples", type=int, default=0, help="Include up to N textual examples per profile row; default: 0.")
    parser.add_argument("--profile-example-context", type=int, default=6, help="Token context size for textual profile examples; default: 6.")
    parser.add_argument("--profile-expand-mwe", choices=["true", "false"], default="false", help="Expand multi-word expression collocates in profile computation; default: false.")
    parser.add_argument("--include-request", action="store_true", help="Include SearchRequest in JSON envelope")
    parser.add_argument("--include-raw", action="store_true", help="Include SearchHit.raw in normalized JSON/JSONL output")
    parser.add_argument("--output", default=None, help="Write output to file instead of stdout")
    parser.add_argument("--subcorpus-output", default=None, help="Create a Parquet subcorpus at FILE using explicit subcorpus selectors")
    parser.add_argument("--subcorpus-query", default=None, help="CQL query used only for subcorpus creation; does not use main --query implicitly")
    parser.add_argument("--subcorpus-author", default=None, help="Metadata selector for subcorpus creation: Autor contains VALUE, case-insensitive")
    parser.add_argument("--subcorpus-title", default=None, help="Metadata selector for subcorpus creation: Tytuł contains VALUE, case-insensitive")
    parser.add_argument("--subcorpus-date-from", default=None, help="Metadata selector for subcorpus creation: Data publikacji >= YYYY-MM-DD/string")
    parser.add_argument("--subcorpus-date-to", default=None, help="Metadata selector for subcorpus creation: Data publikacji <= YYYY-MM-DD/string")
    parser.add_argument("--max-context-chars", type=int, default=None, help="Truncate context string fields to N chars")
    ext_group = parser.add_mutually_exclusive_group()
    ext_group.add_argument("--no-extended-context", action="store_true", help="Omit extended_left/extended_match/extended_right from JSON/JSONL schema output; text RESULTS use left/right context windows")
    parser.add_argument("--fields", default=None, help="Comma-separated result fields to include in JSON/JSONL, e.g. doc_id,match_text,metadata")
    parser.add_argument("--continue-on-error", action="store_true", help="In --query-list mode, emit per-query error records and continue")
    return parser



class _CliProgressSpinner163i:
    """Tiny stderr-only CLI spinner.

    This helper is intentionally local to CLI output handling. It never writes to
    stdout, because stdout carries JSON/JSONL/text result output.
    """

    def __init__(self, desc="Working", enabled=False, delay=0.4, interval=0.1, stream=None):
        self.desc = str(desc or "Working")
        self.enabled = bool(enabled)
        self.delay = float(delay)
        self.interval = float(interval)
        self.stream = stream
        self._thread = None
        self._stop = None
        self._started = False
        self._steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]

    def start(self):
        if not self.enabled:
            return self
        try:
            import sys as _sys
            import threading as _threading
            self.stream = self.stream or _sys.stderr
            self._stop = _threading.Event()
            self._thread = _threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
            self._started = True
        except Exception:
            self.enabled = False
        return self

    def _animate(self):
        try:
            import itertools as _itertools
            import time as _time
            _time.sleep(self.delay)
            if self._stop is None or self._stop.is_set():
                return
            for step in _itertools.cycle(self._steps):
                if self._stop is None or self._stop.is_set():
                    break
                print(f"\r{self.desc} {step}", end="", file=self.stream, flush=True)
                _time.sleep(self.interval)
        except Exception:
            return

    def stop(self):
        if not self.enabled:
            return
        try:
            if self._stop is not None:
                self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.3)
            import shutil as _shutil
            cols = _shutil.get_terminal_size((80, 20)).columns
            print("\r" + " " * cols + "\r", end="", file=self.stream, flush=True)
        except Exception:
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def _cli_progress_enabled(args):
    mode = str(getattr(args, "progress", "auto") or "auto").lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    try:
        import sys as _sys
        return bool(_sys.stderr.isatty())
    except Exception:
        return False

def main(argv: list[str] | None = None) -> int:
    """Run the search CLI and return its process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _cli_subcorpus_validate_args(args, parser)
    if _cli_subcorpus_requested(args):
        return _run_cli_subcorpus_export(args, parser)
    global _CLI_PROFILE_CONFIG
    _CLI_PROFILE_CONFIG = _build_cli_profile_config(args)
    _progress_spinner = _CliProgressSpinner163i("Searching", enabled=_cli_progress_enabled(args))
    try:
        _progress_spinner.start()
        _set_cli_analytics_config_from_args(args)

        _set_cli_search_statistics_args(args)
        corpus_path = Path(args.corpus_path)
        corpus_name = args.corpus_name or corpus_path.stem
        normalize_hits = not bool(args.raw)
        if getattr(args, "query_list", None):
            return _run_query_list_cli(args)

        query = _read_query_file(args.query_file) if args.query_file else str(args.query)
        include_extended_context = not bool(args.no_extended_context)
        fields = _parse_fields(args.fields)

        try:
            ctx = build_search_service_context_from_parquet(
                corpus_path,
                corpus_name=corpus_name,
                limit=args.limit,
                normalize_hits=normalize_hits,
                normalize_fn=normalize_search_results_to_hits if normalize_hits else None,
                full_context_size=args.full_context_size,
                candidate_max_docs=args.candidate_max_docs,
                candidate_stream_batch_docs=args.candidate_stream_batch_docs,
                config={"source": "cli_036L4G47", "format": args.format, "normalized": normalize_hits},
            )
            req = SearchRequest(
                query=query,
                corpus_name=corpus_name,
                left_context=int(args.left_context),
                right_context=int(args.right_context),
                sort_option="Alfabetycznie",
                limit=(None if args.limit is None else int(args.limit)),
                offset=int(args.offset),
                options={"client": "cli", "stage": "036L4G47", "normalized": normalize_hits},
            )
            bundle = run_search_service(req, ctx)
        except Exception as exc:
            err = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "stage": "036L4G47",
                "query_received": locals().get("query", getattr(args, "query", None)),
                "query_file": getattr(args, "query_file", None),
                "corpus_path": str(getattr(args, "corpus_path", "")),
                "corpus_name": locals().get("corpus_name", getattr(args, "corpus_name", None)),
                "normalized": locals().get("normalize_hits", None),
                "format": getattr(args, "format", None),
            }
            _progress_spinner.stop()
            print(json.dumps(err, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2

        if _cli_output_is_table_format(args.format):
            data = _cli_output_schema_data_for_table_format(
                bundle,
                args,
                include_extended_context=include_extended_context,
                max_context_chars=args.max_context_chars,
                fields=fields,
            )
            _progress_spinner.stop()
            _cli_output_write_table_format(data, args.output, args.format)
            return 0
        if args.format == "json":
            output = _print_schema_json(
                bundle,
                pretty=args.pretty,
                include_extended_context=include_extended_context,
                max_context_chars=args.max_context_chars,
                fields=fields,
            )
        elif args.format == "jsonl":
            output = _print_schema_jsonl(
                bundle,
                include_extended_context=include_extended_context,
                max_context_chars=args.max_context_chars,
                fields=fields,
            )
        else:
            output = _print_schema_text(
                bundle,
                include_extended_context=include_extended_context,
                max_context_chars=args.max_context_chars,
                fields=fields,
            )
        _progress_spinner.stop()
        _write_output(output, args.output)
        return 0
    finally:
        _progress_spinner.stop()

# KORPUSUJ_PATCH_137C_DIAGNOSTIC_LOGGING_FLAGS_AND_CONFIG_CLI
# Adds --verbose and --diagnostics-logs to the headless CLI, plus flat config defaults:
#   logging_verbose
#   logging_diagnostics_logs
# Priority: explicit CLI flag > existing env var > config default.
def _install_cli_logging_flags_137c():
    try:
        import os as _os_137c
        import sys as _sys_137c
        import json as _json_137c
        from pathlib import Path as _Path_137c
    except Exception:
        return

    if globals().get("_korpusuj_137c_cli_logging_flags_installed", False):
        return

    TRUTHY = {"1", "true", "yes", "tak", "on", "debug", "verbose"}

    def _truthy_137c(value):
        try:
            if value is True:
                return True
            if isinstance(value, str):
                return value.strip().lower() in TRUTHY
            return bool(value)
        except Exception:
            return False

    def _load_config_137c():
        candidates = []
        try:
            # cli.py -> korpusuj/search/cli.py, project root is parents[2]
            candidates.append(_Path_137c(__file__).resolve().parents[2] / "config.json")
        except Exception:
            pass
        candidates.append(_Path_137c("config.json"))
        for p in candidates:
            try:
                if p.exists():
                    data = _json_137c.loads(p.read_text(encoding="utf-8", errors="replace"))
                    return data if isinstance(data, dict) else {}
            except Exception:
                pass
        return {}

    def _has_env_137c(*names):
        try:
            return any(name in _os_137c.environ for name in names)
        except Exception:
            return False

    def _apply_137c(argv=None):
        argv = list(_sys_137c.argv[1:] if argv is None else argv)
        cfg = _load_config_137c()
        verbose_flag = "--verbose" in argv
        diag_flag = "--diagnostics-logs" in argv

        if (not _has_env_137c("KORPUSUJ_VERBOSE_LOGS", "KORPUSUJ_VERBOSE")) and _truthy_137c(cfg.get("logging_verbose", False)):
            _os_137c.environ["KORPUSUJ_VERBOSE_LOGS"] = "1"
        if (not _has_env_137c("KORPUSUJ_137_DIAGNOSTIC_LOGS")) and _truthy_137c(cfg.get("logging_diagnostics_logs", False)):
            _os_137c.environ["KORPUSUJ_137_DIAGNOSTIC_LOGS"] = "1"

        if verbose_flag:
            _os_137c.environ["KORPUSUJ_VERBOSE_LOGS"] = "1"
        if diag_flag:
            _os_137c.environ["KORPUSUJ_137_DIAGNOSTIC_LOGS"] = "1"
            _os_137c.environ.setdefault("KORPUSUJ_VERBOSE_LOGS", "1")

    orig_build = globals().get("build_arg_parser")
    if callable(orig_build) and not getattr(orig_build, "_korpusuj_137c_logging_flags_wrapped", False):
        def build_arg_parser_with_logging_flags_137c(*args, **kwargs):
            parser = orig_build(*args, **kwargs)
            try:
                existing = set()
                for action in getattr(parser, "_actions", []) or []:
                    for opt in getattr(action, "option_strings", []) or []:
                        existing.add(opt)
                if "--verbose" not in existing:
                    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs for this process")
                if "--diagnostics-logs" not in existing:
                    parser.add_argument("--diagnostics-logs", action="store_true", help="Enable detailed [DIAG ...] execution logs for this process")
            except Exception:
                pass
            return parser
        build_arg_parser_with_logging_flags_137c._korpusuj_137c_logging_flags_wrapped = True
        globals()["build_arg_parser"] = build_arg_parser_with_logging_flags_137c

    orig_main = globals().get("main")
    if callable(orig_main) and not getattr(orig_main, "_korpusuj_137c_logging_flags_wrapped", False):
        def main_with_logging_flags_137c(argv=None):
            _apply_137c(_sys_137c.argv[1:] if argv is None else argv)
            return orig_main(argv)
        main_with_logging_flags_137c._korpusuj_137c_logging_flags_wrapped = True
        globals()["main"] = main_with_logging_flags_137c

    globals()["_korpusuj_137c_cli_logging_flags_installed"] = True

try:
    _install_cli_logging_flags_137c()
except Exception:
    pass
# END KORPUSUJ_PATCH_137C_DIAGNOSTIC_LOGGING_FLAGS_AND_CONFIG_CLI


# KORPUSUJ_PATCH_137D_CLI_SEARCHHIT_JSON_AND_LAZY_FULLTEXT_SANITIZER
# Fixes CLI JSON/JSONL output so SearchHit rows are emitted as dictionaries,
# not repr(SearchHit(...)), and lazy full-text references are omitted from
# extended_* fields instead of leaking internal SearchIndex objects.
def _install_cli_searchhit_json_sanitizer_137d():
    try:
        import dataclasses as _dataclasses_137d
    except Exception:
        _dataclasses_137d = None

    if globals().get("_korpusuj_137d_cli_searchhit_json_sanitizer_installed", False):
        return

    _SearchHit_137d = globals().get("SearchHit")
    _orig_jsonable_137d = globals().get("_jsonable")
    _orig_apply_137d = globals().get("_apply_result_output_controls")
    if not callable(_orig_jsonable_137d) or not callable(_orig_apply_137d):
        return

    _LAZY_SENTINEL_137d = "__KORPUSUJ_LAZY_FULLTEXT_REF_111__"

    def _is_lazy_fulltext_value_137d(value):
        try:
            if value == _LAZY_SENTINEL_137d:
                return True
        except Exception:
            pass
        try:
            if isinstance(value, (list, tuple)) and value and value[0] == _LAZY_SENTINEL_137d:
                return True
        except Exception:
            pass
        try:
            s = repr(value)
            if _LAZY_SENTINEL_137d in s:
                return True
            if "SearchIndex object at" in s:
                return True
        except Exception:
            pass
        return False

    def _safe_scalar_or_jsonable_137d(value):
        if _is_lazy_fulltext_value_137d(value):
            return None
        try:
            return _orig_jsonable_137d(value)
        except Exception:
            try:
                return repr(value)
            except Exception:
                return None

    def _searchhit_to_dict_137d(hit):
        out = {}
        for name in ("doc_id", "start", "end", "match_text", "left_context", "right_context"):
            try:
                value = getattr(hit, name)
            except Exception:
                continue
            if value is not None:
                out[name] = _safe_scalar_or_jsonable_137d(value)

        # Only include extended_* when they are real text/context values. Omit lazy refs,
        # because null would look like a broken field and internal objects must not leak.
        for name in ("extended_left", "extended_match", "extended_right"):
            try:
                value = getattr(hit, name)
            except Exception:
                continue
            if value is None or _is_lazy_fulltext_value_137d(value):
                continue
            out[name] = _safe_scalar_or_jsonable_137d(value)

        try:
            metadata = getattr(hit, "metadata", None)
            if metadata:
                out["metadata"] = _orig_jsonable_137d(metadata)
        except Exception:
            pass

        try:
            raw = getattr(hit, "raw", None)
            if raw is not None:
                out["raw"] = _orig_jsonable_137d(raw)
        except Exception:
            pass
        return out

    def _jsonable_137d(value):
        try:
            if _SearchHit_137d is not None and isinstance(value, _SearchHit_137d):
                return _searchhit_to_dict_137d(value)
        except Exception:
            pass
        # Some SearchHit-like objects may come from compatible modules/classes.
        try:
            if all(hasattr(value, attr) for attr in ("doc_id", "start", "end", "match_text", "left_context", "right_context")):
                return _searchhit_to_dict_137d(value)
        except Exception:
            pass
        return _orig_jsonable_137d(value)

    def _strip_raw_fields_137d(value):
        if isinstance(value, dict):
            return {str(k): _strip_raw_fields_137d(v) for k, v in value.items() if k != "raw"}
        if isinstance(value, list):
            return [_strip_raw_fields_137d(v) for v in value]
        if isinstance(value, tuple):
            return [_strip_raw_fields_137d(v) for v in value]
        return value

    def _drop_extended_context_137d(value):
        if isinstance(value, dict):
            return {str(k): _drop_extended_context_137d(v) for k, v in value.items() if not str(k).startswith("extended_")}
        if isinstance(value, list):
            return [_drop_extended_context_137d(v) for v in value]
        if isinstance(value, tuple):
            return [_drop_extended_context_137d(v) for v in value]
        return value

    def _truncate_str_137d(value, max_chars):
        if max_chars is None:
            return value
        try:
            n = int(max_chars)
        except Exception:
            return value
        if n < 0:
            return value
        if isinstance(value, str) and len(value) > n:
            return value[:n] + "…"
        if isinstance(value, dict):
            return {k: _truncate_str_137d(v, n) for k, v in value.items()}
        if isinstance(value, list):
            return [_truncate_str_137d(v, n) for v in value]
        return value

    def _apply_result_output_controls_137d(row, *, include_raw, include_extended_context, max_context_chars, fields):
        data = _jsonable_137d(row)
        if not include_raw:
            data = _strip_raw_fields_137d(data)
        if not include_extended_context:
            data = _drop_extended_context_137d(data)
        data = _truncate_str_137d(data, max_context_chars)
        if fields and isinstance(data, dict):
            wanted = {str(f) for f in fields}
            data = {k: v for k, v in data.items() if k in wanted}
        return data

    try:
        globals()["_jsonable"] = _jsonable_137d
        globals()["_apply_result_output_controls"] = _apply_result_output_controls_137d
        globals()["_korpusuj_137d_cli_searchhit_json_sanitizer_installed"] = True
    except Exception:
        pass

try:
    _install_cli_searchhit_json_sanitizer_137d()
except Exception:
    pass
# END KORPUSUJ_PATCH_137D_CLI_SEARCHHIT_JSON_AND_LAZY_FULLTEXT_SANITIZER


# KORPUSUJ_PATCH_137E_CLI_FREQUENCY_TOTAL_HITS_AND_EXTENDED_GROUP_SANITIZER
# Fixes two CLI/headless output issues:
# 1) For frequency_* queries, total_hits must be the post-frequency-filter total,
#    not the requested materialization limit.
# 2) Lazy full-text refs in any extended_* component mean the whole extended_* group
#    is internal/lazy and should be omitted from CLI JSON, not partially emitted.
def _install_cli_frequency_total_hits_and_extended_group_sanitizer_137e():
    try:
        import dataclasses as _dataclasses_137e
        import copy as _copy_137e
        import re as _re_137e
    except Exception:
        _dataclasses_137e = None
        _copy_137e = None
        _re_137e = None

    if globals().get("_korpusuj_137e_cli_frequency_total_hits_and_extended_group_sanitizer_installed", False):
        return

    _SearchHit_137e = globals().get("SearchHit")
    _orig_apply_137e = globals().get("_apply_result_output_controls")
    _orig_jsonable_137e = globals().get("_jsonable")
    _orig_run_search_service_137e = globals().get("run_search_service")
    if not callable(_orig_apply_137e) or not callable(_orig_jsonable_137e):
        return

    _LAZY_SENTINEL_137e = "__KORPUSUJ_LAZY_FULLTEXT_REF_111__"

    def _is_frequency_query_137e(query):
        try:
            q = str(query or "")
            return ("frequency_base" in q.lower()) or ("frequency_orth" in q.lower())
        except Exception:
            return False

    def _is_lazy_or_internal_137e(value):
        try:
            if value == _LAZY_SENTINEL_137e:
                return True
        except Exception:
            pass
        try:
            if isinstance(value, (list, tuple)) and value and value[0] == _LAZY_SENTINEL_137e:
                return True
        except Exception:
            pass
        try:
            s = repr(value)
            if _LAZY_SENTINEL_137e in s:
                return True
            if "SearchIndex object at" in s:
                return True
            if "korpusuj.index.sqlite_index.SearchIndex" in s:
                return True
        except Exception:
            pass
        return False

    def _row_has_lazy_extended_group_137e(row):
        # Direct extended fields.
        for name in ("extended_left", "extended_match", "extended_right"):
            try:
                if _is_lazy_or_internal_137e(getattr(row, name)):
                    return True
            except Exception:
                pass
        # Raw SearchCursor tuple contains full_text/lazy payload at index 2.
        try:
            raw = getattr(row, "raw", None)
            if raw is not None and _is_lazy_or_internal_137e(raw):
                return True
        except Exception:
            pass
        # Dict-shaped rows.
        try:
            if isinstance(row, dict):
                for name in ("extended_left", "extended_match", "extended_right", "raw"):
                    if name in row and _is_lazy_or_internal_137e(row.get(name)):
                        return True
        except Exception:
            pass
        return False

    def _drop_extended_group_137e(value):
        if isinstance(value, dict):
            return {str(k): _drop_extended_group_137e(v) for k, v in value.items() if not str(k).startswith("extended_")}
        if isinstance(value, list):
            return [_drop_extended_group_137e(v) for v in value]
        if isinstance(value, tuple):
            return [_drop_extended_group_137e(v) for v in value]
        return value

    def _apply_result_output_controls_137e(row, *, include_raw, include_extended_context, max_context_chars, fields):
        data = _orig_apply_137e(
            row,
            include_raw=include_raw,
            include_extended_context=include_extended_context,
            max_context_chars=max_context_chars,
            fields=fields,
        )
        if _row_has_lazy_extended_group_137e(row):
            data = _drop_extended_group_137e(data)
        # Defensive cleanup in case an earlier sanitizer emitted a partial lazy group.
        if isinstance(data, dict):
            ext_values = [data.get(k) for k in ("extended_left", "extended_match", "extended_right") if k in data]
            if any(_is_lazy_or_internal_137e(v) for v in ext_values):
                data = _drop_extended_group_137e(data)
            # A lone numeric/string extended_right is a strong sign of a split lazy ref.
            if "extended_right" in data and "extended_left" not in data and "extended_match" not in data:
                v = data.get("extended_right")
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
                    data = _drop_extended_group_137e(data)
        return data

    def _get_limit_offset_137e(req):
        try:
            limit = getattr(req, "limit", None)
        except Exception:
            limit = None
        try:
            offset = int(getattr(req, "offset", 0) or 0)
        except Exception:
            offset = 0
        try:
            limit_i = None if limit is None else int(limit)
        except Exception:
            limit_i = None
        return limit_i, max(0, offset)

    def _clone_req_unlimited_137e(req):
        try:
            if _dataclasses_137e is not None and _dataclasses_137e.is_dataclass(req):
                return _dataclasses_137e.replace(req, limit=None, offset=0)
        except Exception:
            pass
        try:
            new = _copy_137e.copy(req) if _copy_137e is not None else req
            try: setattr(new, "limit", None)
            except Exception: pass
            try: setattr(new, "offset", 0)
            except Exception: pass
            return new
        except Exception:
            return req

    def _clone_ctx_unlimited_137e(ctx):
        try:
            if _dataclasses_137e is not None and _dataclasses_137e.is_dataclass(ctx):
                fields = {f.name for f in _dataclasses_137e.fields(ctx)}
                kwargs = {}
                if "limit" in fields:
                    kwargs["limit"] = None
                if kwargs:
                    return _dataclasses_137e.replace(ctx, **kwargs)
        except Exception:
            pass
        try:
            new = _copy_137e.copy(ctx) if _copy_137e is not None else ctx
            if hasattr(new, "limit"):
                try: setattr(new, "limit", None)
                except Exception: pass
            return new
        except Exception:
            return ctx

    def _bundle_with_limited_results_137e(bundle, total_hits, limit, offset):
        results = list(getattr(bundle, "results", []) or [])
        start = max(0, int(offset or 0))
        end = None if limit is None else start + max(0, int(limit))
        page = results[start:end]
        has_more = (start + len(page)) < int(total_hits or 0)
        out_limit = limit
        try:
            if _dataclasses_137e is not None and _dataclasses_137e.is_dataclass(bundle):
                fields = {f.name for f in _dataclasses_137e.fields(bundle)}
                kwargs = {}
                if "results" in fields: kwargs["results"] = page
                if "total_hits" in fields: kwargs["total_hits"] = int(total_hits or 0)
                if "limit" in fields: kwargs["limit"] = out_limit
                if "offset" in fields: kwargs["offset"] = offset
                if "has_more" in fields: kwargs["has_more"] = bool(has_more)
                return _dataclasses_137e.replace(bundle, **kwargs)
        except Exception:
            pass
        try:
            setattr(bundle, "results", page)
        except Exception:
            pass
        for name, value in (("total_hits", int(total_hits or 0)), ("limit", out_limit), ("offset", offset), ("has_more", bool(has_more))):
            try:
                setattr(bundle, name, value)
            except Exception:
                pass
        return bundle

    def run_search_service_137e(req, ctx, *args, **kwargs):
        if not callable(_orig_run_search_service_137e):
            raise RuntimeError("run_search_service unavailable")
        try:
            query = getattr(req, "query", "")
        except Exception:
            query = ""
        if not _is_frequency_query_137e(query):
            return _orig_run_search_service_137e(req, ctx, *args, **kwargs)
        limit, offset = _get_limit_offset_137e(req)
        # For frequency aggregates, first compute the full post-filter result set.
        # Then page it locally so total_hits remains the aggregate total, not LIMIT.
        full_req = _clone_req_unlimited_137e(req)
        full_ctx = _clone_ctx_unlimited_137e(ctx)
        bundle = _orig_run_search_service_137e(full_req, full_ctx, *args, **kwargs)
        full_results = list(getattr(bundle, "results", []) or [])
        try:
            total = int(getattr(bundle, "total_hits", None) or len(full_results))
        except Exception:
            total = len(full_results)
        # If the lower layer still returned a suspicious total equal to the requested
        # limit, trust the fully materialized post-filter result list.
        if total < len(full_results):
            total = len(full_results)
        return _bundle_with_limited_results_137e(bundle, total, limit, offset)

    try:
        globals()["_apply_result_output_controls"] = _apply_result_output_controls_137e
        if callable(_orig_run_search_service_137e):
            globals()["run_search_service"] = run_search_service_137e
        globals()["_korpusuj_137e_cli_frequency_total_hits_and_extended_group_sanitizer_installed"] = True
    except Exception:
        pass

try:
    _install_cli_frequency_total_hits_and_extended_group_sanitizer_137e()
except Exception:
    pass
# END KORPUSUJ_PATCH_137E_CLI_FREQUENCY_TOTAL_HITS_AND_EXTENDED_GROUP_SANITIZER


# KORPUSUJ_PATCH_137I_CLI_CQL_BARE_VALUE_AUTOQUOTE_RETRY
# CLI-only ergonomics fix for shells that strip quotes in --query values, e.g.
#   --query '[base="wojna"]' reaching Python as [base=wojna].
# Tokenizer-aware autoquote for known textual CQL keys inside [] and nested {}.
# It does not quote structural values like dependent={...}, and it does not touch
# frequency tags or top/min/max parameters.
def _install_cli_cql_bare_value_autoquote_137i():
    try:
        import os as _os_137i
        import sys as _sys_137i
        import re as _re_137i
    except Exception:
        return

    g = globals()
    if g.get("_korpusuj_137i_cli_cql_bare_value_autoquote_installed", False):
        return

    AUTOQUOTE_TEXT_KEYS_137I = {
        "orth", "window_base", "window_orth",
        "base", "pos", "upos", "ner",
        "head", "coref", "dependent", "deprel",
        "number", "gender", "degree", "case", "person",
        "accentability", "post-prepositionality",
        "accommodability", "aspect", "vocalicity",
        "agglutination", "negation",
        "autor", "tytuł", "tytul", "children.group",
    }
    NO_AUTOQUOTE_KEYS_137I = {"top", "min", "max", "data", "frequency_base", "frequency_orth", "metadane"}
    TRUTHY_137I = {"1", "true", "yes", "tak", "on"}

    def _enabled_137i():
        try:
            val = str(_os_137i.environ.get("KORPUSUJ_137I_CLI_AUTOQUOTE", "1")).strip().lower()
            return val not in {"0", "false", "no", "nie", "off"}
        except Exception:
            return True

    def _is_key_char_137i(ch):
        return ch.isalnum() or ch in {"_", "-", "."}

    def _skip_ws_137i(s, i):
        n = len(s)
        while i < n and s[i].isspace():
            i += 1
        return i

    def _read_key_137i(s, i):
        n = len(s)
        if i >= n or not (s[i].isalpha() or s[i] in {"_"}):
            return None, i, None
        j = i + 1
        while j < n and _is_key_char_137i(s[j]):
            j += 1
        key = s[i:j]
        # Allow window_base(10)=... and window_orth(10)=..., but normalize key to window_base/window_orth.
        k = _skip_ws_137i(s, j)
        suffix_end = j
        if k < n and s[k] == "(":
            depth = 1
            m = k + 1
            while m < n and depth:
                if s[m] == "(":
                    depth += 1
                elif s[m] == ")":
                    depth -= 1
                m += 1
            if depth == 0:
                suffix_end = m
        return key, j, suffix_end

    def _looks_numeric_137i(value):
        try:
            return bool(_re_137i.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$", value or ""))
        except Exception:
            return False

    def _needs_quote_value_137i(key, value):
        if not value:
            return False
        lk = str(key or "").lower()
        if lk in NO_AUTOQUOTE_KEYS_137I:
            return False
        if lk not in AUTOQUOTE_TEXT_KEYS_137I:
            return False
        if value[0] in {'"', "'", "{", "[", "<"}:
            return False
        # Do not quote obvious booleans/numbers only for non-text future keys; for known text keys quote even pos/number sg etc.
        return True

    def _read_bare_value_137i(s, i):
        n = len(s)
        j = i
        while j < n:
            ch = s[j]
            if ch.isspace() or ch in {"&", "|", "]", "}"}:
                break
            # Do not consume opening structural delimiters as bare scalar values.
            if ch in {"{", "[", "<"}:
                break
            j += 1
        return s[i:j], j

    def _autoquote_cql_137i(query):
        try:
            s = str(query or "")
        except Exception:
            return query
        n = len(s)
        out = []
        i = 0
        quote = None
        square_depth = 0
        brace_depth = 0
        angle_depth = 0
        changed = False

        while i < n:
            ch = s[i]
            if quote:
                out.append(ch)
                if ch == "\\" and i + 1 < n:
                    i += 1
                    out.append(s[i])
                elif ch == quote:
                    quote = None
                i += 1
                continue
            if ch in {'"', "'"}:
                quote = ch
                out.append(ch)
                i += 1
                continue
            if ch == "[":
                square_depth += 1
                out.append(ch)
                i += 1
                continue
            if ch == "]":
                square_depth = max(0, square_depth - 1)
                out.append(ch)
                i += 1
                continue
            if ch == "{":
                brace_depth += 1
                out.append(ch)
                i += 1
                continue
            if ch == "}":
                brace_depth = max(0, brace_depth - 1)
                out.append(ch)
                i += 1
                continue
            if ch == "<":
                angle_depth += 1
                out.append(ch)
                i += 1
                continue
            if ch == ">":
                angle_depth = max(0, angle_depth - 1)
                out.append(ch)
                i += 1
                continue

            # Only inside token/dependency conditions, never inside <frequency ...> tags.
            if (square_depth > 0 or brace_depth > 0) and angle_depth == 0 and (ch.isalpha() or ch == "_"):
                key, key_name_end, key_suffix_end = _read_key_137i(s, i)
                if key:
                    op_pos = _skip_ws_137i(s, key_suffix_end)
                    op = None
                    op_end = op_pos
                    if op_pos < n and s.startswith("!=", op_pos):
                        op = "!="
                        op_end = op_pos + 2
                    elif op_pos < n and s[op_pos] == "=":
                        op = "="
                        op_end = op_pos + 1
                    # Avoid date comparison and other <=/>= forms here.
                    if op:
                        val_pos = _skip_ws_137i(s, op_end)
                        if val_pos < n and s[val_pos] not in {'"', "'", "{", "[", "<"}:
                            value, value_end = _read_bare_value_137i(s, val_pos)
                            if value and _needs_quote_value_137i(key, value):
                                out.append(s[i:val_pos])
                                out.append('"')
                                out.append(value)
                                out.append('"')
                                i = value_end
                                changed = True
                                continue
            out.append(ch)
            i += 1
        return "".join(out) if changed else s

    def _find_query_arg_index_137i(argv):
        try:
            value_flags = {"--query", "--subcorpus-query"}
            inline_prefixes = ("--query=", "--subcorpus-query=")
            for idx, arg in enumerate(argv):
                if arg in value_flags and idx + 1 < len(argv):
                    return idx + 1
                if isinstance(arg, str) and arg.startswith(inline_prefixes):
                    return idx
        except Exception:
            pass
        return None

    def _apply_argv_autoquote_137i(argv=None):
        if not _enabled_137i():
            return None
        if argv is None:
            argv = _sys_137i.argv
        idx = _find_query_arg_index_137i(argv)
        if idx is None:
            return None
        try:
            raw_arg = argv[idx]
            if isinstance(raw_arg, str) and (raw_arg.startswith("--query=") or raw_arg.startswith("--subcorpus-query=")):
                flag_name, original = raw_arg.split("=", 1)
                effective = _autoquote_cql_137i(original)
                if effective != original:
                    argv[idx] = flag_name + "=" + effective
            else:
                original = str(raw_arg)
                effective = _autoquote_cql_137i(original)
                if effective != original:
                    argv[idx] = effective
            if effective != original:
                g["_korpusuj_137i_last_query_original"] = original
                g["_korpusuj_137i_last_query_effective"] = effective
                return {"original": original, "effective": effective}
        except Exception:
            return None
        return None

    orig_main = g.get("main")
    if callable(orig_main) and not getattr(orig_main, "_korpusuj_137i_autoquote_wrapped", False):
        def main_with_cql_autoquote_137i(argv=None):
            try:
                if argv is None:
                    _apply_argv_autoquote_137i(_sys_137i.argv)
                else:
                    local = list(argv)
                    # local argv passed to main() excludes program name.
                    fake = ["<argv>"] + local
                    result = _apply_argv_autoquote_137i(fake)
                    if result is not None:
                        argv = fake[1:]
            except Exception:
                pass
            return orig_main(argv)
        main_with_cql_autoquote_137i._korpusuj_137i_autoquote_wrapped = True
        g["main"] = main_with_cql_autoquote_137i

    orig_bundle_to_dict = g.get("_bundle_to_dict")
    if callable(orig_bundle_to_dict) and not getattr(orig_bundle_to_dict, "_korpusuj_137i_metadata_wrapped", False):
        def _bundle_to_dict_137i(bundle, *args, **kwargs):
            data = orig_bundle_to_dict(bundle, *args, **kwargs)
            try:
                original = g.get("_korpusuj_137i_last_query_original")
                effective = g.get("_korpusuj_137i_last_query_effective")
                if original is not None and effective is not None and original != effective and isinstance(data, dict):
                    meta = data.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                        data["metadata"] = meta
                    meta["query_autoquoted_137i"] = True
                    meta["query_original"] = original
                    meta["query_effective"] = effective
            except Exception:
                pass
            return data
        _bundle_to_dict_137i._korpusuj_137i_metadata_wrapped = True
        g["_bundle_to_dict"] = _bundle_to_dict_137i

    g["_korpusuj_137i_cli_cql_bare_value_autoquote_installed"] = True

try:
    _install_cli_cql_bare_value_autoquote_137i()
except Exception:
    pass
# END KORPUSUJ_PATCH_137I_CLI_CQL_BARE_VALUE_AUTOQUOTE_RETRY


# KORPUSUJ_PATCH_137J_CLI_FREQUENCY_FULL_CONTEXT_BEFORE_PAGING
# CLI-only fix for frequency_* total_hits. For frequency queries, the original
# CLI built SearchBackendContext with limit=args.limit, so the adapter captured
# the page limit before run_search_service wrappers could compute a full count.
# This wrapper rewrites the CLI argv for frequency queries before original main()
# builds context: internal run uses a large full limit and offset=0, then
# run_search_service output is paged back to the user's requested limit/offset.
def _install_cli_frequency_full_context_before_paging_137j():
    try:
        import os as _os_137j
        import sys as _sys_137j
        import dataclasses as _dataclasses_137j
    except Exception:
        _dataclasses_137j = None

    g = globals()
    if g.get("_korpusuj_137j_cli_frequency_full_context_installed", False):
        return

    def _full_limit_137j():
        try:
            return max(1, int(str(_os_137j.environ.get("KORPUSUJ_CLI_FREQUENCY_FULL_LIMIT", "5000000")).strip()))
        except Exception:
            return 5000000

    def _is_frequency_query_137j(query):
        try:
            q = str(query or "").lower()
            return ("frequency_base" in q) or ("frequency_orth" in q)
        except Exception:
            return False

    def _arg_value_137j(argv, name, default=None):
        try:
            prefix = name + "="
            for i, a in enumerate(argv):
                if a == name and i + 1 < len(argv):
                    return argv[i + 1]
                if isinstance(a, str) and a.startswith(prefix):
                    return a.split("=", 1)[1]
        except Exception:
            pass
        return default

    def _set_arg_value_137j(argv, name, value):
        value = str(value)
        prefix = name + "="
        try:
            for i, a in enumerate(argv):
                if a == name and i + 1 < len(argv):
                    argv[i + 1] = value
                    return argv
                if isinstance(a, str) and a.startswith(prefix):
                    argv[i] = prefix + value
                    return argv
            argv.extend([name, value])
        except Exception:
            pass
        return argv

    def _read_query_from_argv_137j(argv):
        q = _arg_value_137j(argv, "--query", None)
        if q is not None:
            return str(q), "query"
        qf = _arg_value_137j(argv, "--query-file", None)
        if qf:
            try:
                text = open(qf, "r", encoding="utf-8-sig", errors="replace").read()
                return text.strip().lstrip("\ufeff").replace('\\"', '"').replace("\\'", "'"), "query-file"
            except Exception:
                return None, "query-file-error"
        return None, None

    def _prepare_argv_137j(argv_with_prog):
        try:
            argv = list(argv_with_prog)
            query, source = _read_query_from_argv_137j(argv[1:])
            if not _is_frequency_query_137j(query):
                return argv, None
            requested_limit_present = any((a == "--limit") or (isinstance(a, str) and a.startswith("--limit=")) for a in argv[1:])
            requested_limit_raw = _arg_value_137j(argv[1:], "--limit", None)
            requested_offset_raw = _arg_value_137j(argv[1:], "--offset", 0)
            try:
                requested_limit = None if not requested_limit_present else int(requested_limit_raw)
            except Exception:
                requested_limit = None
            try:
                requested_offset = int(requested_offset_raw)
            except Exception:
                requested_offset = 0
            requested_limit = None if requested_limit is None else max(1, requested_limit)
            requested_offset = max(0, requested_offset)
            full_limit = _full_limit_137j()
            # Mutate the argv used by original main before it builds context.
            tail = argv[1:]
            tail = _set_arg_value_137j(tail, "--limit", full_limit)
            tail = _set_arg_value_137j(tail, "--offset", 0)
            new_argv = [argv[0]] + tail
            state = {
                "active": True,
                "query": query,
                "query_source": source,
                "requested_limit": requested_limit,
                "requested_offset": requested_offset,
                "full_limit": full_limit,
            }
            return new_argv, state
        except Exception:
            return argv_with_prog, None

    def _page_bundle_137j(bundle, state):
        if not state or not state.get("active"):
            return bundle
        try:
            results = list(getattr(bundle, "results", []) or [])
        except Exception:
            return bundle
        total = len(results)
        requested_limit = int(state.get("requested_limit") or 20)
        requested_offset = int(state.get("requested_offset") or 0)
        start = max(0, requested_offset)
        end = start + max(1, requested_limit)
        page = results[start:end]
        has_more = (start + len(page)) < total
        truncated_by_full_limit = total >= int(state.get("full_limit") or 0)

        def _meta_137j(meta):
            if not isinstance(meta, dict):
                meta = {}
            meta["cli_frequency_full_context_137j"] = True
            meta["frequency_full_result_count_137j"] = int(total)
            meta["frequency_requested_limit_137j"] = int(requested_limit)
            meta["frequency_requested_offset_137j"] = int(requested_offset)
            meta["frequency_internal_full_limit_137j"] = int(state.get("full_limit") or 0)
            if truncated_by_full_limit:
                meta["frequency_full_limit_truncated_137j"] = True
            return meta

        try:
            if _dataclasses_137j is not None and _dataclasses_137j.is_dataclass(bundle):
                fields = {f.name for f in _dataclasses_137j.fields(bundle)}
                kwargs = {}
                if "results" in fields: kwargs["results"] = page
                if "total_hits" in fields: kwargs["total_hits"] = int(total)
                if "limit" in fields: kwargs["limit"] = int(requested_limit)
                if "offset" in fields: kwargs["offset"] = int(requested_offset)
                if "has_more" in fields: kwargs["has_more"] = bool(has_more)
                if "metadata" in fields:
                    kwargs["metadata"] = _meta_137j(getattr(bundle, "metadata", None))
                return _dataclasses_137j.replace(bundle, **kwargs)
        except Exception:
            pass

        try: setattr(bundle, "results", page)
        except Exception: pass
        for name, value in (("total_hits", int(total)), ("limit", int(requested_limit)), ("offset", int(requested_offset)), ("has_more", bool(has_more))):
            try: setattr(bundle, name, value)
            except Exception: pass
        try: setattr(bundle, "metadata", _meta_137j(getattr(bundle, "metadata", None)))
        except Exception: pass
        return bundle

    orig_run = g.get("run_search_service")
    if callable(orig_run) and not getattr(orig_run, "_korpusuj_137j_run_wrapped", False):
        def run_search_service_137j(request, context, *args, **kwargs):
            bundle = orig_run(request, context, *args, **kwargs)
            state = g.get("_korpusuj_137j_frequency_state")
            return _page_bundle_137j(bundle, state)
        run_search_service_137j._korpusuj_137j_run_wrapped = True
        g["run_search_service"] = run_search_service_137j

    orig_main = g.get("main")
    if callable(orig_main) and not getattr(orig_main, "_korpusuj_137j_main_wrapped", False):
        def main_with_frequency_full_context_137j(argv=None):
            old_state = g.get("_korpusuj_137j_frequency_state")
            old_sys_argv = None
            try:
                if argv is None:
                    old_sys_argv = list(_sys_137j.argv)
                    new_argv, state = _prepare_argv_137j(_sys_137j.argv)
                    g["_korpusuj_137j_frequency_state"] = state
                    if state:
                        _sys_137j.argv[:] = new_argv
                    return orig_main(None)
                else:
                    fake = ["<argv>"] + list(argv)
                    new_argv, state = _prepare_argv_137j(fake)
                    g["_korpusuj_137j_frequency_state"] = state
                    return orig_main(new_argv[1:])
            finally:
                if old_sys_argv is not None:
                    try: _sys_137j.argv[:] = old_sys_argv
                    except Exception: pass
                g["_korpusuj_137j_frequency_state"] = old_state
        main_with_frequency_full_context_137j._korpusuj_137j_main_wrapped = True
        g["main"] = main_with_frequency_full_context_137j

    g["_korpusuj_137j_cli_frequency_full_context_installed"] = True

try:
    _install_cli_frequency_full_context_before_paging_137j()
except Exception:
    pass
# END KORPUSUJ_PATCH_137J_CLI_FREQUENCY_FULL_CONTEXT_BEFORE_PAGING


# KORPUSUJ_PATCH_137L_CLI_FREQUENCY_TAG_PARAM_AUTOQUOTE
# CLI-only ergonomics fix: quote bare numeric frequency tag params stripped by shell,
# e.g. <frequency_orth top=10> -> <frequency_orth top="10">.
# Applies to --query and --query-file. It complements 137i, which quotes token
# condition values inside []/{} but intentionally leaves top/min/max untouched.
def _install_cli_frequency_tag_param_autoquote_137l():
    try:
        import os as _os_137l
        import sys as _sys_137l
        import re as _re_137l
    except Exception:
        return

    g = globals()
    if g.get("_korpusuj_137l_cli_frequency_tag_param_autoquote_installed", False):
        return

    _TAG_RE_137L = _re_137l.compile(r"<\s*(frequency(?:_orth|_base)?)\b([^<>]*)>", _re_137l.IGNORECASE | _re_137l.UNICODE)
    _PARAM_RE_137L = _re_137l.compile(r"\b(top|min|max)\s*=\s*([^\s\"'<>/]+)", _re_137l.IGNORECASE | _re_137l.UNICODE)

    def _enabled_137l():
        try:
            val = str(_os_137l.environ.get("KORPUSUJ_137L_CLI_FREQUENCY_PARAM_AUTOQUOTE", "1")).strip().lower()
            return val not in {"0", "false", "no", "nie", "off"}
        except Exception:
            return True

    def _autoquote_frequency_params_137l(query):
        try:
            s = str(query or "")
        except Exception:
            return query
        changed = False

        def repl_tag(m):
            nonlocal changed
            tag_name = m.group(1)
            attrs = m.group(2) or ""

            def repl_param(pm):
                nonlocal changed
                key = pm.group(1)
                val = pm.group(2)
                # Leave already-quoted values alone; regex excludes them, but keep defensive guard.
                if not val or val[0] in {'"', "'"}:
                    return pm.group(0)
                changed = True
                return f'{key}="{val}"'

            new_attrs = _PARAM_RE_137L.sub(repl_param, attrs)
            return "<" + tag_name + new_attrs + ">"

        out = _TAG_RE_137L.sub(repl_tag, s)
        return out if changed else s

    def _find_query_arg_index_137l(argv):
        try:
            for idx, arg in enumerate(argv):
                if arg == "--query" and idx + 1 < len(argv):
                    return idx + 1, False
                if isinstance(arg, str) and arg.startswith("--query="):
                    return idx, True
        except Exception:
            pass
        return None, False

    def _find_query_file_arg_index_137l(argv):
        try:
            for idx, arg in enumerate(argv):
                if arg == "--query-file" and idx + 1 < len(argv):
                    return idx + 1, False
                if isinstance(arg, str) and arg.startswith("--query-file="):
                    return idx, True
        except Exception:
            pass
        return None, False

    def _apply_query_arg_autoquote_137l(argv):
        idx, eq_style = _find_query_arg_index_137l(argv)
        if idx is None:
            return None
        try:
            raw_arg = argv[idx]
            if eq_style:
                original = str(raw_arg).split("=", 1)[1]
                effective = _autoquote_frequency_params_137l(original)
                if effective != original:
                    argv[idx] = "--query=" + effective
            else:
                original = str(raw_arg)
                effective = _autoquote_frequency_params_137l(original)
                if effective != original:
                    argv[idx] = effective
            if effective != original:
                return {"source": "query", "original": original, "effective": effective}
        except Exception:
            return None
        return None

    def _apply_query_file_autoquote_137l(argv):
        idx, eq_style = _find_query_file_arg_index_137l(argv)
        if idx is None:
            return None
        try:
            path = str(argv[idx]).split("=", 1)[1] if eq_style else str(argv[idx])
            if not path:
                return None
            original = open(path, "r", encoding="utf-8-sig", errors="replace").read()
            effective = _autoquote_frequency_params_137l(original)
            if effective == original:
                return None
            tmp_path = path + ".137l.frequency_params.cql"
            with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                f.write(effective)
            if eq_style:
                argv[idx] = "--query-file=" + tmp_path
            else:
                argv[idx] = tmp_path
            return {"source": "query-file", "original": original.strip(), "effective": effective.strip(), "temp_query_file": tmp_path}
        except Exception:
            return None

    def _apply_argv_autoquote_137l(argv=None):
        if not _enabled_137l():
            return None
        if argv is None:
            argv = _sys_137l.argv
        # Prefer direct --query when present; otherwise patch query-file via temp file.
        res = _apply_query_arg_autoquote_137l(argv)
        if res is None:
            res = _apply_query_file_autoquote_137l(argv)
        if res is not None:
            g["_korpusuj_137l_last_query_original"] = res.get("original")
            g["_korpusuj_137l_last_query_effective"] = res.get("effective")
            g["_korpusuj_137l_last_query_source"] = res.get("source")
            if res.get("temp_query_file"):
                g["_korpusuj_137l_last_temp_query_file"] = res.get("temp_query_file")
        return res

    orig_main = g.get("main")
    if callable(orig_main) and not getattr(orig_main, "_korpusuj_137l_frequency_param_autoquote_wrapped", False):
        def main_with_frequency_param_autoquote_137l(argv=None):
            try:
                if argv is None:
                    _apply_argv_autoquote_137l(_sys_137l.argv)
                else:
                    local = list(argv)
                    fake = ["<argv>"] + local
                    result = _apply_argv_autoquote_137l(fake)
                    if result is not None:
                        argv = fake[1:]
            except Exception:
                pass
            return orig_main(argv)
        main_with_frequency_param_autoquote_137l._korpusuj_137l_frequency_param_autoquote_wrapped = True
        g["main"] = main_with_frequency_param_autoquote_137l

    orig_bundle_to_dict = g.get("_bundle_to_dict")
    if callable(orig_bundle_to_dict) and not getattr(orig_bundle_to_dict, "_korpusuj_137l_metadata_wrapped", False):
        def _bundle_to_dict_137l(bundle, *args, **kwargs):
            data = orig_bundle_to_dict(bundle, *args, **kwargs)
            try:
                original = g.get("_korpusuj_137l_last_query_original")
                effective = g.get("_korpusuj_137l_last_query_effective")
                if original is not None and effective is not None and original != effective and isinstance(data, dict):
                    meta = data.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                        data["metadata"] = meta
                    meta["frequency_tag_params_autoquoted_137l"] = True
                    meta["frequency_tag_params_query_source_137l"] = g.get("_korpusuj_137l_last_query_source")
                    meta["frequency_tag_params_query_original_137l"] = original
                    meta["frequency_tag_params_query_effective_137l"] = effective
                    tmp = g.get("_korpusuj_137l_last_temp_query_file")
                    if tmp:
                        meta["frequency_tag_params_temp_query_file_137l"] = tmp
            except Exception:
                pass
            return data
        _bundle_to_dict_137l._korpusuj_137l_metadata_wrapped = True
        g["_bundle_to_dict"] = _bundle_to_dict_137l

    g["_korpusuj_137l_cli_frequency_tag_param_autoquote_installed"] = True

try:
    _install_cli_frequency_tag_param_autoquote_137l()
except Exception:
    pass
# END KORPUSUJ_PATCH_137L_CLI_FREQUENCY_TAG_PARAM_AUTOQUOTE


# KORPUSUJ_PATCH_137N_CLI_DIAGNOSTIC_FILE_LOGGING
# CLI-only diagnostic JSONL file logging. Enabled only with --diagnostics-logs.
# Keeps stdout clean/parseable and injects diagnostics_log_file into JSON metadata.
def _install_cli_diagnostic_file_logging_137n():
    try:
        import os as _os_137n
        import sys as _sys_137n
        import json as _json_137n
        import time as _time_137n
        import datetime as _datetime_137n
        import traceback as _traceback_137n
        from pathlib import Path as _Path_137n
    except Exception:
        return

    g = globals()
    if g.get("_korpusuj_137n_cli_diagnostic_file_logging_installed", False):
        return

    def _truthy_137n(value):
        try:
            if value is True:
                return True
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "tak", "on", "debug", "verbose"}
            return bool(value)
        except Exception:
            return False

    def _arg_value_137n(argv, name, default=None):
        try:
            prefix = name + "="
            for i, a in enumerate(argv):
                if a == name and i + 1 < len(argv):
                    return argv[i + 1]
                if isinstance(a, str) and a.startswith(prefix):
                    return a.split("=", 1)[1]
        except Exception:
            pass
        return default

    def _has_flag_137n(argv, name):
        try:
            return name in argv or any(isinstance(a, str) and a.startswith(name + "=") for a in argv)
        except Exception:
            return False

    def _safe_slug_137n(value, default="corpus"):
        try:
            s = str(value or default)
            out = []
            for ch in s:
                if ch.isalnum() or ch in {"-", "_"}:
                    out.append(ch)
                else:
                    out.append("_")
            slug = "".join(out).strip("_")
            return slug[:80] or default
        except Exception:
            return default

    def _read_query_preview_137n(argv):
        q = _arg_value_137n(argv, "--query", None)
        if q is not None:
            return {"query_source": "query", "query_preview": str(q)[:1000]}
        qf = _arg_value_137n(argv, "--query-file", None)
        if qf:
            try:
                text = _Path_137n(qf).read_text(encoding="utf-8-sig", errors="replace")
                return {"query_source": "query-file", "query_file": str(qf), "query_preview": text.strip()[:1000]}
            except Exception as exc:
                return {"query_source": "query-file", "query_file": str(qf), "query_preview_error": repr(exc)}
        return {"query_source": None, "query_preview": None}

    def _project_root_137n():
        try:
            return _Path_137n(__file__).resolve().parents[2]
        except Exception:
            return _Path_137n.cwd()

    def _make_log_path_137n(argv):
        try:
            corpus_path = _arg_value_137n(argv, "--corpus-path", None)
            corpus_name = _arg_value_137n(argv, "--corpus-name", None)
            if not corpus_name and corpus_path:
                corpus_name = _Path_137n(corpus_path).stem
            slug = _safe_slug_137n(corpus_name or "corpus")
            stamp = _datetime_137n.datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = _os_137n.getpid()
            root = _Path_137n(_os_137n.environ.get("KORPUSUJ_CLI_DIAGNOSTICS_LOG_DIR", "logs/search_cli"))
            if not root.is_absolute():
                root = _project_root_137n() / root
            root.mkdir(parents=True, exist_ok=True)
            return root / f"search_cli_{stamp}_{slug}_{pid}.jsonl"
        except Exception:
            try:
                root = _Path_137n("logs/search_cli")
                root.mkdir(parents=True, exist_ok=True)
                return root / f"search_cli_{int(_time_137n.time())}_{_os_137n.getpid()}.jsonl"
            except Exception:
                return None

    def _jsonable_137n(value, limit=2000):
        try:
            if value is None or isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) > limit:
                    return value[:limit] + "…"
                return value
            if isinstance(value, dict):
                return {str(k): _jsonable_137n(v, limit=limit) for k, v in list(value.items())[:80]}
            if isinstance(value, (list, tuple)):
                return [_jsonable_137n(v, limit=limit) for v in list(value)[:80]]
            return repr(value)[:limit]
        except Exception:
            return None

    def _emit_137n(event, **payload):
        state = g.get("_korpusuj_137n_cli_diag_state") or {}
        path = state.get("log_file")
        if not path:
            return
        try:
            row = {
                "ts": _datetime_137n.datetime.now().isoformat(timespec="milliseconds"),
                "event": event,
                "pid": _os_137n.getpid(),
                "stage": "137n",
            }
            row.update({str(k): _jsonable_137n(v) for k, v in payload.items()})
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write(_json_137n.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        except Exception:
            return

    def _activate_logging_137n(argv):
        try:
            if not _has_flag_137n(argv, "--diagnostics-logs"):
                return None
            log_path = _make_log_path_137n(argv)
            if log_path is None:
                return None
            qinfo = _read_query_preview_137n(argv)
            state = {
                "active": True,
                "log_file": str(log_path),
                "started_perf": _time_137n.perf_counter(),
                "started_wall": _time_137n.time(),
                "argv": list(argv),
                "corpus_path": _arg_value_137n(argv, "--corpus-path", None),
                "corpus_name": _arg_value_137n(argv, "--corpus-name", None),
                "limit": _arg_value_137n(argv, "--limit", None),
                "offset": _arg_value_137n(argv, "--offset", None),
                "format": _arg_value_137n(argv, "--format", None),
                "output": _arg_value_137n(argv, "--output", None),
            }
            state.update(qinfo)
            g["_korpusuj_137n_cli_diag_state"] = state
            _emit_137n(
                "cli.search.start",
                argv=list(argv),
                corpus_path=state.get("corpus_path"),
                corpus_name=state.get("corpus_name"),
                query_source=state.get("query_source"),
                query_file=state.get("query_file"),
                query_preview=state.get("query_preview"),
                limit=state.get("limit"),
                offset=state.get("offset"),
                format=state.get("format"),
                output=state.get("output"),
            )
            return state
        except Exception:
            return None

    orig_run = g.get("run_search_service")
    if callable(orig_run) and not getattr(orig_run, "_korpusuj_137n_diag_file_wrapped", False):
        def run_search_service_137n(request, context, *args, **kwargs):
            state = g.get("_korpusuj_137n_cli_diag_state") or {}
            if state.get("active"):
                try:
                    _emit_137n(
                        "cli.search.service.start",
                        query=getattr(request, "query", None),
                        corpus_name=getattr(request, "corpus_name", None),
                        limit=getattr(request, "limit", None),
                        offset=getattr(request, "offset", None),
                        left_context=getattr(request, "left_context", None),
                        right_context=getattr(request, "right_context", None),
                        options=getattr(request, "options", None),
                        context_metadata=getattr(context, "metadata", None),
                    )
                except Exception:
                    pass
            try:
                bundle = orig_run(request, context, *args, **kwargs)
            except Exception as exc:
                if state.get("active"):
                    _emit_137n(
                        "cli.search.error",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=_traceback_137n.format_exc(limit=20),
                    )
                raise
            if state.get("active"):
                try:
                    results = getattr(bundle, "results", []) or []
                    try:
                        returned = len(results)
                    except Exception:
                        returned = None
                    elapsed_ms = int((_time_137n.perf_counter() - float(state.get("started_perf", _time_137n.perf_counter()))) * 1000)
                    _emit_137n(
                        "cli.search.service.done",
                        total_hits=getattr(bundle, "total_hits", None),
                        returned_hits=returned,
                        limit=getattr(bundle, "limit", None),
                        offset=getattr(bundle, "offset", None),
                        has_more=getattr(bundle, "has_more", None),
                        warnings=getattr(bundle, "warnings", None),
                        messages=getattr(bundle, "messages", None),
                        metadata=getattr(bundle, "metadata", None),
                        elapsed_ms=elapsed_ms,
                    )
                except Exception:
                    pass
            return bundle
        run_search_service_137n._korpusuj_137n_diag_file_wrapped = True
        g["run_search_service"] = run_search_service_137n

    orig_bundle_to_dict = g.get("_bundle_to_dict")
    if callable(orig_bundle_to_dict) and not getattr(orig_bundle_to_dict, "_korpusuj_137n_diag_metadata_wrapped", False):
        def _bundle_to_dict_137n(bundle, *args, **kwargs):
            data = orig_bundle_to_dict(bundle, *args, **kwargs)
            try:
                state = g.get("_korpusuj_137n_cli_diag_state") or {}
                log_file = state.get("log_file")
                if log_file and isinstance(data, dict):
                    meta = data.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                        data["metadata"] = meta
                    meta["diagnostics_log_file"] = log_file
                    meta["diagnostics_log_format"] = "jsonl"
                    meta["diagnostics_log_enabled_137n"] = True
            except Exception:
                pass
            return data
        _bundle_to_dict_137n._korpusuj_137n_diag_metadata_wrapped = True
        g["_bundle_to_dict"] = _bundle_to_dict_137n

    orig_main = g.get("main")
    if callable(orig_main) and not getattr(orig_main, "_korpusuj_137n_diag_file_main_wrapped", False):
        def main_with_diagnostic_file_logging_137n(argv=None):
            old_state = g.get("_korpusuj_137n_cli_diag_state")
            state = None
            try:
                effective_argv = list(_sys_137n.argv[1:] if argv is None else argv)
                state = _activate_logging_137n(effective_argv)
                rc = orig_main(argv)
                if state and state.get("active"):
                    elapsed_ms = int((_time_137n.perf_counter() - float(state.get("started_perf", _time_137n.perf_counter()))) * 1000)
                    _emit_137n("cli.search.done", returncode=rc, elapsed_ms=elapsed_ms)
                return rc
            except Exception as exc:
                if state and state.get("active"):
                    elapsed_ms = int((_time_137n.perf_counter() - float(state.get("started_perf", _time_137n.perf_counter()))) * 1000)
                    _emit_137n(
                        "cli.search.error",
                        returncode=2,
                        elapsed_ms=elapsed_ms,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=_traceback_137n.format_exc(limit=25),
                    )
                raise
            finally:
                # Keep state available until output serialization has happened inside orig_main,
                # then restore to avoid leaking state across in-process calls.
                g["_korpusuj_137n_cli_diag_state"] = old_state
        main_with_diagnostic_file_logging_137n._korpusuj_137n_diag_file_main_wrapped = True
        g["main"] = main_with_diagnostic_file_logging_137n

    g["_korpusuj_137n_cli_diagnostic_file_logging_installed"] = True

try:
    _install_cli_diagnostic_file_logging_137n()
except Exception:
    pass
# END KORPUSUJ_PATCH_137N_CLI_DIAGNOSTIC_FILE_LOGGING





# --- CLI_SENTENCE_OPERATOR_AUTOQUOTE_167G ---
def _install_cli_sentence_operator_autoquote_167g():
    try:
        import os as _os
        import sys as _sys
        import re as _re
    except Exception:
        return
    g = globals()
    if g.get('_cli_sentence_operator_autoquote_167g_installed', False):
        return
    text_keys = {'orth','window_base','window_orth','base','pos','upos','ner','head','coref','dependent','deprel','number','gender','degree','case','person','aspect','autor','tytuł','tytul','children.group'}
    key_pattern = '|'.join(_re.escape(k) for k in sorted(text_keys, key=len, reverse=True))
    pattern = _re.compile('\\b(' + key_pattern + ')\\s*(!=|=)\\s*([^\\s\\]\\}\\)&|<>\\"\']+)')
    def enabled():
        try:
            return str(_os.environ.get('KORPUSUJ_167G_CLI_S_AUTOQUOTE', '1')).strip().lower() not in {'0','false','no','nie','off'}
        except Exception:
            return True
    def quote_bare_values(query):
        s = str(query or '')
        if '<s' not in s:
            return s
        def repl(match):
            key, op, value = match.group(1), match.group(2), match.group(3)
            if not value or value[0] in {'"', "'", '[', '{', '<'}:
                return match.group(0)
            return f'{key}{op}"{value}"'
        return pattern.sub(repl, s)
    def find_query_arg(argv):
        for idx, arg in enumerate(argv or []):
            if arg == '--query' and idx + 1 < len(argv):
                return idx + 1
            if isinstance(arg, str) and arg.startswith('--query='):
                return idx
        return None
    def apply(argv):
        if not enabled():
            return None
        idx = find_query_arg(argv)
        if idx is None:
            return None
        raw = argv[idx]
        if isinstance(raw, str) and raw.startswith('--query='):
            original = raw.split('=', 1)[1]
            effective = quote_bare_values(original)
            if effective != original:
                argv[idx] = '--query=' + effective
        else:
            original = str(raw)
            effective = quote_bare_values(original)
            if effective != original:
                argv[idx] = effective
        if effective != original:
            g['_cli_sentence_operator_autoquote_167g_original'] = original
            g['_cli_sentence_operator_autoquote_167g_effective'] = effective
            return {'original': original, 'effective': effective}
        return None
    original_main = g.get('main')
    if callable(original_main) and not getattr(original_main, '_cli_sentence_operator_autoquote_167g_wrapped', False):
        def main_with_sentence_operator_autoquote_167g(argv=None):
            try:
                if argv is None:
                    apply(_sys.argv)
                else:
                    fake = ['<argv>'] + list(argv)
                    if apply(fake):
                        argv = fake[1:]
            except Exception:
                pass
            return original_main(argv)
        main_with_sentence_operator_autoquote_167g._cli_sentence_operator_autoquote_167g_wrapped = True
        g['main'] = main_with_sentence_operator_autoquote_167g
    g['_cli_sentence_operator_autoquote_167g_installed'] = True
try:
    _install_cli_sentence_operator_autoquote_167g()
except Exception:
    pass
# --- END CLI_SENTENCE_OPERATOR_AUTOQUOTE_167G ---



if __name__ == "__main__":
    raise SystemExit(main())
