# -*- coding: utf-8 -*-
"""Shared helpers for exact hit counting and SearchCursor materialization.

The functions in this module are GUI-independent and preserve cursor-produced
rows, cancellation checks and the distinction between estimates and final hits.
"""
from __future__ import annotations

import logging as _logging
import time as _time
from typing import Any, Callable, Optional


# KORPUSUJ_PATCH_145C5C_RESULT_MATERIALIZATION_VERBOSE_GATE_IMPORT
try:
    from korpusuj.search.diagnostics import korpusuj_verbose_diagnostics_enabled_145c1
except Exception:
    def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
        return False
# END KORPUSUJ_PATCH_145C5C_RESULT_MATERIALIZATION_VERBOSE_GATE_IMPORT

__all__ = [
    "count_searchcursor_hits_036l4g48e",
    "materialize_searchcursor_results_036l4g48e",
]


def _logger_info_036l4g48e(logger: Any, *args: Any, **kwargs: Any) -> None:
    """Best-effort logger.info wrapper preserving legacy swallow-on-error behavior."""
    try:
        info = getattr(logger, "info", None)
        if callable(info):
            info(*args, **kwargs)
    except Exception:
        pass


def count_searchcursor_hits_036l4g48e(
    results: Any,
    *,
    search_token: Any = None,
    logger: Any = None,
    perf_counter: Optional[Callable[[], float]] = None,
) -> dict[str, Any]:
    """Count every final SearchCursor hit exactly once."""
    logger = logger if logger is not None else _logging
    perf_counter = perf_counter if perf_counter is not None else _time.perf_counter
    started = perf_counter()
    total_hits = int(results.count_hits(exact=True))
    finished = perf_counter()
    if korpusuj_verbose_diagnostics_enabled_145c1():
        _logger_info_036l4g48e(
            logger,
            "[DIAG perf.search.count] token=%s stage=count_hits_exact "
            "elapsed=%.6fs value=%r",
            search_token,
            finished - started,
            total_hits,
        )
    return {
        "exact_hits": total_hits,
        "total_hits": total_hits,
        "t_count_exact_start_035d": started,
        "t_count_exact_done_035d": finished,
        "t_count_fast_start_035d": finished,
        "t_count_fast_done_035d": finished,
    }


def count_final_searchcursor_hits(
    results: Any,
    *,
    search_token: Any = None,
    logger: Any = None,
    perf_counter: Optional[Callable[[], float]] = None,
) -> dict[str, Any]:
    """Return a JSON-safe final hit count for SearchCursor-like results.

    Public contract: total_hits is a final post-filter result count. The helper
    may use an exact estimate only when the cursor explicitly declares that the
    estimate is exact. Otherwise it forces final counting through the existing
    GUI-free exact-count helper.
    """
    logger = logger if logger is not None else _logging
    perf_counter = perf_counter if perf_counter is not None else _time.perf_counter

    estimate_is_exact = getattr(results, "count_hits_estimate_is_exact", None)
    estimate = getattr(results, "count_hits_estimate", None)
    if callable(estimate_is_exact) and callable(estimate):
        try:
            if bool(estimate_is_exact()):
                started = perf_counter()
                total = int(estimate() or 0)
                finished = perf_counter()
                if korpusuj_verbose_diagnostics_enabled_145c1():
                    _logger_info_036l4g48e(
                        logger,
                        "[DIAG perf.search.count] token=%s stage=final_count_exact_estimate elapsed=%.6fs value=%r",
                        search_token,
                        finished - started,
                        total,
                    )
                return {
                    "total_hits": total,
                    "source": "final_count",
                    "strategy": "exact_estimate",
                    "t_count_start": started,
                    "t_count_done": finished,
                }
        except Exception:
            pass

    payload = count_searchcursor_hits_036l4g48e(
        results,
        search_token=search_token,
        logger=logger,
        perf_counter=perf_counter,
    )
    data: dict[str, Any] = dict(payload or {})
    total = data.get("total_hits", data.get("exact_hits", None))
    if total is None:
        raise ValueError("final count helper did not return total_hits")
    data["total_hits"] = int(total)
    data["source"] = "final_count"
    data["strategy"] = "exact_materialization"
    return data


def materialize_searchcursor_results_036l4g48e(
    results: Any,
    *,
    cancel_check: Optional[Callable[[], Any]] = None,
    search_token: Any = None,
    logger: Any = None,
    perf_counter: Optional[Callable[[], float]] = None,
) -> dict[str, Any]:
    """Materialize SearchCursor results in the dictionary shape consumed by GUI and export callers.
    
    The function preserves cancellation semantics, fast-materialization detection and the established result keys.
    """
    logger = logger if logger is not None else _logging
    perf_counter = perf_counter if perf_counter is not None else _time.perf_counter

    def _cancelled_036l4g39g() -> bool:
        try:
            return bool(cancel_check()) if cancel_check is not None else False
        except Exception:
            return False

    t_materialize_start_035d = perf_counter()
    fast_materialize_036l4g3 = getattr(results, "materialize_all_grouped_036l4g3", None)
    if callable(fast_materialize_036l4g3):
        materialized_results = fast_materialize_036l4g3(cancel_check=_cancelled_036l4g39g)
        if materialized_results is None:
            _logger_info_036l4g48e(logger, "Materializacja/statystyki przerwane [token=%s]", search_token)
            return {
                "results": None,
                "cancelled": True,
                "t_materialize_start_035d": t_materialize_start_035d,
                "t_materialize_done_035d": perf_counter(),
            }
    else:
        materialized_results = []
        for hit in results:
            if _cancelled_036l4g39g():
                _logger_info_036l4g48e(logger, "Materializacja/statystyki przerwane [token=%s]", search_token)
                return {
                    "results": None,
                    "cancelled": True,
                    "t_materialize_start_035d": t_materialize_start_035d,
                    "t_materialize_done_035d": perf_counter(),
                }
            materialized_results.append(hit)

    t_materialize_done_035d = perf_counter()
    if korpusuj_verbose_diagnostics_enabled_145c1():
        _logger_info_036l4g48e(
            logger,
            "[DIAG perf.search.count] token=%s stage=materialize_cursor elapsed=%.6fs results=%s",
            search_token,
            t_materialize_done_035d - t_materialize_start_035d,
            len(materialized_results),
        )

    return {
        "results": materialized_results,
        "cancelled": False,
        "t_materialize_start_035d": t_materialize_start_035d,
        "t_materialize_done_035d": t_materialize_done_035d,
    }
