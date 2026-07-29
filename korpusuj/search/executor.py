# -*- coding: utf-8 -*-
"""Plan and execute corpus searches through the shared cursor and index interfaces."""
from __future__ import annotations

import time

from korpusuj.search.diagnostics import search_diag_log, summarize_search_plan_for_log
from korpusuj.search.errors import SearchExecutionError
from korpusuj.search.planner import SearchPlanner


_SearchCursor = None
_SearchIndex = None


def configure_search_executor(*, search_cursor_cls=None, search_index_cls=None) -> None:
    global _SearchCursor, _SearchIndex
    if search_cursor_cls is not None:
        _SearchCursor = search_cursor_cls
    if search_index_cls is not None:
        _SearchIndex = search_index_cls


def _make_cursor(*args, **kwargs):
    if _SearchCursor is None:
        raise RuntimeError("SearchCursor dependency not configured for korpusuj.search.executor")
    return _SearchCursor(*args, **kwargs)


def _make_index(*args, **kwargs):
    if _SearchIndex is None:
        raise RuntimeError("SearchIndex dependency not configured for korpusuj.search.executor")
    return _SearchIndex(*args, **kwargs)


class SearchExecutor:
    """Execute planned queries against a configured SearchCursor implementation."""
    def __init__(self, index, corpus_path=None):
        self.index = index
        self.corpus_path = str(corpus_path or "")
        self.planner = SearchPlanner()

    def execute(self, query, left_context_size=10, right_context_size=10):
        """Plan and execute one query with the requested context sizes."""
        t0 = time.perf_counter()
        try:
            indexed_attrs = self.index.meta().get("indexed_attrs", "")
            total_docs = self.index.total_docs
            total_tokens = self.index.total_tokens
        except Exception:
            indexed_attrs = ""
            total_docs = 0
            total_tokens = 0

        search_diag_log(
            "SQLITE_EXEC_START query=%r indexed_attrs=%r total_docs=%s total_tokens=%s",
            query, indexed_attrs, total_docs, total_tokens
        )

        t_plan0 = time.perf_counter()
        plan = self.planner.plan(query, self.index)
        t_plan1 = time.perf_counter()

        search_diag_log(
            "SQLITE_PLAN supported=%s reason=%r plan_time=%.6fs query=%r plan_summary=%r",
            plan.get("supported"), plan.get("reason"), t_plan1 - t_plan0, query, summarize_search_plan_for_log(plan)
        )

        if not plan.get("supported"):
            raise SearchExecutionError(plan.get("reason", "zapytanie nieobsługiwane przez indeks"))

        t_cursor0 = time.perf_counter()
        cursor = _make_cursor(self.index, plan, left_context_size, right_context_size, corpus_path=self.corpus_path)
        t_cursor1 = time.perf_counter()

        search_diag_log(
            "SQLITE_CURSOR_CREATED cursor_time=%.6fs total_prepare_time=%.6fs query=%r cursor_type=%s",
            t_cursor1 - t_cursor0, t_cursor1 - t0, query, type(cursor).__name__
        )
        return cursor


class CorpusSearchExecutor:
    """Bind shared search execution to a loaded corpus and index."""
    def __init__(self, lazy_corpus):
        self.lazy_corpus = lazy_corpus
        self.index = _make_index(lazy_corpus.search_path)
        self.executor = SearchExecutor(self.index, corpus_path=getattr(lazy_corpus, "parquet_path", None))

    def search(self, query, left_context_size=10, right_context_size=10):
        """Execute one query against the bound corpus and index."""
        search_diag_log(
            "CORPUS_EXEC corpus_search_path=%r parquet_path=%r query=%r",
            getattr(self.lazy_corpus, "search_path", None),
            getattr(self.lazy_corpus, "parquet_path", None),
            query
        )
        return self.executor.execute(query, left_context_size, right_context_size)
# Route union plans to UnionSearchCursor while preserving the
# existing _make_cursor implementation for all ordinary plans.
def _install_executor_or_union_lazy_contract():
    original_make_cursor = globals().get("_make_cursor")
    if not callable(original_make_cursor):
        return
    if getattr(original_make_cursor, "_or_union_lazy_contract_wrapped", False):
        return

    def _make_cursor_or_union(index, plan, left_context_size=10, right_context_size=10, corpus_path=None):
        try:
            if isinstance(plan, dict) and str(plan.get("type") or "") == "union":
                try:
                    from korpusuj.search.cursor import UnionSearchCursor
                except Exception:
                    from .cursor import UnionSearchCursor
                return UnionSearchCursor(index, plan, left_context_size, right_context_size, corpus_path=corpus_path)
        except Exception:
            pass
        return original_make_cursor(index, plan, left_context_size, right_context_size, corpus_path=corpus_path)

    _make_cursor_or_union._or_union_lazy_contract_wrapped = True
    _make_cursor_or_union._or_union_lazy_contract_original = original_make_cursor
    globals()["_make_cursor"] = _make_cursor_or_union

try:
    _install_executor_or_union_lazy_contract()
except Exception:
    pass
