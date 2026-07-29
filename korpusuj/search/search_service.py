# -*- coding: utf-8 -*-
"""Stable service-layer entry points for GUI-independent corpus search.

The module exposes the shared headless search contracts without importing the desktop interface.
"""
from __future__ import annotations

from typing import Any

from korpusuj.search.headless import (
    HeadlessSearchNotConfiguredError,
    SearchBackendContext,
    SearchHit,
    SearchMessage,
    SearchRequest,
    SearchResultBundle,
    normalize_search_result_to_hit,
    normalize_search_results_to_hits,
    run_search_headless,
    validate_search_request,
)
from korpusuj.search.headless_runner import build_headless_context_from_parquet

__all__ = [
    "HeadlessSearchNotConfiguredError",
    "SearchBackendContext",
    "SearchHit",
    "SearchMessage",
    "SearchRequest",
    "SearchResultBundle",
    "build_search_service_context_from_parquet",
    "normalize_search_result_to_hit",
    "normalize_search_results_to_hits",
    "run_search_headless",
    "run_search_service",
    "validate_search_request",
]


def run_search_service(
    request: SearchRequest,
    context: SearchBackendContext,
) -> SearchResultBundle:
    """Run search through the current headless adapter contract."""
    return run_search_headless(request, context)


def build_search_service_context_from_parquet(*args: Any, **kwargs: Any) -> SearchBackendContext:
    """Build a non-GUI SearchBackendContext for a Parquet corpus and its .search sidecar.
    
    The function exposes the service-layer entry point to the shared headless context builder.
    """
    return build_headless_context_from_parquet(*args, **kwargs)
