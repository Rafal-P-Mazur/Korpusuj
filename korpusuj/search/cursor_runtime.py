# -*- coding: utf-8 -*-
"""Configure and expose the runtime services used by SearchCursor instances."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class SearchCursorRuntime:
    dependency_cache_corpus_name_from_path: Callable[[str], str]
    get_dependency_cache_ram_mode: Callable[[], str]
    dependency_ram_cache_size_for_corpus: Callable[[str], int]
    put_dependency_ram_cache: Callable[[str, int, Any], None]
    preload_dependency_maps_for_candidates: Callable[..., int]
    dependency_maps_cache: Any
    candidate_max_docs: int = 20000
    candidate_stream_batch_docs: int = 512
    full_context_size: int = 250


_runtime: Optional[SearchCursorRuntime] = None
_full_context_size_provider: Optional[Callable[[], int]] = None


def configure_search_cursor_runtime(
    *,
    dependency_cache_corpus_name_from_path: Callable[[str], str],
    get_dependency_cache_ram_mode: Callable[[], str],
    dependency_ram_cache_size_for_corpus: Callable[[str], int],
    put_dependency_ram_cache: Callable[[str, int, Any], None],
    preload_dependency_maps_for_candidates: Callable[..., int],
    dependency_maps_cache: Any,
    candidate_max_docs: int = 20000,
    candidate_stream_batch_docs: int = 512,
    full_context_size: int = 250,
) -> None:
    global _runtime
    _runtime = SearchCursorRuntime(
        dependency_cache_corpus_name_from_path=dependency_cache_corpus_name_from_path,
        get_dependency_cache_ram_mode=get_dependency_cache_ram_mode,
        dependency_ram_cache_size_for_corpus=dependency_ram_cache_size_for_corpus,
        put_dependency_ram_cache=put_dependency_ram_cache,
        preload_dependency_maps_for_candidates=preload_dependency_maps_for_candidates,
        dependency_maps_cache=dependency_maps_cache,
        candidate_max_docs=int(candidate_max_docs),
        candidate_stream_batch_docs=int(candidate_stream_batch_docs),
        full_context_size=int(full_context_size),
    )


def get_search_cursor_runtime() -> SearchCursorRuntime:
    if _runtime is None:
        raise RuntimeError("SearchCursor runtime not configured. Call configure_search_cursor_runtime(...) first.")
    return _runtime


def dependency_cache_corpus_name_from_path(path: str) -> str:
    return get_search_cursor_runtime().dependency_cache_corpus_name_from_path(path)


def get_dependency_cache_ram_mode() -> str:
    return get_search_cursor_runtime().get_dependency_cache_ram_mode()


def dependency_ram_cache_size_for_corpus(corpus_name: str) -> int:
    return int(get_search_cursor_runtime().dependency_ram_cache_size_for_corpus(corpus_name))


def put_dependency_ram_cache(corpus_name: str, doc_id: int, dep_maps: Any) -> None:
    return get_search_cursor_runtime().put_dependency_ram_cache(corpus_name, doc_id, dep_maps)


def preload_dependency_maps_for_candidate_docs(*args, **kwargs) -> int:
    return int(get_search_cursor_runtime().preload_dependency_maps_for_candidates(*args, **kwargs))


def get_dependency_maps_cache() -> Any:
    return get_search_cursor_runtime().dependency_maps_cache


def candidate_max_docs() -> int:
    return int(get_search_cursor_runtime().candidate_max_docs)


def candidate_stream_batch_docs() -> int:
    return int(get_search_cursor_runtime().candidate_stream_batch_docs)


def full_context_size() -> int:
    return int(get_search_cursor_runtime().full_context_size)



def configure_full_context_size_provider(provider: Optional[Callable[[], int]]) -> None:
    '''Inject a lazy provider for the extended/full context size.

    This must stay lazy because UI settings can change after module import.
    '''
    global _full_context_size_provider
    _full_context_size_provider = provider


def get_full_context_size() -> int:
    '''Return current extended context size from settings/global state.

    Falls back to the value stored in SearchCursorRuntime and finally to 250.
    '''
    try:
        if _full_context_size_provider is not None:
            return int(_full_context_size_provider() or 250)
    except Exception:
        pass
    try:
        return int(get_search_cursor_runtime().full_context_size)
    except Exception:
        return 250
