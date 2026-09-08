from types import SimpleNamespace

from korpusuj.search.cursor_runtime import configure_search_cursor_runtime, get_search_cursor_runtime

def test_runtime_exposes_dependency_cache_limit():
    cache = {}
    configure_search_cursor_runtime(
        dependency_cache_corpus_name_from_path=lambda path: "c",
        get_dependency_cache_ram_mode=lambda: "all",
        dependency_ram_cache_size_for_corpus=lambda name=None: len(cache),
        put_dependency_ram_cache=lambda *args: None,
        preload_dependency_maps_for_candidates=lambda *args, **kwargs: 0,
        dependency_maps_cache=cache,
        dependency_maps_cache_maxsize=7,
    )
    assert get_search_cursor_runtime().dependency_maps_cache is cache
    assert get_search_cursor_runtime().dependency_maps_cache_maxsize == 7
