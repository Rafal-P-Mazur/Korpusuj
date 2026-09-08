import ast
from pathlib import Path
from korpusuj.dependency.runtime_state import configure_dependency_runtime_state, get_dependency_runtime_state
import korpusuj.dependency.runtime as runtime

def make_state(cache=None):
    configure_dependency_runtime_state(dependency_maps_cache={} if cache is None else cache,
        dependency_disk_caches={}, dependency_warmup_threads={}, dependency_warmup_stop_flags={},
        dependency_warmup_lock=None, maps_cache_maxsize=3, candidate_max_docs=2,
        candidate_stream_batch_docs=1, candidate_ram_budget_mb=1, cache_preload_batch_size=1,
        default_ram_mode="none", default_ram_usage_label="Oszczędny",
        ram_usage_labels={"Oszczędny":"none","Maksymalna wydajność":"all"},
        ram_mode_labels={"none":"Oszczędny","all":"Maksymalna wydajność"})
    return get_dependency_runtime_state()

def bind(state, mode):
    runtime.configure_dependency_runtime_bindings_189p(state=state, config_provider=lambda:{},
        corpus_path_provider=lambda name: None, loaded_corpus_provider=lambda name: None,
        ram_mode_provider=lambda: mode, ram_cache_size_provider=lambda name=None: 0,
        progress_reporter=lambda *a,**k: None, legacy_index_ensurer=lambda *a,**k: None,
        diagnostics_enabled=lambda *a,**k: False, verbose_diagnostics_enabled=lambda *a,**k: False)

def test_no_dictionary_wide_namespace_mutation():
    root=Path(__file__).resolve().parents[2]
    assert "globals().update(" not in (root/"korpusuj/dependency/runtime.py").read_text(encoding="utf-8")
    assert "_dependency_runtime.__dict__.update(globals())" not in (root/"engine.py").read_text(encoding="utf-8")

def test_ram_modes_and_two_corpora_are_isolated():
    cache={}; state=make_state(cache); bind(state,"all")
    assert runtime._put_dependency_ram_cache(("A",1),("a",)) is True
    assert runtime._put_dependency_ram_cache(("B",1),("b",)) is True
    runtime._clear_dependency_ram_cache_for_corpus("A")
    assert ("A",1) not in cache and ("B",1) in cache
    bind(state,"none")
    assert runtime._put_dependency_ram_cache(("B",2),("x",)) is False
    assert not any(key[0]=="B" for key in cache)

def test_limit_comes_from_single_runtime_state():
    cache={}; state=make_state(cache); bind(state,"all")
    for i in range(5): runtime._put_dependency_ram_cache(("A",i),i)
    assert len(cache)==3
