import korpusuj.dependency.runtime as runtime

def test_189p2_package_native_dependencies_exist():
    assert runtime.LazyCorpus is not None
    assert runtime.DependencyMapDiskCache is not None
    assert callable(runtime.build_dependency_maps)
    assert runtime.pq is not None
    assert callable(runtime._dependency_cache_path_for_corpus_path)
    assert callable(runtime._cfg_bool)
    assert callable(runtime._as_list_for_warmup)

def test_189p2_helpers_preserve_expected_shapes():
    assert runtime._as_list_for_warmup((1, 2)) == [1, 2]
    assert runtime._as_list_for_warmup(None) == []
