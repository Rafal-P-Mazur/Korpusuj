from korpusuj.search.cursor import (
    LAZY_FULLTEXT_MARKER_111,
    make_lazy_fulltext_ref_111,
    release_materialized_searchcursor_caches_189n2,
)

class FakeIndex:
    index_path = "sample.search"

class FakeLRU:
    def __init__(self):
        self.data = {1: object()}

class FakeCursor:
    def __init__(self):
        self._result_cache = {1: object()}
        self._doc_cache_036l4g7 = {1: object()}
        self._posting_cache_local = {1: object()}
        self._dep_maps_cache = FakeLRU()
        self._hits = [(1, 2, 3)]
        self._hit_iter = iter(())

def test_locator_does_not_own_search_index():
    index = FakeIndex()
    ref = make_lazy_fulltext_ref_111(index, 4, 5, 6)
    assert ref[0] == LAZY_FULLTEXT_MARKER_111
    assert ref[1] == "sample.search"
    assert index not in ref

def test_release_clears_request_only_cursor_state():
    cursor = FakeCursor()
    release_materialized_searchcursor_caches_189n2(cursor)
    assert cursor._result_cache == {}
    assert cursor._doc_cache_036l4g7 == {}
    assert cursor._posting_cache_local == {}
    assert cursor._dep_maps_cache.data == {}
    assert cursor._hits == []
    assert cursor._hit_iter is None
