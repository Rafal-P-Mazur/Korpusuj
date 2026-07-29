"""Regression test for 173b cache-hit telemetry under exact lemma semantics."""
from korpusuj.search import cursor as cursor_module


class FakeIndex:
    def __init__(self):
        self.get_doc_calls = 0
        self.total_docs = 1
        self.doc = {
            "tokens": ["Polsce", "kraju"],
            "lemmas": ["Polska", "kraj"],
            "corefs": [["Head-7"], ["Part-7"]],
        }

    def get_doc(self, doc_id):
        self.get_doc_calls += 1
        return self.doc

    def get_corefs_138i3(self, doc_id):
        return self.doc["corefs"]


class FakeCursor:
    def __init__(self):
        self.index = FakeIndex()


def condition(attr):
    return {"attr": attr, "op": "=", "values": ["Polska"], "match_type": "exact"}


def test_exact_cache_hits_increment_173b_telemetry_without_sqlite_reread():
    cursor = FakeCursor()
    assert cursor_module._coref_condition_positions(cursor, condition("coref"), 0) == {0, 1}
    reads = cursor.index.get_doc_calls
    assert cursor_module._coref_condition_positions(cursor, condition("coref(H)"), 0) == {0}
    assert cursor_module._coref_condition_positions(cursor, condition("coref(P)"), 0) == {1}
    assert cursor.index.get_doc_calls == reads
    assert cursor._coref_doc_index_builds_173b == 1
    assert cursor._coref_doc_index_cache_hits_173b >= 2
