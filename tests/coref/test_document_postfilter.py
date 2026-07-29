# -*- coding: utf-8 -*-
from types import SimpleNamespace

from korpusuj.search import cursor as mod


class FakeIndex:
    def __init__(self):
        self.docs = {
            0: {
                "tokens": ["Polska", "ona", "premier", "Polski", "x"],
                "lemmas": ["Polska", "on", "premier", "Polska", "x"],
                "corefs": [
                    ["Head-1"], ["Part-1"], ["Head-2"],
                    ["Part-2", "Part-2"], [],
                ],
            }
        }
        self.con = SimpleNamespace(execute=lambda sql: [(0,)])

    def get_doc(self, doc_id):
        return self.docs[int(doc_id)]

    def get_corefs_138i3(self, doc_id):
        return self.docs[int(doc_id)]["corefs"]


class FakeCursor:
    def __init__(self):
        self.index = FakeIndex()


def cond(attr, value="Polska", match_type="exact"):
    return {"attr": attr, "values": [value], "op": "=", "match_type": match_type}


def test_173b_role_semantics_and_same_cluster_matching():
    cur = FakeCursor()
    assert mod._coref_condition_positions(cur, cond("coref"), 0) == {0, 1, 2, 3}
    assert mod._coref_condition_positions(cur, cond("coref(H)"), 0) == {0, 2}
    assert mod._coref_condition_positions(cur, cond("coref(P)"), 0) == {1, 3}
    assert mod._coref_condition_positions(cur, cond("coref(M)"), 0) == set()


def test_173b_lemma_match_regex_and_no_duplicate_positions():
    cur = FakeCursor()
    assert mod._coref_condition_positions(cur, cond("coref(P)", "Polsk.*", "regex"), 0) == {1, 3}
    assert mod._coref_condition_positions(cur, cond("coref(P)"), 0) == {1, 3}


def test_173b_document_index_is_built_once_per_cursor_and_document():
    cur = FakeCursor()
    mod._coref_condition_positions(cur, cond("coref"), 0)
    mod._coref_condition_positions(cur, cond("coref(H)"), 0)
    mod._coref_condition_positions(cur, cond("coref(P)"), 0)
    assert cur._coref_doc_index_builds_173b == 1
    assert cur._coref_doc_index_cache_hits_173b >= 2


def test_173b_postings_use_optimized_positions():
    cur = FakeCursor()
    assert mod._coref_condition_postings(cur, cond("coref(H)")) == {0: [0, 2]}
