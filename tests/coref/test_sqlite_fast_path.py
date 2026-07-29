# -*- coding: utf-8 -*-
from korpusuj.search import cursor as mod


class FakeConnection:
    def execute(self, sql, params=()):
        if "FROM terms" in sql:
            return [("base", "Polska"), ("orth", "POLSKA"), ("base", "Niemcy")]
        if "SELECT doc_id FROM docs" in sql:
            return [(0,), (1,), (2,)]
        raise AssertionError(sql)


class FakeIndex:
    def __init__(self):
        self.con = FakeConnection()
        self.docs = {
            0: {"tokens": ["Polska", "ona"], "lemmas": ["Polska", "on"],
                "corefs": [["Head-1"], ["Part-1"]]},
            1: {"tokens": ["Niemcy"], "lemmas": ["Niemcy"], "corefs": [["Head-2"]]},
            2: {"tokens": ["POLSKA", "kraj"], "lemmas": ["Polska", "kraj"],
                "corefs": [["Part-3"], ["Head-3"]]},
        }
        self.get_doc_calls = 0
        self.get_corefs_calls = 0

    def get_postings(self, attr, value):
        if str(value).casefold() == "polska":
            return {0: [0], 2: [0]}
        if str(value).casefold() == "niemcy":
            return {1: [0]}
        return {}

    def get_doc_ids_for_term(self, attr, value):
        return set(self.get_postings(attr, value))

    def get_doc(self, doc_id):
        self.get_doc_calls += 1
        return self.docs[int(doc_id)]

    def get_corefs_138i3(self, doc_id):
        self.get_corefs_calls += 1
        return self.docs[int(doc_id)]["corefs"]


class FakeCursor:
    def __init__(self):
        self.index = FakeIndex()
        self._posting_cache_local = {}


def cond(attr="coref", value="Polska", match_type="exact"):
    return {"attr": attr, "values": [value], "op": "=", "match_type": match_type}


def test_173b2_exact_candidate_prefilter_is_base_orth_union():
    cur = FakeCursor()
    assert mod._coref_exact_candidate_doc_ids_173b2(cur, cond()) == {0, 2}


def test_173b2_final_postings_preserve_cluster_and_role_semantics():
    cur = FakeCursor()
    assert mod._coref_condition_postings(cur, cond("coref")) == {0: [0, 1], 2: [0, 1]}
    cur = FakeCursor()
    assert mod._coref_condition_postings(cur, cond("coref(H)")) == {0: [0], 2: [1]}
    cur = FakeCursor()
    assert mod._coref_condition_postings(cur, cond("coref(P)")) == {0: [1], 2: [0]}


def test_173b2_postings_are_cached_and_revalidation_does_not_reread_sqlite():
    cur = FakeCursor()
    first = mod._coref_condition_postings(cur, cond())
    doc_calls = cur.index.get_doc_calls
    coref_calls = cur.index.get_corefs_calls
    second = mod._coref_condition_postings(cur, cond())
    assert second is first
    assert cur.index.get_doc_calls == doc_calls
    assert cur.index.get_corefs_calls == coref_calls
    assert mod._match_coref_condition_at_pos(cur, cond(), 0, 1) is True
    assert cur.index.get_doc_calls == doc_calls
    assert cur.index.get_corefs_calls == coref_calls


def test_173b2_regex_uses_safe_all_docs_fallback():
    cur = FakeCursor()
    assert mod._coref_exact_candidate_doc_ids_173b2(cur, cond(match_type="regex")) is None
