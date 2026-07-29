
"""Focused cache-contract tests for the 173c2 exact coref semantics."""
from __future__ import annotations

from korpusuj.search import cursor as cursor_module


class FakeIndex:
    def __init__(self):
        self.get_doc_calls = 0
        self.get_corefs_calls = 0
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
        self.get_corefs_calls += 1
        return self.doc["corefs"]


class FakeCursor:
    def __init__(self):
        self.index = FakeIndex()


def cond(attr="coref"):
    return {"attr": attr, "op": "=", "values": ["Polska"], "match_type": "exact"}


def test_exact_H_P_share_one_cached_document_index():
    cursor = FakeCursor()
    assert cursor_module._coref_condition_positions(cursor, cond("coref"), 0) == {0, 1}
    calls_after_first = cursor.index.get_doc_calls
    assert cursor_module._coref_condition_positions(cursor, cond("coref(H)"), 0) == {0}
    assert cursor_module._coref_condition_positions(cursor, cond("coref(P)"), 0) == {1}
    assert cursor.index.get_doc_calls == calls_after_first
    assert cursor._coref_doc_index_builds_173b == 1


def test_173b_builder_is_primed_for_later_revalidation():
    cursor = FakeCursor()
    cursor_module._coref_condition_positions(cursor, cond(), 0)
    calls = cursor.index.get_doc_calls
    cursor_module._build_coref_document_index_173b(cursor, 0)
    assert cursor.index.get_doc_calls == calls
    assert cursor._coref_doc_index_builds_173b == 1


def test_exact_semantics_remain_lemma_first_case_sensitive():
    cursor = FakeCursor()
    index = cursor_module._coref_exact_cached_document_index(cursor, 0)
    assert index["values_by_cluster"]["7"] == {"Polska", "kraj"}
    assert "polska" not in index["values_by_cluster"]["7"]


def test_runtime_block_uses_neutral_names():
    text = open(cursor_module.__file__, "r", encoding="utf-8").read()
    block = text.split("# --- COREF_EXACT_CACHE_CONTRACT ---", 1)[1]
    block = block.split("# --- END COREF_EXACT_CACHE_CONTRACT ---", 1)[0]
    assert "_173c2a" not in block
