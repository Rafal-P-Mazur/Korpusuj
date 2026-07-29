
"""Focused regression tests for lemma-first, case-sensitive exact coref."""
from __future__ import annotations

from korpusuj.search import cursor as cursor_module


class FakeIndex:
    def __init__(self, docs):
        self.docs = docs
        self.total_docs = len(docs)

    def get_doc(self, doc_id):
        return self.docs[int(doc_id)]

    def get_corefs_138i3(self, doc_id):
        return self.docs[int(doc_id)]["corefs"]

    def get_doc_ids_for_term(self, attr, value):
        source_key = "lemmas" if attr == "base" else "tokens"
        return {
            doc_id for doc_id, doc in enumerate(self.docs)
            if any(str(item) == str(value) for item in doc[source_key])
        }


def condition(attr="coref", value="Polska", match_type="exact"):
    return {"attr": attr, "op": "=", "values": [value], "match_type": match_type}


def cursor_for(docs):
    obj = cursor_module.SearchCursor.__new__(cursor_module.SearchCursor)
    obj.index = FakeIndex(docs)
    return obj


def test_exact_matches_inflected_surface_when_lemma_is_Polska():
    tokens = ["Polsce"]
    lemmas = ["Polska"]
    corefs = [["Head-1"]]
    assert cursor_module._coref_exact_matching_cluster_ids(tokens, lemmas, corefs, condition()) == {"1"}


def test_exact_rejects_lowercase_adjective_with_lemma_polski():
    tokens = ["polska"]
    lemmas = ["polski"]
    corefs = [["Head-1"]]
    assert cursor_module._coref_exact_matching_cluster_ids(tokens, lemmas, corefs, condition()) == set()


def test_exact_rejects_sentence_initial_adjective_despite_surface_Polska():
    tokens = ["Polska"]
    lemmas = ["polski"]
    corefs = [["Head-1"]]
    assert cursor_module._coref_exact_matching_cluster_ids(tokens, lemmas, corefs, condition()) == set()


def test_exact_uses_token_only_when_lemma_is_missing():
    tokens = ["Polska"]
    lemmas = [""]
    corefs = [["Head-1"]]
    assert cursor_module._coref_exact_matching_cluster_ids(tokens, lemmas, corefs, condition()) == {"1"}


def test_H_and_P_keep_role_filter_with_same_lemma_contract():
    doc = {
        "tokens": ["Polsce", "kraju"],
        "lemmas": ["Polska", "kraj"],
        "corefs": [["Head-7"], ["Part-7"]],
    }
    cursor = cursor_for([doc])
    assert cursor_module._coref_exact_positions_for_document(cursor, condition("coref(H)"), 0) == {0}
    assert cursor_module._coref_exact_positions_for_document(cursor, condition("coref(P)"), 0) == {1}
    assert cursor_module._coref_exact_positions_for_document(cursor, condition("coref"), 0) == {0, 1}


def test_M_uses_same_matching_clusters_without_changing_span_expansion():
    tokens = ["Była", "ambasador", "Polski", "w", "Rosji"]
    lemmas = ["być", "ambasador", "Polska", "w", "Rosja"]
    corefs = [["Head-214"], ["Part-214"], ["Part-214"], ["Part-214"], ["Part-214"]]
    assert cursor_module._coref_matching_cluster_ids(tokens, lemmas, corefs, condition("coref(M)")) == {"214"}
    assert cursor_module._coref_expand_contiguous_span(corefs, 0, len(corefs)) == 5


def test_literal_candidate_lookup_does_not_generate_case_variants():
    docs = [{"tokens": ["polska"], "lemmas": ["polski"], "corefs": [["Head-1"]]}]
    cursor = cursor_for(docs)
    assert cursor_module._coref_exact_literal_candidate_doc_ids(cursor, condition()) == []


def test_regex_is_delegated_not_redefined_as_exact():
    assert cursor_module._coref_exact_positive_condition(condition(match_type="regex")) is False


def test_runtime_block_uses_neutral_names():
    text = open(cursor_module.__file__, "r", encoding="utf-8").read()
    block = text.split("# --- COREF_EXACT_LEMMA_CASE_SENSITIVE_SEMANTICS ---", 1)[1]
    block = block.split("# --- END COREF_EXACT_LEMMA_CASE_SENSITIVE_SEMANTICS ---", 1)[0]
    assert "_173c2" not in block
