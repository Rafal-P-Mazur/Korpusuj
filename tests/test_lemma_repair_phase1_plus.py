# -*- coding: utf-8 -*-
from korpusuj.corpus.lemma_repair_models import LemmaRepairOptions
from korpusuj.corpus.lemma_repair_policy import _safe_gender_conflict

def test_mode_exists():
    assert LemmaRepairOptions(mode="common-core-plus").mode == "common-core-plus"

def test_safe_noun_gender_conflict():
    candidate={"classification":"MORPH_CONFLICT","sgjp_status":"SGJP_UNIQUE_TARGET"}
    rule={"orth":"czołgami","lemma":"czołgo","upos":"NOUN","replacement":"czołg",
          "compatible_lemmas":["czołg"],"morph_comparisons":[
          {"outcome":"MATCH","best_comparison":{"shared":["number","case","gender"],"matches":{"number":"pl","case":"inst","gender":"m3"},"mismatches":{}}},
          {"outcome":"MISMATCH","best_comparison":{"shared":["number","case","gender"],"matches":{"number":"pl","case":"inst"},"mismatches":{"gender":{"stanza":"f","sgjp":"m3"}}}}]}
    assert _safe_gender_conflict(candidate,rule)

def test_case_conflict_is_rejected():
    candidate={"classification":"MORPH_CONFLICT","sgjp_status":"SGJP_UNIQUE_TARGET"}
    rule={"orth":"formie","lemma":"bledny","upos":"NOUN","replacement":"cel",
          "compatible_lemmas":["cel"],"morph_comparisons":[
          {"outcome":"MISMATCH","best_comparison":{"shared":["number","case","gender"],"matches":{"number":"sg","gender":"m3"},"mismatches":{"case":{"stanza":"loc","sgjp":"dat"}}}}]}
    assert not _safe_gender_conflict(candidate,rule)
