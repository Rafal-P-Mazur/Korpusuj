# -*- coding: utf-8 -*-
from korpusuj.corpus.lemma_repair_policy import _safe_plain_word, _safe_gender_conflict

def test_plain_unicode_words():
    assert _safe_plain_word("czołgami")
    assert _safe_plain_word("hełmach")
    for value in ("a@b.pl","https://x.pl/a","gazety.pl","fake-newsami","abc_1","abc2"):
        assert not _safe_plain_word(value)

def test_artifacts_cannot_be_promoted():
    candidate={"classification":"MORPH_CONFLICT","sgjp_status":"SGJP_UNIQUE_TARGET"}
    details=[{"outcome":"MISMATCH","best_comparison":{"shared":["number","case","gender"],"matches":{"number":"sg","case":"nom"},"mismatches":{"gender":{"stanza":"f","sgjp":"m3"}}}}]
    for orth in ("kontakt@policja.gov.pl","https://example.pl/x","gazety.pl","fake-newsami","abc_1"):
        rule={"orth":orth,"lemma":"bledny","upos":"NOUN","replacement":"cel","compatible_lemmas":["cel"],"morph_comparisons":details}
        assert not _safe_gender_conflict(candidate,rule)
