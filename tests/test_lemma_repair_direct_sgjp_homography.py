# -*- coding: utf-8 -*-
from korpusuj.corpus.lemma_repair_direct_sgjp import _direct_rule


def analysis(lemma, upos, tag):
    return {"orth": "mam", "lemma": lemma, "upos": upos, "tag": tag}


def test_current_lemma_confirmed_in_other_upos_blocks_repair():
    analyses = [
        analysis("mieć", "VERB", "fin:sg:pri:imperf"),
        analysis("mama", "NOUN", "subst:pl:gen:f"),
    ]
    rule = _direct_rule(
        "mam", "mieć", "NOUN", 5, 3,
        {"subst:pl:gen:f": 5}, analyses,
    )
    assert rule is None


def test_cross_upos_homography_blocks_unknown_source():
    analyses = [
        analysis("mieć", "VERB", "fin:sg:pri:imperf"),
        analysis("mama", "NOUN", "subst:pl:gen:f"),
    ]
    rule = _direct_rule(
        "mam", "sztuczny", "NOUN", 5, 3,
        {"subst:pl:gen:f": 5}, analyses,
    )
    assert rule is None


def test_unambiguous_psycholog_repair_still_passes():
    analyses = [
        {"orth": "psychologa", "lemma": "psycholog", "upos": "NOUN", "tag": "subst:sg:gen:m1"},
    ]
    rule = _direct_rule(
        "psychologa", "psycholoeg", "NOUN", 73, 44,
        {"subst:sg:gen:m1": 73}, analyses,
    )
    assert rule is not None
    assert rule["replacement"] == "psycholog"
    assert rule["classification"] == "SAFE_DIRECT_SGJP_REPAIR"


def test_same_upos_multiple_targets_blocks_repair():
    analyses = [
        {"orth": "formie", "lemma": "cel_a", "upos": "NOUN", "tag": "subst:sg:loc:f"},
        {"orth": "formie", "lemma": "cel_b", "upos": "NOUN", "tag": "subst:sg:loc:f"},
    ]
    rule = _direct_rule(
        "formie", "sztuczny", "NOUN", 5, 3,
        {"subst:sg:loc:f": 5}, analyses,
    )
    assert rule is None
