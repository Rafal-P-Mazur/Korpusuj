# -*- coding: utf-8 -*-
from korpusuj.corpus._lemma_repair_d3_engine import (
    compare_tags,
    normalize_lemma,
)
from korpusuj.corpus.lemma_repair_direct_sgjp import _direct_rule


def test_normalize_sgjp_technical_lemma_ids():
    assert normalize_lemma("psycholog:Sm1") == "psycholog"
    assert normalize_lemma("pedagog:Sm1") == "pedagog"
    assert normalize_lemma("hasło:1") == "hasło"
    assert normalize_lemma("zwykły:tekst") == "zwykły:tekst"


def test_case_alternative_gen_acc_matches_gen_and_acc():
    gen = compare_tags("subst:sg:gen:m1", "subst:sg:gen.acc:m1")
    acc = compare_tags("subst:sg:acc:m1", "subst:sg:gen.acc:m1")
    assert not gen["mismatches"]
    assert not acc["mismatches"]


def test_case_alternative_nom_voc_matches_nom_and_voc():
    nom = compare_tags("subst:pl:nom:m1", "subst:pl:nom.voc:m1")
    voc = compare_tags("subst:pl:voc:m1", "subst:pl:nom.voc:m1")
    assert not nom["mismatches"]
    assert not voc["mismatches"]


def test_unrelated_cases_still_conflict():
    result = compare_tags("subst:sg:dat:m1", "subst:sg:gen.acc:m1")
    assert "case" in result["mismatches"]


def test_psycholog_direct_rule_after_normalization():
    analyses = [{
        "orth": "psychologa",
        "lemma": normalize_lemma("psycholog:Sm1"),
        "upos": "NOUN",
        "tag": "subst:sg:gen.acc:m1",
    }]
    rule = _direct_rule(
        "psychologa",
        "psycholoeg",
        "NOUN",
        73,
        44,
        {"subst:sg:gen:m1": 73},
        analyses,
    )
    assert rule is not None
    assert rule["replacement"] == "psycholog"


def test_pedagoga_homography_remains_blocked():
    analyses = [
        {"orth": "pedagoga", "lemma": "pedagoga", "upos": "NOUN", "tag": "subst:sg:nom:m1"},
        {"orth": "pedagoga", "lemma": normalize_lemma("pedagog:Sm1"), "upos": "NOUN", "tag": "subst:sg:gen.acc:m1"},
    ]
    rule = _direct_rule(
        "pedagoga",
        "pedagoeg",
        "NOUN",
        13,
        13,
        {"subst:sg:gen:m1": 13},
        analyses,
    )
    assert rule is None


def test_mam_cross_upos_homography_remains_blocked():
    analyses = [
        {"orth": "mam", "lemma": "mieć", "upos": "VERB", "tag": "fin:sg:pri:imperf"},
        {"orth": "mam", "lemma": "mama", "upos": "NOUN", "tag": "subst:pl:gen:f"},
    ]
    rule = _direct_rule(
        "mam",
        "sztuczny",
        "NOUN",
        5,
        3,
        {"subst:pl:gen:f": 5},
        analyses,
    )
    assert rule is None
