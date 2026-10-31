# -*- coding: utf-8 -*-
"""Regression tests for tag-kind selection and hyphenated acronym repair."""
from __future__ import annotations

from unittest.mock import patch

from korpusuj.corpus import lemma_repair_tokenwise as m


def analysis(lemma, tag, upos, orth="x"):
    return {"orth": orth, "lemma": lemma, "tag": tag, "upos": upos}


def run_unique(analyses, orth, lemma, upos, morph):
    with patch.object(m, "analyses_for_form", return_value=("ok", analyses)):
        return m.unique_full_xpos_target(object(), orth, lemma, upos, morph)


def test_tag_kind_preference():
    target, evidence = run_unique(
        [
            analysis("uzbroić", "ger:sg:nom.acc:n:perf:aff", "NOUN"),
            analysis("uzbrojenie", "subst:sg:nom.acc.voc:n:ncol", "NOUN"),
        ],
        "uzbrojenie", "uzbrojyć", "NOUN", "subst:sg:nom:n:ncol",
    )
    assert target == "uzbrojenie", (target, evidence)
    assert evidence["kind_preference_applied"] is True

    target, evidence = run_unique(
        [
            analysis("uzbroić", "ger:sg:nom.acc:n:perf:aff", "NOUN"),
            analysis("uzbrojenie", "subst:sg:nom.acc.voc:n:ncol", "NOUN"),
        ],
        "uzbrojenie", "uzbrojyć", "NOUN", "ger:sg:nom:n:perf:aff",
    )
    assert target == "uzbroić", (target, evidence)

    target, evidence = run_unique(
        [
            analysis("walczyć", "pact:pl:dat:m1:imperf:aff", "ADJ"),
            analysis("walczący", "adj:pl:dat:m1:pos", "ADJ"),
        ],
        "walczącym", "walczeć", "ADJ", "pact:pl:dat:m1:imperf:aff",
    )
    assert target == "walczyć", (target, evidence)


def test_bad_number_stays_blocked():
    target, evidence = run_unique(
        [analysis("walczyć", "pact:pl:dat:m1:imperf:aff", "ADJ")],
        "walczącym", "walczeć", "ADJ", "pact:sg:dat:m1:imperf:aff",
    )
    assert target is None, (target, evidence)


def test_current_correct_lemma_stays():
    target, evidence = run_unique(
        [
            analysis("walczyć", "pact:pl:dat:m1:imperf:aff", "ADJ"),
            analysis("walczący", "adj:pl:dat:m1:pos", "ADJ"),
        ],
        "walczącym", "walczyć", "ADJ", "pact:pl:dat:m1:imperf:aff",
    )
    assert target is None, (target, evidence)
    assert evidence["current_confirmed"] is True


class AcronymEngine:
    def analyse(self, orth):
        assert orth == "GROM-u"
        return [
            (0, 1, ("GROM", "GROM", "brev:npun")),
            (1, 2, ("-", "-", "interp")),
            (2, 3, ("u", "u", "ign")),
        ]


def test_hyphenated_acronym_inflection():
    target, evidence = m._hyphenated_acronym_inflection_target(
        AcronymEngine(), "GROM-u", "subst:sg:gen:m3",
    )
    assert target == "GROM", (target, evidence)
    assert evidence["split_confirmed"] is True
    assert evidence["repair_kind"] == "acronym_inflection"


def test_hyphenated_acronym_shape_rejections():
    target, evidence = m._hyphenated_acronym_inflection_target(
        AcronymEngine(), "e-mail", "subst:sg:gen:m3",
    )
    assert target is None
    assert evidence["status"] == "shape_mismatch"

    target, evidence = m._hyphenated_acronym_inflection_target(
        AcronymEngine(), "A1-u", "subst:sg:gen:m3",
    )
    assert target is None
    assert evidence["status"] == "insufficient_uppercase"


def test_hyphenated_acronym_rejects_incompatible_xpos():
    target, evidence = m._hyphenated_acronym_inflection_target(
        AcronymEngine(), "GROM-u", "subst:pl:nom:m3",
    )
    assert target is None
    assert evidence["status"] == "suffix_xpos_mismatch"
