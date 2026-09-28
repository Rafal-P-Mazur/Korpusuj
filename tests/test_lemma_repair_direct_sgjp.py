# -*- coding: utf-8 -*-
from korpusuj.corpus.lemma_repair_direct_sgjp import _direct_rule

def analyses(target, upos="NOUN", tag="subst:pl:dat:m1"):
    return [{"orth":"uczniom","lemma":target,"tag":tag,"upos":upos}]

def test_direct_unique_full_match():
    rule=_direct_rule("uczniom","uczni","NOUN",5,3,{"subst:pl:dat:m1":5},analyses("uczeń"))
    assert rule and rule["replacement"]=="uczeń"
    assert rule["classification"]=="SAFE_DIRECT_SGJP_REPAIR"

def test_current_lemma_confirmed_is_not_repaired():
    assert _direct_rule("uczniom","uczeń","NOUN",5,3,{"subst:pl:dat:m1":5},analyses("uczeń")) is None

def test_morph_conflict_is_not_repaired():
    assert _direct_rule("uczniom","uczni","NOUN",5,3,{"subst:sg:gen:m1":5},analyses("uczeń")) is None

def test_ambiguous_target_is_not_repaired():
    a=analyses("uczeń")+analyses("uczyć")
    assert _direct_rule("uczniom","uczni","NOUN",5,3,{"subst:pl:dat:m1":5},a) is None
