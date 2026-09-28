# -*- coding: utf-8 -*-
"""Lekki smoke test importow produkcyjnej uslugi D3."""
from korpusuj.corpus.lemma_repair_models import LemmaRepairOptions, LemmaRepairPaths
from korpusuj.corpus import lemma_repair_service

def test_contract_imports():
    assert LemmaRepairOptions().mode == "common-core"
    assert callable(lemma_repair_service.run)
