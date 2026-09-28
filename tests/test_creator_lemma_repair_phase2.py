# -*- coding: utf-8 -*-
from korpusuj.corpus.creator_core import CreatorRunOptions
from korpusuj.corpus.creator_lemma_repair import VALID_CREATOR_LEMMA_REPAIR_MODES

def test_creator_modes():
    assert VALID_CREATOR_LEMMA_REPAIR_MODES == {"off","common-core","common-core-plus"}
    o=CreatorRunOptions(["x.pdf"],"x.parquet",lemma_repair_mode="common-core-plus")
    assert o.lemma_repair_mode=="common-core-plus"

def test_invalid_mode():
    try: CreatorRunOptions(["x.pdf"],"x.parquet",lemma_repair_mode="invalid")
    except ValueError: pass
    else: raise AssertionError("invalid mode accepted")
