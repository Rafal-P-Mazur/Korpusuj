# -*- coding: utf-8 -*-
"""Shared context checks for lemma-repair rules."""
from __future__ import annotations
import re
from typing import Any, Mapping


def clean(value: Any) -> str:
    return str(value or "").strip()


def ner_broad(value: Any) -> str:
    upper = clean(value).upper()
    if upper in {"", "O", "_"}:
        return "O"
    match = re.match(r"^([BILOUSE])[-_](.+)$", upper)
    core = match.group(2) if match else upper
    if core in {"PER", "PERSON", "PERSNAME", "PERSONNAME"}:
        return "PER"
    if core in {"LOC", "LOCATION", "GPE", "PLACE", "PLACENAME", "GEOG", "GEONAME", "GEOGNAME"}:
        return "LOC"
    if core in {"ORG", "ORGANIZATION", "ORGNAME"}:
        return "ORG"
    if core in {"FAC", "FACILITY"}:
        return "FAC"
    return core


def rule_context_matches(rule: Mapping[str, Any], ner_value: Any, doc_id: int, token_index: int) -> bool:
    required = clean(rule.get("required_ner_broad")).upper()
    if required and ner_broad(ner_value) != required:
        return False
    positions = rule.get("positions") or []
    if positions and not any(
        int(item.get("doc_id", -1)) == int(doc_id)
        and int(item.get("token_index", -1)) == int(token_index)
        for item in positions
    ):
        return False
    return True


def self_test() -> None:
    per = {"required_ner_broad": "PER"}
    assert ner_broad("B-PER") == "PER"
    assert ner_broad("S-orgName") == "ORG"
    assert ner_broad("U-GPE") == "LOC"
    assert rule_context_matches(per, "B-PER", 1, 2)
    assert not rule_context_matches(per, "B-LOC", 1, 2)
    assert not rule_context_matches(per, "O", 1, 2)
    assert rule_context_matches({}, "O", 1, 2)
