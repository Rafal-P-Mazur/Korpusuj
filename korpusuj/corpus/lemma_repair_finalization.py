# -*- coding: utf-8 -*-
"""Final production consolidation and validation of AUTO rules.

This module fixes two failures exposed by the PG2-R builder:
1. same selector with equivalent/case-only targets or different morph_to;
2. AUTO rules that have zero real matches after all augmentation layers.

Genuinely different lemma targets are moved to REVIEW, never guessed.
Preview/apply remain hard safety gates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .lemma_repair_models import LemmaRepairPaths
from .lemma_repair_proper_names import recount_final_auto_rules

TRUST = {
    "PRNG_GENERATED_PARADIGM": 100,
    "PRNG": 95,
    "SGJP_GENDER": 90,
    "SGJP": 85,
    "TOKENWISE_MORFEUSZ_FULL_XPOS": 85,
    "TOKENWISE_BREV_PROTECTION": 85,
    "TOKENWISE_SENTENCE_INITIAL_CASE": 86,
    "TOKENWISE_HYPHENATED_ACRONYM_CASE": 86,
    "LEMMA_CONSISTENCY": 70,
    "RESIDUAL_CROSS_NER": 60,
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def selector(rule: dict[str, Any]):
    return (
        clean(rule.get("orth")), clean(rule.get("lemma")), clean(rule.get("upos")).upper(),
        clean(rule.get("morph_from")), clean(rule.get("required_ner_broad")).upper(),
    )


def score(rule: dict[str, Any]):
    source = clean(rule.get("decision_source")).upper()
    classification = clean(rule.get("classification")).upper()
    trust = max([value for name, value in TRUST.items() if name in source or name in classification] or [0])
    replacement = clean(rule.get("replacement"))
    canonical_case = int(clean(rule.get("upos")).upper() == "PROPN" and replacement[:1].isupper())
    morph = int(bool(clean(rule.get("morph_to"))))
    return (trust, canonical_case, morph, int(rule.get("source_document_count") or 0), int(rule.get("observed_count") or 0))


def consolidate_and_recount_auto_rules(paths: LemmaRepairPaths) -> dict[str, int]:
    auto_path = paths.decisions_auto()
    review_path = paths.decisions_review()
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for rule in auto.get("rules", []):
        grouped[selector(rule)].append(rule)
    kept, moved, resolutions = [], [], []
    for key, group in grouped.items():
        targets = {clean(rule.get("replacement")).casefold() for rule in group}
        if len(targets) > 1:
            for rule in group:
                item = dict(rule)
                item["status"] = "review"; item["decision_bucket"] = "review"
                item["decision_reasons"] = sorted(set((item.get("decision_reasons") or []) + ["CONFLICTING_AUTO_TARGETS"]))
                moved.append(item)
            resolutions.append({"selector": key, "action": "moved_to_review", "targets": sorted(targets)})
            continue
        winner = max(group, key=score)
        kept.append(winner)
        if len(group) > 1:
            resolutions.append({"selector": key, "action": "consolidated", "winner": winner, "discarded": [rule for rule in group if rule is not winner]})
    auto["rules"] = sorted(kept, key=lambda rule: (-int(rule.get("observed_count") or 0), clean(rule.get("orth")).casefold(), clean(rule.get("morph_from"))))
    review["rules"] = list(review.get("rules") or []) + moved
    auto_path.write_text(json.dumps(auto, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    before_recount = len(auto["rules"])
    recount = recount_final_auto_rules(paths)
    after_recount = int(recount.get("auto_rules") or 0)
    report = {
        "schema_version": 1,
        "rules_before_consolidation": sum(len(group) for group in grouped.values()),
        "rules_after_consolidation": before_recount,
        "conflicting_rules_moved_to_review": len(moved),
        "zero_hit_rules_removed": before_recount - after_recount,
        "rules_after_recount": after_recount,
        "resolutions": resolutions,
    }
    paths.artifact_prefix.with_suffix(".auto_finalization.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "auto_conflict_groups": sum(1 for item in resolutions if item["action"] == "moved_to_review"),
        "auto_conflicting_rules_moved_to_review": len(moved),
        "auto_equivalent_groups_consolidated": sum(1 for item in resolutions if item["action"] == "consolidated"),
        "final_auto_zero_hit_removed": before_recount - after_recount,
        "auto_rules": after_recount,
    }
