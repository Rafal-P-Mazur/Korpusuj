# -*- coding: utf-8 -*-
"""Decision safeguards applied after base lemma-repair classification.

Contains the final SGJP safety gate and conservative recovery of review rules.
The functions are moved unchanged from the former small modules.
"""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .lemma_repair_models import LemmaRepairPaths

try:
    import morfeusz2
except Exception:  # pragma: no cover
    morfeusz2 = None

_NOUN_KINDS = {"subst", "depr"}

_ADJ_KINDS = {"adj", "adja", "adjp", "adjc"}

_VERB_KINDS = {
    "fin", "bedzie", "aglt", "praet", "impt", "imps", "inf",
    "pcon", "pant", "winien", "pred", "ppas", "pact",
}

def _clean(value: Any) -> str:
    return str(value or "").strip()

def _lemma(value: Any) -> str:
    return _clean(value).split(":", 1)[0]

def _tag_kind(tag: Any) -> str:
    return _clean(tag).split(":", 1)[0].lower()

def _analysis_parts(item: Any) -> tuple[str, str]:
    try:
        interp = item[2]
        lemma = interp[1]
        tag = interp[2]
        return _lemma(lemma), _clean(tag)
    except Exception:
        return "", ""

def _target_kinds(engine: Any, target: str) -> set[str]:
    kinds: set[str] = set()
    if engine is None or not target:
        return kinds
    try:
        analyses = engine.analyse(target)
    except Exception:
        return kinds
    target_key = target.casefold()
    for item in analyses:
        lemma, tag = _analysis_parts(item)
        if lemma.casefold() == target_key:
            kind = _tag_kind(tag)
            if kind:
                kinds.add(kind)
    return kinds

def _is_sgjp_rule(rule: dict[str, Any]) -> bool:
    source = _clean(rule.get("decision_source")).upper()
    classification = _clean(rule.get("classification")).upper()
    return "SGJP" in source or "SGJP" in classification

def _is_contextual(rule: dict[str, Any]) -> bool:
    source = _clean(rule.get("decision_source")).upper()
    classification = _clean(rule.get("classification")).upper()
    return source == "SGJP_CONTEXT" or classification == "SAFE_PREPOSITIONAL_SGJP_REPAIR"

def _reason(rule: dict[str, Any], engine: Any) -> str:
    if not _is_sgjp_rule(rule):
        return ""

    replacement = _lemma(rule.get("replacement"))
    if not replacement:
        return "EMPTY_SGJP_TARGET"
    rule["replacement"] = replacement

    upos = _clean(rule.get("upos")).upper()
    morph_from = _clean(rule.get("morph_from"))
    morph_to = _clean(rule.get("morph_to"))
    source_kind = _tag_kind(morph_from)
    target_kind = _tag_kind(morph_to)

    # The contextual layer is intentionally narrower than ordinary SGJP.
    # It may disambiguate nominal homographs, but it must not redefine the
    # corpus convention for gerunds, participles, abbreviations or POS.
    if _is_contextual(rule):
        if upos != "NOUN":
            return "CONTEXTUAL_REQUIRES_NOUN"
        if source_kind != "subst" or target_kind != "subst":
            return "CONTEXTUAL_REQUIRES_SUBST_TO_SUBST"
        if _lemma(rule.get("lemma")).casefold() == _lemma(rule.get("orth")).casefold():
            return "CONTEXTUAL_ATTESTED_LEMMA_ALREADY_MATCHES_FORM"

    if source_kind in {"ger", "brev"} or target_kind in {"ger", "brev"}:
        return "GERUND_OR_ABBREVIATION_BLOCKED"

    kinds = _target_kinds(engine, replacement)
    if upos == "ADJ" and kinds and not (kinds & _ADJ_KINDS):
        return "ADJ_TARGET_NOT_ADJECTIVAL"
    if upos == "NOUN" and kinds and not (kinds & _NOUN_KINDS):
        return "NOUN_TARGET_NOT_SUBSTANTIVE"
    if upos == "VERB" and kinds and not (kinds & _VERB_KINDS):
        return "VERB_TARGET_NOT_VERBAL"
    return ""

def filter_auto_rules(paths: Any) -> dict[str, Any]:
    path: Path = paths.decisions_auto()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = list(payload.get("rules") or [])
    engine = morfeusz2.Morfeusz(expand_tags=True) if morfeusz2 is not None else None
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for rule in rules:
        reason = _reason(rule, engine)
        if reason:
            counters[reason] += 1
            removed.append({
                "orth": rule.get("orth"),
                "lemma": rule.get("lemma"),
                "replacement": rule.get("replacement"),
                "upos": rule.get("upos"),
                "morph_from": rule.get("morph_from"),
                "morph_to": rule.get("morph_to"),
                "classification": rule.get("classification"),
                "decision_source": rule.get("decision_source"),
                "observed_count": rule.get("observed_count"),
                "reason": reason,
            })
        else:
            kept.append(rule)

    payload["rules"] = kept
    payload["sgjp_safety_filter"] = {
        "enabled": True,
        "rules_before": len(rules),
        "rules_after": len(kept),
        "rules_removed": len(removed),
        "removed_by_reason": dict(sorted(counters.items())),
        "morfeusz_available": engine is not None,
        "policy": "contextual only NOUN subst->subst; no ger/brev; target POS consistency",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = path.with_name(path.stem.replace(".auto", "") + ".sgjp_safety.json")
    report.write_text(json.dumps({
        "schema_version": 1,
        "summary": payload["sgjp_safety_filter"],
        "removed_rules": removed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload["sgjp_safety_filter"]

VERSION = "1.0.0"

_ALLOWED_REASONS = {
    "CAPITALIZED_FORM_REVIEW_REQUIRED",
    "CAPITALIZED_LEMMA_REVIEW_REQUIRED",
    "COMMON_CORE_UPOS_REVIEW_REQUIRED",
}

_ALLOWED_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "NUM"}

def clean(value: Any) -> str:
    return str(value or "").strip()

def folded_signature(rule: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean(rule.get("orth")).casefold(),
        clean(rule.get("lemma")).casefold(),
        clean(rule.get("replacement")).casefold(),
        clean(rule.get("upos")).upper(),
    )

def exact_signature(rule: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        clean(rule.get("orth")), clean(rule.get("lemma")),
        clean(rule.get("replacement")), clean(rule.get("upos")).upper(),
        clean(rule.get("morph_from")), clean(rule.get("required_ner_broad")).upper(),
    )

def all_upper(value: Any) -> bool:
    text = clean(value)
    return any(char.isalpha() for char in text) and text.isupper()

def recovery_reason(rule: dict[str, Any], auto_folded: set[tuple[str, str, str, str]]) -> str:
    if clean(rule.get("classification")) != "SAFE_FULL_MORPH_REPAIR":
        return ""
    if clean(rule.get("morph_status")) != "FULL_MORPH_MATCH":
        return ""
    if clean(rule.get("sgjp_status")) != "SGJP_UNIQUE_TARGET":
        return ""
    upos = clean(rule.get("upos")).upper()
    if upos == "PROPN" or upos not in _ALLOWED_UPOS:
        return ""
    reasons = {clean(item) for item in (rule.get("decision_reasons") or []) if clean(item)}
    if not reasons or reasons - _ALLOWED_REASONS:
        return ""
    if all_upper(rule.get("orth")):
        return "SAFE_FULL_MORPH_ALL_UPPER_NON_PROPN"
    if folded_signature(rule) in auto_folded:
        return "SAFE_FULL_MORPH_CASEFOLD_COUNTERPART_IN_AUTO"
    return ""

def recover_safe_review_rules(paths: LemmaRepairPaths, reporter: Any = None) -> dict[str, int]:
    auto_path = paths.decisions_auto()
    review_path = paths.decisions_review()
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    auto_rules = list(auto.get("rules") or [])
    review_rules = list(review.get("rules") or [])
    auto_folded = {folded_signature(rule) for rule in auto_rules}
    existing = {exact_signature(rule) for rule in auto_rules}

    promoted = []
    retained = []
    counters: dict[str, int] = {}
    for rule in review_rules:
        reason = recovery_reason(rule, auto_folded)
        if not reason:
            retained.append(rule)
            continue
        promoted_rule = deepcopy(rule)
        promoted_rule["status"] = "accept"
        promoted_rule["decision_bucket"] = "auto"
        promoted_rule["source_classification"] = clean(rule.get("classification"))
        promoted_rule["classification"] = "SAFE_CAPITALIZATION_RECOVERY"
        promoted_rule["decision_source"] = "REVIEW_RECOVERY"
        promoted_rule["decision_reasons"] = list(rule.get("decision_reasons") or []) + [reason]
        key = exact_signature(promoted_rule)
        if key in existing:
            retained.append(rule)
            continue
        existing.add(key)
        promoted.append(promoted_rule)
        counters[reason] = counters.get(reason, 0) + 1

    auto["rules"] = auto_rules + promoted
    auto.setdefault("settings", {})["review_recovery"] = {
        "enabled": True, "version": VERSION,
        "policy": "SAFE_FULL only: all-uppercase non-PROPN or existing casefold counterpart in AUTO",
    }
    auto["review_recovery_promoted_rules"] = len(promoted)
    review["rules"] = retained
    review["review_recovery_promoted_rules"] = len(promoted)
    auto_path.write_text(json.dumps(auto, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    if reporter:
        reporter.status(f"Odzysk bezpiecznych reguł REVIEW: {len(promoted)}...")
    return {
        "review_recovery_promoted_rules": len(promoted),
        "review_recovery_remaining_rules": len(retained),
        **{f"review_recovery_{key.lower()}": value for key, value in counters.items()},
    }
