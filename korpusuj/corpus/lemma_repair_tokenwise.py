# -*- coding: utf-8 -*-
"""Conservative token-level lemma repairs.

Policies:
- common words: accept one Morfeusz lemma compatible with complete XPOS;
- preserve resource spelling; prefer lowercase only when compatible resource
  analyses actually offer it for a non-PROPN token;
- abbreviations: preserve multiword lemmas and every current expansion that has
  any nontechnical lexical analysis; otherwise use FORM as a narrow fallback;
- never infer an abbreviation expansion from a unique Morfeusz `brev` analysis.

Proper names remain handled by the stricter NER/PRNG path.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any

import pyarrow.parquet as pq

from . import lemma_repair_analysis as d3
from .lemma_repair_models import LemmaRepairPaths

VERSION = "1.4.1"
ALLOWED_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}
TECHNICAL_KINDS = {"ign", "xxx", "burk", "interp", "brev"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def key(rule: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        clean(rule.get("orth")), clean(rule.get("lemma")),
        clean(rule.get("upos")).upper(), clean(rule.get("morph_from")),
        clean(rule.get("required_ner_broad")).upper(),
    )


def runtime_engine():
    result = d3.morfeusz_engine()
    return result[0] if isinstance(result, tuple) else result


def analyses_for_form(engine: Any, orth: str) -> tuple[str, list[dict[str, Any]]]:
    status, analyses = d3.analyse_form(engine, orth)
    return status, list(analyses or [])


def choose_resource_case(targets: set[str]) -> tuple[str | None, str]:
    """Choose attested resource spelling without manufacturing lowercase."""
    targets = {clean(value) for value in targets if clean(value)}
    if not targets:
        return None, "no_target"
    lowercase = sorted(value for value in targets if value == value.casefold())
    if lowercase and len({value.casefold() for value in lowercase}) == 1:
        return lowercase[0], "resource_lowercase_available"
    folded = {value.casefold() for value in targets}
    if len(folded) == 1:
        canonical = sorted(targets, key=lambda value: (value != value.upper(), value))[0]
        return canonical, "single_resource_lexeme_case_preserved"
    return None, "ambiguous_resource_lemmas"


def _explicit_nominal_subtype_conflict(stanza_tag: str, resource_tag: str) -> bool:
    """Reject only an explicit ncol/pt contradiction hidden by coarse gender matching."""
    stanza_parts = set(clean(stanza_tag).split(":"))
    resource_parts = set(clean(resource_tag).split(":"))
    return (
        ("ncol" in stanza_parts and "pt" in resource_parts)
        or ("pt" in stanza_parts and "ncol" in resource_parts)
    )


def unique_full_xpos_target(engine: Any, orth: str, lemma: str, upos: str, morph: str):
    status, analyses = analyses_for_form(engine, orth)
    if status != "ok":
        return None, {"status": status, "analyses": analyses}
    compatible = []
    current_confirmed = False
    # Confirmation outside the full-XPOS filter is used only by the explicit
    # ncol/pt safety gate. It must not globally suppress ordinary repairs.
    current_any_nontechnical_confirmed = any(
        d3.normalize_lemma(analysis.get("lemma")).casefold() == lemma.casefold()
        and clean(analysis.get("upos")).upper() == upos
        and clean(analysis.get("tag")).split(":", 1)[0].casefold() not in TECHNICAL_KINDS
        for analysis in analyses
        if d3.normalize_lemma(analysis.get("lemma"))
    )
    nominal_subtype_conflicts = []
    for analysis in analyses:
        target = d3.normalize_lemma(analysis.get("lemma"))
        analysis_upos = clean(analysis.get("upos")).upper()
        analysis_tag = clean(analysis.get("tag"))
        kind = analysis_tag.split(":", 1)[0].casefold()
        if not target or analysis_upos != upos or kind in TECHNICAL_KINDS:
            continue
        if (
            kind == "subst"
            and _explicit_nominal_subtype_conflict(morph, analysis_tag)
            and current_any_nontechnical_confirmed
        ):
            nominal_subtype_conflicts.append({
                "lemma": target,
                "stanza_tag": morph,
                "resource_tag": analysis_tag,
                "reason": "EXPLICIT_NCOL_PT_CONFLICT_CURRENT_LEMMA_CONFIRMED",
            })
            continue
        comparison = d3.compare_tags(morph, analysis_tag)
        if comparison.get("mismatches") or not comparison.get("shared"):
            continue
        compatible.append({"lemma": target, "tag": analysis_tag, "comparison": comparison})
        if target.casefold() == lemma.casefold():
            current_confirmed = True
    targets = {item["lemma"] for item in compatible if item["lemma"]}
    target, case_policy = choose_resource_case(targets)
    if current_confirmed or target is None:
        return None, {
            "status": status, "current_confirmed": current_confirmed,
            "current_any_nontechnical_confirmed": current_any_nontechnical_confirmed,
            "compatible": compatible, "nominal_subtype_conflicts": nominal_subtype_conflicts,
            "case_policy": case_policy,
        }
    return target, {
        "status": status, "current_confirmed": False,
        "current_any_nontechnical_confirmed": current_any_nontechnical_confirmed,
        "compatible": compatible, "nominal_subtype_conflicts": nominal_subtype_conflicts,
        "case_policy": case_policy,
    }


def lemma_has_nontechnical_analysis(engine: Any, lemma: str) -> tuple[bool, list[dict[str, Any]]]:
    """Preserve a current abbreviation expansion if any probe is lexical.

    Identity between the probe and the normalized analysis lemma is deliberately
    not required. Inflected but credible expansions such as `imienia` must be
    protected even when Morfeusz normalizes them to a different base lemma.
    """
    probes = []
    for candidate in (lemma, lemma.replace("_", " "), lemma.replace("_", "")):
        if candidate and candidate not in probes:
            probes.append(candidate)
    evidence = []
    for probe in probes:
        status, analyses = analyses_for_form(engine, probe)
        for analysis in analyses:
            kind = clean(analysis.get("tag")).split(":", 1)[0].casefold()
            evidence.append({"probe": probe, "status": status, "kind": kind, **analysis})
            if kind and kind not in TECHNICAL_KINDS:
                return True, evidence
    return False, evidence


def abbreviation_target(engine: Any, orth: str, lemma: str, morph: str):
    # Multiword expanded lemmas are an established corpus convention. Morfeusz
    # often reports these strings as ign, which must not erase the expansion.
    if "_" in lemma:
        return None, {"policy": "keep_multiword_convention", "lemma": lemma}

    confirmed, lemma_evidence = lemma_has_nontechnical_analysis(engine, lemma)
    if confirmed:
        return None, {
            "policy": "keep_any_nontechnical_current_analysis",
            "lemma_evidence": lemma_evidence,
        }

    # A unique `brev` analysis of FORM is not a safe expansion signal. Single
    # letters and short strings are heavily homonymous (J, G, H, S, Rz, L.A).
    # Therefore an unconfirmed erroneous current lemma falls back only to FORM.
    if lemma.casefold() != orth.casefold():
        status, form_analyses = analyses_for_form(engine, orth)
        return orth, {
            "policy": "surface_fallback_unconfirmed_current",
            "status": status,
            "form_analyses": form_analyses,
            "lemma_evidence": lemma_evidence,
        }
    return None, {
        "policy": "no_safe_change",
        "lemma_evidence": lemma_evidence,
    }




# KORPUSUJ_PATCH_CAPITALIZATION_REPAIRS_20261012
_CASE_ALLOWED_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "INTJ", "PRON"}
_ACRONYM_PREFIX_RE = re.compile(r"^[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+$")


def _sentence_initial_resource_target(engine: Any, orth: str, lemma: str, upos: str, morph: str):
    """Return a lowercase, resource-attested case-only target.

    The current spelling must not itself be attested by a compatible lexical
    analysis.  This deliberately excludes Azjata-like canonical uppercase
    nouns and ambiguous pairs such as Mercedes/mercedes.
    """
    status, analyses = analyses_for_form(engine, orth)
    if status != "ok":
        return None, {"status": status, "analyses": analyses, "policy": "analysis_unavailable"}
    compatible = []
    for analysis in analyses:
        target = d3.normalize_lemma(analysis.get("lemma"))
        analysis_upos = clean(analysis.get("upos")).upper()
        kind = clean(analysis.get("tag")).split(":", 1)[0].casefold()
        if not target or analysis_upos != upos or kind in TECHNICAL_KINDS:
            continue
        comparison = d3.compare_tags(morph, analysis.get("tag", ""))
        if comparison.get("mismatches") or not comparison.get("shared"):
            continue
        compatible.append({"lemma": target, "tag": clean(analysis.get("tag")), "comparison": comparison})
    targets = {item["lemma"] for item in compatible}
    lowercase = sorted(
        target for target in targets
        if target == target.casefold() and target.casefold() == lemma.casefold()
    )
    exact_current = lemma in targets
    if len(lowercase) != 1 or exact_current:
        return None, {
            "status": status,
            "compatible": compatible,
            "exact_current_confirmed": exact_current,
            "lowercase_case_only_targets": lowercase,
            "policy": "ambiguous_or_current_confirmed",
        }
    return lowercase[0], {
        "status": status,
        "compatible": compatible,
        "exact_current_confirmed": False,
        "policy": "gold_like_lowercase_only_resource_target",
    }


def _hyphenated_acronym_target(orth: str, lemma: str):
    """Restore only the acronym prefix spelling, preserving lemma suffix."""
    if "-" not in orth or "-" not in lemma:
        return None
    orth_prefix, _orth_suffix = orth.split("-", 1)
    lemma_prefix, lemma_suffix = lemma.split("-", 1)
    if not orth_prefix or not lemma_prefix or not lemma_suffix:
        return None
    if orth_prefix.casefold() != lemma_prefix.casefold() or orth_prefix == lemma_prefix:
        return None
    if not _ACRONYM_PREFIX_RE.fullmatch(orth_prefix):
        return None
    uppercase_count = sum(char.isupper() for char in orth_prefix)
    if uppercase_count < 2 or orth_prefix.islower():
        return None
    target = orth_prefix + "-" + lemma_suffix
    return target if target.casefold() == lemma.casefold() and target != lemma else None


def augment_capitalization_repairs(
    paths: LemmaRepairPaths,
    engine: Any,
    existing: dict[tuple[str, str, str, str, str], dict[str, Any]],
    counters: Counter,
    promoted: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> None:
    """Add conservative case-only rules using real sentence boundaries and NER.

    A selector is promoted only when every occurrence represented by that exact
    FORM+LEMMA+UPOS+XPOS tuple satisfies the relevant safety policy.  This is
    essential because production rules do not have a sentence-position selector.
    """
    parquet = pq.ParquetFile(paths.parquet)
    columns = set(parquet.schema_arrow.names)
    morph_column = "full_postags" if "full_postags" in columns else ("postags" if "postags" in columns else None)
    if not morph_column or "sentence_starts" not in columns or "ners" not in columns:
        counters["case_scan_unavailable"] += 1
        parquet.close()
        return

    groups = defaultdict(lambda: {
        "count": 0, "docs": set(), "all_sentence_initial": True,
        "all_ner_o": True, "targets": set(), "evidence": [],
        "kind": None, "blocked": Counter(),
    })
    doc_id = 0
    try:
        scan_columns = ["tokens", "lemmas", "upostags", morph_column, "ners", "sentence_starts"]
        for batch in parquet.iter_batches(batch_size=128, columns=scan_columns):
            data = batch.to_pydict()
            for tokens, lemmas, upos_rows, morphs, ners, starts in zip(
                data["tokens"], data["lemmas"], data["upostags"],
                data[morph_column], data["ners"], data["sentence_starts"],
            ):
                start_set = {int(value) for value in (starts or [])}
                for index, (orth, lemma, upos, morph, ner) in enumerate(zip(
                    tokens or [], lemmas or [], upos_rows or [], morphs or [], ners or []
                )):
                    orth, lemma = clean(orth), clean(lemma)
                    upos, morph, ner = clean(upos).upper(), clean(morph), clean(ner).upper()
                    if not orth or not lemma or not morph:
                        continue
                    selector_key = (orth, lemma, upos, morph, "O")
                    item = groups[selector_key]
                    item["count"] += 1
                    item["docs"].add(doc_id)

                    acronym_target = _hyphenated_acronym_target(orth, lemma)
                    if acronym_target and upos == "ADJ" and morph.split(":", 1)[0].casefold() != "brev":
                        item["kind"] = item["kind"] or "hyphenated_acronym"
                        if item["kind"] != "hyphenated_acronym":
                            item["blocked"]["mixed_policy"] += 1
                        item["targets"].add(acronym_target)
                        item["all_ner_o"] = item["all_ner_o"] and ner == "O"
                        continue

                    if upos not in _CASE_ALLOWED_UPOS:
                        item["blocked"]["blocked_by_upos"] += 1
                        continue
                    if upos == "PROPN" or morph.split(":", 1)[0].casefold() == "brev":
                        item["blocked"]["blocked_by_propn_or_brev"] += 1
                        continue
                    if index not in start_set:
                        item["all_sentence_initial"] = False
                        item["blocked"]["not_sentence_initial"] += 1
                    if ner != "O":
                        item["all_ner_o"] = False
                        item["blocked"]["blocked_by_ner"] += 1
                    if not lemma[:1].isupper() or lemma.isupper():
                        item["blocked"]["not_initial_case_pattern"] += 1
                        continue
                    target, evidence = _sentence_initial_resource_target(engine, orth, lemma, upos, morph)
                    item["kind"] = item["kind"] or "sentence_initial"
                    if item["kind"] != "sentence_initial":
                        item["blocked"]["mixed_policy"] += 1
                    if target:
                        item["targets"].add(target)
                        if len(item["evidence"]) < 4:
                            item["evidence"].append(evidence)
                    else:
                        item["blocked"][evidence.get("policy", "resource_rejected")] += 1
                doc_id += 1
    finally:
        parquet.close()

    for selector_key, item in groups.items():
        orth, lemma, upos, morph, required_ner = selector_key
        kind = item["kind"]
        if not kind:
            continue
        counters["case_candidates_scanned"] += 1
        if not item["all_ner_o"]:
            counters["case_blocked_by_ner"] += 1
            continue
        if kind == "sentence_initial" and not item["all_sentence_initial"]:
            counters["case_blocked_by_noninitial_occurrence"] += 1
            continue
        if item["blocked"] or len(item["targets"]) != 1:
            counters["case_blocked_by_resource_or_ambiguity"] += 1
            continue
        target = next(iter(item["targets"]))
        if target == lemma:
            continue
        if kind == "sentence_initial":
            classification = "SAFE_SENTENCE_INITIAL_RESOURCE_CASE"
            source = "TOKENWISE_SENTENCE_INITIAL_CASE"
            reasons = [
                "ALL_SELECTOR_OCCURRENCES_SENTENCE_INITIAL", "NER_O",
                "MORFEUSZ_GOLD_LIKE_CASE_ONLY_TARGET", "CURRENT_CASE_NOT_RESOURCE_CONFIRMED",
            ]
            reason = "Morfeusz potwierdza wyłącznie małoliterowy wariant leksemu na początku zdania."
        else:
            classification = "SAFE_HYPHENATED_ACRONYM_CASE"
            source = "TOKENWISE_HYPHENATED_ACRONYM_CASE"
            reasons = ["HYPHENATED_ACRONYM_PREFIX", "FORM_PREFIX_CASE_PRESERVED", "LEMMA_SUFFIX_PRESERVED"]
            reason = "Kapitalizacja skrótowca przed łącznikiem została zachowana z FORM."
        rule = {
            "orth": orth, "lemma": lemma, "upos": upos, "replacement": target,
            "morph_from": morph, "required_ner_broad": required_ner,
            "status": "accept", "decision_bucket": "auto",
            "classification": classification, "decision_source": source,
            "decision_reasons": reasons, "reason": reason,
            "observed_count": int(item["count"]),
            "source_document_count": len(item["docs"]),
            "capitalization_evidence": {
                "kind": kind, "targets": sorted(item["targets"]),
                "resource": item["evidence"], "all_sentence_initial": item["all_sentence_initial"],
                "all_ner_o": item["all_ner_o"],
            },
        }
        previous = existing.get(key(rule))
        if previous is None:
            existing[key(rule)] = rule
            promoted.append(rule)
            counters["sentence_initial_case_promoted" if kind == "sentence_initial" else "hyphenated_acronym_case_promoted"] += 1
        elif clean(previous.get("replacement")) != target:
            counters["case_conflicts_deferred"] += 1
            diagnostics.append({"rule": rule, "existing": previous})


def augment_tokenwise_repairs(paths: LemmaRepairPaths, reporter: Any = None) -> dict[str, int]:
    auto_path = paths.decisions_auto()
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    existing = {key(rule): rule for rule in auto.get("rules", [])}
    work = sqlite3.connect(paths.workspace)
    engine = runtime_engine()
    counters = Counter()
    promoted = []
    diagnostics = []
    try:
        rows = work.execute(
            "SELECT lemma,upos,orth,morph,token_count,document_count "
            "FROM form_stats ORDER BY orth,lemma,upos,morph"
        )
        for lemma, upos, orth, morph, token_count, document_count in rows:
            lemma, upos, orth, morph = clean(lemma), clean(upos).upper(), clean(orth), clean(morph)
            if not lemma or not orth or not morph or upos not in ALLOWED_UPOS:
                continue
            counters["scanned"] += 1
            target, evidence = unique_full_xpos_target(engine, orth, lemma, upos, morph)
            if evidence.get("nominal_subtype_conflicts"):
                counters["nominal_subtype_conflicts_blocked"] += 1
                diagnostics.append({
                    "orth": orth, "lemma": lemma, "upos": upos, "morph_from": morph,
                    "classification": "BLOCKED_EXPLICIT_NCOL_PT_CONFLICT",
                    "decision_source": "TOKENWISE_MORFEUSZ_FULL_XPOS",
                    "observed_count": int(token_count),
                    "source_document_count": int(document_count),
                    "morfeusz_evidence": evidence,
                })
            if not target or target == lemma:
                continue
            rule = {
                "orth": orth, "lemma": lemma, "upos": upos, "replacement": target,
                "morph_from": morph, "status": "accept", "decision_bucket": "auto",
                "classification": "SAFE_TOKENWISE_MORFEUSZ_FULL_XPOS",
                "decision_source": "TOKENWISE_MORFEUSZ_FULL_XPOS",
                "decision_reasons": ["MORFEUSZ_UNIQUE_FULL_XPOS_MATCH", "MODEL_LEMMA_NOT_CONFIRMED", "RESOURCE_CASE_PRESERVED"],
                "reason": "Jedyny lemat Morfeusza zgodny z pełnym XPOS różni się od lematu modelu.",
                "observed_count": int(token_count), "source_document_count": int(document_count),
                "morfeusz_evidence": evidence,
            }
            previous = existing.get(key(rule))
            if previous is None:
                existing[key(rule)] = rule; promoted.append(rule); counters["morfeusz"] += 1
            elif clean(previous.get("replacement")).casefold() != target.casefold():
                counters["conflicts_deferred"] += 1
                diagnostics.append({"rule": rule, "existing": previous})
    finally:
        work.close()

    augment_capitalization_repairs(
        paths, engine, existing, counters, promoted, diagnostics
    )
    # brev/X is excluded from the family workspace, so scan that class directly.
    brev = defaultdict(lambda: {"count": 0, "docs": set()})
    parquet = pq.ParquetFile(paths.parquet)
    try:
        columns = set(parquet.schema_arrow.names)
        morph_column = "full_postags" if "full_postags" in columns else ("postags" if "postags" in columns else None)
        if morph_column:
            doc_id = 0
            for batch in parquet.iter_batches(batch_size=128, columns=["tokens", "lemmas", "upostags", morph_column]):
                data = batch.to_pydict()
                for tokens, lemmas, upos_rows, morphs in zip(data["tokens"], data["lemmas"], data["upostags"], data[morph_column]):
                    for orth, lemma, upos, morph in zip(tokens or [], lemmas or [], upos_rows or [], morphs or []):
                        orth, lemma, upos, morph = clean(orth), clean(lemma), clean(upos).upper(), clean(morph)
                        if morph.split(":", 1)[0].casefold() != "brev" or not orth or not lemma:
                            continue
                        item = brev[(orth, lemma, upos, morph)]
                        item["count"] += 1; item["docs"].add(doc_id)
                    doc_id += 1
    finally:
        parquet.close()

    brev_diagnostics = []
    for (orth, lemma, upos, morph), item in brev.items():
        target, evidence = abbreviation_target(engine, orth, lemma, morph)
        counters["brev_scanned"] += 1
        counters["brev_" + evidence["policy"]] += 1
        if not target or target == lemma:
            continue
        rule = {
            "orth": orth, "lemma": lemma, "upos": upos, "replacement": target,
            "morph_from": morph, "status": "accept", "decision_bucket": "auto",
            "classification": "SAFE_BREV_SURFACE_FALLBACK",
            "decision_source": "TOKENWISE_BREV_PROTECTION",
            "decision_reasons": ["XPOS_BREV", evidence["policy"].upper(), "NO_AUTO_BREV_EXPANSION"],
            "reason": "Wiarygodne rozwinięcie skrótu pozostaje bez zmian; FORM jest fallbackiem tylko dla niepotwierdzonego błędu.",
            "observed_count": int(item["count"]), "source_document_count": len(item["docs"]),
            "morfeusz_evidence": evidence,
        }
        previous = existing.get(key(rule))
        if previous is None:
            existing[key(rule)] = rule; promoted.append(rule); counters["brev"] += 1
        elif clean(previous.get("replacement")).casefold() != target.casefold():
            counters["brev_conflicts_deferred"] += 1
            brev_diagnostics.append({"rule": rule, "existing": previous})

    auto["rules"] = sorted(existing.values(), key=lambda rule: (-int(rule.get("observed_count") or 0), clean(rule.get("orth")).casefold(), clean(rule.get("morph_from"))))
    auto.setdefault("settings", {})["tokenwise_repairs"] = {
        "version": VERSION, "allowed_upos": sorted(ALLOWED_UPOS),
        "capitalization_policy": "sentence-initial lowercase only when all selector occurrences are sentence-initial, NER=O, and Morfeusz confirms only the lowercase case variant; preserve acronym prefix before hyphen",
        "brev_policy": "keep underscore multiword; keep any nontechnical analysis of current expansion; otherwise FORM fallback; never infer expansion from FORM brev analysis",
        "technical_tags_not_lexical_confirmation": sorted(TECHNICAL_KINDS),
        "non_propn_case_policy": "prefer actual resource lowercase; otherwise preserve sole resource spelling",
        "nominal_subtype_policy": "explicit ncol/pt contradiction is blocked only when the current lemma has another nontechnical Morfeusz analysis; unspecified n remains eligible",
    }
    auto_path.write_text(json.dumps(auto, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "schema_version": 6, "version": VERSION, "counters": dict(counters),
        "rules": promoted, "conflicts": diagnostics, "brev_conflicts": brev_diagnostics,
    }
    paths.artifact_prefix.with_suffix(".tokenwise_repairs.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "tokenwise_scanned": int(counters["scanned"]),
        "tokenwise_morfeusz_promoted_rules": int(counters["morfeusz"]),
        "tokenwise_nominal_subtype_conflicts_blocked": int(counters["nominal_subtype_conflicts_blocked"]),
        "tokenwise_brev_scanned": int(counters["brev_scanned"]),
        "tokenwise_brev_promoted_rules": int(counters["brev"]),
        "tokenwise_brev_keep_nontechnical": int(counters["brev_keep_any_nontechnical_current_analysis"]),
        "tokenwise_brev_keep_multiword": int(counters["brev_keep_multiword_convention"]),
        "tokenwise_brev_surface_fallback": int(counters["brev_surface_fallback_unconfirmed_current"]),
        "tokenwise_conflicts_deferred": int(counters["conflicts_deferred"] + counters["brev_conflicts_deferred"] + counters["case_conflicts_deferred"]),
        "case_candidates_scanned": int(counters["case_candidates_scanned"]),
        "sentence_initial_case_promoted": int(counters["sentence_initial_case_promoted"]),
        "hyphenated_acronym_case_promoted": int(counters["hyphenated_acronym_case_promoted"]),
        "case_blocked_by_ner": int(counters["case_blocked_by_ner"]),
        "case_blocked_by_noninitial_occurrence": int(counters["case_blocked_by_noninitial_occurrence"]),
        "case_blocked_by_resource_or_ambiguity": int(counters["case_blocked_by_resource_or_ambiguity"]),
        "auto_rules": len(auto["rules"]),
    }
