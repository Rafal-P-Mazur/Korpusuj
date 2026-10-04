# -*- coding: utf-8 -*-
"""Final cross-NER consistency checks for lemma-repair AUTO rules.

The layer does not trust NER as an absolute gate. It first applies existing
AUTO rules virtually, then looks for residual lemma variants of the same
orthographic form and morphology. It may also recover capitalized PROPN forms
missed by the NER-based proper-name path when SGJP gives one morphologically
compatible lemma and that lemma is independently attested in the corpus.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import lemma_repair_analysis as d3
from .lemma_repair_models import LemmaRepairPaths
from .lemma_repair_rules import ner_broad, rule_context_matches

VERSION = "1.2.0"
MIN_DOMINANT_SHARE = 0.95
MIN_DOMINANT_COUNT = 3
MIN_DOMINANT_DOCS = 2
MIN_MISSING_COUNT = 2
MIN_MISSING_DOCS = 2
MIN_ATTESTED_TARGET_COUNT = 3
MIN_ATTESTED_TARGET_DOCS = 2


def clean(value: Any) -> str:
    return str(value or "").strip()


def norm(value: Any) -> str:
    return d3.normalize_lemma(value).casefold()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def rule_key(rule: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        clean(rule.get("orth")),
        clean(rule.get("lemma")),
        clean(rule.get("upos")).upper(),
        clean(rule.get("morph_from")),
        clean(rule.get("required_ner_broad")).upper(),
    )


def build_rule_maps(rules: list[dict[str, Any]]):
    exact: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    generic: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for rule in rules:
        orth, lemma, upos, morph, required = rule_key(rule)
        if morph:
            exact[(orth, lemma, upos, morph, required)] = rule
        else:
            generic[(orth, lemma, upos, required)] = rule
    return exact, generic


def match_rule(exact, generic, orth, lemma, upos, morph, ner, doc_id, token_index):
    broad = ner_broad(ner)
    candidates = (
        exact.get((orth, lemma, upos, morph, broad)),
        generic.get((orth, lemma, upos, broad)),
        exact.get((orth, lemma, upos, morph, "")),
        generic.get((orth, lemma, upos, "")),
    )
    for rule in candidates:
        if rule is not None and rule_context_matches(rule, ner, doc_id, token_index):
            return rule
    return None


def compatible_analyses(engine, orth: str, upos: str, morph: str):
    status, analyses = d3.analyse_form(engine, orth)
    if status != "ok":
        return status, []
    output = []
    for analysis in analyses:
        lemma = d3.normalize_lemma(analysis.get("lemma"))
        analysis_upos = clean(analysis.get("upos")).upper()
        allowed_upos = {upos}
        if upos == "PROPN":
            allowed_upos.add("NOUN")
        if not lemma or analysis_upos not in allowed_upos:
            continue
        comparison = d3.compare_tags(morph, analysis.get("tag", "")) if morph else None
        mismatches = set((comparison or {}).get("mismatches") or {})
        # Proper-name tags from Stanza often have the right number/case but a
        # wrong grammatical gender. This layer may still repair the lemma; it
        # does not rewrite the tag. Any mismatch beyond gender remains unsafe.
        if morph and mismatches and not (upos == "PROPN" and mismatches <= {"gender"}):
            continue
        output.append({**analysis, "lemma": lemma, "comparison": comparison})
    return status, output


def _report_paths(paths: LemmaRepairPaths) -> tuple[Path, Path]:
    prefix = paths.artifact_prefix
    return prefix.with_suffix(".lemma_consistency.json"), prefix.with_suffix(".lemma_consistency.md")


def _write_report(paths, payload):
    json_path, md_path = _report_paths(paths)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Końcowa kontrola spójności lematów",
        "",
        f"**Wersja:** {payload['version']}  ",
        f"**Reguły przed warstwą:** {payload['rules_before']:,}  ",
        f"**Nowe reguły AUTO:** {payload['promoted_rules']:,}  ",
        f"**Konflikty pozostawione do REVIEW:** {payload['conflicts']:,}",
        "",
        "## Liczniki",
        "",
    ]
    for key, value in sorted(payload["counters"].items()):
        lines.append(f"- {key}: **{value:,}**")
    lines += ["", "## Nowe reguły", ""]
    for rule in payload["rules"]:
        lines.append(
            f"- `{rule['orth']}` + `{rule['lemma']}` + `{rule['upos']}`"
            f" + `{rule.get('morph_from') or '*'}` -> `{rule['replacement']}`"
            f" ({rule['classification']}; {rule['observed_count']} trafień)"
        )
    lines += ["", "## Rodziny pozostawione do REVIEW", ""]
    for rule in payload.get("review_rows", [])[:500]:
        lines.append(
            f"- `{rule['orth']}` + `{rule['lemma']}` -> `{rule['replacement']}` "
            f"({rule.get('observed_count', 0)} trafień; brak niezależnego mianownika lub wcześniejszego celu)"
        )
    lines += ["", "## Konflikty", ""]
    for item in payload["conflict_rows"][:300]:
        lines.append(
            f"- `{item['orth']}` + `{item['lemma']}` + `{item['upos']}`: "
            f"istniejący cel `{item['existing_target']}`, kandydat `{item['candidate_target']}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def augment_lemma_consistency_repairs(paths: LemmaRepairPaths, reporter: Any = None) -> dict[str, Any]:
    auto_path = paths.decisions_auto()
    payload = json.loads(auto_path.read_text(encoding="utf-8"))
    rules = list(payload.get("rules") or [])
    rules_before = len(rules)
    exact, generic = build_rule_maps(rules)
    accepted_targets = {
        norm(rule.get("replacement"))
        for rule in rules
        if clean(rule.get("replacement"))
        and clean(rule.get("decision_source")).upper() != "LEMMA_CONSISTENCY"
    }

    pf = pq.ParquetFile(paths.parquet)
    columns = set(pf.schema_arrow.names)
    morph_col = "full_postags" if "full_postags" in columns else ("postags" if "postags" in columns else None)
    ner_col = "ners" if "ners" in columns else ("ner" if "ner" in columns else None)
    read = ["tokens", "lemmas", "upostags"] + ([morph_col] if morph_col else []) + ([ner_col] if ner_col else [])

    # Statistics after virtually applying the current AUTO pool.
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    observations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    lemma_attestation: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "docs": set()})
    nominative_attestation: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "docs": set(), "orths": set()})
    malformed = 0
    doc_id = 0
    try:
        for batch in pf.iter_batches(batch_size=128, columns=read):
            data = batch.to_pydict()
            morph_rows = data[morph_col] if morph_col else [None] * batch.num_rows
            ner_rows = data[ner_col] if ner_col else [None] * batch.num_rows
            for tokens, lemmas, upos, morphs, ners in zip(
                data["tokens"], data["lemmas"], data["upostags"], morph_rows, ner_rows
            ):
                ts, ls, us = map(as_list, (tokens, lemmas, upos))
                ms = as_list(morphs) if morph_col else [""] * len(ts)
                ns = as_list(ners) if ner_col else ["O"] * len(ts)
                if not (len(ts) == len(ls) == len(us) == len(ms) == len(ns)):
                    malformed += 1
                    doc_id += 1
                    continue
                seen_lemmas = set()
                for pos, (orth, lemma, part, morph, ner) in enumerate(zip(ts, ls, us, ms, ns)):
                    orth = clean(orth)
                    lemma = clean(lemma)
                    part = clean(part).upper()
                    morph = clean(morph)
                    if not orth or not lemma or not part:
                        continue
                    matched = match_rule(exact, generic, orth, lemma, part, morph, ner, doc_id, pos)
                    final_lemma = d3.normalize_lemma(matched.get("replacement")) if matched else d3.normalize_lemma(lemma)
                    if not final_lemma:
                        final_lemma = lemma
                    final_key = norm(final_lemma)
                    seen_lemmas.add(final_key)
                    parsed_morph = d3.parse_tag(morph) if morph else {}
                    cases = {part for part in clean(parsed_morph.get("case")).casefold().split(".") if part}
                    # Independent nominative evidence means that the surface
                    # form itself is the target lemma in nominative. This does
                    # not rely on NER and therefore can safely validate a
                    # missing family later.
                    if "nom" in cases and norm(orth) == final_key:
                        nominative_attestation[final_key]["count"] += 1
                        nominative_attestation[final_key]["docs"].add(doc_id)
                        nominative_attestation[final_key]["orths"].add(orth)
                    group_key = (orth, part, morph)
                    group = groups.setdefault(group_key, {
                        "lemmas": Counter(), "docs": defaultdict(set), "ners": Counter(),
                        "sources": defaultdict(Counter),
                        "source_docs": defaultdict(lambda: defaultdict(set)),
                        "examples": defaultdict(list),
                    })
                    group["lemmas"][final_lemma] += 1
                    group["docs"][final_lemma].add(doc_id)
                    group["ners"][ner_broad(ner)] += 1
                    group["sources"][final_lemma][lemma] += 1
                    group["source_docs"][final_lemma][lemma].add(doc_id)
                    if len(group["examples"][lemma]) < 4:
                        left, right = max(0, pos - 7), min(len(ts), pos + 8)
                        group["examples"][lemma].append(" ".join(clean(x) for x in ts[left:right]))

                    obs_key = (orth, lemma, part, morph)
                    obs = observations.setdefault(obs_key, {
                        "count": 0, "docs": set(), "ners": Counter(), "examples": [],
                        "already_matched": 0,
                    })
                    obs["count"] += 1
                    obs["docs"].add(doc_id)
                    obs["ners"][ner_broad(ner)] += 1
                    obs["already_matched"] += int(matched is not None)
                    if len(obs["examples"]) < 4:
                        left, right = max(0, pos - 7), min(len(ts), pos + 8)
                        obs["examples"].append(" ".join(clean(x) for x in ts[left:right]))
                for lemma_key in seen_lemmas:
                    lemma_attestation[lemma_key]["docs"].add(doc_id)
                for pos, lemma in enumerate(ls):
                    lemma_key = norm(lemma)
                    if lemma_key:
                        lemma_attestation[lemma_key]["count"] += 1
                doc_id += 1
    finally:
        pf.close()

    engine, _version = d3.morfeusz_engine()
    candidates: list[dict[str, Any]] = []
    counters = Counter()

    def add_candidate(orth, source, upos, morph, target, classification, count, docs, examples, ners, reasons):
        if norm(source) == norm(target):
            return
        candidates.append({
            "orth": orth,
            "lemma": source,
            "upos": upos,
            "replacement": target,
            "reason": "Końcowa kontrola spójności lematów po wszystkich wcześniejszych warstwach.",
            "status": "accept",
            "decision_bucket": "auto",
            "decision_reasons": reasons,
            "classification": classification,
            "decision_source": "LEMMA_CONSISTENCY",
            "morph_from": morph,
            "morph_to": "",
            "observed_count": int(count),
            "source_document_count": int(docs),
            "examples": list(examples)[:6],
            "ner_distribution": dict(ners),
            # Deliberately NER-independent: promotion requires a unique SGJP
            # lemma for the exact orth+UPOS+morph combination.
            "required_ner_broad": "",
        })

    # A. Residual minority lemmas after virtual application of existing rules.
    for (orth, upos, morph), group in groups.items():
        total = sum(group["lemmas"].values())
        if len(group["lemmas"]) < 2 or total < MIN_DOMINANT_COUNT:
            continue
        dominant, dominant_count = group["lemmas"].most_common(1)[0]
        share = dominant_count / total if total else 0.0
        dominant_docs = len(group["docs"][dominant])
        if share < MIN_DOMINANT_SHARE or dominant_count < MIN_DOMINANT_COUNT or dominant_docs < MIN_DOMINANT_DOCS:
            counters["multi_lemma_review"] += 1
            continue
        status, analyses = compatible_analyses(engine, orth, upos, morph)
        targets = {norm(a["lemma"]): a["lemma"] for a in analyses}
        if len(targets) != 1 or norm(dominant) not in targets:
            counters["dominant_not_unique_in_sgjp"] += 1
            continue
        for final_lemma, minority_count in group["lemmas"].items():
            if final_lemma == dominant:
                continue
            for source_lemma, source_count in group["sources"][final_lemma].items():
                source_docs = len(group["source_docs"][final_lemma][source_lemma])
                if norm(source_lemma) in targets:
                    counters["minority_supported_by_sgjp"] += 1
                    continue
                add_candidate(
                    orth, source_lemma, upos, morph, dominant,
                    "MINORITY_LEMMA_NER_INDEPENDENT_REPAIR",
                    source_count, source_docs,
                    group["examples"].get(source_lemma, []), group["ners"],
                    ["DOMINANT_LEMMA_SHARE_95", "UNIQUE_SGJP_TARGET", "NER_NOT_REQUIRED"],
                )
                counters["minority_candidates"] += 1

    # B. Capitalized PROPN missed by the NER path.
    #
    # Morfeusz is the first gate: after UPOS/morph filtering there must be
    # exactly one normalized lemma. For PROPN the target must preserve
    # capitalization, which blocks common-homograph degradations such as
    # RAZEM -> raz, Sera -> ser, Swini -> swinia.
    #
    # A unique analysis of one isolated form is still insufficient. AUTO
    # requires at least two distinct orthographic forms and two documents
    # supporting the same target paradigm. All remaining rows stay only in
    # the diagnostic report.
    missing_rows = []
    target_families = defaultdict(lambda: {"orths": set(), "docs": set(), "rows": 0})
    for (orth, source, upos, morph), obs in observations.items():
        if upos != "PROPN" or not orth[:1].isupper() or not any(ch.isalpha() for ch in orth):
            continue
        if obs["already_matched"] >= obs["count"]:
            continue
        _status, analyses = compatible_analyses(engine, orth, upos, morph)
        targets = {norm(a["lemma"]): a["lemma"] for a in analyses}
        if len(targets) != 1:
            counters["missing_multiple_morfeusz_lemmas"] += 1
            continue
        target = next(iter(targets.values()))
        if not target[:1].isupper():
            counters["missing_common_homograph_for_propn"] += 1
            continue
        if norm(source) in targets or norm(source) == norm(target):
            continue
        row = {
            "orth": orth, "source": source, "upos": upos, "morph": morph,
            "target": target, "obs": obs,
        }
        missing_rows.append(row)
        family = target_families[norm(target)]
        family["orths"].add(orth)
        family["docs"].update(obs["docs"])
        family["rows"] += 1

    missing_review = []
    for row in missing_rows:
        target_key = norm(row["target"])
        family = target_families[target_key]
        if len(family["orths"]) < 2 or len(family["docs"]) < 2:
            counters["missing_without_two_form_family"] += 1
            continue
        obs = row["obs"]
        nominative = nominative_attestation.get(target_key, {"count": 0, "docs": set(), "orths": set()})
        evidence = []
        if target_key in accepted_targets:
            evidence.append("EARLIER_ACCEPTED_TARGET")
        if int(nominative.get("count", 0)) >= 1 and len(nominative.get("docs", set())) >= 1:
            evidence.append("CORPUS_ATTESTED_NOMINATIVE")
        if not evidence:
            missing_review.append({
                "orth": row["orth"],
                "lemma": row["source"],
                "upos": row["upos"],
                "replacement": row["target"],
                "reason": "Rodzina ma co najmniej dwie formy, ale brak niezależnego mianownika lub wcześniejszego zaakceptowanego celu.",
                "status": "review",
                "decision_bucket": "review",
                "decision_reasons": [
                    "UNIQUE_MORFEUSZ_LEMMA_AFTER_MORPH_FILTER",
                    "TWO_DISTINCT_FORMS_ONE_TARGET",
                    "MISSING_INDEPENDENT_NOMINATIVE_OR_ACCEPTED_TARGET",
                ],
                "classification": "MISSING_FROM_REPAIR_PIPELINE_FAMILY_REVIEW",
                "decision_source": "LEMMA_CONSISTENCY",
                "morph_from": row["morph"],
                "morph_to": "",
                "observed_count": int(obs["count"]),
                "source_document_count": len(obs["docs"]),
                "examples": list(obs["examples"])[:6],
                "ner_distribution": dict(obs["ners"]),
                "required_ner_broad": "",
                "family_distinct_forms": sorted(family["orths"]),
                "family_document_count": len(family["docs"]),
            })
            counters["missing_family_review_no_independent_evidence"] += 1
            continue
        add_candidate(
            row["orth"], row["source"], row["upos"], row["morph"], row["target"],
            "MISSING_FROM_REPAIR_PIPELINE_FAMILY_CONFIRMED",
            obs["count"], len(obs["docs"]), obs["examples"], obs["ners"],
            [
                "UNIQUE_MORFEUSZ_LEMMA_AFTER_MORPH_FILTER",
                "CAPITALIZED_TARGET_FOR_PROPN",
                "TWO_DISTINCT_FORMS_ONE_TARGET",
                "AT_LEAST_TWO_DOCUMENTS",
                *evidence,
                "NER_NOT_REQUIRED",
            ],
        )
        counters["missing_family_confirmed_candidates"] += 1

    # Deduplicate and protect existing targets.
    existing_by_key = {rule_key(rule): rule for rule in rules}
    promoted = []
    conflicts = []
    for rule in candidates:
        key = rule_key(rule)
        previous = existing_by_key.get(key)
        if previous is not None:
            if norm(previous.get("replacement")) != norm(rule.get("replacement")):
                conflicts.append({
                    "orth": rule["orth"], "lemma": rule["lemma"], "upos": rule["upos"],
                    "existing_target": previous.get("replacement"),
                    "candidate_target": rule.get("replacement"),
                })
            continue
        existing_by_key[key] = rule
        promoted.append(rule)

    if missing_review:
        review_path = paths.decisions_review()
        review_payload = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {
            "schema_version": 1,
            "bucket": "review",
            "rules": [],
        }
        review_rules = list(review_payload.get("rules") or [])
        review_keys = {
            (
                clean(rule.get("orth")), clean(rule.get("lemma")), clean(rule.get("upos")).upper(),
                clean(rule.get("morph_from")), clean(rule.get("replacement")), clean(rule.get("classification")),
            )
            for rule in review_rules
        }
        for rule in missing_review:
            key = (
                clean(rule.get("orth")), clean(rule.get("lemma")), clean(rule.get("upos")).upper(),
                clean(rule.get("morph_from")), clean(rule.get("replacement")), clean(rule.get("classification")),
            )
            if key not in review_keys:
                review_rules.append(rule)
                review_keys.add(key)
        review_payload["rules"] = review_rules
        review_payload["lemma_consistency_review_rules"] = len(missing_review)
        review_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    payload["rules"] = sorted(
        existing_by_key.values(),
        key=lambda r: (-int(r.get("observed_count") or 0), clean(r.get("orth")).casefold(), clean(r.get("lemma")).casefold()),
    )
    payload.setdefault("settings", {})["lemma_consistency"] = {
        "enabled": True,
        "version": VERSION,
        "minimum_dominant_share": MIN_DOMINANT_SHARE,
        "minimum_dominant_count": MIN_DOMINANT_COUNT,
        "minimum_dominant_documents": MIN_DOMINANT_DOCS,
        "missing_policy": "unique Morfeusz lemma + capitalized PROPN target + two distinct forms + two documents + independent nominative or earlier accepted target",
        "ner_policy": "diagnostic signal; not a gate after Morfeusz and family confirmation",
    }
    payload["lemma_consistency_promoted_rules"] = len(promoted)
    auto_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "schema_version": 1,
        "version": VERSION,
        "rules_before": rules_before,
        "rules_after": len(payload["rules"]),
        "promoted_rules": len(promoted),
        "review_rules": len(missing_review),
        "conflicts": len(conflicts),
        "malformed_documents": malformed,
        "counters": dict(counters),
        "rules": promoted,
        "review_rows": missing_review,
        "conflict_rows": conflicts,
    }
    _write_report(paths, report)
    if reporter:
        reporter.status(f"Końcowa kontrola spójności: dodano {len(promoted)} reguł...")
    return {
        "lemma_consistency_promoted_rules": len(promoted),
        "lemma_consistency_review_rules": len(missing_review),
        "lemma_consistency_conflicts": len(conflicts),
        "auto_rules": len(payload["rules"]),
    }
