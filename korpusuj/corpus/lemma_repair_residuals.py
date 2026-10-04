# -*- coding: utf-8 -*-
"""Residual lemma repairs after all ordinary repair layers.

Two conservative mechanisms are implemented:
1. Cross-morph fallback for an orth+source lemma+UPOS family whose existing
   accepted rules all have one target and whose remaining token has no
   competing Morfeusz lemma.
2. Geographic form matching generated from a canonical PRNG name already
   accepted by an earlier rule. The observed form and grammatical case must
   exactly match one generated form and exactly one canonical target.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import lemma_repair_analysis as d3
from .lemma_repair_models import LemmaRepairPaths
from korpusuj.runtime_paths import resource_root
from .lemma_repair_rules import ner_broad, rule_context_matches

VERSION = "1.1.0"


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


def split_values(value: Any) -> set[str]:
    return {part for part in clean(value).casefold().split(".") if part}



def case_values_overlap(left: Any, right: Any) -> bool:
    """Compare case collections regardless of JSON/runtime container type."""
    def values(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return split_values(value)
        if isinstance(value, (set, frozenset, list, tuple)):
            output = set()
            for item in value:
                output.update(split_values(item))
            return output
        return split_values(value)
    return bool(values(left) & values(right))

def rule_key(rule: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        clean(rule.get("orth")), clean(rule.get("lemma")),
        clean(rule.get("upos")).upper(), clean(rule.get("morph_from")),
        clean(rule.get("required_ner_broad")).upper(),
    )


def build_rule_maps(rules):
    exact, generic = {}, {}
    for rule in rules:
        orth, lemma, upos, morph, required = rule_key(rule)
        if morph:
            exact[(orth, lemma, upos, morph, required)] = rule
        else:
            generic[(orth, lemma, upos, required)] = rule
    return exact, generic


def match_rule(exact, generic, orth, lemma, upos, morph, ner, doc_id, token_index):
    broad = ner_broad(ner)
    for rule in (
        exact.get((orth, lemma, upos, morph, broad)),
        generic.get((orth, lemma, upos, broad)),
        exact.get((orth, lemma, upos, morph, "")),
        generic.get((orth, lemma, upos, "")),
    ):
        if rule is not None and rule_context_matches(rule, ner, doc_id, token_index):
            return rule
    return None


def morfeusz_analyses(engine, orth: str, upos: str) -> set[str]:
    status, analyses = d3.analyse_form(engine, orth)
    if status != "ok":
        return set()
    allowed = {upos}
    if upos == "PROPN":
        allowed.add("NOUN")
    return {
        norm(item.get("lemma"))
        for item in analyses
        if d3.normalize_lemma(item.get("lemma"))
        and clean(item.get("upos")).upper() in allowed
    }


def generated_rows(engine, canonical: str):
    try:
        raw = engine.generate(canonical)
    except Exception:
        return []
    output = []
    for item in raw or []:
        form = lemma = tag = ""
        if isinstance(item, dict):
            form = clean(item.get("orth") or item.get("form"))
            lemma = d3.normalize_lemma(item.get("lemma"))
            tag = clean(item.get("tag"))
        elif isinstance(item, (list, tuple)):
            # Morfeusz2 commonly returns (form, lemma, tag, labels, qualifiers).
            if len(item) >= 3:
                form = clean(item[0])
                lemma = d3.normalize_lemma(item[1])
                tag = clean(item[2])
        if not form or not tag:
            continue
        output.append({"form": form, "lemma": lemma or canonical, "tag": tag})
    return output


def canonical_exists(con: sqlite3.Connection, canonical: str) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM objects WHERE canonical_norm=? LIMIT 1", (norm(canonical),)
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def raw_generator_lexemes(engine, forms: list[str], canonical: str) -> dict[str, set[str]]:
    """Return raw SGJP lemmas whose normalized lemma equals canonical."""
    found: dict[str, set[str]] = defaultdict(set)
    for form in forms:
        try:
            analyses = list(engine.analyse(form))
        except Exception:
            continue
        for item in analyses:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            interp = item[2]
            if not isinstance(interp, (list, tuple)) or len(interp) < 3:
                continue
            raw_lemma = clean(interp[1])
            tag = clean(interp[2])
            if not raw_lemma or d3.normalize_lemma(raw_lemma).casefold() != canonical.casefold():
                continue
            if d3.tag_upos(tag) not in {"NOUN", "PROPN"}:
                continue
            found[raw_lemma].add(form)
    return found





def prng_objects(con: sqlite3.Connection, canonicals: set[str]) -> list[dict[str, Any]]:
    """Read PRNG families using only the verified production columns.

    The production database contains:
      geographic_names, metadata, name_forms

    Verified name_forms columns:
      id, geographic_name_id, prng_id, canonical_name, form, form_key,
      form_kind, object_type, name_status

    No assumed normalized canonical-name column is used. Rows are selected by
    the real `canonical_name` column and grouped by `geographic_name_id`.
    """
    actual_columns = {
        clean(row[1])
        for row in con.execute("PRAGMA table_info(name_forms)").fetchall()
    }
    required_columns = {
        "geographic_name_id", "prng_id", "canonical_name", "form",
        "form_kind", "object_type", "name_status",
    }
    missing = sorted(required_columns - actual_columns)
    if missing:
        raise RuntimeError(
            "Tabela name_forms nie ma wymaganych kolumn: " + ", ".join(missing)
            + "; dostępne: " + ", ".join(sorted(actual_columns))
        )

    output = []
    seen_families = set()
    for canonical in sorted(canonicals, key=str.casefold):
        rows = con.execute(
            """
            SELECT geographic_name_id,prng_id,canonical_name,form,form_kind,
                   object_type,name_status
            FROM name_forms
            WHERE canonical_name=?
            ORDER BY geographic_name_id,
                     CASE form_kind
                       WHEN 'canonical' THEN 0
                       WHEN 'genitive' THEN 1
                       WHEN 'locative' THEN 2
                       WHEN 'adjective' THEN 3
                       WHEN 'variant' THEN 4
                       ELSE 5
                     END,
                     id
            """,
            (canonical,),
        ).fetchall()

        # A canonical target normally originates from this same PRNG database,
        # so exact equality is the primary contract. If spelling/case differs,
        # inspect matching canonical names in Python with the shared normalizer,
        # still without assuming another database column.
        if not rows:
            candidates = con.execute(
                """
                SELECT geographic_name_id,prng_id,canonical_name,form,form_kind,
                       object_type,name_status
                FROM name_forms
                ORDER BY geographic_name_id,id
                """
            ).fetchall()
            rows = [row for row in candidates if norm(row[2]) == norm(canonical)]

        by_family = defaultdict(lambda: {
            "canonical": "", "forms": defaultdict(list),
            "object_type": "", "name_status": "", "prng_id": "",
        })
        for geographic_name_id, prng_id, canonical_name, form, form_kind, object_type, name_status in rows:
            family_key = (int(geographic_name_id), clean(prng_id), norm(canonical_name))
            data = by_family[family_key]
            data["object_id"] = int(geographic_name_id)
            data["prng_id"] = clean(prng_id)
            data["canonical"] = clean(canonical_name)
            data["object_type"] = clean(object_type)
            data["name_status"] = clean(name_status)
            value = clean(form)
            kind = clean(form_kind)
            if value and kind:
                data["forms"][kind].append(value)

        for family_key, data in by_family.items():
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
            data["forms"] = {
                kind: list(dict.fromkeys(values))
                for kind, values in data["forms"].items()
            }
            output.append(data)
    return output


def conservative_prng_instrumentals(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive only the missing instrumental from the PRNG nom/gen/loc triad.

    This is not an open-ended declension engine. A form is emitted only for
    three tightly validated signatures, each constrained by all three PRNG
    forms. The evidence is stored in the output for auditing.
    """
    canonical = clean(record.get("canonical"))
    forms = record.get("forms") or {}
    genitives = forms.get("genitive") or []
    locatives = forms.get("locative") or []
    output = []
    for genitive in genitives:
        for locative in locatives:
            generated = ""
            signature = ""
            # Cherson: Chersonia, Chersoniu -> Chersoniem
            if canonical.endswith("ń") and genitive.endswith("nia") and locative.endswith("niu"):
                if genitive[:-2].casefold() == locative[:-2].casefold():
                    generated = genitive[:-1] + "em"
                    signature = "N_NIA_NIU_NIEM"
            # Donieck/Lugansk/Smolensk: -ka/-ku -> -kiem; -ska/-sku -> -skiem
            elif genitive.endswith("a") and locative.endswith("u"):
                gen_stem = genitive[:-1]
                loc_stem = locative[:-1]
                if gen_stem.casefold() == loc_stem.casefold() and gen_stem.endswith(("k", "g")):
                    generated = gen_stem + "iem"
                    signature = "K_G_A_U_IEM"
            # Kijow: Kijowa, Kijowie -> Kijowem
            if not generated and canonical.endswith("ów") and genitive.endswith("owa") and locative.endswith("owie"):
                if genitive[:-1].casefold() == locative[:-2].casefold():
                    generated = genitive[:-1] + "em"
                    signature = "OW_OWA_OWIE_OWEM"
            if generated:
                output.append({
                    "form": generated,
                    "lemma": canonical,
                    "tag": "subst:sg:inst:m3",
                    "cases": {"inst"},
                    "source": "PRNG_TRIAD_SIGNATURE",
                    "signature": signature,
                    "canonical": canonical,
                    "genitive": genitive,
                    "locative": locative,
                    "object_id": record.get("object_id"),
                })
    unique = {}
    for item in output:
        key = (item["form"].casefold(), item["canonical"].casefold(), item["signature"])
        unique[key] = item
    return list(unique.values())


def json_safe(value: Any) -> Any:
    """Recursively convert runtime containers to deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item) for item in value), key=lambda item: str(item))
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value

def report_paths(paths: LemmaRepairPaths):
    return (
        paths.artifact_prefix.with_suffix(".residual_repairs.json"),
        paths.artifact_prefix.with_suffix(".residual_repairs.md"),
    )


def write_report(paths, payload):
    json_path, md_path = report_paths(paths)
    json_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Audyt i naprawy resztkowe lematów", "",
        f"Wersja: `{payload['version']}`  ",
        f"Reguły przed warstwą: `{payload['rules_before']}`  ",
        f"Nowe reguły AUTO: `{payload['promoted_rules']}`  ",
        f"Pozostałe obserwacje diagnostyczne: `{payload['residual_observations']}`", "",
        "## Liczniki", "",
    ]
    for key, value in sorted(payload["counters"].items()):
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Nowe reguły", ""]
    for rule in payload["rules"]:
        lines.append(
            f"- `{rule['orth']}` + `{rule['lemma']}` + `{rule['upos']}` + "
            f"`{rule.get('morph_from') or '*'}` -> `{rule['replacement']}` "
            f"({rule['classification']}; {rule['observed_count']} trafień)"
        )
    lines += ["", "## Nierozstrzygnięte resztki", ""]
    for row in payload["residual_rows"][:500]:
        lines.append(
            f"- `{row['orth']}` + `{row['lemma']}` + `{row['upos']}` + "
            f"`{row.get('morph') or '*'}`: {row['count']} trafień; {row['reason']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def augment_residual_repairs(paths: LemmaRepairPaths, reporter: Any = None) -> dict[str, Any]:
    auto_path = paths.decisions_auto()
    payload = json.loads(auto_path.read_text(encoding="utf-8"))
    rules = list(payload.get("rules") or [])
    rules_before = len(rules)
    exact, generic = build_rule_maps(rules)

    cross_families = defaultdict(lambda: {"targets": set(), "morphs": set(), "rules": []})
    accepted_geo_targets = set()
    for rule in rules:
        orth, lemma, upos, morph, required = rule_key(rule)
        target = d3.normalize_lemma(rule.get("replacement"))
        if upos == "PROPN" and orth and lemma and target:
            family = cross_families[(orth, lemma, upos)]
            family["targets"].add(target)
            if morph:
                family["morphs"].add(morph)
            family["rules"].append(rule)
        if target and clean(rule.get("required_ner_broad")).upper() == "LOC":
            accepted_geo_targets.add(target)
        if target and clean(rule.get("decision_source")).upper() == "PRNG":
            accepted_geo_targets.add(target)

    pf = pq.ParquetFile(paths.parquet)
    columns = set(pf.schema_arrow.names)
    morph_col = "full_postags" if "full_postags" in columns else ("postags" if "postags" in columns else None)
    ner_col = "ners" if "ners" in columns else ("ner" if "ner" in columns else None)
    read = ["tokens", "lemmas", "upostags"] + ([morph_col] if morph_col else []) + ([ner_col] if ner_col else [])
    residual = defaultdict(lambda: {"count": 0, "docs": set(), "ners": Counter(), "positions": [], "examples": []})
    doc_id = malformed = 0
    try:
        for batch in pf.iter_batches(batch_size=128, columns=read):
            data = batch.to_pydict()
            morph_rows = data[morph_col] if morph_col else [None] * batch.num_rows
            ner_rows = data[ner_col] if ner_col else [None] * batch.num_rows
            for tokens, lemmas, upos, morphs, ners in zip(data["tokens"], data["lemmas"], data["upostags"], morph_rows, ner_rows):
                ts, ls, us = map(as_list, (tokens, lemmas, upos))
                ms = as_list(morphs) if morph_col else [""] * len(ts)
                ns = as_list(ners) if ner_col else ["O"] * len(ts)
                if not (len(ts) == len(ls) == len(us) == len(ms) == len(ns)):
                    malformed += 1; doc_id += 1; continue
                for pos, (orth, lemma, upos_value, morph, ner) in enumerate(zip(ts, ls, us, ms, ns)):
                    orth, lemma = clean(orth), clean(lemma)
                    upos_value, morph = clean(upos_value).upper(), clean(morph)
                    if not orth or not lemma or not upos_value:
                        continue
                    if match_rule(exact, generic, orth, lemma, upos_value, morph, ner, doc_id, pos) is not None:
                        continue
                    row = residual[(orth, lemma, upos_value, morph)]
                    row["count"] += 1; row["docs"].add(doc_id); row["ners"][ner_broad(ner)] += 1
                    if len(row["positions"]) < 10000:
                        row["positions"].append({"doc_id": doc_id, "token_index": pos})
                    if len(row["examples"]) < 5:
                        left, right = max(0, pos - 7), min(len(ts), pos + 8)
                        row["examples"].append(" ".join(clean(x) for x in ts[left:right]))
                doc_id += 1
    finally:
        pf.close()

    engine, _version = d3.morfeusz_engine()
    counters, candidates, diagnostics = Counter(), [], []

    def add_rule(orth, lemma, upos, morph, target, classification, source, row, reasons, evidence=None):
        candidates.append({
            "orth": orth, "lemma": lemma, "upos": upos, "replacement": target,
            "reason": "Naprawa resztkowa po wirtualnym zastosowaniu wszystkich wcześniejszych reguł.",
            "status": "accept", "decision_bucket": "auto", "decision_reasons": reasons,
            "classification": classification, "decision_source": source,
            "morph_from": morph, "morph_to": "", "observed_count": int(row["count"]),
            "source_document_count": len(row["docs"]), "examples": list(row["examples"]),
            "ner_distribution": dict(row["ners"]), "required_ner_broad": "", **(evidence or {}),
        })

    # Cross-morph is deliberately restricted to proper names.
    for (orth, lemma, upos_value, morph), row in residual.items():
        if upos_value != "PROPN":
            continue
        family = cross_families.get((orth, lemma, upos_value))
        if not family or len(family["targets"]) != 1 or not family["morphs"]:
            continue
        target = next(iter(family["targets"]))
        analyses = morfeusz_analyses(engine, orth, upos_value)
        if analyses and analyses != {norm(target)}:
            counters["cross_morph_competing_morfeusz_lemmas"] += 1
            continue
        add_rule(orth, lemma, upos_value, "", target,
                 "SAFE_CROSS_MORPH_UNIQUE_LEMMA_REPAIR", "RESIDUAL_CROSS_MORPH", row,
                 ["PROPN_ONLY", "EXISTING_EXACT_RULES_ONE_TARGET", "NO_COMPETING_MORFEUSZ_LEMMA", "MORPH_FALLBACK"],
                 {"covered_source_morphs": sorted(family["morphs"]), "residual_morph": morph})
        counters["cross_morph_candidates"] += 1


    # Cross-NER fallback: an already accepted SGJP/PRNG rule has one target,
    # but an otherwise identical residual observation was blocked only by an
    # incorrect NER class. The residual rule keeps its exact morphology.
    trusted_sources = {"SGJP", "PRNG", "PRNG_GENERATED_PARADIGM"}
    trusted_classes = {"SAFE_NER_SGJP_REPAIR", "PRNG_SAFE_LEMMA_REPAIR", "SAFE_PRNG_PARADIGM_REPAIR"}
    cross_ner_families = defaultdict(lambda: {"targets": set(), "sources": set(), "rules": []})
    for accepted_rule in rules:
        required = clean(accepted_rule.get("required_ner_broad")).upper()
        source = clean(accepted_rule.get("decision_source")).upper()
        classification = clean(accepted_rule.get("classification")).upper()
        if not required:
            continue
        if source not in trusted_sources and classification not in trusted_classes:
            continue
        key = (
            clean(accepted_rule.get("orth")), clean(accepted_rule.get("lemma")),
            clean(accepted_rule.get("upos")).upper(),
        )
        target = d3.normalize_lemma(accepted_rule.get("replacement"))
        if not all(key) or not target:
            continue
        family = cross_ner_families[key]
        family["targets"].add(target)
        family["sources"].add(source or classification)
        family["rules"].append(accepted_rule)

    for (orth, lemma, upos_value, morph), row in residual.items():
        family = cross_ner_families.get((orth, lemma, upos_value))
        if not family or len(family["targets"]) != 1:
            continue
        target = next(iter(family["targets"]))
        confirmed = False
        evidence = []
        if any(source.startswith("PRNG") for source in family["sources"]):
            confirmed = True
            evidence.append("PRNG_EXACT_ACCEPTED_FAMILY")
        if any(source == "SGJP" for source in family["sources"]):
            analyses = morfeusz_analyses(engine, orth, upos_value)
            if analyses == {norm(target)}:
                confirmed = True
                evidence.append("UNIQUE_MORFEUSZ_TARGET")
        if not confirmed:
            counters["cross_ner_not_independently_confirmed"] += 1
            continue
        add_rule(
            orth, lemma, upos_value, morph, target,
            "SAFE_CROSS_NER_UNIQUE_LEMMA_REPAIR", "RESIDUAL_CROSS_NER", row,
            ["EXISTING_NER_SCOPED_RULE_ONE_TARGET", "EXACT_RESIDUAL_MORPH", *evidence, "NER_FALLBACK"],
            {
                "covered_required_ners": sorted({
                    clean(item.get("required_ner_broad")).upper()
                    for item in family["rules"] if clean(item.get("required_ner_broad"))
                }),
                "trusted_sources": sorted(family["sources"]),
                "residual_ner_distribution": dict(row["ners"]),
            },
        )
        counters["cross_ner_candidates"] += 1

    prng_path = resource_root() / "temp" / "prng_world.sqlite"
    generated_index = defaultdict(list)
    lexeme_audit = []
    prng_status = "missing"
    if prng_path.exists():
        con = sqlite3.connect(prng_path)
        try:
            records = prng_objects(con, accepted_geo_targets)
            prng_status = "ok"
            for record in records:
                canonical = record["canonical"]
                family_forms = []
                for kind in ("canonical", "genitive", "locative", "variant"):
                    family_forms.extend(record["forms"].get(kind, []))
                raw_lexemes = raw_generator_lexemes(engine, list(dict.fromkeys(family_forms)), canonical)
                generated = []
                for raw_lemma, found_from in raw_lexemes.items():
                    rows = generated_rows(engine, raw_lemma)
                    generated.extend(rows)
                    lexeme_audit.append({
                        "canonical": canonical, "object_id": record["object_id"],
                        "raw_lemma": raw_lemma, "found_from": sorted(found_from),
                        "generated_count": len(rows),
                    })
                if not generated:
                    generated.extend(conservative_prng_instrumentals(record))
                    if generated:
                        counters["prng_triad_instrumental_fallback"] += 1
                for item in generated:
                    parsed = d3.parse_tag(item["tag"])
                    cases = set(item.get("cases") or split_values(parsed.get("case")))
                    numbers = split_values(parsed.get("number")) if "cases" not in item else {"sg"}
                    if "sg" not in numbers or not cases:
                        continue
                    generated_index[item["form"].casefold()].append({
                        "canonical": canonical, "tag": item["tag"], "cases": sorted(cases),
                        "object_id": record["object_id"],
                        "generation_source": item.get("source", "SGJP_RAW_LEXEME"),
                        "signature": item.get("signature", ""),
                        "prng_genitive": item.get("genitive", ""),
                        "prng_locative": item.get("locative", ""),
                    })
        finally:
            con.close()

    for (orth, lemma, upos_value, morph), row in residual.items():
        if upos_value != "PROPN" or not orth[:1].isupper():
            continue
        hits = generated_index.get(orth.casefold(), [])
        if not hits:
            continue
        observed = d3.parse_tag(morph) if morph else {}
        observed_cases = split_values(observed.get("case"))
        if observed_cases:
            hits = [hit for hit in hits if case_values_overlap(hit.get("cases"), observed_cases)]
        unique = {norm(hit["canonical"]): hit["canonical"] for hit in hits}
        if len(unique) != 1:
            counters["prng_generated_multiple_canonicals"] += 1
            diagnostics.append({"orth": orth, "lemma": lemma, "upos": upos_value, "morph": morph,
                                "count": row["count"], "reason": "PRNG_MULTIPLE_CANONICALS",
                                "canonical_candidates": sorted(unique.values(), key=str.casefold)})
            continue
        target = next(iter(unique.values()))
        if norm(lemma) == norm(target):
            continue
        add_rule(orth, lemma, upos_value, morph, target,
                 "SAFE_PRNG_PARADIGM_REPAIR", "PRNG_GENERATED_PARADIGM", row,
                 ["PRNG_CANONICAL_CONFIRMED", "EXACT_GENERATED_FORM", "CASE_MATCH", "UNIQUE_CANONICAL"],
                 {"prng_canonical": target, "generated_matches": hits})
        counters["prng_paradigm_candidates"] += 1

    existing = {rule_key(rule): rule for rule in rules}
    promoted, conflicts = [], []
    for rule in candidates:
        key = rule_key(rule); previous = existing.get(key)
        if previous is not None:
            if norm(previous.get("replacement")) != norm(rule.get("replacement")):
                conflicts.append({"orth": rule["orth"], "lemma": rule["lemma"], "upos": rule["upos"],
                                  "existing_target": previous.get("replacement"), "candidate_target": rule.get("replacement")})
            continue
        existing[key] = rule; promoted.append(rule)

    payload["rules"] = sorted(existing.values(), key=lambda r: (-int(r.get("observed_count") or 0), clean(r.get("orth")).casefold(), clean(r.get("lemma")).casefold()))
    payload.setdefault("settings", {})["residual_repairs"] = {
        "enabled": True, "version": "1.1.0", "cross_morph": "PROPN only",
        "prng_paradigm": "raw SGJP lexeme from PRNG family; conservative nom/gen/loc instrumental fallback",
        "prng_status": prng_status, "prng_path": str(prng_path),
    }
    payload["residual_promoted_rules"] = len(promoted)
    auto_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def is_covered(key):
        orth, lemma, upos_value, morph = key
        for rule in promoted:
            if rule["orth"] == orth and rule["lemma"] == lemma and rule["upos"] == upos_value:
                if not clean(rule.get("morph_from")) or clean(rule.get("morph_from")) == morph:
                    return True
        return False

    residual_rows = list(diagnostics)
    for key, row in residual.items():
        orth, lemma, upos_value, morph = key
        if is_covered(key):
            continue
        residual_rows.append({"orth": orth, "lemma": lemma, "upos": upos_value, "morph": morph,
                              "count": row["count"], "documents": len(row["docs"]),
                              "ner_distribution": dict(row["ners"]), "examples": row["examples"],
                              "reason": "RESIDUAL_UNRESOLVED"})
    residual_rows.sort(key=lambda r: (-int(r.get("count") or 0), clean(r.get("orth")).casefold()))
    report = {"schema_version": 1, "version": "1.1.0", "rules_before": rules_before,
              "rules_after": len(payload["rules"]), "promoted_rules": len(promoted),
              "conflicts": len(conflicts), "malformed_documents": malformed,
              "prng_status": prng_status, "residual_observations": len(residual_rows),
              "counters": dict(counters), "rules": promoted, "conflict_rows": conflicts,
              "generator_lexeme_audit": lexeme_audit, "residual_rows": residual_rows}
    write_report(paths, report)
    if reporter:
        reporter.status(f"Naprawy resztkowe: dodano {len(promoted)} reguł...")
    return {"residual_promoted_rules": len(promoted), "residual_conflicts": len(conflicts),
            "residual_unresolved": len(residual_rows), "auto_rules": len(payload["rules"])}
