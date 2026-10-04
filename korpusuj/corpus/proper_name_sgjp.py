# -*- coding: utf-8 -*-
"""Eksperymentalny audyt lematyzacji nazw własnych v0.2.

Narzędzie diagnostyczne. Nie modyfikuje Parquetu ani indeksów. Zachowuje
oryginalne etykiety NER i dodaje wyłącznie pomocniczą klasyfikację leksykalną.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

VERSION = "0.2.0-experimental"
LOCATION_LABELS = {
    "LOC", "GPE", "LOCATION", "PLACE", "PLACENAME", "GEOG", "GEONAME",
    "GEOGNAME", "FAC", "FACILITY",
}
PERSON_LABELS = {"PER", "PERSON", "PERSNAME"}
ORG_LABELS = {"ORG", "ORGANIZATION", "ORGNAME"}
TARGET_UPOS = {"PROPN", "NOUN"}
SGJP_TO_UPOS = {
    "subst": "NOUN", "depr": "NOUN", "adj": "ADJ", "adja": "ADJ",
    "adjp": "ADJ", "adjc": "ADJ", "adv": "ADV", "fin": "VERB",
    "bedzie": "AUX", "aglt": "AUX", "praet": "VERB", "impt": "VERB",
    "imps": "VERB", "inf": "VERB", "pcon": "VERB", "pant": "VERB",
    "winien": "VERB", "pred": "VERB", "pact": "ADJ", "ppas": "ADJ",
    "num": "NUM", "ppron12": "PRON", "ppron3": "PRON", "prep": "ADP",
    "conj": "CCONJ", "comp": "SCONJ", "qub": "PART", "interj": "INTJ",
    "interp": "PUNCT", "brev": "X", "burk": "X", "xxx": "X", "ign": "X",
}
TAG_LAYOUTS = {
    "subst": ("number", "case", "gender"),
    "depr": ("number", "case", "gender"),
    "adj": ("number", "case", "gender", "degree"),
}
TECHNICAL_SGJP_KINDS = {"ign", "xxx", "burk", "interp", "brev"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalized_lemma(value: Any) -> str:
    """Usuń cały techniczny sufiks SGJP od pierwszego dwukropka."""
    return clean(value).split(":", 1)[0]


def lemma_key(value: Any) -> str:
    return normalized_lemma(value).casefold()


def ner_parts(value: Any) -> tuple[str, str, str]:
    """Zwróć: surową etykietę, prefiks BIO/BILOU/IOBES i klasę ogólną."""
    raw = clean(value)
    upper = raw.upper()
    if upper in {"", "O", "_"}:
        return raw, "O", "O"
    match = re.match(r"^([BILOUSE])[-_](.+)$", upper)
    prefix, core = (match.group(1), match.group(2)) if match else ("", upper)
    if core in LOCATION_LABELS:
        broad = "LOC"
    elif core in PERSON_LABELS:
        broad = "PER"
    elif core in ORG_LABELS:
        broad = "ORG"
    else:
        broad = core
    return raw, prefix, broad


def starts_new_span(previous: tuple[str, str, str] | None, current: tuple[str, str, str]) -> bool:
    raw, prefix, broad = current
    if broad == "O":
        return False
    if previous is None:
        return True
    _praw, _pprefix, pbroad = previous
    if prefix in {"B", "U", "S"}:
        return True
    if broad != pbroad:
        return True
    if prefix in {"I", "L", "E"}:
        return False
    # Bez BIO rekonstruujemy span jako maksymalny ciąg tej samej klasy.
    return False


def span_ranges(ners: list[Any]) -> tuple[list[tuple[int, int, str]], list[int | None]]:
    spans: list[tuple[int, int, str]] = []
    token_to_span: list[int | None] = [None] * len(ners)
    start: int | None = None
    previous: tuple[str, str, str] | None = None
    current_broad = "O"
    for index, value in enumerate(ners):
        parts = ner_parts(value)
        if parts[2] == "O":
            if start is not None:
                spans.append((start, index, current_broad))
            start, previous, current_broad = None, None, "O"
            continue
        if start is None or starts_new_span(previous, parts):
            if start is not None:
                spans.append((start, index, current_broad))
            start, current_broad = index, parts[2]
        previous = parts
        if parts[1] in {"U", "S"}:
            spans.append((start, index + 1, current_broad))
            start, previous, current_broad = None, None, "O"
        elif parts[1] in {"L", "E"} and start is not None:
            spans.append((start, index + 1, current_broad))
            start, previous, current_broad = None, None, "O"
    if start is not None:
        spans.append((start, len(ners), current_broad))
    for span_id, (begin, end, _broad) in enumerate(spans):
        for index in range(begin, end):
            token_to_span[index] = span_id
    return spans, token_to_span


def tag_upos(tag: str) -> str:
    return clean(SGJP_TO_UPOS.get(clean(tag).split(":", 1)[0].casefold()))


def parse_tag(tag: str) -> dict[str, str]:
    parts = [part.casefold() for part in clean(tag).split(":") if part]
    if not parts:
        return {"kind": "unknown", "raw": clean(tag)}
    output = {"kind": parts[0], "raw": clean(tag)}
    for name, value in zip(TAG_LAYOUTS.get(parts[0], ()), parts[1:]):
        output[name] = value
    return output


def value_set(value: Any) -> set[str]:
    return {part for part in clean(value).casefold().split(".") if part}


def tags_compatible(observed: str, sgjp: str) -> tuple[bool, dict[str, Any]]:
    left, right = parse_tag(observed), parse_tag(sgjp)
    shared = sorted((set(left) & set(right)) - {"kind", "raw"})
    matches, mismatches = {}, {}
    for feature in shared:
        overlap = value_set(left[feature]) & value_set(right[feature])
        if overlap:
            matches[feature] = sorted(overlap)
        else:
            mismatches[feature] = {"observed": left[feature], "sgjp": right[feature]}
    return bool(shared) and not mismatches, {
        "observed": left, "sgjp": right, "shared": shared,
        "matches": matches, "mismatches": mismatches,
    }


def morfeusz_engine() -> Any:
    try:
        import morfeusz2
    except ImportError as exc:
        raise SystemExit("Brak morfeusz2 w aktywnym środowisku.") from exc
    return morfeusz2.Morfeusz()


def analyses_for(engine: Any, orth: str) -> tuple[str, list[dict[str, str]]]:
    try:
        raw = list(engine.analyse(orth))
    except Exception as exc:
        return "ERROR", [{"error": f"{type(exc).__name__}: {exc}"}]
    output, seen, mismatch = [], set(), False
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        interp = item[2]
        if not isinstance(interp, (list, tuple)) or len(interp) < 3:
            continue
        try:
            mismatch |= int(item[0]) != 0 or int(item[1]) != 1
        except Exception:
            pass
        raw_lemma = clean(interp[1])
        tag = clean(interp[2])
        analysis = {
            "orth": clean(interp[0]),
            "raw_lemma": raw_lemma,
            "normalized_lemma": normalized_lemma(raw_lemma),
            "normalized_key": lemma_key(raw_lemma),
            "tag": tag,
            "upos": tag_upos(tag),
        }
        key = tuple(analysis.values())
        if key not in seen:
            seen.add(key)
            output.append(analysis)
    if not output:
        return "NO_ANALYSIS", []

    # Sprawdz profile rodzaju dla formy haslowej. Ten sam napis lematu moze
    # odpowiadac kilku leksemom SGJP, np. nazwie geograficznej i nazwisku.
    profile_cache: dict[str, list[str]] = {}
    for analysis in output:
        key = analysis.get("normalized_key", "")
        lemma = analysis.get("normalized_lemma", "")
        if not key or key in profile_cache:
            continue
        profiles = set()
        try:
            lemma_raw = list(engine.analyse(lemma))
        except Exception:
            lemma_raw = []
        for lemma_item in lemma_raw:
            if not isinstance(lemma_item, (list, tuple)) or len(lemma_item) < 3:
                continue
            lemma_interp = lemma_item[2]
            if not isinstance(lemma_interp, (list, tuple)) or len(lemma_interp) < 3:
                continue
            raw_lemma = clean(lemma_interp[1])
            tag = clean(lemma_interp[2])
            parsed = parse_tag(tag)
            if lemma_key(raw_lemma) != key:
                continue
            if parsed.get("kind") in TECHNICAL_SGJP_KINDS:
                continue
            gender = parsed.get("gender", "")
            if gender:
                profiles.add(gender)
        profile_cache[key] = sorted(profiles)
    for analysis in output:
        analysis["lemma_gender_profiles"] = profile_cache.get(analysis.get("normalized_key", ""), [])

    return ("TOKENIZATION_MISMATCH" if mismatch else "OK"), output


@dataclass
class Observation:
    orth: str
    lemma: str
    upos: str
    morph: str
    ner_raw: str
    ner_broad: str
    span_kind: str
    span_text: str
    span_length: int
    count: int = 0
    documents: set[int] = field(default_factory=set)
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceName:
    canonical: str
    kind: str = ""
    source: str = ""
    genitive: str = ""


def load_reference(path: Path | None) -> tuple[dict[str, set[ReferenceName]], int]:
    index: dict[str, set[ReferenceName]] = defaultdict(set)
    if path is None:
        return index, 0
    rows = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split("\t")]
        if not cells or not cells[0]:
            continue
        ref = ReferenceName(
            canonical=cells[0], kind=cells[1] if len(cells) > 1 else "",
            source=cells[2] if len(cells) > 2 else "",
            genitive=cells[3] if len(cells) > 3 else "",
        )
        index[ref.canonical.casefold()].add(ref)
        if ref.genitive:
            index[ref.genitive.casefold()].add(ref)
        rows += 1
    return index, rows


def aligned_columns(names: list[str]) -> dict[str, str]:
    aliases = {
        "tokens": ("tokens",), "lemmas": ("lemmas",),
        "upos": ("upostags", "upos"),
        "morph": ("full_postags", "postags", "xpostags"),
        "ner": ("ners", "ner"),
    }
    found: dict[str, str] = {}
    for logical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in names:
                found[logical] = candidate
                break
    missing = {"tokens", "lemmas", "upos", "ner"} - set(found)
    if missing:
        raise SystemExit("Brak wymaganych kolumn: " + ", ".join(sorted(missing)))
    return found


def context(tokens: list[Any], index: int, width: int) -> str:
    start, end = max(0, index - width), min(len(tokens), index + width + 1)
    values = [clean(token).replace("\n", "\\n") for token in tokens[start:end]]
    values[index - start] = "[" + values[index - start] + "]"
    return " ".join(values)


def collect(path: Path, batch_size: int, width: int, max_examples: int) -> tuple[dict[tuple[str, ...], Observation], Counter, Counter, int, int]:
    parquet = pq.ParquetFile(path)
    mapping = aligned_columns(list(parquet.schema_arrow.names))
    columns = list(dict.fromkeys(mapping.values()))
    observations: dict[tuple[str, ...], Observation] = {}
    ner_counts: Counter[str] = Counter()
    span_counts: Counter[str] = Counter()
    document_id = 0
    token_total = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pydict()
            for row in range(batch.num_rows):
                tokens = list(data[mapping["tokens"]][row] or [])
                lemmas = list(data[mapping["lemmas"]][row] or [])
                upos = list(data[mapping["upos"]][row] or [])
                ners = list(data[mapping["ner"]][row] or [])
                morph = list(data[mapping["morph"]][row] or []) if "morph" in mapping else [""] * len(tokens)
                if len({len(tokens), len(lemmas), len(upos), len(ners), len(morph)}) != 1:
                    document_id += 1
                    continue
                token_total += len(tokens)
                spans, token_to_span = span_ranges(ners)
                for begin, end, broad in spans:
                    span_counts[f"{broad}:{'SINGLE' if end - begin == 1 else 'MULTI'}"] += 1
                for index, (orth, lemma, pos, tag, ner_value) in enumerate(zip(tokens, lemmas, upos, morph, ners)):
                    raw_label, _prefix, broad = ner_parts(ner_value)
                    ner_counts[raw_label or "O"] += 1
                    span_id = token_to_span[index]
                    # SGJP obsluguje wszystkie rzeczywiste kategorie NER.
                    # Tokeny O nie maja span_id, wiec nadal sa pomijane.
                    # Ograniczenie geograficzne pozostaje w adapterze PRNG.
                    if span_id is None:
                        continue
                    begin, end, _ = spans[span_id]
                    span_length = end - begin
                    span_kind = "SINGLE_TOKEN_ENTITY" if span_length == 1 else "MULTI_TOKEN_ENTITY_MEMBER"
                    span_text = " ".join(clean(value) for value in tokens[begin:end])
                    orth_s, lemma_s, upos_s, morph_s = clean(orth), clean(lemma), clean(pos).upper(), clean(tag)
                    if not orth_s or not lemma_s or upos_s not in TARGET_UPOS:
                        continue
                    key = (orth_s, lemma_s, upos_s, morph_s, raw_label, span_kind, span_text)
                    item = observations.get(key)
                    if item is None:
                        item = Observation(orth_s, lemma_s, upos_s, morph_s, raw_label, broad, span_kind, span_text, span_length)
                        observations[key] = item
                    item.count += 1
                    item.documents.add(document_id)
                    if len(item.examples) < max_examples:
                        item.examples.append(context(tokens, index, width))
                document_id += 1
    finally:
        parquet.close()
    return observations, ner_counts, span_counts, document_id, token_total


def analytical_scope(obs: Observation) -> str:
    """Techniczny zakres raportowy; nie jest nowa klasyfikacja NER."""
    if obs.span_kind == "MULTI_TOKEN_ENTITY_MEMBER":
        return "MULTI_TOKEN_NAME_MEMBER"
    return "SINGLE_TOKEN_NAMED_ENTITY"


def reconstructed_gender_tag(observed_tag: str, sgjp_tag: str) -> str:
    """Zachowaj kontekstowa liczbe i przypadek, podmien rodzaj z SGJP."""
    observed_parts = [part for part in clean(observed_tag).split(":") if part]
    sgjp_parts = [part for part in clean(sgjp_tag).split(":") if part]
    observed = parse_tag(observed_tag)
    sgjp = parse_tag(sgjp_tag)
    if not observed_parts or not sgjp_parts:
        return ""
    kind = observed.get("kind", observed_parts[0])
    number = observed.get("number", "")
    case = observed.get("case", "")
    gender = sgjp.get("gender", "")
    if not (kind and number and case and gender):
        return ""
    layout = TAG_LAYOUTS.get(sgjp.get("kind", ""), ())
    try:
        gender_position = layout.index("gender") + 1
    except ValueError:
        return ""
    extras = sgjp_parts[gender_position + 1:]
    return ":".join([kind, number, case, gender, *extras])


def classify(obs: Observation, analyses: list[dict[str, str]], status: str, reference: dict[str, set[ReferenceName]]) -> dict[str, Any]:
    current_key = lemma_key(obs.lemma)
    # Analizy techniczne Morfeusza nie sa dowodem leksykalnym. Zachowujemy je
    # diagnostycznie w `analyses`, ale nie budujemy z nich celu korekty.
    lexical_analyses = [
        analysis for analysis in analyses
        if parse_tag(analysis.get("tag", "")).get("kind") not in TECHNICAL_SGJP_KINDS
        and analysis.get("upos") not in {"", "X", "PUNCT"}
        and analysis.get("normalized_key")
    ]
    target_variants: dict[str, set[str]] = defaultdict(set)
    for analysis in lexical_analyses:
        target_variants[analysis["normalized_key"]].add(analysis["normalized_lemma"])

    def preferred_variant(key: str, variants: set[str]) -> str:
        ordered = sorted(variants, key=lambda value: (value.casefold(), value))
        if obs.orth[:1].isupper():
            capitalized = [value for value in ordered if value[:1].isupper()]
            if capitalized:
                return capitalized[0]
        return ordered[0]

    all_targets = {
        key: preferred_variant(key, variants)
        for key, variants in target_variants.items()
    }
    current_confirmed = current_key in all_targets
    compatible_variants: dict[str, set[str]] = defaultdict(set)
    compatible: dict[str, str] = {}
    comparisons = []
    same_upos = [analysis for analysis in lexical_analyses if analysis.get("upos") in {obs.upos, "NOUN"}]
    for analysis in same_upos:
        ok, details = tags_compatible(obs.morph, analysis.get("tag", "")) if obs.morph else (False, {})
        comparisons.append({
            "raw_lemma": analysis.get("raw_lemma"),
            "normalized_lemma": analysis.get("normalized_lemma"),
            "tag": analysis.get("tag"), "compatible": ok, "details": details,
        })
        if ok:
            compatible_variants[analysis["normalized_key"]].add(analysis["normalized_lemma"])
    compatible = {
        key: preferred_variant(key, variants)
        for key, variants in compatible_variants.items()
    }
    # Bezpieczny szum rodzaju: liczba i przypadek musza byc zgodne, a jedyna
    # niezgodna cecha wspolna moze byc rodzaj. Zachowujemy konkretny tag SGJP,
    # aby raportowac jawna propozycje morph_from -> morph_to.
    gender_noise_candidates = []
    for comparison in comparisons:
        details = comparison.get("details") or {}
        mismatches = details.get("mismatches") or {}
        matches = details.get("matches") or {}
        if set(mismatches) == {"gender"} and {"number", "case"}.issubset(matches):
            gender_noise_candidates.append(comparison)

    refs = sorted({
        ref.canonical
        for key in {obs.orth.casefold(), current_key, *all_targets.keys()}
        for ref in reference.get(key, set())
    })
    target = ""
    morph_repair = None
    decision_reasons = []
    if obs.span_kind == "MULTI_TOKEN_ENTITY_MEMBER":
        category, decision = "MULTI_TOKEN_NAME_MEMBER", "SEPARATE_SECTION"
    elif status == "TOKENIZATION_MISMATCH":
        category, decision = "TOKENIZATION_MISMATCH", "REJECT"
    elif not lexical_analyses:
        category, decision = "EXTERNAL_REGISTRY_CANDIDATE", "EXTERNAL_REGISTRY_CANDIDATE"
    elif current_confirmed:
        exact = any(a.get("normalized_lemma") == obs.lemma for a in analyses)
        category = "CURRENT_LEMMA_CONFIRMED" if exact else "CURRENT_LEMMA_CONFIRMED_CASE_INSENSITIVE"
        decision = "KEEP"
    elif len(all_targets) > 1:
        category, decision = "CROSS_LEMMA_HOMOGRAPHY", "REVIEW"
    elif len(compatible) == 1:
        target = next(iter(compatible.values()))
        # Wielkoliterowy token z pelnego spanu NER nie moze zostac
        # automatycznie sprowadzony do maloliterowego homografu pospolitego,
        # np. Kataru -> katar, Bostonie -> boston, Telegramie -> telegram.
        if obs.orth[:1].isupper() and target[:1].islower():
            category = "PROPER_NAME_CASE_CONFLICT"
            decision = "EXTERNAL_REGISTRY_CANDIDATE"
        else:
            category = "SAFE_NER_SGJP_REPAIR"
            decision = "AUTO_CANDIDATE"
    elif len(all_targets) == 1 and gender_noise_candidates:
        target_key = next(iter(all_targets))
        target = all_targets[target_key]
        matching_gender_candidates = [
            item for item in gender_noise_candidates
            if lemma_key(item.get("normalized_lemma", "")) == target_key
        ]
        morph_targets = sorted({item.get("tag", "") for item in matching_gender_candidates if item.get("tag")})
        lemma_gender_profiles = sorted({
            gender
            for analysis in lexical_analyses
            if analysis.get("normalized_key") == target_key
            for gender in analysis.get("lemma_gender_profiles", [])
            if gender
        })
        candidate_gender_profiles = sorted({
            parse_tag(item.get("tag", "")).get("gender", "")
            for item in matching_gender_candidates
            if parse_tag(item.get("tag", "")).get("gender", "")
        })
        effective_gender_profiles = lemma_gender_profiles or candidate_gender_profiles

        if obs.orth[:1].isupper() and target[:1].islower():
            category = "PROPER_NAME_CASE_CONFLICT"
            decision = "EXTERNAL_REGISTRY_CANDIDATE"
            decision_reasons = ["CAPITALIZED_NER_WITH_LOWERCASE_SGJP_TARGET"]
        elif len(effective_gender_profiles) > 1:
            category = "AMBIGUOUS_GENDER_REPAIR"
            decision = "REVIEW"
            decision_reasons = ["COMPETING_SGJP_GENDER_PROFILES"]
        elif len(morph_targets) == 1:
            observed_tag = parse_tag(obs.morph)
            source_target_tag = morph_targets[0]
            target_tag = parse_tag(source_target_tag)
            reconstructed_tag = reconstructed_gender_tag(obs.morph, source_target_tag)
            if not reconstructed_tag:
                category = "AMBIGUOUS_GENDER_REPAIR"
                decision = "REVIEW"
                decision_reasons = ["CANNOT_RECONSTRUCT_CONTEXTUAL_MORPH_TAG"]
            else:
                morph_repair = {
                    "from": obs.morph,
                    "to": reconstructed_tag,
                    "sgjp_source_tag": source_target_tag,
                    "matched_features": ["number", "case"],
                    "changed_features": {
                        "gender": {
                            "from": observed_tag.get("gender", ""),
                            "to": target_tag.get("gender", ""),
                        }
                    },
                }
                category = "SAFE_NER_SGJP_LEMMA_AND_GENDER_REPAIR"
                decision = "AUTO_CANDIDATE"
                decision_reasons = [
                    "UNIQUE_SGJP_LEMMA",
                    "NUMBER_AND_CASE_MATCH",
                    "TOLERATED_MODEL_GENDER_MISMATCH",
                    "CONTEXTUAL_MORPH_TAG_RECONSTRUCTED",
                ]
        else:
            category, decision = "AMBIGUOUS_GENDER_REPAIR", "REVIEW"
            decision_reasons = ["MULTIPLE_MATCHING_SGJP_MORPH_TAGS"]
    elif len(all_targets) == 1:
        target = next(iter(all_targets.values()))
        category, decision = "UNIQUE_LEMMA_MORPH_CONFLICT", "REVIEW"
    else:
        category, decision = "EXTERNAL_REGISTRY_CANDIDATE", "EXTERNAL_REGISTRY_CANDIDATE"

    raw_targets = sorted({a.get("raw_lemma", "") for a in analyses if a.get("raw_lemma")})
    normalized_targets = sorted(all_targets.values(), key=str.casefold)
    scope = analytical_scope(obs)
    if target and refs and target.casefold() not in {value.casefold() for value in refs}:
        category, decision = "REFERENCE_CONFLICT", "REVIEW"
    elif target and target.casefold() in {value.casefold() for value in refs}:
        category = "REGISTRY_CONFIRMED_" + category
    return {
        "orth": obs.orth, "model_lemma": obs.lemma, "model_lemma_key": current_key,
        "upos": obs.upos, "morph": obs.morph,
        "ner_raw": obs.ner_raw, "ner_broad": obs.ner_broad,
        "span_kind": obs.span_kind, "span_text": obs.span_text, "span_length": obs.span_length,
        "analytical_scope": scope,
        "token_count": obs.count, "document_count": len(obs.documents),
        "status": status, "category": category, "decision": decision, "target": target,
        "decision_reasons": decision_reasons,
        "morph_repair": morph_repair,
        "raw_sgjp_lemmas": raw_targets,
        "normalized_sgjp_lemmas": normalized_targets,
        "lexical_sgjp_lemmas": sorted({a["normalized_lemma"] for a in lexical_analyses}, key=lambda value: (value.casefold(), value)),
        "sgjp_lemma_variants": {
            key: sorted(values, key=lambda value: (value.casefold(), value))
            for key, values in sorted(target_variants.items())
        },
        "preferred_sgjp_lemmas": dict(sorted(all_targets.items())),
        "technical_sgjp_analyses": [
            a for a in analyses
            if parse_tag(a.get("tag", "")).get("kind") in TECHNICAL_SGJP_KINDS
            or a.get("upos") in {"", "X", "PUNCT"}
        ],
        "lemma_gender_profiles": {
            key: sorted({
                gender
                for analysis in lexical_analyses
                if analysis.get("normalized_key") == key
                for gender in analysis.get("lemma_gender_profiles", [])
                if gender
            })
            for key in sorted(all_targets)
        },
        "compatible_lemmas": sorted(compatible.values(), key=str.casefold),
        "reference_hits": refs, "analyses": analyses, "comparisons": comparisons,
        "examples": obs.examples,
    }


def aggregate_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agreguj po unikalnej parze lemat modelu -> cel, niezależnie od tagu."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["decision"] != "AUTO_CANDIDATE" or not row["target"]:
            continue
        key = (row["model_lemma_key"], lemma_key(row["target"]))
        group = groups.get(key)
        if group is None:
            group = {
                "model_lemma": row["model_lemma"], "target": row["target"],
                "analytical_scopes": set(), "categories": set(), "forms": Counter(),
                "morph_repairs": Counter(), "token_count": 0,
                "morph_token_count": 0, "documents_upper_bound": 0, "observations": 0,
            }
            groups[key] = group
        group["analytical_scopes"].add(row["analytical_scope"])
        group["categories"].add(row["category"])
        group["forms"][row["orth"]] += row["token_count"]
        group["token_count"] += row["token_count"]
        if row.get("morph_repair"):
            repair = row["morph_repair"]
            repair_key = repair["from"] + " -> " + repair["to"]
            group["morph_repairs"][repair_key] += row["token_count"]
            group["morph_token_count"] += row["token_count"]
        group["documents_upper_bound"] += row["document_count"]
        group["observations"] += 1
    output = []
    for group in groups.values():
        output.append({
            "model_lemma": group["model_lemma"], "target": group["target"],
            "analytical_scopes": sorted(group["analytical_scopes"]),
            "categories": sorted(group["categories"]),
            "forms": dict(group["forms"].most_common()),
            "morph_repairs": dict(group["morph_repairs"].most_common()),
            "token_count": group["token_count"],
            "morph_token_count": group["morph_token_count"],
            "documents_upper_bound": group["documents_upper_bound"],
            "observations": group["observations"],
        })
    return sorted(output, key=lambda item: (-item["token_count"], item["model_lemma"].casefold(), item["target"].casefold()))


def escape(value: Any) -> str:
    return clean(value).replace("|", "\\|").replace("\n", " ")


def add_rows_table(lines: list[str], title: str, rows: list[dict[str, Any]], limit: int = 300) -> None:
    lines += ["", f"## {title}", "", "| Forma | Lemat modelu | Cel | NER | Zakres analityczny | Span | Tokeny | Dokumenty | Kategoria |", "|---|---|---|---|---|---|---:|---:|---|"]
    for row in sorted(rows, key=lambda item: (-item["token_count"], item["orth"].casefold()))[:limit]:
        lines.append("| " + " | ".join([
            escape(row["orth"]), escape(row["model_lemma"]), escape(row["target"]),
            escape(row["ner_raw"]), escape(row["analytical_scope"]), escape(row["span_text"]),
            f"{row['token_count']:,}", f"{row['document_count']:,}", escape(row["category"]),
        ]) + " |")


def write_report(path: Path, parquet: Path, rows: list[dict[str, Any]], rules: list[dict[str, Any]], ner_counts: Counter, span_counts: Counter, documents: int, tokens: int, reference_rows: int) -> None:
    decisions = Counter(row["decision"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    single = [row for row in rows if row["span_kind"] == "SINGLE_TOKEN_ENTITY"]
    multi = [row for row in rows if row["span_kind"] == "MULTI_TOKEN_ENTITY_MEMBER"]
    external = [row for row in rows if row["decision"] == "EXTERNAL_REGISTRY_CANDIDATE"]
    changed_tokens = sum(rule["token_count"] for rule in rules)
    changed_morph_tokens = sum(rule.get("morph_token_count", 0) for rule in rules)
    gender_repair_rules = sum(bool(rule.get("morph_repairs")) for rule in rules)
    lines = [
        "# Eksperymentalny audyt lematyzacji nazw własnych v0.2", "",
        f"**Wersja:** `{VERSION}`  ",
        f"**Wygenerowano:** `{datetime.now(timezone.utc).isoformat()}`  ",
        f"**Parquet:** `{parquet}`", "",
        "> Eksperyment diagnostyczny. Nie zmienia korpusu ani etykiet NER.", "",
        "## Podsumowanie", "",
        f"- Dokumenty: **{documents:,}**",
        f"- Tokeny korpusu: **{tokens:,}**",
        f"- Obserwacje po progach: **{len(rows):,}**",
        f"- Pełne encje jednotokenowe: **{len(single):,}**",
        f"- Człony nazw wieloczłonowych: **{len(multi):,}**",
        f"- Unikalne reguły lemat modelu → cel: **{len(rules):,}**",
        f"- Tokeny ze zmianą lematu: **{changed_tokens:,}**",
        f"- Tokeny także ze zmianą tagu morfologicznego: **{changed_morph_tokens:,}**",
        f"- Reguły zawierające korektę rodzaju: **{gender_repair_rules:,}**",
        f"- Kandydaci do rejestrów zewnętrznych: **{len(external):,}**",
        f"- Wiersze lokalnego TSV: **{reference_rows:,}**", "",
        "## Decyzje", "",
    ]
    for key, value in decisions.most_common():
        lines.append(f"- `{key}`: **{value:,}**")
    lines += ["", "## Kategorie", ""]
    for key, value in categories.most_common():
        lines.append(f"- `{key}`: **{value:,}**")
    lines += ["", "## Spany NER", ""]
    for key, value in span_counts.most_common():
        lines.append(f"- `{key}`: **{value:,}**")

    lines += ["", "## Reguły zagregowane według lemat modelu → cel", "", "| Lemat modelu | Cel | Zakresy analityczne | Formy | Korekty morfologiczne | Tokeny lematu | Tokeny morfologii | Obserwacje |", "|---|---|---|---|---|---:|---:|---:|"]
    for rule in rules[:500]:
        forms = ", ".join(f"{form} ({count})" for form, count in list(rule["forms"].items())[:8])
        morph_repairs = ", ".join(f"{repair} ({count})" for repair, count in list(rule.get("morph_repairs", {}).items())[:8])
        lines.append("| " + " | ".join([
            escape(rule["model_lemma"]), escape(rule["target"]),
            escape(", ".join(rule["analytical_scopes"])), escape(forms), escape(morph_repairs),
            f"{rule['token_count']:,}", f"{rule.get('morph_token_count', 0):,}", f"{rule['observations']:,}",
        ]) + " |")

    add_rows_table(lines, "Pełne nazwy jednotokenowe", [row for row in single if row["decision"] == "AUTO_CANDIDATE"])
    add_rows_table(lines, "Człony nazw wieloczłonowych, bez automatycznej decyzji", multi)
    add_rows_table(lines, "EXTERNAL_REGISTRY_CANDIDATE", external)
    add_rows_table(lines, "Pozostałe przypadki do przeglądu", [row for row in rows if row["decision"] == "REVIEW"])

    lines += ["", "## Interpretacja", "",
        "- Oryginalne etykiety NER są zachowane w polu `ner_raw`; eksperyment nie tworzy własnej klasyfikacji semantycznej nazw.",
        "- Jednoznaczne korekty SGJP pozostają `AUTO_CANDIDATE` niezależnie od tego, czy dotyczą toponimu, etnonimu, nazwy mieszkańca lub innego leksemu objętego NER.",
        "- Reguły są liczone według pary lemat modelu → znormalizowany cel, niezależnie od kombinacji tagu.",
        "- `SAFE_NER_SGJP_LEMMA_AND_GENDER_REPAIR` wymaga zgodności liczby i przypadka, konfliktu wyłącznie rodzaju oraz jednego profilu rodzaju SGJP.",
        "- Docelowy tag zachowuje kontekstową liczbę i przypadek z korpusu, a z SGJP pobiera rodzaj i jego rozszerzenia, np. `ncol` lub `pt`.",
        "- Blokada konfliktu kapitalizacji obowiązuje także w gałęzi korekty rodzaju.",
        "- Eksperyment nadal niczego nie zapisuje do Parquetu.",
        "- Człony nazw wieloczłonowych nie otrzymują automatycznej decyzji w tej wersji.",
        "- `EXTERNAL_REGISTRY_CANDIDATE` obejmuje brak użytecznej analizy leksykalnej oraz konflikt wielkoliterowej nazwy z małoliterowym homografem SGJP.",
        "- Analizy techniczne `ign`, `xxx`, `burk`, `interp` i `brev` pozostają w diagnostyce, ale nie tworzą celu korekty.",
        "- Jeśli SGJP podaje warianty różniące się wyłącznie wielkością litery, wielkoliterowy pełny span NER preferuje wariant wielkoliterowy, np. `Kataru` → `Katar`.",
        "- Kandydaci zewnętrzni są przeznaczeni do późniejszego sprawdzenia w PRNG/KSNG/Wikidata.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference-tsv", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--min-docs", type=int, default=2)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    parquet = args.parquet.expanduser().resolve()
    if not parquet.is_file():
        raise SystemExit(f"Brak Parquetu: {parquet}")
    report = (args.report or parquet.with_name(parquet.stem + ".proper_name_lemma_audit_v2.md")).expanduser().resolve()
    reference_path = args.reference_tsv.expanduser().resolve() if args.reference_tsv else None
    if reference_path and not reference_path.is_file():
        raise SystemExit(f"Brak TSV: {reference_path}")

    observations, ner_counts, span_counts, documents, tokens = collect(parquet, args.batch_size, args.context, args.examples)
    reference, reference_rows = load_reference(reference_path)
    engine = morfeusz_engine()
    cache: dict[str, tuple[str, list[dict[str, str]]]] = {}
    selected = [item for item in observations.values() if item.count >= args.min_count and len(item.documents) >= args.min_docs]
    rows = []
    for index, obs in enumerate(sorted(selected, key=lambda item: (-item.count, item.orth.casefold())), 1):
        if obs.orth not in cache:
            cache[obs.orth] = analyses_for(engine, obs.orth)
        status, analyses = cache[obs.orth]
        rows.append(classify(obs, analyses, status, reference))
        if index % 500 == 0:
            print(f"[proper-name-audit-v2] {index:,}/{len(selected):,}", file=sys.stderr, flush=True)
    rules = aggregate_rules(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, parquet, rows, rules, ner_counts, span_counts, documents, tokens, reference_rows)
    if args.json:
        report.with_suffix(".json").write_text(json.dumps({
            "schema_version": 2, "version": VERSION, "parquet": str(parquet),
            "documents": documents, "tokens": tokens, "ner_counts": dict(ner_counts),
            "span_counts": dict(span_counts), "rules": rules, "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "success": True, "experimental": True, "version": VERSION,
        "report": str(report), "rows": len(rows), "rules": len(rules),
        "changed_tokens": sum(rule["token_count"] for rule in rules),
        "changed_morph_tokens": sum(rule.get("morph_token_count", 0) for rule in rules),
        "gender_repair_rules": sum(bool(rule.get("morph_repairs")) for rule in rules),
        "external_registry_candidates": sum(row["decision"] == "EXTERNAL_REGISTRY_CANDIDATE" for row in rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
