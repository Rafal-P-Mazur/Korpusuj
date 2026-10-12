# -*- coding: utf-8 -*-
"""Bezpieczne reguly PROPN z jednoznacznej analizy Morfeusza zgodnej z pelnym XPOS."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import lemma_repair_analysis as analysis
from . import lemma_repair_proper_names as proper_names
from .lemma_repair_models import LemmaRepairPaths

PLACE_NER_MARKERS = ("place", "geog", "loc")
REQUIRED_SHARED_FEATURES = {"case", "gender", "number"}
CLASSIFICATION = "PROPN_MORFEUSZ_FULL_XPOS_PRNG"
DECISION_SOURCE = "exact_morfeusz_full_xpos_prng"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _clean(value).casefold()


def _lemma(value: Any) -> str:
    return _clean(value).split(":", 1)[0]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value if isinstance(value, list) else []


def _place_ner(value: Any) -> bool:
    folded = _norm(value)
    return any(marker in folded for marker in PLACE_NER_MARKERS)


def _prng_path() -> Path:
    # Uzywamy tego samego resolvera zasobow co produkcyjny modul nazw wlasnych.
    return proper_names.resource_root() / "temp" / "prng_world.sqlite"


def _open_prng() -> sqlite3.Connection | None:
    # Nie duplikujemy kontraktu bazy PRNG. Produkcyjny open_prng wykonuje
    # wlasciwa walidacje wersji, schematu i trybu tylko do odczytu.
    path = _prng_path()
    try:
        connection, status = proper_names.open_prng(path)
    except Exception:
        return None
    if connection is None or str(status).strip().casefold() != "ok":
        if connection is not None:
            connection.close()
        return None
    connection.row_factory = sqlite3.Row
    return connection


def _canonical_prng_row(connection: sqlite3.Connection, candidate: str) -> dict[str, Any] | None:
    # Korzystamy wylacznie z produkcyjnego API PRNG. Indeks runtime ma tabele
    # name_forms, a jego schemat nie jest kontraktem tego modulu.
    hits = proper_names.prng_hits(connection, candidate)
    canonical_hits = [
        hit for hit in hits
        if _norm(hit.get("canonical_name")) == _norm(candidate)
        and _norm(hit.get("kind")) == "canonical"
    ]
    canonical_ids = {
        (_clean(hit.get("prng_id")), _norm(hit.get("canonical_name")))
        for hit in canonical_hits
    }
    if len(canonical_ids) != 1:
        return None
    # Zwracamy znormalizowany dowod API, nie surowy rekord SQLite.
    return {
        "candidate": candidate,
        "canonical_name": canonical_hits[0].get("canonical_name", ""),
        "prng_id": canonical_hits[0].get("prng_id", ""),
        "object_type": canonical_hits[0].get("object_type", ""),
        "name_status": canonical_hits[0].get("name_status", ""),
        "hits": canonical_hits,
        "evidence": "PRNG_CANONICAL_EXACT_VIA_API",
    }


def _morfeusz_engine():
    # Produkcyjne API zwraca (engine, status), a nie sam obiekt Morfeusza.
    result = analysis.morfeusz_engine()
    if isinstance(result, tuple):
        engine = result[0] if result else None
        status = result[1] if len(result) > 1 else ""
    else:
        engine = result
        status = "legacy_direct_engine"
    if engine is None or not callable(getattr(engine, "analyse", None)):
        raise RuntimeError(
            "Produkcyjny morfeusz_engine() nie zwrocil obiektu z metoda analyse; "
            f"status={status!r}, type={type(engine).__name__}"
        )
    return engine


def _raw_analyses(engine: Any, orth: str) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for raw in engine.analyse(orth):
        interpretation = raw[2] if len(raw) >= 3 else raw[-1]
        if not isinstance(interpretation, (list, tuple)) or len(interpretation) < 3:
            continue
        lemma = _lemma(interpretation[1])
        tag = _clean(interpretation[2])
        if lemma and tag:
            output.append((lemma, tag))
    return output


def _is_full_xpos_match(stanza_tag: str, morfeusz_tag: str) -> tuple[bool, dict[str, Any]]:
    comparison = analysis.compare_tags(stanza_tag, morfeusz_tag)
    if not isinstance(comparison, dict):
        return False, {"raw": repr(comparison)}
    mismatches = comparison.get("mismatches") or {}
    shared = set(comparison.get("shared") or [])
    matches = comparison.get("matches") or {}
    valid = not mismatches and REQUIRED_SHARED_FEATURES <= shared
    for feature in REQUIRED_SHARED_FEATURES:
        pair = matches.get(feature)
        if isinstance(pair, (list, tuple)) and len(pair) >= 2 and _norm(pair[0]) != _norm(pair[1]):
            valid = False
    return valid, comparison


def _existing_rule_key(rule: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _clean(rule.get("orth")), _clean(rule.get("lemma")), _clean(rule.get("upos")).upper(),
        _clean(rule.get("morph_from")), _clean(rule.get("replacement")),
    )


def augment_exact_morfeusz_propn(paths: LemmaRepairPaths, reporter: Any = None) -> dict[str, Any]:
    """Dodaj tylko reguly z jednym lematem Morfeusza zgodnym z pelnym XPOS i PRNG."""
    stats: dict[str, Any] = {
        "exact_morfeusz_propn_scanned": 0,
        "exact_morfeusz_propn_named_place": 0,
        "exact_morfeusz_propn_unique_xpos": 0,
        "exact_morfeusz_propn_prng_confirmed": 0,
        "exact_morfeusz_propn_promoted_rules": 0,
        "exact_morfeusz_propn_promoted_tokens": 0,
        "exact_morfeusz_propn_prng_available": False,
        "exact_morfeusz_propn_prng_path": str(_prng_path()),
    }
    if not paths.decisions_auto().is_file():
        return stats
    connection = _open_prng()
    if connection is None:
        return stats
    stats["exact_morfeusz_propn_prng_available"] = True
    engine = _morfeusz_engine()

    observations: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    documents: dict[tuple[str, str, str, str, str], set[int]] = defaultdict(set)
    examples: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    parquet = pq.ParquetFile(paths.parquet)
    available = set(parquet.schema_arrow.names)
    ner_column = "ners" if "ners" in available else ("nertags" if "nertags" in available else None)
    morph_column = "full_postags" if "full_postags" in available else ("postags" if "postags" in available else None)
    required = {"tokens", "lemmas", "upostags"}
    if not required <= available or ner_column is None or morph_column is None:
        connection.close()
        return stats

    columns = ["tokens", "lemmas", "upostags", morph_column, ner_column]
    doc_id = -1
    try:
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                doc_id += 1
                tokens = _as_list(row.get("tokens"))
                lemmas = _as_list(row.get("lemmas"))
                upos = _as_list(row.get("upostags"))
                morph = _as_list(row.get(morph_column))
                ner = _as_list(row.get(ner_column))
                if not (len(tokens) == len(lemmas) == len(upos) == len(morph) == len(ner)):
                    continue
                for position, (orth, model_lemma, token_upos, token_xpos, token_ner) in enumerate(
                    zip(tokens, lemmas, upos, morph, ner, strict=True)
                ):
                    if _clean(token_upos).upper() != "PROPN":
                        continue
                    if not _place_ner(token_ner):
                        continue
                    if not _clean(token_xpos) or not _clean(orth) or not _clean(model_lemma):
                        continue
                    stats["exact_morfeusz_propn_named_place"] += 1
                    raw = _raw_analyses(engine, _clean(orth))
                    stats["exact_morfeusz_propn_scanned"] += 1
                    compatible: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for candidate, morfeusz_tag in raw:
                        valid, comparison = _is_full_xpos_match(_clean(token_xpos), morfeusz_tag)
                        if valid:
                            compatible[candidate].append({
                                "morfeusz_tag": morfeusz_tag,
                                "comparison": comparison,
                            })
                    if len(compatible) != 1:
                        continue
                    replacement = next(iter(compatible))
                    if _norm(replacement) == _norm(model_lemma):
                        continue
                    stats["exact_morfeusz_propn_unique_xpos"] += 1
                    prng_row = _canonical_prng_row(connection, replacement)
                    if prng_row is None:
                        continue
                    stats["exact_morfeusz_propn_prng_confirmed"] += 1
                    key = (_clean(orth), _clean(model_lemma), "PROPN", _clean(token_xpos), replacement)
                    if key not in observations:
                        observations[key] = {
                            "analyses": compatible[replacement],
                            "prng": prng_row,
                        }
                    documents[key].add(doc_id)
                    if len(examples[key]) < 8:
                        left = max(0, position - 8)
                        right = min(len(tokens), position + 9)
                        examples[key].append({
                            "document": doc_id,
                            "token_index": position,
                            "context": " ".join(_clean(value) for value in tokens[left:right]),
                            "ner": _clean(token_ner),
                            "xpos": _clean(token_xpos),
                        })
    finally:
        connection.close()

    payload = json.loads(paths.decisions_auto().read_text(encoding="utf-8"))
    rules = list(payload.get("rules") or [])
    existing = {_existing_rule_key(rule) for rule in rules}
    promoted = []
    for key in sorted(observations):
        orth, model_lemma, token_upos, token_xpos, replacement = key
        if key in existing:
            continue
        evidence = observations[key]
        rule = {
            "orth": orth,
            "lemma": model_lemma,
            "upos": token_upos,
            "replacement": replacement,
            "reason": "Jedyny lemat Morfeusza zgodny z pelnym XPOS; nazwa miejsca potwierdzona w PRNG.",
            "status": "accept",
            "decision_bucket": "auto",
            "decision_reasons": [
                "MORFEUSZ_UNIQUE_FULL_XPOS_MATCH",
                "NER_PLACE",
                "PRNG_CANONICAL_EXACT",
            ],
            "classification": CLASSIFICATION,
            "decision_source": DECISION_SOURCE,
            "morph_from": token_xpos,
            "morph_to": token_xpos,
            "observed_count": sum(1 for _ in examples[key]) if len(documents[key]) == 0 else 0,
            "source_document_count": len(documents[key]),
            "examples": examples[key],
            "evidence": evidence,
        }
        # observed_count musi odpowiadac wszystkim trafieniom, nie tylko probce.
        rule["observed_count"] = sum(
            1 for batch in pq.ParquetFile(paths.parquet).iter_batches(batch_size=128, columns=columns)
            for row in batch.to_pylist()
            for o, l, u, m in zip(
                _as_list(row.get("tokens")), _as_list(row.get("lemmas")),
                _as_list(row.get("upostags")), _as_list(row.get(morph_column)), strict=True
            )
            if _clean(o) == orth and _clean(l) == model_lemma and _clean(u).upper() == token_upos and _clean(m) == token_xpos
        )
        if rule["observed_count"] <= 0:
            continue
        rules.append(rule)
        promoted.append(rule)
        existing.add(key)

    payload["rules"] = rules
    payload["exact_morfeusz_propn_policy"] = {
        "classification": CLASSIFICATION,
        "requirements": [
            "UPOS_PROPN", "NER_PLACE", "UNIQUE_MORFEUSZ_LEMMA",
            "FULL_XPOS_CASE_NUMBER_GENDER", "PRNG_CANONICAL_EXACT",
        ],
        "promoted_rules": len(promoted),
    }
    paths.decisions_auto().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostic = {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        "stats": stats,
        "promoted_rules": promoted,
    }
    paths.artifact_prefix.with_suffix(".exact_morfeusz_propn.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats["exact_morfeusz_propn_promoted_rules"] = len(promoted)
    stats["exact_morfeusz_propn_promoted_tokens"] = sum(int(rule["observed_count"]) for rule in promoted)
    if reporter and promoted:
        reporter.status(f"Dodano {len(promoted)} bezpiecznych regul PROPN z pelnego XPOS i PRNG.")
    return stats
