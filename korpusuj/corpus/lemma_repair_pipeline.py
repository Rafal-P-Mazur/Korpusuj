# -*- coding: utf-8 -*-
"""Jawny pipeline przygotowania decyzji naprawy lematyzacji."""
from __future__ import annotations

from dataclasses import dataclass

PIPELINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class LemmaRepairStage:
    key: str
    phase: str
    callable_name: str
    description: str
    mutates_parquet: bool = False


WORKFLOW_STAGES = (
    LemmaRepairStage("audit", "analysis", "run_audit", "Agregacja statystyk, SGJP i kandydaci D3."),
    LemmaRepairStage("base_decisions", "prepare", "engine.main", "Bazowy podział AUTO, REVIEW i REJECTED."),
    LemmaRepairStage("common_core_plus", "prepare", "_augment_plus", "Bezpieczne rozszerzenie common-core-plus."),
    LemmaRepairStage("direct_sgjp", "prepare", "augment_direct_sgjp", "Bezpośrednia walidacja form w SGJP."),
    LemmaRepairStage("contextual_sgjp", "prepare", "augment_contextual_sgjp", "Kontekstowe rozstrzyganie SGJP."),
    LemmaRepairStage("proper_names", "prepare", "augment_proper_name_repairs", "Nazwy własne w SGJP i PRNG."),
    LemmaRepairStage("finalize_sgjp", "prepare", "finalize_sgjp_replacements", "Normalizacja finalnych celów SGJP."),
    LemmaRepairStage("safety", "prepare", "filter_auto_rules", "Filtr bezpieczeństwa reguł AUTO."),
    LemmaRepairStage("consistency", "prepare", "augment_lemma_consistency_repairs", "Końcowa spójność lematów."),
    LemmaRepairStage("review_recovery", "prepare", "recover_safe_review_rules", "Odzysk ściśle bezpiecznych reguł REVIEW."),
    LemmaRepairStage("residual_repairs", "prepare", "augment_residual_repairs", "Domknięcie bezpiecznych resztek."),
    LemmaRepairStage("final_recount", "prepare", "recount_final_auto_rules", "Końcowe przeliczenie zasięgu reguł."),
    LemmaRepairStage("preview", "validation", "preview", "Dry-run i kontrola liczników."),
    LemmaRepairStage("apply", "rewrite", "apply", "Zapis poprawionego Parquetu.", True),
)

PREPARE_STAGE_KEYS = tuple(stage.key for stage in WORKFLOW_STAGES if stage.phase == "prepare")
EXPECTED_PREPARE_STAGE_KEYS = (
    "base_decisions",
    "common_core_plus",
    "direct_sgjp",
    "contextual_sgjp",
    "proper_names",
    "finalize_sgjp",
    "safety",
    "consistency",
    "review_recovery",
    "residual_repairs",
    "final_recount",
)


def pipeline_manifest() -> tuple[LemmaRepairStage, ...]:
    """Return the immutable public description of the current workflow."""
    return WORKFLOW_STAGES


def validate_pipeline_manifest() -> None:
    """Fail fast if keys, order, or destructive-stage declarations drift."""
    keys = tuple(stage.key for stage in WORKFLOW_STAGES)
    if len(keys) != len(set(keys)):
        raise RuntimeError("Powielone klucze etapów lemma repair.")
    if PREPARE_STAGE_KEYS != EXPECTED_PREPARE_STAGE_KEYS:
        raise RuntimeError(
            "Zmieniono kolejność etapów prepare bez aktualizacji kontraktu regresyjnego: "
            f"{PREPARE_STAGE_KEYS}"
        )
    mutating = tuple(stage.key for stage in WORKFLOW_STAGES if stage.mutates_parquet)
    if mutating != ("apply",):
        raise RuntimeError(f"Tylko apply może modyfikować Parquet; otrzymano: {mutating}")


validate_pipeline_manifest()

import json
import re
from collections import defaultdict
from typing import Any
from .lemma_repair_models import LemmaRepairError, LemmaRepairOptions, LemmaRepairPaths
from . import lemma_repair_base_decisions as engine
from .lemma_repair_sgjp import augment_contextual_sgjp, augment_direct_sgjp, finalize_sgjp_replacements
from .lemma_repair_proper_names import augment_proper_name_repairs
from .lemma_repair_decisions import filter_auto_rules
from .lemma_repair_consistency import augment_lemma_consistency_repairs
from .lemma_repair_decisions import recover_safe_review_rules
from .lemma_repair_residuals import augment_residual_repairs


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _key(rule: dict[str, Any]) -> tuple[str, str, str]:
    return (_clean(rule.get("orth")), _clean(rule.get("lemma")), _clean(rule.get("upos")).upper())


def _enrich(candidate: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "orth": _clean(rule.get("orth")),
        "lemma": _clean(rule.get("lemma")),
        "upos": _clean(rule.get("upos")).upper(),
        "replacement": _clean(rule.get("replacement")),
        "reason": "Jednoznaczny cel SGJP; konflikt tagu Stanza ograniczony do rodzaju gramatycznego.",
        "status": "accept",
        "decision_bucket": "auto",
        "decision_reasons": ["D3_UNIQUE_TARGET_SAFE_GENDER_CONFLICT"],
        "classification": "SAFE_LEMMA_REPAIR_WITH_TAG_CONFLICT",
        "source_classification": _clean(candidate.get("classification")),
        "sgjp_status": _clean(candidate.get("sgjp_status") or rule.get("sgjp_status")),
        "morph_status": _clean(rule.get("morph_status")),
        "compatible_lemmas": rule.get("compatible_lemmas") or [],
        "stanza_morph_counts": rule.get("stanza_morph_counts") or {},
        "morph_comparisons": rule.get("morph_comparisons") or [],
        "observed_count": int(rule.get("observed_count") or 0),
        "source_count": int(candidate.get("source_count") or 0),
        "target_count": int(candidate.get("target_count") or 0),
        "source_document_count": int(candidate.get("source_docs") or 0),
        "target_document_count": int(candidate.get("target_docs") or 0),
        "source_form_count": int(candidate.get("source_forms") or 0),
        "target_form_count": int(candidate.get("target_forms") or 0),
        "examples": list(candidate.get("examples") or [])[:6],
    }


_LETTER_WORD_RE = re.compile(r"^[^\W\d_]+$", re.UNICODE)

def _safe_plain_word(value: str) -> bool:
    """Jedno slowo zlozone wylacznie z liter Unicode."""
    return bool(value and _LETTER_WORD_RE.fullmatch(value))


def _safe_gender_conflict(candidate: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Dopuszcza tylko jednoznaczne rzeczowniki z konfliktem ograniczonym do gender.

    Liczba i przypadek musza byc porownywalne i zgodne dla kazdego
    zaobserwowanego tagu. Nie dopuszczamy braku analizy, konfliktu UPOS ani
    konfliktu case/number. To obejmuje m.in. czołgami/czołgo i hełmy/hełma.
    """
    replacement = _clean(rule.get("replacement"))
    orth, lemma, upos = _key(rule)
    if _clean(candidate.get("classification")) != "MORPH_CONFLICT":
        return False
    if _clean(candidate.get("sgjp_status")) != "SGJP_UNIQUE_TARGET":
        return False
    if upos != "NOUN" or not all((orth, lemma, replacement)):
        return False
    if not orth[0].islower() or not lemma[0].islower() or not replacement[0].islower():
        return False
    # Odrzucamy e-maile, URL-e, domeny, cyfry, podkreslenia, laczniki
    # i pozostale artefakty tokenizacji.
    if not all(_safe_plain_word(value) for value in (orth, lemma, replacement)):
        return False
    compatible = sorted({_clean(x) for x in (rule.get("compatible_lemmas") or []) if _clean(x)})
    if compatible != [replacement]:
        return False
    details = rule.get("morph_comparisons") or []
    if not details:
        return False
    saw_gender_conflict = False
    for detail in details:
        if _clean(detail.get("outcome")) not in {"MATCH", "MISMATCH"}:
            return False
        best = detail.get("best_comparison")
        if not isinstance(best, dict):
            return False
        shared = set(best.get("shared") or [])
        matches = set((best.get("matches") or {}).keys())
        mismatches = set((best.get("mismatches") or {}).keys())
        if not {"number", "case"}.issubset(shared):
            return False
        if not {"number", "case"}.issubset(matches):
            return False
        if mismatches - {"gender"}:
            return False
        if "gender" in mismatches:
            saw_gender_conflict = True
    return saw_gender_conflict


def _write_payload(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _augment_plus(paths: LemmaRepairPaths) -> dict[str, int]:
    audit = json.loads(paths.audit_json().read_text(encoding="utf-8"))
    auto = json.loads(paths.decisions_auto().read_text(encoding="utf-8"))
    review = json.loads(paths.decisions_review().read_text(encoding="utf-8"))

    existing = {_key(r): r for r in auto.get("rules", [])}
    review_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in review.get("rules", []):
        review_by_key[_key(item)].append(item)

    promoted: list[dict[str, Any]] = []
    for candidate in audit.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        for rule in candidate.get("rules", []):
            if not isinstance(rule, dict) or not _safe_gender_conflict(candidate, rule):
                continue
            key = _key(rule)
            target = _clean(rule.get("replacement"))
            current = existing.get(key)
            if current and _clean(current.get("replacement")) != target:
                raise LemmaRepairError(f"Konflikt celu common-core-plus dla {key}")
            if current:
                continue
            enriched = _enrich(candidate, rule)
            existing[key] = enriched
            promoted.append(enriched)

    promoted_keys = {_key(r) for r in promoted}
    auto["rules"] = sorted(existing.values(), key=lambda r: (-int(r.get("observed_count") or 0), r["upos"], r["orth"].casefold()))
    auto["bucket"] = "auto"
    auto.setdefault("settings", {})["policy_mode"] = "common-core-plus"
    auto["settings"]["safe_tag_conflicts"] = {
        "upos": ["NOUN"],
        "allowed_mismatch_features": ["gender"],
        "required_matching_features": ["number", "case"],
        "orthography": "unicode_letters_only",
        "rejects": ["email", "url", "domain", "digits", "underscore", "hyphen", "tokenization_artifact"],
    }
    auto["plus_promoted_rules"] = len(promoted)

    review["rules"] = [r for r in review.get("rules", []) if _key(r) not in promoted_keys]
    review.setdefault("settings", {})["policy_mode"] = "common-core-plus"
    review["plus_promoted_rules"] = len(promoted)

    _write_payload(paths.decisions_auto(), auto)
    _write_payload(paths.decisions_review(), review)
    return {
        "plus_promoted_rules": len(promoted),
        "auto_rules": len(auto["rules"]),
        "review_rules": len(review["rules"]),
    }


def prepare_policy(paths: LemmaRepairPaths, options: LemmaRepairOptions, reporter: Any = None) -> dict:
    if not paths.audit_json().is_file():
        raise LemmaRepairError(f"Brak audytu D3: {paths.audit_json()}")
    if reporter:
        reporter.status("Przygotowywanie bezpiecznych korekt lematow...")
    argv = ["--d3-json", str(paths.audit_json()), "--parquet", str(paths.parquet),
            "--output-prefix", str(paths.artifact_prefix), "--seed", str(options.seed),
            "--allow-auto-without-entity-reference"]
    if options.mode in {"common-core", "common-core-plus"}:
        argv.append("--common-core-only")
    try:
        engine.main(argv)
    except SystemExit as exc:
        if int(exc.code or 0) != 0:
            raise LemmaRepairError(f"Selekcja decyzji zakonczyla sie kodem {exc.code}")
    if options.mode == "common-core-plus":
        extra = _augment_plus(paths)
        if reporter:
            reporter.status("Bezposrednia walidacja obserwowanych form w SGJP...")
        direct = augment_direct_sgjp(paths, reporter)
        if reporter:
            reporter.status("Kontekstowe rozstrzyganie SGJP...")
        contextual = augment_contextual_sgjp(paths, reporter)
        if reporter:
            reporter.status("Walidacja jednotokenowych nazw własnych w SGJP i PRNG...")
        proper_names = augment_proper_name_repairs(paths, reporter)
        normalized_sgjp = finalize_sgjp_replacements(paths)
        filter_auto_rules(paths)
        consistency = augment_lemma_consistency_repairs(paths, reporter)
        review_recovery = recover_safe_review_rules(paths, reporter)
        residual = augment_residual_repairs(paths, reporter)
        # Finalizacja może zmienić cel lub usunąć reguły puste. Liczniki AUTO
        # muszą więc zostać policzone jeszcze raz według dokładnie tej samej
        # hierarchii dopasowania co preview/apply, łącznie z regułami pozycyjnymi.
        from .lemma_repair_proper_names import recount_final_auto_rules
        recounted = recount_final_auto_rules(paths)
        extra = {**extra, **direct, **contextual, **proper_names, **normalized_sgjp, **consistency, **review_recovery, **residual, **recounted}
    else:
        extra = {
            "plus_promoted_rules": 0,
            "direct_sgjp_promoted_rules": 0,
            "auto_rules": len(json.loads(paths.decisions_auto().read_text(encoding="utf-8")).get("rules", [])),
        }
    return {**extra, "auto_path": str(paths.decisions_auto()), "mode": options.mode}
