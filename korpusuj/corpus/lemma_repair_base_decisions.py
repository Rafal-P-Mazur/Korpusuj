# -*- coding: utf-8 -*-
"""Przygotowuje konserwatywne decyzje korekt na podstawie audytu D3.

Wejście:
- JSON wygenerowany przez experimental_stanza_lemma_repair_sqlite_d3.py;
- źródłowy Parquet, którego korekty mają dotyczyć;
- opcjonalnie named_entity_reference.sqlite z update_named_entity_reference.py.

Wyjście:
- *.auto.json      reguły automatyczne, status=accept;
- *.review.json    reguły do dalszej oceny, status=review;
- *.rejected.json  reguły wyłączone, status=reject;
- *.sample.md      deterministyczna, warstwowa próbka walidacyjna puli AUTO;
- *.summary.md     pełne podsumowanie selekcji.

Skrypt niczego nie zmienia w Parquecie ani w .search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

VERSION = "1.1.0"
AUTO_CLASS = "SAFE_FULL_MORPH_REPAIR"
PARTIAL_CLASS = "SAFE_PARTIAL_MORPH_REPAIR"
REJECT_CLASSES = {
    "CASE_NORMALIZATION_ONLY",
    "CONVENTION_GERUND",
    "CONVENTION_PARTICIPLE_ACTIVE",
    "CONVENTION_PARTICIPLE_PASSIVE",
    "AMBIGUOUS_OR_HOMOGRAPHIC",
}
ENTITY_UPOS = {"PROPN"}
ARTIFACT_RE = re.compile(r"(?:\d|[—–/\\|@#%]|\.{2,}|^-|-$)")
WORD_RE = re.compile(r"^[^\W\d_]+(?:-[^\W\d_]+)*$", re.UNICODE)


class DecisionError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    return str(value or "").strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DecisionError(f"Brak pliku: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise DecisionError(f"Nie można odczytać JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DecisionError(f"Główny element JSON musi być obiektem: {path}")
    return value


def validate_d3(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tool = clean(payload.get("tool"))
    version = clean(payload.get("version"))
    candidates = payload.get("candidates")
    if tool != "experimental_stanza_lemma_repair_sqlite":
        raise DecisionError(f"Nieobsługiwane narzędzie źródłowe: {tool!r}")
    if not version.startswith("3."):
        raise DecisionError(f"Oczekiwano raportu D3, otrzymano wersję: {version!r}")
    if not isinstance(candidates, list):
        raise DecisionError("Brak listy candidates w JSON D3.")
    return [item for item in candidates if isinstance(item, dict)]


def entity_keys(reference: Path | None) -> tuple[set[tuple[str, str, str]], dict[str, int]]:
    """Zwraca kombinacje orth+lemma+upos zaobserwowane wewnątrz encji."""
    if reference is None:
        return set(), {"available": 0, "rows": 0}
    if not reference.is_file():
        raise DecisionError(f"Brak bazy nazw własnych: {reference}")
    con = sqlite3.connect(f"file:{reference.as_posix()}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"entity_token_stats", "corpus_entities"} <= tables:
            raise DecisionError("Baza nazw własnych nie zawiera oczekiwanych tabel.")
        keys: set[tuple[str, str, str]] = set()
        rows = 0
        query = """
            SELECT DISTINCT t.orth,t.lemma,t.upos
            FROM entity_token_stats t
            JOIN corpus_entities e ON e.id=t.entity_id
            WHERE e.source_method IN ('NER','PROPN_SEQUENCE')
        """
        for orth, lemma, upos in con.execute(query):
            keys.add((clean(orth), clean(lemma), clean(upos).upper()))
            rows += 1
        return keys, {"available": 1, "rows": rows}
    finally:
        con.close()


def rule_identity(rule: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean(rule.get("orth")),
        clean(rule.get("lemma")),
        clean(rule.get("upos")).upper(),
        clean(rule.get("replacement")),
    )


def artifact_reasons(rule: dict[str, Any]) -> list[str]:
    orth, lemma, upos, replacement = rule_identity(rule)
    reasons = []
    if not all((orth, lemma, upos, replacement)):
        reasons.append("MISSING_REQUIRED_FIELD")
        return reasons
    if ARTIFACT_RE.search(orth) or not WORD_RE.fullmatch(orth):
        reasons.append("TOKENIZATION_OR_ORTHOGRAPHY_ARTIFACT")
    if lemma.casefold() == replacement.casefold():
        reasons.append("CASE_NORMALIZATION_ONLY")
    if any(ch.isspace() for ch in orth):
        reasons.append("MULTITOKEN_ORTH")
    return reasons


def candidate_rules(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rules = candidate.get("rules")
    return [r for r in rules if isinstance(r, dict)] if isinstance(rules, list) else []


def candidate_examples(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    value = candidate.get("examples") or candidate.get("concordances") or []
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def classify_rule(
    candidate: dict[str, Any],
    rule: dict[str, Any],
    entity_key_set: set[tuple[str, str, str]],
    reference_available: bool,
    common_core_only: bool,
) -> tuple[str, list[str]]:
    classification = clean(candidate.get("classification") or rule.get("classification"))
    sgjp_status = clean(candidate.get("sgjp_status") or rule.get("sgjp_status"))
    morph_status = clean(rule.get("morph_status"))
    orth, lemma, upos, replacement = rule_identity(rule)
    reasons = artifact_reasons(rule)

    if upos in ENTITY_UPOS:
        reasons.append("PROPN_REVIEW_REQUIRED")
    if (orth, lemma, upos) in entity_key_set:
        reasons.append("ENTITY_REFERENCE_HIT")

    if common_core_only and classification == AUTO_CLASS:
        if upos not in {"NOUN", "VERB"}:
            reasons.append("COMMON_CORE_UPOS_REVIEW_REQUIRED")
        if orth and orth[0].isupper():
            reasons.append("CAPITALIZED_FORM_REVIEW_REQUIRED")
        if lemma and lemma[0].isupper():
            reasons.append("CAPITALIZED_LEMMA_REVIEW_REQUIRED")

    if classification in REJECT_CLASSES:
        reasons.append(classification)
        return "rejected", sorted(set(reasons))

    if reasons:
        # Artefakty i encje nie są automatycznie odrzucane semantycznie.
        # Trafiają do REVIEW, chyba że sama klasa D3 jest diagnostyczna.
        if classification in {AUTO_CLASS, PARTIAL_CLASS, "MORPH_CONFLICT", "REVIEW_OTHER"}:
            return "review", sorted(set(reasons))
        return "rejected", sorted(set(reasons))

    if classification == AUTO_CLASS:
        if sgjp_status != "SGJP_UNIQUE_TARGET":
            return "review", ["SGJP_NOT_UNIQUE"]
        if morph_status != "FULL_MORPH_MATCH":
            return "review", ["MORPH_NOT_FULL_MATCH"]
        compatible = rule.get("compatible_lemmas") or []
        compatible = sorted({clean(x) for x in compatible if clean(x)})
        if compatible != [replacement]:
            return "review", ["TARGET_NOT_SOLE_COMPATIBLE_LEMMA"]
        if not reference_available:
            return "review", ["ENTITY_REFERENCE_NOT_AVAILABLE"]
        return "auto", ["D3_FULL_MORPH_SGJP_UNIQUE", "ENTITY_FILTER_PASSED"]

    if classification == PARTIAL_CLASS:
        return "review", ["PARTIAL_MORPH_MATCH"]
    if classification == "MORPH_CONFLICT":
        return "review", ["MORPH_CONFLICT"]
    return "review", [classification or "UNCLASSIFIED"]


def enrich_rule(candidate: dict[str, Any], rule: dict[str, Any], bucket: str, reasons: list[str]) -> dict[str, Any]:
    source_count = int(candidate.get("source_count") or 0)
    target_count = int(candidate.get("target_count") or 0)
    examples = candidate_examples(candidate)
    return {
        "orth": clean(rule.get("orth")),
        "lemma": clean(rule.get("lemma")),
        "upos": clean(rule.get("upos")).upper(),
        "replacement": clean(rule.get("replacement")),
        "reason": clean(rule.get("reason")) or "Kandydat przygotowany z audytu D3 SGJP/morfologia.",
        "status": "accept" if bucket == "auto" else ("review" if bucket == "review" else "reject"),
        "decision_bucket": bucket,
        "decision_reasons": reasons,
        "classification": clean(candidate.get("classification") or rule.get("classification")),
        "sgjp_status": clean(candidate.get("sgjp_status") or rule.get("sgjp_status")),
        "morph_status": clean(rule.get("morph_status")),
        "compatible_lemmas": rule.get("compatible_lemmas") or [],
        "stanza_morph_counts": rule.get("stanza_morph_counts") or {},
        "morph_comparisons": rule.get("morph_comparisons") or [],
        "observed_count": int(rule.get("observed_count") or 0),
        "source_count": source_count,
        "target_count": target_count,
        "source_document_count": int(candidate.get("source_docs") or candidate.get("source_doc_count") or 0),
        "target_document_count": int(candidate.get("target_docs") or candidate.get("target_doc_count") or 0),
        "source_form_count": int(candidate.get("source_forms") or candidate.get("source_form_count") or 0),
        "target_form_count": int(candidate.get("target_forms") or candidate.get("target_form_count") or 0),
        "examples": examples[:6],
    }


def deduplicate_rules(rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        grouped[(rule["orth"], rule["lemma"], rule["upos"])].append(rule)
    accepted = []
    conflicts = []
    for key, group in grouped.items():
        targets = {rule["replacement"] for rule in group}
        if len(targets) == 1:
            best = max(group, key=lambda r: (r["observed_count"], r["target_count"], r["replacement"]))
            accepted.append(best)
        else:
            for rule in group:
                rule = dict(rule)
                rule["status"] = "review"
                rule["decision_bucket"] = "review"
                rule["decision_reasons"] = sorted(set(rule["decision_reasons"] + ["CONFLICTING_TARGETS"]))
                conflicts.append(rule)
    accepted.sort(key=lambda r: (-r["observed_count"], r["upos"], r["orth"].casefold(), r["lemma"].casefold()))
    conflicts.sort(key=lambda r: (-r["observed_count"], r["orth"].casefold()))
    return accepted, conflicts


def stratified_sample(rules: list[dict[str, Any]], seed: int, per_upos: int, rare_n: int, high_n: int, distant_n: int) -> list[dict[str, Any]]:
    if not rules:
        return []
    rng = random.Random(seed)
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add(items: Iterable[dict[str, Any]], stratum: str) -> None:
        for item in items:
            key = rule_identity(item)
            if key not in selected:
                selected[key] = dict(item)
                selected[key]["sample_strata"] = [stratum]
            elif stratum not in selected[key]["sample_strata"]:
                selected[key]["sample_strata"].append(stratum)

    by_upos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        by_upos[rule["upos"]].append(rule)
    for upos, group in sorted(by_upos.items()):
        n = min(per_upos, len(group))
        add(rng.sample(group, n), f"RANDOM_{upos}")

    high = sorted(rules, key=lambda r: (-r["observed_count"], r["orth"].casefold()))[:high_n]
    rare = sorted(rules, key=lambda r: (r["observed_count"], r["orth"].casefold()))[:rare_n]

    def edit_proxy(rule: dict[str, Any]) -> float:
        a = rule["lemma"].casefold(); b = rule["replacement"].casefold()
        common = sum(1 for x, y in zip(a, b) if x == y)
        return 1.0 - (2.0 * common / max(1, len(a) + len(b)))

    distant = sorted(rules, key=lambda r: (-edit_proxy(r), -r["observed_count"]))[:distant_n]
    add(high, "HIGHEST_FREQUENCY")
    add(rare, "LOWEST_FREQUENCY")
    add(distant, "LARGEST_STRING_DIFFERENCE")
    return list(selected.values())


def decision_payload(
    bucket: str,
    rules: list[dict[str, Any]],
    d3_path: Path,
    d3_sha: str,
    parquet: Path,
    parquet_sha: str,
    reference: Path | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "tool": "prepare_d3_lemma_repair_decisions",
        "tool_version": VERSION,
        "command": "decisions",
        "bucket": bucket,
        "generated_at": now(),
        "source": {
            "path": str(parquet),
            "sha256": parquet_sha,
            "bytes": parquet.stat().st_size,
        },
        "d3_audit": {"path": str(d3_path), "sha256": d3_sha},
        "named_entity_reference": (
            {"path": str(reference), "sha256": sha256_file(reference)} if reference else None
        ),
        "settings": settings,
        "rules": rules,
    }
    payload["config_sha256"] = sha256_json({
        "schema_version": payload["schema_version"],
        "bucket": bucket,
        "source": payload["source"],
        "settings": settings,
        "rules": rules,
    })
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_sample(path: Path, sample: list[dict[str, Any]], summary: dict[str, Any], seed: int) -> None:
    lines = [
        "# Próba walidacyjna automatycznych korekt D3",
        "",
        f"**Wygenerowano:** `{now()}`  ",
        f"**Seed:** `{seed}`  ",
        f"**Liczba reguł AUTO:** **{summary['auto_rules']:,}**  ",
        f"**Liczebność próbki:** **{len(sample):,}**",
        "",
        "Dla każdej pozycji oceń regułę jako `ACCEPT` albo `REJECT`. Próbka nie zmienia pliku decyzji automatycznie.",
        "",
    ]
    for index, rule in enumerate(sample, 1):
        lines += [
            f"## {index}. `{rule['orth']}` + `{rule['lemma']}` + `{rule['upos']}` → `{rule['replacement']}`",
            "",
            "**Decyzja ręczna:** `TODO`  ",
            f"**Warstwy próbki:** `{', '.join(rule.get('sample_strata', []))}`  ",
            f"**Wystąpienia reguły:** **{rule['observed_count']:,}**  ",
            f"**Klasa:** `{rule['classification']}`  ",
            f"**Morfologia:** `{rule['morph_status']}`  ",
            f"**SGJP:** `{rule['sgjp_status']}`",
            "",
        ]
        examples = rule.get("examples") or []
        if examples:
            lines.append("**Przykłady:**")
            for example in examples[:3]:
                label = clean(example.get("document") or example.get("Oryginalna_nazwa_pliku") or example.get("doc_id"))
                context = clean(example.get("context"))
                lines.append(f"- `{label}`: {context}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, summary: dict[str, Any], reason_counts: Counter[str], outputs: dict[str, Path]) -> None:
    lines = [
        "# Przygotowanie decyzji korekt D3",
        "",
        f"**Wygenerowano:** `{now()}`",
        "",
        "## Wynik",
        "",
        f"- Reguły AUTO: **{summary['auto_rules']:,}**",
        f"- Reguły REVIEW: **{summary['review_rules']:,}**",
        f"- Reguły REJECTED: **{summary['rejected_rules']:,}**",
        f"- Konflikty celu przesunięte do REVIEW: **{summary['target_conflicts']:,}**",
        f"- Kandydaci D3: **{summary['d3_candidates']:,}**",
        f"- Surowe reguły D3: **{summary['raw_rules']:,}**",
        "",
        "## Pliki",
        "",
    ]
    for name, output in outputs.items():
        lines.append(f"- {name}: `{output}`")
    lines += ["", "## Powody klasyfikacji", ""]
    for reason, count in reason_counts.most_common():
        lines.append(f"- `{reason}`: **{count:,}**")
    lines += [
        "",
        "## Bezpieczeństwo",
        "",
        "Pula AUTO zawiera tylko reguły D3 z pełną zgodnością morfologiczną, jednoznacznym celem SGJP i bez artefaktów tokenizacji. W trybie `--common-core-only` pula jest dodatkowo ograniczona do małoliterowych rzeczowników i czasowników. Przed zastosowaniem całej puli należy ocenić wygenerowaną próbkę walidacyjną.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dzieli reguły D3 na AUTO, REVIEW i REJECTED oraz tworzy próbkę walidacyjną.",
        allow_abbrev=False,
    )
    parser.add_argument("--d3-json", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--named-entity-reference")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--sample-per-upos", type=int, default=40)
    parser.add_argument("--sample-rare", type=int, default=25)
    parser.add_argument("--sample-high-frequency", type=int, default=25)
    parser.add_argument("--sample-distant", type=int, default=25)
    parser.add_argument(
        "--allow-auto-without-entity-reference",
        action="store_true",
        help="Nie blokuj puli AUTO przy braku bazy nazw własnych. Opcja eksperymentalna.",
    )
    parser.add_argument(
        "--common-core-only",
        action="store_true",
        help="Pula AUTO tylko dla małoliterowych NOUN/VERB; ADJ i formy kapitalizowane trafiają do REVIEW.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    d3_path = Path(args.d3_json).resolve()
    parquet = Path(args.parquet).resolve()
    reference = Path(args.named_entity_reference).resolve() if args.named_entity_reference else None
    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if not parquet.is_file():
        raise DecisionError(f"Brak końcowego Parquetu: {parquet}")

    d3 = load_json(d3_path)
    candidates = validate_d3(d3)
    keys, reference_info = entity_keys(reference)
    reference_available = bool(reference_info["available"] or args.allow_auto_without_entity_reference)

    buckets: dict[str, list[dict[str, Any]]] = {"auto": [], "review": [], "rejected": []}
    reason_counts: Counter[str] = Counter()
    raw_rules = 0
    for candidate in candidates:
        for rule in candidate_rules(candidate):
            raw_rules += 1
            bucket, reasons = classify_rule(candidate, rule, keys, reference_available, args.common_core_only)
            reason_counts.update(reasons)
            buckets[bucket].append(enrich_rule(candidate, rule, bucket, reasons))

    auto, auto_conflicts = deduplicate_rules(buckets["auto"])
    review, review_conflicts = deduplicate_rules(buckets["review"] + auto_conflicts)
    rejected, rejected_conflicts = deduplicate_rules(buckets["rejected"])
    # Konflikty w odrzuconych też zachowujemy w REVIEW, bo nie wolno ich zgubić.
    review.extend(rejected_conflicts)
    review.sort(key=lambda r: (-r["observed_count"], r["upos"], r["orth"].casefold()))

    settings = {
        "required_auto_classification": AUTO_CLASS,
        "required_sgjp_status": "SGJP_UNIQUE_TARGET",
        "required_morph_status": "FULL_MORPH_MATCH",
        "entity_reference_required": not args.allow_auto_without_entity_reference,
        "artifact_pattern": ARTIFACT_RE.pattern,
        "seed": args.seed,
        "common_core_only": args.common_core_only,
        "common_core_allowed_upos": ["NOUN", "VERB"] if args.common_core_only else None,
        "common_core_requires_lowercase_orth": args.common_core_only,
        "common_core_requires_lowercase_lemma": args.common_core_only,
    }
    parquet_sha = sha256_file(parquet)
    d3_sha = sha256_file(d3_path)
    outputs = {
        "AUTO": default_output(prefix, ".auto.json"),
        "REVIEW": default_output(prefix, ".review.json"),
        "REJECTED": default_output(prefix, ".rejected.json"),
        "SAMPLE": default_output(prefix, ".sample.md"),
        "SUMMARY": default_output(prefix, ".summary.md"),
    }
    write_json(outputs["AUTO"], decision_payload("auto", auto, d3_path, d3_sha, parquet, parquet_sha, reference, settings))
    write_json(outputs["REVIEW"], decision_payload("review", review, d3_path, d3_sha, parquet, parquet_sha, reference, settings))
    write_json(outputs["REJECTED"], decision_payload("rejected", rejected, d3_path, d3_sha, parquet, parquet_sha, reference, settings))

    summary = {
        "d3_candidates": len(candidates),
        "raw_rules": raw_rules,
        "auto_rules": len(auto),
        "review_rules": len(review),
        "rejected_rules": len(rejected),
        "target_conflicts": len(auto_conflicts) + len(review_conflicts) + len(rejected_conflicts),
        "entity_reference_rows": reference_info["rows"],
    }
    sample = stratified_sample(
        auto,
        args.seed,
        args.sample_per_upos,
        args.sample_rare,
        args.sample_high_frequency,
        args.sample_distant,
    )
    write_sample(outputs["SAMPLE"], sample, summary, args.seed)
    write_summary(outputs["SUMMARY"], summary, reason_counts, outputs)

    print(json.dumps({
        "success": True,
        **summary,
        "sample_rules": len(sample),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DecisionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("ERROR: Przerwano przez użytkownika.", file=sys.stderr)
        raise SystemExit(130)
