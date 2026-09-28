# -*- coding: utf-8 -*-
"""Eksperymentalny audyt i kontrolowana korekta lematów w korpusie Korpusuj.

Skrypt działa na gotowym pliku Parquet. Nie uruchamia Stanza ani spaCy.
Oferuje trzy komendy:

1. audit   - wykrywa możliwe rozszczepienia paradygmatów i tworzy raport MD + JSON;
2. prepare - tworzy edytowalny plik decyzji na podstawie wyników audytu;
3. apply   - stosuje wyłącznie reguły ze statusem "accept" do NOWEGO Parquetu.

Przykład:
    python experimental_stanza_lemma_repair.py audit --parquet korpus.parquet
    python experimental_stanza_lemma_repair.py prepare --audit-json korpus.lemma_audit.json
    # ręcznie zmień wybrane statusy z "review" na "accept"
    python experimental_stanza_lemma_repair.py apply \
        --parquet korpus.parquet \
        --decisions korpus.lemma_decisions.json \
        --output korpus_lemma_repaired.parquet

Po apply należy przebudować artefakty pochodne:
    python -m korpusuj.index.cli create korpus_lemma_repaired.parquet --progress on --pretty
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_VERSION = "1.0.0"
KORPUS_META_KEY = b"korpus_meta"
REQUIRED_COLUMNS = ("tokens", "lemmas", "upostags")
OPTIONAL_MORPH_COLUMNS = ("full_postags", "postags")
DEFAULT_REPORT_LIMIT = 300

# Funkcyjne i techniczne klasy rzadko tworzą użyteczne paradygmaty fleksyjne.
DEFAULT_EXCLUDED_UPOS = {
    "ADP", "AUX", "CCONJ", "DET", "INTJ", "PART", "PRON", "PUNCT",
    "SCONJ", "SYM", "X",
}


class LemmaRepairError(RuntimeError):
    pass


@dataclass
class FormStats:
    count: int = 0
    docs: set[int] = field(default_factory=set)
    morph: Counter[str] = field(default_factory=Counter)


@dataclass
class LemmaStats:
    lemma: str
    upos: str
    count: int = 0
    docs: set[int] = field(default_factory=set)
    forms: dict[str, FormStats] = field(default_factory=dict)
    morph: Counter[str] = field(default_factory=Counter)

    @property
    def form_count(self) -> int:
        return len(self.forms)

    @property
    def doc_count(self) -> int:
        return len(self.docs)


@dataclass
class Candidate:
    source_lemma: str
    target_lemma: str
    upos: str
    source_count: int
    target_count: int
    source_doc_count: int
    target_doc_count: int
    source_form_count: int
    target_form_count: int
    similarity: float
    source_forms: list[dict[str, Any]]
    target_forms: list[dict[str, Any]]
    evidence: list[str]
    cautions: list[str]
    classification: str
    rules: list[dict[str, Any]]
    concordances: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_lemma": self.source_lemma,
            "target_lemma": self.target_lemma,
            "upos": self.upos,
            "source_count": self.source_count,
            "target_count": self.target_count,
            "source_doc_count": self.source_doc_count,
            "target_doc_count": self.target_doc_count,
            "source_form_count": self.source_form_count,
            "target_form_count": self.target_form_count,
            "similarity": round(self.similarity, 6),
            "source_forms": self.source_forms,
            "target_forms": self.target_forms,
            "evidence": self.evidence,
            "cautions": self.cautions,
            "classification": self.classification,
            "rules": self.rules,
            "concordances": self.concordances,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_for_similarity(value: str) -> str:
    value = clean_text(value).casefold()
    # Numery sensów bywają zapisane po dwukropku, np. zamek:1.
    value = re.sub(r":\d+$", "", value)
    return value


def common_prefix_length(a: str, b: str) -> int:
    a = normalize_for_similarity(a)
    b = normalize_for_similarity(b)
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def lemma_similarity(a: str, b: str) -> float:
    a_n = normalize_for_similarity(a)
    b_n = normalize_for_similarity(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def probable_same_family(source: LemmaStats, target: LemmaStats) -> bool:
    """Konserwatywny filtr kandydatów, nie ostateczna decyzja o korekcie."""
    if source.upos != target.upos or source.lemma == target.lemma:
        return False
    a = normalize_for_similarity(source.lemma)
    b = normalize_for_similarity(target.lemma)
    similarity = lemma_similarity(a, b)
    prefix = common_prefix_length(a, b)
    shortest = min(len(a), len(b))
    # Dopuszczamy np. kpieć/kpić, ale ograniczamy łączenie przypadkowych krótkich słów.
    return similarity >= 0.72 and (prefix >= 3 or (shortest <= 5 and prefix >= 2))


def compact_morph(value: Any) -> str:
    text = clean_text(value)
    return text if text else "<brak>"


def ensure_parquet(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise LemmaRepairError(f"Plik nie istnieje: {path}")
    if path.suffix.lower() != ".parquet":
        raise LemmaRepairError(f"Oczekiwano pliku .parquet: {path}")
    return path


def schema_and_columns(path: Path) -> tuple[pa.Schema, list[str], str | None]:
    pf = pq.ParquetFile(path)
    try:
        schema = pf.schema_arrow
        columns = list(schema.names)
    finally:
        try:
            pf.close()
        except Exception:
            pass
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise LemmaRepairError(f"Brak wymaganych kolumn: {', '.join(missing)}")
    morph_column = next((name for name in OPTIONAL_MORPH_COLUMNS if name in columns), None)
    return schema, columns, morph_column


def iter_rows(path: Path, columns: Sequence[str], batch_size: int) -> Iterable[tuple[int, dict[str, Any]]]:
    pf = pq.ParquetFile(path)
    doc_id = 0
    try:
        for batch in pf.iter_batches(batch_size=max(1, batch_size), columns=list(columns)):
            data = batch.to_pydict()
            for row_index in range(batch.num_rows):
                yield doc_id, {name: data[name][row_index] for name in columns}
                doc_id += 1
    finally:
        try:
            pf.close()
        except Exception:
            pass


def collect_statistics(
    path: Path,
    *,
    batch_size: int,
    excluded_upos: set[str],
) -> tuple[dict[tuple[str, str], LemmaStats], dict[str, Any], str | None]:
    _schema, columns, morph_column = schema_and_columns(path)
    read_columns = ["tokens", "lemmas", "upostags"]
    if morph_column:
        read_columns.append(morph_column)

    stats: dict[tuple[str, str], LemmaStats] = {}
    total_tokens = 0
    malformed_documents = 0
    excluded_tokens = 0

    for doc_id, row in iter_rows(path, read_columns, batch_size):
        tokens = as_list(row["tokens"])
        lemmas = as_list(row["lemmas"])
        upos = as_list(row["upostags"])
        morphs = as_list(row[morph_column]) if morph_column else [""] * len(tokens)
        lengths = {len(tokens), len(lemmas), len(upos), len(morphs)}
        if len(lengths) != 1:
            malformed_documents += 1
            continue
        total_tokens += len(tokens)
        for orth_raw, lemma_raw, upos_raw, morph_raw in zip(tokens, lemmas, upos, morphs):
            orth = clean_text(orth_raw)
            lemma = clean_text(lemma_raw)
            tag = clean_text(upos_raw).upper()
            if not orth or not lemma or not tag or tag in excluded_upos:
                excluded_tokens += 1
                continue
            key = (lemma, tag)
            item = stats.get(key)
            if item is None:
                item = LemmaStats(lemma=lemma, upos=tag)
                stats[key] = item
            item.count += 1
            item.docs.add(doc_id)
            morph = compact_morph(morph_raw)
            item.morph[morph] += 1
            form = item.forms.get(orth)
            if form is None:
                form = FormStats()
                item.forms[orth] = form
            form.count += 1
            form.docs.add(doc_id)
            form.morph[morph] += 1

    diagnostics = {
        "documents": doc_id + 1 if "doc_id" in locals() else 0,
        "tokens": total_tokens,
        "lemma_upos_groups": len(stats),
        "malformed_documents_skipped": malformed_documents,
        "excluded_tokens": excluded_tokens,
        "morph_column": morph_column,
    }
    return stats, diagnostics, morph_column


def top_forms(item: LemmaStats, limit: int = 20) -> list[dict[str, Any]]:
    ordered = sorted(item.forms.items(), key=lambda pair: (-pair[1].count, pair[0].casefold(), pair[0]))
    out = []
    for orth, info in ordered[:limit]:
        out.append({
            "orth": orth,
            "count": info.count,
            "document_count": len(info.docs),
            "morph": [
                {"value": key, "count": count}
                for key, count in info.morph.most_common(5)
            ],
        })
    return out


def build_candidate(source: LemmaStats, target: LemmaStats) -> Candidate | None:
    if not probable_same_family(source, target):
        return None
    if target.count <= source.count:
        return None
    if target.form_count <= source.form_count:
        return None

    evidence: list[str] = []
    cautions: list[str] = []
    similarity = lemma_similarity(source.lemma, target.lemma)

    if source.form_count <= 2:
        evidence.append("Lemat źródłowy ma szczątkowy paradygmat: najwyżej dwie formy powierzchniowe.")
    else:
        cautions.append("Lemat źródłowy ma więcej niż dwie formy, więc może być samodzielnym leksemem.")

    richness_ratio = target.form_count / max(1, source.form_count)
    if richness_ratio >= 3:
        evidence.append("Lemat docelowy ma co najmniej trzykrotnie bogatszy obserwowany paradygmat.")

    if target.doc_count > source.doc_count:
        evidence.append("Lemat docelowy ma większe rozproszenie dokumentowe.")

    source_morph = set(source.morph)
    target_morph = set(target.morph)
    missing_in_target = sorted(source_morph - target_morph)
    if missing_in_target:
        evidence.append("Formy źródłowe realizują co najmniej jedną kategorię morfologiczną nieobecną w obserwowanym paradygmacie docelowym.")
    else:
        cautions.append("Kategorie morfologiczne źródła nie wypełniają wyraźnej luki w paradygmacie docelowym.")

    source_orths = set(source.forms)
    target_orths = set(target.forms)
    if source_orths & target_orths:
        cautions.append("Ta sama forma powierzchniowa występuje już z oboma lematami; możliwa jest rzeczywista wieloznaczność.")

    if similarity >= 0.84:
        evidence.append("Lematy są bardzo podobne znakowo.")
    else:
        cautions.append("Podobieństwo lematów jest umiarkowane, nie bardzo wysokie.")

    # Klasyfikacja jest oparta na jawnych przesłankach, nie na ukrytej sumie wag.
    strong_conditions = (
        source.form_count <= 2
        and richness_ratio >= 3
        and target.doc_count >= source.doc_count
        and not (source_orths & target_orths)
        and similarity >= 0.78
    )
    if strong_conditions and missing_in_target:
        classification = "strong_review_candidate"
    elif strong_conditions:
        classification = "review_candidate"
    else:
        classification = "weak_review_candidate"

    rules = []
    for orth, form in sorted(source.forms.items(), key=lambda pair: (-pair[1].count, pair[0].casefold())):
        rules.append({
            "orth": orth,
            "lemma": source.lemma,
            "upos": source.upos,
            "replacement": target.lemma,
            "reason": (
                f"Eksperymentalny kandydat: szczątkowy paradygmat {source.lemma!r} "
                f"może być odszczepioną częścią paradygmatu {target.lemma!r}."
            ),
            "status": "review",
            "observed_count": form.count,
            "observed_document_count": len(form.docs),
        })

    return Candidate(
        source_lemma=source.lemma,
        target_lemma=target.lemma,
        upos=source.upos,
        source_count=source.count,
        target_count=target.count,
        source_doc_count=source.doc_count,
        target_doc_count=target.doc_count,
        source_form_count=source.form_count,
        target_form_count=target.form_count,
        similarity=similarity,
        source_forms=top_forms(source),
        target_forms=top_forms(target),
        evidence=evidence,
        cautions=cautions,
        classification=classification,
        rules=rules,
    )


def candidate_sort_key(item: Candidate) -> tuple[Any, ...]:
    rank = {
        "strong_review_candidate": 0,
        "review_candidate": 1,
        "weak_review_candidate": 2,
    }.get(item.classification, 9)
    return (
        rank,
        -item.source_count,
        -item.source_doc_count,
        -item.similarity,
        item.source_lemma.casefold(),
        item.target_lemma.casefold(),
    )


def discover_candidates(
    stats: dict[tuple[str, str], LemmaStats],
    *,
    min_source_count: int,
    min_source_docs: int,
    max_source_forms: int,
    max_candidates_per_source: int,
) -> list[Candidate]:
    by_upos: dict[str, list[LemmaStats]] = defaultdict(list)
    for item in stats.values():
        by_upos[item.upos].append(item)

    results: list[Candidate] = []
    for upos, items in by_upos.items():
        sources = [
            item for item in items
            if item.count >= min_source_count
            and item.doc_count >= min_source_docs
            and item.form_count <= max_source_forms
        ]
        for source in sources:
            possible: list[Candidate] = []
            for target in items:
                candidate = build_candidate(source, target)
                if candidate is not None:
                    possible.append(candidate)
            possible.sort(key=candidate_sort_key)
            results.extend(possible[:max_candidates_per_source])

    # Jeden źródłowy lemat może mieć kilka podobnych kandydatów. Zachowujemy je do ręcznej oceny.
    results.sort(key=candidate_sort_key)
    return results


def collect_concordances(
    path: Path,
    candidates: list[Candidate],
    *,
    max_per_rule: int,
    context_tokens: int,
    batch_size: int,
) -> None:
    wanted: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        for rule in candidate.rules:
            wanted[(rule["orth"], rule["lemma"], rule["upos"])].append(candidate)
    if not wanted:
        return

    _schema, columns, _morph = schema_and_columns(path)
    read_columns = ["tokens", "lemmas", "upostags"]
    for optional in ("Oryginalna_nazwa_pliku", "Tytuł", "Autor"):
        if optional in columns:
            read_columns.append(optional)

    counts: Counter[tuple[str, str, str]] = Counter()
    for doc_id, row in iter_rows(path, read_columns, batch_size):
        tokens = as_list(row["tokens"])
        lemmas = as_list(row["lemmas"])
        upos = as_list(row["upostags"])
        if not (len(tokens) == len(lemmas) == len(upos)):
            continue
        for index, (orth_raw, lemma_raw, upos_raw) in enumerate(zip(tokens, lemmas, upos)):
            key = (clean_text(orth_raw), clean_text(lemma_raw), clean_text(upos_raw).upper())
            if key not in wanted or counts[key] >= max_per_rule:
                continue
            left = max(0, index - context_tokens)
            right = min(len(tokens), index + context_tokens + 1)
            context = " ".join(clean_text(token) for token in tokens[left:right])
            entry = {
                "doc_id": doc_id,
                "token_index": index,
                "context": context,
            }
            for optional in ("Oryginalna_nazwa_pliku", "Tytuł", "Autor"):
                if optional in row:
                    entry[optional] = clean_text(row[optional])
            for candidate in wanted[key]:
                candidate.concordances.append(entry)
            counts[key] += 1

        if wanted and all(counts[key] >= max_per_rule for key in wanted):
            break


def default_audit_paths(parquet: Path) -> tuple[Path, Path]:
    base = parquet.with_suffix("")
    return (
        base.with_name(base.name + ".lemma_audit.md"),
        base.with_name(base.name + ".lemma_audit.json"),
    )


def write_audit_report(path: Path, payload: Mapping[str, Any], report_limit: int) -> None:
    summary = payload["summary"]
    candidates = list(payload["candidates"])
    lines = [
        "# Audyt eksperymentalny lematyzacji Stanza",
        "",
        f"**Status:** `{payload['status']}`  ",
        f"**Wygenerowano:** `{payload['generated_at']}`  ",
        f"**Parquet:** `{payload['source']['path']}`  ",
        f"**SHA-256 źródła:** `{payload['source']['sha256']}`  ",
        "",
        "## Zastrzeżenie",
        "",
        "Raport wykrywa niespójności obserwowanych paradygmatów. Nie stanowi złotego standardu i sam nie dowodzi, że proponowany lemat jest poprawny. Wszystkie reguły mają domyślnie status `review`.",
        "",
        "## Podsumowanie",
        "",
        f"- Dokumenty: **{summary['documents']:,}**",
        f"- Tokeny: **{summary['tokens']:,}**",
        f"- Grupy lemma + UPOS: **{summary['lemma_upos_groups']:,}**",
        f"- Kandydaci mocni: **{summary['strong_candidates']:,}**",
        f"- Kandydaci zwykli: **{summary['review_candidates']:,}**",
        f"- Kandydaci słabi: **{summary['weak_candidates']:,}**",
        f"- Pominięte dokumenty z nierównoległymi tablicami: **{summary['malformed_documents_skipped']:,}**",
        "",
        "## Następny krok",
        "",
        "1. Uruchom komendę `prepare`.",
        "2. W pliku decyzji zmień `status` tylko wybranych reguł z `review` na `accept`.",
        "3. Uruchom `apply`, wskazując nową ścieżkę wyniku.",
        "",
        "## Kandydaci",
        "",
    ]

    if not candidates:
        lines.append("Nie znaleziono kandydatów przy bieżących warunkach audytu.")
    for index, candidate in enumerate(candidates[:report_limit], start=1):
        lines.extend([
            f"### {index}. `{candidate['source_lemma']}` → `{candidate['target_lemma']}` ({candidate['upos']})",
            "",
            f"**Klasyfikacja:** `{candidate['classification']}`  ",
            f"**Podobieństwo napisów:** `{candidate['similarity']:.3f}`  ",
            f"**Źródło:** {candidate['source_count']} wystąpień, {candidate['source_doc_count']} dokumentów, {candidate['source_form_count']} form  ",
            f"**Cel:** {candidate['target_count']} wystąpień, {candidate['target_doc_count']} dokumentów, {candidate['target_form_count']} form",
            "",
            "**Dowody:**",
        ])
        lines.extend(f"- {item}" for item in candidate["evidence"])
        if candidate["cautions"]:
            lines.append("")
            lines.append("**Ostrzeżenia:**")
            lines.extend(f"- {item}" for item in candidate["cautions"])
        lines.extend(["", "**Formy źródłowe:**"])
        for form in candidate["source_forms"]:
            morph = "; ".join(f"{x['value']} ({x['count']})" for x in form["morph"])
            lines.append(f"- `{form['orth']}`: {form['count']} wystąpień, {form['document_count']} dokumentów; {morph}")
        lines.extend(["", "**Najczęstsze formy docelowe:**"])
        for form in candidate["target_forms"][:12]:
            lines.append(f"- `{form['orth']}`: {form['count']} wystąpień, {form['document_count']} dokumentów")
        if candidate["concordances"]:
            lines.extend(["", "**Przykłady:**"])
            for example in candidate["concordances"][:8]:
                label = example.get("Oryginalna_nazwa_pliku") or example.get("Tytuł") or f"doc_id={example['doc_id']}"
                lines.append(f"- `{label}`: {example['context']}")
        lines.extend(["", "**Proponowane reguły:**"])
        for rule in candidate["rules"]:
            lines.append(
                f"- `{rule['orth']}` + `{rule['lemma']}` + `{rule['upos']}` → `{rule['replacement']}` "
                f"({rule['observed_count']} wystąpień; status: `{rule['status']}`)"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> int:
    parquet = ensure_parquet(args.parquet)
    report_default, json_default = default_audit_paths(parquet)
    report_path = Path(args.report).expanduser().resolve() if args.report else report_default
    json_path = Path(args.json).expanduser().resolve() if args.json else json_default
    excluded = {item.strip().upper() for item in args.exclude_upos.split(",") if item.strip()}

    stats, diagnostics, morph_column = collect_statistics(
        parquet,
        batch_size=args.batch_size,
        excluded_upos=excluded,
    )
    candidates = discover_candidates(
        stats,
        min_source_count=args.min_source_count,
        min_source_docs=args.min_source_docs,
        max_source_forms=args.max_source_forms,
        max_candidates_per_source=args.max_candidates_per_source,
    )
    collect_concordances(
        parquet,
        candidates[: args.concordance_candidate_limit],
        max_per_rule=args.examples_per_rule,
        context_tokens=args.context_tokens,
        batch_size=args.batch_size,
    )

    counts = Counter(item.classification for item in candidates)
    payload = {
        "schema_version": 1,
        "tool": "experimental_stanza_lemma_repair",
        "tool_version": SCRIPT_VERSION,
        "command": "audit",
        "status": "review_required",
        "generated_at": utc_now(),
        "source": {
            "path": str(parquet),
            "sha256": sha256_file(parquet),
            "bytes": parquet.stat().st_size,
        },
        "settings": {
            "min_source_count": args.min_source_count,
            "min_source_docs": args.min_source_docs,
            "max_source_forms": args.max_source_forms,
            "max_candidates_per_source": args.max_candidates_per_source,
            "excluded_upos": sorted(excluded),
            "morph_column": morph_column,
        },
        "summary": {
            **diagnostics,
            "candidates": len(candidates),
            "strong_candidates": counts["strong_review_candidate"],
            "review_candidates": counts["review_candidate"],
            "weak_candidates": counts["weak_review_candidate"],
        },
        "candidates": [item.to_dict() for item in candidates],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_audit_report(report_path, payload, args.report_limit)

    print(json.dumps({
        "success": True,
        "report": str(report_path),
        "json": str(json_path),
        "candidates": len(candidates),
        "strong_candidates": counts["strong_review_candidate"],
    }, ensure_ascii=False, indent=2))
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    audit_path = Path(args.audit_json).expanduser().resolve()
    if not audit_path.is_file():
        raise LemmaRepairError(f"Brak pliku audytu: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if audit.get("command") != "audit" or not isinstance(audit.get("candidates"), list):
        raise LemmaRepairError("Plik nie jest wynikiem komendy audit tego narzędzia.")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else audit_path.with_name(audit_path.name.replace(".lemma_audit.json", ".lemma_decisions.json"))
    )
    rules: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in audit["candidates"]:
        for rule in candidate.get("rules", []):
            key = (rule["orth"], rule["lemma"], rule["upos"], rule["replacement"])
            if key in seen:
                continue
            seen.add(key)
            rules.append({
                **rule,
                "candidate_classification": candidate.get("classification"),
                "source_form_count": candidate.get("source_form_count"),
                "target_form_count": candidate.get("target_form_count"),
                "status": "review",
            })

    payload = {
        "schema_version": 1,
        "tool": "experimental_stanza_lemma_repair",
        "tool_version": SCRIPT_VERSION,
        "command": "decisions",
        "generated_at": utc_now(),
        "source": audit["source"],
        "audit_json": str(audit_path),
        "instructions": "Zmień status wyłącznie zaakceptowanych reguł z 'review' na 'accept'. Dopuszczalne: review, accept, reject.",
        "rules": rules,
    }
    payload["config_sha256"] = sha256_json({"schema_version": 1, "source": payload["source"], "rules": rules})
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "output": str(output), "rules": len(rules)}, ensure_ascii=False, indent=2))
    return 0


def load_decisions(path: Path, source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise LemmaRepairError(f"Brak pliku decyzji: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("command") != "decisions" or payload.get("schema_version") != 1:
        raise LemmaRepairError("Nieobsługiwany plik decyzji.")
    allowed = {"review", "accept", "reject"}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise LemmaRepairError("Pole rules musi być listą.")
    for index, rule in enumerate(rules):
        status = rule.get("status")
        if status not in allowed:
            raise LemmaRepairError(f"rules[{index}].status ma niedozwoloną wartość: {status!r}")
        for key in ("orth", "lemma", "upos", "replacement", "reason"):
            if not isinstance(rule.get(key), str) or not rule[key]:
                raise LemmaRepairError(f"rules[{index}].{key} musi być niepustym tekstem")
    declared = payload.get("source") or {}
    actual_sha = sha256_file(source)
    if declared.get("sha256") and declared["sha256"] != actual_sha:
        raise LemmaRepairError(
            "SHA-256 wejściowego Parquet nie zgadza się z plikiem decyzji. "
            "Decyzje muszą być stosowane do dokładnie tego korpusu, który audytowano."
        )
    accepted = [rule for rule in rules if rule.get("status") == "accept"]
    if not accepted:
        raise LemmaRepairError("Brak reguł ze statusem 'accept'. Niczego nie zapisano.")
    return accepted, payload


def read_korpus_meta(schema: pa.Schema) -> dict[str, Any]:
    raw = (schema.metadata or {}).get(KORPUS_META_KEY)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        raise LemmaRepairError(f"Nie można odczytać korpus_meta: {exc}") from exc


def rewrite_with_rules(
    source: Path,
    output: Path,
    decisions_path: Path,
    accepted: list[dict[str, Any]],
    decision_payload: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if output == source:
        raise LemmaRepairError("Output musi być innym plikiem niż źródło.")
    if output.exists():
        raise LemmaRepairError(f"Output już istnieje: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_schema, columns, _morph = schema_and_columns(source)
    source_meta = read_korpus_meta(source_schema)
    physical_metadata = {k: v for k, v in (source_schema.metadata or {}).items() if k != KORPUS_META_KEY}
    physical_schema = source_schema.with_metadata(physical_metadata)

    rule_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rule in accepted:
        key = (rule["orth"], rule["lemma"], rule["upos"].upper())
        previous = rule_map.get(key)
        if previous and previous["replacement"] != rule["replacement"]:
            raise LemmaRepairError(f"Sprzeczne zaakceptowane reguły dla {key!r}")
        rule_map[key] = rule

    stage_data = output.with_name(output.name + ".lemma_repair_stage")
    stage_final = output.with_name(output.name + ".lemma_repair_final_stage")
    for stage in (stage_data, stage_final):
        if stage.exists():
            stage.unlink()

    base_tf: Counter[str] = Counter()
    corrected_by_rule: Counter[str] = Counter()
    affected_docs: set[int] = set()
    total_docs = 0
    total_tokens = 0
    changed_tokens = 0

    pf = pq.ParquetFile(source)
    writer = pq.ParquetWriter(stage_data, physical_schema, compression="snappy")
    try:
        doc_base = 0
        for batch in pf.iter_batches(batch_size=max(1, batch_size)):
            data = batch.to_pydict()
            for row_index in range(batch.num_rows):
                doc_id = doc_base + row_index
                tokens = as_list(data["tokens"][row_index])
                lemmas = as_list(data["lemmas"][row_index])
                upos = as_list(data["upostags"][row_index])
                if not (len(tokens) == len(lemmas) == len(upos)):
                    raise LemmaRepairError(
                        f"Nierównoległe tablice w doc_id={doc_id}: "
                        f"tokens={len(tokens)}, lemmas={len(lemmas)}, upostags={len(upos)}"
                    )
                mutable = list(lemmas)
                for pos, (orth_raw, lemma_raw, upos_raw) in enumerate(zip(tokens, lemmas, upos)):
                    key = (clean_text(orth_raw), clean_text(lemma_raw), clean_text(upos_raw).upper())
                    rule = rule_map.get(key)
                    if rule is None:
                        continue
                    replacement = rule["replacement"]
                    if mutable[pos] != replacement:
                        mutable[pos] = replacement
                        changed_tokens += 1
                        affected_docs.add(doc_id)
                        rule_id = "|".join((*key, replacement))
                        corrected_by_rule[rule_id] += 1
                data["lemmas"][row_index] = mutable
                base_tf.update(clean_text(value) for value in mutable)
                total_tokens += len(tokens)
            table = pa.Table.from_pydict(data, schema=physical_schema)
            writer.write_table(table)
            doc_base += batch.num_rows
        total_docs = doc_base
    except Exception:
        writer.close()
        pf.close()
        for stage in (stage_data, stage_final):
            try:
                if stage.exists():
                    stage.unlink()
            except OSError:
                pass
        raise
    else:
        writer.close()
        pf.close()

    unused = []
    for key, rule in rule_map.items():
        rule_id = "|".join((*key, rule["replacement"]))
        if corrected_by_rule[rule_id] == 0:
            unused.append(rule_id)
    if changed_tokens == 0:
        stage_data.unlink(missing_ok=True)
        raise LemmaRepairError("Żadna zaakceptowana reguła nie pasowała do wejściowego Parquet.")

    source_sha = sha256_file(source)
    repair_meta = {
        "enabled": True,
        "schema_version": 1,
        "tool_version": SCRIPT_VERSION,
        "generated_at": utc_now(),
        "source_parquet_path": str(source),
        "source_parquet_sha256": source_sha,
        "decisions_path": str(decisions_path),
        "decisions_sha256": sha256_file(decisions_path),
        "accepted_rules": len(accepted),
        "corrected_tokens": changed_tokens,
        "affected_documents": len(affected_docs),
        "counts_by_rule": dict(sorted(corrected_by_rule.items())),
        "unused_accepted_rules": sorted(unused),
    }
    final_meta = dict(source_meta)
    final_meta["base_tf"] = dict(sorted(base_tf.items()))
    final_meta["total_tokens"] = total_tokens
    final_meta["experimental_lemma_repairs"] = repair_meta

    final_metadata = dict(physical_metadata)
    final_metadata[KORPUS_META_KEY] = json.dumps(final_meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    final_schema = source_schema.with_metadata(final_metadata)

    src_stage = pq.ParquetFile(stage_data)
    writer2 = pq.ParquetWriter(stage_final, final_schema, compression="snappy")
    try:
        for batch in src_stage.iter_batches(batch_size=max(1, batch_size)):
            table = pa.Table.from_batches([batch], schema=final_schema)
            writer2.write_table(table)
    finally:
        writer2.close()
        src_stage.close()

    check = pq.ParquetFile(stage_final)
    try:
        if int(check.metadata.num_rows) != total_docs:
            raise LemmaRepairError("Walidacja wyniku: zmieniła się liczba dokumentów.")
        checked_meta = read_korpus_meta(check.schema_arrow)
        if int(checked_meta.get("total_tokens", -1)) != total_tokens:
            raise LemmaRepairError("Walidacja wyniku: niespójne total_tokens.")
        if sum(int(value) for value in checked_meta.get("base_tf", {}).values()) != total_tokens:
            raise LemmaRepairError("Walidacja wyniku: suma base_tf nie zgadza się z liczbą tokenów.")
        stored_repair = checked_meta.get("experimental_lemma_repairs") or {}
        if int(stored_repair.get("corrected_tokens", -1)) != changed_tokens:
            raise LemmaRepairError("Walidacja wyniku: niespójny licznik korekt.")
    finally:
        check.close()

    os.replace(stage_final, output)
    stage_data.unlink(missing_ok=True)
    result_sha = sha256_file(output)
    return {
        "success": True,
        "source": str(source),
        "source_sha256": source_sha,
        "output": str(output),
        "output_sha256": result_sha,
        "documents": total_docs,
        "tokens": total_tokens,
        "accepted_rules": len(accepted),
        "corrected_tokens": changed_tokens,
        "affected_documents": len(affected_docs),
        "counts_by_rule": dict(sorted(corrected_by_rule.items())),
        "unused_accepted_rules": sorted(unused),
        "rebuild_required": [str(output.with_suffix(".search")), str(output.with_suffix(".dep_cache"))],
    }


def write_apply_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Eksperymentalna korekta lematów",
        "",
        "**Status:** `success`  ",
        f"**Źródło:** `{result['source']}`  ",
        f"**Wynik:** `{result['output']}`  ",
        f"**SHA-256 wyniku:** `{result['output_sha256']}`",
        "",
        "## Podsumowanie",
        "",
        f"- Dokumenty: **{result['documents']:,}**",
        f"- Tokeny: **{result['tokens']:,}**",
        f"- Zaakceptowane reguły: **{result['accepted_rules']:,}**",
        f"- Poprawione tokeny: **{result['corrected_tokens']:,}**",
        f"- Zmienione dokumenty: **{result['affected_documents']:,}**",
        "",
        "## Zastosowane reguły",
        "",
    ]
    for key, count in result["counts_by_rule"].items():
        lines.append(f"- `{key}`: **{count}**")
    if result["unused_accepted_rules"]:
        lines.extend(["", "## Zaakceptowane reguły bez dopasowań", ""])
        lines.extend(f"- `{item}`" for item in result["unused_accepted_rules"])
    lines.extend([
        "",
        "## Wymagane po korekcie",
        "",
        "Dla nowego Parquetu zbuduj nową parę `.search` i `.dep_cache`. Nie kopiuj artefaktów starego korpusu.",
        "",
        f"```powershell\npython -m korpusuj.index.cli create \"{result['output']}\" --progress on --pretty\n```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_apply(args: argparse.Namespace) -> int:
    source = ensure_parquet(args.parquet)
    decisions = Path(args.decisions).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".parquet":
        raise LemmaRepairError("--output musi kończyć się rozszerzeniem .parquet")
    accepted, payload = load_decisions(decisions, source)
    result = rewrite_with_rules(
        source,
        output,
        decisions,
        accepted,
        payload,
        batch_size=args.batch_size,
    )
    report = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output.with_name(output.stem + ".lemma_repair_report.md")
    )
    write_apply_report(report, result)
    result = dict(result)
    result["report"] = str(report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experimental_stanza_lemma_repair.py",
        description="Audyt rozszczepionych paradygmatów i kontrolowana korekta lematów w gotowym Parquet Korpusuj.",
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Wykryj kandydatów i utwórz raport; nie modyfikuje korpusu.")
    audit.add_argument("--parquet", required=True)
    audit.add_argument("--report")
    audit.add_argument("--json")
    audit.add_argument("--batch-size", type=int, default=128)
    audit.add_argument("--min-source-count", type=int, default=2)
    audit.add_argument("--min-source-docs", type=int, default=2)
    audit.add_argument("--max-source-forms", type=int, default=2)
    audit.add_argument("--max-candidates-per-source", type=int, default=3)
    audit.add_argument("--examples-per-rule", type=int, default=4)
    audit.add_argument("--context-tokens", type=int, default=12)
    audit.add_argument("--concordance-candidate-limit", type=int, default=300)
    audit.add_argument("--report-limit", type=int, default=DEFAULT_REPORT_LIMIT)
    audit.add_argument(
        "--exclude-upos",
        default=",".join(sorted(DEFAULT_EXCLUDED_UPOS)),
        help="Lista UPOS pomijanych przy indukcji paradygmatów.",
    )
    audit.set_defaults(func=run_audit)

    prepare = sub.add_parser("prepare", help="Utwórz edytowalny plik decyzji z audytu.")
    prepare.add_argument("--audit-json", required=True)
    prepare.add_argument("--output")
    prepare.set_defaults(func=run_prepare)

    apply_cmd = sub.add_parser("apply", help="Zastosuj reguły accept do nowego Parquetu.")
    apply_cmd.add_argument("--parquet", required=True)
    apply_cmd.add_argument("--decisions", required=True)
    apply_cmd.add_argument("--output", required=True)
    apply_cmd.add_argument("--report")
    apply_cmd.add_argument("--batch-size", type=int, default=128)
    apply_cmd.set_defaults(func=run_apply)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LemmaRepairError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: Przerwano przez użytkownika.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
