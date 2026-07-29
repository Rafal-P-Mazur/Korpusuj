"""Shared search-frequency and statistics computation for GUI and CLI callers."""
from __future__ import annotations

"""Search statistics helpers.

This module is the first boundary extracted from ``engine.py`` for the search
result/statistics pipeline. It intentionally starts small: at this migration
stage it owns only pure/stateless helpers and a data container used to describe
where search statistics should live next.

The current GUI still computes most frequency tables in ``engine.py``. Future
steps should move those computations here after tests confirm that concordance
results can remain stable while statistics are updated separately.
"""

from dataclasses import dataclass, field
from numbers import Number
from typing import Any


@dataclass
class SearchStatistics:
    """Container for search statistics computed after concordance hits."""

    true_monthly_totals: dict[str, int] = field(default_factory=dict)
    monthly_freq_for_use: dict[str, Any] = field(default_factory=dict)
    monthly_tfidf_for_use: dict[str, Any] = field(default_factory=dict)
    monthly_zscore_for_use: dict[str, Any] = field(default_factory=dict)

    fq_data: list[Any] = field(default_factory=list)
    fq_data_token: list[Any] = field(default_factory=list)
    fq_data_month: list[Any] = field(default_factory=list)

    s_lemma_total_freq: list[Any] = field(default_factory=list)
    s_lemma_global_pmw: list[Any] = field(default_factory=list)
    s_lemma_global_tfidf: list[Any] = field(default_factory=list)
    s_lemma_monthly_trends: list[Any] = field(default_factory=list)
    s_lemma_monthly_tfidf: list[Any] = field(default_factory=list)
    s_lemma_monthly_zscore: list[Any] = field(default_factory=list)

    has_dates: bool = False


def normalize_monthly_token_counts_for_search(raw_monthly_counts: Any) -> tuple[list[tuple[str, str, int]], dict[str, int]]:
    """Normalize corpus monthly token counts used by search statistics.

    Accepted input shapes mirror the historical ``engine.py`` implementation:
    - flat mapping: ``{"YYYY-MM": count}``
    - nested mapping: ``{YYYY: {MM: count}}``
    """

    import re

    true_monthly_totals: dict[str, int] = {}
    flattened: list[tuple[str, str, int]] = []

    def add_month_count(year: Any, month: Any, count: Any) -> None:
        try:
            y = int(year)
            m = int(month)
            c = int(count or 0)
        except Exception:
            return
        key = f"{y}-{m}"
        true_monthly_totals[key] = true_monthly_totals.get(key, 0) + c
        flattened.append((str(y), str(m), c))

    if isinstance(raw_monthly_counts, dict):
        for year_or_key, months_or_count in raw_monthly_counts.items():
            if isinstance(months_or_count, Number):
                match = re.match(r"^(\d{4})[-/](\d{1,2})$", str(year_or_key))
                if match:
                    add_month_count(match.group(1), match.group(2), months_or_count)
                continue

            if isinstance(months_or_count, dict):
                for month, count in months_or_count.items():
                    add_month_count(year_or_key, month, count)

    return flattened, true_monthly_totals


@dataclass
class SearchFrequencyInputs:
    """Intermediate frequency inputs derived from concordance hits."""

    unique_matched_tokens: dict[Any, int] = field(default_factory=dict)
    unique_lemmas: set[Any] = field(default_factory=set)
    monthly_lemma_freq: dict[str, dict[Any, int]] = field(default_factory=dict)
    exact_orth_df: dict[Any, set[Any]] = field(default_factory=dict)
    exact_lemma_df: dict[Any, set[Any]] = field(default_factory=dict)


def collect_search_frequency_inputs(results_sorted: Any) -> SearchFrequencyInputs:
    """Collect first-stage frequency inputs from sorted concordance results.
    
    The function prepares dictionaries used by PMW, TF-IDF and z-score calculations and accepts both established tuple rows and schema-v1 result dictionaries.
    """
    import re

    from datetime import datetime, timedelta

    def month_from_text(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"(\d{4})[-/.](\d{1,2})", text)
        if match:
            return f"{int(match.group(1)):04d}-{int(match.group(2))}"
        match = re.search(r"\b(\d{4})(\d{2})\b", text)
        if match:
            return f"{int(match.group(1)):04d}-{int(match.group(2))}"
        return text

    def normalize_lemma_value(value: Any, fallback: Any) -> Any:
        if value is None or value == "":
            return fallback
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return value

    def row_values(row: Any) -> tuple[Any, Any, Any, Any]:
        """Return matched_text, matched_lemmas, month_key, row_idx."""
        if isinstance(row, dict):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            matched_text = row.get("match_text") or metadata.get("matched_text") or ""
            matched_lemmas = normalize_lemma_value(metadata.get("matched_lemmas"), matched_text)
            month_key = (
                metadata.get("month_key")
                or metadata.get("month")
                or metadata.get("Data publikacji")
                or metadata.get("date")
                or metadata.get("publication_date")
                or ""
            )
            month_key = month_from_text(month_key)
            row_idx = row.get("doc_id", row.get("row_idx", row.get("source_doc_id", None)))
            return matched_text, matched_lemmas, month_key, row_idx

        try:
            (
                publication_date,
                context,
                full_text,
                matched_text,
                matched_lemmas,
                month_key,
                title,
                author,
                additional_metadata,
                left_context,
                right_context,
                row_idx,
                start_idx_val,
                end_idx_val,
            ) = row
            return matched_text, matched_lemmas, month_key, row_idx
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Unsupported search result row shape: {type(row).__name__}: {exc}") from exc

    out = SearchFrequencyInputs()

    for row in results_sorted or []:
        matched_text, matched_lemmas, month_key, row_idx = row_values(row)
        matched_text = matched_text if matched_text is not None else ""
        matched_lemmas = matched_lemmas if matched_lemmas is not None else matched_text

        out.exact_orth_df.setdefault(matched_text, set()).add(row_idx)
        out.exact_lemma_df.setdefault(matched_lemmas, set()).add(row_idx)

        token_key = matched_text
        out.unique_matched_tokens[token_key] = out.unique_matched_tokens.get(token_key, 0) + 1
        out.unique_lemmas.add(matched_lemmas)

        try:
            year, month_val = str(month_key).split("-")[:2]
            normalized_key = f"{year}-{int(month_val)}"
        except Exception:
            normalized_key = str(month_key or "")

        if normalized_key:
            if normalized_key not in out.monthly_lemma_freq:
                out.monthly_lemma_freq[normalized_key] = {}
            out.monthly_lemma_freq[normalized_key][matched_lemmas] = out.monthly_lemma_freq[normalized_key].get(
                matched_lemmas, 0
            ) + 1

    date_keys = []
    for key in out.monthly_lemma_freq.keys():
        try:
            year, month = map(int, str(key).split("-")[:2])
            date_keys.append(datetime(year, month, 1))
        except Exception:
            continue

    if date_keys:
        start_date = min(date_keys)
        end_date = max(date_keys)
        current_date = start_date
        while current_date <= end_date:
            key = f"{current_date.year}-{current_date.month}"
            if key not in out.monthly_lemma_freq:
                out.monthly_lemma_freq[key] = {lemma: 0 for lemma in out.unique_lemmas}
            current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)

    return out

@dataclass
class GlobalFrequencyTables:
    """Global frequency/statistics tables derived from frequency inputs."""

    fq_data_token: list[Any] = field(default_factory=list)
    fq_data: list[Any] = field(default_factory=list)
    s_lemma_total_freq: list[Any] = field(default_factory=list)
    s_lemma_global_pmw: list[Any] = field(default_factory=list)
    s_lemma_global_tfidf: list[Any] = field(default_factory=list)


def build_global_frequency_tables(
    *,
    unique_matched_tokens: dict[Any, int],
    monthly_lemma_freq: dict[str, dict[Any, int]],
    exact_orth_df: dict[Any, set[Any]],
    exact_lemma_df: dict[Any, set[Any]],
    total_token_count: int | float,
    total_docs: int | float,
    df_for_matched_key,
) -> GlobalFrequencyTables:
    """Build global token/lemma frequency tables for search statistics.

    This mirrors the global frequency-table block previously embedded in
    ``engine.py::search_thread``. It intentionally accepts ``df_for_matched_key``
    as a callback so the current engine-specific exact/phrase document-frequency
    behavior remains unchanged.
    """

    import math

    out = GlobalFrequencyTables()

    for idx, (token, frequency) in enumerate(
        sorted(unique_matched_tokens.items(), key=lambda x: x[1], reverse=True), start=1
    ):
        frequency_normalized = (frequency / total_token_count) * 1_000_000 if total_token_count > 0 else 0.0
        tf_global = (frequency / total_token_count) if total_token_count > 0 else 0
        df_val_orth = df_for_matched_key(token, "orth", exact_orth_df)
        idf_global = math.log10(total_docs / df_val_orth) if df_val_orth > 0 else 0
        global_tfidf_orth = tf_global * idf_global * 100000
        out.fq_data_token.append([
            idx,
            token,
            frequency,
            round(frequency_normalized, 2),
            df_val_orth,
            round(global_tfidf_orth, 2),
        ])

    lemma_total_freq: dict[Any, int] = {}
    for month_data in monthly_lemma_freq.values():
        for lemma, count in month_data.items():
            lemma_total_freq[lemma] = lemma_total_freq.get(lemma, 0) + count

    out.s_lemma_total_freq = sorted(lemma_total_freq.items(), key=lambda x: x[1], reverse=True)

    for idx, (lemma, frequency) in enumerate(out.s_lemma_total_freq, start=1):
        frequency_normalized = (frequency / total_token_count) * 1_000_000 if total_token_count > 0 else 0.0
        tf_global = (frequency / total_token_count) if total_token_count > 0 else 0
        df_val_lemma = df_for_matched_key(lemma, "base", exact_lemma_df)
        idf_global = math.log10(total_docs / df_val_lemma) if df_val_lemma > 0 else 0
        global_tfidf_lemma = tf_global * idf_global * 100000
        out.fq_data.append([
            idx,
            lemma,
            frequency,
            round(frequency_normalized, 2),
            df_val_lemma,
            round(global_tfidf_lemma, 2),
        ])

    out.s_lemma_global_pmw = sorted([(r[1], r[3]) for r in out.fq_data], key=lambda x: x[1], reverse=True)
    out.s_lemma_global_tfidf = sorted([(r[1], r[5]) for r in out.fq_data], key=lambda x: x[1], reverse=True)

    return out


@dataclass
class MonthlyFrequencyTables:
    """Monthly frequency/statistics tables derived from search frequency inputs."""

    monthly_freq_for_use: dict[str, dict[Any, float]] = field(default_factory=dict)
    monthly_tfidf_for_use: dict[str, dict[Any, float]] = field(default_factory=dict)
    monthly_zscore_for_use: dict[str, dict[Any, float]] = field(default_factory=dict)
    fq_data_month: list[Any] = field(default_factory=list)


def build_monthly_frequency_tables(
    *,
    monthly_lemma_freq: dict[str, dict[Any, int]],
    unique_lemmas: set[Any],
    true_monthly_totals: dict[str, int],
    total_docs: int | float,
    exact_lemma_df: dict[Any, set[Any]],
    df_for_matched_key,
    calc_z_score_func,
) -> MonthlyFrequencyTables:
    """Build monthly PMW/TF-IDF/z-score tables for search statistics.

    This mirrors the monthly statistics block previously embedded in
    ``engine.py::search_thread``. The engine-specific document-frequency and
    z-score behavior are preserved through callbacks.
    """

    import math
    import numpy as np

    out = MonthlyFrequencyTables()

    def valid_calendar_month_key(value: Any) -> bool:
        """Return True only for a real calendar month encoded as YYYY-M(M)."""
        import re

        match = re.fullmatch(r"(\d{4})-(\d{1,2})", str(value or "").strip())
        if match is None:
            return False
        month = int(match.group(2))
        return 1 <= month <= 12

    # Missing publication dates are valid corpus data. They may be represented
    # by buckets such as "Unknown", but those buckets have no place on a time
    # axis and must not abort concordance display or affect monthly z-scores.
    dated_monthly_lemma_freq = {
        month_key: lemma_counts
        for month_key, lemma_counts in monthly_lemma_freq.items()
        if valid_calendar_month_key(month_key)
    }

    for month_key, lemma_counts in dated_monthly_lemma_freq.items():
        total = true_monthly_totals.get(month_key, 0)
        if total > 0:
            out.monthly_freq_for_use[month_key] = {
                lemma: (count / total) * 1_000_000 for lemma, count in lemma_counts.items()
            }
        else:
            out.monthly_freq_for_use[month_key] = {lemma: 0.0 for lemma in lemma_counts}

    # Szybki bufor DF: frazy z faktycznych trafień, pojedyncze lematy z indeksu globalnego.
    global_lemma_df_cache: dict[Any, int | float] = {}
    for lemma in unique_lemmas:
        global_lemma_df_cache[lemma] = df_for_matched_key(lemma, "base", exact_lemma_df)

    for month_key, lemma_counts in dated_monthly_lemma_freq.items():
        total = true_monthly_totals.get(month_key, 0)
        out.monthly_tfidf_for_use[month_key] = {}
        for lemma, count in lemma_counts.items():
            tf = (count / total) if total > 0 else 0
            df_val = global_lemma_df_cache.get(lemma, 1)
            idf = math.log10(total_docs / df_val) if df_val > 0 else 0
            out.monthly_tfidf_for_use[month_key][lemma] = tf * idf * 100000

    lemma_norm_values = {lemma: [] for lemma in unique_lemmas}
    for month_key in dated_monthly_lemma_freq.keys():
        for lemma in unique_lemmas:
            lemma_norm_values[lemma].append(out.monthly_freq_for_use[month_key].get(lemma, 0.0))

    lemma_stats = {}
    for lemma, vals in lemma_norm_values.items():
        mean_val = np.mean(vals) if vals else 0.0
        std_val = np.std(vals) if vals else 0.0
        lemma_stats[lemma] = (mean_val, std_val)

    for month_key in dated_monthly_lemma_freq.keys():
        out.monthly_zscore_for_use[month_key] = {}
        for lemma in unique_lemmas:
            val = out.monthly_freq_for_use[month_key].get(lemma, 0.0)
            mean_val, std_val = lemma_stats[lemma]
            out.monthly_zscore_for_use[month_key][lemma] = calc_z_score_func(val, mean_val, std_val)

    sorted_month_keys = sorted(
        dated_monthly_lemma_freq.keys(),
        key=lambda k: (int(k.split("-")[0]), int(k.split("-")[1])),
    )
    for month_key in sorted_month_keys:
        year_str, month_str = month_key.split("-")
        raw_counts = dated_monthly_lemma_freq[month_key]
        norm_counts = out.monthly_freq_for_use[month_key]
        for lemma in sorted(raw_counts.keys()):
            raw = raw_counts[lemma]
            norm = norm_counts.get(lemma, 0.0)
            tfidf = out.monthly_tfidf_for_use[month_key].get(lemma, 0.0)
            zscore = out.monthly_zscore_for_use[month_key].get(lemma) or 0.0
            out.fq_data_month.append([
                int(year_str),
                int(month_str),
                lemma,
                raw,
                round(norm, 2) if isinstance(norm, (int, float)) else None,
                round(tfidf, 2) if isinstance(tfidf, (int, float)) else None,
                round(zscore, 2) if isinstance(zscore, (int, float)) else None,
            ])

    return out
