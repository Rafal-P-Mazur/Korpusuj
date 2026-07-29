"""Corpus information helpers for Korpusuj.

Pure or nearly-pure helpers used by the corpus information UI.
No customtkinter/tkinter/messagebox/application globals here.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CORPUS_INFO_EXCLUDE_COLS = {
    "Data publikacji", "Tytuł", "Autor", "tags", "Treść", "token_counts",
    "tokens", "lemmas", "deprels", "postags", "full_postags",
    "word_ids", "sentence_ids", "head_ids", "start_ids", "end_ids", "ners", "upostags",
    "corefs", "srl", "srls", "srl_frames",
}

TECHNICAL_PARQUET_INDEX_COLS = {"__index_level_0__", "__index_level_1__", "index", "level_0"}


@dataclass(frozen=True)
class CorpusInfoModel:
    total_docs: int
    total_tokens: int
    unique_lemmas: int
    unique_orths: int
    date_range: str
    monthly_stats_str: str
    meta_cols: list[str]
    meta_str: str


def safe_len_or_zero(obj) -> int:
    try:
        return len(obj)
    except Exception:
        return 0


def safe_total_tokens_for_corpus_info(inv_idx, df=None) -> int:
    try:
        if isinstance(inv_idx, dict):
            val = inv_idx.get("total_tokens", 0)
            if val:
                return int(val)
    except Exception:
        pass
    try:
        val = getattr(inv_idx, "total_tokens", 0)
        if val:
            return int(val)
    except Exception:
        pass
    try:
        val = getattr(df, "total_tokens", 0)
        if val:
            return int(val)
    except Exception:
        pass
    try:
        if df is not None and "tokens" in df.columns:
            total = 0
            for tokens in df["tokens"]:
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                total += len(tokens or [])
            return int(total)
    except Exception:
        pass
    return 0


def safe_unique_values_from_df_for_corpus_info(df, column_name) -> int:
    try:
        if df is None or column_name not in df.columns:
            return 0
        values = set()
        for item in df[column_name]:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, (list, tuple, set)):
                values.update(item)
            elif item is not None:
                values.add(item)
        return len(values)
    except Exception:
        return 0


def safe_lazy_term_index_count_for_corpus_info(term_index, inv_idx=None, attr=None, df=None, df_column=None) -> int:
    """Return unique term count for dict indexes and Korpusuj LazyTermIndex.

    Real contract from korpusuj.index.sqlite_index / builder:
    - LazyTermIndex has: index_path, attr, _idx and methods _open(), get(), __contains__().
    - SQLite table terms has one row per unique (attr, value):
      terms(attr TEXT, value TEXT, df INTEGER, cf INTEGER, postings BLOB,
            PRIMARY KEY (attr, value)).
    Therefore unique lemma/orth count is:
      SELECT COUNT(*) FROM terms WHERE attr=?
    """
    if term_index is None:
        return 0

    # Old in-memory dict index: len(dict) is the desired number of unique terms.
    try:
        return len(term_index)
    except Exception:
        pass

    attr_value = attr
    try:
        attr_value = attr_value or getattr(term_index, "attr", None)
    except Exception:
        pass

    # Precise LazyTermIndex path-based contract.
    index_path = None
    for path_attr in ("index_path", "search_path"):
        try:
            value = getattr(term_index, path_attr, None)
            if value:
                index_path = value
                break
        except Exception:
            pass

    if index_path and attr_value:
        try:
            import sqlite3
            with sqlite3.connect(str(index_path)) as con:
                row = con.execute("SELECT COUNT(*) FROM terms WHERE attr=?", (attr_value,)).fetchone()
                return int(row[0] or 0) if row is not None else 0
        except Exception:
            pass

    # Precise LazyTermIndex _open/_idx/SearchIndex connection contract.
    try:
        opener = getattr(term_index, "_open", None)
        if callable(opener):
            opener()
        idx = getattr(term_index, "_idx", None)
        con = getattr(idx, "con", None)
        if con is not None and attr_value:
            row = con.execute("SELECT COUNT(*) FROM terms WHERE attr=?", (attr_value,)).fetchone()
            return int(row[0] or 0) if row is not None else 0
    except Exception:
        pass

    # Generic count-like fallbacks for future wrappers.
    for name in ("count", "size", "term_count", "num_terms", "n_terms"):
        try:
            obj = getattr(term_index, name, None)
            if callable(obj):
                value = obj()
                if value is not None:
                    return int(value)
            elif obj is not None:
                return int(obj)
        except Exception:
            pass

    # Materialized DataFrame fallback.
    return safe_unique_values_from_df_for_corpus_info(df, df_column)


def parse_year_month_for_corpus_info(year_value, month_value=None):
    try:
        if isinstance(year_value, bytes):
            year_value = year_value.decode("utf-8", errors="replace")
        if isinstance(month_value, bytes):
            month_value = month_value.decode("utf-8", errors="replace")
    except Exception:
        pass
    if isinstance(year_value, (tuple, list)) and len(year_value) >= 2:
        try:
            return int(year_value[0]), int(year_value[1])
        except Exception:
            return None
    year_text = str(year_value or "").strip()
    month_text = str(month_value or "").strip() if month_value is not None else ""
    if month_text:
        try:
            return int(year_text), int(month_text)
        except Exception:
            pass
    try:
        import re
        m = re.match(r"^\s*(\d{4})(?:[-/.](\d{1,2}))?", year_text)
        if m:
            year = int(m.group(1))
            month = int(m.group(2) or 1)
            if 1 <= month <= 12:
                return year, month
    except Exception:
        pass
    return None


def normalize_monthly_counts_for_corpus_info(monthly_counts) -> list[tuple[int, int, int]]:
    normalized = []
    try:
        items = monthly_counts.items()
    except Exception:
        return normalized
    for y_key, value in items:
        if isinstance(value, dict):
            for m_key, count in value.items():
                ym = parse_year_month_for_corpus_info(y_key, m_key)
                if ym is None:
                    continue
                try:
                    normalized.append((ym[0], ym[1], int(count or 0)))
                except Exception:
                    normalized.append((ym[0], ym[1], 0))
        else:
            ym = parse_year_month_for_corpus_info(y_key)
            if ym is None:
                continue
            try:
                normalized.append((ym[0], ym[1], int(value or 0)))
            except Exception:
                normalized.append((ym[0], ym[1], 0))
    merged = {}
    for y, m, count in normalized:
        merged[(y, m)] = merged.get((y, m), 0) + count
    return [(y, m, count) for (y, m), count in sorted(merged.items())]


def get_corpus_metadata_columns(df, exclude_cols=None) -> list[str]:
    exclude = set(DEFAULT_CORPUS_INFO_EXCLUDE_COLS if exclude_cols is None else exclude_cols)
    out = []
    try:
        columns = list(df.columns)
    except Exception:
        return out
    for c in columns:
        c_str = str(c)
        if c in exclude:
            continue
        if c in TECHNICAL_PARQUET_INDEX_COLS:
            continue
        if c_str.startswith("__index_level_"):
            continue
        out.append(c)
    return out


def build_corpus_info_model(df, inv_idx) -> CorpusInfoModel:
    total_docs = safe_len_or_zero(df)
    total_tokens = safe_total_tokens_for_corpus_info(inv_idx, df)
    get_attr = inv_idx.get if hasattr(inv_idx, "get") else (lambda key, default=None: default)

    unique_lemmas = safe_lazy_term_index_count_for_corpus_info(
        get_attr("base", {}), inv_idx=inv_idx, attr="base", df=df, df_column="lemmas"
    )
    unique_orths = safe_lazy_term_index_count_for_corpus_info(
        get_attr("orth", {}), inv_idx=inv_idx, attr="orth", df=df, df_column="tokens"
    )

    date_range = "Brak danych o dacie"
    monthly_stats_str = ""
    monthly_counts = get_attr("monthly_token_counts", {}) or {}
    monthly_items = normalize_monthly_counts_for_corpus_info(monthly_counts)
    monthly_stats_list = []
    dates = []
    for y, m, count in monthly_items:
        dates.append((int(y), int(m)))
        monthly_stats_list.append(f"  • {int(m):02d}.{int(y)}: {int(count):,} tokenów")
    if dates:
        min_d = min(dates)
        max_d = max(dates)
        date_range = f"{min_d[1]:02d}.{min_d[0]} - {max_d[1]:02d}.{max_d[0]}"
    if monthly_stats_list:
        monthly_stats_str = "LICZBA TOKENÓW NA MIESIĄC:\n" + "\n".join(monthly_stats_list)

    meta_cols = get_corpus_metadata_columns(df)
    meta_str = "\n  • ".join(meta_cols) if meta_cols else "  Brak dodatkowych metadanych"
    return CorpusInfoModel(
        total_docs=int(total_docs), total_tokens=int(total_tokens),
        unique_lemmas=int(unique_lemmas), unique_orths=int(unique_orths),
        date_range=date_range, monthly_stats_str=monthly_stats_str,
        meta_cols=list(meta_cols), meta_str=meta_str,
    )
