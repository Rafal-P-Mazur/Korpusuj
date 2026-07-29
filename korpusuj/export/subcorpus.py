"""Helpery eksportu podkorpusów Korpusuj.

Moduł zawiera czystą logikę danych dla eksportu podkorpusów:
- wybór wierszy z wyników wyszukiwania,
- filtrowanie po metadanych,
- przeliczanie metadanych frekwencyjnych,
- zapis Parquet z metadanymi schematu.

Nie zawiera kodu GUI: filedialog, messagebox, CTk, threading i app.after zostają w engine.py.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Any

import pandas as pd


# --- KORPUSUJ_MIGRATION_023_SUBCORPUS_HELPERS ---


def _as_plain_list(value):
    if value is None:
        return []
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _extract_result_row_position(result) -> int | None:
    """
    Zwraca pozycyjny row_idx/doc_id z wyniku wyszukiwania.

    Kontrakt engine/SearchCursor:
        result[11] = row_idx / doc_id / pozycja dokumentu w DataFrame
    Nie jest to bezpiecznie etykieta df.index.
    """
    try:
        if isinstance(result, dict):
            for key in ("row_idx", "row_index", "doc_id"):
                if key in result:
                    return int(result[key])
            return None
        return int(result[11])
    except Exception:
        return None


def _ensure_pandas_dataframe(df: Any) -> pd.DataFrame:
    """Return a pandas DataFrame for subcorpus export helper operations.

    GUI search can keep corpora as LazyCorpus objects. Subcorpus export needs
    DataFrame semantics because it filters columns and writes a Parquet corpus
    artifact. If an object exposes a real class-defined materialize() method,
    call it explicitly. This avoids speculative getattr/hasattr calls that can
    trigger LazyCorpus.__getattr__ unexpectedly.
    """
    if isinstance(df, pd.DataFrame):
        return df

    materialize = None
    try:
        for cls in type(df).mro():
            candidate = cls.__dict__.get("materialize")
            if callable(candidate):
                materialize = candidate
                break
    except Exception:
        materialize = None

    if materialize is not None:
        materialized = materialize(df)
        if not isinstance(materialized, pd.DataFrame):
            raise TypeError(
                "subcorpus export expected materialize() to return pandas.DataFrame; "
                f"got {type(materialized).__name__}"
            )
        return materialized

    if hasattr(df, "iloc") and hasattr(df, "columns"):
        return df

    raise TypeError(f"subcorpus export requires pandas.DataFrame-like object; got {type(df).__name__}")


def select_rows_from_search_results(df: pd.DataFrame, results: Iterable[Any]) -> pd.DataFrame:
    """
    Wybiera dokumenty z DataFrame na podstawie wyników wyszukiwania.

    WAŻNE: row_idx z wyników wyszukiwania jest traktowany jako pozycja/doc_id,
    więc używamy df.iloc, nie df.loc. To zabezpiecza realne korpusy Parquet,
    które mogą mieć nieciągły indeks etykietowy.
    """
    df = _ensure_pandas_dataframe(df)
    positions = []

    for result in results or []:
        pos = _extract_result_row_position(result)
        if pos is None:
            continue
        if 0 <= pos < len(df):
            positions.append(pos)

    # Zachowaj kolejność pierwszego wystąpienia i usuń duplikaty.
    seen = set()
    unique_positions = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)

    return df.iloc[unique_positions].copy()


def filter_dataframe_by_metadata(
    df: pd.DataFrame,
    date_from: str | None = None,
    date_to: str | None = None,
    author: str | None = None,
    title: str | None = None,
) -> pd.DataFrame:
    """
    Filtruje DataFrame po podstawowych metadanych.

    Semantyka zgodna z dotychczasowym engine.py:
    - Data publikacji: porównanie tekstowe na stringach,
    - Autor/Tytuł: contains case-insensitive.
    """
    df = _ensure_pandas_dataframe(df)
    mask = pd.Series(True, index=df.index)

    date_from = str(date_from or "").strip()
    date_to = str(date_to or "").strip()
    author = str(author or "").strip()
    title = str(title or "").strip()

    if "Data publikacji" in df.columns:
        date_series = df["Data publikacji"].astype(str)
        if date_from:
            mask &= date_series >= date_from
        if date_to:
            mask &= date_series <= date_to

    if "Autor" in df.columns and author:
        mask &= df["Autor"].astype(str).str.contains(author, case=False, na=False)

    if "Tytuł" in df.columns and title:
        mask &= df["Tytuł"].astype(str).str.contains(title, case=False, na=False)

    return df[mask].copy()


def compute_subcorpus_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """
    Przelicza metadane frekwencyjne dla podkorpusu.

    Zwraca strukturę zgodną z dotychczasowym metadata_export:
    - base_tf,
    - orth_tf,
    - total_tokens,
    - monthly_token_counts.
    """
    df = _ensure_pandas_dataframe(df)
    base_tf = Counter()
    orth_tf = Counter()
    total_tokens = 0
    monthly_token_counts: dict[str, dict[str, int]] = {}

    for row in df.itertuples():
        tokens = _as_plain_list(getattr(row, "tokens", None))
        lemmas = _as_plain_list(getattr(row, "lemmas", None))

        orth_tf.update(tokens)
        base_tf.update(lemmas)
        total_tokens += len(tokens)

        if "Data publikacji" in df.columns:
            pub_date = str(getattr(row, "_4", getattr(row, "Data publikacji", "0000-00-00"))).strip()
            parts = pub_date.split("-")
            year = parts[0] if len(parts) > 0 and parts[0] else "0000"
            month = parts[1] if len(parts) > 1 and parts[1] else "00"
            monthly_token_counts.setdefault(year, {}).setdefault(month, 0)
            monthly_token_counts[year][month] += len(tokens)

    return {
        "base_tf": dict(base_tf),
        "orth_tf": dict(orth_tf),
        "total_tokens": total_tokens,
        "monthly_token_counts": monthly_token_counts,
    }


def write_subcorpus_parquet(df: pd.DataFrame, file_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """
    Zapisuje DataFrame do Parquet z metadanymi Korpusuj w schema.metadata[b"korpus_meta"].
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = _ensure_pandas_dataframe(df)
    metadata = metadata if metadata is not None else compute_subcorpus_metadata(df)
    meta_json_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")

    table_pa = pa.Table.from_pandas(df, preserve_index=False)
    existing_meta = table_pa.schema.metadata or {}
    merged_meta = {**existing_meta, b"korpus_meta": meta_json_bytes}
    table_pa = table_pa.replace_schema_metadata(merged_meta)

    pq.write_table(table_pa, str(file_path), compression="snappy")


def export_dataframe_to_subcorpus_parquet(df: pd.DataFrame, file_path: str | Path) -> dict[str, Any]:
    """
    Wygodny helper: przelicz metadane i zapisz podkorpus.

    Zwraca metadane, żeby testy i ewentualny kod UI mogły je zweryfikować.
    """
    df = _ensure_pandas_dataframe(df)
    metadata = compute_subcorpus_metadata(df)
    write_subcorpus_parquet(df, file_path, metadata)
    return metadata
# --- END KORPUSUJ_MIGRATION_023_SUBCORPUS_HELPERS ---
