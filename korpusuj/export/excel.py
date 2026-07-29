"""Build tabular data for Excel exports of search results, tables and collocational profiles.

The helpers are independent of GUI dialogs and application state.
"""

import pandas as pd


def clean_for_excel(df):
    df_cleaned = df.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', regex=True)
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[col] = df_cleaned[col].apply(
            lambda x: f"'{x}" if isinstance(x, str) and str(x).startswith(('=', '-', '+', '@')) else x
        )
    return df_cleaned


# --- KORPUSUJ_MIGRATION_022_EXCEL_SHEET_BUILDERS ---
# Helpery eksportu danych do Excela/CSV.
# Cel: trzymać logikę DataFrame/arkuszy poza engine.py, ale zostawić dialogi UI
# i odczyt globalnego stanu aplikacji w adapterze engine.export_data().

DEFAULT_SEARCH_EXPORT_COLUMNS = [
    "Data publikacji", "context", "full_text_with_markers",
    "Rezultat", "matched_lemmas",
    "month_key", "Tytuł", "Autor", "additional_metadata",
    "Lewy kontekst", "Prawy kontekst", "row_index", "start_idx", "end_idx",
]

DEFAULT_SEARCH_EXPORT_VISIBLE_COLUMNS = [
    "Data publikacji", "Autor", "Tytuł",
    "Lewy kontekst", "Rezultat", "Prawy kontekst",
]

LEMMA_FREQUENCY_HEADERS = [
    "Nr", "Forma podstawowa (base)", "Liczba wystąpień", "Częstość względna",
    "Rozproszenie (DF)", "Ogólne TF-IDF",
]

TOKEN_FREQUENCY_HEADERS = [
    "Nr", "Forma tekstowa (orth)", "Liczba wystąpień", "Częstość względna",
    "Rozproszenie (DF)", "Ogólne TF-IDF",
]

MONTH_FREQUENCY_HEADERS = [
    "Rok", "Miesiąc", "Forma podstawowa", "Liczba wystąpień", "Częstość względna",
    "TF-IDF", "Z-score",
]

COLLOCATION_HEADERS = [
    "Nr", "Kolokat", "f(nc)", "f(c)", "Log-Likelihood", "MI Score", "T-score", "Log-Dice",
]

PROFILE_HEADERS = [
    "Nr", "Kolokat", "Relacja składniowa", "Współwyst.", "Zasięg (Dok.)", "Freq. Glob.",
    "Log-Likelihood", "MI Score", "T-score", "Log-Dice",
]


def build_search_results_export_df(results, all_columns=None, visible_columns=None):
    # Buduje główny DataFrame eksportu wyników wyszukiwania.
    all_columns = list(all_columns or DEFAULT_SEARCH_EXPORT_COLUMNS)
    visible_columns = list(visible_columns or DEFAULT_SEARCH_EXPORT_VISIBLE_COLUMNS)

    df_export = pd.DataFrame(list(results or []), columns=all_columns)

    if "additional_metadata" in df_export.columns:
        meta_df = pd.json_normalize(df_export["additional_metadata"])
        df_flat = pd.concat([df_export.drop(columns=["additional_metadata"]), meta_df], axis=1)
        ordered_columns = list(meta_df.columns) + visible_columns
    else:
        df_flat = df_export
        ordered_columns = visible_columns

    existing_columns = [col for col in ordered_columns if col in df_flat.columns]
    out = df_flat[existing_columns] if existing_columns else df_flat
    return clean_for_excel(out)


def build_table_export_df(data_rows, headers):
    # Buduje prosty DataFrame arkusza z danych tabel/paginatorów.
    return clean_for_excel(pd.DataFrame(list(data_rows or []), columns=list(headers or [])))


def _profile_row_value(row_obj, attr, default=""):
    try:
        return getattr(row_obj, attr, default)
    except Exception:
        return default


def build_profile_export_df(profile_dict):
    # Buduje DataFrame pełnego profilu kolokacyjnego ze słownika relacji.
    all_merged_rows = []

    for _rel_name, rows in (profile_dict or {}).items():
        all_merged_rows.extend(list(rows or []))

    all_merged_rows.sort(
        key=lambda r: (
            _profile_row_value(r, "log_dice", 0) or 0,
            _profile_row_value(r, "cooc_freq", 0) or 0,
        ),
        reverse=True,
    )

    table_rows = []

    for i, row_obj in enumerate(all_merged_rows):
        display_colloc = _profile_row_value(row_obj, "collocate", "")
        collocate_upos = _profile_row_value(row_obj, "collocate_upos", "")

        if collocate_upos:
            display_colloc = f"{display_colloc} [{collocate_upos}]"

        table_rows.append([
            i + 1,
            display_colloc,
            _profile_row_value(row_obj, "relation", ""),
            _profile_row_value(row_obj, "cooc_freq", 0),
            _profile_row_value(row_obj, "doc_freq", 0),
            _profile_row_value(row_obj, "global_freq", 0),
            _profile_row_value(row_obj, "ll_score", 0),
            _profile_row_value(row_obj, "mi_score", 0),
            _profile_row_value(row_obj, "t_score", 0),
            _profile_row_value(row_obj, "log_dice", 0),
        ])

    return build_table_export_df(table_rows, PROFILE_HEADERS)


def write_excel_workbook(file_path, sheets):
    # Zapisuje wiele arkuszy XLSX. sheets: iterable par (sheet_name, dataframe).
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            if df is None:
                continue
            clean_for_excel(df).to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)


def write_csv_export(file_path, dataframe):
    # Zapisuje główną tabelę eksportu do CSV.
    clean_for_excel(dataframe).to_csv(file_path, index=False)
# --- END KORPUSUJ_MIGRATION_022_EXCEL_SHEET_BUILDERS ---

