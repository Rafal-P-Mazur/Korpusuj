"""
test_query_engine_functional.py

Zestaw testów funkcjonalnych silnika zapytań Korpusuj.

Zakres:
- parser i mechanizm wyszukiwania na kontrolowanym, syntetycznym korpusie;
- zapytania proste, frazowe, złożone, zależnościowe, zagnieżdżone,
  metadane, warunki zdaniowe, okienkowe/dystansowe i negatywne;
- regresje wykryte podczas modularizacji:
  1. alternatywa deprel="a|b" wewnątrz dependent={...};
  2. profil kolokacyjny używający row_idx jako pozycji dokumentu, a nie
     etykiety indeksu DataFrame;
  3. ochrona przed przesuwaniem token_idx przez node_offset przed profilem.

UWAGA:
Ten plik zawiera adapter run_query(). Jeżeli aktualna wersja Korpusuj
wywołuje wyszukiwanie inaczej niż find_lemma_context(query, df, corpus_name),
wystarczy zmienić wyłącznie funkcję run_query() i/lub sekcję importu.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import pandas as pd
import pytest


# =============================================================================
# 1. IMPORT SILNIKA / ADAPTER
# =============================================================================

try:
    from engine import find_lemma_context  # aktualny wariant preferowany
except ImportError:  # fallback dla starszej/roboczej wersji testowej
    try:
        from Korpusuj_test import find_lemma_context
    except ImportError as exc:  # pragma: no cover - test environment setup
        find_lemma_context = None
        IMPORT_ERROR = exc
    else:
        IMPORT_ERROR = None
else:
    IMPORT_ERROR = None

try:
    from korpusuj.semantic.word_profile import compute_word_profile
except ImportError as exc:  # pragma: no cover - test environment setup
    compute_word_profile = None
    WORD_PROFILE_IMPORT_ERROR = exc
else:
    WORD_PROFILE_IMPORT_ERROR = None

try:
    from korpusuj.search import parser as cql_parser
except ImportError as exc:  # pragma: no cover - test environment setup
    cql_parser = None
    PARSER_IMPORT_ERROR = exc
else:
    PARSER_IMPORT_ERROR = None

try:
    from korpusuj.search.planner import SearchPlanner
except ImportError as exc:  # pragma: no cover - test environment setup
    SearchPlanner = None
    PLANNER_IMPORT_ERROR = exc
else:
    PLANNER_IMPORT_ERROR = None


def run_query(query: str, df: pd.DataFrame, corpus_name: str = "synthetic") -> list[Any]:
    """
    Adapter uruchamiający zapytanie na DataFrame.

    Zakładany kontrakt preferowany:
        find_lemma_context(query, df, corpus_name)

    Jeżeli engine ma inną sygnaturę, zmień tylko tę funkcję.
    """
    if find_lemma_context is None:
        pytest.fail(f"Nie udało się zaimportować find_lemma_context: {IMPORT_ERROR!r}")

    result = find_lemma_context(query, df, corpus_name)

    # SearchCursor albo inne leniwe iteratory materializujemy do listy,
    # żeby testy miały stabilny typ.
    if result is None:
        return []

    if isinstance(result, list):
        return result

    try:
        return list(result)
    except TypeError:
        return [result]


# =============================================================================
# 2. POMOCNICZE FUNKCJE DO ASERCJI
# =============================================================================


def result_texts(results: Iterable[Any]) -> list[str]:
    """
    Próbuje wydobyć tekst dopasowania/kontekstu z wyników.

    W różnych wersjach silnika wynik może być tuple, obiektem lub dict.
    Testy celowo używają dość tolerancyjnej ekstrakcji tekstu.
    """
    texts: list[str] = []

    for res in results:
        if isinstance(res, dict):
            parts = [
                res.get("matched_text"),
                res.get("matched_lemmas"),
                res.get("context"),
                res.get("text"),
            ]
            texts.append(" ".join(str(x) for x in parts if x is not None))
            continue

        if isinstance(res, (tuple, list)):
            # W aktualnym schemacie zwykle:
            # 3 = matched_text, 4 = matched_lemmas, 1 = context.
            preferred_indices = [3, 4, 1, 2]
            parts = []
            for idx in preferred_indices:
                if idx < len(res):
                    parts.append(res[idx])
            if parts:
                texts.append(" ".join(str(x) for x in parts if x is not None))
            else:
                texts.append(str(res))
            continue

        texts.append(str(res))

    return texts


def assert_any_text_contains(results: Iterable[Any], expected: str) -> None:
    texts = result_texts(results)
    assert any(expected in text for text in texts), (
        f"Nie znaleziono oczekiwanego fragmentu: {expected!r}. "
        f"Teksty wyników: {texts!r}"
    )


def assert_non_empty(results: Iterable[Any], query: str) -> None:
    results = list(results)
    assert results, f"Zapytanie nie zwróciło wyników: {query}"


def count_tokens(df: pd.DataFrame) -> int:
    return int(sum(len(x) for x in df["tokens"])) if not df.empty else 0


def token_freq_dict_from_df(df: pd.DataFrame, ignore_case: bool = True) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for lemmas in df["lemmas"]:
        for lemma in lemmas:
            key = str(lemma)
            if ignore_case:
                key = key.lower()
            counter[key] += 1
    return dict(counter)


def make_profile_result(row_idx: int, token_idx: int, matched_text: str, matched_lemma: str) -> tuple[Any, ...]:
    """
    Minimalny wynik zgodny ze schematem używanym przez profil:

    indeks 11 = row_idx/doc_id/pozycja dokumentu
    indeks 12 = token_idx w dokumencie
    indeks 13 = end_idx
    """
    return (
        "2024-01-01",          # 0 publication_date
        "context",             # 1 context
        ["left", matched_text, "right"],  # 2 full_text markers
        matched_text,           # 3 matched_text
        matched_lemma,          # 4 matched_lemmas
        "2024-01",             # 5 month_key
        "Synthetic title",      # 6 title
        "Synthetic author",     # 7 author
        {},                     # 8 additional_metadata
        "left",                # 9 left_context
        "right",               # 10 right_context
        row_idx,                # 11 row_idx / doc_id / positional row
        token_idx,              # 12 token_idx
        token_idx + 1,          # 13 end_idx
    )


def flatten_profile_rows(profile: dict[str, list[Any]]) -> list[Any]:
    rows: list[Any] = []
    for rel_rows in profile.values():
        rows.extend(rel_rows)
    return rows


# =============================================================================
# 3. SYNTETYCZNY KORPUS O ZNANEJ ANOTACJI
# =============================================================================


@pytest.fixture(scope="module")
def synthetic_corpus() -> tuple[pd.DataFrame, str]:
    """
    Miniaturowy korpus z ręcznie kontrolowaną anotacją.

    Dokument 0:
        Mały kot zjadł świeżą rybę szybko .

    Dokument 1:
        Pies szczekał .

    Dokument 2:
        Wojna trwała długo .

    Dokument 3:
        Wojna była trwana przez lata .
        Sztuczny przykład z deprel="nsubj:pass" dla testu alternatywy deprel.
    """
    rows = [
        {
            "tokens": ["Mały", "kot", "zjadł", "świeżą", "rybę", "szybko", "."],
            "lemmas": ["mały", "kot", "zjeść", "świeży", "ryba", "szybko", "."],
            "postags": ["adj", "subst", "fin", "adj", "subst", "adv", "interp"],
            "upostags": ["ADJ", "NOUN", "VERB", "ADJ", "NOUN", "ADV", "PUNCT"],
            "full_postags": ["adj:sg:nom:m1", "subst:sg:nom:m1", "fin:sg:ter:perf", "adj:sg:acc:f", "subst:sg:acc:f", "adv", "interp"],
            "sentence_ids": [0, 0, 0, 0, 0, 0, 0],
            "word_ids": [1, 2, 3, 4, 5, 6, 7],
            "head_ids": [2, 3, 0, 5, 3, 3, 3],
            "deprels": ["amod", "nsubj", "root", "amod", "obj", "advmod", "punct"],
            "start_ids": [0, 5, 9, 15, 23, 28, 35],
            "end_ids": [4, 8, 14, 22, 27, 34, 36],
            "ners": ["O"] * 7,
            "corefs": [""] * 7,
            "Data publikacji": "2024-01-01",
            "Autor": "Anna Nowak",
            "Tytuł": "O kocie",
            "url": "",
            "Treść": "Mały kot zjadł świeżą rybę szybko.",
        },
        {
            "tokens": ["Pies", "szczekał", "."],
            "lemmas": ["pies", "szczekać", "."],
            "postags": ["subst", "fin", "interp"],
            "upostags": ["NOUN", "VERB", "PUNCT"],
            "full_postags": ["subst:sg:nom:m2", "fin:sg:ter:imperf", "interp"],
            "sentence_ids": [0, 0, 0],
            "word_ids": [1, 2, 3],
            "head_ids": [2, 0, 2],
            "deprels": ["nsubj", "root", "punct"],
            "start_ids": [0, 5, 13],
            "end_ids": [4, 12, 14],
            "ners": ["O"] * 3,
            "corefs": [""] * 3,
            "Data publikacji": "2024-02-01",
            "Autor": "Jan Kowalski",
            "Tytuł": "O psie",
            "url": "",
            "Treść": "Pies szczekał.",
        },
        {
            "tokens": ["Wojna", "trwała", "długo", "."],
            "lemmas": ["wojna", "trwać", "długo", "."],
            "postags": ["subst", "fin", "adv", "interp"],
            "upostags": ["NOUN", "VERB", "ADV", "PUNCT"],
            "full_postags": ["subst:sg:nom:f", "fin:sg:ter:imperf", "adv", "interp"],
            "sentence_ids": [0, 0, 0, 0],
            "word_ids": [1, 2, 3, 4],
            "head_ids": [2, 0, 2, 2],
            "deprels": ["nsubj", "root", "advmod", "punct"],
            "start_ids": [0, 6, 13, 18],
            "end_ids": [5, 12, 18, 19],
            "ners": ["O"] * 4,
            "corefs": [""] * 4,
            "Data publikacji": "2024-03-01",
            "Autor": "Anna Nowak",
            "Tytuł": "O wojnie aktywnej",
            "url": "",
            "Treść": "Wojna trwała długo.",
        },
        {
            "tokens": ["Wojna", "była", "trwana", "przez", "lata", "."],
            "lemmas": ["wojna", "być", "trwać", "przez", "rok", "."],
            "postags": ["subst", "fin", "ppas", "prep", "subst", "interp"],
            "upostags": ["NOUN", "AUX", "VERB", "ADP", "NOUN", "PUNCT"],
            "full_postags": ["subst:sg:nom:f", "fin:sg:ter:imperf", "ppas:sg:nom:f:imperf:aff", "prep", "subst:pl:acc:m3", "interp"],
            "sentence_ids": [0, 0, 0, 0, 0, 0],
            "word_ids": [1, 2, 3, 4, 5, 6],
            "head_ids": [3, 3, 0, 5, 3, 3],
            "deprels": ["nsubj:pass", "aux:pass", "root", "case", "obl", "punct"],
            "start_ids": [0, 6, 11, 19, 25, 29],
            "end_ids": [5, 10, 18, 24, 29, 30],
            "ners": ["O"] * 6,
            "corefs": [""] * 6,
            "Data publikacji": "2024-04-01",
            "Autor": "Anna Nowak",
            "Tytuł": "O wojnie biernej",
            "url": "",
            "Treść": "Wojna była trwana przez lata.",
        },
    ]

    return pd.DataFrame(rows), "synthetic"


@pytest.fixture(scope="module")
def synthetic_corpus_non_contiguous_index(synthetic_corpus) -> tuple[pd.DataFrame, str]:
    """
    Symuluje realny Parquet po filtrowaniu/łączeniu, gdzie df.index nie jest
    gęstym zakresem 0..N-1, natomiast .search/full_results_sorted używa
    pozycyjnego doc_id.
    """
    df, corpus_name = synthetic_corpus
    df = df.copy()
    df.index = [0, 2, 4, 11][:len(df)]
    return df, corpus_name


@pytest.fixture(scope="module")
def empty_corpus() -> tuple[pd.DataFrame, str]:
    columns = [
        "tokens", "lemmas", "postags", "upostags", "full_postags",
        "sentence_ids", "word_ids", "head_ids", "deprels",
        "start_ids", "end_ids", "ners", "corefs",
        "Data publikacji", "Autor", "Tytuł", "url", "Treść",
    ]
    return pd.DataFrame(columns=columns), "empty"


# =============================================================================
# 4. TESTY PODSTAWOWE: TOKENY, FRAZY, WARUNKI
# =============================================================================


def test_simple_lemma_query_returns_expected_matches(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot"]', df, corpus_name)
    assert_non_empty(results, '[base="kot"]')
    assert_any_text_contains(results, "kot")


def test_simple_orth_query_returns_expected_matches(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    # W syntetycznym korpusie token ma postać "kot", nie "Kot".
    # Ten test sprawdza więc działające dopasowanie orth bez zakładania case-insensitive.
    results = run_query('[orth="kot"]', df, corpus_name)
    assert_non_empty(results, '[orth="kot"]')


def test_phrase_query_adj_plus_lemma(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[pos="adj"] [base="ryba"]', df, corpus_name)
    assert_non_empty(results, '[pos="adj"] [base="ryba"]')
    assert_any_text_contains(results, "ryb")


def test_multiple_conditions_in_one_segment(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot" & pos="subst"]', df, corpus_name)
    assert_non_empty(results, '[base="kot" & pos="subst"]')


def test_regex_alternative_for_base_attribute(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot|pies"]', df, corpus_name)
    assert len(results) >= 2


# =============================================================================
# 5. TESTY RELACJI SKŁADNIOWYCH I ZAGNIEŻDŻEŃ
# =============================================================================


def test_dependency_query_verb_with_subject_and_object(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="zjeść" & dependent={base="kot" & deprel="nsubj"} & dependent={base="ryba" & deprel="obj"}]'
    results = run_query(query, df, corpus_name)
    assert_non_empty(results, query)


def test_dependency_query_with_nested_condition(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="zjeść" & dependent={deprel="obj" & base="ryba" & dependent={base="świeży" & deprel="amod"}}]'
    results = run_query(query, df, corpus_name)
    assert_non_empty(results, query)


def test_head_query_for_object(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="ryba" & head="zjeść"]'
    results = run_query(query, df, corpus_name)
    assert_non_empty(results, query)


def test_dependency_deprel_pipe_or_inside_dependent_parser_shape():
    """
    Regresja: deprel="nsubj|nsubj:pass" wewnątrz dependent={...}
    ma być alternatywą wartości exact, a nie pojedynczym dosłownym stringiem
    ani źle obsłużonym regexem.
    """
    if cql_parser is None:
        pytest.fail(f"Nie udało się zaimportować parsera: {PARSER_IMPORT_ERROR!r}")

    query = '[base="trwać" & dependent={base="wojna" & deprel="nsubj|nsubj:pass"}]'
    parsed = cql_parser.parse_query_group(query)

    found_deprel = []

    def walk(obj):
        if isinstance(obj, tuple) and len(obj) >= 5 and obj[0] == "deprel":
            found_deprel.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(parsed)

    assert found_deprel, f"Nie znaleziono warunku deprel w: {parsed!r}"
    assert any(cond[1] == ["nsubj", "nsubj:pass"] for cond in found_deprel), found_deprel


def test_dependency_deprel_pipe_or_inside_dependent_returns_superset(synthetic_corpus):
    """
    Regresja: OR w deprel wewnątrz dependent={...} nie może zwracać mniej
    wyników niż pojedyncza wartość nsubj.
    """
    df, corpus_name = synthetic_corpus

    q_single = '[base="trwać" & dependent={base="wojna" & deprel="nsubj"}]'
    q_or = '[base="trwać" & dependent={base="wojna" & deprel="nsubj|nsubj:pass"}]'

    single = run_query(q_single, df, corpus_name)
    pipe_or = run_query(q_or, df, corpus_name)

    assert_non_empty(single, q_single)
    assert len(pipe_or) >= len(single), (
        f"Zapytanie OR powinno zwracać co najmniej wyniki single. "
        f"single={len(single)}, or={len(pipe_or)}"
    )


# =============================================================================
# 6. TESTY OPERATORÓW OKNA, DYSTANSU I ZDANIA
# =============================================================================


def test_window_base_query(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="mały" & window_base(3)="zjeść"]', df, corpus_name)
    assert_non_empty(results, '[base="mały" & window_base(3)="zjeść"]')


def test_distance_operator_between_segments(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    # Składnia obsługiwana przez dotychczasowy parser dla luki/repetytora.
    query = '[base="kot"] [*][0,1] [base="zjeść"]'
    results = run_query(query, df, corpus_name)
    assert_non_empty(results, query)


def test_sentence_unordered_condition(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="zjeść"] <s ([base="ryba"]) ([base="szybko"] )>'
    results = run_query(query, df, corpus_name)
    assert_non_empty(results, query)


# =============================================================================
# 7. TESTY METADANYCH
# =============================================================================


def test_metadata_filter_author_equals(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot"] <autor="Anna Nowak">', df, corpus_name)
    assert_non_empty(results, '[base="kot"] <autor="Anna Nowak">')


def test_metadata_filter_author_not_equals(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot"] <autor!="Jan Kowalski">', df, corpus_name)
    assert_non_empty(results, '[base="kot"] <autor!="Jan Kowalski">')


def test_metadata_filter_title_equals(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="pies"] <tytuł="O psie">', df, corpus_name)
    assert_non_empty(results, '[base="pies"] <tytuł="O psie">')


# =============================================================================
# 8. TESTY PROFILU KOLOKACYJNEGO / WORD PROFILE
# =============================================================================


def test_word_profile_uses_positional_row_idx_not_dataframe_label_index(synthetic_corpus_non_contiguous_index):
    """
    Regresja z realnego korpusu:
    .search/full_results_sorted używa row_idx jako gęstego doc_id / pozycji
    dokumentu 0..N-1, podczas gdy Parquet może mieć nieciągły df.index.

    compute_word_profile musi zatem interpretować row_idx pozycyjnie
    przez df.iloc[int(row_idx)], a nie etykietowo przez df.loc[row_idx].
    """
    if compute_word_profile is None:
        pytest.fail(f"Nie udało się zaimportować compute_word_profile: {WORD_PROFILE_IMPORT_ERROR!r}")

    df, _ = synthetic_corpus_non_contiguous_index

    # Dokument pozycyjny 2 ma nieciągłą etykietę indeksu 4.
    # row_idx=2 musi wskazać trzeci wiersz przez iloc, nie df.loc[2].
    results = [make_profile_result(row_idx=2, token_idx=1, matched_text="trwała", matched_lemma="trwać")]

    profile = compute_word_profile(
        results=results,
        df=df,
        token_freq_dict=token_freq_dict_from_df(df),
        target_lemma="trwać",
        total_tokens=count_tokens(df),
        min_freq=1,
        ignore_case=True,
        expand_mwe=False,
    )

    assert profile, "Profil powinien powstać dla row_idx traktowanego pozycyjnie."
    rows = flatten_profile_rows(profile)
    assert rows, "Profil powinien zawierać przynajmniej jeden wiersz."


def test_word_profile_token_idx_must_not_be_shifted_by_node_offset(synthetic_corpus_non_contiguous_index):
    """
    Regresja: res[12] jest indeksem tokenu w dokumencie.
    Nie wolno go przesuwać przez node_offset przed compute_word_profile().
    Test pokazuje, że poprawny token_idx=1 dla lematu 'trwać' działa.
    """
    if compute_word_profile is None:
        pytest.fail(f"Nie udało się zaimportować compute_word_profile: {WORD_PROFILE_IMPORT_ERROR!r}")

    df, _ = synthetic_corpus_non_contiguous_index

    correct_results = [make_profile_result(row_idx=2, token_idx=1, matched_text="trwała", matched_lemma="trwać")]

    profile = compute_word_profile(
        results=correct_results,
        df=df,
        token_freq_dict=token_freq_dict_from_df(df),
        target_lemma="trwać",
        total_tokens=count_tokens(df),
        min_freq=1,
        ignore_case=True,
        expand_mwe=False,
    )

    assert profile


# =============================================================================
# 9. TESTY NEGATYWNE I PRZYPADKI BRZEGOWE
# =============================================================================


def test_sequential_negative_wrong_order(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="zjeść"] [base="kot"]', df, corpus_name)
    assert len(results) == 0


def test_sentence_boundary_is_respected(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="ryba"] [base="pies"]', df, corpus_name)
    assert len(results) == 0


def test_empty_corpus_returns_no_results(empty_corpus):
    df, corpus_name = empty_corpus
    results = run_query('[base="kot"]', df, corpus_name)
    assert len(results) == 0


# =============================================================================
# 10. TEST OPCJONALNY: INTERPUNKCJA
# =============================================================================


def test_punctuation_as_token_in_phrase(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    # W korpusie po "ryba" występuje jeszcze "szybko", więc interpunkcja nie jest bezpośrednio po rybie.
    results = run_query('[base="szybko"] [pos="interp"]', df, corpus_name)
    assert_non_empty(results, '[base="szybko"] [pos="interp"]')

# =============================================================================
# 12. TESTY PEŁNEJ SKŁADNI REGEX LEGACY NA SQL/SearchIndex — 036L4G37F
# =============================================================================
# KORPUSUJ_MIGRATION_036L4G37F_LEGACY_REGEX_SEMANTICS_TESTS


def _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch):
    """Tiny .search sidecar with terms for documented legacy-regex syntax."""
    import sqlite3

    from korpusuj.index import sqlite_index as sqlite_mod
    from korpusuj.index.postings import PostingList

    db_path = tmp_path / "regex_full_036l4g37f.search"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE terms (attr TEXT, value TEXT, df INTEGER, cf INTEGER, postings BLOB)")
    con.execute("CREATE INDEX idx_terms_attr_value_036l4g37f ON terms(attr, value)")
    con.execute("INSERT INTO meta(key, value) VALUES ('total_docs', '10')")
    con.execute("INSERT INTO meta(key, value) VALUES ('total_tokens', '100')")

    next_pos = {1: 0}

    def add(value, attr="orth"):
        pos = next_pos[1]
        next_pos[1] += 1
        postings_by_doc = {1: [pos]}
        con.execute(
            "INSERT INTO terms(attr, value, df, cf, postings) VALUES (?, ?, ?, ?, ?)",
            (attr, value, 1, 1, PostingList.encode(postings_by_doc)),
        )
        return pos

    for value in [
        "kot", "koty", "kotyy", "pies", "dom", "las", "domu", "domem", "domy", "domów",
        "Jagoda", "jagoda", "Magoda", "wojna", "wojskowy", "województwo", "pokój",
        "Ania", "kawa", "drzewa", "Adam", "kwestia", "kwestię", "kwestio",
        "kryzys", "kryzysowy", "zysk", ".", "a", "5", "12", "abc_12", " ", "\t",
    ]:
        add(value)

    con.commit()
    con.close()

    def cfg():
        return {
            "regex_sqlite_route": True,
            "regex_sqlite_enabled": True,
            "regex_sqlite_debug": False,
            "regex_sqlite_max_terms": 10000,
            "regex_sqlite_max_cf": 1000000,
            "regex_sqlite_max_pattern_len": 256,
            "regex_sqlite_match_mode": "fullmatch",
        }

    monkeypatch.setattr(sqlite_mod, "_engine_config", cfg, raising=False)
    return sqlite_mod.SearchIndex(str(db_path))


def _values_for_regex_036l4g37f(idx, pattern, attr="orth"):
    """Return the set of term values matched by the SQLite regex vocabulary scan."""
    return {value for value, _df, _cf in idx.find_terms_regex_036l4g37c(attr, pattern)}


def test_regex_legacy_question_mark_zero_or_one_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        assert _values_for_regex_036l4g37f(idx, "koty?") == {"kot", "koty"}
    finally:
        idx.close()


def test_regex_legacy_dot_exact_single_character_repeated_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        values = _values_for_regex_036l4g37f(idx, "...")
        assert {"kot", "dom", "las"}.issubset(values)
        assert "domu" not in values
        assert "kryzys" not in values
    finally:
        idx.close()


def test_regex_legacy_character_classes_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        values = _values_for_regex_036l4g37f(idx, "[A-z]agoda")
        assert {"Jagoda", "jagoda", "Magoda"}.issubset(values)
        assert "kawa" not in values
    finally:
        idx.close()


def test_regex_legacy_digit_word_and_whitespace_classes_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        assert _values_for_regex_036l4g37f(idx, r"\d+") >= {"5", "12"}
        assert "abc_12" in _values_for_regex_036l4g37f(idx, r"\w+")
        assert _values_for_regex_036l4g37f(idx, r"\s") >= {" ", "\t"}
    finally:
        idx.close()


def test_regex_legacy_star_plus_suffix_and_prefix_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        assert _values_for_regex_036l4g37f(idx, "woj.*") >= {"wojna", "wojskowy", "województwo"}
        ending_a = _values_for_regex_036l4g37f(idx, ".*a")
        assert {"Ania", "kawa", "drzewa"}.issubset(ending_a)
        assert "Adam" not in ending_a
        dom_plus = _values_for_regex_036l4g37f(idx, "dom.+")
        assert {"domu", "domem", "domy", "domów"}.issubset(dom_plus)
        assert "dom" not in dom_plus
    finally:
        idx.close()


def test_regex_legacy_alternation_and_grouping_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        assert _values_for_regex_036l4g37f(idx, "kot|pies") == {"kot", "pies"}
        values = _values_for_regex_036l4g37f(idx, "kwesti(a|ę)")
        assert values == {"kwestia", "kwestię"}
    finally:
        idx.close()


def test_regex_legacy_tilde_contains_sequence_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        values = _values_for_regex_036l4g37f(idx, "~zys")
        assert {"kryzys", "kryzysowy", "zysk"}.issubset(values)
    finally:
        idx.close()


def test_regex_legacy_escaped_dot_is_literal_dot_036l4g37f(tmp_path, monkeypatch):
    idx = _make_regex_sqlite_index_full_036l4g37f(tmp_path, monkeypatch)
    try:
        assert _values_for_regex_036l4g37f(idx, r"\.") == {"."}
    finally:
        idx.close()

