"""
test_query_engine_functional.py

Czysty zestaw testów funkcjonalnych silnika zapytań Korpusuj.

Cel testów:
- sprawdzenie poprawności działania parsera i mechanizmu wyszukiwania
  względem kontrolowanego, syntetycznego korpusu o znanej anotacji;
- oddzielenie testowania silnika zapytań od oceny jakości zewnętrznych
  modeli anotacyjnych NLP;
- zapewnienie minimalnego, uruchamialnego zestawu testów dla zapytań:
  prostych, frazowych, złożonych, zależnościowych, zagnieżdżonych,
  metadanych, zdaniowych, okienkowych/dystansowych oraz negatywnych.

UWAGA:
Ten plik zawiera adapter run_query(). Jeżeli aktualna wersja Korpusuj
wywołuje wyszukiwanie inaczej niż find_lemma_context(query, df, corpus_name),
wystarczy zmienić wyłącznie funkcję run_query() i/lub sekcję importu.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import pytest


# =============================================================================
# 1. IMPORT SILNIKA / ADAPTER
# =============================================================================

# Dostosuj tę sekcję, jeśli aktualna funkcja wyszukująca znajduje się
# w innym module albo wymaga innej inicjalizacji.
try:
    from engine import find_lemma_context  # aktualny wariant preferowany
except ImportError:  # fallback dla starszej/roboczej wersji testowej
    try:
        from Korpusuj_test import find_lemma_context
    except ImportError as exc:
        find_lemma_context = None
        IMPORT_ERROR = exc
    else:
        IMPORT_ERROR = None
else:
    IMPORT_ERROR = None


def run_query(query: str, df: pd.DataFrame, corpus_name: str = "synthetic") -> list[Any]:
    """
    Adapter uruchamiający zapytanie na DataFrame.

    Domyślnie zakłada starsze/obecne API:
        find_lemma_context(query, df, corpus_name)

    Jeśli silnik wymaga budowania indeksów, rejestracji DataFrame w globalnym
    stanie albo użycia klasy SearchState, należy zmienić tylko tę funkcję.
    """
    if find_lemma_context is None:
        raise ImportError(
            "Nie udało się zaimportować find_lemma_context. "
            "Dostosuj sekcję importu w teście."
        ) from IMPORT_ERROR

    result = find_lemma_context(query, df, corpus_name)
    return list(result or [])


# =============================================================================
# 2. POMOCNICZE FUNKCJE DO ASERCJI
# =============================================================================


def result_texts(results: Iterable[Any]) -> list[str]:
    """
    Próbuje wydobyć tekst dopasowania/kontekstu z wyników.

    W starszych testach tekst dopasowania znajdował się pod indeksem 3.
    Ponieważ format wyniku może się zmieniać, funkcja jest defensywna:
    - jeśli wynik jest dict, szuka typowych kluczy;
    - jeśli wynik jest listą/tuplą, próbuje indeks 3;
    - w ostateczności zwraca str(result).
    """
    texts: list[str] = []
    for item in results:
        if isinstance(item, dict):
            for key in ("match", "matched_text", "text", "context", "kwic"):
                if key in item and item[key] is not None:
                    texts.append(str(item[key]))
                    break
            else:
                texts.append(str(item))
        elif isinstance(item, (list, tuple)):
            if len(item) > 3:
                texts.append(str(item[3]))
            else:
                texts.append(str(item))
        else:
            texts.append(str(item))
    return texts


def assert_any_text_contains(results: Iterable[Any], expected: str) -> None:
    texts = result_texts(results)
    assert any(expected in text for text in texts), (
        f"Nie znaleziono oczekiwanego fragmentu: {expected!r}. "
        f"Teksty wyników: {texts!r}"
    )


# =============================================================================
# 3. SYNTETYCZNY KORPUS O ZNANEJ ANOTACJI
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_corpus() -> tuple[pd.DataFrame, str]:
    """
    Tworzy miniaturowy, ręcznie kontrolowany korpus.

    Dokument 1 / zdanie 1:
        "Mały kot szybko zjadł świeżą rybę."

    Dokument 2 / zdanie 2:
        "Duży pies goni kota."

    Dokument 3 / dwa zdania w jednym wierszu:
        "Kot je rybę. Pies szczeka."

    Konwencja head_ids:
    - wartości odpowiadają word_id nadrzędnika;
    - 0 oznacza root.
    """
    df = pd.DataFrame(
        {
            "tokens": [
                ["Mały", "kot", "szybko", "zjadł", "świeżą", "rybę", "."],
                ["Duży", "pies", "goni", "kota", "."],
                ["Kot", "je", "rybę", ".", "Pies", "szczeka", "."],
            ],
            "lemmas": [
                ["mały", "kot", "szybko", "zjeść", "świeży", "ryba", "."],
                ["duży", "pies", "gonić", "kot", "."],
                ["kot", "jeść", "ryba", ".", "pies", "szczekać", "."],
            ],
            "postags": [
                ["adj", "subst", "adv", "verb", "adj", "subst", "interp"],
                ["adj", "subst", "verb", "subst", "interp"],
                ["subst", "verb", "subst", "interp", "subst", "verb", "interp"],
            ],
            "upostags": [
                ["ADJ", "NOUN", "ADV", "VERB", "ADJ", "NOUN", "PUNCT"],
                ["ADJ", "NOUN", "VERB", "NOUN", "PUNCT"],
                ["NOUN", "VERB", "NOUN", "PUNCT", "NOUN", "VERB", "PUNCT"],
            ],
            "full_postags": [
                ["adj:sg:nom:m1:pos", "subst:sg:nom:m2", "adv:pos", "praet:sg:m1:perf", "adj:sg:acc:f:pos", "subst:sg:acc:f", "interp"],
                ["adj:sg:nom:m2:pos", "subst:sg:nom:m2", "fin:sg:ter:imperf", "subst:sg:acc:m2", "interp"],
                ["subst:sg:nom:m2", "fin:sg:ter:imperf", "subst:sg:acc:f", "interp", "subst:sg:nom:m2", "fin:sg:ter:imperf", "interp"],
            ],
            "sentence_ids": [
                [1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 4, 4, 4],
            ],
            "word_ids": [
                [1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 6, 7],
            ],
            "head_ids": [
                [2, 4, 4, 0, 6, 4, 4],  # kot -> zjadł; rybę -> zjadł
                [2, 3, 0, 3, 3],        # pies -> goni; kota -> goni
                [2, 0, 2, 2, 6, 0, 6],  # Kot -> je; rybę -> je; Pies -> szczeka
            ],
            "deprels": [
                ["amod", "nsubj", "advmod", "root", "amod", "obj", "punct"],
                ["amod", "nsubj", "root", "obj", "punct"],
                ["nsubj", "root", "obj", "punct", "nsubj", "root", "punct"],
            ],
            "start_ids": [
                [0, 5, 9, 16, 22, 29, 33],
                [0, 5, 10, 15, 19],
                [0, 4, 7, 12, 14, 19, 26],
            ],
            "end_ids": [
                [4, 8, 15, 21, 28, 33, 34],
                [4, 9, 14, 19, 20],
                [3, 6, 11, 13, 18, 25, 27],
            ],
            "ners": [
                ["0", "0", "0", "0", "0", "0", "0"],
                ["0", "0", "0", "0", "0"],
                ["0", "0", "0", "0", "0", "0", "0"],
            ],
            "corefs": [
                [[], [], [], [], [], [], []],
                [[], [], [], [], []],
                [[], [], [], [], [], [], []],
            ],
            "Data publikacji": ["2023-10-15", "2023-10-16", "2023-10-17"],
            "Autor": ["Jan Kowalski", "Anna Nowak", "Jan Kowalski"],
            "Tytuł": ["O kocie", "O psie", "Dwa zdania"],
            "url": ["", "", ""],
            "Treść": [
                "Mały kot szybko zjadł świeżą rybę.",
                "Duży pies goni kota.",
                "Kot je rybę. Pies szczeka.",
            ],
        }
    )
    return df, "synthetic"


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

    # "kot" występuje jako lemma: kot, kota, Kot = 3 dopasowania.
    assert len(results) == 3


def test_simple_orth_query_is_case_sensitive_or_exact_if_engine_uses_exact_mode(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[orth="Kot"]', df, corpus_name)

    # Oczekiwany co najmniej jeden wynik: token "Kot" w trzecim dokumencie.
    # Jeśli silnik ignoruje wielkość liter globalnie, wyników może być więcej;
    # dlatego test sprawdza minimalną własność, a nie dokładną liczbę.
    assert len(results) >= 1
    assert_any_text_contains(results, "Kot")


def test_phrase_query_adj_plus_lemma(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[pos="adj"] [base="ryba"]', df, corpus_name)

    assert len(results) == 1
    assert_any_text_contains(results, "świeżą rybę")


def test_multiple_conditions_in_one_segment(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot" & pos="subst"]', df, corpus_name)

    assert len(results) == 3


def test_regex_alternative_for_base_attribute(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot|pies"]', df, corpus_name)

    assert len(results) == 5


# =============================================================================
# 5. TESTY RELACJI SKŁADNIOWYCH I ZAGNIEŻDŻEŃ
# =============================================================================


def test_dependency_query_verb_with_subject_and_object(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="zjeść" & dependent={base="kot" & deprel="nsubj"} & dependent={base="ryba" & deprel="obj"}]'
    results = run_query(query, df, corpus_name)

    assert len(results) == 1
    assert_any_text_contains(results, "zjadł")


def test_dependency_query_with_nested_condition(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="zjeść" & dependent={deprel="obj" & base="ryba" & dependent={base="świeży" & deprel="amod"}}]'
    results = run_query(query, df, corpus_name)

    assert len(results) == 1
    assert_any_text_contains(results, "zjadł")


def test_head_query_for_object(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="ryba" & head="zjeść"]'
    results = run_query(query, df, corpus_name)

    # W dokumencie 1 "rybę" ma head = zjadł/zjeść.
    assert len(results) >= 1
    assert_any_text_contains(results, "rybę")


# =============================================================================
# 6. TESTY OPERATORÓW OKNA, DYSTANSU I ZDANIA
# =============================================================================


def test_window_base_query(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="mały" & window_base(3)="zjeść"]', df, corpus_name)

    assert len(results) == 1
    assert_any_text_contains(results, "Mały")


def test_distance_operator_between_segments(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot"] [*][1,1] [base="zjeść"]', df, corpus_name)

    # "kot szybko zjadł" — między kot i zjadł jest dokładnie jeden token.
    assert len(results) == 1
    assert_any_text_contains(results, "kot szybko zjadł")


def test_sentence_unordered_condition(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    query = '[base="zjeść"] <s ([base="ryba"]) ([base="szybko"] )>'
    results = run_query(query, df, corpus_name)

    assert len(results) == 1
    assert_any_text_contains(results, "zjadł")


# =============================================================================
# 7. TESTY METADANYCH
# =============================================================================


def test_metadata_filter_author_equals(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot"] <autor="Anna Nowak">', df, corpus_name)

    # W dokumencie Anny Nowak jest "kota" z lematem kot.
    assert len(results) == 1


def test_metadata_filter_author_not_equals(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="kot"] <autor!="Jan Kowalski">', df, corpus_name)

    assert len(results) == 1


def test_metadata_filter_title_equals(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="pies"] <tytuł="O psie">', df, corpus_name)

    assert len(results) == 1
    assert_any_text_contains(results, "pies")


# =============================================================================
# 8. TESTY NEGATYWNE I PRZYPADKI BRZEGOWE
# =============================================================================


def test_sequential_negative_wrong_order(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="zjeść"] [base="kot"]', df, corpus_name)

    # W korpusie nie ma sekwencji "zjeść kot".
    assert len(results) == 0


def test_sentence_boundary_is_respected(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="ryba"] [base="pies"]', df, corpus_name)

    # W trzecim dokumencie "rybę" i "Pies" są w sąsiednich zdaniach,
    # ale nie powinny tworzyć zwykłej frazy tokenowej.
    assert len(results) == 0


def test_empty_corpus_returns_no_results(empty_corpus):
    df, corpus_name = empty_corpus
    results = run_query('[base="kot"]', df, corpus_name)

    assert results == []


# =============================================================================
# 9. TEST OPCJONALNY: INTERPUNKCJA
# =============================================================================


def test_punctuation_as_token_in_phrase(synthetic_corpus):
    df, corpus_name = synthetic_corpus
    results = run_query('[base="ryba"] [pos="interp"]', df, corpus_name)

    # Dwa wystąpienia: "rybę." w dokumencie 1 i "rybę." w dokumencie 3.
    assert len(results) == 2
