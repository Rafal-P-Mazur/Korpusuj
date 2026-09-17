# Wyszukiwanie uruchamiane z GUI

## Wejście

Wyszukiwanie rozpoczyna callback związany z przyciskiem lub klawiszem Enter. Callback odczytuje tekst zapytania, wybrany korpus, szerokość kontekstu i ustawienia sortowania. Następnie rejestruje nową operację i uruchamia worker.

## Przygotowanie backendu

Dla załadowanego korpusu `engine.py` pobiera `LazyCorpus` ze stanu. Executor korzysta z `.search` przypisanego do tego obiektu. Parser i planner pochodzą z `korpusuj.search`; wrappery w `engine.py` zachowują nazwy używane przez starszy kod GUI.

## Routing

Standardowo zapytanie trafia do ścieżki indeksowanej. Przed wykonaniem routing może skierować rozpoznany, nieobsługiwany bezpiecznie przypadek do adaptera fallbacku. Adapter wywołuje `_legacy_find_lemma_context(...)` na zmaterializowanych danych.

Informacja o wybranej trasie jest zapisywana w diagnostyce. Fallback nie jest wybierany jako równorzędna optymalizacja.

## Worker i publikacja

Worker wykonuje zapytanie i otrzymuje listę albo `SearchCursor`. Przed zapisaniem wyniku sprawdza token aktywnej operacji. Wynik nieaktualnego workera jest pomijany.

Aktualny wynik zostaje przypisany do `full_results_sorted`, numer strony jest ustawiany na początek, a `display_page(...)` jest planowane w wątku GUI.

## `display_page(query, selected_corpus)`

Funkcja oblicza indeks początkowy i końcowy bieżącej strony. Dla kursora wywołuje `get_range(start, end)`. Dla listy wykonuje zwykły wycinek.

Po zbudowaniu wierszy aktualizuje tabelę i nawigację. Przycisk następnej strony pozostaje aktywny, jeśli bieżąca strona jest pełna i kursor może mieć dalsze trafienia. Przycisk ostatniej strony wymaga znanej dokładnej liczby trafień.

## Sortowanie

Sortowanie widocznej strony nie wymaga pobrania wszystkich wyników. Sortowanie globalne może zmaterializować cały kursor, ponieważ kolejność trzeba ustalić przed wybraniem strony.
