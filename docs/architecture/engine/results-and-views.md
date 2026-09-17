# Wyniki i widoki w `engine.py`

## Nawigacja

`next_page()`, `prev_page()`, `first_page()` i `last_page()` zmieniają `current_page` i ponownie wywołują `display_page(...)`. `last_page()` działa dopiero po ustaleniu dokładnej liczby trafień.

`go_to_page(...)`, `next_p(...)` i pokrewne callbacki obsługują paginator tabeli. `update_table(...)` synchronizuje tabelę po zmianie strony lub sortowania.

## Pełny dokument

Wiersz konkordancji zawiera `doc_id`, pozycje tokenów i leniwe odwołanie do pełnego tekstu. Kliknięcie wiersza odczytuje dokument z `LazyCorpus`, wyznacza zakres dopasowania i aktualizuje panel pełnego kontekstu.

NER i koreferencja są podświetlane na podstawie `start_ids`, `end_ids`, `ners`, `corefs` i `coref_mentions` dokumentu. Brak odpowiedniej warstwy w `korpus_meta` powoduje pominięcie podświetlenia.

## Statystyki i kolokacje

GUI przekazuje wynik wyszukiwania do funkcji z `korpusuj.search.statistics` i `korpusuj.search.collocations`. Kolokacje składniowe korzystają z dependency runtime dla dokumentów należących do trafień.

## Eksport

Eksport konkordancji rozwiązuje pełne konteksty tylko dla eksportowanych wierszy i przekazuje je do `korpusuj.export.excel`. Tworzenie podkorpusu wybiera pełne dokumenty, a nie pojedyncze trafienia, i zapisuje nowy Parquet przez `korpusuj.export.subcorpus`.
