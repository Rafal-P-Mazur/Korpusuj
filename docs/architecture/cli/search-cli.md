# `korpusuj.search.cli`

## Parser

`build_arg_parser()` definiuje trzy wzajemnie wykluczające źródła zapytania: `--query`, `--query-file` i `--query-list`. Pozostałe grupy argumentów kontrolują stronicowanie, kontekst, format wyniku, statystyki, kolokacje, profil kolokacyjny, podkorpus i diagnostykę.

## `main(argv=None)`

Po sparsowaniu argumentów `main()` rozdziela wykonanie listy zapytań od pojedynczego zapytania. Dla pojedynczego zapytania buduje kontekst bez GUI przez `build_headless_context_from_parquet(...)`, a następnie przekazuje wykonanie do `_run_cli_search(...)`.

## `_run_cli_search(args, context, query, corpus_name)`

Funkcja buduje `SearchRequest`, uruchamia wyszukiwanie w kontekście headless i otrzymuje wynik z zachowanym `total_hits`. Następnie, zależnie od argumentów, dołącza statystyki, kolokacje albo profil i decyduje, czy zwrócić także wiersze konkordancji.

Pełne konteksty są rozwiązywane dla wierszy, które znajdą się w odpowiedzi. Kontrole `--fields`, `--no-extended-context` i `--max-context-chars` są stosowane po zbudowaniu znormalizowanego wyniku.

## Lista zapytań

`_run_query_list_cli(...)` odczytuje niepuste wiersze pliku i uruchamia każde zapytanie osobno. `--continue-on-error` zamienia błąd pojedynczego zapytania na rekord błędu i pozwala przejść do następnego. Bez tej opcji pierwszy błąd kończy batch.

JSONL zawiera osobny rekord dla każdego zapytania. JSON zawiera zbiorczą strukturę. Format tekstowy generuje czytelne sekcje dla kolejnych zapytań.

## Formaty wyjścia

JSON, JSONL i tekst są wypisywane na `stdout` albo do `--output`. CSV i XLSX wymagają pliku docelowego i korzystają z modułów `korpusuj.export`.

Postęp, spinner i diagnostyka trafiają na `stderr`. Dzięki temu `stdout` pozostaje poprawnym dokumentem JSON lub strumieniem JSONL.
