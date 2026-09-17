# `korpusuj.search.headless_runner`

## Cel modułu

Moduł tworzy środowisko wyszukiwania bez importowania `engine.py` i GUI. Jest używany przez CLI oraz testy integracyjne.

## `configure_non_gui_search_cursor_runtime(...)`

Przekazuje `SearchCursor` wartości potrzebne bez GUI: rozmiar pełnego kontekstu, limity partii kandydatów i opcjonalny cache dependency. Funkcje zależne od kontrolek GUI są zastępowane stałymi lub lokalnymi adapterami.

## `build_lazy_corpus_for_headless(...)`

Tworzy `LazyCorpus` dla Parquet i `.search`. Jeśli lista kolumn, liczba dokumentów lub metadane nie zostały przekazane, funkcja odczytuje je ze źródła.

## `build_corpus_search_executor_for_headless(...)`

Konfiguruje zależności `SearchExecutor` i zwraca `CorpusSearchExecutor` związany z utworzonym `LazyCorpus`.

## `build_non_gui_find_lemma_context_adapter(...)`

Tworzy funkcję o interfejsie zgodnym z wywołaniami oczekującymi `find_lemma_context`. Adapter uruchamia executor, zachowuje dokładne `total_hits`, stosuje limit i offset, a opcjonalnie normalizuje wiersze.

## `MaterializedSearchResults`

Jest listą z dodatkowymi polami `total_hits`, źródłem liczby i strategią liczenia. Dzięki temu ograniczona lista zwróconych wierszy nie traci informacji o liczbie wszystkich trafień.

## `build_headless_context_from_parquet(...)`

Składa pełny `SearchBackendContext`: `LazyCorpus`, executor, adapter wyszukiwania, metadane oraz funkcje potrzebne CLI. Zwrócony kontekst jest wejściem `_run_cli_search(...)`.
