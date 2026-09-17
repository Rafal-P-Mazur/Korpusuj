# Mapa pakietów i ważnych modułów

## Uruchamianie i konfiguracja

- `Korpusuj.py`: punkt wejścia GUI.
- `engine.py`: stan GUI, obsługa zdarzeń, konfiguracja runtime'u i prezentacja wyników.
- `korpusuj.runtime_paths`: ścieżki konfiguracji, logów, modeli, pamięci podręcznych i zasobów.
- `korpusuj.config`: wspólne ustawienia aplikacji.

## `korpusuj.corpus`

- `creator.py`: okno creatora oraz adapter istniejącego GUI.
- `creator_core.py`: `CreatorRunOptions` i kontrakt reportera.
- `creator_gui_adapter.py`: reporter dla interfejsu graficznego.
- `creator_orchestration.py`: `run_creator_job` i przebieg całego zadania.
- `creator_io.py`: odczyt wejść, XLSX i bezpieczne rozpakowywanie ZIP.
- `creator_chunking.py`: podział dokumentów na fragmenty.
- `creator_nlp.py`: inicjalizacja i stan modeli NLP.
- `lemma_corrections.py`: reguły korekty lematów.
- `loading.py`: `LoadedCorpusBundle`, kontrola `.search` i utworzenie `LazyCorpus`.
- `info.py`: dane prezentowane w informacji o korpusie.
- `merger.py`: walidacja i scalanie zgodnych korpusów.
- `merger_cli.py`: interfejs terminalowy mergera.

## `korpusuj.index`

- `builder.py`: budowa `.search` z Parquet.
- `sqlite_index.py`: schemat SQLite, `SearchIndex` i `LazyTermIndex`.
- `postings.py`: kodowanie i odczyt postingów.
- `status.py`: kontrola świeżości i integralności `.search`.
- `cli.py`: polecenia `create`, `status` i `rebuild` dla pary sidecarów.
- `lru.py`: ograniczone pamięci podręczne indeksu.

## `korpusuj.dependency`

- `maps.py`: mapy nadrzędników i podrzędników.
- `disk_cache.py`: baza `.dep_cache`.
- `lifecycle.py`: budowa, walidacja i publikacja `.search` oraz `.dep_cache`.
- `runtime_state.py`: kolekcje i parametry dependency runtime.
- `runtime.py`: odczyt, preładowanie i warmup map.
- `policy.py`: stałe formatu i ustawienia techniczne.

## `korpusuj.search`

- `parser.py`: składnia CQL.
- `planner.py`: plan wykonania zapytania.
- `executor.py`: wykonanie planu.
- `backend.py`: `LazyCorpus` i odczyt dokumentów.
- `cursor.py`: `SearchCursor`, stronicowanie i leniwe konteksty.
- `cursor_runtime.py`: funkcje dependency przekazywane kursorowi.
- `result_materialization.py`: dokładne liczenie i przygotowanie wyników.
- `statistics.py`: statystyki trafień.
- `collocations.py`: kolokacje liniowe i składniowe.
- `models.py`: modele stanu wyszukiwania.
- `output_schema.py`: publiczny format wyniku CLI.
- `headless_runner.py`: wykonanie bez GUI.
- `legacy_adapter.py`: wywołanie fallbacku zgodnościowego.
- `cli.py`: wyszukiwanie, analizy i eksport z terminala.

## Pozostałe pakiety

- `korpusuj.export`: eksport tabel i tworzenie podkorpusów.
- `korpusuj.semantic`: model i raporty semantyczne oraz widok sieci.
- `korpusuj.topics`: integracja BERTopic.
- `korpusuj.ui`: komponenty interfejsu, tabele, podpowiedzi, fiszki i widoki pomocnicze.
- `korpusuj.utils`: małe funkcje współdzielone przez kilka pakietów.

`legacy_engine.py` jest historyczną kopią aplikacji sprzed indeksów SQLite i modularyzacji. Nie uczestniczy w uruchomieniu aktualnej wersji.
