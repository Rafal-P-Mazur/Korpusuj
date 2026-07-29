# Mapa modułów

## GUI

- `Korpusuj.py` — punkt uruchomienia interfejsu graficznego;
- `engine.py` — integracja interfejsu graficznego ze wspólnym rdzeniem aplikacji;
- `korpusuj/ui/` — komponenty widoków.

## `korpusuj/corpus/`

Tworzenie korpusu z poziomu GUI i CLI, opcje uruchomienia, odczyt i zapis danych, przetwarzanie porcjami, modele NLP, `run_creator_job`, obsługa postępu, odczyt korpusu i informacje techniczne.

## `korpusuj/index/`

Tworzenie i udostępnianie indeksu `.search`:

- `builder.py`;
- `sqlite_index.py`;
- `status.py`;
- `cli.py`;
- `postings.py`;
- `lru.py`.

## `korpusuj/dependency/`

Tworzenie map zależności składniowych i pliku `.dep_cache`:

- `maps.py`;
- `disk_cache.py`;
- `lifecycle.py`;
- `runtime.py`.

## `korpusuj/search/`

- `parser.py` — analiza składni CQL;
- `planner.py` — przygotowanie planu wykonania zapytania;
- `backend.py` — dostęp do korpusu Parquet i indeksu `.search`;
- `cursor.py` — udostępnianie wyników przez `SearchCursor`;
- `result_materialization.py` — zliczanie i materializacja wyników;
- `statistics.py` i `collocations.py` — statystyki i kolokacje;
- `cli.py` — publiczny interfejs wiersza poleceń;
- `output_schema.py`, `diagnostics.py` i `errors.py` — schemat wyników, diagnostyka oraz obsługa błędów.

## Pozostałe pakiety

- `korpusuj/export/` — eksport wyników i podkorpusów;
- `korpusuj/semantic/` — sieci semantyczne, wektory, profile i raporty;
- `korpusuj/topics/` — modelowanie tematyczne za pomocą BERTopic;
- `korpusuj/utils/` — współdzielone narzędzia pomocnicze.

## Podział odpowiedzialności

Nowa logika powinna trafiać do modułu odpowiedzialnego za daną funkcję. Plik `engine.py` integruje funkcje z interfejsem graficznym, ale nie jest domyślnym miejscem dla nowej logiki dziedzinowej.
