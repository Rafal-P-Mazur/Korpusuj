# Stan projektu w `engine.py`

## Korpusy

`files` mapuje nazwę wyświetlaną w GUI na ścieżkę Parquet. Ta sama nazwa jest kluczem w pozostałych kolekcjach związanych z projektem.

`dataframes` zachowuje historyczną nazwę. Po bieżącym ładowaniu wartością jest zwykle `LazyCorpus`, nie pełny DataFrame.

`inverted_indexes` przechowuje słownik zgodnościowy utworzony przez `build_lazy_corpus_bundle(...)`. Zawiera `LazyTermIndex` dla `base` i `orth` oraz statystyki frekwencyjne.

## Wynik wyszukiwania

`current_state` jest instancją `SearchState` i zawiera parametry bieżącego zapytania. `full_results_sorted` przechowuje wynik używany przez paginator. Może być listą albo obiektem kursora udostępniającym `get_range(...)`.

`current_page` i `rows_per_page` określają zakres przekazywany do `display_page(...)`. Dokładna liczba stron zależy od `_count_cache` kursora albo od estymacji oznaczonej jako dokładna.

## Ochrona współbieżnego wyszukiwania

`search_guard`, blokada stanu, znacznik trwającej operacji i token wyszukiwania zapobiegają publikacji wyniku przez nieaktualny worker. Nowe wyszukiwanie otrzymuje nowy token; worker sprawdza go przed zapisaniem wyniku w stanie GUI.

## Dependency

`dependency_maps_cache` przechowuje mapy dokumentów w RAM. `dependency_disk_caches` przechowuje otwarte obiekty `DependencyMapDiskCache`. Osobne kolekcje zawierają wątki warmupu i sygnały zatrzymania.

`DependencyRuntimeState` nie kopiuje tych danych. Otrzymuje te same kolekcje podczas konfiguracji, więc zmiany wykonane przez runtime są widoczne w `engine.py`.

## Zamknięcie korpusu

Usunięcie korpusu ze stanu obejmuje zatrzymanie jego warmupu, zamknięcie `.dep_cache`, usunięcie map RAM oraz zamknięcie obiektów dostępu do `.search` i Parquet. Nazwa korpusu jest kluczem pozwalającym usunąć powiązane elementy z kilku kolekcji.
