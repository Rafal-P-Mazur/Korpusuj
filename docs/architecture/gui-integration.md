# Połączenie modułów z GUI

## Otwarcie projektu

Po wyborze Parquet GUI wywołuje `prepare_loaded_corpus_bundle`. Z bundle'a zapisuje:

- ścieżkę korpusu;
- `LazyCorpus`;
- `LazyTermIndex` dla `base` i `orth`;
- metadane frekwencyjne;
- listę kolumn i informacje o warstwach anotacji.

## Uruchomienie wyszukiwania

GUI zapisuje tekst zapytania i ustawienia kontekstu w stanie wyszukiwania, a następnie uruchamia wykonanie w tle. Wynik trafia do `full_results_sorted` jako lista albo kursor.

`display_page` pobiera tylko wiersze bieżącej strony. Każdy wiersz otrzymuje tekst kontekstu, metadane oraz informacje potrzebne do późniejszego pobrania pełnego dokumentu.

## Widok pełnego dokumentu

Po kliknięciu konkordancji GUI odczytuje pełny tekst dokumentu. Pozycje tokenów i znaków służą do zaznaczenia dopasowania. Jeśli korpus zawiera NER lub koreferencję, widok korzysta z tablic anotacji tego samego dokumentu.

## Creator

Okno creatora zbiera pliki, ustawienia modelu i warstw oraz ścieżkę wyniku. Następnie tworzy `CreatorRunOptions` i wywołuje `run_creator_job`. `GuiProgressReporter` przekazuje komunikaty i wartości paska postępu do głównego wątku GUI.

## Raporty HTML

Raporty semantyczne, BERTopic i inne widoki HTML są otwierane w pywebview. W wersji instalacyjnej osobny proces jest uruchamiany przez ten sam plik EXE z argumentem określającym zadanie.

## Logi

Błędy GUI są zapisywane w pliku logu wyznaczonym przez `runtime_paths`. Diagnostyka wyszukiwania zapisuje dodatkowe informacje o wybranej trasie, planie i czasach etapów.
