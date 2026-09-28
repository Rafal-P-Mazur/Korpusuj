# Obraz systemu

## Punkty wejścia

`Korpusuj.py` uruchamia interfejs graficzny. `engine.py` tworzy okno aplikacji, przechowuje stan otwartego projektu i łączy zdarzenia interfejsu z modułami pakietu `korpusuj`.

Polecenia CLI uruchamiają bezpośrednio moduły pakietu:

```text
python -m korpusuj.corpus.creator_cli
python -m korpusuj.corpus.merger_cli
python -m korpusuj.index.cli
python -m korpusuj.search.cli
```

GUI i CLI korzystają z tych samych modułów tworzenia korpusu, indeksowania i wyszukiwania. `engine.py` nadal zawiera część kodu zgodnościowego oraz konfigurację obiektów runtime potrzebnych przez wyszukiwarkę.

## Pliki należące do korpusu

```text
korpus.parquet
korpus.search
korpus.dep_cache
```

`korpus.parquet` przechowuje dokumenty, tokeny, anotacje i metadane. Jest jedynym z tych trzech plików, którego nie można odtworzyć bez ponownego przygotowania korpusu.

`korpus.search` jest indeksem SQLite używanym do wyboru dokumentów i pozycji tokenowych pasujących do warunków zapytania.

`korpus.dep_cache` jest bazą SQLite z mapami relacji nadrzędnik-podrzędnik. Korzystają z niej zapytania składniowe i kolokacje składniowe.

## Tworzenie korpusu

```text
pliki źródłowe
  -> odczyt tekstu i metadanych
  -> normalizacja technicznych znaków Unicode
  -> podział długich tekstów
  -> analiza Stanza albo spaCy
  -> opcjonalne NER i koreferencja
  -> opcjonalne korekty lematów
  -> zapis częściowy
  -> finalny Parquet
```

GUI creatora i creator CLI wywołują `run_creator_job` z obiektem `CreatorRunOptions`. Różnią się sposobem zbierania opcji i prezentowania postępu, ale nie przebiegiem anotacji.

## Otwieranie korpusu

`prepare_loaded_corpus_bundle` odczytuje schemat Parquet i metadane `korpus_meta`. Następnie sprawdza plik `.search`. Brakujący lub nieaktualny indeks zostaje zbudowany przed zwróceniem `LoadedCorpusBundle`.

Bundle zawiera `LazyCorpus`, dlatego otwarcie projektu nie wymaga wczytania wszystkich dokumentów do jednego DataFrame. Dokumenty są odczytywane z Parquet wtedy, gdy potrzebuje ich wyszukiwanie, widok pełnego tekstu albo analiza.

## Wyszukiwanie

```text
CQL
  -> parser
  -> planner
  -> executor
  -> SearchCursor
  -> dokładne sprawdzenie kandydatów
  -> wyniki dla GUI albo CLI
```

Indeks wskazuje kandydatów do sprawdzenia. `SearchCursor` sprawdza na danych dokumentu warunki, których sam posting nie rozstrzyga, na przykład sekwencję tokenów, relacje składniowe lub warunki zdaniowe.

`_legacy_find_lemma_context` jest fallbackiem zgodnościowym. Jest używany tylko wtedy, gdy routing odrzuci wykonanie przez ścieżkę indeksowaną. Nie jest drugim standardowym backendem wyszukiwania.

## Analizy i eksport

Graf eksploracyjny i raport analityczny znajdują się w `korpusuj.semantic`. Raport buduje graf mutual 5-NN, używa `SenseInducer.chinese_whispers(...)` do podziału i osobno opisuje członków klastrów oraz relacje pozostałych lematów do centroidów. Modelowanie tematyczne znajduje się w `korpusuj.topics`.
