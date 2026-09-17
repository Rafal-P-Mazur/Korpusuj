# `korpusuj.corpus.creator_cli`

## `build_arg_parser()`

Tworzy parser dla wejść, outputu, backendu NLP, metadanych, mapowania kolumn, wznowienia, warstw NER i koreferencji, korekt lematów oraz formatu statusu.

`--input` jest opcją powtarzalną. `_expand_input_paths(...)` rozwija katalog do obsługiwanych plików bez rekurencyjnego przechodzenia po podkatalogach. Duplikaty ścieżek są usuwane z zachowaniem kolejności.

## Konfiguracja metadanych

`_load_mapping_json(...)` odczytuje obiekt JSON i sprawdza, że klucze oraz wartości są napisami. `_read_metadata_columns(...)` odczytuje nagłówki XLSX. `_build_metadata_configuration(...)` łączy oba źródła i zwraca ścieżkę metadanych oraz mapowanie w formacie wymaganym przez creator.

## `StderrProgressReporter`

Reporter implementuje protokół creatora. Komunikaty statusu, ostrzeżenia i błędy wypisuje na `stderr` z prefiksem poziomu. Metody liczbowego postępu zachowują ostatnie wartości, ale nie mieszają ich z końcowym JSON w `stdout`.

## `main(argv=None)`

Kolejność wykonania:

1. parsuje argumenty;
2. rozwija wejścia;
3. sprawdza ścieżkę `.parquet` outputu;
4. przygotowuje metadane i mapowanie;
5. tworzy `CreatorRunOptions`;
6. tworzy `StderrProgressReporter`, chyba że użyto `--quiet`;
7. wywołuje `run_creator_job(...)`;
8. zmienia `CreatorRunResult` na słownik statusu;
9. wypisuje status jako JSON albo tekst.

Błąd konfiguracji argumentów zwraca kod 2. Nieobsłużony wyjątek wykonania zwraca kod 1. Pomyślne zadanie zwraca 0.
