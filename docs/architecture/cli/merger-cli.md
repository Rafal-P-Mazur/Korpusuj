# `korpusuj.corpus.merger_cli`

## `parser()`

Parser wymaga co najmniej dwóch wystąpień `--input` i jednego `--output`. Dodatkowe opcje wybierają raport, zastąpienie istniejącego pliku, rozmiar partii, kontrolę duplikatów i tryb historycznych korpusów bez `annotation_layers`.

## `main(argv=None)`

Funkcja parsuje argumenty, sprawdza liczbę wejść i dodatni `batch_size`, a następnie wywołuje `merge_corpora(...)`.

Callback postępu wypisuje `[merge] wykonane/razem` na `stderr`. Sukces wypisuje `result.to_dict()` jako JSON na `stdout` i zwraca kod 0.

`CorpusMergeError` jest oczekiwanym błędem walidacji wejść i zwraca kod 2. Inny wyjątek zwraca kod 1. Oba rodzaje błędów są serializowane jako JSON zawierający `success`, `error_type` i `error`.
