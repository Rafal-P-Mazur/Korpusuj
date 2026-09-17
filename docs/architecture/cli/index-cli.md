# `korpusuj.index.cli`

## Parser podkomend

CLI udostępnia `create`, `status` i `rebuild`. Wspólne argumenty wskazują Parquet, opcjonalną ścieżkę `.search`, profil indeksu albo listę atrybutów oraz format odpowiedzi.

`create` i `rebuild` przyjmują także `--force` i ustawienie postępu. `status` tylko odczytuje stan.

## `main(argv=None)`

Funkcja normalizuje ścieżki i wyprowadza ścieżkę `.dep_cache` z Parquet. Dla `status` wywołuje `inspect_index_artifacts(...)` i emituje otrzymany stan.

Dla `create` funkcja najpierw sprawdza istniejący zestaw. Obecność dowolnego sidecara blokuje zapis bez `--force`. Dla `rebuild` świeży zestaw nie jest budowany ponownie bez `--force`.

Budowa wywołuje `build_index_artifacts_atomic(...)`. Callback postępu zapisuje informacje na `stderr`; wynik JSON lub tekst trafia na `stdout`.

`CliInputError` oznacza błąd parametrów lub niedozwolony stan wejściowy. Taki błąd zwraca kod 2. Nieobsłużony wyjątek zwraca 1. Sukces zwraca 0. `status` może zwracać kod zależny od stanu zestawu, aby skrypt automatyzujący odróżnił `fresh` od brakującego lub uszkodzonego artefaktu.
