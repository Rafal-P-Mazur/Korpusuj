# Korpus i artefakty indeksowe

## Zestaw

```text
korpus.parquet
korpus.search
korpus.dep_cache
```

## Parquet

Parquet jest kanonicznym źródłem dokumentów, tokenów, anotacji i metadanych. Creator zapisuje Parquet, a indeksowanie i wyszukiwanie traktują go jako źródło.

## `.search`

Pochodny indeks SQLite z terminami, postingami, danymi dokumentów i metadanymi zgodności. Nazwa jest wyprowadzana ze ścieżki Parquet.

## `.dep_cache`

Pochodny magazyn SQLite map zależności składniowych budowanych z Parquet.

## Lifecycle

```powershell
python -m korpusuj.index.cli create korpus.parquet
python -m korpusuj.index.cli status korpus.parquet
python -m korpusuj.index.cli rebuild korpus.parquet
```

Zestaw jest budowany przez staging i publikowany po walidacji obu artefaktów. `--force` wymusza przebudowę.

## Stany

- `fresh` — zgodny i kompletny;
- `missing` — brak artefaktu;
- `stale` — zmieniony Parquet;
- `incompatible` — niezgodny kontrakt;
- `corrupt` — błąd integralności lub kompletności.

## Odtwarzalność

`.search` i `.dep_cache` są odtwarzalne z Parquet. Kopie zapasowe powinny przede wszystkim chronić Parquet i dane źródłowe.
