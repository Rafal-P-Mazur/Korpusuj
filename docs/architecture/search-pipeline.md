# Pipeline wyszukiwania

## Przebieg

```text
CQL -> parser -> planner -> backend + SearchCursor
    -> końcowe liczenie -> materializacja -> GUI / CLI / analityka / eksport
```

## Parser i planner

`parser.py` rozpoznaje CQL. `planner.py` przygotowuje plan wykonania i wskazuje warunki wymagające dokładnej weryfikacji.

## Backend i `SearchCursor`

`backend.py` wiąże Parquet z `.search`. `SearchCursor` pobiera kandydatów z indeksu i dokładnie weryfikuje warunki. Dependency korzysta z `.dep_cache`.

## Dokładne `total_hits`

`count_final_searchcursor_hits` centralizuje końcowe liczenie. Wyniku publicznego nie wolno obcinać limitem strony, podglądu ani prefetchu. Estymacja jest używana tylko wtedy, gdy cursor gwarantuje jej dokładność.

## Materializacja

`result_materialization.py` zmienia leniwy wynik w reprezentację potrzebną konsumentowi. GUI może pobierać stronę, a pełny eksport lub sortowanie mogą wymagać szerszej materializacji.

## Analityka i eksport

`statistics.py` oblicza statystyki, `collocations.py` kolokacje, a `korpusuj/export/` zapisuje wyniki i subkorpusy. GUI i CLI są konsumentami tych modułów.

## Granica GUI

`engine.py` uruchamia zadania w tle i publikuje wynik do widoków. Parser, planner, cursor, końcowe liczenie i główne obliczenia analityczne mają właścicieli w `korpusuj/search/`.
