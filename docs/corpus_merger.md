# Scalanie gotowych korpusów

Moduł `korpusuj.corpus.merger` łączy co najmniej dwa gotowe korpusy Korpusuj w jeden kanoniczny plik Parquet. Wykorzystuje istniejące dokumenty i anotacje. Nie uruchamia tokenizacji, lematyzacji, analizy zależnościowej, NER, koreferencji ani modeli NLP.

## Miejsce w architekturze

```text
korpus_a.parquet + korpus_b.parquet
    -> korpusuj.corpus.merger
    -> korpus_wynikowy.parquet
    -> nowy korpus_wynikowy.search + korpus_wynikowy.dep_cache
```

Parquet pozostaje kanonicznym źródłem danych. `.search` i `.dep_cache` są artefaktami pochodnymi. Merger ich nie kopiuje ani nie scala. Po utworzeniu wyniku należy zbudować nowy zestaw indeksów.

## Wymagania zgodności

Wejścia muszą być zgodnymi korpusami Korpusuj, a nie dowolnymi plikami Parquet. Merger sprawdza:

- wymagane kolumny dokumentowe i językowe;
- nazwy, kolejność oraz bezpieczną zgodność typów Arrow;
- zgodność `annotation_layers`;
- długości tablic anotacji względem `tokens`;
- współrzędne `coref_mentions`;
- kolizje `Oryginalna_nazwa_pliku`;
- identyczne treści wykrywane przez SHA-256;
- rozdzielność inputów i outputu.

Puste typy Arrow oparte na `null` mogą zostać uzgodnione z odpowiadającym typem zawierającym dane. Inne niezgodności blokują operację.

## Podstawowe użycie

Polecenie należy uruchomić z rootu projektu:

```powershell
python -m korpusuj.corpus.merger_cli `
    --input ".\korpusy\prasa_2022_2025.parquet" `
    --input ".\korpusy\prasa_2026.parquet" `
    --output ".\korpusy\prasa_2022_2026.parquet" `
    --report ".\korpusy\prasa_2022_2026_merge_report.md"
```

Wersja jednowierszowa:

```powershell
python -m korpusuj.corpus.merger_cli --input ".\korpusy\prasa_2022_2025.parquet" --input ".\korpusy\prasa_2026.parquet" --output ".\korpusy\prasa_2022_2026.parquet" --report ".\korpusy\prasa_2022_2026_merge_report.md"
```

Postęp trafia na `stderr`, a końcowy wynik JSON na `stdout`.

## Wiele wejść i kolejność dokumentów

`--input` jest powtarzalne:

```powershell
python -m korpusuj.corpus.merger_cli `
    --input ".\korpusy\a.parquet" `
    --input ".\korpusy\b.parquet" `
    --input ".\korpusy\c.parquet" `
    --output ".\korpusy\abc.parquet"
```

Kolejność wejść jest częścią kontraktu. Wynik zawiera kolejno wszystkie dokumenty z pierwszego, drugiego i następnych wejść. Merger nie sortuje dokumentów i nie odtwarza historycznej kolejności sprzed wcześniejszego podziału korpusu.

Parquet nie przechowuje globalnego `doc_id` nadawanego przez wyszukiwarkę. Nową numerację tworzy builder `.search` po scaleniu.

## Argumenty CLI

- `--input FILE.parquet` wskazuje wejście i musi wystąpić co najmniej dwa razy.
- `--output FILE.parquet` wskazuje nowy plik wynikowy.
- `--report FILE.md` ustala raport Markdown. Domyślnie raport powstaje obok wyniku.
- `--replace` pozwala zastąpić istniejący output dopiero po przygotowaniu i walidacji nowego pliku.
- `--batch-size N` ustala liczbę dokumentów w partii. Domyślnie `128`.
- `--no-duplicate-check` wyłącza kontrolę nazw i treści. Jest to opcja ryzykowna i niezalecana.
- `--allow-undeclared-annotation-layers` włącza tryb historyczny.

```powershell
python -m korpusuj.corpus.merger_cli --help
```

## Historyczne korpusy bez `annotation_layers`

Nowe korpusy powinny deklarować w `korpus_meta`:

```json
{
  "annotation_layers": {
    "ner": true,
    "coreference": true
  }
}
```

Jeżeli wszystkie świadomie zweryfikowane wejścia historyczne nie mają tego pola, można użyć:

```powershell
python -m korpusuj.corpus.merger_cli `
    --input ".\korpusy\historyczny_a.parquet" `
    --input ".\korpusy\historyczny_b.parquet" `
    --output ".\korpusy\historyczny_merged.parquet" `
    --allow-undeclared-annotation-layers
```

Tryb nie pozwala mieszać wejść deklarowanych i niedeklarowanych. Nadal wymaga zgodnego schematu i poprawnych danych. Wynik otrzymuje konserwatywnie `ner: false` oraz `coreference: false`, ponieważ brakujących informacji nie można udowodnić. Dla nowych korpusów nie należy używać tej opcji.

## Kontrola duplikatów

Domyślnie merger blokuje powtórzoną `Oryginalna_nazwa_pliku` oraz identyczną treść dokumentu. Nie deduplikuje automatycznie. `--no-duplicate-check` należy stosować wyłącznie po osobnej analizie wejść.

## Przeliczane metadane

Merger przelicza z gotowych kolumn:

- `total_tokens` z długości `tokens`;
- `base_tf` z `lemmas`;
- `orth_tf` z `tokens`;
- `monthly_token_counts` z `Data publikacji` oraz długości `tokens`.

Nie jest to ponowne NLP. Nierozpoznane daty są zgłaszane jako ostrzeżenia i pomijane w statystykach miesięcznych.

## Pliki etapowe i publikacja

Obok outputu mogą powstać:

```text
output.parquet.merge_stage
output.parquet.final_stage
```

Merger nie korzysta z systemowego `%TEMP%`. Finalny plik jest publikowany dopiero po ponownym otwarciu i walidacji. Wejścia nie są modyfikowane. Bez `--replace` istniejący output blokuje operację.

## Raport i kontrola wyniku

Raport zawiera listę wejść, liczbę dokumentów, `total_tokens`, ostrzeżenia oraz informację o konieczności odbudowania sidecarów.

```powershell
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile(r'.\korpusy\prasa_2022_2026.parquet'); print('rows=',p.metadata.num_rows,'row_groups=',p.metadata.num_row_groups,'columns=',len(p.schema_arrow.names)); p.close()"
```

Rozmiar pliku i liczba row groups mogą różnić się od sumy wejść z powodu ponownego kodowania i kompresji. Nie jest to samo w sobie oznaką utraty danych.

## Budowa nowych `.search` i `.dep_cache`

Nie należy kopiować ani scalać sidecarów wejściowych. Nowy zestaw tworzy index CLI:

```powershell
python -m korpusuj.index.cli create ".\korpusy\prasa_2022_2026.parquet" --progress on --pretty
```

Kontrola:

```powershell
python -m korpusuj.index.cli status ".\korpusy\prasa_2022_2026.parquet" --pretty
```

## Typowe błędy

- `Brak korpus_meta`: wejście nie jest obsługiwanym korpusem Korpusuj.
- `Wszystkie wejścia nie deklarują annotation_layers`: po osobnej weryfikacji można użyć trybu historycznego.
- `Nie można mieszać wejść z zadeklarowanym i niezadeklarowanym annotation_layers`: zestaw jest niejednorodny.
- `Niezgodne annotation_layers`: wejścia deklarują różne warstwy.
- `Niezgodne nazwy lub kolejność kolumn` lub `Niezgodny typ`: schematy są niekompatybilne.
- `Kolizja Oryginalna_nazwa_pliku`: powtórzona nazwa źródłowa.
- `Duplikat treści`: identyczna treść w co najmniej dwóch dokumentach.
- `Niezgodna długość .../tokens`: uszkodzona równoległość anotacji.
- `Output już istnieje`: wybierz inną ścieżkę lub świadomie użyj `--replace`.
- `Output nie może być inputem`: wynik musi mieć inną ścieżkę.

## Ograniczenia

Merger nie obsługuje dowolnych Parquetów, nie migruje starych schematów, nie naprawia anotacji, nie uruchamia NLP, nie deduplikuje automatycznie, nie sortuje dokumentów, nie odtwarza historycznych `doc_id`, nie scala sidecarów i nie gwarantuje identycznego fizycznego rozmiaru ani podziału na row groups.

## Powiązana dokumentacja

- [Korzystanie z interfejsu wiersza poleceń](cli.md)
- [Korpus i artefakty indeksowe](architecture/corpus-and-index-artifacts.md)
- [Pipeline creatora](architecture/creator-pipeline.md)
