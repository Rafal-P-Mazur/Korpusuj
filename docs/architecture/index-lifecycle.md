# Indeks wyszukiwania i pamięć zależnościowa

## Dwa artefakty budowane razem

`korpusuj.index.cli` zarządza zestawem:

```text
korpus.search
korpus.dep_cache
```

Polecenia `create` i `rebuild` wywołują `build_index_artifacts_atomic`. Funkcja tworzy oba pliki etapowe, sprawdza je i publikuje razem.

## Budowa `.search`

`SearchIndexBuilder` czyta Parquet partiami. Dla indeksowanych atrybutów zapisuje terminy i postingi zawierające `doc_id` oraz pozycje tokenów.

Profil `compact` indeksuje `base` i `orth`. Profil `full` dodaje `pos`, `upos`, `deprel` i `ner`.

Indeks zapisuje także dane dokumentowe oraz informacje pozwalające sprawdzić, czy pochodzi z bieżącej wersji Parquet.

## Budowa `.dep_cache`

Builder dependency czyta `sentence_ids`, `word_ids` i `head_ids`. Dla każdego tokenu ustala pozycję nadrzędnika w tablicy dokumentu i zapisuje wynik jako zwartą tablicę liczb int32.

Lista podrzędników nie jest zapisywana osobno. `LazyChildrenLookup` tworzy ją z tablicy nadrzędników podczas pierwszego użycia.

## Publikacja atomowa

Nowe pliki powstają jako:

```text
korpus.search.stage
korpus.dep_cache.stage
```

Przed publikacją oba przechodzą kontrolę integralności i zgodności z Parquet. Istniejący zestaw jest tymczasowo przenoszony do plików backupu publikacji. Jeśli publikacja któregokolwiek nowego pliku nie powiedzie się, poprzedni zestaw jest przywracany.

## Stany

- `fresh`: plik jest kompletny i odpowiada bieżącemu Parquet;
- `missing`: pliku nie ma;
- `stale`: Parquet zmienił się po utworzeniu pliku;
- `incompatible`: wersja formatu lub wymagany kontrakt nie odpowiada bieżącej aplikacji;
- `corrupt`: SQLite albo zapisane dane nie przeszły kontroli integralności.

Status całego zestawu odpowiada gorszemu stanowi jednego z dwóch plików.
