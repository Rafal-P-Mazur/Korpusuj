# Format korpusu

## Jeden wiersz na dokument

Plik Parquet zawiera jeden wiersz dla każdego dokumentu. `Oryginalna_nazwa_pliku` identyfikuje materiał źródłowy, a `Treść` zawiera pełny tekst dokumentu.

Aktualny merger wymaga następujących kolumn językowych:

```text
tokens
lemmas
postags
full_postags
upostags
deprels
word_ids
sentence_ids
head_ids
start_ids
end_ids
ners
corefs
coref_mentions
```

## Tablice tokenowe

`tokens` wyznacza liczbę tokenów dokumentu. Każda z poniższych kolumn ma dokładnie tyle samo elementów:

```text
lemmas
postags
full_postags
upostags
deprels
word_ids
sentence_ids
head_ids
start_ids
end_ids
ners
corefs
```

`start_ids` i `end_ids` wskazują pozycje znakowe tokenu w `Treść`. `sentence_ids`, `word_ids` i `head_ids` opisują miejsce tokenu w strukturze zależnościowej.

## Koreferencja

`coref_mentions` zawiera pełne wzmianki koreferencyjne. Każda wzmianka ma identyfikator klastra, zakres tokenów i token główny. Zakres jest półotwarty: `start` należy do wzmianki, a `end` wskazuje pierwszą pozycję za wzmianką. Token `head` musi spełniać warunek `start <= head < end`.

`corefs` jest równoległą do tokenów reprezentacją używaną przez część kodu zgodnościowego i prezentację.

## Metadane dokumentu

Poza kolumnami technicznymi Parquet może zawierać autora, tytuł, datę publikacji, gatunek i własne pola użytkownika. Wyszukiwanie metadanych działa na wartościach zapisanych w wierszu dokumentu.

## `korpus_meta`

Metadane schematu Parquet zawierają obiekt `korpus_meta`. Przechowuje on między innymi:

- całkowitą liczbę tokenów;
- frekwencje lematów i form tekstowych;
- miesięczne liczby tokenów;
- deklarację warstw NER i koreferencji;
- informacje o zastosowanych regułach korekty lematów.

Creator zapisuje te dane podczas publikacji korpusu. Merger przelicza wartości frekwencyjne na podstawie łączonych dokumentów.

## Identyfikator dokumentu w indeksie

`doc_id` używany przez `.search` odpowiada pozycji dokumentu w Parquet. Nie jest trwałym identyfikatorem zapisanym w samym korpusie. Po zmianie kolejności dokumentów lub utworzeniu podkorpusu indeks musi zostać zbudowany ponownie.
