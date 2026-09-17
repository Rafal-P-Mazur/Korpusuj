# Wykonanie zapytania

## Główna ścieżka

Standardowe zapytanie wykonuje ścieżka indeksowana:

```text
tekst CQL
  -> parser
  -> plan zapytania
  -> wybór kandydatów z .search
  -> sprawdzenie kandydatów przez SearchCursor
  -> wyniki
```

Parser tworzy strukturę zapytania. Planner określa, które warunki mogą użyć indeksu i jakie dane dokumentu będą potrzebne do sprawdzenia wyniku.

## Kandydat i trafienie

Posting indeksu wskazuje dokument i pozycję, które mogą spełniać warunek. Jest kandydatem, dopóki `SearchCursor` nie sprawdzi wszystkich warunków zapytania.

Dodatkowego sprawdzenia wymagają między innymi:

- kolejność elementów sekwencji;
- luki między elementami;
- relacje `head` i `dependent`;
- warunki dotyczące całego zdania;
- zagnieżdżone warunki zależnościowe;
- filtry metadanych;
- role i zakresy koreferencji.

Pozytywnie zweryfikowany kandydat staje się trafieniem widocznym w wyniku.

## Atrybuty indeksowane

Dostępność postingów zależy od profilu `.search`. `base` i `orth` są dostępne w profilu compact. Profil full dodaje `pos`, `upos`, `deprel` i `ner`.

Brak postingów dla danego atrybutu nie zmienia znaczenia zapytania. Planner wybiera inną kotwicę, a warunek sprawdza na danych dokumentu.

## Zapytania zależnościowe

Warunki `head`, `dependent` i zagnieżdżone relacje używają map dependency. Kursor pobiera mapę konkretnego dokumentu z pamięci RAM, `.dep_cache` albo tworzy ją z kolumn Parquet.

## Alternatywy i filtry końcowe

Gałęzie połączone `||` mogą być wykonywane osobno i łączone w kursorze sumującym. Filtry frekwencji działają na zbiorze trafień ustalonym przez zapytanie podstawowe, dlatego wymagają policzenia wartości `base` albo `orth` w tym zbiorze.

## Fallback zgodnościowy

`_legacy_find_lemma_context` nie uczestniczy w zwykłym wykonaniu. Routing uruchamia tę funkcję tylko dla przypadku, którego ścieżka indeksowana nie przyjmuje jako bezpiecznie obsługiwanego.

Fallback materializuje potrzebne dane do DataFrame i wykonuje starszy matcher. Diagnostyka zapisuje fakt przełączenia i jego przyczynę.

## Parametry wydajności

`candidate_max_docs`, rozmiar partii strumieniowania i limity cache sterują sposobem pobierania danych. Nie określają liczby zwracanych trafień. Gdy kandydatów jest więcej niż mieści pojedyncza partia, kolejne partie są przetwarzane aż do wyczerpania zbioru kandydatów.
