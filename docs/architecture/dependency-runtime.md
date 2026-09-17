# Dostęp do relacji zależnościowych

## Mapa dokumentu

Dla każdego tokenu mapa przechowuje indeks jego bezpośredniego nadrzędnika. Token będący korzeniem zdania ma wartość ujemną. Na podstawie tej tablicy `LazyChildrenLookup` ustala podrzędniki danego tokenu.

## Źródła mapy

Podczas zapytania mapa dokumentu jest pobierana w następującej kolejności:

1. z cache w pamięci RAM;
2. z `.dep_cache`;
3. przez obliczenie z `sentence_ids`, `word_ids` i `head_ids` w Parquet.

Obliczona mapa może zostać zapisana do cache zgodnie z aktywną polityką pamięci.

## `DependencyRuntimeState`

Obiekt przechowuje:

- mapy znajdujące się w RAM;
- otwarte połączenia do `.dep_cache`;
- wątki przygotowujące mapy;
- sygnały zatrzymania;
- limity pamięci i liczby preładowywanych dokumentów.

`engine.py` tworzy kolekcje stanu i przekazuje runtime'owi funkcje dostępu do ścieżek korpusów oraz ustawień użytkownika.

## Tryby pamięci

W trybie bez cache RAM mapa jest odczytywana z dysku lub tworzona dla bieżącego dokumentu. W trybie cache aplikacja zachowuje ograniczoną liczbę map w pamięci i usuwa najstarsze wpisy po przekroczeniu budżetu.

Preładowanie map kandydatów skraca czas zapytania składniowego, ale nie wybiera kompletnego zbioru wyników. Kursor nadal przetwarza wszystkie partie kandydatów.

## Zamknięcie korpusu

Po zamknięciu korpusu runtime zatrzymuje jego worker, zamyka połączenie do `.dep_cache` i usuwa mapy tego korpusu z RAM. Dzięki temu kolejny korpus nie korzysta z map przypisanych do wcześniejszej ścieżki Parquet.
