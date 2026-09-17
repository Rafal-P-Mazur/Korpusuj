# Wyniki, liczba trafień i eksport

## Wynik leniwy

`SearchCursor` nie tworzy od razu listy wszystkich trafień. Przetwarza kandydatów w kolejności potrzebnej do pobrania żądanego zakresu. GUI pobiera stronę przez `get_range(start, end)`.

Dzięki temu wyświetlenie pierwszej strony nie wymaga utworzenia wszystkich wierszy konkordancji.

## Dokładna liczba trafień

Kursor przechowuje znalezione trafienia w `_hits`. Gdy iterator kandydatów dojdzie do końca, ustawia `_count_cache` na liczbę elementów w `_hits`. Ta wartość jest dokładną liczbą trafień po sprawdzeniu wszystkich warunków zapytania.

Przed wyczerpaniem iteratora `_count_cache` ma wartość `None`. W tym stanie `len(cursor)` zwraca większą z dwóch liczb:

- liczby trafień znalezionych dotychczas;
- estymacji zwróconej przez `count_hits_estimate()`.

Dlatego `len(cursor)` nie jest wtedy dokładnym `total_hits`.

`count_final_searchcursor_hits` zwraca dokładną liczbę w jednej z dwóch sytuacji:

1. kursor ma już `_count_cache`;
2. kursor oznacza swoją estymację jako dokładną.

Jeżeli żaden warunek nie jest spełniony, funkcja prosi kursor o kolejne zakresy aż do wyczerpania iteratora. Wyczerpanie ustawia `_count_cache`, a funkcja zwraca tę wartość.

## Ostatnia strona w GUI

GUI włącza przejście do ostatniej strony dopiero wtedy, gdy `_count_cache` istnieje albo kursor potwierdza, że estymacja jest dokładna. Wcześniej interfejs może wyświetlać kolejne strony, ale nie zna pewnego numeru strony ostatniej.

## Krótki i pełny kontekst

Wiersz konkordancji zawiera krótki lewy i prawy kontekst. Pełny tekst dokumentu może być zapisany jako leniwe odwołanie zawierające `doc_id` i pozycję trafienia.

Po wybraniu wiersza GUI rozwiązuje odwołanie i odczytuje dokument z `LazyCorpus`. Eksport rozwiązuje te odwołania dla eksportowanych wierszy przed utworzeniem tabeli.

## Eksport

`korpusuj.export.excel` zapisuje konkordancje, statystyki, kolokacje i profile kolokacyjne. Przed zapisem usuwa znaki niedozwolone w XLSX i zabezpiecza tekst rozpoczynający się od znaku interpretowanego jako formuła.

`korpusuj.export.subcorpus` wybiera pełne dokumenty i zapisuje nowy Parquet. Sidecary źródłowego korpusu nie są kopiowane; dla podkorpusu trzeba zbudować nowy `.search` i `.dep_cache`.
