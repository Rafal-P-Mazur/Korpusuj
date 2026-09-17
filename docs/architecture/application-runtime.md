# Uruchamianie i stan aplikacji

## Uruchomienie GUI

`Korpusuj.py` uruchamia `engine.py`. Podczas inicjalizacji `engine.py`:

- ładuje konfigurację użytkownika;
- wyznacza katalogi modeli, pamięci podręcznych i logów;
- konfiguruje logowanie;
- tworzy stan wyszukiwania i widoki;
- przekazuje wyszukiwarce funkcje dostępu do dependency runtime.

## Stan otwartego projektu

`engine.py` przechowuje między innymi:

- mapowanie nazwy korpusu na plik Parquet;
- załadowany `LazyCorpus`;
- leniwe indeksy terminów `base` i `orth`;
- bieżący `SearchState`;
- wynik ostatniego zapytania;
- numer strony i liczbę wierszy na stronie;
- cache relacji zależnościowych i obiekty `.dep_cache`.

Nazwy `dataframes` i `inverted_indexes` pochodzą ze starszej architektury. W bieżącym kodzie `dataframes` może zawierać `LazyCorpus`, a pola `base` i `orth` w `inverted_indexes` mogą być instancjami `LazyTermIndex`.

## Konfiguracja runtime wyszukiwarki

Przy imporcie `engine.py` przekazuje do modułów wyszukiwania:

- klasy `SearchCursor` i `SearchIndex` używane przez executor;
- funkcje pobierające mapy zależnościowe;
- słowniki cache dependency;
- limity preładowania kandydatów;
- bieżący rozmiar pełnego kontekstu.

Moduły wyszukiwania są wydzielone do pakietu `korpusuj.search`, ale część ich zależności pochodzi nadal ze stanu tworzonego w `engine.py`.

## Zadania wykonywane poza wątkiem GUI

Wyszukiwanie, tworzenie korpusu, przygotowanie dependency i część analiz mogą trwać długo. Są uruchamiane w wątku roboczym albo osobnym procesie. Informacje o postępie i wyniku wracają do interfejsu przez funkcje zaplanowane za pomocą `app.after`.

## Dodatkowe tryby procesu

Ten sam program obsługuje argumenty uruchamiające osobne zadania, między innymi webview dla raportów HTML oraz procesy związane z analizą semantyczną. W wersji PyInstaller wykonuje je ten sam plik EXE z innym argumentem. Przy uruchomieniu ze źródeł używany jest interpreter Pythona.

## Zamykanie projektu

Zamknięcie lub zastąpienie projektu wymaga zamknięcia `LazyCorpus`, połączeń `SearchIndex` i obiektów `DependencyMapDiskCache` związanych z tym korpusem. Wątki przygotowujące dependency otrzymują sygnał zatrzymania, a wpisy cache tego korpusu są usuwane.
