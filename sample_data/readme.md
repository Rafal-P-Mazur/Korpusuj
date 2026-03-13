
# Korpus demonstracyjny: interpelacje sejmowe

Ten katalog zawiera przykładowy korpus emonstracyjny w formacie `.parquet`, przeznaczony do szybkiego przetestowania aplikacji bez konieczności samodzielnego budowania korpusu.

## Źródło danych

Korpus został wygenerowany z publicznego API Sejmu RP na podstawie interpelacji sejmowych.

Wykorzystany zasób API udostępnia m.in.:
- numer interpelacji,
- tytuł,
- autorów,
- adresata,
- datę otrzymania i datę wysłania,
- linki do treści dokumentu.

## Cel

Korpus demonstracyjny został przygotowany do szybkiego testowania następujących funkcji aplikacji:
- importu plików tekstowych,
- dołączania metadanych z pliku Excel,
- filtrowania wyników po dacie, autorze i adresacie,
- wizualizacji trendów czasowych.

## Struktura metadanych

Korpus zawiera następujące metadane
- `Tytuł`
- `Data publikacji`
- `Autor`
- `Numer interpelacji`
- `Adresat`
- `Źródło`
- `Kadencja`
- `URL treści`

## Jak użyć w Korpusuj

1. uruchom aplikację,
2. wybierz `Plik -> Nowy projekt`,
3. wskaż plik `sample_data/demo_korpus.parquet`,
4. wybierz aktywny korpus z listy rozwijanej w głównym oknie.

## Sposób wygenerowania

Korpus został wygenerowany skryptem:

`scripts/download_sejm_interpellations_txt_random.py`

Parametry generacji:
- kadencja: 10
- rok: 2024
- liczba dokumentów na miesiąc: 5
- tryb doboru: losowy
- adresat: minister zdrowia


## Uwagi

Jest to zestaw demonstracyjny o małej skali. Jego celem nie jest reprezentatywność, lecz umożliwienie szybkiego przetestowania funkcji aplikacji.
