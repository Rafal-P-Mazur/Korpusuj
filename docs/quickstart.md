# Pierwsze kroki

Ten przewodnik pokazuje najkrótszą drogę od uruchomienia aplikacji do pierwszego wyszukiwania. Pełny opis funkcji znajduje się w instrukcjach [interfejsu graficznego](gui.md), [wiersza poleceń](cli.md) i [języka CQL](cql.md).

## Uruchamianie aplikacji

- **Windows CPU, instalator:** uruchom program ze skrótu albo przez `Korpusuj.exe`.
- **Windows CPU, wersja portable:** rozpakuj całe archiwum ZIP i uruchom `Korpusuj.exe`.
- **Windows GPU, instalator:** uruchom program ze skrótu albo przez `Korpusuj.exe`. Instalacja wymaga pobrania dodatkowych komponentów obsługi GPU.
- **macOS ARM64:** uruchom aplikację ze źródeł w środowisku Python 3.11 albo użyj gotowego pakietu, jeżeli został dołączony do wydania.

Jeżeli komputer z systemem Windows nie ma zgodnej karty NVIDIA, wybierz wersję CPU.

## Otwarcie istniejącego korpusu

1. Wybierz **Nowy projekt**.
2. Wskaż plik korpusu w formacie `.parquet`.
3. Poczekaj na zakończenie wczytywania i przygotowania projektu.
4. W polu wyszukiwania wpisz:

```cql
[base="wojna"]
```

5. Naciśnij Enter lub wybierz przycisk uruchamiający wyszukiwanie.

Zapytanie znajduje wszystkie wystąpienia, którym podczas anotacji przypisano lemat `wojna`. Wyniki mogą obejmować różne formy fleksyjne. Po wyświetleniu wyników można przeglądać konkordancje, otworzyć szerszy kontekst, sprawdzić statystyki albo przejść do analizy kolokacji.

## Utworzenie nowego korpusu

Jeżeli nie masz gotowego pliku `.parquet`:

1. Wybierz **Utwórz korpus**.
2. Wybierz Stanza albo spaCy.
3. Zdecyduj, czy korpus ma zawierać rozpoznawanie jednostek nazwanych i koreferencję.
4. Wybierz pliki źródłowe.
5. Pozostaw zaznaczone dokumenty przeznaczone do przetworzenia.
6. Wybierz **Przetwórz pliki** i wskaż miejsce zapisania korpusu.

Kreator zapisuje wynik w pliku `.parquet`. Pierwsze użycie wybranych funkcji może wymagać pobrania modeli językowych. Podstawowy polski pakiet Stanza zajmuje około 413 MB. Koreferencja może dodatkowo pobrać adapter o wielkości około 132 MB i model XLM-RoBERTa Large o wielkości około 2,24 GB.

## Log diagnostyczny w systemie Windows

Szczegółowy log interfejsu graficznego znajduje się w:

```text
%LOCALAPPDATA%\Korpusuj\logs\gui\korpusuj.log
```

Katalog można otworzyć, wpisując w pasku adresu Eksploratora:

```text
%LOCALAPPDATA%\Korpusuj\logs\gui
```

## Uruchamianie ze źródeł

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
python Korpusuj.py
```

### macOS ARM64

```bash
source .venv/bin/activate
python Korpusuj.py
```

Sposób utworzenia środowiska CPU, GPU lub macOS opisuje [instrukcja instalacji](installation.md).

## Podstawowy przebieg CLI

Poniższe polecenia wymagają poprawnie skonfigurowanego środowiska źródłowego.

### 1. Utworzenie korpusu

```text
python -m korpusuj.corpus.creator_cli --input dokumenty --output korpusy/test.parquet
```

Opcje tworzenia korpusu można wyświetlić poleceniem:

```text
python -m korpusuj.corpus.creator_cli --help
```

### 2. Utworzenie indeksów

```text
python -m korpusuj.index.cli create korpusy/test.parquet --progress on --pretty
python -m korpusuj.index.cli status korpusy/test.parquet --pretty
```

### 3. Pierwsze wyszukiwanie

```text
python -m korpusuj.search.cli --corpus-path korpusy/test.parquet --query "[base='wojna']" --format text
```

## Co dalej

- Naucz się tworzyć bardziej złożone zapytania w [przewodniku CQL](cql.md).
- Poznaj statystyki, wykresy, kolokacje, sieć semantyczną i modelowanie tematyczne w [instrukcji GUI](gui.md).
- Skorzystaj z automatyzacji opisanej w [instrukcji CLI](cli.md).
