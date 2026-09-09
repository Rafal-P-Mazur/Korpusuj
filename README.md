# Korpusuj

Korpusuj to aplikacja do tworzenia, przeszukiwania i analizowania anotowanych korpusów języka polskiego. Udostępnia interfejs graficzny oraz narzędzia wiersza poleceń, dzięki czemu może służyć do interaktywnej pracy z wynikami i automatyzacji zadań badawczych.

## Najważniejsze możliwości

Korpusuj umożliwia:

- tworzenie korpusów z plików TXT, DOCX, PDF i XLSX oraz z dokumentów zgromadzonych w archiwach ZIP;
- dołączanie metadanych, takich jak autor, tytuł, data publikacji, gatunek i własne kategorie badawcze;
- analizę językową za pomocą Stanza albo spaCy;
- opcjonalne rozpoznawanie jednostek nazwanych (NER) i koreferencji;
- wyszukiwanie w języku CQL według form tekstowych, lematów, części mowy, cech morfologicznych, relacji składniowych, jednostek nazwanych, koreferencji i metadanych;
- przeglądanie konkordancji i szerszego kontekstu trafień;
- generowanie statystyk, wykresów, kolokacji i profili kolokacyjnych;
- tworzenie sieci semantycznych i raportów semantycznych;
- modelowanie tematyczne za pomocą BERTopic;
- eksport wyników i tworzenie podkorpusów;
- wykonywanie zadań z interfejsu graficznego albo z wiersza poleceń.

## Wersje programu

### Windows CPU: instalator

Wersja CPU wykonuje obliczenia związane z modelami językowymi za pomocą procesora głównego. Nie wymaga karty NVIDIA i jest odpowiednia dla większości komputerów z systemem Windows.

Instalator kopiuje program do wybranego katalogu, tworzy skróty i pozwala wskazać osobny katalog do przechowywania modeli językowych.

### Windows CPU: wersja portable

Wersja portable ma te same możliwości obliczeniowe co instalacyjna wersja CPU, ale nie wymaga instalowania programu. Po pobraniu należy rozpakować całe archiwum ZIP i uruchomić `Korpusuj.exe`.

Modele językowe i ich pamięci podręczne są przechowywane w katalogu `models` obok programu. Cały rozpakowany katalog można przenieść na inny dysk lub komputer.

### Windows GPU: instalator

Wersja GPU może wykorzystywać zgodną kartę NVIDIA do przyspieszania obsługiwanych modeli językowych i obliczeń opartych na PyTorch. Wymaga odpowiednio nowego sterownika NVIDIA.

Instalator pobiera podczas instalacji około 2,42 GiB oficjalnych komponentów PyTorch dla CUDA 12.6, dlatego wymaga połączenia z Internetem. Jeżeli komputer nie ma zgodnej karty NVIDIA, należy wybrać wersję CPU.

### macOS ARM64

Korpusuj jest dostępny również jako gotowy pakiet aplikacji dla komputerów Mac z procesorami Apple Silicon. Obsługiwane modele mogą korzystać z akceleracji MPS. Aplikację można także uruchomić ze źródeł przy użyciu Pythona 3.11.

Szczegółowe informacje zawiera [instrukcja instalacji](docs/installation.md).

## Pierwsze wyszukiwanie

Po uruchomieniu aplikacji:

1. Wybierz **Nowy projekt**.
2. Wskaż korpus zapisany w pliku `.parquet`.
3. Poczekaj na zakończenie wczytywania projektu.
4. Wpisz w polu wyszukiwania:

```cql
[base="wojna"]
```

5. Naciśnij Enter albo wybierz przycisk uruchamiający wyszukiwanie.

Zapytanie znajduje wystąpienia, którym podczas anotacji przypisano lemat `wojna`, dlatego wyniki mogą obejmować różne formy fleksyjne tego słowa.

Jeżeli nie masz jeszcze korpusu, wybierz **Utwórz korpus** i skorzystaj z wbudowanego kreatora. Skrócony opis całego przebiegu znajduje się w dokumencie [Pierwsze kroki](docs/quickstart.md).

## Interfejs wiersza poleceń

Korpusuj udostępnia trzy główne moduły CLI:

```text
python -m korpusuj.corpus.creator_cli --help
python -m korpusuj.index.cli --help
python -m korpusuj.search.cli --help
```

Służą one do tworzenia korpusów, zarządzania indeksami oraz wykonywania zapytań, analiz i eksportu wyników. Pełny opis znajduje się w [instrukcji CLI](docs/cli.md).

## Pliki korpusu i indeksów

Treść korpusu oraz jego anotacje są zapisywane w pliku `.parquet`. Dla korpusu mogą zostać utworzone dwa dodatkowe pliki:

```text
korpus.parquet
korpus.search
korpus.dep_cache
```

- `.parquet` zawiera dokumenty, tokeny, anotacje i metadane;
- `.search` zawiera indeks przyspieszający wyszukiwanie;
- `.dep_cache` zawiera dane potrzebne do wykonywania zapytań składniowych.

## Uruchamianie ze źródeł

Projekt wymaga Pythona 3.11.

### Windows CPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-cpu-py311.txt
python Korpusuj.py
```

### Windows GPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-gpu-py311.txt
python Korpusuj.py
```

Sprawdzenie dostępności CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Brak dostępnego GPU')"
```

### macOS ARM64

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-macos-arm64-py311.txt
python Korpusuj.py
```

Sprawdzenie dostępności MPS:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

## Dokumentacja

- [Spis dokumentacji](docs/index.md)
- [Instalacja](docs/installation.md)
- [Pierwsze kroki](docs/quickstart.md)
- [Instrukcja interfejsu graficznego](docs/gui.md)
- [Instrukcja interfejsu wiersza poleceń](docs/cli.md)
- [Przewodnik po języku zapytań CQL](docs/cql.md)
- [Architektura aplikacji](docs/architecture/overview.md)

## Rozwój i testowanie

Informacje dla osób rozwijających aplikację znajdują się w [dokumentacji architektury](docs/architecture/overview.md).

Zestaw testów można uruchomić poleceniem:

```text
python -m pytest -q -p no:cacheprovider tests
```
