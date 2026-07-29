# Korpusuj

Korpusuj to aplikacja do tworzenia, przeszukiwania i analizowania anotowanych korpusów języka polskiego. Udostępnia interfejs graficzny oraz narzędzia wiersza poleceń, dzięki czemu może służyć zarówno do interaktywnej pracy z wynikami, jak i do automatyzacji większych zadań badawczych.

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

## Instalacja

Program Korpusuj jest dostępny dla systemu Windows w trzech wariantach.

### Wersja CPU — instalator

Wersja CPU wykonuje obliczenia związane z modelami językowymi za pomocą procesora głównego komputera. Nie wymaga karty NVIDIA i jest odpowiednia dla większości komputerów.

Instalator kopiuje program do wybranego katalogu, tworzy skróty i pozwala wskazać osobny katalog do przechowywania modeli językowych.

### Wersja CPU — portable

Wersja portable ma te same możliwości obliczeniowe co instalacyjna wersja CPU, ale nie wymaga instalowania programu. Po rozpakowaniu archiwum należy uruchomić `Korpusuj.exe`.

Modele językowe i ich cache są przechowywane w katalogu `models` obok programu. Cały katalog wersji portable można przenieść na inny dysk lub komputer.

### Wersja GPU — instalator

Wersja GPU może wykorzystywać zgodną kartę NVIDIA do przyspieszania obsługiwanych modeli językowych i obliczeń opartych na PyTorch. Jest przeznaczona dla komputerów ze zgodną kartą NVIDIA oraz odpowiednio nowym sterownikiem.

Instalator pobiera podczas instalacji około 2,42 GiB oficjalnych komponentów PyTorch dla CUDA 12.6. Wymaga to połączenia z Internetem.

Jeżeli komputer nie ma zgodnej karty NVIDIA, należy wybrać wersję CPU.

Więcej informacji o wymaganiach sprzętowych, instalacji oraz przechowywaniu i pobieraniu modeli zawiera [instrukcja instalacji programu Korpusuj](docs/installation.md).

## Pierwsze wyszukiwanie

Po uruchomieniu aplikacji:

1. wybierz **Nowy projekt**;
2. wskaż korpus zapisany w pliku `.parquet`;
3. poczekaj na zakończenie wczytywania projektu;
4. wpisz w polu wyszukiwania:

```cql
[base="wojna"]
```

5. naciśnij `Enter` albo wybierz przycisk uruchamiający wyszukiwanie.

Zapytanie znajduje wystąpienia, którym podczas anotacji przypisano lemat `wojna`, dlatego wyniki mogą obejmować różne formy fleksyjne tego słowa.

Jeżeli nie masz jeszcze korpusu, wybierz **Utwórz korpus** i skorzystaj z wbudowanego kreatora. Skrócony opis całego przebiegu znajduje się w dokumencie [Pierwsze kroki](docs/quickstart.md).

## Interfejs wiersza poleceń

Korpusuj udostępnia trzy główne polecenia:

```powershell
python -m korpusuj.corpus.creator_cli --help
python -m korpusuj.index.cli --help
python -m korpusuj.search.cli --help
```

Służą one odpowiednio do tworzenia korpusów, zarządzania indeksami oraz wykonywania zapytań, analiz i eksportu wyników. Pełny opis argumentów i przykładowych przebiegów znajduje się w [instrukcji CLI](docs/cli.md).

## Pliki korpusu i indeksów

Treść korpusu oraz jego anotacje są zapisywane w pliku `.parquet`. Dla korpusu mogą zostać utworzone dwa dodatkowe pliki:

```text
korpus.parquet
korpus.search
korpus.dep_cache
```

- `.parquet` zawiera dokumenty, tokeny, anotacje i metadane;
- `.search` zawiera indeks przyspieszający wyszukiwanie;
- `.dep_cache` zawiera dane potrzebne do sprawnego wykonywania zapytań składniowych.

Pliki `.search` i `.dep_cache` można ponownie utworzyć na podstawie pliku `.parquet` za pomocą narzędzia do zarządzania indeksami.

## Dokumentacja

- [Spis dokumentacji](docs/index.md)
- [Instalacja](docs/installation.md)
- [Pierwsze kroki](docs/quickstart.md)
- [Instrukcja interfejsu graficznego](docs/gui.md)
- [Instrukcja interfejsu wiersza poleceń](docs/cli.md)
- [Przewodnik po języku zapytań CQL](docs/cql.md)
- [Architektura aplikacji](docs/architecture/overview.md)

## Uruchamianie ze źródeł

Projekt wymaga systemu Windows i Pythona 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Korpusuj.py
```

Wersja uruchamiana ze źródeł przechowuje modele i cache w katalogu:

```text
<root projektu>\models
```

## Rozwój i testowanie

Informacje przeznaczone dla osób rozwijających aplikację znajdują się w [dokumentacji architektury](docs/architecture/overview.md).

Zestaw testów można uruchomić poleceniem:

```powershell
python -m pytest -q -p no:cacheprovider tests
```
