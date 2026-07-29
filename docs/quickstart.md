# Pierwsze kroki

Ten przewodnik pokazuje najkrótszą drogę od uruchomienia gotowej aplikacji do pierwszego wyszukiwania. Pełny opis funkcji znajduje się w instrukcjach [interfejsu graficznego](gui.md), [wiersza poleceń](cli.md) i [języka CQL](cql.md).

## Uruchamianie aplikacji

- **Wersja CPU — instalator:** uruchom program ze skrótu albo przez `Korpusuj.exe`.
- **Wersja CPU — portable:** rozpakuj archiwum ZIP i uruchom `Korpusuj.exe`.
- **Wersja GPU — instalator:** uruchom program ze skrótu albo przez `Korpusuj.exe`. Podczas instalacji pobierane są dodatkowe komponenty obsługi GPU.

Jeżeli komputer nie ma zgodnej karty NVIDIA, wybierz wersję CPU.

Instrukcje uruchamiania przez Python dotyczą wyłącznie pracy ze źródłami i znajdują się w dalszej części dokumentu.

## Otwarcie istniejącego korpusu

- Wybierz **Nowy projekt**.
- Wskaż plik korpusu w formacie `.parquet`.
- Poczekaj na zakończenie wczytywania i przygotowania projektu.
- W polu wyszukiwania wpisz:

```text
[base="wojna"]
```

- Naciśnij Enter lub wybierz przycisk uruchamiający wyszukiwanie.

Zapytanie znajduje wszystkie wystąpienia, którym podczas anotacji przypisano lemat `wojna`. Wyniki mogą obejmować różne formy fleksyjne. Po wyświetleniu wyników można przeglądać konkordancje, otworzyć szerszy kontekst, sprawdzić statystyki albo przejść do analizy kolokacji.

## Utworzenie nowego korpusu

Jeżeli nie masz gotowego pliku `.parquet`:

- Wybierz **Utwórz korpus**.
- Wybierz mechanizm analizy językowej: Stanza albo spaCy.
- Zdecyduj, czy korpus ma zawierać rozpoznawanie jednostek nazwanych i koreferencję.
- Wybierz pliki źródłowe.
- Pozostaw zaznaczone dokumenty przeznaczone do przetworzenia.
- Wybierz **Przetwórz pliki** i wskaż miejsce zapisania korpusu.

Kreator zapisuje wynik w pliku `.parquet`. Pierwsze użycie wybranego mechanizmu może wymagać pobrania modelu językowego. Podstawowa polska Stanza pobiera około 413 MB. Koreferencja może dodatkowo pobrać adapter około 132 MB i XLM-RoBERTa Large (`model.safetensors`) około 2,24 GB. Dane trafiają do wybranego katalogu modeli wraz z cache'em Hugging Face.

## Log diagnostyczny

Jeżeli aplikacja wyświetli błąd, szczegółowe informacje znajdziesz w:

```text
%LOCALAPPDATA%\Korpusuj\logs\gui\korpusuj.log
```

Katalog można otworzyć, wpisując w pasku adresu Eksploratora:

```text
%LOCALAPPDATA%\Korpusuj\logs\gui
```

## Uruchamianie ze źródeł

Tylko podczas pracy z repozytorium aktywuj środowisko i uruchom aplikację przez Pythona:

```powershell
.\.venv\Scripts\Activate.ps1
python Korpusuj.py
```

### Podstawowy przebieg CLI ze źródeł

CLI nie jest podstawowym sposobem obsługi zainstalowanej wersji GUI. Poniższe polecenia wymagają środowiska źródłowego.

#### 1. Utworzenie korpusu

```powershell
python -m korpusuj.corpus.creator_cli `
  --input .\dokumenty `
  --output .\korpusy\test.parquet
```

Domyślnym mechanizmem jest Stanza. Aby użyć spaCy, dodaj `--model spacy`. Rozpoznawanie jednostek nazwanych i koreferencję można wyłączyć odpowiednio przez `--no-ner` i `--no-coreference`.

#### 2. Utworzenie indeksów

```powershell
python -m korpusuj.index.cli create .\korpusy\test.parquet --progress on --pretty
python -m korpusuj.index.cli status .\korpusy\test.parquet --pretty
```

#### 3. Pierwsze wyszukiwanie

```powershell
python -m korpusuj.search.cli `
  --corpus-path .\korpusy\test.parquet `
  --query '[base="wojna"]' `
  --format text
```

## Co dalej

- Naucz się tworzyć bardziej złożone zapytania w [przewodniku CQL](cql.md).
- Poznaj statystyki, wykresy, kolokacje, sieć semantyczną i modelowanie tematyczne w [instrukcji GUI](gui.md).
- Skorzystaj z automatyzacji opisanej w [instrukcji CLI](cli.md), jeżeli pracujesz ze źródłami lub przygotowaną osobno dystrybucją CLI.
