
# Korpusuj

Korpusuj to aplikacja badawcza do budowy, przeszukiwania i analizy anotowanych korpusów tekstowych. Program umożliwia przetwarzanie plików źródłowych do formatu `.parquet`, wyszukiwanie z użyciem autorskiego języka zapytań typu CQL oraz analizę frekwencyjną, kolokacyjną i diachroniczną.

## Status projektu

Projekt ma charakter badawczy i jest rozwijany jako narzędzie do pracy z korpusami anotowanymi. Repozytorium obejmuje zarówno główną aplikację, jak i kreator korpusów oraz dokumentację języka zapytań.

## Główne możliwości

- budowa korpusów z plików `.txt`, `.docx`, `.pdf`, `.xlsx` oraz archiwów `.zip`
- automatyczna anotacja z użyciem modeli **Stanza** lub **spaCy**
- tokenizacja, lematyzacja, tagowanie morfosyntaktyczne, rozpoznawanie jednostek nazwanych, analiza zależnościowa oraz obsługa koreferencji
- obsługa plików PDF z ekstrakcją tekstu, a w razie potrzeby także z OCR
- mapowanie kolumn w plikach Excel oraz import osobnego pliku metadanych
- filtrowanie wyników z użyciem metadanych podstawowych i własnych kolumn użytkownika
- wyszukiwanie z użyciem rozbudowanego języka zapytań
- konstruktor zapytań dostępny z poziomu interfejsu graficznego
- analiza konkordancyjna, frekwencyjna, kolokacyjna i diachroniczna
- wizualizacja drzewa zależności, podświetlanie NER i klastrów koreferencyjnych
- eksport wyników do Excela
- zapisywanie wybranych fragmentów jako fiszek

## Dane wejściowe i format korpusu

Kreator korpusów zapisuje wynik przetwarzania do pliku `.parquet`. W pliku tym przechowywane są:

- tekst źródłowy
- metadane
- tokeny
- lematy
- tagi morfosyntaktyczne
- relacje zależnościowe
- identyfikatory zdań
- jednostki nazwane
- informacje o koreferencji

Pliki Excel mogą być używane zarówno jako źródła tekstów, jak i jako osobny plik metadanych. Kreator pozwala mapować kolumny na pola standardowe:

- `Nazwa pliku`
- `Tytuł`
- `Treść`
- `Data publikacji`
- `Autor`

Dodatkowe kolumny są zachowywane i mogą być później używane do filtrowania wyników.

## Modele językowe

Program obsługuje dwa tryby anotacji:

- **Stanza** — pełny pipeline z tokenizacją, tagowaniem, lematyzacją, NER, analizą zależnościową i koreferencją
- **spaCy** — alternatywny pipeline z lokalnie pobieranym modelem języka polskiego

Modele są pobierane przy pierwszym użyciu i przechowywane lokalnie w katalogu `models/`.

## Wyszukiwanie

Silnik zapytań obsługuje:

- dopasowanie po `orth`, `base`, `pos`, `upos`, `deprel`, `ner`
- warunki zagnieżdżone dla relacji `head` i `dependent`
- atrybuty cech morfosyntaktycznych, np. `case`, `number`, `gender`, `person`, `aspect`
- koreferencję (`coref`, `coref(H)`, `coref(P)`)
- okna kontekstowe (`window_base`, `window_orth`)
- ograniczenie wyszukiwania do jednego zdania przy użyciu operatora `<s>`
- filtrowanie po frekwencji (`<frequency_base>`, `<frequency_orth>`)
- filtrowanie po metadanych (`<autor>`, `<tytuł>`, `<data>`, `<metadane:...>`)
- wyrażenia regularne
- uproszczony tryb szybkiego wyszukiwania

Aplikacja waliduje składnię zapytania przed uruchomieniem wyszukiwania i przechowuje historię ostatnich zapytań.

## Analiza wyników

### Konkordancje

Wyniki wyszukiwania są prezentowane jako konkordancje z lewym i prawym kontekstem. Po kliknięciu w wiersz można wyświetlić rozszerzony kontekst w pełnym tekście.

Wyniki można sortować według:

- daty publikacji
- autora
- tytułu
- porządku alfabetycznego
- lewego kontekstu
- prawego kontekstu

### Statystyki

Program generuje zestawienia frekwencyjne dla:

- form podstawowych
- form ortograficznych
- rozkładu w czasie
- kolokacji

W tabelach wykorzystywane są m.in. następujące miary:

- PMW
- TF-IDF
- document frequency
- Z-score

### Kolokacje

Moduł kolokacji obsługuje analizę:

- liniową
- składniową

Dostępne są m.in. następujące ustawienia:

- rozmiar okna
- ograniczenie do jednego zdania
- filtry POS, UPOS i deprel
- progi minimalnej frekwencji
- progi minimalnego rozproszenia

Dostępne miary asocjacji:

- Log-Likelihood
- MI Score
- T-score
- Log-Dice

### Trendy

Zakładka trendów pozwala generować wykresy dla różnych interwałów czasowych:

- dzień
- miesiąc
- rok

Dostępne typy wartości:

- liczba wystąpień
- częstość względna
- TF-IDF
- Z-score

Użytkownik może grupować etykiety i zmieniać nazwy serii na wykresie.

## Interfejs i narzędzia dodatkowe

Aplikacja udostępnia również:

- konstruktor zapytań
- wizualizację drzewa zależności
- podświetlanie jednostek nazwanych
- podświetlanie klastrów koreferencyjnych
- fiszki z zapisywaniem zaznaczonych fragmentów tekstu
- historię zapytań
- trwałe ustawienia aplikacji zapisywane w `config.json`

## Instalacja

### Uruchamianie ze źródeł

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/Rafal-P-Mazur/Korpusuj.git
   cd Korpusuj
   ``
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ``
3. Uruchom aplikację:
   ```bash
   python Korpusuj_beta.py
   ``

### Windows (.exe)
1. Pobierz najnowszą wersję z [Releases](https://github.com/Rafal-P-Mazur/Korpusuj/releases).
2. Rozpakuj archiwum ZIP.
3. Uruchom `Korpusuj.exe` klikając dwukrotnie.


## Uwagi dotyczące wydajności

- przy pierwszym użyciu kreatora aplikacja może pobrać wymagane modele językowe
- przy dużych korpusach czas anotacji może być znaczny
- w środowisku Python można korzystać z konfiguracji Torch zgodnej z GPU
- wersja exe działa wyłącznie w trybie CPU

## Tworzenie korpusu

Aby utworzyć nowy korpus:

1. uruchom aplikację
2. wybierz `Plik -> Utwórz korpus`
3. wskaż pliki źródłowe
4. wybierz model anotacji
5. opcjonalnie dodaj osobny plik metadanych
6. uruchom przetwarzanie
7. zapisz wynik do pliku `.parquet`

Kreator obsługuje:

- mapowanie kolumn w arkuszach Excel
- dodatkowe metadane
- automatyczne rozpakowywanie archiwów `.zip`
- checkpointy i wznawianie przerwanego przetwarzania

## Wczytanie korpusu

Aby rozpocząć pracę z gotowym korpusem:

1. wybierz `Plik -> Nowy projekt`
2. wskaż plik `.parquet`
3. wybierz aktywny korpus z listy w głównym oknie

Aplikacja umożliwia pracę z wieloma korpusami.

## Eksport

Wyniki można eksportować do pliku Excel. Eksport może obejmować m.in.:

- wyniki wyszukiwania
- frekwencję lematów
- frekwencję form ortograficznych
- rozkład w czasie
- kolokacje

## Struktura plików
* `Korpusuj.exe` – Plik wykonywalny aplikacji.
* `models/` – Katalog z lokalnymi modelami NLP (Stanza/SpaCy).
* `config.json` — zapis preferencji użytkownika
* `korpusuj.log` – Plik dziennika błędów i operacji.
* `temp/` – Pliki tymczasowe generowane podczas sesji.

## Ograniczenia

Aplikacja jest rozwijana jako narzędzie badawcze. Wydajność anotacji i wyszukiwania zależy od wielkości danych, wybranego modelu oraz konfiguracji środowiska. Przy dużych korpusach czas działania może być znaczny.

## Licencja

Projekt jest udostępniany na licencji MIT.
``
