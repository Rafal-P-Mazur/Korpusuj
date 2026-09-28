# Korekta lematyzacji gotowego korpusu

Korpusuj może przeanalizować lematy aktywnego korpusu i utworzyć nowy plik Parquet z bezpiecznie zatwierdzonymi korektami. Funkcja jest przeznaczona do naprawy systematycznych błędów powstałych podczas anotacji przez Stanza albo spaCy. Nie wykonuje ponownej tokenizacji i nie zmienia tekstu, form `tokens`, tagów, zależności, NER, koreferencji ani metadanych dokumentów.

## Uruchomienie w GUI

1. Otwórz pierwotny plik `.parquet` jako aktywny korpus.
2. Wybierz **Plik → Popraw lematyzację aktywnego korpusu**.
3. Wybierz **Analizuj lematyzację**.
4. Przejrzyj podsumowanie oraz reguły automatyczne.
5. Wybierz **Zastosuj korekty**, aby zapisać oddzielny plik wynikowy.
6. Otwórz wynikowy korpus i używaj odpowiadającego mu nowego `.search` oraz `.dep_cache`.

Plik źródłowy nie jest nadpisywany. Katalog roboczy `<nazwa>.lemma_repair` powstaje obok korpusu i zawiera workspace SQLite, raporty oraz pliki decyzji.

## Jak wybierane są korekty

Podstawowa ścieżka wykrywa szczątkowe lub rozszczepione paradygmaty i porównuje obserwowane formy z analizami Morfeusza 2 oraz danymi fleksyjnymi SGJP. Tryb `common-core-plus` uzupełnia ją bezpośrednią kontrolą każdej wystarczająco częstej trójki:

```text
forma tekstowa + obecny lemat + UPOS
```

Automatyczna reguła może powstać tylko wtedy, gdy obecny lemat nie jest potwierdzony przez SGJP, cel jest jednoznaczny, a dostępne cechy morfosyntaktyczne zapewniają wymagany poziom zgodności. Techniczne identyfikatory haseł SGJP, np. `psycholog:Sm1`, są normalizowane do hasła bazowego. Alternatywne wartości tagów SGJP, np. `gen.acc` i `nom.voc`, są traktowane jako zbiory dopuszczalnych wartości.

Mechanizm nie zawiera słownika ręcznie wpisanych poprawek. Przykłady takie jak błędny lemat `psycholoeg` są rozstrzygane na podstawie rzeczywistej formy, znacznika i analiz SGJP.

## Ochrona przed błędnymi zmianami

Korekta jest blokowana między innymi wtedy, gdy:

- SGJP potwierdza obecny lemat w dowolnej części mowy;
- forma jest homograficzna i prowadzi do kilku lematów, np. `mam` może odpowiadać formom leksemów `mieć`, `mama` i `mamić`;
- po ograniczeniu do właściwego UPOS pozostaje kilka celów;
- nie ma żadnego wiarygodnego dopasowania morfologicznego;
- forma jest artefaktem tokenizacji, adresem, domeną albo zawiera niedozwolony układ znaków;
- przypadek występuje zbyt rzadko lub w zbyt małej liczbie dokumentów.

Morfeusz 2 jest analizatorem morfologicznym, a nie kontekstowym dezambiguatorem. Dlatego sama analiza formy nie rozstrzyga homografii. Korpusuj wykorzystuje informacje z korpusu, lecz zachowuje konserwatywne blokady i nie zmienia automatycznie form wieloznacznych.

## Artefakty analizy

Najważniejsze pliki w katalogu roboczym:

- `*.audit.md` i `*.audit.json` — audyt kandydatów;
- `*.auto.json` — reguły zaakceptowane automatycznie;
- `*.review.json` — reguły wymagające oceny;
- `*.rejected.json` — reguły odrzucone;
- `*.direct_sgjp.json` — reguły dodane przez bezpośrednią ścieżkę SGJP;
- `*.preview.json` i `*.status.json` — walidacja dopasowania reguł;
- `*.apply.json` — wynik zastosowania.

Raport bazowy D3 jest tworzony przed późniejszym uzupełnieniem puli przez direct-SGJP. Ostateczną liczbę zaakceptowanych i dopasowanych reguł należy odczytywać z podglądu lub pliku statusu.

## Walidacja wyniku

Przed zastąpieniem używanego korpusu należy sprawdzić:

- identyczną liczbę dokumentów i tokenów;
- identyczny schemat kolumn;
- identyczność wszystkich kolumn poza `lemmas`;
- zgodność liczby zmienionych tokenów z raportem;
- świeżość nowego `.search` i `.dep_cache`.

Fizyczny rozmiar wynikowego Parquetu może być mniejszy lub większy od źródła, ponieważ plik jest ponownie kodowany i kompresowany. Sama różnica rozmiaru nie świadczy o utracie danych.

## Zależność i atrybucja

Funkcja korzysta z **Morfeusza 2** i danych fleksyjnych **SGJP**. Pakiet `morfeusz2` musi być dostępny zarówno przy uruchamianiu ze źródeł, jak i w artefaktach binarnych CPU i GPU. Pełna nota licencyjna znajduje się w głównym pliku `THIRD_PARTY_NOTICES.md` dystrybucji.

W publikacjach opisujących wyniki uzyskane przy użyciu Morfeusza 2 zalecane jest cytowanie pracy:

> Witold Kieraś, Marcin Woliński. *Morfeusz 2 – analizator i generator fleksyjny dla języka polskiego*. Język Polski, XCVII(1), 75–83, 2017.
