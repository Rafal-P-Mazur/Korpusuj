# Korzystanie z interfejsu wiersza poleceń

Niniejszy przewodnik opisuje obsługę aplikacji Korpusuj z poziomu terminala. Obejmuje tworzenie korpusu, zarządzanie indeksami, wyszukiwanie, eksport wyników, statystyki, kolokacje, profile kolokacyjne i tworzenie podkorpusów.

> **Uwaga:** Składnia języka CQL została opisana w osobnym przewodniku. Instrukcja interfejsu graficznego znajduje się w pliku `gui.md`.

---

## 1. Rozpoczęcie pracy

Polecenia należy wykonywać w głównym katalogu projektu, w środowisku Pythona zawierającym zależności Korpusuj. Dostępne są trzy publiczne moduły CLI:

```powershell
python -m korpusuj.corpus.creator_cli --help
python -m korpusuj.index.cli --help
python -m korpusuj.search.cli --help
```

Pierwszy moduł tworzy korpus, drugi zarządza zestawem indeksów, a trzeci wykonuje wyszukiwanie i analizy.

### Ścieżki zawierające spacje

Ścieżkę zawierającą spacje należy ująć w cudzysłów:

```powershell
python -m korpusuj.search.cli `
    --corpus-path "D:\Korpusy badawcze\prasa.parquet" `
    --query '[base="wojna"]'
```

W PowerShell znak `` ` `` umożliwia kontynuowanie polecenia w następnym wierszu. Polecenie może być również zapisane w jednym wierszu.

### `stdout`, `stderr` i kod zakończenia

Dane wynikowe i końcowy status są kierowane do standardowego wyjścia (`stdout`) albo do pliku wskazanego opcją `--output`. Komunikaty postępu, ostrzeżenia i diagnostyka mogą trafiać do `stderr`. Dzięki temu wynik JSON lub JSONL można przekazać do innego programu bez łączenia go z komunikatami diagnostycznymi.

Kod `0` oznacza pomyślne zakończenie. Kod różny od `0` oznacza błąd, nieprawidłowe argumenty albo niepełny stan wymaganych artefaktów. W automatyzacji należy sprawdzać zarówno kod zakończenia, jak i treść wyniku.

---

## 2. Tworzenie korpusu

Kreator przetwarza pliki źródłowe, wykonuje skonfigurowaną analizę językową i zapisuje gotowy korpus w formacie `.parquet`. Do uruchomienia kreatora służy moduł:

```powershell
python -m korpusuj.corpus.creator_cli
```

### Najprostsze polecenie

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus.parquet
```

`--input` wskazuje plik albo katalog wejściowy. Opcję można podać wielokrotnie. Katalogi są rozwijane nierekurencyjnie, dlatego pliki znajdujące się w podkatalogach nie są automatycznie dołączane.

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dane\teksty `
    --input dane\uzupelnienie.xlsx `
    --output korpus.parquet
```

### Argumenty kreatora

- `--input PATH` — wymagany plik lub katalog wejściowy; opcja powtarzalna.
- `--output FILE.parquet` — wymagany docelowy plik korpusu.
- `--model {stanza,spacy}` — mechanizm analizy językowej; domyślnie `stanza`.
- `--metadata FILE.xlsx` — opcjonalny zewnętrzny arkusz metadanych.
- `--mapping FILE.json` — mapowanie pól Korpusuj na rzeczywiste nazwy kolumn arkusza XLSX.
- `--resume` — wznowienie zgodnego, wcześniej przerwanego zadania.
- `--no-ner` — wyłączenie rozpoznawania jednostek nazwanych.
- `--no-coreference` — wyłączenie rozwiązywania koreferencji.
- `--format {json,text}` — format końcowego statusu w `stdout`; domyślnie `json`.
- `--pretty` — czytelne formatowanie statusu JSON.
- `--quiet` — wyciszenie komunikatów postępu w `stderr`; nie zmienia końcowego statusu w `stdout`.

### Wybór mechanizmu analizy

Domyślnie używana jest Stanza:

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus_stanza.parquet `
    --model stanza
```

Aby użyć spaCy:

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus_spacy.parquet `
    --model spacy
```

Wyniki Stanza i spaCy mogą się różnić. Korpusy przeznaczone do bezpośredniego porównania najlepiej przygotowywać za pomocą tego samego mechanizmu i tego samego zestawu warstw.

### NER i koreferencja

NER i koreferencja są domyślnie włączone. Aby wyłączyć jedną albo obie warstwy:

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus.parquet `
    --no-coreference
```

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus.parquet `
    --no-ner `
    --no-coreference
```

Wyłączenie warstwy skraca przetwarzanie i zmniejsza plik wynikowy, ale funkcje wymagające tej anotacji nie będą później dostępne.

### Metadane i mapowanie kolumn

Zewnętrzny arkusz metadanych można wskazać opcją `--metadata`:

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --metadata metadane.xlsx `
    --output korpus.parquet
```

Jeżeli nazwy kolumn nie odpowiadają polom oczekiwanym przez Korpusuj, należy przekazać plik JSON przez `--mapping`:

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dane.xlsx `
    --mapping mapowanie.json `
    --output korpus.parquet
```

Plik mapowania musi być zapisany w UTF-8. Kluczem jest nazwa pola Korpusuj, a wartością rzeczywista nazwa kolumny w XLSX. Przykładowa postać:

```json
{
  "Treść": "tekst_dokumentu",
  "Nazwa pliku": "identyfikator",
  "Autor": "autor_dokumentu",
  "Data publikacji": "data"
}
```

### Wznawianie przerwanego zadania

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus.parquet `
    --resume
```

Wznowienie wymaga zgodnych plików częściowych lub istniejącego wyniku oraz takich samych ustawień przetwarzania. Nie należy zmieniać mechanizmu NLP ani zestawu warstw między przerwaniem a wznowieniem.

### Status tekstowy i JSON

Domyślnym formatem końcowego statusu jest JSON:

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus.parquet `
    --pretty
```

Status tekstowy:

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus.parquet `
    --format text
```

W automatyzacji zalecany jest JSON. `--quiet` wycisza postęp, ale zachowuje końcowy status.

---

## 3. Zarządzanie indeksami

Indeksy przyspieszają wyszukiwanie i udostępniają dane potrzebne do analiz składniowych. Są artefaktami powiązanymi z konkretnym plikiem korpusu i powinny być tworzone, sprawdzane oraz przebudowywane jako jeden zestaw. Do zarządzania nimi służy:

```powershell
python -m korpusuj.index.cli
```

Dostępne podkomendy:

```text
create
status
rebuild
```

Zestaw obejmuje indeks wyszukiwania `.search` oraz pamięć zależności składniowych `.dep_cache`. Oba artefakty należy traktować jako jeden zestaw powiązany z plikiem `.parquet`.

### Tworzenie indeksów

```powershell
python -m korpusuj.index.cli create korpus.parquet
```

Własna ścieżka indeksu wyszukiwania:

```powershell
python -m korpusuj.index.cli create korpus.parquet `
    --index indeksy\korpus.search
```

### Sprawdzanie stanu

```powershell
python -m korpusuj.index.cli status korpus.parquet
```

Status w czytelnej postaci JSON:

```powershell
python -m korpusuj.index.cli status korpus.parquet `
    --format json `
    --pretty
```

Polecenie sprawdza cały zestaw artefaktów. W automatyzacji należy interpretować kod zakończenia oraz pola `status`, `artifacts`, `reasons` i wyniki kontroli integralności w zwróconym obiekcie.

### Przebudowa indeksów

```powershell
python -m korpusuj.index.cli rebuild korpus.parquet
```

Przebudowa odtwarza zestaw indeksów. Opcja `--force` pozwala wymusić operację w sytuacji, w której istniejące artefakty normalnie powstrzymałyby zapis:

```powershell
python -m korpusuj.index.cli rebuild korpus.parquet `
    --force `
    --progress on
```

### Profile indeksu

Dostępne są profile:

```text
compact
full
```

Przykład:

```powershell
python -m korpusuj.index.cli create korpus.parquet `
    --profile compact
```

Profil `full` obejmuje pełny standardowy zestaw atrybutów, natomiast `compact` ogranicza rozmiar indeksu. Zamiast profilu można przekazać własny zestaw atrybutów przez `--attrs`. Opcje `--profile` i `--attrs` są wzajemnie wykluczające.

```powershell
python -m korpusuj.index.cli create korpus.parquet `
    --attrs base,orth,upos
```

### Opcje wspólne

- `parquet` — wymagany argument pozycyjny wskazujący korpus.
- `--index INDEX_PATH` — opcjonalna ścieżka indeksu `.search`.
- `--profile {compact,full}` — standardowy profil indeksu.
- `--attrs ATTRS` — własny zestaw atrybutów.
- `--format {json,text}` — format wyniku; domyślnie `json`.
- `--pretty` — czytelne formatowanie JSON.

Dla `create` i `rebuild` dostępne są ponadto:

- `--force` — wymuszenie operacji;
- `--progress {auto,off,on}` — sterowanie komunikatami postępu.

`status` nie modyfikuje indeksów i nie udostępnia `--force` ani `--progress`.

---

## 4. Wyszukiwanie

Moduł wyszukiwania wykonuje zapytania CQL, zwraca konkordancje i może uruchamiać dodatkowe analizy. Można przekazać pojedyncze zapytanie, odczytać je z pliku albo przetworzyć listę zapytań. Moduł uruchamia się poleceniem:

```powershell
python -m korpusuj.search.cli
```

`--corpus-path` jest zawsze wymagane. Zapytanie można przekazać bezpośrednio, odczytać z jednego pliku albo uruchomić listę zapytań.

### Pojedyncze zapytanie

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]'
```

Opcjonalna nazwa korpusu:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --corpus-name "Korpus prasy" `
    --query '[base="wojna"]'
```

Jeżeli `--corpus-name` nie zostanie podane, używana jest nazwa pliku `.parquet` bez rozszerzenia.

### Zapytanie w pliku

Plik UTF-8 zawierający jedno zapytanie:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query-file zapytanie.cql
```

### Lista zapytań

Plik listy powinien zawierać po jednym zapytaniu CQL w każdym niepustym wierszu:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query-list zapytania.txt `
    --format jsonl `
    --output wyniki.jsonl
```

`--continue-on-error` powoduje zapis rekordu błędu dla niepoprawnego zapytania i przejście do następnej pozycji:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query-list zapytania.txt `
    --continue-on-error `
    --format jsonl `
    --output wyniki.jsonl
```

Opcje `--query`, `--query-file` i `--query-list` są wzajemnie wykluczające.

### Kontekst, limit i przesunięcie

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --left-context 15 `
    --right-context 15 `
    --limit 100 `
    --offset 0
```

- `--left-context` — liczba tokenów lewego kontekstu; domyślnie `10`.
- `--right-context` — liczba tokenów prawego kontekstu; domyślnie `10`.
- `--full-context-size` — wewnętrzne okno pełnego kontekstu; domyślnie `250`.
- `--limit` — maksymalna liczba zwróconych wierszy konkordancji; domyślnie zwracane są wszystkie.
- `--offset` — przesunięcie początku zwracanej strony; domyślnie `0`.

`limit` ogranicza liczbę zwracanych wierszy, ale nie powinien być interpretowany jako całkowita liczba trafień. Pole `total_hits` opisuje pełną liczbę dopasowań.

### Parametry wydajności zapytań składniowych

- `--candidate-max-docs` — budżet dokumentów wstępnie ładowanych dla kandydatów zależnościowych; domyślnie `3000`.
- `--candidate-stream-batch-docs` — rozmiar partii strumieniowania kandydatów; domyślnie `256`.

> **Ważne:** są to parametry wydajności i pamięci podręcznej, a nie limity wyników. Zmiana tych wartości nie może służyć do obcinania liczby trafień.

### Formaty wyniku

Dostępne formaty:

```text
json
jsonl
text
xlsx
csv
```

Domyślnie używany jest JSON. Dla XLSX i CSV wymagane jest `--output`.

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --format xlsx `
    --output wyniki.xlsx
```

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --format csv `
    --output wyniki.csv
```

`--pretty` formatuje JSON z wcięciami. `--normalized` zwraca ujednolicony schemat wyników i jest trybem domyślnym. `--raw` zwraca surowe wiersze w kształcie wewnętrznym i jest przeznaczone głównie do diagnostyki lub zgodności ze starszymi integracjami.

### Wybór pól i kontekstu rozszerzonego

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --fields doc_id,match_text,left_context,right_context,metadata `
    --no-extended-context
```

- `--fields` — lista pól JSON lub JSONL oddzielonych przecinkami.
- `--no-extended-context` — pomija pola `extended_left`, `extended_match` i `extended_right`.
- `--max-context-chars N` — skraca tekstowe pola kontekstu do najwyżej `N` znaków.
- `--include-request` — dołącza obiekt żądania do koperty JSON.
- `--include-raw` — dołącza pole `raw` do wyniku znormalizowanego.

---

## 5. Statystyki

Analiza statystyczna podsumowuje zbiór trafień znalezionych przez zapytanie. Może być zwracana razem z konkordancjami albo samodzielnie. Statystyki razem z wynikami oblicza się następująco:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics statistics `
    --pretty
```

Aby pominąć wiersze konkordancji i zwrócić samą analizę:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics statistics `
    --analytics-only `
    --pretty
```

Zakres analizy określa `--analytics-scope`:

- `all-matches` — analiza wszystkich dopasowań; wartość domyślna;
- `returned-results` — analiza tylko zwracanej strony wyników.

Jeżeli celem jest opis całego zbioru trafień, należy zachować `all-matches`. `returned-results` jest użyteczne przy analizie konkretnej strony lub próbki.

---

## 6. Kolokacje

Analiza kolokacji wskazuje wyrazy współwystępujące z trafieniem zapytania. Korpusuj obsługuje dwa sposoby wyznaczania kolokatów: liniowy, oparty na położeniu tokenów w tekście, oraz składniowy, oparty na relacjach zależnościowych.

### Kolokacje liniowe

Kolokacje liniowe są wyznaczane w oknie tokenów po lewej i prawej stronie trafienia. Wielkość obu części okna można ustawić niezależnie, a analizę można ograniczyć do granic zdania.

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics collocations `
    --colloc-mode linear `
    --colloc-left-span 5 `
    --colloc-right-span 5 `
    --colloc-sentence-bound true `
    --colloc-sort log-dice `
    --pretty
```

### Kolokacje składniowe

Kolokacje składniowe są wyznaczane na podstawie bezpośrednich relacji zależnościowych, a nie odległości tokenów w tekście. Pozwala to odnajdywać powiązane gramatycznie elementy także wtedy, gdy nie stoją obok siebie.

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics collocations `
    --colloc-mode syntactic `
    --colloc-syn-dir both `
    --colloc-deprel amod `
    --colloc-sort log-dice `
    --pretty
```

Kolokacje składniowe wymagają anotacji zależnościowych w korpusie oraz gotowej i zgodnej pamięci `.dep_cache`.

### Parametry kolokacji

Sposób wyznaczania kolokatów określa opcja `--colloc-mode`. Wartość `linear` wyszukuje wyrazy występujące w określonym otoczeniu tekstowym trafienia, natomiast `syntactic` uwzględnia wyrazy połączone z trafieniem relacją zależnościową. Domyślnie używany jest tryb liniowy.

W trybie liniowym opcje `--colloc-left-span` i `--colloc-right-span` określają liczbę tokenów analizowanych odpowiednio po lewej i prawej stronie trafienia. Domyślna wartość obu parametrów wynosi `5`. Opcja `--colloc-sentence-bound` decyduje, czy wyszukiwanie kolokatów ma zostać ograniczone do granic zdania. Przy wartości `true`, używanej domyślnie, tokeny z sąsiednich zdań nie są uwzględniane; wartość `false` pozwala przekroczyć granicę zdania w ramach zadanego zakresu liniowego.

W trybie składniowym opcja `--colloc-syn-dir` określa kierunek analizowanej relacji. Wartość `dependent` uwzględnia podrzędniki trafienia, `head` — jego nadrzędnik, natomiast `both` — oba kierunki. Domyślnie uwzględniane są podrzędniki. Opcja `--colloc-deprel` pozwala dodatkowo ograniczyć wyniki do wskazanego typu relacji zależnościowej, na przykład `amod`. Jeżeli opcja nie zostanie podana, uwzględniane są wszystkie relacje. Kolokacje składniowe wymagają odpowiednich anotacji korpusu oraz poprawnej pamięci `.dep_cache`.

Opcja `--colloc-form` określa postać, według której kolokaty są grupowane i prezentowane. Wartość `base` oznacza lemat, a `orth` — formę tekstową występującą w korpusie. Domyślnie używany jest lemat. Ustawienie to wpływa także na interpretację etykiet przekazywanych przez `--collocate-concordance`.

Dwa progi pozwalają odrzucić rzadkie kolokaty. `--colloc-min-freq` określa minimalną liczbę współwystąpień kolokatu z trafieniem. `--colloc-min-range` określa natomiast minimalny zasięg, czyli liczbę odrębnych dokumentów lub wyników, w których kolokat musi wystąpić. Domyślna wartość obu progów wynosi `1`.

Kolejność wierszy w tabeli kolokacji ustala opcja `--colloc-sort`. Dostępne miary to `log-likelihood`, `mi`, `t-score` i `log-dice`; domyślnie używana jest miara `log-dice`. Opcja `--colloc-limit` ogranicza liczbę wierszy wyświetlanych w tabeli dopiero po obliczeniu miar i uporządkowaniu kolokatów. Nie ogranicza liczby trafień głównego zapytania ani liczby konkordancji.

Pełna lista opcji:

- `--colloc-mode {linear,syntactic}` — wybiera liniowy albo składniowy sposób wyznaczania kolokatów; domyślnie `linear`.
- `--colloc-syn-dir {dependent,head,both}` — w trybie składniowym wybiera podrzędniki, nadrzędnik albo oba kierunki relacji; domyślnie `dependent`.
- `--colloc-deprel RELACJA` — w trybie składniowym ogranicza kolokaty do wskazanej relacji zależnościowej, np. `amod`; brak wartości oznacza wszystkie relacje.
- `--colloc-form {base,orth}` — grupuje kolokaty według lematu albo formy tekstowej; domyślnie `base`.
- `--colloc-left-span N` — w trybie liniowym określa liczbę tokenów analizowanych po lewej stronie trafienia; domyślnie `5`.
- `--colloc-right-span N` — w trybie liniowym określa liczbę tokenów analizowanych po prawej stronie trafienia; domyślnie `5`.
- `--colloc-sentence-bound {true,false}` — określa, czy liniowe okno kolokacji może przekraczać granice zdania; domyślnie `true`.
- `--colloc-min-freq N` — ustala minimalną liczbę współwystąpień wymaganą do uwzględnienia kolokatu; domyślnie `1`.
- `--colloc-min-range N` — ustala minimalną liczbę odrębnych dokumentów lub wyników, w których kolokat musi wystąpić; domyślnie `1`.
- `--colloc-sort {log-likelihood,mi,t-score,log-dice}` — wybiera miarę służącą do uporządkowania tabeli kolokacji; domyślnie `log-dice`.
- `--colloc-limit N` — ogranicza liczbę wierszy tabeli po rankingu; nie ogranicza trafień zapytania ani konkordancji.

### Filtrowanie kandydatów

Filtrowanie kandydatów pozwala ograniczyć analizę do kolokatów o wskazanych cechach morfosyntaktycznych, na przykład tylko do rzeczowników lub przymiotników. `--collocate-filter` jest opcją powtarzalną. W pojedynczej grupie warunki są łączone logicznym AND; kolejne grupy są łączone logicznym OR.

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics collocations `
    --collocate-filter upos=NOUN,pos=subst `
    --collocate-filter upos=ADJ `
    --pretty
```

### Konkordancje kolokatów

Standardowo konkordancje odnoszą się do trafień głównego zapytania. Można jednak zwrócić konteksty wybranych kolokatów, aby zobaczyć ich rzeczywiste użycia w materiale. Aby zwrócić wystąpienia kolokatów zamiast podstawowych trafień zapytania:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics collocations `
    --concordance-of collocates `
    --collocate-concordance konflikt `
    --pretty
```

`--collocate-concordance` można powtarzać. Etykieta kolokatu jest interpretowana zgodnie z `--colloc-form`.

---

## 7. Profil kolokacyjny

Profil kolokacyjny porządkuje powiązania badanego lematu według relacji składniowych i miar asocjacyjnych. Ułatwia dzięki temu porównywanie typowych sposobów użycia wyrazu.

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --profile collocational `
    --profile-sort log-dice `
    --profile-min-freq 2 `
    --pretty
```

Aby zwrócić wyłącznie profil:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --profile collocational `
    --profile-only `
    --pretty
```

### Zapytania wieloelementowe

Dla zapytania zawierającego kilka tokenów należy wskazać token centralny. Numeracja zaczyna się od `1`:

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wypowiedzieć"] [base="wojna"]' `
    --profile collocational `
    --profile-target-token 2 `
    --pretty
```

`--profile-target-lemma` pozwala jawnie podać lemat, jeżeli nie można go jednoznacznie wywnioskować z docelowego tokenu.

### Parametry profilu

- `--profile {collocational}` — włączenie profilu kolokacyjnego.
- `--profile-only` — pominięcie głównych wierszy konkordancji.
- `--profile-target-token N` — numer tokenu centralnego, liczony od `1`.
- `--profile-target-lemma LEMAT` — opcjonalne jawne wskazanie lematu.
- `--profile-sort {log-dice,log-likelihood,mi,t-score,frequency}` — sposób sortowania.
- `--profile-min-freq N` — minimalna frekwencja; domyślnie `2`.
- `--profile-max-rows-per-relation N` — maksymalna liczba wierszy w relacji; domyślnie bez limitu.
- `--profile-layout {tree,flat}` — układ drzewa zgodny z grupami GUI albo płaska tabela; domyślnie `tree`.
- `--profile-example-refs N` — liczba technicznych odwołań do przykładów; domyślnie `0`.
- `--profile-examples N` — liczba przykładów tekstowych; domyślnie `0`.
- `--profile-example-context N` — szerokość kontekstu przykładu; domyślnie `6` tokenów.
- `--profile-expand-mwe {true,false}` — rozwijanie wielowyrazowych kolokatów; domyślnie `false`.

---

## 8. Tworzenie podkorpusu

Podkorpus jest nowym plikiem `.parquet` zawierającym wybraną część korpusu źródłowego. Można wybrać dokumenty na podstawie zapytania CQL, metadanych albo połączenia dostępnych selektorów. Podkorpus tworzy się przez podanie `--subcorpus-output` oraz co najmniej jednego jawnego selektora. Główne `--query` nie jest automatycznie używane do tworzenia podkorpusu.

### Podkorpus według zapytania

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --subcorpus-output podkorpus.parquet `
    --subcorpus-query '[base="wojna"]'
```

### Podkorpus według metadanych

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --subcorpus-output podkorpus.parquet `
    --subcorpus-author "Kowalski" `
    --subcorpus-date-from 2020-01-01 `
    --subcorpus-date-to 2024-12-31
```

Dostępne selektory:

- `--subcorpus-query CQL` — zapytanie używane wyłącznie do tworzenia podkorpusu;
- `--subcorpus-author VALUE` — pole `Autor` zawiera wartość, bez rozróżniania wielkości liter;
- `--subcorpus-title VALUE` — pole `Tytuł` zawiera wartość, bez rozróżniania wielkości liter;
- `--subcorpus-date-from VALUE` — dolna granica pola `Data publikacji`;
- `--subcorpus-date-to VALUE` — górna granica pola `Data publikacji`.

---

## 9. Automatyzacja i diagnostyka

W skryptach automatyzujących należy oddzielać dane wynikowe od komunikatów diagnostycznych, kontrolować kod zakończenia procesu i jawnie wskazywać format oraz plik wyjściowy.

### Zapis wyniku do pliku

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --format json `
    --pretty `
    --output wynik.json
```

### Sterowanie postępem

`--progress` przyjmuje:

- `auto` — postęp tylko w interaktywnym terminalu;
- `off` — bez wskaźnika postępu;
- `on` — wymuszenie wskaźnika.

Do automatyzacji zalecane jest `--progress off`, aby diagnostyka nie zakłócała logów procesu.

### Logi

- `--verbose` — włącza rozszerzone komunikaty działania procesu.
- `--diagnostics-logs` — włącza szczegółowe wpisy wykonawcze oznaczone jako `[DIAG ...]`.

Logi diagnostyczne należy włączać podczas badania konkretnego błędu. Nie są wymagane w zwykłej pracy.

### Przykład PowerShell z kontrolą kodu zakończenia

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --format json `
    --output wynik.json `
    --progress off

if ($LASTEXITCODE -ne 0) {
    Write-Error "Wyszukiwanie zakończyło się błędem: $LASTEXITCODE"
    exit $LASTEXITCODE
}
```

---

## 10. Rozwiązywanie problemów

### Polecenie nie rozpoznaje modułu `korpusuj`

Uruchom polecenie z głównego katalogu projektu i upewnij się, że aktywne środowisko Pythona zawiera zależności aplikacji.

### Ścieżka nie jest rozpoznawana

Ujmij ścieżkę w cudzysłów, szczególnie jeśli zawiera spacje lub znaki specjalne.

### Creator wyświetla ostrzeżenie `HERFERENCE_LAZY_IMPORT`

Ostrzeżenie może pojawić się w `stderr` podczas ładowania modułu koreferencji. Jeżeli polecenie `--help` lub właściwe zadanie kończy się kodem `0`, samo ostrzeżenie nie oznacza niepowodzenia. W automatyzacji należy oddzielać `stdout` od `stderr`.

### Nie można wznowić tworzenia korpusu

Sprawdź ścieżkę wyjściową, obecność zgodnych plików częściowych oraz zgodność mechanizmu NLP i warstw anotacji z poprzednim uruchomieniem.

### Status indeksów nie jest poprawny

Uruchom:

```powershell
python -m korpusuj.index.cli status korpus.parquet `
    --format json `
    --pretty
```

Sprawdź status obu artefaktów i listę przyczyn. W razie potrzeby wykonaj `rebuild`.

### Wyszukiwanie zwraca zero trafień

Sprawdź składnię CQL, nazwę atrybutu, pisownię wartości, dostępność wymaganej warstwy anotacji oraz to, czy wskazano właściwy korpus.

### Wynik XLSX lub CSV nie został zapisany

Dla formatów `xlsx` i `csv` wymagane jest `--output` z odpowiednim plikiem docelowym.

### Statystyki obejmują tylko część wyników

Sprawdź `--analytics-scope`. Aby analizować wszystkie dopasowania niezależnie od `--limit`, użyj:

```text
--analytics-scope all-matches
```

### Profil nie może ustalić tokenu centralnego

Dla zapytania wieloelementowego ustaw `--profile-target-token`. Numeracja tokenów zaczyna się od `1`.

### Kolokacje składniowe są niedostępne

Sprawdź, czy korpus zawiera anotacje składniowe i czy zestaw indeksów obejmuje poprawną pamięć `.dep_cache`.

---

## 11. Skrócona lista poleceń

### Tworzenie korpusu

```powershell
python -m korpusuj.corpus.creator_cli `
    --input dokumenty `
    --output korpus.parquet
```

### Utworzenie indeksów

```powershell
python -m korpusuj.index.cli create korpus.parquet
```

### Sprawdzenie indeksów

```powershell
python -m korpusuj.index.cli status korpus.parquet `
    --pretty
```

### Wyszukiwanie

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --format text
```

### Statystyki

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics statistics `
    --analytics-only `
    --pretty
```

### Kolokacje

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --analytics collocations `
    --colloc-sort log-dice `
    --pretty
```

### Profil kolokacyjny

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --query '[base="wojna"]' `
    --profile collocational `
    --profile-only `
    --pretty
```

### Podkorpus

```powershell
python -m korpusuj.search.cli `
    --corpus-path korpus.parquet `
    --subcorpus-output podkorpus.parquet `
    --subcorpus-query '[base="wojna"]'
```
