# Przewodnik po języku zapytań wyszukiwarki Korpusuj

Język CQL (*Corpus Query Language*) w aplikacji Korpusuj służy do wyszukiwania tokenów i sekwencji tokenów w anotowanym korpusie. Zapytania mogą wykorzystywać formy tekstowe, lematy, części mowy, cechy morfologiczne, relacje składniowe, jednostki nazwane, koreferencję, treść zdań, frekwencję oraz metadane dokumentów.

Działanie zapytania zależy od warstw zapisanych w korpusie. Zapytania o lemat korzystają z lematyzacji, zapytania składniowe — z anotacji zależnościowej, `ner` — z warstwy jednostek nazwanych, a `coref` — z warstwy koreferencji.

---

## Podstawowa składnia języka zapytań

Zapytanie dotyczące pojedynczego tokenu zapisuje się w nawiasach kwadratowych. Tokenem może być słowo, liczba, znak interpunkcyjny albo inny element wydzielony podczas analizy tekstu.

Podstawowy warunek ma postać:

```cql
[atrybut="wartość"]
```

Atrybut wskazuje informację, która ma zostać sprawdzona, a wartość określa poszukiwaną cechę. Przykładowo:

```cql
[base="wojna"]
```

Atrybut `base` oznacza lemat, czyli formę podstawową. Zapytanie znajduje wystąpienia, którym podczas anotacji przypisano lemat `wojna`. Wyniki mogą obejmować różne formy fleksyjne, takie jak „wojna”, „wojny”, „wojnę” i „wojną”, o ile występują w korpusie i zostały prawidłowo zanalizowane.

Aby wyszukać konkretną formę zapisaną w tekście, należy użyć atrybutu `orth`:

```cql
[orth="wojną"]
```

### Równość i wykluczenie wartości

Operator `=` wybiera tokeny spełniające warunek:

```cql
[upos="NOUN"]
```

Operator `!=` wybiera tokeny, które nie mają wskazanej wartości:

```cql
[upos!="NOUN"]
```

### Łączenie warunków dotyczących jednego tokenu

Kilka warunków dotyczących tego samego tokenu łączy się operatorem `&`:

```cql
[orth="mam" & pos="subst"]
```

Zapytanie znajduje tokeny o formie `mam`, którym przypisano część mowy `subst`.

Można połączyć większą liczbę cech:

```cql
[base="bohater" & case="gen" & number="pl"]
```

Wszystkie warunki wewnątrz jednego segmentu muszą być spełnione przez ten sam token.

### Sekwencje tokenów

Segmenty zapisane kolejno tworzą sekwencję:

```cql
[base="wypowiedzieć"] [base="wojna"]
```

Zapytanie znajduje fragmenty, w których token o lemacie `wypowiedzieć` znajduje się bezpośrednio przed tokenem o lemacie `wojna`.

Można łączyć różne atrybuty:

```cql
[base="móc"] [orth="zjeść"] [orth="obiad"] [orth="\\."]
```

Wynikiem sekwencji jest cały dopasowany fragment — od pierwszego do ostatniego tokenu.

### Alternatywa kompletnych zapytań

Dwie kompletne gałęzie zapytania można rozdzielić operatorem `||`:

```cql
([orth="władza"] [orth="mediów"]) || ([base="władza"] [base="partia"])
```

Aplikacja wykonuje obie gałęzie i łączy ich wyniki. Nawiasy okrągłe wyznaczają granice poszczególnych sekwencji.

Operatora `||` nie należy mylić z pojedynczym znakiem `|` używanym wewnątrz wartości:

```cql
[base="kot|pies"]
```

Pojedynczy znak `|` oznacza tutaj alternatywę wartości: lemat `kot` albo `pies`.

---

## Podstawowe atrybuty

### `orth` — forma tekstowa

```cql
[orth="Anię"]
```

Wyszukuje formę zapisaną w tekście jako `Anię`.

### `base` — lemat

```cql
[base="Ania"]
```

Wyszukuje wystąpienia, którym przypisano lemat `Ania`, niezależnie od ich formy fleksyjnej.

### `pos` — część mowy w tagsecie modelu

```cql
[pos="subst"]
```

Wyszukuje tokeny oznaczone wartością `subst`. Zestaw wartości zależy od modelu użytego do utworzenia korpusu.

### `upos` — uniwersalna część mowy

```cql
[upos="VERB"]
```

Wyszukuje tokeny oznaczone jako czasowniki w tagsecie Universal POS.

### `deprel` — relacja zależnościowa

```cql
[deprel="nsubj"]
```

Wyszukuje tokeny pełniące wskazaną funkcję składniową względem nadrzędnika.

### `head` — nadrzędnik tokenu

```cql
[base="ryba" & head="zjeść"]
```

Wyszukuje tokeny o lemacie `ryba`, których bezpośrednim nadrzędnikiem jest token pasujący do wartości `zjeść`.

### `dependent` — podrzędnik tokenu

```cql
[base="miasto" & dependent="piękny"]
```

Wyszukuje tokeny o lemacie `miasto`, które mają bezpośredni podrzędnik pasujący do wartości `piękny`.

### `ner` — jednostka nazwana

```cql
[ner="S-persName"]
```

Wyszukuje tokeny oznaczone wskazaną etykietą jednostki nazwanej.

### `coref` — koreferencja

```cql
[coref(H)="Polska"]
```

Wyszukuje elementy klastra koreferencyjnego zgodnie z wybraną rolą.

### `window_base` i `window_orth` — obecność w kontekście

```cql
[base="pies" & window_base(5)="kot"]
```

Sprawdza, czy w odległości do pięciu tokenów od badanego tokenu znajduje się wskazany lemat.

---

## Składnia uproszczona

Korpusuj pozwala wpisywać proste zapytania bez nawiasów kwadratowych. Tekst:

```text
kot je rybę
```

jest interpretowany jako sekwencja form tekstowych:

```cql
[orth="kot"] [orth="je"] [orth="rybę"]
```

Interpunkcja przylegająca do wyrazu jest wydzielana jako osobny token. Zapytanie:

```text
Kto wygrał mecz?
```

odpowiada sekwencji:

```cql
[orth="Kto"] [orth="wygrał"] [orth="mecz"] [orth="\\?"]
```

Można połączyć pełny segment CQL ze zwykłym tekstem:

```text
[base="móc"] zjeść obiad.
```

Pierwszy element jest wyszukiwany według lematu, a pozostałe jako formy tekstowe:

```cql
[base="móc"] [orth="zjeść"] [orth="obiad"] [orth="\\."]
```

Gwiazdka wpisana jako osobny element oznacza dokładnie jeden dowolny token:

```text
Polska * Niemcy
```

co odpowiada zapytaniu:

```cql
[orth="Polska"] [*] [orth="Niemcy"]
```

---

## Wyrażenia regularne

Wyrażenia regularne są obsługiwane wewnątrz segmentów CQL. Najczęściej używane metaznaki to:

- `.` — dowolny pojedynczy znak;
- `?` — zero albo jedno wystąpienie poprzedzającego elementu;
- `*` — zero albo więcej wystąpień poprzedzającego elementu;
- `+` — jedno albo więcej wystąpień poprzedzającego elementu;
- `[a-z]` — znak z podanego zakresu;
- `\\d` — cyfra;
- `\\w` — znak słowa;
- `\\s` — biały znak;
- `|` — alternatywa;
- `( )` — grupa wzorca;
- `\\` — potraktowanie następnego znaku dosłownie.

```cql
[orth="koty?"]
```

Znajduje formę `kot` albo `koty`.

```cql
[orth="..."]
```

Znajduje tokeny składające się z dokładnie trzech znaków.

```cql
[orth="woj.*"]
```

Znajduje formy rozpoczynające się od `woj`, na przykład „wojna”, „wojskowy” lub „województwo”.

```cql
[orth=".*a"]
```

Znajduje formy kończące się literą `a`.

```cql
[orth="dom.+"]
```

Znajduje formy rozpoczynające się od `dom` i zawierające co najmniej jeden dalszy znak.

```cql
[orth="kwesti(a|ę)"]
```

Znajduje formę `kwestia` albo `kwestię`.

```cql
[orth="\\."]
```

Znajduje kropkę. Ukośnik odwrotny powoduje, że znak `.` jest rozumiany dosłownie.

### Wyszukiwanie fragmentu wartości

Wartość poprzedzona znakiem `~` służy do wyszukiwania ciągu wewnątrz tokenu:

```cql
[orth="~zys"]
```

Zapytanie może znaleźć formy takie jak „kryzys”, „kryzysowy” i „zysk”.

---

## Operator zakresu

Segment:

```cql
[*]
```

oznacza jeden dowolny token.

```cql
[orth="Polska"] [*] [orth="Niemcy"]
```

Zapytanie znajduje sekwencje, w których pomiędzy formami `Polska` i `Niemcy` znajduje się dokładnie jeden token.

Zakres liczby dowolnych tokenów zapisuje się bezpośrednio po luce:

```cql
[base="Ania"] [*][1,3] [base="Tomek"]
```

Zapytanie dopuszcza od jednego do trzech tokenów pomiędzy elementami.

Zakres może obejmować zero:

```cql
[base="kot"] [*][0,1] [base="zjeść"]
```

Elementy mogą znajdować się bezpośrednio obok siebie albo być rozdzielone jednym tokenem.

---

## Atrybuty cech morfosyntaktycznych

Cechy morfosyntaktyczne można łączyć z lematem lub częścią mowy. Najczęściej używane atrybuty to:

- `case` — przypadek;
- `number` — liczba;
- `gender` — rodzaj;
- `person` — osoba;
- `aspect` — aspekt.

```cql
[base="bohater" & case="gen" & number="pl"]
```

Zapytanie wyszukuje wystąpienia lematu `bohater` w dopełniaczu liczby mnogiej.

```cql
[upos="VERB" & person="pri" & number="sg" & aspect="imperf"]
```

Zapytanie wyszukuje czasowniki niedokonane w pierwszej osobie liczby pojedynczej.

Dokładne nazwy i wartości cech zależą od modelu językowego i warstw zapisanych w korpusie.

---

## Jednostki nazwane

Atrybut `ner` wyszukuje tokeny na podstawie etykiet jednostek nazwanych:

```cql
[ner="S-persName"]
```

Zestaw etykiet zależy od modelu użytego podczas tworzenia korpusu. Dla języka polskiego mogą występować typy takie jak:

- `persName` — nazwa osoby;
- `orgName` — nazwa organizacji;
- `geogName` — nazwa geograficzna;
- `placeName` — nazwa miejsca;
- `date` — data;
- `time` — czas.

Stanza dodaje do typu jednostki prefiks opisujący pozycję tokenu:

- `S-` — jednostka jednotokenowa;
- `B-` — początek jednostki wielotokenowej;
- `I-` — token wewnątrz jednostki;
- `E-` — koniec jednostki wielotokenowej;
- `O` — token nienależący do jednostki nazwanej.

Zapytanie:

```cql
[ner=".-persName"]
```

wykorzystuje kropkę jako dowolny pojedynczy znak. Dzięki temu znajduje etykiety osób z dowolnym jednoterowym prefiksem, na przykład `S-persName`, `B-persName`, `I-persName` i `E-persName`.

---

## Koreferencja

Koreferencja łączy wzmianki odnoszące się do tego samego obiektu, osoby lub pojęcia. Zapytania koreferencyjne wymagają korpusu z warstwą koreferencji.

### Dowolny element klastra — `coref`

```cql
[coref="Polska"]
```

Aplikacja odnajduje klaster zawierający wzmiankę pasującą do wartości `Polska`, a następnie zwraca tokeny należące do tego klastra.

### Głowa — `coref(H)`

```cql
[coref(H)="Kowalski"]
```

Zwraca tokeny oznaczone jako głowy wzmianek należących do dopasowanego klastra.

### Część — `coref(P)`

```cql
[pos="pron" & coref(P)="Warszawa"]
```

Wyszukuje zaimki oznaczone jako części wzmianek w klastrze związanym z wartością `Warszawa`.

### Pełna wzmianka — `coref(M)`

```cql
[coref(M)="Polska"]
```

Zwraca pełną wzmiankę należącą do dopasowanego klastra koreferencyjnego.

---

## Odległość i pozycja nadrzędnika lub podrzędnika

Atrybuty `head(...)` i `dependent(...)` mogą ograniczać pozycję elementu relacji względem badanego tokenu.

Wartość dodatnia oznacza pozycję po prawej stronie, a ujemna — po lewej. Można również stosować operatory porównania odległości.

```cql
[head(1)="makaron"]
```

Wyszukuje tokeny, których nadrzędnik pasujący do wartości `makaron` znajduje się o jedną pozycję tokenową po prawej stronie.

```cql
[dependent(<2)="smaczny"]
```

Wyszukuje tokeny mające podrzędnik `smaczny` spełniający podany warunek odległości.

---

## Analiza kontekstu

Atrybuty `window_base` i `window_orth` sprawdzają obecność wskazanego słowa w otoczeniu badanego tokenu — po lewej albo prawej stronie.

```cql
[base="pies" & window_base(5)="kot"]
```

Wyszukuje tokeny o lemacie `pies`, jeżeli lemat `kot` występuje nie dalej niż pięć tokenów od nich.

```cql
[orth="Polska" & window_orth(10)="gospodarka"]
```

Wyszukuje formę `Polska`, jeżeli w odległości do dziesięciu tokenów znajduje się forma `gospodarka`.

Jeżeli odległość nie zostanie podana, aplikacja przeszukuje domyślnie 50 tokenów po lewej i 50 tokenów po prawej stronie:

```cql
[pos="subst" & window_base="wybory"]
```

---

## Zagnieżdżanie warunków

Zagnieżdżone warunki w nawiasach klamrowych pozwalają dokładnie opisywać relacje `head` i `dependent`.

```cql
[base="trwać" & dependent={base="wojna" & deprel="nsubj"}]
```

Zapytanie znajduje token o lemacie `trwać`, jeżeli ma podrzędnik `wojna` w relacji `nsubj`.

Można wymagać kilku podrzędników:

```cql
[base="zjeść" & dependent={base="kot" & deprel="nsubj"} & dependent={base="ryba" & deprel="obj"}]
```

Możliwe jest zagnieżdżenie wielopoziomowe:

```cql
[base="zjeść" & dependent={deprel="obj" & base="ryba" & dependent={base="świeży" & deprel="amod"}}]
```

Zapytanie opisuje czasownik `zjeść`, jego dopełnienie `ryba` oraz zależny od rzeczownika modyfikator `świeży`.

Relację można zanegować:

```cql
[base="być" & dependent!={orth="nie"}]
```

Negacja oznacza brak podrzędnika spełniającego cały warunek zagnieżdżony.

---

## Filtrowanie po treści zdań

Operator `<s>` ogranicza dopasowanie do jednego zdania albo nakłada dodatkowe warunki na zdanie zawierające główne trafienie.

### Cała sekwencja w jednym zdaniu

```cql
[base="wygrać"] [*][1,3] [base="wojna"] <s>
```

Sekwencja musi w całości znajdować się w obrębie jednego zdania.

### Dodatkowy element w tym samym zdaniu

```cql
[base="wygrać"] <s [base="wojna"]>
```

Zapytanie znajduje token `wygrać` w zdaniu zawierającym również token o lemacie `wojna`. Kolejność obu elementów jest dowolna.

### Kilka elementów w dowolnej kolejności

Warunki zapisane w nawiasach okrągłych wymagają obecności wszystkich podanych elementów w tym samym zdaniu, ale nie ustalają ich kolejności:

```cql
[base="wygrać"] <s ([base="Chinka"]) ([base="set"])>
```

Zdanie musi zawierać zarówno lemat `Chinka`, jak i lemat `set`.

---

## Filtrowanie po frekwencji

Operatory `<frequency_base ...>` i `<frequency_orth ...>` ograniczają wyniki według częstości lematów albo form tekstowych w zbiorze wyników.

### Najczęstsze wartości — `top`

```cql
[base="pies"] <frequency_orth top="3">
```

Zapytanie zachowuje wyniki należące do trzech najczęstszych form tekstowych lematu `pies`.

```cql
[upos="NOUN"] <frequency_base top="10">
```

Zapytanie zachowuje wyniki należące do dziesięciu najczęstszych lematów wśród dopasowanych rzeczowników.

### Dolna i górna granica — `min` i `max`

```cql
[upos="VERB"] <frequency_base min="2" max="10">
```

Zapytanie pozostawia wyniki, których lemat występuje w rozpatrywanym zbiorze co najmniej 2 i nie więcej niż 10 razy.

Filtr frekwencji jest stosowany po wyszukaniu podstawowego zbioru wyników. Następnie aplikacja zlicza wartości `base` albo `orth` i stosuje parametry `top`, `min` i `max`.

---

## Filtrowanie po metadanych

Metadane opisują dokument jako całość. Filtr metadanych ogranicza dokumenty, w których wykonywana jest część tokenowa zapytania.

### Autor

```cql
[base="Tadeusz"] <autor="Mickiewicz">
```

Wyszukuje lemat `Tadeusz` w dokumentach, których pole autora spełnia podany warunek.

### Tytuł

```cql
[base="dzień"] <tytuł="~sen">
```

Wyszukuje lemat `dzień` w dokumentach, których tytuł zawiera ciąg `sen`.

### Data

```cql
[base="pies"] <data="2024-04-.*">
```

Wyszukuje lemat `pies` w dokumentach z datą pasującą do podanego wzorca.

Zakres dat można opisać dwoma warunkami:

```cql
[base="kot"] <data >= "2024-01-20"> <data <= "2025-02-25">
```

Zapytanie ogranicza wyniki do dokumentów z datą mieszczącą się w podanym przedziale, włącznie z wartościami granicznymi.

### Własne pola metadanych

Dodatkowe kolumny zapisane w korpusie są dostępne przez prefiks `metadane:`:

```cql
[base="Duda"] <metadane:portal="Wyborcza">
```

Wyszukuje lemat `Duda` w dokumentach, których pole `portal` ma wartość `Wyborcza`.

Dla pól liczbowych lub porządkowalnych można użyć operatorów porównania:

```cql
[base="mecz"] <metadane:rok > "2010">
```

---

## Złożone przykłady

### Czasownik z podmiotem i dopełnieniem

```cql
[base="zjeść" & dependent={base="kot" & deprel="nsubj"} & dependent={base="ryba" & deprel="obj"}]
```

### Dopełnienie z modyfikatorem

```cql
[base="zjeść" & dependent={deprel="obj" & base="ryba" & dependent={base="świeży" & deprel="amod"}}]
```

### Alternatywne etykiety podmiotu

```cql
[base="trwać" & dependent={base="wojna" & deprel="nsubj|nsubj:pass"}]
```

### Osoba związana składniowo z nazwą organizacji

```cql
[ner=".-persName" & dependent={orth="prezes" & dependent={ner=".-orgName"}}] [ner=".-persName"]
```

### Kilka lematów w jednym zdaniu

```cql
[base="wygrać"] <s ([base="Chinka"]) ([base="set"])>
```

### Zapytanie z datą i własną metadaną

```cql
[base="mecz"] <data >= "2020-01-01"> <metadane:gatunek="artykuł">
```

---

## Powiązana dokumentacja

- `gui.md` — wykonywanie zapytań i przeglądanie wyników w interfejsie graficznym;
- `cli.md` — uruchamianie zapytań, list zapytań, analiz i eksportu z wiersza poleceń;
- `quickstart.md` — podstawowy przebieg pracy z aplikacją;
- `architecture/search-pipeline.md` — techniczny opis przepływu wyszukiwania dla osób rozwijających aplikację.
