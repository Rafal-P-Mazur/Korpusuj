Niniejszy przewodnik opisuje pracę z aplikacją Korpusuj. Obejmuje wszystkie etapy: od tworzenia i otwierania korpusów, przez wyszukiwanie i przeglądanie konkordancji, aż po generowanie statystyk, wykresów, analizę kolokacji i profili kolokacyjnych. Omówione zostało również badanie sieci semantycznych, filtrowanie wyników według ram znaczeniowych, modelowanie tematyczne oraz eksportowanie danych.

*Uwaga: Składnia zapytań CQL oraz instrukcje korzystania z interfejsu wiersza poleceń zostały opisane w osobnych częściach dokumentacji.*

---

## 1. Rozpoczęcie pracy

### Uruchamianie aplikacji

Po zainstalowaniu programu uruchom Korpusuj ze skrótu albo przez plik `Korpusuj.exe`.

W wersji portable rozpakuj całe archiwum i uruchom znajdujący się w nim plik `Korpusuj.exe`.

Wersja GPU wymaga zgodnej karty NVIDIA. Jeżeli komputer nie ma takiej karty, użyj wersji CPU.

Uruchamianie poleceniem:

```powershell
python Korpusuj.py
```

jest przeznaczone wyłącznie do pracy ze źródłami.

### Otwieranie korpusu
Otwieranie gotowego korpusu przebiega następująco:
1. Z menu należy wybrać opcję **Nowy projekt**.
2. Wskazać plik korpusu w formacie `.parquet`.
3. Poczekać na zakończenie ładowania i przygotowywania projektu.
4. Po pomyślnym wczytaniu danych można rozpocząć wyszukiwanie lub uruchomić wybraną analizę.

**Ważne:** Pierwsze otwarcie dużego korpusu może potrwać dłużej, ponieważ aplikacja przygotowuje indeksy przyspieszające wyszukiwanie. Jeżeli korpus zawiera anotacje składniowe, mogą być również przygotowywane dane pomocnicze umożliwiające ich sprawne przeszukiwanie.

Jeżeli otwieranie zakończy się błędem, warto sprawdzić, czy plik istnieje i czy jest prawidłowym korpusem obsługiwanym przez aplikację. Szczegółowe informacje o ewentualnych błędach zapisywane są w pliku logów:

```text
logs/gui/korpusuj.log
```

### Okno główne aplikacji
W głównym oknie aplikacji znajdują się między innymi:
* pole do wprowadzania zapytań wyszukiwania,
* przycisk uruchamiający wyszukiwanie oraz **Konstruktor zapytań**,
* ustawienia szerokości lewego i prawego kontekstu,
* opcje sortowania wyników,
* tabela konkordancji z nawigacją między stronami,
* panele analityczne: statystyki, wykresy, kolokacje oraz profil kolokacyjny,
* przyciski dedykowane eksportowi danych i uruchamianiu analiz dodatkowych.

---

## 2. Tworzenie nowego korpusu

Wbudowany kreator umożliwia tworzenie korpusu na podstawie wskazanych dokumentów. Wynikiem jest pojedynczy plik `.parquet`, który można później otworzyć za pomocą opcji **Nowy projekt**.

W tym celu należy wybrać **Utwórz korpus**, określić mechanizm analizy językowej i potrzebne warstwy analizy, a następnie kliknąć **Wybierz pliki**. Program obsługuje różnorodne formaty: pliki tekstowe (TXT), dokumenty Word (DOCX), pliki PDF, arkusze kalkulacyjne (XLSX) oraz archiwa ZIP. Po załadowaniu materiałów na liście pozostawia się zaznaczone tylko te pozycje, które mają zostać przetworzone (zmiana strony nie usuwa zaznaczeń).

*  **TXT i DOCX:** Pliki czytane są jako czysty tekst; z dokumentów DOCX pobierane są kolejne akapity.
*  **PDF:** Aplikacja najpierw próbuje odczytać warstwę tekstową pliku PDF. Jeżeli dokument jest skanem, może użyć OCR opartego na EasyOCR, o ile wymagane komponenty i modele są dostępne. Bez OCR można przetwarzać pliki PDF zawierające zwykłą warstwę tekstową.
*  **XLSX:** Każdy wiersz jest traktowany jako odrębny dokument. W oknie mapowania należy wskazać kolumnę zawierającą treść oraz kolumnę Nazwa pliku, która identyfikuje dokument. Arkusz może zawierać również dodatkowe kolumny z metadanymi np. tytuł, data publikacji, autor itp.
*  **ZIP:** Archiwum jest rozpakowywane, a zawarte w nim obsługiwane dokumenty przetwarzane są osobno.

Po wczytaniu plików wybiera się polecenie **Przetwórz pliki**. Dalsze kroki zależą od typu dokumentów: w przypadku zwykłych tekstów pojawia się pytanie o metadane, natomiast dla arkuszy XLSX wyświetla się okno mapowania kolumn.

### Dołączanie metadanych z zewnętrznego pliku
Metadane to informacje opisujące dokument, niebędące jego główną treścią (np. autor, data, gatunek, źródło). Dołączenie do korpusu metadanych ułatwia późniejsze filtrowanie wyników i umożliwia generowanie zestawień zmian częstości w czasie.

Jeżeli informacje opisujące dokumenty znajdują się w osobnym arkuszu XLSX, można dołączyć go niezależnie od formatu plików źródłowych. Po kliknięciu "Przetwórz pliki" program wyświetli pytanie:
```text
Czy dodać osobny plik z metadanymi (np. metadane.xlsx)?
```
Aby dołączyć plik z metadanymi należy wybrać **Tak** i wskazać odpowiedni dokument. Plik z metadanymi powinien zawierać kolumnę **Nazwa pliku** (na jej podstawie metadane łączone są z plikami zawierającymi treść tekstów), kolejne kolumny powinny zawierać dodatkowe informacje opisujące dokumenty np. **Tytuł**, **Data publikacji**, **Autor**, **Gatunek**. Każdy wiersz powinien opisywać jeden dokument.

*Zalecenia dotyczące tworzenia pliku z metadanymi:*
* Najpewniejszą metodą jest wpisywanie w arkuszu pełnych nazw plików wraz z rozszerzeniami (np. `raport_01.pdf`).
* Każda nazwa powinna występować w arkuszu tylko raz. W przypadku powtórzeń, późniejszy wiersz nadpisuje wcześniejsze dane.
* Brak przypisanego wiersza w pliku metadanych nie wstrzymuje przetwarzania danego dokumentu. Zostanie on załadowany z domyślnymi wartościami (np. data ``0000-00-00``).
* Najlepiej stosować jednolity format daty (np. ``YYYY-MM-DD``). Aplikacja rozpoznaje polskie nazwy miesięcy, jednak ujednolicony zapis gwarantuje poprawne wyświetlanie na późniejszych wykresach.

### Analiza językowa i zapis

Przed rozpoczęciem należy wybrać mechanizm analizy językowej: [Stanza](https://stanfordnlp.github.io/stanza/) lub [spaCy](https://spacy.io/). Można także wybrać dodatkowe warstwy przetwarzania tekstu, w tym rozpoznawanie jednostek nazwanych i koreferencję. Koreferencja języka polskiego w spaCy korzysta z [Herference](https://github.com/ipipan/herference).

Pierwsze użycie wybranej funkcji może wymagać pobrania modeli. Podstawowy polski pakiet Stanza ma około 413 MB. Koreferencja Stanza może dodatkowo pobrać adapter około 132 MB oraz model XLM-RoBERTa Large około 2,24 GB. Duży plik `model.safetensors` zawiera wagi modelu bazowego i jest oczekiwanym elementem pobierania.

Modele i cache są przechowywane w katalogu modeli wybranym podczas instalacji, w katalogu `models` wersji portable albo w katalogu `models` w rootcie projektu przy uruchamianiu ze źródeł.

Podczas pracy wyświetlany jest pasek postępu. Wyniki są zapisywane na bieżąco w plikach `.part_*`. Po przerwaniu przetwarzania można wznowić zadanie, wskazując tę samą ścieżkę i używając identycznych ustawień anotacji.

## 3. Wyszukiwanie

### Zapytanie w języku CQL
Wyszukiwanie odbywa się przy pomocy języka CQL (*Corpus Query Language*).

Przykład zapytania o lemat:
```text
[`base`="wojna"]
```
Zapytanie znajduje wszystkie wystąpienia w korpusie, którym podczas anotacji przypisano lemat „wojna”. Wyniki mogą obejmować różne formy fleksyjne, na przykład „wojna”, „wojny” lub „wojną”.

### Konstruktor zapytań i kontekst
Przed zatwierdzeniem wyszukiwania można ustawić wielkość **Lewego kontekstu** i **Prawego kontekstu**. Definiują one szerokość fragmentu tekstu wyświetlanego wokół znalezionego słowa. Ustawienie to wpływa wyłącznie na prezentację wyników.

Wyszukiwanie uruchamia się klawiszem `Enter` lub przyciskiem w interfejsie. Kombinacja klawiszy `Shift+Enter` wstawia w polu wyszukiwania nową linię.

W celu sformułowania zapytania w języku CQL można skorzystać z pomocy **Konstruktora**. Po wskazaniu w nim odpowiedniego atrybutu, operatora i wartości, system wygeneruje i wstawi gotowy fragment zapytania do pola wyszukiwania.

### Porządkowanie wyników
Kolejność wyświetlania wyników konkordancji można zmienić za pomocą opcji **Sortuj wyniki:**.
Dostępne opcje obejmują sortowanie alfabetyczne, sortowanie po lewym lub prawym kontekście, według metadanych (autor, data) oraz według frekwencji lematu lub formy tekstowej.

---

## 4. Przeglądanie wyników

Wyniki zapytania wyświetlane są jako konkordancje w formacie Key Word in Context (KWIC). W kolejnych kolumnach wyświetlane są: metadane, kontekst lewy, dopasowanie i kontekst prawy.

Kliknięcie danego wiersza spowoduje wyświetlenie szerszego kontekstu w oknie pełnego widoku po prawej stronie. Przyciski znajdujące się pod oknem pełnego widoku umożliwiają wygenerowanie grafu zależności składniowych, a także podświetlenie klastrów koreferencyjnych i jednostek nazwanych (jeśli korpus ma wygenerowane takie warstwy).

Zaznaczony wiersz w tabeli konkordancji lub tekst w oknie pełnego widoku można skopiować przy użyciu skrótu `Ctrl+C`.

---

## 5. Statystyki

Zakładka **Statystyki** służy do analizy słownictwa pojawiającego się w wynikach wyszukiwania. Narzędzie generuje niezależne tabele dla lematów (`base`) i dokładnych form z tekstu (`orth`).

### Interpretacja danych w tabelach
Tabela prezentuje następujące parametry statystyczne:
*  **Liczba wystąpień:** Łączna liczba wystąpień danej pozycji w zbiorze wyników.
*  **Częstość względna:** Liczba wystąpień odniesiona do liczby tokenów w analizowanym materiale. Pozwala porównywać zbiory lub okresy różniące się objętością.
*  **Rozproszenie (DF):** Liczba odrębnych dokumentów, w których występuje dana pozycja.
*  **TF-IDF:** Miara zwiększająca wagę pozycji charakterystycznych dla określonego dokumentu lub okresu, a zmniejszająca wagę pozycji szeroko rozpowszechnionych w analizowanym zbiorze.
*  **Z-score:** Miara wskazująca na stopień odchylenia popularności danego słowa od jego przeciętnego poziomu występowania.

Kliknięcie nagłówka kolumny zmienia kierunek sortowania tabeli.

---

## 6. Wykresy

Wykresy przedstawiają zmiany częstości wybranych form na osi czasu. Wykorzystanie tej funkcji wymaga posiadania w korpusie poprawnie sformatowanych metadanych z datami publikacji.

### Konfiguracja wykresu
Aby przygotować wykres, należy zaznaczyć w statystykach wybrane pozycje, które staną się liniami na wykresie. W **Preferencjach** możliwe jest ustawienie minimalnego progu wystąpień tokenów. Podniesienie tego progu ukrywa najrzadsze formy, co może zwiększyć czytelność analizy głównych trendów.

Dostępne parametry konfiguracyjne to:
*  **Wybierz typ wykresu:**
  * *Liczba wystąpień* (przydatna, gdy objętość materiału w porównywanych okresach jest podobna).
  * *Częstość względna* (zalecana przy porównywaniu okresów o różnej ilości tekstu).
  * *TF-IDF* lub *Z-score*.
*  **Interwał:** Grupowanie danych po dniach, miesiącach lub latach. Należy dobrać interwał odpowiednio do gęstości danych, by uniknąć fragmentacji wykresu.
*  **Zakres dat:** Umożliwia zdefiniowanie analizowanego okresu (opcja **Niestandardowy zakres dat**).
*  **Skalowanie osi Y:** Tryb automatyczny lub ręczny. Tryb ręczny wymaga podania wartości w polu **Górny limit** i sprawdza się w przypadku porównywania kilku osobnych wykresów w równej skali.
*  **Zaznaczone elementy:** Pozwalają wybrać słowa lub frazy, dla których zostanie wygenerowany wykres. Aby dla kilku pozycji utworzyć jedną serię, należy wybrać je z listy, wpisać ich wspólną nazwę w polu **Nowa nazwa dla zaznaczonych** i wybrać **Grupuj/Zmień nazwę**.


Zmiany ustawień zatwierdza się przyciskiem **Odśwież wykres**. Wykres można zapisać na dysku jako plik graficzny poleceniem **Zapisz wykres**.

---

## 7. Kolokacje

Analiza kolokacji pomaga znaleźć słowa często współwystępujące z badanym wyrażeniem.

### Rodzaje kontekstu
*  **Kontekst liniowy:** Analizuje wyrazy znajdujące się w określonej odległości od trafienia. Wykorzystuje się tu parametry **L-span** i **R-span**. Zaznaczenie **Ogranicz do zdań** zapobiega łączeniu wyrazów przekraczających granice zdania.
*  **Kontekst składniowy:** Analizuje relacje składniowe niezależnie od liniowej odległości między wyrazami (wymaga odpowiednich warstw w korpusie). Można w tym miejscu określić **Kierunek** relacji (nadrzędnik, podrzędnik) oraz zastosować filtry części mowy.

### Filtry i miary
*  **Forma kolokatu:** Analiza może obejmować lematy (`base`) lub dokładne formy (`orth`).
*  **Progi minimalne:** Wymagane jest ustalenie progu minimalnych współwystąpień (`Min f`) oraz minimalnego rozproszenia w dokumentach (`Min r`).
*  **Miary asocjacyjne:** Do oceny siły przyciągania słów dostępne są miary takie jak **Log-Dice**, **MI Score**, **T-score** oraz **Log-Likelihood**.

Po ustawieniu parametrów należy wybrać przycisk **Oblicz**. Tabela wyników umożliwia interaktywne przejście do tabeli konkordancji przez kliknięcie w menu rozwijanym (otwieranym przez kliknięcie prawym przyciskiem myszy) opcji **Wyszukaj kolokację**.

---

## 8. Profil kolokacyjny

Profil kolokacyjny ułatwia analizę kolokacji danego wyrazu, kategoryzując powiązania wyrazowe ze względu na pełnione funkcje składniowe. W efekcie zostaje utworzony uporządkowany zestaw typowych nadrzędników i podrzędników przypisanych do badanej formy.

Generowanie profilu:
1. Należy wykonać wyszukiwanie główne.
2. Wskazać słowo stanowiące podstawę analizy w polu **Słowo centralne (węzeł)**. W przypadku zapytań złożonych z wielu słów konieczne jest precyzyjne określenie, który token stanowi podstawę profilu (np. **Token 1**).
3. Podać **Minimalną frekwencję (`Min f`)** (usuwa z profilu połączenia występujące rzadziej niż wskazany próg).
4. Ustawić filtry wielkości liter i kategorie składniowe.
5. Kliknąć **Generuj**.

Podsumowanie profilu dzieli połączenia na grupy odpowiadające relacjom składniowym. W każdej grupie aplikacja przedstawia najlepiej ocenione pozycje, między innymi ranking Top 5 (wg Log-Dice), oraz umożliwia otwarcie pełnej listy.

---

## 9. Eksploracja i analiza otoczenia semantycznego

Moduł semantyczny służy do eksploracyjnego badania słownictwa funkcjonującego w korpusie w kontekstach podobnych do kontekstów wybranego lematu. Nie zastępuje konkordancji, kolokacji ani jakościowej interpretacji tekstów. Pomaga odnaleźć jednostki, które mogą posłużyć do formułowania dalszych pytań i precyzyjniejszych zapytań korpusowych.

### Model dystrybucyjny

Dla lematów tworzony jest model FastText wytrenowany na wybranym korpusie. Lematy występujące w podobnych kontekstach otrzymują zbliżone reprezentacje wektorowe, a ich podobieństwo jest mierzone cosinusem.

FastText wykorzystuje również znakowe fragmenty wyrazów. Bliskość wektorowa może więc częściowo wynikać z podobieństwa formalnego, zwłaszcza w wypadku jednej rodziny słowotwórczej albo błędnych wariantów lematyzacyjnych. Podobieństwo dystrybucyjne nie określa automatycznie rodzaju relacji językowej.

### Graf eksploracyjny

Graf eksploracyjny wspiera analizę kierowaną przez użytkownika:

1. użytkownik wskazuje lemat początkowy;
2. program proponuje dystrybucyjnie podobne lematy;
3. użytkownik wybiera węzeł, którego otoczenie chce rozwinąć;
4. program dobiera kolejne jednostki z uwzględnieniem rozwijanego węzła, lematu początkowego i dotychczas zbudowanej części grafu.

W tabeli kandydatów należy rozróżnić bezpośrednie podobieństwo do rozwijanego lematu, wynik służący do uporządkowania kandydatów, ogólność wynikającą z rozpowszechnienia na listach sąsiedztwa oraz frekwencję korpusową. Ogólność i frekwencja nie są tą samą miarą.

### Raport analityczny

Raport bada szerszy zbiór dostępnych sąsiadów badanego lematu. W aktualnym wariancie:

1. wykorzystywana jest pełna lista sąsiadów dostępna w artefakcie modelu;
2. budowany jest ważony graf wzajemnych pięciu najbliższych sąsiadów, czyli mutual 5-NN;
3. dwa lematy są połączone, jeżeli każdy należy do pięciu najbliższych sąsiadów drugiego;
4. wagą krawędzi jest podobieństwo cosinusowe;
5. graf jest dzielony przez Chinese Whispers;
6. prezentowane są klastry mające co najmniej trzy lematy.

Wartość `k=5` przyjęto na podstawie testów granulacji jako wariant zachowujący szczegółowe struktury lokalne. Większe wartości `k` zmniejszały liczbę izolatów, ale łączyły też część czytelnych grup i obniżały ich średnią zwartość.

Chinese Whispers używa seedu referencyjnego `42`, maksymalnie `100` iteracji i zatrzymuje się wcześniej po pełnej iteracji bez zmian etykiet. Seed umożliwia odtworzenie przebiegu referencyjnego przy tym samym grafie i implementacji, ale sam nie jest oceną stabilności względem innych seedów.

### Status grup

Algorytm wyodrębnia lokalne grupy podobieństwa dystrybucyjnego. Określenie **rama** ma w raporcie charakter operacyjny. Nie oznacza, że każda grupa automatycznie odpowiada ramie semantycznej w sensie określonej teorii językoznawczej.

Raport rozróżnia:

- **członka klastra** (`graph_cluster`), należącego do klastra Chinese Whispers;
- **lemat powiązany z ramą** (`frame_relation`), który nie należy do prezentowanego klastra, lecz jest opisany przez podobieństwo do dwóch najbliższych centroidów.

Lematy `frame_relation` nie są automatycznie dołączane do ram, nie zmieniają ich składu i nie wpływają na centroidy.

### Miary

- **Typowość ramowa**: podobieństwo lematu do centroidu ramy odniesienia.
- **Swoistość ramowa**: różnica między podobieństwem do ramy odniesienia i największym podobieństwem do innej ramy.
- **Centralność pola**: podobieństwo lematu do centroidu całego analizowanego pola.
- **Specyficzność pola**: centralność pola pomniejszona odpowiednio do ogólności lematu.

Raport nie oblicza jednego agregowanego indeksu nośności. Poszczególne miary należy interpretować osobno.

### Relacje między grupami

Raport porównuje centroidy grup. Pozwala to rozpoznawać szersze obszary organizacji otoczenia dystrybucyjnego bez automatycznego scalania grup i bez budowania formalnej hierarchii ram.

### Ograniczenia i dalsza analiza

Wynik zależy od korpusu, frekwencji, lematyzacji, konfiguracji FastText, subwordów, pojemności listy sąsiadów, parametru `k`, minimalnego rozmiaru klastra i przebiegu grupowania. Opisuje konkretny korpus i model, a nie kompletny system semantyczny języka.

Zalecany przebieg pracy:

```text
graf lub raport analityczny
→ wybór lematów i grup
→ hipoteza
→ zapytania CQL
→ konkordancje i kolokacje
→ interpretacja użyć
```

## 10. Modelowanie tematyczne (BERTopic)

Korpusuj wykorzystuje bibliotekę [BERTopic](https://maartengr.github.io/BERTopic/index.html). Oficjalna dokumentacja opisuje pełne możliwości biblioteki; nie wszystkie z nich są dostępne bezpośrednio w interfejsie Korpusuj.

Modelowanie tematyczne (BERTopic) umożliwia wyszukiwanie ukrytych wzorców i grupowanie dokumentów według rozpoznanych automatycznie tematów. Wynikiem analizy jest raport HTML prezentujący wizualizację klastrów i słowa kluczowe opisujące domeny tekstu.

Pierwsze użycie BERTopic może pobrać model Sentence Transformers. Model oraz cache są przechowywane w efektywnym katalogu modeli, a nie w katalogu programu. Cache nie jest automatycznie usuwany po analizie.

Tryby wyznaczania liczby tematów obejmują tryb domyślny bez narzuconego limitu, automatyczną redukcję zbliżonych tematów oraz ręczne podanie docelowej liczby tematów.

Dostępne opcje przygotowania tekstu obejmują:
*  Używanie form zlematyzowanych (podstawowych).
*  Filtrowanie polskich słów funkcyjnych (stop-words).
*  Mechanizm różnorodności słów (ogranicza powtarzanie bardzo podobnych wyrazów w opisie tematu, dzięki czemu etykieta może obejmować szerszy zakres charakterystycznego słownictwa).

Polecenie **Rozpocznij analizę** inicjuje złożone obliczenia. W trakcie analizy wyświetlane są komunikaty o postępie operacji. Temat o identyfikatorze `-1` obejmuje dokumenty lub fragmenty, których model nie przypisał do żadnego wyodrębnionego tematu. Są one traktowane jako szum.

Jeżeli pojawi się komunikat **Brak tematów**, warto użyć większego zbioru, zmienić tryb ustalania liczby tematów, wypróbować automatyczną redukcję oraz porównać wynik z włączoną i wyłączoną lematyzacją lub filtracją stop-words.

---

## 11. Eksport i podkorpus

Aplikacja umożliwia eksportowanie uzyskanych danych.

Polecenie **Eksportuj wyniki** pozwala zapisać bieżących konkordancji i zestawień tabelarycznych w zewnętrznym pliku.

Istnieje możliwość ograniczenia przeszukiwanego materiału poprzez utworzenie podkorpusu:
*  **Utwórz podkorpus z wyników** – pozwala utworzyć mniejszego korpusu ograniczonego do bieżących wyników wyszukiwania.
*  **Utwórz podkorpus po metadanych** – pozwala wybrać dokumenty na podstawie dostępnych pól, na przykład daty, autora, tytułu, gatunku lub innych metadanych zapisanych w korpusie.

---

## 12. Informacje, preferencje, pomoc i skróty

**Informacje o korpusie**
Moduł pokazuje podstawowe informacje o zawartości korpusu, między innymi liczbę dokumentów i tokenów, liczbę unikatowych lematów i form tekstowych, zakres dat publikacji oraz dostępne pola metadanych.

**Preferencje**
Sekcja ta służy do zmiany konfiguracji aplikacji. Można w niej dostosować motyw, czcionki, styl wykresów, zachowanie pamięci RAM oraz minimalny próg pozycji uwzględnianych na wykresach. Opcje potwierdza się przyciskiem zapisu, a powrót do ustawień podstawowych realizowany jest poleceniem **Domyślne**.

**Pomoc i Skróty**
W menu Pomoc dostępne są dokumenty: **Instrukcja użytkownika** i **Przewodnik po języku zapytań**.
Istotne skróty klawiaturowe to:
*  `Enter` — wykonanie zapytania,
*  `Shift+Enter` — wymuszenie nowej linii,
*  `Ctrl+Z` — cofnij,
*  `Ctrl+Y` — ponów,
*  `Ctrl+C` — kopiuj.

---

## 13. Rozwiązywanie problemów

- **Nieudane lub niekompletne pobranie Stanza:** Zamknij aplikację, usuń wyłącznie `<models_dir>\stanza` i ponów pobieranie. Zachowaj `.huggingface`, jeśli chcesz zachować pobrany XLM-RoBERTa i możliwość ponownego wykorzystania cache'u.
- **Brak miejsca podczas pobierania modeli:** Zwolnij miejsce na dysku zawierającym `models_dir`; koreferencja może wymagać kilku dodatkowych gigabajtów.

*  **Długi czas wczytywania korpusu po otwarciu:** Podczas pierwszego otwarcia dużego korpusu tworzone są indeksy pomocnicze; proces ten należy doprowadzić do końca.
*  **Brak wyników wyszukiwania (0 trafień):** Należy sprawdzić składnię, wybrany atrybut, pisownię wartości, filtry metadanych oraz to, czy korpus zawiera wymaganą warstwę anotacji.
*  **Puste tabele kolokacji:** Wymagane jest wcześniejsze wykonanie wyszukiwania. Należy sprawdzić progi `Min f` oraz `Min r` (zbyt rygorystyczne wykluczają wyniki).
*  **Błąd ustalenia lematu przez profil kolokacyjny:** Przy złożonych zapytaniach należy upewnić się, że w polu **Słowo centralne (węzeł)** został wybrany odpowiedni token.
*  **Pusty wykres:** Do utworzenia wykresu wymagana jest obecność dat w metadanych dołączonych do korpusu
*  **Błąd budowania sieci lub raportu semantycznego:** Brak wyniku może być spowodowany niewystarczającym rozmiarem korpusu. Wszelkie błędy działania aplikacji zapisywane są na bieżąco w pliku `%LOCALAPPDATA%\Korpusuj\logs\gui\korpusuj.log`.

- **Nieudane lub niekompletne pobranie Stanza:** zamknij program, usuń wyłącznie `<katalog modeli>\stanza`, uruchom program ponownie i ponów pobieranie.
- **Brak miejsca podczas pobierania modeli:** zwolnij miejsce na dysku zawierającym katalog modeli. Koreferencja może wymagać kilku dodatkowych gigabajtów.

Szczegółowe informacje o błędach interfejsu graficznego są zapisywane w:

```text
%LOCALAPPDATA%\Korpusuj\logs\gui\korpusuj.log
```

Katalog logów można otworzyć w Eksploratorze, wpisując:

```text
%LOCALAPPDATA%\Korpusuj\logs\gui
```

albo w PowerShellu:

```powershell
explorer "$env:LOCALAPPDATA\Korpusuj\logs\gui"
```

#### Korekta lematyzacji gotowego korpusu

Polecenie **Plik → Popraw lematyzację aktywnego korpusu** analizuje lematy otwartego korpusu z użyciem Morfeusza 2 i danych SGJP. Program najpierw przygotowuje audyt i bezpieczną pulę reguł, a dopiero po zatwierdzeniu zapisuje oddzielny plik Parquet. Plik źródłowy nie jest nadpisywany.

Mechanizm poprawia lematy bez ponownej tokenizacji i bez zmiany tagów, zależności, NER ani koreferencji. Blokuje formy homograficzne i przypadki, w których SGJP nie wskazuje jednoznacznego celu. Po utworzeniu wyniku należy pracować na nowym korpusie i jego nowych indeksach. Szczegółowy opis kryteriów, artefaktów i walidacji znajduje się w [dokumentacji korekty lematyzacji](lemma-repair.md).
