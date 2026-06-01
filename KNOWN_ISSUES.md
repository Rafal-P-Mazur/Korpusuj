# Korpusuj — znane ograniczenia i kwestie do dalszego rozwoju

Ten dokument zawiera listę znanych ograniczeń programu **Korpusuj** oraz obszarów, które wymagają dalszych testów, optymalizacji lub rozwoju. Lista nie oznacza błędów uniemożliwiających korzystanie z programu, lecz wskazuje kwestie istotne dla interpretacji wyników, stabilności pracy i planowania badań korpusowych.

## 1. Status programu

Korpusuj jest rozwijanym jednoosobowo narzędziem badawczym. Należy traktować go jako zaawansowany prototyp przeznaczony do pracy naukowej i eksploracyjnej, a nie jako komercyjny system produkcyjny o gwarantowanej stabilności we wszystkich środowiskach sprzętowych i systemowych.

**Konsekwencje praktyczne:**

- program może wymagać dostosowania środowiska uruchomieniowego, zwłaszcza w wariancie źródłowym;
- część funkcji może zachowywać się odmiennie w zależności od konfiguracji systemu, wersji bibliotek Python i dostępności modeli NLP;
- przed użyciem programu w większym projekcie badawczym zaleca się wykonanie prób na reprezentatywnej próbce materiału.

## 2. Zależność od jakości automatycznej anotacji

Korpusuj integruje wyniki generowane przez zewnętrzne modele i biblioteki NLP. Program nie jest autorskim modelem lematyzacji, rozpoznawania jednostek nazwanych, analizy zależnościowej ani koreferencji.

**Znane ograniczenia:**

- błędy lematyzacji lub tagowania morfosyntaktycznego mogą wpływać na wyniki zapytań po `base`, `pos`, `upos` i cechach fleksyjnych;
- błędy parsera zależnościowego mogą wpływać na wyniki zapytań wykorzystujących `head`, `dependent`, `deprel` i warunki zagnieżdżone;
- błędy rozpoznawania jednostek nazwanych mogą powodować zarówno wyniki fałszywie pozytywne, jak i fałszywie negatywne w zapytaniach wykorzystujących `ner`;
- automatyczna koreferencja jest szczególnie wrażliwa na długość tekstu, złożoność składniową i niejednoznaczność odniesień.

**Zalecenie:**

Wyniki zapytań wykorzystujących warstwy składniowe, NER i koreferencję należy traktować jako materiał do dalszej kontroli badawczej, a nie jako automatyczne rozstrzygnięcie interpretacyjne.

## 3. Koreferencja

Warstwa koreferencyjna jest jedną z najbardziej eksperymentalnych warstw anotacyjnych używanych przez program.

**Możliwe problemy:**

- niepełne klastry koreferencyjne;
- błędne łączenie wzmianek odnoszących się do różnych referentów;
- pomijanie zaimków, deskrypcji lub nazw własnych;
- obniżona skuteczność w bardzo długich dokumentach;
- problemy z wielowyrazowymi wzmiankami i ustalaniem ich elementu nadrzędnego.

**Zalecenie:**

Zapytania wykorzystujące `coref`, `coref(H)`, `coref(P)` i `coref(M)` powinny być traktowane jako narzędzia eksploracyjne. W zastosowaniach interpretacyjnych zalecana jest ręczna kontrola wyników.

## 4. Skalowalność i zużycie pamięci

Korpusuj jest aplikacją lokalną. Przetwarzanie i wyszukiwanie odbywa się na komputerze użytkownika, a część operacji wykonywana jest w pamięci operacyjnej.

**Znane ograniczenia:**

- bardzo duże korpusy mogą wymagać znacznej ilości pamięci RAM;
- czas ładowania korpusu zależy od rozmiaru pliku Parquet, liczby dokumentów, liczby tokenów i liczby warstw anotacji;
- zapytania generujące bardzo szeroki zbiór kandydatów mogą wykonywać się dłużej nawet wtedy, gdy liczba wyników końcowych nie jest bardzo duża;
- tworzenie korpusu, zwłaszcza z pełną anotacją składniową, NER i koreferencją, może być czasochłonne.

**Zalecenie:**

Przy pracy z dużymi korpusami warto najpierw przetestować działanie programu na mniejszym podzbiorze danych oraz kontrolować zużycie pamięci RAM.

## 5. Wydajność zapytań

Czas wykonania zapytania zależy nie tylko od liczby wyników, lecz także od selektywności warunków początkowych i liczby kandydatów wymagających dalszej weryfikacji.

**Przykładowe czynniki spowalniające zapytania:**

- bardzo ogólne warunki, np. zapytania rozpoczynające się od szerokich klas części mowy;
- zapytania wielosegmentowe z dużymi zakresami dystansu;
- zapytania z wieloma warunkami zagnieżdżonymi;
- zapytania odwołujące się do relacji składniowych w dużych zbiorach danych;
- zapytania wykorzystujące koreferencję.

**Zalecenie:**

Jeśli to możliwe, warto formułować zapytania z selektywną kotwicą leksykalną, np. łączyć warunki morfosyntaktyczne z `base` lub `orth`.

## 6. Kreator korpusów

Kreator korpusów obsługuje wiele formatów wejściowych, jednak jakość wynikowego korpusu zależy od jakości i struktury danych wejściowych.

**Możliwe problemy:**

- dokumenty PDF mogą wymagać OCR, którego jakość zależy od jakości skanu;
- pliki DOCX i PDF o niestandardowej strukturze mogą wymagać ręcznej kontroli po imporcie;
- niejednolite metadane mogą wymagać wcześniejszego uporządkowania;
- bardzo długie dokumenty bez wyraźnych granic zdań mogą wymagać dodatkowego dzielenia na fragmenty;
- archiwa ZIP zawierające zagnieżdżone katalogi lub nietypowe kodowanie nazw plików mogą wymagać kontroli po rozpakowaniu.

**Zalecenie:**

Przed zbudowaniem dużego korpusu warto sprawdzić próbkę danych wejściowych i poprawność powiązania dokumentów z metadanymi.

## 7. Metadane

Filtrowanie metadanych opiera się na kolumnach dostępnych w pliku korpusowym. Program zachowuje dodatkowe kolumny zdefiniowane przez użytkownika, ale nie rozstrzyga automatycznie niespójności metadanych.

**Możliwe problemy:**

- różne formaty dat w jednej kolumnie;
- niejednolite zapisy nazw autorów lub źródeł;
- puste wartości w kluczowych polach;
- rozbieżności między nazwami plików w metadanych i rzeczywistymi nazwami dokumentów.

**Zalecenie:**

Przed importem warto ujednolicić nazwy plików, daty, autorów i inne podstawowe pola metadanych.

## 8. Interfejs i środowisko uruchomieniowe

Program wykorzystuje interfejs graficzny oraz liczne biblioteki zewnętrzne. Stabilność działania może zależeć od systemu operacyjnego, wersji Pythona, wersji bibliotek i sposobu instalacji modeli NLP.

**Możliwe problemy i różnice między wariantami uruchomienia:**

- różnice między wersją wykonywalną a uruchomieniem ze źródeł;
- problemy z dostępnością modeli spaCy, Stanza, Herference lub innych komponentów NLP;
- w środowisku Python można korzystać z konfiguracji PyTorch zgodnej z GPU, o ile użytkownik samodzielnie zainstaluje odpowiednią wersję bibliotek i sterowników;
- wersja wykonywalna `.exe` działa wyłącznie w trybie CPU;
- dłuższy czas pierwszego uruchomienia funkcji wymagających załadowania modeli.

**Zalecenie:**

W przypadku uruchamiania wersji źródłowej zalecane jest korzystanie z odizolowanego środowiska Python i instalowanie zależności zgodnie z instrukcją w repozytorium. Jeśli użytkownik chce korzystać z GPU, powinien przygotować środowisko PyTorch zgodne z posiadaną kartą graficzną, wersją CUDA i sterownikami. Wersję `.exe` należy traktować jako wariant CPU-only.

## 9. Testy funkcjonalne

W repozytorium znajdują się lub zostaną udostępnione testy funkcjonalne obejmujące:

- kreator korpusów;
- silnik wyszukiwania;
- syntetyczny korpus testowy o kontrolowanej anotacji;
- wybrane przypadki brzegowe.

Testy te służą sprawdzeniu działania mechanizmów programu względem kontrolowanych danych. Nie stanowią ewaluacji jakości modeli NLP używanych do automatycznej anotacji.

## 10. Planowane kierunki rozwoju

Możliwe kierunki dalszego rozwoju obejmują:

- dalszą optymalizację wydajności zapytań złożonych;
- rozszerzenie zestawu testów automatycznych;
- lepszą dokumentację języka zapytań;
- uzupełnienie listy przykładów zapytań badawczych;
- dalszą stabilizację obsługi dużych korpusów;
- rozwój mechanizmów raportowania błędów i ostrzeżeń;
- rozbudowę dokumentacji ograniczeń poszczególnych warstw anotacyjnych.

## 11. Zgłaszanie problemów

Problemy, błędy i sugestie rozwoju można zgłaszać przez mechanizm Issues w repozytorium projektu.

Przy zgłoszeniu warto podać:

- wersję programu lub datę pobrania kodu;
- system operacyjny;
- sposób uruchomienia programu, np. wersja wykonywalna lub kod źródłowy;
- rozmiar i typ korpusu;
- treść zapytania, jeśli problem dotyczy wyszukiwania;
- krótki opis oczekiwanego i uzyskanego wyniku;
- komunikat błędu lub fragment logu, jeśli jest dostępny;
- plik `korpusuj.log`, jeśli problem dotyczy działania aplikacji, ładowania korpusu, tworzenia korpusu, wyszukiwania lub eksportu danych.
