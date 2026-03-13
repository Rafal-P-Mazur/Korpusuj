# Korpusuj

Aplikacja do przetwarzania języka naturalnego (NLP). Umożliwia tworzenie automatycznie anotowanych korpusów na podstawie dokumentów TXT, DOCX, XLSX i PDF, przeszukiwanie utworzonego korpusu z wykorzystaniem autorskiego języka CQL (Corpus Query Language) oraz wizualizację danych frekwencyjnych.

## ⚠️ Status projektu i wydajność
Projekt ten jest autorskim narzędziem badawczym stworzonym przez językoznawcę przy wsparciu AI. Nie jest to profesjonalny produkt komercyjny.

* **Faza Beta:** Aplikacja działa stabilnie w środowisku testowym autora. Jako że konfiguracje sprzętowe użytkowników mogą się różnić, mogą wystąpić nieprzewidziane błędy. Zachęcam do zgłaszania ich w zakładce [Issues](https://github.com/Rafal-P-Mazur/Korpusuj/issues).
* **Kod źródłowy:** Kod źródłowy odzwierciedla proces uczenia się i eksperymentowania z AI, stąd może nie spełniać wszystkich standardów inżynierii oprogramowania. Kod jest otwarty i dostępny dla każdego, kto chciałby go poprawiać lub rozwijać
* **Wydajność (CPU vs GPU):**
    * **Wersja .exe:** Ze względu na rozmiar pliku, gotowa aplikacja korzysta z biblioteki Torch w wersji **CPU-only**. Przetwarzanie dużych korpusów może zająć sporo czasu.
    * **Wersja Python:** Użytkownicy uruchamiający kod ze źródeł mogą zainstalować wersję Torch z obsługą **CUDA (GPU)**. Stanza wykorzysta wtedy kartę graficzną, co wielokrotnie przyspieszy proces anotacji (tworzenia korpusu).

![Tag](https://img.shields.io/github/v/tag/Rafal-P-Mazur/Korpusuj)
---

![image alt](https://github.com/Rafal-P-Mazur/Korpusuj/blob/main/Images/1.png?raw=true)
![image alt](https://github.com/Rafal-P-Mazur/Korpusuj/blob/main/Images/2.png?raw=true)
![image alt](https://github.com/Rafal-P-Mazur/Korpusuj/blob/main/Images/3.png?raw=true)

## Główne funkcje
- **Budowa korpusów:** Konwersja surowych plików tekstowych do formatu `.parquet` zawierającego pełną anotację morfosyntaktyczną.
- **Wyszukiwanie:** Zaawansowany silnik zapytań obsługujący warunki zagnieżdżone, relacje zależnościowe (dependency parsing) i wyrażenia regularne.
- **Analiza:** Generowanie tabel frekwencyjnych dla lematów i form ortograficznych.
- **Wizualizacja:** Interaktywne wykresy trendów czasowych (diachroniczne).
- **Dostępność:** Gotowy plik `.exe` dla systemu Windows (nie wymaga instalacji Pythona).
---

## Instalacja

### Windows (.exe)
1. Pobierz najnowszą wersję z [Releases](https://github.com/Rafal-P-Mazur/Korpusuj/releases).
2. Rozpakuj archiwum ZIP.
3. Uruchom `Korpusuj.exe` klikając dwukrotnie.

⚠️ Plik nie jest podpisany certyfikatem Windows – system może wyświetlić ostrzeżenie bezpieczeństwa. Jest to normalne.

ℹ️ Plik `.exe` zawiera wersję biblioteki **Torch** skonfigurowaną do pracy **tylko na CPU**.  

---

### Python
1. Zainstaluj Python 3.10.
2. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/Rafal-P-Mazur/Korpusuj.git
3. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt

   ⚠️ Aby podczas tworzenia korpusu wykorzystywać GPU, należy pobrać zgodną z kartą graficzną wersję Torch

4. Uruchom aplikację:
   ```bash
   python Korpusuj_beta.py

---

## Instrukcja użytkowania

### Krok 1: Tworzenie korpusu
1.  W menu głównym należy wybrać: **Plik -> Utwórz korpus**.
2.  **Wybór plików:**
    * **Pliki tekstowe:** Obsługiwane formaty to `.txt`, `.docx` oraz `.xlsx` (w przypadku plików Excel wymagane są kolumny: *Tytuł, Treść, Data publikacji*).
    * **PDF:** Aplikacja posiada wbudowany moduł **OCR**. W przypadku plików PDF będących skanami, tekst zostanie rozpoznany automatycznie (może to wpłynąć na czas przetwarzania).
    * **Archiwum ZIP:** Możliwe jest wczytywanie plików tekstowych spakowanych w archiwa `.zip`. Aplikacja automatycznie rozpakuje archiwum i przetworzy zawarte w nim pliki.
3.  **Wybór modelu językowego:**
    * **Stanza:** Zapewnia wyższą precyzję analizy składniowej (zalecane).
    * **spaCy:** Oferuje szybsze przetwarzanie danych.
4.  **Metadane (Opcjonalne):**
    * Istnieje możliwość załączenia pliku `metadane.xlsx` w celu automatycznego przypisania informacji bibliograficznych.
    * **Wymagane kolumny:** Arkusz musi zawierać kolumnę **"Nazwa pliku"** (odpowiadającą nazwie pliku źródłowego, np. `tekst1.txt`) oraz kolumny podstawowe: **"Tytuł"**, **"Data publikacji"**, **"Autor"**.
    * **Format daty:** Zaleca się, aby kolumna "Data publikacji" miała w Excelu ustawiony format **Data**. Aplikacja automatycznie rozpozna i sformatuje tak wprowadzone daty, co zapewni poprawne działanie wykresów trendów.
    * **Własne pola:** Użytkownik może dodać dowolne inne kolumny (np. "Gatunek", "Źródło", "Portal"), które zostaną zaimportowane i umożliwią później filtrowanie wyników w wyszukiwarce.
5.  **Przetwarzanie plików:**
    * Proces uruchamia się przyciskiem **Przetwórz pliki**.
    * > **Ważne:** Przy **pierwszym użyciu** kreatora aplikacja automatycznie pobierze niezbędne modele językowe (Stanza lub SpaCy) dla języka polskiego. Czas operacji zależy od szybkości łącza internetowego. Kolejne uruchomienia nie wymagają ponownego pobierania.
6.  Wynikowy plik `.parquet` należy zapisać na dysku.

### Krok 2: Wczytanie projektu
1.  Należy powrócić do głównego okna aplikacji.
2.  Wybrać opcję **Plik -> Nowy projekt**.
3.  Wskazać utworzony wcześniej plik `.parquet`.
4.  Aktywny korpus wybiera się z listy rozwijanej w lewym górnym rogu ("Wybierz korpus").

### Krok 3: Wyszukiwanie
Zapytania wprowadza się w głównym polu tekstowym. Aplikacja obsługuje autorski język zapytań **CQL** z obsługą wyrażeń regularnych.

* **Przykłady zapytań:**
    * `[base="dom"]` – wyszukiwanie wszystkich form fleksyjnych słowa "dom".
    * `[pos="adj"][base="chmura"]` – wyszukiwanie przymiotnika stojącego bezpośrednio przed słowem "chmura".
    * `[pos="subst" & dependent={base="piękny"}]` – wyszukiwanie rzeczownika określanego przez przymiotnik "piękny" (niezależnie od szyku w zdaniu).
* **Pomoc:** Pełna dokumentacja składni dostępna jest w menu **Pomoc -> Przewodnik po języku zapytań**.

### Krok 4: Analiza wyników
Wyniki prezentowane są w trzech zakładkach:

#### 1. Wyniki wyszukiwania (Konkordancje)
* W lewym panelu znajduje się tabela, w której wyświetlane są wyniki wyszukiwania wraz z lewym i prawym kontekstem.
* Kliknięcie w wiersz powoduje wyświetlenie **pełnego tekstu** artykułu w pprawym panelu.
* Znalezione frazy są podświetlone kolorami w celu łatwiejszej identyfikacji.

#### 2. Statystyki
Zakładka zawiera tabele frekwencyjne generowane automatycznie dla wyników wyszukiwania. Dostępne są następujące widoki:
* **Formy podstawowe:** Ranking lematów (form podstawowych).
* **Formy ortograficzne:** Ranking dokładnych form tekstowych.
* **Częstość w czasie:** Liczba wystąpień w poszczególnych miesiącach.

#### 3. Trendy (Wykresy)
Wizualizacja zmienności użycia słów w czasie.
* **Tryby:** Miesięczny/Roczny oraz Surowy/Znormalizowany (liczba wystąpień na milion słów).
* **Legenda:** Elementy dodaje się do wykresu poprzez zaznaczenie ich na liście po lewej stronie.
* **Edycja:** Pole edycji zapewnia możliwość łączenia synonimów w jedną linię na wykresie (np. sumowanie słów "auto" i "samochód").

### Krok 5: Narzędzia dodatkowe

* **Fiszki:** Aby zapisać fragment tekstu, należy zaznaczyć go w podglądzie pełnego tekstu, wpisać nazwę w polu "Nazwa fiszki" i kliknąć **Zapisz fiszkę**. Fragment zostanie zapisany w pliku tekstowym w folderze `/fiszki`.
* **Eksport:** Opcja **Plik -> Eksportuj wyniki** umożliwia zapisanie wszystkich tabel i konkordancji do jednego pliku Excel (`.xlsx`).

---

## ⚙️ Konfiguracja
W menu **Ustawienia -> Preferencje** możliwa jest modyfikacja następujących parametrów:
* **Motyw:** Ciemny / Jasny.
* **Czcionka:** Rozmiar i krój czcionki.
* **Kontekst:** Domyślna liczba słów wyświetlana w wynikach wyszukiwania.

## 📂 Struktura plików
* `Korpusuj.exe` – Plik wykonywalny aplikacji.
* `models/` – Katalog z lokalnymi modelami NLP (Stanza/SpaCy).
* `korpusuj.log` – Plik dziennika błędów i operacji.
* `temp/` – Pliki tymczasowe generowane podczas sesji.
* `stanza_resources/` (w katalogu użytkownika) – Pobrane modele językowe.

## 📜 Licencja
Projekt udostępniony na licencji MIT.
