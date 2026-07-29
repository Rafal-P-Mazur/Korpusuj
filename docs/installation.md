# Instalacja i uruchamianie

## Wybór wersji programu

### Wersja CPU

Wersja CPU korzysta z procesora głównego komputera. Nie wymaga karty NVIDIA i działa na większości komputerów z systemem Windows.

Wersję CPU należy wybrać, jeżeli:

- komputer nie ma zgodnej karty NVIDIA;
- akceleracja GPU nie jest potrzebna;
- preferowana jest prostsza instalacja;
- program ma działać w wersji portable.

Wersja CPU jest dostępna jako instalator oraz archiwum portable.

### Wersja GPU

Wersja GPU może wykorzystywać zgodną kartę NVIDIA do przyspieszania obsługiwanych modeli i obliczeń PyTorch.

Wersję GPU należy wybrać, jeżeli:

- komputer ma zgodną kartę NVIDIA;
- zainstalowany jest odpowiednio nowy sterownik;
- użytkownik chce korzystać z akceleracji GPU;
- podczas instalacji dostępne jest połączenie z Internetem.

Instalator wersji GPU pobiera około 2,42 GiB oficjalnych komponentów PyTorch dla CUDA 12.6.

Brak zgodnej karty NVIDIA nie uniemożliwia korzystania z programu Korpusuj — w takim przypadku należy użyć wersji CPU.

## Uruchamianie aplikacji

### Instalacyjna wersja CPU lub GPU

Uruchom instalator, wybierz katalog programu i katalog modeli, a po zakończeniu otwórz program Korpusuj ze skrótu albo przez plik `Korpusuj.exe`.

### Wersja CPU portable

Rozpakuj archiwum ZIP i uruchom znajdujący się w nim plik `Korpusuj.exe`. Nie przenoś samego pliku EXE bez pozostałej zawartości katalogu.

## Katalog modeli językowych

Modele Stanza, spaCy, EasyOCR, Sentence Transformers i inne modele używane przez program Korpusuj nie są dołączone do instalatora. Są pobierane przy pierwszym użyciu odpowiedniej funkcji.

W wersji instalacyjnej katalog modeli wybiera się podczas instalacji. Można wskazać inny dysk, co jest zalecane, jeśli na dysku systemowym jest mało wolnego miejsca.

Pozostałe warianty używają następujących lokalizacji:

```text
wersja CPU portable    <katalog Korpusuj.exe>\models
wersja źródłowa        <root projektu>\models
```

W katalogu modeli program przechowuje również cache pobranych modeli. Cache nie jest usuwany po zamknięciu programu, dzięki czemu tych samych danych nie trzeba pobierać ponownie.

### Wielkość pobieranych modeli

Orientacyjne wielkości pierwszego pobrania:

- podstawowy polski pakiet Stanza: około 413 MB;
- adapter polskiej koreferencji Stanza: około 132 MB;
- model XLM-RoBERTa Large wymagany przez koreferencję: około 2,24 GB;
- modele spaCy, EasyOCR i Sentence Transformers: zależnie od wybranego modelu.

Plik `model.safetensors` o wielkości około 2,24 GB jest modelem bazowym XLM-RoBERTa Large używanym przez koreferencję Stanza. Jego pobieranie jest prawidłowym zachowaniem.

Przed włączeniem koreferencji należy zapewnić co najmniej 4–5 GB wolnego miejsca na dysku zawierającym katalog modeli.

### Niekompletne pobranie Stanza

Jeżeli pobieranie Stanza zostało przerwane, a model nie daje się później załadować:

1. zamknij program Korpusuj;
2. usuń katalog `<katalog modeli>\stanza`;
3. uruchom program ponownie;
4. ponów pobieranie modelu.

Nie usuwaj całego katalogu modeli, jeśli chcesz zachować modele innych bibliotek.

### Informacje techniczne o cache

Cache bibliotek Hugging Face, Transformers, Sentence Transformers i Torch znajduje się wewnątrz wybranego katalogu modeli, między innymi w podkatalogach `.huggingface`, `sentence-transformers` i `torch`.

## Uruchamianie ze źródeł

Poniższe instrukcje dotyczą pracy ze źródłami, a nie gotowej wersji użytkowej. Projekt wymaga systemu Windows i Pythona 3.11. W głównym katalogu repozytorium wykonaj:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Korpusuj.py
```

Jeżeli PowerShell blokuje aktywowanie środowiska, można zmienić zasady tylko dla bieżącej sesji:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Polecenia CLI przeznaczone dla uruchomienia ze źródeł:

```powershell
python -m korpusuj.corpus.creator_cli --help
python -m korpusuj.index.cli --help
python -m korpusuj.search.cli --help
```

Szczegółowy opis znajduje się w [instrukcji CLI](cli.md).

## Deinstalacja i zachowanie modeli

Deinstalator usuwa katalog programu wraz z plikami zainstalowanymi lub pobranymi podczas instalacji.

Podczas deinstalacji można osobno zdecydować:

- czy usunąć konfigurację, logi i pliki tymczasowe programu;
- czy usunąć zewnętrzny katalog modeli.

Domyślnie katalog modeli pozostaje na dysku. Pozwala to użyć pobranych modeli po ponownej instalacji programu bez pobierania ich od początku.

Usunięcie katalogu modeli usuwa również cache Hugging Face, modele Stanza, spaCy, EasyOCR, Sentence Transformers oraz inne dane modelowe przechowywane w tym katalogu.

## Sprawdzenie

W wersji użytkowej otwórz `Korpusuj.exe` i sprawdź, czy pojawia się główne okno. Podczas pracy ze źródłami uruchom `python Korpusuj.py`. Osoby rozwijające aplikację mogą dodatkowo wykonać:

```powershell
python -m pytest -q -p no:cacheprovider tests
```

Ostrzeżenia bibliotek zewnętrznych nie zawsze oznaczają niepowodzenie; o wyniku testów decyduje końcowy komunikat programu pytest.
