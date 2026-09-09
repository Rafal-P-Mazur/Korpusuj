# Instalacja i uruchamianie

## Wybór wersji programu

### Windows CPU

Wersja CPU korzysta z procesora głównego. Nie wymaga karty NVIDIA i działa na większości komputerów z systemem Windows.

Wybierz wersję CPU, jeżeli:

- komputer nie ma zgodnej karty NVIDIA;
- akceleracja GPU nie jest potrzebna;
- program ma działać w wersji portable.

Wersja CPU jest dostępna jako instalator oraz archiwum portable.

### Windows GPU

Wersja GPU może wykorzystywać zgodną kartę NVIDIA do przyspieszania obsługiwanych modeli i obliczeń PyTorch. Wymaga odpowiednio nowego sterownika NVIDIA oraz połączenia z Internetem podczas instalacji.

Instalator pobiera około 2,42 GiB oficjalnych komponentów PyTorch dla CUDA 12.6. Jeżeli komputer nie ma zgodnej karty NVIDIA, należy użyć wersji CPU.

### macOS ARM64

Kod źródłowy można uruchamiać na komputerach Mac z procesorami Apple Silicon przy użyciu Pythona 3.11. Obsługiwane modele mogą korzystać z akceleracji MPS. Gotowy pakiet aplikacji dla macOS może nie być dołączony do każdego wydania.

## Uruchamianie gotowej aplikacji

### Instalacyjna wersja CPU lub GPU

Uruchom instalator, wybierz katalog programu i katalog modeli, a po zakończeniu otwórz Korpusuj ze skrótu albo przez plik `Korpusuj.exe`.

Instalator GPU pobiera i weryfikuje komponenty PyTorch przed zakończeniem instalacji. Nie zamykaj instalatora w trakcie tego procesu.

### Wersja CPU portable

Rozpakuj całe archiwum ZIP i uruchom znajdujący się w nim plik `Korpusuj.exe`. Nie przenoś samego pliku EXE bez katalogu `_internal` i pozostałej zawartości dystrybucji.

## Katalog modeli językowych

Modele Stanza, spaCy, EasyOCR, Sentence Transformers i inne modele używane przez Korpusuj nie są dołączone do instalatora. Są pobierane przy pierwszym użyciu odpowiedniej funkcji.

W wersji instalacyjnej katalog modeli wybiera się podczas instalacji. Jeśli na dysku systemowym jest mało wolnego miejsca, można wskazać inny dysk.

Wersja CPU portable używa katalogu:

```text
<katalog Korpusuj.exe>\models
```

Położenie katalogu modeli w wersji uruchamianej ze źródeł zależy od konfiguracji aplikacji i trybu uruchomienia.

Pobrane modele i pamięci podręczne pozostają na dysku po zamknięciu programu, dzięki czemu nie trzeba pobierać ich ponownie.

### Wielkość pobieranych modeli

Orientacyjne wielkości pierwszego pobrania:

- podstawowy polski pakiet Stanza: około 413 MB;
- adapter polskiej koreferencji: około 132 MB;
- model XLM-RoBERTa Large wymagany przez koreferencję: około 2,24 GB;
- modele spaCy, EasyOCR i Sentence Transformers: zależnie od wybranego modelu.

Plik `model.safetensors` o wielkości około 2,24 GB jest modelem bazowym XLM-RoBERTa Large używanym przez koreferencję. Jego pobieranie jest prawidłowym zachowaniem.

Przed włączeniem koreferencji zapewnij co najmniej 4 do 5 GB wolnego miejsca na dysku zawierającym katalog modeli.

### Niekompletne pobranie Stanza

Jeżeli pobieranie modelu Stanza zostało przerwane i model nie daje się później załadować:

1. Zamknij Korpusuj.
2. Usuń katalog `<katalog modeli>\stanza`.
3. Uruchom program ponownie.
4. Ponów pobieranie modelu.

Nie usuwaj całego katalogu modeli, jeśli chcesz zachować dane innych bibliotek.

## Uruchamianie ze źródeł

Poniższe instrukcje dotyczą pracy ze źródłami. Projekt wymaga Pythona 3.11.

### Windows CPU

W głównym katalogu repozytorium wykonaj:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-cpu-py311.txt
python Korpusuj.py
```

Jeżeli PowerShell blokuje aktywowanie środowiska, zmień zasady tylko dla bieżącej sesji:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows GPU

Wariant GPU wymaga zgodnej karty NVIDIA i odpowiednio nowego sterownika:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-gpu-py311.txt
python Korpusuj.py
```

Sprawdzenie środowiska GPU:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Dostępna:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Brak')"
```

### macOS ARM64

Na komputerze Mac z procesorem Apple Silicon wykonaj:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-macos-arm64-py311.txt
python Korpusuj.py
```

Sprawdzenie dostępności MPS:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

### Polecenia CLI

```text
python -m korpusuj.corpus.creator_cli --help
python -m korpusuj.index.cli --help
python -m korpusuj.search.cli --help
```

Szczegółowy opis znajduje się w [instrukcji CLI](cli.md).

## Deinstalacja i zachowanie modeli

Deinstalator usuwa pliki programu. Podczas deinstalacji można osobno zdecydować:

- czy usunąć konfigurację, logi i pliki tymczasowe programu;
- czy usunąć zewnętrzny katalog modeli.

Katalog modeli jest domyślnie zachowywany, aby po ponownej instalacji nie trzeba było pobierać modeli od początku. Jego usunięcie usuwa również przechowywane w nim pamięci podręczne i dane modeli.

## Sprawdzenie

W wersji użytkowej uruchom `Korpusuj.exe` i sprawdź, czy pojawia się główne okno. Podczas pracy ze źródłami uruchom:

```text
python Korpusuj.py
```

Osoby rozwijające aplikację mogą dodatkowo wykonać:

```text
python -m pytest -q -p no:cacheprovider tests
```

Ostrzeżenia bibliotek zewnętrznych nie zawsze oznaczają niepowodzenie. O wyniku testów decyduje końcowy komunikat programu pytest.
