# Tworzenie korpusu

## Wspólny przebieg GUI i CLI

GUI creatora i `korpusuj.corpus.creator_cli` wywołują `run_creator_job`. Parametry są przekazywane w `CreatorRunOptions`, a postęp przez obiekt reportera.

`CreatorRunOptions` zawiera:

- listę wejść;
- ścieżkę wynikowego Parquet;
- wybór Stanza albo spaCy;
- ustawienia NER i koreferencji;
- opcjonalny XLSX metadanych;
- mapowanie kolumn XLSX;
- tryb wznowienia;
- opcjonalny plik korekt lematów;
- katalog modeli.

## Odczyt wejść

- TXT jest odczytywany jako tekst.
- DOCX jest odczytywany z kolejnych akapitów.
- PDF jest najpierw odczytywany z warstwy tekstowej. Strony bez tekstu mogą zostać przekazane do EasyOCR.
- XLSX jest przekształcany zgodnie z mapowaniem kolumn.
- ZIP jest sprawdzany przed rozpakowaniem. Wpisy wychodzące poza katalog docelowy są odrzucane.

## Normalizacja tekstu

Przed analizą językową creator usuwa techniczne znaki Unicode, które nie powinny tworzyć tokenów, między innymi soft hyphen oraz wybrane znaki zero-width i kierunku tekstu. Zwykłe znaki treści nie są zastępowane ani transliterowane.

## Podział długich dokumentów

`creator_chunking.py` dzieli dokument na fragmenty mieszczące się w limicie pipeline'u NLP. Fragment pamięta przesunięcie względem początku dokumentu. Po analizie creator dodaje to przesunięcie do pozycji znakowych i łączy identyfikatory zdań oraz wzmianek.

## Modele NLP

`CreatorModelState` przechowuje załadowane pipeline'y między kolejnymi dokumentami. Dzięki temu Stanza lub spaCy nie są inicjalizowane dla każdego pliku osobno.

Analiza tworzy tokeny, lematy, tagi, cechy morfologiczne i relacje zależnościowe. NER i koreferencja są uruchamiane zgodnie z opcjami zadania.

Jeśli NER jest wyłączony, `ners` zawiera `O` dla każdego tokenu. Jeśli koreferencja jest wyłączona, `corefs` i `coref_mentions` są puste.

## Korekty lematów

Opcjonalny plik JSON zawiera reguły dopasowujące formę tekstową, lemat i UPOS. Reguła zmienia lemat po wykonaniu NLP. Creator zapisuje hash pliku reguł i statystyki zastosowań w `korpus_meta`.

## Zapis częściowy i wznowienie

Dokumenty są zapisywane partiami do plików częściowych. Wznowienie korzysta z istniejących części tylko wtedy, gdy zgadzają się ustawienia warstw anotacji i konfiguracja korekt lematów. Niezgodne części nie są dołączane do nowego wyniku.

## Finalny Parquet

Po przetworzeniu wszystkich dokumentów creator scala części, przelicza metadane zbiorcze i zapisuje finalny Parquet. Budowa `.search` i `.dep_cache` jest osobną operacją.


## `CreatorRunOptions`

`CreatorRunOptions` jest dataclassą z `slots=True` zdefiniowaną w `korpusuj/corpus/creator_core.py`. Konstruktor przyjmuje następujące pola:

```python
CreatorRunOptions(
    input_files: list[str],
    output_parquet_file: str,
    metadata_path: str | None = None,
    model_name: str = "stanza",
    excel_mappings: dict[str, Any] | None = None,
    resume_mode: bool = False,
    processed_set: set[str] | None = None,
    enable_ner: bool = True,
    enable_coreference: bool = True,
    lemma_corrections_path: str | None = None,
)
```

`__post_init__()` zamienia elementy `input_files` oraz ścieżki plików na napisy, kopiuje `excel_mappings` do nowego słownika i `processed_set` do nowego zbioru. Dzięki temu późniejsza zmiana kolekcji przekazanej przez wywołującego nie zmienia opcji już zapisanych w obiekcie.

### Znaczenie pól

- `input_files` zawiera ścieżki plików wybranych do przetworzenia. `run_creator_job()` odrzuca pustą listę przed uruchomieniem właściwego pipeline'u.
- `output_parquet_file` jest ścieżką finalnego Parquet. Pliki częściowe są wyprowadzane przez orkiestrator i nie są podawane w opcjach.
- `metadata_path` wskazuje zewnętrzny XLSX z metadanymi. `None` oznacza brak osobnego arkusza.
- `model_name` wybiera backend. `run_creator_job()` akceptuje po normalizacji wyłącznie `stanza` albo `spacy`.
- `excel_mappings` mapuje pola rozpoznawane przez Korpusuj na nagłówki kolumn arkusza. `None` jest przekazywane dalej jako pusty słownik.
- `resume_mode` zezwala orkiestratorowi na wykorzystanie zgodnych plików częściowych. Samo `True` nie pomija kontroli warstw anotacji i korekt lematów.
- `processed_set` jest deklarowane w dataclassie, ale bieżący `run_creator_job()` nie przekazuje tego pola do `_run_creator_job_impl()`. Pole nie steruje więc aktualnym przebiegiem uruchamianym przez tę funkcję.
- `enable_ner` i `enable_coreference` określają warstwy zapisywane w korpusie i w `korpus_meta.annotation_layers`.
- `lemma_corrections_path` wskazuje plik JSON ładowany przed rozpoczęciem przetwarzania. Błąd tego pliku kończy zadanie przed wywołaniem `_run_creator_job_impl()`.

`CreatorRunOptions` nie deklaruje pola `models_dir`. `run_creator_job()` przyjmuje katalog modeli osobnym argumentem `models_dir`. Wewnętrzna próba odczytu `models_dir` z obiektu opcji obsługuje inne obiekty o podobnym kontrakcie, ale standardowej dataclassy nie można skonstruować z takim argumentem.

## `run_creator_job(...)`

```python
run_creator_job(
    options,
    reporter=None,
    *,
    model_state=None,
    models_dir=None,
    cancel_requested=None,
)
```

Funkcja wykonuje przed właściwym pipeline'em następujące kroki:

1. wybiera przekazany reporter albo `NullProgressReporter`;
2. wybiera przekazany `CreatorModelState` albo tworzy nowy;
3. ustala katalog modeli przez `models_root(...)`;
4. zapisuje ustawienia NER i koreferencji w stanie orkiestratora;
5. ładuje plik korekt lematów;
6. odrzuca pustą listę wejść;
7. sprawdza `cancel_requested()`;
8. normalizuje `model_name` i odrzuca wartość inną niż `stanza` lub `spacy`;
9. buduje stan aktywnych wejść;
10. wywołuje `_run_creator_job_impl(...)`.

Callback `completed(...)` zbiera wynik `_run_creator_job_impl()`. `run_creator_job()` zwraca `CreatorRunResult` z polami `success`, `output_file` i `error_message`.

## Pozostałe główne elementy

- `_run_creator_job_impl(...)` wykonuje walidację plików, odczyt źródeł, inicjalizację modeli, anotację, zapis części i publikację Parquet.
- `CreatorModelState` przechowuje zainicjalizowane pipeline'y NLP między dokumentami.
- `initialize_stanza(...)` i `initialize_spacy(...)` tworzą pipeline wybranego backendu.
- `process_single_text(...)` i `process_single_text_spacy(...)` zwracają tablice anotacji dla jednego tekstu lub chunka.
- `chunk_text_safe(...)` dzieli tekst i zachowuje informacje potrzebne do odtworzenia pozycji w całym dokumencie.
- `_write_creator_part(...)` zapisuje partię dokumentów wraz z metadanymi używanymi podczas wznowienia.
