# Pipeline creatora

## Wspólny runtime

GUI i creator CLI przekazują `CreatorRunOptions` do wspólnego `run_creator_job`.

```text
GUI creator --\
              -> run_creator_job -> Parquet
creator CLI --/
```

## Etapy

1. Walidacja TXT, DOCX, PDF, XLSX i ZIP oraz ścieżki wyjściowej.
2. Odczyt i normalizacja przez `creator_io.py` i orkiestrator.
3. Chunking tekstu przez `creator_chunking.py`.
4. Inicjalizacja Stanza lub spaCy w `creator_nlp.py`; `CreatorModelState` utrzymuje stan modelu.
5. Anotacja morfosyntaktyczna i dependency oraz opcjonalne NER i koreferencja.
6. Łączenie metadanych XLSX przez wymagany klucz `Nazwa pliku`.
7. Zapis checkpointów i wznowienie zgodnego zadania.
8. Scalenie części do finalnego Parquet.

## OCR

PDF najpierw korzysta z warstwy tekstowej. Opcjonalny OCR zależy od dostępności bibliotek i zasobów i nie jest warunkiem podstawowego pipeline.

## Metadane schematu

Aktywne warstwy anotacji są zapisywane w metadanych Parquet i służą kontroli zgodności wznowienia.

## Wynik

Creator kończy na kanonicznym Parquet. `.search` i `.dep_cache` tworzy osobno index CLI.

## Postęp

`creator_core.py` definiuje kontrakt reportera. CLI używa stderr, a adapter GUI planuje aktualizacje widżetów na wątku interfejsu.
