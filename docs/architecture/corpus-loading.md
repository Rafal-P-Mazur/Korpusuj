# Otwieranie korpusu

## `prepare_loaded_corpus_bundle`

Funkcja `prepare_loaded_corpus_bundle` przygotowuje dane potrzebne GUI i CLI:

1. odczytuje schemat i `korpus_meta` z Parquet;
2. ustala domyślną ścieżkę `.search`;
3. sprawdza zgodność indeksu z Parquet;
4. buduje `.search`, jeśli indeks nie istnieje albo jest nieaktualny;
5. odczytuje metadane indeksu;
6. tworzy `LoadedCorpusBundle`.

## `LoadedCorpusBundle`

`LoadedCorpusBundle` jest niemodyfikowalną dataclassą zwracaną przez `prepare_loaded_corpus_bundle(...)`:

```python
@dataclass(frozen=True)
class LoadedCorpusBundle:
    name: str
    parquet_path: str
    search_path: str
    columns: list[str]
    total_docs: int
    total_tokens: int
    monthly_token_counts: dict
    korpus_meta: dict
    search_meta: dict
    dataframe: object
    inverted_index: dict
```

`dataframe` zawiera `LazyCorpus`. Nazwa pola zachowuje interfejs oczekiwany przez istniejący kod `engine.py`; pole nie zawiera pełnego `pandas.DataFrame`.

`inverted_index` również zachowuje wcześniejszy interfejs GUI. `build_lazy_corpus_bundle(...)` tworzy dokładnie taki słownik:

```python
{
    "base": LazyTermIndex(search_path, "base"),
    "orth": LazyTermIndex(search_path, "orth"),
    "base_tf": korpus_meta.get("base_tf", {}),
    "orth_tf": korpus_meta.get("orth_tf", {}),
    "total_tokens": int(total_tokens or 0),
    "monthly_token_counts": monthly_counts,
}
```

Pola `base` i `orth` są obiektami dostępu do `.search`. Pozostałe pola są statystykami odczytanymi z `korpus_meta` albo wyliczonymi podczas ładowania.

### `LazyTermIndex`

`LazyTermIndex(index_path, attr)` zapisuje ścieżkę `.search` i nazwę atrybutu. Konstruktor nie otwiera SQLite. Pierwsze wywołanie `get(...)` albo operatora `in` tworzy `SearchIndex` i zapisuje go w `_idx`; następne operacje tego samego obiektu używają tego połączenia.

```python
inverted_index["base"].get("wojna", set())
```

zwraca zbiór `doc_id` dokumentów mających dokładny termin `wojna` w atrybucie `base`. Metoda pobiera postingi, ale zwraca tylko ich klucze, bez pozycji tokenowych. Dla `orth` zachowanie jest identyczne, lecz termin jest formą tekstową.

```python
"wojna" in inverted_index["base"]
```

sprawdza, czy informacja o terminie ma `df > 0`. Błąd odczytu jest zamieniany na pusty wynik: `get(...)` zwraca wartość domyślną albo pusty zbiór, a operator `in` zwraca `False`.

`base_tf` i `orth_tf` nie wykonują zapytań do SQLite. Są słownikami łącznych frekwencji zapisanymi w metadanych Parquet. `total_tokens` jest liczbą tokenów całego korpusu, a `monthly_token_counts` przechowuje liczby tokenów według okresów wyprowadzonych z dat publikacji.

## `LazyCorpus`

`LazyCorpus` przechowuje ścieżkę Parquet, ścieżkę `.search`, liczbę dokumentów i metadane. Udostępnia dokumenty po `doc_id` bez ładowania całego korpusu do pamięci.

Funkcje wymagające pełnego DataFrame mogą jawnie zmaterializować dane. Nie dzieje się to przy zwykłym otwarciu projektu.

## Zakres automatycznej kontroli

Ten przebieg zapewnia świeżość `.search`. Stan `.dep_cache` jest sprawdzany i przygotowywany przez lifecycle dependency. Brak świeżego `.dep_cache` nie blokuje otwarcia korpusu, ale wpływa na gotowość i szybkość funkcji składniowych.
