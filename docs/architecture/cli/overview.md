# Architektura interfejsów CLI

Korpusuj ma cztery niezależne wejścia CLI:

```text
korpusuj.corpus.creator_cli
korpusuj.corpus.merger_cli
korpusuj.index.cli
korpusuj.search.cli
```

Każdy moduł ma własny parser argumentów i funkcję `main(argv=None)`, która zwraca kod zakończenia. Blok `if __name__ == "__main__"` przekazuje ten kod do `SystemExit`.

Dane przeznaczone do dalszego przetwarzania są wypisywane na `stdout`. Postęp i komunikaty diagnostyczne trafiają na `stderr`. Dzięki temu JSON lub JSONL można przekierować do pliku bez domieszki informacji o postępie.

`korpusuj.search.headless_runner` składa wyszukiwarkę bez zależności od `engine.py`. Tworzy `LazyCorpus`, executor, runtime kursora i adapter potrzebny kodowi oczekującemu funkcji `find_lemma_context`.
