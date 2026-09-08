from __future__ import annotations

from types import SimpleNamespace

from korpusuj.search.backend import LazyCorpus
from korpusuj.search.collocations import _background_frequency


class FakeIndex:
    total_tokens = 1000

    def get_docs_many(self, doc_ids, chunk_size=800):
        return {
            int(doc_id): {
                "doc_id": int(doc_id),
                "metadata": {},
                "text": "",
                "tokens": ["A", "B"],
                "lemmas": ["a", "b"],
                "sentence_ids": [0, 0],
            }
            for doc_id in doc_ids
        }


class FakeDependencyCache:
    def __init__(self):
        self.get_many_calls = []
        self.get_calls = 0

    def get_many(self, doc_ids, batch_size=800):
        ids = list(doc_ids)
        self.get_many_calls.append((ids, batch_size))
        return {int(doc_id): ([-1, 0], None) for doc_id in ids}

    def get(self, doc_id):
        self.get_calls += 1
        raise AssertionError("per-document dependency get() must not be used")


class FakeFrequencyProvider:
    def __init__(self):
        self.calls = []

    def get_background_frequencies(self, attr, values, ignore_case=False):
        values = list(values)
        self.calls.append((attr, values, ignore_case))
        return {str(value): 7 for value in values}


def test_lazy_corpus_batches_dependency_maps_for_many_documents():
    corpus = LazyCorpus("corpus.parquet", "corpus.search", total_docs=2)
    corpus._search_index = FakeIndex()
    corpus._dependency_cache = FakeDependencyCache()

    docs = corpus.get_docs_many([1, 2], chunk_size=32)

    assert corpus._dependency_cache.get_many_calls == [([1, 2], 32)]
    assert corpus._dependency_cache.get_calls == 0
    assert docs[1]["word_ids"] == [1, 2]
    assert docs[1]["head_ids"] == [0, 1]
    assert docs[2]["word_ids"] == [1, 2]
    assert docs[2]["head_ids"] == [0, 1]


def test_background_frequency_requests_only_candidate_values():
    provider = FakeFrequencyProvider()
    frequencies, total = _background_frequency(
        {"frequency_provider": provider, "total_tokens": 1000},
        "Lemat (base)",
        True,
        required_values=["uczeń", "nauczyciel"],
    )

    assert provider.calls == [("base", ["uczeń", "nauczyciel"], True)]
    assert frequencies == {"uczeń": 7, "nauczyciel": 7}
    assert total == 1000


def test_background_frequency_preserves_dictionary_compatibility():
    frequencies, total = _background_frequency(
        {"base_tf": {"Uczeń": 2, "uczeń": 3}, "total_tokens": 10},
        "Lemat (base)",
        True,
        required_values=["uczeń"],
    )
    assert frequencies == {"uczeń": 5}
    assert total == 10
