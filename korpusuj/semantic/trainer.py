from __future__ import annotations
import argparse
import ast
import json
import logging
import math
import os
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import pyarrow.parquet as pq

try:
    from gensim.models import Word2Vec, FastText
except Exception as e:  # pragma: no cover
    Word2Vec = None
    FastText = None
    _GENSIM_IMPORT_ERROR = e
else:
    _GENSIM_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Konfiguracja logowania
# -----------------------------------------------------------------------------
def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Helpery parsowania / normalizacji
# -----------------------------------------------------------------------------
_PUNCT_SET = set(string.punctuation) | {
    "—", "–", "…", "„", "”", "«", "»", "’", "‘", "‚", "·", "•",
    "´", "`", "´", "°", "№", "§", "¶", "※", "•", "▪", "●", "○",
}


def _is_nan_like(value) -> bool:
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except Exception:
        return value is None


def parse_maybe_list(value):
    if _is_nan_like(value): return []
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list): return converted
        except Exception:
            pass
    if isinstance(value, str):
        value = value.strip()
        if not value: return []
        if (value.startswith("[") and value.endswith("]")) or (value.startswith("(") and value.endswith(")")):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)): return list(parsed)
            except Exception:
                pass
        return [value]
    return [value]


def normalize_lemma(lemma: str, lower: bool = True, strip_whitespace: bool = True) -> str:
    if lemma is None:
        return ""
    lemma = str(lemma)
    if strip_whitespace:
        lemma = lemma.strip()
    if lower:
        lemma = lemma.lower()
    return lemma


def is_punctuation_like(token: str) -> bool:
    token = str(token).strip()
    if not token:
        return True
    if token in _PUNCT_SET:
        return True
    if all(ch in _PUNCT_SET for ch in token):
        return True
    return False


def is_numeric_like(token: str) -> bool:
    token = str(token).strip()
    if not token:
        return False
    # obejmuje liczby, daty, godziny, zakresy, formaty z separatorami
    return bool(re.fullmatch(r"[\d\s.,:/\-]+", token))


def has_letter(token: str) -> bool:
    token = str(token)
    return any(ch.isalpha() for ch in token)


def clean_ner_piece(token: str) -> str:
    """
    Czyści pojedynczy segment encji przed sklejeniem:
    - usuwa białe znaki,
    - obcina śmieci interpunkcyjne z początku/końca,
    - normalizuje wielokrotne podkreślniki.
    """
    token = str(token).strip()
    if not token:
        return ""

    # usuń śmieciowe znaki z brzegów
    token = token.strip(" \t\r\n\"'`„”‚’()[]{}<>|")

    # usuń wielokrotne podkreślniki
    token = re.sub(r"_+", "_", token)

    # jeszcze raz obetnij brzegi po normalizacji
    token = token.strip("_- ")

    return token


def is_source_signature_like(token: str) -> bool:
    """
    Wykrywa ciągi typu:
    - olnk/PAP/wPolityce.pl
    - red/wPolityce.pl/PAP/X/Fb/media
    - PAP/ans
    - URL-like / source-line / tag redakcyjny
    """
    token = str(token).strip()
    if not token:
        return False

    lowered = token.lower()

    # wiele slashy = bardzo silny sygnał źródłowy / linkowy
    if lowered.count("/") >= 2:
        return True

    # typowe znaczniki redakcyjno-linkowe
    if any(x in lowered for x in [
        ".pl", ".com", ".net", "http://", "https://", "www.",
        "/pap", "pap/", "/x/", "/fb/", "red/", "media/",
        "wpolityce", "rmf", "onet", "tvn", "interia"
    ]):
        return True

    # coś wygląda jak "PAP/ans", "kk/DW", "aja/RMF"
    if "/" in lowered and len(lowered) <= 40:
        parts = [p for p in lowered.split("/") if p]
        if 2 <= len(parts) <= 5 and all(len(p) <= 20 for p in parts):
            return True

    return False


def is_valid_ner_piece(token: str) -> bool:
    """
    Czy pojedynczy segment nadaje się do bycia częścią encji?
    """
    token = clean_ner_piece(token)
    if not token:
        return False
    if is_punctuation_like(token):
        return False
    if is_numeric_like(token):
        return False
    if is_source_signature_like(token):
        return False
    if not has_letter(token):
        return False
    return True


def build_entity_lemma(parts: List[str]) -> str:
    """
    Buduje bezpieczną postać MWE/encji.
    Jeśli segmenty są brudne, czyści je.
    Jeśli po czyszczeniu nic nie zostaje, zwraca pusty string.
    """
    cleaned = [clean_ner_piece(p) for p in parts]
    cleaned = [p for p in cleaned if is_valid_ner_piece(p)]

    if not cleaned:
        return ""

    entity = "_".join(cleaned)
    entity = re.sub(r"_+", "_", entity).strip("_- ")

    # ostatni bezpiecznik przeciwko śmieciowym początkom i końcom
    entity = entity.strip(" \t\r\n\"'`„”‚’()[]{}<>|")

    if not entity:
        return ""

    if is_source_signature_like(entity):
        return ""

    if is_numeric_like(entity):
        return ""

    if not has_letter(entity):
        return ""

    return entity


@dataclass
class CorpusStats:
    documents_seen: int = 0
    documents_used: int = 0
    sentences_seen: int = 0
    sentences_used: int = 0
    tokens_seen: int = 0
    tokens_used: int = 0
    unique_lemmas: int = 0
    filtered_out_punct: int = 0
    filtered_out_numeric: int = 0
    filtered_out_upos: int = 0
    filtered_out_noise: int = 0
    empty_sentences_after_filter: int = 0



@dataclass
class TrainingConfig:
    parquet_path: str
    output_dir: str
    algo: str = "word2vec"
    vector_size: int = 150
    window: int = 15
    min_count: int = 5
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    sg: int = 1
    epochs: int = 10
    negative: int = 10
    sample: float = 1e-5
    seed: int = 42
    lower: bool = True
    keep_punct: bool = False
    keep_numeric: bool = False
    allowed_upos: Optional[List[str]] = None
    batch_size: int = 512
    save_full_model: bool = True
    save_text_vectors: bool = False
    precompute_neighbors: int = 0
    neighbors_for_top_vocab: int = 2000


# -----------------------------------------------------------------------------
# Iterator zdań z Parqueta
# -----------------------------------------------------------------------------
class ParquetSentenceIterator:
    def __init__(self, parquet_path: Path, config: TrainingConfig):
        self.parquet_path = Path(parquet_path)
        self.config = config
        self.stats = CorpusStats()
        self.lemma_counter = Counter()
        self._validated_columns: Optional[List[int]] = None  # <--- ZMIANA: przechowujemy INDEKSY
        self._is_first_pass = True

    def _validate_schema(self) -> List[int]:
        parquet_file = pq.ParquetFile(self.parquet_path)
        names = parquet_file.schema.names

        col_map_lower = {n.lower(): i for i, n in enumerate(names)}

        lemma_idx = -1
        sent_idx = -1
        upos_idx = -1
        ner_idx = -1

        # Szukamy po poprawnych nazwach
        if "lemmas" in col_map_lower:
            lemma_idx = col_map_lower["lemmas"]
        elif "base" in col_map_lower:
            lemma_idx = col_map_lower["base"]

        if "sentence_ids" in col_map_lower:
            sent_idx = col_map_lower["sentence_ids"]
        elif "sentence_id" in col_map_lower:
            sent_idx = col_map_lower["sentence_id"]
        elif "sentence" in col_map_lower:
            sent_idx = col_map_lower["sentence"]

        if "upostags" in col_map_lower:
            upos_idx = col_map_lower["upostags"]
        elif "upos" in col_map_lower:
            upos_idx = col_map_lower["upos"]

        if "ners" in col_map_lower:
            ner_idx = col_map_lower["ners"]
        elif "ner" in col_map_lower:
            ner_idx = col_map_lower["ner"]

        # MAGIA: Obejście błędu "element". Szukamy kolumn wygenerowanych anonimowo przez Pandas!
        if lemma_idx == -1 or sent_idx == -1:
            element_indices = [i for i, n in enumerate(names) if n == "element"]
            if len(element_indices) >= 13:
                logging.info("Wykryto kolumny 'element'. Mapuję twarde indeksy według struktury kreatora...")
                # Kolejność w creatorze: 0:tokens, 1:lemmas, 2:postags, 3:full_postags,
                # 4:deprels, 5:word_ids, 6:sentence_ids, ..., 11:upostags
                lemma_idx = element_indices[1]
                sent_idx = element_indices[6]
                ner_idx = element_indices[10]
                upos_idx = element_indices[11]
            else:
                raise ValueError(f"Nie odnaleziono kolumn NLP. Nazwy w pliku: {names}")

        if lemma_idx == -1 or sent_idx == -1:
            raise ValueError(f"Brak krytycznych kolumn. Nazwy w pliku: {names}")

        self._validated_columns = [lemma_idx, sent_idx, upos_idx, ner_idx]
        return self._validated_columns

    def __iter__(self) -> Iterator[List[str]]:
        indices = self._validated_columns or self._validate_schema()
        lemma_idx, sent_idx, upos_idx, ner_idx = indices

        parquet_file = pq.ParquetFile(self.parquet_path)
        is_first = self._is_first_pass

        for batch in parquet_file.iter_batches(batch_size=self.config.batch_size):
            lemmas_col = batch.column(lemma_idx).to_pylist()
            sentence_ids_col = batch.column(sent_idx).to_pylist()
            upostags_col = batch.column(upos_idx).to_pylist() if upos_idx != -1 else [None] * len(lemmas_col)
            ners_col = batch.column(ner_idx).to_pylist() if ner_idx != -1 else [None] * len(lemmas_col)

            for lemmas_raw, sentence_ids_raw, upos_raw, ners_raw in zip(lemmas_col, sentence_ids_col, upostags_col,
                                                                        ners_col):
                if is_first:
                    self.stats.documents_seen += 1

                lemmas = parse_maybe_list(lemmas_raw)
                sentence_ids = parse_maybe_list(sentence_ids_raw)
                upostags = parse_maybe_list(upos_raw) if upos_raw is not None else []
                ners = parse_maybe_list(ners_raw) if ners_raw is not None else []

                if not lemmas or not sentence_ids:
                    continue

                current_sent_id = None
                current_sentence: List[str] = []
                current_len_before_filter = 0

                # Bufor dla wielowyrazowych nazw własnych
                entity_buffer = []

                for idx, lemma in enumerate(lemmas):
                    sent_id = sentence_ids[idx]
                    ner = str(ners[idx]).strip() if ners and idx < len(ners) else "O"
                    upos = str(upostags[idx]).strip().upper() if upostags and idx < len(upostags) else ""

                    if current_sent_id is None:
                        current_sent_id = sent_id
                    elif sent_id != current_sent_id:
                        # Zmiana zdania - jeśli coś zostało w buforze encji, zrzuć to
                        if entity_buffer:
                            entity_lemma = build_entity_lemma(entity_buffer)
                            if entity_lemma:
                                current_sentence.append(entity_lemma)
                            entity_buffer = []

                        if is_first:
                            self.stats.sentences_seen += 1
                            self.stats.tokens_seen += current_len_before_filter
                        if current_sentence:
                            if is_first:
                                self.stats.sentences_used += 1
                                self.stats.tokens_used += len(current_sentence)
                                self.lemma_counter.update(current_sentence)
                            yield current_sentence

                        current_sentence = []
                        current_sent_id = sent_id
                        current_len_before_filter = 0

                    current_len_before_filter += 1
                    normalized = normalize_lemma(lemma, lower=self.config.lower)
                    if not normalized:
                        continue

                    normalized = clean_ner_piece(normalized)
                    if not normalized:
                        continue

                    # 1) Filtr interpunkcji
                    if not self.config.keep_punct and is_punctuation_like(normalized):
                        if is_first:
                            self.stats.filtered_out_punct += 1
                        continue

                    # 2) Filtr tokenów wyłącznie liczbowych / godzinowych / datopodobnych
                    if not self.config.keep_numeric and is_numeric_like(normalized):
                        if is_first:
                            self.stats.filtered_out_numeric += 1
                        continue

                    # 3) Filtr source-signature / URL-like / tagów redakcyjnych
                    if is_source_signature_like(normalized):
                        if is_first:
                            self.stats.filtered_out_noise += 1
                        continue

                    # 4) Filtr UPOS dla zwykłych tokenów (poza NER)
                    if self.config.allowed_upos and ner == "O":
                        if upos not in self.config.allowed_upos:
                            if is_first:
                                self.stats.filtered_out_upos += 1
                            continue

                    # 5) Czy segment nadaje się do bycia częścią encji?
                    valid_for_entity = is_valid_ner_piece(normalized)

                    # -------------------------------------------------
                    # Logika BIOES łącząca NER-y (bardziej konserwatywna)
                    # -------------------------------------------------
                    if ner.startswith("B-"):
                        if entity_buffer:
                            entity_lemma = build_entity_lemma(entity_buffer)
                            if entity_lemma:
                                current_sentence.append(entity_lemma)


                        if valid_for_entity:
                            entity_buffer = [normalized]
                        else:
                            entity_buffer = []

                    elif ner.startswith("I-"):
                        if valid_for_entity:
                            if entity_buffer:
                                entity_buffer.append(normalized)
                            else:
                                # parser dał I- bez B-, więc traktujemy ostrożnie jak pojedynczy token
                                current_sentence.append(normalized)
                        else:
                            if entity_buffer:
                                entity_lemma = build_entity_lemma(entity_buffer)
                                if entity_lemma:
                                    current_sentence.append(entity_lemma)
                                entity_buffer = []

                    elif ner.startswith("E-"):
                        if valid_for_entity:
                            if entity_buffer:
                                entity_buffer.append(normalized)
                                entity_lemma = build_entity_lemma(entity_buffer)
                                if entity_lemma:
                                    current_sentence.append(entity_lemma)
                                entity_buffer = []
                            else:
                                # parser dał E- bez B-/I-, więc zachowujemy token jako zwykły
                                current_sentence.append(normalized)
                        else:
                            if entity_buffer:
                                entity_lemma = build_entity_lemma(entity_buffer)
                                if entity_lemma:
                                    current_sentence.append(entity_lemma)
                                entity_buffer = []

                    elif ner.startswith("S-"):
                        if entity_buffer:
                            entity_lemma = build_entity_lemma(entity_buffer)
                            if entity_lemma:
                                current_sentence.append(entity_lemma)
                            entity_buffer = []

                        if valid_for_entity:
                            current_sentence.append(normalized)

                    else:  # ner == "O" lub błąd
                        if entity_buffer:
                            entity_lemma = build_entity_lemma(entity_buffer)
                            if entity_lemma:
                                current_sentence.append(entity_lemma)
                            entity_buffer = []

                        current_sentence.append(normalized)

                # Obsługa końca dokumentu
                if current_sent_id is not None:
                    if entity_buffer:
                        entity_lemma = build_entity_lemma(entity_buffer)
                        if entity_lemma:
                            current_sentence.append(entity_lemma)


                    if is_first:
                        self.stats.sentences_seen += 1
                        self.stats.tokens_seen += current_len_before_filter
                    if current_sentence:
                        if is_first:
                            self.stats.sentences_used += 1
                            self.stats.tokens_used += len(current_sentence)
                            self.lemma_counter.update(current_sentence)
                        yield current_sentence

        if is_first:
            self.stats.unique_lemmas = len(self.lemma_counter)
            self._is_first_pass = False


# -----------------------------------------------------------------------------
# Trening modelu
# -----------------------------------------------------------------------------
def ensure_gensim_available() -> None:
    if Word2Vec is None or FastText is None:
        raise RuntimeError(
            "Nie udało się zaimportować gensim.models.Word2Vec/FastText. "
            f"Szczegóły: {_GENSIM_IMPORT_ERROR}"
        )


def train_embedding_model(sentences: Iterable[List[str]], config: TrainingConfig):
    ensure_gensim_available()

    common_kwargs = dict(
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        sg=config.sg,
        negative=config.negative,
        sample=config.sample,
        seed=config.seed,
    )

    algo = config.algo.lower().strip()
    logging.info(f"Inicjalizacja pustego modelu {algo.upper()}...")

    if algo == "word2vec":
        model = Word2Vec(**common_kwargs)
    elif algo == "fasttext":
        model = FastText(**common_kwargs)
    else:
        raise ValueError("Nieobsługiwany algorytm. Użyj: word2vec albo fasttext")

    # ETAP 1: Słownik (Tylko Pass 1)
    logging.info("ETAP 1/2: Budowanie słownika z korpusu (Pass 1)...")
    model.build_vocab(sentences)
    logging.info(f"Słownik zbudowany. Liczba unikalnych form (vocab): {len(model.wv)}")

    # ETAP 2: Trening wlasciwy
    logging.info(f"ETAP 2/2: Trening modelu (Liczba epok: {config.epochs}). To może potrwać...")
    model.train(
        corpus_iterable=sentences,
        total_examples=model.corpus_count,
        epochs=config.epochs
    )

    return model


# -----------------------------------------------------------------------------
# Zapis artefaktów
# -----------------------------------------------------------------------------
def build_output_prefix(parquet_path: Path, output_dir: Path, algo: str) -> Path:
    stem = parquet_path.stem
    return output_dir / f"{stem}.semantic.{algo.lower()}"


def save_metadata(prefix: Path, config: TrainingConfig, stats: CorpusStats, top_lemmas: Sequence) -> Path:
    metadata_path = prefix.with_suffix(".meta.json")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trainer": "semantic_trainer.py",
        "config": asdict(config),
        "stats": asdict(stats),
        "top_lemmas": [{"lemma": lemma, "freq": int(freq)} for lemma, freq in top_lemmas],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return metadata_path


def save_neighbors(prefix: Path, model, lemma_counter: Counter, n_neighbors: int, top_vocab: int) -> Optional[Path]:
    if n_neighbors <= 0: return None
    import pandas as pd

    rows = []
    actual_vocab = model.wv.key_to_index.keys()
    candidate_lemmas = [lemma for lemma, _ in lemma_counter.most_common(top_vocab) if lemma in actual_vocab]

    for lemma in candidate_lemmas:
        try:
            neighbors = model.wv.most_similar(lemma, topn=n_neighbors)
            for rank, (neighbor, score) in enumerate(neighbors, start=1):
                if neighbor in actual_vocab:
                    rows.append({
                        "lemma": lemma,
                        "lemma_freq": lemma_counter[lemma],
                        "neighbor": neighbor,
                        "neighbor_freq": lemma_counter[neighbor],
                        "similarity": float(score),
                        "rank": rank,
                    })
        except KeyError:
            continue

    if not rows: return None
    neighbors_path = prefix.with_suffix(".neighbors.parquet")
    pd.DataFrame(rows).to_parquet(neighbors_path, engine="pyarrow", compression="snappy")
    return neighbors_path

def save_lightweight_vectors(
        prefix: Path,
        model,
        lemma_counter: Counter,
        top_vocab: int,
        n_neighbors: int = 0
) -> Optional[Path]:
    if top_vocab <= 0: return None

    import pandas as pd
    import numpy as np

    actual_vocab = set(model.wv.key_to_index.keys())

    # 1) Seed lemmata = to samo źródło, z którego budujemy neighbors
    seed_lemmas = [
        lemma
        for lemma, _ in lemma_counter.most_common(top_vocab)
        if lemma in actual_vocab
    ]

    # 2) Startujemy od seedów
    vector_lemmas = set(seed_lemmas)

    # 3) Dociągamy wszystkie słowa, które pojawiają się jako neighbors tych seedów
    if n_neighbors > 0:
        for lemma in seed_lemmas:
            try:
                neighbors = model.wv.most_similar(lemma, topn=n_neighbors)
            except KeyError:
                continue

            for neighbor, _score in neighbors:
                if neighbor in actual_vocab:
                    vector_lemmas.add(neighbor)

    if not vector_lemmas:
        return None

    rows = [
        {
            "lemma": lemma,
            "vector": model.wv[lemma].astype(np.float32)  # float32 drastycznie zmniejsza wagę pliku
        }
        for lemma in sorted(vector_lemmas)
    ]

    vectors_path = prefix.with_suffix(".vectors.parquet")
    pd.DataFrame(rows).to_parquet(vectors_path, engine="pyarrow", compression="snappy")

    logging.info(
        f"Zapisano lekkie wektory dla {len(rows)} słów do {vectors_path.name} "
        f"(seedy: {len(seed_lemmas)}, rozszerzenie o neighbors: {len(vector_lemmas) - len(seed_lemmas)})"
    )

    return vectors_path


def save_model_artifacts(prefix: Path, model, config: TrainingConfig, iterator: ParquetSentenceIterator) -> dict:
    output_paths = {}

    kv_path = prefix.with_suffix(".kv")
    model.wv.save(str(kv_path))
    output_paths["keyed_vectors"] = str(kv_path)

    if config.save_full_model:
        model_path = prefix.with_suffix(".model")
        model.save(str(model_path))
        output_paths["full_model"] = str(model_path)

    if config.save_text_vectors:
        txt_path = prefix.with_suffix(".vectors.txt")
        model.wv.save_word2vec_format(str(txt_path), binary=False)
        output_paths["text_vectors"] = str(txt_path)

    top_lemmas = iterator.lemma_counter.most_common(200)
    metadata_path = save_metadata(prefix, config, iterator.stats, top_lemmas)
    output_paths["metadata"] = str(metadata_path)

    neighbors_path = save_neighbors(
        prefix=prefix,
        model=model,
        lemma_counter=iterator.lemma_counter,
        n_neighbors=config.precompute_neighbors,
        top_vocab=config.neighbors_for_top_vocab,
    )
    if neighbors_path:
        output_paths["neighbors"] = str(neighbors_path)

    vectors_path = save_lightweight_vectors(
        prefix=prefix,
        model=model,
        lemma_counter=iterator.lemma_counter,
        top_vocab=config.neighbors_for_top_vocab,
        n_neighbors=config.precompute_neighbors,
    )
    if vectors_path:
        output_paths["vectors"] = str(vectors_path)

    return output_paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trenuje model semantyczny Word2Vec/FastText na gotowym korpusie Parquet.")
    parser.add_argument("--parquet", required=True, help="Ścieżka do pliku .parquet z korpusem")
    parser.add_argument("--output-dir", default=None, help="Folder wyjściowy dla artefaktów")
    parser.add_argument("--algo", choices=["word2vec", "fasttext"], default="word2vec", help="Algorytm")
    parser.add_argument("--vector-size", type=int, default=150, help="Wymiar wektora")
    parser.add_argument("--window", type=int, default=15, help="Rozmiar okna kontekstowego")
    parser.add_argument("--min-count", type=int, default=5, help="Minimalna frekwencja lemy w słowniku")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Liczba wątków")
    parser.add_argument("--sg", type=int, choices=[0, 1], default=1, help="1=skip-gram, 0=CBOW")
    parser.add_argument("--epochs", type=int, default=10, help="Liczba epok")
    parser.add_argument("--negative", type=int, default=10, help="Liczba negatywnych próbek")
    parser.add_argument("--sample", type=float, default=1e-5, help="Subsampling częstych słów")
    parser.add_argument("--seed", type=int, default=42, help="Seed treningu")
    parser.add_argument("--keep-punct", action="store_true", help="Nie usuwaj interpunkcji")
    parser.add_argument("--keep-numeric", action="store_true", help="Nie usuwaj tokenów liczbowych")
    parser.add_argument("--allowed-upos", nargs="+", default=None, help="Lista dozwolonych UPOS")
    parser.add_argument("--batch-size", type=int, default=512, help="Rozmiar partii odczytu Parquet")
    parser.add_argument("--no-full-model", action="store_true", help="Nie zapisuj pełnego modelu")
    parser.add_argument("--save-text-vectors", action="store_true", help="Zapisz także .vectors.txt")
    parser.add_argument("--precompute-neighbors", type=int, default=0, help="Prekomputacja sąsiadów")
    parser.add_argument("--neighbors-for-top-vocab", type=int, default=2000, help="Dla ilu lematów prekomputować")
    parser.add_argument("--verbose", action="store_true", help="Logowanie DEBUG")
    parser.add_argument("--no-lower", action="store_true", help="Nie zamieniaj lematów na małe litery")
    return parser


# -----------------------------------------------------------------------------
# API wysokiego poziomu
# -----------------------------------------------------------------------------
def train_from_parquet(config: TrainingConfig) -> dict:
    import tempfile
    import zipfile

    parquet_path = Path(config.parquet_path).resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku Parquet: {parquet_path}")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nasz nowy, czysty plik wyjściowy
    wektor_path = output_dir / f"{parquet_path.stem}.wektor"

    # Tworzymy folder tymczasowy, żeby nie śmiecić obok korpusu
    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        tmp_path = Path(tmpdir)
        iterator = ParquetSentenceIterator(parquet_path=parquet_path, config=config)
        prefix = build_output_prefix(parquet_path=parquet_path, output_dir=tmp_path, algo=config.algo)

        logging.info("Rozpoczynam odczyt i trening na pliku: %s", parquet_path)

        model = train_embedding_model(sentences=iterator, config=config)

        if iterator.stats.unique_lemmas == 0:
            iterator.stats.unique_lemmas = len(iterator.lemma_counter)

        # Trener zapisuje całą chmurę plików do tmp_path
        save_model_artifacts(prefix=prefix, model=model, config=config, iterator=iterator)

        logging.info(f"Pakowanie modelu do archiwum: {wektor_path.name}")

        # Pakujemy wszystko do jednego pliku .wektor (Z kompresją, żeby zaoszczędzić miejsce)
        with zipfile.ZipFile(wektor_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmp_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmp_path)
                    zf.write(file_path, arcname)

    # Blok 'with tempfile...' automatycznie kasuje folder tmpdir po wyjściu z niego!

    logging.info("Trening i sprzątanie zakończone pomyślnie.")
    return {
        "output_paths": {"wektor_archive": str(wektor_path)},
        "stats": asdict(iterator.stats),
        "vocab_size": len(model.wv),
        "prefix": str(wektor_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    config = TrainingConfig(
        parquet_path=args.parquet,
        output_dir=args.output_dir or str(Path(args.parquet).resolve().parent),
        algo=args.algo,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        epochs=args.epochs,
        negative=args.negative,
        sample=args.sample,
        seed=args.seed,
        lower=not args.no_lower,
        keep_punct=args.keep_punct,
        keep_numeric=args.keep_numeric,
        allowed_upos=[x.upper() for x in args.allowed_upos] if args.allowed_upos else None,
        batch_size=args.batch_size,
        save_full_model=not args.no_full_model,
        save_text_vectors=args.save_text_vectors,
        precompute_neighbors=args.precompute_neighbors,
        neighbors_for_top_vocab=args.neighbors_for_top_vocab,
    )

    try:
        result = train_from_parquet(config)
    except KeyboardInterrupt:
        logging.warning("Przerwano przez użytkownika.")
        return 130
    except Exception as e:
        logging.exception("Błąd podczas treningu semantycznego: %s", e)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())