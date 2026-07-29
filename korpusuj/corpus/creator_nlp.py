# -*- coding: utf-8 -*-
"""Lightweight creator NLP state types.

Heavy NLP libraries and model initialization intentionally remain in the
existing creator module until the stateful extraction design is approved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import gc
import logging
from korpusuj.runtime_paths import configure_ml_cache_environment as _configure_ml_cache_environment_182n
_configure_ml_cache_environment_182n()

import torch



@dataclass(slots=True)
class CreatorModelState:
    """Owns NLP/SRL session objects explicitly instead of relying on globals."""

    nlp_stanza: Any = None
    nlp_spacy: Any = None

    def clear_all(self) -> None:
        self.nlp_stanza = None
        self.nlp_spacy = None

# --- Pure coreference helpers extracted from creator.py (track 172f3) ---
def _coref_anchor_append_unique_138c(coref_anchors, key, label):
    """Append coref label once per token/key."""
    try:
        labels = coref_anchors.setdefault(key, [])
        if label not in labels:
            labels.append(label)
    except Exception:
        try:
            coref_anchors.setdefault(key, []).append(label)
        except Exception:
            pass


def _coref_span_crosses_sentence_boundary(span):
    """Return True when tokens in one mention belong to different sentences."""
    sentence_starts = set()
    for token in span:
        try:
            sentence_starts.add(int(token.sent.start))
        except Exception:
            return True
    return len(sentence_starts) != 1


def _positions_are_contiguous(positions):
    """Require a complete continuous token mapping for a mention."""
    if not positions:
        return False
    ordered = sorted(int(position) for position in positions)
    return ordered == list(range(ordered[0], ordered[-1] + 1))

# --- NKJP full-tag reconstruction helper extracted from creator.py ---
def reconstruct_nkjp_tag(tag_str, morph_obj):
    base = tag_str.lower()

    def m(key, mapping):
        vals = morph_obj.get(key)
        val = vals[0] if vals else ""
        # Jeśli brak cechy lub nie ma jej w mapowaniu, dajemy "_"
        return mapping.get(val, "_")

    num = m("Number", {"Sing": "sg", "Plur": "pl", "Ptan": "ptan"})
    case = m("Case",
             {"Nom": "nom", "Gen": "gen", "Dat": "dat", "Acc": "acc", "Ins": "inst", "Loc": "loc", "Voc": "voc"})

    gen_vals = morph_obj.get("Gender")
    gen_val = gen_vals[0] if gen_vals else ""
    anim_vals = morph_obj.get("Animacy")
    anim_val = anim_vals[0] if anim_vals else ""

    if gen_val == "Masc":
        if anim_val == "Hum":
            gen = "m1"
        elif anim_val == "Nhum":
            gen = "m2"
        elif anim_val == "Inan":
            gen = "m3"
        else:
            gen = "m1"
    elif gen_val == "Fem":
        gen = "f"
    elif gen_val == "Neut":
        gen = "n"
    else:
        gen = ""

    person = m("Person", {"1": "pri", "2": "sec", "3": "ter"})
    aspect = m("Aspect", {"Imp": "imperf", "Perf": "perf"})
    degree = m("Degree", {"Pos": "pos", "Cmp": "com", "Sup": "sup"})

    # Składanie łańcucha z zachowaniem ścisłej, pozycyjnej kolejności NKJP
    parts = [base]
    if base in ("subst", "depr"):
        parts.extend([num, case, gen])
    elif base in ("adj", "adja", "adjp"):
        parts.extend([num, case, gen, degree])
    elif base in ("ppron12", "ppron3"):
        parts.extend([num, case, gen, person, "", ""])
    elif base == "num":
        parts.extend([num, case, gen, ""])
    elif base in ("fin", "bedzie", "impt"):
        parts.extend([num, person, aspect])
    elif base in ("praet", "winien"):
        parts.extend([num, gen, aspect, ""])
    elif base in ("inf", "pcon", "pant", "imps"):
        parts.extend([aspect])
    elif base in ("ger", "pact", "ppas"):
        parts.extend([num, case, gen, aspect, ""])
    elif base == "adv":
        parts.extend([degree])

    if len(parts) == 1:
        return base
    return ":".join(parts)

def initialize_stanza(
    state: CreatorModelState,
    reporter,
    *,
    stanza_module,
    models_dir: str,
    enable_ner: bool = True,
    enable_coreference: bool = True,
):
    """Load Stanza into ``state`` without tkinter/messagebox dependencies."""
    import os

    if stanza_module is None:
        reporter.error("Biblioteka Stanza nie jest zainstalowana.")
        return False

    stanza_dir = os.path.join(models_dir, "stanza")
    os.makedirs(stanza_dir, exist_ok=True)
    model_path = os.path.join(stanza_dir, "pl")

    if not os.path.exists(model_path):
        try:
            reporter.status("Proszę czekać - pobieram model Stanza (ok. 500 MB)...")
            reporter.tick()
            stanza_module.download("pl", model_dir=stanza_dir)
        except Exception as exc:
            reporter.error("Nie udało się pobrać modelu Stanza", exc)
            return False

    reporter.status("Ładuję model Stanza z folderu 'models' - proszę czekać.")
    reporter.tick()
    try:
        processors = ["tokenize", "pos", "lemma"]
        if enable_ner:
            processors.append("ner")
        processors.append("depparse")
        if enable_coreference:
            processors.append("coref")
        state.nlp_stanza = stanza_module.Pipeline(
            "pl",
            model_dir=stanza_dir,
            processors=",".join(processors),
            use_gpu=bool(torch.cuda.is_available()),
            n_process=1,
        )
        reporter.status("Model Stanza załadowany pomyślnie")
        return True
    except Exception as exc:
        state.nlp_stanza = None
        try:
            import importlib.metadata as importlib_metadata
            import sys

            def _package_version_182y(distribution_name):
                try:
                    return importlib_metadata.version(distribution_name)
                except Exception:
                    return "unavailable"

            logging.getLogger(__name__).exception(
                "STANZA_INITIALIZATION_FAILURE_182Y | frozen=%s | executable=%r | "
                "meipass=%r | models_dir=%r | stanza_dir=%r | processors=%r | "
                "enable_ner=%s | enable_coreference=%s | HF_HOME=%r | "
                "HF_HUB_CACHE=%r | HUGGINGFACE_HUB_CACHE=%r | HF_XET_CACHE=%r | "
                "TRANSFORMERS_CACHE=%r | stanza=%s | transformers=%s | peft=%s | "
                "torch=%s | huggingface-hub=%s | hf-xet=%s",
                bool(getattr(sys, "frozen", False)),
                sys.executable,
                getattr(sys, "_MEIPASS", None),
                models_dir,
                stanza_dir,
                processors,
                enable_ner,
                enable_coreference,
                os.environ.get("HF_HOME"),
                os.environ.get("HF_HUB_CACHE"),
                os.environ.get("HUGGINGFACE_HUB_CACHE"),
                os.environ.get("HF_XET_CACHE"),
                os.environ.get("TRANSFORMERS_CACHE"),
                _package_version_182y("stanza"),
                _package_version_182y("transformers"),
                _package_version_182y("peft"),
                _package_version_182y("torch"),
                _package_version_182y("huggingface-hub"),
                _package_version_182y("hf-xet"),
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "STANZA_INITIALIZATION_DIAGNOSTIC_FAILURE_182Y"
            )
        reporter.error("Nie udało się załadować Stanza", exc)
        return False


def initialize_spacy(
    state: CreatorModelState,
    reporter,
    *,
    spacy_module,
    herference_module,
    requests_module,
    models_dir: str,
    enable_ner: bool = True,
    enable_coreference: bool = True,
):
    """Load local SpaCy/herference resources into ``state`` without GUI calls."""
    import os
    import sys
    import zipfile

    model_name = "pl_core_news_lg"
    model_version = "3.8.0"
    spacy_dir = os.path.join(models_dir, "spacy")

    try:
        reporter.status("Sprawdzam model SpaCy w folderze 'models'...")
        reporter.tick()
        if spacy_dir not in sys.path:
            sys.path.insert(0, spacy_dir)

        if not os.path.exists(os.path.join(spacy_dir, model_name, "__init__.py")):
            reporter.status(
                f"Pobieram model SpaCy ({model_name}). To może potrwać (ok. 500MB)..."
            )
            reporter.tick()
            url = (
                "https://github.com/explosion/spacy-models/releases/download/"
                f"{model_name}-{model_version}/{model_name}-{model_version}-py3-none-any.whl"
            )
            os.makedirs(spacy_dir, exist_ok=True)
            whl_path = os.path.join(spacy_dir, "temp_model.whl")
            response = requests_module.get(url, stream=True)
            response.raise_for_status()
            with open(whl_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)
            reporter.status("Rozpakowuję i instaluję model SpaCy w folderze 'models'...")
            reporter.tick()
            with zipfile.ZipFile(whl_path, "r") as archive:
                archive.extractall(spacy_dir)
            os.remove(whl_path)
            reporter.status("Model SpaCy zainstalowany pomyślnie.")
            reporter.tick()

        reporter.status("Ładuję model SpaCy (proszę czekać)...")
        reporter.tick()
        disabled_components = [] if enable_ner else ["ner"]
        state.nlp_spacy = spacy_module.load(model_name, disable=disabled_components)

        if enable_coreference and herference_module is not None:
            try:
                reporter.status("Podpinam model herference (koreferencje)...")
                reporter.tick()
                state.nlp_spacy.add_pipe("herference")
            except Exception as exc:
                logging.exception(
                    "SpaCy załadowano, ale podpinanie Herference zakończyło się błędem"
                )
                reporter.error(
                    "SpaCy załadowano, ale podpinanie herference zakończyło się błędem",
                    exc,
                )

        reporter.status("Model SpaCy załadowany pomyślnie z folderu lokalnego.")
        return state.nlp_spacy
    except Exception as exc:
        state.nlp_spacy = None
        reporter.error("Wystąpił błąd podczas pobierania lub ładowania modelu SpaCy", exc)
        return None

from korpusuj.corpus.creator_chunking import chunk_text_safe

# --- COREF_MENTIONS_CANONICAL_CREATOR_174J1 ---
class _ProcessedTokensWithCorefMentions174J1(list):
    """List-compatible NLP result carrying document-level mention records."""

    def __init__(self):
        super().__init__()
        self.coref_mentions = []


def _append_coref_mention_174j1(processed_tokens, cluster_id, mention_id, start, end, head):
    """Append a validated canonical mention without altering legacy corefs."""
    start, end, head = int(start), int(end), int(head)
    if not (0 <= start < end and start <= head < end):
        raise ValueError(
            f"invalid coref mention coordinates: cluster={cluster_id!r}, "
            f"mention={mention_id}, span=({start}, {end}), head={head}"
        )
    processed_tokens.coref_mentions.append({
        "cluster_id": str(cluster_id),
        "mention_id": int(mention_id),
        "start": start,
        "end": end,
        "head": head,
    })
# --- END COREF_MENTIONS_CANONICAL_CREATOR_174J1 ---

# --- Stateful text processors (track 172g4) ---
def process_single_text(text, filename, state: CreatorModelState, reporter):
    import torch
    if not text.strip(): return None
    chunks = chunk_text_safe(text, chunk_size=15000)
    all_processed_tokens = _ProcessedTokensWithCorefMentions174J1()
    global_mention_id_counter = 1
    global_sent_id_offset = 0
    global_char_offset = 0
    total_chunks = len(chunks)
    reporter.current(0)

    # --- NOWE: GLOBALNY LICZNIK DLA CAŁEGO PLIKU (Rozwiązuje problem kolizji chunków) ---
    global_cluster_id_counter = 1

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            global_char_offset += len(chunk)
            continue
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            reporter.status(f"Przetwarzam: {filename} (Część {i + 1}/{total_chunks})"); reporter.tick()
            doc = state.nlp_stanza(chunk)
        except Exception as e:
            logging.warning(f"Błąd Stanza (część {i + 1}): {e}")
            global_char_offset += len(chunk)
            continue

        if not doc.sentences:
            global_char_offset += len(chunk)
            continue

        # --- BUDOWANIE MAPY KOTWIC (HYBRYDA: ROLA + ID) DLA STANZA ---
        coref_anchors = {}
        chunk_token_offset = len(all_processed_tokens)
        stanza_word_to_global_174j1 = {}
        _chunk_local_token_index_174j1 = 0
        for _sentence_index_174j1, _sentence_174j1 in enumerate(doc.sentences):
            for _word_174j1 in _sentence_174j1.words:
                if isinstance(getattr(_word_174j1, "id", None), int):
                    stanza_word_to_global_174j1[(_sentence_index_174j1, _word_174j1.id)] = (
                        chunk_token_offset + _chunk_local_token_index_174j1
                    )
                _chunk_local_token_index_174j1 += 1
        try:
            # Stanza trzyma gotowe klastry w doc.coref: lista CorefChain (doc-level)
            if hasattr(doc, "coref") and doc.coref:
                for chain in doc.coref:
                    c_id_str = str(global_cluster_id_counter)
                    global_cluster_id_counter += 1

                    for mention in getattr(chain, "mentions", []):
                        s_idx = getattr(mention, "sentence", getattr(mention, "sent_id", None))
                        if s_idx is None:
                            continue

                        # Zabezpieczenie na zero-anaphora / nietypowe indeksy
                        if not isinstance(getattr(mention, "start_word", None), int) or not isinstance(
                                getattr(mention, "end_word", None), int):
                            continue

                        sentence_obj = doc.sentences[s_idx]
                        span_words = sentence_obj.words[mention.start_word:mention.end_word]
                        if not span_words:
                            continue
                        # --- Heurystyka wyznaczania Head ---
                        mention_ids = {w.id for w in span_words if isinstance(w.id, int)}
                        anchor = None
                        for w in span_words:
                            if not isinstance(w.id, int):
                                continue
                            if w.head not in mention_ids:
                                anchor = w
                                break
                        if anchor is None:
                            # fallback: ostatni "normalny" word, jeśli istnieje
                            anchor = next((w for w in reversed(span_words) if isinstance(w.id, int)), None)
                            if anchor is None:
                                continue

                        mapped_span_words = [
                            w for w in span_words
                            if isinstance(w.id, int) and (s_idx, w.id) in stanza_word_to_global_174j1
                        ]
                        if len(mapped_span_words) != len(span_words):
                            continue
                        _mention_positions_174j1 = [
                            stanza_word_to_global_174j1[(s_idx, w.id)]
                            for w in mapped_span_words
                        ]
                        if (
                            (s_idx, anchor.id) not in stanza_word_to_global_174j1
                            or not _positions_are_contiguous(_mention_positions_174j1)
                        ):
                            continue
                        _append_coref_mention_174j1(
                            all_processed_tokens, c_id_str, global_mention_id_counter,
                            min(_mention_positions_174j1), max(_mention_positions_174j1) + 1,
                            stanza_word_to_global_174j1[(s_idx, anchor.id)],
                        )
                        global_mention_id_counter += 1

                        # --- Role Head/Part ---
                        for w in span_words:
                            if not isinstance(w.id, int):  # pomijamy puste węzły
                                continue
                            role = "Head" if w.id == anchor.id else "Part"
                            _coref_anchor_append_unique_138c(coref_anchors, (s_idx, w.id), f"{role}-{c_id_str}")

        except Exception as e:
            logging.warning(f"Błąd mapowania koreferencji Stanza: {e}")
        # -----------------------------------------------------------------



        chunk_char_pos = 0
        for sent_idx, sentence in enumerate(doc.sentences, start=1):
            real_sent_id = sent_idx + global_sent_id_offset
            # Current file progress
            current_progress = (i / total_chunks) + ((sent_idx / len(doc.sentences)) / total_chunks)
            if sent_idx % 10 == 0:
                reporter.current(current_progress)
                reporter.tick()

            word_to_ner = {word.id: token.ner for token in sentence.tokens for word in token.words}
            for word in sentence.words:
                start_idx_local = chunk.find(word.text, chunk_char_pos)
                if start_idx_local == -1: start_idx_local = chunk_char_pos
                end_idx_local = start_idx_local + len(word.text) - 1
                chunk_char_pos = end_idx_local + 1

                start_idx_global = start_idx_local + global_char_offset
                end_idx_global = end_idx_local + global_char_offset

                # Szybki odczyt gotowej etykiety
                sent_idx_stanza = sent_idx - 1
                coref_val = coref_anchors.get((sent_idx_stanza, word.id), [])

                all_processed_tokens.append({
                    "token": word.text,
                    "lemma": word.lemma,
                    "sentenceID": real_sent_id,
                    "wordID": word.id,
                    "headID": word.head,
                    "deprel": word.deprel,
                    "postag": word.xpos,
                    "start": start_idx_global,
                    "end": end_idx_global,
                    "ner": word_to_ner.get(word.id, "0"),
                    "upos": word.upos,
                    "coref": coref_val
                })
        global_sent_id_offset += len(doc.sentences)
        global_char_offset += len(chunk)
        del doc
    return all_processed_tokens

def process_single_text_spacy(text, filename, state: CreatorModelState, reporter):
    if not text.strip(): return None
    state.nlp_spacy.max_length = 2000000
    chunks = chunk_text_safe(text, chunk_size=15000)
    all_processed_tokens = _ProcessedTokensWithCorefMentions174J1()
    global_mention_id_counter = 1
    global_sent_id_offset = 0
    global_char_offset = 0
    total_chunks = len(chunks)
    reporter.current(0)

    # --- NOWE: GLOBALNY LICZNIK DLA CAŁEGO PLIKU ---
    global_cluster_id_counter = 1

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            global_char_offset += len(chunk)
            continue
        try:
            reporter.status(f"Przetwarzam: {filename} (Część {i + 1}/{total_chunks})"); reporter.tick()
            doc = state.nlp_spacy(chunk)

            chunk_token_offset = len(all_processed_tokens)

            spacy_i_to_global = {
                token.i: chunk_token_offset + local_i
                for local_i, token in enumerate(doc)
            }

        except Exception:
            global_char_offset += len(chunk)
            continue

        sentences = list(doc.sents)
        if not sentences:
            global_char_offset += len(chunk)
            continue

        # --- DODANE: MAPA KOTWIC (HERFERENCE: ROLA + ID) ---
        coref_anchors = {}
        if hasattr(doc._, "coref") and doc._.coref:
            try:
                text_obj = doc._.coref  # api.Text
                for cluster in text_obj.clusters:
                    if not getattr(cluster, "mentions", None):
                        continue

                    c_id_str = str(global_cluster_id_counter)
                    global_cluster_id_counter += 1

                    for mention in cluster.mentions:
                        span = getattr(mention, "span", None)

                        # Fallback: jeśli align nie ustawił span, spróbuj indices
                        if span is None:
                            idx = getattr(mention, "indices", None)
                            if idx and len(idx) == 2:
                                start, end = idx  # end w api.Mention jest inkluzywny
                                # defensywnie: end może być < start, sprawdzamy też granice dokumentu
                                if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(doc):
                                    span = doc[start:end + 1]

                        if not span:
                            continue
                        if _coref_span_crosses_sentence_boundary(span):
                            logging.warning(
                                "[APP corpus.creator.coref_sentence_boundary] "
                                "backend=%r cluster=%r filename=%r span=%r skipped",
                                "herference", c_id_str, filename,
                                str(getattr(span, "text", ""))[:300],
                            )
                            continue

                        # --- Head/Part heurystyka ---
                        mention_tokens = set(span)
                        anchor = None
                        for token in span:
                            # Szukamy korzenia (słowa, którego nadrzędnik jest poza wzmianką)
                            if token.head not in mention_tokens or token.head == token:
                                anchor = token
                                break

                        if anchor is None:
                            anchor = span[-1]

                        mapped_span_tokens = [
                            token for token in span if token.i in spacy_i_to_global
                        ]
                        if len(mapped_span_tokens) != len(span):
                            continue
                        _mention_positions_174j1 = [
                            spacy_i_to_global[token.i] for token in mapped_span_tokens
                        ]
                        if (
                            anchor.i not in spacy_i_to_global
                            or not _positions_are_contiguous(_mention_positions_174j1)
                        ):
                            continue
                        _append_coref_mention_174j1(
                            all_processed_tokens, c_id_str, global_mention_id_counter,
                            min(_mention_positions_174j1), max(_mention_positions_174j1) + 1,
                            spacy_i_to_global[anchor.i],
                        )
                        global_mention_id_counter += 1

                        for token in span:
                            role = "Head" if token.i == anchor.i else "Part"
                            _coref_anchor_append_unique_138c(coref_anchors, token.i, f"{role}-{c_id_str}")

            except Exception as e:
                logging.warning(f"Błąd mapowania koreferencji (herference): {e}")
        # ----------------------------------------

        for sent_idx, sentence in enumerate(sentences, start=1):
            real_sent_id = sent_idx + global_sent_id_offset
            current_progress = (i / total_chunks) + ((sent_idx / len(sentences)) / total_chunks)
            if sent_idx % 20 == 0:
                reporter.current(current_progress)
                reporter.tick()

            # --- SPAKOWANIE DANYCH ---
            for token in sentence:
                start_idx_global = token.idx + global_char_offset
                end_idx_global = start_idx_global + len(token.text) - 1

                coref_val = coref_anchors.get(token.i, [])
                full_nkjp_tag = reconstruct_nkjp_tag(token.tag_, token.morph)

                all_processed_tokens.append({
                    "token": token.text,
                    "lemma": token.lemma_,
                    "sentenceID": real_sent_id,
                    "wordID": token.i + 1,
                    "headID": token.head.i + 1 if token.head != token else 0,
                    "deprel": token.dep_,
                    "postag": full_nkjp_tag,
                    "start": start_idx_global,
                    "end": end_idx_global,
                    "ner": token.ent_type_ if token.ent_type_ else "O",
                    "upos": token.pos_,
                    "coref": coref_val
                })
        global_sent_id_offset += len(sentences)
        global_char_offset += len(chunk)

        del doc
        gc.collect()
    return all_processed_tokens

__all__ = [
    'CreatorModelState',
    '_coref_anchor_append_unique_138c',
    '_coref_span_crosses_sentence_boundary',
    '_positions_are_contiguous',
    'reconstruct_nkjp_tag',
    'initialize_stanza',
    'initialize_spacy',
    'process_single_text',
    'process_single_text_spacy',
]
