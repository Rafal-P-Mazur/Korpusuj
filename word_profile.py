import math
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Callable, Optional, Iterable


def safe_ll(o: float, e: float) -> float:
    """Bezpieczne log-likelihood bez ryzyka dzielenia przez zero."""
    return o * math.log(o / e) if o > 0 and e > 0 else 0.0


# ==============================================================================
# 1. MODELE DANYCH
# ==============================================================================

@dataclass(frozen=True)
class WordProfileHit:
    row_idx: int
    token_idx: int


@dataclass
class WordProfileRow:
    relation: str
    collocate: str
    cooc_freq: int
    doc_freq: int
    global_freq: int
    log_dice: float
    mi_score: float
    t_score: float
    ll_score: float
    collocate_upos: str = ""
    example_refs: List[Tuple[int, int, int]] = field(default_factory=list)
    display_collocate: str = ""  # <--- NOWE POLE (Tylko do wyświetlania w GUI)


# ==============================================================================
# 2. GRAMATYKI PROFILU SKŁADNIOWEGO
# ==============================================================================

PROFILE_GRAMMARS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "NOUN": {
        "Modyfikatory liczebnikowe (nummod)": {"target_is": "head", "deprels": ["nummod", "nummod:gov"]},
        "Frazy przyimkowe (nmod)": {"target_is": "head", "deprels": ["nmod"], "cascade_case": True, "prepend_case": True},
        "Modyfikatory przymiotnikowe (amod)": {"target_is": "head", "deprels": ["amod"]},
        "Modyfikatory rzeczowne w dopełniaczu (nmod)": {"target_is": "head", "deprels": ["nmod"], "req_case": "Gen", "exclude_child_deprel": ["case"]},
        "Modyfikatory rzeczowne w celowniku (nmod)": {"target_is": "head", "deprels": ["nmod"], "req_case": "Dat", "exclude_child_deprel": ["case"]},
        "Modyfikatory rzeczowne w narzędniku (nmod)": {"target_is": "head", "deprels": ["nmod"], "req_case": "Ins", "exclude_child_deprel": ["case"]},
        "Zdania przydawkowe (acl / acl:relcl)": {"target_is": "head", "deprels": ["acl", "acl:relcl"], "req_upos_in": ["VERB", "AUX", "ADJ"]},
        "Terminy wielowyrazowe i złożenia (fixed / flat / compound)": {"target_is": "head", "deprels": ["fixed", "flat", "compound"]},
        "Wystąpienia w roli podmiotu (nsubj)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"]},
        "Wystąpienia w roli dopełnienia (obj / iobj)": {"target_is": "child", "deprels": ["obj", "iobj"]},
        "Połączenia współrzędne / szeregi (conj)": {"target_is": "symmetric", "deprels": ["conj"], "req_upos": "NOUN"},
        "Apozycje i dopowiedzenia (appos)": {"target_is": "head", "deprels": ["appos"]},
        "Zaimki określające (det)": {"target_is": "head", "deprels": ["det", "det:poss"]},
        "Czym jest? (orzecznik rzeczowny)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "positive", "req_upos": "NOUN"},
        "Jaki jest? (orzecznik przymiotnikowy)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "positive", "req_upos": "ADJ"},
        "Czym nie jest? (orzecznik rzeczowny)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "negative", "req_upos": "NOUN"},
        "Jaki nie jest? (orzecznik przymiotnikowy)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "negative", "req_upos": "ADJ"},
        "Porównania z czasownikiem (obl:cmpr)": {
            "target_is": "sibling",
            "target_deprels": ["nsubj", "nsubj:pass"],
            "deprels": ["obl:cmpr", "obl"],
            "capture_child_lemma_from_deprels": ["mark", "case"],
            "capture_child_lemma_allow": ["jak", "niż", "jakby", "niby"],
            "relation_name_template": "Porównanie '{marker}' (czas.)",
            "exclude_shared_head_child_deprel": ["obj"],
            "exclude_child_deprel": ["advmod:emph", "parataxis"],
            "exclude_child_lemma": ["raz", "razie", "dotąd"],
            "req_head_upos": ["VERB", "AUX"],
            "req_upos_in": ["NOUN", "PROPN", "PRON"],
            "req_case": "Nom"
        },
        "Porównania z przyimkiem 'od' (obl:cmpr)": {
            "target_is": "sibling",
            "target_deprels": ["nsubj", "nsubj:pass"],
            "deprels": ["obl:cmpr", "obl"],
            "capture_child_lemma_from_deprels": ["case"],
            "capture_child_lemma_allow": ["od"],
            "relation_name_template": "Porównanie '{marker}' (przym.)",
            "req_head_upos": ["ADJ", "ADV"],
            "req_head_feature": "Degree=Cmp",
            "req_upos_in": ["NOUN", "PROPN", "PRON"]
        },
        "Porównania ze spójnikiem 'jak/niż' (obl:cmpr)": {
            "target_is": "sibling",
            "target_deprels": ["nsubj", "nsubj:pass"],
            "deprels": ["obl:cmpr", "obl"],
            "capture_child_lemma_from_deprels": ["mark"],
            "capture_child_lemma_allow": ["jak", "niż", "jakby", "niby"],
            "relation_name_template": "Porównanie '{marker}' (przym.)",
            "req_head_upos": ["ADJ", "ADV"],
            "req_upos_in": ["NOUN", "PROPN", "PRON"],
            "req_case": "Nom"
        },
        "Określniki ilościowe (det:nummod / det:numgov)": {
            "target_is": "child",
            "deprels": ["det:nummod", "det:numgov"]
        },
        "Elementy modyfikowane przez rzeczownik (nmod)": {
            "target_is": "child",
            "deprels": ["nmod"],
            "req_upos": "NOUN"
        },
        "Elementy modyfikowane przez rzeczownik (obl)": {
            "target_is": "child",
            "deprels": ["obl", "obl:arg"],
            "req_upos_in": ["VERB", "AUX", "ADJ"]
        },

    },
    "PROPN": {
        "Człony nazwy własnej (flat / fixed / compound)": {"target_is": "head", "deprels": ["flat", "fixed", "compound"]},
        "Modyfikatory liczebnikowe (nummod)": {"target_is": "head", "deprels": ["nummod", "nummod:gov"]},
        "Frazy przyimkowe (nmod)": {"target_is": "head", "deprels": ["nmod"], "cascade_case": True, "prepend_case": True},
        "Modyfikatory przymiotnikowe (amod)": {"target_is": "head", "deprels": ["amod"]},
        "Modyfikatory rzeczowne w dopełniaczu (nmod)": {"target_is": "head", "deprels": ["nmod"], "req_case": "Gen", "exclude_child_deprel": ["case"]},
        "Modyfikatory rzeczowne w celowniku (nmod)": {"target_is": "head", "deprels": ["nmod"], "req_case": "Dat", "exclude_child_deprel": ["case"]},
        "Modyfikatory rzeczowne w narzędniku (nmod)": {"target_is": "head", "deprels": ["nmod"], "req_case": "Ins", "exclude_child_deprel": ["case"]},
        "Zdania przydawkowe (acl / acl:relcl)": {"target_is": "head", "deprels": ["acl", "acl:relcl"], "req_upos_in": ["VERB", "AUX", "ADJ"]},
        "Wystąpienia w roli podmiotu (nsubj)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"]},
        "Wystąpienia w roli dopełnienia (obj / iobj)": {"target_is": "child", "deprels": ["obj", "iobj"]},
        "Połączenia współrzędne / szeregi (conj)": {"target_is": "symmetric", "deprels": ["conj"], "req_upos": "NOUN"},
        "Apozycje i dopowiedzenia (appos)": {"target_is": "head", "deprels": ["appos"]},
        "Zaimki określające (det)": {"target_is": "head", "deprels": ["det", "det:poss"]},
        "Czym jest? (orzecznik rzeczowny)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "positive", "req_upos": "NOUN"},
        "Jaki jest? (orzecznik przymiotnikowy)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "positive", "req_upos": "ADJ"},
        "Czym nie jest? (orzecznik rzeczowny)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "negative", "req_upos": "NOUN"},
        "Jaki nie jest? (orzecznik przymiotnikowy)": {"target_is": "child", "deprels": ["nsubj", "nsubj:pass"], "requires_copula": True, "copula_polarity": "negative", "req_upos": "ADJ"},
        "Porównania z czasownikiem (obl:cmpr)": {
            "target_is": "sibling",
            "target_deprels": ["nsubj", "nsubj:pass"],
            "deprels": ["obl:cmpr", "obl"],
            "capture_child_lemma_from_deprels": ["mark", "case"],
            "capture_child_lemma_allow": ["jak", "niż", "jakby", "niby"],
            "relation_name_template": "Porównanie '{marker}' (czas.)",
            "exclude_shared_head_child_deprel": ["obj"],
            "exclude_child_deprel": ["advmod:emph", "parataxis"],
            "exclude_child_lemma": ["raz", "razie", "dotąd"],
            "req_head_upos": ["VERB", "AUX"],
            "req_upos_in": ["NOUN", "PROPN", "PRON"],
            "req_case": "Nom"
        },
        "Porównania z przyimkiem 'od' (obl:cmpr)": {
            "target_is": "sibling",
            "target_deprels": ["nsubj", "nsubj:pass"],
            "deprels": ["obl:cmpr", "obl"],
            "capture_child_lemma_from_deprels": ["case"],
            "capture_child_lemma_allow": ["od"],
            "relation_name_template": "Porównanie '{marker}' (przym.)",
            "req_head_upos": ["ADJ", "ADV"],
            "req_head_feature": "Degree=Cmp",
            "req_upos_in": ["NOUN", "PROPN", "PRON"]
        },
        "Porównania ze spójnikiem 'jak/niż' (obl:cmpr)": {
            "target_is": "sibling",
            "target_deprels": ["nsubj", "nsubj:pass"],
            "deprels": ["obl:cmpr", "obl"],
            "capture_child_lemma_from_deprels": ["mark"],
            "capture_child_lemma_allow": ["jak", "niż", "jakby", "niby"],
            "relation_name_template": "Porównanie '{marker}' (przym.)",
            "req_head_upos": ["ADJ", "ADV"],
            "req_upos_in": ["NOUN", "PROPN", "PRON"],
            "req_case": "Nom"
        },
        "Określniki ilościowe (det:nummod / det:numgov)": {
            "target_is": "child",
            "deprels": ["det:nummod", "det:numgov"]
        },
        "Elementy modyfikowane przez nazwę własną (nmod)": {
            "target_is": "child",
            "deprels": ["nmod"],
            "req_upos": "NOUN"
        },
        "Elementy modyfikowane przez nazwę własną (obl)": {
            "target_is": "child",
            "deprels": ["obl", "obl:arg"],
            "req_upos_in": ["VERB", "AUX", "ADJ"]
        },
    },

    "VERB": {
        "Podmioty (nsubj)": {"target_is": "head", "deprels": ["nsubj", "nsubj:pass"]},
        "Podmioty zdaniowe (csubj)": {"target_is": "head", "deprels": ["csubj", "csubj:pass"]},
        "Argumenty syntetyczne w bierniku (obj)": {"target_is": "head", "deprels": ["obj"], "req_case": "Acc"},
        "Argumenty syntetyczne w dopełniaczu (obj)": {"target_is": "head", "deprels": ["obj"], "req_case": "Gen"},
        "Argumenty syntetyczne w celowniku (iobj)": {"target_is": "head", "deprels": ["iobj"], "req_case": "Dat"},
        "Argumenty syntetyczne w narzędniku (iobj)": {"target_is": "head", "deprels": ["iobj"], "req_case": "Ins"},
        "Argumenty analityczne (obl:arg)": {"target_is": "head", "deprels": ["obl:arg"], "cascade_case": True},
        "Modyfikatory syntetyczne w narzędniku (obl)": {"target_is": "head", "deprels": ["obl", "obl:arg"], "req_case": "Ins",
                                            "exclude_child_deprel": ["case"]},
        "Modyfikatory syntetyczne w bierniku (obl)": {"target_is": "head", "deprels": ["obl"], "req_case": "Acc", "exclude_child_deprel": ["case"]},
        "Modyfikatory syntetyczne w dopełniaczu (obl)": {"target_is": "head", "deprels": ["obl"], "req_case": "Gen", "exclude_child_deprel": ["case"]},
        "Modyfikatory analityczne (obl)": {"target_is": "head", "deprels": ["obl"], "cascade_case": True},
        "Modyfikatory przysłówkowe (advmod)": {"target_is": "head", "deprels": ["advmod"]},
        "Agens strony biernej (obl:agent)": {
            "target_is": "head",
            "deprels": ["obl", "obl:agent"],
            "requires_child_lemma": ["przez"],
            "capture_child_lemma_from_deprels": ["case", "mark"],
            "capture_child_lemma_allow": ["przez"],
            "relation_name_template": "Agens '{marker}'",
            "prepend_case": True
        },
        "Porównania zdaniowe (advcl / advcl:cmpr)": {
            "target_is": "child",
            "deprels": ["advcl:cmpr", "advcl"],
            "capture_child_lemma_from_deprels": ["mark"],
            "capture_child_lemma_allow": ["jak", "jakby", "niż", "niby"],
            "relation_name_template": "Porównanie '{marker}'"
        },
        "Modyfikatory zdaniowe (advcl)": {
            "target_is": "child",
            "deprels": ["advcl"],
            "exclude_child_lemma": ["jak", "jakby", "niż", "niby"],
            "capture_child_lemma_from_deprels": ["mark"],
            "relation_name_template": "Okolicznik zdaniowy '{marker}'"
        },
        "Argumenty zdaniowe (ccomp / xcomp)": {"target_is": "head", "deprels": ["ccomp", "xcomp"]},
        "Parataksa / Konstrukcje luźno powiązane (parataxis)": {
            "target_is": "symmetric",
            "deprels": ["parataxis"],
            "req_upos": "VERB"
        },
        "Partykuła zwrotna 'się' (expl:pv)": {"target_is": "head", "deprels": ["expl:pv"]},
        "Połączenia współrzędne / szeregi (conj)": {"target_is": "symmetric", "deprels": ["conj"], "req_upos": "VERB"},
        "Dołączenia bezokolicznikowe (xcomp)": {"target_is": "head", "deprels": ["xcomp"], "req_upos": "VERB"},
    },
    "ADJ": {
        "Elementy modyfikowane przez przymiotnik (amod)": {"target_is": "child", "deprels": ["amod"]},
        "Modyfikatory przymiotnika (advmod)": {"target_is": "head", "deprels": ["advmod"]},
        "Wzmacniacze / Intensyfikatory (advmod)": {"target_is": "head", "deprels": ["advmod"], "req_upos": "ADV"},
        "Operatory stopniowania (advmod)": {"target_is": "head", "deprels": ["advmod"], "req_upos": "ADV", "req_lemma": ["bardziej", "mniej", "najbardziej", "najmniej", "coraz"]},
        "Frazy przyimkowe (obl / nmod)": {"target_is": "head", "deprels": ["obl", "nmod"], "cascade_case": True, "prepend_case": True},
        "Punkt odniesienia w porównaniach (obl / nmod / advcl)": {
            "target_is": "head",
            "deprels": ["obl", "nmod", "advcl"],
            "capture_child_lemma_from_deprels": ["case", "mark"],
            "capture_child_lemma_allow": ["niż", "jak", "od", "jakby", "niby"],
            "relation_name_template": "Punkt odniesienia w porównaniach '{marker}'"
        },
        "Połączenia współrzędne / szeregi (conj)": {"target_is": "symmetric", "deprels": ["conj"], "req_upos": "ADJ"},
        "Podmioty orzecznika (nsubj + cop)": {
            "target_is": "child",
            "deprels": ["nsubj", "nsubj:pass"],
            "requires_child_deprel": ["cop"]
        },
    },
    "ADV": {
        "Elementy modyfikowane przez przysłówek (advmod)": {"target_is": "child", "deprels": ["advmod"]},
        "Modyfikatory przysłówka (advmod)": {"target_is": "head", "deprels": ["advmod"]},
        "Operatory stopniowania (advmod)": {"target_is": "head", "deprels": ["advmod"], "req_upos": "ADV", "req_lemma": ["bardziej", "mniej", "najbardziej", "najmniej", "coraz"]},
        "Punkt odniesienia w porównaniach (obl / nmod / advcl)": {
            "target_is": "head",
            "deprels": ["obl", "nmod", "advcl"],
            "capture_child_lemma_from_deprels": ["case", "mark"],
            "capture_child_lemma_allow": ["niż", "jak", "od", "jakby", "niby"],
            "relation_name_template": "Punkt odniesienia w porównaniach '{marker}'"
        }
    },
    "NUM": {
        "Elementy modyfikowane przez liczebnik (nummod / nummod:gov)": {
            "target_is": "child",
            "deprels": ["nummod", "nummod:gov"]
        },
        "Modyfikatory liczebnika (advmod / case)": {
            "target_is": "child",
            "deprels": ["advmod", "case", "dep"]
        },
        "Połączenia współrzędne / szeregi (conj)": {
            "target_is": "symmetric",
            "deprels": ["conj"],
            "req_upos": "NUM"
        }
    },
    "PRON": {
        "Czynności, których jest podmiotem (nsubj)": {"target_is": "head", "deprels": ["nsubj", "nsubj:pass"], "req_upos_in": ["VERB", "AUX"]},
        "Czynności, których jest dopełnieniem (obj / iobj)": {"target_is": "head", "deprels": ["obj", "iobj"], "req_upos_in": ["VERB", "AUX"]},
        "Modyfikatory przymiotnikowe (amod)": {"target_is": "child", "deprels": ["amod"]},
        "Zdania przydawkowe (acl:relcl)": {"target_is": "child", "deprels": ["acl:relcl"]},
    },
}


# ==============================================================================
# 3. ADAPTERY / POMOCNICZE
# ==============================================================================

def get_mwe_phrase(head_idx: int, word_ids: List[Any], children_by_head: Dict[int, List[int]], deprels: List[Any],
                   lemmas: List[Any], ignore_case: bool = True) -> str:
    """Zbiera i skleja wielowyrazowe jednostki (MWE) na podstawie drzewa zależności."""
    mwe_indices = {head_idx}
    queue = [head_idx]
    # Relacje UD oznaczające wielowyrazowe jednostki
    allowed_mwe_deprels = {"flat", "fixed", "compound"}

    while queue:
        current_idx = queue.pop(0)
        # KLUCZOWE: Musimy zapytać o dzieci używając word_id, a nie indeksu tablicy!
        current_word_id = word_ids[current_idx]

        for child_idx in children_by_head.get(current_word_id, []):
            if str(deprels[child_idx]) in allowed_mwe_deprels:
                if child_idx not in mwe_indices:
                    mwe_indices.add(child_idx)
                    queue.append(child_idx)

    # Jeśli nie ma innych członów, zwracamy po prostu jedno słowo
    if len(mwe_indices) == 1:
        part = str(lemmas[head_idx])
        return part.lower() if ignore_case else part

    # Sortujemy indeksy, by zachować oryginalną kolejność słów w zdaniu
    sorted_indices = sorted(list(mwe_indices))

    phrase_parts = []
    for idx in sorted_indices:
        part = str(lemmas[idx])
        phrase_parts.append(part.lower() if ignore_case else part)

    return " ".join(phrase_parts)

def unpack_word_profile_hit(res: Any) -> WordProfileHit:
    return WordProfileHit(row_idx=res[11], token_idx=res[12])


def default_case_extractor(full_postag: str) -> str:
    if not isinstance(full_postag, str):
        return ""
    lowered = full_postag.lower()
    if "case=" in lowered:
        for part in lowered.split("|"):
            if part.startswith("case="):
                return part.split("=")[1].capitalize()[:3]
    parts = lowered.split(":")
    pos = parts[0]
    feats = parts[1:]
    case_index_map = {
        "subst": 1, "depr": 1, "adj": 1, "ppron12": 1, "ppron3": 1,
        "num": 1, "numcol": 1, "ger": 1, "pact": 1, "ppas": 1,
        "siebie": 0, "prep": 0
    }
    if pos in case_index_map:
        idx = case_index_map[pos]
        if idx < len(feats):
            val = feats[idx]
            nkjp_to_ud = {
                "nom": "Nom", "gen": "Gen", "dat": "Dat", "acc": "Acc",
                "inst": "Ins", "ins": "Ins", "loc": "Loc", "voc": "Voc"
            }
            return nkjp_to_ud.get(val, val.capitalize()[:3])
    return ""


def find_sentence_bounds(sentence_ids: List[Any], token_idx: int) -> Tuple[int, int]:
    sent_id = sentence_ids[token_idx]
    start = token_idx
    end = token_idx
    while start > 0 and sentence_ids[start - 1] == sent_id: start -= 1
    while end < len(sentence_ids) and sentence_ids[end] == sent_id: end += 1
    return start, end


# --- ZOPTYMALIZOWANE WYSZUKIWANIE PRZYIMKÓW (O(1) w pętli) ---
def find_preposition_for_token_fast(
        token_word_id: int, children_by_head: Dict[int, List[int]],
        deprels: List[Any], lemmas: List[Any], ignore_case: bool = True
) -> str:
    for child_idx in children_by_head.get(token_word_id, []):
        if deprels[child_idx] == "case":
            prep = str(lemmas[child_idx])
            return prep.lower() if ignore_case else prep
    return ""


# --- INTELIGENTNIEJSZE FILTROWANIE ---
def normalize_lemma(val: Any, ignore_case: bool = True) -> str:
    txt = str(val)
    return txt.lower() if ignore_case else txt

def get_child_indices_for_word_id(
    token_word_id: int,
    children_by_head: Dict[int, List[int]]
) -> List[int]:
    return children_by_head.get(token_word_id, [])

def child_lemmas_for_word_id(
    token_word_id: int,
    children_by_head: Dict[int, List[int]],
    lemmas: List[Any],
    deprels: List[Any],
    ignore_case: bool = True,
    only_deprels: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    allowed = set(only_deprels) if only_deprels else None
    for child_idx in get_child_indices_for_word_id(token_word_id, children_by_head):
        child_dep = str(deprels[child_idx])
        if allowed is not None and child_dep not in allowed:
            continue
        out.append((normalize_lemma(lemmas[child_idx], ignore_case), child_dep, child_idx))
    return out

def first_child_lemma_matching(
    token_word_id: int,
    children_by_head: Dict[int, List[int]],
    lemmas: List[Any],
    deprels: List[Any],
    ignore_case: bool = True,
    only_deprels: Optional[Iterable[str]] = None,
    allow_lemmas: Optional[Iterable[str]] = None,
) -> str:
    allow = {normalize_lemma(x, ignore_case) for x in allow_lemmas} if allow_lemmas else None
    for child_lemma, child_dep, _ in child_lemmas_for_word_id(
        token_word_id=token_word_id,
        children_by_head=children_by_head,
        lemmas=lemmas,
        deprels=deprels,
        ignore_case=ignore_case,
        only_deprels=only_deprels,
    ):
        if allow is None or child_lemma in allow:
            return child_lemma
    return ""

def candidate_word_id_for_rule(
    rule: Dict[str, Any],
    candidate_idx: int,
    target_idx: int,
    word_ids: List[Any],
) -> int:
    return word_ids[candidate_idx]


def check_extended_rule_filters(
        *,
        rule: Dict[str, Any],
        candidate_idx: int,
        target_idx: int,
        word_ids: List[Any],
        head_ids: List[Any],
        children_by_head: Dict[int, List[int]],
        lemmas: List[Any],
        deprels: List[Any],
        upostags: List[Any],
        feats: List[Any],
        idx_by_word_id: Dict[Any, int],  # <--- NOWY ARGUMENT
        ignore_case: bool,
) -> bool:
    cand_lemma = normalize_lemma(lemmas[candidate_idx], ignore_case)
    cand_upos = str(upostags[candidate_idx]).upper()

    if "req_lemma" in rule:
        # Używamy np. id(rule) albo konkretnej nazwy relacji jako klucza
        req_lemmas = get_rule_set(f"req_lemma_{id(rule)}", rule["req_lemma"], ignore_case)
        if cand_lemma not in req_lemmas:
            return False

    if "exclude_lemma" in rule:
        # POBIERAMY Z CACHE
        bad_lemmas = get_rule_set(f"exclude_lemma_{id(rule)}", rule["exclude_lemma"], ignore_case)
        if cand_lemma in bad_lemmas:
            return False

    if "req_upos_in" in rule:
        # POBIERAMY Z CACHE (Uposy w gramatyce są już wielkimi literami, więc dajemy ignore_case=False)
        req_upos = get_rule_set(f"req_upos_in_{id(rule)}", rule["req_upos_in"], ignore_case=False)
        if cand_upos not in req_upos:
            return False

    cand_word_id = candidate_word_id_for_rule(rule, candidate_idx, target_idx, word_ids)

    if "requires_child_lemma" in rule:
        # POBIERAMY Z CACHE
        req_child_lemmas = get_rule_set(f"req_child_lemma_{id(rule)}", rule["requires_child_lemma"], ignore_case)
        # To musi zostać liczone w pętli, bo dzieci kandydata są unikalne dla każdego słowa
        child_lemmas = {
            lemma for lemma, _, _ in child_lemmas_for_word_id(
                cand_word_id, children_by_head, lemmas, deprels, ignore_case
            )
        }
        if child_lemmas.isdisjoint(req_child_lemmas):
            return False

    if "requires_child_deprel" in rule:
        # POBIERAMY Z CACHE (deprele to małe litery z UD, ignorujemy ignore_case)
        req_child_deps = get_rule_set(f"req_child_deprel_{id(rule)}", rule["requires_child_deprel"], ignore_case=False)
        # To musi zostać liczone w pętli
        child_deps = {
            dep for _, dep, _ in child_lemmas_for_word_id(
                cand_word_id, children_by_head, lemmas, deprels, ignore_case
            )
        }
        if child_deps.isdisjoint(req_child_deps):
            return False

    if "requires_child_deprel" in rule:
        # POBIERAMY Z CACHE (deprele to małe litery z UD, ignorujemy ignore_case)
        req_child_deps = get_rule_set(f"req_child_deprel_{id(rule)}", rule["requires_child_deprel"], ignore_case=False)
        # To musi zostać liczone w pętli
        child_deps = {
            dep for _, dep, _ in child_lemmas_for_word_id(
                cand_word_id, children_by_head, lemmas, deprels, ignore_case
            )
        }
        if child_deps.isdisjoint(req_child_deps):
            return False

        # === NOWE FILTRY WYKLUCZAJĄCE ===
    if "exclude_child_lemma" in rule:
        bad_child_lemmas = get_rule_set(f"exclude_child_lemma_{id(rule)}", rule["exclude_child_lemma"], ignore_case)
        child_lemmas = {
            lemma for lemma, _, _ in child_lemmas_for_word_id(
                cand_word_id, children_by_head, lemmas, deprels, ignore_case
            )
        }
        if not child_lemmas.isdisjoint(bad_child_lemmas):
            return False

    if "exclude_child_deprel" in rule:
        bad_child_deps = get_rule_set(f"exclude_child_deprel_{id(rule)}", rule["exclude_child_deprel"],
                                      ignore_case=False)
        child_deps = {
            dep for _, dep, _ in child_lemmas_for_word_id(
                cand_word_id, children_by_head, lemmas, deprels, ignore_case
            )
        }
        if not child_deps.isdisjoint(bad_child_deps):
            return False
    # ================================


    # --- BEZPIECZNE SPRAWDZANIE GŁOWY (NADRZĘDNIKA) ---
    shared_head_word_id = head_ids[candidate_idx]

    if "exclude_shared_head_child_deprel" in rule:
        # POBIERAMY Z CACHE
        bad_deps = get_rule_set(f"exclude_shared_head_child_deprel_{id(rule)}",
                                rule["exclude_shared_head_child_deprel"], ignore_case=False)
        for child_idx in children_by_head.get(shared_head_word_id, []):
            if deprels[child_idx] in bad_deps:
                return False


    # Tłumaczymy word_id na fizyczny indeks na liście
    head_idx = idx_by_word_id.get(shared_head_word_id)

    if "req_head_upos" in rule:
        if head_idx is None:
            return False
        head_upos = str(upostags[head_idx]).upper()
        if head_upos not in {str(x).upper() for x in rule["req_head_upos"]}:
            return False

    if "req_head_feature" in rule:
        if head_idx is None:
            return False
        req_feat = rule["req_head_feature"]
        if req_feat not in str(feats[head_idx]):
            return False

    return True

def build_dynamic_relation_name(
    *,
    relation_name: str,
    rule: Dict[str, Any],
    candidate_idx: int,
    target_idx: int,
    word_ids: List[Any],
    children_by_head: Dict[int, List[int]],
    lemmas: List[Any],
    deprels: List[Any],
    ignore_case: bool,
) -> Optional[str]:
    dynamic_relation_name = relation_name

    if rule.get("cascade_case") is True:
        prep_search_word_id = word_ids[candidate_idx] if rule["target_is"] == "head" else word_ids[target_idx]
        prep = find_preposition_for_token_fast(
            token_word_id=prep_search_word_id,
            children_by_head=children_by_head,
            deprels=deprels,
            lemmas=lemmas,
            ignore_case=ignore_case
        )
        if not prep:
            return None
        dynamic_relation_name = f"{relation_name} '{prep}'"

    if "capture_child_lemma_from_deprels" in rule:
        marker = first_child_lemma_matching(
            token_word_id=word_ids[candidate_idx],
            children_by_head=children_by_head,
            lemmas=lemmas,
            deprels=deprels,
            ignore_case=ignore_case,
            only_deprels=rule["capture_child_lemma_from_deprels"],
            allow_lemmas=rule.get("capture_child_lemma_allow")
        )
        if not marker:
            return None
        tpl = rule.get("relation_name_template", relation_name + " '{marker}'")
        dynamic_relation_name = tpl.format(marker=marker)

    return dynamic_relation_name

def is_valid_collocate(lemma: str, upos: str) -> bool:
    if not lemma or lemma == "_": return False
    # Odrzucanie interpunkcji i symboli (np. emoji)
    if upos in {"PUNCT", "SYM"}: return False
    # Zabezpieczenie przed "śmieciami" z tokenizatora bez żadnej litery (np. "--")
    if not any(ch.isalpha() for ch in lemma): return False
    return True


def match_rule(*, rule: Dict[str, Any], candidate_idx: int, target_idx: int,
               word_ids: List[Any], head_ids: List[Any], deprels: List[Any]) -> bool:
    target_w_id = word_ids[target_idx]
    target_h_id = head_ids[target_idx]
    target_deprel = deprels[target_idx]
    target_is = rule["target_is"]
    allowed_deprels = rule["deprels"]

    if target_is == "head":
        return head_ids[candidate_idx] == target_w_id and deprels[candidate_idx] in allowed_deprels
    if target_is == "child":
        return word_ids[candidate_idx] == target_h_id and target_deprel in allowed_deprels
    if target_is == "symmetric":
        match_a = head_ids[candidate_idx] == target_w_id and deprels[candidate_idx] in allowed_deprels
        match_b = word_ids[candidate_idx] == target_h_id and target_deprel in allowed_deprels
        return match_a or match_b
    if target_is == "sibling":
        # Węzeł badany i kandydujący muszą mieć tego samego rodzica
        same_head = (head_ids[candidate_idx] == target_h_id) and (target_h_id != 0)
        cand_deprel_ok = deprels[candidate_idx] in allowed_deprels
        # Możemy dodatkowo wymusić, jaką relację musi mieć węzeł badany
        target_deprel_ok = target_deprel in rule.get("target_deprels", [target_deprel])
        return same_head and cand_deprel_ok and target_deprel_ok

    return False


def compute_log_dice(cooc_freq: int, target_global_freq: int, collocate_global_freq: int) -> float:
    divisor = target_global_freq + collocate_global_freq
    if cooc_freq <= 0 or divisor <= 0: return 0.0
    return 14 + math.log2((2 * cooc_freq) / divisor)

# Szybki cache prekompilowanych zbiorów dla reguł (żeby nie tworzyć setów w pętli)
_RULE_CACHE = {}

def get_rule_set(rule_id_key, values_list, ignore_case):
    """Zwraca gotowy zbiór set() dla danej reguły, zapamiętując go przy pierwszym użyciu."""
    cache_key = f"{rule_id_key}_{ignore_case}"
    if cache_key not in _RULE_CACHE:
        if ignore_case:
            _RULE_CACHE[cache_key] = {str(x).lower() for x in values_list}
        else:
            _RULE_CACHE[cache_key] = {str(x) for x in values_list}
    return _RULE_CACHE[cache_key]
# ==============================================================================
# 4. GŁÓWNY SILNIK (Zaktualizowana sygnatura i miary)
# ==============================================================================
def compute_word_profile(
        results: Iterable[Any], df, token_freq_dict: Dict[str, int], target_lemma: str,
        total_tokens: int,
        min_freq: int = 2, max_rows_per_relation: Optional[int] = None,
        keep_examples: int = 5, case_extractor: Callable[[str], str] = default_case_extractor,
        ignore_case: bool = True, expand_mwe: bool = False
) -> Dict[str, List[WordProfileRow]]:
    hits = [unpack_word_profile_hit(res) for res in results]
    if not hits: return {}

    counters: Dict[str, Counter] = defaultdict(Counter)
    upos_trackers: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    doc_trackers: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    example_trackers: Dict[str, Dict[str, List[Tuple[int, int, int]]]] = defaultdict(lambda: defaultdict(list))
    seen_signatures = set()

    # NOWOŚĆ: Przechowuje gotowy tekst do wyświetlenia (np. "z wschód") dla czystego lematu (np. "wschód")
    display_map: Dict[str, Dict[str, str]] = defaultdict(dict)

    # NOWOŚĆ: Mapa zapamiętująca głowę frazy do obliczeń statystycznych
    mwe_head_map: Dict[str, str] = {}

    target_lemma_norm = str(target_lemma).lower() if ignore_case else str(target_lemma)
    target_global_freq = token_freq_dict.get(target_lemma_norm, 1)

    # --- NOWOŚĆ: Grupowanie trafień po dokumencie (row_idx) ---
    hits_by_row = defaultdict(list)
    for hit in hits:
        hits_by_row[hit.row_idx].append(hit.token_idx)

    # Główna pętla idzie po unikalnych dokumentach, a nie pojedynczych hitach
    for row_idx, token_indices in hits_by_row.items():

        # 1. POBIERAMY I KONWERTUJEMY DANE TYLKO RAZ NA CAŁY TEKST!
        row_data = df.loc[row_idx]

        word_ids = row_data.word_ids.tolist() if hasattr(row_data.word_ids, "tolist") else row_data.word_ids
        head_ids = row_data.head_ids.tolist() if hasattr(row_data.head_ids, "tolist") else row_data.head_ids
        deprels = row_data.deprels.tolist() if hasattr(row_data.deprels, "tolist") else row_data.deprels
        lemmas = row_data.lemmas.tolist() if hasattr(row_data.lemmas, "tolist") else row_data.lemmas
        upostags = row_data.upostags.tolist() if hasattr(row_data.upostags, "tolist") else row_data.upostags
        full_postags = row_data.full_postags.tolist() if hasattr(row_data.full_postags,
                                                                 "tolist") else row_data.full_postags
        sentence_ids = row_data.sentence_ids.tolist() if hasattr(row_data.sentence_ids,
                                                                 "tolist") else row_data.sentence_ids
        feats = row_data.feats.tolist() if "feats" in df.columns and hasattr(row_data.feats, "tolist") else [""] * len(
            lemmas)

        # 2. Pętla po wszystkich trafieniach w tym JEDNYM tekście
        for target_idx in token_indices:
            target_upos = str(upostags[target_idx]).upper()
            grammar = PROFILE_GRAMMARS.get(target_upos)
            if not grammar: continue

            sent_start, sent_end = find_sentence_bounds(sentence_ids, target_idx)

            # Błyskawiczna mapa dzieci dla całego zdania
            children_by_head = defaultdict(list)
            for i in range(sent_start, sent_end):
                children_by_head[head_ids[i]].append(i)

            idx_by_word_id = {word_ids[i]: i for i in range(sent_start, sent_end)}

            for cand_idx in range(sent_start, sent_end):
                if cand_idx == target_idx: continue

                cand_lemma = str(lemmas[cand_idx])
                if ignore_case: cand_lemma = cand_lemma.lower()

                cand_upos = str(upostags[cand_idx]).upper()
                if not is_valid_collocate(cand_lemma, cand_upos): continue

                for relation_name, rule in grammar.items():
                    if not match_rule(rule=rule, candidate_idx=cand_idx, target_idx=target_idx,
                                      word_ids=word_ids, head_ids=head_ids, deprels=deprels):
                        continue

                    if "req_case" in rule:
                        if case_extractor(str(full_postags[cand_idx])) != rule["req_case"]: continue

                    if "req_upos" in rule:
                        if cand_upos != rule["req_upos"]: continue

                    if "req_upos_in" in rule:
                        if cand_upos not in {str(x).upper() for x in rule["req_upos_in"]}: continue

                    if not check_extended_rule_filters(
                            rule=rule,
                            candidate_idx=cand_idx,
                            target_idx=target_idx,
                            word_ids=word_ids,
                            head_ids=head_ids,  # <--- DODANE
                            children_by_head=children_by_head,
                            lemmas=lemmas,
                            deprels=deprels,
                            upostags=upostags,
                            feats=feats,  # <--- DODANE
                            idx_by_word_id=idx_by_word_id,
                            ignore_case=ignore_case
                    ):
                        continue

                    if rule.get("requires_copula"):
                        cand_word_id = word_ids[cand_idx]
                        has_cop = False
                        is_negated = False

                        # Sprawdzamy dzieci naszego kandydata (orzecznika)
                        for child_idx in children_by_head.get(cand_word_id, []):
                            child_deprel = deprels[child_idx]
                            child_lemma = str(lemmas[child_idx]).lower()

                            # ZMIANA: Uwzględniamy "to" jako pełnoprawny łącznik (copula) w UD
                            if child_deprel == "cop" and child_lemma in ("być", "to"):
                                has_cop = True
                            if child_lemma == "nie":  # wyłapuje partykuły przeczące
                                is_negated = True

                        if not has_cop:
                            continue
                        polarity = rule.get("copula_polarity", "positive")
                        if polarity == "positive" and is_negated:
                            continue
                        if polarity == "negative" and not is_negated:
                            continue


                    # 1. NAJPIERW wyliczamy i sprawdzamy nazwę relacji
                    dynamic_relation_name = build_dynamic_relation_name(
                        relation_name=relation_name,
                        rule=rule,
                        candidate_idx=cand_idx,
                        target_idx=target_idx,
                        word_ids=word_ids,
                        children_by_head=children_by_head,
                        lemmas=lemmas,
                        deprels=deprels,
                        ignore_case=ignore_case
                    )
                    if not dynamic_relation_name: continue

                    # 2. DOPIERO TERAZ ustalamy lematy, bo zmienna 'dynamic_relation_name' już istnieje!
                    if expand_mwe:
                        collocate_text = get_mwe_phrase(cand_idx, word_ids, children_by_head, deprels, lemmas,
                                                        ignore_case)
                    else:
                        collocate_text = cand_lemma

                    # Zapisujemy czysty lemat
                    base_collocate = collocate_text

                    # Generujemy tekst z przyimkiem na pokaz
                    display_colloc = base_collocate
                    if rule.get("prepend_case"):
                        prep = find_preposition_for_token_fast(
                            token_word_id=word_ids[cand_idx], children_by_head=children_by_head,
                            deprels=deprels, lemmas=lemmas, ignore_case=ignore_case
                        )
                        if prep:
                            display_colloc = f"{prep} {base_collocate}"

                    # Zapamiętujemy głowę
                    if base_collocate not in mwe_head_map:
                        mwe_head_map[base_collocate] = cand_lemma

                    # 3. Zapisujemy z użyciem prawidłowej zmiennej
                    display_map[dynamic_relation_name][base_collocate] = display_colloc

                    signature = (row_idx, target_idx, cand_idx, dynamic_relation_name)
                    if signature in seen_signatures: continue
                    seen_signatures.add(signature)

                    # 4. Inkrementacja trackerów
                    counters[dynamic_relation_name][base_collocate] += 1
                    upos_trackers[dynamic_relation_name][base_collocate][cand_upos] += 1
                    doc_trackers[dynamic_relation_name][base_collocate].add(row_idx)

                    refs = example_trackers[dynamic_relation_name][base_collocate]
                    if len(refs) < keep_examples: refs.append((row_idx, target_idx, cand_idx))

    final_profiles: Dict[str, List[WordProfileRow]] = {}
    for dynamic_relation_name, rel_counter in counters.items():
        rows: List[WordProfileRow] = []
        for collocate_text, cooc_freq in rel_counter.items():
            if cooc_freq < min_freq: continue

            head_lemma = mwe_head_map.get(collocate_text, collocate_text)
            global_freq = token_freq_dict.get(head_lemma, 1)
            log_dice = compute_log_dice(cooc_freq, target_global_freq, global_freq)
            doc_freq = len(doc_trackers[dynamic_relation_name].get(collocate_text, set()))

            c_upos = ""
            if upos_trackers[dynamic_relation_name].get(collocate_text):
                c_upos = upos_trackers[dynamic_relation_name][collocate_text].most_common(1)[0][0]

            # --- NOWE OBLICZENIA STATYSTYCZNE ---
            # Obliczanie expected w relacji do węzła centralnego (jak w Word Sketch)
            expected = (target_global_freq * global_freq) / total_tokens if total_tokens > 0 else 1

            mi_score = math.log2(cooc_freq / expected) if cooc_freq > expected else 0
            t_score = (cooc_freq - expected) / math.sqrt(cooc_freq) if cooc_freq > 0 else 0

            O11 = cooc_freq
            O12 = max(0, global_freq - cooc_freq)
            O21 = max(0, target_global_freq - cooc_freq)
            O22 = max(0, total_tokens - global_freq - target_global_freq + cooc_freq)

            E11 = expected
            E12 = (global_freq * (total_tokens - target_global_freq)) / total_tokens if total_tokens > 0 else 1
            E21 = ((total_tokens - global_freq) * target_global_freq) / total_tokens if total_tokens > 0 else 1
            E22 = ((total_tokens - global_freq) * (
                        total_tokens - target_global_freq)) / total_tokens if total_tokens > 0 else 1

            ll_score = 2 * (safe_ll(O11, E11) + safe_ll(O12, E12) + safe_ll(O21, E21) + safe_ll(O22, E22))

            # Wyciągamy tekst z przyimkiem dla wiersza
            disp_text = display_map[dynamic_relation_name].get(collocate_text, collocate_text)

            rows.append(WordProfileRow(
                relation=dynamic_relation_name, collocate=collocate_text, cooc_freq=cooc_freq,
                doc_freq=doc_freq, global_freq=global_freq, log_dice=round(log_dice, 2),
                mi_score=round(mi_score, 2), t_score=round(t_score, 2), ll_score=round(ll_score, 2),
                collocate_upos=c_upos, example_refs=example_trackers[dynamic_relation_name].get(collocate_text, []),
                display_collocate=disp_text  # <--- Brakujący argument!
            ))

        # Domyślnie sortujemy po Log-Dice, tak jak było
        rows.sort(key=lambda r: (r.log_dice, r.cooc_freq, r.doc_freq), reverse=True)
        if max_rows_per_relation is not None: rows = rows[:max_rows_per_relation]
        if rows: final_profiles[dynamic_relation_name] = rows

    return final_profiles


def flatten_word_profile(profile: Dict[str, List[WordProfileRow]]) -> List[WordProfileRow]:
    flat: List[WordProfileRow] = []
    for relation_name, rows in profile.items():
        flat.extend(rows)
    flat.sort(key=lambda r: (r.relation, -r.log_dice, -r.cooc_freq))
    return flat