# -*- coding: utf-8 -*-
"""GUI-free collocation computation for Korpusuj search results.

This module owns pure collocation collection, scoring, sorting and legacy row
conversion. It intentionally contains no GUI imports, no widget access, no
threading, no GUI callback scheduling, and no GUI-state or paginator writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from collections import Counter
import math
import string




@dataclass
class CollocateFilterGroup:
    '''One public CLI collocate-filter group.

    Conditions inside one group are AND-ed; multiple groups are OR-ed.
    pos matches postags[i], with fallback to the first full_postags segment.
    tag matches the tail of full_postags after the first segment.
    '''
    upos: str | None = None
    pos: str | None = None
    tag: str | None = None

@dataclass
class CollocationOptions:
    """Configure lexical or syntactic collocation computation over search matches."""
    mode: str = "Liniowe"
    upos_filter: str = "Wszystkie"
    pos_filter: str = "Wszystkie"
    form_mode: str = "Lemat (base)"
    ignore_case: bool = False
    use_sentence_bound: bool = False
    sort_mode: str = "Log-Likelihood"
    active_feat_filters: dict[str, str] = field(default_factory=dict)
    collocate_filter_groups: list[CollocateFilterGroup] = field(default_factory=list)
    min_freq: int = 1
    min_range: int = 1
    l_span: int = 5
    r_span: int = 5
    syn_dir: str = "Podrzędnik"
    deprel_filter: str = "Wszystkie"


@dataclass
class CollocationRow:
    rank: int
    colloc: str
    fnc: int
    fc: int
    ll: float
    mi: float
    t: float
    log_dice: float


@dataclass
class CollocationTable:
    rows: list[CollocationRow]
    total_actual_slots: int
    total_results: int
    options: CollocationOptions



@dataclass
class CollocateOccurrence:
    """Concrete token occurrence counted as a collocate of a source query match."""

    source_row_idx: int
    source_match_start_idx: int
    source_match_end_idx: int
    source_match_text: str | None
    collocate_idx: int
    collocate_end_idx: int
    collocate: str
    collocate_form: str
    collocate_rank: int | None = None
    mode: str = "linear"
    direction: str = "unknown"
    distance: int | None = None
    deprel: str | None = None


@dataclass
class CollocateOccurrenceTable:
    """Concrete collocate occurrences selected from the same slots as aggregate collocations."""

    rows: list[CollocateOccurrence]
    options: CollocationOptions
    selected_collocates: list[str]
    source_total_results: int

def _safe_ll(o: float, e: float) -> float:
    return o * math.log(o / e) if o > 0 and e > 0 else 0.0


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)  # NaN
    except Exception:
        return False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if hasattr(value, "tolist"):
            return value.tolist()
    except Exception:
        pass
    try:
        return list(value)
    except Exception:
        return []


def get_clean_colloc(word: Any) -> str:
    """Return normalized collocate text or empty string for invalid/punctuation-only items."""
    if _is_na(word):
        return ""
    w = str(word).replace("\u200b", "").replace("\xad", "").replace("\xa0", "").strip()
    if not w or w == "_":
        return ""
    if all(c in string.punctuation or c in "„”«»–—…" for c in w):
        return ""
    return w


def _result_indices(result: Any) -> tuple[Any, int, int]:
    """Return (row_idx, start_idx, end_idx) from the legacy GUI result row shape."""
    if isinstance(result, dict):
        row_idx = result.get("row_idx", result.get("doc_id"))
        start_idx = result.get("start_idx", result.get("start", 0))
        end_idx = result.get("end_idx", result.get("end", int(start_idx) + 1))
        return row_idx, int(start_idx), int(end_idx)
    row_idx, start_idx, end_idx = result[11], result[12], result[13]
    return row_idx, int(start_idx), int(end_idx)


def _row_by_index(df: Any, row_idx: Any) -> Any:
    """Return corpus row for a search-result row index.

    Search result ``row_idx`` / ``doc_id`` values are positional DataFrame row
    numbers produced by the search layer. They must therefore be resolved with
    ``iloc``. Using label-based ``loc`` can read a different row when the
    DataFrame index is non-contiguous, filtered, or otherwise not equal to
    positional row numbers; that breaks syntactic collocation dependency lookup
    and occurrence metadata.
    """
    try:
        idx = int(row_idx)
    except Exception:
        idx = int(float(row_idx))

    if hasattr(df, "iloc"):
        return df.iloc[idx]

    # Fallback for list-like test doubles. This is intentionally positional.
    try:
        return df[idx]
    except Exception:
        # Last-resort compatibility for unusual non-pandas objects. This path
        # should not be used for the normal corpus DataFrame runtime.
        if hasattr(df, "loc"):
            return df.loc[row_idx]
        raise

def _row_value(row_data: Any, name: str, default: Any = None) -> Any:
    try:
        if hasattr(row_data, name):
            return getattr(row_data, name)
    except Exception:
        pass
    try:
        if hasattr(row_data, "get"):
            return row_data.get(name, default)
    except Exception:
        pass
    try:
        return row_data[name]
    except Exception:
        return default


def _token_at(values: Any, idx: int, default: Any = None) -> Any:
    try:
        return values[idx]
    except Exception:
        return default




def _split_full_pos_tag(full_tag: Any) -> tuple[str, str]:
    if full_tag is None:
        return "", ""
    parts = str(full_tag).split(":")
    if not parts:
        return "", ""
    return parts[0], ":".join(parts[1:]) if len(parts) > 1 else ""


def _tag_tail_prefix_matches(actual_tail: str, required_tail: str | None) -> bool:
    if not required_tail:
        return True
    actual_tail = str(actual_tail or "")
    required_tail = str(required_tail or "")
    return actual_tail == required_tail or actual_tail.startswith(required_tail + ":")


def _matches_collocate_filter_group(
    idx: int,
    row_upostags: Any,
    row_postags: Any,
    row_full_postags: Any,
    group: CollocateFilterGroup,
) -> bool:
    if group.upos:
        if row_upostags is None or _token_at(row_upostags, idx) != group.upos:
            return False
    full_pos = ""
    full_tail = ""
    if row_full_postags is not None:
        full_pos, full_tail = _split_full_pos_tag(_token_at(row_full_postags, idx, ""))
    if group.pos:
        row_pos = _token_at(row_postags, idx, None)
        if row_pos is None:
            row_pos = full_pos
        if row_pos != group.pos:
            return False
    if group.tag:
        if not _tag_tail_prefix_matches(full_tail, group.tag):
            return False
    return True

def _matches_filter(
    idx: int,
    row_upostags: Any,
    row_postags: Any,
    row_full_postags: Any,
    options: CollocationOptions,
    feat_mapping: dict[str, dict[str, int]] | None,
) -> bool:
    filter_groups = getattr(options, "collocate_filter_groups", None) or []
    if filter_groups:
        return any(
            _matches_collocate_filter_group(idx, row_upostags, row_postags, row_full_postags, group)
            for group in filter_groups
        )

    upos_filter = options.upos_filter
    pos_filter = options.pos_filter
    active_feat_filters = options.active_feat_filters or {}
    u_match = (upos_filter == "Wszystkie") if row_upostags is None else (
        upos_filter == "Wszystkie" or _token_at(row_upostags, idx) == upos_filter
    )
    p_match = (pos_filter == "Wszystkie" or _token_at(row_postags, idx) == pos_filter)
    if not (u_match and p_match):
        return False
    if active_feat_filters and row_full_postags is not None:
        full_tag = str(_token_at(row_full_postags, idx, ""))
        tag_parts = full_tag.split(":")
        tag_pos = tag_parts[0] if tag_parts else ""
        tag_feats = tag_parts[1:] if len(tag_parts) > 1 else []
        mapping = (feat_mapping or {}).get(tag_pos, {})
        for feat, req_val in active_feat_filters.items():
            if feat in mapping:
                f_idx = mapping[feat]
                if f_idx < len(tag_feats) and tag_feats[f_idx] == req_val:
                    continue
            return False
    return True


def _matches_basic_pos(idx: int, row_upostags: Any, row_postags: Any, options: CollocationOptions) -> bool:
    u_match = (options.upos_filter == "Wszystkie") if row_upostags is None else (
        options.upos_filter == "Wszystkie" or _token_at(row_upostags, idx) == options.upos_filter
    )
    p_match = (options.pos_filter == "Wszystkie" or _token_at(row_postags, idx) == options.pos_filter)
    return bool(u_match and p_match)


def _is_punct(idx: int, upostags: Any, tokens: Any) -> bool:
    try:
        if upostags is not None:
            return _token_at(upostags, idx) == "PUNCT"
        return _token_at(tokens, idx, "") in string.punctuation
    except Exception:
        return False


def _sentence_bounds(start_idx: int, lemmas: list[Any], sentence_ids: Any, use_sentence_bound: bool) -> tuple[int, int]:
    if not use_sentence_bound or sentence_ids is None:
        return 0, len(lemmas)
    try:
        sent_id = sentence_ids[start_idx]
    except Exception:
        return 0, len(lemmas)
    sent_start = start_idx
    while sent_start > 0 and sentence_ids[sent_start - 1] == sent_id:
        sent_start -= 1
    sent_end = start_idx
    while sent_end < len(sentence_ids) and sentence_ids[sent_end] == sent_id:
        sent_end += 1
    return sent_start, sent_end


def _background_frequency(inverted_index: dict[str, Any], form_mode: str, ignore_case: bool) -> tuple[dict[Any, int], int]:
    bg_tf_raw = inverted_index.get("base_tf", {}) if form_mode == "Lemat (base)" else inverted_index.get("orth_tf", {})
    total_tokens = int(inverted_index.get("total_tokens", 1) or 1)
    if total_tokens == 0:
        total_tokens = 1
    if ignore_case:
        bg_tf: dict[str, int] = {}
        for k, v in (bg_tf_raw or {}).items():
            kl = str(k).lower()
            try:
                iv = int(v)
            except Exception:
                iv = 0
            bg_tf[kl] = bg_tf.get(kl, 0) + iv
        return bg_tf, total_tokens
    return dict(bg_tf_raw or {}), total_tokens


def _add_collocate(
    *,
    colloc_counter: Counter,
    colloc_doc_tracker: dict[Any, set],
    colloc_value: Any,
    row_idx: Any,
    ignore_case: bool,
) -> None:
    clean_w = get_clean_colloc(colloc_value)
    if clean_w:
        colloc = clean_w.lower() if ignore_case else clean_w
        colloc_counter[colloc] += 1
        colloc_doc_tracker.setdefault(colloc, set()).add(row_idx)



def _occurrence_text_from_tokens(tokens: list[Any], start_idx: int, end_idx: int) -> str | None:
    try:
        start = max(0, int(start_idx))
        end = max(start, int(end_idx))
        parts = [str(x) for x in list(tokens or [])[start:end] if not _is_na(x)]
        return " ".join(parts) if parts else None
    except Exception:
        return None


def _occurrence_clean_collocate(value: Any, ignore_case: bool = False) -> str | None:
    # Keep occurrence selected-collocate matching aligned with aggregate
    # collocation labels. Some corpus lemmas/tokens contain invisible soft
    # hyphens (U+00AD), e.g. "wy\u00adbu\u00adch"; aggregate rows expose
    # these as plain labels like "wybuch". Remove only invisible word-break
    # formatting characters here, then apply the existing ignore_case rule.
    if _is_na(value):
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    text = text.replace("\u00ad", "")
    if not text:
        return None
    return text.lower() if ignore_case else text


def _occurrence_selected_map(selected_collocates: Any, ignore_case: bool = False) -> tuple[set[str] | None, dict[str, int]]:
    if selected_collocates is None:
        return None, {}
    selected: set[str] = set()
    ranks: dict[str, int] = {}
    try:
        items = list(selected_collocates or [])
    except Exception:
        items = []
    for item in items:
        rank = None
        value = None
        if isinstance(item, dict):
            value = item.get("collocate", item.get("colloc"))
            rank = item.get("rank")
        else:
            value = getattr(item, "colloc", getattr(item, "collocate", item))
            rank = getattr(item, "rank", None)
        label = _occurrence_clean_collocate(value, ignore_case=ignore_case)
        if label is None:
            continue
        selected.add(label)
        try:
            if rank is not None:
                ranks[label] = int(rank)
        except Exception:
            pass
    return selected, ranks


def _append_collocate_occurrence(
    rows: list[CollocateOccurrence],
    *,
    selected: set[str] | None,
    ranks: dict[str, int],
    row_idx: int,
    source_start_idx: int,
    source_end_idx: int,
    source_match_text: str | None,
    collocate_idx: int,
    collocate_value: Any,
    tokens: list[Any],
    options: CollocationOptions,
    mode: str,
    direction: str,
    distance: int | None = None,
    deprel: str | None = None,
) -> None:
    label = _occurrence_clean_collocate(collocate_value, ignore_case=options.ignore_case)
    if label is None or (selected is not None and label not in selected):
        return
    colloc_text = _occurrence_text_from_tokens(tokens, collocate_idx, int(collocate_idx) + 1)
    rows.append(
        CollocateOccurrence(
            source_row_idx=int(row_idx),
            source_match_start_idx=int(source_start_idx),
            source_match_end_idx=int(source_end_idx),
            source_match_text=source_match_text,
            collocate_idx=int(collocate_idx),
            collocate_end_idx=int(collocate_idx) + 1,
            collocate=str(label),
            collocate_form=colloc_text or str(label),
            collocate_rank=ranks.get(str(label)),
            mode=str(mode),
            direction=str(direction),
            distance=distance,
            deprel=deprel,
        )
    )


def collect_collocate_occurrences(
    results: Iterable[Any],
    df: Any,
    options: CollocationOptions,
    selected_collocates: Any = None,
    feat_mapping: dict[str, dict[str, int]] | None = None,
) -> CollocateOccurrenceTable:
    """Collect concrete candidate slots actually counted as collocates.

    This does not search globally for collocate labels. It mirrors the candidate
    slot logic of compute_collocations and returns only query-bound collocate
    occurrences.
    """
    results_list = list(results or [])
    selected, ranks = _occurrence_selected_map(selected_collocates, ignore_case=options.ignore_case)
    rows: list[CollocateOccurrence] = []
    seen_slots: set[Any] = set()

    for res in results_list:
        row_idx, start_idx, end_idx = _result_indices(res)
        row_data = _row_by_index(df, row_idx)
        lemmas = _as_list(_row_value(row_data, "lemmas", []))
        tokens = _as_list(_row_value(row_data, "tokens", []))
        postags = _as_list(_row_value(row_data, "postags", []))
        upostags = _row_value(row_data, "upostags", None)
        if upostags is not None:
            upostags = _as_list(upostags)
        full_postags = _row_value(row_data, "full_postags", None)
        if full_postags is not None:
            full_postags = _as_list(full_postags)
        sentence_ids = _row_value(row_data, "sentence_ids", None)
        if sentence_ids is not None:
            sentence_ids = _as_list(sentence_ids)
        form_array = lemmas if options.form_mode == "Lemat (base)" else tokens
        effective_sentence_bound = True if options.mode == "Składniowe" else options.use_sentence_bound
        sent_start, sent_end = _sentence_bounds(start_idx, lemmas, sentence_ids, effective_sentence_bound)
        source_match_text = _occurrence_text_from_tokens(tokens, start_idx, end_idx)

        if options.mode == "Liniowe":
            window_indices = list(range(max(sent_start, start_idx - int(options.l_span)), start_idx)) + list(range(end_idx, min(sent_end, end_idx + int(options.r_span))))
            for i in window_indices:
                if (row_idx, i) in seen_slots:
                    continue
                seen_slots.add((row_idx, i))
                if _is_punct(i, upostags, tokens):
                    continue
                if not _matches_filter(i, upostags, postags, full_postags, options, feat_mapping):
                    continue
                direction = "left" if i < start_idx else "right"
                try:
                    distance = int(start_idx - i) if i < start_idx else int(i - end_idx + 1)
                except Exception:
                    distance = None
                _append_collocate_occurrence(rows, selected=selected, ranks=ranks, row_idx=row_idx, source_start_idx=start_idx, source_end_idx=end_idx, source_match_text=source_match_text, collocate_idx=i, collocate_value=_token_at(form_array, i), tokens=tokens, options=options, mode="linear", direction=direction, distance=distance, deprel=None)
        else:
            word_ids = _as_list(_row_value(row_data, "word_ids", []))
            head_ids = _as_list(_row_value(row_data, "head_ids", []))
            deprels = _as_list(_row_value(row_data, "deprels", []))
            for i in range(start_idx, end_idx):
                w_id = _token_at(word_ids, i)
                h_id = _token_at(head_ids, i, 0)
                if options.syn_dir in ["Nadrzędnik", "Oba"] and h_id != 0:
                    for j in range(sent_start, sent_end):
                        if _token_at(word_ids, j) == h_id and j not in range(start_idx, end_idx):
                            if (row_idx, j) in seen_slots:
                                continue
                            seen_slots.add((row_idx, j))
                            dep = _token_at(deprels, i)
                            if options.deprel_filter != "Wszystkie" and dep != options.deprel_filter:
                                continue
                            if _is_punct(j, upostags, tokens):
                                continue
                            if not _matches_filter(j, upostags, postags, full_postags, options, feat_mapping):
                                continue
                            _append_collocate_occurrence(rows, selected=selected, ranks=ranks, row_idx=row_idx, source_start_idx=start_idx, source_end_idx=end_idx, source_match_text=source_match_text, collocate_idx=j, collocate_value=_token_at(form_array, j), tokens=tokens, options=options, mode="syntactic", direction="head", distance=None, deprel=str(dep) if dep is not None else None)
                if options.syn_dir in ["Podrzędnik", "Oba"]:
                    for j in range(sent_start, sent_end):
                        if j in range(start_idx, end_idx):
                            continue
                        if _token_at(head_ids, j) != w_id:
                            continue
                        if (row_idx, j) in seen_slots:
                            continue
                        seen_slots.add((row_idx, j))
                        dep = _token_at(deprels, j)
                        if options.deprel_filter != "Wszystkie" and dep != options.deprel_filter:
                            continue
                        if _is_punct(j, upostags, tokens):
                            continue
                        if not _matches_filter(j, upostags, postags, full_postags, options, feat_mapping):
                            continue
                        _append_collocate_occurrence(rows, selected=selected, ranks=ranks, row_idx=row_idx, source_start_idx=start_idx, source_end_idx=end_idx, source_match_text=source_match_text, collocate_idx=j, collocate_value=_token_at(form_array, j), tokens=tokens, options=options, mode="syntactic", direction="dependent", distance=None, deprel=str(dep) if dep is not None else None)

    return CollocateOccurrenceTable(rows=rows, options=options, selected_collocates=(sorted(selected) if selected is not None else []), source_total_results=len(results_list))

def compute_collocations(
    results: Iterable[Any],
    df: Any,
    inverted_index: dict[str, Any],
    options: CollocationOptions,
    feat_mapping: dict[str, dict[str, int]] | None = None,
) -> CollocationTable:
    """Compute collocation table from concordance result rows.

    The implementation intentionally preserves the legacy GUI semantics of
    engine.calculate_collocs, including result tuple indices, seen_slots
    deduplication, total_actual_slots behavior, score formulas, and row sorting.
    """
    results_list = list(results or [])
    colloc_counter: Counter = Counter()
    colloc_doc_tracker: dict[Any, set] = {}
    total_actual_slots = 0
    seen_slots: set[Any] = set()

    for res in results_list:
        row_idx, start_idx, end_idx = _result_indices(res)
        row_data = _row_by_index(df, row_idx)
        lemmas = _as_list(_row_value(row_data, "lemmas", []))
        tokens = _as_list(_row_value(row_data, "tokens", []))
        postags = _as_list(_row_value(row_data, "postags", []))
        upostags = _row_value(row_data, "upostags", None)
        if upostags is not None:
            upostags = _as_list(upostags)
        sentence_ids = _row_value(row_data, "sentence_ids", None)
        if sentence_ids is not None:
            sentence_ids = _as_list(sentence_ids)
        form_array = lemmas if options.form_mode == "Lemat (base)" else tokens
        effective_sentence_bound = True if options.mode == "Składniowe" else options.use_sentence_bound
        sent_start, sent_end = _sentence_bounds(start_idx, lemmas, sentence_ids, effective_sentence_bound)

        if options.mode == "Liniowe":
            window_indices = list(range(max(sent_start, start_idx - int(options.l_span)), start_idx)) + list(
                range(end_idx, min(sent_end, end_idx + int(options.r_span)))
            )
            for i in window_indices:
                if (row_idx, i) in seen_slots:
                    continue
                seen_slots.add((row_idx, i))
                is_punct = _is_punct(i, upostags, tokens)
                total_actual_slots += 1
                if not is_punct:
                    full_postags = _row_value(row_data, "full_postags", None)
                    if full_postags is not None:
                        full_postags = _as_list(full_postags)
                    if _matches_filter(i, upostags, postags, full_postags, options, feat_mapping):
                        _add_collocate(
                            colloc_counter=colloc_counter,
                            colloc_doc_tracker=colloc_doc_tracker,
                            colloc_value=_token_at(form_array, i),
                            row_idx=row_idx,
                            ignore_case=options.ignore_case,
                        )
        else:
            word_ids = _as_list(_row_value(row_data, "word_ids", []))
            head_ids = _as_list(_row_value(row_data, "head_ids", []))
            deprels = _as_list(_row_value(row_data, "deprels", []))
            for i in range(start_idx, end_idx):
                w_id = _token_at(word_ids, i)
                h_id = _token_at(head_ids, i, 0)
                if options.syn_dir in ["Nadrzędnik", "Oba"] and h_id != 0:
                    for j in range(sent_start, sent_end):
                        if _token_at(word_ids, j) == h_id and j not in range(start_idx, end_idx):
                            if (row_idx, j) in seen_slots:
                                continue
                            seen_slots.add((row_idx, j))
                            if options.deprel_filter == "Wszystkie" or _token_at(deprels, i) == options.deprel_filter:
                                is_punct = _is_punct(j, upostags, tokens)
                                if not is_punct:
                                    total_actual_slots += 1
                                    full_postags = _row_value(row_data, "full_postags", None)
                                    if full_postags is not None:
                                        full_postags = _as_list(full_postags)
                                    if _matches_filter(j, upostags, postags, full_postags, options, feat_mapping):
                                        _add_collocate(
                                            colloc_counter=colloc_counter,
                                            colloc_doc_tracker=colloc_doc_tracker,
                                            colloc_value=_token_at(form_array, j),
                                            row_idx=row_idx,
                                            ignore_case=options.ignore_case,
                                        )
                if options.syn_dir in ["Podrzędnik", "Oba"]:
                    for j in range(sent_start, sent_end):
                        if _token_at(head_ids, j) == w_id and j not in range(start_idx, end_idx):
                            if (row_idx, j, "dep") in seen_slots:
                                continue
                            seen_slots.add((row_idx, j, "dep"))
                            if options.deprel_filter == "Wszystkie" or _token_at(deprels, j) == options.deprel_filter:
                                is_punct = _is_punct(j, upostags, tokens)
                                if not is_punct:
                                    total_actual_slots += 1
                                    # Preserve legacy behavior: dependent branch checks only UPOS/POS,
                                    # not active full_postags feature filters.
                                    if _matches_basic_pos(j, upostags, postags, options):
                                        _add_collocate(
                                            colloc_counter=colloc_counter,
                                            colloc_doc_tracker=colloc_doc_tracker,
                                            colloc_value=_token_at(form_array, j),
                                            row_idx=row_idx,
                                            ignore_case=options.ignore_case,
                                        )

    bg_tf, total_tokens = _background_frequency(inverted_index, options.form_mode, options.ignore_case)
    fn = len(results_list)
    if total_actual_slots == 0:
        total_actual_slots = 1

    rows: list[CollocationRow] = []
    for colloc, fnc in colloc_counter.items():
        if fnc < int(options.min_freq) or len(colloc_doc_tracker.get(colloc, set())) < int(options.min_range):
            continue
        fc = int(bg_tf.get(colloc, 1) or 1)
        expected = (total_actual_slots * fc) / total_tokens
        if expected <= 0:
            continue
        mi_score = math.log2(fnc / expected)
        t_score = (fnc - expected) / math.sqrt(fnc) if fnc > 0 else 0
        log_dice = 14 + math.log2((2 * fnc) / (fn + fc))
        O11 = fnc
        O12 = max(0, fc - fnc)
        O21 = max(0, total_actual_slots - fnc)
        O22 = max(0, total_tokens - fc - total_actual_slots + fnc)
        E11 = expected
        E12 = (fc * (total_tokens - total_actual_slots)) / total_tokens
        E21 = ((total_tokens - fc) * total_actual_slots) / total_tokens
        E22 = ((total_tokens - fc) * (total_tokens - total_actual_slots)) / total_tokens
        ll_score = 2 * (_safe_ll(O11, E11) + _safe_ll(O12, E12) + _safe_ll(O21, E21) + _safe_ll(O22, E22))
        rows.append(
            CollocationRow(
                rank=0,
                colloc=str(colloc),
                fnc=int(fnc),
                fc=int(fc),
                ll=round(ll_score, 2),
                mi=round(mi_score, 2),
                t=round(t_score, 2),
                log_dice=round(log_dice, 2),
            )
        )

    sort_attr = {
        "Log-Likelihood": "ll",
        "MI Score": "mi",
        "T-score": "t",
        "Log-Dice": "log_dice",
    }.get(options.sort_mode, "ll")
    rows.sort(key=lambda r: getattr(r, sort_attr), reverse=True)
    for i, row in enumerate(rows):
        row.rank = i + 1
    return CollocationTable(rows=rows, total_actual_slots=total_actual_slots, total_results=fn, options=options)


def collocation_table_to_legacy_rows(table: CollocationTable) -> list[list[Any]]:
    """Return GUI/export-compatible rows: [rank, colloc, fnc, fc, ll, mi, t, log_dice]."""
    return [[r.rank, r.colloc, r.fnc, r.fc, r.ll, r.mi, r.t, r.log_dice] for r in table.rows]
