# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Any, Dict, Optional, Sequence, Tuple

def split_sentence_operator_query(query: Any) -> Optional[Dict[str, str]]:
    q = str(query or '').strip()
    if '<s' not in q: return None
    token_part, sentence_part = q.split('<s', 1)
    sentence_part = sentence_part.strip()
    if sentence_part.endswith('>'): sentence_part = sentence_part[:-1].strip()
    token_part = token_part.strip()
    if not token_part or not sentence_part: return None
    return {'token_part': token_part, 'sentence_part': sentence_part}

def with_sentence_operator_metadata(plan: Any, query: Any, parse_sentence_conditions) -> Any:
    split = split_sentence_operator_query(query)
    if not split or not isinstance(plan, dict): return plan
    ordered, conditions = parse_sentence_conditions(split['sentence_part'])
    if ordered is None or conditions is None: raise ValueError('invalid <s> sentence conditions')
    out = dict(plan)
    out['sentence_operator'] = {'token_part': split['token_part'], 'sentence_part': split['sentence_part'], 'ordered': bool(ordered), 'conditions': conditions}
    return out

def as_list(value: Any):
    if value is None: return []
    if hasattr(value, 'tolist'):
        try: return value.tolist()
        except Exception: pass
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    if isinstance(value, (str, bytes)): return [value]
    try: return list(value)
    except Exception: return [value]

def sentence_bounds(sentence_ids: Sequence[Any], token_index: int) -> Tuple[int, int]:
    idx = int(token_index); sid = sentence_ids[idx]
    start = idx
    while start > 0 and sentence_ids[start - 1] == sid: start -= 1
    end = idx
    while end < len(sentence_ids) and sentence_ids[end] == sid: end += 1
    return start, end

def text_value_matches(candidate: Any, values: Sequence[Any], operator: str = '=', match_type: str = 'exact') -> bool:
    cand = str(candidate or ''); matched = False
    for raw in values or []:
        val = str(raw or '')
        if match_type == 'exact':
            if cand.lower() == val.lower(): matched = True; break
        elif match_type == 'regex':
            try:
                if re.fullmatch(val, cand, flags=re.IGNORECASE): matched = True; break
            except Exception: pass
        elif match_type == 'regex_search':
            try:
                if re.search(val, cand, flags=re.IGNORECASE): matched = True; break
            except Exception: pass
    return (not matched) if str(operator) == '!=' else matched

def token_attr(doc: Dict[str, Any], key: str, token_index: int) -> Any:
    field = {'orth':'tokens','base':'lemmas','pos':'postags','upos':'upostags','deprel':'deprels','ner':'ners'}.get(str(key))
    if field is None: return None
    seq = as_list(doc.get(field))
    return seq[token_index] if 0 <= token_index < len(seq) else None

def condition_parts(condition: Any):
    if isinstance(condition, dict):
        key = condition.get('key') or condition.get('attr') or condition.get('field')
        values = condition.get('values')
        if values is None:
            value = condition.get('value'); values = [] if value is None else [value]
        return key, values, condition.get('operator') or condition.get('op') or '=', bool(condition.get('is_nested') or condition.get('nested')), condition.get('match_type') or 'exact'
    if isinstance(condition, (tuple, list)):
        if len(condition) >= 5: return condition[0], condition[1], condition[2], condition[3], condition[4]
        if len(condition) >= 4: return condition[0], condition[1], condition[2], condition[3], 'exact'
    return None, [], '=', False, 'exact'

def token_matches_condition_group(doc: Dict[str, Any], token_index: int, group: Any) -> bool:
    conditions = group if isinstance(group, list) else [group]
    for condition in conditions:
        key, values, operator, is_nested, match_type = condition_parts(condition)
        if key is None or is_nested: return False
        value = token_attr(doc, str(key), int(token_index))
        if value is None or not text_value_matches(value, values, str(operator), str(match_type or 'exact')): return False
    return True

def match_pattern_in_range(doc: Dict[str, Any], start_index: int, conditions: Sequence[Any], end_limit: int) -> Optional[int]:
    if not conditions: return int(start_index)
    if int(start_index) >= int(end_limit): return None
    first = conditions[0]
    if isinstance(first, tuple) and first and first[0] == 'repeat':
        base_cond, min_rep, max_rep = first[1], int(first[2]), int(first[3])
        for count in range(max_rep, min_rep - 1, -1):
            idx = int(start_index); ok = True
            for _ in range(count):
                if idx >= int(end_limit) or not token_matches_condition_group(doc, idx, base_cond): ok = False; break
                idx += 1
            if ok:
                rest = match_pattern_in_range(doc, idx, conditions[1:], end_limit)
                if rest is not None: return rest
        return None
    if not token_matches_condition_group(doc, int(start_index), first): return None
    return match_pattern_in_range(doc, int(start_index) + 1, conditions[1:], end_limit)

def sentence_satisfies_conditions(doc: Dict[str, Any], match_start: int, ordered: bool, conditions: Sequence[Any]) -> bool:
    sentence_ids = as_list(doc.get('sentence_ids'))
    if not sentence_ids: return False
    start, end = sentence_bounds(sentence_ids, int(match_start))
    if ordered: return any(match_pattern_in_range(doc, idx, conditions, end) is not None for idx in range(start, end))
    return all(any(token_matches_condition_group(doc, idx, group) for idx in range(start, end)) for group in conditions)

def hit_parts(hit: Any):
    if isinstance(hit, dict): return hit.get('doc_id', hit.get('row_idx')), hit.get('start', hit.get('start_idx')), hit.get('end', hit.get('end_idx'))
    if isinstance(hit, (tuple, list)) and len(hit) >= 3: return hit[0], hit[1], hit[2]
    return None, None, None
