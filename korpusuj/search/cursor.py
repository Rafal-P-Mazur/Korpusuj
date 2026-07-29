# -*- coding: utf-8 -*-
"""Lazy search cursors for discovering, counting, paging and materializing query matches."""
from __future__ import annotations
# KORPUSUJ_PATCH_138Y_INLINE_COREF_SEMANTICS_INTO_CURSOR_BEGIN
# Coref CQL runtime helpers inlined from korpusuj/search/coref_legacy_semantics_138i3.py.
# 138y deliberately preserves existing helper names and semantics.
# Do not rename _138i3/_138t2 helpers in this patch; do that only after 138x is green.
# COREF_CQL_SEARCHCURSOR_ENABLED helper
import re


# KORPUSUJ_PATCH_145C1_SAFE_DIAGNOSTICS_IMPORT
try:
    from korpusuj.search.diagnostics import (
        korpusuj_diagnostics_enabled_145c1,
        korpusuj_verbose_diagnostics_enabled_145c1,
    )
except Exception:
    def korpusuj_diagnostics_enabled_145c1(config_obj=None):
        return False
    def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
        return False
# END KORPUSUJ_PATCH_145C1_SAFE_DIAGNOSTICS_IMPORT
COREF_CQL_SEARCHCURSOR_ENABLED = True
_INVALID = {'', '0', 'O', '_', 'None', 'none', 'nan', 'NaN', '[]', None}
def _as_list(v):
    if v is None: return []
    if isinstance(v, list): return v
    if isinstance(v, tuple): return list(v)
    if isinstance(v, set): return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s in _INVALID: return []
        if s.startswith('[') and s.endswith(']'): return [x.strip().strip("'\\\"") for x in re.split(r'[,;]', s[1:-1]) if x.strip()]
        if ',' in s or ';' in s: return [x.strip() for x in re.split(r'[,;]', s) if x.strip()]
        return [s]
    return [v]
def _parse_coref_key(key):
    k = str(key or '').strip().replace(' ', '').lower()
    if k == 'coref': return None, False
    if k == 'coref(h)': return 'Head', False
    if k == 'coref(p)': return 'Part', False
    if k == 'coref(m)': return None, True
    return None
def _is_coref_condition(cond):
    if isinstance(cond, dict): key = cond.get('key', cond.get('attr', cond.get('field')))
    elif isinstance(cond, (list, tuple)) and cond: key = cond[0]
    else: key = getattr(cond, 'key', getattr(cond, 'attr', getattr(cond, 'field', None)))
    return _parse_coref_key(key) is not None
def condition_parts_138i3(cond):
    key = value = None; op = '='; mode = 'exact'
    if isinstance(cond, dict):
        key = cond.get('key', cond.get('attr', cond.get('field'))) ; value = cond.get('value', cond.get('val', cond.get('pattern')))
        op = cond.get('op', cond.get('operator', op)) or op ; mode = cond.get('match_type', cond.get('mode', mode)) or mode
    elif isinstance(cond, (list, tuple)):
        if len(cond) >= 1: key = cond[0]
        if len(cond) >= 3 and str(cond[1]) in {'=', '!=', '<>', '~', '!~', 'regex', 'regex_search'}: op, value = cond[1], cond[2]
        elif len(cond) >= 2: value = cond[1]
        if len(cond) >= 4: mode = cond[3]
    else:
        key = getattr(cond, 'key', getattr(cond, 'attr', getattr(cond, 'field', None)))
        value = getattr(cond, 'value', getattr(cond, 'val', getattr(cond, 'pattern', None)))
        op = getattr(cond, 'op', getattr(cond, 'operator', op)) or op ; mode = getattr(cond, 'match_type', getattr(cond, 'mode', mode)) or mode
    if str(op) in {'~', 'regex'}: mode, op = 'regex', '='
    elif str(op) == '!~': mode, op = 'regex', '!='
    return key, value, str(op), str(mode)
def _parse_coref_label(label):
    if label in _INVALID: return None
    s = str(label).strip()
    if not s or s in _INVALID: return None
    if '-' in s:
        role, cid = s.split('-', 1); role, cid = role.strip(), cid.strip()
        if not cid or cid in _INVALID: return None
        rl = role.lower()
        if rl.startswith('head'): role = 'Head'
        elif rl.startswith('part'): role = 'Part'
        elif rl.startswith('mention'): role = 'Mention'
        return role, cid
    return '', s
def _coref_token_cluster_ids(corefs, idx, required_role=None):
    ids = set()
    try: labs = _as_list(corefs[int(idx)])
    except Exception: labs = []
    for lab in labs:
        parsed = _parse_coref_label(lab)
        if not parsed: continue
        role, cid = parsed
        if required_role and role != required_role: continue
        ids.add(str(cid))
    return ids
def _build_coref_clusters(tokens, lemmas, corefs):
    clusters = {}; n = max(len(tokens or []), len(lemmas or []), len(corefs or []))
    for i in range(n):
        tok = tokens[i] if i < len(tokens or []) else None; lem = lemmas[i] if i < len(lemmas or []) else None
        for cid in _coref_token_cluster_ids(corefs, i):
            b = clusters.setdefault(str(cid), set())
            for v in (tok, lem):
                if v is not None:
                    s = str(v).strip().lower()
                    if s: b.add(s)
    return clusters
def _vals(v): return [str(x).strip().strip("'\"").lower() for x in _as_list(v) if str(x).strip()]
def _match(words, vals, mode='exact'):
    mode = str(mode or 'exact').lower()
    for raw in vals:
        if mode in {'regex', 'regex_search'}:
            try:
                rx = re.compile(raw, re.I)
                if any((rx.fullmatch(w) if mode == 'regex' else rx.search(w)) for w in words): return True
            except Exception: pass
        elif raw in words: return True
    return False
def _get_doc(cursor_obj, doc_id):
    try:
        if hasattr(cursor_obj, 'get_doc'): return cursor_obj.get_doc(int(doc_id))
        if hasattr(cursor_obj, '_get_doc_cached_036l4g7'): return cursor_obj._get_doc_cached_036l4g7(int(doc_id))
        return getattr(cursor_obj, 'index').get_doc(int(doc_id))
    except Exception: return {}
def _get_coref_doc_arrays(cursor_obj, doc_id):
    doc = _get_doc(cursor_obj, doc_id)
    if not isinstance(doc, dict): doc = {}
    tokens = doc.get('tokens') or doc.get('orths') or []; lemmas = doc.get('lemmas') or doc.get('bases') or []; corefs = doc.get('corefs') or []
    if not corefs:
        try:
            idx = getattr(cursor_obj, 'index', None)
            if idx is not None and hasattr(idx, 'get_corefs_138i3'): corefs = idx.get_corefs_138i3(int(doc_id)) or []
        except Exception: pass
    return tokens, lemmas, corefs
def _match_coref_condition_at_pos(cursor_obj, cond, doc_id, pos):
    key, value, op, mode = condition_parts_138i3(cond); parsed = _parse_coref_key(key)
    if parsed is None: return None
    role, _expand = parsed; tokens, lemmas, corefs = _get_coref_doc_arrays(cursor_obj, int(doc_id))
    try: pos = int(pos)
    except Exception: return False
    cids = _coref_token_cluster_ids(corefs, pos, role)
    if not cids: return op in {'!=', '<>'}
    clusters = _build_coref_clusters(tokens, lemmas, corefs)
    found = any(_match(clusters.get(cid, set()), _vals(value), mode) for cid in cids)
    return (not found) if op in {'!=', '<>'} else found
def _coref_condition_positions(cursor_obj, cond, doc_id):
    if not _is_coref_condition(cond): return None
    tokens, lemmas, corefs = _get_coref_doc_arrays(cursor_obj, int(doc_id)); n = max(len(tokens or []), len(lemmas or []), len(corefs or []))
    return {pos for pos in range(n) if _match_coref_condition_at_pos(cursor_obj, cond, doc_id, pos)}
def iter_index_doc_ids_138i3(cursor_obj):
    idx = getattr(cursor_obj, 'index', None)
    if idx is None: return []
    con = getattr(idx, 'con', None)
    if con is not None:
        try: return [int(r[0]) for r in con.execute('SELECT doc_id FROM docs').fetchall()]
        except Exception: pass
    total = getattr(idx, 'total_docs', None) or getattr(cursor_obj, 'total_docs', None)
    return list(range(int(total))) if total else []
def _coref_condition_postings(cursor_obj, cond):
    if not _is_coref_condition(cond): return None
    out = {}
    for doc_id in iter_index_doc_ids_138i3(cursor_obj):
        pos = _coref_condition_positions(cursor_obj, cond, int(doc_id))
        if pos: out[int(doc_id)] = sorted(int(p) for p in pos)
    return out

# COREF_CQL_SAME_CLUSTER_ROLE_SEMANTICS_138T2
# Implements documented same-cluster role semantics for coref/coref(H)/coref(P)/coref(M).
# Composite labels like "Part-52' 'Head-65' 'Part-89" are split into separate
# role/id pairs, but IDs are NOT connected transitively. A match requires the
# requested value to be present in the SAME cluster id as the current role.
try:
    _orig_match_coref_condition_pos_138t2 = _match_coref_condition_at_pos
except Exception:
    _orig_match_coref_condition_pos_138t2 = None
try:
    _orig_coref_condition_positions_138t2 = _coref_condition_positions
except Exception:
    _orig_coref_condition_positions_138t2 = None
try:
    _orig_coref_condition_postings_138t2 = _coref_condition_postings
except Exception:
    _orig_coref_condition_postings_138t2 = None

import re as _re_138t2

_COREF_LABEL_RE = _re_138t2.compile(r"\b(Head|Part|Mention|H|P|M)\s*-\s*([0-9A-Za-z_.:-]+)", _re_138t2.I)


def _coref_attr_kind(cond):
    attr = str((cond or {}).get("attr") or (cond or {}).get("key") or "").strip().replace(" ", "")
    low = attr.lower()
    if low == "coref":
        return "bare"
    if low.startswith("coref(") and low.endswith(")"):
        return low[6:-1].strip().lower()
    return None


def _coref_role_allowed(kind):
    if kind in {"h", "head"}:
        return {"head"}
    if kind in {"p", "part"}:
        return {"part"}
    if kind in {"m", "mention"}:
        return {"mention"}
    if kind == "bare":
        return {"head", "part", "mention"}
    return set()


def _coref_condition_values(cond):
    vals = (cond or {}).get("values")
    if vals is None:
        val = (cond or {}).get("value")
        vals = [] if val is None else [val]
    return [str(v).strip() for v in vals if str(v).strip()]


def _coref_condition_is_positive(cond):
    op = str((cond or {}).get("op") or (cond or {}).get("operator") or (cond or {}).get("match_op") or "=").strip()
    if op in {"!=", "not"}:
        return False
    if bool((cond or {}).get("negated") or (cond or {}).get("negative")):
        return False
    return True


def _coref_as_label_strings(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            out.extend(_coref_as_label_strings(x))
        return out
    s = str(v).strip()
    if not s or s in {"0", "O", "_", "None", "none", "[]"}:
        return []
    return [s]


def _parse_coref_components(v):
    out = []
    for lab in _coref_as_label_strings(v):
        for m in _COREF_LABEL_RE.finditer(lab):
            role = m.group(1).lower()
            cid = m.group(2).strip("'\" ,;[]()")
            if role == "h":
                role = "head"
            elif role == "p":
                role = "part"
            elif role == "m":
                role = "mention"
            if cid:
                out.append({"role": role, "cid": cid, "raw": lab})
    return out


def _coref_get_doc(cursor, doc_id):
    for obj in (getattr(cursor, "index", None), cursor):
        if obj is None:
            continue
        for name in ("get_doc", "doc", "get_document"):
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    d = fn(int(doc_id))
                    if d is not None:
                        return d
                except Exception:
                    pass
    return None


def _coref_get_corefs(cursor, doc_id):
    for obj in (getattr(cursor, "index", None), cursor):
        if obj is None:
            continue
        for name in ("get_corefs_138i3", "get_corefs", "corefs"):
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    v = fn(int(doc_id))
                    if v is not None:
                        return v
                except Exception:
                    pass
    d = _coref_get_doc(cursor, doc_id)
    if isinstance(d, dict):
        return d.get("corefs") or []
    return []


def _coref_doc_arrays(cursor, doc_id):
    d = _coref_get_doc(cursor, doc_id)
    if isinstance(d, dict):
        toks = d.get("tokens") or d.get("orths") or d.get("orth") or []
        lems = d.get("lemmas") or d.get("bases") or d.get("base") or []
        return toks or [], lems or []
    return [], []


def _coref_text_matches(text, values, match_type="exact"):
    s = str(text or "")
    sl = s.lower()
    for v in values:
        vl = str(v or "").lower()
        if not vl:
            continue
        if match_type in {"regex", "regex_search"}:
            try:
                if _re_138t2.search(str(v), s, flags=_re_138t2.I):
                    return True
            except Exception:
                pass
        elif sl == vl:
            return True
    return False


def _coref_cluster_has_value(cursor, doc_id, cid, values, match_type="exact"):
    corefs = _coref_get_corefs(cursor, doc_id) or []
    toks, lems = _coref_doc_arrays(cursor, doc_id)
    for p, labs in enumerate(corefs):
        comps = _parse_coref_components(labs)
        if not any(c.get("cid") == cid for c in comps):
            continue
        candidates = []
        if p < len(toks):
            candidates.append(toks[p])
        if p < len(lems):
            candidates.append(lems[p])
        for val in candidates:
            if _coref_text_matches(val, values, match_type=match_type):
                return True
    return False


def _match_coref_same_cluster_role(cursor, cond, doc_id, pos):
    kind = _coref_attr_kind(cond)
    allowed = _coref_role_allowed(kind)
    if not allowed:
        return None
    if not _coref_condition_is_positive(cond):
        return None
    values = _coref_condition_values(cond)
    if not values:
        return False
    corefs = _coref_get_corefs(cursor, doc_id) or []
    try:
        current = _parse_coref_components(corefs[int(pos)])
    except Exception:
        return False
    if not current:
        return False
    match_type = (cond or {}).get("match_type") or "exact"
    for comp in current:
        if comp.get("role") in allowed:
            if _coref_cluster_has_value(cursor, doc_id, comp.get("cid"), values, match_type=match_type):
                return True
    return False


def _match_coref_condition_at_pos(cursor, cond, doc_id, pos):
    """Check whether a token position satisfies the requested same-cluster coreference role."""
    try:
        r = _match_coref_same_cluster_role(cursor, cond, doc_id, pos)
        if r is not None:
            return bool(r)
    except Exception:
        pass
    if callable(_orig_match_coref_condition_pos_138t2):
        return _orig_match_coref_condition_pos_138t2(cursor, cond, doc_id, pos)
    return None


def _coref_condition_positions(cursor, cond, doc_id):
    try:
        kind = _coref_attr_kind(cond)
        if _coref_role_allowed(kind) and _coref_condition_is_positive(cond):
            corefs = _coref_get_corefs(cursor, doc_id) or []
            out = set()
            for p in range(len(corefs)):
                try:
                    if _match_coref_condition_at_pos(cursor, cond, doc_id, p):
                        out.add(int(p))
                except Exception:
                    pass
            return out
    except Exception:
        pass
    if callable(_orig_coref_condition_positions_138t2):
        return _orig_coref_condition_positions_138t2(cursor, cond, doc_id)
    return None


def _coref_iter_doc_ids(cursor):
    idx = getattr(cursor, "index", None)
    for obj in (idx, cursor):
        if obj is None:
            continue
        for name in ("iter_doc_ids", "doc_ids", "get_doc_ids"):
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    return list(fn())
                except Exception:
                    pass
    con = getattr(idx, "con", None) or getattr(idx, "conn", None) or getattr(idx, "connection", None)
    if con is not None:
        try:
            return [int(r[0]) for r in con.execute("SELECT doc_id FROM docs")]
        except Exception:
            pass
    return []


def _coref_condition_postings(cursor, cond):
    try:
        kind = _coref_attr_kind(cond)
        if _coref_role_allowed(kind) and _coref_condition_is_positive(cond):
            merged = {}
            for doc_id in _coref_iter_doc_ids(cursor):
                ps = sorted(_coref_condition_positions(cursor, cond, doc_id) or [])
                if ps:
                    merged[int(doc_id)] = ps
            return merged
    except Exception:
        pass
    if callable(_orig_coref_condition_postings_138t2):
        return _orig_coref_condition_postings_138t2(cursor, cond)
    return None
# KORPUSUJ_PATCH_173B_OPTIMIZE_COREF_DOCUMENT_POSTFILTER_BEGIN
# Build one reusable coreference index per document.  The former 138t2 path
# called _coref_cluster_has_value() for every token position, and that helper
# rescanned every token in the document.  This preserves the documented
# same-cluster role semantics while making each document linear to index.

_COREF_DOC_INDEX_CACHE_LIMIT_173B = 128


def _coref_normalize_role_173b(role):
    low = str(role or "").strip().lower()
    if low in {"h", "head"}:
        return "head"
    if low in {"p", "part"}:
        return "part"
    if low in {"m", "mention"}:
        return "mention"
    return low


def _coref_doc_index_cache_173b(cursor):
    cache = getattr(cursor, "_coref_doc_index_cache_173b", None)
    if cache is None:
        cache = {}
        try:
            cursor._coref_doc_index_cache_173b = cache
        except Exception:
            pass
    return cache


def _build_coref_document_index_173b(cursor, doc_id):
    """Return normalized values and role positions grouped by cluster id."""
    try:
        cache = _coref_doc_index_cache_173b(cursor)
        key = int(doc_id)
        if key in cache:
            try:
                cursor._coref_doc_index_cache_hits_173b = (
                    getattr(cursor, "_coref_doc_index_cache_hits_173b", 0) + 1
                )
            except Exception:
                pass
            return cache[key]
    except Exception:
        cache = None
        key = doc_id

    try:
        tokens, lemmas = _coref_doc_arrays(cursor, doc_id)
    except Exception:
        tokens, lemmas = [], []
    try:
        corefs = _coref_get_corefs(cursor, doc_id) or []
    except Exception:
        corefs = []

    values_by_cluster = {}
    positions_by_cluster_role = {}
    n = max(len(tokens or []), len(lemmas or []), len(corefs or []))
    for pos in range(n):
        try:
            components = _parse_coref_components(corefs[pos] if pos < len(corefs) else [])
        except Exception:
            components = []
        if not components:
            continue

        normalized_values = set()
        for seq in (tokens, lemmas):
            try:
                value = seq[pos] if pos < len(seq) else None
            except Exception:
                value = None
            text = str(value or "").strip().casefold()
            if text:
                normalized_values.add(text)

        # A token may contain duplicate labels.  Index each (cluster, role,
        # position) once so result positions remain set-like.
        seen_components = set()
        for component in components:
            cid = str(component.get("cid") or "").strip()
            role = _coref_normalize_role_173b(component.get("role"))
            if not cid or not role:
                continue
            marker = (cid, role)
            if marker in seen_components:
                continue
            seen_components.add(marker)
            values_by_cluster.setdefault(cid, set()).update(normalized_values)
            role_map = positions_by_cluster_role.setdefault(
                cid, {"head": set(), "part": set(), "mention": set(), "all": set()}
            )
            role_map.setdefault(role, set()).add(int(pos))
            role_map["all"].add(int(pos))

    built = {
        "values_by_cluster": values_by_cluster,
        "positions_by_cluster_role": positions_by_cluster_role,
        "token_count": n,
    }
    try:
        cursor._coref_doc_index_builds_173b = (
            getattr(cursor, "_coref_doc_index_builds_173b", 0) + 1
        )
    except Exception:
        pass
    if cache is not None:
        try:
            if len(cache) >= _COREF_DOC_INDEX_CACHE_LIMIT_173B:
                cache.pop(next(iter(cache)))
            cache[key] = built
        except Exception:
            pass
    return built


def _coref_matching_cluster_ids_173b(doc_index, values, match_type="exact"):
    wanted = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not wanted:
        return set()
    mode = str(match_type or "exact").lower()
    out = set()
    for cid, cluster_values in (doc_index.get("values_by_cluster") or {}).items():
        matched = False
        for raw in wanted:
            if mode in {"regex", "regex_search"}:
                try:
                    rx = _re_138t2.compile(raw, _re_138t2.I)
                    matcher = rx.fullmatch if mode == "regex" else rx.search
                    matched = any(matcher(value) is not None for value in cluster_values)
                except Exception:
                    matched = False
            else:
                matched = raw.casefold() in cluster_values
            if matched:
                out.add(str(cid))
                break
    return out


def _coref_condition_positions_173b(cursor, cond, doc_id):
    kind = _coref_attr_kind(cond)
    allowed = _coref_role_allowed(kind)
    if not allowed:
        return None
    # Preserve the old fallback for negative coreference conditions.
    if not _coref_condition_is_positive(cond):
        if callable(_orig_coref_condition_positions_138t2):
            return _orig_coref_condition_positions_138t2(cursor, cond, doc_id)
        return None
    values = _coref_condition_values(cond)
    if not values:
        return set()
    doc_index = _build_coref_document_index_173b(cursor, int(doc_id))
    match_type = (cond or {}).get("match_type") or "exact"
    cluster_ids = _coref_matching_cluster_ids_173b(doc_index, values, match_type)
    positions = set()
    role_maps = doc_index.get("positions_by_cluster_role") or {}
    for cid in cluster_ids:
        role_map = role_maps.get(str(cid)) or {}
        for role in allowed:
            positions.update(role_map.get(role) or set())
    return positions


def _coref_condition_postings_173b(cursor, cond):
    kind = _coref_attr_kind(cond)
    if not _coref_role_allowed(kind) or not _coref_condition_is_positive(cond):
        if callable(_orig_coref_condition_postings_138t2):
            return _orig_coref_condition_postings_138t2(cursor, cond)
        return None
    merged = {}
    for doc_id in _coref_iter_doc_ids(cursor):
        positions = _coref_condition_positions_173b(cursor, cond, int(doc_id)) or set()
        if positions:
            merged[int(doc_id)] = sorted(int(pos) for pos in positions)
    return merged


# Install optimized helpers under the names used by SearchCursor execution.
_coref_condition_positions = _coref_condition_positions_173b
_coref_condition_postings = _coref_condition_postings_173b
# KORPUSUJ_PATCH_173B2_COREF_DIRECT_POSTINGS_FASTPATH_BEGIN
# The 173b postings are final same-cluster/role matches.  Before 173b2 the
# generic SearchCursor path revalidated every returned position through the
# old per-position helper, causing repeated SQLite get_doc/get_corefs reads.


def _coref_posting_cache_key_173b2(cond):
    try:
        values = tuple(str(v) for v in (_coref_condition_values(cond) or []))
        return (
            "coref_final_173b2",
            str((cond or {}).get("attr") or (cond or {}).get("key") or "coref").lower(),
            str((cond or {}).get("op") or (cond or {}).get("operator") or "="),
            str((cond or {}).get("match_type") or "exact").lower(),
            values,
        )
    except Exception:
        return None


def _coref_exact_candidate_doc_ids_173b2(cursor, cond):
    """Return a safe exact-query doc prefilter or None when not provable.

    A cluster can contain value X only in a document containing token or lemma
    X.  Therefore the union of orth/base term postings is a necessary document
    condition.  Term spelling is discovered from SQLite and compared with
    Python casefold so coref's case-insensitive exact semantics are preserved.
    """
    if not _coref_condition_is_positive(cond):
        return None
    if str((cond or {}).get("match_type") or "exact").lower() != "exact":
        return None
    values = [str(v).strip() for v in (_coref_condition_values(cond) or []) if str(v).strip()]
    if not values:
        return set()
    index = getattr(cursor, "index", None)
    if index is None:
        return None

    candidate_ids = set()
    proved_any = False
    wanted = {value.casefold() for value in values}

    # Prefer discovering the actual indexed spelling.  This avoids assuming
    # whether base/orth terms were stored as Polska, polska, or another case.
    con = getattr(index, "con", None)
    if con is not None:
        try:
            rows = con.execute(
                "SELECT attr, value FROM terms WHERE attr IN ('base', 'orth')"
            ).fetchall()
            matching_terms = []
            for row in rows:
                try:
                    attr, indexed_value = row[0], row[1]
                except Exception:
                    try:
                        attr, indexed_value = row["attr"], row["value"]
                    except Exception:
                        continue
                if str(indexed_value).casefold() in wanted:
                    matching_terms.append((str(attr), str(indexed_value)))
            for attr, indexed_value in matching_terms:
                postings = index.get_postings(attr, indexed_value)
                if postings is not None:
                    proved_any = True
                    try:
                        candidate_ids.update(int(doc_id) for doc_id in postings.keys())
                    except Exception:
                        pass
            # An empty set is safe only when the terms query itself succeeded.
            return candidate_ids if proved_any or not matching_terms else None
        except Exception:
            pass

    # Conservative API fallback.  If it cannot prove candidates, use None so
    # the caller scans all documents rather than risking false negatives.
    getter = getattr(index, "get_doc_ids_for_term", None)
    if callable(getter):
        spellings = set(values)
        for value in values:
            spellings.update({value.lower(), value.upper(), value.casefold(), value.capitalize()})
        try:
            for attr in ("base", "orth"):
                for spelling in spellings:
                    docs = getter(attr, spelling)
                    if docs is not None:
                        proved_any = True
                        candidate_ids.update(int(doc_id) for doc_id in docs)
            return candidate_ids if proved_any else None
        except Exception:
            return None
    return None


def _coref_condition_postings_173b2(cursor, cond):
    key = _coref_posting_cache_key_173b2(cond)
    cache = getattr(cursor, "_posting_cache_local", None)
    if cache is None:
        cache = {}
        try:
            cursor._posting_cache_local = cache
        except Exception:
            pass
    if key is not None and key in cache:
        return cache[key]

    kind = _coref_attr_kind(cond)
    if not _coref_role_allowed(kind) or not _coref_condition_is_positive(cond):
        result = _coref_condition_postings_173b(cursor, cond)
        if key is not None:
            cache[key] = result
        return result

    candidate_ids = _coref_exact_candidate_doc_ids_173b2(cursor, cond)
    if candidate_ids is None:
        doc_ids = _coref_iter_doc_ids(cursor)
        try:
            cursor._coref_candidate_source_173b2 = "all_docs_fallback"
        except Exception:
            pass
    else:
        doc_ids = sorted(candidate_ids)
        try:
            cursor._coref_candidate_source_173b2 = "base_orth_exact_union"
            cursor._coref_candidate_doc_count_173b2 = len(doc_ids)
        except Exception:
            pass

    merged = {}
    for doc_id in doc_ids:
        positions = _coref_condition_positions_173b(cursor, cond, int(doc_id)) or set()
        if positions:
            merged[int(doc_id)] = sorted(int(pos) for pos in positions)
    if key is not None:
        cache[key] = merged
    return merged


def _match_coref_condition_at_pos_173b2(cursor, cond, doc_id, pos):
    """Revalidate from the cached 173b document index without SQLite reads."""
    kind = _coref_attr_kind(cond)
    allowed = _coref_role_allowed(kind)
    if not allowed or not _coref_condition_is_positive(cond):
        if callable(_orig_match_coref_condition_pos_138t2):
            return _orig_match_coref_condition_pos_138t2(cursor, cond, doc_id, pos)
        return None
    values = _coref_condition_values(cond)
    if not values:
        return False
    doc_index = _build_coref_document_index_173b(cursor, int(doc_id))
    matching_ids = _coref_matching_cluster_ids_173b(
        doc_index, values, (cond or {}).get("match_type") or "exact"
    )
    try:
        position = int(pos)
    except Exception:
        return False
    role_maps = doc_index.get("positions_by_cluster_role") or {}
    for cid in matching_ids:
        role_map = role_maps.get(str(cid)) or {}
        for role in allowed:
            if position in (role_map.get(role) or set()):
                return True
    return False


# Install both aliases used by SearchCursor.  Standalone coref positions now
# come directly from final postings, while any later generic revalidation is
# cache-only and performs no repeated SQLite document/corefs lookup.
_coref_condition_postings = _coref_condition_postings_173b2
_match_coref_condition_at_pos = _match_coref_condition_at_pos_173b2
# KORPUSUJ_PATCH_173B2_COREF_DIRECT_POSTINGS_FASTPATH_END

# KORPUSUJ_PATCH_173B_OPTIMIZE_COREF_DOCUMENT_POSTFILTER_END

# KORPUSUJ_PATCH_138Y_INLINE_COREF_SEMANTICS_INTO_CURSOR_END

import os
import json

import re

from korpusuj.dependency.disk_cache import DependencyMapDiskCache
from korpusuj.index.lru import LRUCache
from korpusuj.search.diagnostics import search_diag_log
from korpusuj.search.cursor_runtime import (
    candidate_max_docs,
    candidate_stream_batch_docs,
    dependency_cache_corpus_name_from_path as _dependency_cache_corpus_name_from_path,
    dependency_ram_cache_size_for_corpus as _dependency_ram_cache_size_for_corpus,
    get_dependency_cache_ram_mode as _get_dependency_cache_ram_mode,
    get_dependency_maps_cache,
    get_search_cursor_runtime,
    get_full_context_size,
    preload_dependency_maps_for_candidate_docs as preload_dependency_maps_for_candidates,
    put_dependency_ram_cache as _put_dependency_ram_cache,
)


# KORPUSUJ_MIGRATION_036L4G7_SEARCHCURSOR_DOC_CACHE
def _doc_cache_limit_036l4g7():
    try:
        return max(0, int(os.environ.get("KORPUSUJ_036L4G7_DOC_CACHE_SIZE", "4096") or 4096))
    except Exception:
        return 4096


def _get_doc_cached_036l4g7(cursor, doc_id):
    limit = _doc_cache_limit_036l4g7()
    if limit <= 0:
        return cursor.index.get_doc(doc_id)
    try:
        doc_id = int(doc_id)
    except Exception:
        pass
    cache = getattr(cursor, "_doc_cache_036l4g7", None)
    if cache is None:
        cache = {}
        cursor._doc_cache_036l4g7 = cache
    if doc_id in cache:
        try:
            cursor._doc_cache_hits_036l4g7 = getattr(cursor, "_doc_cache_hits_036l4g7", 0) + 1
        except Exception:
            pass
        return cache[doc_id]
    try:
        cursor._doc_cache_misses_036l4g7 = getattr(cursor, "_doc_cache_misses_036l4g7", 0) + 1
    except Exception:
        pass
    doc = cursor.index.get_doc(doc_id)
    try:
        if len(cache) >= limit:
            cache.pop(next(iter(cache)))
        cache[doc_id] = doc
    except Exception:
        pass
    return doc
# END KORPUSUJ_MIGRATION_036L4G7_SEARCHCURSOR_DOC_CACHE

    # KORPUSUJ_MIGRATION_036L4G8_PREFETCH_DOCS_FOR_MATERIALIZATION
    def _prefetch_docs_for_hits_036l4g8(self):
        """Prefetch unique docs for already collected hits before _result materialization."""
        try:
            hits = getattr(self, "_hits", []) or []
            if not hits:
                return 0
            cache = getattr(self, "_doc_cache_036l4g7", None)
            if cache is None:
                cache = {}
                self._doc_cache_036l4g7 = cache
            ids = []
            seen = set()
            for h in hits:
                try:
                    doc_id = int(h[0])
                except Exception:
                    continue
                if doc_id in seen or doc_id in cache:
                    continue
                seen.add(doc_id)
                ids.append(doc_id)
            if not ids:
                return 0
            loader = getattr(self.index, "get_docs_many", None) or getattr(self.index, "get_docs_many_036l4g8", None)
            if not callable(loader):
                return 0
            docs = loader(ids)
            if not isinstance(docs, dict):
                return 0
            for doc_id, doc in docs.items():
                try:
                    cache[int(doc_id)] = doc
                except Exception:
                    cache[doc_id] = doc
            return len(docs)
        except Exception:
            return 0



# KORPUSUJ_MIGRATION_PATCH_111_LAZY_FULLTEXT_CONTEXT_ON_CLICK
LAZY_FULLTEXT_MARKER_111 = "__KORPUSUJ_LAZY_FULLTEXT_REF_111__"


def make_lazy_fulltext_ref_111(index, doc_id, start, end, left_context_size=10, right_context_size=10, full_context_size=250):
    """Return a lightweight reference for extended/full context."""
    try:
        doc_id = int(doc_id)
    except Exception:
        pass
    try:
        start = int(start)
    except Exception:
        pass
    try:
        end = int(end)
    except Exception:
        pass
    try:
        left_context_size = int(left_context_size or 10)
    except Exception:
        left_context_size = 10
    try:
        right_context_size = int(right_context_size or 10)
    except Exception:
        right_context_size = 10
    try:
        full_context_size = int(full_context_size or 250)
    except Exception:
        full_context_size = 250
    return (
        LAZY_FULLTEXT_MARKER_111,
        index,
        doc_id,
        start,
        end,
        left_context_size,
        right_context_size,
        full_context_size,
    )


def is_lazy_fulltext_ref_111(value):
    try:
        return isinstance(value, (tuple, list)) and len(value) >= 8 and value[0] == LAZY_FULLTEXT_MARKER_111
    except Exception:
        return False


def _int_at_lazy_fulltext_111(seq, idx, fallback):
    try:
        return int(seq[idx])
    except Exception:
        try:
            return int(fallback)
        except Exception:
            return 0


def resolve_lazy_fulltext_ref_111(full_text_or_ref, context=None):
    """Resolve a lazy fulltext ref to [full_left, matched, full_right].

    The GUI row-click renderer inserts:
        full_text[0] + result[0] + result[1] + result[2] + full_text[2]

    Therefore this resolver must keep the returned full_text slices and the
    mutable result/context payload consistent.  In particular, when the result
    table context was produced from token arrays because the cached result-table
    document did not include full text, never subtract that token-joined context
    from full-text slices by exact string matching.  Use character offsets from
    start_ids/end_ids instead.
    """
    if not is_lazy_fulltext_ref_111(full_text_or_ref):
        return full_text_or_ref
    try:
        _marker, index, doc_id, start, end, left_ctx, right_ctx, full_ctx = full_text_or_ref[:8]
        doc_id = int(doc_id)
        start = int(start)
        end = int(end)
        left_ctx = int(left_ctx or 0)
        right_ctx = int(right_ctx or 0)
        full_ctx = int(full_ctx or 250)
    except Exception:
        index = None
        doc_id = None
        start = 0
        end = 0
        left_ctx = 0
        right_ctx = 0
        full_ctx = 250

    context_left = ""
    context_match = ""
    context_right = ""
    try:
        if isinstance(context, (list, tuple)):
            context_left = str(context[0]) if len(context) > 0 else ""
            context_match = str(context[1]) if len(context) > 1 else ""
            context_right = str(context[2]) if len(context) > 2 else ""
    except Exception:
        context_left = context_match = context_right = ""

    def _clamp_char(value, text_len):
        try:
            value = int(value)
        except Exception:
            value = 0
        return max(0, min(value, text_len))

    def _token_start(starts, token_idx, fallback, text_len):
        return _clamp_char(_int_at_lazy_fulltext_111(starts, token_idx, fallback), text_len)

    def _token_end_exclusive(ends, token_idx, fallback, text_len):
        return _clamp_char(_int_at_lazy_fulltext_111(ends, token_idx, fallback) + 1, text_len)

    def _limit_after_token_window(starts, token_idx, fallback, text_len):
        try:
            token_idx = int(token_idx)
        except Exception:
            token_idx = 0
        if token_idx < len(starts):
            return _token_start(starts, token_idx, fallback, text_len)
        return text_len

    def _rewrite_mutable_context(short_left, matched, short_right):
        if not isinstance(context, list):
            return
        try:
            while len(context) < 3:
                context.append("")
            context[0] = short_left
            context[1] = matched
            context[2] = short_right
        except Exception:
            pass

    try:
        getter = getattr(index, "get_doc", None)
        doc = getter(doc_id) if callable(getter) else None
        doc = doc or {}
        tokens = doc.get("tokens", []) or []
        starts = doc.get("start_ids", []) or []
        ends = doc.get("end_ids", []) or []
        text = doc.get("text", "") or ""
        text_len = len(text)

        if not text or not starts or not ends or end <= 0:
            if not context_match:
                try:
                    context_match = " ".join(tokens[start:end])
                except Exception:
                    context_match = ""
            _rewrite_mutable_context(context_left, context_match, context_right)
            return ["", context_match, ""]

        token_count = len(starts)
        start = max(0, min(int(start), token_count))
        end = max(start, min(int(end), token_count))

        match_start = _token_start(starts, start, start, text_len)
        match_end = _token_end_exclusive(ends, end - 1, match_start, text_len) if end > start else match_start
        match_end = max(match_start, match_end)
        matched = text[match_start:match_end]

        short_left_token = max(0, start - max(0, left_ctx))
        short_left_start = _token_start(starts, short_left_token, 0, text_len) if start > 0 else match_start
        short_left_start = min(short_left_start, match_start)
        short_left = text[short_left_start:match_start] if start > 0 else ""

        short_right_limit_token = min(token_count, end + max(0, right_ctx))
        short_right_end = _limit_after_token_window(starts, short_right_limit_token, match_end, text_len)
        short_right_end = max(match_end, short_right_end)
        short_right = text[match_end:short_right_end]

        full_left_token = max(0, start - max(0, full_ctx))
        full_left_start = _token_start(starts, full_left_token, 0, text_len) if start > 0 else match_start
        full_left_start = min(full_left_start, short_left_start)
        full_left = text[full_left_start:short_left_start] if start > 0 else ""

        full_right_limit_token = min(token_count, end + max(0, right_ctx) + max(0, full_ctx))
        full_right_end = _limit_after_token_window(starts, full_right_limit_token, short_right_end, text_len)
        full_right_end = max(short_right_end, full_right_end)
        full_right = text[short_right_end:full_right_end]

        _rewrite_mutable_context(short_left, matched, short_right)
        return [full_left, matched, full_right]
    except Exception:
        pass

    _rewrite_mutable_context(context_left, context_match, context_right)
    return ["", context_match, ""]


def resolve_result_row_fulltext_111(row):
    """Return row with result[2] resolved if it is a lazy fulltext ref."""
    try:
        if not isinstance(row, (tuple, list)) or len(row) <= 2:
            return row
        if not is_lazy_fulltext_ref_111(row[2]):
            return row
        resolved = resolve_lazy_fulltext_ref_111(row[2], row[1] if len(row) > 1 else None)
        out = list(row)
        out[2] = resolved
        return tuple(out) if isinstance(row, tuple) else out
    except Exception:
        return row

class SearchCursor:
    """Lazily discover, count, page and materialize final matches for a planned query."""
    def __init__(self, index, plan, left_context_size=10, right_context_size=10, corpus_path=None):
        self.index = index
        self.plan = plan
        self.left_context_size = int(left_context_size or 10)
        self.right_context_size = int(right_context_size or 10)
        try: self.full_context_size = int(get_search_cursor_runtime().full_context_size)
        except Exception: self.full_context_size = 250
        self.corpus_path = str(corpus_path or "")
        self._hit_iter = None
        self._hits = []
        self._count_cache = None
        self._exhausted = False
        self._result_cache = {}
        self._metadata_doc_filter = None
        self._posting_cache_local = {}
        self._dep_cache = None
        self._dep_maps_cache = LRUCache(512)
        # KORPUSUJ_MIGRATION_036L4G7_SEARCHCURSOR_DOC_CACHE
        self._doc_cache_036l4g7 = {}
        self._doc_cache_hits_036l4g7 = 0
        self._doc_cache_misses_036l4g7 = 0
        self.corpus_name = _dependency_cache_corpus_name_from_path(self.corpus_path)
        # 3m1 hotfix: pola muszą istnieć dla każdej instancji SearchCursor.
        self._dep_stream_batches = 0
        self._dep_stream_preloaded = 0


    def __bool__(self): return self._count_cache is None or self._count_cache > 0
    def __len__(self): return self._count_cache if self._count_cache is not None else max(len(self._hits), int(self.count_hits_estimate() or 0))
    def __iter__(self):
        self._ensure_all()
        try:
            self._prefetch_docs_for_hits_036l4g8()
        except Exception:
            pass
        return iter(self.get_range(0, len(self._hits)))
    def __getitem__(self, item):
        if isinstance(item, slice): return self.get_range(item.start or 0, item.stop if item.stop is not None else len(self))
        return self.get_range(item, item + 1)[0]

    def _plan_uses_dependency(self):
        return bool(self.plan.get("uses_dependency"))

    # KORPUSUJ_MIGRATION_036L4G6D_CANDIDATE_STREAM_BATCH_SIZE
    def _candidate_stream_batch_size(self):
        """Return batch size for streaming dependency candidate postings."""
        try:
            value = candidate_stream_batch_docs()
        except Exception:
            value = None
        try:
            value = int(value or 0)
        except Exception:
            value = 0
        return value if value > 0 else 256

    # KORPUSUJ_MIGRATION_036L4G6E_DEP_STREAM_COUNTER_METHODS
    def _dep_stream_add_batch(self, n=1):
        """Increment dependency streaming batch counter used by stream diagnostics."""
        try:
            self._dep_stream_batches = int(getattr(self, "_dep_stream_batches", 0) or 0) + int(n or 1)
        except Exception:
            self._dep_stream_batches = 1
        return self._dep_stream_batches

    def _dep_stream_add_preloaded(self, n=0):
        """Increment dependency streaming preload counter used by stream diagnostics."""
        try:
            self._dep_stream_preloaded = int(getattr(self, "_dep_stream_preloaded", 0) or 0) + int(n or 0)
        except Exception:
            self._dep_stream_preloaded = int(n or 0) if str(n or "0").isdigit() else 0
        return self._dep_stream_preloaded

    def count_hits_estimate(self):
        # Dla dependency nie pokazujemy estymat z kotwicy; dokładne liczenie robi count_hits(exact=True).
        """Return the inexpensive hit estimate available for the current cursor."""
        if self._plan_uses_dependency():
            return self._count_cache if self._count_cache is not None else len(self._hits)
        groups = self.plan.get("token_groups") or []

        # KORPUSUJ_MIGRATION_PATCH_118_FAST_REGEX_CF_COUNT_FROM_FIND_TERMS
        # Count-only fast path for simple indexed regex CQL. Public get_term_info()
        # must keep returning both df and cf, but GUI/CLI count wrappers need only
        # exact total_hits/cf. find_terms_regex_036l4g37c returns per-term cf and
        # avoids computing df via doc_id/postings union.
        try:
            if len(groups) == 1 and len(groups[0].get("conds", [])) == 1 and not self.plan.get("metadata_filters"):
                c118 = (groups[0].get("conds", []) or [None])[0]
                if isinstance(c118, dict):
                    match_type_118 = c118.get("match_type") or "exact"
                    if match_type_118 in ("regex", "regex_search"):
                        attr_118 = c118.get("attr")
                        values_118 = c118.get("values", None)
                        if values_118 is None:
                            values_118 = [c118.get("value")]
                        if attr_118 and values_118 and len(values_118) == 1 and values_118[0] is not None:
                            finder_118 = getattr(self.index, "find_terms_regex_036l4g37c", None)
                            if callable(finder_118):
                                raw_118 = values_118[0]
                                search_mode_118 = (match_type_118 == "regex_search")
                                try:
                                    pattern_118 = self._regex_pattern_for_index_090(raw_118, search_mode=search_mode_118)
                                except Exception:
                                    pattern_118 = raw_118
                                terms_118 = None
                                for call_118 in (
                                    lambda: finder_118(attr_118, pattern_118, search_mode=search_mode_118),
                                    lambda: finder_118(attr_118, pattern_118),
                                ):
                                    try:
                                        terms_118 = list(call_118())
                                        break
                                    except TypeError:
                                        continue
                                if terms_118 is not None:
                                    cf_total_118 = 0
                                    for item_118 in terms_118:
                                        cf_118 = None
                                        if isinstance(item_118, dict):
                                            cf_118 = item_118.get("cf")
                                        elif isinstance(item_118, (tuple, list)):
                                            if len(item_118) >= 4 and str(item_118[0]) == str(attr_118):
                                                cf_118 = item_118[3]
                                            elif len(item_118) >= 3:
                                                cf_118 = item_118[2]
                                        try:
                                            cf_total_118 += int(cf_118 or 0)
                                        except Exception:
                                            pass
                                    try:
                                        import logging
                                        if korpusuj_verbose_diagnostics_enabled_145c1():
                                            logging.info(
                                                "[DIAG perf.search.count] event=%r data=%r",
                                                "used_find_terms_cf_sum",
                                                {
                                                    "attr": attr_118,
                                                    "pattern": str(pattern_118),
                                                    "match_type": match_type_118,
                                                    "terms": len(terms_118),
                                                    "cf": int(cf_total_118),
                                                },
                                            )
                                    except Exception:
                                        pass
                                    return int(cf_total_118)
        except Exception as _exc_118:
            try:
                import logging
                if korpusuj_verbose_diagnostics_enabled_145c1():
                    logging.info(
                        "[DIAG perf.search.count] event=%r data=%r",
                        "fast_regex_cf_failed_fallback_get_term_info",
                        {"error": repr(_exc_118)},
                    )
            except Exception:
                pass

        infos = [self.index.get_term_info(c["attr"], v)
                 for el in groups for c in el.get("conds", []) for v in c.get("values", [c.get("value")])]
        if not infos:
            return 0
        if len(groups) == 1 and len(groups[0].get("conds", [])) == 1 and not self.plan.get("metadata_filters"):
            return int(infos[0].get("cf", 0) or 0)
        return min(int(info.get("df", 0) or 0) for info in infos)

    def count_hits_estimate_is_exact(self):
        """Return whether count_hits_estimate is guaranteed to equal the final hit count."""
        groups = self.plan.get("token_groups") or []

        # KORPUSUJ_PATCH_173B6_COREF_EXACT_COUNT_CONTRACT
        # Coref is a docs.corefs post-filter, not an indexed term attribute.
        # get_term_info("coref", value) may therefore yield cf=0 even when the
        # final coref postings contain hits. Do not classify that term estimate
        # as exact; centralized counting must call count_hits(exact=True), which
        # exhausts the final SearchCursor and populates _count_cache correctly.
        try:
            if any(
                _is_coref_condition(cond)
                for group in groups
                for cond in (group.get("conds", []) if isinstance(group, dict) else [])
            ):
                return False
        except Exception:
            # Conservative policy: if plan inspection fails, keep the previous
            # behavior for non-coref plans rather than changing all count paths.
            pass

        return (not self._plan_uses_dependency()
                and len(groups) == 1
                and len(groups[0].get("conds", [])) == 1
                and not self.plan.get("metadata_filters"))

    def count_hits(self, exact=False):
        """Return the hit count, using exact final-match counting when exact is true."""
        if exact and self._count_cache is None:
            self._ensure_all()
        return self._count_cache if self._count_cache is not None else max(len(self._hits), int(self.count_hits_estimate() or 0))
    def get_page(self, page=0, page_size=100):
        start = int(page) * int(page_size); return self.get_range(start, start + int(page_size))
    def get_range(self, start, stop):
        self._ensure_until(stop); return [self._result(i) for i in range(start, min(stop, len(self._hits)))]
    def _ensure_until(self, stop):
        if self._hit_iter is None: self._hit_iter = self._iter_hits()
        while len(self._hits) < stop:
            try: self._hits.append(next(self._hit_iter))
            except StopIteration: self._exhausted = True; self._count_cache = len(self._hits); break
    def _ensure_all(self): self._ensure_until(10 ** 18)
    def _meta_docs(self):
        if self._metadata_doc_filter is None: self._metadata_doc_filter = self.index.filter_docs_by_metadata(self.plan.get("metadata_filters") or [])
        return self._metadata_doc_filter


    # KORPUSUJ_MIGRATION_036L4F2_SIMPLE_INDEXED_MORPH_FASTPATH
    _INDEXED_ATTRS_036L4F2 = {"base", "orth", "pos", "upos", "deprel", "ner"}

    def _fastpath_simple_indexed_morph_parts_036l4f2(self):
        try:
            if self._plan_uses_dependency():
                return None
            groups = self.plan.get("token_groups", [])
            if len(groups) != 1:
                return None
            group = groups[0]
            conds = list(group.get("conds", []) or [])
            if not conds:
                return None
            indexed = []
            morph = []
            for cond in conds:
                if not isinstance(cond, dict):
                    return None
                attr = str(cond.get("attr", ""))
                values = list(cond.get("values", []) or [])
                if not attr or not values:
                    return None
                op = str(cond.get("op", "=") or "=")
                if op not in {"=", "=="}:
                    return None
                if self._is_morph_feature_condition_036l1b(cond):
                    morph.append(cond)
                elif attr in self._INDEXED_ATTRS_036L4F2:
                    indexed.append(cond)
                else:
                    return None
            if not indexed or not morph:
                return None
            return group, indexed, morph
        except Exception:
            return None

    def _intersect_postings_036l4f2(self, left, right):
        if not left or not right:
            return {}
        if len(left) > len(right):
            left, right = right, left
        out = {}
        for doc_id, lpos in left.items():
            rpos = right.get(int(doc_id))
            if not rpos:
                continue
            if len(lpos) <= len(rpos):
                s = set(int(p) for p in lpos)
                inter = [int(p) for p in rpos if int(p) in s]
            else:
                s = set(int(p) for p in rpos)
                inter = [int(p) for p in lpos if int(p) in s]
            if inter:
                out[int(doc_id)] = sorted(set(inter))
        return out

    def _indexed_candidate_postings_036l4f2(self, indexed_conds):
        ordered = sorted(indexed_conds, key=lambda c: self._condition_df(c))
        current = None
        for cond in ordered:
            p = self._condition_postings(cond)
            if current is None:
                current = {int(d): list(ps) for d, ps in p.items()}
            else:
                current = self._intersect_postings_036l4f2(current, p)
            if not current:
                return {}
        return current or {}

    def _morph_conditions_match_full_tag_036l4f2(self, full_tag, morph_conds):
        full_tag = str(full_tag or "")
        parts_set = None
        for cond in morph_conds:
            attr = str(cond.get("attr", ""))
            wanted = set(self._condition_values_036l1b(cond))
            if not wanted:
                return False
            value = self._morph_feature_from_full_postag_036l1b(full_tag, attr)
            if value in wanted:
                continue
            if attr in self._MORPH_FEATURE_ATTRS_036L1B:
                if parts_set is None:
                    parts_set = set(full_tag.split(":")[1:])
                if parts_set & wanted:
                    continue
            return False
        return True


    # KORPUSUJ_MIGRATION_036L4F4_FULL_POSTAGS_CACHE
    def _full_postags_cache_036l4f4(self):
        cache = getattr(self, "_full_postags_cache_036l4f4_store", None)
        if cache is None:
            cache = {}
            self._full_postags_cache_036l4f4_store = cache
        return cache

    def _get_full_postags_cached_036l4f4(self, doc_id):
        doc_id = int(doc_id)
        cache = self._full_postags_cache_036l4f4()
        if doc_id in cache:
            return cache[doc_id]
        getter = getattr(self.index, "get_full_postags_036l4f4", None)
        if callable(getter):
            arr = getter(doc_id) or []
        else:
            doc = self._get_doc_cached_036l4b(doc_id)
            arr = doc.get("full_postags", []) or []
        cache[doc_id] = arr
        return arr
    # END KORPUSUJ_MIGRATION_036L4F4_FULL_POSTAGS_CACHE

    def _iter_hits_simple_indexed_morph_036l4f2(self):
        parts = self._fastpath_simple_indexed_morph_parts_036l4f2()
        if parts is None:
            return None
        _group, indexed_conds, morph_conds = parts
        meta_docs = self._meta_docs()
        candidates = self._indexed_candidate_postings_036l4f2(indexed_conds)
        def _gen():
            for doc_id, positions in candidates.items():
                doc_id = int(doc_id)
                if meta_docs is not None and doc_id not in meta_docs:
                    continue
                arr = self._get_full_postags_cached_036l4f4(doc_id)
                n = len(arr)
                for pos in positions:
                    ipos = int(pos)
                    if ipos < 0 or ipos >= n:
                        continue
                    if self._morph_conditions_match_full_tag_036l4f2(arr[ipos], morph_conds):
                        yield (doc_id, ipos, ipos + 1)
        return _gen()
    # END KORPUSUJ_MIGRATION_036L4F2_SIMPLE_INDEXED_MORPH_FASTPATH

    def _iter_hits(self):
        fast_iter_036l4f2 = self._iter_hits_simple_indexed_morph_036l4f2()
        if fast_iter_036l4f2 is not None:
            yield from fast_iter_036l4f2
            return
        # KORPUSUJ_MIGRATION_PATCH_115_PURE_INDEXED_SINGLE_CONDITION_DIRECT_POSTINGS_FASTPATH
        # Pure indexed single-condition queries, e.g. [base="wojna"] or
        # [base=".*a.*"], do not use the existing indexed+morph fastpath because
        # that path intentionally requires both indexed and morph conditions.
        # Without this direct path, they fall through to generic anchor matching,
        # which re-validates every posting through _match_at/_condition_matches_pos.
        # Here postings are the source of truth, so for this very narrow positive
        # single-condition case we can yield them directly.
        try:
            groups_115 = self.plan.get("token_groups") or []
            if (
                not self._plan_uses_dependency()
                and len(groups_115) == 1
                and not self.plan.get("metadata_filters")
            ):
                group_115 = groups_115[0] or {}
                conds_115 = list(group_115.get("conds", []) or [])
                if (
                    len(conds_115) == 1
                    and not (group_115.get("neg_conds") or [])
                    and not (group_115.get("dep_conds") or [])
                    and not (group_115.get("dep_neg_conds") or [])
                ):
                    cond_115 = conds_115[0]
                    if isinstance(cond_115, dict) and not self._is_morph_feature_condition_036l1b(cond_115):
                        attr_115 = str(cond_115.get("attr", "") or "")
                        indexed_attrs_115 = getattr(self, "_INDEXED_ATTRS_036L4F2", {"base", "orth", "pos", "upos", "deprel", "ner"})
                        op_115 = str(cond_115.get("op", "=") or "=")
                        values_115 = cond_115.get("values", None)
                        if values_115 is None:
                            values_115 = [cond_115.get("value")]
                        if attr_115 in indexed_attrs_115 and op_115 in {"=", "=="} and any(v is not None for v in (values_115 or [])):
                            postings_115 = self._condition_postings(cond_115)
                            for doc_id_115, positions_115 in postings_115.items():
                                for pos_115 in positions_115:
                                    ipos_115 = int(pos_115)
                                    if ipos_115 >= 0:
                                        yield (int(doc_id_115), ipos_115, ipos_115 + 1)
                            return
        except Exception:
            # Conservative fallback: keep legacy generic matcher semantics if the
            # direct postings path cannot prove safety at runtime.
            pass
        groups = self.plan["token_groups"]
        if self._plan_uses_dependency() and len(groups) == 1:
            seed_iter = self._dependency_seed_position_iterator_for_token(groups[0])
            if seed_iter is not None:
                meta_docs = self._meta_docs()
                seen = set()
                for doc_id, pos in seed_iter:
                    doc_id = int(doc_id); pos = int(pos)
                    if meta_docs is not None and doc_id not in meta_docs:
                        continue
                    key = (doc_id, pos)
                    if key in seen:
                        continue
                    seen.add(key)
                    if pos >= 0 and self._match_token_at(doc_id, pos, groups[0]):
                        yield (doc_id, pos, pos + 1)
                search_diag_log(
                    "DEP_STREAM_DONE corpus=%r seen=%s batches=%s preloaded=%s",
                    self.corpus_name, len(seen), getattr(self, "_dep_stream_batches", 0), getattr(self, "_dep_stream_preloaded", 0)
                )
                return

        anchors = []
        for gi, el in enumerate(groups):
            for cond in el.get("conds", []):
                df = self._condition_df(cond)
                if df > 0:
                    anchors.append((df, gi, cond))
        anchors.sort(key=lambda x: x[0])

        if not anchors:
            if not self._plan_uses_dependency():
                return
            meta_docs = self._meta_docs()
            total_docs = int(getattr(self.index, "total_docs", 0) or 0)
            for doc_id in range(total_docs):
                if meta_docs is not None and doc_id not in meta_docs: continue
                doc = self.index.get_doc(doc_id) or {}
                token_count = len(doc.get("tokens", []) or [])
                for pos in range(token_count):
                    if self._match_token_at(doc_id, pos, groups[0]):
                        yield (doc_id, pos, pos + 1)
            return

        _df, anchor_group_idx, anchor_cond = anchors[0]
        meta_docs = self._meta_docs()
        anchor_postings = self._condition_postings(anchor_cond)
        # 3m1: nie robimy dużego preloadu anchor_scan z góry; dependency seed stream obsługuje zapytania jednoelementowe.
        for doc_id, positions in anchor_postings.items():
            if meta_docs is not None and doc_id not in meta_docs: continue
            for anchor_pos in positions:
                start = int(anchor_pos) - anchor_group_idx
                if start >= 0 and self._match_at(int(doc_id), start, groups):
                    yield (int(doc_id), start, start + len(groups))


    def _match_at(self, doc_id, start, groups):
        for offset, el in enumerate(groups):
            if not self._match_token_at(doc_id, start + offset, el):
                return False
        return True

    def _match_token_at(self, doc_id, pos, el):
        for cond in el.get("conds", []):
            if not self._condition_matches_pos(cond, doc_id, pos): return False
        for cond in el.get("neg_conds", []):
            if self._condition_matches_pos(cond, doc_id, pos): return False
        if el.get("dep_conds") or el.get("dep_neg_conds"):
            doc = self.index.get_doc(doc_id)
            if not doc: return False
            for cond in el.get("dep_conds", []):
                if not self._dependency_condition_matches(doc_id, pos, cond, doc): return False
            for cond in el.get("dep_neg_conds", []):
                if self._dependency_condition_matches(doc_id, pos, cond, doc): return False
        return True


    # KORPUSUJ_MIGRATION_036L1B_MORPH_FEATURE_DOC_FILTER
    _MORPH_FEATURE_ATTRS_036L1B = {
        "number", "case", "gender", "person", "aspect", "degree",
        "accentability", "post-prepositionality", "accommodability",
        "vocalicity", "agglutination", "negation", "fullstoppedness",
    }
    _MORPH_FEATURE_MAP_036L1B = {
        "subst": {"number": 0, "case": 1, "gender": 2},
        "depr": {"number": 0, "case": 1, "gender": 2},
        "adj": {"number": 0, "case": 1, "gender": 2, "degree": 3},
        "adja": {}, "adjp": {}, "adjc": {},
        "ppron12": {"number": 0, "case": 1, "gender": 2, "person": 3, "accentability": 4},
        "ppron3": {"number": 0, "case": 1, "gender": 2, "person": 3, "accentability": 4, "post-prepositionality": 5},
        "siebie": {"case": 0},
        "num": {"number": 0, "case": 1, "gender": 2, "accommodability": 3},
        "numcol": {"number": 0, "case": 1, "gender": 2, "accommodability": 3},
        "fin": {"number": 0, "person": 1, "aspect": 2},
        "bedzie": {"number": 0, "person": 1, "aspect": 2},
        "aglt": {"number": 0, "person": 1, "aspect": 2, "vocalicity": 3},
        "praet": {"number": 0, "gender": 1, "aspect": 2, "agglutination": 3},
        "winien": {"number": 0, "gender": 1, "aspect": 2},
        "impt": {"number": 0, "person": 1, "aspect": 2},
        "imps": {"aspect": 0}, "inf": {"aspect": 0}, "pcon": {"aspect": 0}, "pant": {"aspect": 0},
        "ger": {"number": 0, "case": 1, "gender": 2, "aspect": 3, "negation": 4},
        "pact": {"number": 0, "case": 1, "gender": 2, "aspect": 3, "negation": 4},
        "ppas": {"number": 0, "case": 1, "gender": 2, "aspect": 3, "negation": 4},
        "prep": {"case": 0}, "adv": {"degree": 0}, "brev": {"fullstoppedness": 0},
    }


    # KORPUSUJ_MIGRATION_036L4B_DOC_CACHE_AND_GENERIC_MORPH
    def _doc_cache_036l4b(self):
        cache = getattr(self, "_doc_cache_036l4b_store", None)
        if cache is None:
            cache = {}
            self._doc_cache_036l4b_store = cache
            self._doc_cache_036l4b_order = []
        return cache

    def _get_doc_cached_036l4b(self, doc_id):
        doc_id = int(doc_id)
        cache = self._doc_cache_036l4b()
        if doc_id in cache:
            return cache[doc_id]
        doc = self.index.get_doc(doc_id) or {}
        cache[doc_id] = doc
        order = getattr(self, "_doc_cache_036l4b_order", [])
        order.append(doc_id)
        self._doc_cache_036l4b_order = order
        max_size = 4096
        if len(order) > max_size:
            old = order.pop(0)
            cache.pop(old, None)
        return doc
    # END KORPUSUJ_MIGRATION_036L4B_DOC_CACHE_AND_GENERIC_MORPH

    def _condition_values_036l1b(self, cond):
        vals = cond.get("values", None)
        if vals is None:
            vals = cond.get("value", [])
        if vals is None:
            return []
        if isinstance(vals, (list, tuple, set)):
            return [str(v) for v in vals]
        return [str(vals)]

    def _is_morph_feature_condition_036l1b(self, cond):
        return str(cond.get("attr", "")) in self._MORPH_FEATURE_ATTRS_036L1B

    def _morph_feature_from_full_postag_036l1b(self, full_tag, feature):
        parts = str(full_tag or "").split(":")
        if not parts:
            return ""
        pos = parts[0]
        feats = parts[1:]
        mapping = self._MORPH_FEATURE_MAP_036L1B.get(pos, {})
        idx = mapping.get(feature)
        if idx is not None and 0 <= idx < len(feats):
            return str(feats[idx])
        return ""

    def _morph_feature_condition_matches_pos_036l1b(self, cond, doc_id, pos):
        doc = self._get_doc_cached_036l4b(int(doc_id))
        arr = doc.get("full_postags", []) or []
        ipos = int(pos)
        if ipos < 0 or ipos >= len(arr):
            return False
        attr = str(cond.get("attr", ""))
        wanted = set(self._condition_values_036l1b(cond))
        if not wanted:
            return False
        full_tag = str(arr[ipos] or "")
        value = self._morph_feature_from_full_postag_036l1b(full_tag, attr)
        if value in wanted:
            return True
        if attr in self._MORPH_FEATURE_ATTRS_036L1B:
            return bool(set(full_tag.split(":")[1:]) & wanted)
        return False

    def _morph_feature_condition_postings_036l1b(self, cond):
        # Only used when a morph feature condition is the only possible anchor.
        # For normal queries such as [base="wojna" & pos="subst" & case="gen"],
        # indexed conditions provide candidates and this method is not needed.
        merged = {}
        meta_docs = self._meta_docs()
        total = getattr(self.index, "total_docs", 0)
        try:
            total = total() if callable(total) else total
        except Exception:
            total = 0
        try:
            total = int(total or 0)
        except Exception:
            total = 0
        for doc_id in range(total):
            if meta_docs is not None and doc_id not in meta_docs:
                continue
            doc = self.index.get_doc(doc_id) or {}
            arr = doc.get("full_postags", []) or []
            positions = []
            for pos in range(len(arr)):
                if self._morph_feature_condition_matches_pos_036l1b(cond, doc_id, pos):
                    positions.append(pos)
            if positions:
                merged[int(doc_id)] = positions
        return merged
    # END KORPUSUJ_MIGRATION_036L1B_MORPH_FEATURE_DOC_FILTER

    def _condition_matches_pos(self, cond, doc_id, pos):
        # COREF_CQL_SEARCHCURSOR_ENABLED: coref doc-array post-filter.
        try:
            _coref_138i3 = _match_coref_condition_at_pos(self, cond, doc_id, pos)
            if _coref_138i3 is not None:
                return bool(_coref_138i3)
        except Exception:
            pass
        if self._is_morph_feature_condition_036l1b(cond):
            return self._morph_feature_condition_matches_pos_036l1b(cond, doc_id, pos)
        return int(pos) in self._condition_positions(cond, doc_id)

    def _condition_positions(self, cond, doc_id):
        # COREF_CQL_SEARCHCURSOR_ENABLED: coref positions.
        try:
            _coref_pos_138i3 = _coref_condition_positions(self, cond, doc_id)
            if _coref_pos_138i3 is not None:
                return _coref_pos_138i3
        except Exception:
            pass
        return set(self._condition_postings(cond).get(int(doc_id), []))
    def _condition_df(self, cond):
        # COREF_CQL_SEARCHCURSOR_138N: coref is a docs-payload post-filter, not an anchor.
        try:
            if _is_coref_condition(cond):
                return 10**12
        except Exception:
            pass
        if self._is_morph_feature_condition_036l1b(cond):
            # 036L4B: morph features are non-indexed doc-array filters.
            # They must not become anchors when indexed attrs are available.
            return 10**12
        return len(self._condition_postings(cond))

    def _condition_postings(self, cond):
        # COREF_CQL_SEARCHCURSOR_ENABLED: standalone coref postings via docs scan.
        try:
            _coref_post_138i3 = _coref_condition_postings(self, cond)
            if _coref_post_138i3 is not None:
                return _coref_post_138i3
        except Exception:
            pass
        key = json.dumps(cond, sort_keys=True, default=str)
        if key in self._posting_cache_local:
            return self._posting_cache_local[key]
        if self._is_morph_feature_condition_036l1b(cond):
            merged = self._morph_feature_condition_postings_036l1b(cond)
            self._posting_cache_local[key] = merged
            return merged

        match_type = cond.get("match_type") or "exact"
        if match_type in ("regex", "regex_search"):
            merged = self._condition_postings_regex_087(cond, search_mode=(match_type == "regex_search"))
            self._posting_cache_local[key] = merged
            return merged

        merged = {}
        for v in cond.get("values", []):
            for d, ps in self.index.get_postings(cond.get("attr"), v).items():
                s = merged.setdefault(int(d), set())
                s.update(int(p) for p in ps)
        merged = {d: sorted(ps) for d, ps in merged.items()}
        self._posting_cache_local[key] = merged
        return merged

    # KORPUSUJ_MIGRATION_PATCH_90_REGEX_REAL_INDEX_API
    # KORPUSUJ_MIGRATION_PATCH_91_PROPAGATE_REGEX_TOO_BROAD
    def _condition_postings_regex_087(self, cond, search_mode=False):
        """Resolve postings for a regular-expression token condition."""
        attr = cond.get("attr")
        values = self._regex_values_090(cond)
        if not attr or not values:
            return {}

        compiled = []
        for raw in values:
            try:
                compiled.append((str(raw), re.compile(str(raw))))
            except re.error:
                return {}

        merged = {}

        for raw, pat in compiled:
            before_count = sum(len(v) for v in merged.values())

            # Patch 91: _call_regex_postings_index_api_090 now raises on
            # TooBroad instead of returning state="too_broad". This prevents
            # broad valid regexes from becoming false zero-hit results.
            postings, state = self._call_regex_postings_index_api_090(attr, raw, search_mode=search_mode)
            if postings:
                self._merge_postings_into_090(merged, postings)
                continue

            try:
                for term in self._iter_index_terms_for_attr_087(attr, pattern=raw, search_mode=search_mode):
                    term_s = str(term)
                    if not (pat.search(term_s) if search_mode else pat.fullmatch(term_s)):
                        continue
                    try:
                        exact_postings = self.index.get_postings(attr, term)
                    except Exception as exc:
                        if self._is_regex_too_broad_exception_090(exc):
                            self._raise_regex_too_broad_091(attr, raw, exc, source="exact_postings_after_term_regex")
                        exact_postings = {}
                    self._merge_postings_into_090(merged, exact_postings)
            except Exception as exc:
                if self._is_regex_too_broad_exception_090(exc):
                    self._raise_regex_too_broad_091(attr, raw, exc, source="term_regex_iterator")
                raise

            after_count = sum(len(v) for v in merged.values())
            if after_count == before_count:
                # Compatibility fallback for non-SQLite/test indexes only.
                # We reach this point only when no TooBroad signal was raised.
                self._merge_postings_into_090(
                    merged,
                    self._condition_postings_regex_scan_docs_087(attr, [(raw, pat)], search_mode),
                )

        return {int(d): sorted({int(p) for p in ps}) for d, ps in merged.items() if ps}

    def _regex_values_090(self, cond):
        vals = cond.get("values", None)
        if vals is None:
            vals = cond.get("value", [])
        if vals is None:
            return []
        if isinstance(vals, (list, tuple, set)):
            return [str(v) for v in vals if v is not None]
        return [str(vals)]

    def _merge_postings_into_090(self, merged, postings):
        if not postings:
            return merged
        try:
            iterator = postings.items()
        except Exception:
            return merged
        for d, ps in iterator:
            try:
                doc_id = int(d)
            except Exception:
                continue
            s = merged.setdefault(doc_id, set())
            try:
                for p in ps:
                    s.add(int(p))
            except Exception:
                try:
                    s.add(int(ps))
                except Exception:
                    pass
        return merged

    def _regex_pattern_for_index_090(self, raw, search_mode=False):
        raw = str(raw)
        if not search_mode:
            return raw
        return ".*(?:" + raw + ").*"

    def _is_regex_too_broad_exception_090(self, exc):
        name = type(exc).__name__
        msg = repr(exc)
        return (
            "TooBroad" in name
            or "too_broad" in msg
            or "too broad" in msg.lower()
            or "too many terms" in msg.lower()
            or "max_terms" in msg
        )

    def _raise_regex_too_broad_091(self, attr, raw, exc, source="index_api"):
        """Propagate broad-regex policy signal instead of returning false zero."""
        try:
            search_diag_log(
                "REGEX_CURSOR_TOO_BROAD_091 attr=%r pattern=%r source=%r exc=%r",
                attr, raw, source, exc,
            )
        except Exception:
            pass
        # Preserve the original exception type/message so upstream layers that
        # already know RegexSQLiteTooBroadError can still recognize it.
        raise exc

    def _call_regex_postings_index_api_090(self, attr, raw, search_mode=False):
        """Call the index regular-expression postings API with normalized search-mode handling."""
        pattern = self._regex_pattern_for_index_090(raw, search_mode=search_mode)
        first_empty = None

        method = getattr(self.index, "get_postings_regex_036l4g37c", None)
        if callable(method):
            for call in (
                lambda: method(attr, pattern, search_mode=search_mode),
                lambda: method(attr, pattern),
            ):
                try:
                    res = call()
                    if isinstance(res, dict):
                        if res:
                            return res, "ok"
                        first_empty = res
                        # Empty regex-specific result is inconclusive. Continue
                        # to get_postings(), because that path owns broad policy
                        # in the observed SQLite SearchIndex.
                        continue
                except TypeError:
                    continue
                except Exception as exc:
                    if self._is_regex_too_broad_exception_090(exc):
                        self._raise_regex_too_broad_091(attr, raw, exc, source="get_postings_regex_036l4g37c")
                    return {}, "error"

        method = getattr(self.index, "get_postings", None)
        if callable(method):
            try:
                res = method(attr, pattern)
                if isinstance(res, dict):
                    return res, "ok"
            except Exception as exc:
                if self._is_regex_too_broad_exception_090(exc):
                    self._raise_regex_too_broad_091(attr, raw, exc, source="get_postings")
                return {}, "error"

        if first_empty is not None:
            return first_empty, "ok_empty_regex_api"
        return {}, "unavailable"
    def _iter_index_terms_for_attr_087(self, attr, pattern=None, search_mode=False):
        """Yield terms for attr, preferring the real SQLite regex term API."""
        seen = set()

        def emit_one(item):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                if str(item[0]) == str(attr):
                    item = item[1]
                else:
                    return None
            key = str(item)
            if key in seen:
                return None
            seen.add(key)
            return item

        def emit_many(values):
            if values is None:
                return
            iterator = values.keys() if isinstance(values, dict) else values
            for item in iterator:
                emitted = emit_one(item)
                if emitted is not None:
                    yield emitted

        if pattern is not None:
            sqlite_pattern = self._regex_pattern_for_index_090(pattern, search_mode=search_mode)
            finder = getattr(self.index, "find_terms_regex_036l4g37c", None)
            if callable(finder):
                for call in (
                    lambda: finder(attr, sqlite_pattern, search_mode=search_mode),
                    lambda: finder(attr, sqlite_pattern),
                ):
                    try:
                        result = call()
                    except TypeError:
                        continue
                    except Exception as exc:
                        if self._is_regex_too_broad_exception_090(exc):
                            self._raise_regex_too_broad_091(attr, pattern, exc, source="find_terms_regex_036l4g37c")
                        raise
                    else:
                        yield from emit_many(result)
                        return

        for method_name in ("iter_terms", "terms", "get_terms", "vocab", "term_values", "values"):
            method = getattr(self.index, method_name, None)
            if callable(method):
                for args in ((attr,), ()): 
                    try:
                        result = method(*args)
                    except TypeError:
                        continue
                    except Exception:
                        continue
                    yield from emit_many(result)

        for attr_name in ("postings", "_postings", "inverted_index", "_inverted_index", "index", "_index", "term_postings", "_term_postings"):
            obj = getattr(self.index, attr_name, None)
            if isinstance(obj, dict):
                nested = obj.get(attr)
                if isinstance(nested, dict):
                    yield from emit_many(nested.keys())
                for key in obj.keys():
                    emitted = emit_one(key)
                    if emitted is not None:
                        yield emitted

    def _doc_attr_values_for_regex_scan_090(self, doc, attr):
        attr = str(attr or "")
        if not isinstance(doc, dict):
            return []
        if attr in ("base", "lemma", "lemmas"):
            return doc.get("lemmas", []) or []
        if attr in ("orth", "token", "tokens", "form"):
            return doc.get("tokens", []) or []
        if attr in ("pos", "postag", "postags"):
            return doc.get("postags", []) or []
        if attr in ("upos", "upostag", "upostags"):
            return doc.get("upostags", []) or []
        if attr in ("deprel", "deprels"):
            return doc.get("deprels", []) or []
        if attr in ("ner", "ners"):
            return doc.get("ner", []) or doc.get("ners", []) or []
        vals = doc.get(attr, [])
        if isinstance(vals, (list, tuple)):
            return vals
        return []

    def _condition_postings_regex_scan_docs_087(self, attr, compiled, search_mode=False):
        """Compatibility fallback for non-SQLite/test indexes."""
        merged = {}
        try:
            total_docs = int(getattr(self.index, "total_docs", 0) or 0)
        except Exception:
            total_docs = 0
        for doc_id in range(total_docs):
            try:
                doc = self.index.get_doc(doc_id) or {}
            except Exception:
                continue
            values = self._doc_attr_values_for_regex_scan_090(doc, attr)
            for pos, val in enumerate(values):
                if isinstance(val, dict):
                    val = val.get(attr)
                if val is None:
                    continue
                val_s = str(val)
                for _raw, pat in compiled:
                    if (pat.search(val_s) if search_mode else pat.fullmatch(val_s)):
                        merged.setdefault(int(doc_id), set()).add(int(pos))
                        break
        return merged
    # END KORPUSUJ_MIGRATION_PATCH_91_PROPAGATE_REGEX_TOO_BROAD
    def _iter_posting_batches(self, postings, batch_size=None):
        batch_size = int(batch_size or self._candidate_stream_batch_size() or 1)
        batch = []
        try:
            iterator = sorted(postings.items(), key=lambda kv: int(kv[0]))
        except Exception:
            iterator = postings.items()
        for doc_id, positions in iterator:
            batch.append((int(doc_id), positions))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _dependency_seed_position_iterator_for_token(self, el):
        candidates = []
        for cond in el.get("dep_conds", []):
            attr = str(cond.get("attr", ""))
            if not (attr.startswith("dependent") or attr.startswith("head")):
                continue
            nested = cond.get("nested_element")
            try:
                if nested is not None:
                    anchor = self._best_nested_anchor_cond(nested)
                    est = self._condition_df(anchor) if anchor is not None else 10**12
                else:
                    est = min((int(self.index.get_term_info("base", v).get("df", 0) or 0)
                               for v in cond.get("values", [cond.get("value")]) if v is not None), default=10**12)
            except Exception:
                est = 10**12
            candidates.append((est, cond))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        cond = candidates[0][1]
        attr = str(cond.get("attr", ""))
        if attr.startswith("dependent"):
            return self._iter_parent_positions_from_dependent_cond_stream(cond)
        if attr.startswith("head"):
            return self._iter_child_positions_from_head_cond_stream(cond)
        return None

    def _iter_parent_positions_from_dependent_cond_stream(self, cond):
        nested = cond.get("nested_element")
        child_postings = {}
        if nested is not None:
            anchor = self._best_nested_anchor_cond(nested)
            if anchor is None:
                return
            child_postings = self._condition_postings(anchor)
        else:
            for v in cond.get("values", [cond.get("value")]):
                for d, ps in self.index.get_postings("base", v).items():
                    s = child_postings.setdefault(int(d), set())
                    for p in ps: s.add(int(p))
            child_postings = {d: sorted(ps) for d, ps in child_postings.items()}
        pos_count = 0; yielded = 0
        for batch in self._iter_posting_batches(child_postings):
            doc_ids = [doc_id for doc_id, _positions in batch]
            self._dep_stream_add_batch()
            self._dep_stream_add_preloaded(self._candidate_preload_dependency_docs(doc_ids, reason="dependent_stream"))
            for doc_id, child_positions in batch:
                dep_maps = self._dependency_maps(doc_id)
                if dep_maps is None: continue
                parent_idx, _children_lookup = dep_maps
                for child in child_positions:
                    pos_count += 1
                    child = int(child)
                    if child < 0 or child >= len(parent_idx): continue
                    parent = int(parent_idx[child])
                    if parent < 0: continue
                    if not self._dep_distance_ok(cond.get("distance"), parent, child, kind="dependent"): continue
                    if nested is not None and not self._match_token_at(int(doc_id), child, nested): continue
                    yielded += 1
                    yield (int(doc_id), parent)
        search_diag_log("DEP_STREAM_ANCHOR kind=dependent corpus=%r docs=%s positions=%s yielded=%s batches=%s preloaded=%s",
                        self.corpus_name, len(child_postings), pos_count, yielded, getattr(self, "_dep_stream_batches", 0), getattr(self, "_dep_stream_preloaded", 0))

    def _iter_child_positions_from_head_cond_stream(self, cond):
        nested = cond.get("nested_element")
        parent_postings = {}
        if nested is not None:
            anchor = self._best_nested_anchor_cond(nested)
            if anchor is None:
                return
            parent_postings = self._condition_postings(anchor)
        else:
            for v in cond.get("values", [cond.get("value")]):
                for d, ps in self.index.get_postings("base", v).items():
                    s = parent_postings.setdefault(int(d), set())
                    for p in ps: s.add(int(p))
            parent_postings = {d: sorted(ps) for d, ps in parent_postings.items()}
        yielded = 0
        for batch in self._iter_posting_batches(parent_postings):
            doc_ids = [doc_id for doc_id, _positions in batch]
            self._dep_stream_add_batch()
            self._dep_stream_add_preloaded(self._candidate_preload_dependency_docs(doc_ids, reason="head_stream"))
            for doc_id, parent_positions in batch:
                dep_maps = self._dependency_maps(doc_id)
                if dep_maps is None: continue
                _parent_idx, children_lookup = dep_maps
                for parent in parent_positions:
                    parent = int(parent)
                    if nested is not None and not self._match_token_at(int(doc_id), parent, nested): continue
                    for child in children_lookup[parent]:
                        child = int(child)
                        if self._dep_distance_ok(cond.get("distance"), parent, child, kind="head"):
                            yielded += 1
                            yield (int(doc_id), child)
        search_diag_log("DEP_STREAM_ANCHOR kind=head corpus=%r docs=%s yielded=%s batches=%s preloaded=%s",
                        self.corpus_name, len(parent_postings), yielded, getattr(self, "_dep_stream_batches", 0), getattr(self, "_dep_stream_preloaded", 0))

    def _candidate_preload_dependency_docs(self, doc_ids, reason=""):
        if _get_dependency_cache_ram_mode() != "candidate":
            return 0
        try:
            ids = sorted({int(x) for x in doc_ids})
        except Exception:
            ids = []
        if not ids:
            return 0
        try:
            cfg = globals().get("config", {}) or {}
            max_docs = max(1, int(cfg.get("dependency_candidate_max_docs", candidate_max_docs()) or candidate_max_docs()))
        except Exception:
            max_docs = candidate_max_docs()
        before = _dependency_ram_cache_size_for_corpus(self.corpus_name)
        remaining = max(0, max_docs - before)
        if remaining <= 0:
            search_diag_log("DEP_CANDIDATE_CURSOR_PRELOAD_SKIP_FULL reason=%r corpus=%r candidates=%s ram_size=%s max_docs=%s",
                            reason, self.corpus_name, len(ids), before, max_docs)
            return 0
        loaded = preload_dependency_maps_for_candidates(self.corpus_name, ids[:remaining])
        after = _dependency_ram_cache_size_for_corpus(self.corpus_name)
        search_diag_log("DEP_CANDIDATE_CURSOR_PRELOAD reason=%r corpus=%r candidates=%s submitted=%s loaded=%s ram_before=%s ram_after=%s",
                        reason, self.corpus_name, len(ids), min(len(ids), remaining), loaded, before, after)
        return loaded


    def _dependency_maps(self, doc_id):
        doc_id = int(doc_id)
        cached = self._dep_maps_cache.get(doc_id)
        if cached is not None:
            return cached

        mode = _get_dependency_cache_ram_mode()
        cache_key = (self.corpus_name, doc_id)
        if mode != "none":
            cached = get_dependency_maps_cache().get(cache_key)
            if cached is not None:
                self._dep_maps_cache.put(doc_id, cached)
                return cached

        if self._dep_cache is None:
            if not self.corpus_path:
                return None
            try:
                self._dep_cache = DependencyMapDiskCache(self.corpus_path)
            except Exception as e:
                search_diag_log("DEP_CACHE_OPEN_FAIL corpus_path=%r reason=%r", self.corpus_path, e)
                return None
        dep_maps = self._dep_cache.get(doc_id)
        if dep_maps is not None:
            self._dep_maps_cache.put(doc_id, dep_maps)
            # Candidate ma trzymać w globalnym RAM tylko jawnie preloadowany podzbiór.
            # All może dopisywać brakujące rekordy, none nie zapisuje nic.
            if mode == "all":
                _put_dependency_ram_cache(cache_key, dep_maps)
        return dep_maps

    def _dependency_condition_matches(self, doc_id, pos, cond, doc):
        dep_maps = self._dependency_maps(doc_id)
        if dep_maps is None: return False
        parent_idx, children_lookup = dep_maps
        pos = int(pos)
        if pos < 0 or pos >= len(parent_idx): return False
        attr = cond.get("attr", "")
        nested_el = cond.get("nested_element")
        lemmas = doc.get("lemmas", []) or doc.get("tokens", []) or []

        if attr.startswith("head"):
            parent = int(parent_idx[pos])
            if parent < 0: return False
            if not self._dep_distance_ok(cond.get("distance"), parent, pos, kind="head"): return False
            if nested_el is not None:
                return self._match_token_at(doc_id, parent, nested_el)
            return parent < len(lemmas) and self._dep_value_ok(lemmas[parent], cond)

        if attr.startswith("dependent"):
            for child in children_lookup[pos]:
                child = int(child)
                if not self._dep_distance_ok(cond.get("distance"), pos, child, kind="dependent"): continue
                if nested_el is not None:
                    if self._match_token_at(doc_id, child, nested_el): return True
                else:
                    if child < len(lemmas) and self._dep_value_ok(lemmas[child], cond): return True
            return False
        return False

    def _dep_value_ok(self, actual, cond):
        return str(actual) in {str(v) for v in cond.get("values", [cond.get("value")]) if v is not None}
    def _dep_distance_ok(self, spec, parent_pos, child_pos, kind="dependent"):
        if not spec: return True
        dist = int(parent_pos) - int(child_pos) if kind == "head" else int(child_pos) - int(parent_pos)
        op = spec.get("op", "="); val = int(spec.get("value", 0))
        if op == "=": return dist == val
        if op == ">": return dist > val
        if op == "<": return dist < val
        if op == ">=": return dist >= val
        if op == "<=": return dist <= val
        return True

    def _best_nested_anchor_cond(self, nested_el):
        best, best_df = None, None
        for cond in nested_el.get("conds", []):
            df = self._condition_df(cond)
            if best is None or df < best_df:
                best, best_df = cond, df
        return best

    def _candidate_parent_positions_from_dependent_cond(self, cond):
        nested = cond.get("nested_element")
        child_postings = {}
        if nested is not None:
            anchor = self._best_nested_anchor_cond(nested)
            if anchor is None: return None
            child_postings = self._condition_postings(anchor)
        else:
            # Semantyka legacy: dependent="X" sprawdza lemat zależnego tokenu.
            for v in cond.get("values", [cond.get("value")]):
                for d, ps in self.index.get_postings("base", v).items():
                    s = child_postings.setdefault(int(d), set())
                    for p in ps: s.add(int(p))
            child_postings = {d: sorted(ps) for d, ps in child_postings.items()}

        self._candidate_preload_dependency_docs(child_postings.keys(), reason="dependent_seed")
        out = set(); pos_count = 0
        for doc_id, child_positions in child_postings.items():
            dep_maps = self._dependency_maps(doc_id)
            if dep_maps is None: continue
            parent_idx, _children_lookup = dep_maps
            doc = self.index.get_doc(doc_id) or {}
            for child in child_positions:
                pos_count += 1
                child = int(child)
                if child < 0 or child >= len(parent_idx): continue
                parent = int(parent_idx[child])
                if parent < 0: continue
                if not self._dep_distance_ok(cond.get("distance"), parent, child, kind="dependent"): continue
                if nested is not None and not self._match_token_at(int(doc_id), child, nested): continue
                out.add((int(doc_id), parent))
        search_diag_log("DEP_ANCHOR cond_attr=%r docs=%s positions=%s parents=%s", cond.get("attr"), len(child_postings), pos_count, len(out))
        return sorted(out)

    def _candidate_child_positions_from_head_cond(self, cond):
        nested = cond.get("nested_element")
        parent_postings = {}
        if nested is not None:
            anchor = self._best_nested_anchor_cond(nested)
            if anchor is None: return None
            parent_postings = self._condition_postings(anchor)
        else:
            # Semantyka legacy: head="X" sprawdza lemat nadrzędnika.
            for v in cond.get("values", [cond.get("value")]):
                for d, ps in self.index.get_postings("base", v).items():
                    s = parent_postings.setdefault(int(d), set())
                    for p in ps: s.add(int(p))
            parent_postings = {d: sorted(ps) for d, ps in parent_postings.items()}

        self._candidate_preload_dependency_docs(parent_postings.keys(), reason="head_seed")
        out = set()
        for doc_id, parent_positions in parent_postings.items():
            dep_maps = self._dependency_maps(doc_id)
            if dep_maps is None: continue
            _parent_idx, children_lookup = dep_maps
            for parent in parent_positions:
                parent = int(parent)
                if nested is not None:
                    doc = self.index.get_doc(doc_id) or {}
                    if not self._match_token_at(int(doc_id), parent, nested): continue
                for child in children_lookup[parent]:
                    if self._dep_distance_ok(cond.get("distance"), parent, int(child), kind="head"):
                        out.add((int(doc_id), int(child)))
        return sorted(out)

    def _dependency_seed_positions_for_token(self, el):
        seeds = []
        for cond in el.get("dep_conds", []):
            if str(cond.get("attr", "")).startswith("dependent"):
                cand = self._candidate_parent_positions_from_dependent_cond(cond)
            elif str(cond.get("attr", "")).startswith("head"):
                cand = self._candidate_child_positions_from_head_cond(cond)
            else:
                cand = None
            if cand is not None:
                seeds.append(set(cand))
        if not seeds: return None
        seeds.sort(key=len)
        out = seeds[0]
        for s in seeds[1:]: out = out & s
        return sorted(out)

    def _result(self, i):
        if i in self._result_cache: return self._result_cache[i]
        doc_id, start, end = self._hits[i]
        doc = _get_doc_cached_036l4g7(self, doc_id) or {}; tokens = doc.get("tokens", []) or []; lemmas = doc.get("lemmas", []) or tokens; meta = doc.get("metadata", {}) or {}
        starts = doc.get("start_ids", []) or []; ends = doc.get("end_ids", []) or []; text = doc.get("text", "") or ""
        def _int_at(seq, idx, fallback):
            try: return int(seq[idx])
            except Exception: return int(fallback)
        char_start = _int_at(starts, start, start); char_end_excl = _int_at(ends, end - 1, end - 1) + 1 if end > 0 else char_start
        char_start = max(0, min(char_start, len(text))); char_end_excl = max(char_start, min(char_end_excl, len(text)))
        matched_text_actual = text[char_start:char_end_excl] if text else " ".join(tokens[start:end]); matched_lemmas = " ".join(lemmas[start:end]) if lemmas else matched_text_actual
        if text and starts and ends:
            left_token = max(0, start - self.left_context_size); left_start = _int_at(starts, left_token, 0) if start > 0 else char_start
            left_context = text[max(0, left_start):char_start] if start > 0 else ""
            right_token = min(len(starts) - 1, end - 1 + self.right_context_size + 1); right_limit = _int_at(starts, right_token, char_end_excl) if right_token < len(starts) else char_end_excl
            right_context = text[char_end_excl:max(char_end_excl, min(right_limit, len(text)))]
        else:
            left_context = " ".join(tokens[max(0, start - self.left_context_size):start]); right_context = " ".join(tokens[end:end + self.right_context_size])
        try: k_full_context_111 = int(get_full_context_size())
        except Exception: k_full_context_111 = 250
        publication_date = str(meta.get("Data publikacji", "")); month_key = publication_date[:7] if re.match(r"\d{4}-\d{2}", publication_date) else "Unknown"
        title = str(meta.get("Tytuł", "")); author = str(meta.get("Autor", "")); additional_metadata = {k: v for k, v in meta.items() if k not in ("Data publikacji", "Tytuł", "Autor", "doc_id")}
        # KORPUSUJ_MIGRATION_PATCH_111_LAZY_FULLTEXT_CONTEXT_ON_CLICK
        full_text_ref_111 = make_lazy_fulltext_ref_111(
            self.index,
            doc_id,
            start,
            end,
            self.left_context_size,
            self.right_context_size,
            k_full_context_111,
        )
        res = (publication_date, [left_context, matched_text_actual, right_context], full_text_ref_111, matched_text_actual, matched_lemmas, month_key, title, author, additional_metadata, left_context, right_context, doc_id, start, end)
        self._result_cache[i] = res; return res




# KORPUSUJ_MIGRATION_036L4D_SEARCHCURSOR_SAFE_RUNTIME_TIMING
def _install_searchcursor_timing_036l4d():
    import os
    import time
    import logging
    from functools import wraps

    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_timing_036l4d_installed", False):
        return

    def _enabled(self):
        val = str(os.environ.get("KORPUSUJ_SEARCHCURSOR_TIMING_036L4D", "1")).strip().lower()
        return val not in {"0", "false", "no", "off"}

    def _stats(self):
        d = getattr(self, "_timing_036l4d_stats", None)
        if d is None:
            d = {}
            self._timing_036l4d_stats = d
        return d

    def _bump(self, name, dt):
        if not _enabled(self):
            return
        d = _stats(self)
        rec = d.setdefault(name, {"calls": 0, "time": 0.0})
        rec["calls"] += 1
        rec["time"] += float(dt)

    def _wrap_method(name):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_timing_036l4d_wrapped", False):
            return
        @wraps(orig)
        def wrapper(self, *args, **kwargs):
            if not _enabled(self):
                return orig(self, *args, **kwargs)
            t0 = time.perf_counter()
            try:
                return orig(self, *args, **kwargs)
            finally:
                _bump(self, name, time.perf_counter() - t0)
        wrapper._timing_036l4d_wrapped = True
        setattr(cls, name, wrapper)

    # IMPORTANT: do NOT wrap _iter_hits. It is a generator; wrapping it caused
    # RuntimeError("generator raised StopIteration") in 036L4C2.
    for method_name in [
        "_condition_df",
        "_condition_postings",
        "_condition_matches_pos",
        "_morph_feature_condition_matches_pos_036l1b",
        "_morph_feature_condition_postings_036l1b",
        "_get_doc_cached_036l4b",
        "_ensure_until",
    ]:
        _wrap_method(method_name)

    orig_ensure_all = getattr(cls, "_ensure_all", None)
    if orig_ensure_all is not None and not getattr(orig_ensure_all, "_timing_036l4d_wrapped", False):
        @wraps(orig_ensure_all)
        def ensure_all_wrapper(self, *args, **kwargs):
            if not _enabled(self):
                return orig_ensure_all(self, *args, **kwargs)
            t0 = time.perf_counter()
            try:
                return orig_ensure_all(self, *args, **kwargs)
            finally:
                total = time.perf_counter() - t0
                _bump(self, "_ensure_all", total)
                try:
                    s = _stats(self)
                    hits = getattr(self, "_hits", None)
                    hit_count = len(hits) if hasattr(hits, "__len__") else "unknown"
                    parts = []
                    for k, v in sorted(s.items(), key=lambda kv: kv[1].get("time", 0.0), reverse=True):
                        parts.append(f"{k}: calls={v.get('calls')} time={v.get('time'):.6f}s")
                    if korpusuj_verbose_diagnostics_enabled_145c1():
                        logging.info("[DIAG perf.search.materialization] hits=%s total=%.6fs | %s", hit_count, total, " | ".join(parts))
                except Exception:
                    pass
        ensure_all_wrapper._timing_036l4d_wrapped = True
        setattr(cls, "_ensure_all", ensure_all_wrapper)

    cls._timing_036l4d_installed = True

_install_searchcursor_timing_036l4d()
# END KORPUSUJ_MIGRATION_036L4D_SEARCHCURSOR_SAFE_RUNTIME_TIMING

# KORPUSUJ_MIGRATION_036L4G6C_CURSOR_MATERIALIZATION_PROFILE_RUNTIME
def _install_cursor_materialization_profile_036l4g6c():
    import os, time, logging
    from functools import wraps
    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_profile_036l4g6c_installed", False):
        return

    def enabled():
        return str(os.environ.get("KORPUSUJ_036L4G6_PROFILE", "")).strip().lower() in ("1", "true", "yes", "tak", "on")

    def stats(self):
        d = getattr(self, "_profile_036l4g6c_stats", None)
        if d is None:
            d = {
                "iter_calls": 0, "iter_total": 0.0,
                "count_hits_calls": 0, "count_hits_total": 0.0,
                "ensure_calls": 0, "ensure_total": 0.0, "ensure_hits_added": 0,
                "get_range_calls": 0, "get_range_total": 0.0,
                "result_calls": 0, "result_total": 0.0,
                "result_cache_hits": 0, "result_cache_misses": 0,
                "get_doc_calls": 0, "get_doc_total": 0.0,
                "doc_hit_counts": {},
            }
            self._profile_036l4g6c_stats = d
        return d

    def add(d, k, v):
        try: d[k] = d.get(k, 0) + v
        except Exception: pass

    def report(self, stage):
        if not enabled():
            return
        try:
            d = stats(self)
            doc_counts = d.get("doc_hit_counts") or {}
            top_docs = sorted(doc_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            rc = int(d.get("result_calls", 0) or 0)
            gc = int(d.get("get_doc_calls", 0) or 0)
            if korpusuj_verbose_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG perf.search.materialization] "
                    "stage=%s cursor_id=%s hits=%s exhausted=%s count_cache=%s "
                    "iter_calls=%s iter_total=%.6fs count_hits_calls=%s count_hits_total=%.6fs "
                    "ensure_calls=%s ensure_total=%.6fs ensure_hits_added=%s "
                    "get_range_calls=%s get_range_total=%.6fs "
                    "result_calls=%s result_total=%.6fs result_avg=%.6fs "
                    "result_cache_hits=%s result_cache_misses=%s "
                    "get_doc_calls=%s get_doc_total=%.6fs get_doc_avg=%.6fs "
                    "unique_docs=%s top_docs=%r",
                    stage, id(self), len(getattr(self, "_hits", []) or []),
                    getattr(self, "_exhausted", None), getattr(self, "_count_cache", None),
                    d.get("iter_calls", 0), float(d.get("iter_total", 0.0) or 0.0),
                    d.get("count_hits_calls", 0), float(d.get("count_hits_total", 0.0) or 0.0),
                    d.get("ensure_calls", 0), float(d.get("ensure_total", 0.0) or 0.0), d.get("ensure_hits_added", 0),
                    d.get("get_range_calls", 0), float(d.get("get_range_total", 0.0) or 0.0),
                    rc, float(d.get("result_total", 0.0) or 0.0), (float(d.get("result_total", 0.0) or 0.0) / rc) if rc else 0.0,
                    d.get("result_cache_hits", 0), d.get("result_cache_misses", 0),
                    gc, float(d.get("get_doc_total", 0.0) or 0.0), (float(d.get("get_doc_total", 0.0) or 0.0) / gc) if gc else 0.0,
                    len(doc_counts), top_docs,
                )
        except Exception:
            pass

    def wrap_method(name, total_key, calls_key, stage=None, track_hits=False):
        orig = getattr(cls, name, None)
        if not callable(orig) or getattr(orig, "_profile_036l4g6c_wrapped", False):
            return
        @wraps(orig)
        def wrapper(self, *args, **kwargs):
            if not enabled():
                return orig(self, *args, **kwargs)
            d = stats(self)
            before = len(getattr(self, "_hits", []) or []) if track_hits else 0
            t0 = time.perf_counter()
            try:
                return orig(self, *args, **kwargs)
            finally:
                add(d, total_key, time.perf_counter() - t0)
                add(d, calls_key, 1)
                if track_hits:
                    try: add(d, "ensure_hits_added", max(0, len(getattr(self, "_hits", []) or []) - before))
                    except Exception: pass
                if stage:
                    report(self, stage)
        wrapper._profile_036l4g6c_wrapped = True
        setattr(cls, name, wrapper)

    wrap_method("_ensure_until", "ensure_total", "ensure_calls", None, True)
    wrap_method("get_range", "get_range_total", "get_range_calls", "get_range", False)
    wrap_method("count_hits", "count_hits_total", "count_hits_calls", "count_hits", False)
    wrap_method("__iter__", "iter_total", "iter_calls", "__iter__", False)

    orig_result = getattr(cls, "_result", None)
    if callable(orig_result) and not getattr(orig_result, "_profile_036l4g6c_wrapped", False):
        @wraps(orig_result)
        def result_wrapper(self, i, *args, **kwargs):
            if not enabled():
                return orig_result(self, i, *args, **kwargs)
            d = stats(self)
            add(d, "result_calls", 1)
            try:
                if i in getattr(self, "_result_cache", {}):
                    add(d, "result_cache_hits", 1)
                else:
                    add(d, "result_cache_misses", 1)
                    doc_id = int((getattr(self, "_hits", []) or [])[int(i)][0])
                    counts = d.get("doc_hit_counts") or {}
                    counts[doc_id] = counts.get(doc_id, 0) + 1
                    d["doc_hit_counts"] = counts
            except Exception:
                pass

            idx = getattr(self, "index", None)
            orig_get_doc = getattr(idx, "get_doc", None) if idx is not None else None
            patched = False
            if callable(orig_get_doc):
                def profiled_get_doc(doc_id, *gd_args, **gd_kwargs):
                    tg = time.perf_counter()
                    try:
                        return orig_get_doc(doc_id, *gd_args, **gd_kwargs)
                    finally:
                        add(d, "get_doc_calls", 1)
                        add(d, "get_doc_total", time.perf_counter() - tg)
                try:
                    setattr(idx, "get_doc", profiled_get_doc)
                    patched = True
                except Exception:
                    patched = False
            t0 = time.perf_counter()
            try:
                return orig_result(self, i, *args, **kwargs)
            finally:
                add(d, "result_total", time.perf_counter() - t0)
                if patched:
                    try: setattr(idx, "get_doc", orig_get_doc)
                    except Exception: pass
        result_wrapper._profile_036l4g6c_wrapped = True
        setattr(cls, "_result", result_wrapper)

    cls._profile_036l4g6c_installed = True

_install_cursor_materialization_profile_036l4g6c()
# END KORPUSUJ_MIGRATION_036L4G6C_CURSOR_MATERIALIZATION_PROFILE_RUNTIME

# KORPUSUJ_MIGRATION_036L4G8B_RUNTIME_PREFETCH_SEARCHCURSOR
def _install_searchcursor_prefetch_036l4g8b():
    """Install batched document prefetch support on SearchCursor."""
    import os as _os_036l4g8b
    import logging as _logging_036l4g8b

    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_prefetch_036l4g8b_installed", False):
        return

    def _enabled():
        try:
            return str(_os_036l4g8b.environ.get("KORPUSUJ_036L4G8_PREFETCH", "1")).strip().lower() not in ("0", "false", "no", "off")
        except Exception:
            return True

    def _limit():
        try:
            return max(0, int(_os_036l4g8b.environ.get("KORPUSUJ_036L4G8_PREFETCH_LIMIT", "0") or 0))
        except Exception:
            return 0

    def _prefetch_docs_for_hits_036l4g8b(self):
        try:
            hits = getattr(self, "_hits", []) or []
            if not hits:
                return 0
            cache = getattr(self, "_doc_cache_036l4g7", None)
            if cache is None:
                cache = {}
                self._doc_cache_036l4g7 = cache

            ids = []
            seen = set()
            lim = _limit()
            for h in hits:
                try:
                    doc_id = int(h[0])
                except Exception:
                    continue
                if doc_id in seen or doc_id in cache:
                    continue
                seen.add(doc_id)
                ids.append(doc_id)
                if lim and len(ids) >= lim:
                    break
            if not ids:
                return 0

            index = getattr(self, "index", None)
            loader = getattr(index, "get_docs_many_for_result_table", None)
            if not callable(loader):
                loader = getattr(index, "get_docs_many_for_results_036l4g9", None)
            if not callable(loader):
                loader = getattr(index, "get_docs_many", None)
            if not callable(loader):
                loader = getattr(index, "get_docs_many_036l4g8", None)
            if not callable(loader):
                try:
                    if korpusuj_verbose_diagnostics_enabled_145c1():
                        _logging_036l4g8b.info("[DIAG perf.search.prefetch] skipped reason=no_batch_loader ids=%s", len(ids))
                except Exception:
                    pass
                return 0

            docs = loader(ids)
            if not isinstance(docs, dict):
                return 0

            added = 0
            for doc_id, doc in docs.items():
                try:
                    cache[int(doc_id)] = doc
                except Exception:
                    cache[doc_id] = doc
                added += 1

            try:
                self._prefetch_036l4g8b_last = {"requested": len(ids), "added": added, "cache_size": len(cache)}
                if korpusuj_verbose_diagnostics_enabled_145c1():
                    _logging_036l4g8b.info("[DIAG perf.search.prefetch] requested_docs=%s loaded_docs=%s cache_size=%s", len(ids), added, len(cache))
            except Exception:
                pass
            return added
        except Exception as exc:
            try:
                if korpusuj_verbose_diagnostics_enabled_145c1():
                    _logging_036l4g8b.info("[DIAG perf.search.prefetch] failed reason=%r", exc, exc_info=True)
            except Exception:
                pass
            return 0

    setattr(cls, "_prefetch_docs_for_hits_036l4g8b", _prefetch_docs_for_hits_036l4g8b)
    setattr(cls, "_prefetch_docs_for_hits_036l4g8", _prefetch_docs_for_hits_036l4g8b)

    current_iter = getattr(cls, "__iter__", None)
    if callable(current_iter):
        base_iter = getattr(current_iter, "__wrapped__", current_iter)
    else:
        base_iter = None

    def __iter___036l4g8b(self):
        if not _enabled():
            if callable(base_iter):
                return base_iter(self)
            self._ensure_all()
            return iter(self.get_range(0, len(self._hits)))
        self._ensure_all()
        try:
            self._prefetch_docs_for_hits_036l4g8b()
        except Exception:
            pass
        return iter(self.get_range(0, len(self._hits)))

    __iter___036l4g8b.__name__ = "__iter__"
    setattr(cls, "__iter__", __iter___036l4g8b)
    cls._prefetch_036l4g8b_installed = True

_install_searchcursor_prefetch_036l4g8b()
# END KORPUSUJ_MIGRATION_036L4G8B_RUNTIME_PREFETCH_SEARCHCURSOR



# KORPUSUJ_PATCH_134_CURSOR_WINDOW_BASE_ORTH_LAZY_FILTER
def _install_searchcursor_window_conditions_134():
    try:
        import re as _re_134
        import logging as _logging_134
    except Exception:
        return
    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_window_conditions_134_installed", False):
        return
    WIN_RE = _re_134.compile(r"^window_(base|orth)(?:\((\d+)\))?$")
    RX_META = _re_134.compile(r"[\.\^\$\*\+\?\{\}\[\]\|\\\(\)]")
    INDEXED = {"base", "orth", "pos", "upos", "deprel", "ner"}

    def _attr(cond):
        try: return str(cond.get("attr") or cond.get("key") or "")
        except Exception: return ""
    def _values(cond):
        try:
            vals = cond.get("values", None)
            if vals is None: vals = [cond.get("value")]
            if isinstance(vals, (str, bytes)): vals = [vals]
            return [str(v) for v in (vals or []) if v is not None]
        except Exception: return []
    def _op(cond, neg=False):
        try: op = str(cond.get("op") or cond.get("operator") or "=")
        except Exception: op = "="
        if op == "==": op = "="
        if neg and op == "=": op = "!="
        return op
    def _mt(cond, vals):
        try: mt = str(cond.get("match_type") or "")
        except Exception: mt = ""
        if mt in {"exact", "regex", "regex_search"}: return mt
        for v in vals:
            s = str(v)
            if s.startswith("~") and len(s) > 1: return "regex_search"
            if RX_META.search(s): return "regex"
        return "exact"
    def _spec(cond, neg=False):
        if not isinstance(cond, dict): return None
        m = WIN_RE.match(_attr(cond))
        if not m: return None
        vals = _values(cond)
        try: dist = int(m.group(2)) if m.group(2) else 50
        except Exception: dist = 50
        return {"kind": m.group(1), "distance": max(0, dist), "values": vals, "operator": _op(cond, neg), "match_type": _mt(cond, vals)}
    def _split(group):
        normal, wins = [], []
        for c in list((group or {}).get("conds") or []):
            s = _spec(c, False)
            (wins if s else normal).append(s or c)
        for c in list((group or {}).get("neg_conds") or []):
            s = _spec(c, True)
            (wins if s else normal).append(s or c)
        return normal, wins
    def _doc(self, doc_id):
        for n in ("_get_doc_cached_036l4b", "_get_doc_cached"):
            f = getattr(self, n, None)
            if callable(f):
                try: return f(int(doc_id)) or {}
                except Exception: pass
        h = globals().get("_get_doc_cached_036l4g7")
        if callable(h):
            try: return h(self, int(doc_id)) or {}
            except Exception: pass
        try: return self.index.get_doc(int(doc_id)) or {}
        except Exception: return {}
    def _val_match(val, wanted, mt):
        val = str(val)
        if mt == "exact": return val in {str(v) for v in wanted}
        for raw in wanted:
            pat = str(raw)
            if mt == "regex_search" and pat.startswith("~") and len(pat) > 1: pat = pat[1:]
            try: rx = _re_134.compile(pat)
            except Exception: continue
            if (rx.search(val) if mt == "regex_search" else rx.fullmatch(val)): return True
        return False
    def _win_match(self, doc_id, pos, spec):
        d = _doc(self, doc_id)
        toks = d.get("tokens", []) or []
        lems = d.get("lemmas", []) or d.get("bases", []) or toks
        arr = lems if spec.get("kind") == "base" else toks
        try: ipos = int(pos)
        except Exception: return False
        if ipos < 0 or ipos >= len(arr): return False
        dist = int(spec.get("distance", 50) or 50)
        found = False
        for wi in range(max(0, ipos - dist), min(len(arr), ipos + dist + 1)):
            if wi == ipos: continue
            if _val_match(arr[wi], spec.get("values") or [], spec.get("match_type") or "exact"):
                found = True; break
        return found if spec.get("operator") == "=" else (not found if spec.get("operator") == "!=" else False)
    def _intersect(self, anchors):
        f = getattr(self, "_indexed_candidate_postings_036l4f2", None)
        if callable(f): return f(anchors) or {}
        cur = None
        for c in sorted(anchors, key=lambda x: len(self._condition_postings(x) or {})):
            p = {int(d): sorted({int(q) for q in ps}) for d, ps in (self._condition_postings(c) or {}).items() if ps}
            if cur is None: cur = p; continue
            out = {}
            for d, ps in cur.items():
                inter = sorted(set(ps) & set(p.get(int(d), [])))
                if inter: out[int(d)] = inter
            cur = out
            if not cur: return {}
        return cur or {}

    orig = getattr(cls, "_iter_hits", None)
    if not callable(orig): return
    def _iter_hits_window_134(self):
        try:
            groups = list(((getattr(self, "plan", {}) or {}).get("token_groups")) or [])
            if len(groups) != 1: yield from orig(self); return
            group = groups[0] or {}
            if group.get("dep_conds") or group.get("dep_neg_conds"): yield from orig(self); return
            normal, wins = _split(group)
            if not wins: yield from orig(self); return
            anchors = [c for c in normal if isinstance(c, dict) and _attr(c).split("(", 1)[0] in INDEXED]
            if not anchors: raise RuntimeError("PATCH_134_UNANCHORED_WINDOW_REQUIRES_LEGACY_FALLBACK")
            cand = _intersect(self, anchors)
            try: meta = self._meta_docs()
            except Exception: meta = None
            checked = yielded = 0
            for doc_id, positions in cand.items():
                doc_id = int(doc_id)
                if meta is not None and doc_id not in meta: continue
                for p in positions:
                    ip = int(p); checked += 1
                    if not all(_win_match(self, doc_id, ip, s) for s in wins): continue
                    ok = True
                    for c in normal:
                        try:
                            if not self._condition_matches_pos(c, doc_id, ip): ok = False; break
                        except Exception: ok = False; break
                    if ok:
                        yielded += 1
                        yield (doc_id, ip, ip + 1)
            try: korpusuj_verbose_diagnostics_enabled_145c1() and _logging_134.info("[DIAG perf.search.window_cursor] checked=%s yielded=%s", checked, yielded)
            except Exception: pass
            return
        except Exception:
            yield from orig(self)
    setattr(cls, "_iter_hits", _iter_hits_window_134)
    odf = getattr(cls, "_condition_df", None)
    if callable(odf) and not getattr(odf, "_patch_134_window_wrapped", False):
        def _condition_df_window_134(self, cond):
            try:
                if WIN_RE.match(_attr(cond)): return 10 ** 12
            except Exception: pass
            return odf(self, cond)
        _condition_df_window_134._patch_134_window_wrapped = True
        setattr(cls, "_condition_df", _condition_df_window_134)
    cls._window_conditions_134_installed = True
try:
    _install_searchcursor_window_conditions_134()
except Exception:
    pass
# END KORPUSUJ_PATCH_134_CURSOR_WINDOW_BASE_ORTH_LAZY_FILTER


# KORPUSUJ_PATCH_135_WINDOW_EXACT_DOC_AND_POSITION_PREFILTER
# Exact positive window_base/window_orth prefilter for broad anchored window queries.
# Scope:
# - anchored queries only;
# - exact window values only;
# - operator '=' only;
# - window_base uses base postings; window_orth uses orth postings;
# - preserves patch 134 fallback for regex, regex_search, !=, and unanchored queries.
def _install_searchcursor_window_exact_prefilter_135():
    try:
        import logging as _logging_135
    except Exception:
        return

    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_window_exact_prefilter_135_installed", False):
        return

    INDEXED_135 = {"base", "orth", "pos", "upos", "deprel", "ner"}

    def _attr_135(cond):
        try:
            return str(cond.get("attr") or cond.get("key") or "")
        except Exception:
            return ""

    def _values_135(cond):
        try:
            vals = cond.get("values", None)
            if vals is None:
                vals = [cond.get("value")]
            if isinstance(vals, (str, bytes)):
                vals = [vals]
            return [str(v) for v in (vals or []) if v is not None]
        except Exception:
            return []

    def _operator_135(cond, neg=False):
        try:
            op = str(cond.get("op") or cond.get("operator") or "=")
        except Exception:
            op = "="
        if op == "==":
            op = "="
        if neg and op == "=":
            op = "!="
        return op

    def _window_spec_135(cond, neg=False):
        if not isinstance(cond, dict):
            return None
        attr = _attr_135(cond)
        # No re dependency here: fast manual parse for window_base, window_base(N), etc.
        if attr.startswith("window_base"):
            kind = "base"
            prefix = "window_base"
        elif attr.startswith("window_orth"):
            kind = "orth"
            prefix = "window_orth"
        else:
            return None
        dist = 50
        suffix = attr[len(prefix):]
        if suffix:
            if not (suffix.startswith("(") and suffix.endswith(")")):
                return None
            try:
                dist = int(suffix[1:-1])
            except Exception:
                return None
        vals = _values_135(cond)
        mt = str(cond.get("match_type") or "exact")
        return {
            "attr": attr,
            "kind": kind,
            "distance": max(0, dist),
            "values": vals,
            "operator": _operator_135(cond, neg=neg),
            "match_type": mt,
            "prefilter_attr": "base" if kind == "base" else "orth",
        }

    def _split_135(group):
        normal, windows = [], []
        for c in list((group or {}).get("conds") or []):
            s = _window_spec_135(c, neg=False)
            if s is not None:
                windows.append(s)
            else:
                normal.append(c)
        # Negative buckets are deliberately not optimized here; keep patch 134 path.
        for c in list((group or {}).get("neg_conds") or []):
            s = _window_spec_135(c, neg=True)
            if s is not None:
                windows.append(s)
            else:
                normal.append(c)
        return normal, windows

    def _is_exact_positive_window_135(spec):
        if not spec:
            return False
        if spec.get("operator") != "=":
            return False
        if spec.get("match_type") != "exact":
            return False
        vals = spec.get("values") or []
        return len(vals) == 1 and vals[0] not in (None, "")

    def _intersect_anchor_postings_135(self, anchors):
        f = getattr(self, "_indexed_candidate_postings_036l4f2", None)
        if callable(f):
            return f(anchors) or {}
        current = None
        for c in sorted(anchors, key=lambda x: len(self._condition_postings(x) or {})):
            postings = self._condition_postings(c) or {}
            postings = {int(d): sorted({int(p) for p in ps}) for d, ps in postings.items() if ps}
            if current is None:
                current = postings
                continue
            out = {}
            for d, ps in current.items():
                inter = sorted(set(ps) & set(postings.get(int(d), [])))
                if inter:
                    out[int(d)] = inter
            current = out
            if not current:
                return {}
        return current or {}

    def _window_postings_135(self, spec):
        attr = spec.get("prefilter_attr")
        value = (spec.get("values") or [None])[0]
        if not attr or value is None:
            return {}
        try:
            return self.index.get_postings(attr, value) or {}
        except Exception:
            return {}

    def _positions_near_135(anchor_positions, window_positions, dist):
        # Both inputs are sorted-ish lists. Use two-pointer scan for O(a+w).
        a = sorted({int(x) for x in anchor_positions})
        w = sorted({int(x) for x in window_positions})
        if not a or not w:
            return []
        out = []
        j = 0
        n = len(w)
        for ap in a:
            while j < n and w[j] < ap - dist:
                j += 1
            k = j
            ok = False
            while k < n and w[k] <= ap + dist:
                if w[k] != ap:
                    ok = True
                    break
                k += 1
            if ok:
                out.append(ap)
        return out

    orig_iter_hits = getattr(cls, "_iter_hits", None)
    if not callable(orig_iter_hits):
        return

    def _iter_hits_window_exact_prefilter_135(self):
        try:
            plan = getattr(self, "plan", {}) or {}
            groups = list(plan.get("token_groups") or [])
            if len(groups) != 1:
                yield from orig_iter_hits(self); return
            group = groups[0] or {}
            if group.get("dep_conds") or group.get("dep_neg_conds"):
                yield from orig_iter_hits(self); return
            normal, windows = _split_135(group)
            if not windows or not all(_is_exact_positive_window_135(s) for s in windows):
                yield from orig_iter_hits(self); return
            anchors = [c for c in normal if isinstance(c, dict) and _attr_135(c).split("(", 1)[0] in INDEXED_135]
            if not anchors:
                yield from orig_iter_hits(self); return

            candidates = _intersect_anchor_postings_135(self, anchors)
            if not candidates:
                try:
                    korpusuj_verbose_diagnostics_enabled_145c1() and _logging_135.info("[DIAG perf.search.window_prefilter] anchor_candidates=0 checked=0 yielded=0")
                except Exception:
                    pass
                return

            anchor_docs_initial = len(candidates)
            anchor_positions_initial = sum(len(v) for v in candidates.values())
            filtered = {int(d): list(ps) for d, ps in candidates.items() if ps}
            window_docs_debug = []

            for spec in windows:
                wp = _window_postings_135(self, spec)
                window_docs_debug.append(len(wp))
                if not wp:
                    filtered = {}
                    break
                dist = int(spec.get("distance", 50) or 50)
                out = {}
                # Doc prefilter + position-band prefilter.
                for d, aps in filtered.items():
                    wps = wp.get(int(d))
                    if not wps:
                        continue
                    near = _positions_near_135(aps, wps, dist)
                    if near:
                        out[int(d)] = near
                filtered = out
                if not filtered:
                    break

            try:
                meta_docs = self._meta_docs()
            except Exception:
                meta_docs = None

            checked = yielded = 0
            for doc_id, positions in filtered.items():
                doc_id = int(doc_id)
                if meta_docs is not None and doc_id not in meta_docs:
                    continue
                for p in positions:
                    ip = int(p)
                    checked += 1
                    ok = True
                    # Re-check normal conditions exactly as patch 134 does.
                    for c in normal:
                        try:
                            if not self._condition_matches_pos(c, doc_id, ip):
                                ok = False; break
                        except Exception:
                            ok = False; break
                    if ok:
                        yielded += 1
                        yield (doc_id, ip, ip + 1)
            try:
                korpusuj_verbose_diagnostics_enabled_145c1() and _logging_135.info(
                    "[DIAG perf.search.window_prefilter] anchor_docs=%s anchor_positions=%s window_docs=%s checked=%s yielded=%s",
                    anchor_docs_initial,
                    anchor_positions_initial,
                    window_docs_debug,
                    checked,
                    yielded,
                )
            except Exception:
                pass
            return
        except Exception:
            # Safety: preserve existing patch 134/general cursor behavior.
            yield from orig_iter_hits(self)

    setattr(cls, "_iter_hits", _iter_hits_window_exact_prefilter_135)
    cls._window_exact_prefilter_135_installed = True

try:
    _install_searchcursor_window_exact_prefilter_135()
except Exception:
    pass
# END KORPUSUJ_PATCH_135_WINDOW_EXACT_DOC_AND_POSITION_PREFILTER


# KORPUSUJ_PATCH_136_FREQUENCY_RESULT_SET_FILTER_WITH_TOTAL_HITS_FIX
# Implements result-set scoped frequency_base/frequency_orth aggregate filtering
# for SearchCursor plans that carry frequency_operator_136 from planner.py.
def _install_searchcursor_frequency_filter_136():
    try:
        from collections import Counter as _Counter_136
        import logging as _logging_136
    except Exception:
        return

    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_frequency_filter_136_installed", False):
        return

    def _freq_136(self):
        try:
            f = (getattr(self, "plan", {}) or {}).get("frequency_operator_136")
            return f if isinstance(f, dict) and f.get("feature") in ("frequency_base", "frequency_orth") else None
        except Exception:
            return None

    def _doc_136(self, doc_id):
        for n in ("_get_doc_cached_036l4b", "_get_doc_cached"):
            f = getattr(self, n, None)
            if callable(f):
                try:
                    return f(int(doc_id)) or {}
                except Exception:
                    pass
        h = globals().get("_get_doc_cached_036l4g7")
        if callable(h):
            try:
                return h(self, int(doc_id)) or {}
            except Exception:
                pass
        try:
            return self.index.get_doc(int(doc_id)) or {}
        except Exception:
            return {}

    def _hit_value_136(self, hit, feature):
        try:
            doc_id, start, end = hit
            doc = _doc_136(self, int(doc_id)) or {}
            toks = doc.get("tokens", []) or []
            lems = doc.get("lemmas", []) or doc.get("bases", []) or toks
            start = int(start); end = int(end)
            if feature == "frequency_base":
                arr = lems
            else:
                arr = toks
            return " ".join(str(x) for x in arr[start:end]) if arr else ""
        except Exception:
            return ""

    def _allowed_values_136(counter, params):
        allowed = set(counter.keys())
        try:
            if "top" in params:
                allowed = {v for v, _c in counter.most_common(int(params.get("top") or 0))}
        except Exception:
            pass
        try:
            if "min" in params:
                mn = int(params.get("min") or 0)
                allowed = {v for v in allowed if int(counter.get(v, 0)) >= mn}
        except Exception:
            pass
        try:
            if "max" in params:
                mx = int(params.get("max") or 0)
                allowed = {v for v in allowed if int(counter.get(v, 0)) <= mx}
        except Exception:
            pass
        return allowed

    orig_iter_hits = getattr(cls, "_iter_hits", None)
    orig_count_hits_estimate = getattr(cls, "count_hits_estimate", None)
    orig_count_hits_estimate_is_exact = getattr(cls, "count_hits_estimate_is_exact", None)
    orig_len = getattr(cls, "__len__", None)
    if not callable(orig_iter_hits):
        return

    def _iter_hits_frequency_filter_136(self):
        freq = _freq_136(self)
        if not freq:
            yield from orig_iter_hits(self)
            return
        feature = freq.get("feature")
        params = dict(freq.get("params") or {})
        t0 = None
        try:
            import time as _time_136
            t0 = _time_136.perf_counter()
        except Exception:
            pass
        try:
            base_hits = list(orig_iter_hits(self))
            counter = _Counter_136(_hit_value_136(self, h, feature) for h in base_hits)
            allowed = _allowed_values_136(counter, params)
            yielded = 0
            for h in base_hits:
                if _hit_value_136(self, h, feature) in allowed:
                    yielded += 1
                    yield h
            try:
                elapsed = None
                if t0 is not None:
                    import time as _time_136
                    elapsed = _time_136.perf_counter() - t0
                korpusuj_verbose_diagnostics_enabled_145c1() and _logging_136.info(
                    "[DIAG perf.search.frequency_filter] feature=%s params=%r base_hits=%s unique=%s allowed=%s yielded=%s elapsed=%s",
                    feature, params, len(base_hits), len(counter), len(allowed), yielded,
                    f"{elapsed:.6f}s" if elapsed is not None else None,
                )
            except Exception:
                pass
            return
        except Exception as exc:
            # Conservative fallback: do not break search if aggregate code fails.
            try:
                korpusuj_verbose_diagnostics_enabled_145c1() and _logging_136.warning("[DIAG perf.search.frequency_filter] fallback reason=%r", exc, exc_info=True)
            except Exception:
                pass
            yield from orig_iter_hits(self)
            return

    def _ensure_all_frequency_136(self):
        f = getattr(self, "_ensure_all", None)
        if callable(f):
            return f()
        # Very old cursor fallback.
        if getattr(self, "_hit_iter", None) is None:
            self._hit_iter = self._iter_hits()
        while not getattr(self, "_exhausted", False):
            try:
                self._hits.append(next(self._hit_iter))
            except StopIteration:
                self._exhausted = True
                self._count_cache = len(getattr(self, "_hits", []) or [])
                break

    def count_hits_estimate_frequency_136(self):
        if _freq_136(self):
            _ensure_all_frequency_136(self)
            return len(getattr(self, "_hits", []) or [])
        if callable(orig_count_hits_estimate):
            return orig_count_hits_estimate(self)
        return len(getattr(self, "_hits", []) or [])

    def count_hits_estimate_is_exact_frequency_136(self):
        if _freq_136(self):
            return True
        if callable(orig_count_hits_estimate_is_exact):
            return orig_count_hits_estimate_is_exact(self)
        return False

    def len_frequency_136(self):
        if _freq_136(self):
            _ensure_all_frequency_136(self)
            return len(getattr(self, "_hits", []) or [])
        if callable(orig_len):
            return orig_len(self)
        return len(getattr(self, "_hits", []) or [])

    try:
        setattr(cls, "_iter_hits", _iter_hits_frequency_filter_136)
        setattr(cls, "count_hits_estimate", count_hits_estimate_frequency_136)
        setattr(cls, "count_hits_estimate_is_exact", count_hits_estimate_is_exact_frequency_136)
        setattr(cls, "__len__", len_frequency_136)
        cls._frequency_filter_136_installed = True
    except Exception:
        pass

try:
    _install_searchcursor_frequency_filter_136()
except Exception:
    pass
# END KORPUSUJ_PATCH_136_FREQUENCY_RESULT_SET_FILTER_WITH_TOTAL_HITS_FIX


# KORPUSUJ_PATCH_136B_FREQUENCY_FILTER_CACHE_REUSE
# Reuses filtered frequency hit lists across count/materialization cursors so
# patch 136 aggregate computation is not repeated for the same plan/query.
def _install_searchcursor_frequency_filter_cache_reuse_136b():
    try:
        import os as _os_136b
        import json as _json_136b
        import logging as _logging_136b
        from collections import OrderedDict as _OrderedDict_136b
    except Exception:
        return

    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_frequency_filter_cache_136b_installed", False):
        return

    orig_iter_hits = getattr(cls, "_iter_hits", None)
    if not callable(orig_iter_hits):
        return

    def _enabled_136b():
        try:
            return str(_os_136b.environ.get("KORPUSUJ_136B_FREQUENCY_CACHE", "1")).strip().lower() not in {"0", "false", "no", "off"}
        except Exception:
            return True

    def _max_entries_136b():
        try:
            return max(0, int(_os_136b.environ.get("KORPUSUJ_136B_FREQUENCY_CACHE_MAX", "8") or 8))
        except Exception:
            return 8

    def _freq_136b(self):
        try:
            f = (getattr(self, "plan", {}) or {}).get("frequency_operator_136")
            return f if isinstance(f, dict) and f.get("feature") in ("frequency_base", "frequency_orth") else None
        except Exception:
            return None

    def _index_key_136b(self):
        idx = getattr(self, "index", None)
        parts = [type(idx).__name__]
        for attr in ("path", "db_path", "index_path", "corpus_path", "name"):
            try:
                v = getattr(idx, attr, None)
                if v:
                    parts.append(f"{attr}={v}")
            except Exception:
                pass
        try:
            meta = idx.meta() if idx is not None and callable(getattr(idx, "meta", None)) else None
            if isinstance(meta, dict):
                for k in ("corpus_id", "corpus", "name", "total_docs", "total_tokens"):
                    if k in meta:
                        parts.append(f"meta.{k}={meta.get(k)}")
        except Exception:
            pass
        if len(parts) == 1:
            parts.append(f"id={id(idx)}")
        return "|".join(str(p) for p in parts)

    def _stable_plan_136b(self):
        try:
            p = dict(getattr(self, "plan", {}) or {})
            # Keep frequency operator and token/metadata plan. Drop fields that can be diagnostic only.
            keep = {}
            for k in ("token_groups", "metadata_filters", "uses_dependency", "frequency_operator_136", "frequency_base_query_136"):
                if k in p:
                    keep[k] = p[k]
            return keep
        except Exception:
            return getattr(self, "plan", {}) or {}

    def _cache_key_136b(self):
        try:
            payload = {
                "index": _index_key_136b(self),
                "plan": _stable_plan_136b(self),
            }
            return _json_136b.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return repr((_index_key_136b(self), _stable_plan_136b(self)))

    cache = getattr(cls, "_frequency_filter_cache_136b", None)
    if cache is None:
        cache = _OrderedDict_136b()
        setattr(cls, "_frequency_filter_cache_136b", cache)

    def _remember_136b(key, hits):
        try:
            max_entries = _max_entries_136b()
            if max_entries <= 0:
                return
            cache[key] = tuple((int(d), int(s), int(e)) for d, s, e in hits)
            cache.move_to_end(key)
            while len(cache) > max_entries:
                cache.popitem(last=False)
        except Exception:
            pass

    def _cached_136b(key):
        try:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        except Exception:
            pass
        return None

    def _iter_hits_frequency_cache_136b(self):
        freq = _freq_136b(self)
        if not freq or not _enabled_136b():
            yield from orig_iter_hits(self)
            return
        key = _cache_key_136b(self)
        cached = _cached_136b(key)
        if cached is not None:
            try:
                korpusuj_verbose_diagnostics_enabled_145c1() and _logging_136b.info(
                    "[DIAG perf.search.frequency_cache] event=hit feature=%s params=%r hits=%s cache_size=%s",
                    freq.get("feature"), freq.get("params"), len(cached), len(cache),
                )
            except Exception:
                pass
            for h in cached:
                yield h
            return

        hits = list(orig_iter_hits(self))
        _remember_136b(key, hits)
        try:
            korpusuj_verbose_diagnostics_enabled_145c1() and _logging_136b.info(
                "[DIAG perf.search.frequency_cache] event=store feature=%s params=%r hits=%s cache_size=%s",
                freq.get("feature"), freq.get("params"), len(hits), len(cache),
            )
        except Exception:
            pass
        for h in hits:
            yield h

    try:
        setattr(cls, "_iter_hits", _iter_hits_frequency_cache_136b)
        cls._frequency_filter_cache_136b_installed = True
    except Exception:
        pass

try:
    _install_searchcursor_frequency_filter_cache_reuse_136b()
except Exception:
    pass
# END KORPUSUJ_PATCH_136B_FREQUENCY_FILTER_CACHE_REUSE


# patch_145c3b_remove_cursor_137_137b_logging_monkeypatches: removed patch_137 global logging.info monkeypatch from cursor.py.
# Logging diagnostics are now source-gated; no runtime marker alias rewrite here.


# patch_145c3b_remove_cursor_137_137b_logging_monkeypatches: removed patch_137b global logging.info monkeypatch from cursor.py.
# Logging diagnostics are now source-gated; no runtime marker alias rewrite here.
# Executes planner plans carrying `gaps=[{"after_group", "min", "max"}, ...]`
# using offset-aware token-group matching.  Existing adjacent/no-gap execution is
# preserved by delegating to the previously installed _iter_hits.
def _install_searchcursor_gap_range_lazy_contract():
    cls = globals().get("SearchCursor")
    if cls is None or getattr(cls, "_gap_range_lazy_contract_installed", False):
        return
    original_iter_hits = getattr(cls, "_iter_hits", None)
    if not callable(original_iter_hits):
        return

    def _gap_specs(plan, group_count):
        raw = list((plan or {}).get("gaps") or [])
        if not raw:
            return None
        by_after = {}
        try:
            for g in raw:
                after = int(g.get("after_group"))
                min_gap = int(g.get("min", 0))
                max_gap = int(g.get("max", min_gap))
                if after < 0 or after >= group_count - 1 or min_gap < 0 or max_gap < min_gap:
                    return None
                by_after[after] = (min_gap, max_gap)
        except Exception:
            return None
        return by_after

    def _offset_vectors(group_count, by_after, max_vectors=10000):
        # Offsets are token positions relative to group 0.  For no explicit gap
        # between group i and i+1, adjacent semantics means gap=0 and next offset
        # increases by 1.  For [*][m,n], next offset increases by 1 + gap.
        vectors = [[0]]
        for i in range(group_count - 1):
            min_gap, max_gap = by_after.get(i, (0, 0))
            new_vectors = []
            for vec in vectors:
                base = int(vec[-1]) + 1
                for gap in range(int(min_gap), int(max_gap) + 1):
                    new_vectors.append(vec + [base + int(gap)])
                    if len(new_vectors) > max_vectors:
                        return None
            vectors = new_vectors
        return vectors

    def _match_at_offsets(self, doc_id, span_start, groups, offsets):
        try:
            span_start = int(span_start)
        except Exception:
            return False
        if span_start < 0:
            return False
        for gi, el in enumerate(groups):
            try:
                pos = span_start + int(offsets[gi])
            except Exception:
                return False
            if pos < 0:
                return False
            if not self._match_token_at(int(doc_id), pos, el):
                return False
        return True

    def _iter_hits_gap_range(self):
        plan = getattr(self, "plan", {}) or {}
        gaps_raw = plan.get("gaps") or []
        if not gaps_raw:
            yield from original_iter_hits(self)
            return

        groups = list(plan.get("token_groups") or [])
        if len(groups) < 2:
            yield from original_iter_hits(self)
            return
        if self._plan_uses_dependency():
            # Keep dependency semantics conservative for this bounded patch.
            yield from original_iter_hits(self)
            return

        by_after = _gap_specs(plan, len(groups))
        if not by_after:
            yield from original_iter_hits(self)
            return
        vectors = _offset_vectors(len(groups), by_after)
        if not vectors:
            yield from original_iter_hits(self)
            return

        anchors = []
        try:
            for gi, el in enumerate(groups):
                for cond in list((el or {}).get("conds") or []):
                    df = self._condition_df(cond)
                    if df > 0:
                        anchors.append((df, gi, cond))
        except Exception:
            anchors = []
        anchors.sort(key=lambda x: x[0])
        if not anchors:
            yield from original_iter_hits(self)
            return

        _df, anchor_group_idx, anchor_cond = anchors[0]
        try:
            meta_docs = self._meta_docs()
        except Exception:
            meta_docs = None
        anchor_postings = self._condition_postings(anchor_cond)
        seen = set()
        for doc_id, positions in (anchor_postings or {}).items():
            doc_id = int(doc_id)
            if meta_docs is not None and doc_id not in meta_docs:
                continue
            for anchor_pos in positions:
                try:
                    anchor_pos = int(anchor_pos)
                except Exception:
                    continue
                for offsets in vectors:
                    try:
                        span_start = anchor_pos - int(offsets[anchor_group_idx])
                        span_end = span_start + int(offsets[-1]) + 1
                    except Exception:
                        continue
                    if span_start < 0 or span_end <= span_start:
                        continue
                    key = (doc_id, span_start, span_end)
                    if key in seen:
                        continue
                    if _match_at_offsets(self, doc_id, span_start, groups, offsets):
                        seen.add(key)
                        yield key
        return

    _iter_hits_gap_range._gap_range_lazy_contract_wrapped = True
    setattr(cls, "_iter_hits", _iter_hits_gap_range)
    cls._gap_range_lazy_contract_installed = True

try:
    _install_searchcursor_gap_range_lazy_contract()
except Exception:
    pass
# Lazy compound cursor for top-level || plans.  It wraps ordinary SearchCursor
# children and returns their original row tuples, preserving lazy fulltext refs.
class UnionSearchCursor:
    """Merge top-level OR branches while preserving lazy paging and duplicate elimination."""
    def __init__(self, index, plan, left_context_size=10, right_context_size=10, corpus_path=None):
        self.index = index
        self.plan = plan or {}
        self.left_context_size = int(left_context_size or 10)
        self.right_context_size = int(right_context_size or 10)
        self.corpus_path = corpus_path
        self.corpus_name = corpus_path
        self._children = []
        self._items = None  # list[(child_cursor, child_index, (doc_id,start,end))]
        self._count_cache = None
        self._result_cache = {}
        self._build_children()

    def _build_children(self):
        branches = list((self.plan or {}).get("branches") or [])
        self._children = [
            SearchCursor(
                self.index,
                branch,
                self.left_context_size,
                self.right_context_size,
                corpus_path=self.corpus_path,
            )
            for branch in branches
            if isinstance(branch, dict) and branch.get("supported")
        ]

    def _hit_key_from_row(self, row):
        try:
            if isinstance(row, (tuple, list)) and len(row) > 13:
                return (int(row[11]), int(row[12]), int(row[13]))
        except Exception:
            pass
        return None

    def _ensure_child_hits(self, child):
        try:
            ensure = getattr(child, "_ensure_all", None)
            if callable(ensure):
                ensure()
        except Exception:
            pass
        try:
            hits = list(getattr(child, "_hits", []) or [])
            if hits:
                return hits
        except Exception:
            pass
        # Conservative fallback: use rows to infer keys.  This may be slower, but
        # it still preserves lazy fulltext refs because row[2] is not resolved.
        hits = []
        try:
            n = len(child)
        except Exception:
            n = 0
        try:
            rows = child.get_range(0, n) if hasattr(child, "get_range") else [child[i] for i in range(n)]
            for row in rows:
                key = self._hit_key_from_row(row)
                if key is not None:
                    hits.append(key)
        except Exception:
            pass
        return hits

    def _ensure_items(self):
        if self._items is not None:
            return
        seen = set()
        items = []
        for child in list(self._children or []):
            hits = self._ensure_child_hits(child)
            for child_index, hit in enumerate(hits):
                try:
                    key = (int(hit[0]), int(hit[1]), int(hit[2]))
                except Exception:
                    try:
                        row = child[child_index]
                    except Exception:
                        continue
                    key = self._hit_key_from_row(row)
                    if key is None:
                        continue
                if key in seen:
                    continue
                seen.add(key)
                items.append((child, int(child_index), key))
        self._items = items
        self._count_cache = len(items)

    def _ensure_all(self):
        self._ensure_items()

    def __len__(self):
        self._ensure_items()
        return int(self._count_cache or 0)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop is not None else len(self)
            return self.get_range(start, stop)
        i = int(item)
        self._ensure_items()
        if i in self._result_cache:
            return self._result_cache[i]
        child, child_index, _key = self._items[i]
        row = child[child_index]
        self._result_cache[i] = row
        return row

    def get_range(self, start, stop):
        self._ensure_items()
        start = max(0, int(start or 0))
        stop = max(start, int(stop if stop is not None else len(self)))
        stop = min(stop, len(self._items or []))
        return [self[i] for i in range(start, stop)]

    def get_page(self, page=0, page_size=100):
        start = int(page) * int(page_size)
        return self.get_range(start, start + int(page_size))

    def count_hits_estimate(self):
        # Sum child estimates cheaply. Dedupe requires _ensure_items().
        """Return the combined inexpensive estimate for the union branches."""
        total = 0
        for child in list(self._children or []):
            try:
                total += int(child.count_hits_estimate())
            except Exception:
                try:
                    total += int(len(child))
                except Exception:
                    pass
        return total

    def count_hits_estimate_is_exact(self):
        """Return whether the union estimate is guaranteed to be the final count."""
        return False

    def count_hits(self, exact=False):
        """Return the deduplicated union hit count, exactly when requested."""
        if exact or self._count_cache is None:
            self._ensure_items()
        return int(self._count_cache if self._count_cache is not None else self.count_hits_estimate())

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def _hits(self):
        self._ensure_items()
        return [key for _child, _idx, key in (self._items or [])]

try:
    globals()["UnionSearchCursor"] = UnionSearchCursor
except Exception:
    pass
# GUI sorting calls results.sort(key=..., reverse=...).  UnionSearchCursor
# is list-like but initially did not expose .sort(), so top-level || failed in
# engine.sort_search_results_in_place().  This compatibility method reorders the
# internal deduplicated item list using row tuples from child SearchCursors while
# preserving lazy fulltext refs.
def _install_union_searchcursor_sort_compat():
    cls = globals().get("UnionSearchCursor")
    if cls is None or getattr(cls, "_sort_compat_installed", False):
        return

    def sort(self, key=None, reverse=False):
        try:
            ensure = getattr(self, "_ensure_items", None)
            if callable(ensure):
                ensure()
            items = list(getattr(self, "_items", None) or [])
            decorated = []
            for pos, item in enumerate(items):
                child, child_index, hit_key = item
                try:
                    row = child[child_index]
                except Exception:
                    row = None
                try:
                    sort_key = key(row) if callable(key) else row
                except Exception:
                    sort_key = ""
                decorated.append((sort_key, pos, item))
            decorated.sort(key=lambda x: (x[0], x[1]), reverse=bool(reverse))
            self._items = [item for _sort_key, _pos, item in decorated]
            try:
                self._count_cache = len(self._items)
            except Exception:
                pass
            try:
                self._result_cache.clear()
            except Exception:
                self._result_cache = {}
            return None
        except Exception:
            # Keep list.sort-like behavior: no return value.  If sorting cannot be
            # applied, leave the cursor order unchanged rather than failing GUI.
            return None

    setattr(cls, "sort", sort)
    cls._sort_compat_installed = True

try:
    _install_union_searchcursor_sort_compat()
except Exception:
    pass
try:
    if "UnionSearchCursor" in globals():
        UnionSearchCursor._is_union_searchcursor = True
except Exception:
    pass
#
# Policy decided after bench_table_context_source_performance.py:
#
#   * Table/concordance context (row[1], row[9], row[10]) should prefer
#     document text sliced by start_ids/end_ids when doc["text"] and offsets are
#     available.
#   * Token reconstruction is only a fallback for missing text/offsets.
#   * Full text payload (row[2]) remains a lazy fulltext ref created by
#     make_lazy_fulltext_ref_111(...); it is resolved only on row click/export.
#   * UnionSearchCursor must simply return child SearchCursor rows, so it
#     inherits the same table-context policy.
#
# Rationale:
#   Text-offset contexts preserve original punctuation/spacing and benchmarked as
#   comparable to or faster than token reconstruction on the user-provided corpus.
#   Therefore there is no performance reason to force token reconstruction.

def get_table_context_source_policy():
    """Return the documented table-context source policy for scanners/tests."""
    return {
        "policy": "text_offsets_preferred",
        "table_context": "doc_text_start_ids_end_ids_when_available",
        "token_reconstruction": "fallback_only_when_text_or_offsets_missing",
        "full_text_payload": "lazy_ref_row_2_resolved_on_click_or_export",
        "applies_to": ["SearchCursor", "UnionSearchCursor_child_rows"],
        "benchmark": "bench_table_context_source_performance",
    }

try:
    TABLE_CONTEXT_SOURCE_POLICY = get_table_context_source_policy()
except Exception:
    TABLE_CONTEXT_SOURCE_POLICY = {"policy": "text_offsets_preferred"}

# --- SENTENCE_OPERATOR_CURSOR_FILTER_167G ---
def _install_sentence_operator_cursor_filter_167g():
    try:
        from korpusuj.search.sentence_operator import hit_parts, sentence_satisfies_conditions
    except Exception:
        return
    cls = globals().get('SearchCursor')
    if cls is None or getattr(cls, '_sentence_operator_cursor_filter_167g_installed', False): return
    original_iter_hits = getattr(cls, '_iter_hits', None)
    if not callable(original_iter_hits): return
    def _doc_for_sentence_operator_167g(cursor, doc_id):
        try:
            fn = globals().get('_get_doc_cached_036l4g7')
            if callable(fn): return fn(cursor, int(doc_id))
        except Exception: pass
        try: return cursor.index.get_doc(int(doc_id))
        except Exception: return None
    def iter_hits_with_sentence_operator_167g(self, *args, **kwargs):
        spec = None
        try:
            plan = getattr(self, 'plan', None)
            if isinstance(plan, dict): spec = plan.get('sentence_operator')
        except Exception: spec = None
        if not spec:
            yield from original_iter_hits(self, *args, **kwargs); return
        ordered = bool(spec.get('ordered')); conditions = spec.get('conditions') or []
        for hit in original_iter_hits(self, *args, **kwargs):
            doc_id, start, _end = hit_parts(hit)
            if doc_id is None or start is None: continue
            doc = _doc_for_sentence_operator_167g(self, doc_id)
            if not isinstance(doc, dict): continue
            try:
                if sentence_satisfies_conditions(doc, int(start), ordered, conditions): yield hit
            except Exception: continue
    setattr(cls, '_iter_hits', iter_hits_with_sentence_operator_167g)
    setattr(cls, '_sentence_operator_cursor_filter_167g_installed', True)
try: _install_sentence_operator_cursor_filter_167g()
except Exception: pass
# --- END SENTENCE_OPERATOR_CURSOR_FILTER_167G ---

# --- COREF_M_CONTIGUOUS_SHARED_CLUSTER_SPAN ---
# coref(M) is a mention-span mode inherited from the legacy engine. It is not
# an explicit Mention-* label-role filter. For a positive standalone query,
# start at a token whose cluster contains the requested token/lemma and extend
# right while consecutive tokens retain at least one shared cluster id.

def _coref_label_pairs(value):
    out = []
    seen = set()
    stack = list(value) if isinstance(value, (list, tuple, set)) else [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple, set)):
            stack[:0] = list(item)
            continue
        text = str(item or "").strip()
        if not text or text in {"0", "O", "_", "None", "none", "[]"}:
            continue
        matches = list(_COREF_LABEL_RE.finditer(text)) if "_COREF_LABEL_RE" in globals() else []
        if matches:
            for match in matches:
                role = str(match.group(1) or "").strip().lower()
                role = {"h": "head", "p": "part", "m": "mention"}.get(role, role)
                pair = (role, str(match.group(2)))
                if pair not in seen:
                    seen.add(pair)
                    out.append(pair)
            continue
        if "-" in text:
            role, cluster_id = text.split("-", 1)
            role = role.strip().lower()
            role = {"h": "head", "p": "part", "m": "mention"}.get(role, role)
            pair = (role, cluster_id.strip())
            if pair[1] and pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out


def _coref_cluster_ids_at(corefs, position):
    try:
        labels = corefs[int(position)]
    except Exception:
        return set()
    return {cluster_id for _role, cluster_id in _coref_label_pairs(labels)}


def _coref_standalone_mention_condition(plan):
    if not isinstance(plan, dict) or plan.get("sentence_operator") or plan.get("metadata_filters"):
        return None
    groups = plan.get("token_groups") or []
    if len(groups) != 1:
        return None
    group = groups[0] or {}
    conditions = list(group.get("conds", []) or [])
    if (
        len(conditions) != 1
        or group.get("neg_conds")
        or group.get("dep_conds")
        or group.get("dep_neg_conds")
    ):
        return None
    condition = conditions[0]
    if not isinstance(condition, dict):
        return None
    kind = _coref_attr_kind(condition) if "_coref_attr_kind" in globals() else None
    if kind not in {"m", "mention"}:
        return None
    if "_coref_condition_is_positive" in globals() and not _coref_condition_is_positive(condition):
        return None
    return condition


def _coref_condition_match_type(condition):
    return str(
        condition.get("match_type")
        or condition.get("mode")
        or condition.get("value_match_type")
        or "exact"
    ).strip().lower()


def _coref_document_arrays(cursor, doc_id):
    # Do not blindly reuse similarly named historical helpers: in some active
    # cursor versions _coref_doc_arrays returns a two-element internal index,
    # not (tokens, lemmas, corefs). Accept a helper only when its contract is
    # explicitly the required three-array tuple.
    for helper_name in ("_get_coref_doc_arrays", "_coref_doc_arrays"):
        helper = globals().get(helper_name)
        if not callable(helper):
            continue
        try:
            payload = helper(cursor, int(doc_id))
        except Exception:
            payload = None
        if isinstance(payload, (list, tuple)) and len(payload) == 3:
            tokens, lemmas, corefs = payload
            return list(tokens or []), list(lemmas or []), list(corefs or [])

    doc = None
    try:
        cached = globals().get("_get_doc_cached_036l4g7")
        if callable(cached):
            doc = cached(cursor, int(doc_id))
    except Exception:
        doc = None
    if not isinstance(doc, dict):
        try:
            doc = cursor.index.get_doc(int(doc_id)) or {}
        except Exception:
            doc = {}

    tokens = doc.get("tokens") or doc.get("orths") or []
    lemmas = doc.get("lemmas") or doc.get("bases") or []
    corefs = doc.get("corefs") or []
    if not corefs:
        try:
            getter = getattr(cursor.index, "get_corefs_138i3", None)
            if callable(getter):
                corefs = getter(int(doc_id)) or []
        except Exception:
            corefs = []
    return list(tokens or []), list(lemmas or []), list(corefs or [])

def _coref_matching_cluster_ids(tokens, lemmas, corefs, condition):
    values = _coref_condition_values(condition) if "_coref_condition_values" in globals() else []
    match_type = _coref_condition_match_type(condition)
    values_by_cluster = {}
    size = max(len(tokens), len(lemmas), len(corefs))
    for position in range(size):
        candidates = []
        if position < len(tokens):
            candidates.append(tokens[position])
        if position < len(lemmas):
            candidates.append(lemmas[position])
        for cluster_id in _coref_cluster_ids_at(corefs, position):
            bucket = values_by_cluster.setdefault(str(cluster_id), set())
            for candidate in candidates:
                text = str(candidate or "").strip()
                if text:
                    bucket.add(text)
    result = set()
    for cluster_id, candidates in values_by_cluster.items():
        if "_coref_text_matches" in globals():
            if any(_coref_text_matches(candidate, values, match_type) for candidate in candidates):
                result.add(cluster_id)
        elif "_match" in globals():
            normalized = {str(value).casefold() for value in candidates}
            wanted = [str(value).casefold() for value in values]
            if _match(normalized, wanted, match_type):
                result.add(cluster_id)
        elif any(str(value).casefold() in {str(x).casefold() for x in candidates} for value in values):
            result.add(cluster_id)
    return result


def _coref_expand_contiguous_span(corefs, start, end_limit):
    active_cluster_ids = _coref_cluster_ids_at(corefs, start)
    end = int(start) + 1
    if not active_cluster_ids:
        return end
    while end < int(end_limit):
        shared = active_cluster_ids.intersection(_coref_cluster_ids_at(corefs, end))
        if not shared:
            break
        active_cluster_ids = shared
        end += 1
    return end


def _coref_candidate_document_ids(cursor, condition):
    # Exact positive values are safely prefiltered by indexed base/orth terms.
    # Failure or an empty candidate set falls back to all docs to avoid false negatives.
    match_type = _coref_condition_match_type(condition)
    values = _coref_condition_values(condition) if "_coref_condition_values" in globals() else []
    candidates = set()
    if match_type == "exact":
        index = getattr(cursor, "index", None)
        for value in values:
            for attr in ("base", "orth"):
                for method_name in ("get_doc_ids_for_term", "get_postings"):
                    method = getattr(index, method_name, None)
                    if not callable(method):
                        continue
                    try:
                        payload = method(attr, value)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        candidates.update(int(doc_id) for doc_id in payload.keys())
                    else:
                        for item in payload or []:
                            if isinstance(item, (list, tuple)):
                                candidates.add(int(item[0]))
                            else:
                                candidates.add(int(item))
                    if candidates:
                        break
    if candidates:
        return sorted(candidates)
    iterator = globals().get("_coref_iter_doc_ids") or globals().get("iter_index_doc_ids_138i3")
    if callable(iterator):
        try:
            return [int(doc_id) for doc_id in iterator(cursor)]
        except Exception:
            pass
    total = int(getattr(getattr(cursor, "index", None), "total_docs", 0) or 0)
    return list(range(total))


def _iter_standalone_coref_mention_spans(cursor, condition):
    for doc_id in _coref_candidate_document_ids(cursor, condition):
        tokens, lemmas, corefs = _coref_document_arrays(cursor, doc_id)
        if not corefs:
            continue
        matching_cluster_ids = _coref_matching_cluster_ids(tokens, lemmas, corefs, condition)
        if not matching_cluster_ids:
            continue
        size = max(len(tokens), len(lemmas), len(corefs))
        position = 0
        while position < size:
            if _coref_cluster_ids_at(corefs, position).intersection(matching_cluster_ids):
                end = _coref_expand_contiguous_span(corefs, position, size)
                if end > position:
                    yield (int(doc_id), int(position), int(end))
                    position = end
                    continue
            position += 1


def _install_coref_mention_span_cursor_path():
    cursor_class = globals().get("SearchCursor")
    if cursor_class is None or getattr(cursor_class, "_coref_mention_span_installed", False):
        return
    original_iter_hits = getattr(cursor_class, "_iter_hits", None)
    if not callable(original_iter_hits):
        return

    def iter_hits_with_coref_mention_spans(self, *args, **kwargs):
        condition = _coref_standalone_mention_condition(getattr(self, "plan", None))
        if condition is None:
            yield from original_iter_hits(self, *args, **kwargs)
            return
        yield from _iter_standalone_coref_mention_spans(self, condition)

    cursor_class._iter_hits = iter_hits_with_coref_mention_spans
    cursor_class._coref_mention_span_installed = True


try:
    _install_coref_mention_span_cursor_path()
except Exception:
    pass
# --- END COREF_M_CONTIGUOUS_SHARED_CLUSTER_SPAN ---

# --- COREF_EXACT_LEMMA_CASE_SENSITIVE_SEMANTICS ---
# Exact coreference values are linguistic values, not identifiers to normalize.
# Prefer the lemma at every token position; use the surface token only when the
# lemma is genuinely absent.  Regex modes deliberately retain their established
# behavior and are delegated to the previously active implementations.

_previous_coref_condition_positions = globals().get("_coref_condition_positions_173b")
_previous_coref_exact_candidate_doc_ids = globals().get("_coref_exact_candidate_doc_ids_173b2")
_previous_coref_matching_cluster_ids = globals().get("_coref_matching_cluster_ids")
_previous_coref_candidate_document_ids = globals().get("_coref_candidate_document_ids")


def _coref_exact_positive_condition(condition):
    if not isinstance(condition, dict):
        return False
    # This predicate controls coreference-only document-array routes.  Without
    # the attribute guard, every positive exact condition (for example POS)
    # is misclassified as exact coreference during positional revalidation.
    if not _is_coref_condition(condition):
        return False
    match_type = str(
        condition.get("match_type")
        or condition.get("mode")
        or condition.get("value_match_type")
        or "exact"
    ).strip().lower()
    if match_type != "exact":
        return False
    helper = globals().get("_coref_condition_is_positive")
    if callable(helper):
        try:
            return bool(helper(condition))
        except Exception:
            pass
    operator = str(
        condition.get("op")
        or condition.get("operator")
        or condition.get("match_op")
        or "="
    ).strip().lower()
    return operator not in {"!=", "<>", "not"} and not bool(
        condition.get("negated") or condition.get("negative")
    )


def _coref_exact_query_values(condition):
    helper = globals().get("_coref_condition_values")
    if callable(helper):
        try:
            values = helper(condition)
        except Exception:
            values = None
    else:
        values = None
    if values is None:
        values = condition.get("values")
    if values is None:
        value = condition.get("value")
        values = [] if value is None else [value]
    return {str(value).strip() for value in values if str(value).strip()}


def _coref_exact_document_arrays(cursor, doc_id):
    # Use only helpers whose runtime contract is exactly three arrays.
    for helper_name in ("_coref_document_arrays", "_get_coref_doc_arrays"):
        helper = globals().get(helper_name)
        if not callable(helper):
            continue
        try:
            payload = helper(cursor, int(doc_id))
        except Exception:
            payload = None
        if isinstance(payload, (list, tuple)) and len(payload) == 3:
            tokens, lemmas, corefs = payload
            return list(tokens or []), list(lemmas or []), list(corefs or [])

    doc = None
    try:
        cached = globals().get("_get_doc_cached_036l4g7")
        if callable(cached):
            doc = cached(cursor, int(doc_id))
    except Exception:
        doc = None
    if not isinstance(doc, dict):
        try:
            doc = cursor.index.get_doc(int(doc_id)) or {}
        except Exception:
            doc = {}
    tokens = doc.get("tokens") or doc.get("orths") or []
    lemmas = doc.get("lemmas") or doc.get("bases") or []
    corefs = doc.get("corefs") or []
    if not corefs:
        try:
            getter = getattr(cursor.index, "get_corefs_138i3", None)
            if callable(getter):
                corefs = getter(int(doc_id)) or []
        except Exception:
            corefs = []
    return list(tokens or []), list(lemmas or []), list(corefs or [])


def _coref_exact_label_pairs(value):
    helper = globals().get("_coref_label_pairs")
    if callable(helper):
        try:
            return list(helper(value) or [])
        except Exception:
            pass
    parser = globals().get("_parse_coref_components")
    if callable(parser):
        try:
            return list(parser(value) or [])
        except Exception:
            pass
    return []


def _coref_exact_cluster_ids_at(corefs, position):
    try:
        labels = corefs[int(position)]
    except Exception:
        return set()
    return {str(cluster_id) for _role, cluster_id in _coref_exact_label_pairs(labels)}


def _coref_exact_semantic_value(tokens, lemmas, position):
    lemma = lemmas[position] if position < len(lemmas) else None
    if lemma is not None and str(lemma).strip() != "":
        return str(lemma).strip()
    token = tokens[position] if position < len(tokens) else None
    if token is not None and str(token).strip() != "":
        return str(token).strip()
    return None


def _coref_exact_matching_cluster_ids(tokens, lemmas, corefs, condition):
    wanted = _coref_exact_query_values(condition)
    if not wanted:
        return set()
    values_by_cluster = {}
    size = max(len(tokens), len(lemmas), len(corefs))
    for position in range(size):
        semantic_value = _coref_exact_semantic_value(tokens, lemmas, position)
        if semantic_value is None:
            continue
        for cluster_id in _coref_exact_cluster_ids_at(corefs, position):
            values_by_cluster.setdefault(cluster_id, set()).add(semantic_value)
    return {
        cluster_id
        for cluster_id, values in values_by_cluster.items()
        if values.intersection(wanted)
    }


def _coref_exact_role_kind(condition):
    helper = globals().get("_coref_attr_kind")
    if callable(helper):
        try:
            return helper(condition)
        except Exception:
            pass
    attribute = str(condition.get("attr") or condition.get("key") or "").strip().replace(" ", "").lower()
    if attribute == "coref":
        return "bare"
    if attribute.startswith("coref(") and attribute.endswith(")"):
        return attribute[6:-1].strip().lower()
    return None


def _coref_exact_positions_for_document(cursor, condition, doc_id):
    tokens, lemmas, corefs = _coref_exact_document_arrays(cursor, doc_id)
    matching_cluster_ids = _coref_exact_matching_cluster_ids(tokens, lemmas, corefs, condition)
    if not matching_cluster_ids:
        return set()
    kind = _coref_exact_role_kind(condition)
    allowed_roles = None
    if kind in {"h", "head"}:
        allowed_roles = {"head", "h"}
    elif kind in {"p", "part"}:
        allowed_roles = {"part", "p"}
    elif kind in {"m", "mention"}:
        # M owns a dedicated span iterator and must not be reduced to positions.
        return set()
    result = set()
    for position in range(max(len(tokens), len(lemmas), len(corefs))):
        for role, cluster_id in _coref_exact_label_pairs(corefs[position] if position < len(corefs) else []):
            if str(cluster_id) not in matching_cluster_ids:
                continue
            normalized_role = str(role or "").strip().lower()
            if allowed_roles is None or normalized_role in allowed_roles:
                result.add(position)
                break
    return result


def _coref_condition_positions_173b(cursor, condition, doc_id):
    if _coref_exact_positive_condition(condition) and _coref_exact_role_kind(condition) not in {"m", "mention"}:
        return _coref_exact_positions_for_document(cursor, condition, doc_id)
    if callable(_previous_coref_condition_positions):
        return _previous_coref_condition_positions(cursor, condition, doc_id)
    return set()


def _coref_matching_cluster_ids(tokens, lemmas, corefs, condition):
    if _coref_exact_positive_condition(condition):
        return _coref_exact_matching_cluster_ids(tokens, lemmas, corefs, condition)
    if callable(_previous_coref_matching_cluster_ids):
        return _previous_coref_matching_cluster_ids(tokens, lemmas, corefs, condition)
    return set()


def _coref_exact_literal_candidate_doc_ids(cursor, condition):
    values = _coref_exact_query_values(condition)
    index = getattr(cursor, "index", None)
    candidates = set()
    successful_lookup = False
    if index is not None:
        for value in values:
            # base is the semantic index. orth is only a conservative fallback
            # candidate source; the final matcher accepts it only if lemma is absent.
            for attribute in ("base", "orth"):
                for method_name in ("get_doc_ids_for_term", "get_postings"):
                    method = getattr(index, method_name, None)
                    if not callable(method):
                        continue
                    try:
                        payload = method(attribute, value)
                    except Exception:
                        continue
                    successful_lookup = True
                    if isinstance(payload, dict):
                        candidates.update(int(doc_id) for doc_id in payload.keys())
                    else:
                        for item in payload or []:
                            if isinstance(item, (list, tuple)):
                                candidates.add(int(item[0]))
                            else:
                                candidates.add(int(item))
                    break
    # An exact empty lookup is meaningful only when the index API was available.
    if successful_lookup:
        return sorted(candidates)
    iterator = globals().get("_coref_iter_doc_ids") or globals().get("iter_index_doc_ids_138i3")
    if callable(iterator):
        try:
            return [int(doc_id) for doc_id in iterator(cursor)]
        except Exception:
            pass
    total = int(getattr(index, "total_docs", 0) or 0)
    return list(range(total))


def _coref_exact_candidate_doc_ids_173b2(cursor, condition):
    if _coref_exact_positive_condition(condition):
        return set(_coref_exact_literal_candidate_doc_ids(cursor, condition))
    if callable(_previous_coref_exact_candidate_doc_ids):
        return _previous_coref_exact_candidate_doc_ids(cursor, condition)
    return None


def _coref_candidate_document_ids(cursor, condition):
    if _coref_exact_positive_condition(condition):
        return _coref_exact_literal_candidate_doc_ids(cursor, condition)
    if callable(_previous_coref_candidate_document_ids):
        return _previous_coref_candidate_document_ids(cursor, condition)
    iterator = globals().get("_coref_iter_doc_ids") or globals().get("iter_index_doc_ids_138i3")
    return list(iterator(cursor)) if callable(iterator) else []


# Rebind aliases used by older generic paths.  The 173b2 postings wrapper looks
# up `_coref_condition_positions_173b` dynamically, while direct revalidation
# may use the unsuffixed alias.
_coref_condition_positions = _coref_condition_positions_173b
# --- END COREF_EXACT_LEMMA_CASE_SENSITIVE_SEMANTICS ---

# --- COREF_EXACT_CACHE_CONTRACT ---
# Preserve the pre-existing 173b invariant: one document-index build per cursor
# and document, then cache-only positional revalidation.  The semantic exact
# values remain lemma-first and case-sensitive; this block changes only loading
# and cache reuse.

_previous_coref_exact_positions_for_document = globals().get("_coref_exact_positions_for_document")


def _coref_exact_document_index_cache(cursor):
    cache = getattr(cursor, "_coref_exact_document_index_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(cursor, "_coref_exact_document_index_cache", cache)
    return cache


def _coref_exact_cached_document_index(cursor, doc_id):
    document_id = int(doc_id)
    cache = _coref_exact_document_index_cache(cursor)
    if document_id in cache:
        # Keep the historical 173b hit telemetry and cache contract observable.
        # The builder is already primed, so this call is cache-only and performs
        # no document/corefs read.
        legacy_builder = globals().get("_build_coref_document_index_173b")
        if callable(legacy_builder):
            try:
                legacy_builder(cursor, document_id)
            except Exception:
                pass
        return cache[document_id]

    # Prime the established 173b cache and telemetry.  The later generic
    # revalidation path calls this builder, so priming it here prevents a new
    # SQLite read after postings have already been computed.
    legacy_builder = globals().get("_build_coref_document_index_173b")
    if callable(legacy_builder):
        try:
            legacy_builder(cursor, document_id)
        except Exception:
            pass

    tokens, lemmas, corefs = _coref_exact_document_arrays(cursor, document_id)
    values_by_cluster = {}
    size = max(len(tokens), len(lemmas), len(corefs))
    for position in range(size):
        semantic_value = _coref_exact_semantic_value(tokens, lemmas, position)
        if semantic_value is None:
            continue
        for cluster_id in _coref_exact_cluster_ids_at(corefs, position):
            values_by_cluster.setdefault(str(cluster_id), set()).add(semantic_value)

    index = {
        "tokens": tokens,
        "lemmas": lemmas,
        "corefs": corefs,
        "values_by_cluster": values_by_cluster,
        "size": size,
    }
    cache[document_id] = index
    return index


def _coref_exact_positions_for_document(cursor, condition, doc_id):
    document_index = _coref_exact_cached_document_index(cursor, doc_id)
    wanted = _coref_exact_query_values(condition)
    matching_cluster_ids = {
        cluster_id
        for cluster_id, values in document_index["values_by_cluster"].items()
        if values.intersection(wanted)
    }
    if not matching_cluster_ids:
        return set()

    kind = _coref_exact_role_kind(condition)
    allowed_roles = None
    if kind in {"h", "head"}:
        allowed_roles = {"head", "h"}
    elif kind in {"p", "part"}:
        allowed_roles = {"part", "p"}
    elif kind in {"m", "mention"}:
        return set()

    corefs = document_index["corefs"]
    result = set()
    for position in range(document_index["size"]):
        labels = corefs[position] if position < len(corefs) else []
        for role, cluster_id in _coref_exact_label_pairs(labels):
            if str(cluster_id) not in matching_cluster_ids:
                continue
            normalized_role = str(role or "").strip().lower()
            if allowed_roles is None or normalized_role in allowed_roles:
                result.add(position)
                break
    return result


# Rebind the wrappers installed by 173c2 so they resolve the cache-preserving
# exact-position helper above.
def _coref_condition_positions_173b(cursor, condition, doc_id):
    if _coref_exact_positive_condition(condition) and _coref_exact_role_kind(condition) not in {"m", "mention"}:
        return _coref_exact_positions_for_document(cursor, condition, doc_id)
    previous = globals().get("_previous_coref_condition_positions")
    if callable(previous):
        return previous(cursor, condition, doc_id)
    return set()


_coref_condition_positions = _coref_condition_positions_173b
# --- END COREF_EXACT_CACHE_CONTRACT ---

# --- COREF_MIXED_EXACT_POSTINGS_AND_REVALIDATION ---
# Positive exact coref / coref(H) / coref(P) conditions are final positional
# postings, not approximate document anchors.  Let the generic multi-condition
# iterator compare their true posting count with indexed token conditions and
# revalidate them through the same lemma-first, case-sensitive helper used by
# standalone coref.  coref(M), regex and negative conditions keep their existing
# dedicated paths.

_previous_condition_df_for_mixed_coref = SearchCursor._condition_df
_previous_condition_matches_pos_for_mixed_coref = SearchCursor._condition_matches_pos


def _coref_exact_mixed_condition_eligible(condition):
    if not isinstance(condition, dict):
        return False
    exact_helper = globals().get("_coref_exact_positive_condition")
    role_helper = globals().get("_coref_exact_role_kind")
    if not callable(exact_helper) or not callable(role_helper):
        return False
    try:
        if not exact_helper(condition):
            return False
        kind = role_helper(condition)
    except Exception:
        return False
    return kind in {"bare", "h", "head", "p", "part"}


def _coref_exact_mixed_positions(cursor, condition, doc_id):
    helper = globals().get("_coref_exact_positions_for_document")
    if not callable(helper):
        return None
    try:
        return set(int(position) for position in (helper(cursor, condition, int(doc_id)) or set()))
    except Exception:
        return None


def _coref_exact_mixed_condition_df(cursor, condition):
    if not _coref_exact_mixed_condition_eligible(condition):
        return None
    try:
        postings = cursor._condition_postings(condition) or {}
    except Exception:
        return None
    # SearchCursor._condition_df() uses len(postings) for ordinary indexed
    # conditions, i.e. document frequency.  Keep the same unit for exact coref
    # so anchor selection compares like with like.  Counting coref positions
    # here made broad POS postings look artificially cheaper.
    return len(postings)


def _condition_df_with_exact_coref_anchor(self, condition):
    exact_df = _coref_exact_mixed_condition_df(self, condition)
    if exact_df is not None:
        return exact_df
    return _previous_condition_df_for_mixed_coref(self, condition)


# KORPUSUJ_PATCH_174C_REUSE_FINAL_COREF_POSTINGS_FOR_REVALIDATION
# Exact positive bare/H/P coref postings are final positional results.  Once
# _condition_postings() has placed them in the cursor-local postings cache,
# positional revalidation must reuse that result instead of recomputing all
# matching coref positions for the document for every candidate position.
def _coref_final_posting_positions_cache(cursor):
    cache = getattr(cursor, "_coref_final_posting_positions_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        cursor._coref_final_posting_positions_cache = cache
    return cache


def _coref_cached_final_posting_positions(cursor, condition, doc_id):
    """Return cached final positions, or None when final postings are unavailable.

    The cache belongs to one SearchCursor and disappears with that cursor.  It is
    populated lazily from the already computed _posting_cache_local payload.  No
    SQLite access, coref document decoding, cluster-index construction, or disk
    persistence is performed here.
    """
    posting_key_helper = globals().get("_coref_posting_cache_key_173b2")
    if not callable(posting_key_helper):
        return None
    try:
        posting_key = posting_key_helper(condition)
    except Exception:
        return None
    if posting_key is None:
        return None

    posting_cache = getattr(cursor, "_posting_cache_local", None)
    if not isinstance(posting_cache, dict) or posting_key not in posting_cache:
        return None
    postings = posting_cache.get(posting_key)
    if not isinstance(postings, dict):
        return None

    try:
        document_id = int(doc_id)
    except Exception:
        document_id = doc_id
    cache_key = (posting_key, document_id)
    cache = _coref_final_posting_positions_cache(cursor)
    if cache_key not in cache:
        raw_positions = postings.get(document_id)
        if raw_positions is None and document_id != doc_id:
            raw_positions = postings.get(doc_id)
        cache[cache_key] = frozenset(int(pos) for pos in (raw_positions or ()))
    return cache[cache_key]

def _condition_matches_pos_with_exact_coref(self, condition, doc_id, position):
    if _coref_exact_mixed_condition_eligible(condition):
        # Fast path: _condition_df/_condition_postings has already constructed
        # exact final postings for this condition.  Convert one document bucket
        # to a frozenset once, then use O(1) membership for every later candidate.
        cached_positions = _coref_cached_final_posting_positions(
            self, condition, doc_id
        )
        if cached_positions is not None:
            try:
                return int(position) in cached_positions
            except Exception:
                return False

        # Safe fallback for any call path that reaches positional validation
        # before the final postings payload is available.
        positions = _coref_exact_mixed_positions(self, condition, doc_id)
        if positions is not None:
            try:
                return int(position) in positions
            except Exception:
                return False
    return _previous_condition_matches_pos_for_mixed_coref(
        self, condition, doc_id, position
    )


SearchCursor._condition_df = _condition_df_with_exact_coref_anchor
SearchCursor._condition_matches_pos = _condition_matches_pos_with_exact_coref
# --- END COREF_MIXED_EXACT_POSTINGS_AND_REVALIDATION ---

# --- EXACT_COREF_M_CANONICAL_AND_FLAT_COMPATIBILITY_RUNTIME ---
# Exact coref(M) is reconstructed per matching cluster from raw Head-C/Part-C
# roles.  Ordinary mixed-group conditions are evaluated on Head-C; the complete
# Part* Head Part* mention is emitted.  No shared-cluster expansion is used.

_coref_m_previous_iter_hits = SearchCursor._iter_hits


def _coref_m_exact_values(condition):
    values = condition.get("values") if isinstance(condition, dict) else None
    if values is None and isinstance(condition, dict):
        values = [condition.get("value")]
    return [str(value) for value in (values or []) if value is not None and str(value) != ""]


def _coref_m_is_condition(condition):
    if not isinstance(condition, dict):
        return False
    attribute = str(condition.get("attr") or condition.get("attribute") or "").replace(" ", "").lower()
    return attribute in {"coref(m)", "coref(mention)"}


def _coref_m_is_positive_exact(condition):
    if not _coref_m_is_condition(condition):
        return False
    match_type = str(
        condition.get("match_type")
        or condition.get("mode")
        or condition.get("value_match_type")
        or "exact"
    ).strip().lower()
    if match_type != "exact":
        return False
    if condition.get("negative") or condition.get("negated") or condition.get("exclude"):
        return False
    return bool(_coref_m_exact_values(condition))


def _coref_m_supported_query_shape(plan):
    """Return (mention_condition, ordinary_conditions) for supported shapes."""
    if not isinstance(plan, dict):
        return None
    if plan.get("metadata_filters") or plan.get("uses_dependency"):
        return None
    groups = plan.get("token_groups") or []
    if len(groups) != 1 or not isinstance(groups[0], dict):
        return None
    group = groups[0]
    if str(group.get("type") or "token").lower() != "token":
        return None
    if group.get("neg_conds") or group.get("dep_conds") or group.get("dep_neg_conds"):
        return None
    conditions = list(group.get("conds") or [])
    mention_conditions = [condition for condition in conditions if _coref_m_is_positive_exact(condition)]
    if len(mention_conditions) != 1:
        return None
    if sum(1 for condition in conditions if _coref_m_is_condition(condition)) != 1:
        return None
    ordinary = [condition for condition in conditions if not _coref_m_is_condition(condition)]
    if any(_is_coref_condition(condition) for condition in ordinary):
        return None
    return mention_conditions[0], ordinary


def _coref_m_condition_postings(cursor, condition):
    try:
        postings = cursor._condition_postings(condition)
    except Exception:
        return None
    if not isinstance(postings, dict):
        return None
    normalized = {}
    for doc_id, positions in postings.items():
        try:
            normalized[int(doc_id)] = {int(position) for position in (positions or [])}
        except Exception:
            return None
    return normalized


def _coref_m_ordinary_head_positions(cursor, conditions):
    """Return doc -> positions satisfying every ordinary condition."""
    if not conditions:
        return None
    maps = []
    for condition in conditions:
        postings = _coref_m_condition_postings(cursor, condition)
        if postings is None:
            return False
        maps.append(postings)
    documents = set(maps[0])
    for postings in maps[1:]:
        documents.intersection_update(postings)
    result = {}
    for doc_id in documents:
        positions = set(maps[0].get(doc_id) or set())
        for postings in maps[1:]:
            positions.intersection_update(postings.get(doc_id) or set())
        if positions:
            result[int(doc_id)] = positions
    return result


def _coref_m_candidate_documents(cursor, mention_condition, ordinary_heads):
    """Obtain complete candidate documents without an internal hit limit."""
    if ordinary_heads is not None:
        return sorted(int(doc_id) for doc_id in ordinary_heads)
    helper = globals().get("_coref_candidate_document_ids")
    if callable(helper):
        try:
            return sorted({int(doc_id) for doc_id in helper(cursor, mention_condition)})
        except Exception:
            pass
    postings = _coref_m_condition_postings(cursor, mention_condition)
    if postings is not None:
        return sorted(postings)
    return None


def _coref_m_contiguous_components(positions):
    ordered = sorted({int(position) for position in positions})
    if not ordered:
        return ()
    output = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position != previous + 1:
            output.append((start, previous + 1))
            start = position
        previous = position
    output.append((start, previous + 1))
    return tuple(output)


def _coref_m_flat_cluster_mentions(corefs, cluster_id, doc_id):
    """Yield (start, end, head) for one cluster, or fail explicitly."""
    wanted = str(cluster_id)
    roles_by_position = {}
    for position, cell in enumerate(corefs):
        roles = set()
        for role, current_cluster_id in _coref_exact_label_pairs(cell):
            if str(current_cluster_id) != wanted:
                continue
            normalized = str(role or "").strip().lower()
            if normalized in {"head", "h"}:
                roles.add("head")
            elif normalized in {"part", "p"}:
                roles.add("part")
        if roles:
            roles_by_position[int(position)] = roles

    for start, end in _coref_m_contiguous_components(roles_by_position):
        heads = [
            position for position in range(start, end)
            if "head" in roles_by_position.get(position, set())
        ]
        parts = [
            position for position in range(start, end)
            if "part" in roles_by_position.get(position, set())
        ]
        if len(heads) == 1:
            yield int(start), int(end), int(heads[0])
            continue
        if len(heads) > 1 and not parts:
            for head in heads:
                yield int(head), int(head) + 1, int(head)
            continue
        if not heads:
            raise RuntimeError(
                "174g malformed exact coref(M) component without Head-C: "
                f"doc={doc_id}, cluster={wanted}, span=({start}, {end})"
            )
        raise RuntimeError(
            "174g ambiguous exact coref(M) component with multiple Head-C and Part-C: "
            f"doc={doc_id}, cluster={wanted}, span=({start}, {end}), "
            f"heads={heads}, parts={parts}"
        )


def _coref_m_flat_compatibility_coordinates(cursor, mention_condition, ordinary_conditions):
    ordinary_heads = _coref_m_ordinary_head_positions(cursor, ordinary_conditions)
    if ordinary_heads is False:
        return None
    document_ids = _coref_m_candidate_documents(cursor, mention_condition, ordinary_heads)
    if document_ids is None:
        return None

    coordinates = set()
    for doc_id in document_ids:
        tokens, lemmas, corefs = _coref_exact_document_arrays(cursor, int(doc_id))
        tokens = list(tokens or [])
        lemmas = list(lemmas or [])
        corefs = list(corefs or [])
        if not corefs:
            continue
        matching_cluster_ids = {
            str(cluster_id)
            for cluster_id in _coref_exact_matching_cluster_ids(
                tokens, lemmas, corefs, mention_condition
            )
        }
        for cluster_id in sorted(matching_cluster_ids):
            for start, end, head in _coref_m_flat_cluster_mentions(
                corefs, cluster_id, int(doc_id)
            ):
                if ordinary_heads is not None and head not in ordinary_heads.get(int(doc_id), set()):
                    continue
                coordinates.add((int(doc_id), int(start), int(end)))
    return tuple(sorted(coordinates))


def _iter_hits_with_exact_coref_m(self, *args, **kwargs):
    shape = _coref_m_supported_query_shape(getattr(self, "plan", None))
    if shape is None:
        yield from _coref_m_previous_iter_hits(self, *args, **kwargs)
        return
    mention_condition, ordinary_conditions = shape
    coordinates = _coref_m_coordinates(
        self, mention_condition, ordinary_conditions
    )
    if coordinates is None:
        yield from _coref_m_previous_iter_hits(self, *args, **kwargs)
        return
    yield from coordinates


SearchCursor._iter_hits = _iter_hits_with_exact_coref_m
SearchCursor._exact_coref_m_flat_compatibility_available = True

# Preserve flat reconstruction as the explicit compatibility path for old source Parquets.


def _coref_m_source_has_canonical_mentions(cursor):
    """Return True/False from the source Parquet schema, or None if unknown.

    A rebuilt SQLite sidecar always has the physical coref_mentions column, so
    neither PRAGMA nor per-document key presence can distinguish old Parquet.
    Cache the authoritative source-schema answer on this SearchCursor.
    """
    cache_name = "_coref_m_source_has_canonical_mentions"
    cached = getattr(cursor, cache_name, None)
    if cached in (True, False):
        return cached

    candidate_paths = []
    cursor_path = str(getattr(cursor, "corpus_path", "") or "").strip()
    if cursor_path:
        candidate_paths.append(cursor_path)

    index = getattr(cursor, "index", None)
    connection = getattr(index, "con", None)
    if connection is not None:
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key='source_parquet_path'"
            ).fetchone()
            if row and row[0]:
                candidate_paths.append(str(row[0]))
        except Exception:
            pass

    seen = set()
    for raw_path in candidate_paths:
        path = os.path.abspath(os.path.expanduser(str(raw_path)))
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            import pyarrow.parquet as _pq_174k4
            parquet_file = _pq_174k4.ParquetFile(path)
            names = set(getattr(parquet_file.schema_arrow, "names", ()) or ())
            result = "coref_mentions" in names
            setattr(cursor, cache_name, result)
            return result
        except Exception:
            continue

    # Unknown must preserve 174j3's prior behavior.  Do not infer old format
    # merely because this query's candidate documents have no mentions.
    return None


def _coref_m_coordinates(cursor, mention_condition, ordinary_conditions):
    """Return canonical stored M spans, or delegate the whole query to flat compatibility."""
    ordinary_heads = _coref_m_ordinary_head_positions(
        cursor, ordinary_conditions
    )
    if ordinary_heads is False:
        return None

    document_ids = _coref_m_candidate_documents(
        cursor, mention_condition, ordinary_heads
    )
    if document_ids is None:
        return None
    document_ids = tuple(int(doc_id) for doc_id in document_ids)

    source_has_canonical_mentions = (
        _coref_m_source_has_canonical_mentions(cursor)
    )
    if source_has_canonical_mentions is False:
        return _coref_m_flat_compatibility_coordinates(
            cursor, mention_condition, ordinary_conditions
        )

    # Avoid mixing exact stored spans with lossy reconstruction in one result.
    documents = {}
    for doc_id in document_ids:
        doc = cursor.index.get_doc(int(doc_id))
        if not isinstance(doc, dict) or "coref_mentions" not in doc:
            return _coref_m_flat_compatibility_coordinates(
                cursor, mention_condition, ordinary_conditions
            )
        documents[int(doc_id)] = doc

    coordinates = set()
    for doc_id in document_ids:
        doc = documents[int(doc_id)]
        mentions = doc.get("coref_mentions")
        if not isinstance(mentions, list):
            raise RuntimeError(
                "174j3 invalid coref_mentions payload: expected list, "
                f"doc={doc_id}, type={type(mentions).__name__}"
            )

        tokens, lemmas, corefs = _coref_exact_document_arrays(cursor, doc_id)
        matching_cluster_ids = _coref_exact_matching_cluster_ids(
            tokens, lemmas, corefs, mention_condition
        )
        matching_cluster_ids = {str(cid) for cid in matching_cluster_ids}
        if not matching_cluster_ids:
            continue

        accepted_heads = None
        if ordinary_heads is not None:
            accepted_heads = {
                int(position)
                for position in (ordinary_heads.get(int(doc_id)) or set())
            }
            if not accepted_heads:
                continue

        token_count = len(doc.get("tokens") or tokens or [])
        for ordinal, mention in enumerate(mentions):
            if not isinstance(mention, dict):
                raise RuntimeError(
                    "174j3 invalid coref mention record: expected mapping, "
                    f"doc={doc_id}, ordinal={ordinal}, "
                    f"type={type(mention).__name__}"
                )
            try:
                cluster_id = str(mention["cluster_id"])
                start = int(mention["start"])
                end = int(mention["end"])
                head = int(mention["head"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "174j3 malformed coref mention record: "
                    f"doc={doc_id}, ordinal={ordinal}, record={mention!r}"
                ) from exc

            if cluster_id not in matching_cluster_ids:
                continue
            if not (0 <= start < end <= token_count):
                raise RuntimeError(
                    "174j3 coref mention span outside document tokens: "
                    f"doc={doc_id}, ordinal={ordinal}, span={(start, end)}, "
                    f"token_count={token_count}"
                )
            if not (start <= head < end):
                raise RuntimeError(
                    "174j3 coref mention head outside span: "
                    f"doc={doc_id}, ordinal={ordinal}, span={(start, end)}, "
                    f"head={head}"
                )
            if accepted_heads is not None and head not in accepted_heads:
                continue
            coordinates.add((int(doc_id), start, end))

    return tuple(sorted(coordinates))


SearchCursor._exact_coref_m_canonical_sidecar_available = True
# --- END EXACT_COREF_M_CANONICAL_AND_FLAT_COMPATIBILITY_RUNTIME ---


