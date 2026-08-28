# -*- coding: utf-8 -*-
"""Translate parsed CQL conditions into executable plans for the SQLite search index."""
from __future__ import annotations

import re
from korpusuj.search.parser import parse_single_condition, split_sentence_operator_query

from korpusuj.index.sqlite_index import DEFAULT_INDEXED_ATTRS


# KORPUSUJ_MIGRATION_036L2B_PLANNER_DOC_FILTER_ATTRS
# Non-indexed token attributes handled by SearchCursor via docs.full_postags.
# They are intentionally NOT part of .search/terms indexed_attrs.
DOC_FILTER_ATTRS_036L2B = {
        "number", "case", "gender", "person", "aspect", "degree",
        "accentability", "post-prepositionality", "accommodability",
        "vocalicity", "agglutination", "negation", "fullstoppedness",
    }
# END KORPUSUJ_MIGRATION_036L2B_PLANNER_DOC_FILTER_ATTRS

def _split_top_level_or(query):
    """Split only top-level || operators, outside CQL containers and strings."""
    text = str(query or "")
    parts = []
    start = 0
    index = 0
    square_depth = 0
    brace_depth = 0
    round_depth = 0
    angle_depth = 0
    quote = None
    escaped = False

    while index < len(text):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char in ('"', "'"):
            quote = char
        elif char == "[":
            square_depth += 1
        elif char == "]" and square_depth:
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif char == "(":
            round_depth += 1
        elif char == ")" and round_depth:
            round_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif (
            char == "|"
            and index + 1 < len(text)
            and text[index + 1] == "|"
            and square_depth == 0
            and brace_depth == 0
            and round_depth == 0
            and angle_depth == 0
        ):
            parts.append(text[start:index].strip())
            index += 2
            start = index
            continue
        index += 1

    parts.append(text[start:].strip())
    if len(parts) <= 1 or any(not part for part in parts):
        return None
    return parts


def _strip_balanced_outer_parentheses(query):
    """Remove one outer (...) pair only when it encloses the complete query."""
    text = str(query or "").strip()
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return text

    depth = 0
    quote = None
    escaped = False
    square_depth = 0
    brace_depth = 0
    angle_depth = 0

    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ('"', "'"):
            quote = char
            continue
        if char == "[":
            square_depth += 1
            continue
        if char == "]" and square_depth:
            square_depth -= 1
            continue
        if char == "{":
            brace_depth += 1
            continue
        if char == "}" and brace_depth:
            brace_depth -= 1
            continue
        if char == "<":
            angle_depth += 1
            continue
        if char == ">" and angle_depth:
            angle_depth -= 1
            continue
        if square_depth or brace_depth or angle_depth:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return text
            if depth == 0 and index != len(text) - 1:
                return text

    if quote is not None or depth != 0:
        return text
    return text[1:-1].strip()


class SearchPlanner:
    """Translate parsed CQL into executable indexed-search plans."""
    INDEXABLE_ATTRS = set(DEFAULT_INDEXED_ATTRS)
    DEP_ATTR_RE = re.compile(r'^(head|dependent)(?:\(([^)]*)\))?$', re.IGNORECASE)

    def plan(self, query, index):
        """Create an executable plan, including native top-level OR unions."""
        plan = self._plan_non_union(query, index)
        if not (
            isinstance(plan, dict)
            and plan.get("supported") is False
            and str(plan.get("reason") or "") == "elementy złożone CQL"
        ):
            return plan

        branches = _split_top_level_or(query)
        if not branches:
            return plan

        branch_plans = []
        branch_queries = []
        uses_dependency = False
        for branch_query in branches:
            normalized_query = _strip_balanced_outer_parentheses(branch_query)
            branch_plan = self._plan_non_union(normalized_query, index)
            if not (isinstance(branch_plan, dict) and branch_plan.get("supported")):
                return plan
            if str(branch_plan.get("type") or "") == "union":
                return plan
            branch_copy = dict(branch_plan)
            branch_copy["or_union_branch_query"] = branch_query
            branch_copy["or_union_normalized_query"] = normalized_query
            branch_plans.append(branch_copy)
            branch_queries.append(branch_query)
            uses_dependency = bool(uses_dependency or branch_copy.get("uses_dependency"))

        return {
            "supported": True,
            "type": "union",
            "branches": branch_plans,
            "branch_queries": branch_queries,
            "metadata_filters": [],
            "uses_dependency": uses_dependency,
            "or_union_lazy_contract": True,
        }

    def _plan_non_union(self, query, index):
        """Create an ordinary non-union plan for a query and search index."""
        sentence_operator = split_sentence_operator_query(query)
        if sentence_operator:
            lhs_plan = self.plan(sentence_operator["token_part"], index)
            rhs_plan = self.plan(sentence_operator["sentence_part"], index)
            if not isinstance(lhs_plan, dict) or not isinstance(rhs_plan, dict):
                raise ValueError("sentence operator requires plannable LHS and RHS queries")
            combined_plan = dict(lhs_plan)
            combined_plan["sentence_operator"] = {
                "token_part": sentence_operator["token_part"],
                "sentence_part": sentence_operator["sentence_part"],
                "rhs_plan": rhs_plan,
            }
            return combined_plan
        available_attrs = set(self.INDEXABLE_ATTRS)
        try:
            meta_attrs = (index.meta().get("indexed_attrs", "") if index is not None else "")
            if meta_attrs:
                available_attrs = {a.strip() for a in meta_attrs.split(",") if a.strip()}
        except Exception:
            pass

        q = re.sub(r"<frequency_(?:orth|base)\s+[^>]+>", "", query or "", flags=re.IGNORECASE).strip()
        q, metadata_filters = self._extract_metadata(q)

        try:
            contents = self._extract_square_brackets(q)
            if not contents:
                return {"supported": False, "reason": "brak prostych nawiasów tokenowych"}
            if self._strip_square_brackets(q).strip():
                return {"supported": False, "reason": "elementy złożone CQL"}
            groups = [self._parse_token_element(content, available_attrs) for content in contents]
        except ValueError as e:
            return {"supported": False, "reason": str(e)}

        return {
            "supported": True,
            "token_groups": groups,
            "metadata_filters": metadata_filters,
            "uses_dependency": any(self._element_uses_dependency(g) for g in groups),
        }

    def _extract_square_brackets(self, q):
        out = []
        i = 0
        while i < len(q):
            if q[i] != "[":
                i += 1
                continue
            j = i + 1
            in_str = False
            esc = False
            brace_depth = 0
            while j < len(q):
                ch = q[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth = max(0, brace_depth - 1)
                    elif ch == "]" and brace_depth == 0:
                        out.append(q[i + 1:j])
                        i = j + 1
                        break
                j += 1
            else:
                raise ValueError("niedomknięty nawias []")
        return out

    def _strip_square_brackets(self, q):
        s = list(q)
        i = 0
        while i < len(q):
            if q[i] != "[":
                i += 1
                continue
            j = i + 1
            in_str = False
            esc = False
            brace_depth = 0
            while j < len(q):
                ch = q[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth = max(0, brace_depth - 1)
                    elif ch == "]" and brace_depth == 0:
                        for k in range(i, j + 1):
                            s[k] = " "
                        i = j + 1
                        break
                j += 1
            else:
                break
        return "".join(s)

    def _split_top_level_amp(self, s):
        parts = []
        buf = []
        in_str = False
        esc = False
        brace_depth = 0
        for ch in s:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                buf.append(ch)
            elif ch == "{":
                brace_depth += 1
                buf.append(ch)
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
                buf.append(ch)
            elif ch == "&" and brace_depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        part = "".join(buf).strip()
        if part:
            parts.append(part)
        return parts

    def _parse_token_element(self, content, available_attrs):
        el = {"type": "token", "conds": [], "neg_conds": [], "dep_conds": [], "dep_neg_conds": []}
        for part in self._split_top_level_amp(content.strip()):
            cond = self._parse_condition(part, available_attrs)
            is_dep = cond.pop("is_dependency", False)
            is_neg = cond.pop("neg", False)
            key = "dep_conds" if is_dep else "conds"
            if is_neg:
                key = "dep_neg_conds" if is_dep else "neg_conds"
            el[key].append(cond)
        return el

    @staticmethod
    def _classify_token_match_type_087(value):
        if not isinstance(value, str):
            return "exact"
        if value.startswith("~") and len(value) > 1:
            return "regex_search"
        if re.search(r"[\[\]\\.\^\$\*\+\?\{\}\(\)]", value):
            return "regex"
        return "exact"

    def _parse_condition(self, part, available_attrs):
        # head={...}, dependent={...}, head!={...}, dependent!={...}
        m_nested = re.fullmatch(r'([A-Za-z_][\w.\-]*(?:\([^)]*\))?)\s*(!=|=)\s*\{(.*)\}\s*', part, flags=re.S)
        if m_nested:
            raw_attr, op, inner = m_nested.group(1), m_nested.group(2), m_nested.group(3)
            dep_kind, dist = self._parse_dep_attr(raw_attr)
            if not dep_kind:
                raise ValueError("zagnieżdżenie {...} obsługiwane tylko dla head/dependent")
            nested_el = self._parse_token_element(inner, available_attrs)
            return {
                "attr": dep_kind,
                "raw_attr": raw_attr,
                "match_type": "nested",
                "nested_element": nested_el,
                "distance": dist,
                "neg": op == "!=",
                "is_dependency": True,
            }

        # attr="value" lub attr!="value"
        m = re.fullmatch(r'([A-Za-z_][\w.\-]*(?:\([^)]*\))?)\s*(=|!=)\s*"([^"]*)"', part.strip())
        if not m:
            raise ValueError("warunek spoza prostego indeksu")
        raw_attr, op, value = m.group(1), m.group(2), m.group(3)
        match_type_087 = self._classify_token_match_type_087(value)
        if match_type_087 == "regex_search":
            value = value[1:]
        dep_kind, dist = self._parse_dep_attr(raw_attr)
        if dep_kind:
            parsed_attr, parsed_values, parsed_op, is_nested, parsed_match_type = (
                parse_single_condition(part.strip())
            )
            if is_nested or str(parsed_attr).strip() != str(raw_attr).strip():
                raise ValueError("niespójny prosty warunek zależnościowy")
            values = list(parsed_values or [])
            if not values:
                raise ValueError("pusta wartość prostego warunku zależnościowego")
            return {
                "attr": dep_kind,
                "raw_attr": raw_attr,
                "values": values,
                "value": values[0],
                "match_type": parsed_match_type,
                "distance": dist,
                "neg": parsed_op == "!=",
                "is_dependency": True,
            }
        if raw_attr not in available_attrs and raw_attr not in DOC_FILTER_ATTRS_036L2B:
            raise ValueError("warunek nieindeksowalny w aktywnym profilu indeksu")
        return {
            "attr": raw_attr,
            "values": [value],
            "value": value,
            "match_type": match_type_087,
            "neg": op == "!=",
            "is_dependency": False,
        }

    def _parse_dep_attr(self, raw_attr):
        m = self.DEP_ATTR_RE.fullmatch(str(raw_attr or "").strip())
        if not m:
            return None, None
        return m.group(1).lower(), self._parse_distance(m.group(2)) if m.group(2) else None

    def _parse_distance(self, raw):
        raw = str(raw or "").strip()
        m = re.fullmatch(r'(<=|>=|<|>|=)?\s*(-?\d+)', raw)
        if not m:
            return None
        return {"op": m.group(1) or "=", "value": int(m.group(2))}

    def _element_uses_dependency(self, el):
        if el.get("dep_conds") or el.get("dep_neg_conds"):
            return True
        for cond in el.get("dep_conds", []) + el.get("dep_neg_conds", []):
            nested = cond.get("nested_element")
            if nested and self._element_uses_dependency(nested):
                return True
        return False

    # KORPUSUJ_MIGRATION_036L4G42_METADATA_OPERATOR_NORMALIZATION
    def _extract_metadata(self, q):
        """Extract and normalize metadata filters from CQL."""
        filters = []
        key_map = {"autor": "Autor", "tytuł": "Tytuł", "tytul": "Tytuł", "data": "Data publikacji"}
        field_pat = r'(autor|tytuł|tytul|data|metadane:[^\s<>!=]+)'

        def classify(v):
            if v.startswith("~") and len(v) > 1:
                return "regex_search", v[1:]
            if re.search(r"[\.\^\$\*\+\?\{\}\[\]\|\\\(\)]", v):
                return "regex", v
            return "exact", v

        def normalize_key(raw):
            return key_map.get(str(raw).lower(), raw)

        def add_filter(raw_key, op, raw_value):
            key = normalize_key(raw_key)
            mt, val = classify(raw_value)
            filters.append((key, op, val, mt))
            return ""

        # KORPUSUJ_MIGRATION_036L4G42E_SPLIT_GT
        # Explicit split forms:
        #   <data>="..."> means >=
        #   <data>"...">  means >
        split_ge = re.compile(r'<\s*' + field_pat + r'\s*>\s*=\s*"([^"]*)"\s*>', flags=re.IGNORECASE)
        q = split_ge.sub(lambda m: add_filter(m.group(1), ">=", m.group(2)), q)

        split_gt = re.compile(r'<\s*' + field_pat + r'\s*>\s*"([^"]*)"\s*>', flags=re.IGNORECASE)
        q = split_gt.sub(lambda m: add_filter(m.group(1), ">", m.group(2)), q)

        # Canonical in-tag operator form.
        canonical = re.compile(
            r'<\s*' + field_pat + r'\s*(?:(<=|>=|!=|=)\s*|([<>])\s+)"([^"]*)"\s*>',
            flags=re.IGNORECASE,
        )
        q = canonical.sub(lambda m: add_filter(m.group(1), m.group(2) or m.group(3), m.group(4)), q)
        return q.strip(), filters

# --- KORPUSUJ_FIX_016_DEPREL_PIPE_OR ---
# Compatibility fix:
# Legacy parser treats deprel="a|b" as exact OR values ["a", "b"].
# SearchPlanner may classify "|" as regex, which breaks dependency nested matching.
# This post-processes the produced plan and normalizes only deprel pipe values.
def _korpusuj_fix_016_split_deprel_pipe_value(value):
    if not isinstance(value, str):
        return None

    value = value.strip()

    if "|" not in value:
        return None

    # Do not rewrite explicit regex-search notation.
    if value.startswith("~"):
        return None

    # Do not rewrite slash-wrapped regex notation if such notation is used.
    if len(value) >= 2 and value.startswith("/") and value.endswith("/"):
        return None

    parts = [part.strip() for part in value.split("|") if part.strip()]

    if len(parts) <= 1:
        return None

    return parts


def _korpusuj_fix_016_normalize_deprel_pipe_or(obj):
    """
    Recursively normalize SearchPlanner output.

    Converts deprel values like:
        {"attr": "deprel", "value": "nsubj|nsubj:pass", "match_type": "regex"}

    into:
        {"attr": "deprel", "value": "nsubj", "values": ["nsubj", "nsubj:pass"], "match_type": "exact"}

    This is intentionally limited to attr == "deprel".
    """
    if isinstance(obj, dict):
        attr = str(obj.get("attr", "")).strip().lower()

        if attr == "deprel":
            raw_values = []

            if isinstance(obj.get("values"), list):
                raw_values.extend(obj.get("values") or [])
            elif "value" in obj:
                raw_values.append(obj.get("value"))

            normalized_values = []
            changed = False

            for value in raw_values:
                split_values = _korpusuj_fix_016_split_deprel_pipe_value(value)

                if split_values:
                    normalized_values.extend(split_values)
                    changed = True
                else:
                    normalized_values.append(value)

            if changed:
                normalized_values = [
                    str(v).strip()
                    for v in normalized_values
                    if str(v).strip()
                ]

                # Preserve order, remove duplicates.
                seen = set()
                deduped = []

                for value in normalized_values:
                    if value not in seen:
                        seen.add(value)
                        deduped.append(value)

                obj["values"] = deduped
                obj["value"] = deduped[0] if deduped else obj.get("value")
                obj["match_type"] = "exact"

        for value in list(obj.values()):
            _korpusuj_fix_016_normalize_deprel_pipe_or(value)

    elif isinstance(obj, list):
        for item in obj:
            _korpusuj_fix_016_normalize_deprel_pipe_or(item)

    elif isinstance(obj, tuple):
        for item in obj:
            _korpusuj_fix_016_normalize_deprel_pipe_or(item)

    return obj


try:
    _KORPUSUJ_FIX_016_ORIGINAL_PLAN = SearchPlanner.plan

    def _korpusuj_fix_016_plan(self, *args, **kwargs):
        plan = _KORPUSUJ_FIX_016_ORIGINAL_PLAN(self, *args, **kwargs)
        return _korpusuj_fix_016_normalize_deprel_pipe_or(plan)

    SearchPlanner.plan = _korpusuj_fix_016_plan

except Exception:
    # Planner import should never fail because of the compatibility patch.
    pass
# --- END KORPUSUJ_FIX_016_DEPREL_PIPE_OR ---


# KORPUSUJ_PATCH_134_PLANNER_WINDOW_ATTR_SUPPORT
def _install_searchplanner_window_attr_support_134():
    try: import re as _re_134
    except Exception: return
    WIN = {"window_base", "window_orth"}; ANCH = {"base", "orth", "pos", "upos", "deprel", "ner"}
    CRE = _re_134.compile(r"\b([A-Za-z_][A-Za-z0-9_.-]*)(?:\(\d+\))?\s*(?:!?=)")
    def anchored(q):
        attrs = set(m.group(1) for m in CRE.finditer(str(q or "")))
        return bool(attrs & WIN) and bool(attrs & ANCH)
    def widen(obj):
        undo=[]
        for target in (globals(), obj, obj.__class__):
            try:
                items = target.items() if isinstance(target, dict) else ((n, getattr(target, n)) for n in dir(target))
                for name, val in list(items):
                    if not isinstance(val, (set, frozenset, tuple, list)): continue
                    if not any(x in str(name).lower() for x in ("attr", "key", "support", "index")): continue
                    try: s=set(val)
                    except Exception: continue
                    if not (s & ANCH or {"number", "case", "gender"} & s): continue
                    if WIN <= s: continue
                    new=set(s)|WIN; undo.append((target,name,val))
                    if isinstance(target, dict): target[name]=type(val)(new) if not isinstance(val, frozenset) else frozenset(new)
                    else: setattr(target,name,type(val)(new) if not isinstance(val, frozenset) else frozenset(new))
            except Exception: pass
        return undo
    def restore(undo):
        for target,name,old in reversed(undo or []):
            try:
                if isinstance(target, dict): target[name]=old
                else: setattr(target,name,old)
            except Exception: pass
    patched=0
    for name,obj in list(globals().items()):
        if not isinstance(obj,type) or not hasattr(obj,"plan"): continue
        orig=getattr(obj,"plan",None)
        if not callable(orig) or getattr(orig,"_patch_134_window_wrapped",False): continue
        def make(orig_plan):
            def plan_window_134(self, query, *args, **kwargs):
                if not anchored(query): return orig_plan(self, query, *args, **kwargs)
                undo=widen(self)
                try:
                    plan=orig_plan(self, query, *args, **kwargs)
                    if isinstance(plan,dict): plan["patch_134_window_conditions"]=True
                    return plan
                finally: restore(undo)
            plan_window_134._patch_134_window_wrapped=True
            return plan_window_134
        try: setattr(obj,"plan",make(orig)); patched+=1
        except Exception: pass
    globals()["_patch_134_planner_window_attr_support_count"] = patched
try:
    _install_searchplanner_window_attr_support_134()
except Exception:
    pass
# END KORPUSUJ_PATCH_134_PLANNER_WINDOW_ATTR_SUPPORT


# KORPUSUJ_PATCH_134B_WINDOW_DISTANCE_ATTR_PLANNER_SUPPORT
def _install_searchplanner_window_distance_attr_support_134b():
    try:
        import re as _re_134b
    except Exception:
        return
    WIN = {"window_base", "window_orth"}
    ANCH = {"base", "orth", "pos", "upos", "deprel", "ner"}
    ATTR_RE = _re_134b.compile(r"\b([A-Za-z_][A-Za-z0-9_.-]*(?:\(\d+\))?)\s*(?:!?=)")
    WIN_DIST_RE = _re_134b.compile(r"^window_(?:base|orth)\(\d+\)$")

    def query_dynamic_window_attrs(q):
        attrs = set(m.group(1) for m in ATTR_RE.finditer(str(q or "")))
        bases = {a.split("(", 1)[0] for a in attrs}
        dyn = {a for a in attrs if WIN_DIST_RE.match(a)}
        if dyn and (bases & ANCH):
            return dyn
        return set()

    def container_name_ok(name):
        n = str(name).lower()
        return any(x in n for x in ("attr", "key", "support", "index", "valid", "allowed"))

    def container_relevant(val):
        try: s = set(val)
        except Exception: return False
        return bool((s & ANCH) or (s & WIN) or ({"number", "case", "gender"} & s))

    def same_type(old, new):
        if isinstance(old, frozenset): return frozenset(new)
        if isinstance(old, tuple): return tuple(new)
        if isinstance(old, list): return list(new)
        return set(new)

    def widen(instance, dyn):
        undo = []
        if not dyn: return undo
        add = set(dyn) | WIN
        for target in (globals(), instance, instance.__class__):
            try:
                items = target.items() if isinstance(target, dict) else ((n, getattr(target, n)) for n in dir(target))
                for name, val in list(items):
                    if not isinstance(val, (set, frozenset, tuple, list)): continue
                    if not container_name_ok(name): continue
                    if not container_relevant(val): continue
                    try: s = set(val)
                    except Exception: continue
                    if add <= s: continue
                    undo.append((target, name, val))
                    if isinstance(target, dict): target[name] = same_type(val, s | add)
                    else: setattr(target, name, same_type(val, s | add))
            except Exception:
                pass
        return undo

    def restore(undo):
        for target, name, old in reversed(undo or []):
            try:
                if isinstance(target, dict): target[name] = old
                else: setattr(target, name, old)
            except Exception:
                pass

    patched = 0
    for _name, _obj in list(globals().items()):
        try:
            if not isinstance(_obj, type): continue
            orig = getattr(_obj, "plan", None)
            if not callable(orig) or getattr(orig, "_patch_134b_window_distance_wrapped", False): continue
            def make_wrapper(orig_plan):
                def plan_window_distance_134b(self, query, *args, **kwargs):
                    dyn = query_dynamic_window_attrs(query)
                    if not dyn:
                        return orig_plan(self, query, *args, **kwargs)
                    undo = widen(self, dyn)
                    try:
                        result = orig_plan(self, query, *args, **kwargs)
                        if isinstance(result, dict):
                            result["patch_134b_window_distance_attrs"] = sorted(dyn)
                        return result
                    finally:
                        restore(undo)
                plan_window_distance_134b._patch_134b_window_distance_wrapped = True
                return plan_window_distance_134b
            setattr(_obj, "plan", make_wrapper(orig))
            patched += 1
        except Exception:
            pass
    globals()["_patch_134b_window_distance_attr_support_count"] = patched
try:
    _install_searchplanner_window_distance_attr_support_134b()
except Exception:
    pass
# END KORPUSUJ_PATCH_134B_WINDOW_DISTANCE_ATTR_PLANNER_SUPPORT


# KORPUSUJ_PATCH_136_FREQUENCY_OPERATOR_PLAN_CAPTURE
# Captures legacy XML frequency tags as aggregate operator config on the plan.
def _install_frequency_operator_plan_capture_136():
    try:
        import re as _re_136
    except Exception:
        return

    TAG_RE = _re_136.compile(r"<\s*(frequency_base|frequency_orth)\s+([^>]*)>", _re_136.IGNORECASE)
    PARAM_RE = _re_136.compile(r"\b(top|min|max)\s*=\s*\"(\d+)\"", _re_136.IGNORECASE)

    def _extract_frequency_operator_136(query):
        q = query or ""
        matches = list(TAG_RE.finditer(q))
        if not matches:
            return None
        # Bounded patch: one operator only. If multiple are present, keep the first
        # and mark ambiguity for downstream diagnostics rather than rejecting here.
        m = matches[0]
        params = {k.lower(): int(v) for k, v in PARAM_RE.findall(m.group(2) or "")}
        return {
            "feature": m.group(1).lower(),
            "params": params,
            "raw_tag": m.group(0),
            "tag_count": len(matches),
            "patch": "136",
        }

    for _name, _obj in list(globals().items()):
        if not isinstance(_obj, type):
            continue
        plan_fn = getattr(_obj, "plan", None)
        if not callable(plan_fn) or getattr(plan_fn, "_frequency_operator_136_wrapped", False):
            continue

        def _make_wrapper_136(orig):
            def plan_with_frequency_operator_136(self, query, index, *args, **kwargs):
                freq = _extract_frequency_operator_136(query)
                plan = orig(self, query, index, *args, **kwargs)
                if isinstance(plan, dict) and freq:
                    plan["frequency_operator_136"] = freq
                    # Historical planner strips tags before token planning. Keep the
                    # original query for diagnostics and the stripped base query for
                    # downstream parity/debugging.
                    try:
                        plan["frequency_base_query_136"] = TAG_RE.sub("", query or "").strip()
                    except Exception:
                        pass
                return plan
            plan_with_frequency_operator_136._frequency_operator_136_wrapped = True
            return plan_with_frequency_operator_136

        try:
            setattr(_obj, "plan", _make_wrapper_136(plan_fn))
        except Exception:
            pass

try:
    _install_frequency_operator_plan_capture_136()
except Exception:
    pass
# END KORPUSUJ_PATCH_136_FREQUENCY_OPERATOR_PLAN_CAPTURE

# COREF_CQL_PLANNER_DOC_FILTER_138L2:
# coref/coref(H/P/M) are SearchCursor doc-array post-filter attributes.
# They are intentionally NOT physical SQLite indexed attrs.
COREF_CQL_DOC_FILTER_ATTRS_138L2 = (
    "coref", "coref(H)", "coref(P)", "coref(M)",
    "coref(h)", "coref(p)", "coref(m)",
)
try:
    DOC_FILTER_ATTRS_036L2B = tuple(dict.fromkeys(tuple(DOC_FILTER_ATTRS_036L2B) + COREF_CQL_DOC_FILTER_ATTRS_138L2))
except Exception:
    DOC_FILTER_ATTRS_036L2B = COREF_CQL_DOC_FILTER_ATTRS_138L2

def is_coref_doc_filter_attr_138l2(attr):
    return str(attr or "").strip().replace(" ", "") in COREF_CQL_DOC_FILTER_ATTRS_138L2
# Adds bounded support for CQL gap/range sequences [A][*][m,n][B] by returning
# an ordinary SearchCursor-compatible token_groups plan plus explicit gap
# metadata.  Top-level OR (||) is deliberately not handled here.
def _install_searchplanner_gap_range_lazy_contract():
    try:
        import re as _re
    except Exception:
        return
    cls = globals().get("SearchPlanner")
    if cls is None or getattr(cls, "_gap_range_lazy_contract_installed", False):
        return
    original_plan = getattr(cls, "plan", None)
    if not callable(original_plan):
        return

    _RANGE_RE = _re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")

    def _available_attrs(self, index):
        available = set(getattr(self, "INDEXABLE_ATTRS", set()))
        try:
            meta_attrs = (index.meta().get("indexed_attrs", "") if index is not None else "")
            if meta_attrs:
                available = {a.strip() for a in meta_attrs.split(",") if a.strip()}
        except Exception:
            pass
        return available

    def _gap_plan(self, query, index):
        available_attrs = _available_attrs(self, index)
        q = _re.sub(r"<frequency_(?:orth|base)\s+[^>]+>", "", query or "", flags=_re.IGNORECASE).strip()
        q, metadata_filters = self._extract_metadata(q)
        contents = self._extract_square_brackets(q)
        if not contents:
            return None
        # Keep top-level OR / other complex CQL out of this patch lane.
        if self._strip_square_brackets(q).strip():
            return None

        token_groups = []
        gaps = []
        i = 0
        while i < len(contents):
            content = str(contents[i]).strip()
            if content == "*":
                if not token_groups:
                    return None
                if i + 1 >= len(contents):
                    return None
                m = _RANGE_RE.fullmatch(str(contents[i + 1]).strip())
                if not m:
                    return None
                min_gap = int(m.group(1))
                max_gap = int(m.group(2))
                if min_gap < 0 or max_gap < min_gap:
                    return None
                gaps.append({"after_group": len(token_groups) - 1, "min": min_gap, "max": max_gap})
                i += 2
                # Require a token group after the gap/range pair.
                if i >= len(contents):
                    return None
                continue
            # A bare numeric range without a preceding [*] remains unsupported.
            if _RANGE_RE.fullmatch(content):
                return None
            token_groups.append(self._parse_token_element(content, available_attrs))
            i += 1

        if not gaps or len(token_groups) < 2:
            return None
        for gap in gaps:
            if int(gap.get("after_group", -1)) >= len(token_groups) - 1:
                return None

        return {
            "supported": True,
            "token_groups": token_groups,
            "gaps": gaps,
            "metadata_filters": metadata_filters,
            "uses_dependency": any(self._element_uses_dependency(g) for g in token_groups),
            "gap_range_lazy_contract": True,
        }

    def plan_gap_range(self, query, index, *args, **kwargs):
        plan = original_plan(self, query, index, *args, **kwargs)
        try:
            # Only recover the exact rejection produced by [*][m,n] bracket
            # contents. Do not override supported plans or unrelated failures.
            if not (isinstance(plan, dict) and plan.get("supported") is False):
                return plan
            if str(plan.get("reason") or "") != "warunek spoza prostego indeksu":
                return plan
            q = str(query or "")
            if "[*]" not in q:
                return plan
            gap_plan = _gap_plan(self, query, index)
            return gap_plan if isinstance(gap_plan, dict) else plan
        except Exception:
            return plan

    plan_gap_range._gap_range_lazy_contract_wrapped = True
    setattr(cls, "plan", plan_gap_range)
    cls._gap_range_lazy_contract_installed = True

try:
    _install_searchplanner_gap_range_lazy_contract()
except Exception:
    pass
