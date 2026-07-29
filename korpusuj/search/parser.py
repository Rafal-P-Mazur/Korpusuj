# -*- coding: utf-8 -*-
"""Parse CQL into token, sentence and frequency conditions used by search planning."""
from __future__ import annotations

import ast
import logging
import re


class QueryParseError(Exception):
    """Fallback parse error; overwritten by engine.QueryParseError at runtime."""
    pass

def split_top_level(s, delimiter):
    # Splits string s on delimiter characters that are not inside nested { }.
    parts = []
    current = []
    level = 0
    i = 0
    while i < len(s):
        if s[i] == '{':
            level += 1
            current.append(s[i])
        elif s[i] == '}':
            level -= 1
            current.append(s[i])
        elif s[i:i + len(delimiter)] == delimiter and level == 0:
            parts.append("".join(current).strip())
            current = []
            i += len(delimiter) - 1
        else:
            current.append(s[i])
        i += 1
    if current:
        parts.append("".join(current).strip())
    return parts

def find_top_level_operator(s, op):
    # Znajduje indeks operatora op (np. "!=" lub "=") na poziomie zewnętrznym.
    level = 0
    i = 0
    while i < len(s):
        if s[i] == '{':
            level += 1
        elif s[i] == '}':
            level -= 1
        elif level == 0 and s[i:i + len(op)] == op:
            return i
        i += 1
    return -1

def parse_single_condition(s):
    # Parses a single condition supporting '=' and '!=' operators as well as
    # new operators for prefix/suffix matching and regular expressions.

    s = s.strip()

    # Check for repetition operator pattern, e.g., "1,3"
    if re.match(r'^\d+\s*,\s*\d+$', s):
        parts = s.split(',')
        min_repeat = int(parts[0].strip())
        max_repeat = int(parts[1].strip())
        return ("repeat", (min_repeat, max_repeat), False)

    # Wildcard: if string starts and ends with an asterisk, return an empty tuple.
    if s.startswith("*") and s.endswith("*"):
        return (), None

    # Look for the outer-level "!=" operator first.
    op_index = find_top_level_operator(s, "!=")
    operator = None
    if op_index != -1:
        operator = "!="
    else:
        op_index = find_top_level_operator(s, "=")
        if op_index != -1:
            operator = "="

    if operator is None:
        raise QueryParseError(f"Niepoprawny warunek: {s}")

    key = s[:op_index].strip()
    rest = s[op_index + len(operator):].strip()

    # Handle nested queries – if the value is enclosed in { } then leave it intact.
    if rest.startswith("{") and rest.endswith("}"):
        value_content = rest  # keep the braces
    # Support quoted string values, including escaped characters like \"
    elif rest[0] in ('"', "'") and rest[-1] == rest[0]:
        try:
            value_content = ast.literal_eval(rest)  # properly parses escape sequences
        except Exception as e:
            raise QueryParseError(f"Niepoprawna wartość tekstowa w warunku: {s!r}: {e}")

    else:
        raise QueryParseError(f"Niepoprawny warunek: {s}")

    regex_meta_pattern = re.compile(r'[\[\]\\\.\^\$\*\+\?\{\}\(\)]')

    # Check for regex notation or prefix/suffix operators, but only for literal (non-nested) values.
    if not (value_content.startswith("{") and value_content.endswith("}")):
        # Regex pattern if wrapped in forward slashes.

        if value_content.startswith("~")  and len(value_content) > 1:
            match_type = "regex_search"
            value_content = value_content[1:]  # strip /.../
        elif regex_meta_pattern.search(value_content):
            match_type = "regex"
        else:
            match_type = "exact"
    else:
        match_type = "exact"


    # Process nested conditions if the value is wrapped in { }.
    if value_content.startswith("{") and value_content.endswith("}"):
        inner = value_content[1:-1].strip()
        nested_conditions = parse_conditions(inner)
        return (key, nested_conditions, operator, True, match_type)
    else:
        # For regex patterns, do not split on '|' because it might be part of the expression.
        if match_type == "regex":
            values = [value_content]
        else:
            # Support OR conditions separated by "|"
            values = [v.strip() for v in value_content.split("|")]
        return (key, values, operator, False, match_type)

def parse_conditions(s):
    # Splits a condition string on top-level '&' (ignoring those inside { })
    # and parses each condition.
    # Returns a list of condition tuples.
    # Modified to allow a bracket to consist solely of a repetition operator.

    s = s.strip()
    if re.match(r'^\d+\s*,\s*\d+$', s):
        parts = s.split(',')
        min_repeat = int(parts[0].strip())
        max_repeat = int(parts[1].strip())
        return [("repeat", (min_repeat, max_repeat), False)]
    parts = split_top_level(s, "&")
    conditions = []
    for part in parts:
        part = part.strip()
        cond = parse_single_condition(part)
        if isinstance(cond, tuple) and cond and cond[0] == "repeat":
            if not conditions:
                logging.warning("Repetition operator with no preceding condition")
                return None
            prev = conditions.pop()
            rep_cond = ("repeat", prev, cond[1][0], cond[1][1])
            conditions.append(rep_cond)
        else:
            conditions.append(cond)
    return conditions

def extract_square_brackets(s: str):
    """
    Extracts top-level [ ... ] groups.
    Wszelki tekst poza nawiasami jest automatycznie rozbijany na słowa
    i traktowany jako dopasowanie ortograficzne (orth="...").
    """
    parts = []
    current = []
    level = 0
    in_quotes = False
    quote_char = None

    def flush_naked():

        naked = "".join(current).strip()
        # 1. Usuwamy nawiasy okrągłe, bo służą do grupowania logicznego, a nie wyszukiwania
        naked = naked.replace("(", "").replace(")", "")

        if naked:
            tokens = re.findall(r'\w+|[^\w\s]', naked)
            for token in tokens:
                if token == "*":
                    parts.append("*")
                else:
                    # 2. Uciekamy (escape) znaki specjalne regex, żeby nie wysadziły silnika
                    meta_chars = r'.^$*+?{}[]\|'
                    token_clean = "".join(["\\" + ch if ch in meta_chars else ch for ch in token])
                    token_clean = token_clean.replace('"', '\\"')
                    parts.append(f'orth="{token_clean}"')
        current.clear()

    for c in s:
        if in_quotes:
            current.append(c)
            if c == quote_char:
                in_quotes = False
        else:
            if c in ('"', "'"):
                in_quotes = True
                quote_char = c
                current.append(c)
            elif c == "[":
                if level == 0:
                    flush_naked()
                else:
                    current.append(c)
                level += 1
            elif c == "]":
                level -= 1
                if level == 0:
                    parts.append("".join(current).strip())
                    current = []
                else:
                    current.append(c)
            else:
                current.append(c)

    if level == 0:
        flush_naked()

    return parts

def parse_query_group(group):
    # Given a query group (a string like '[lemma="Ania"][*][1,3][lemma="Tomek"]'),
    # extract the list of bracket conditions.
    # If a bracket contains only a repetition operator, it is attached
    # to the previous bracket.

    group_conditions = []
    for cond_str in extract_square_brackets(group):
        cond_str = cond_str.strip()
        if cond_str.startswith("*") and cond_str.endswith("*"):
            group_conditions.append(())
        else:
            parsed_conditions = parse_conditions(cond_str)
            if (len(parsed_conditions) == 1 and isinstance(parsed_conditions[0], tuple) and
                    parsed_conditions[0] and parsed_conditions[0][0] == "repeat"):
                if not group_conditions:
                    logging.warning("Repetition operator with no preceding bracket")
                    return None
                prev = group_conditions.pop()
                rep_op = parsed_conditions[0]
                new_cond = ("repeat", prev, rep_op[1][0], rep_op[1][1])
                group_conditions.append(new_cond)
            else:
                group_conditions.append(parsed_conditions)
    return group_conditions

def parse_sentence_conditions(s):
    # Parses the conditions provided to the <s> operator.
    # Jeśli warunki są podane w okrągłych nawiasach, traktujemy je jako nieuporządkowane
    # (kolejność nie ma znaczenia). Jeśli w nawiasach kwadratowych – wymagana jest fraza.
    # Zwraca krotkę (ordered, conditions) gdzie:
    #   - ordered = True  => frazowe, tokeny muszą wystąpić kolejno
    #   - ordered = False => nieuporządkowane, tokeny mogą wystąpić w dowolnej kolejności.

    s = s.strip()
    if s.startswith("("):
        unordered_conditions = []
        for m in re.finditer(r'\(([^\)]+)\)', s):
            content = m.group(1).strip()
            conds = parse_query_group(content)
            if conds is None:
                return None, None
            unordered_conditions.extend(conds)
        return False, unordered_conditions
    else:
        conditions = parse_query_group(s)
        return True, conditions

def parse_frequency_attributes(query, attr="frequency"):
    # Extract frequency options from a tag such as:
    #   <frequency top="100" min="1" max="20">
    # Returns a dict with keys "top", "min", "max" (if found), else None.

    pattern = rf'<{attr}\s+([^>]+)>'
    match = re.search(pattern, query)
    if not match:
        return None
    attributes = match.group(1)
    freq_opts = {}
    top_match = re.search(r'top="(\d+)"', attributes)
    if top_match:
        freq_opts["top"] = int(top_match.group(1))
    min_match = re.search(r'min="(\d+)"', attributes)
    if min_match:
        freq_opts["min"] = int(min_match.group(1))
    max_match = re.search(r'max="(\d+)"', attributes)
    if max_match:
        freq_opts["max"] = int(max_match.group(1))
    return freq_opts

def parse_frequency_attribute(query):
    # For backwards compatibility (token frequency using top=)
    opts = parse_frequency_attributes(query, "frequency_orth")
    if opts and "top" in opts:
        return opts["top"]
    return None

def parse_frequency_base_attribute(query):
    # For lemma frequency using top= in <frequency_base>
    opts = parse_frequency_attributes(query, "frequency_base")
    if opts and "top" in opts:
        return opts["top"]
    return None
