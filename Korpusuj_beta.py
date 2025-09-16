import os
import pandas as pd
import numpy as np
import customtkinter as ctk
from CTkListbox import *
import tkinter as tk
from tkinter import filedialog
import re
import json
import threading
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import shutil
from collections import Counter
import sys
import warnings
import math
from datetime import datetime, timedelta
import ast
from ctypes import windll
import string
from functools import lru_cache
import webview
import creator
import fiszki_tkinter
import table

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
windll.shcore.SetProcessDpiAwareness(2)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
text_widgets = []
dataframes = {}
files = {}
corpus_options = []
lemma_vars_raw = {}
merge_entry_vars_raw = {}
lemma_vars_norm = {}
merge_entry_vars_norm = {}
monthly_lemma_freq = {}
temp_clipboard = ""

global monthly_freq_for_use, true_monthly_totals
monthly_freq_for_use = {}
true_monthly_totals = {}
styl_wykresow = None  # set in UI

wykres_sort = None  # set in UI
has_month_column = False

# Determine the base directory for the fonts
if getattr(sys, 'frozen', False):  # If running as a PyInstaller .exe
    BASE_DIR = sys._MEIPASS
    BASE_DIR_CORP = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)
    BASE_DIR_CORP = os.path.dirname(os.path.abspath(__file__))

# Paths and defaults
CONFIG_PATH = os.path.join(BASE_DIR_CORP, 'config.json')
DEFAULT_SETTINGS = {
    'font_family': 'Verdana',
    'fontsize': 14,
    'styl_wykresow': 'ciemny',
    'motyw': 'ciemny',
    'plotting': 'Tak',
    'kontekst': 250
}

# Load or initialize config at startup
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        config = DEFAULT_SETTINGS.copy()
else:
    config = DEFAULT_SETTINGS.copy()
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# Define font paths
FONT1_PATH = os.path.join(BASE_DIR, "fonts", "JetBrainsMono-Bold.ttf")
FONT2_PATH = os.path.join(BASE_DIR, "fonts", "JetBrainsMono-Regular.ttf")

# Load both fonts
ctk.FontManager.load_font(FONT1_PATH)
ctk.FontManager.load_font(FONT2_PATH)

file_path = 'temp/temp_plot.png'
if os.path.exists(file_path):
    os.remove(file_path)
    # print("File removed successfully.")

file_path2 = 'temp/temp_plot_norm.png'
if os.path.exists(file_path2):
    os.remove(file_path2)
    # print("File removed successfully.")

# ---------------------------
# Global pagination variables
# ---------------------------
current_page = 0
rows_per_page = 100
full_results_sorted = []
global_query = ""
global_selected_corpus = ""
search_status = 0
current_p_lemma = 0
current_p_token = 0

# Mapping for morphological features: for each pos, a dictionary mapping feature names
# to the index (0-indexed in the features list; i.e. after splitting and dropping the pos).
FEAT_MAPPING = {
    "subst": {"number": 0, "case": 1, "gender": 2},
    "depr": {"number": 0, "case": 1, "gender": 2},
    "adj": {"number": 0, "case": 1, "gender": 2, "degree": 3},
    "adja": {},
    "adjp": {},
    "adjc": {},
    "conj": {},
    "ppron12": {"number": 0, "case": 1, "gender": 2, "person": 3, "accentability": 4},
    "ppron3": {"number": 0, "case": 1, "gender": 2, "person": 3, "accentability": 4, "post-prepositionality": 5},
    "siebie": {"case": 0},
    "num": {"number": 0, "case": 1, "gender": 2, "accommodability": 3},
    "fin": {"number": 0, "person": 1, "aspect": 2},
    "bedzie": {"number": 0, "person": 1, "aspect": 2},
    "aglt": {"number": 0, "person": 1, "aspect": 2, "vocalicity": 3},
    "praet": {"number": 0, "gender": 1, "aspect": 2, "agglutination": 3},
    "impt": {"number": 0, "person": 1, "aspect": 2},
    "imps": {"aspect": 0},
    "inf": {"aspect": 0},
    "pcon": {"aspect": 0},
    "pant": {"aspect": 0},
    "ger": {"number": 0, "case": 1, "gender": 2, "aspect": 3, "negation": 4},
    "pact": {"number": 0, "case": 1, "gender": 2, "aspect": 3, "negation": 4},
    "ppas": {"number": 0, "case": 1, "gender": 2, "aspect": 3, "negation": 4},
    "winien": {"number": 0, "gender": 1, "aspect": 2},
    "adv": {"degree": 0},
    "prep": {"case": 0},
    "comp": {},
    "qub": {},
    "interj": {},
    "brev": {"fullstoppedness": 0},
    "burk": {},
    "interp": {},
    "xxx": {},
    "ign": {}
}


def build_dependency_maps(sentence_ids, word_ids, head_ids):
    # Build efficient parent/child lookup tables.
    # Returns:
    #     parent_idx: list[int] -> parent index of each token (-1 if none)
    #     children_lookup: list[list[int]] -> children indices for each token

    num_tokens = len(word_ids)
    parent_idx = [-1] * num_tokens
    children_lookup = [[] for _ in range(num_tokens)]

    # Map (sentence, word_id) -> index
    parent_lookup = {(sentence_ids[i], word_ids[i]): i for i in range(num_tokens)}

    for i in range(num_tokens):
        key = (sentence_ids[i], head_ids[i])
        p = parent_lookup.get(key, -1)
        parent_idx[i] = p
        if p >= 0:
            children_lookup[p].append(i)

    return parent_idx, children_lookup


def prepare_match_functions(tokens, lemmas, postags, upostags, deprels, ners, full_postags, parent_idx):
    # Returns a cached version of match_conditions to avoid recomputation.

    @lru_cache(maxsize=None)
    def match_conditions(token_idx, cond_tuple):

        # Evaluate conditions for a single token.
        # cond_tuple must be hashable (e.g. tuple of tuples instead of list of lists).

        for cond in cond_tuple:
            if not cond:
                continue

            if len(cond) >= 5:
                key, values, operator, is_nested, match_type = cond
            else:
                key, values, operator, is_nested = cond
                match_type = "exact"

            # --- direct token attributes ---
            if key == "orth":
                attr = tokens[token_idx]
            elif key == "base":
                attr = lemmas[token_idx]
            elif key == "pos":
                attr = postags[token_idx]
            elif key == "upos":
                attr = upostags[token_idx] if upostags else ""
            elif key == "deprel":
                attr = deprels[token_idx]
            elif key == "ner":
                attr = ners[token_idx]
            elif key in ("children", "children.group"):
                p_idx = parent_idx[token_idx]
                if p_idx == -1:
                    return False
                if is_nested:
                    if not match_conditions(p_idx, tuple(values)):
                        return False
                else:
                    if lemmas[p_idx] not in values:
                        return False
                continue
            else:
                # morphological feature lookup (using full_postags)
                full_tag = full_postags[token_idx]
                tag_parts = full_tag.split(":")
                pos = tag_parts[0] if tag_parts else ""
                feats = tag_parts[1:] if len(tag_parts) > 1 else []
                feat_index = FEAT_MAPPING.get(pos, {}).get(key, -1)
                token_feat = feats[feat_index] if feat_index >= 0 and feat_index < len(feats) else ""
                attr = token_feat

            # --- comparison ---
            if operator == "=":
                if match_type == "exact" and attr not in values:
                    return False
                if match_type == "regex" and not any(re.fullmatch(v, attr) for v in values):
                    return False
                if match_type == "regex_search" and not any(re.search(v, attr) for v in values):
                    return False
            elif operator == "!=":
                if match_type == "exact" and attr in values:
                    return False
                if match_type == "regex" and any(re.fullmatch(v, attr) for v in values):
                    return False
                if match_type == "regex_search" and any(re.search(v, attr) for v in values):
                    return False

        return True

    return match_conditions

def resource_path(relative_path):
    # Get absolute path to resource, works for dev and PyInstaller
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Funkcja do wczytania pliku JSON na podstawie wybranego korpusu
def load_data():
    selected_corpus = corpus_var.get()
    input_file = files[selected_corpus]
    return pd.read_parquet(input_file)

# ---------------------------
# Pagination Functions
# ---------------------------
def display_page(query, selected_corpus):
    global current_page, rows_per_page, full_results_sorted, data, search_status
    # Clear and insert header into the text widget.

    # Check for search status

    if search_status == 1:
        text_result.set_data([("Proszę czekać!", "Przeszukuję korpus w poszukiwaniu:", "", query)])
        text_result.set_fulltext_data([])

        page_label.configure(text="0/0")
        button_first.configure(state="disabled")
        button_prev.configure(state="disabled")
        button_next.configure(state="disabled")
        button_last.configure(state="disabled")
        return

    # Check for empty results
    if search_status == 0:
        if not full_results_sorted:
            text_result.set_data([("", "Brak wyników dla zapytania:", query, "")])
            text_result.set_fulltext_data([])

            page_label.configure(text="0/0")
            button_first.configure(state="disabled")
            button_prev.configure(state="disabled")
            button_next.configure(state="disabled")
            button_last.configure(state="disabled")
            return

    start_index = current_page * rows_per_page
    end_index = start_index + rows_per_page
    line_number = 3  # Starting line number (header occupies first two lines)
    new_data = []
    full_data = []

    # Iterate over only the slice for the current page.
    for idx, (publication_date, context, full_text, matched_text, matched_lemmas, month_key, title, author,
              additional_metadata,
              left_context, right_context) in enumerate(
        full_results_sorted[start_index:end_index], start=start_index + 1):
        matched_text = matched_text.replace("\n", " ")
        left_context = left_context.replace("\n", " ")
        right_context = right_context.replace("\n", " ")
        if len(title) > 15:
            metadata = f"{idx}. (A: {author}; T: {title[:15]}...; D: {publication_date})"
        else:
            metadata = f"{idx}. (A: {author}; T: {title[:15]}; D: {publication_date})"
        row_data = (metadata, left_context, matched_text, right_context)
        new_data.append(row_data)
        row_full_data = (full_text, context, publication_date, title, author, additional_metadata)
        full_data.append(row_full_data)

    text_result.set_data(new_data)
    text_result.set_fulltext_data(full_data)

    def handle_row_click(row_index):
        if 0 <= row_index - 1 < len(text_result.fulltext_data):
            fdata = text_result.fulltext_data[row_index - 1]
            display_full_text(fdata[0], fdata[1],  fdata[2], fdata[3], fdata[4], fdata[5])

    text_result.set_additional_event(handle_row_click)

    # Update pagination button states.
    total_pages = math.ceil(len(full_results_sorted) / rows_per_page) if full_results_sorted else 1
    page_label.configure(text=f"{current_page + 1}/{total_pages}")
    if current_page == 0:
        button_first.configure(state="disabled")
        button_prev.configure(state="disabled")
    else:
        button_first.configure(state="normal")
        button_prev.configure(state="normal")
    if end_index >= len(full_results_sorted):
        button_next.configure(state="disabled")
        button_last.configure(state="disabled")
    else:
        button_next.configure(state="normal")
        button_last.configure(state="normal")

# wyniki paginacja
def next_page():
    global current_page, rows_per_page, full_results_sorted, global_query, global_selected_corpus
    if (current_page + 1) * rows_per_page < len(full_results_sorted):
        current_page += 1
        display_page(global_query, global_selected_corpus)

def prev_page():
    global current_page, global_query, global_selected_corpus
    if current_page > 0:
        current_page -= 1
        display_page(global_query, global_selected_corpus)


def first_page():
    global current_page, global_query, global_selected_corpus
    if current_page != 0:
        current_page = 0
        display_page(global_query, global_selected_corpus)


def last_page():
    global current_page, rows_per_page, full_results_sorted, global_query, global_selected_corpus
    total_pages = math.ceil(len(full_results_sorted) / rows_per_page) if full_results_sorted else 1
    if current_page != total_pages - 1:
        current_page = total_pages - 1
        display_page(global_query, global_selected_corpus)


def update_table(paginator):
    data = paginator["data"]
    page_ref = paginator["current_page"]
    table = paginator["table"]
    label = paginator["label"]
    items_per_page = paginator["items_per_page"]

    total_items = len(data)
    max_page = (total_items - 1) // items_per_page

    # Clamp the page number within bounds
    page = max(0, min(page_ref[0], max_page))
    page_ref[0] = page

    start = page * items_per_page
    end = min(start + items_per_page, total_items)

    table.set_data(data[start:end])
    label.configure(text=f"{page + 1}/{max_page + 1}")


def go_to_page(paginator, page_num):
    paginator["current_page"][0] = page_num
    update_table(paginator)


def next_p(paginator):
    paginator["current_page"][0] += 1
    update_table(paginator)


def prev_p(paginator):
    paginator["current_page"][0] -= 1
    update_table(paginator)


def first_p(paginator):
    go_to_page(paginator, 0)


def last_p(paginator):
    total_items = len(paginator["data"])
    last = (total_items - 1) // paginator["items_per_page"]
    go_to_page(paginator, last)

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
        return ("repeat", (min_repeat, max_repeat), False), None

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
        print(f"Invalid condition format in part: {s}")
        return None, f"Invalid condition format in part: {s}"

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
            print(f"Invalid quoted value in part: {s!r}  — {e}")
            return None, f"Invalid quoted value in part: {s!r}"
    else:
        print(f"Invalid condition format in part: {s}")
        return None, f"Invalid condition format in part: {s}"
    regex_meta_pattern = re.compile(r'[\[\]\\\.\^\$\*\+\?\{\}\|\(\)]')

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
        if nested_conditions is None:
            return None, f"Error parsing nested conditions in part: {s}"
        return (key, nested_conditions, operator, True, match_type), None
    else:
        # For regex patterns, do not split on '|' because it might be part of the expression.
        if match_type == "regex":
            values = [value_content]
        else:
            # Support OR conditions separated by "|"
            values = [v.strip() for v in value_content.split("|")]
        return (key, values, operator, False, match_type), None


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
        cond, err = parse_single_condition(part)
        if cond is None:
            print(err)
            return None
        if isinstance(cond, tuple) and cond and cond[0] == "repeat":
            if not conditions:
                print("Repetition operator with no preceding condition")
                return None
            prev = conditions.pop()
            rep_cond = ("repeat", prev, cond[1][0], cond[1][1])
            conditions.append(rep_cond)
        else:
            conditions.append(cond)
    return conditions

def extract_square_brackets(s: str):
    """Extracts top-level [ ... ] groups from a query string,
    without breaking on ] that appear inside quoted strings or regex char classes."""
    parts = []
    current = []
    level = 0
    in_quotes = False
    quote_char = None

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
                if level > 0:
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
                if level > 0:
                    current.append(c)

    return parts

def parse_query_group(group):
    # Given a query group (a string like '[lemma="Ania"][*][1,3][lemma="Tomek"]'),
    # extract the list of bracket conditions.
    # If a bracket contains only a repetition operator, it is attached
    # to the previous bracket.

    group_conditions = []
    for cond_str in extract_square_brackets(group):
        cond_str = cond_str.strip()
        print(cond_str)
        if cond_str.startswith("*") and cond_str.endswith("*"):
            group_conditions.append(())
        else:
            parsed_conditions = parse_conditions(cond_str)
            if parsed_conditions is None:
                return None
            if (len(parsed_conditions) == 1 and isinstance(parsed_conditions[0], tuple) and
                    parsed_conditions[0] and parsed_conditions[0][0] == "repeat"):
                if not group_conditions:
                    print("Repetition operator with no preceding bracket")
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

# --- Main Function: find_lemma_context ---
def find_lemma_context(query, df, left_context_size=10, right_context_size=10):
    global search_status

    # Searches for matching tokens in the DataFrame df based on the query.

    selected_corpus = corpus_var.get() if corpus_var else "default_corpus"

    results = []

    text_result.after(0, lambda: display_page(query, selected_corpus))

    # Extract frequency options for tokens and for lemmas
    freq_opts = parse_frequency_attributes(query, "frequency_orth")
    freq_base_opts = parse_frequency_attributes(query, "frequency_base")

    # These variables are used if only a top value is provided (backward compatibility)
    frequency_limit = freq_opts.get("top") if freq_opts else None
    frequency_base_limit = freq_base_opts.get("top") if freq_base_opts else None

    # Extract and remove date filter from query ---
    date_filters = []  # list of (operator, date_str)
    metadata_filters = []

    def extract_filters(q, tag):

        # Generic extractor for <tag op "value"> and [tag op "value"].
        # Returns: new_q, list of (op, value, match_type).

        # regex to catch e.g. <autor = "Smith"> or [data != "/^2021-/"]
        pattern = rf'''
            (?:
        <\s*{tag}\s*(<=|>=|!=|=|<|>)\s*"([^"]+)"\s*>   # angle-bracket
    )

        '''
        filters = []

        # pre‑compile once
        regex_meta = re.compile(r'[\.\^\$\*\+\?\{\}\[\]\|\\\(\)]')

        def classify(val):
            # ~ syntax → regex_search
            if val.startswith("~") and len(val) > 1:
                return "regex_search", val[1:]
            # contains regex meta characters → regex
            if regex_meta.search(val):
                return "regex", val
            # plain exact match
            return "exact", val

        for m in re.finditer(pattern, q, flags=re.VERBOSE | re.IGNORECASE):
            op = m.group(1) or m.group(3)
            raw = m.group(2) or m.group(4)
            mt, norm = classify(raw)
            filters.append((op, norm, mt))

        # strip them out
        new_q = re.sub(pattern, "", q, flags=re.VERBOSE | re.IGNORECASE).strip()
        return new_q, filters

    # usage:
    query, author_filters = extract_filters(query, "autor")
    query, title_filters = extract_filters(query, "tytuł")
    query, date_filters = extract_filters(query, "data")

    def extract_metadane_filters(q):

        # Extract filters of form <metadane:column op "value"> or [metadane:column op "value"].
        # Returns: new_q, list of (column, op, value, match_type).

        # Match both angle and square brackets
        pattern = r'''
            (?:<\s*metadane:(\w+)\s*(=|!=|<=|>=|<|>)\s*"([^"]+)"\s*>)  # angle-bracket

        '''
        filters = []

        regex_meta = re.compile(r'[\.\^\$\*\+\?\{\}\[\]\|\\\(\)]')

        def classify(val):
            # ~ syntax → regex_search
            if val.startswith("~")  and len(val) > 1:
                return "regex_search", val[1:]
            # contains regex meta characters → regex
            if regex_meta.search(val):
                return "regex", val
            # plain exact match
            return "exact", val

        for m in re.finditer(pattern, q, flags=re.VERBOSE | re.IGNORECASE):
            col = m.group(1) or m.group(4)
            op = m.group(2) or m.group(5)
            raw = m.group(3) or m.group(6)
            mt, norm = classify(raw)
            filters.append((col, op, norm, mt))

        # Clean query
        new_q = re.sub(pattern, "", q, flags=re.VERBOSE | re.IGNORECASE).strip()
        return new_q, filters

    query, metadata_filters = extract_metadane_filters(query)

    temp_results = []  # Temporary list to store all detailed match results.

    token_counter = Counter()  # Counter for matched tokens/phrases
    lemma_counter = Counter()  # Counter for matched lemmas

    # --- Parse Query Groups with optional <s> operator ---
    query_groups = [group.strip() for group in query.split("||")]
    parsed_query_groups = []
    for group in query_groups:
        if "<s" in group:
            token_part, sentence_part = group.split("<s", 1)
            sentence_part = sentence_part.strip()
            if sentence_part.endswith(">"):
                sentence_part = sentence_part[:-1].strip()
            token_query_conditions = parse_query_group(token_part)
            s_ordered, sentence_query_conditions = parse_sentence_conditions(sentence_part)
            if token_query_conditions is None or sentence_query_conditions is None:
                return []
            parsed_query_groups.append((token_query_conditions, s_ordered, sentence_query_conditions))
        else:
            token_query_conditions = parse_query_group(group)
            if token_query_conditions is None:
                return []
            parsed_query_groups.append((token_query_conditions, None, None))

    # --- Vectorized pre-filtering using token-level conditions only ---
    # For regex conditions we skip pre-filtering (they are handled row-by-row)
    for token_query_conditions, s_ordered, sentence_query_conditions in parsed_query_groups:
        masks = []
        for cond in token_query_conditions:
            compound_mask = pd.Series(True, index=df.index)
            for subcond in (cond if isinstance(cond, list) else [cond]):
                if not subcond:
                    continue
                if len(subcond) >= 5:
                    key, values, operator, is_nested, match_type = subcond
                else:
                    key, values, operator, is_nested = subcond
                    match_type = "exact"
                # Skip prefiltering for children/parent keys and regex conditions.
                if key in ("children", "parent") or match_type == "regex":
                    continue
                query_to_df = {"orth": "token", "base": "lemma", "pos": "postag", "upos": "upostag", "deprel": "deprel",
                               "ner": "ner"}
                if key in query_to_df:
                    col = query_to_df[key]
                    if match_type == "exact":
                        regex = f"{col}:(?:{'|'.join(re.escape(v) for v in values)})\\b"

            masks.append(compound_mask)
        if masks:
            combined_mask = masks[0] if masks else pd.Series(True, index=df.index)
            for mask in masks[1:]:
                combined_mask &= mask
            filtered_df = df[combined_mask]
        else:
            filtered_df = df

        # ✅ Build a single mask for all filters
        mask = pd.Series(True, index=filtered_df.index)

        # --- Author filters ---
        if author_filters:
            author_series = filtered_df['Autor'].astype(str).str.lower()
            for op, value, match_type in author_filters:
                val = value.lower()
                if match_type == "exact":
                    submask = author_series == val
                else:  # regex
                    submask = author_series.str.contains(value, regex=True, flags=re.IGNORECASE)
                if op == "!=":
                    submask = ~submask
                mask &= submask

        # --- Title filters ---
        if title_filters:
            title_series = filtered_df['Tytuł'].astype(str).str.lower()
            for op, value, match_type in title_filters:
                val = value.lower()
                if match_type == "exact":
                    submask = title_series == val

                else:
                    submask = title_series.str.contains(value, regex=True, flags=re.IGNORECASE)
                if op == "!=":
                    submask = ~submask
                mask &= submask

        # --- Date filters ---
        if date_filters:
            date_series = filtered_df['Data publikacji'].astype(str).str[:10]
            for op, value, match_type in date_filters:
                if op == '<':
                    submask = date_series < value
                elif op == '<=':
                    submask = date_series <= value
                elif op == '>':
                    submask = date_series > value
                elif op == '>=':
                    submask = date_series >= value
                else:  # '=' or '!='
                    if match_type == "exact":
                        submask = date_series == value
                    else:
                        submask = date_series.str.contains(value, regex=True, flags=re.IGNORECASE)
                    if op == "!=":
                        submask = ~submask
                mask &= submask

        # --- Metadata filters ---
        if metadata_filters:
            for column, op, value, match_type in metadata_filters:
                if column not in filtered_df.columns:
                    continue
                series = filtered_df[column].astype(str).str.lower()
                val = value.lower()
                if op in ("<", "<=", ">", ">="):
                    if op == "<":
                        submask = series < val
                    elif op == "<=":
                        submask = series <= val
                    elif op == ">":
                        submask = series > val
                    else:  # ">="
                        submask = series >= val
                else:  # '=' or '!='
                    if match_type == "exact":
                        submask = series == val
                    elif match_type == "regex":
                        # fullmatch for regex
                        submask = series.apply(lambda x: bool(re.fullmatch(value, x, flags=re.IGNORECASE)))
                    else:  # regex_search
                        submask = series.str.contains(value, regex=True, flags=re.IGNORECASE)

                    if op == "!=":
                        submask = ~submask

                mask &= submask

        # ✅ Apply all filters at once
        filtered_df = filtered_df[mask]

        # --- Apply generalized metadane filters ---
        for column, op, value, match_type in metadata_filters:
            if column not in filtered_df.columns:
                continue  # optional: warn

            series = filtered_df[column].astype(str).str.lower()
            val = value.lower()

            if op in ("<", "<=", ">", ">="):
                mask = series < val if op == "<" else \
                    series <= val if op == "<=" else \
                        series > val if op == ">" else \
                            series >= val
            else:
                if match_type == "exact":
                    mask = series == val
                else:  # regex
                    mask = series.str.contains(value, regex=True, flags=re.IGNORECASE)

                if op == "!=":
                    mask = ~mask

            filtered_df = filtered_df[mask]

        for row in filtered_df.itertuples(index=True):
            original_row_index = row.Index

            # ✅ use pre-parsed lists directly
            tokens = row.tokens
            lemmas = row.lemmas
            deprels = row.deprels
            postags = row.postags
            upostags = getattr(row, "upostags", None)
            full_postags = row.full_postags
            word_ids = row.word_ids
            sentence_ids = row.sentence_ids
            head_ids = row.head_ids
            start_ids = row.start_ids
            end_ids = row.end_ids
            ners = row.ners

            num_tokens = len(tokens)
            if num_tokens == 0:
                continue

                # Metadata fields (must exist in df BEFORE filtering)
                publication_date = getattr(row, "publication_date", None)
                title = getattr(row, "title", None)
                author = getattr(row, "author", None)
                additional_metadata = getattr(row, "additional_metadata", None)


            parent_idx, children_lookup = build_dependency_maps(
                sentence_ids, word_ids, head_ids
            )

            match_conditions = prepare_match_functions(
                tokens, lemmas, postags, upostags, deprels, ners, full_postags, parent_idx
            )

            def match_conditions(token_idx, conditions):
                if not conditions:
                    return True
                for cond in conditions:
                    if not cond:
                        continue
                    if len(cond) >= 5:
                        key, values, operator, is_nested, match_type = cond
                    else:
                        key, values, operator, is_nested = cond
                        match_type = "exact"
                    if key in ("orth", "base", "pos", "deprel", "ner", "upos"):
                        if key == "orth":
                            attr = tokens[token_idx]
                        elif key == "base":
                            attr = lemmas[token_idx]
                        elif key == "pos":
                            attr = postags[token_idx]
                        elif key == "upos":
                            attr = upostags[token_idx]
                        elif key == "deprel":
                            attr = deprels[token_idx]
                        elif key == "ner":
                            attr = ners[token_idx]
                        if operator == "=":
                            if match_type == "exact" and attr not in values:

                                return False

                            elif match_type == "regex" and not any(re.fullmatch(v, attr) for v in values):

                                return False
                            elif match_type == "regex_search" and not any(re.search(v, attr) for v in values):
                                return False

                        elif operator == "!=":
                            if match_type == "exact" and attr in values:
                                return False

                            elif match_type == "regex" and any(re.fullmatch(v, attr) for v in values):
                                return False
                            elif match_type == "regex_search" and any(re.search(v, attr) for v in values):
                                return False
                    elif key in ("children", "children.group"):
                        parent = parent_idx[token_idx]
                        if parent < 0:  # <- fix from None
                            return False
                        if is_nested:
                            if not match_conditions(parent, tuple(values)):
                                return False
                        else:
                            parent_attr = lemmas[parent]
                            if operator == "=":
                                if match_type == "exact" and parent_attr not in values:
                                    return False
                                elif match_type == "regex":
                                    if not any(re.fullmatch(v, parent_attr) for v in values):
                                        return False
                                elif match_type == "regex_search":
                                    if not any(re.search(v, parent_attr) for v in values):
                                        return False
                            elif operator == "!=":
                                if match_type == "exact" and parent_attr in values:
                                    return False
                                elif match_type == "regex":
                                    if any(re.fullmatch(v, parent_attr) for v in values):
                                        return False
                                elif match_type == "regex_search":
                                    if any(re.search(v, parent_attr) for v in values):
                                        return False


                    elif key == "parent":
                        if operator == "=":
                            found = False
                            for child in children_lookup[token_idx]:
                                if is_nested:
                                    if match_conditions(child, tuple(values)):
                                        found = True
                                        break
                                else:
                                    child_attr = lemmas[child]
                                    if match_type == "exact" and child_attr in values:
                                        found = True
                                        break
                                    elif match_type == "regex" and any(re.fullmatch(v, child_attr) for v in values):
                                        found = True
                                        break
                                    elif match_type == "regex_search" and any(re.search(v, child_attr) for v in values):
                                        found = True
                                        break

                            if not found:
                                return False

                        elif operator == "!=":
                            for child in children_lookup[token_idx]:
                                if is_nested:
                                    if match_conditions(child, tuple(values)):
                                        return False
                                else:
                                    child_attr = lemmas[child]
                                    if match_type == "exact" and child_attr in values:
                                        return False
                                    elif match_type == "regex" and any(re.fullmatch(v, child_attr) for v in values):
                                        return False
                                    elif match_type == "rege_search" and any(re.search(v, child_attr) for v in values):
                                        return False


                    else:
                        full_tag = full_postags[token_idx]
                        tag_parts = full_tag.split(":")
                        pos = tag_parts[0] if tag_parts else ""
                        feats = tag_parts[1:] if len(tag_parts) > 1 else []
                        mapping = FEAT_MAPPING.get(pos, {})
                        if key not in mapping:
                            return False
                        feat_index = mapping[key]
                        token_feat = feats[feat_index] if feat_index < len(feats) else ""
                        if operator == "=":
                            if match_type == "exact" and token_feat not in values:
                                return False
                            elif match_type == "regex" and not any(re.fullmatch(v, attr) for v in values):
                                pass
                            elif match_type == "regex_search" and not any(re.search(v, attr) for v in values):
                                pass


                        elif operator == "!=":
                            if match_type == "exact" and token_feat in values:
                                return False
                            elif match_type == "regex" and any(re.fullmatch(v, token_feat) for v in values):
                                return False
                            elif match_type == "regex_search" and any(re.search(v, token_feat) for v in values):
                                return False

                return True

            def match_pattern(start_idx, cond_list):
                if not cond_list:
                    return start_idx
                first = cond_list[0]
                if isinstance(first, tuple) and first and first[0] == "repeat":
                    base_cond = first[1]
                    min_rep = first[2]
                    max_rep = first[3]
                    for count in range(min_rep, max_rep + 1):
                        new_idx = start_idx
                        valid = True
                        for _ in range(count):
                            if new_idx >= num_tokens or not match_conditions(new_idx, base_cond if isinstance(base_cond,
                                                                                                              list) else [
                                base_cond]):
                                valid = False
                                break
                            new_idx += 1
                        if valid:
                            remainder = match_pattern(new_idx, cond_list[1:])
                            if remainder is not None:
                                return remainder
                    return None
                else:
                    if start_idx >= num_tokens or not match_conditions(start_idx, first):
                        return None
                    return match_pattern(start_idx + 1, cond_list[1:])

            def match_pattern_in_range(start_idx, cond_list, end_limit):
                if not cond_list:
                    return start_idx
                first = cond_list[0]
                if isinstance(first, tuple) and first and first[0] == "repeat":
                    base_cond = first[1]
                    min_rep = first[2]
                    max_rep = first[3]
                    for count in range(min_rep, max_rep + 1):
                        new_idx = start_idx
                        valid = True
                        for _ in range(count):
                            if new_idx >= end_limit or not match_conditions(new_idx, base_cond if isinstance(base_cond,
                                                                                                             list) else [
                                base_cond]):
                                valid = False
                                break
                            new_idx += 1
                        if valid:
                            remainder = match_pattern_in_range(new_idx, cond_list[1:], end_limit)
                            if remainder is not None:
                                return remainder
                    return None
                else:
                    if start_idx >= end_limit or not match_conditions(start_idx, first):
                        return None
                    return match_pattern_in_range(start_idx + 1, cond_list[1:], end_limit)

            def match_pattern_in_sentence(start_idx, cond_list, sentence_ids):
                sent_id = sentence_ids[start_idx]
                # find sentence boundaries
                sent_start = start_idx
                while sent_start > 0 and sentence_ids[sent_start - 1] == sent_id:
                    sent_start -= 1
                sent_end = start_idx
                while sent_end < len(sentence_ids) and sentence_ids[sent_end] == sent_id:
                    sent_end += 1
                return match_pattern_in_range(start_idx, cond_list, sent_end)

            def sentence_contains_conditions(sent_start, sent_end, conditions):
                for idx in range(sent_start, sent_end):
                    if match_pattern_in_range(idx, conditions, sent_end) is not None:
                        return True
                return False

            def sentence_matches(sent_start, sent_end, conditions_groups):
                for group in conditions_groups:
                    group_satisfied = False
                    for token_idx in range(sent_start, sent_end):
                        if match_conditions(token_idx, group):
                            group_satisfied = True
                            break
                    if not group_satisfied:
                        return False
                return True

            def build_children_lookup(num_tokens, sentence_ids, word_ids, head_ids):

                # Build a mapping from each token index to a list of its children indices.

                children_lookup = {i: [] for i in range(num_tokens)}
                parent_lookup = {(sentence_ids[i], word_ids[i]): i for i in range(num_tokens)}
                for i in range(num_tokens):
                    parent_idx = parent_lookup.get((sentence_ids[i], head_ids[i]))
                    if parent_idx is not None:
                        children_lookup[parent_idx].append(i)
                return children_lookup

            def get_dependency_paths(start_idx, children_lookup):

                # Recursively collect all dependency paths starting from start_idx.
                # Each path is a list of token indices.

                if not children_lookup.get(start_idx):
                    return [[start_idx]]
                paths = []
                for child in children_lookup[start_idx]:
                    for sub_path in get_dependency_paths(child, children_lookup):
                        paths.append([start_idx] + sub_path)
                return paths

            # --- Processing children.group conditions ---
            condition_groups = []
            if isinstance(token_query_conditions, list) and token_query_conditions and isinstance(
                    token_query_conditions[0], (list, tuple)):
                condition_groups = token_query_conditions
            else:
                condition_groups = [token_query_conditions]
            for group in condition_groups:
                group = group if isinstance(group, list) else [group]
                children_cond = None
                extra_conditions = []
                for cond in group:
                    if cond and cond[0] == "children.group":
                        children_cond = cond
                    else:
                        extra_conditions.append(cond)
                if children_cond:
                    if len(children_cond) >= 5:
                        key, target_values, operator, is_nested, match_type = children_cond
                    else:
                        key, target_values, operator, is_nested = children_cond
                        match_type = "exact"
                    children_lookup = build_children_lookup(num_tokens, sentence_ids, word_ids, head_ids)
                    if all(isinstance(tv, str) for tv in target_values):
                        for target in target_values:
                            if isinstance(target, tuple):
                                target = target[0]
                            target_indices = [idx for idx, lemma in enumerate(lemmas)
                                              if isinstance(lemma, str) and lemma.lower() == target.lower()]
                            if not target_indices:
                                continue
                            for t_idx in target_indices:
                                paths = get_dependency_paths(t_idx, children_lookup)
                                for path in paths:
                                    if len(path) <= 1:
                                        continue
                                    direct_parent_idx = path[1]
                                    if extra_conditions and not match_conditions(direct_parent_idx, extra_conditions):
                                        continue
                                    formatted_path = " -> ".join(tokens[idx] for idx in path)
                                    if frekw_dane:
                                        frekw_dane.insert(ctk.END, f"{target}: {formatted_path}\n")
                    else:
                        target_indices = [idx for idx in range(num_tokens) if match_conditions(idx, target_values)]
                        for t_idx in target_indices:
                            paths = get_dependency_paths(t_idx, children_lookup)
                            for path in paths:
                                if len(path) <= 1:
                                    continue
                                direct_parent_idx = path[1]
                                if extra_conditions and not match_conditions(direct_parent_idx, extra_conditions):
                                    continue
                                formatted_path = " -> ".join(tokens[idx] for idx in path)
                                if frekw_dane:
                                    frekw_dane.insert(ctk.END, f"{tokens[t_idx]}: {formatted_path}\n")

            # --- End children.group processing ---

            for i in range(num_tokens):
                if s_ordered or sentence_query_conditions:
                    sent_start = i
                    while sent_start > 0 and sentence_ids[sent_start - 1] == sentence_ids[i]:
                        sent_start -= 1
                    sent_end = i
                    while sent_end < len(sentence_ids) and sentence_ids[sent_end] == sentence_ids[i]:
                        sent_end += 1

                    # check that the sentence also contains the `<s>` conditions
                    if sentence_query_conditions and not sentence_contains_conditions(sent_start, sent_end,
                                                                                      sentence_query_conditions):
                        continue  # skip this match, sentence doesn't have `zabić`

                    end_idx = match_pattern_in_range(i, token_query_conditions, sent_end)
                else:
                    # no sentence restriction
                    end_idx = match_pattern(i, token_query_conditions)

                if end_idx is not None:
                    # [ ... build contexts same as before, using start_ids / end_ids ... ]
                    left_context = row.Treść[
                                   max(0, start_ids[max(0, i - left_context_size)]): start_ids[i]
                                   ] if i > 0 else ""

                    matched_text = row.Treść[start_ids[i]: end_ids[end_idx - 1] + 1]

                    right_context = row.Treść[
                                    end_ids[end_idx - 1] + 1: start_ids[
                                        min(len(start_ids) - 1, i + right_context_size + 1)]
                                    ]

                    matched_lemmas = " ".join(lemmas[i:end_idx])
                    # context = f"{left_context} |* {matched_text} *|{right_context}"
                    context = [left_context, matched_text, right_context]

                    full_left_context = row.Treść[
                                        max(0, start_ids[max(0, i - kontekst)]): start_ids[i]
                                        ] if i > 0 else ""
                    full_left_context = full_left_context[:-len(left_context)] if left_context else full_left_context

                    full_right_context = row.Treść[
                                         end_ids[end_idx - 1] + 1: start_ids[min(len(start_ids) - 1, i + kontekst)]
                                         ]
                    full_right_context = full_right_context[
                                         len(right_context):] if right_context else full_right_context

                    # full_text_with_markers = f"{full_left_context} |*{matched_text} *|{full_right_context}"
                    full_text_with_markers = [full_left_context, matched_text, full_right_context]

                    token_counter[matched_text] += 1
                    lemma_counter[matched_lemmas] += 1

                    if "Data publikacji" in df.columns:
                        raw_date = df.loc[row.Index, "Data publikacji"]
                    else:
                        raw_date = ""
                    publication_date = raw_date.split(" ")[0] if isinstance(raw_date, str) else "Brak danych"
                    if selected_corpus == "wpolityceU.parquet" or selected_corpus == "onetU.parquet":

                        year = str(getattr(row, "rok"))
                        month = str(getattr(row, "miesiąc")).zfill(2)
                        month_key = f"{year}-{month}"
                    else:
                        try:
                            if publication_date:
                                parts = publication_date.split("-")
                                if len(parts) == 2:
                                    year, month = parts
                                elif len(parts) == 1:
                                    year, month = parts[0], "1"  # tylko rok → wpiszmy styczeń
                                else:
                                    year, month, _ = parts
                                month_key = f"{year}-{month}"
                            else:
                                month_key = "Unknown"
                        except Exception:
                            month_key = "Unknown"

                    title = df.loc[row.Index, "Tytuł"] if "Tytuł" in df.columns else " "
                    author = df.loc[row.Index, "Autor"] if "Autor" in df.columns else " "

                    exclude_cols = {
                        "Data publikacji", "Tytuł", "Autor", "tags", "Treść", "token_counts",
                        "tokens", "lemmas", "deprels", "postags", "full_postags",
                        "word_ids", "sentence_ids", "head_ids", "start_ids", "end_ids", "ners", "upostags"
                    }

                    additional_metadata = {
                        col: df.loc[row.Index, col]
                        for col in df.columns
                        if col not in exclude_cols
                    }


                    temp_results.append(((matched_text, matched_lemmas),
                                         (publication_date, context, full_text_with_markers,
                                          matched_text, matched_lemmas,
                                          month_key, title, author, additional_metadata, left_context, right_context)))

        final_results = []
        # Filtering based on frequency_base options (lemma frequency)
        if freq_base_opts:
            # If "top" is provided, get the top lemmas; otherwise consider all lemmas.
            if "top" in freq_base_opts:
                top_lemmas = {lemma for lemma, _ in lemma_counter.most_common(freq_base_opts["top"])}
            else:
                top_lemmas = set(lemma_counter.keys())
            for (token_key, lemma_key), detailed_result in temp_results:
                count = lemma_counter[lemma_key]
                if (lemma_key in top_lemmas and
                        ("min" not in freq_base_opts or count >= freq_base_opts["min"]) and
                        ("max" not in freq_base_opts or count <= freq_base_opts["max"])):
                    final_results.append(detailed_result)
        # Else if only frequency (token) filtering is used.
        elif freq_opts:
            if "top" in freq_opts:
                top_tokens = {token for token, _ in token_counter.most_common(freq_opts["top"])}
            else:
                top_tokens = set(token_counter.keys())
            for (token_key, lemma_key), detailed_result in temp_results:
                count = token_counter[token_key]
                if (token_key in top_tokens and
                        ("min" not in freq_opts or count >= freq_opts["min"]) and
                        ("max" not in freq_opts or count <= freq_opts["max"])):
                    final_results.append(detailed_result)
        else:
            # No frequency filtering, return all results.
            final_results = [detailed for _, detailed in temp_results]

    return final_results

selected_tag = None
original_colors = {}


# Funkcja obsługująca wyszukiwanie
def search():
    theme = THEMES[motyw.get()]
    global search_status
    search_status = 1

    # Clear the checkboxes and result text widget.
    for frame in [checkboxes_frame]:
        for child in frame.winfo_children():
            child.destroy()
    checkboxes_frame.update_idletasks()

    button_search.configure(state="disabled")

    def check_brackets(query):
        stack = []
        in_single_quote = False
        in_double_quote = False

        for char in query:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == '[' and not in_single_quote and not in_double_quote:
                stack.append('[')
            elif char == ']' and not in_single_quote and not in_double_quote:
                if not stack:
                    return False  # Zamknięto nawias, którego nie było
                stack.pop()

        return len(stack) == 0

    def search_thread():  # Run the search in a separate thread.

        try:
            print(f"Search started in thread: {threading.current_thread().name}")
            query = entry_query.get("1.0", ctk.END).strip()  # Remove trailing newline
            if check_brackets(query):
                left_context_size = int(entry_left_context.get())
                right_context_size = int(entry_right_context.get())

                selected_corpus = corpus_var.get()

                df = dataframes[selected_corpus]
                try:
                    results = find_lemma_context(query, df, left_context_size, right_context_size)
                    if not results:
                        print("No results found.")
                    else:
                        print("Number of results:", len(results))
                except:
                    text_result.set_data([("","Bład zapytania", "", "")])
                    return
            else:
                text_result.set_data([("","Bład zapytania:", "", "")])
                return

            def update_text_result():
                global search_status, monthly_lemma_freq, lemma_vars_raw, lemma_vars_norm, merge_entry_vars_raw, merge_entry_vars_norm, has_month_column, true_monthly_totals, monthly_freq_for_use, fq_data
                fq_data = []
                fq_data_token = []

                def first_real_token(text):
                    if not text:
                        return ""
                    for tok in text.split():
                        cleaned = tok.strip(string.punctuation).lower()
                        if cleaned:  # skip if it was only punctuation
                            return cleaned
                    return ""  # fallback if nothing left

                def last_real_token(text):
                    if not text:
                        return ""
                    for tok in reversed(text.split()):
                        cleaned = tok.strip(string.punctuation).lower()
                        if cleaned:
                            return cleaned
                    return ""

                sort_option = sort_option_var.get()
                if sort_option == "Data publikacji":
                    results_sorted = sorted(results, key=lambda x: x[0])
                elif sort_option == "Tytuł":
                    results_sorted = sorted(results, key=lambda x: x[6])
                elif sort_option == "Autor":
                    results_sorted = sorted(results, key=lambda x: x[7])
                elif sort_option == "Alfabetycznie":
                    results_sorted = sorted(results, key=lambda x: x[3])
                elif sort_option == "Prawy kontekst":
                    results_sorted = sorted(results, key=lambda x: first_real_token(x[10]))
                elif sort_option == "Lewy kontekst":
                    results_sorted = sorted(results, key=lambda x: last_real_token(x[9]))


                else:
                    results_sorted = results  # Fallback
                if results_sorted:

                    # Store globals for pagination.
                    global global_query, global_selected_corpus, full_results_sorted, current_page, search_status
                    global_query = query
                    global_selected_corpus = selected_corpus
                    full_results_sorted = results_sorted
                    current_page = 0

                    # Display only the first 100 rows (first page).
                    search_status = 0

                    text_result.after(300, lambda: display_page(query, selected_corpus))

                    if "Data publikacji" in df.columns:
                        # ------------- Frequency and Plot Calculations (unchanged) -------------
                        # First, check if the DataFrame has a "miesiąc" column.
                        if "miesiąc" in df.columns and (
                                global_selected_corpus == "wpolityceU" or global_selected_corpus == "onetU"):

                            def get_month_label(month_key):
                                try:
                                    # Here month_key is expected to be in the format "year_group-period_index"
                                    year_group, period_index = month_key.split('-')
                                    year_group = int(year_group)
                                    period_index = int(period_index)
                                except Exception:
                                    # If the key isn't in the expected format, return it unchanged.
                                    return month_key

                                base_year = 2021 + year_group

                                # Calculate start month.
                                if period_index + 1 <= 12:
                                    start_month = period_index + 1
                                    start_year = base_year
                                else:
                                    start_month = (period_index + 1) - 12
                                    start_year = base_year + 1

                                # Calculate end month.
                                if start_month + 1 <= 12:
                                    end_month = start_month + 1
                                    end_year = start_year
                                else:
                                    end_month = (start_month + 1) - 12
                                    end_year = start_year + 1

                                return f"24.{start_month:02d}.{start_year} - 23.{end_month:02d}.{end_year}"
                        else:
                            def get_month_label(month_key):
                                try:
                                    # Assume month_key is already in the format "year-month" where year is full and month is numeric.
                                    year, month = month_key.split('-')
                                    month = int(month)
                                    return f"{month:02d}.{year}"
                                except Exception:
                                    return month_key

                        unique_matched_tokens = {}
                        unique_lemmas = set()  # Set to store individual lemmas

                        monthly_lemma_freq.clear()
                        lemma_vars_raw.clear()
                        merge_entry_vars_raw.clear()

                        # Process each result and normalize the month key.
                        for publication_date, context, full_text, matched_text, matched_lemmas, month_key, title, author, additional_metadata, left_context, right_context in results:
                            # token_key = matched_text.lower()
                            token_key = matched_text
                            unique_matched_tokens[token_key] = unique_matched_tokens.get(token_key, 0) + 1

                            # Add each individual lemma to the unique set.
                            # for lemma in matched_lemmas.split():
                            unique_lemmas.add(matched_lemmas)

                            # Normalize the month_key.
                            try:
                                year, month_val = month_key.split('-')
                                normalized_key = f"{year}-{int(month_val)}"
                            except Exception:
                                normalized_key = month_key

                            # Initialize monthly frequency for the key if not already present
                            if normalized_key not in monthly_lemma_freq:
                                monthly_lemma_freq[normalized_key] = {}

                            monthly_lemma_freq[normalized_key][matched_lemmas] = (
                                    monthly_lemma_freq[normalized_key].get(matched_lemmas, 0) + 1
                            )
                        print(global_selected_corpus)
                        if "miesiąc" in df.columns and (
                                global_selected_corpus == "wpolityceU.parquet" or global_selected_corpus == "onetU.parquet"):

                            # ---- Backfill: Ensure every expected month has all lemmas ----
                            for year in ['1', '2']:
                                for month in range(1, 13):
                                    key = f"{year}-{month}"
                                    if key not in monthly_lemma_freq:
                                        monthly_lemma_freq[key] = {}
                                    for lemma in unique_lemmas:
                                        if lemma not in monthly_lemma_freq[key]:
                                            monthly_lemma_freq[key][lemma] = 0
                        else:
                            # Convert your keys to datetime objects.
                            date_keys = []
                            for key in monthly_lemma_freq.keys():
                                try:
                                    year, month = map(int, key.split('-'))
                                    date_keys.append(datetime(year, month, 1))
                                except Exception:
                                    continue

                            if date_keys:
                                start_date = min(date_keys)
                                end_date = max(date_keys)

                                # Iterate month by month from start_date to end_date.
                                current_date = start_date
                                while current_date <= end_date:
                                    key = f"{current_date.year}-{current_date.month}"
                                    if key not in monthly_lemma_freq:
                                        # Create an entry for this month with 0 counts for each lemma.
                                        monthly_lemma_freq[key] = {lemma: 0 for lemma in unique_lemmas}
                                    # Move to the next month.
                                    # This trick ensures correct rollover at December.
                                    current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)

                        def extract_year_month_counts(token_dict):
                            data = []
                            for year, months in token_dict.items():
                                for month, count in months.items():
                                    data.append((year, month, count))
                            return data

                        def update_data_tables():
                            global monthly_freq_for_use, true_monthly_totals
                            true_monthly_totals.clear()

                            if "token_counts" not in df.columns:
                                return []

                            # Extract and flatten year-month-count triples
                            filtered = df[df["token_counts"].notna()]
                            extracted = filtered["token_counts"].apply(
                                lambda s: extract_year_month_counts(json.loads(s)))
                            flattened = [item for sublist in extracted for item in sublist]

                            # Fill true_monthly_totals dictionary
                            for year, month, count in flattened:
                                key = f"{year}-{int(month)}"
                                true_monthly_totals[key] = true_monthly_totals.get(key, 0) + count

                            # Optional: return list of (year, month, count)
                            return flattened

                        update_data_tables()

                        total_token_count = sum(true_monthly_totals.values())

                        # 2) Build orthographic frequencies table
                        frekw_dane_text = "\n\n\nFrekwencja (orth):\n"

                        for idx, (token, frequency) in enumerate(
                                sorted(unique_matched_tokens.items(), key=lambda x: x[1], reverse=True), start=1):
                            frequency_normalized = (
                                (frequency / total_token_count) * 1_000_000 if total_token_count > 0 else 0.0
                            )
                            fq_data_token.append([idx, token, frequency, round(frequency_normalized, 2)])
                            frekw_dane_text += f"{token}: {frequency}\n"

                        paginator_token["data"] = fq_data_token
                        update_table(paginator_token)

                        frekw_dane_tabela_orth.set_data(fq_data_token[:15])

                        # 3) Total lemma frequencies across all months
                        lemma_total_freq = {}
                        for month_data in monthly_lemma_freq.values():
                            for lemma, count in month_data.items():
                                lemma_total_freq[lemma] = lemma_total_freq.get(lemma, 0) + count

                        s_lemma_total_freq = sorted(lemma_total_freq.items(), key=lambda x: x[1], reverse=True)
                        fq_data = []
                        for idx, (lemma, frequency) in enumerate(s_lemma_total_freq, start=1):
                            frequency_normalized = (
                                (frequency / total_token_count) * 1_000_000 if total_token_count > 0 else 0.0
                            )
                            fq_data.append([idx, lemma, frequency, round(frequency_normalized, 2)])
                            frekw_dane_text += f"{lemma}: {frequency}\n"

                        paginator_fq["data"] = fq_data

                        frekw_dane_tabela.set_data(fq_data[:15])
                        update_table(paginator_fq)

                        print("obliczyłem frequency do tabel")

                        # 4) Normalize monthly frequencies

                        monthly_freq_for_use = {}
                        for month_key, lemma_counts in monthly_lemma_freq.items():
                            total = true_monthly_totals.get(month_key, 0)
                            if total > 0:
                                monthly_freq_for_use[month_key] = {
                                    lemma: (count / total) * 1_000_000
                                    for lemma, count in lemma_counts.items()
                                }
                            else:
                                monthly_freq_for_use[month_key] = {lemma: 0.0 for lemma in lemma_counts}

                        # 5) Month-level frequency table
                        fq_data_month = []
                        sorted_month_keys = sorted(
                            monthly_lemma_freq.keys(), key=lambda k: (int(k.split('-')[0]), int(k.split('-')[1]))
                        )
                        for month_key in sorted_month_keys:
                            year_str, month_str = month_key.split('-')
                            raw_counts = monthly_lemma_freq[month_key]
                            norm_counts = monthly_freq_for_use[month_key]
                            for lemma in sorted(raw_counts.keys()):
                                raw = raw_counts[lemma]
                                norm = norm_counts.get(lemma, 0.0)
                                fq_data_month.append([int(year_str), int(month_str), lemma, raw, round(norm, 2)])

                        paginator_month["data"] = fq_data_month
                        update_table(paginator_month)
                        frekw_dane_tabela_month.set_data(fq_data_month[:15])
                        update_table(paginator_month)

                        is_month_view = False

                        if plotting.get() == 'Tak':

                            # Plotting
                            if is_month_view:
                                keys = sorted(monthly_freq_for_use.keys(),
                                              key=lambda k: (int(k.split('-')[0]), int(k.split('-')[1])))
                                x_labels = [get_month_label(k) for k in keys]
                                plot_data = monthly_freq_for_use
                            else:
                                yearly_grouped = {}
                                for key, data in monthly_freq_for_use.items():
                                    year, month = key.split('-')
                                    if year == '0000' or month == '0':
                                        continue
                                    yearly_grouped.setdefault(year, {})
                                    for lemma, val in data.items():
                                        yearly_grouped[year][lemma] = yearly_grouped[year].get(lemma, 0) + val
                                keys = sorted(yearly_grouped.keys(), key=int)
                                x_labels = keys
                                plot_data = yearly_grouped

                            x = np.arange(len(keys))

                            if styl_wykresow.get() == "ciemny":
                                plt.style.use('dark_background')
                                fig, ax = plt.subplots(figsize=(12, 7), facecolor='#2C2F33')
                                ax.set_facecolor('#2C2F33')
                            else:
                                plt.style.use('default')
                                fig, ax = plt.subplots(figsize=(12, 7))

                            colors = plt.cm.tab20.colors

                            # Enable minor ticks (so unlabeled ticks still exist for gridlines)
                            ax.grid(True, which='major', axis='both', linestyle='--', linewidth = 0.5, alpha=0.2)
                            ax.set_xlabel('Miesiąc' if is_month_view else 'Rok')
                            ylabel = 'Frekwencja'
                            ax.set_ylabel(ylabel)
                            max_labels = 24  # tweak this depending on your preference
                            n_labels = len(x_labels)

                            if n_labels > max_labels:
                                step = int(np.ceil(n_labels / max_labels))
                            else:
                                step = 1

                            # --- Decide which ticks get labels ---
                            labeled_idx = set([0, n_labels - 1] + list(range(0, n_labels, step)))

                            labels = []
                            for i, lbl in enumerate(x_labels):
                                if i in labeled_idx:
                                    labels.append(lbl)  # show label
                                else:
                                    labels.append("")  # hide label

                            # --- Set ticks ---
                            ax.set_xticks(x)  # all ticks visible
                            ax.set_xticklabels(labels, rotation=45 if is_month_view else 0, ha='right')

                            ax.set_xlim(x[0] - 1, x[-1] + 1)

                            # --- Different tick lengths ---
                            for tick, label in zip(ax.xaxis.get_major_ticks(), labels):
                                if label == "":  # unlabeled → short tick
                                    tick.tick1line.set_markersize(3)
                                    tick.tick2line.set_markersize(3)
                                else:  # labeled → longer tick
                                    tick.tick1line.set_markersize(7)
                                    tick.tick2line.set_markersize(7)

                            renderer = fig.canvas.get_renderer()
                            bboxes = [lbl.get_window_extent(renderer) for lbl in ax.get_xticklabels()]
                            if bboxes:
                                maxw = max(b.width for b in bboxes)
                                bottom = min(0.05 + (maxw / fig.get_dpi()), 0.85)
                            else:
                                bottom = 0.10
                            fig.subplots_adjust(bottom=bottom)

                            ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
                            plt.tight_layout(rect=[0, 0, 1, 0.85])
                            os.makedirs('temp', exist_ok=True)
                            plt.savefig('temp/temp_plot.png', bbox_inches='tight')
                            plt.close(fig)

                            text_result.after(0, update_plot_images)

                            # Post-processing UI
                            if global_selected_corpus in ("wpolityceU.parquet", "onetU.parquet"):
                                has_month_column = 'miesiąc' in df.columns
                            else:
                                has_month_column = False

                            for child in checkboxes_frame.winfo_children():

                                child.destroy()

                            lemma_vars_raw.clear()
                            lemma_vars_norm.clear()
                            merge_entry_vars_raw.clear()
                            merge_entry_vars_norm.clear()


                            # Clear before use:
                            lemma_vars = {}
                            merge_vars = {}

                            def build_listbox_ui(parent_frame, sorted_lemma_freq, vars_dict, merge_dict,
                                                 update_plot_callback, items_per_page=100):
                                """
                                Build a paginated listbox UI with global selection and rename support.
                                Returns: container_frame, listbox_widget, rename_entry
                                """

                                vars_dict.clear()
                                merge_dict.clear()
                                update_job = {"after_id": None}
                                current_page = {"idx": 0}

                                total_pages = max(1, math.ceil(len(sorted_lemma_freq) / items_per_page))

                                # Initialize selection & merge vars
                                for lemma, _ in sorted_lemma_freq:
                                    vars_dict[lemma] = ctk.BooleanVar(value=False)
                                    merge_dict[lemma] = ctk.StringVar(value=lemma)

                                # Main container
                                container = ctk.CTkFrame(parent_frame, fg_color=theme["frame_fg"], corner_radius=15)
                                container.pack(fill="x", expand=False, padx=10, pady=5)

                                # ---- Rename entry + button ----
                                rename_entry = ctk.CTkEntry(
                                    container,
                                    placeholder_text="Nowa nazwa etykiety",
                                    font=("JetBrains Mono", 12),
                                    fg_color=theme["subframe_fg"],
                                    corner_radius=8,
                                    height=35,
                                )
                                rename_entry.pack(fill="x", padx=10, pady=(5, 5))

                                rename_btn = ctk.CTkButton(
                                    container,
                                    text="Edytuj zaznaczone etykiety",
                                    font=("Verdana", 12, 'bold'),
                                    fg_color=theme["button_fg"],
                                    hover_color=theme["button_hover"],
                                    text_color=theme["button_text"],
                                    corner_radius=8,
                                    height=35,
                                )
                                rename_btn.pack(fill="x", padx=10, pady=(5, 10))

                                # ---- Pagination Frame ----
                                nav_frame = ctk.CTkFrame(container, fg_color=theme["subframe_fg"], corner_radius=12)
                                nav_frame.pack(fill="x", padx=10, pady=(0, 10))
                                nav_frame.grid_columnconfigure(0, weight=0)
                                nav_frame.grid_columnconfigure(1, weight=1)
                                nav_frame.grid_columnconfigure(2, weight=0)

                                prev_btn = ctk.CTkButton(nav_frame, text="<", width=40, height=35,
                                                         fg_color=theme["button_fg"],
                                                         hover_color=theme["button_hover"],
                                                         text_color=theme["button_text"],
                                                         corner_radius=8)
                                prev_btn.grid(row=0, column=0, sticky="w", padx=5, pady=5)

                                page_label = ctk.CTkLabel(nav_frame, text=f"1 / {total_pages}",
                                                          font=("Verdana", 12, 'bold'),
                                                          text_color=theme["label_text"], anchor="center")
                                page_label.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

                                next_btn = ctk.CTkButton(nav_frame, text=">", width=40, height=35,
                                                         fg_color=theme["button_fg"],
                                                         hover_color=theme["button_hover"],
                                                         text_color=theme["button_text"],
                                                         corner_radius=8)
                                next_btn.grid(row=0, column=2, sticky="e", padx=5, pady=5)

                                # ---- Listbox Frame (scrollable / fixed height) ----
                                listbox_frame = ctk.CTkFrame(container, fg_color=theme["subframe_fg"], corner_radius=8)
                                listbox_frame.pack(fill="both", expand=True)

                                listbox = CTkListbox(
                                    listbox_frame,
                                    multiple_selection=True,
                                    height=300,  # fixed height to leave space for buttons
                                    fg_color=theme["subframe_fg"],
                                    border_color=theme["button_fg"],
                                    border_width=2,
                                    text_color=theme["label_text"],
                                    hover=True,
                                    hover_color=theme["button_hover"],
                                    button_color=theme["frame_fg"],
                                    highlight_color="#4E8752",
                                    font=("Verdana", 12)
                                )
                                listbox.pack(fill="both", expand=True)



                                # ---- Functions ----
                                def show_page(page_idx):
                                    """Populate the listbox with items of the current page without destroying it"""
                                    listbox.delete(0, "end")
                                    start = page_idx * items_per_page
                                    end = min(start + items_per_page, len(sorted_lemma_freq))
                                    page_items = sorted_lemma_freq[start:end]

                                    for lemma, total in page_items:
                                        display_name = merge_dict[lemma].get()
                                        listbox.insert("end", f"{display_name} ({total})")

                                    # Restore selection from vars_dict
                                    for idx, (lemma, _) in enumerate(page_items):
                                        if vars_dict[lemma].get():
                                            listbox.selection_set(idx)

                                    page_label.configure(text=f"{page_idx + 1} / {total_pages}")
                                    current_page["idx"] = page_idx

                                def delayed_update(event=None):
                                    """Update global vars from the listbox selection (debounced)"""
                                    start = current_page["idx"] * items_per_page
                                    end = min(start + items_per_page, len(sorted_lemma_freq))
                                    page_items = sorted_lemma_freq[start:end]

                                    # Update vars_dict only for items that are currently selected
                                    for idx, (lemma, _) in enumerate(page_items):
                                        vars_dict[lemma].set(idx in listbox.curselection())

                                    # Debounce plot update
                                    if update_job["after_id"]:
                                        container.after_cancel(update_job["after_id"])
                                    update_job["after_id"] = container.after(200, update_plot_callback)

                                listbox.bind("<<ListboxSelect>>", delayed_update)

                                def rename_selected():
                                    new_text = rename_entry.get().strip()
                                    if not new_text:
                                        return

                                    start = current_page["idx"] * items_per_page
                                    end = min(start + items_per_page, len(sorted_lemma_freq))
                                    page_items = sorted_lemma_freq[start:end]

                                    for idx in list(listbox.curselection()):
                                        lemma = page_items[idx][0]
                                        merge_dict[lemma].set(new_text)

                                    show_page(current_page["idx"])
                                    delayed_update()

                                rename_btn.configure(command=rename_selected)

                                def prev_page():
                                    if current_page["idx"] > 0:
                                        show_page(current_page["idx"] - 1)

                                def next_page():
                                    if current_page["idx"] < total_pages - 1:
                                        show_page(current_page["idx"] + 1)

                                prev_btn.configure(command=prev_page)
                                next_btn.configure(command=next_page)

                                # Show first page
                                show_page(0)

                                return container, listbox, rename_entry

                            container_raw, listbox_raw, rename_entry_raw = build_listbox_ui(
                                checkboxes_frame, s_lemma_total_freq, lemma_vars_raw, merge_entry_vars_raw, update_plot
                            )

                            container_norm, listbox_norm, rename_entry_norm = build_listbox_ui(
                                checkboxes_frame, s_lemma_total_freq, lemma_vars_norm, merge_entry_vars_norm,
                                update_plot
                            )

                            def toggle_listboxes(*args):
                                mode = wykres_sort_mode.get()
                                if mode.endswith("_raw"):
                                    container_raw.pack(fill="both", expand=True)
                                    container_norm.pack_forget()
                                else:
                                    container_norm.pack(fill="both", expand=True)
                                    container_raw.pack_forget()

                            wykres_sort_mode.trace_add("write", toggle_listboxes)
                            toggle_listboxes()



                            print("zbudowałem czekboksy")

                        update_data_tables()  # build lemma frequencies etc.



                else:
                    full_results_sorted = []
                    print("Brak wyników zmieniam status na 0")
                    search_status = 0
                    text_result.after(0, lambda: display_page(query, selected_corpus))

                # print(f"Search finished in thread: {threading.current_thread().name}")

            text_result.after(0, update_text_result)
        finally:
            # text_result.tag_raise("sel")
            button_search.configure(state="normal")

    print(f"Search function called in thread: {threading.current_thread().name}")
    thread = threading.Thread(target=search_thread, daemon=True)
    thread.start()
    print("Search thread started.")

cached_image = None
resize_timer = None
target_widget = None  # global variable to hold the widget.

def load_image(img_path):
    global cached_image
    try:
        if not os.path.exists(img_path):
            print("Plot image not found!")
            return None

        img = Image.open(img_path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


# Global variable to store the timer ID for debouncing
debounce_timer = None
def update_plot_images():
    try:
        target_img = "temp/temp_plot.png"
        target_label = frekw_wykresy

        if not os.path.exists(target_img):
            print(f"Plot image not found: {target_img}")
            return

        img = Image.open(target_img).convert("RGBA")

        # Ensure layout is updated to get correct sizes
        target_label.update_idletasks()

        widget_width = target_label.winfo_width()
        widget_height = target_label.winfo_height()

        if widget_width <= 1 or widget_height <= 1:
            return

        # Adjust for other frames (like plot_options_frame) if necessary
        parent_width = target_label.master.winfo_width()
        if 'plot_options_frame' in globals():  # if frame exists
            available_width = parent_width - plot_options_frame.winfo_width()
            widget_width = min(widget_width, available_width)

        # Get Tk scaling
        scaling = float(target_label.winfo_toplevel().tk.call('tk', 'scaling'))
        base_scaling = 1.33
        margin = 1 - 0.6 * (scaling - base_scaling)

        # Apply high-DPI factor for crispness
        high_res_factor = max(1, scaling)
        max_width = int(widget_width * margin * high_res_factor)
        max_height = int(widget_height * margin * high_res_factor)

        # Resize proportionally
        scale = min(max_width / img.width, max_height / img.height)
        final_width = max(1, int(img.width * scale))
        final_height = max(1, int(img.height * scale))
        img_resized = img.resize((final_width, final_height), Image.LANCZOS)

        # Paste into background matching the label size
        final_img = Image.new("RGBA", (widget_width, widget_height), (0, 0, 0, 0))
        paste_x = (widget_width - int(final_width / high_res_factor)) // 2
        paste_y = (widget_height - int(final_height / high_res_factor)) // 2
        final_img.paste(
            img_resized.resize(
                (int(final_width / high_res_factor), int(final_height / high_res_factor)),
                Image.LANCZOS
            ),
            (paste_x, paste_y)
        )

        # Update CTkImage
        img_ctk = ctk.CTkImage(light_image=final_img, size=final_img.size)
        target_label.configure(image=img_ctk, text="")
        target_label.image = img_ctk
        target_label._image_ref = final_img

    except Exception as e:
        print(f"Error loading image: {e}")


# Helper to format month labels depending on corpus logic
def get_month_label(month_key):
    if has_month_column:

        # Original 'miesiąc' column logic: month_key like '1-5'
        try:
            year_group, period_index = month_key.split('-')
            yg = int(year_group)
            pi = int(period_index)
            base_year = 2021 + yg
            # start month/year
            if pi + 1 <= 12:
                sm = pi + 1;
                sy = base_year
            else:
                sm = pi + 1 - 12;
                sy = base_year + 1
            # end month/year
            if sm + 1 <= 12:
                em = sm + 1;
                ey = sy
            else:
                em = sm + 1 - 12;
                ey = sy + 1
            return f"24.{sm:02d}.{sy} - 23.{em:02d}.{ey}"
        except Exception:
            return month_key
    else:
        # Standard year-month labels
        try:
            year, month = month_key.split('-')
            m = int(month)
            return f"{m:02d}.{year}"
        except Exception:
            return month_key

def update_plot():
    global monthly_lemma_freq, monthly_freq_for_use
    global lemma_vars_raw, lemma_vars_norm
    global merge_entry_vars_raw, merge_entry_vars_norm

    # Get current mode from OptionMenu
    mode = wykres_sort_mode.get()
    normalized = mode.endswith("_norm")
    is_month_view = mode.startswith("Miesiąc")

    # Pick correct data & vars
    if normalized:
        freq_dict = monthly_freq_for_use
        lemma_vars_local = lemma_vars_norm
        merge_vars_local = merge_entry_vars_norm
    else:
        freq_dict = monthly_lemma_freq
        lemma_vars_local = lemma_vars_raw
        merge_vars_local = merge_entry_vars_raw

    target_img = "temp/temp_plot.png"

    def generate_and_save_plot(freq_dict, lemma_vars_local, merge_vars_local,
                               target_img, normalized=True, sort_mode="month"):

        # --- Group data monthly or yearly ---
        if sort_mode == "month":
            grouped = {}
            for k, v in freq_dict.items():
                try:
                    year, month = k.split('-')
                    if month != '0':
                        grouped[k] = v
                except Exception:
                    continue
            keys = sorted(grouped.keys(), key=lambda k: (int(k.split('-')[0]), int(k.split('-')[1])))
            x_labels = [get_month_label(k) for k in keys]
        else:
            yearly = {}
            for key, data in freq_dict.items():
                try:
                    year, month = key.split('-')
                    if year == '0000' or month == '0':
                        continue
                    yearly.setdefault(year, {})
                    for lemma, val in data.items():
                        yearly[year][lemma] = yearly[year].get(lemma, 0) + val
                except Exception:
                    continue
            grouped = yearly
            keys = sorted(grouped.keys(), key=int)
            x_labels = keys

        x = np.arange(len(keys))

        # --- Plot style ---
        if styl_wykresow.get() == "ciemny":
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='#2C2F33')
            ax.set_facecolor('#2C2F33')
        else:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 7))

        # --- Group selected lemmas (with renames) ---
        colors = plt.cm.tab20.colors
        groups = {}
        for lemma, var in lemma_vars_local.items():
            if var.get():
                key = merge_vars_local[lemma].get().strip() or lemma
                groups.setdefault(key, []).append(lemma)

        # --- Plot lines ---
        for idx, (grp, lemmas) in enumerate(sorted(groups.items())):
            y = [sum(grouped[k].get(l, 0) for l in lemmas) for k in keys]
            ax.plot(x, y, marker='o', label=grp, color=colors[idx % len(colors)])

        # Enable minor ticks (so unlabeled ticks still exist for gridlines)
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth = 0.5, alpha=0.2)
        ax.set_xlabel('Miesiąc' if sort_mode == "month" else 'Rok')
        ylabel = 'Frekwencja' + (' (per mln)' if normalized else '')
        ax.set_ylabel(ylabel)
        max_labels = 24  # tweak this depending on your preference
        n_labels = len(x_labels)

        if n_labels > max_labels:
            step = int(np.ceil(n_labels / max_labels))
        else:
            step = 1

        # --- Decide which ticks get labels ---
        labeled_idx = set([0, n_labels - 1] + list(range(0, n_labels, step)))

        labels = []
        for i, lbl in enumerate(x_labels):
            if i in labeled_idx:
                labels.append(lbl)  # show label
            else:
                labels.append("")  # hide label

        # --- Set ticks ---
        ax.set_xticks(x)  # all ticks visible
        ax.set_xticklabels(labels, rotation=45 if is_month_view else 0, ha='right')

        ax.set_xlim(x[0] - 1, x[-1] + 1)

        # --- Different tick lengths ---
        for tick, label in zip(ax.xaxis.get_major_ticks(), labels):
            if label == "":  # unlabeled → short tick
                tick.tick1line.set_markersize(3)
                tick.tick2line.set_markersize(3)
            else:  # labeled → longer tick
                tick.tick1line.set_markersize(7)
                tick.tick2line.set_markersize(7)

        # Adjust margins
        renderer = fig.canvas.get_renderer()
        bboxes = [lbl.get_window_extent(renderer) for lbl in ax.get_xticklabels()]
        if bboxes:
            max_w = max(b.width for b in bboxes)
            bottom = min(0.05 + (max_w / fig.get_dpi()), 0.85)
        else:
            bottom = 0.10
        fig.subplots_adjust(bottom=bottom)
        ax.tick_params(axis='x', labelsize=9)

        plt.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.32), frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.85])

        os.makedirs('temp', exist_ok=True)
        plt.savefig(target_img, bbox_inches='tight')
        plt.close(fig)

    # ✅ Call with actual parameters
    generate_and_save_plot(freq_dict, lemma_vars_local, merge_vars_local,
                           target_img, normalized=normalized,
                           sort_mode="month" if is_month_view else "year")

    update_plot_images()

def on_resize(event):
    global debounce_timer
    # Cancel the previous timer if it exists
    if debounce_timer:
        app.after_cancel(debounce_timer)

    # Set a new timer to call update_plot_image after 100ms (or another time of your choosing)
    debounce_timer = app.after(100, lambda: (update_plot_images(), update_plot()))


# Function to save the plot locally
def save_plot_locally():
    # Open a save-as file dialog.
    file_path = filedialog.asksaveasfilename(
        title="Save Plot As",
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All Files", "*.*")]
    )
    if file_path:
        try:
            shutil.copy("temp/temp_plot.png", file_path)
            print(f"Plot saved to {file_path}")
        except Exception as e:
            print("Error saving plot:", e)


# Function to save the plot locally
def save_plot_locally_norm():
    # Open a save-as file dialog.
    file_path = filedialog.asksaveasfilename(
        title="Save Plot As",
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All Files", "*.*")]
    )
    if file_path:
        try:
            # Copy the temp_plot.png file to the user-selected location.
            shutil.copy("temp/temp_plot_norm.png", file_path)
            print(f"Plot saved to {file_path}")
        except Exception as e:
            print("Error saving plot:", e)


def display_full_text(full_text, result, publication_date, title, author, additional_metadata):

    text_full.delete("1.0", ctk.END)

    text_full.insert(ctk.END,
                     f'Data publikacji: {publication_date}, Tytuł: {title}, Autor: {author}')
    # Append additional (dynamic) fields
    if additional_metadata:
        extra_fields = ', '.join(f', {key}: {value}' for key, value in additional_metadata.items())
        text_full.insert(ctk.END, extra_fields)

    text_full.insert(ctk.END, "\n\n")
    text_full.tag_add("text_style", "1.0", ctk.END)
    text_full.insert(ctk.END, full_text[0], "text_style")
    text_full.insert(ctk.END, result[0], "highlight")
    highlight_index = text_full.index(ctk.END)
    text_full.insert(ctk.END, result[1], "highlight_keyword")
    text_full.insert(ctk.END, result[2], "highlight")
    text_full.insert(ctk.END, full_text[2], "text_style")

    # Konfiguracja tagów
    text_full.tag_config("highlight", foreground=highlight_color, spacing1=15, spacing2=10, lmargin1=50, lmargin2=50, rmargin=50)
    text_full.tag_config("highlight_keyword", foreground=highlight_keyword, spacing1=15, spacing2=10, lmargin1=50, lmargin2=50,
                         rmargin=50)
    text_full.tag_config("text_style", spacing1=15, spacing2=10, lmargin1=50, lmargin2=50, rmargin=50)
    text_full.see(highlight_index)


# Function to highlight the specified elements
def highlight_entry(event=None):
    query = entry_query.get("1.0", ctk.END)

    # Reset all tags
    for tag in entry_query.tag_names():
        if tag != "sel":  # don't remove selection
            entry_query.tag_remove(tag, "1.0", ctk.END)

    # --- Highlight keywords first ---
    keywords = ["orth=", "orth!=", "base=", "base!=", "pos=", "pos!=", "ner=", "ner!=", "head=", "head!=",
                "children=", "children!=", "parent=", "parent!=", "deprel=", "deprel!=", "number=", "number!=",
                "gender=", "gender!=", "degree=", "degree!=", "case=", "case!=", "person=", "person!=",
                "accentability=", "accentability!=", "post-prepositionality=", "post-prepositionality!=",
                "accommodability=", "accommodability!=", "aspect=", "aspect!=", "vocalicity=", "vocalicity!=",
                "agglutination=", "agglutination!=", "negation=", "negation!=", "||", "data>", "data<", "data=",
                "data!=", "data<=", "data>=", "autor=", "autor!=", "metadane:", "tytuł=", "tytuł!=",
                "children.group=", "frequency_base", "frequency_orth", "top=", "min=", "max=", "<s>"]

    for term in keywords:
        start_idx = "1.0"
        while True:
            start_idx = entry_query.search(term, start_idx, ctk.END)
            if not start_idx:
                break
            end_idx = f"{start_idx} + {len(term)}c"
            entry_query.tag_add(term, start_idx, end_idx)
            start_idx = end_idx
        entry_query.tag_config(term, foreground=keywords_color)

    # --- Highlight single-character punctuation ---
    punctuation = ["[", "]", "<", ">", "{", "}", "&", '"', "'"]
    for char in punctuation:
        start_idx = "1.0"
        while True:
            start_idx = entry_query.search(char, start_idx, ctk.END)
            if not start_idx:
                break
            end_idx = f"{start_idx} + 1c"
            entry_query.tag_add(char, start_idx, end_idx)
            start_idx = end_idx
        entry_query.tag_config(char, foreground=punctuation_color)

    # --- Highlight text inside quotes ---
    start_idx = "1.0"
    while True:
        first_q = entry_query.search('"', start_idx, ctk.END)
        if not first_q:
            break
        second_q = entry_query.search('"', f"{first_q} + 1c", ctk.END)
        if not second_q:
            break
        entry_query.tag_add("question", f"{first_q} + 1c", second_q)
        start_idx = f"{second_q} + 1c"
    entry_query.tag_config("question", foreground=text_inside_quotation_color)

    # --- Highlight text inside single quotes ---
    start_idx = "1.0"
    while True:
        first_q = entry_query.search("'", start_idx, ctk.END)
        if not first_q:
            break
        second_q = entry_query.search("'", f"{first_q} + 1c", ctk.END)
        if not second_q:
            break
        entry_query.tag_add("question", f"{first_q} + 1c", second_q)
        start_idx = f"{second_q} + 1c"

    entry_query.tag_config("question", foreground=text_inside_quotation_color)


def undo(event=None):
    try:
        entry_query.edit_undo()  # Perform undo
    except:
        pass  # Ignore if no more actions to undo


def redo(event=None):
    try:
        entry_query.edit_redo()  # Perform redo
    except:
        pass  # Ignore if no more actions to redo


def export_data():

    try:
        all_columns = [
            "Data publikacji", "context", "full_text_with_markers",
            "Rezultat", "matched_lemmas",
            "month_key", "Tytuł", "Autor", "additional_metadata",
            "Lewy kontekst", "Prawy kontekst"
        ]

        df_export = pd.DataFrame(full_results_sorted, columns=all_columns)
        meta_df = pd.json_normalize(df_export["additional_metadata"])
        df_flat = pd.concat([df_export.drop(columns=["additional_metadata"]), meta_df], axis=1)
        df_export_slice = df_flat[list(meta_df.columns) + ["Data publikacji", "Autor", "Tytuł",
                                                           "Lewy kontekst", "Rezultat", "Prawy kontekst"]]

        # Use safe default directory
        initial_dir = BASE_DIR if 'BASE_DIR' in globals() else os.path.expanduser("~")

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            initialdir=initial_dir
        )

        if not file_path:
            return  # user cancelled

        # Ensure folder exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Export to Excel with two sheets
        if file_path.lower().endswith(".xlsx"):
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Sheet 1: main export
                df_export_slice.to_excel(writer, sheet_name="Wyniki wyszukiwania", index=False)

                # Sheet 2: paginator_fq['data']
                if 'data' in paginator_fq:
                    data_rows = paginator_fq['data']
                    headers = ["Nr", "Forma podstawowa (base)", "Liczba wystąpień", "Częstość względna"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    df_data.to_excel(writer, sheet_name="Częstość lematów", index=False)
                # Sheet 3: paginator_fq['data']
                if 'data' in paginator_token:
                    data_rows = paginator_token['data']
                    headers = ["Nr", "Forma tekstowa (orth)", "Liczba wystąpień", "Częstość względna"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    df_data.to_excel(writer, sheet_name="Częstość tokenów", index=False)
                # Sheet 4: paginator_fq['data']
                if 'data' in paginator_month:
                    data_rows = paginator_month['data']
                    headers = ["Rok.", "Miesiąc", "Forma podstawowa", "Liczba wystąpień", "Częstość względna"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    df_data.to_excel(writer, sheet_name="Częstość w czasie", index=False)
        else:
            # fallback CSV export (single sheet)
            df_export_slice.to_csv(file_path, index=False)

    except Exception as e:
        print("Export error:", e)

def show_loading_screen():
    loading_win = ctk.CTkToplevel(app)
    loading_win.title("Ładowanie korpusów...")

    width, height = 300, 100

    # Update main app size first
    app.update_idletasks()
    app_width = app.winfo_width()
    app_height = app.winfo_height()
    app_x = app.winfo_x()
    app_y = app.winfo_y()

    # Center relative to app
    x = app_x + (app_width // 2) - (width // 2)
    y = app_y + (app_height // 2) - (height // 2)
    loading_win.geometry(f"{width}x{height}+{x}+{y}")

    # Ensure it's always on top
    loading_win.transient(app)
    loading_win.grab_set()
    loading_win.lift()

    loading_win.protocol("WM_DELETE_WINDOW", lambda: None)

    loading_label = ctk.CTkLabel(loading_win, text="Ładowanie danych, proszę czekać...")
    loading_label.pack(expand=True, pady=20)

    loading_win.update()  # Force redraw
    return loading_win, loading_label  # Return both

def on_entry_click(event=None):
    # Function to clear the placeholder text when the user clicks inside the entry
    if entry_query.get("1.0",
                       ctk.END).strip() == 'Podaj zapytanie np.: [orth="miasta"][pos="prep"][base="Polska"]':
        entry_query.delete("1.0", ctk.END)

def on_focus_out(event=None):
    # Function to reset the placeholder text if the textbox is empty when the user clicks outside
    if not entry_query.get("1.0", ctk.END).strip():
        entry_query.insert("1.0", 'Podaj zapytanie np.: [orth="miasta"][pos="prep"][base="Polska"]')


def keep_selection(event):
    global temp_clipboard
    widget = event.widget  # the textbox that triggered the event
    try:
        temp_clipboard = widget.get("sel.first", "sel.last")
        widget.tag_add("selection", "sel.first", "sel.last")
        widget.tag_config("selection", background="#0078D7", foreground="#ffffff")
    except tk.TclError:
        pass

def remove_selection(event):
    global temp_clipboard
    temp_clipboard = ""
    # Recursively remove selection from all CTkTextboxes starting from root
    def remove_from_children(widget):
        for child in widget.winfo_children():
            if isinstance(child, ctk.CTkTextbox):
                child.tag_remove("selection", "1.0", "end")
            remove_from_children(child)
    remove_from_children(app)  # start from the root window


def save_config():
    data = {
        'font_family': font_family.get(),
        'fontsize': fontsize,
        'styl_wykresow': styl_wykresow.get(),
        'motyw': motyw.get(),
        'plotting': plotting.get(),
        'kontekst': kontekst

    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Settings window
def settings_window():
    global settings_popup, fontsize, font_family, plotting, kontekst
    theme = THEMES[motyw.get()]

    # Callbacks
    def restore_defaults():
        global settings_popup, fontsize, font_family, kontekst, plotting
        font_family.set(DEFAULT_SETTINGS['font_family'])
        styl_wykresow.set(DEFAULT_SETTINGS['styl_wykresow'])
        motyw.set(DEFAULT_SETTINGS['motyw'])
        fontsize_entry.delete(0, 'end')
        fontsize_entry.insert(0, str(DEFAULT_SETTINGS['fontsize']))
        fontsize = DEFAULT_SETTINGS['fontsize']
        plotting.set(DEFAULT_SETTINGS['plotting'])
        kontekst_entry.delete(0, 'end')
        kontekst_entry.insert(0, str(DEFAULT_SETTINGS['kontekst']))
        kontekst = DEFAULT_SETTINGS['kontekst']
        apply_theme()
        save_config()
        settings_popup.destroy()
        settings_popup = None

    def on_save():
        global settings_popup, fontsize, font_family, kontekst
        try:
            fontsize = int(fontsize_entry.get())
        except ValueError:
            fontsize = DEFAULT_SETTINGS['fontsize']
        try:
            kontekst = int(kontekst_entry.get())
        except ValueError:
            kontekst = DEFAULT_SETTINGS['kontekst']
        apply_theme()
        save_config()
        try:
            font_tuple = (font_family.get(), fontsize)
            frekw_dane_tabela.set_font(font_tuple)
            frekw_dane_tabela_orth.set_font(font_tuple)
            frekw_dane_tabela_month.set_font(font_tuple)
        except NameError:
            pass
        settings_popup.destroy()
        settings_popup = None

    if settings_popup and settings_popup.winfo_exists():
        settings_popup.lift()
        return

    settings_popup = ctk.CTkToplevel(app)
    settings_popup.title('Ustawienia')
    settings_popup.geometry('420x660')
    settings_popup.grab_set()
    settings_popup.configure(fg_color=theme["app_bg"])  # use theme

    # Frame for all settings
    settings_frame = ctk.CTkFrame(settings_popup, fg_color=theme["subframe_fg"], corner_radius=15)
    settings_frame.pack(fill="both", expand=True, padx=15, pady=15)

    entry_height = 35  # consistent with rest of app
    button_height = 35

    # Font size
    ctk.CTkLabel(settings_frame, text='Rozmiar czcionki:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    fontsize_entry = ctk.CTkEntry(settings_frame, width=150, height=entry_height, font=("Verdana", 12),
                                  fg_color=theme["frame_fg"], corner_radius=8)
    fontsize_entry.insert(0, str(fontsize))
    fontsize_entry.pack(pady=5)

    # Font family
    ctk.CTkLabel(settings_frame, text='Czcionka:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    font_options = ['Verdana', 'JetBrains Mono', 'Arial', 'Tahoma', 'Times New Roman', 'Lato', 'Segoe']
    ctk.CTkComboBox(settings_frame, values=font_options, variable=font_family,
                     fg_color=theme["button_fg"], dropdown_fg_color=theme["dropdown_fg"],
                     dropdown_hover_color=theme["dropdown_hover"], text_color=theme["button_text"],
                     font=("Verdana", 12), height=entry_height).pack(pady=5)

    # Chart style
    ctk.CTkLabel(settings_frame, text='Styl wykresów:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    ctk.CTkComboBox(settings_frame, values=['jasny', 'ciemny'], variable=styl_wykresow,
                     fg_color=theme["button_fg"], dropdown_fg_color=theme["dropdown_fg"],
                     dropdown_hover_color=theme["dropdown_hover"], text_color=theme["button_text"],
                     font=("Verdana", 12), height=entry_height).pack(pady=5)

    # Theme
    ctk.CTkLabel(settings_frame, text='Motyw:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    ctk.CTkComboBox(settings_frame, values=['jasny', 'ciemny'], variable=motyw,
                     fg_color=theme["button_fg"], dropdown_fg_color=theme["dropdown_fg"],
                     dropdown_hover_color=theme["dropdown_hover"], text_color=theme["button_text"],
                     font=("Verdana", 12), height=entry_height).pack(pady=5)

    # Plots
    ctk.CTkLabel(settings_frame, text='Rysowanie wykresów:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    plot_options = ['Tak', 'Nie']
    ctk.CTkComboBox(settings_frame, values=plot_options, variable=plotting,
                     fg_color=theme["button_fg"], dropdown_fg_color=theme["dropdown_fg"],
                     dropdown_hover_color=theme["dropdown_hover"], text_color=theme["button_text"],
                     font=("Verdana", 12), height=entry_height).pack(pady=5)

    # Context
    ctk.CTkLabel(settings_frame, text='Liczba tokenów w rozszerzonym kontekście:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    kontekst_entry = ctk.CTkEntry(settings_frame, width=150, height=entry_height, font=("Verdana", 12),
                                  fg_color=theme["frame_fg"], corner_radius=8)
    kontekst_entry.insert(0, str(kontekst))
    kontekst_entry.pack(pady=5)

    # Buttons frame
    button_frame = ctk.CTkFrame(settings_frame, fg_color=theme["subframe_fg"], corner_radius=12)
    button_frame.pack(pady=20)

    ctk.CTkButton(
        button_frame,
        text='Domyślne',
        fg_color=theme["button_fg"],
        hover_color=theme["button_hover"],
        text_color=theme["button_text"],
        font=("Verdana", 12, "bold"),
        height=button_height,
        corner_radius=8,
        command=restore_defaults
    ).grid(row=0, column=0, padx=10)

    ctk.CTkButton(
        button_frame,
        text='Zapisz',
        fg_color=theme["button_fg"],
        hover_color=theme["button_hover"],
        text_color=theme["button_text"],
        font=("Verdana", 12, "bold"),
        height=button_height,
        corner_radius=8,
        command=on_save
    ).grid(row=0, column=1, padx=10)


# Callback to update rows per page selection.
def update_rows_per_page(selected_value):
    global rows_per_page, current_page, global_query, global_selected_corpus
    rows_per_page = int(selected_value)
    current_page = 0  # Reset to first page when rows per page changes.
    text_result.set_rows_number(rows_per_page)
    if global_query and global_selected_corpus:
        display_page(global_query, global_selected_corpus)


def save_to_file():
    # Get the filename from the entry box
    file_name = fiszka_entrybox.get()

    # Get the selected text from the text_result
    try:
        # First try the widget's internal selection
        selected_text = text_full.get("sel.first", "sel.last")
    except tk.TclError:
        try:
            # Fallback to system selection
            selected_text = text_full.get("selection.first", "selection.last")
        except tk.TclError:
            # Nothing selected at all
            selected_text = ""

    # Check if there's any selected text and filename is not empty
    if selected_text and file_name:
        try:
            # Open the file in append mode (creates the file if it doesn't exist)

            with open(f'fiszki/{file_name}.txt', 'a', encoding='utf-8') as file:
                # Append the selected text to the file, followed by a newline
                first_line = (text_full.get("1.0", "end-1c")).split('\n')[0]
                file.write(f'Korpus: {corpus_var.get()}, {first_line}\n\n{selected_text}<br><br>')
            print(f"Selected text successfully appended to {file_name}.")
            update_dropdown()
            flash_button(save_selection_button, "green")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        if not selected_text:
            print("No text selected to save.")
        elif not file_name:
            print("Please provide a valid filename.")

# Function to get list of txt files
def get_txt_files():
    folder_path = "fiszki"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create folder if not exists
    return [f[:-4] for f in os.listdir(folder_path) if f.endswith(".txt")]

webview_thread = None  # Global variable to track the thread

#Przewodnik po języku zapytań
def open_webview_window():
    global webview_thread

    # If thread exists and is alive, don't open a new window
    if webview_thread and webview_thread.is_alive():
        print("Webview is already running.")
        return webview_thread

    def worker_przewodnik():
        file_path = os.path.join(BASE_DIR, "Przewodnik po języku zapytań.html")
        window = webview.create_window(
            "Przewodnik Korpusuj",
            url=f"file://{file_path}",
            width=1200,
            height=800,
            resizable=True,
            text_select=True,
        )

        webview.start(debug=False)

    # Start the webview in a new thread
    webview_thread = threading.Thread(target=worker_przewodnik, name="MainThread")
    webview_thread.start()
    return webview_thread

def fiszki_load_file_content(value):
    """Uruchamia load_file_content w osobnym wątku (jeden na raz)."""
    global webview_thread

    # sprawdzamy czy już działa
    if webview_thread is not None and webview_thread.is_alive():
        print("Loader already running.")
        return webview_thread

    def worker():
        fiszki_tkinter.load_file_content(value)

    # inicjalizacja nowego wątku
    webview_thread = threading.Thread(target=worker, name="MainThread", daemon=True)
    webview_thread.start()

    return webview_thread

def flash_button(button, color):
    original_color = button.cget("fg_color")  # Store original color
    button.configure(fg_color=color)  # Change to success/error color
    app.after(500, lambda: button.configure(fg_color=original_color))  # Reset after 2s


def copy_text(event):
    global temp_clipboard
    widget = event.widget

    # Check if the widget has 'get' and supports selection
    if hasattr(widget, "get"):
        try:
            selected_text = widget.get("sel.first", "sel.last")
            widget.clipboard_clear()
            widget.clipboard_append(selected_text)
            return "break"  # stop further processing
        except tk.TclError:
            pass  # no selection, fall back to temp_clipboard

    # Fallback: use temp_clipboard if not empty
    if temp_clipboard:
        # Use root to access clipboard safely
        root = widget.winfo_toplevel()
        root.clipboard_clear()
        root.clipboard_append(temp_clipboard)
        return "break"

def add_textbox_context_menu(widget, allow_paste=False):
    # Attach a right-click context menu to a CTkTextbox.

    menu = tk.Menu(widget, tearoff=0)

    # Copy
    def copy():
        try:
            selected_text = widget.get("sel.first", "sel.last")
            # selected_text = selected_text.replace("\n", "\r\n")
        except tk.TclError:
            selected_text = widget.get("1.0", "end-1c")
        widget.clipboard_clear()
        widget.clipboard_append(selected_text)

    # Select All
    def select_all():
        widget.tag_add("sel", "1.0", "end")
        widget.mark_set("insert", "1.0")
        widget.see("insert")

    # Paste
    def paste():
        try:
            text_to_insert = widget.clipboard_get()
            widget.insert("insert", text_to_insert)
        except tk.TclError:
            pass  # clipboard empty

    menu.add_command(label="Kopiuj", command=copy)
    if allow_paste:
        menu.add_command(label="Wklej", command=paste)
    menu.add_command(label="Zaznacz wszystko", command=select_all)

    def show_menu(event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    widget.bind("<Button-3>", show_menu)

# Funkcja obsługująca klawisz Enter
def on_enter(event=None):
    search()

def update_dropdown():
    new_values = get_txt_files()  # Fetch latest file list
    dropdown.configure(values=new_values)  # Update dropdown list

def load_data_on_startup(loading_label=None):
    global dataframes
    for i, (name, path) in enumerate(files.items(), start=1):
        try:
            if loading_label:
                loading_label.configure(text=f"Ładowanie {name} ({i}/{len(files)})...")
                loading_label.update()

            df = pd.read_parquet(path)  # Changed to read_parquet

            dataframes[name] = df
            print(f"{name} loaded and preprocessed: {len(df)} rows")
        except FileNotFoundError:
            print(f"Error: {path} not found!")
        except ValueError:
            print(f"Error: {path} is not a valid Parquet file!")


def load_corpora():
    global files, corpus_options, corpus_var, dataframes
    # Use BASE_DIR to locate corpus_files folder
    corpus_dir = os.path.join(BASE_DIR_CORP)

    # Ask user to select files
    file_paths = filedialog.askopenfilenames(
        title="Wybierz plik(i) korpusu",
        initialdir=corpus_dir if os.path.exists(corpus_dir) else os.path.expanduser("~"),
        filetypes=[("Parquet files", "*.parquet")]
    )

    if not file_paths:
        print("Nie wybrano plików.")
        return

    loading_screen, loading_label = show_loading_screen()

    # Extract corpus names
    corpus_options = [os.path.basename(f).replace(".parquet", "") for f in file_paths]
    corpus_var.set(corpus_options[0])

    # Map names to paths
    files = {name: path for name, path in zip(corpus_options, file_paths)}

    # Load
    load_data_on_startup(loading_label=loading_label)

    # Update dropdown
    option_corpus.configure(values=corpus_options, variable=corpus_var)

    # Close loading window
    loading_screen.destroy()
    print("Korpusy załadowane.")

def register_text_widget(widget):
    text_widgets.append(widget)
    # apply initial font
    widget.configure(font=(font_family.get(), fontsize))


def apply_plot_style():
    plt.style.use('dark_background' if styl_wykresow.get() == 'ciemny' else 'default')

THEMES = {
    "ciemny": {
        # Base
        "app_bg": "#1F2328",

        # Tables
        "row_colors": ("#2C2F33", "#33373D"),
        "text_colors": ["#FFFFFF", "#FFFFFF", "#65A46F", "#FFFFFF"],
        "text_colors_month": ["white", "white", "white", "#65a46f", "white"],
        "selected_row": "#3A75C4",
        "canvas_bg": "#2C2F33",

        # Widgets
        "frame_fg": "#2C2F33",
        "subframe_fg": "#1F2328",
        "button_fg": "#4B6CB7",
        "button_hover": "#5B7CD9",
        "button_text": "white",
        "label_text": "white",
        "dropdown_fg": "#4B6CB7",
        "dropdown_hover": "#5B7CD9",

        # Syntax highlighting
        "highlight": "#65a46f",
        "highlight_keyword": "#37E152",
        "question": "#eed870",
        "keywords": "#a393ca",
        "punctuation": "#669a5d",
        "quotation": "#e68672",
    },
    "jasny": {
    # Base
    "app_bg": "#CED3D3",           # light gray main background

    # Tables
    "row_colors": ("#E6E8E8", "#F2F4F4"),      # alternating light rows
    "text_colors": ["black", "black", "#000DFF", "black"],
    "text_colors_month": ["black", "black", "black", "#000DFF", "black"],
    "selected_row": "#A3C9F1",                   # highlight selected row
    "canvas_bg": "#E6E8E8",

    # Widgets
    "frame_fg": "#F5F7F7",        # frame backgrounds slightly lighter than app_bg
    "subframe_fg": "#E6E8E8",
    "button_fg": "#6BA6F7",       # lighter blue buttons
    "button_hover": "#89BDFA",    # hover effect
    "button_text": "black",        # dark text
    "label_text": "black",
    "dropdown_fg": "#6BA6F7",      # lighter blue dropdown buttons
    "dropdown_hover": "#89BDFA",   # hover effect for dropdown
    "dropdown_text": "black",      # dropdown list text

        # Syntax highlighting
        "highlight": "#000DFF",
        "highlight_keyword": "#D400FF",
        "question": "#0084bc",
        "keywords": "#ac35aa",
        "punctuation": "#986801",
        "quotation": "#50a14f",
    },
}

def apply_theme():
    global highlight_color, highlight_keyword, question_marks_color, keywords_color, punctuation_color, text_inside_quotation_color

    theme = THEMES[motyw.get()]  # pick "ciemny" or "jasny"

    ctk.set_appearance_mode("dark" if motyw.get() == "ciemny" else "light")
    app.configure(fg_color=theme["app_bg"])

    # Style menu bar and menu buttons
    menu._menu_bar.configure(fg_color=theme["app_bg"])
    for m in menu._menu_widgets:
        m.configure(
            bg=theme["app_bg"],
            fg=theme["button_text"],
            activebackground=theme["button_fg"],
            activeforeground=theme["button_text"]
        )
        m.menu.configure(
            bg=theme["app_bg"],
            fg=theme["button_text"],
            activebackground=theme["button_fg"],
            activeforeground=theme["button_text"]
        )

    # --- Frames ---
    top_frame_container.configure(fg_color=theme["frame_fg"])
    lemma_frame.configure(fg_color=theme["frame_fg"])
    month_frame.configure(fg_color=theme["frame_fg"])
    orth_frame.configure(fg_color=theme["frame_fg"])
    result_frame.configure(fg_color=theme["frame_fg"])
    tabview.configure(fg_color=theme["frame_fg"])

    # Subframes
    for frame in [
        pagination_frame, entry_button_frame, pagination_lemma_frame, pagination_orth_frame,
        pagination_month_frame, plot_options_frame,
        saveplot_button_frame,
         checkboxes_frame,
    ]:
        frame.configure(fg_color=theme["subframe_fg"])

    # --- Buttons ---
    for button in [
        button_search, settings_button, button_first, button_prev, button_next, button_last,
        button_first_lemma, button_prev_lemma, button_next_lemma, button_last_lemma,
        button_first_orth, button_prev_orth, button_next_orth, button_last_orth,
        button_first_month, button_prev_month, button_next_month, button_last_month,
        button_save_plot,  save_selection_button
    ]:
        button.configure(
            fg_color=theme["button_fg"],
            hover_color=theme["button_hover"],
            text_color=theme["button_text"]
        )

    # --- Labels ---
    for label in [
        label_corpus, label_left_context, label_right_context, label_sort,
        page_label, page_label_lemma, page_label_orth, page_label_month, plot_type_label,
                frekw_wykresy, rows_label
    ]:
        label.configure(text_color=theme["label_text"])

    # --- OptionMenus ---
    for option in [option_corpus, option_sort, dropdown_rows, dropdown, table_selector, wykres_sort_menu]:
        option.configure(
            fg_color=theme["dropdown_fg"],
            dropdown_fg_color=theme["dropdown_fg"],
            dropdown_hover_color=theme["dropdown_hover"],
            text_color=theme["button_text"],
            dropdown_text_color = theme["button_text"]
        )

    # --- Entries / Textboxes ---
    for entry in [entry_query, entry_left_context, entry_right_context, fiszka_entrybox, text_full]:
        entry.configure(
            fg_color=theme["subframe_fg"],
            text_color=theme["label_text"]
        )

    # --- Tabview ---
    tabview._segmented_button.configure(
        fg_color=theme["frame_fg"],  # tab container background
        selected_color=theme["button_fg"],  # active tab
        unselected_color=theme["subframe_fg"],  # inactive tabs
        text_color=theme["button_text"],  # inactive tab text
        selected_hover_color=theme["button_hover"],  # hover on active tab
        unselected_hover_color=theme["dropdown_hover"],  # hover on inactive tab
    )

    # Fonts
    font_tuple = (font_family.get(), fontsize)
    for tbl in (text_result, frekw_dane_tabela, frekw_dane_tabela_orth, frekw_dane_tabela_month):
        tbl.set_header_font(font_tuple)

    # Tables
    for tbl in (text_result, frekw_dane_tabela, frekw_dane_tabela_orth):
        tbl.set_row_colors(*theme["row_colors"])
        tbl.set_text_colors(theme["text_colors"])
        tbl.set_selected_row_color(theme["selected_row"])
        tbl.set_canvas_background(theme["canvas_bg"])

    frekw_dane_tabela_month.set_text_colors(theme["text_colors_month"])
    frekw_dane_tabela_month.set_row_colors(*theme["row_colors"])
    frekw_dane_tabela_month.set_selected_row_color(theme["selected_row"])
    frekw_dane_tabela_month.set_canvas_background(theme["canvas_bg"])

    # Syntax highlighting
    highlight_color = theme["highlight"]
    highlight_keyword = theme["highlight_keyword"]
    question_marks_color = theme["question"]
    keywords_color = theme["keywords"]
    punctuation_color = theme["punctuation"]
    text_inside_quotation_color = theme["quotation"]


    register_text_widget(text_full)

def show_table(choice):
    lemma_frame.grid_remove()
    orth_frame.grid_remove()
    month_frame.grid_remove()

    if choice == "Formy podstawowe (base)":
        lemma_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Formy ortograficzne (orth)":
        orth_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Częstość w czasie":
        month_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)



class Menu:
    """
    Custom Menu Class with theme support.
    """

    def __init__(self, root: ctk.CTk, theme=None) -> None:
        self._root = root
        self._theme = theme  # store theme dictionary
        self._menu_bar = ctk.CTkFrame(self._root, cursor="hand2")
        self._menu_bar.pack(side="top", fill="x")
        self._menu_widgets: list[tk.Menubutton] = []
        self._theme = theme or {
            "frame_fg": "#CED3D3",
            "button_fg": "#357EDD",
            "button_text": "black",
            "label_text": "black"
        }

    def menu_bar(self, text: str, **kwargs) -> tk.Menu:
        menu = tk.Menubutton(
            self._menu_bar,
            text=text,
            bg=self._theme["frame_fg"],
            fg=self._theme["label_text"],
            activebackground=self._theme["button_fg"],
            activeforeground=self._theme["button_text"]
        )
        menu.menu = tk.Menu(menu, **kwargs)
        menu["menu"] = menu.menu
        menu.pack(side="left", padx=2, pady=2)
        self._menu_widgets.append(menu)

        # Apply theme to menu items
        menu.menu.configure(
            bg=self._theme["frame_fg"],
            fg=self._theme["label_text"],
            activebackground=self._theme["button_fg"],
            activeforeground=self._theme["button_text"]
        )
        return menu.menu

    def update_theme(self, theme: dict):
        """Call this to update the menu when theme changes."""
        self._theme = theme
        self._menu_bar.configure(fg_color=self._theme["frame_fg"])
        for menu in self._menu_widgets:
            menu.configure(
                bg=self._theme["frame_fg"],
                fg=self._theme["label_text"],
                activebackground=self._theme["button_fg"],
                activeforeground=self._theme["button_text"],
            )
            menu.menu.configure(
                bg=self._theme["frame_fg"],
                fg=self._theme["label_text"],
                activebackground=self._theme["button_fg"],
                activeforeground=self._theme["button_text"],
            )



# Tworzenie interfejsu GUI
app = ctk.CTk()
app.withdraw()
splash = ctk.CTkToplevel(app)

splash.overrideredirect(True)  # Removes window decorations
# Desired splash size
width, height = 400, 400

# Get screen dimensions
screen_width = splash.winfo_screenwidth()
screen_height = splash.winfo_screenheight()

# Calculate position for center
x = (screen_width - width) // 2
y = (screen_height - height) // 2

splash.geometry(f"{width}x{height}+{x}+{y}")

logo_image = ctk.CTkImage(
    Image.open("temp/logo.png"),  # path to your image
    size=(400, 400)          # resize to fit splash screen
)

# Create a label with the image
splash_label = ctk.CTkLabel(splash, image=logo_image, text="")  # text="" removes text
splash_label.pack(expand=True)
splash.update()

menu = Menu(app)

file_menu = menu.menu_bar(text="Plik", tearoff=0)
file_menu.add_command(label="Nowy projekt", command=load_corpora)
file_menu.add_command(label="Utwórz korpus", command=lambda: creator.main())
file_menu.add_command(label="Eksportuj wyniki", command=export_data)
file_menu.add_separator()
file_menu.add_command(label="Zamknij", command=lambda: exit())
file_menu = menu.menu_bar(text="Edytuj", tearoff=0)
file_menu.add_command(label="Cofnij", command=lambda: undo())
file_menu.add_command(label="Ponów", command=lambda: redo())
file_menu = menu.menu_bar(text="Ustawienia", tearoff=0)
file_menu.add_command(label="Preferencje", command=settings_window)
file_menu = menu.menu_bar(text="Pomoc", tearoff=0)
file_menu.add_command(label="Przewodnik po języku zapytań", command=open_webview_window)

app.title("Korpusuj")
icon_path = os.path.join(BASE_DIR, "favicon.ico")
app.iconbitmap(icon_path)


# Global vars
font_family = ctk.StringVar(value=config['font_family'])
fontsize = config['fontsize']
styl_wykresow = ctk.StringVar(value=config['styl_wykresow'])
motyw = ctk.StringVar(value=config['motyw'])
plotting = ctk.StringVar(value=config.get('plotting', DEFAULT_SETTINGS['plotting']))
kontekst = config.get('kontekst', DEFAULT_SETTINGS['kontekst'])
settings_popup = None

# menu = CTkTitleMenu(app)
# button_1 = menu.add_cascade("Plik")
# button_2 = menu.add_cascade("Edycja")
# button_3 = menu.add_cascade("Ustawienia")
# button_4 = menu.add_cascade("Pomoc")

corpus_var = ctk.StringVar(value="")

# ---------- Top Query/Settings Frame ----------
top_frame_container = ctk.CTkFrame(app, fg_color="#2C2F33", corner_radius=15)
top_frame_container.pack(fill="x", side="top", padx=10, pady=(10,5))

# Keep original column weights
top_frame_container.grid_columnconfigure(0, weight=1)
top_frame_container.grid_columnconfigure(1, weight=1)
top_frame_container.grid_columnconfigure(2, weight=18)
top_frame_container.grid_columnconfigure(3, weight=1)
top_frame_container.grid_columnconfigure(4, weight=1)
top_frame_container.grid_columnconfigure(5, weight=1)
top_frame_container.grid_columnconfigure(6, weight=1)
top_frame_container.grid_columnconfigure(7, weight=1)

# Corpus selection
label_corpus = ctk.CTkLabel(top_frame_container, text="Wybierz korpus:", font=("Verdana", 12, 'bold'), text_color="white")
label_corpus.grid(row=1, column=1, padx=1, pady=1, sticky="w")

option_corpus = ctk.CTkOptionMenu(
    top_frame_container,
    values=corpus_options,
    variable=corpus_var,
    font=("Verdana", 12, 'bold'),
    fg_color="#4B6CB7",
    dropdown_fg_color="#4B6CB7",
    dropdown_hover_color="#5B7CD9",
    text_color="white",
    dropdown_font=("Verdana", 12, 'bold'),
    width=120,
    height=35,
    corner_radius=8
)
option_corpus.grid(row=2, column=1, padx=1, pady=1, sticky="w")

# Query widget (keep background)
entry_query = ctk.CTkTextbox(
    top_frame_container, height=100, font=("JetBrains Mono Bold", 14),
    exportselection=False, corner_radius=12, fg_color="#1F2328"
)
entry_query.grid(row=0, rowspan=4, column=2, padx=15,  pady=(5,5), sticky="ew")
entry_query.insert("1.0", 'Podaj zapytanie np.: [orth="miasta"][pos="prep"][base="Polska"]')
entry_query.bind("<FocusIn>", on_entry_click)
entry_query.bind("<FocusOut>", on_focus_out)
entry_query.bind("<KeyRelease>", highlight_entry)

# Search button
s_img = ctk.CTkImage(dark_image=Image.open("temp/s.png"), size=(50, 50))
button_search = ctk.CTkButton(
    top_frame_container, text="", image=s_img,
    fg_color="#4B6CB7", hover_color="#5B7CD9", width=50, height=50, command=search
)
button_search.grid(row=1, rowspan=2, column=3, pady=1, sticky="w")

# Left/Right Context
label_left_context = ctk.CTkLabel(top_frame_container, text="Kontekst (l):", font=("Verdana", 12, 'bold'), text_color="white")
label_left_context.grid(row=1, column=4, padx=1, pady=1, sticky="w")
entry_left_context = ctk.CTkEntry(top_frame_container, width=40, fg_color="#2C2F33", text_color="white", corner_radius=8)
entry_left_context.grid(row=1, column=5, padx=1, pady=1, sticky="w")
entry_left_context.insert(0, "10")

label_right_context = ctk.CTkLabel(top_frame_container, text="Kontekst (r):", font=("Verdana", 12, 'bold'), text_color="white")
label_right_context.grid(row=2, column=4, padx=1, pady=1, sticky="w")
entry_right_context = ctk.CTkEntry(top_frame_container, width=40, fg_color="#2C2F33", text_color="white", corner_radius=8)
entry_right_context.grid(row=2, column=5, padx=1, pady=1, sticky="w")
entry_right_context.insert(0, "10")

# Sort options
label_sort = ctk.CTkLabel(top_frame_container, text="Sortuj wyniki:", font=("Verdana", 12, 'bold'), text_color="white")
label_sort.grid(row=1, column=6, padx=1, pady=1, sticky="w")
sort_option_var = tk.StringVar(value="Alfabetycznie")
option_sort = ctk.CTkOptionMenu(
    top_frame_container,
    values=["Alfabetycznie", "Lewy kontekst", "Prawy kontekst", "Autor", "Tytuł", "Data publikacji"],
    variable=sort_option_var,
    font=("Verdana", 12, 'bold'),
    fg_color="#4B6CB7",
    dropdown_fg_color="#4B6CB7",
    dropdown_hover_color="#3E3782",
    dropdown_font=("Verdana", 12, 'bold'),
    text_color="white",
    width=120,
    height=35,
    corner_radius=8
)
option_sort.grid(row=2, column=6, padx=1, pady=1, sticky="w")

# Settings button
settings_icon = ctk.CTkImage(dark_image=Image.open("temp/u.png"), size=(50, 50))
settings_button = ctk.CTkButton(top_frame_container, image=settings_icon, text="", fg_color="#4B6CB7", hover_color="#5B7CD9", width=50, height=50, command=settings_window)
settings_button.grid(row=1, rowspan=2, column=7, pady=1, sticky="w")


# Create tab view
tabview = ctk.CTkTabview(
    app,
    corner_radius=12,
    border_width=0,
    fg_color="#2C2F33",  # background of tabview and tabs
    border_color="#3E3F42",  # subtle border
    segmented_button_fg_color="#2C2F33",
    segmented_button_selected_color="#4B6CB7",           # active tab color
    segmented_button_selected_hover_color="#5B7CD9",     # hover when selected
    segmented_button_unselected_color="#2C2F33",         # inactive tab color
    segmented_button_unselected_hover_color="#3E3F42",   # hover inactive
    text_color="white",                                   # inactive tab text
    anchor="n",
    state="normal"
)

tabview.pack(fill="both", expand=True, padx=10, pady=(0,10))

# Add tabs
tab_wyniki = tabview.add("Wyniki wyszukiwania")
tab_wyniki_frekw = tabview.add("Statystyki")
tab_wyniki_wykresy = tabview.add("Trendy")
tab_wyniki_wykresy.pack_propagate(False)

tabview._segmented_button.configure(
    font=ctk.CTkFont(family="Verdana", size=13, weight='bold'),
    fg_color="#2C2F33",               # tab background
    selected_color="#4B6CB7",         # selected tab
    text_color="white",                # unselected tab text
    corner_radius=8,
    border_width=0

)

# Optionally, make the tabview border subtle
tabview.configure(border_width=0, border_color="#3E3F42")

# ------------------------------
# Main Page
# ------------------------------
# Main result frame
result_frame = ctk.CTkFrame(tab_wyniki, corner_radius=15, fg_color="#2C2F33")
result_frame.pack(fill="both", expand=True, padx=15, pady=15)

# Top frame for pagination + entry/buttons
top_frame = ctk.CTkFrame(result_frame, fg_color="transparent")
top_frame.pack(fill="x", padx=10, pady=(10, 5))
top_frame.grid_columnconfigure(0, weight=1, uniform="group1")
top_frame.grid_columnconfigure(1, weight=1, uniform="group1")

# Middle frame for tables/text
middle_frame = ctk.CTkFrame(result_frame, fg_color="transparent")
middle_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
middle_frame.grid_columnconfigure(0, weight=1, uniform="group1")
middle_frame.grid_columnconfigure(1, weight=1, uniform="group1")
middle_frame.grid_rowconfigure(0, weight=1)

# ------------------------------
# Pagination Frame (Original Layout Preserved)
# ------------------------------
pagination_frame = ctk.CTkFrame(top_frame, fg_color="#1F2328", corner_radius=12)
pagination_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# Modern button styling
button_kwargs = dict(
    width=35,
    height=35,
    corner_radius=8,
    border_width=0,
    fg_color="#4B6CB7",               # dark blue
    hover_color="#5B7CD9",            # lighter blue on hover
    border_color=None,
    text_color="white",
    font=("Verdana", 12, 'bold'),
    anchor="center",
    hover=True,
    state="normal"
)

# Pagination buttons (using Unicode arrows)
button_first = ctk.CTkButton(pagination_frame, text="|<", command=first_page, **button_kwargs)
button_first.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

button_prev = ctk.CTkButton(pagination_frame, text="<", command=prev_page, **button_kwargs)
button_prev.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

page_label = ctk.CTkLabel(pagination_frame, text="1/1", font=("Verdana", 12, 'bold'), text_color="#FFFFFF")
page_label.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

button_next = ctk.CTkButton(pagination_frame, text=">", command=next_page, **button_kwargs)
button_next.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

button_last = ctk.CTkButton(pagination_frame, text=">|", command=last_page, **button_kwargs)
button_last.grid(row=1, column=4, padx=5, pady=5, sticky="ew")

# Rows dropdown
rows_label = ctk.CTkLabel(pagination_frame, text="Liczba wierszy na stronie:", font=("Verdana", 12, 'bold'), text_color="#FFFFFF")
rows_label.grid(row=1, column=5, padx=5, pady=5, sticky="e")

rows_options = ["10", "50", "100", "250", "500", "1000"]
rows_var = ctk.StringVar(value="100")
dropdown_rows = ctk.CTkOptionMenu(pagination_frame, font=("Verdana", 12, 'bold'), values=rows_options, variable=rows_var,
                                  command=update_rows_per_page, width=120, height=35,  corner_radius=8,
                                  fg_color="#4B6CB7", dropdown_fg_color="#4B6CB7", dropdown_hover_color="#3E3782", dropdown_font=("Verdana", 12, 'bold'))
dropdown_rows.grid(row=1, column=6, padx=5, pady=5, sticky="ew")

# Make the single row expand vertically
pagination_frame.grid_rowconfigure(1, weight=1)  # row 1 contains all buttons and dropdown

# Make columns expand evenly (you already did this)
[pagination_frame.grid_columnconfigure(i, weight=1) for i in range(7)]

# ------------------------------
# Entry and Buttons Frame
# ------------------------------
entry_button_frame = ctk.CTkFrame(top_frame, fg_color="#1F2328", corner_radius=12)
entry_button_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

fiszka_entrybox = ctk.CTkEntry(entry_button_frame, placeholder_text="Nazwa fiszki",
                               font=("Verdana", 12, 'bold'), height=35, corner_radius=8, fg_color="#2C2F33")
fiszka_entrybox.pack(pady=10, padx=10, fill="x", expand=True, side="left")

selected_file = ctk.StringVar(value="Otwórz fiszkę")
dropdown = ctk.CTkOptionMenu(
    entry_button_frame,
    variable=selected_file,
    values=get_txt_files(),
    command=fiszki_load_file_content,
    font=("Verdana", 12, 'bold'),
    corner_radius=8,
    width=120,
    height=35,  # ⬅ match pagination
    fg_color="#4B6CB7",
    dropdown_fg_color="#4B6CB7",
    dropdown_hover_color="#3E3782",
    dropdown_font=("Verdana", 12, 'bold')
)
dropdown.pack(pady=10, padx=5, side="right")

save_selection_button = ctk.CTkButton(
    entry_button_frame,
    text="Zapisz fiszkę",
    font=("Verdana", 13, 'bold'),
    corner_radius=8,
    width=120,
    height=35,
    fg_color="#4E8752",
    hover_color="#57965C",
    command=save_to_file
)
save_selection_button.pack(pady=10, padx=5, side="right")


# ------------------------------
# Middle Frame: Table and Full Text
# ------------------------------
min_column_widths = [150, 150, 100, 150]
justify_list = ["center", "right", "center", "left"]
headers = ["Metadane", "Lewy Kontekst", "Rezultat", "Prawy Kontekst"]
data = []

text_result = table.CustomTable(middle_frame, headers, data, min_column_widths, justify_list,
                                rows_per_page, fulltext_data=[])
text_result.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=10)
text_result.set_text_anchor(["center", "e", "center", "w"])

text_full = ctk.CTkTextbox(middle_frame, font=(font_family.get(), fontsize),
                           wrap="word", exportselection=False, corner_radius=12, fg_color="#1F2328")
text_full.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=10)
text_full.bind("<FocusOut>", keep_selection)

# Context menus
add_textbox_context_menu(text_full, allow_paste=False)
add_textbox_context_menu(entry_query, allow_paste=True)

# Equal resizing
result_frame.columnconfigure(0, weight=1)

# ------------------------------
# Tables
# ------------------------------
# Frequency data

tab_wyniki_frekw.grid_rowconfigure(0, weight=0)  # OptionMenu row
tab_wyniki_frekw.grid_rowconfigure(1, weight=1)  # Tables row
tab_wyniki_frekw.grid_columnconfigure(0, weight=1)

selected_table = tk.StringVar(value="Formy podstawowe (base)")

table_selector = ctk.CTkOptionMenu(
    tab_wyniki_frekw,
    values=["Formy podstawowe (base)", "Formy ortograficzne (orth)", "Częstość w czasie"],
    variable=selected_table,
    command=lambda choice: show_table(choice),
    font=("Verdana", 12, 'bold'),
    corner_radius=8,
    width=120,
    height=35,  # ⬅ match pagination
    fg_color="#4B6CB7",
    dropdown_fg_color="#4B6CB7",
    dropdown_hover_color="#3E3782",
    dropdown_font=("Verdana", 12, 'bold')
)
table_selector.grid(row=0, column=0, pady=(30,5), padx=10)

# --- Shared styles ---
button_kwargs_small = dict(
    width=35,
    height=35,
    corner_radius=8,
    border_width=0,
    fg_color="#4B6CB7",
    hover_color="#5B7CD9",
    text_color="white",
    font=("Verdana", 12, 'bold'),
    anchor="center"
)

label_kwargs_small = dict(
    font=("Verdana", 12, 'bold'),
    text_color="white"
)

fq_headers = ["Nr", "Forma podstawowa (base)", "Liczba wystąpień", "Częstość wzgędna (na 1000 000 segmentów)"]
fq_data = []

fq_min_column_widths = [150, 150, 100, 150]
fq_justify_list = ["center", "center", "center", "center"]

lemma_frame = ctk.CTkFrame(
    tab_wyniki_frekw,
    fg_color="#2C2F33",       # match main result frame
    corner_radius=15           # rounded corners like main frames
)
lemma_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

pagination_lemma_frame = ctk.CTkFrame(
    lemma_frame,
    fg_color="#1F2328",        # dark background to match theme
    corner_radius=12
)
pagination_lemma_frame.grid(row=0, column=0, sticky="ew", pady=5, padx=5)

for col in range(5):
    pagination_lemma_frame.columnconfigure(col, weight=1)


pagination_lemma_frame.grid_rowconfigure(0, pad=5)
pagination_lemma_frame.grid_columnconfigure(0, pad=5)

button_first_lemma = ctk.CTkButton(pagination_lemma_frame, text="|<", command=lambda: first_p(paginator_fq), **button_kwargs_small)
button_first_lemma.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

button_prev_lemma = ctk.CTkButton(pagination_lemma_frame, text="<", command=lambda: prev_p(paginator_fq), **button_kwargs_small)
button_prev_lemma.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

page_label_lemma = ctk.CTkLabel(pagination_lemma_frame, text="1/1", **label_kwargs_small)
page_label_lemma.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

button_next_lemma = ctk.CTkButton(pagination_lemma_frame, text=">", command=lambda: next_p(paginator_fq), **button_kwargs_small)
button_next_lemma.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

button_last_lemma = ctk.CTkButton(pagination_lemma_frame, text=">|", command=lambda: last_p(paginator_fq), **button_kwargs_small)
button_last_lemma.grid(row=1, column=4, padx=5, pady=5, sticky="ew")

frekw_dane_tabela = table.CustomTable(lemma_frame, fq_headers, fq_data, fq_min_column_widths,
                                      fq_justify_list, 15, fulltext_data=[])
frekw_dane_tabela.grid(row=1, column=0, sticky="nsew", pady=0)

paginator_fq = {
    "data": fq_data,
    "current_page": [0],
    "table": frekw_dane_tabela,
    "label": page_label_lemma,
    "items_per_page": 15
}

lemma_frame.rowconfigure(0, weight=1, uniform="group1")
lemma_frame.rowconfigure(1, weight=5, uniform="group1")
lemma_frame.columnconfigure(0, weight=1)


orth_frame = ctk.CTkFrame(
    tab_wyniki_frekw,
    fg_color="#2C2F33",
    corner_radius=15
)
orth_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

pagination_orth_frame = ctk.CTkFrame(
    orth_frame,
    fg_color="#1F2328",
    corner_radius=12
)
pagination_orth_frame.grid(row=0, column=0, sticky="ew", pady=5, padx=5)

for col in range(5):
    pagination_orth_frame.columnconfigure(col, weight=1)


pagination_orth_frame.grid_rowconfigure(0, pad=5)
pagination_orth_frame.grid_columnconfigure(0, pad=5)

fq_headers_token = ["Nr", "Forma tekstowa (orth)", "Liczba wystąpień", "Częstość wzgędna (na 1000 000 segmentów)"]
fq_data_token = []

# Create the buttons, labels, and dropdown in the pagination_frame using grid
button_first_orth = ctk.CTkButton(pagination_orth_frame, text="|<", command=lambda: first_p(paginator_token), **button_kwargs_small)
button_first_orth.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

button_prev_orth = ctk.CTkButton(pagination_orth_frame, text="<", command=lambda: prev_p(paginator_token), **button_kwargs_small)
button_prev_orth.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

page_label_orth = ctk.CTkLabel(pagination_orth_frame, text="1/1", **label_kwargs_small)
page_label_orth.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

button_next_orth = ctk.CTkButton(pagination_orth_frame, text=">", command=lambda: next_p(paginator_token), **button_kwargs_small)
button_next_orth.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

button_last_orth = ctk.CTkButton(pagination_orth_frame, text=">|", command=lambda: last_p(paginator_token), **button_kwargs_small)
button_last_orth.grid(row=1, column=4, padx=5, pady=5, sticky="ew")

frekw_dane_tabela_orth = table.CustomTable(orth_frame, fq_headers_token, fq_data_token,
                                           fq_min_column_widths, fq_justify_list, 15, fulltext_data=[])
frekw_dane_tabela_orth.grid(row=1, column=0, sticky="nsew", pady=0)

paginator_token = {
    "data": fq_data_token,
    "current_page": [0],
    "table": frekw_dane_tabela_orth,
    "label": page_label_orth,
    "items_per_page": 15
}

orth_frame.rowconfigure(0, weight=1, uniform="group1")
orth_frame.rowconfigure(1, weight=5, uniform="group1")
orth_frame.columnconfigure(0, weight=1)

# month table
fq_headers_month = ["Rok", "Miesiąc", "Forma podstawowa", "Liczba wystąpień",
                    "Częstość wzgędna (na 1000 000 segmentów)"]
fq_data_month = []

month_frame = ctk.CTkFrame(
    tab_wyniki_frekw,
    fg_color="#2C2F33",
    corner_radius=15
)
month_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

pagination_month_frame = ctk.CTkFrame(
    month_frame,
    fg_color="#1F2328",
    corner_radius=12
)
pagination_month_frame.grid(row=0, column=0, sticky="ew", pady=5, padx=5)

for col in range(5):
    pagination_month_frame.columnconfigure(col, weight=1)
pagination_month_frame.grid_rowconfigure(0, pad=5)
pagination_month_frame.grid_columnconfigure(0, pad=5)

# Create the buttons, labels, and dropdown in the pagination_frame using grid
button_first_month = ctk.CTkButton(pagination_month_frame, text="|<", command=lambda: first_p(paginator_month), **button_kwargs_small)
button_first_month.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

button_prev_month = ctk.CTkButton(pagination_month_frame, text="<", command=lambda: prev_p(paginator_month), **button_kwargs_small)
button_prev_month.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

page_label_month = ctk.CTkLabel(pagination_month_frame, text="1/1", **label_kwargs_small)
page_label_month.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

button_next_month = ctk.CTkButton(pagination_month_frame, text=">", command=lambda: next_p(paginator_month), **button_kwargs_small)
button_next_month.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

button_last_month = ctk.CTkButton(pagination_month_frame, text=">|", command=lambda: last_p(paginator_month), **button_kwargs_small)
button_last_month.grid(row=1, column=4, padx=5, pady=5, sticky="ew")

frekw_dane_tabela_month = table.CustomTable(month_frame, fq_headers_month, fq_data_month, [100] * 5,
                                            ["center"] * 5, 15, fulltext_data=[])
frekw_dane_tabela_month.grid(row=1, column=0, sticky="nsew", pady=0)

paginator_month = {
    "data": fq_data_month,
    "current_page": [0],
    "table": frekw_dane_tabela_month,
    "label": page_label_month,
    "items_per_page": 15
}

month_frame.rowconfigure(0, weight=1, uniform="group1")
month_frame.rowconfigure(1, weight=5, uniform="group1")
month_frame.columnconfigure(0, weight=1)

# ------------------------------
# Plots
# ------------------------------
plot_options_frame = ctk.CTkFrame(tab_wyniki_wykresy, fg_color="#2C2F33", corner_radius=15)
plot_options_frame.pack(pady=10, padx=10, side="left")

# Save button container
saveplot_button_frame = ctk.CTkFrame(plot_options_frame, fg_color="#1F2328", corner_radius=12)
saveplot_button_frame.pack(pady=5, padx=5, fill="x")

# Plot type label
plot_type_label = ctk.CTkLabel(saveplot_button_frame, text="Wybierz typ wykresu:", font=("Verdana", 13, 'bold'))
plot_type_label.pack(pady=5, padx=5, fill="x")

# Single mode variable (StringVar) for plot type
wykres_sort_mode = ctk.StringVar(value="Miesiąc_raw")

# Dropdown menu for choosing plot mode
wykres_sort_menu = ctk.CTkOptionMenu(
    saveplot_button_frame,   # or whichever frame holds your plot controls
    variable=wykres_sort_mode,
    values=[
        "Miesiąc_raw",     # miesięcznie, nieznormalizowane
        "Rok_raw",    # miesięcznie, znormalizowane
        "Miesiąc_norm",      # rocznie, nieznormalizowane
        "Rok_norm"      # rocznie, znormalizowane
    ],
    command=lambda _: update_plot(),# refresh when changed
    font=("Verdana", 12, 'bold'),
    corner_radius=8,
    width=120,
    height=35,
    fg_color="#4B6CB7",
    dropdown_fg_color="#4B6CB7",
    dropdown_hover_color="#3E3782",
    dropdown_font=("Verdana", 12, 'bold')
)
wykres_sort_menu.pack(pady=5, padx=5, fill="x")

# Checkboxes frame (we will swap between raw/normalized listboxes here)
checkboxes_frame = ctk.CTkFrame(plot_options_frame, fg_color="#1F2328", corner_radius=12)
checkboxes_frame.pack(pady=5, padx=5, fill="x")

# Save plot button
button_save_plot = ctk.CTkButton(
    saveplot_button_frame,
    text="Zapisz wykres",
    font=("Verdana", 12, 'bold'),
    fg_color="#4B6CB7",
    hover_color="#5B7CD9",
    text_color="white",
    corner_radius=8,
    height=35,
    command=lambda: save_plot_locally()
)
button_save_plot.pack(padx=5, pady=5, fill="x")

frekw_wykresy = ctk.CTkLabel(tab_wyniki_wykresy, text="", font=("Verdana", 16, 'bold'))
frekw_wykresy.pack(fill="both", expand=True)
frekw_wykresy.bind("<Configure>", on_resize)

# Update dropdown values on startup
dropdown.configure(values=get_txt_files())

# Przypisanie Enter do pola wpisywania lematu i całej aplikacji
entry_query.bind("<Return>", on_enter)
entry_left_context.bind("<Return>", on_enter)
entry_right_context.bind("<Return>", on_enter)


# Enable undo/redo (since CTkEntry is based on tk.Entry)
entry_query.configure(undo=True)
# Bind Ctrl+Z to undo
entry_query.bind("<Control-z>", undo)
entry_query.bind("<Control-Z>", undo)  # For some systems

# Bind Ctrl+Y to redo
entry_query.bind("<Control-y>", redo)
entry_query.bind("<Control-Y>", redo)  # For some systems

# Show "Lematy" by default
show_table("Formy podstawowe (base)")
app.bind("<Button-1>", remove_selection)
app.bind_all("<Control-c>", copy_text)

# Apply on startup
apply_theme()
apply_plot_style()
app.state("zoomed")
splash.destroy()
app.deiconify()
app.update()

app.mainloop()