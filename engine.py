import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import logging
from logging.handlers import RotatingFileHandler
import traceback
import pandas as pd
import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import re
import json
import threading
from PIL import Image
import shutil
from collections import Counter
import warnings
import math
from datetime import datetime, timedelta
import ast
import string
import table
from tkinter import messagebox
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta
import calendar

def notify_status(msg):
    # Sprawdzamy, czy launcher jest uruchomiony i ma funkcję update_status
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'update_status'):
        sys.modules['__main__'].update_status(msg)

notify_status("Wczytywanie bibliotek systemowych...")

# ==========================================
# LAZY LOADERY (Wczytywanie na żądanie)
# ==========================================
_creator_module = None
_creator_module = None


def get_creator_module():
    global _creator_module
    if _creator_module is None:
        # 1. Tworzymy tymczasowe okienko informacyjne
        loading_win = ctk.CTkToplevel(app)
        loading_win.title("Ładowanie...")
        loading_win.geometry("300x120")
        loading_win.attributes("-topmost", True)
        loading_win.overrideredirect(True)  # Opcjonalnie: usuwa ramkę okna

        # Centrowanie względem okna głównego
        x = app.winfo_x() + (app.winfo_width() // 2) - 150
        y = app.winfo_y() + (app.winfo_height() // 2) - 60
        loading_win.geometry(f"+{x}+{y}")

        lbl = ctk.CTkLabel(loading_win, text="Ładowanie modułu kreatora korpusów...\nMoże to potrwać kilka sekund.",
                           font=("Verdana", 12))
        lbl.pack(expand=True, pady=20)

        # 2. Wymuszamy natychmiastowe narysowanie okienka
        loading_win.update()

        # 3. Zmieniamy kursor na "oczekiwanie" (kółko/klepsydra)
        app.configure(cursor="wait")

        # 4. Właściwy ciężki import modułu
        import creator
        _creator_module = creator

        # 5. Sprzątanie: zamykamy okienko i przywracamy kursor
        loading_win.destroy()
        app.configure(cursor="")

    return _creator_module

_fiszki_module = None
def get_fiszki_module():
    global _fiszki_module
    if _fiszki_module is None:
        import fiszki_tkinter
        _fiszki_module = fiszki_tkinter
    return _fiszki_module

_plot_stack = None
def get_plot_stack():
    global _plot_stack
    if _plot_stack is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.cm as cm
        _plot_stack = {
            "plt": plt,
            "Figure": Figure,
            "FigureCanvasAgg": FigureCanvasAgg,
            "cm": cm
        }
    return _plot_stack
# ==========================================


warnings.filterwarnings("ignore")
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except (ImportError, AttributeError):
    pass # Zignoruj na Mac/Linux
from dataclasses import dataclass, field

@dataclass
class SearchState:
    query: str = ""
    corpus: str = ""
    results: list = field(default_factory=list)
    monthly_lemma_freq: dict = field(default_factory=dict)
    true_monthly_totals: dict = field(default_factory=dict)
    monthly_freq_for_use: dict = field(default_factory=dict)
    monthly_tfidf_for_use: dict = field(default_factory=dict)
    monthly_zscore_for_use: dict = field(default_factory=dict)
    lemma_df_cache: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    fq_data: list = field(default_factory=list)
    fq_data_token: list = field(default_factory=list)
    fq_data_month: list = field(default_factory=list)
    s_lemma_total_freq: list = field(default_factory=list)
    s_lemma_global_pmw: list = field(default_factory=list)
    s_lemma_global_tfidf: list = field(default_factory=list)
    unique_lemmas: set = field(default_factory=set)
    has_dates: bool = False

current_state = SearchState()
state_lock = threading.Lock()
text_widgets = []
dataframes = {}
inverted_indexes = {}
files = {}
corpus_options = []
lemma_vars = {}
merge_entry_vars = {}
monthly_lemma_freq = {}
temp_clipboard = ""
lemma_df_cache = {}

global monthly_freq_for_use, true_monthly_totals
monthly_freq_for_use = {}
true_monthly_totals = {}
styl_wykresow = None  # set in UI

wykres_sort = None  # set in UI



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




LOG_PATH = os.path.join(BASE_DIR_CORP, "korpusuj.log")

# Konfiguracja root loggera z rotacją plików
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Wyczyść istniejące handlery, żeby uniknąć duplikatów wpisów
root_logger.handlers.clear()

log_handler = RotatingFileHandler(
    LOG_PATH,
    mode="a",
    maxBytes=1_000_000,   # 1 MB na plik
    backupCount=5,        # trzymaj 5 archiwów: .1 ... .5
    encoding="utf-8"
)

log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))

root_logger.addHandler(log_handler)

logging.info("Logger initialized")



notify_status("Ładowanie zasobów i czcionek...")
# Define font paths
FONT1_PATH = os.path.join(BASE_DIR, "fonts", "JetBrainsMono-Bold.ttf")
FONT2_PATH = os.path.join(BASE_DIR, "fonts", "JetBrainsMono-Regular.ttf")

# Load both fonts
ctk.FontManager.load_font(FONT1_PATH)
ctk.FontManager.load_font(FONT2_PATH)

file_path = 'temp/temp_plot.png'
if os.path.exists(file_path):
    os.remove(file_path)


# ---------------------------
# Global pagination variables
# ---------------------------
current_page = 0
rows_per_page = 100
full_results_sorted = []
global_query = ""
global_selected_corpus = ""
search_status = 0


# ---------------------------
# Globalne bezpieczeństwo wyszukiwania i komunikaty
# ---------------------------
search_guard = threading.Lock()
search_in_progress = False
active_search_token = 0

last_search_warnings = []
last_search_error = ""

class QueryValidationError(Exception):
    pass

class SearchExecutionError(Exception):
    pass

class QueryParseError(Exception):
    pass



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
    "numcol": {"number": 0, "case": 1, "gender": 2, "accommodability": 3},
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


def calc_z_score(val, mean_val, std_val):
    """Zwraca Z-score lub None w przypadku braku wariancji."""
    return ((val - mean_val) / std_val) if std_val > 0 else None

def safe_ll(o, e):
    """Bezpieczne log-likelihood bez ryzyka dzielenia przez zero."""
    return o * math.log(o / e) if o > 0 and e > 0 else 0.0

def calc_pmw(frequency, total_tokens):
    """Częstość na milion słów (Per Million Words)."""
    return (frequency / total_tokens) * 1_000_000 if total_tokens > 0 else 0.0


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
    new_data = []
    full_data = []


    # Iterate over only the slice for the current page.
    for idx, (publication_date, context, full_text, matched_text, matched_lemmas, month_key, title, author,
              additional_metadata,
              left_context, right_context, row_idx, start_idx_val, end_idx_val) in enumerate(
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
        row_full_data = (full_text, context, publication_date, title, author, additional_metadata, row_idx,
                         start_idx_val)
        full_data.append(row_full_data)

    text_result.set_data(new_data)
    text_result.set_fulltext_data(full_data)

    def handle_row_click(row_index):
        if 0 <= row_index - 1 < len(text_result.fulltext_data):
            fdata = text_result.fulltext_data[row_index - 1]
            # Przekazujemy wszystkie 8 parametrów (dodane fdata[6] i fdata[7])
            display_full_text(fdata[0], fdata[1],  fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7])

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


def global_sort_callback(paginator, col_index, ascending):
    """Sortuje całokształt danych w paginatorze i odświeża widok na 1 stronę."""

    def sort_key(row):
        val = row[col_index] if col_index < len(row) else ""
        return val if val is not None else ""

    try:
        paginator["data"].sort(key=sort_key, reverse=not ascending)
    except TypeError:
        # Fallback w przypadku wymieszania str i int
        paginator["data"].sort(key=lambda x: str(sort_key(x)), reverse=not ascending)

    # Po posortowaniu przenieś użytkownika z powrotem na pierwszą stronę
    paginator["current_page"][0] = 0
    update_table(paginator)

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
        import re
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

# --- Main Function: find_lemma_context ---

def find_lemma_context(query, df, selected_corpus, left_context_size=10, right_context_size=10, warnings_list=None):
    if warnings_list is None:
        warnings_list = []

    global search_status




    # Wymuszenie odświeżenia UI (pokazanie ekranu ładowania)
    text_result.after(0, lambda: display_page(query, selected_corpus))

    # Pobranie opcji frekwencyjnych z zapytania
    freq_opts = parse_frequency_attributes(query, "frequency_orth")
    freq_base_opts = parse_frequency_attributes(query, "frequency_base")

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

    def get_rows_for_conditions(conditions_list, corpus_index, total_rows):
        """
        Recursively extracts exact required words from conditions and
        returns the intersecting set of valid row IDs.
        """
        if not conditions_list:
            return set(total_rows)

        block_rows = set(total_rows)
        has_constraints = False

        for cond in conditions_list:
            if not cond: continue

            # Handle repetitions [n, m] safely
            if isinstance(cond, tuple) and cond[0] == "repeat":
                min_rep = cond[2]
                if min_rep == 0:
                    continue  # SKIP: This token is optional {0,x}, we can't filter by it!

                inner_cond = cond[1] if isinstance(cond[1], list) else [cond[1]]
                inner_rows = get_rows_for_conditions(inner_cond, corpus_index, total_rows)
                if inner_rows is not None:
                    block_rows.intersection_update(inner_rows)
                    has_constraints = True
                continue

            if len(cond) < 5: continue
            key, values, operator, is_nested, match_type = cond

            # We ONLY filter on exact, required matches
            if operator == "=":
                if is_nested:
                    # Recursive call: peek inside the dependent={...} or head={...}
                    inner_rows = get_rows_for_conditions(values, corpus_index, total_rows)
                    if inner_rows is not None:
                        block_rows.intersection_update(inner_rows)
                        has_constraints = True
                elif match_type == "exact" and key in ("base", "orth"):
                    # Handle multiple values (e.g. orth="Polska|Niemcy")
                    val_rows = set()
                    for val in values:
                        val_rows.update(corpus_index[key].get(val, set()))
                    block_rows.intersection_update(val_rows)
                    has_constraints = True

        return block_rows if has_constraints else None

    def get_prefiltered_rows(parsed_groups, corpus_name, total_rows):
        """Applies the pre-filter to all groups (including || OR groups) and <s> tags."""
        corpus_index = inverted_indexes.get(corpus_name)
        if not corpus_index:
            return set(total_rows)

        final_valid_rows = set()

        for token_query_conditions, s_ordered, sentence_query_conditions in parsed_groups:
            group_rows = set(total_rows)

            # 1. Process standard token conditions
            if token_query_conditions:
                for bracket in token_query_conditions:
                    bracket_conds = bracket if isinstance(bracket, list) else [bracket]
                    b_rows = get_rows_for_conditions(bracket_conds, corpus_index, total_rows)
                    if b_rows is not None:
                        group_rows.intersection_update(b_rows)

            # 2. Process sentence <s> conditions (e.g. <s [base="wojna"]>)
            if sentence_query_conditions:
                for bracket in sentence_query_conditions:
                    bracket_conds = bracket if isinstance(bracket, list) else [bracket]
                    b_rows = get_rows_for_conditions(bracket_conds, corpus_index, total_rows)
                    if b_rows is not None:
                        group_rows.intersection_update(b_rows)

            # Union the results (because parsed_groups are separated by ||)
            final_valid_rows.update(group_rows)

        return final_valid_rows

    # --- Vectorized pre-filtering using token-level conditions only ---
    for token_query_conditions, s_ordered, sentence_query_conditions in parsed_query_groups:
        # 1. Use the inverted index to instantly throw away irrelevant rows!
        valid_row_ids = get_prefiltered_rows(
            parsed_query_groups,
            selected_corpus,
            df.index
        )

        # 2. Filter the dataframe down immediately
        filtered_df = df.loc[list(valid_row_ids)].copy()

        # 3. Create the mask for the REMAINING rows for author/date filters
        mask = pd.Series(True, index=filtered_df.index)


        # Apply author/title/date/metadata filters using `mask` here

        filtered_df = filtered_df[mask]

        # --- Author filters ---
        if author_filters:
            if 'Autor' not in filtered_df.columns:
                add_warning(warnings_list, 'Filtr "autor" został pominięty: w korpusie brak kolumny "Autor".')
            else:
                author_series = filtered_df['Autor'].astype(str).str.lower()
                for op, value, match_type in author_filters:
                    val = value.lower()
                    if match_type == "exact":
                        submask = author_series == val
                    else:
                        submask = author_series.str.contains(value, regex=True, flags=re.IGNORECASE, na=False)
                    if op == "!=":
                        submask = ~submask
                    mask &= submask

        # --- Title filters ---
        if title_filters:
            if 'Tytuł' not in filtered_df.columns:
                add_warning(warnings_list, 'Filtr "tytuł" został pominięty: w korpusie brak kolumny "Tytuł".')
            else:
                title_series = filtered_df['Tytuł'].astype(str).str.lower()
                for op, value, match_type in title_filters:
                    val = value.lower()
                    if match_type == "exact":
                        submask = title_series == val
                    else:
                        submask = title_series.str.contains(value, regex=True, flags=re.IGNORECASE, na=False)
                    if op == "!=":
                        submask = ~submask
                    mask &= submask

        # --- Date filters ---
        if date_filters:
            if 'Data publikacji' not in filtered_df.columns:
                add_warning(warnings_list, 'Filtr "data" został pominięty: w korpusie brak kolumny "Data publikacji".')
            else:
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
                    else:
                        if match_type == "exact":
                            submask = date_series == value
                        else:
                            submask = date_series.str.contains(value, regex=True, flags=re.IGNORECASE, na=False)
                    if op == "!=":
                        submask = ~submask
                    mask &= submask

        # --- Metadata filters ---

        if metadata_filters:
            for column, op, value, match_type in metadata_filters:
                if column not in filtered_df.columns:
                    add_warning(warnings_list, f'Filtr metadanych został pominięty: brak kolumny "{column}".')
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
                    else:
                        submask = series >= val
                else:
                    if match_type == "exact":
                        submask = series == val
                    elif match_type == "regex":
                        submask = series.apply(lambda x: bool(re.fullmatch(value, x, flags=re.IGNORECASE)))
                    else:
                        submask = series.str.contains(value, regex=True, flags=re.IGNORECASE, na=False)

                if op == "!=":
                    submask = ~submask

                mask &= submask

        # ✅ Apply all filters at once
        filtered_df = filtered_df[mask]



        # Tworzymy szybkie słowniki dla kolumn metadanych przed pętlą
        dates_dict = df["Data publikacji"].to_dict() if "Data publikacji" in df.columns else {}
        titles_dict = df["Tytuł"].to_dict() if "Tytuł" in df.columns else {}
        authors_dict = df["Autor"].to_dict() if "Autor" in df.columns else {}

        # Z góry definiujemy kolumny do zignorowania w additional_metadata
        exclude_cols = {
            "Data publikacji", "Tytuł", "Autor", "tags", "Treść", "token_counts",
            "tokens", "lemmas", "deprels", "postags", "full_postags",
            "word_ids", "sentence_ids", "head_ids", "start_ids", "end_ids", "ners", "upostags",
            "corefs"
        }

        # Słownik słowników dla reszty metadanych
        meta_columns = [col for col in df.columns if col not in exclude_cols]
        meta_dicts = {col: df[col].to_dict() for col in meta_columns}
        # -----------------------------------------------------------------------

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
            # Pobieramy corefs, z zabezpieczeniem dla starych korpusów
            corefs = getattr(row, "corefs", None)
            # --- Pre-kompilacja klastrów koreferencji dla danego wiersza ---
            coref_lemma_clusters = {}
            if corefs is not None:
                for idx, c_tags in enumerate(corefs):
                    if c_tags is None: continue
                    if isinstance(c_tags, str): c_tags = [c_tags]  # Bezpiecznik dla bardzo starych plików

                    for c_tag in c_tags:
                        if c_tag in ("0", "O", "_", None): continue

                        parts = c_tag.split("-", 1)
                        c_id_str = parts[1] if len(parts) > 1 else c_tag

                        if c_id_str not in coref_lemma_clusters:
                            coref_lemma_clusters[c_id_str] = set()

                        coref_lemma_clusters[c_id_str].add(str(lemmas[idx]).lower())
                        coref_lemma_clusters[c_id_str].add(str(tokens[idx]).lower())
            # ------------------------------------------------------------------------

            num_tokens = len(tokens)
            if num_tokens == 0:
                continue


            parent_idx, children_lookup = build_dependency_maps(
                sentence_ids, word_ids, head_ids
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
                    if key in ("orth", "base", "pos", "deprel", "ner", "upos") or key.startswith("coref"):
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
                        elif key.startswith("coref"):
                            c_tags = corefs[token_idx] if corefs is not None else []
                            if isinstance(c_tags, str): c_tags = [c_tags]

                            match_found = False

                            for c_tag in c_tags:
                                if c_tag in ("0", "O", "_", None): continue

                                tag_parts = c_tag.split("-", 1)
                                token_role = tag_parts[0] if len(tag_parts) > 1 else ""
                                c_id = tag_parts[1] if len(tag_parts) > 1 else c_tag

                                required_role = ""
                                if "(H)" in key:
                                    required_role = "Head"
                                elif "(P)" in key:
                                    required_role = "Part"

                                if required_role and token_role != required_role:
                                    continue

                                cluster_words = coref_lemma_clusters.get(c_id, set())

                                for val in values:
                                    val_lower = val.lower()
                                    if match_type == "exact" and val_lower in cluster_words:
                                        match_found = True
                                    elif match_type == "regex" and any(
                                        re.fullmatch(val_lower, w, re.IGNORECASE) for w in cluster_words):
                                        match_found = True
                                    elif match_type == "regex_search" and any(
                                        re.search(val_lower, w, re.IGNORECASE) for w in cluster_words):
                                        match_found = True

                                if match_found:
                                    break

                            if operator == "=" and not match_found:
                                return False
                            elif operator == "!=" and match_found:
                                return False

                            continue
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
                    elif key.startswith("head") or key.startswith("head.group"):
                        # 'children' in your existing code checks the *parent* of token_idx, keep that behaviour.
                        parent = parent_idx[token_idx]

                        # If there's no parent: "=" must fail, "!=" should pass (nothing to forbid)
                        if parent is None or parent < 0:
                            if operator == "=":
                                return False
                            else:  # operator == "!="
                                continue

                        # Parse children(...) syntax: children(N), children(<N), children(>N)
                        m = re.match(r'head(?:\.group)?(?:\((<|>|=)?(-?\d+)\))?$', key)
                        dist_op = m.group(1) if m and m.group(1) else None
                        dist_val = int(m.group(2)) if m and m.group(2) else None

                        # Helper: check distance filter; returns True if parent/child distance passes the dist filter
                        def _distance_matches_child(dist_val, dist_op):
                            if dist_val is None:
                                return True
                            distance =  word_ids[parent]  - word_ids[token_idx]# <-- same formula as children_dist
                            if dist_op in (None, "="):
                                return distance == dist_val
                            elif dist_op == "<":
                                return distance < dist_val
                            elif dist_op == ">":
                                return distance > dist_val
                            return False

                        if operator == "=":
                            # require that parent (optionally constrained by distance) matches values/nested
                            if not _distance_matches_child(dist_val, dist_op):
                                return False

                            if is_nested:
                                # nested: parent must satisfy nested conditions
                                if not match_conditions(parent, tuple(values)):
                                    return False
                            else:
                                parent_attr = lemmas[parent]
                                if match_type == "exact":
                                    if parent_attr not in values:
                                        return False
                                elif match_type == "regex":
                                    if not any(re.fullmatch(v, parent_attr) for v in values):
                                        return False
                                elif match_type == "regex_search":
                                    if not any(re.search(v, parent_attr) for v in values):
                                        return False

                        elif operator == "!=":
                            # require that parent (within optional distance constraint) does NOT match values/nested
                            # If distance constraint is present but doesn't match, the negation is vacuously satisfied:
                            if dist_val is not None and not _distance_matches_child(dist_val, dist_op):
                                # no parent at that specified distance → nothing to forbid
                                continue

                            if is_nested:
                                # if parent satisfies nested => fail the whole condition
                                if match_conditions(parent, tuple(values)):
                                    return False
                            else:
                                parent_attr = lemmas[parent]
                                if match_type == "exact":
                                    if parent_attr in values:
                                        return False
                                elif match_type == "regex":
                                    if any(re.fullmatch(v, parent_attr) for v in values):
                                        return False
                                elif match_type == "regex_search":
                                    if any(re.search(v, parent_attr) for v in values):
                                        return False

                        else:
                            # unknown operator (safe-fail)
                            return False


                    elif key.startswith("dependent"):
                        # Combined parent + parent_dist logic with correct nested semantics.
                        # children_lookup[token_idx] are the token_idx's children (original code's semantics).
                        children = children_lookup[token_idx]
                        if not children:
                            # If looking for existence and there are no children -> fail.
                            # If it's a negation (operator !=), absence of children satisfies the condition.
                            if operator == "=":
                                return False
                            else:  # operator == "!="
                                continue  # no children -> nothing to forbid
                        # parse optional distance filter: parent(3), parent(-2), parent(<3), parent(>1)
                        m = re.match(r'dependent(?:\((<|>|=)?(-?\d+)\))?$', key)
                        dist_op = m.group(1) if m and m.group(1) else None
                        dist_val = int(m.group(2)) if m and m.group(2) else None
                        if operator == "=":
                            # We require that at least one child (optionally satisfying distance) matches values/nested.
                            found = False
                            for child in children:
                                # apply distance filter (if present). distance = child - parent (linear)
                                if dist_val is not None:
                                    distance = word_ids[child] - word_ids[token_idx]
                                    if dist_op in (None, "=") and distance != dist_val:
                                        continue
                                    elif dist_op == "<" and not (distance < dist_val):
                                        continue
                                    elif dist_op == ">" and not (distance > dist_val):
                                        continue

                                # nested conditions: evaluate recursively on the child
                                if is_nested:
                                    if match_conditions(child, tuple(values)):
                                        found = True
                                        break
                                else:
                                    # simple attribute match on the child (original behavior)
                                    child_attr = lemmas[child]
                                    if match_type == "exact":
                                        if child_attr in values:
                                            found = True
                                            break

                                    elif match_type == "regex":
                                        if any(re.fullmatch(v, child_attr) for v in values):
                                            found = True
                                            break

                                    elif match_type == "regex_search":
                                        if any(re.search(v, child_attr) for v in values):
                                            found = True
                                            break
                            if not found:
                                return False



                        elif operator == "!=":
                            # We require that NO child (that passes optional distance filter) matches values/nested.
                            for child in children:
                                # distance filter (if present)
                                if dist_val is not None:
                                    distance = word_ids[child] - word_ids[token_idx]
                                    if dist_op in (None, "=") and distance != dist_val:
                                        continue
                                    elif dist_op == "<" and not (distance < dist_val):
                                        continue
                                    elif dist_op == ">" and not (distance > dist_val):
                                        continue
                                if is_nested:
                                    # if any child satisfies the nested condition -> the entire parent!= fails
                                    if match_conditions(child, tuple(values)):
                                        return False
                                else:
                                    child_attr = lemmas[child]
                                    if match_type == "exact":
                                        if child_attr in values:
                                            return False
                                    elif match_type == "regex":
                                        if any(re.fullmatch(v, child_attr) for v in values):
                                            return False
                                    elif match_type == "regex_search":
                                        if any(re.search(v, child_attr) for v in values):
                                            return False
                            # no forbidden child found => OK (do nothing)
                        else:
                            # if some other operator appears (shouldn't), fail safe
                            return False

                    elif key.startswith("window_base") or key.startswith("window_orth"):
                        # Parsowanie klucza
                        m = re.match(r'window_(base|orth)(?:\((\d+)\))?$', key)
                        if not m:
                            return False

                        w_type = m.group(1)  # "base" lub "orth"
                        dist = int(m.group(2)) if m.group(2) else 50

                        # Bezpieczne granice dla pętli
                        start_w = max(0, token_idx - dist)
                        end_w = min(num_tokens, token_idx + dist + 1)

                        found = False
                        for w_i in range(start_w, end_w):
                            if w_i == token_idx:
                                continue  # Nie sprawdzamy głównego tokenu

                            val = lemmas[w_i] if w_type == "base" else tokens[w_i]

                            if match_type == "exact":
                                if val in values:
                                    found = True
                                    break
                            elif match_type == "regex":
                                if any(re.fullmatch(v, val) for v in values):
                                    found = True
                                    break
                            elif match_type == "regex_search":
                                if any(re.search(v, val) for v in values):
                                    found = True
                                    break

                        # Logika przepuszczania / odrzucania
                        if operator == "=":
                            if not found:
                                return False
                            else:
                                continue

                        elif operator == "!=":
                            if found:
                                return False
                            else:
                                continue
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
                            elif match_type == "regex" and not any(re.fullmatch(v, token_feat) for v in values):
                                return False
                            elif match_type == "regex_search" and not any(re.search(v, token_feat) for v in values):
                                return False


                        elif operator == "!=":
                            if match_type == "exact" and token_feat in values:
                                return False
                            elif match_type == "regex" and any(re.fullmatch(v, token_feat) for v in values):
                                return False
                            elif match_type == "regex_search" and any(re.search(v, token_feat) for v in values):
                                return False

                return True

            def expand_mention(s_idx, e_limit, current_conds):
                current_cond_list = current_conds if isinstance(current_conds, list) else [current_conds]
                is_coref_m = False
                for c in current_cond_list:
                    if c and len(c) >= 1 and c[0] == "coref(M)":
                        is_coref_m = True
                        break

                if not is_coref_m:
                    return s_idx + 1  # Standardowy skok o 1 słowo

                n_idx = s_idx + 1
                c_tags = corefs[s_idx] if corefs is not None else []
                if isinstance(c_tags, str): c_tags = [c_tags]

                # Zbieramy ID klastra dla bieżącego słowa
                active_c_ids = {t.split("-")[-1] for t in c_tags if t not in ("0", "O", "_", None)}
                if not active_c_ids:
                    return n_idx

                # Pożeramy w prawo tak długo, jak długo kolejne słowa mają ten sam ID klastra
                while n_idx < e_limit:
                    next_tags = corefs[n_idx] if corefs is not None else []
                    if isinstance(next_tags, str): next_tags = [next_tags]
                    next_active = {t.split("-")[-1] for t in next_tags if t not in ("0", "O", "_", None)}

                    shared = active_c_ids.intersection(next_active)
                    if shared:
                        active_c_ids = shared
                        n_idx += 1
                    else:
                        break
                return n_idx

            def match_pattern(start_idx, cond_list):
                if not cond_list: return start_idx
                first = cond_list[0]

                if isinstance(first, tuple) and first and first[0] == "repeat":
                    base_cond = first[1]
                    min_rep = first[2]
                    max_rep = first[3]

                    for count in range(max_rep, min_rep - 1, -1):
                        new_idx, valid = start_idx, True
                        for _ in range(count):
                            if new_idx >= num_tokens:
                                valid = False; break
                            base_cond_list = base_cond if isinstance(base_cond, list) else [base_cond]
                            if not match_conditions(new_idx, base_cond_list):
                                valid = False; break
                            # --- Używamy ekspansji zamiast new_idx += 1 ---
                            new_idx = expand_mention(new_idx, num_tokens, base_cond_list)

                        if valid:
                            remainder = match_pattern(new_idx, cond_list[1:])
                            if remainder is not None:
                                return remainder
                    return None
                else:
                    first_cond_list = first if isinstance(first, list) else [first]
                    if start_idx >= num_tokens or not match_conditions(start_idx, first_cond_list):
                        return None
                    # --- Używamy ekspansji zamiast start_idx + 1 ---
                    new_idx = expand_mention(start_idx, num_tokens, first_cond_list)
                    return match_pattern(new_idx, cond_list[1:])

            def match_pattern_in_range(start_idx, cond_list, end_limit):
                if not cond_list: return start_idx
                first = cond_list[0]
                if isinstance(first, tuple) and first and first[0] == "repeat":
                    base_cond = first[1]
                    min_rep = first[2]
                    max_rep = first[3]

                    for count in range(max_rep, min_rep - 1, -1):
                        new_idx = start_idx
                        valid = True
                        for _ in range(count):
                            if new_idx >= end_limit:
                                valid = False; break
                            base_cond_list = base_cond if isinstance(base_cond, list) else [base_cond]
                            if not match_conditions(new_idx, base_cond_list):
                                valid = False; break
                            # --- Używamy ekspansji zamiast new_idx += 1 ---
                            new_idx = expand_mention(new_idx, end_limit, base_cond_list)
                        if valid:
                            remainder = match_pattern_in_range(new_idx, cond_list[1:], end_limit)
                            if remainder is not None:
                                return remainder
                    return None
                else:
                    first_cond_list = first if isinstance(first, list) else [first]
                    if start_idx >= end_limit or not match_conditions(start_idx, first_cond_list):
                        return None
                    # --- Używamy ekspansji zamiast start_idx + 1 ---
                    new_idx = expand_mention(start_idx, end_limit, first_cond_list)
                    return match_pattern_in_range(new_idx, cond_list[1:], end_limit)

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
                    if cond and cond[0] == "head.group":
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


            # --- End children.group processing ---

            i = 0
            while i < num_tokens:
                if s_ordered or sentence_query_conditions:
                    sent_start = i
                    while sent_start > 0 and sentence_ids[sent_start - 1] == sentence_ids[i]:
                        sent_start -= 1
                    sent_end = i
                    while sent_end < len(sentence_ids) and sentence_ids[sent_end] == sentence_ids[i]:
                        sent_end += 1

                    # check that the sentence also contains the `<s>` conditions
                    if sentence_query_conditions:
                        if s_ordered:
                            if not sentence_contains_conditions(sent_start, sent_end, sentence_query_conditions):
                                i += 1
                                continue
                        else:
                            if not sentence_matches(sent_start, sent_end, sentence_query_conditions):
                                i += 1
                                continue

                    end_idx = match_pattern_in_range(i, token_query_conditions, sent_end)
                else:
                    # no sentence restriction
                    end_idx = match_pattern(i, token_query_conditions)

                if end_idx is not None and end_idx > i:

                    left_context = row.Treść[
                        max(0, start_ids[max(0, i - left_context_size)]): start_ids[i]
                    ] if i > 0 else ""

                    matched_text = row.Treść[start_ids[i]: end_ids[end_idx - 1] + 1]

                    right_context = row.Treść[
                        end_ids[end_idx - 1] + 1: start_ids[
                            min(len(start_ids) - 1, end_idx - 1 + right_context_size + 1)]
                    ]

                    matched_lemmas = " ".join(lemmas[i:end_idx])

                    context = [left_context, matched_text, right_context]

                    full_left_context = row.Treść[
                        max(0, start_ids[max(0, i - kontekst)]): start_ids[i]
                    ] if i > 0 else ""
                    full_left_context = full_left_context[
                        :-len(left_context)] if left_context else full_left_context

                    full_right_context = row.Treść[
                        end_ids[end_idx - 1] + 1: start_ids[min(len(start_ids) - 1, end_idx - 1 + kontekst)]
                    ]
                    full_right_context = full_right_context[
                        len(right_context):] if right_context else full_right_context

                    full_text_with_markers = [full_left_context, matched_text, full_right_context]

                    token_counter[matched_text] += 1
                    lemma_counter[matched_lemmas] += 1

                    # --- ZMIANA NR 3: Korzystamy ze słowników (O(1)) zamiast .loc ---
                    if "Data publikacji" in df.columns:
                        raw_date = dates_dict.get(original_row_index, "")
                    else:
                        raw_date = ""
                    publication_date = raw_date.split(" ")[0] if isinstance(raw_date, str) else "Brak danych"

                    try:
                        if publication_date and publication_date != "Brak danych":
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

                    title = titles_dict.get(original_row_index, " ")
                    author = authors_dict.get(original_row_index, " ")

                    # Szybkie pobranie additional_metadata z wcześniej przygotowanego meta_dicts
                    additional_metadata = {
                        col: meta_dicts[col].get(original_row_index, " ")
                        for col in meta_dicts
                    }

                    temp_results.append(((matched_text, matched_lemmas),
                                         (publication_date, context, full_text_with_markers,
                                          matched_text, matched_lemmas,
                                          month_key, title, author, additional_metadata, left_context,
                                          right_context, row.Index, i, end_idx)))  # <-- Dodane 3 parametry

                    # --- SKOK ZA DOPASOWANIE ---
                    i = end_idx
                else:
                    # --- PRZESUNIĘCIE O 1, JEŚLI NIC NIE ZNALEZIONO ---
                    i += 1

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

# --- HISTORIA WYSZUKIWAŃ (Z CACHE WYNIKÓW) ---
search_history = []
MAX_HISTORY = 10

def add_to_history(state: SearchState):
    """Dodaje pełny stan wyszukiwania (zapytanie + wyliczone wyniki) do historii."""
    if not state.query or state.query.startswith('Podaj zapytanie np.:'):
        return

    global search_history
    # Usuń z historii duplikat (takie samo zapytanie i korpus), by zaktualizowany wynik wskoczył na górę
    search_history = [s for s in search_history if not (s.query == state.query and s.corpus == state.corpus)]
    search_history.append(state)

    if len(search_history) > MAX_HISTORY:
        search_history.pop(0)

    update_history_menu()

def update_history_menu():
    """Odświeża listę zapytań w zakładce Historia w górnym Menu."""
    if 'history_menu' not in globals():
        return

    history_menu.delete(0, tk.END)
    if not search_history:
        history_menu.add_command(label="Brak historii", state="disabled")
    else:
        for state in reversed(search_history):
            # Etykieta: "[KORPUS] fragment_zapytania..."
            q = state.query
            display_label = f"[{state.corpus}] {q[:45]}..." if len(q) > 45 else f"[{state.corpus}] {q}"
            history_menu.add_command(label=display_label, command=lambda st=state: restore_from_history(st))

        history_menu.add_separator()
        history_menu.add_command(label="Wyczyść historię", command=clear_history)

def clear_history():
    search_history.clear()
    update_history_menu()

def restore_from_history(state: SearchState):
    """Natychmiastowo ładuje zapytanie i pełne WYNIKI z cache'u, bez ponownego parsowania."""
    global current_state, global_query, global_selected_corpus, full_results_sorted
    global monthly_lemma_freq, monthly_freq_for_use, monthly_tfidf_for_use, monthly_zscore_for_use
    global fq_data, fq_data_token, fq_data_month, true_monthly_totals, lemma_df_cache
    global search_status

    # 1. Zaktualizuj UI zapytania
    entry_query.delete("1.0", ctk.END)
    entry_query.insert("1.0", state.query)
    corpus_var.set(state.corpus)
    highlight_entry()

    # 2. Błyskawiczne nadpisanie globalnego stanu pamięci RAM
    with state_lock:
        current_state = state
        global_query = state.query
        global_selected_corpus = state.corpus
        full_results_sorted = list(state.results)
        monthly_lemma_freq = dict(state.monthly_lemma_freq)
        monthly_freq_for_use = dict(state.monthly_freq_for_use)
        monthly_tfidf_for_use = dict(state.monthly_tfidf_for_use)
        monthly_zscore_for_use = dict(state.monthly_zscore_for_use)
        true_monthly_totals = dict(state.true_monthly_totals)
        lemma_df_cache = dict(state.lemma_df_cache)
        fq_data = list(state.fq_data)
        fq_data_token = list(state.fq_data_token)
        fq_data_month = list(state.fq_data_month)

    # 3. Natychmiastowe Renderowanie Paginacji Wyników (Z głównej tabeli)
    search_status = 0
    liczba = len(full_results_sorted)
    label_results_count.configure(text=f"Znaleziono trafień: {liczba:,}".replace(',', ' '))
    display_page(global_query, global_selected_corpus)

    # 4. Natychmiastowe Renderowanie Statystyk (Jeśli mają daty)
    if getattr(state, "has_dates", False):
        paginator_token["data"] = fq_data_token
        paginator_token["current_page"][0] = 0
        update_table(paginator_token)

        paginator_fq["data"] = fq_data
        paginator_fq["current_page"][0] = 0
        update_table(paginator_fq)

        paginator_month["data"] = fq_data_month
        paginator_month["current_page"][0] = 0
        update_table(paginator_month)

        # Usunięcie starych checkboxów wykresów po lewej
        for child in checkboxes_frame.winfo_children():
            child.destroy()

        lemma_vars.clear()
        merge_entry_vars.clear()

        # Odbudowanie widgetów do klikania wykresów (Kopia czystej logiki z search())
        def build_listbox_ui_local(parent_frame, sorted_lemma_freq, vars_dict, merge_dict, update_plot_callback, items_per_page=100):
            vars_dict.clear()
            merge_dict.clear()
            update_job = {"after_id": None}
            current_page_idx = {"idx": 0}
            local_state_data = {"data": sorted_lemma_freq}
            theme = THEMES[motyw.get()]

            for lemma, _ in sorted_lemma_freq:
                vars_dict[lemma] = ctk.BooleanVar(value=False)
                merge_dict[lemma] = ctk.StringVar(value=lemma)

            container = ctk.CTkFrame(parent_frame, fg_color=theme["frame_fg"], corner_radius=15)
            rename_entry = ctk.CTkEntry(container, placeholder_text="Nowa nazwa dla zaznaczonych", font=("JetBrains Mono", 12), fg_color=theme["subframe_fg"], corner_radius=8, height=35)
            rename_entry.pack(fill="x", padx=10, pady=(5, 5))
            rename_btn = ctk.CTkButton(container, text="Grupuj/Zmień nazwę", font=("Verdana", 12, 'bold'), fg_color=theme["button_fg"], hover_color=theme["button_hover"], text_color=theme["button_text"], corner_radius=8, height=35)
            rename_btn.pack(fill="x", padx=10, pady=(5, 10))

            nav_frame = ctk.CTkFrame(container, fg_color=theme["subframe_fg"], corner_radius=12)
            nav_frame.pack(fill="x", padx=10, pady=(0, 10))
            nav_frame.grid_columnconfigure(0, weight=0); nav_frame.grid_columnconfigure(1, weight=1); nav_frame.grid_columnconfigure(2, weight=0)

            prev_btn = ctk.CTkButton(nav_frame, text="<", width=40, height=35, fg_color=theme["button_fg"], hover_color=theme["button_hover"], text_color=theme["button_text"], corner_radius=8)
            prev_btn.grid(row=0, column=0, sticky="w", padx=5, pady=5)
            lbl_page = ctk.CTkLabel(nav_frame, text="1 / 1", font=("Verdana", 12, 'bold'), text_color=theme["label_text"])
            lbl_page.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
            next_btn = ctk.CTkButton(nav_frame, text=">", width=40, height=35, fg_color=theme["button_fg"], hover_color=theme["button_hover"], text_color=theme["button_text"], corner_radius=8)
            next_btn.grid(row=0, column=2, sticky="e", padx=5, pady=5)

            listbox_frame = ctk.CTkScrollableFrame(container, fg_color=theme["subframe_fg"], corner_radius=8, height=300)
            listbox_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

            def delayed_update():
                if update_job["after_id"]: container.after_cancel(update_job["after_id"])
                update_job["after_id"] = container.after(300, update_plot_callback)

            def show_page(page_idx):
                try:
                    if not listbox_frame.winfo_exists(): return
                except: return
                for widget in listbox_frame.winfo_children(): widget.destroy()

                current_data = local_state_data["data"]
                total_pages = max(1, math.ceil(len(current_data) / items_per_page))
                if page_idx >= total_pages: page_idx = max(0, total_pages - 1)

                start = page_idx * items_per_page
                end = min(start + items_per_page, len(current_data))
                for lemma, score in current_data[start:end]:
                    display_name = merge_dict[lemma].get()
                    score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                    cb = ctk.CTkCheckBox(listbox_frame, text=f"{display_name} ({score_str})", variable=vars_dict[lemma], command=delayed_update, font=("Verdana", 12), fg_color=theme["button_fg"], hover_color=theme["button_hover"], text_color=theme["label_text"])
                    cb.pack(anchor="w", pady=4, padx=5)

                lbl_page.configure(text=f"{page_idx + 1} / {total_pages}")
                current_page_idx["idx"] = page_idx

            def rename_selected():
                new_text = rename_entry.get().strip()
                if not new_text: return
                renamed_any = False
                for lemma, _ in local_state_data["data"]:
                    if vars_dict[lemma].get():
                        merge_dict[lemma].set(new_text)
                        renamed_any = True
                if renamed_any:
                    show_page(current_page_idx["idx"])
                    delayed_update()

            rename_btn.configure(command=rename_selected)
            prev_btn.configure(command=lambda: show_page(current_page_idx["idx"] - 1) if current_page_idx["idx"] > 0 else None)
            next_btn.configure(command=lambda: show_page(current_page_idx["idx"] + 1) if current_page_idx["idx"] < max(1, math.ceil(len(local_state_data["data"]) / items_per_page)) - 1 else None)

            def set_data(new_sorted_data):
                local_state_data["data"] = new_sorted_data
                show_page(0)

            show_page(0)
            return container, set_data

        container_listbox, set_data_listbox = build_listbox_ui_local(
            checkboxes_frame, state.s_lemma_total_freq, lemma_vars, merge_entry_vars, update_plot
        )
        container_listbox.pack(fill="both", expand=True)

        def toggle_listboxes(*args):
            mode = wykres_sort_mode.get()
            def get_max_scores(freq_dict):
                scores = {}
                for lemma in state.unique_lemmas:
                    max_val = max((freq_dict[m].get(lemma, 0) for m in freq_dict), default=0)
                    scores[lemma] = max_val
                return sorted(scores.items(), key=lambda x: x[1], reverse=True)

            if mode == "TF-IDF":
                set_data_listbox(state.s_lemma_global_tfidf)
            elif mode == "Z-score":
                set_data_listbox(get_max_scores(state.monthly_zscore_for_use))
            elif mode == "Częstość względna":
                set_data_listbox(state.s_lemma_global_pmw)
            else:
                set_data_listbox(state.s_lemma_total_freq)

        # Podmiana eventów na nowy lokalny toggle_listboxes
        for trace_id in wykres_sort_mode.trace_info():
            wykres_sort_mode.trace_remove(*trace_id[0:2])
        wykres_sort_mode.trace_add("write", toggle_listboxes)
        toggle_listboxes()

        # Odbudowanie wykresów z uwzględnieniem danych ze zbuforowanego stanu!
        force_recalculate_plot()
    else:
        # Wyczyść listboxy jeśli brak dat w wybranym korpusie
        for child in checkboxes_frame.winfo_children():
            child.destroy()


def log_exception(context: str, exc: Exception, user_message: str = None):
    logging.error("%s: %s\n%s", context, exc, traceback.format_exc())
    if user_message:
        try:
            messagebox.showerror("Błąd", user_message)
        except Exception:
            # ostateczny fallback - nie blokuj aplikacji, jeśli messagebox też zawiedzie
            logging.error("Nie udało się pokazać messagebox dla błędu: %s", context)

def add_warning(warnings_list, msg):
    if warnings_list is None:
        return
    if msg not in warnings_list:
        warnings_list.append(msg)
        logging.warning(msg)

def show_search_error(msg: str):
    global search_status
    search_status = 0
    label_results_count.configure(text="")
    text_result.set_data([("", "Błąd zapytania", msg, "")])
    text_result.set_fulltext_data([])
    page_label.configure(text="0/0")
    button_first.configure(state="disabled")
    button_prev.configure(state="disabled")
    button_next.configure(state="disabled")
    button_last.configure(state="disabled")


def show_search_warnings(warnings_list):
    if 'warning_label' not in globals():
        return
    if not warnings_list:
        warning_label.configure(text="")
        # Chowamy etykietę i odzyskujemy miejsce!
        warning_label.pack_forget()
        return

    warning_label.configure(text=" | ".join(warnings_list[:3]))
    # Pokazujemy etykietę (wymuszając jej pozycję tuż nad paned_window)
    warning_label.pack(fill="x", padx=10, pady=(0, 5), before=paned_window)

def check_brackets(query):
    stack = []
    in_single_quote = False
    in_double_quote = False

    for char in query:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char in ('[', '{') and not in_single_quote and not in_double_quote:
            stack.append(char)
        elif char in (']', '}') and not in_single_quote and not in_double_quote:
            if not stack:
                return False  # Zamknięto nawias, którego nie było
            last = stack.pop()
            if char == ']' and last != '[': return False
            if char == '}' and last != '{': return False

    return len(stack) == 0

def validate_query_for_ui(query: str):
    """
    Lekka, praktyczna walidacja zapytania przed uruchomieniem wyszukiwania.
    Nie zastępuje parsera, ale wyłapuje błędy składniowe i literówki w nazwach atrybutów.
    """
    if not query or not query.strip():
        raise QueryValidationError("Zapytanie jest puste.")

    if not check_brackets(query):
        raise QueryValidationError("Niezgodna liczba nawiasów kwadratowych lub błędne cudzysłowy w zapytaniu.")

    # --- Zbiór dozwolonych atrybutów dla tokenów ---
    VALID_KEYS = {
        "orth", "base", "pos", "upos", "deprel", "ner",
        "number", "case", "gender", "degree", "person", "aspect", "negation",
        "accentability", "post-prepositionality", "accommodability",
        "vocalicity", "agglutination", "fullstoppedness"
    }

    def validate_conditions(conds):
        for cond in conds:
            if not cond: continue
            if isinstance(cond, list):
                validate_conditions(cond)
            elif isinstance(cond, tuple):
                if cond[0] == "repeat":
                    # Sprawdzanie warunków zagnieżdżonych w operatorze powtórzeń np. [1,3]
                    if len(cond) > 1 and isinstance(cond[1], (list, tuple)):
                        validate_conditions(cond[1] if isinstance(cond[1], list) else [cond[1]])
                else:
                    key = cond[0]
                    # Usuwamy ewentualne parametry w nawiasach (np. head(3) -> head)
                    base_key = key.split("(")[0].split(".")[0]

                    # Jeśli klucz nie jest na liście i nie jest atrybutem relacyjnym - wyrzuć błąd!
                    if base_key not in VALID_KEYS and not base_key.startswith(
                            ("coref", "head", "dependent", "window_base", "window_orth")):
                        raise QueryValidationError(
                            f"Nieznany atrybut w zapytaniu: '{key}'")

                    # Walidacja głębszych zagnieżdżeń (np. dla dependent={...})
                    if len(cond) >= 4 and cond[3]:
                        validate_conditions(cond[1])

    # Dzielimy zapytanie na grupy (obsługa operatora LUB '||')
    raw_groups = query.split("||")
    query_groups = []

    for g in raw_groups:
        cleaned = g.strip()
        if not cleaned:
            raise QueryValidationError(
                "Wykryto pustą grupę zapytania.")

        query_groups.append(cleaned)

    for idx, group in enumerate(query_groups, start=1):
        try:
            if "<s" in group:
                token_part, sentence_part = group.split("<s", 1)

                token_conds = parse_query_group(token_part)
                if token_conds is None:
                    raise QueryValidationError(f"Błąd składni w grupie {idx}: niepoprawna część tokenowa.")
                validate_conditions(token_conds)  # Wywołanie naszej nowej weryfikacji słownikiem

                sentence_part = sentence_part.strip()
                if sentence_part.endswith(">"):
                    sentence_part = sentence_part[:-1].strip()

                s_ordered, sentence_conds = parse_sentence_conditions(sentence_part)
                if sentence_conds is None:
                    raise QueryValidationError(f"Błąd składni w grupie {idx}: niepoprawna część <s>.")
                validate_conditions(sentence_conds)
            else:
                token_conds = parse_query_group(group)
                if token_conds is None:
                    raise QueryValidationError(f"Błąd składni w grupie {idx}: {group}")

                # --- KLUCZOWA POPRAWKA ---
                if not token_conds:
                    raise QueryValidationError(
                        f"Grupa {idx} ('{group}') nie zawiera żadnego zdefiniowanego segmentu w nawiasach kwadratowych.")
                validate_conditions(token_conds)
        except QueryValidationError:
            raise
        except Exception as e:
            raise QueryValidationError(f"Nie udało się przeanalizować grupy {idx}: {e}")


# Funkcja obsługująca wyszukiwanie

def search():
    theme = THEMES[motyw.get()]
    global search_status, precalculated_bins
    global search_in_progress, active_search_token, last_search_error, last_search_warnings

    with search_guard:
        if search_in_progress:
            return
        search_in_progress = True
        active_search_token += 1
        local_token = active_search_token

    search_status = 1
    last_search_error = ""
    last_search_warnings = []
    precalculated_bins = []

    # --- POBRANIE ZAPYTANIA (BEZ ZAPISU DO HISTORII) ---
    current_query = entry_query.get("1.0", ctk.END).strip()

    # --- POBRANIE STANU GUI DO ZMIENNYCH LOKALNYCH (ZAMROŻENIE W GŁÓWNYM WĄTKU) ---
    try:
        left_ctx_val = int(entry_left_context.get() or "10")
        right_ctx_val = int(entry_right_context.get() or "10")
    except ValueError:
        left_ctx_val, right_ctx_val = 10, 10

    # Zamrażamy wszystkie wybory, których wątek roboczy potrzebuje
    gui_state = {
        "query": current_query,
        "selected_corpus": corpus_var.get(),
        "left_context": left_ctx_val,
        "right_context": right_ctx_val,
        "sort_option": sort_option_var.get(),
        "is_plotting": plotting.get(),
        "plot_style": styl_wykresow.get()
    }

    # Clear the checkboxes and result text widget.
    for frame in [checkboxes_frame]:
        for child in frame.winfo_children():
            child.destroy()
    checkboxes_frame.update_idletasks()

    button_search.configure(state="disabled")

    if 'paginator_colloc' in globals():
        paginator_colloc["data"] = []
        paginator_colloc["current_page"][0] = 0
        update_table(paginator_colloc)

    # Wstrzykujemy stan GUI jako drugi argument do funkcji
    def search_thread(search_token, ui_state):

        try:
            logging.info("Search started in thread: %s [token=%s]", threading.current_thread().name, search_token)

            # Wyciągamy wartości BEZPIECZNIE ze słownika, zamiast z GUI!
            query = ui_state["query"]
            validate_query_for_ui(query)

            left_context_size = ui_state["left_context"]
            right_context_size = ui_state["right_context"]
            selected_corpus = ui_state["selected_corpus"]
            sort_option = ui_state["sort_option"]

            df = dataframes[selected_corpus]

            warnings_list = []

            # Lokalny, kompletny stan jednego przebiegu:
            local_state = SearchState()
            local_state.query = query
            local_state.corpus = selected_corpus
            # Przekazujemy selected_corpus zgodnie z nową definicją z Kroku 2!
            results = find_lemma_context(
                query,
                df,
                selected_corpus,
                left_context_size,
                right_context_size,
                warnings_list=warnings_list
            )

            # Jeśli to już nie jest aktualne wyszukiwanie, niczego nie nadpisuj.
            if search_token != active_search_token:
                logging.info("Discarding stale search results [token=%s]", search_token)
                return

            global last_search_warnings
            last_search_warnings = warnings_list

            if not results:
                logging.info("No results found [token=%s]", search_token)
            else:
                logging.info("Number of results: %s [token=%s]", len(results), search_token)

            # ======================================================================================
            # --- ZMIANA NR 2: MATEMATYKA ZOSTUJE W TLE, DO GUI PRZEKAZUJEMY TYLKO GOTOWE WYNIKI ---
            # ======================================================================================

            global search_status, monthly_lemma_freq, lemma_vars, merge_entry_vars
            global true_monthly_totals, monthly_freq_for_use, monthly_tfidf_for_use, monthly_zscore_for_use
            global fq_data, full_results_sorted, current_page, global_query, global_selected_corpus, lemma_df_cache

            monthly_tfidf_for_use = {}
            monthly_zscore_for_use = {}
            fq_data = []
            fq_data_token = []
            fq_data_month = []

            def first_real_token(text):
                if not text: return ""
                for tok in text.split():
                    cleaned = tok.strip(string.punctuation).lower()
                    if cleaned: return cleaned
                return ""

            def last_real_token(text):
                if not text: return ""
                for tok in reversed(text.split()):
                    cleaned = tok.strip(string.punctuation).lower()
                    if cleaned: return cleaned
                return ""

            # Pobranie opcji przed rozpoczęciem tła
            sort_option = ui_state["sort_option"]

            # --- SORTOWANIE (WYKONUJE SIĘ W TLE) ---
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
                results_sorted = results

            if results_sorted:
                # wyniki najpierw w lokalnym stanie:
                local_state.results = results_sorted

                # atomowa podmiana stanu i TYLKO wtedy aktualizacja GUI
                with state_lock:
                    globals()['current_state'] = local_state
                    # Kompatybilność z istniejącym GUI (czytającym ze "starych" globali):
                    globals()['global_query'] = local_state.query
                    globals()['global_selected_corpus'] = local_state.corpus
                    globals()['full_results_sorted'] = list(local_state.results)

                auto_fill_dates(results_sorted)
                search_status = 0
                text_result.after(300, lambda: display_page(local_state.query, local_state.corpus))

                # --- AGREGACJA STATYSTYK (WYKONUJE SIĘ W TLE) ---
                if "Data publikacji" in df.columns:


                    unique_matched_tokens = {}
                    unique_lemmas = set()
                    monthly_lemma_freq.clear()

                    exact_orth_df = {}
                    exact_lemma_df = {}

                    for publication_date, context, full_text, matched_text, matched_lemmas, month_key, title, author, additional_metadata, left_context, right_context, row_idx, start_idx_val, end_idx_val in results_sorted:
                        exact_orth_df.setdefault(matched_text, set()).add(row_idx)
                        exact_lemma_df.setdefault(matched_lemmas, set()).add(row_idx)

                        token_key = matched_text
                        unique_matched_tokens[token_key] = unique_matched_tokens.get(token_key, 0) + 1
                        unique_lemmas.add(matched_lemmas)

                        try:
                            year, month_val = month_key.split('-')
                            normalized_key = f"{year}-{int(month_val)}"
                        except Exception:
                            normalized_key = month_key

                        if normalized_key not in monthly_lemma_freq:
                            monthly_lemma_freq[normalized_key] = {}
                        monthly_lemma_freq[normalized_key][matched_lemmas] = monthly_lemma_freq[normalized_key].get(
                            matched_lemmas, 0) + 1

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
                        current_date = start_date
                        while current_date <= end_date:
                            key = f"{current_date.year}-{current_date.month}"
                            if key not in monthly_lemma_freq:
                                monthly_lemma_freq[key] = {lemma: 0 for lemma in unique_lemmas}
                            current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)

                    def update_data_tables():
                        global true_monthly_totals
                        true_monthly_totals.clear()
                        raw_monthly_counts = inverted_indexes[global_selected_corpus].get("monthly_token_counts", {})
                        flattened = []
                        for year, months in raw_monthly_counts.items():
                            for month, count in months.items():
                                key = f"{year}-{int(month)}"
                                true_monthly_totals[key] = true_monthly_totals.get(key, 0) + count
                                flattened.append((year, month, count))
                        return flattened

                    update_data_tables()
                    total_token_count = sum(true_monthly_totals.values())
                    total_docs = len(df)

                    # Teraz pobieramy dane z całego korpusu (indeksu), więc TF-IDF będzie rzetelny
                    lemma_df_cache = {
                        lemma: len(inverted_indexes[global_selected_corpus]["base"].get(lemma, set()))
                        for lemma in unique_lemmas
                    }


                    for idx, (token, frequency) in enumerate(
                            sorted(unique_matched_tokens.items(), key=lambda x: x[1], reverse=True), start=1):
                        frequency_normalized = (
                            (frequency / total_token_count) * 1_000_000 if total_token_count > 0 else 0.0)
                        tf_global = (frequency / total_token_count) if total_token_count > 0 else 0
                        global_docs_set_orth = inverted_indexes[global_selected_corpus]["orth"].get(token, set())
                        df_val_orth = len(global_docs_set_orth) if global_docs_set_orth else 1
                        idf_global = math.log10(total_docs / df_val_orth) if df_val_orth > 0 else 0
                        global_tfidf_orth = tf_global * idf_global * 100000
                        fq_data_token.append([idx, token, frequency, round(frequency_normalized, 2), df_val_orth,
                                              round(global_tfidf_orth, 2)])

                    lemma_total_freq = {}
                    for month_data in monthly_lemma_freq.values():
                        for lemma, count in month_data.items():
                            lemma_total_freq[lemma] = lemma_total_freq.get(lemma, 0) + count

                    s_lemma_total_freq = sorted(lemma_total_freq.items(), key=lambda x: x[1], reverse=True)

                    for idx, (lemma, frequency) in enumerate(s_lemma_total_freq, start=1):
                        frequency_normalized = (
                            (frequency / total_token_count) * 1_000_000 if total_token_count > 0 else 0.0)
                        tf_global = (frequency / total_token_count) if total_token_count > 0 else 0
                        global_docs_set_lemma = inverted_indexes[global_selected_corpus]["base"].get(lemma, set())
                        df_val_lemma = len(global_docs_set_lemma) if global_docs_set_lemma else 1
                        idf_global = math.log10(total_docs / df_val_lemma) if df_val_lemma > 0 else 0
                        global_tfidf_lemma = tf_global * idf_global * 100000
                        fq_data.append([idx, lemma, frequency, round(frequency_normalized, 2), df_val_lemma,
                                        round(global_tfidf_lemma, 2)])
                    s_lemma_global_pmw = sorted([(r[1], r[3]) for r in fq_data], key=lambda x: x[1], reverse=True)
                    s_lemma_global_tfidf = sorted([(r[1], r[5]) for r in fq_data], key=lambda x: x[1], reverse=True)


                    local_state.monthly_lemma_freq = dict(monthly_lemma_freq)
                    monthly_freq_for_use = {}
                    for month_key, lemma_counts in monthly_lemma_freq.items():
                        total = true_monthly_totals.get(month_key, 0)
                        if total > 0:
                            monthly_freq_for_use[month_key] = {lemma: (count / total) * 1_000_000 for lemma, count in
                                                               lemma_counts.items()}
                        else:
                            monthly_freq_for_use[month_key] = {lemma: 0.0 for lemma in lemma_counts}

                    # Szybki bufor z prawdziwym (globalnym) DF dla unikalnych lematów
                    global_lemma_df_cache = {}
                    for lemma in unique_lemmas:
                        global_docs_set = inverted_indexes[global_selected_corpus]["base"].get(lemma, set())
                        global_lemma_df_cache[lemma] = len(global_docs_set) if global_docs_set else 1

                    for month_key, lemma_counts in monthly_lemma_freq.items():
                        total = true_monthly_totals.get(month_key, 0)
                        monthly_tfidf_for_use[month_key] = {}
                        for lemma, count in lemma_counts.items():
                            tf = (count / total) if total > 0 else 0

                            # Używamy globalnego DF zamiast lokalnego (błędnego) cache'u
                            df_val = global_lemma_df_cache.get(lemma, 1)

                            idf = math.log10(total_docs / df_val) if df_val > 0 else 0
                            monthly_tfidf_for_use[month_key][lemma] = tf * idf * 100000

                    lemma_norm_values = {lemma: [] for lemma in unique_lemmas}
                    for month_key in monthly_lemma_freq.keys():
                        for lemma in unique_lemmas:
                            lemma_norm_values[lemma].append(monthly_freq_for_use[month_key].get(lemma, 0.0))

                    lemma_stats = {}
                    for lemma, vals in lemma_norm_values.items():
                        mean_val = np.mean(vals) if vals else 0.0
                        std_val = np.std(vals) if vals else 0.0
                        lemma_stats[lemma] = (mean_val, std_val)

                    for month_key in monthly_lemma_freq.keys():
                        monthly_zscore_for_use[month_key] = {}
                        for lemma in unique_lemmas:
                            val = monthly_freq_for_use[month_key].get(lemma, 0.0)
                            mean_val, std_val = lemma_stats[lemma]
                            z = calc_z_score(val, mean_val, std_val)
                            monthly_zscore_for_use[month_key][lemma] = z

                    sorted_month_keys = sorted(monthly_lemma_freq.keys(),
                                               key=lambda k: (int(k.split('-')[0]), int(k.split('-')[1])))
                    for month_key in sorted_month_keys:
                        year_str, month_str = month_key.split('-')
                        raw_counts = monthly_lemma_freq[month_key]
                        norm_counts = monthly_freq_for_use[month_key]
                        for lemma in sorted(raw_counts.keys()):
                            raw = raw_counts[lemma]
                            norm = norm_counts.get(lemma, 0.0)
                            tfidf = monthly_tfidf_for_use[month_key].get(lemma, 0.0)
                            zscore = monthly_zscore_for_use[month_key].get(lemma, 0.0)
                            fq_data_month.append([
                                int(year_str),
                                int(month_str),
                                lemma,
                                raw,
                                round(norm, 2) if isinstance(norm, (int, float)) else None,
                                round(tfidf, 2) if isinstance(tfidf, (int, float)) else None,
                                round(zscore, 2) if isinstance(zscore, (int, float)) else None
                            ])

                    if ui_state["is_plotting"] == 'Tak':
                        # Wykres również generowany w tle - dzięki API obiektowemu Matplotlib
                        plot_stack = get_plot_stack()
                        Figure = plot_stack["Figure"]
                        FigureCanvasAgg = plot_stack["FigureCanvasAgg"]
                        yearly_grouped = {}
                        for key, data_ in monthly_freq_for_use.items():
                            year, month = key.split('-')
                            if year == '0000' or month == '0': continue
                            yearly_grouped.setdefault(year, {})
                            for lemma, val in data_.items():
                                yearly_grouped[year][lemma] = yearly_grouped[year].get(lemma, 0) + val
                        keys = sorted(yearly_grouped.keys(), key=int)
                        x_labels = keys
                        x = np.arange(len(keys))

                        fig = Figure(figsize=(12, 7), dpi=100)
                        ax = fig.add_subplot(111)

                        if ui_state["plot_style"] == "ciemny":
                            fig.patch.set_facecolor('#2C2F33')
                            ax.set_facecolor('#2C2F33')
                            ax.tick_params(colors='white')
                            ax.xaxis.label.set_color('white')
                            ax.yaxis.label.set_color('white')
                            for spine in ax.spines.values(): spine.set_edgecolor('white')
                        else:
                            fig.patch.set_facecolor('white')
                            ax.set_facecolor('white')
                            ax.tick_params(colors='black')

                        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.2)
                        ax.set_xlabel('Rok')
                        ax.set_ylabel('Frekwencja')

                        max_labels = 24
                        n_labels = len(x_labels)
                        step = int(np.ceil(n_labels / max_labels)) if n_labels > max_labels else 1
                        labeled_idx = set([0, n_labels - 1] + list(range(0, n_labels, step)))
                        labels = [lbl if i in labeled_idx else "" for i, lbl in enumerate(x_labels)]

                        ax.set_xticks(x)
                        ax.set_xticklabels(labels, rotation=0, ha='right')
                        if len(x) > 0: ax.set_xlim(x[0] - 1, x[-1] + 1)
                        for tick, label in zip(ax.xaxis.get_major_ticks(), labels):
                            size = 3 if label == "" else 7
                            tick.tick1line.set_markersize(size)
                            tick.tick2line.set_markersize(size)

                        ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
                        fig.tight_layout(rect=[0, 0, 1, 0.85])
                        os.makedirs('temp', exist_ok=True)
                        canvas = FigureCanvasAgg(fig)
                        fig.savefig('temp/temp_plot.png', bbox_inches='tight')

                # =========================================================================
                # --- AKTUALIZACJA GUI (WYKONUJE SIĘ BEZPIECZNIE W GŁÓWNYM WĄTKU) ---
                # =========================================================================
                def update_gui():
                    global search_status
                    search_status = 0
                    liczba = len(full_results_sorted)
                    label_results_count.configure(text=f"Znaleziono trafień: {liczba:,}".replace(',', ' '))
                    display_page(global_query, global_selected_corpus)

                    if "Data publikacji" in df.columns:
                        # Tabele
                        paginator_token["data"] = fq_data_token
                        update_table(paginator_token)
                        frekw_dane_tabela_orth.set_data(fq_data_token[:15])

                        paginator_fq["data"] = fq_data
                        update_table(paginator_fq)
                        frekw_dane_tabela.set_data(fq_data[:15])

                        paginator_month["data"] = fq_data_month
                        update_table(paginator_month)
                        frekw_dane_tabela_month.set_data(fq_data_month[:15])

                        if ui_state["is_plotting"] == 'Tak':
                            update_plot_images()



                        for child in checkboxes_frame.winfo_children():
                            child.destroy()

                        lemma_vars.clear()
                        merge_entry_vars.clear()

                        def build_listbox_ui(parent_frame, sorted_lemma_freq, vars_dict, merge_dict,
                                             update_plot_callback, items_per_page=100):
                            vars_dict.clear()
                            merge_dict.clear()
                            update_job = {"after_id": None}
                            current_page_idx = {"idx": 0}
                            state = {"data": sorted_lemma_freq}
                            theme = THEMES[motyw.get()]

                            for lemma, _ in sorted_lemma_freq:
                                vars_dict[lemma] = ctk.BooleanVar(value=False)
                                merge_dict[lemma] = ctk.StringVar(value=lemma)

                            container = ctk.CTkFrame(parent_frame, fg_color=theme["frame_fg"], corner_radius=15)
                            rename_entry = ctk.CTkEntry(container, placeholder_text="Nowa nazwa dla zaznaczonych",
                                                        font=("JetBrains Mono", 12), fg_color=theme["subframe_fg"],
                                                        corner_radius=8, height=35)
                            rename_entry.pack(fill="x", padx=10, pady=(5, 5))
                            rename_btn = ctk.CTkButton(container, text="Grupuj/Zmień nazwę",
                                                       font=("Verdana", 12, 'bold'), fg_color=theme["button_fg"],
                                                       hover_color=theme["button_hover"],
                                                       text_color=theme["button_text"], corner_radius=8, height=35)
                            rename_btn.pack(fill="x", padx=10, pady=(5, 10))

                            nav_frame = ctk.CTkFrame(container, fg_color=theme["subframe_fg"], corner_radius=12)
                            nav_frame.pack(fill="x", padx=10, pady=(0, 10))
                            nav_frame.grid_columnconfigure(0, weight=0);
                            nav_frame.grid_columnconfigure(1, weight=1);
                            nav_frame.grid_columnconfigure(2, weight=0)

                            prev_btn = ctk.CTkButton(nav_frame, text="<", width=40, height=35,
                                                     fg_color=theme["button_fg"], hover_color=theme["button_hover"],
                                                     text_color=theme["button_text"], corner_radius=8)
                            prev_btn.grid(row=0, column=0, sticky="w", padx=5, pady=5)
                            lbl_page = ctk.CTkLabel(nav_frame, text="1 / 1", font=("Verdana", 12, 'bold'),
                                                    text_color=theme["label_text"])
                            lbl_page.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
                            next_btn = ctk.CTkButton(nav_frame, text=">", width=40, height=35,
                                                     fg_color=theme["button_fg"], hover_color=theme["button_hover"],
                                                     text_color=theme["button_text"], corner_radius=8)
                            next_btn.grid(row=0, column=2, sticky="e", padx=5, pady=5)

                            listbox_frame = ctk.CTkScrollableFrame(container, fg_color=theme["subframe_fg"],
                                                                   corner_radius=8, height=300)
                            listbox_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

                            def delayed_update():
                                if update_job["after_id"]: container.after_cancel(update_job["after_id"])
                                update_job["after_id"] = container.after(300, update_plot_callback)

                            def show_page(page_idx):
                                try:
                                    if not listbox_frame.winfo_exists(): return
                                except:
                                    return
                                for widget in listbox_frame.winfo_children(): widget.destroy()

                                current_data = state["data"]
                                total_pages = max(1, math.ceil(len(current_data) / items_per_page))
                                if page_idx >= total_pages: page_idx = total_pages - 1

                                start = page_idx * items_per_page
                                end = min(start + items_per_page, len(current_data))
                                for lemma, score in current_data[start:end]:
                                    display_name = merge_dict[lemma].get()
                                    score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                                    cb = ctk.CTkCheckBox(listbox_frame, text=f"{display_name} ({score_str})",
                                                         variable=vars_dict[lemma], command=delayed_update,
                                                         font=("Verdana", 12), fg_color=theme["button_fg"],
                                                         hover_color=theme["button_hover"],
                                                         text_color=theme["label_text"])
                                    cb.pack(anchor="w", pady=4, padx=5)

                                lbl_page.configure(text=f"{page_idx + 1} / {total_pages}")
                                current_page_idx["idx"] = page_idx

                            def rename_selected():
                                new_text = rename_entry.get().strip()
                                if not new_text: return
                                renamed_any = False
                                for lemma, _ in state["data"]:
                                    if vars_dict[lemma].get():
                                        merge_dict[lemma].set(new_text)
                                        renamed_any = True
                                if renamed_any:
                                    show_page(current_page_idx["idx"])
                                    delayed_update()

                            rename_btn.configure(command=rename_selected)
                            prev_btn.configure(
                                command=lambda: show_page(current_page_idx["idx"] - 1) if current_page_idx[
                                                                                              "idx"] > 0 else None)
                            next_btn.configure(
                                command=lambda: show_page(current_page_idx["idx"] + 1) if current_page_idx["idx"] < max(
                                    1, math.ceil(len(state["data"]) / items_per_page)) - 1 else None)

                            def set_data(new_sorted_data):
                                state["data"] = new_sorted_data
                                show_page(0)

                            show_page(0)
                            return container, listbox_frame, rename_entry, set_data

                        container_listbox, listbox_frame, rename_entry, set_data_listbox = build_listbox_ui(
                            checkboxes_frame, s_lemma_total_freq, lemma_vars, merge_entry_vars, update_plot
                        )
                        container_listbox.pack(fill="both", expand=True)

                        def toggle_listboxes(*args):
                            mode = wykres_sort_mode.get()

                            def get_max_scores(freq_dict):
                                scores = {}
                                for lemma in unique_lemmas:
                                    max_val = max((freq_dict[m].get(lemma, 0) for m in freq_dict), default=0)
                                    scores[lemma] = max_val
                                return sorted(scores.items(), key=lambda x: x[1], reverse=True)

                            # Teraz używamy miar globalnych przygotowanych wyżej
                            if mode == "TF-IDF":
                                set_data_listbox(s_lemma_global_tfidf)
                            elif mode == "Z-score":
                                # Dla Z-score zostawiamy Max, bo miara globalna dla Z-score nie istnieje
                                set_data_listbox(get_max_scores(monthly_zscore_for_use))
                            elif mode == "Częstość względna":
                                set_data_listbox(s_lemma_global_pmw)
                            else:
                                set_data_listbox(s_lemma_total_freq)



                        for trace_id in wykres_sort_mode.trace_info():
                            wykres_sort_mode.trace_remove(*trace_id[0:2])
                        wykres_sort_mode.trace_add("write", toggle_listboxes)
                        toggle_listboxes()

                        # Zapisz wartości do lokalnego stanu
                        local_state.monthly_freq_for_use = dict(monthly_freq_for_use)
                        local_state.monthly_tfidf_for_use = dict(monthly_tfidf_for_use)
                        local_state.monthly_zscore_for_use = dict(monthly_zscore_for_use)

                        # --- NOWE: DODANIE DO STANU STATYSTYK FREKWENCYJNYCH ---
                        local_state.has_dates = True
                        local_state.fq_data = fq_data
                        local_state.fq_data_token = fq_data_token
                        local_state.fq_data_month = fq_data_month
                        local_state.s_lemma_total_freq = s_lemma_total_freq
                        local_state.s_lemma_global_pmw = s_lemma_global_pmw
                        local_state.s_lemma_global_tfidf = s_lemma_global_tfidf
                        local_state.unique_lemmas = unique_lemmas
                        local_state.true_monthly_totals = true_monthly_totals
                        local_state.lemma_df_cache = lemma_df_cache

                        # Atomowo ustanów najnowszy stan i zasil istniejące globalne aliasy używane w GUI:
                        with state_lock:
                            globals()['current_state'] = local_state
                            globals()['global_query'] = local_state.query
                            globals()['global_selected_corpus'] = local_state.corpus
                            globals()['full_results_sorted'] = list(local_state.results)
                            globals()['monthly_lemma_freq'] = dict(local_state.monthly_lemma_freq)
                            globals()['monthly_freq_for_use'] = dict(local_state.monthly_freq_for_use)
                            globals()['monthly_tfidf_for_use'] = dict(local_state.monthly_tfidf_for_use)
                            globals()['monthly_zscore_for_use'] = dict(local_state.monthly_zscore_for_use)

                app.after(0, lambda: show_search_warnings(last_search_warnings))

                # --- NOWE: ZAPIS DO HISTORII CAŁEGO STANU NA SAM KONIEC ---
                app.after(0, lambda: add_to_history(local_state))

                # Delegowanie pracy z UI do głównego wątku
                app.after(0, update_gui)


            else:
                # Brak wyników
                def update_no_results():
                    global search_status, full_results_sorted
                    full_results_sorted = []
                    search_status = 0
                    label_results_count.configure(text="Znaleziono trafień: 0")
                    display_page(query, selected_corpus)

                app.after(0, lambda: show_search_warnings(last_search_warnings))
                app.after(0, update_no_results)



        except (QueryValidationError, QueryParseError) as e:

            logging.warning("Validation or Parse error in search thread [token=%s]: %s", search_token, e)

            if search_token == active_search_token:
                # 1. Zapisujemy treść błędu do bezpiecznej zmiennej tekstowej
                error_msg = str(e)
                label_results_count.after(0, lambda: label_results_count.configure(text=""))

                # 2. Przekazujemy zmienną do lambdy przez domyślny argument (msg=error_msg)
                app.after(0, lambda msg=error_msg: show_search_error(msg))

        except Exception as e:

            logging.exception("Error in search thread [token=%s]", search_token)

            if search_token == active_search_token:
                # To samo tutaj - zapisujemy sformatowany tekst przed wrzuceniem do lambdy
                error_msg = f"Nie udało się wykonać wyszukiwania.\nSzczegóły: {e}"
                label_results_count.after(0, lambda: label_results_count.configure(text=""))

                app.after(0, lambda msg=error_msg: show_search_error(msg))

        finally:

            with search_guard:
                global search_in_progress
                if search_token == active_search_token:
                    search_in_progress = False
            if search_token == active_search_token:
                app.after(0, lambda: button_search.configure(state="normal"))

    print(f"Search function called in thread: {threading.current_thread().name}")

    thread = threading.Thread(target=search_thread, args=(local_token, gui_state), daemon=True)
    thread.start()

    print("Search thread started.")


def parse_date_safe(s):
    """Parsuje daty wpisane przez użytkownika."""
    if not s or not isinstance(s, str):
        return None

    formats = ("%d-%m-%Y", "%d.%m.%Y", "%m-%Y", "%m.%Y", "%Y-%m-%d", "%Y-%m", "%Y")
    for fmt in formats:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue

    logging.info("parse_date_safe: nie rozpoznano formatu daty: %r", s)
    return None


def auto_fill_dates(results):
    """Automatycznie uzupełnia pola dat na podstawie znalezionych wyników."""
    if not results: return
    dates = []
    for res in results:
        d = parse_date_safe(res[0])
        if d: dates.append(d)
    if dates:
        min_date, max_date = min(dates), max(dates)

        # Zapisz aktualny stan i tymczasowo odblokuj, żeby móc wpisać
        current_state = date_start_entry.cget("state")
        date_start_entry.configure(state="normal")
        date_end_entry.configure(state="normal")

        date_start_entry.delete(0, 'end')
        date_start_entry.insert(0, min_date.strftime("%d-%m-%Y"))
        date_end_entry.delete(0, 'end')
        date_end_entry.insert(0, max_date.strftime("%d-%m-%Y"))

        # Przywróć poprzedni stan
        date_start_entry.configure(state=current_state)
        date_end_entry.configure(state=current_state)


def get_effective_total_n(bin_start, bin_end, monthly_totals):
    """Oblicza sumę wagową tokenów dla zadanego przedziału dat."""
    effective_n = 0
    curr = bin_start
    while curr < bin_end:
        last_day = calendar.monthrange(curr.year, curr.month)[1]
        month_end = datetime(curr.year, curr.month, last_day)
        month_next = month_end + timedelta(days=1)
        overlap_start = max(curr, bin_start)
        overlap_end = min(month_next, bin_end)

        if overlap_start < overlap_end:
            days_in_overlap = (overlap_end - overlap_start).days
            m_key = f"{curr.year}-{curr.month}"
            total_m = monthly_totals.get(m_key, 0)
            effective_n += (days_in_overlap / last_day) * total_m
        curr = month_next
    return effective_n


# --- GLOBALE DLA CACHE'OWANIA WYKRESÓW ---
precalculated_bins = []
precalculated_bin_totals = []
precalculated_lemma_counts = {}


def calculate_bins():
    global precalculated_bins, precalculated_bin_totals, precalculated_lemma_counts
    global full_results_sorted, true_monthly_totals

    precalculated_bins, precalculated_bin_totals, precalculated_lemma_counts = [], [], {}
    if not full_results_sorted: return

    try:
        multiplier = int(interval_mult_entry.get())
    except:
        multiplier = 1
    unit = interval_unit_var.get()

    results_dates = []
    for res in full_results_sorted:
        d = parse_date_safe(res[0])
        if d: results_dates.append(d)
    if not results_dates: return

    # Bierzemy daty z pól TYLKO jeśli checkbox jest zaznaczony
    if custom_date_var.get():
        u_start = parse_date_safe(date_start_entry.get())
        u_end = parse_date_safe(date_end_entry.get())
    else:
        u_start = None
        u_end = None

    corpus_dates = []
    for k in true_monthly_totals.keys():
        try:
            y, m = map(int, k.split('-'))
            corpus_dates.append(datetime(y, m, 1))
        except:
            pass

    if corpus_dates:
        corpus_min = min(corpus_dates)
        corpus_max_month = max(corpus_dates)
        last_day = calendar.monthrange(corpus_max_month.year, corpus_max_month.month)[1]
        corpus_max = datetime(corpus_max_month.year, corpus_max_month.month, last_day)
    else:
        corpus_min, corpus_max = min(results_dates), max(results_dates)

    start_dt = u_start if u_start else min(corpus_min, min(results_dates))
    end_dt = u_end if u_end else max(corpus_max, max(results_dates))

    if not u_start and unit in ["Miesiąc", "Rok"]:
        start_dt = start_dt.replace(day=1)

    curr = start_dt
    limit = end_dt + timedelta(days=1)

    while curr < limit:
        if unit == "Dzień":
            nxt = curr + timedelta(days=multiplier)
        elif unit == "Miesiąc":
            nxt = curr + relativedelta(months=multiplier)
        else:
            nxt = curr + relativedelta(years=multiplier)
        precalculated_bins.append((curr, nxt))
        curr = nxt

    precalculated_bin_totals = [get_effective_total_n(b[0], b[1], true_monthly_totals) for b in precalculated_bins]

    unique_lemmas = set(res[4] for res in full_results_sorted)
    for lemma in unique_lemmas:
        precalculated_lemma_counts[lemma] = [0] * len(precalculated_bins)

    for res in full_results_sorted:
        res_dt = parse_date_safe(res[0])
        res_lemma = res[4]
        if not res_dt: continue
        for i, (b_s, b_e) in enumerate(precalculated_bins):
            if b_s <= res_dt < b_e:
                precalculated_lemma_counts[res_lemma][i] += 1
                break


def force_recalculate_plot(*args):
    calculate_bins()
    update_plot()
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


def update_plot():
    global full_results_sorted, true_monthly_totals, lemma_vars, merge_entry_vars, lemma_df_cache, global_selected_corpus
    global precalculated_bins, precalculated_bin_totals, precalculated_lemma_counts

    plot_stack = get_plot_stack()
    Figure = plot_stack["Figure"]
    FigureCanvasAgg = plot_stack["FigureCanvasAgg"]
    cm = plot_stack["cm"]

    # 1. Obiektowa funkcja do rysowania pustego ekranu
    def draw_empty(message):
        fig = Figure(figsize=(12, 7), dpi=100)
        ax = fig.add_subplot(111)

        is_dark = styl_wykresow.get() == "ciemny"
        bg_color = '#2C2F33' if is_dark else 'white'
        text_color = 'white' if is_dark else 'black'

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16, alpha=0.5, color=text_color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        os.makedirs('temp', exist_ok=True)
        canvas = FigureCanvasAgg(fig)
        fig.savefig('temp/temp_plot.png', bbox_inches='tight')
        update_plot_images()

    # Weryfikacja danych
    if not full_results_sorted:
        draw_empty("Brak wyników wyszukiwania dla podanego zapytania.")
        return

    if not precalculated_bins:
        calculate_bins()
    if not precalculated_bins:
        draw_empty("Brak danych w wybranym przedziale czasowym.")
        return

    mode = wykres_sort_mode.get()
    unit = interval_unit_var.get()

    groups = {}
    for lemma, var in lemma_vars.items():
        if var.get():
            g_name = merge_entry_vars[lemma].get() or lemma
            groups.setdefault(g_name, []).append(lemma)

    if not groups:
        draw_empty("Zaznacz elementy na liście poniżej, aby narysować wykres.")
        return

    num_bins = len(precalculated_bins)
    plot_data_raw = {g: [0] * num_bins for g in groups}

    for g_name, lems in groups.items():
        for lemma in lems:
            if lemma in precalculated_lemma_counts:
                for i in range(num_bins):
                    plot_data_raw[g_name][i] += precalculated_lemma_counts[lemma][i]

    total_docs = len(dataframes[global_selected_corpus])

    # 2. KONFIGURACJA PŁÓTNA - CZYSTE API OBIEKTOWE (Z kopii 4)
    fig = Figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)

    is_dark = styl_wykresow.get() == "ciemny"
    if is_dark:
        fig.patch.set_facecolor('#2C2F33')
        ax.set_facecolor('#2C2F33')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('white')
        text_color = 'white'
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        text_color = 'black'

    x_indices = range(num_bins)
    x_labels = [b[0].strftime("%d.%m.%Y") for b in precalculated_bins]

    if mode == "Częstość względna":
        ylabel = "Częstość względna (PMW)"
    elif mode == "TF-IDF":
        ylabel = "TF-IDF (Ważony)"
    elif mode == "Z-score":
        ylabel = "Z-score (Dynamika zmian)"
    else:
        ylabel = "Liczba wystąpień"

    colors = cm.tab20.colors  # Paleta kolorów uratowana z kopii 4!

    for idx, (g_name, raw_values) in enumerate(plot_data_raw.items()):
        pmw_values = [(v / (precalculated_bin_totals[i] / 1e6)) if precalculated_bin_totals[i] > 0 else 0 for i, v in
                      enumerate(raw_values)]

        if mode == "Częstość względna":
            final_vals = pmw_values
        elif mode == "TF-IDF":
            total_idf = sum(math.log10(total_docs / lemma_df_cache.get(l, 1)) for l in groups[g_name])
            avg_idf = total_idf / len(groups[g_name]) if groups[g_name] else 0
            final_vals = []
            for i, v in enumerate(raw_values):
                tf = (v / precalculated_bin_totals[i]) if precalculated_bin_totals[i] > 0 else 0
                final_vals.append(tf * avg_idf * 100000)
        elif mode == "Z-score":
            mean_v = np.mean(pmw_values)
            std_v = np.std(pmw_values)
            final_vals = [(v - mean_v) / std_v if std_v > 0 else 0 for v in pmw_values]
        else:
            final_vals = raw_values

        ax.plot(x_indices, final_vals, marker='o', label=g_name, color=colors[idx % len(colors)])

    max_labels = 24
    n_labels = len(x_labels)
    step = int(np.ceil(n_labels / max_labels)) if n_labels > max_labels else 1

    labeled_idx = set([0, n_labels - 1] + list(range(0, n_labels, step)))
    final_labels = [lbl if i in labeled_idx else "" for i, lbl in enumerate(x_labels)]

    ax.set_xticks(list(x_indices))
    ax.set_xticklabels(final_labels, rotation=45, ha='right')
    if len(x_indices) > 0:
        ax.set_xlim(x_indices[0] - 1, x_indices[-1] + 1)

    for tick, label in zip(ax.xaxis.get_major_ticks(), final_labels):
        size = 3 if label == "" else 7
        tick.tick1line.set_markersize(size)
        tick.tick2line.set_markersize(size)

    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.2)

    # Przeniesione idealne układanie legendy z kopii 4
    ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.32), frameon=False, labelcolor=text_color)

    fig.tight_layout(rect=[0, 0, 1, 0.85])

    # Bezpieczny zapis z użyciem FigureCanvasAgg
    os.makedirs('temp', exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    fig.savefig('temp/temp_plot.png', bbox_inches='tight')
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
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać wykresu.\nSzczegóły: {e}")


def show_dependency_graph():
    global current_graph_row_idx, current_graph_start_idx, global_selected_corpus
    if current_graph_row_idx is None or current_graph_start_idx is None:
        return

    plot_stack = get_plot_stack()
    Figure = plot_stack["Figure"]
    FigureCanvasAgg = plot_stack["FigureCanvasAgg"]
    plt = plot_stack["plt"]

    # Pobranie danych o zdaniu
    df = dataframes[global_selected_corpus]
    row_data = df.loc[current_graph_row_idx]

    sentence_ids = row_data.sentence_ids
    word_ids = row_data.word_ids
    head_ids = row_data.head_ids
    tokens = row_data.tokens
    upostags = row_data.upostags
    deprels = row_data.deprels

    token_idx = current_graph_start_idx
    sent_id = sentence_ids[token_idx]

    # Znalezienie granic zdania
    start = token_idx
    while start > 0 and sentence_ids[start - 1] == sent_id:
        start -= 1
    end = token_idx
    while end < len(sentence_ids) and sentence_ids[end] == sent_id:
        end += 1

    długość_zdania = end - start

    # Pobranie aktualnego motywu
    theme = THEMES.get(motyw.get(), THEMES["jasny"])
    bg_color = theme["app_bg"]
    text_color = theme["label_text"]
    line_color = theme["button_fg"]
    tag_color = theme["highlight"]
    label_bg = theme["subframe_fg"]

    # 1. ZBIERANIE I PRZETWARZANIE KRAWĘDZI
    edges = []
    roots = []

    for i in range(start, end):
        head = head_ids[i]
        dep_idx = i - start

        if head == 0:
            roots.append({'dep': dep_idx, 'label': deprels[i]})
            continue

        head_idx = None
        for j in range(start, end):
            if word_ids[j] == head:
                head_idx = j - start
                break

        if head_idx is not None:
            left = min(head_idx, dep_idx)
            right = max(head_idx, dep_idx)
            edges.append({
                'head': head_idx,
                'dep': dep_idx,
                'left': left,
                'right': right,
                'dist': right - left,
                'label': deprels[i]
            })

    edges.sort(key=lambda e: e['dist'])

    levels = []
    for edge in edges:
        left, right = edge['left'], edge['right']
        placed = False

        for level_idx, level_spans in enumerate(levels):
            overlap = False
            for s_left, s_right in level_spans:
                # Kolizja zachodzi, gdy odcinki nachodzą na siebie (minimalny bufor)
                if max(left, s_left) < min(right, s_right):
                    overlap = True
                    break

            if not overlap:
                level_spans.append((left, right))
                edge['level'] = level_idx
                placed = True
                break

        if not placed:
            levels.append([(left, right)])
            edge['level'] = len(levels) - 1

    num_levels = len(levels)

    # 2. USTAWIENIA RYSOWANIA
    base_h = 0.8
    step_h = 0.65
    max_height = base_h + (num_levels * step_h)
    if roots:
        max_height += step_h

    # Obliczamy "fizyczny" rozmiar płótna Matplotlib.
    # Mnożnik * 1.2 wymusza odpowiednio szeroki margines na każde słowo!
    width_in_inches = max(12, długość_zdania * 1.2)
    height_in_inches = max(5, max_height + 1.0)

    # 3. UTWORZENIE OKNA Z PASKAMI PRZEWIJANIA (SCROLLBAR)
    graph_win = ctk.CTkToplevel(app)
    graph_win.title("Graf zależności (Składnia)")
    graph_win.geometry("1100x650")  # Startowy rozmiar okna aplikacji
    graph_win.grab_set()

    # Główny kontener
    container = ctk.CTkFrame(graph_win, fg_color=bg_color)
    container.pack(fill="both", expand=True, padx=10, pady=10)

    # Płótno Tkintera (które połyka za duże elementy)
    canvas_tk = tk.Canvas(container, bg=bg_color, highlightthickness=0)

    # Paski nawigacyjne w stylu CustomTkinter
    hbar = ctk.CTkScrollbar(container, orientation="horizontal", command=canvas_tk.xview)
    vbar = ctk.CTkScrollbar(container, orientation="vertical", command=canvas_tk.yview)

    canvas_tk.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

    # Układ kontenera
    hbar.pack(side="bottom", fill="x")
    vbar.pack(side="right", fill="y")
    canvas_tk.pack(side="left", fill="both", expand=True)

    # Wewnętrzna ramka siedząca wewnątrz Canvasu (na nią wgramy Matplotlib)
    inner_frame = ctk.CTkFrame(canvas_tk, fg_color=bg_color)
    canvas_window = canvas_tk.create_window((0, 0), window=inner_frame, anchor="nw")

    def configure_scrollregion(event):
        canvas_tk.configure(scrollregion=canvas_tk.bbox("all"))

    inner_frame.bind("<Configure>", configure_scrollregion)

    # 4. RYSOWANIE MATPLOTLIB
    fig = Figure(figsize=(width_in_inches, height_in_inches), dpi=100)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')

    # Słowa
    for i in range(start, end):
        x = i - start
        word = tokens[i]
        tag = upostags[i]

        ax.text(x, 0, word, ha='center', va='bottom', fontsize=12, fontweight='bold', color=text_color, zorder=3)
        ax.text(x, -0.3, tag, ha='center', va='top', fontsize=10, color=tag_color, zorder=3)

    # Krawędzie
    for edge in edges:
        head_idx = edge['head']
        dep_idx = edge['dep']

        h = base_h + (edge['level'] * step_h)

        ax.plot([head_idx, head_idx], [0.3, h], color=line_color, lw=1.5, zorder=1)
        ax.plot([head_idx, dep_idx], [h, h], color=line_color, lw=1.5, zorder=1)

        ax.annotate("", xy=(dep_idx, 0.3), xytext=(dep_idx, h),
                    arrowprops=dict(arrowstyle="->", color=line_color, lw=1.5), zorder=1)

        mid_x = (head_idx + dep_idx) / 2
        ax.text(mid_x, h, edge['label'], ha='center', va='center', fontsize=9, color=text_color,
                bbox=dict(boxstyle="round,pad=0.2", fc=label_bg, ec=line_color, lw=1, alpha=1.0),
                zorder=2)

    # Root
    root_h = max_height
    for r in roots:
        dep_idx = r['dep']
        ax.annotate("", xy=(dep_idx, 0.3), xytext=(dep_idx, root_h),
                    arrowprops=dict(arrowstyle="->", color=tag_color, lw=2.0), zorder=1)
        ax.text(dep_idx, root_h, r['label'], ha='center', va='center', fontsize=10, fontweight='bold', color=text_color,
                bbox=dict(boxstyle="round,pad=0.3", fc=label_bg, ec=tag_color, lw=1.5, alpha=1.0),
                zorder=2)

    # Skalowanie obszaru rysowania
    ax.set_xlim(-0.5, długość_zdania - 0.5)
    ax.set_ylim(-1, max_height + 0.5)

    plt.tight_layout()

    # 5. OSADZENIE W INTERFEJSIE
    # 5. OSADZENIE W INTERFEJSIE (Wersja stabilna - renderowanie do obrazu)
    os.makedirs('temp', exist_ok=True)
    temp_graph_path = "temp/temp_graph.png"

    # Renderujemy graf do pliku tymczasowego
    canvas_render = FigureCanvasAgg(fig)
    fig.savefig(temp_graph_path, bbox_inches='tight', facecolor=fig.get_facecolor())

    # Wczytujemy jako CTkImage, aby zachować skalowanie DPI
    img_pil = Image.open(temp_graph_path)

    # Obliczamy rozmiar (możesz dostosować mnożnik 0.8 dla wielkości grafu)
    display_w = int(img_pil.width)
    display_h = int(img_pil.height)

    graph_img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(display_w, display_h))

    # Wyświetlamy w labelu wewnątrz przewijanej ramy
    graph_label = ctk.CTkLabel(inner_frame, image=graph_img_ctk, text="")
    graph_label.pack(fill="both", expand=True)

    # Sprzątanie pamięci
    plt.close(fig)

    # Mouse wheel binding (opcjonalnie do przewijania pionowego/poziomego myszką)
    def _on_mousewheel(event):
        canvas_tk.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shiftmouse(event):
        canvas_tk.xview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas_tk.bind_all("<MouseWheel>", _on_mousewheel)
    canvas_tk.bind_all("<Shift-MouseWheel>", _on_shiftmouse)


# --- NOWE ZMIENNE GLOBALNE DLA HIGHLIGHTINGU ---
show_ner_active = False
show_coref_active = False
current_display_row_idx = None
current_display_start_idx = None
current_body_start_mark = None

def toggle_ner():
    global show_ner_active
    show_ner_active = not show_ner_active
    if show_ner_active:
        button_toggle_ner.configure(fg_color="#4E8752", hover_color="#57965C")  # Zielony aktywny
    else:
        button_toggle_ner.configure(fg_color="#4B6CB7", hover_color="#5B7CD9")  # Domyślny niebieski
    update_highlights()

def toggle_coref():
    global show_coref_active
    show_coref_active = not show_coref_active
    if show_coref_active:
        button_toggle_coref.configure(fg_color="#4E8752", hover_color="#57965C") # Zielony aktywny
    else:
        button_toggle_coref.configure(fg_color="#4B6CB7", hover_color="#5B7CD9") # Domyślny niebieski
    update_highlights()


def update_highlights():
    if current_display_row_idx is None:
        return

    # Usunięcie starych tagów NER i Coref
    for tag in text_full.tag_names():
        if tag.startswith("ner_") or tag.startswith("coref_"):
            text_full.tag_remove(tag, "1.0", ctk.END)

    if not show_ner_active and not show_coref_active:
        return

    df = dataframes[global_selected_corpus]
    row_data = df.loc[current_display_row_idx]
    start_ids = row_data.start_ids
    end_ids = row_data.end_ids
    ners = row_data.ners
    corefs = getattr(row_data, "corefs", None)

    # Funkcja dobierająca kolor w zależności od motywu
    def get_color(text, is_ner=True):
        current_theme = motyw.get()  # Pobieramy aktualny motyw ("ciemny" lub "jasny")

        if current_theme == "ciemny":
            colors_ner = ["#FF9D9A", "#9DFF9A", "#9A9DFF", "#FF9AFF", "#FFFF9A", "#9AFFFF", "#FFC99A", "#C99AFF",
                          "#FFA07A", "#20B2AA", "#778899", "#9370DB"]
            colors_coref = ["#FFB347", "#84B6F4", "#FDCAE1", "#CFCFC4", "#B0E0E6", "#FDFD96", "#FF6961", "#77DD77",
                            "#F08080", "#E6E6FA", "#DDA0DD", "#40E0D0", "#FFDAB9", "#98FB98", "#AFEEEE", "#DB7093",
                            "#F0E68C", "#E0FFFF"]
        else:
            colors_ner = ["#B30000", "#006600", "#0000B3", "#800080", "#B35900", "#008080", "#D2143A", "#5900B3",
                          "#8B0000", "#556B2F", "#2F4F4F", "#483D8B"]
            colors_coref = ["#A0522D", "#2F4F4F", "#483D8B", "#8B0000", "#556B2F", "#008B8B", "#8B4513", "#4B0082",
                            "#B22222", "#006400", "#8B008B", "#D2691E", "#0000CD", "#228B22", "#4682B4", "#C71585",
                            "#DAA520", "#1E90FF"]

        palette = colors_ner if is_ner else colors_coref
        idx = sum(ord(c) for c in str(text)) % len(palette)
        return palette[idx]

    global_start_token_idx = max(0, current_display_start_idx - kontekst)
    global_start_char = max(0, start_ids[global_start_token_idx]) if current_display_start_idx > 0 else 0

    for k in range(len(start_ids)):
        token_start = start_ids[k]
        token_end = end_ids[k] + 1
        if token_end < global_start_char:
            continue

        offset_start = token_start - global_start_char
        offset_end = token_end - global_start_char
        idx_start = f"{current_body_start_mark} + {offset_start}c"
        idx_end = f"{current_body_start_mark} + {offset_end}c"

        # Kolorowanie NER (Zmiana koloru tekstu + Podkreślenie)
        if show_ner_active and ners[k] not in ("0", "O", "_", None):
            ner_type = ners[k].split("-")[-1] if "-" in ners[k] else ners[k]
            tag_name = f"ner_{ner_type}"
            text_full.tag_add(tag_name, idx_start, idx_end)
            text_full.tag_config(
                tag_name,
                foreground=get_color(ner_type, is_ner=True),
                underline=True
            )

        # Kolorowanie Klastrów Koreferencyjnych (Zmiana koloru tekstu + Podkreślenie)
        if show_coref_active and corefs is not None:
            c_tags = corefs[k]
            if isinstance(c_tags, str): c_tags = [c_tags]

            for c_tag in c_tags:
                if c_tag not in ("0", "O", "_", None):
                    cluster_id = c_tag.split("-")[-1] if "-" in c_tag else c_tag
                    tag_name = f"coref_{cluster_id}"
                    text_full.tag_add(tag_name, idx_start, idx_end)
                    text_full.tag_config(
                        tag_name,
                        foreground=get_color(cluster_id, is_ner=False),
                        underline=True
                    )

def display_full_text(full_text, result, publication_date, title, author, additional_metadata, row_idx=None, start_idx=None):
    global current_graph_row_idx, current_graph_start_idx
    global current_display_row_idx, current_display_start_idx, current_body_start_mark

    current_graph_row_idx = row_idx
    current_graph_start_idx = start_idx
    current_display_row_idx = row_idx
    current_display_start_idx = start_idx

    text_full.delete("1.0", ctk.END)

    text_full.insert(ctk.END,
                     f'Data publikacji: {publication_date}, Tytuł: {title}, Autor: {author}')
    if additional_metadata:
        joined_meta = ', '.join(f'{key}: {value}' for key, value in additional_metadata.items())
        extra_fields = f', {joined_meta}'
        text_full.insert(ctk.END, extra_fields)

    text_full.insert(ctk.END, "\n\n")
    text_full.tag_add("text_style", "1.0", ctk.END)

    # -------------------------------------------------------------
    # WAŻNE: pobieramy znacznik początku właściwego tekstu korpusu
    current_body_start_mark = text_full.index("end-1c")
    # -------------------------------------------------------------

    text_full.insert(ctk.END, full_text[0].replace("\r", ""), "text_style")
    text_full.insert(ctk.END, result[0].replace("\r", ""), "highlight")
    highlight_index = text_full.index(ctk.END)
    text_full.insert(ctk.END, result[1].replace("\r", ""), "highlight_keyword")
    text_full.insert(ctk.END, result[2].replace("\r", ""), "highlight")
    text_full.insert(ctk.END, full_text[2].replace("\r", ""), "text_style")

    # Konfiguracja tagów podstawowych
    text_full.tag_config("highlight", foreground=highlight_color, lmargin1=50, lmargin2=50, rmargin=50)

    text_full.tag_config("highlight_keyword", foreground=highlight_keyword, lmargin1=50, lmargin2=50, rmargin=50)

    text_full.tag_config("text_style", lmargin1=50, lmargin2=50, rmargin=50)

    text_full.see(highlight_index)

    # Aktywacja wszystkich trzech przycisków
    button_draw_graph.configure(state="normal")
    button_toggle_ner.configure(state="normal")
    button_toggle_coref.configure(state="normal")

    # Odświeżenie kolorów na nowym tekście
    update_highlights()

# Function to highlight the specified elements
def highlight_entry(event=None):


    # Reset all tags
    for tag in entry_query.tag_names():
        if tag != "sel":  # don't remove selection
            entry_query.tag_remove(tag, "1.0", ctk.END)

    # --- Highlight keywords first ---
    keywords = ["orth=", "orth!=", "base=", "base!=", "pos=", "pos!=", "upos=", "upos!=" "ner=", "ner!=", "head=", "head!=", "coref=", "coref!=",
                "head=", "head!=", "dependent=", "dependent!=", "deprel=", "deprel!=", "number=", "number!=", "ner=", "ner!="
                "window_base=", "window_base!=", "window_orth=", "window_orth!=",
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

    # Highlight dynamic keys like children(...) and parent(...) with operators
    new_dynamic_keys = ["head(", "dependent(", "window_base(", "window_orth(", "coref("]

    for term in new_dynamic_keys:
        start_idx = "1.0"
        while True:
            start_idx = entry_query.search(term, start_idx, ctk.END)
            if not start_idx:
                break

            # Find the closing parenthesis
            close_idx = entry_query.search(")", start_idx, ctk.END)
            if not close_idx:
                close_idx = f"{start_idx} + {len(term)}c"  # fallback
            else:
                close_idx = f"{close_idx} + 1c"

            # Include operator = or != immediately after the closing parenthesis
            operator_match = entry_query.search("!=|=", close_idx, ctk.END, regexp=True)
            if operator_match:
                # Extend close_idx to include operator
                op_end = f"{operator_match} + {2 if entry_query.get(operator_match, f'{operator_match} + 2c') == '!=' else 1}c"
                close_idx = op_end

            entry_query.tag_add(term, start_idx, close_idx)
            start_idx = close_idx

        entry_query.tag_config(term, foreground=keywords_color)

    # Highlight dynamic keys like children(...) and parent(...)
    for term in new_dynamic_keys:
        start_idx = "1.0"
        while True:
            start_idx = entry_query.search(term, start_idx, ctk.END)
            if not start_idx:
                break

            # Find the closing bracket or next operator
            # Look for ')' after start_idx
            close_idx = entry_query.search(")", start_idx, ctk.END)
            if not close_idx:
                close_idx = f"{start_idx} + {len(term)}c"  # fallback
            else:
                close_idx = f"{close_idx} + 1c"

            entry_query.tag_add(term, start_idx, close_idx)
            start_idx = close_idx

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


    # --- Kolorowanie ról koreferencyjnych (H), (P) i (M) ---
    role_tags = {"(H)": "#D400FF", "(P)": "#00D4FF",
                 "(M)": "#FFD400"}  # Fioletowy (Head), Turkusowy (Part), Żółty (Mention)

    for role, color in role_tags.items():
        start_idx = "1.0"
        while True:
            start_idx = entry_query.search(role, start_idx, ctk.END)
            if not start_idx:
                break
            end_idx = f"{start_idx} + {len(role)}c"

            # Tworzymy unikalny tag dla roli
            tag_name = f"role_{role}"
            entry_query.tag_add(tag_name, start_idx, end_idx)
            entry_query.tag_config(tag_name, foreground=color)

            start_idx = end_idx


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


# --- Słowniki pomocnicze dla UI ---
UPOS_DICT = [
    "Wszystkie", "ADJ (przymiotnik)", "ADP (przyimek)", "ADV (przysłówek)", "AUX (czas. posiłkowy)",
    "CCONJ (spójnik współrzędny)", "DET (określnik)", "INTJ (wykrzyknik)", "NOUN (rzeczownik)",
    "NUM (liczebnik)", "PART (partykuła)", "PRON (zaimek)", "PROPN (nazwa własna)",
    "PUNCT (interpunkcja)", "SCONJ (spójnik podrzędny)", "SYM (symbol)", "VERB (czasownik)", "X (inne)"
]

POS_NKJP_DICT = [
    "Wszystkie", "subst (rzeczownik)", "depr (rzecz. deprecjatywny)", "adj (przymiotnik)",
    "adja (przymiotnik przyprzym.)", "adjp (przymiotnik poprzyimkowy)", "adjc (przymiotnik predykatywny)",
    "conj (spójnik współrzędny)", "comp (spójnik podrzędny)", "ppron12 (zaimek os. 1/2)",
    "ppron3 (zaimek os. 3)", "siebie (zaimek SIEBIE)", "num (liczebnik główny)",
    "numcol (liczebnik zbiorowy)", "fin (czasownik - f. nieprzeszła)", "bedzie (czas. być - f. przyszła)",
    "aglt (aglutynant BYĆ)", "praet (pseudoimiesłów)", "impt (rozkaźnik)", "imps (bezosobnik)",
    "inf (bezokolicznik)", "pcon (im. przys. współczesny)", "pant (im. przys. uprzedni)",
    "ger (odsłownik)", "pact (im. przym. czynny)", "ppas (im. przym. bierny)",
    "winien (czas. winien)", "adv (przysłówek)", "prep (przyimek)",
    "qub (partykuła)", "interj (wykrzyknik)", "brev (skrót)", "burk (burkinostka)", "interp (interpunkcja)",
    "xxx (obce/nieznane)", "ign (ignorowany)"
]

# Drzewiasta struktura depreli (Ujednolicona terminologia)
DEPREL_TREE = [
    "Wszystkie",
    "root - głowa drzewa",
    "nsubj - podmiot rzeczowny",
    "  ├─ nsubj:pass - podmiot (str. bierna)",
    "csubj - podmiot zdaniowy",
    "  ├─ csubj:pass - podmiot zdaniowy (str. bierna)",
    "obj - dopełnienie bliższe",
    "ioobj - dopełnienie dalsze",
    "xcomp - dopełnienie otwarte / orzecznik",
    "  ├─ xcomp:pred - dopełnienie orzecznikowe",
    "  ├─ xcomp:obj - dopełnienie bezokolicznikowe",
    "  ├─ xcomp:subj - podmiot bezokolicznikowy",
    "ccomp - dopełnienie zdaniowe",
    "  ├─ ccomp:obj - dopełnienie zdaniowe czasownika",
    "amod - modyfikator przymiotnikowy",
    "  ├─ amod:flat - przymiotnikowa nazwa własna",
    "nmod - modyfikator rzeczowny",
    "  ├─ nmod:arg - wymagany zależnik rzeczowny",
    "  ├─ nmod:poss - modyfikator dzierżawczy",
    "nummod - modyfikator liczebnikowy",
    "  ├─ nummod:gov - liczebnik (rządzący)",
    "det - określnik",
    "acl - modyfikator zdaniowy (klauza)",
    "  ├─ acl:relcl - zdanie względne",
    "advmod - modyfikator przysłówkowy",
    "  ├─ advmod:emph - partykuła wzmacniająca",
    "  ├─ advmod:neg - partykuła przecząca",
    "advcl - zdanie okolicznikowe",
    "obl - argument ukośny (przyimkowy)",
    "  ├─ obl:arg - argument przyimkowy czasownika",
    "aux - czasownik posiłkowy",
    "  ├─ aux:pass - czasownik posiłkowy (str. bierna)",
    "cop - łącznik",
    "case - wskaźnik przypadka (przyimek)",
    "mark - wskaźnik zespolenia",
    "cc - spójnik współrzędny",
    "conj - element dołączony spójnikiem"
]

# --- Słowniki jednostek nazwanych (NER) ---
NER_PREFIXES = [
    "Brak (SpaCy / Dowolny)",
    "B- (początek - Stanza)",
    "I- (wnętrze - Stanza)",
    "S- (pojedynczy - Stanza)"
]

NER_TYPES = [
    "persName - osoba",
    "orgName - organizacja / instytucja",
    "geogName - obiekt geograficzny",
    "placeName - miejsce",
    "date - data",
    "time - czas",
    "O - poza jednostką nazwaną"
]


class RegexHelperWindow(ctk.CTkToplevel):
    def __init__(self, parent, target_entry, theme):
        super().__init__(parent)
        self.target_entry = target_entry
        self.title("Pomocnik Regex")
        self.geometry("380x450")
        self.configure(fg_color=theme["app_bg"])
        self.attributes("-topmost", True)  # Zawsze na wierzchu kreatora

        lbl = ctk.CTkLabel(self, text="Kliknij symbol, aby wstawić do pola:", font=("Verdana", 12, "bold"),
                           text_color=theme["label_text"])
        lbl.pack(pady=10, padx=10, anchor="w")

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Słownik wyrażeń regex na podstawie Twojej listy
        regex_items = [
            ("?", "Zero lub jedno wystąpienie"),
            (".", "Dowolny pojedynczy znak"),
            ("[a-z]", "Dowolna mała litera"),
            ("[A-Z]", "Dowolna wielka litera"),
            ("[A-Za-z]", "Dowolna litera"),
            ("\\d", "Dowolna cyfra (0-9)"),
            ("\\w", "Znak alfanumeryczny"),
            ("\\s", "Biały znak (spacja, tab)"),
            ("*", "Zero lub więcej wystąpień"),
            ("+", "Jedno lub więcej wystąpień"),
            ("|", "Alternatywa (LUB)"),
            ("()", "Podgrupa (wstawia kursor do środka)"),
            ("\\", "Traktuj kolejny znak dosłownie")
        ]

        for symbol, desc in regex_items:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)

            btn = ctk.CTkButton(
                row, text=symbol, width=70, font=("JetBrains Mono", 12, "bold"),
                fg_color=theme["button_fg"], hover_color=theme["button_hover"],
                command=lambda s=symbol: self.insert_regex(s)
            )
            btn.pack(side="left", padx=(0, 10))

            lbl_desc = ctk.CTkLabel(row, text=desc, text_color=theme["label_text"], font=("Verdana", 11))
            lbl_desc.pack(side="left")

    def insert_regex(self, text):
        # Pobieranie aktualnej pozycji kursora
        idx = self.target_entry.index(tk.INSERT)
        self.target_entry.insert(idx, text)

        # Jeśli to nawiasy, wstaw kursor w środek
        if text == "()":
            self.target_entry.icursor(idx + 1)

        self.target_entry.focus()


# --- Słowniki cech morfologicznych NKJP ---
MORPH_DICTS = {
    "case": ["nom (mianownik)", "gen (dopełniacz)", "dat (celownik)", "acc (biernik)", "inst (narzędnik)", "loc (miejscownik)", "voc (wołacz)"],
    "number": ["sg (pojedyncza)", "pl (mnoga)"],
    "gender": ["m1 (męskoosobowy)", "m2 (męskozwierzęcy)", "m3 (męskorzeczowy)", "f (żeński)", "n (nijaki)"],
    "degree": ["pos (równy)", "com (wyższy)", "sup (najwyższy)"],
    "person": ["pri (pierwsza)", "sec (druga)", "ter (trzecia)"],
    "aspect": ["imperf (niedokonany)", "perf (dokonany)"],
    "negation": ["aff (niezanegowana - pisanie, czytanego)", "neg (zanegowana - niepisanie, nieczytanego)"],
    "accentability": ["akc (akcentowana - jego, niego, tobie)", "nakc (nieakcentowana - go, -ń, ci)"],
    "post-prepositionality": ["praep (poprzyimkowa - niego, -ń)", "npraep (niepoprzyimkowa - jego, go)"],
    "accommodability": ["congr (uzgadniająca - dwaj, pięcioma)", "rec (rządząca - dwóch, dwu, pięciorgiem)"],
    "vocalicity": ["wok (wokaliczna - -em)", "nwok (niewokaliczna - -m)"],
    "agglutination": ["agl (aglutynacyjna - niósł)", "nagl (nieaglutynacyjna - niosł-)"],
    "fullstoppedness": ["pun (z następującą kropką - tzn)", "npun (bez kropki - wg)"]
}


class ConditionRow(ctk.CTkFrame):
    def __init__(self, parent, theme, remove_callback, depth=0):
        super().__init__(parent, fg_color="transparent")
        self.theme = theme
        self.remove_callback = remove_callback
        self.depth = depth

        self.attr_map = {
            "Forma ortograficzna (orth)": "orth",
            "Forma podstawowa (base)": "base",
            "Część mowy NKJP (pos)": "pos",
            "Część mowy UDP (upos)": "upos",
            "Nadrzędnik (head)": "head",
            "Podrzędnik (dependent)": "dependent",
            "Relacja (deprel)": "deprel",
            "Jednostka nazwana (ner)": "ner",
            "Koreferencja (coref)": "coref",
            "Przypadek (case)": "case",
            "Liczba (number)": "number",
            "Rodzaj (gender)": "gender",
            "Stopień (degree)": "degree",
            "Osoba (person)": "person",
            "Aspekt (aspect)": "aspect",
            "Zanegowanie (negation)": "negation",
            "Akcentowość (accentability)": "accentability",
            "Poprzyimkowość (post-prep)": "post-prepositionality",
            "Akomodacyjność (accommodability)": "accommodability",
            "Wokaliczność (vocalicity)": "vocalicity",
            "Aglutynacyjność (agglutination)": "agglutination",
            "Kropkowalność (fullstoppedness)": "fullstoppedness",
            "Okno lematu (window_base)": "window_base",
            "Okno ortograficzne (window_orth)": "window_orth"

        }

        self.op_map = {
            "Równa się (=)": "=",
            "Nie równa się (!=)": "!=",
            "Zawiera tekst (~ partial)": "CONTAINS"
        }

        self.attr_var = ctk.StringVar(value="Forma ortograficzna (orth)")
        self.op_var = ctk.StringVar(value="Równa się (=)")

        self.text_entry = None
        self.nested_conditions = []

        self.setup_ui()

    def setup_ui(self):
        self.main_row = ctk.CTkFrame(self, fg_color="transparent")
        self.main_row.pack(fill="x", pady=2)

        dropdown_kwargs = dict(fg_color=self.theme["dropdown_fg"], button_color=self.theme["button_fg"],
                               text_color=self.theme["button_text"])

        self.attr_menu = ctk.CTkOptionMenu(self.main_row, variable=self.attr_var, values=list(self.attr_map.keys()),
                                           width=210, command=self.on_attr_change, **dropdown_kwargs)
        self.attr_menu.pack(side="left", padx=(0, 5))

        self.op_menu = ctk.CTkOptionMenu(self.main_row, variable=self.op_var, values=list(self.op_map.keys()),
                                         width=180, **dropdown_kwargs)
        self.op_menu.pack(side="left", padx=5)

        self.val_container = ctk.CTkFrame(self.main_row, fg_color="transparent")
        self.val_container.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_del = ctk.CTkButton(self.main_row, text="✖", width=30, fg_color="#D9534F", hover_color="#C9302C",
                                     command=lambda: self.remove_callback(self))
        self.btn_del.pack(side="left", padx=(5, 0))

        self.nested_container = ctk.CTkFrame(self, fg_color="transparent")
        self.on_attr_change(self.attr_var.get())

    def on_attr_change(self, selected_attr):
        # Czyszczenie interfejsu
        for widget in self.val_container.winfo_children():
            widget.destroy()
        for widget in self.nested_container.winfo_children():
            widget.destroy()
        self.nested_container.pack_forget()

        self.nested_conditions.clear()
        self.text_entry = None

        internal_attr = self.attr_map[selected_attr]
        dropdown_kwargs = dict(fg_color=self.theme["dropdown_fg"], button_color=self.theme["button_fg"],
                               text_color=self.theme["button_text"])

        # --- ZAGNIEŻDŻENIA (HEAD / DEPENDENT) ---
        if internal_attr in ["head", "dependent"]:
            self.op_menu.configure(state="normal", values=["Równa się (=)", "Nie równa się (!=)"])
            self.op_var.set("Równa się (=)")

            ctk.CTkLabel(self.val_container, text="Dystans:", font=("Verdana", 11, "bold"),
                         text_color=self.theme["label_text"]).pack(side="left", padx=(5, 2))
            self.dist_op_var = ctk.StringVar(value="Dowolny")
            ctk.CTkOptionMenu(self.val_container, variable=self.dist_op_var,
                              values=["Dowolny", "Równy (=)", "Mniejszy niż (<)", "Większy niż (>)"], width=130,
                              height=24, **dropdown_kwargs).pack(side="left", padx=5)

            self.dist_val_var = ctk.StringVar(value="1")
            ctk.CTkEntry(self.val_container, textvariable=self.dist_val_var, width=50, height=24,
                         fg_color=self.theme["frame_fg"]).pack(side="left", padx=5)

            self.nested_container.configure(fg_color=self.theme["app_bg"], corner_radius=6)
            self.nested_container.pack(fill="x", expand=False, padx=(40, 5), pady=(2, 5), ipady=5)

            self.nested_rules_frame = ctk.CTkFrame(self.nested_container, fg_color="transparent")
            self.nested_rules_frame.pack(fill="x", expand=False, padx=5, pady=2)

            ctk.CTkButton(self.nested_container, text="+ Dodaj atrybut zagnieżdżony", height=24, width=180,
                          fg_color=self.theme["button_fg"], command=self.add_nested_rule).pack(anchor="w", padx=10,
                                                                                               pady=2)
            self.add_nested_rule()

        # --- KOREFERENCJA (COREF) ---
        elif internal_attr == "coref":
            self.op_menu.configure(state="normal", values=list(self.op_map.keys()))
            self.coref_role_var = ctk.StringVar(value="Dowolna ranga")
            ctk.CTkOptionMenu(self.val_container, variable=self.coref_role_var,
                              # Dodajemy opcję (M)
                              values=["Dowolna ranga", "(H) - Głowa", "(P) - Część", "(M) - Cała wzmianka"], width=150,
                              **dropdown_kwargs).pack(side="left", padx=(0, 5))

            self.text_entry = ctk.CTkEntry(self.val_container, placeholder_text="Powiązane słowo...",
                                           fg_color=self.theme["frame_fg"])
            self.text_entry.pack(side="left", fill="x", expand=True)

            ctk.CTkButton(self.val_container, text="[.*] Regex", width=70, fg_color=self.theme["frame_fg"],
                          border_width=1, border_color=self.theme["button_fg"], text_color=self.theme["label_text"],
                          hover_color=self.theme["subframe_fg"],
                          command=lambda: self.open_regex_helper(self.text_entry)).pack(side="left", padx=(5, 0))

        # --- NER Z POMOCNIKAMI ---
        elif internal_attr == "ner":
            self.op_menu.configure(state="normal", values=list(self.op_map.keys()))

            self.text_entry = ctk.CTkEntry(self.val_container, placeholder_text="Tag jednostki...",
                                           fg_color=self.theme["frame_fg"])
            self.text_entry.pack(side="left", fill="x", expand=True)

            def insert_ner(choice, menu_var, default_text):
                clean_val = choice.split(" ")[0]
                if clean_val not in ["Brak", "Wszystkie"]:
                    self.text_entry.insert(tk.INSERT, clean_val)
                    self.text_entry.focus()
                menu_var.set(default_text)  # Reset napisu

            ner_pref_var = ctk.StringVar(value="➕ Prefiks")
            ctk.CTkOptionMenu(self.val_container, variable=ner_pref_var, values=NER_PREFIXES, width=105,
                              command=lambda c: insert_ner(c, ner_pref_var, "➕ Prefiks"), **dropdown_kwargs).pack(
                side="left", padx=(5, 0))

            ner_type_var = ctk.StringVar(value="➕ Typ")
            ctk.CTkOptionMenu(self.val_container, variable=ner_type_var, values=NER_TYPES, width=80,
                              command=lambda c: insert_ner(c, ner_type_var, "➕ Typ"), **dropdown_kwargs).pack(
                side="left", padx=(5, 0))

            ctk.CTkButton(self.val_container, text="[.*] Regex", width=70, fg_color=self.theme["frame_fg"],
                          border_width=1, border_color=self.theme["button_fg"], text_color=self.theme["label_text"],
                          hover_color=self.theme["subframe_fg"],
                          command=lambda: self.open_regex_helper(self.text_entry)).pack(side="left", padx=(5, 0))

        # --- LISTY TAGÓW JAKO POMOCNIKI (POS, UPOS, DEPREL, CECHY MORF) ---
         # --- OKNO SŁOWA (WINDOW) ---
        elif internal_attr in ["window_base", "window_orth"]:
            self.op_menu.configure(state="normal", values=list(self.op_map.keys()))

            ctk.CTkLabel(self.val_container, text="Dystans (±):", font=("Verdana", 11, "bold"),
                         text_color=self.theme["label_text"]).pack(side="left", padx=(5, 2))

            self.window_size_var = ctk.StringVar(value="50")
            ctk.CTkEntry(self.val_container, textvariable=self.window_size_var, width=40, height=24,
                         fg_color=self.theme["frame_fg"]).pack(side="left", padx=(0, 5))

            self.text_entry = ctk.CTkEntry(self.val_container, placeholder_text="Szukane słowo...",
                                           fg_color=self.theme["frame_fg"])
            self.text_entry.pack(side="left", fill="x", expand=True)

            ctk.CTkButton(self.val_container, text="[.*] Regex", width=70, fg_color=self.theme["frame_fg"],
                          border_width=1, border_color=self.theme["button_fg"], text_color=self.theme["label_text"],
                          hover_color=self.theme["subframe_fg"],
                          command=lambda: self.open_regex_helper(self.text_entry)).pack(side="left", padx=(5, 0))

        # --- ZWYKŁY TEKST (ORTH, BASE) ---
        else:
            self.op_menu.configure(state="normal", values=list(self.op_map.keys()))
            self.text_entry = ctk.CTkEntry(self.val_container, placeholder_text="Wpisz wartość lub stwórz regex...",
                                           fg_color=self.theme["frame_fg"])
            self.text_entry.pack(side="left", fill="x", expand=True)

            ctk.CTkButton(self.val_container, text="[.*] Regex", width=70, fg_color=self.theme["frame_fg"],
                          border_width=1, border_color=self.theme["button_fg"], text_color=self.theme["label_text"],
                          hover_color=self.theme["subframe_fg"],
                          command=lambda: self.open_regex_helper(self.text_entry)).pack(side="left", padx=(5, 0))

    def open_regex_helper(self, entry_widget):
        if hasattr(self, "regex_window") and self.regex_window is not None and self.regex_window.winfo_exists():
            self.regex_window.target_entry = entry_widget
            self.regex_window.lift()
        else:
            self.regex_window = RegexHelperWindow(self.winfo_toplevel(), entry_widget, self.theme)

    def add_nested_rule(self):
        row = ConditionRow(self.nested_rules_frame, self.theme, self.remove_nested_rule, depth=self.depth + 1)
        row.pack(fill="x", pady=2)
        self.nested_conditions.append(row)

    def remove_nested_rule(self, row):
        row.destroy()
        if row in self.nested_conditions:
            self.nested_conditions.remove(row)

    def get_query_string(self):
        attr = self.attr_map[self.attr_var.get()]
        op_selection = self.op_map[self.op_var.get()]

        # 1. Zagnieżdżenia (head, dependent)
        if attr in ["head", "dependent"]:
            if not self.nested_conditions: return None
            inner_queries = [r.get_query_string() for r in self.nested_conditions if r.get_query_string()]
            if not inner_queries: return None

            dist_str = ""
            dist_op = self.dist_op_var.get()
            if dist_op != "Dowolny":
                dist_val = self.dist_val_var.get().strip()
                if dist_val:
                    if dist_op == "Równy (=)":
                        dist_str = f"({dist_val})"
                    elif dist_op == "Mniejszy niż (<)":
                        dist_str = f"(<{dist_val})"
                    elif dist_op == "Większy niż (>)":
                        dist_str = f"(>{dist_val})"

            return f"{attr}{dist_str}{op_selection}{{{' & '.join(inner_queries)}}}"

        # 2. Koreferencja (Złożenie roli i wartości z pola)
        if attr == "coref":
            role = self.coref_role_var.get()
            if role == "(H) - Głowa":
                attr = "coref(H)"
            elif role == "(P) - Część":
                attr = "coref(P)"
            elif role == "(M) - Cała wzmianka":
                attr = "coref(M)"

        # 3. Parametry okna (window_base / window_orth)
        if attr in ["window_base", "window_orth"]:
            if not self.text_entry: return None
            val = self.text_entry.get().strip()
            if not val: return None

            dist = self.window_size_var.get().strip()
            attr_str = f"{attr}({dist})" if dist else attr

            if op_selection == "CONTAINS":
                return f'{attr_str}="~{val}"'
            else:
                return f'{attr_str}{op_selection}"{val}"'

        # Każdy inny atrybut pobiera teraz wartość prosto z pola tekstowego (text_entry)
        if not self.text_entry: return None
        val = self.text_entry.get().strip()
        if not val: return None

        # Formatowanie końcowego atrybutu (Wspólne dla wszystkich pól tekstowych)
        if op_selection == "CONTAINS":
            return f'{attr}="~{val}"'
        else:
            return f'{attr}{op_selection}"{val}"'


class GapBlock(ctk.CTkFrame):
    def __init__(self, parent, theme, remove_callback):
        super().__init__(parent, fg_color=theme["frame_fg"], corner_radius=12, border_width=1, border_color="#D9A04F")
        self.theme = theme

        lbl = ctk.CTkLabel(self, text="⬌ Odstęp (Dystans)", font=("Verdana", 12, "bold"), text_color="#D9A04F")
        lbl.pack(side="left", padx=10, pady=5)

        ctk.CTkLabel(self, text="od:", text_color=theme["label_text"]).pack(side="left", padx=(10, 2))
        self.min_entry = ctk.CTkEntry(self, width=40, height=28)
        self.min_entry.insert(0, "1")
        self.min_entry.pack(side="left")

        ctk.CTkLabel(self, text="do:", text_color=theme["label_text"]).pack(side="left", padx=(10, 2))
        self.max_entry = ctk.CTkEntry(self, width=40, height=28)
        self.max_entry.insert(0, "3")
        self.max_entry.pack(side="left")

        ctk.CTkLabel(self, text="słów", text_color=theme["label_text"]).pack(side="left", padx=(5, 10))

        btn_del = ctk.CTkButton(self, text="✖", width=30, height=28, fg_color="#D9534F", hover_color="#C9302C",
                                command=lambda: remove_callback(self))
        btn_del.pack(side="right", padx=10, pady=5)

    def get_query_string(self):
        min_v = self.min_entry.get().strip() or "0"
        max_v = self.max_entry.get().strip() or "1"
        return f"[*][{min_v},{max_v}]"


class MetaBlock(ctk.CTkFrame):
    def __init__(self, parent, theme, remove_callback):
        super().__init__(parent, fg_color=theme["frame_fg"], corner_radius=12, border_width=1, border_color="#9A5BB6")
        self.theme = theme
        self.is_gap = False
        self.is_meta = True

        # Nagłówek
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header_frame, text="⚙ Filtr", font=("Verdana", 12, "bold"),
                     text_color="#9A5BB6").pack(side="left")
        ctk.CTkButton(header_frame, text="Usuń", width=60, height=24, fg_color="#D9534F", hover_color="#C9302C",
                      command=lambda: remove_callback(self)).pack(side="right")

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.type_var = ctk.StringVar(value="Autor")
        self.types = ["Autor", "Tytuł", "Data publikacji", "Inne metadane", "Frekwencja lematów (base)",
                      "Frekwencja form (orth)", "W jednym zdaniu (<s>)"]

        ctk.CTkOptionMenu(self.content_frame, variable=self.type_var, values=self.types, command=self.on_type_change,
                          fg_color=theme["dropdown_fg"], button_color=theme["button_fg"],
                          text_color=theme["button_text"]).pack(side="left", padx=(0, 10))

        self.dynamic_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.dynamic_frame.pack(side="left", fill="x", expand=True)

        self.on_type_change("Autor")

    def on_type_change(self, selected_type):
        for w in self.dynamic_frame.winfo_children():
            w.destroy()

        self.op_var = ctk.StringVar(value="=")
        ops = ["=", "!=", "<", ">", "<=", ">=", "~ (zawiera)"]
        dropdown_kwargs = dict(fg_color=self.theme["dropdown_fg"], button_color=self.theme["button_fg"],
                               text_color=self.theme["button_text"])

        if selected_type in ["Autor", "Tytuł", "Data publikacji"]:
            ctk.CTkOptionMenu(self.dynamic_frame, variable=self.op_var, values=ops, width=80, **dropdown_kwargs).pack(
                side="left", padx=5)
            self.val_entry = ctk.CTkEntry(self.dynamic_frame, placeholder_text="Wartość...",
                                          fg_color=self.theme["frame_fg"])
            self.val_entry.pack(side="left", fill="x", expand=True, padx=5)

        elif selected_type == "Inne metadane":
            self.key_entry = ctk.CTkEntry(self.dynamic_frame, width=120, placeholder_text="Klucz (np. gazeta)",
                                          fg_color=self.theme["frame_fg"])
            self.key_entry.pack(side="left", padx=5)
            ctk.CTkOptionMenu(self.dynamic_frame, variable=self.op_var, values=ops, width=80, **dropdown_kwargs).pack(
                side="left", padx=5)
            self.val_entry = ctk.CTkEntry(self.dynamic_frame, placeholder_text="Wartość...",
                                          fg_color=self.theme["frame_fg"])
            self.val_entry.pack(side="left", fill="x", expand=True, padx=5)

        elif selected_type in ["Frekwencja lematów (base)", "Frekwencja form (orth)"]:
            ctk.CTkLabel(self.dynamic_frame, text="Top:", text_color=self.theme["label_text"]).pack(side="left", padx=2)
            self.top_entry = ctk.CTkEntry(self.dynamic_frame, width=50, fg_color=self.theme["frame_fg"])
            self.top_entry.pack(side="left", padx=2)

            ctk.CTkLabel(self.dynamic_frame, text="Min f:", text_color=self.theme["label_text"]).pack(side="left",
                                                                                                      padx=2)
            self.min_entry = ctk.CTkEntry(self.dynamic_frame, width=50, fg_color=self.theme["frame_fg"])
            self.min_entry.pack(side="left", padx=2)

            ctk.CTkLabel(self.dynamic_frame, text="Max f:", text_color=self.theme["label_text"]).pack(side="left",
                                                                                                      padx=2)
            self.max_entry = ctk.CTkEntry(self.dynamic_frame, width=50, fg_color=self.theme["frame_fg"])
            self.max_entry.pack(side="left", padx=2)

        elif selected_type == "W jednym zdaniu (<s>)":
            ctk.CTkLabel(self.dynamic_frame, text="Ogranicza całą znalezioną sekwencję do jednego zdania.",
                         text_color=self.theme["label_text"]).pack(side="left", padx=5)

    def get_query_string(self):
        t = self.type_var.get()

        def get_op():
            op = self.op_var.get()
            return "=" if op == "~ (zawiera)" else op

        def get_val():
            val = self.val_entry.get().strip()
            return f"~{val}" if self.op_var.get() == "~ (zawiera)" else val

        if t == "Autor":
            return f'<autor{get_op()}"{get_val()}">' if get_val() else ""
        elif t == "Tytuł":
            return f'<tytuł{get_op()}"{get_val()}">' if get_val() else ""
        elif t == "Data publikacji":
            return f'<data{get_op()}"{get_val()}">' if get_val() else ""
        elif t == "Inne metadane":
            k, v = self.key_entry.get().strip(), get_val()
            return f'<metadane:{k}{get_op()}"{v}">' if (k and v) else ""
        elif t in ["Frekwencja lematów (base)", "Frekwencja form (orth)"]:
            tag = "frequency_base" if "base" in t else "frequency_orth"
            attrs = []
            if top := self.top_entry.get().strip(): attrs.append(f'top="{top}"')
            if min_f := self.min_entry.get().strip(): attrs.append(f'min="{min_f}"')
            if max_f := self.max_entry.get().strip(): attrs.append(f'max="{max_f}"')
            return f'<{tag} {" ".join(attrs)}>' if attrs else ""
        elif t == "W jednym zdaniu (<s>)":
            return "<s>"
        return ""

class QueryBuilderWindow(ctk.CTkToplevel):
    def __init__(self, parent, target_textbox, theme):
        super().__init__(parent)
        self.target_textbox = target_textbox
        self.theme = theme

        self.title("Konstruktor zapytań")
        self.geometry("900x700")
        self.configure(fg_color=self.theme["app_bg"])
        self.grab_set()

        self.blocks = []

        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.bottom_frame = ctk.CTkFrame(self, fg_color=self.theme["subframe_fg"], corner_radius=10)
        self.bottom_frame.pack(fill="x", side="bottom", padx=15, pady=15, ipadx=10, ipady=10)

        self.btn_add_token = ctk.CTkButton(self.bottom_frame, text="➕ Dodaj Segment (Token)",
                                           font=("Verdana", 12, "bold"), fg_color=self.theme["button_fg"],
                                           command=self.add_token_block)
        self.btn_add_token.pack(side="left", padx=5)

        self.btn_add_gap = ctk.CTkButton(self.bottom_frame, text="⬌ Dodaj Odstęp", font=("Verdana", 12, "bold"),
                                         fg_color="#D9A04F", hover_color="#B8863A", text_color="black",
                                         command=self.add_gap_block)
        self.btn_add_gap.pack(side="left", padx=5)

        self.btn_add_meta = ctk.CTkButton(self.bottom_frame, text="⚙ Dodaj filtr",
                                          font=("Verdana", 12, "bold"),
                                          fg_color="#9A5BB6", hover_color="#8E44AD", text_color="white",
                                          command=self.add_meta_block)
        self.btn_add_meta.pack(side="left", padx=5)

        self.btn_generate = ctk.CTkButton(self.bottom_frame, text="✅ Gotowe - Wstaw zapytanie",
                                          font=("Verdana", 13, "bold"), fg_color="#4E8752", hover_color="#57965C",
                                          command=self.generate_and_insert)
        self.btn_generate.pack(side="right", padx=5)

        self.add_token_block()

    def add_meta_block(self):
        meta = MetaBlock(self.scroll_frame, self.theme, self.remove_block)
        meta.pack(fill="x", pady=(0, 10), ipadx=5)
        self.blocks.append(meta)

    def add_token_block(self):
        # Tworzymy główną kartę segmentu (usunąłem ipadx/ipady, bo CTk czasem głupieje przy ramkach)
        card = ctk.CTkFrame(self.scroll_frame, fg_color=self.theme["subframe_fg"], corner_radius=12, border_width=1,
                            border_color="#3E3F42")
        card.pack(fill="x", pady=(0, 15), padx=5)
        card.is_gap = False

        # 1. Nagłówek (dodany bezpieczny wewnętrzny margines: padx=15, pady=10)
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(header_frame, text="[ Segment / Słowo ]", font=("Verdana", 13, "bold"),
                     text_color=self.theme["label_text"]).pack(side="left")
        ctk.CTkButton(header_frame, text="Usuń segment", width=100, height=28, fg_color="#D9534F",
                      hover_color="#C9302C", command=lambda: self.remove_block(card)).pack(side="right")

        # 2. Kontener na reguły (dodany padx=15)
        card.rules_container = ctk.CTkFrame(card, fg_color="transparent")
        card.rules_container.pack(fill="x", padx=15, pady=5)
        card.rules_list = []

        self.add_rule(card)

        # 3. Przycisk dodawania atrybutu (dodany padx=15 i dolny margines pady=15, żeby nie dotykał spodu ramki)
        ctk.CTkButton(card, text="➕ Dodaj atrybut (AND)", width=160, height=28, fg_color="transparent", border_width=1,
                      border_color=self.theme["button_fg"], text_color=self.theme["label_text"],
                      command=lambda: self.add_rule(card)).pack(anchor="w", padx=15, pady=(5, 15))

        self.blocks.append(card)

    def add_gap_block(self):
        gap = GapBlock(self.scroll_frame, self.theme, self.remove_block)
        gap.pack(fill="x", pady=(0, 10), ipadx=10)
        gap.is_gap = True
        self.blocks.append(gap)

    def add_rule(self, card):
        row = ConditionRow(card.rules_container, self.theme, lambda r: self.remove_rule(card, r))
        row.pack(fill="x", pady=3)
        card.rules_list.append(row)

    def remove_rule(self, card, row):
        row.destroy()
        if row in card.rules_list:
            card.rules_list.remove(row)

    def remove_block(self, block):
        block.destroy()
        if block in self.blocks:
            self.blocks.remove(block)

    def generate_and_insert(self):
        token_parts = []
        meta_parts = []
        sentence_bound = False
        ignored_blocks = 0

        for block in self.blocks:
            if hasattr(block, 'is_meta') and block.is_meta:
                q = block.get_query_string()
                if q == "<s>":
                    sentence_bound = True
                elif q:
                    meta_parts.append(q)
                else:
                    ignored_blocks += 1

            elif getattr(block, 'is_gap', False):
                token_parts.append(block.get_query_string())

            else:
                token_conditions = [r.get_query_string() for r in block.rules_list if r.get_query_string()]
                if token_conditions:
                    token_parts.append(f"[{' & '.join(token_conditions)}]")
                else:
                    ignored_blocks += 1
                    token_parts.append("[*]")

        tokens_query = "".join(token_parts)

        if sentence_bound:
            tokens_query = f"{tokens_query} <s >"

        final_query = (tokens_query + " " + " ".join(meta_parts)).strip()

        if not final_query:
            messagebox.showwarning("Puste zapytanie", "Nie udało się zbudować zapytania.")
            return

        if ignored_blocks > 0:
            messagebox.showinfo(
                "Uwaga",
                f"Pominięto {ignored_blocks} pustych lub niekompletnych bloków podczas budowania zapytania."
            )

        self.target_textbox.delete("1.0", ctk.END)
        self.target_textbox.insert("1.0", final_query)
        self.target_textbox.event_generate("<KeyRelease>")
        self.destroy()


def export_data():
    try:
        all_columns = [
            "Data publikacji", "context", "full_text_with_markers",
            "Rezultat", "matched_lemmas",
            "month_key", "Tytuł", "Autor", "additional_metadata",
            "Lewy kontekst", "Prawy kontekst", "row_index", "start_idx", "end_idx"
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

        # Export to Excel with multiple sheets
        if file_path.lower().endswith(".xlsx"):
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Sheet 1: main export
                df_export_slice.to_excel(writer, sheet_name="Wyniki wyszukiwania", index=False)

                # Sheet 2: paginator_fq['data']
                if 'data' in paginator_fq and paginator_fq['data']:
                    data_rows = paginator_fq['data']
                    headers = ["Nr", "Forma podstawowa (base)", "Liczba wystąpień", "Częstość względna"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    df_data.to_excel(writer, sheet_name="Częstość lematów", index=False)

                # Sheet 3: paginator_token['data']
                if 'data' in paginator_token and paginator_token['data']:
                    data_rows = paginator_token['data']
                    headers = ["Nr", "Forma tekstowa (orth)", "Liczba wystąpień", "Częstość względna"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    df_data.to_excel(writer, sheet_name="Częstość tokenów", index=False)

                # Sheet 4: paginator_month['data']
                if 'data' in paginator_month and paginator_month['data']:
                    data_rows = paginator_month['data']
                    headers = ["Rok", "Miesiąc", "Forma podstawowa", "Liczba wystąpień", "Częstość względna"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    df_data.to_excel(writer, sheet_name="Częstość w czasie", index=False)

                # --- NOWOŚĆ: Sheet 5: Kolokacje ---
                # --- NOWOŚĆ: Sheet 5: Kolokacje ---
                if 'paginator_colloc' in globals() and 'data' in paginator_colloc and paginator_colloc['data']:
                    data_rows = paginator_colloc['data']
                    headers = ["Nr", "Kolokat", "f(nc)", "f(c)", "Log-Likelihood", "MI Score", "T-score",
                               "Log-Dice"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    df_data.to_excel(writer, sheet_name="Kolokacje", index=False)

        else:
            # fallback CSV export (single sheet - tylko główne wyniki)
            df_export_slice.to_csv(file_path, index=False)

    except Exception as e:
        messagebox.showerror("Błąd eksportu",
                             f"Nie udało się zapisać pliku. Upewnij się, że plik nie jest otwarty w innym programie.\n\nSzczegóły: {e}")

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
            log_exception(
                "save_to_file",
                e,
                f"Nie udało się zapisać zaznaczenia do fiszki.\nSzczegóły: {e}"
            )

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


# Zmieniona funkcja przyjmująca nazwę pliku jako argument
def open_webview_window(file_name):
    global webview_thread

    if webview_thread and webview_thread.is_alive():
        print("Webview is already running.")
        return webview_thread

    def worker():
        import webview
        # Używamy przekazanej nazwy pliku
        file_path = os.path.join(BASE_DIR, f"temp/{file_name}")
        window = webview.create_window(
            "Pomoc Korpusuj",
            url=f"file://{file_path}",
            width=1200,
            height=800,
            resizable=True,
            text_select=True,
        )
        webview.start(debug=False)

    webview_thread = threading.Thread(target=worker, name="MainThread")
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
        get_fiszki_module().load_file_content(value)


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
    global dataframes, inverted_indexes
    for i, (name, path) in enumerate(files.items(), start=1):
        try:
            if loading_label:
                loading_label.configure(text=f"Ładowanie {name} ({i}/{len(files)})...")
                loading_label.update()

            # Szybki odczyt metadanych ze schematu Parquet (bez wczytywania całej tabeli do RAM)
            schema = pq.read_schema(path)
            schema_meta = schema.metadata

            if schema_meta and b"korpus_meta" in schema_meta:
                korpus_meta = json.loads(schema_meta[b"korpus_meta"].decode('utf-8'))
                base_tf = korpus_meta.get("base_tf", {})
                orth_tf = korpus_meta.get("orth_tf", {})
                total_tokens = korpus_meta.get("total_tokens", 0)
                monthly_token_counts = korpus_meta.get("monthly_token_counts", {})
            else:
                base_tf, orth_tf, total_tokens, monthly_token_counts = {}, {}, 0, {}

            df = pd.read_parquet(path)
            dataframes[name] = df

            print(f"Budowanie indeksu odwróconego dla {name}...")
            base_idx = {}
            orth_idx = {}

            # Budujemy jedynie indeks wierszy do wyszukiwarki (bez zliczania tokenów!)
            for row in df.itertuples():
                row_id = row.Index
                for lemma in set(row.lemmas):
                    base_idx.setdefault(lemma, set()).add(row_id)
                for token in set(row.tokens):
                    orth_idx.setdefault(token, set()).add(row_id)

            inverted_indexes[name] = {
                "base": base_idx,
                "orth": orth_idx,
                "base_tf": base_tf,
                "orth_tf": orth_tf,
                "total_tokens": total_tokens,
                "monthly_token_counts": monthly_token_counts
            }

            print(f"{name} loaded and indexed: {len(df)} rows")
        except Exception as e:
            messagebox.showerror("Błąd ładowania korpusu",
                                 f"Nie udało się wczytać pliku {name}.\nPlik może być uszkodzony.\nSzczegóły: {e}")

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



THEMES = {
    "ciemny": {
        # Base
        "app_bg": "#1F2328",

        # Tables
        "row_colors": ("#2C2F33", "#33373D"),
        "text_colors": ["#FFFFFF", "#FFFFFF", "#65A46F", "#FFFFFF"],
        "text_colors_month": ["white", "white", "white", "#65a46f", "white"],
        "text_colors_colloc": ["#FFFFFF", "#65A46F", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"], # <--- NOWE
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
        "app_bg": "#CED3D3",

        # Tables
        "row_colors": ("#E6E8E8", "#F2F4F4"),
        "text_colors": ["black", "black", "#000DFF", "black"],
        "text_colors_month": ["black", "black", "black", "#000DFF", "black"],
        "text_colors_colloc": ["black", "#000DFF", "black", "black", "black", "black", "black", "black"], # <--- NOWE
        "selected_row": "#A3C9F1",
        "canvas_bg": "#E6E8E8",

        # Widgets
        "frame_fg": "#F5F7F7",
        "subframe_fg": "#E6E8E8",
        "button_fg": "#6BA6F7",
        "button_hover": "#89BDFA",
        "button_text": "black",
        "label_text": "black",
        "dropdown_fg": "#6BA6F7",
        "dropdown_hover": "#89BDFA",
        "dropdown_text": "black",

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
    colloc_frame.configure(fg_color=theme["frame_fg"])
    tabview.configure(fg_color=theme["frame_fg"])

    # Subframes (Widoczne, zaokrąglone kafelki z zawartością)
    for frame in [
        pagination_frame, entry_button_frame, pagination_lemma_frame, pagination_orth_frame,
        pagination_month_frame, plot_options_frame, saveplot_button_frame, checkboxes_frame,
        colloc_controls, pagination_colloc_frame, date_settings_frame
    ]:
        frame.configure(fg_color=theme["subframe_fg"], border_color=theme["subframe_fg"])

    # Kontenery strukturalne (MUSZĄ być przezroczyste, by było widać między nimi tło okna)
    for frame in [left_pane, right_pane, right_subframe, buttons_action_frame]:
        frame.configure(fg_color="transparent")

    # --- Zmiana motywu dynamicznych kontrolek (Obejście dla anonimowych etykiet i menu) ---
    def update_frame_children(parent_frame):
        for child in parent_frame.winfo_children():
            if isinstance(child, ctk.CTkLabel):
                child.configure(text_color=theme["label_text"])
            elif isinstance(child, ctk.CTkOptionMenu):
                child.configure(
                    fg_color=theme["button_fg"], button_color=theme["button_fg"],
                    dropdown_fg_color=theme["dropdown_fg"], dropdown_hover_color=theme["dropdown_hover"],
                    text_color=theme["button_text"], dropdown_text_color=theme["button_text"]
                )
            elif isinstance(child, ctk.CTkEntry):
                child.configure(fg_color=theme["frame_fg"], text_color=theme["label_text"])
            elif isinstance(child, ctk.CTkFrame):
                update_frame_children(child)  # Zmiana dla dzieci w zagnieżdżonych ramkach

    update_frame_children(colloc_controls)
    update_frame_children(date_settings_frame) # <--- DODANO magiczne kolorowanie dat i interwałów!

    # --- Buttons ---
    for button in [
        button_search, settings_button, button_first, button_prev, button_next, button_last,
        button_first_lemma, button_prev_lemma, button_next_lemma, button_last_lemma,
        button_first_orth, button_prev_orth, button_next_orth, button_last_orth,
        button_first_month, button_prev_month, button_next_month, button_last_month,
        button_save_plot, save_selection_button,
        button_first_colloc, button_prev_colloc, button_next_colloc, button_last_colloc, btn_calc_colloc,
        btn_refresh_plot # <--- DODANO nowy przycisk odświeżania
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
        frekw_wykresy, rows_label, page_label_colloc
    ]:
        label.configure(text_color=theme["label_text"])

    # --- OptionMenus ---
    for option in [option_corpus, option_sort, dropdown_rows, dropdown, plot_type_menu]: # <--- ZMIENIONO WYKRES_SORT_MENU NA PLOT_TYPE_MENU
        option.configure(
            fg_color=theme["dropdown_fg"],
            dropdown_fg_color=theme["dropdown_fg"],
            dropdown_hover_color=theme["dropdown_hover"],
            text_color=theme["button_text"],
            dropdown_text_color=theme["button_text"]
        )

    # --- Entries / Textboxes ---
    for entry in [entry_query, entry_left_context, entry_right_context, fiszka_entrybox, text_full]:
        entry.configure(
            fg_color=theme["subframe_fg"],
            text_color=theme["label_text"],
            border_color=theme["subframe_fg"]  # <--- DODANO border_color
        )

    # --- Tabview ---
    tabview._segmented_button.configure(
        fg_color=theme["frame_fg"],
        selected_color=theme["button_fg"],
        unselected_color=theme["subframe_fg"],
        text_color=theme["button_text"],
        selected_hover_color=theme["button_hover"],
        unselected_hover_color=theme["dropdown_hover"],
    )

    # --- Selektor tabel (Segmented Button) ---
    table_selector.configure(
        selected_color=theme["button_fg"],
        unselected_color=theme["subframe_fg"],
        selected_hover_color=theme["button_hover"],
        unselected_hover_color=theme["dropdown_hover"],
        text_color=theme["button_text"]
    )
    # --- PanedWindow (Przeciągany separator) ---
    paned_window.configure(bg=theme["frame_fg"])

    # KLUCZOWE: Nadpisanie zbuforowanego koloru tła (bg_color),
    # który ujawnia się podczas zmiany rozmiaru/przeciągania ramki.
    left_pane.configure(bg_color=theme["frame_fg"])
    right_pane.configure(bg_color=theme["frame_fg"])
    right_subframe.configure(bg_color=theme["subframe_fg"])

    # Fonts
    font_tuple = (font_family.get(), fontsize)
    for tbl in (text_result, frekw_dane_tabela, frekw_dane_tabela_orth, frekw_dane_tabela_month,
                colloc_table):
        tbl.set_header_font(font_tuple)
        tbl.set_font(font_tuple)

    # Tables - Standardowe 4-kolumnowe
    for tbl in (text_result, frekw_dane_tabela, frekw_dane_tabela_orth):
        tbl.set_row_colors(*theme["row_colors"])
        tbl.set_text_colors(theme["text_colors"])
        tbl.set_selected_row_color(theme["selected_row"])
        tbl.set_canvas_background(theme["canvas_bg"])

        try:
            tbl.configure(bg=theme["canvas_bg"])
        except Exception:
            pass
        try:
            tbl.configure(bg_color=theme["canvas_bg"])
        except Exception:
            pass

    # Tabela 5-kolumnowa
    frekw_dane_tabela_month.set_text_colors(theme["text_colors_month"])
    frekw_dane_tabela_month.set_row_colors(*theme["row_colors"])
    frekw_dane_tabela_month.set_selected_row_color(theme["selected_row"])
    frekw_dane_tabela_month.set_canvas_background(theme["canvas_bg"])

    # Tabela 7-kolumnowa (Kolokacje)
    colloc_table.set_text_colors(theme["text_colors_colloc"])
    colloc_table.set_row_colors(*theme["row_colors"])
    colloc_table.set_selected_row_color(theme["selected_row"])
    colloc_table.set_canvas_background(theme["canvas_bg"])

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
    colloc_frame.grid_remove()

    if choice == "Formy podstawowe (base)":
        lemma_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Formy ortograficzne (orth)":
        orth_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Częstość w czasie":
        month_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Kolokacje":
        colloc_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)



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


def calculate_collocs():
    if not full_results_sorted:
        messagebox.showinfo("Brak wyników", "Najpierw wyszukaj frazę, aby móc obliczyć jej kolokacje.")
        return

    # 1. Pobieranie ustawień GUI (musi zostać w głównym wątku!)
    mode = colloc_mode_var.get()
    upos_filter = upos_var.get()
    pos_filter = pos_var.get()
    form_mode = colloc_form_var.get()
    ignore_case = colloc_ignore_case_var.get()
    use_sentence_bound = sentence_boundary_var.get()
    sort_mode = colloc_sort_var.get()

    try:
        min_freq = int(entry_min_freq.get() or "1")
        min_range = int(entry_min_range.get() or "1")
        l_span = int(entry_l_span.get() or "5")
        r_span = int(entry_r_span.get() or "5")
    except ValueError:
        messagebox.showerror("Błąd", "Wartości muszą być liczbami całkowitymi.")
        return

    syn_dir = syn_dir_var.get()
    raw_deprel = syn_deprel_var.get()
    deprel_filter = raw_deprel.replace("├─", "").strip().split(" ")[0]

    # Zablokuj przycisk, żeby ktoś nie kliknął 5 razy podczas liczenia
    btn_calc_colloc.configure(state="disabled", text="Liczenie...")

    def worker():
        try:
            colloc_counter = Counter()
            colloc_doc_tracker = {}
            df = dataframes[global_selected_corpus]
            total_actual_slots = 0
            seen_slots = set()

            # --- NOWOŚĆ: KULOODPORNY FILTR KOLOKACJI ---
            def get_clean_colloc(word):
                if pd.isna(word) or word is None: return ""
                # Usuwamy niewidzialne znaki Unicode i twarde spacje
                w = str(word).replace('\u200b', '').replace('\xad', '').replace('\xa0', '').strip()
                # Odrzucamy całkowicie puste lub nierozpoznane lematy "_"
                if not w or w == "_": return ""
                # Odrzucamy, jeśli znak to wyłącznie interpunkcja (nawet ta polska)
                import string
                if all(c in string.punctuation or c in '„”«»–—…' for c in w): return ""
                return w

            # --- 1. ZBIERANIE KOLOKATÓW ---
            for res in full_results_sorted:
                row_idx, start_idx, end_idx = res[11], res[12], res[13]
                row_data = df.loc[row_idx]
                lemmas = row_data.lemmas
                tokens = row_data.tokens
                postags = row_data.postags
                upostags = getattr(row_data, "upostags", None)
                sentence_ids = row_data.sentence_ids

                form_array = lemmas if form_mode == "Lemat (base)" else tokens

                # Granice okna (Zdanie lub cały plik)
                if use_sentence_bound:
                    sent_id = sentence_ids[start_idx]
                    sent_start = start_idx
                    while sent_start > 0 and sentence_ids[sent_start - 1] == sent_id:
                        sent_start -= 1
                    sent_end = start_idx
                    while sent_end < len(sentence_ids) and sentence_ids[sent_end] == sent_id:
                        sent_end += 1
                else:
                    sent_start, sent_end = 0, len(lemmas)

                if mode == "Liniowe":
                    window_indices = list(range(max(sent_start, start_idx - l_span), start_idx)) + \
                                     list(range(end_idx, min(sent_end, end_idx + r_span)))

                    for i in window_indices:
                        if (row_idx, i) in seen_slots:
                            continue
                        seen_slots.add((row_idx, i))

                        is_punct = (upostags[i] == "PUNCT") if upostags is not None else (tokens[i] in string.punctuation)
                        total_actual_slots += 1

                        if not is_punct:
                            u_match = (upos_filter == "Wszystkie") if upostags is None else (
                                        upos_filter == "Wszystkie" or upostags[i] == upos_filter)
                            p_match = (pos_filter == "Wszystkie" or postags[i] == pos_filter)

                            if u_match and p_match:
                                clean_w = get_clean_colloc(form_array[i])
                                if clean_w:
                                    colloc = clean_w.lower() if ignore_case else clean_w
                                    colloc_counter[colloc] += 1
                                    colloc_doc_tracker.setdefault(colloc, set()).add(row_idx)
                else:
                    # Tryb: Składniowe (Zależności)
                    word_ids = row_data.word_ids
                    head_ids = row_data.head_ids
                    deprels = row_data.deprels

                    for i in range(start_idx, end_idx):
                        w_id = word_ids[i]
                        h_id = head_ids[i]

                        if syn_dir in ["Nadrzędnik", "Oba"] and h_id != 0:
                            for j in range(sent_start, sent_end):
                                if word_ids[j] == h_id and j not in range(start_idx, end_idx):
                                    if (row_idx, j) in seen_slots:
                                        continue
                                    seen_slots.add((row_idx, j))

                                    if deprel_filter == "Wszystkie" or deprels[i] == deprel_filter:
                                        is_punct = (upostags[j] == "PUNCT") if upostags is not None else (
                                                    tokens[j] in string.punctuation)
                                        if not is_punct:
                                            total_actual_slots += 1
                                            u_match = (upos_filter == "Wszystkie") if upostags is None else (
                                                        upos_filter == "Wszystkie" or upostags[j] == upos_filter)
                                            p_match = (pos_filter == "Wszystkie" or postags[j] == pos_filter)
                                            if u_match and p_match:
                                                clean_w = get_clean_colloc(form_array[j])
                                                if clean_w:
                                                    colloc = clean_w.lower() if ignore_case else clean_w
                                                    colloc_counter[colloc] += 1
                                                    colloc_doc_tracker.setdefault(colloc, set()).add(row_idx)

                        if syn_dir in ["Podrzędnik", "Oba"]:
                            for j in range(sent_start, sent_end):
                                if head_ids[j] == w_id and j not in range(start_idx, end_idx):
                                    if (row_idx, j, "dep") in seen_slots:
                                        continue
                                    seen_slots.add((row_idx, j, "dep"))

                                    if deprel_filter == "Wszystkie" or deprels[j] == deprel_filter:
                                        is_punct = (upostags[j] == "PUNCT") if upostags is not None else (
                                                    tokens[j] in string.punctuation)
                                        if not is_punct:
                                            total_actual_slots += 1
                                            u_match = (upos_filter == "Wszystkie") if upostags is None else (
                                                        upos_filter == "Wszystkie" or upostags[j] == upos_filter)
                                            p_match = (pos_filter == "Wszystkie" or postags[j] == pos_filter)
                                            if u_match and p_match:
                                                clean_w = get_clean_colloc(form_array[j])
                                                if clean_w:
                                                    colloc = clean_w.lower() if ignore_case else clean_w
                                                    colloc_counter[colloc] += 1
                                                    colloc_doc_tracker.setdefault(colloc, set()).add(row_idx)
            # --- 2. OBLICZANIE STATYSTYK ---
            inv_idx_data = inverted_indexes[global_selected_corpus]
            bg_tf_raw = inv_idx_data['base_tf'] if form_mode == "Lemat (base)" else inv_idx_data['orth_tf']
            total_tokens = inv_idx_data['total_tokens']

            # Dynamiczne tworzenie małych liter dla frekwencji globalnej korpusu
            if ignore_case:
                bg_tf = {}
                for k, v in bg_tf_raw.items():
                    kl = str(k).lower()
                    bg_tf[kl] = bg_tf.get(kl, 0) + v
            else:
                bg_tf = bg_tf_raw

            fn = len(full_results_sorted)
            if total_actual_slots == 0: total_actual_slots = 1

            colloc_stats = []

            for colloc, fnc in colloc_counter.items():
                if fnc < min_freq or len(colloc_doc_tracker[colloc]) < min_range:
                    continue

                fc = bg_tf.get(colloc, 1)
                expected = (total_actual_slots * fc) / total_tokens
                if expected <= 0: continue

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


                ll_score = 2 * (safe_ll(O11, E11) + safe_ll(O12, E12) + safe_ll(O21, E21) + safe_ll(O22, E22))

                colloc_stats.append(
                    [0, colloc, fnc, fc, round(ll_score, 2), round(mi_score, 2), round(t_score, 2), round(log_dice, 2)])

            # --- 3. SORTOWANIE I WYŚWIETLANIE ---
            sort_idx = {"Log-Likelihood": 4, "MI Score": 5, "T-score": 6, "Log-Dice": 7}.get(sort_mode, 4)
            colloc_stats.sort(key=lambda x: x[sort_idx], reverse=True)

            for i, row in enumerate(colloc_stats):
                row[0] = i + 1

            # Wrzucenie gotowych wyników do tabeli (Przekazanie do głównego wątku)
            def update_ui():
                paginator_colloc["data"] = colloc_stats
                paginator_colloc["current_page"][0] = 0
                update_table(paginator_colloc)
                btn_calc_colloc.configure(state="normal", text="Oblicz")

            app.after(0, update_ui)

        except Exception as e:
            logging.exception("Błąd kolokacji")

            def on_error():
                btn_calc_colloc.configure(state="normal", text="Oblicz")
                messagebox.showerror("Błąd kolokacji", f"Nie udało się obliczyć kolokacji.\nSzczegóły: {e}")

            app.after(0, on_error)



    threading.Thread(target=worker, daemon=True).start()

def search_from_table(selected_word):
    if not selected_word or not selected_word.strip():
        return

    selected_word = selected_word.strip()

    # 1. Pobieramy oryginalne zapytanie i rozbijamy na grupy (np. po ||)
    original_query = entry_query.get("1.0", ctk.END).strip()
    if not original_query:
        return

    # Pobieramy aktualne ustawienia kolokacji z GUI
    form_mode = colloc_form_var.get()
    attr = "base" if form_mode == "Lemat (base)" else "orth"
    mode = colloc_mode_var.get()
    ignore_case = colloc_ignore_case_var.get() # <--- Sprawdzamy stan checkboxa

    # --- NOWOŚĆ: Wstrzykiwanie regexa (Case-Insensitive) ---
    if ignore_case:
        import re
        # Używamy flagi (?i) do ignorowania wielkości liter.
        # Escapujemy też samo słowo, żeby ew. znaki (np. myślnik) nie zepsuły wzorca.
        query_val = f"(?i){re.escape(selected_word)}"
    else:
        query_val = selected_word

    query_groups = [g.strip() for g in original_query.split("||")]
    new_query_groups = []

    # Pomocnicza funkcja do wyciągania zawartości z pierwotnego nawiasu [...]
    def extract_core(q):
        if "<s" in q:
            q = q.split("<s")[0]
        m = re.search(r'\[(.*?)\]', q)
        if m:
            core = m.group(1).strip()
            return "" if core == "*" else core
        return ""

    def join_rules(*rules):
        # Łączy reguły operatorem '&', pomijając puste
        return " & ".join(r for r in rules if r)

    # --- TRYB LINIOWY (Szukamy po prostu w pobliżu) ---
    if mode == "Liniowe":
        try:
            val_l = int(entry_l_span.get() or "5")
            val_r = int(entry_r_span.get() or "5")
        except ValueError:
            val_l, val_r = 5, 5

        for qg in query_groups:
            if val_l > 0:
                new_query_groups.append(f'[{attr}="{query_val}"] [*][0,{val_l - 1}] {qg}')
            if val_r > 0:
                new_query_groups.append(f'{qg} [*][0,{val_r - 1}] [{attr}="{query_val}"]')

    # --- TRYB SKŁADNIOWY (Odwracanie ról i wstrzykiwanie relacji) ---
    else:
        syn_dir = syn_dir_var.get()
        raw_deprel = syn_deprel_var.get()
        deprel = raw_deprel.replace("├─", "").strip().split(" ")[0]

        for qg in query_groups:
            core_str = extract_core(qg)

            if syn_dir in ["Nadrzędnik", "Oba"]:
                # Kolokat to nadrzędnik (głowa).
                dep_rule = join_rules(core_str, f'deprel="{deprel}"' if deprel != "Wszystkie" else "")

                if dep_rule:
                    rule = f'[{attr}="{query_val}" & dependent={{{dep_rule}}}]'
                else:
                    rule = f'[{attr}="{query_val}"]'
                new_query_groups.append(rule)

            if syn_dir in ["Podrzędnik", "Oba"]:
                # Kolokat to podrzędnik (zależnik).
                main_rules = join_rules(f'{attr}="{query_val}"',
                                        f'deprel="{deprel}"' if deprel != "Wszystkie" else "")
                head_rule = f'head={{{core_str}}}' if core_str else ""

                rule = f'[{join_rules(main_rules, head_rule)}]'
                new_query_groups.append(rule)

    # Łączymy wszystkie opcje operatorem "LUB"
    new_query = " || ".join(new_query_groups)

    if new_query:
        # Przełączenie zakładki, aktualizacja pola tekstowego i wymuszenie wyszukiwania
        tabview.set("Wyniki wyszukiwania")
        entry_query.delete("1.0", ctk.END)
        entry_query.insert("1.0", new_query)
        search()



# Tworzenie interfejsu GUI
notify_status("Inicjalizacja silnika graficznego...")
app = ctk.CTk()
import tkinter as tk
tk._default_root = app
app.withdraw()

menu = Menu(app)

file_menu = menu.menu_bar(text="Plik", tearoff=0)
file_menu.add_command(label="Nowy projekt", command=load_corpora)
file_menu.add_command(label="Eksportuj wyniki", command=export_data)
file_menu.add_separator()
file_menu.add_command(label="Utwórz korpus", command=lambda: get_creator_module().main(app))
file_menu.add_separator()
file_menu.add_command(label="Zamknij", command=lambda: exit())
file_menu = menu.menu_bar(text="Edytuj", tearoff=0)
file_menu.add_command(label="Cofnij", command=lambda: undo())
file_menu.add_command(label="Ponów", command=lambda: redo())
history_menu = menu.menu_bar(text="Historia", tearoff=0)
update_history_menu()
file_menu = menu.menu_bar(text="Ustawienia", tearoff=0)
file_menu.add_command(label="Preferencje", command=settings_window)
file_menu = menu.menu_bar(text="Pomoc", tearoff=0)
# Przekazujemy konkretne nazwy plików do funkcji
file_menu.add_command(label="Instrukcja użytkownika",
                      command=lambda: open_webview_window("Instrukcja_uzytkownika.html"))
file_menu.add_command(label="Przewodnik po języku zapytań",
                      command=lambda: open_webview_window("Przewodnik po języku zapytań.html"))

app.title("Korpusuj")
icon_path = os.path.join(BASE_DIR, "favicon.ico")
try:
    app.iconbitmap(icon_path)
except Exception as e:
    print(f"Ostrzeżenie: Nie udało się załadować ikony: {e}")


# Global vars
font_family = ctk.StringVar(value=config['font_family'])
fontsize = config['fontsize']
styl_wykresow = ctk.StringVar(value=config['styl_wykresow'])
motyw = ctk.StringVar(value=config['motyw'])
plotting = ctk.StringVar(value=config.get('plotting', DEFAULT_SETTINGS['plotting']))
kontekst = config.get('kontekst', DEFAULT_SETTINGS['kontekst'])
settings_popup = None


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

def open_query_builder():
    current_theme = THEMES[motyw.get()]
    QueryBuilderWindow(app, entry_query, current_theme)

# --- Pływający przycisk Kreatora na polu tekstowym ---
builder_button = ctk.CTkButton(
    entry_query,             # Przycisk "rodzicem" czyni pole tekstowe
    text="✨ Konstruktor",
    font=("Verdana", 10, "bold"),
    fg_color="#37E152",      # Wyrazisty kolor, by był widoczny na ciemnym tle
    text_color="#1F2328",
    hover_color="#2DBF42",
    width=80,
    height=24,
    corner_radius=6,
    command=open_query_builder
)

# Pozycjonujemy go w prawym dolnym rogu (relx=1.0, rely=1.0)
# z małym marginesem (x=-10, y=-10) i kotwicą w prawym dolnym rogu (anchor="se")
builder_button.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor="se")

entry_query.insert("1.0", 'Podaj zapytanie np.: [orth="miasta"][pos="prep"][base="Polska"]')
entry_query.bind("<FocusIn>", on_entry_click)
entry_query.bind("<FocusOut>", on_focus_out)
entry_query.bind("<KeyRelease>", highlight_entry)

search_path = os.path.join(BASE_DIR, "temp/s.png")
try:
    img_search = Image.open(search_path).convert("RGBA")
    s_img = ctk.CTkImage(light_image=img_search, dark_image=img_search, size=(50, 50))
except Exception:
    s_img = None

button_search = ctk.CTkButton(
    top_frame_container, text="" if s_img else "Szukaj", image=s_img,
    fg_color="#4B6CB7", hover_color="#5B7CD9", width=50, height=50, command=search
)
if s_img:
    button_search.image = s_img  # Twarde przypisanie (ochrona przed usunięciem z RAM)
button_search.grid(row=1, rowspan=2, column=3, pady=1, sticky="w")

label_results_count = ctk.CTkLabel(
    master=app, # Zmień 'app' na nazwę ramki, w której masz przycisk Szukaj (np. left_frame, search_frame)
    text="",
    font=("Verdana", 12, "bold"),
    text_color="#888888" # Szary, żeby nie odciągał za bardzo uwagi
)
label_results_count.pack(pady=(5, 5)) # Jeśli używasz .grid(), zmień na .grid(row=X, column=Y)

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

settings_path = os.path.join(BASE_DIR, "temp/u.png")
try:
    img_settings = Image.open(settings_path).convert("RGBA")
    settings_icon_img = ctk.CTkImage(light_image=img_settings, dark_image=img_settings, size=(50, 50))
except Exception:
    settings_icon_img = None

settings_button = ctk.CTkButton(
    top_frame_container, text="" if settings_icon_img else "Opcje", image=settings_icon_img,
    fg_color="#4B6CB7", hover_color="#5B7CD9", width=50, height=50, command=settings_window
)
if settings_icon_img:
    settings_button.image = settings_icon_img  # Twarde przypisanie
settings_button.grid(row=1, rowspan=2, column=7, pady=1, sticky="w")


# Create tab view
notify_status("Budowanie interfejsu użytkownika...")
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
result_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))


warning_label = ctk.CTkLabel(
    result_frame,
    text="",
    font=("Verdana", 11, "italic"),
    text_color="#D9A04F",
    anchor="w"
)
#warning_label.pack(fill="x", padx=10, pady=(0, 5))



# Utworzenie PanedWindow (widżetu z przeciąganym separatorem)
# Top frame for pagination + entry/buttons
paned_window = tk.PanedWindow(result_frame, orient="horizontal", bg="#2C2F33", bd=0, sashwidth=8, sashcursor="size_we", opaqueresize=False)
paned_window.pack(fill="both", expand=True, padx=10, pady=(0, 10))

# Utworzenie dwóch głównych kontenerów dla lewej i prawej strony (one będą zmieniane przez separator)
left_pane = ctk.CTkFrame(paned_window, fg_color="transparent")
right_pane = ctk.CTkFrame(paned_window, fg_color="transparent")

paned_window.add(left_pane, minsize=400, stretch="always")
paned_window.add(right_pane, minsize=400, stretch="always")

# ==========================================
# LEWA STRONA (Paginacja + Tabela Wyników)
# ==========================================
pagination_frame = ctk.CTkFrame(left_pane, fg_color="#1F2328", corner_radius=12)
pagination_frame.pack(fill="x", padx=5, pady=(0, 5))

# Wspólny styl dla przycisków nawigacji
button_kwargs = dict(
    width=35,
    height=35,
    corner_radius=8,
    border_width=0,
    fg_color="#4B6CB7",
    hover_color="#5B7CD9",
    border_color=None,
    text_color="white",
    font=("Verdana", 12, 'bold'),
    anchor="center",
    hover=True,
    state="normal"
)

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

rows_label = ctk.CTkLabel(pagination_frame, text="Liczba wierszy na stronie:", font=("Verdana", 12, 'bold'), text_color="#FFFFFF")
rows_label.grid(row=1, column=5, padx=5, pady=5, sticky="e")

rows_options = ["10", "50", "100", "250", "500", "1000"]
rows_var = ctk.StringVar(value="100")
dropdown_rows = ctk.CTkOptionMenu(pagination_frame, font=("Verdana", 12, 'bold'), values=rows_options, variable=rows_var,
                                  command=update_rows_per_page, width=120, height=35,  corner_radius=8,
                                  fg_color="#4B6CB7", dropdown_fg_color="#4B6CB7", dropdown_hover_color="#3E3782", dropdown_font=("Verdana", 12, 'bold'))
dropdown_rows.grid(row=1, column=6, padx=5, pady=5, sticky="ew")

pagination_frame.grid_rowconfigure(1, weight=1)
[pagination_frame.grid_columnconfigure(i, weight=1) for i in range(7)]

# Tabela wyników zajmuje resztę lewego panelu
min_column_widths = [150, 150, 100, 150]
justify_list = ["center", "right", "center", "left"]
headers = ["Metadane", "Lewy Kontekst", "Rezultat", "Prawy Kontekst"]
data = []

text_result = table.CustomTable(left_pane, headers, data, min_column_widths, justify_list,
                                rows_per_page, fulltext_data=[], sortable=False)
text_result.set_text_anchor(["center", "e", "center", "w"])
text_result.pack(fill="both", expand=True, padx=5, pady=0)

# ==========================================
# PRAWA STRONA (Fiszki + Tekst Pełny)
# ==========================================
entry_button_frame = ctk.CTkFrame(right_pane, fg_color="#1F2328", corner_radius=12)
entry_button_frame.pack(fill="x", padx=5, pady=(0, 5))

fiszka_entrybox = ctk.CTkEntry(entry_button_frame, placeholder_text="Nazwa fiszki",
                               font=("Verdana", 12, 'bold'), height=35, corner_radius=8, fg_color="#2C2F33")
# Zmiana: pady=5
fiszka_entrybox.pack(pady=5, padx=10, fill="x", expand=True, side="left")

selected_file = ctk.StringVar(value="Otwórz fiszkę")
dropdown = ctk.CTkOptionMenu(
    entry_button_frame,
    variable=selected_file,
    values=get_txt_files(),
    command=fiszki_load_file_content,
    font=("Verdana", 12, 'bold'),
    corner_radius=8,
    width=120,
    height=35,
    fg_color="#4B6CB7",
    dropdown_fg_color="#4B6CB7",
    dropdown_hover_color="#3E3782",
    dropdown_font=("Verdana", 12, 'bold')
)
# Zmiana: pady=5
dropdown.pack(pady=5, padx=5, side="right")

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
# Zmiana: pady=5
save_selection_button.pack(pady=5, padx=5, side="right")

right_subframe = ctk.CTkFrame(right_pane, fg_color="transparent")
right_subframe.pack(fill="both", expand=True, padx=5, pady=0)
right_subframe.grid_rowconfigure(0, weight=1)
right_subframe.grid_columnconfigure(0, weight=1)

text_full = ctk.CTkTextbox(right_subframe, font=(font_family.get(), fontsize),
                           wrap="word", exportselection=False, corner_radius=12, fg_color="#1F2328")
text_full.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
text_full.bind("<FocusOut>", keep_selection)
text_full._textbox.configure(spacing1=15, spacing2=10)

buttons_action_frame = ctk.CTkFrame(right_subframe, fg_color="transparent")
buttons_action_frame.grid(row=1, column=0, sticky="ew")
buttons_action_frame.grid_columnconfigure((0, 1, 2), weight=1)

button_draw_graph = ctk.CTkButton(
    buttons_action_frame,
    text="Pokaż graf\ndrzewa zależności",
    command=show_dependency_graph,
    state="disabled",
    fg_color="#4B6CB7", hover_color="#5B7CD9", font=("Verdana", 11, 'bold')
)
button_draw_graph.grid(row=0, column=0, sticky="ew", padx=(0, 2))

button_toggle_ner = ctk.CTkButton(
    buttons_action_frame,
    text="Zaznacz jednostki\nnazwane (NER)",
    command=toggle_ner,
    state="disabled",
    fg_color="#4B6CB7", hover_color="#5B7CD9", font=("Verdana", 11, 'bold')
)
button_toggle_ner.grid(row=0, column=1, sticky="ew", padx=2)

button_toggle_coref = ctk.CTkButton(
    buttons_action_frame,
    text="Zaznacz klastry\nkoreferencyjne",
    command=toggle_coref,
    state="disabled",
    fg_color="#4B6CB7", hover_color="#5B7CD9", font=("Verdana", 11, 'bold')
)
button_toggle_coref.grid(row=0, column=2, sticky="ew", padx=(2, 0))


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

selected_table = ctk.StringVar(value="Formy podstawowe (base)")

table_selector = ctk.CTkSegmentedButton(
    tab_wyniki_frekw,
    variable=selected_table,
    values=["Formy podstawowe (base)", "Formy ortograficzne (orth)", "Częstość w czasie", "Kolokacje"],
    command=show_table,
    font=("Verdana", 12, 'bold')
)
# sticky="ew" każe mu wypełnić przestrzeń w poziomie, a pady=(10,5) usuwa wielką dziurę na górze
table_selector.grid(row=0, column=0, pady=(2, 5), padx=10, sticky="ew")

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

fq_headers = ["Nr", "Forma podstawowa (base)", "Liczba wystąpień", "Częstość wzgędna (na 1000 000 segmentów)", "Rozproszenie (DF)", "Ogólne TF-IDF"]
fq_data = []

fq_min_column_widths = [50, 150, 100, 150, 100, 100]
fq_justify_list = ["center", "center", "center", "center", "center", "center"]

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

fq_headers_token = ["Nr", "Forma tekstowa (orth)", "Liczba wystąpień", "Częstość wzgędna (na 1000 000 segmentów)", "Rozproszenie (DF)", "Ogólne TF-IDF"]
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
                    "Częstość względna", "TF-IDF", "Z-score"]
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

frekw_dane_tabela_month = table.CustomTable(month_frame, fq_headers_month, fq_data_month, [60, 60, 120, 80, 100, 80, 80],
                                            ["center"] * 7, 15, fulltext_data=[])
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

# --- PODPIĘCIE SORTOWANIA PO KLIKNIĘCIU W NAGŁÓWEK ---
frekw_dane_tabela.sort_callback = lambda col, asc: global_sort_callback(paginator_fq, col, asc)
frekw_dane_tabela_orth.sort_callback = lambda col, asc: global_sort_callback(paginator_token, col, asc)
frekw_dane_tabela_month.sort_callback = lambda col, asc: global_sort_callback(paginator_month, col, asc)

all_upos = [
    "Wszystkie", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

all_pos = [
    "Wszystkie", "subst", "depr", "adj", "adja", "adjp", "adjc", "conj", "ppron12",
    "ppron3", "siebie", "num", "fin", "bedzie", "aglt", "praet", "impt", "imps",
    "inf", "pcon", "pant", "ger", "pact", "ppas", "winien", "adv", "prep", "comp",
    "qub", "interj", "brev", "burk", "interp", "xxx", "ign"
]

# --- Ramka dla kolokacji ---
colloc_frame = ctk.CTkFrame(tab_wyniki_frekw, fg_color="#2C2F33", corner_radius=15)

colloc_controls = ctk.CTkFrame(colloc_frame, fg_color="#1F2328", corner_radius=12)
colloc_controls.grid(row=0, column=0, sticky="ew", pady=10, padx=10)

# --- Zmienne sterujące (Dodane granice zdania) ---
colloc_sort_var = ctk.StringVar(master=app, value="Log-Dice")
colloc_mode_var = ctk.StringVar(master=app, value="Liniowe")
syn_dir_var = ctk.StringVar(master=app, value="Podrzędnik")
syn_deprel_var = ctk.StringVar(master=app, value="Wszystkie")
upos_var = ctk.StringVar(master=app, value="Wszystkie")
pos_var = ctk.StringVar(master=app, value="Wszystkie")
colloc_form_var = ctk.StringVar(master=app, value="Lemat (base)")
sentence_boundary_var = ctk.BooleanVar(master=app, value=True)
colloc_ignore_case_var = ctk.BooleanVar(master=app, value=True)
colloc_dedup_var = ctk.BooleanVar(master=app, value=False)

font_ui = ("Verdana", 11, 'bold')
fg_opt = "#4B6CB7"

# GWARANCJA RÓWNEGO PODZIAŁU: 6 identycznych kolumn i 2 elastyczne wiersze
for i in range(6):
    colloc_controls.grid_columnconfigure(i, weight=1, uniform="colloc_cols")

colloc_controls.grid_rowconfigure(0, weight=1)
colloc_controls.grid_rowconfigure(1, weight=1)

# ================= WIERZ 0 =================
# Wiersz 0, Kolumna 0: Typ
cell_00 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_00.grid(row=0, column=0, sticky="ew", padx=5, pady=(10, 5))
ctk.CTkLabel(cell_00, text="Typ:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
ctk.CTkOptionMenu(cell_00, variable=colloc_mode_var, values=["Liniowe", "Składniowe"],
                  command=lambda e: toggle_colloc_mode(), width=90, height=30, fg_color=fg_opt,
                  button_color=fg_opt).pack(side="left", fill="x", expand=True)

# Wiersz 0, Kolumn 1 i 2: Dynamiczna ramka (zajmuje 2 kolumny)
cell_01 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_01.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=(10, 5))

frame_linear = ctk.CTkFrame(cell_01, fg_color="transparent")
ctk.CTkLabel(frame_linear, text="L-span:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
entry_l_span = ctk.CTkEntry(frame_linear, width=40, height=30, fg_color="#2C2F33", text_color="white", corner_radius=8)
entry_l_span.insert(0, "5")
entry_l_span.pack(side="left", padx=(0, 15))

ctk.CTkLabel(frame_linear, text="R-span:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
entry_r_span = ctk.CTkEntry(frame_linear, width=40, height=30, fg_color="#2C2F33", text_color="white", corner_radius=8)
entry_r_span.insert(0, "5")
entry_r_span.pack(side="left")

frame_syntactic = ctk.CTkFrame(cell_01, fg_color="transparent")
ctk.CTkLabel(frame_syntactic, text="Rel.:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
ctk.CTkOptionMenu(frame_syntactic, variable=syn_dir_var, values=["Podrzędnik", "Nadrzędnik", "Oba"], width=90,
                  height=30, fg_color=fg_opt, button_color=fg_opt).pack(side="left", padx=(0, 15))
ctk.CTkLabel(frame_syntactic, text="Deprel:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
ctk.CTkOptionMenu(frame_syntactic, variable=syn_deprel_var, values=DEPREL_TREE, width=120, height=30, fg_color=fg_opt,
                  button_color=fg_opt).pack(side="left", fill="x", expand=True)


def toggle_colloc_mode(*args):
    if colloc_mode_var.get() == "Liniowe":
        frame_linear.pack(side="left", fill="both", expand=True)
        frame_syntactic.pack_forget()
    else:
        frame_linear.pack_forget()
        frame_syntactic.pack(side="left", fill="both", expand=True)


toggle_colloc_mode()

# Wiersz 0, Kolumna 3: UPOS
cell_03 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_03.grid(row=0, column=3, sticky="ew", padx=5, pady=(10, 5))
ctk.CTkLabel(cell_03, text="UPOS:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
ctk.CTkOptionMenu(cell_03, variable=upos_var, values=all_upos, width=80, height=30, fg_color=fg_opt,
                  button_color=fg_opt).pack(side="left", fill="x", expand=True)

# Wiersz 0, Kolumna 4: POS
cell_04 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_04.grid(row=0, column=4, sticky="ew", padx=5, pady=(10, 5))
ctk.CTkLabel(cell_04, text="POS:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
ctk.CTkOptionMenu(cell_04, variable=pos_var, values=all_pos, width=80, height=30, fg_color=fg_opt,
                  button_color=fg_opt).pack(side="left", fill="x", expand=True)

# ================= WIERZ 1 =================
# Wiersz 1, Kolumna 0: Sort
cell_10 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_10.grid(row=1, column=0, sticky="ew", padx=5, pady=(5, 10))
ctk.CTkLabel(cell_10, text="Sort:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
ctk.CTkOptionMenu(cell_10, variable=colloc_sort_var, values=["Log-Dice", "MI Score", "T-score", "Log-Likelihood"],
                  width=90, height=30, fg_color=fg_opt, button_color=fg_opt).pack(side="left", fill="x", expand=True)

# Wiersz 1, Kolumna 1: Forma
cell_11 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_11.grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 10))
ctk.CTkLabel(cell_11, text="Forma:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
ctk.CTkOptionMenu(cell_11, variable=colloc_form_var, values=["Lemat (base)", "Token (orth)"], width=90, height=30,
                  fg_color=fg_opt, button_color=fg_opt).pack(side="left", fill="x", expand=True)

# Wiersz 1, Kolumna 2: Freq & Range
cell_12 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_12.grid(row=1, column=2, sticky="ew", padx=5, pady=(5, 10))
ctk.CTkLabel(cell_12, text="Min f:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
entry_min_freq = ctk.CTkEntry(cell_12, width=40, height=30, fg_color="#2C2F33", text_color="white", corner_radius=8)
entry_min_freq.insert(0, "1")
entry_min_freq.pack(side="left", padx=(0, 10))
ctk.CTkLabel(cell_12, text="Min r:", font=font_ui, text_color="white").pack(side="left", padx=(0, 5))
entry_min_range = ctk.CTkEntry(cell_12, width=40, height=30, fg_color="#2C2F33", text_color="white", corner_radius=8)
entry_min_range.insert(0, "1")
entry_min_range.pack(side="left")

# Wiersz 1, Kolumna 3: Checkbox Ogranicz do zdań
cell_13 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_13.grid(row=1, column=3, sticky="ew", padx=5, pady=(5, 10))
chk_sentence_bound = ctk.CTkCheckBox(cell_13, text="Ogranicz do zdań", variable=sentence_boundary_var, font=font_ui, text_color="white", width=20, height=20, fg_color="#4E8752", hover_color="#57965C")
chk_sentence_bound.pack(side="left", fill="x", expand=True)

# Wiersz 1, Kolumna 4: Checkbox Ignoruj wielkość liter
cell_14 = ctk.CTkFrame(colloc_controls, fg_color="transparent")
cell_14.grid(row=1, column=4, sticky="ew", padx=5, pady=(5, 10))
chk_ignore_case = ctk.CTkCheckBox(cell_14, text="Ignoruj wielk. liter", variable=colloc_ignore_case_var, font=font_ui, text_color="white", width=20, height=20, fg_color="#4E8752", hover_color="#57965C")
chk_ignore_case.pack(side="left", fill="x", expand=True)

# ================= PRZYCISK OBLICZ (Kolumna 5, Wiersze 0 i 1) =================
# rowspan=2 sprawia, że przycisk wchodzi na dwa rzędy. sticky="nsew" każe mu wypełnić całą tę przestrzeń (kwadraciak)
btn_calc_colloc = ctk.CTkButton(colloc_controls, text="Oblicz", command=lambda: calculate_collocs(), corner_radius=8,
                                fg_color="#4E8752", hover_color="#57965C", font=("Verdana", 14, 'bold'))
btn_calc_colloc.grid(row=0, column=5, rowspan=2, sticky="nsew", padx=(5, 10), pady=10)


# --- NOWE NAGŁÓWKI I SZEROKOŚCI TABELI ---
colloc_headers = ["Nr", "Kolokat", "Współwystąpienia", "Frekw. kolokatu", "Log-Likelihood", "MI Score", "T-score", "Log-Dice"]
colloc_widths = [50, 150, 100, 100, 120, 100, 100, 100]
colloc_justify = ["center", "center", "center", "center", "center", "center", "center", "center"]
colloc_data = []

# --- 1. NOWA RAMKA PAGINACJI DLA KOLOKACJI ---
pagination_colloc_frame = ctk.CTkFrame(colloc_frame, fg_color="#1F2328", corner_radius=12)
pagination_colloc_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)

for col in range(5):
    pagination_colloc_frame.columnconfigure(col, weight=1)

# Przyciski nawigacyjne
button_first_colloc = ctk.CTkButton(pagination_colloc_frame, text="|<", command=lambda: first_p(paginator_colloc), **button_kwargs_small)
button_first_colloc.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

button_prev_colloc = ctk.CTkButton(pagination_colloc_frame, text="<", command=lambda: prev_p(paginator_colloc), **button_kwargs_small)
button_prev_colloc.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

page_label_colloc = ctk.CTkLabel(pagination_colloc_frame, text="1/1", **label_kwargs_small)
page_label_colloc.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

button_next_colloc = ctk.CTkButton(pagination_colloc_frame, text=">", command=lambda: next_p(paginator_colloc), **button_kwargs_small)
button_next_colloc.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

button_last_colloc = ctk.CTkButton(pagination_colloc_frame, text=">|", command=lambda: last_p(paginator_colloc), **button_kwargs_small)
button_last_colloc.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

# --- 2. TABELA KOLOKACJI ---
colloc_table = table.CustomTable(
    colloc_frame, colloc_headers, colloc_data, colloc_widths, colloc_justify, 15,
    fulltext_data=[],
    search_callback=search_from_table
)
colloc_table.grid(row=2, column=0, sticky="nsew", pady=0)
colloc_table.grid(row=2, column=0, sticky="nsew", pady=0)

# --- 3. DEFINICJA PAGINATORA ---
paginator_colloc = {
    "data": colloc_data,
    "current_page": [0],
    "table": colloc_table,
    "label": page_label_colloc,
    "items_per_page": 15
}

colloc_table.sort_callback = lambda col, asc: global_sort_callback(paginator_colloc, col, asc)

# Konfiguracja wierszy i kolumn ramki głównej kolokacji
colloc_frame.rowconfigure(0, weight=0) # Panel sterowania
colloc_frame.rowconfigure(1, weight=0) # Paginacja
colloc_frame.rowconfigure(2, weight=1) # Tabela
colloc_frame.columnconfigure(0, weight=1)

# ------------------------------
# Plots
# ------------------------------
plot_options_frame = ctk.CTkFrame(tab_wyniki_wykresy, fg_color="#2C2F33", corner_radius=15)
plot_options_frame.pack(pady=(5, 10), padx=10, side="left")

# Save button container
saveplot_button_frame = ctk.CTkFrame(plot_options_frame, fg_color="#1F2328", corner_radius=12)
saveplot_button_frame.pack(pady=5, padx=5, fill="x")

# Plot type label
plot_type_label = ctk.CTkLabel(saveplot_button_frame, text="Wybierz typ wykresu:", font=("Verdana", 13, 'bold'))
plot_type_label.pack(pady=5, padx=5, fill="x")
# Single mode variable (StringVar) for plot type
wykres_sort_mode = ctk.StringVar(value="Liczba wystąpień")

# Rozwijane menu do wyboru statystyk
plot_type_menu = ctk.CTkOptionMenu(
    saveplot_button_frame,
    variable=wykres_sort_mode,
    values=["Liczba wystąpień", "Częstość względna", "TF-IDF", "Z-score"],
    font=("Verdana", 12, 'bold'),
    fg_color="#4B6CB7", dropdown_fg_color="#4B6CB7", dropdown_hover_color="#5B7CD9",
    command=lambda _: force_recalculate_plot()
)
plot_type_menu.pack(pady=5, padx=5, fill="x")

# Kontener na daty i interwał
date_settings_frame = ctk.CTkFrame(plot_options_frame, fg_color="#1F2328", corner_radius=12)
date_settings_frame.pack(pady=5, padx=5, fill="x")

# Checkbox aktywujący niestandardowe daty
custom_date_var = ctk.BooleanVar(value=False)

def toggle_custom_dates():
    state = "normal" if custom_date_var.get() else "disabled"
    date_start_entry.configure(state=state)
    date_end_entry.configure(state=state)
    force_recalculate_plot() # Opcjonalnie: odświeża od razu po kliknięciu

chk_custom_dates = ctk.CTkCheckBox(date_settings_frame, text="Niestandardowy zakres dat",
                                   variable=custom_date_var, command=toggle_custom_dates,
                                   font=("Verdana", 11, "bold"))
chk_custom_dates.pack(pady=(10, 2), anchor="w", padx=10)

# Daty w jednym wierszu
dates_row_frame = ctk.CTkFrame(date_settings_frame, fg_color="transparent")
dates_row_frame.pack(fill="x", padx=10, pady=2)

date_start_entry = ctk.CTkEntry(dates_row_frame, placeholder_text="Od (np. 01-01-2024)", height=28, state="disabled")
date_start_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)

date_end_entry = ctk.CTkEntry(dates_row_frame, placeholder_text="Do (np. 31-12-2024)", height=28, state="disabled")
date_end_entry.pack(side="left", padx=(5, 0), fill="x", expand=True)

ctk.CTkLabel(date_settings_frame, text="Interwał:", font=("Verdana", 11, "bold")).pack(pady=(5, 2))
interval_controls = ctk.CTkFrame(date_settings_frame, fg_color="transparent")
interval_controls.pack(fill="x", padx=10, pady=5)

interval_mult_entry = ctk.CTkEntry(interval_controls, width=45, height=28)
interval_mult_entry.insert(0, "1")
interval_mult_entry.pack(side="left", padx=(0, 5))

interval_unit_var = ctk.StringVar(value="Miesiąc")
interval_unit_menu = ctk.CTkOptionMenu(
    interval_controls, variable=interval_unit_var, values=["Dzień", "Miesiąc", "Rok"],
    height=28, command=lambda _: force_recalculate_plot(), fg_color="#4B6CB7"
)
interval_unit_menu.pack(side="left", fill="x", expand=True)

btn_refresh_plot = ctk.CTkButton(
    date_settings_frame, text="🔄 Odśwież wykres", font=("Verdana", 12, "bold"), height=35,
    command=force_recalculate_plot, fg_color="#4E8752", hover_color="#57965C", corner_radius=8
)
btn_refresh_plot.pack(pady=10, padx=10, fill="x")

# Checkboxes frame
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
notify_status("Przygotowywanie widoku...")
apply_theme()



# --- NOWE: Wymuszenie podziału PanedWindow na idealne 50/50 po załadowaniu okna ---
def set_initial_pane_ratio():
    try:
        # Pobieramy całkowitą szerokość kontenera z panelami
        total_width = paned_window.winfo_width()
        if total_width > 10:
            # Ustawiamy suwak (sash) o indeksie 0 dokładnie w połowie ekranu
            paned_window.sash_place(0, total_width // 2, 0)
    except Exception:
        pass


# ----------------------------------------------------------------------------------

def on_closing():
    try:
        app.quit()
        app.destroy()
    except Exception:
        pass
    finally:
        os._exit(0)

def main():
    # Pokazujemy gotowe, w pełni wyrenderowane okno
    app.state("zoomed")
    app.deiconify()
    app.update()

    # Ustawiamy proporcje po załadowaniu
    app.after(50, set_initial_pane_ratio)

    # Podpinamy zamykanie
    app.protocol("WM_DELETE_WINDOW", on_closing)

    # Odpalamy główną pętlę!
    app.mainloop()


if __name__ == "__main__":
    main()