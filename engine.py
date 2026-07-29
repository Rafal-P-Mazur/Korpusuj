"""Desktop GUI orchestration for Korpusuj.

This module owns interface state, widgets, background-task coordination and
presentation. Reusable corpus, index, search, export and semantic computation
remains in the :mod:`korpusuj` package.
"""
from korpusuj.runtime_paths import (
    ensure_user_config,
    flashcards_root,
    gui_log_dir,
    resource_root,
    writable_temp_root,
)
import os
import sys
from pathlib import Path
import subprocess
import logging
import array
import struct
# KORPUSUJ_MIGRATION_027_CORPUS_INFO_HELPERS
from korpusuj.corpus.info import (
    safe_len_or_zero as _safe_len_or_zero,
    safe_total_tokens_for_corpus_info as _safe_total_tokens_for_corpus_info,
    safe_unique_values_from_df_for_corpus_info as _safe_unique_values_from_df_for_corpus_info,
    safe_lazy_term_index_count_for_corpus_info as _safe_lazy_term_index_count_for_corpus_info,
    parse_year_month_for_corpus_info as _parse_year_month_for_corpus_info,
    normalize_monthly_counts_for_corpus_info as _normalize_monthly_counts_for_corpus_info,
    build_corpus_info_model,
)

# KORPUSUJ_MIGRATION_029B_CORPUS_LOADING_HELPERS
from korpusuj.corpus.loading import (
    search_sidecar_path as _corpus_loading_search_sidecar_path,
    prepare_loaded_corpus_bundle,
)

# KORPUSUJ_MIGRATION_030_WIRE_DEPENDENCY_RUNTIME
import korpusuj.dependency.runtime as _dependency_runtime

# KORPUSUJ_MIGRATION_031_SEARCH_STATISTICS_BOUNDARY
from korpusuj.search.statistics import (
    SearchStatistics,
    build_global_frequency_tables,
    build_monthly_frequency_tables,
    collect_search_frequency_inputs,
    normalize_monthly_token_counts_for_search,
)



os.environ["PYTHONIOENCODING"] = "utf-8"


def launch_webview(target_path: str):
    import os
    import sys
    import subprocess
    from pathlib import Path

    absolute_path = str(Path(target_path).resolve())
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--run-webview", absolute_path]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), "--run-webview", absolute_path]

    logging.info(f"launch_webview -> {absolute_path}")
    subprocess.Popen(cmd, creationflags=creationflags)

if "--run-webview" in sys.argv:
    try:
        raw_target = sys.argv[sys.argv.index("--run-webview") + 1]

        import webview
        import platform
        import urllib.parse
        from pathlib import Path

        # 1. Jeśli już dostaliśmy gotowy file:// URI -> użyj bez zmian
        if str(raw_target).startswith("file://"):
            file_url = raw_target
            parsed = urllib.parse.urlparse(raw_target)
            absolute_path = urllib.parse.unquote(parsed.path)
            if os.name == "nt" and absolute_path.startswith("/"):
                absolute_path = absolute_path.lstrip("/")
        else:
            # 2. Normalizacja ścieżki lokalnej (względnej lub bezwzględnej)
            candidate = Path(raw_target)

            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                search_bases = []

                # katalog roboczy
                search_bases.append(Path.cwd())

                # katalog skryptu / exe
                if getattr(sys, "frozen", False):
                    search_bases.append(Path(os.path.dirname(sys.executable)))
                else:
                    search_bases.append(Path(os.path.dirname(os.path.abspath(__file__))))

                # katalog zasobów PyInstaller
                if hasattr(sys, "_MEIPASS"):
                    search_bases.append(Path(sys._MEIPASS))

                resolved = None
                for base in search_bases:
                    probe = (base / raw_target).resolve()
                    if probe.exists():
                        resolved = probe
                        break

                if resolved is None:
                    resolved = (search_bases[0] / raw_target).resolve()

            absolute_path = str(resolved)
            file_url = "file:///" + urllib.parse.quote(
                absolute_path.replace("\\", "/").lstrip("/")
            )

        if os.path.exists(absolute_path):
            title = Path(absolute_path).name
            if title.lower() == "report.html":
                title = "Raport semantyczny"

            webview.create_window(
                title,
                url=file_url,
                width=1400,
                height=900,
                resizable=True,
                text_select=True
            )
        else:
            webview.create_window(
                "Błąd",
                html=f"""
                <html>
                  <body style="font-family: Arial; padding: 24px;">
                    <h2>Nie znaleziono pliku</h2>
                    <p>{absolute_path}</p>
                  </body>
                </html>
                """
            )

        if platform.system() == "Darwin":
            webview.start(gui="cocoa", debug=False)
        else:
            webview.start(debug=False)
        sys.exit(0)

    except Exception as e:
        logging.error(f"Błąd Webview: {e}")
        sys.exit(0)

if "--run-semantic-trainer" in sys.argv:
    try:
        sys.argv.remove("--run-semantic-trainer")

        # 2. Importujemy moduł (dzięki temu PyInstaller wie, że ma go spakować do .exe!)
        from korpusuj.semantic import trainer as semantic_trainer

        exit_code = semantic_trainer.main()
        sys.exit(exit_code)
    except Exception as e:

        logging.info(f"Krytyczny błąd w procesie podrzędnym trainera: {e}")
        sys.exit(1)

if "--run-semantic-report" in sys.argv:
    try:
        sys.argv.remove("--run-semantic-report")
        from korpusuj.semantic import reports_analytical_v7_1 as semantic_report
        exit_code = semantic_report.main()
        sys.exit(exit_code)
    except Exception as e:
        logging.info(f"Błąd procesu raportu semantycznego: {e}")
        sys.exit(1)

if "--run-fiszki" in sys.argv:
    try:
        val = sys.argv[sys.argv.index("--run-fiszki") + 1]
        from korpusuj.ui import fiszki_tkinter

        fiszki_tkinter.load_file_content(val)
    except Exception as e:
        logging.info(f"Błąd Fiszek: {e}")
    sys.exit(0)
# =========================================================================

os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stderr is not None:
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from logging.handlers import RotatingFileHandler
import traceback
import pandas as pd
import networkx as nx
import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import re
import json
import pickle
import threading
from PIL import Image
import shutil
from collections import Counter
import warnings
import math
from datetime import datetime, timedelta
import ast
import string
from korpusuj.ui import tables as table
from tkinter import messagebox
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta
import calendar
import time
from korpusuj.semantic.word_profile import compute_word_profile, flatten_word_profile
from korpusuj.semantic.sense_inducer import SenseInducer

def notify_status(msg):
    # Sprawdzamy, czy launcher jest uruchomiony i ma funkcję update_status
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'update_status'):
        sys.modules['__main__'].update_status(msg)

notify_status("Wczytywanie bibliotek systemowych...")


from korpusuj.topics.engine import TopicEngine

# ==========================================
# KOMPATYBILNE WRAPPERY DLA RESZTY PROGRAMU
# ==========================================
from korpusuj.semantic.engine import SemanticEngine
semantic_engine = SemanticEngine()


def on_training_success():
    notify_status("Sieć semantyczna wygenerowana! Ładowanie danych...")
    smart_show_semantic_network()


def load_semantic_neighbors():
    current_corpus_name = corpus_var.get()
    current_corpus_path = files.get(current_corpus_name)
    semantic_engine.load_neighbors(current_corpus_path)


def get_semantic_neighbors(word, top_n=25):
    return semantic_engine.get_neighbors(word, top_n)


def is_mutual_knn(u: str, v: str) -> bool:
    return semantic_engine.is_mutual_knn(u, v)


def dynamic_bridge_threshold(freq_u: int, freq_v: int, base: float = 0.55) -> float:
    return SemanticEngine.dynamic_bridge_threshold(freq_u, freq_v, base)


def smart_show_semantic_network():
    """Inteligentna funkcja łącząca w sobie logikę pytania o budowanie sieci i uruchamiania widoku."""
    current_corpus_name = corpus_var.get()
    current_corpus_path = files.get(current_corpus_name)

    if not current_corpus_path:
        messagebox.showwarning("Brak danych", "Najpierw wybierz korpus z menu po lewej stronie!")
        return

    # Jeśli nie ma na dysku plików sieci semantycznej
    if not semantic_engine.network_exists(current_corpus_path):
        ans = messagebox.askyesno(
            "Brak sieci semantycznej",
            f"Dla korpusu '{current_corpus_name}' nie wygenerowano jeszcze sieci semantycznej.\n\nCzy chcesz ją teraz zbudować?"
        )
        if ans:
            theme = THEMES[motyw.get()]
            semantic_engine.open_training_setup(app, current_corpus_name, current_corpus_path, theme,
                                                on_training_success)
        return

    # Jeśli sieć semantyczna istnieje, ładujemy ją i wyświetlamy
    if semantic_engine.index is None:
        load_semantic_neighbors()

    theme = THEMES[motyw.get()]

    def insert_to_query(w):
        current_q = entry_query.get("1.0", tk.END).strip()
        if 'Podaj zapytanie' in current_q: current_q = ""
        entry_query.delete("1.0", tk.END)
        entry_query.insert("1.0", current_q + (" || " if current_q else "") + f'[base="{w}"]')
        highlight_entry()

    from korpusuj.ui.semantic_network_viewer import SemanticNetworkViewer
    SemanticNetworkViewer(
        app,
        semantic_engine,
        theme,
        insert_to_query,
        current_corpus_name_provider=lambda: corpus_var.get(),
        current_corpus_path_provider=lambda corpus_name: files.get(corpus_name),
        open_report_callback=open_webview_window,
    )

# ==========================================
# LAZY LOADERY (Wczytywanie na żądanie)
# ==========================================
from korpusuj.ui.plots import get_plot_stack
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
        from korpusuj.corpus import creator
        _creator_module = creator

        # 5. Sprzątanie: zamykamy okienko i przywracamy kursor
        loading_win.destroy()
        app.configure(cursor="")

    return _creator_module

_fiszki_module = None
def get_fiszki_module():
    global _fiszki_module
    if _fiszki_module is None:
        from korpusuj.ui import fiszki_tkinter
        _fiszki_module = fiszki_tkinter
    return _fiszki_module



warnings.filterwarnings("ignore")
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except (ImportError, AttributeError):
    pass # Zignoruj na Mac/Linux
from korpusuj.search.models import SearchState

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

dependency_maps_cache = {}
dependency_warmup_threads = {}
dependency_warmup_stop_flags = {}
dependency_warmup_lock = threading.Lock()
DEPENDENCY_MAPS_CACHE_MAXSIZE = 50000


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
CONFIG_PATH = str(ensure_user_config(resource_root() / "config.json"))
DEFAULT_SETTINGS = {
    'font_family': 'Verdana',
    'fontsize': 14,
    'styl_wykresow': 'ciemny',
    'motyw': 'ciemny',
    'plotting': 'Tak',
    'kontekst': 250,
    'min_tokens_threshold': 0,
    'index_profile': 'full',
    'index_batch_docs': 5000,
    'dependency_cache_warmup': True,
    'dependency_cache_warmup_build_maps': True,
    'dependency_cache_warmup_materialize': False,
    'dependency_cache_ram_mode': 'none',
    'gui_search_boundary_enabled': True,
    'gui_search_boundary_fallback_on_error': True,
    'gui_search_boundary_allow_broad_regex': True,
    'gui_search_boundary_max_raw_hits': 0,
}

def _write_config_atomic(config_data):
    """Write user configuration atomically on the destination volume."""
    config_path = Path(CONFIG_PATH)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_name(config_path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, config_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# Merge a valid user mapping over the complete defaults contract. This also
# supports installer-created config files containing only ``models_dir``.
loaded_config = {}
config_needs_writeback = False
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded_value = json.load(f)
        if isinstance(loaded_value, dict):
            loaded_config = loaded_value
        else:
            config_needs_writeback = True
    except (json.JSONDecodeError, OSError):
        config_needs_writeback = True
else:
    config_needs_writeback = True

config = DEFAULT_SETTINGS.copy()
config.update(loaded_config)
if config != loaded_config:
    config_needs_writeback = True

if config_needs_writeback:
    try:
        _write_config_atomic(config)
    except OSError as exc:
        logging.warning("Nie udało się zapisać uzupełnionej konfiguracji: %s", exc)


# 3m1: backend/benchmark bez pełnego GUI też potrzebuje globalnego kontekstu.
try:
    kontekst = int((config or {}).get('kontekst', DEFAULT_SETTINGS.get('kontekst', 250)) or 250)
except Exception:
    kontekst = 250


# KORPUSUJ_PATCH_137P_GUI_LOG_PATH_TO_LOGS_GUI


# GUI logs are kept outside project root noise; old root korpusuj.log files remain historical.


LOG_DIR = str(gui_log_dir())


os.makedirs(LOG_DIR, exist_ok=True)


LOG_PATH = os.path.join(LOG_DIR, "korpusuj.log")
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

# patch_145c3c2_remove_engine_035x_130_filter_with_clean_verbose_helper: removed transitional 035X/130 handler-level logging filter.
# Diagnostics and verbose/performance logs are now source-gated.

root_logger.addHandler(log_handler)

logging.info("Logger initialized")

# KORPUSUJ_PATCH_145C1_SAFE_ENGINE_DIAGNOSTICS_IMPORT
try:
    from korpusuj.search.diagnostics import (
        korpusuj_diagnostics_enabled_145c1,
        korpusuj_verbose_diagnostics_enabled_145c1,
        korpusuj_verbose_log_145c2,
    )
except Exception:
    def korpusuj_diagnostics_enabled_145c1(config_obj=None):
        try:
            for env_name in (
                "KORPUSUJ_VERBOSE_DIAGNOSTICS", "KORPUSUJ_SEARCH_VERBOSE", "KORPUSUJ_SEARCH_MIGRATION_DEBUG",
                "KORPUSUJ_137_DIAGNOSTIC_LOGS", "KORPUSUJ_VERBOSE_EXECUTION_DIAGNOSTICS",
                "KORPUSUJ_VERBOSE_LOGS", "KORPUSUJ_VERBOSE", "KORPUSUJ_DEBUG_LOGS",
            ):
                if str(os.environ.get(env_name, "")).strip().lower() in {"1", "true", "yes", "tak", "on", "debug", "verbose"}:
                    return True
        except Exception:
            pass
        return False
    def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
        return korpusuj_diagnostics_enabled_145c1(config_obj=config_obj)
    def korpusuj_verbose_log_145c2(marker, semantic_event, message, *args, **kwargs):
        return None
# END KORPUSUJ_PATCH_145C1_SAFE_ENGINE_DIAGNOSTICS_IMPORT




notify_status("Ładowanie zasobów i czcionek...")
# Define font paths
FONT1_PATH = os.path.join(BASE_DIR, "fonts", "JetBrainsMono-Bold.ttf")
FONT2_PATH = os.path.join(BASE_DIR, "fonts", "JetBrainsMono-Regular.ttf")

# Load both fonts
ctk.FontManager.load_font(FONT1_PATH)
ctk.FontManager.load_font(FONT2_PATH)

file_path = writable_temp_root() / "temp_plot.png"
if file_path.exists():
    file_path.unlink()


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

from korpusuj.search.errors import QueryValidationError, SearchExecutionError, QueryParseError


# =========================================================================
# NOWA WYSZUKIWARKA KORPUSOWA: SQLite .search, staging postingów, async GUI
# =========================================================================
import sqlite3
import zlib
from collections import OrderedDict, defaultdict

SEARCH_INDEX_VERSION="1.8.3-profile-doc-arrays"
ENGINE_PATCH_LEVEL = "4e1-dependency-runtime-state-scaffold"
INDEX_PROFILES = {
    "compact": ("base", "orth"),
    "full": ("base", "orth", "pos", "upos", "deprel", "ner"),
}
DEFAULT_INDEXED_ATTRS = INDEX_PROFILES["full"]

# =========================================================================
# DIAGNOSTYKA WYSZUKIWANIA SQLite / legacy
# Włączanie:
#   - zmienna środowiskowa: KORPUSUJ_SEARCH_DIAG=1
#   - albo config.json: "search_diag": true
# =========================================================================
from korpusuj.search.diagnostics import (
    SEARCH_DIAG_ENV,
    configure_search_diagnostics,
    search_diag_enabled,
    search_diag_log,
)

configure_search_diagnostics(config_provider=lambda: globals().get("config", {}) or {})

from korpusuj.search.cursor_runtime import configure_search_cursor_runtime, configure_full_context_size_provider

def get_search_indexed_attrs(profile=None):
    if profile is None:
        cfg = globals().get("config", {}) or {}
        profile = os.environ.get("KORPUSUJ_INDEX_PROFILE") or cfg.get("index_profile", "full")
    profile = str(profile or "full").strip().lower()
    if profile in INDEX_PROFILES:
        return tuple(INDEX_PROFILES[profile])
    allowed = set(INDEX_PROFILES["full"])
    attrs = tuple(a.strip() for a in profile.split(",") if a.strip() in allowed)
    return attrs or DEFAULT_INDEXED_ATTRS
LINGUISTIC_COLUMNS_EXCLUDED_FROM_METADATA = {
    "Data publikacji", "Tytuł", "Autor", "tags", "Treść", "token_counts",
    "tokens", "lemmas", "deprels", "deprel", "postags", "pos", "full_postags",
    "word_ids", "sentence_ids", "head_ids", "start_ids", "start_id", "end_ids", "end_id",
    "ners", "ner", "upostags", "upos", "corefs", "srl", "srls", "srl_frames",
}


def _search_sidecar_path(parquet_path):
    return _corpus_loading_search_sidecar_path(parquet_path)


from korpusuj.utils.serialization import _json_zlib_dumps, _json_zlib_loads, _as_plain_list, _safe_scalar

def _set_loading_status(loading_label=None, message="", progress_bar=None):
    def _apply():
        try:
            if loading_label is not None:
                loading_label.configure(text=str(message))
            if progress_bar is not None:
                try:
                    progress_bar.start()
                except Exception:
                    pass
        except Exception:
            pass
    try:
        app.after(0, _apply)
    except Exception:
        _apply()


from korpusuj.index.lru import LRUCache

from korpusuj.index.postings import PostingList

from korpusuj.index.builder import SearchIndexBuilder
from korpusuj.index.sqlite_index import SearchIndex, LazyTermIndex, get_search_indexed_attrs, _search_sidecar_path






from korpusuj.search.backend import LazyCorpus


# KORPUSUJ_MIGRATION_035B_SEARCH_ENTRYPOINT_LAZYCORPUS

def _make_lazy_corpus_for_search(selected_corpus, fallback_df=None):
    """Return a LazyCorpus for GUI search when the SQLite .search sidecar is available.
    
    The indexed search path uses LazyCorpus, SearchIndex and the dependency cache, while materialized dataframes remain available to statistics and other GUI operations.
    """
    try:
        corpus_path = None
        try:
            corpus_path = (globals().get("files", {}) or {}).get(selected_corpus)
        except Exception:
            corpus_path = None
        if not corpus_path:
            search_diag_log(
                "LAZYCORPUS_SKIP corpus=%r reason=no_files_path fallback_type=%s",
                selected_corpus, type(fallback_df).__name__
            )
            return fallback_df

        try:
            corpus_path_obj = Path(corpus_path)
        except Exception:
            corpus_path_obj = corpus_path

        try:
            search_path = _search_sidecar_path(corpus_path_obj)
        except Exception:
            search_path = Path(corpus_path_obj).with_suffix(".search")

        try:
            if not Path(search_path).exists():
                search_diag_log(
                    "LAZYCORPUS_SKIP corpus=%r parquet_path=%r search_path=%r reason=missing_search_sidecar",
                    selected_corpus, str(corpus_path_obj), str(search_path)
                )
                return fallback_df
        except Exception:
            pass

        attempts = []
        try:
            import inspect
            sig = inspect.signature(LazyCorpus)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            names = {p.name for p in params}
            kwargs = {}
            if "corpus_name" in names:
                kwargs["corpus_name"] = selected_corpus
            if "name" in names:
                kwargs["name"] = selected_corpus
            if "parquet_path" in names:
                kwargs["parquet_path"] = corpus_path_obj
            if "corpus_path" in names:
                kwargs["corpus_path"] = corpus_path_obj
            if "path" in names:
                kwargs["path"] = corpus_path_obj
            if "search_path" in names:
                kwargs["search_path"] = search_path
            if "index_path" in names:
                kwargs["index_path"] = search_path
            if "dataframe" in names:
                kwargs["dataframe"] = fallback_df
            if "df" in names:
                kwargs["df"] = fallback_df
            if kwargs:
                attempts.append(((), kwargs, "signature_kwargs"))
        except Exception:
            pass

        attempts.extend([
            ((corpus_path_obj,), {}, "parquet_path"),
            ((str(corpus_path_obj),), {}, "parquet_path_str"),
            ((corpus_path_obj, search_path), {}, "parquet_path_search_path"),
            ((str(corpus_path_obj), str(search_path)), {}, "parquet_path_search_path_str"),
            ((selected_corpus, corpus_path_obj), {}, "name_parquet_path"),
            ((selected_corpus, corpus_path_obj, search_path), {}, "name_parquet_path_search_path"),
        ])

        last_error = None
        for args, kwargs, label in attempts:
            try:
                lazy = LazyCorpus(*args, **kwargs)
                search_diag_log(
                    "LAZYCORPUS_READY corpus=%r constructor=%s parquet_path=%r search_path=%r lazy_type=%s",
                    selected_corpus, label, str(corpus_path_obj), str(search_path), type(lazy).__name__
                )
                return lazy
            except Exception as e:
                last_error = e

        search_diag_log(
            "LAZYCORPUS_FAIL corpus=%r parquet_path=%r search_path=%r reason=%r fallback_type=%s",
            selected_corpus, str(corpus_path_obj), str(search_path), last_error, type(fallback_df).__name__
        )
        return fallback_df
    except Exception as e:
        search_diag_log(
            "LAZYCORPUS_FAIL_UNEXPECTED corpus=%r reason=%r fallback_type=%s",
            selected_corpus, e, type(fallback_df).__name__
        )
        return fallback_df



from korpusuj.search import parser as cql_parser
from korpusuj.search.planner import SearchPlanner

def _sync_cql_parser_exceptions():
    try:
        cql_parser.QueryParseError = QueryParseError
    except Exception:
        pass

def split_top_level(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.split_top_level(*args, **kwargs)

def find_top_level_operator(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.find_top_level_operator(*args, **kwargs)

def parse_single_condition(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.parse_single_condition(*args, **kwargs)

def parse_conditions(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.parse_conditions(*args, **kwargs)

def extract_square_brackets(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.extract_square_brackets(*args, **kwargs)

def parse_query_group(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.parse_query_group(*args, **kwargs)

def parse_sentence_conditions(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.parse_sentence_conditions(*args, **kwargs)

def parse_frequency_attributes(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.parse_frequency_attributes(*args, **kwargs)

def parse_frequency_attribute(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.parse_frequency_attribute(*args, **kwargs)

def parse_frequency_base_attribute(*args, **kwargs):
    _sync_cql_parser_exceptions()
    return cql_parser.parse_frequency_base_attribute(*args, **kwargs)


from korpusuj.search.cursor import SearchCursor
def _is_searchcursor_like(value):
    """True for ordinary SearchCursor and UnionSearchCursor lazy cursors."""
    try:
        if isinstance(value, SearchCursor):
            return True
    except Exception:
        pass
    try:
        if getattr(value, "_is_union_searchcursor", False):
            return True
    except Exception:
        pass
    try:
        if type(value).__name__ == "UnionSearchCursor":
            return True
    except Exception:
        pass
    return False



from korpusuj.search.diagnostics import summarize_search_plan_for_log

from korpusuj.search.executor import SearchExecutor, CorpusSearchExecutor, configure_search_executor

configure_search_executor(
    search_cursor_cls=SearchCursor,
    search_index_cls=SearchIndex,
)


def ensure_legacy_inverted_index_for_corpus(corpus_name, df):
    idx = inverted_indexes.get(corpus_name, {})
    if isinstance(idx.get("base"), dict) and isinstance(idx.get("orth"), dict):
        return

    def _iter_values_safe(value):
        if value is None:
            return []
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            pass
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, str):
            return [value]
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        return [value]

    base_idx, orth_idx = {}, {}
    for row in df.itertuples():
        row_id = row.Index
        for lemma in set(_iter_values_safe(getattr(row, "lemmas", None))):
            base_idx.setdefault(lemma, set()).add(row_id)
        for token in set(_iter_values_safe(getattr(row, "tokens", None))):
            orth_idx.setdefault(token, set()).add(row_id)
    idx["base"] = base_idx
    idx["orth"] = orth_idx
    inverted_indexes[corpus_name] = idx


# =========================================================================
from korpusuj.dependency.runtime_state import configure_dependency_runtime_state
# BACKGROUND DEPENDENCY CACHE WARMUP
# =========================================================================

def _cfg_bool(name, default=False):
    try:
        cfg = globals().get("config", {}) or {}
        val = cfg.get(name, default)
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "tak", "on")
        return bool(val)
    except Exception:
        return bool(default)


def _safe_dependency_progress(message, progress_callback=None):
    """Thread-safe status dla dependency cache."""
    msg = str(message)
    if progress_callback is not None:
        try:
            progress_callback(msg)
            return
        except Exception:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] msg=%r", msg, exc_info=True)
    try:
        app.after(0, lambda m=msg: notify_status(m))
    except Exception:
        try:
            notify_status(msg)
        except Exception:
            pass


DEPENDENCY_DISK_CACHE_VERSION = "depmap-3f-parent-int32-v1"
DEPENDENCY_LEGACY_DISK_CACHE_VERSION = "depmap-3d-pickle-v1"
DEPENDENCY_PARENT_MAGIC = b"DP3F_PARENT_I32_V1\x00"
dependency_disk_caches = {}

DEPENDENCY_RAM_USAGE_LABELS = {
    "Oszczędny": "none",
    "Maksymalna wydajność": "all",
}
DEPENDENCY_RAM_MODE_LABELS = {v: k for k, v in DEPENDENCY_RAM_USAGE_LABELS.items()}
DEFAULT_DEPENDENCY_RAM_USAGE_LABEL = "Oszczędny"
DEFAULT_DEPENDENCY_RAM_MODE = "none"
DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE = 500
DEPENDENCY_CANDIDATE_RAM_BUDGET_MB = 512
DEPENDENCY_CANDIDATE_MAX_DOCS = 3000
DEPENDENCY_CANDIDATE_STREAM_BATCH_DOCS = 256


# 4e.1: dependency runtime state scaffold. This mirrors existing globals without
# changing runtime behavior. Actual helper migration is deferred to later 4e.x.
configure_dependency_runtime_state(
    dependency_maps_cache=dependency_maps_cache,
    dependency_disk_caches=dependency_disk_caches,
    dependency_warmup_threads=dependency_warmup_threads,
    dependency_warmup_stop_flags=dependency_warmup_stop_flags,
    dependency_warmup_lock=dependency_warmup_lock,
    maps_cache_maxsize=DEPENDENCY_MAPS_CACHE_MAXSIZE,
    candidate_max_docs=DEPENDENCY_CANDIDATE_MAX_DOCS,
    candidate_stream_batch_docs=DEPENDENCY_CANDIDATE_STREAM_BATCH_DOCS,
    candidate_ram_budget_mb=DEPENDENCY_CANDIDATE_RAM_BUDGET_MB,
    cache_preload_batch_size=DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE,
    default_ram_mode=DEFAULT_DEPENDENCY_RAM_MODE,
    default_ram_usage_label=DEFAULT_DEPENDENCY_RAM_USAGE_LABEL,
    ram_usage_labels=DEPENDENCY_RAM_USAGE_LABELS,
    ram_mode_labels=DEPENDENCY_RAM_MODE_LABELS,
    disk_cache_version=DEPENDENCY_DISK_CACHE_VERSION,
    legacy_disk_cache_version=DEPENDENCY_LEGACY_DISK_CACHE_VERSION,
    parent_magic=DEPENDENCY_PARENT_MAGIC,
)


def _dependency_label_to_mode(label):
    mode = DEPENDENCY_RAM_USAGE_LABELS.get(str(label or "").strip(), DEFAULT_DEPENDENCY_RAM_MODE)
    return mode if mode in ("none", "all") else DEFAULT_DEPENDENCY_RAM_MODE


def _dependency_mode_to_label(mode):
    mode = str(mode or "").strip().lower()
    if mode == "candidate": mode = "none"
    return DEPENDENCY_RAM_MODE_LABELS.get(mode, DEFAULT_DEPENDENCY_RAM_USAGE_LABEL)


def _get_dependency_cache_ram_mode():
    try:
        var = globals().get("dependency_ram_usage_var")
        if var is not None:
            mode = _dependency_label_to_mode(var.get())
            return mode if mode in ("none", "all") else DEFAULT_DEPENDENCY_RAM_MODE
    except Exception: pass
    try:
        cfg = globals().get("config", {}) or {}
        mode = str(cfg.get("dependency_cache_ram_mode", DEFAULT_DEPENDENCY_RAM_MODE) or DEFAULT_DEPENDENCY_RAM_MODE).strip().lower()
        if mode == "candidate": mode = "none"
        if mode in ("none", "all"): return mode
    except Exception: pass
    return DEFAULT_DEPENDENCY_RAM_MODE

try:
    if str((globals().get("config", {}) or {}).get("dependency_cache_ram_mode", DEFAULT_DEPENDENCY_RAM_MODE)).strip().lower() == "candidate":
        config["dependency_cache_ram_mode"] = DEFAULT_DEPENDENCY_RAM_MODE
except Exception: pass


def _dependency_cache_corpus_name_from_path(corpus_path):
    """Stabilny klucz RAM cache dla map dependency.

    Warmup używa nazwy korpusu, natomiast SearchCursor zna głównie ścieżkę parquetu.
    Dopasowujemy więc ścieżkę do globalnego files, żeby tryby none/all korzystały
    z tego samego dependency_maps_cache.
    """
    try:
        target = str(Path(corpus_path).resolve()) if corpus_path else ""
        for name, path in (globals().get("files", {}) or {}).items():
            try:
                if str(Path(path).resolve()) == target:
                    return name
            except Exception:
                if str(path) == str(corpus_path):
                    return name
    except Exception:
        pass
    return str(corpus_path or "")


def _dependency_ram_cache_size_for_corpus(corpus_name=None):
    try:
        if corpus_name is None:
            return len(dependency_maps_cache)
        return sum(1 for k in dependency_maps_cache if isinstance(k, tuple) and k and k[0] == corpus_name)
    except Exception:
        return 0


def _get_search_index_cache_sizes():
    """3e: rozmiary cache SQLite dopasowane do trybu użycia RAM.

    Wyszukiwanie prostych zapytań nadal opiera się o plik .search/SQLite, więc w trybie
    Oszczędny możemy znacząco zmniejszyć LRU bez dużej utraty szybkości.
    """
    mode = _get_dependency_cache_ram_mode()
    if mode == "all":
        return 1024, 4096
    return 32, 64


def _get_lazy_term_index_cache_sizes():
    """3e: mniejszy cache dla pojedynczych LazyTermIndex w trybie Oszczędny."""
    mode = _get_dependency_cache_ram_mode()
    if mode == "none":
        return 16, 16
    if mode == "all":
        return 128, 128
    return 64, 32


def _query_uses_dependency_maps(query):
    try:
        q = str(query or "").lower()
        return ("dependent" in q) or ("head" in q)
    except Exception:
        return False


def _as_list_for_warmup(value):
    if value is None:
        return []
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


from korpusuj.dependency.maps import LazyChildrenLookup, build_dependency_maps
from korpusuj.dependency.disk_cache import DependencyMapDiskCache, _dependency_cache_source_signature, _dependency_cache_path_for_corpus_path
from korpusuj.dependency.policy import DEPENDENCY_DISK_CACHE_VERSION, DEPENDENCY_LEGACY_DISK_CACHE_VERSION, DEPENDENCY_PARENT_MAGIC


def _dependency_runtime_call(func_name, *args, **kwargs):
    impl_map = {
        "get_dependency_disk_cache_for_corpus": "_get_dependency_disk_cache_for_corpus_impl",
        "_clear_dependency_ram_cache_for_corpus": "__clear_dependency_ram_cache_for_corpus_impl",
        "_put_dependency_ram_cache": "__put_dependency_ram_cache_impl",
        "preload_dependency_maps_for_candidates": "_preload_dependency_maps_for_candidates_impl",
        "preload_all_dependency_maps_for_corpus": "_preload_all_dependency_maps_for_corpus_impl",
        "_cache_dependency_maps_for_row": "__cache_dependency_maps_for_row_impl",
        "_select_dependency_parquet_columns": "__select_dependency_parquet_columns_impl",
        "build_dependency_cache_from_parquet_batches": "_build_dependency_cache_from_parquet_batches_impl",
        "warm_dependency_cache_for_corpus": "_warm_dependency_cache_for_corpus_impl",
        "start_dependency_cache_warmup": "_start_dependency_cache_warmup_impl",
    }
    impl_name = impl_map.get(func_name)
    if impl_name and hasattr(_dependency_runtime, impl_name):
        impl = getattr(_dependency_runtime, impl_name)
        try:
            _dependency_runtime.__dict__.update(globals())
        except Exception:
            pass
        return impl(*args, **kwargs)
    func = getattr(_dependency_runtime, func_name)
    import inspect
    try:
        params = list(inspect.signature(func).parameters)
        if params and params[0] == "engine_globals":
            return func(globals(), *args, **kwargs)
    except Exception:
        pass
    return func(*args, **kwargs)

def get_dependency_disk_cache_for_corpus(corpus_name):
    return _dependency_runtime_call("get_dependency_disk_cache_for_corpus", corpus_name)


def _clear_dependency_ram_cache_for_corpus(corpus_name=None):
    return _dependency_runtime_call("_clear_dependency_ram_cache_for_corpus", corpus_name)


def _put_dependency_ram_cache(cache_key, dep_maps):
    return _dependency_runtime_call("_put_dependency_ram_cache", cache_key, dep_maps)


def preload_dependency_maps_for_candidates(corpus_name, doc_ids, diag=None, batch_size=DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE):
    return _dependency_runtime_call(
        "preload_dependency_maps_for_candidates", corpus_name, doc_ids, diag=diag, batch_size=batch_size
    )


# 4d.2: SearchCursor runtime boundary. Dependency runtime stays in engine.py.
configure_search_cursor_runtime(
    dependency_cache_corpus_name_from_path=_dependency_cache_corpus_name_from_path,
    get_dependency_cache_ram_mode=_get_dependency_cache_ram_mode,
    dependency_ram_cache_size_for_corpus=_dependency_ram_cache_size_for_corpus,
    put_dependency_ram_cache=_put_dependency_ram_cache,
    preload_dependency_maps_for_candidates=preload_dependency_maps_for_candidates,
    dependency_maps_cache=dependency_maps_cache,
    candidate_max_docs=DEPENDENCY_CANDIDATE_MAX_DOCS,
    candidate_stream_batch_docs=DEPENDENCY_CANDIDATE_STREAM_BATCH_DOCS,
    full_context_size=globals().get("kontekst", 250),
)

# 4d.3 note: SearchCursor is externalized; keep dependency runtime in engine.py.
# 4d.2.1: keep extended context dynamic; UI/settings can change after startup.
configure_full_context_size_provider(
    lambda: int(globals().get("kontekst", (globals().get("config", {}) or {}).get("kontekst", 250)) or 250)
)


def preload_all_dependency_maps_for_corpus(corpus_name, disk_cache=None, diag=None):
    return _dependency_runtime_call(
        "preload_all_dependency_maps_for_corpus", corpus_name, disk_cache=disk_cache, diag=diag
    )


def _cache_dependency_maps_for_row(corpus_name, row, diag=None, disk_cache=None, store_ram=False, commit=True):
    return _dependency_runtime_call(
        "_cache_dependency_maps_for_row", corpus_name, row, diag=diag, disk_cache=disk_cache, store_ram=store_ram, commit=commit
    )



def _select_dependency_parquet_columns(parquet_path):
    return _dependency_runtime_call("_select_dependency_parquet_columns", parquet_path)


def build_dependency_cache_from_parquet_batches(
    corpus_name,
    parquet_path,
    disk_cache=None,
    batch_docs=5000,
    progress_callback=None,
    diag=None,
    stop_flag_getter=None,
):
    return _dependency_runtime_call(
        "build_dependency_cache_from_parquet_batches",
        corpus_name,
        parquet_path,
        disk_cache=disk_cache,
        batch_docs=batch_docs,
        progress_callback=progress_callback,
        diag=diag,
        stop_flag_getter=stop_flag_getter,
    )

def warm_dependency_cache_for_corpus(corpus_name, build_maps=True, materialize=False, progress_callback=None):
    return _dependency_runtime_call(
        "warm_dependency_cache_for_corpus",
        corpus_name,
        build_maps=build_maps,
        materialize=materialize,
        progress_callback=progress_callback,
    )


def start_dependency_cache_warmup(corpus_name, build_maps=None, materialize=None):
    return _dependency_runtime_call(
        "start_dependency_cache_warmup", corpus_name, build_maps=build_maps, materialize=materialize
    )




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


# ==========================================
# CQL AUTOCOMPLETE
# ==========================================
from korpusuj.ui.autocomplete import CQLAutocomplete

# ==========================================
# UI TOOLTIP
# ==========================================
from korpusuj.ui.tooltip import ToolTip

def calc_z_score(val, mean_val, std_val):
    """Zwraca Z-score lub None w przypadku braku wariancji."""
    return ((val - mean_val) / std_val) if std_val > 0 else None

def safe_ll(o, e):
    """Bezpieczne log-likelihood bez ryzyka dzielenia przez zero."""
    return o * math.log(o / e) if o > 0 and e > 0 else 0.0

def calc_pmw(frequency, total_tokens):
    """Częstość na milion słów (Per Million Words)."""
    return (frequency / total_tokens) * 1_000_000 if total_tokens > 0 else 0.0




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
# KORPUSUJ_MIGRATION_PATCH_111_LAZY_FULLTEXT_CONTEXT_ON_CLICK

def _resolve_lazy_fulltext_payload_111(full_text_or_ref, context=None):
    """Resolve lazy SearchCursor fulltext payload for row-click display."""
    try:
        from korpusuj.search.cursor import resolve_lazy_fulltext_ref_111
        return resolve_lazy_fulltext_ref_111(full_text_or_ref, context)
    except Exception:
        return full_text_or_ref


def _resolve_lazy_fulltext_rows_for_export_111(rows):
    """Resolve lazy fulltext/context rows only for explicit export."""
    try:
        from korpusuj.search.cursor import resolve_result_row_fulltext_111
    except Exception:
        return rows
    try:
        return [resolve_result_row_fulltext_111(row) for row in rows]
    except Exception:
        try:
            return list(rows)
        except Exception:
            return rows

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


    # SearchCursor: leniwe stronicowanie; kontekst i pełny tekst cięte po indeksach znaków.
    if _is_searchcursor_like(full_results_sorted):
        start_index = current_page * rows_per_page
        end_index = start_index + rows_per_page
        page_results = full_results_sorted.get_range(start_index, end_index)
        if not page_results:
            text_result.set_data([("", "Brak wyników dla zapytania:", query, "")])
            text_result.set_fulltext_data([])
            page_label.configure(text="0/0")
            button_first.configure(state="disabled"); button_prev.configure(state="disabled")
            button_next.configure(state="disabled"); button_last.configure(state="disabled")
            return
        new_data, full_data = [], []
        for idx, (publication_date, context, full_text, matched_text, matched_lemmas, month_key, title, author,
                  additional_metadata, left_context, right_context, row_idx, start_idx_val, end_idx_val) in enumerate(page_results, start=start_index + 1):
            matched_text = str(matched_text).replace("\n", " ")
            left_context = str(left_context).replace("\n", " ")
            right_context = str(right_context).replace("\n", " ")
            title_s = str(title)
            metadata = f"{idx}. (A: {author}; T: {title_s[:15]}{'...' if len(title_s) > 15 else ''}; D: {publication_date})"
            new_data.append((metadata, left_context, matched_text, right_context))
            full_data.append((full_text, context, publication_date, title, author, additional_metadata, row_idx, start_idx_val))
        text_result.set_data(new_data)
        text_result.set_fulltext_data(full_data)
        def handle_row_click(row_index):
            if 0 <= row_index - 1 < len(text_result.fulltext_data):
                fdata = text_result.fulltext_data[row_index - 1]
                full_text_111 = _resolve_lazy_fulltext_payload_111(fdata[0], fdata[1])

                display_full_text(full_text_111, fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7])
        text_result.set_additional_event(handle_row_click)
        total_est = len(full_results_sorted)
        total_pages = max(1, math.ceil(total_est / rows_per_page)) if total_est else 1
        page_label.configure(text=f"{current_page + 1}/{total_pages}")
        button_first.configure(state="disabled" if current_page == 0 else "normal")
        button_prev.configure(state="disabled" if current_page == 0 else "normal")
        has_next = len(page_results) == rows_per_page and (end_index < total_est or getattr(full_results_sorted, '_count_cache', None) is None)
        button_next.configure(state="normal" if has_next else "disabled")
        count_known = getattr(full_results_sorted, "_count_cache", None) is not None

        estimate_exact = (
                hasattr(full_results_sorted, "count_hits_estimate_is_exact")
                and full_results_sorted.count_hits_estimate_is_exact()
        )

        can_jump_to_last = count_known or estimate_exact

        button_last.configure(
            state="normal"
            if can_jump_to_last and current_page + 1 < total_pages
            else "disabled"
        )
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
            full_text_111 = _resolve_lazy_fulltext_payload_111(fdata[0], fdata[1])

            display_full_text(full_text_111, fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7])

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

    if _is_searchcursor_like(full_results_sorted):
        count_known = getattr(full_results_sorted, "_count_cache", None) is not None

        estimate_exact = (
            hasattr(full_results_sorted, "count_hits_estimate_is_exact")
            and full_results_sorted.count_hits_estimate_is_exact()
        )

        if not count_known and not estimate_exact:
            # Dla złożonych zapytań dokładna ostatnia strona wymaga przeliczenia całości.
            # Możesz albo zostawić return, albo wymusić dokładne liczenie:
            return

            # Alternatywnie, jeśli akceptujesz koszt czasu:
            # full_results_sorted.count_hits(exact=True)

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


# --- Main Function: find_lemma_context ---
def _legacy_find_lemma_context(query, df, selected_corpus, left_context_size=10, right_context_size=10, warnings_list=None):
    t_find_start = time.perf_counter()
    legacy_diag = {
        "query": query,
        "rows_total": 0,
        "rows_prefilter_union": 0,
        "rows_after_metadata": 0,
        "rows_scanned": 0,
        "tokens_checked": 0,
        "head_checks": 0,
        "dependent_checks": 0,
        "dep_maps_built": 0,
        "dep_maps_ram_hits": 0,
        "anchor_candidates": 0,
        "anchor_mode_docs": 0,
        "matches": 0,
        "time_prefilter": 0.0,
        "time_dep_maps": 0.0,
        "time_match_conditions_success_path": 0.0,
        "time_total": 0.0,
    }
    try:
        legacy_diag["rows_total"] = len(df)
    except Exception:
        pass
    if warnings_list is None:
        warnings_list = []

    global search_status

    # Wymuszenie odświeżenia UI (pokazanie ekranu ładowania) tylko gdy działa GUI.
    # Benchmark/backend bez Tkintera nie powinien wymagać text_result.
    try:
        tr = globals().get("text_result")
        if tr is not None:
            tr.after(0, lambda: display_page(query, selected_corpus))
    except Exception:
        pass

    # Pobranie opcji frekwencyjnych z zapytania
    freq_opts = parse_frequency_attributes(query, "frequency_orth")
    freq_base_opts = parse_frequency_attributes(query, "frequency_base")

    # NAPRAWA: Usunięcie tagów frekwencyjnych z zapytania po ich wczytaniu
    query = re.sub(r'<frequency_orth\s+[^>]+>', '', query, flags=re.IGNORECASE).strip()
    query = re.sub(r'<frequency_base\s+[^>]+>', '', query, flags=re.IGNORECASE).strip()

    # KORPUSUJ_MIGRATION_036L4G42_LEGACY_METADATA_OPERATOR_NORMALIZATION
    def extract_filters(q, tag):
        filters = []
        regex_meta = re.compile(r'[\.\^\$\*\+\?\{\}\[\]\|\\\(\)]')

        def classify(val):
            if val.startswith("~") and len(val) > 1:
                return "regex_search", val[1:]
            if regex_meta.search(val):
                return "regex", val
            return "exact", val

        def add_filter(op, raw):
            mt, norm = classify(raw)
            filters.append((op, norm, mt))
            return ""

        # KORPUSUJ_MIGRATION_036L4G42E_LEGACY_SPLIT_GT
        split_ge = re.compile(r'<\s*' + re.escape(tag) + r'\s*>\s*=\s*"([^"]+)"\s*>', flags=re.IGNORECASE)
        q = split_ge.sub(lambda m: add_filter(">=", m.group(1)), q)

        split_gt = re.compile(r'<\s*' + re.escape(tag) + r'\s*>\s*"([^"]+)"\s*>', flags=re.IGNORECASE)
        q = split_gt.sub(lambda m: add_filter(">", m.group(1)), q)

        canonical = re.compile(
            r'<\s*' + re.escape(tag) + r'\s*(?:(<=|>=|!=|=)\s*|([<>])\s+)"([^"]+)"\s*>',
            flags=re.IGNORECASE,
        )
        q = canonical.sub(lambda m: add_filter(m.group(1) or m.group(2), m.group(3)), q)
        return q.strip(), filters

    # usage:
    query, author_filters = extract_filters(query, "autor")
    query, title_filters = extract_filters(query, "tytuł")
    query, date_filters = extract_filters(query, "data")

    # KORPUSUJ_MIGRATION_036L4G42_LEGACY_METADANE_OPERATOR_NORMALIZATION
    def extract_metadane_filters(q):
        filters = []
        regex_meta = re.compile(r'[\.\^\$\*\+\?\{\}\[\]\|\\\(\)]')
        col_pat = r'(\w+)'

        def classify(val):
            if val.startswith("~") and len(val) > 1:
                return "regex_search", val[1:]
            if regex_meta.search(val):
                return "regex", val
            return "exact", val

        def add_filter(col, op, raw):
            mt, norm = classify(raw)
            filters.append((col, op, norm, mt))
            return ""

        split_ge = re.compile(r'<\s*metadane:' + col_pat + r'\s*>\s*=\s*"([^"]+)"\s*>', flags=re.IGNORECASE)
        q = split_ge.sub(lambda m: add_filter(m.group(1), ">=", m.group(2)), q)

        split_gt = re.compile(r'<\s*metadane:' + col_pat + r'\s*>\s*"([^"]+)"\s*>', flags=re.IGNORECASE)
        q = split_gt.sub(lambda m: add_filter(m.group(1), ">", m.group(2)), q)

        canonical = re.compile(
            r'<\s*metadane:' + col_pat + r'\s*(?:(<=|>=|!=|=)\s*|([<>])\s+)"([^"]+)"\s*>',
            flags=re.IGNORECASE,
        )
        q = canonical.sub(lambda m: add_filter(m.group(1), m.group(2) or m.group(3), m.group(4)), q)
        return q.strip(), filters

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
                        val_rows.update(corpus_index.get(key, {}).get(val, set()))
                    block_rows.intersection_update(val_rows)
                    has_constraints = True
                elif match_type == "exact" and (key.startswith("head") or key.startswith("dependent")):
                    # Bezpieczny prefiltr dokumentowy dla relacji składniowych:
                    # jeśli token ma head/dependent="X", dokument musi zawierać lemat X.
                    # Relację właściwą nadal weryfikuje legacy matcher.
                    val_rows = set()
                    for val in values:
                        val_rows.update(corpus_index.get("base", {}).get(val, set()))
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

    # --- PREFILTER: liczony raz per grupa, a maska metadanych raz globalnie ---
    group_jobs = []
    all_valid_row_ids = set()

    _pref_t0 = time.perf_counter()
    for group_tuple in parsed_query_groups:
        # Używamy prefiltru tylko dla tej konkretnej grupy
        group_row_ids = get_prefiltered_rows([group_tuple], selected_corpus, df.index)
        group_jobs.append((*group_tuple, group_row_ids))
        if group_row_ids:
            all_valid_row_ids.update(group_row_ids)
    _pref_t1 = time.perf_counter()

    legacy_diag["time_prefilter"] += (_pref_t1 - _pref_t0)
    legacy_diag["rows_prefilter_union"] = len(all_valid_row_ids)
    search_diag_log(
        "LEGACY_PREFILTER query=%r rows_total=%s rows_prefilter_union=%s time=%.6fs",
        query, legacy_diag.get("rows_total"), legacy_diag.get("rows_prefilter_union"), legacy_diag.get("time_prefilter")
    )

    if not all_valid_row_ids:
        legacy_diag["time_total"] = time.perf_counter() - t_find_start
        search_diag_log("LEGACY_DONE_NO_PREFILTER_HITS query=%r diag=%r", query, legacy_diag)
        return []

    # 1) Jeden wspólny koszyk kandydatów dla wszystkich grup
    filtered_df_base = df.loc[list(all_valid_row_ids)].copy()

    # 2) Jedna maska metadanych liczona tylko raz
    mask = pd.Series(True, index=filtered_df_base.index)

    # --- Author filters ---
    if author_filters:
        if 'Autor' not in filtered_df_base.columns:
            add_warning(warnings_list, 'Filtr "autor" został pominięty: w korpusie brak kolumny "Autor".')
        else:
            author_series = filtered_df_base['Autor'].astype(str).str.lower()
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
        if 'Tytuł' not in filtered_df_base.columns:
            add_warning(warnings_list, 'Filtr "tytuł" został pominięty: w korpusie brak kolumny "Tytuł".')
        else:
            title_series = filtered_df_base['Tytuł'].astype(str).str.lower()
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
        if 'Data publikacji' not in filtered_df_base.columns:
            add_warning(warnings_list, 'Filtr "data" został pominięty: w korpusie brak kolumny "Data publikacji".')
        else:
            date_series = filtered_df_base['Data publikacji'].astype(str).str[:10]
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
            if column not in filtered_df_base.columns:
                add_warning(warnings_list, f'Filtr metadanych został pominięty: brak kolumny "{column}".')
                continue
            series = filtered_df_base[column].astype(str).str.lower()
            val = value.lower()
            if op in ("<", "<=", ">", ">="):
                # KORPUSUJ_MIGRATION_036L4G42G_LEGACY_METADATA_RANGE_DATE_THRESHOLD_NORMALIZATION
                try:
                    _m_date_threshold_036l4g42g = re.fullmatch(r"(\d{4})[-./](\d{2})[-./](\d{2})", val.strip())
                    if _m_date_threshold_036l4g42g:
                        val = "-".join(_m_date_threshold_036l4g42g.groups())
                except Exception:
                    pass
                # KORPUSUJ_MIGRATION_036L4G42F_LEGACY_METADATA_RANGE_WILDCARD_THRESHOLDS
                wildcard_prefix = None
                try:
                    if val.endswith(".*") and len(val) > 2:
                        wildcard_prefix = val[:-2]
                except Exception:
                    wildcard_prefix = None
                if wildcard_prefix is not None:
                    in_bucket = series.str.startswith(wildcard_prefix, na=False)
                    if op == ">=":
                        submask = in_bucket | (series > wildcard_prefix)
                    elif op == ">":
                        submask = (~in_bucket) & (series > wildcard_prefix)
                    elif op == "<=":
                        submask = in_bucket | (series < wildcard_prefix)
                    else:
                        submask = (~in_bucket) & (series < wildcard_prefix)
                elif op == "<":
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

    # ✅ Apply all metadata filters once
    filtered_df_base = filtered_df_base[mask]
    try:
        legacy_diag["rows_after_metadata"] = len(filtered_df_base)
    except Exception:
        pass
    search_diag_log(
        "LEGACY_AFTER_METADATA query=%r rows_after_metadata=%s",
        query, legacy_diag.get("rows_after_metadata")
    )

    if _query_uses_dependency_maps(query) and _get_dependency_cache_ram_mode() == "candidate":
        preload_dependency_maps_for_candidates(selected_corpus, filtered_df_base.index, diag=legacy_diag)

    # 3) Słowniki metadanych budujemy też tylko raz, na odfiltrowanym koszyku
    dates_dict = filtered_df_base[
        "Data publikacji"].to_dict() if "Data publikacji" in filtered_df_base.columns else {}
    titles_dict = filtered_df_base["Tytuł"].to_dict() if "Tytuł" in filtered_df_base.columns else {}
    authors_dict = filtered_df_base["Autor"].to_dict() if "Autor" in filtered_df_base.columns else {}

    exclude_cols = {
        "Data publikacji", "Tytuł", "Autor", "tags", "Treść", "token_counts",
        "tokens", "lemmas", "deprels", "postags", "full_postags",
        "word_ids", "sentence_ids", "head_ids", "start_ids", "end_ids", "ners", "upostags",
        "corefs", "srl", "srls", "srl_frames"
    }
    meta_columns = [col for col in filtered_df_base.columns if col not in exclude_cols]
    meta_dicts = {col: filtered_df_base[col].to_dict() for col in meta_columns}

    t_prefilter = time.perf_counter()

    # 4) Pętla po poszczególnych zapytaniach (grupach ||)
    for token_query_conditions, s_ordered, sentence_query_conditions, group_row_ids in group_jobs:
        if not group_row_ids:
            continue

        # Bierzemy tylko te wiersze z koszyka, które pasują do danej grupy
        group_index = filtered_df_base.index.intersection(group_row_ids)
        if len(group_index) == 0:
            continue

        filtered_df = filtered_df_base.loc[group_index]

        for row in filtered_df.itertuples(index=True):
            legacy_diag["rows_scanned"] += 1
            original_row_index = row.Index

            # --- 1. SZYBKIE LISTY PYTHONOWE ---
            tokens = row.tokens.tolist() if hasattr(row.tokens, "tolist") else row.tokens
            lemmas = row.lemmas.tolist() if hasattr(row.lemmas, "tolist") else row.lemmas
            deprels = row.deprels.tolist() if hasattr(row.deprels, "tolist") else row.deprels
            postags = row.postags.tolist() if hasattr(row.postags, "tolist") else row.postags

            upostags = getattr(row, "upostags", None)
            if upostags is not None: upostags = upostags.tolist() if hasattr(upostags, "tolist") else upostags

            srls = getattr(row, "srls", None)
            if srls is not None:
                srls = srls.tolist() if hasattr(srls, "tolist") else srls
            if srls is not None and len(srls) != len(tokens):
                srls = None

            srl_frames = getattr(row, "srl_frames", None)

            if srl_frames is None:
                srl_frames = []
            elif hasattr(srl_frames, "item"):
                srl_frames = srl_frames.item()

            if isinstance(srl_frames, str):
                try:
                    srl_frames = json.loads(srl_frames)
                except Exception:
                    srl_frames = []
            elif hasattr(srl_frames, "tolist"):
                srl_frames = srl_frames.tolist()

            if not isinstance(srl_frames, list):
                srl_frames = []

            full_postags = row.full_postags.tolist() if hasattr(row.full_postags, "tolist") else row.full_postags
            word_ids = row.word_ids.tolist() if hasattr(row.word_ids, "tolist") else row.word_ids
            sentence_ids = row.sentence_ids.tolist() if hasattr(row.sentence_ids, "tolist") else row.sentence_ids
            head_ids = row.head_ids.tolist() if hasattr(row.head_ids, "tolist") else row.head_ids
            start_ids = row.start_ids.tolist() if hasattr(row.start_ids, "tolist") else row.start_ids
            end_ids = row.end_ids.tolist() if hasattr(row.end_ids, "tolist") else row.end_ids
            ners = row.ners.tolist() if hasattr(row.ners, "tolist") else row.ners

            corefs = getattr(row, "corefs", None)
            if corefs is not None: corefs = corefs.tolist() if hasattr(corefs, "tolist") else corefs
            # -------------------------------------------------------------------------

            num_tokens = len(tokens)
            if num_tokens == 0:
                continue

            # --- 2. LENIWE ŁADOWANIE (Drzewa/klastry TYLKO gdy są potrzebne) ---
            _deps_cache = None

            def get_deps():
                nonlocal _deps_cache

                if _deps_cache is not None:
                    return _deps_cache

                cache_key = (selected_corpus, int(original_row_index))

                if _get_dependency_cache_ram_mode() != "none":
                    cached = dependency_maps_cache.get(cache_key)
                    if cached is not None:
                        legacy_diag["dep_maps_ram_hits"] += 1
                        _deps_cache = cached
                        return _deps_cache

                disk_cache = get_dependency_disk_cache_for_corpus(selected_corpus)
                if disk_cache is not None:
                    _dep_disk_t0 = time.perf_counter()
                    cached_disk = disk_cache.get(int(original_row_index))
                    _dep_disk_t1 = time.perf_counter()
                    if cached_disk is not None:
                        legacy_diag["dep_maps_disk_hits"] = legacy_diag.get("dep_maps_disk_hits", 0) + 1
                        legacy_diag["time_dep_maps_disk"] = legacy_diag.get("time_dep_maps_disk", 0.0) + (_dep_disk_t1 - _dep_disk_t0)
                        _deps_cache = cached_disk
                        _put_dependency_ram_cache(cache_key, _deps_cache)
                        return _deps_cache

                _dep_t0 = time.perf_counter()
                _deps_cache = build_dependency_maps(sentence_ids, word_ids, head_ids)
                _dep_t1 = time.perf_counter()

                legacy_diag["dep_maps_built"] += 1
                legacy_diag["time_dep_maps"] += (_dep_t1 - _dep_t0)

                if disk_cache is not None:
                    disk_cache.put(int(original_row_index), _deps_cache, commit=True)
                    legacy_diag["dep_maps_disk_written"] = legacy_diag.get("dep_maps_disk_written", 0) + 1

                _put_dependency_ram_cache(cache_key, _deps_cache)

                return _deps_cache

            _coref_cache = None

            def get_coref_clusters():
                nonlocal _coref_cache
                if _coref_cache is None:
                    _coref_cache = {}
                    if corefs is not None:
                        for c_idx, c_tags in enumerate(corefs):
                            if c_tags is None: continue
                            if isinstance(c_tags, str): c_tags = [c_tags]
                            for c_tag in c_tags:
                                if c_tag in ("0", "O", "_", None): continue
                                parts = c_tag.split("-", 1)
                                c_id_str = parts[1] if len(parts) > 1 else c_tag
                                if c_id_str not in _coref_cache:
                                    _coref_cache[c_id_str] = set()
                                _coref_cache[c_id_str].add(str(lemmas[c_idx]).lower())
                                _coref_cache[c_id_str].add(str(tokens[c_idx]).lower())
                return _coref_cache

            # ----------------------------------------------------------------------

            # --- 3. METADANE I DATY POBIERAMY RAZ NA CAŁY DOKUMENT! ---
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
                        year, month = parts[0], "1"
                    else:
                        year, month, _ = parts
                    month_key = f"{year}-{month}"
                else:
                    month_key = "Unknown"
            except Exception:
                month_key = "Unknown"

            title = titles_dict.get(original_row_index, " ")
            author = authors_dict.get(original_row_index, " ")
            additional_metadata = {col: meta_dicts[col].get(original_row_index, " ") for col in meta_dicts}


            # ----------------------------------------------------------------------
            def normalize_srl_role(role):
                role = str(role or "").upper()

                if role.startswith("B-") or role.startswith("I-"):
                    role = role[2:]

                if role.startswith("ARGM-"):
                    role = role.replace("ARGM-", "", 1)

                return role

            def get_frame_for_predicate(token_idx):
                for frame in srl_frames:
                    try:
                        pred_id = int(frame.get("pred_id"))
                    except Exception:
                        continue

                    if pred_id == token_idx:
                        return frame

                return None

            def token_is_srl_predicate(token_idx):
                return get_frame_for_predicate(token_idx) is not None

            def value_matches_text(value, candidate, match_type="exact"):
                value = str(value)
                candidate = str(candidate or "")

                if match_type == "exact":
                    return candidate.lower() == value.lower()

                if match_type == "regex":
                    return re.fullmatch(value, candidate, flags=re.IGNORECASE) is not None

                if match_type == "regex_search":
                    return re.search(value, candidate, flags=re.IGNORECASE) is not None

                return False

            def value_matches_any(value, candidates, match_type="exact"):
                if candidates is None:
                    candidates = []

                for candidate in candidates:
                    if value_matches_text(value, candidate, match_type):
                        return True

                return False



            def parse_srl_argument_key(key):
                key = str(key).lower()
                parts = key.split("_")

                if len(parts) < 2 or parts[0] != "srl":
                    return None, None

                role = parts[1].upper()
                field = "_".join(parts[2:]) if len(parts) >= 3 else "base"
                return role, field

            def frame_has_argument(frame, role, field, value, match_type="exact"):
                """
                role: ARG0, ARG1, ARG2, TMP, LOC, MNR...
                field: base/lemma/lemmas, orth/token/tokens, text, ner, head...
                """
                if not frame:
                    return False

                wanted_role = normalize_srl_role(role)

                for arg in frame.get("arguments", []):
                    arg_role = normalize_srl_role(arg.get("role"))
                    arg_role_full = normalize_srl_role(arg.get("role_full"))

                    if wanted_role not in {arg_role, arg_role_full}:
                        continue

                    field = field.lower()

                    if field in ("base", "lemma", "lemmas"):
                        candidates = [str(x).lower() for x in arg.get("lemmas", [])]
                        if match_type == "exact":
                            if str(value).lower() in candidates:
                                return True
                        else:
                            if value_matches_any(value, candidates, match_type):
                                return True

                    elif field in ("orth", "token", "tokens"):
                        candidates = [str(x) for x in arg.get("tokens", [])]
                        if match_type == "exact":
                            if str(value).lower() in [x.lower() for x in candidates]:
                                return True
                        else:
                            if value_matches_any(value, candidates, match_type):
                                return True

                    elif field == "text":
                        if value_matches_text(value, arg.get("text", ""), match_type):
                            return True

                    elif field == "ner":
                        candidates = [str(x) for x in arg.get("ners", [])]
                        if match_type == "exact":
                            if str(value) in candidates:
                                return True
                        else:
                            if value_matches_any(value, candidates, match_type):
                                return True

                    elif field == "upos":
                        candidates = [str(x) for x in arg.get("upostags", [])]
                        if match_type == "exact":
                            if str(value).upper() in [x.upper() for x in candidates]:
                                return True
                        else:
                            if value_matches_any(value, candidates, match_type):
                                return True

                    # --- TUTAJ JEST TWOJA ORYGINALNA, ZNAKOMITA LOGIKA GŁÓW ---
                    elif field in ("head_base", "head_lemma", "head"):
                        candidate = arg.get("head_lemma", "")
                        if value_matches_text(value, candidate, match_type):
                            return True

                    elif field in ("head_text", "head_orth", "head_token"):
                        candidate = arg.get("head_text", "")
                        if value_matches_text(value, candidate, match_type):
                            return True

                    elif field == "head_ner":
                        candidate = arg.get("head_ner", "")
                        if value_matches_text(value, candidate, match_type):
                            return True

                    elif field == "head_pos":
                        candidate = arg.get("head_pos", "")
                        if value_matches_text(value, candidate, match_type):
                            return True

                return False

            def check_frame_against_srl_conds(role_of_token, frame_dict, srl_conds):
                for cond in srl_conds:
                    if len(cond) >= 5:
                        key, values, operator, is_nested, match_type = cond
                    else:
                        key, values, operator, is_nested = cond
                        match_type = "exact"

                    match_found = False

                    if key in ("srl", "srl_role"):
                        for val in values:
                            val_norm = normalize_srl_role(val)
                            if match_type == "exact" and role_of_token == val_norm:
                                match_found = True;
                                break
                            elif match_type == "regex" and re.fullmatch(str(val), role_of_token, flags=re.IGNORECASE):
                                match_found = True;
                                break
                            elif match_type == "regex_search" and re.search(str(val), role_of_token,
                                                                            flags=re.IGNORECASE):
                                match_found = True;
                                break

                    elif key in ("srl_pred", "srl_pred_lemma"):
                        pred_id = int(frame_dict.get("pred_id", -1))
                        pred_lemma = frame_dict.get("pred_lemma", "")
                        if not pred_lemma and 0 <= pred_id < num_tokens:
                            pl = lemmas[pred_id]
                            if hasattr(pl, "item"): pl = pl.item()
                            pred_lemma = pl
                        pred_lemma = str(pred_lemma or "").strip().lower()

                        for val in values:
                            v_str = str(val).strip().lower()
                            if match_type == "exact" and pred_lemma == v_str:
                                match_found = True;
                                break
                            elif match_type == "regex" and re.fullmatch(str(val), pred_lemma, flags=re.IGNORECASE):
                                match_found = True;
                                break
                            elif match_type == "regex_search" and re.search(str(val), pred_lemma, flags=re.IGNORECASE):
                                match_found = True;
                                break

                    elif key == "srl_is_pred":
                        expected_true = any(str(v).lower() in ("true", "1", "yes", "tak") for v in values)
                        actual_true = (role_of_token == "PRED")
                        match_found = (actual_true == expected_true)

                    elif key.startswith("srl_"):
                        role, field = parse_srl_argument_key(key)
                        if role is not None and field is not None:
                            for val in values:
                                if frame_has_argument(frame_dict, role, field, val, match_type):
                                    match_found = True
                                    break

                    if operator == "=" and not match_found:
                        return False
                    elif operator == "!=" and match_found:
                        return False

                return True

            def match_conditions(token_idx, conditions):
                legacy_diag["tokens_checked"] += 1
                _mc_t0 = time.perf_counter()
                if not conditions:
                    return True

                srl_conds = []
                other_conds = []

                for cond in conditions:
                    if not cond: continue
                    if cond[0].startswith("srl"):
                        srl_conds.append(cond)
                    else:
                        other_conds.append(cond)

                for cond in other_conds:
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
                            if required_role and token_role != required_role: continue
                            cluster_words = get_coref_clusters().get(c_id, set())
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
                            if match_found: break
                        if operator == "=" and not match_found:
                            return False
                        elif operator == "!=" and match_found:
                            return False

                    elif key.startswith("head") or key.startswith("head.group"):
                        legacy_diag["head_checks"] += 1
                        p_idx_map, _ = get_deps()
                        parent = p_idx_map[token_idx]
                        if parent is None or parent < 0:
                            if operator == "=":
                                return False
                            else:
                                continue
                        m = re.match(r'head(?:\.group)?(?:\((<|>|=)?(-?\d+)\))?$', key)
                        dist_op = m.group(1) if m and m.group(1) else None
                        dist_val = int(m.group(2)) if m and m.group(2) else None

                        def _distance_matches_child(dist_val, dist_op):
                            if dist_val is None: return True
                            distance = word_ids[parent] - word_ids[token_idx]
                            if dist_op in (None, "="):
                                return distance == dist_val
                            elif dist_op == "<":
                                return distance < dist_val
                            elif dist_op == ">":
                                return distance > dist_val
                            return False

                        if operator == "=":
                            if not _distance_matches_child(dist_val, dist_op): return False
                            if is_nested:
                                if not match_conditions(parent, tuple(values)): return False
                            else:
                                parent_attr = lemmas[parent]
                                if match_type == "exact":
                                    if parent_attr not in values: return False
                                elif match_type == "regex":
                                    if not any(re.fullmatch(v, parent_attr) for v in values): return False
                                elif match_type == "regex_search":
                                    if not any(re.search(v, parent_attr) for v in values): return False
                        elif operator == "!=":
                            if dist_val is not None and not _distance_matches_child(dist_val, dist_op):
                                continue
                            if is_nested:
                                if match_conditions(parent, tuple(values)): return False
                            else:
                                parent_attr = lemmas[parent]
                                if match_type == "exact":
                                    if parent_attr in values: return False
                                elif match_type == "regex":
                                    if any(re.fullmatch(v, parent_attr) for v in values): return False
                                elif match_type == "regex_search":
                                    if any(re.search(v, parent_attr) for v in values): return False

                    elif key.startswith("dependent"):
                        legacy_diag["dependent_checks"] += 1
                        _, c_lookup_map = get_deps()
                        children = c_lookup_map[token_idx]
                        if children is None or len(children) == 0:
                            if operator == "=":
                                return False
                            else:
                                continue
                        m = re.match(r'dependent(?:\((<|>|=)?(-?\d+)\))?$', key)
                        dist_op = m.group(1) if m and m.group(1) else None
                        dist_val = int(m.group(2)) if m and m.group(2) else None
                        if operator == "=":
                            found = False
                            for child in children:
                                if dist_val is not None:
                                    distance = word_ids[child] - word_ids[token_idx]
                                    if dist_op in (None, "=") and distance != dist_val:
                                        continue
                                    elif dist_op == "<" and not (distance < dist_val):
                                        continue
                                    elif dist_op == ">" and not (distance > dist_val):
                                        continue
                                if is_nested:
                                    if match_conditions(child, tuple(values)):
                                        found = True;
                                        break
                                else:
                                    child_attr = lemmas[child]
                                    if match_type == "exact":
                                        if child_attr in values: found = True; break
                                    elif match_type == "regex":
                                        if any(re.fullmatch(v, child_attr) for v in values): found = True; break
                                    elif match_type == "regex_search":
                                        if any(re.search(v, child_attr) for v in values): found = True; break
                            if not found: return False
                        elif operator == "!=":
                            for child in children:
                                if dist_val is not None:
                                    distance = word_ids[child] - word_ids[token_idx]
                                    if dist_op in (None, "=") and distance != dist_val:
                                        continue
                                    elif dist_op == "<" and not (distance < dist_val):
                                        continue
                                    elif dist_op == ">" and not (distance > dist_val):
                                        continue
                                if is_nested:
                                    if match_conditions(child, tuple(values)): return False
                                else:
                                    child_attr = lemmas[child]
                                    if match_type == "exact":
                                        if child_attr in values: return False
                                    elif match_type == "regex":
                                        if any(re.fullmatch(v, child_attr) for v in values): return False
                                    elif match_type == "regex_search":
                                        if any(re.search(v, child_attr) for v in values): return False

                    elif key.startswith("window_base") or key.startswith("window_orth"):
                        m = re.match(r'window_(base|orth)(?:\((\d+)\))?$', key)
                        if not m: return False
                        w_type = m.group(1)
                        dist = int(m.group(2)) if m.group(2) else 50
                        start_w = max(0, token_idx - dist)
                        end_w = min(num_tokens, token_idx + dist + 1)
                        found = False
                        for w_i in range(start_w, end_w):
                            if w_i == token_idx: continue
                            val = lemmas[w_i] if w_type == "base" else tokens[w_i]
                            if match_type == "exact":
                                if val in values: found = True; break
                            elif match_type == "regex":
                                if any(re.fullmatch(v, val) for v in values): found = True; break
                            elif match_type == "regex_search":
                                if any(re.search(v, val) for v in values): found = True; break
                        if operator == "=":
                            if not found: return False
                        elif operator == "!=":
                            if found: return False

                    else:
                        full_tag = full_postags[token_idx]
                        tag_parts = full_tag.split(":")
                        pos = tag_parts[0] if tag_parts else ""
                        feats = tag_parts[1:] if len(tag_parts) > 1 else []
                        mapping = FEAT_MAPPING.get(pos, {})
                        if key not in mapping: return False
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

                if srl_conds:
                    valid_frame_found = False
                    candidate_frames = []

                    if token_is_srl_predicate(token_idx):
                        f = get_frame_for_predicate(token_idx)
                        if f: candidate_frames.append(("PRED", f.get("pred_id", token_idx), f))

                    t_roles = srls[token_idx] if srls is not None else []
                    if hasattr(t_roles, "tolist"): t_roles = t_roles.tolist()
                    if not isinstance(t_roles, (list, tuple)): t_roles = [t_roles]

                    for r_item in t_roles:
                        if isinstance(r_item, dict):
                            role_str = str(r_item.get("role", r_item.get(b"role", "")))
                            raw_pred_id = r_item.get("pred_id", r_item.get(b"pred_id"))
                            try:
                                p_id = int(raw_pred_id) if raw_pred_id is not None else None
                            except:
                                p_id = None
                        else:
                            role_str = str(r_item)
                            p_id = None

                        if role_str in ("0", "O", "_", "None", ""): continue
                        clean_role = normalize_srl_role(role_str)
                        if p_id is not None:
                            f = get_frame_for_predicate(p_id)
                            if f: candidate_frames.append((clean_role, p_id, f))

                    for role_of_token, p_id, f_dict in candidate_frames:
                        if check_frame_against_srl_conds(role_of_token, f_dict, srl_conds):
                            valid_frame_found = True
                            break

                    if not valid_frame_found:
                        return False

                legacy_diag["time_match_conditions_success_path"] += (time.perf_counter() - _mc_t0)
                return True

            def expand_mention(s_idx, e_limit, current_conds):
                current_cond_list = current_conds if isinstance(current_conds, list) else [current_conds]
                is_coref_m = False
                is_srl = False
                srl_conds = []
                for c in current_cond_list:
                    if c and len(c) >= 1:
                        if c[0] == "coref(M)":
                            is_coref_m = True
                        elif c[0].startswith("srl"):
                            is_srl = True
                            srl_conds.append(c)

                if is_srl:
                    n_idx = s_idx + 1

                    def get_valid_signatures_for_token(t_idx, require_i=False):
                        sigs = set()

                        if not require_i:
                            if token_is_srl_predicate(t_idx):
                                f = get_frame_for_predicate(t_idx)
                                if f and check_frame_against_srl_conds("PRED", f, srl_conds):
                                    sigs.add((t_idx, "PRED"))

                        t_roles = srls[t_idx] if srls is not None else []
                        if hasattr(t_roles, "tolist"): t_roles = t_roles.tolist()
                        if not isinstance(t_roles, (list, tuple)): t_roles = [t_roles]

                        for r in t_roles:
                            if isinstance(r, dict):
                                raw_val = r.get("raw_role", r.get(b"raw_role", None))
                                role_val = r.get("role", r.get(b"role", ""))
                                role_full_val = r.get("role_full", r.get(b"role_full", role_val))
                                bio_val = r.get("bio", r.get(b"bio", None))
                                raw_pred = r.get("pred_id", r.get(b"pred_id"))
                                try:
                                    p_id = int(raw_pred) if raw_pred is not None else None
                                except:
                                    p_id = None
                            else:
                                raw_val = str(r)
                                role_val = raw_val
                                role_full_val = raw_val
                                bio_val = ""
                                p_id = None

                            raw_val = str(raw_val or role_val or "")
                            bio_val = str(bio_val or "")

                            if require_i:
                                if not (raw_val.startswith("I-") or bio_val == "I"):
                                    continue
                            else:
                                if not (raw_val.startswith("B-") or raw_val.startswith("I-") or bio_val in ("B", "I")):
                                    pass  # Złagodzenie dla jednowyrazowych, gołych tagów z parsowania

                            clean = normalize_srl_role(role_full_val or role_val or raw_val)

                            if p_id is not None:
                                f = get_frame_for_predicate(p_id)
                                if f and check_frame_against_srl_conds(clean, f, srl_conds):
                                    sigs.add((p_id, clean))

                        return sigs

                    active_srls = get_valid_signatures_for_token(s_idx, require_i=False)
                    if not active_srls:
                        return n_idx

                    while n_idx < e_limit:
                        next_active = get_valid_signatures_for_token(n_idx, require_i=True)
                        shared = active_srls.intersection(next_active)
                        if shared:
                            active_srls = shared
                            n_idx += 1
                        else:
                            break

                    return n_idx

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

            def _morph_feature_at_for_anchor(tok_idx, feat_key):
                try:
                    full_tag = full_postags[tok_idx]
                    tag_parts = str(full_tag).split(":")
                    tag_pos = tag_parts[0] if tag_parts else ""
                    tag_feats = tag_parts[1:] if len(tag_parts) > 1 else []
                    mapping = FEAT_MAPPING.get(tag_pos, {})
                    if feat_key not in mapping:
                        return ""
                    feat_idx = mapping[feat_key]
                    return tag_feats[feat_idx] if feat_idx < len(tag_feats) else ""
                except Exception:
                    return ""

            def _token_attr_for_anchor(tok_idx, attr_key):
                try:
                    if attr_key == "orth":
                        return tokens[tok_idx]
                    if attr_key == "base":
                        return lemmas[tok_idx]
                    if attr_key == "pos":
                        return postags[tok_idx]
                    if attr_key == "upos":
                        return upostags[tok_idx]
                    if attr_key == "deprel":
                        return deprels[tok_idx]
                    if attr_key == "ner":
                        return ners[tok_idx]
                    return _morph_feature_at_for_anchor(tok_idx, attr_key)
                except Exception:
                    return ""

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
            # --- CACHEWARM_ANCHOR: kotwica pozycyjna dla pierwszego segmentu ---
            # Dotąd fast-forward działał tylko dla base/orth. Tu rozszerzamy go na
            # pos/upos/deprel/ner oraz cechy morfologiczne (np. number="sg", case="gen").
            anchor_type = None
            anchor_values = set()
            anchor_filters = []

            if token_query_conditions and len(token_query_conditions) > 0:
                first_cond_group = token_query_conditions[0]
                first_conds = first_cond_group if isinstance(first_cond_group, list) else [first_cond_group]

                morph_keys = set()
                try:
                    for _m in FEAT_MAPPING.values():
                        morph_keys.update(_m.keys())
                except Exception:
                    morph_keys = set()

                for cond in first_conds:
                    if cond and len(cond) >= 5:
                        key, values, operator, is_nested, match_type = cond
                        if operator == "=" and match_type == "exact" and not is_nested:
                            if key in ("orth", "base", "pos", "upos", "deprel", "ner") or key in morph_keys:
                                vals = set(v for v in values if isinstance(v, str))
                                if vals:
                                    anchor_filters.append((key, vals))

            anchor_indices = []
            if anchor_filters:
                anchor_type = "multi"
                anchor_values = {"__multi__"}
                # Intersekcja wszystkich tanich warunków pierwszego segmentu.
                # Dla np. [pos="subst" & number="sg" & dependent=...] sprawdzamy dependency
                # tylko na tokenach subst:sg, a nie na wszystkich tokenach dokumentu.
                for _idx in range(num_tokens):
                    _ok = True
                    for _key, _vals in anchor_filters:
                        if str(_token_attr_for_anchor(_idx, _key)) not in _vals:
                            _ok = False
                            break
                    if _ok:
                        anchor_indices.append(_idx)
                legacy_diag["anchor_candidates"] += len(anchor_indices)
                legacy_diag["anchor_mode_docs"] += 1

            # ---------------------------------------------------------------

            i = 0
            # Jeśli znaleźliśmy precyzyjne indeksy kotwicy, i w ogóle one istnieją w tym zdaniu
            anchor_pointer = 0
            i = 0
            while i < num_tokens:

                # --- NOWOŚĆ: Błyskawiczny przeskok (Fast-Forward) ---
                if anchor_type and anchor_values:
                    # Przesuwamy wskaźnik do najbliższej znalezionej pozycji kotwicy
                    while anchor_pointer < len(anchor_indices) and anchor_indices[anchor_pointer] < i:
                        anchor_pointer += 1

                    if anchor_pointer < len(anchor_indices):
                        # Przeskakujemy od razu do właściwego słowa!
                        i = anchor_indices[anchor_pointer]
                    else:
                        # Brak więcej wystąpień kotwicy w tym dokumencie -> kończymy sprawdzanie dokumentu
                        break
                        # -----------------------------------------------------

                if s_ordered or sentence_query_conditions:
                    sent_start = i
                    while sent_start > 0 and sentence_ids[sent_start - 1] == sentence_ids[i]:
                        sent_start -= 1
                    sent_end = i
                    while sent_end < len(sentence_ids) and sentence_ids[sent_end] == sentence_ids[i]:
                        sent_end += 1

                    if sentence_query_conditions:
                        if s_ordered:
                            if not sentence_contains_conditions(sent_start, sent_end, sentence_query_conditions):
                                i = sent_end  # Przeskok na koniec zdania (optymalizacja!)
                                continue
                        else:
                            if not sentence_matches(sent_start, sent_end, sentence_query_conditions):
                                i = sent_end  # Przeskok na koniec zdania
                                continue

                    end_idx = match_pattern_in_range(i, token_query_conditions, sent_end)
                else:
                    end_idx = match_pattern(i, token_query_conditions)

                if end_idx is not None and end_idx > i:
                    # Wyciągamy same słowa z listy (dużo szybsze niż cięcie długich stringów)
                    matched_text = " ".join(tokens[i:end_idx]) if end_idx - i > 1 else str(tokens[i])
                    matched_lemmas = " ".join(lemmas[i:end_idx]) if end_idx - i > 1 else str(lemmas[i])

                    token_counter[matched_text] += 1
                    lemma_counter[matched_lemmas] += 1

                    # Zapisujemy w pamięci same lekkie "namiary" na dopasowanie
                    temp_results.append((
                        matched_text, matched_lemmas, row.Index, i, end_idx,
                        publication_date, month_key, title, author, additional_metadata
                    ))

                    i = end_idx
                else:
                    i += 1  # Jeśli się nie udało, idziemy oczko dalej (lub do kolejnej kotwicy w następnym obrocie)


        # --- 1. SZYBKIE FILTROWANIE SUROWYCH WYNIKÓW ---
        filtered_raw_results = []
        if freq_base_opts:
            if "top" in freq_base_opts:
                top_lemmas = {lemma for lemma, _ in lemma_counter.most_common(freq_base_opts["top"])}
            else:
                top_lemmas = set(lemma_counter.keys())
            for item in temp_results:
                matched_text, matched_lemmas = item[0], item[1]
                count = lemma_counter[matched_lemmas]
                if (matched_lemmas in top_lemmas and
                        ("min" not in freq_base_opts or count >= freq_base_opts["min"]) and
                        ("max" not in freq_base_opts or count <= freq_base_opts["max"])):
                    filtered_raw_results.append(item)
        elif freq_opts:
            if "top" in freq_opts:
                top_tokens = {token for token, _ in token_counter.most_common(freq_opts["top"])}
            else:
                top_tokens = set(token_counter.keys())
            for item in temp_results:
                matched_text, matched_lemmas = item[0], item[1]
                count = token_counter[matched_text]
                if (matched_text in top_tokens and
                        ("min" not in freq_opts or count >= freq_opts["min"]) and
                        ("max" not in freq_opts or count <= freq_opts["max"])):
                    filtered_raw_results.append(item)
        else:
            filtered_raw_results = temp_results

        # --- 2. LENIWE BUDOWANIE KONTEKSTÓW (TYLKO DLA ZAAKCEPTOWANYCH) ---
        final_results = []
        for (matched_text_real, matched_lemmas, row_idx, i, end_idx,
             pub_date, m_key, title, author, add_meta) in filtered_raw_results:
            row = filtered_df_base.loc[row_idx]
            start_ids = row.start_ids.tolist() if hasattr(row.start_ids, "tolist") else row.start_ids
            end_ids = row.end_ids.tolist() if hasattr(row.end_ids, "tolist") else row.end_ids
            tresc = row.Treść

            left_context = tresc[max(0, start_ids[max(0, i - left_context_size)]): start_ids[i]] if i > 0 else ""

            # Właściwy tekst dopasowania z oryginalnymi znakami
            matched_text_actual = tresc[start_ids[i]: end_ids[end_idx - 1] + 1]

            right_limit = start_ids[min(len(start_ids) - 1, end_idx - 1 + right_context_size + 1)]
            right_context = tresc[end_ids[end_idx - 1] + 1: right_limit]

            context = [left_context, matched_text_actual, right_context]

            # Pełny kontekst
            global kontekst
            try:
                k_full_context = int(kontekst)
            except Exception:
                k_full_context = int((globals().get("config", {}) or {}).get("kontekst", 250) or 250)
            full_left = tresc[max(0, start_ids[max(0, i - k_full_context)]): start_ids[i]] if i > 0 else ""
            full_left = full_left[:-len(left_context)] if left_context else full_left

            full_right_limit = start_ids[min(len(start_ids) - 1, end_idx - 1 + k_full_context)]
            full_right = tresc[end_ids[end_idx - 1] + 1: full_right_limit]
            full_right = full_right[len(right_context):] if right_context else full_right

            full_text_with_markers = [full_left, matched_text_actual, full_right]

            legacy_diag["matches"] += 1
            final_results.append((
                pub_date, context, full_text_with_markers,
                matched_text_actual, matched_lemmas,
                m_key, title, author, add_meta,
                left_context, right_context, row_idx, i, end_idx
            ))

        t_find_end = time.perf_counter()
        legacy_diag["time_total"] = t_find_end - t_find_start
        search_diag_log("LEGACY_DONE query=%r diag=%r", query, legacy_diag)
        search_diag_log("LEGACY_TIMING prefilter=%.4fs token_loop=%.4fs", t_prefilter - t_find_start, t_find_end - t_prefilter)
        return final_results




# KORPUSUJ_MIGRATION_036L4G36U3_FINAL_REGEX_CQL_TO_LEGACY_ROUTE
# KORPUSUJ_MIGRATION_036L4G36V_FIX_U3_REGEX_ROUTE_DETECTOR
def _regex_legacy_route_enabled_036l4g36u3():
    try:
        import os
        return str(os.environ.get('KORPUSUJ_REGEX_LEGACY_ROUTE_036L4G36U3', '1')).lower() not in {'0', 'false', 'no', 'off'}
    except Exception:
        return True


def _query_requires_legacy_regex_backend_036l4g36u3(query):
    try:
        q = str(query or '')
        if '[' not in q or ']' not in q:
            return False
        regex_meta = set('.^$*+?{}[]|\\()')
        parts = []
        i = 0
        while i < len(q):
            if q[i] != '[':
                i += 1
                continue
            j = q.find(']', i + 1)
            if j < 0:
                break
            parts.append(q[i:j+1])
            i = j + 1
        for part in parts:
            vals = []
            i = 0
            while i < len(part):
                if part[i] != chr(34):
                    i += 1
                    continue
                i += 1
                buf = []
                esc = False
                while i < len(part):
                    ch = part[i]
                    if esc:
                        buf.append(ch)
                        esc = False
                    elif ch == chr(92):
                        esc = True
                    elif ch == chr(34):
                        vals.append(''.join(buf))
                        break
                    else:
                        buf.append(ch)
                    i += 1
                i += 1
            for val in vals:
                if val.startswith('~') and len(val) > 1:
                    return True
                if any(ch in regex_meta for ch in val):
                    return True
        return False
    except Exception:
        return False


def _regex_legacy_route_log_036l4g36u3(event, **data):
    try:
        import logging
        if korpusuj_diagnostics_enabled_145c1():
            logging.info('[DIAG regex.legacy_route] event=%r data=%r', event, data)
    except Exception:
        pass






# KORPUSUJ_MIGRATION_036L4G37C_REGEX_SQLITE_CONFIG_ROUTE
# Config-driven bypass of regex->legacy route when active search object is LazyCorpus.

# KORPUSUJ_MIGRATION_036L4G79_LEGACY_RUNTIME_LABELS
def _legacy_runtime_label_log_036l4g79(event, **data):
    # Best-effort structured diagnostic log for legacy fallback activation.
    # This helper must never influence search semantics.
    try:
        import logging
        if korpusuj_diagnostics_enabled_145c1():
            logging.info('[DIAG legacy.runtime_label] event=%r data=%r', event, data)
    except Exception:
        pass


def _legacy_runtime_df_kind_036l4g79(df):
    try:
        return type(df).__name__
    except Exception:
        return 'unknown'


def _legacy_runtime_query_shape_036l4g79(query):
    try:
        q = str(query or '')
        q_lower = q.lower()
        if not q:
            return 'unknown'
        if q.count('[') != q.count(']') or q.count('{') != q.count('}'):
            return 'malformed_regex'
        if 'dependent=' in q_lower or 'dependent={' in q_lower or 'deprel=' in q_lower:
            return 'dependency_cql'
        if '!=' in q or ' not ' in q_lower or 'negative_regex' in q_lower:
            return 'negative_regex'
        if '.*' in q:
            return 'broad_regex'
        regex_meta = set('.^$*+?{}[]|\\()')
        if '[' in q and ']' in q and any(ch in q for ch in regex_meta):
            return 'regex'
        return 'normal'
    except Exception:
        return 'unknown'
# END KORPUSUJ_MIGRATION_036L4G79_LEGACY_RUNTIME_LABELS


# END KORPUSUJ_MIGRATION_036L4G37C_REGEX_SQLITE_CONFIG_ROUTE


# KORPUSUJ_MIGRATION_036L4G38A_REGEX_SQLITE_BROAD_POLICY
# Preflight for regex-SQL route. Broad regexes fall back to legacy before
# SearchCursor starts materializing hits.
def _regex_sqlite_cfg_036l4g38a(name, default=None):
    try:
        cfg = globals().get('config', {}) or {}
        defaults = {
            'regex_sqlite_route': True,
            'regex_sqlite_enabled': True,
            'regex_sqlite_debug': False,
            'regex_sqlite_max_terms': 5000,
            'regex_sqlite_max_cf': 2000000,
            'regex_sqlite_broad_policy': 'legacy',
        }
        if default is None:
            default = defaults.get(name)
        return cfg.get(name, default)
    except Exception:
        return default

def _regex_sqlite_bool_cfg_036l4g38a(name, default=True):
    val = _regex_sqlite_cfg_036l4g38a(name, default)
    if isinstance(val, str):
        return val.strip().lower() in {'1', 'true', 'yes', 'tak', 'on'}
    return bool(val)

def _regex_sqlite_int_cfg_036l4g38a(name, default):
    try:
        return int(_regex_sqlite_cfg_036l4g38a(name, default))
    except Exception:
        return int(default)

def _regex_sqlite_broad_policy_036l4g38a():
    val = str(_regex_sqlite_cfg_036l4g38a('regex_sqlite_broad_policy', 'legacy') or 'legacy').strip().lower()
    return val if val in {'legacy', 'error', 'allow'} else 'legacy'

def _regex_sqlite_log_036l4g38a(event, **data):
    try:
        import logging as _logging_036l4g38a
        if korpusuj_diagnostics_enabled_145c1():
            if korpusuj_diagnostics_enabled_145c1():
                _logging_036l4g38a.info('[DIAG regex.sqlite.broad_route] event=%r data=%r', event, data)
    except Exception:
        pass

def _regex_sqlite_is_regex_value_036l4g38a(value):
    try:
        value = str(value or '')
        if value.startswith('~') and len(value) > 1:
            return True
        return any(ch in set('.^$*+?[]|\\()') for ch in value)
    except Exception:
        return False

def _regex_sqlite_extract_conditions_036l4g38a(query):
    try:
        import re as _re_036l4g38a, ast as _ast_036l4g38a
        q = str(query or '')
        out = []
        # Extract simple token-attribute conditions from square brackets.
        for m in _re_036l4g38a.finditer(r'\[([^\[\]]+)\]', q):
            part = m.group(1)
            for cm in _re_036l4g38a.finditer(r'\b(base|orth|pos|upos|deprel|ner)\s*(!=|=)\s*("(?:\\.|[^"\\])*")', part):
                attr = cm.group(1)
                op = cm.group(2)
                raw = cm.group(3)
                try:
                    value = _ast_036l4g38a.literal_eval(raw)
                except Exception:
                    value = raw[1:-1]
                out.append((attr, op, value))
        return out
    except Exception:
        return []

def _regex_sqlite_preflight_036l4g38a(df, query):
    try:
        import sqlite3 as _sqlite3_036l4g38a, re as _re_036l4g38a
        search_path = getattr(df, 'search_path', None)
        if not search_path:
            return 'unsupported'
        conditions = _regex_sqlite_extract_conditions_036l4g38a(query)
        regex_conditions = [(a, o, v) for a, o, v in conditions if _regex_sqlite_is_regex_value_036l4g38a(v)]
        if not regex_conditions:
            return 'ok'
        max_terms = _regex_sqlite_int_cfg_036l4g38a('regex_sqlite_max_terms', 5000)
        max_cf = _regex_sqlite_int_cfg_036l4g38a('regex_sqlite_max_cf', 2000000)

        con = _sqlite3_036l4g38a.connect(str(search_path))
        con.row_factory = _sqlite3_036l4g38a.Row
        try:
            for attr, op, pattern_raw in regex_conditions:
                # Negative regex is not candidate-producing in SQL route; use legacy for safety.
                if op == '!=':
                    _regex_sqlite_log_036l4g38a('fallback_to_legacy', reason='negative_regex', attr=attr, pattern=pattern_raw)
                    return 'unsupported'
                pattern_raw = str(pattern_raw or '')
                tilde = pattern_raw.startswith('~') and len(pattern_raw) > 1
                pattern = pattern_raw[1:] if tilde else pattern_raw
                try:
                    rx = _re_036l4g38a.compile(pattern)
                except Exception as e:
                    _regex_sqlite_log_036l4g38a('fallback_to_legacy', reason='compile_error', attr=attr, pattern=pattern_raw, error=repr(e))
                    return 'unsupported'
                matched = 0
                total_cf = 0
                for row in con.execute('SELECT value, cf FROM terms WHERE attr=?', (attr,)): 
                    value = str(row['value'])
                    ok = (rx.search(value) is not None) if tilde else (rx.fullmatch(value) is not None)
                    if ok:
                        matched += 1
                        try:
                            total_cf += int(row['cf'] or 0)
                        except Exception:
                            pass
                        if matched > max_terms:
                            _regex_sqlite_log_036l4g38a('too_broad', reason='max_terms', attr=attr, pattern=pattern_raw, matched=matched, max_terms=max_terms)
                            return 'too_broad'
                        if total_cf > max_cf:
                            _regex_sqlite_log_036l4g38a('too_broad', reason='max_cf', attr=attr, pattern=pattern_raw, total_cf=total_cf, max_cf=max_cf)
                            return 'too_broad'
                _regex_sqlite_log_036l4g38a('preflight_ok', attr=attr, pattern=pattern_raw, matched=matched, total_cf=total_cf)
        finally:
            try:
                con.close()
            except Exception:
                pass
        return 'ok'
    except Exception as e:
        _regex_sqlite_log_036l4g38a('fallback_to_legacy', reason='preflight_exception', error=repr(e))
        return 'unsupported'

def _regex_sqlite_route_enabled_036l4g38a(df=None, query=None):
    try:
        if not _regex_sqlite_bool_cfg_036l4g38a('regex_sqlite_route', True):
            return False
        if not _regex_sqlite_bool_cfg_036l4g38a('regex_sqlite_enabled', True):
            return False
        lazy_cls = globals().get('LazyCorpus')
        is_lazy = False
        if lazy_cls is not None and df is not None and isinstance(df, lazy_cls):
            is_lazy = True
        elif df is not None and hasattr(df, 'search_path') and hasattr(df, 'parquet_path'):
            is_lazy = True
        if not is_lazy:
            return False
        policy = _regex_sqlite_broad_policy_036l4g38a()
        if policy == 'allow':
            return True
        status = _regex_sqlite_preflight_036l4g38a(df, query)
        if status == 'too_broad' and policy == 'legacy':
            _regex_sqlite_log_036l4g38a('fallback_to_legacy', reason='too_broad_policy_legacy', query=query)
            return False
        if status == 'unsupported':
            return False
        return True
    except Exception:
        return False
# END KORPUSUJ_MIGRATION_036L4G38A_REGEX_SQLITE_BROAD_POLICY


# KORPUSUJ_MIGRATION_036L4G38B_REGEX_LEGACY_ROUTE_RESET_DF
def _normalize_regex_legacy_route_df_036l4g38b(route_df, selected_corpus=None, query=None):
    try:
        if hasattr(route_df, "reset_index") and hasattr(route_df, "iloc"):
            try:
                return route_df.reset_index(drop=True)
            except Exception:
                return route_df.copy().reset_index(drop=True)
    except Exception:
        pass
    return route_df

def _force_rebuild_legacy_inverted_index_for_regex_036l4g38b(selected_corpus=None):
    try:
        inv = globals().get("inverted_indexes", None)
        if isinstance(inv, dict) and selected_corpus in inv:
            inv.pop(selected_corpus, None)
    except Exception:
        pass
# END KORPUSUJ_MIGRATION_036L4G38B_REGEX_LEGACY_ROUTE_RESET_DF


# KORPUSUJ_MIGRATION_PATCH_103_GUI_SQLITE_REGEX_ROUTE_AFTER_NO_LIMIT
# Allow LazyCorpus/.search SQLite route to try valid indexed regex queries after
# patch_96 removed SQLite regex term/cf execution limits. This only bypasses the
# early regex-to-legacy route; existing sqlite-exception fallback to legacy stays.
def _gui_sqlite_regex_route_available_after_patch96_103(df, query):
    try:
        lazy_cls = globals().get("LazyCorpus")
        if lazy_cls is None or not isinstance(df, lazy_cls):
            return False
        if not _query_requires_legacy_regex_backend_036l4g36u3(query):
            return False
        search_path = getattr(df, "search_path", None)
        if not search_path:
            return False
        try:
            from pathlib import Path as _Path_103
            if not _Path_103(str(search_path)).exists():
                return False
        except Exception:
            return False
        supported = {"base", "orth", "pos", "upos", "deprel", "ner"}
        extractor = globals().get("_regex_sqlite_extract_conditions_036l4g38a")
        if callable(extractor):
            try:
                conds = extractor(query) or []
            except Exception:
                return False
            if not conds:
                return False
            for item in conds:
                try:
                    attr, op, _pattern = item
                except Exception:
                    return False
                if str(attr) not in supported:
                    return False
                if str(op) == "!=":
                    # Negative regex remains legacy-owned for now.
                    return False
        try:
            import logging as _logging_103
            if korpusuj_diagnostics_enabled_145c1():
                if korpusuj_diagnostics_enabled_145c1():
                    _logging_103.info(
                        "[DIAG regex.sqlite.gui_route] event=%r data=%r",
                        "allow_sqlite_instead_of_early_legacy",
                        {
                            "query": query,
                            "df_type": type(df).__name__,
                            "search_path": str(search_path),
                            "reason": "lazycorpus_search_sidecar_available_patch96_no_limit",
                        },
                    )
        except Exception:
            pass
        return True
    except Exception:
        return False
# END KORPUSUJ_MIGRATION_PATCH_103_GUI_SQLITE_REGEX_ROUTE_AFTER_NO_LIMIT



# KORPUSUJ_MIGRATION_PATCH_121_LEGACY_ADAPTER_FACADE
def _legacy_adapter_call_121(
    legacy_impl, query, df, selected_corpus,
    left_context_size=10, right_context_size=10, warnings_list=None,
    *, legacy_source, legacy_reason, route_name=None, extra=None,
):
    """Route all legacy activations through a stable adapter boundary."""
    _legacy_route_observability_122(
        legacy_source,
        legacy_reason,
        query,
        selected_corpus,
        df,
        route_name=route_name,
        event="route_enter",
        extra=extra,
    )
    try:
        from korpusuj.search.legacy_adapter import call_legacy_find_lemma_context_121
    except Exception as _legacy_adapter_import_exc_121:
        try:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG legacy.adapter] event=%r data=%r",
                    "adapter_import_failed_fallback_direct",
                    {
                        "legacy_source": legacy_source,
                        "legacy_reason": legacy_reason,
                        "route_name": route_name or legacy_source,
                        "query": query,
                        "selected_corpus": selected_corpus,
                        "df_type": type(df).__name__ if df is not None else None,
                        "error": repr(_legacy_adapter_import_exc_121),
                    },
                )
        except Exception:
            pass
        return legacy_impl(query, df, selected_corpus, left_context_size, right_context_size, warnings_list)
    return call_legacy_find_lemma_context_121(
        legacy_impl, query, df, selected_corpus, left_context_size, right_context_size, warnings_list,
        legacy_source=legacy_source, legacy_reason=legacy_reason, route_name=route_name, logger=logging, extra=extra,
    )


# KORPUSUJ_MIGRATION_PATCH_122_STRICT_SQLITE_EXCEPTION_AND_LEGACY_ROUTE_OBSERVABILITY_NO_BODY_SLICING
def _legacy_sqlite_exception_fallback_enabled_122():
    """Return whether SQLite exception fallback to legacy is enabled."""
    try:
        from korpusuj.search.legacy_adapter import legacy_fallback_on_sqlite_exception_enabled_121
        return bool(legacy_fallback_on_sqlite_exception_enabled_121(globals().get("config", None)))
    except Exception:
        return True


def _legacy_sqlite_exception_strict_snapshot_122():
    """Return strict-mode config snapshot for logging/probes."""
    try:
        from korpusuj.search.legacy_adapter import legacy_strict_config_snapshot_122
        return legacy_strict_config_snapshot_122(globals().get("config", None))
    except Exception as _snapshot_exc_122:
        return {"fallback_enabled": True, "strict_mode": False, "snapshot_error": repr(_snapshot_exc_122)}


def _legacy_route_observability_122(legacy_source, legacy_reason, query, selected_corpus, df, route_name=None, event="route_enter", extra=None):
    """Log route observability without touching the legacy matcher body."""
    try:
        from korpusuj.search.legacy_adapter import log_legacy_route_observability_122
        return log_legacy_route_observability_122(
            logging,
            legacy_source=legacy_source,
            legacy_reason=legacy_reason,
            query=query,
            selected_corpus=selected_corpus,
            df=df,
            route_name=route_name,
            event=event,
            extra=extra,
        )
    except Exception as _obs_exc_122:
        payload = {
            "legacy_source": legacy_source,
            "legacy_reason": legacy_reason,
            "route_name": route_name or legacy_source,
            "query": query,
            "selected_corpus": selected_corpus,
            "df_type": type(df).__name__ if df is not None else None,
            "event": event,
            "observability_error": repr(_obs_exc_122),
            "patch": "122",
        }
        try:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG legacy.route] event=%r data=%r", event, payload)
        except Exception:
            pass
        return payload

def find_lemma_context(query, df, selected_corpus, left_context_size=10, right_context_size=10, warnings_list=None):
    # KORPUSUJ_MIGRATION_036L4G36U3_FINAL_REGEX_CQL_TO_LEGACY_ROUTE
    if _regex_legacy_route_enabled_036l4g36u3() and _query_requires_legacy_regex_backend_036l4g36u3(query) and not _gui_sqlite_regex_route_available_after_patch96_103(df, query) and not _regex_sqlite_route_enabled_036l4g38a(df, query):
        try:
            if warnings_list is None:
                warnings_list = []
            route_df = df
            try:
                lazy_cls = globals().get("LazyCorpus")
                if lazy_cls is not None and isinstance(df, lazy_cls):
                    route_df = (globals().get("dataframes", {}) or {}).get(selected_corpus, df)
            except Exception:
                route_df = df
            _regex_legacy_route_log_036l4g36u3("route_to_legacy", query=query, selected_corpus=selected_corpus, df_type=type(df).__name__, route_df_type=type(route_df).__name__)
            try:
                # KORPUSUJ_MIGRATION_036L4G38B_LEGACY_ROUTE_DF_NORMALIZED
                route_df = _normalize_regex_legacy_route_df_036l4g38b(route_df, selected_corpus, query)
                _force_rebuild_legacy_inverted_index_for_regex_036l4g38b(selected_corpus)
                ensure_legacy_inverted_index_for_corpus(selected_corpus, route_df)
            except Exception as _idx_exc:
                _regex_legacy_route_log_036l4g36u3("legacy_index_prepare_failed", query=query, selected_corpus=selected_corpus, reason=repr(_idx_exc))
            _legacy_runtime_label_log_036l4g79(
                "legacy_activate",
                legacy_source="regex_route_dataframe",
                legacy_reason="query_requires_legacy_regex_backend+sqlite_route_disabled_or_unavailable",
                legacy_df_kind=_legacy_runtime_df_kind_036l4g79(route_df),
                legacy_query_shape=_legacy_runtime_query_shape_036l4g79(query),
                query=query,
                selected_corpus=selected_corpus,
                df_type=type(df).__name__,
                route_df_type=type(route_df).__name__,
            )
            return _legacy_adapter_call_121(
                _legacy_find_lemma_context,
                query,
                route_df,
                selected_corpus,
                left_context_size,
                right_context_size,
                warnings_list,
                legacy_source="regex_route_dataframe",
                legacy_reason="query_requires_legacy_regex_backend+sqlite_route_disabled_or_unavailable",
                route_name="regex_route_dataframe",
                extra={"original_df_type": type(df).__name__, "route_df_type": type(route_df).__name__},
            )
        except Exception as _route_exc:
            _regex_legacy_route_log_036l4g36u3("route_to_legacy_failed_falling_back", query=query, selected_corpus=selected_corpus, reason=repr(_route_exc))
    if warnings_list is None:
        warnings_list = []

    t0 = time.perf_counter()
    search_diag_log(
        "FIND_START corpus=%r query=%r df_type=%s left=%s right=%s",
        selected_corpus, query, type(df).__name__, left_context_size, right_context_size
    )

    if isinstance(df, LazyCorpus):
        try:
            search_diag_log(
                "BACKEND_TRY sqlite corpus=%r query=%r search_path=%r parquet_path=%r",
                selected_corpus, query, getattr(df, "search_path", None), getattr(df, "parquet_path", None)
            )
            result = CorpusSearchExecutor(df).search(query, left_context_size, right_context_size)
            search_diag_log(
                "BACKEND_SUCCESS sqlite corpus=%r query=%r result_type=%s elapsed=%.6fs",
                selected_corpus, query, type(result).__name__, time.perf_counter() - t0
            )
            return result
        except Exception as e:
            search_diag_log(
                "BACKEND_FAIL sqlite corpus=%r query=%r reason=%r elapsed_before_fallback=%.6fs",
                selected_corpus, query, e, time.perf_counter() - t0
            )
            logging.info("SQLite SearchCursor nie obsłużył zapytania; fallback do legacy matcher: %s", e)
            # KORPUSUJ_MIGRATION_PATCH_121_LEGACY_ADAPTER_FACADE
            # Compatibility default: keep fallback to legacy after SQLite/SearchCursor
            # exception. In strict mode, re-raise instead of masking backend bugs.
            _legacy_fallback_enabled_121 = _legacy_sqlite_exception_fallback_enabled_122()
            _legacy_strict_snapshot_122 = _legacy_sqlite_exception_strict_snapshot_122()
            try:
                if korpusuj_diagnostics_enabled_145c1():
                    logging.info("[DIAG legacy.strict] event=%r data=%r", "sqlite_exception_policy", _legacy_strict_snapshot_122)
            except Exception:
                pass
            if not _legacy_fallback_enabled_121:
                try:
                    _legacy_runtime_label_log_036l4g79(
                        "legacy_suppressed",
                        legacy_source="sqlite_fail_materialized",
                        legacy_reason="strict_no_legacy_on_sqlite_exception",
                        legacy_df_kind=_legacy_runtime_df_kind_036l4g79(df),
                        legacy_query_shape=_legacy_runtime_query_shape_036l4g79(query),
                        query=query,
                        selected_corpus=selected_corpus,
                        original_df_type=type(df).__name__,
                        sqlite_exception=repr(e),
                    )
                    if korpusuj_diagnostics_enabled_145c1():
                        logging.info(
                            "[DIAG legacy.adapter] event=%r data=%r",
                            "strict_sqlite_exception_re_raise",
                            {
                                "legacy_source": "sqlite_fail_materialized",
                                "legacy_reason": "strict_no_legacy_on_sqlite_exception",
                                "query": query,
                                "selected_corpus": selected_corpus,
                                "df_type": type(df).__name__,
                                "sqlite_exception": repr(e),
                            },
                        )
                except Exception:
                    pass
                raise

            t_mat0 = time.perf_counter()
            real_df = df.materialize()
            t_mat1 = time.perf_counter()
            search_diag_log(
                "MATERIALIZE_PARQUET corpus=%r query=%r rows=%s time=%.6fs",
                selected_corpus, query, len(real_df) if hasattr(real_df, "__len__") else "unknown", t_mat1 - t_mat0
            )

            t_idx0 = time.perf_counter()
            ensure_legacy_inverted_index_for_corpus(selected_corpus, real_df)
            t_idx1 = time.perf_counter()
            search_diag_log(
                "LEGACY_INDEX_READY corpus=%r query=%r time=%.6fs",
                selected_corpus, query, t_idx1 - t_idx0
            )
            search_diag_log("BACKEND_USE legacy_after_sqlite_fail corpus=%r query=%r", selected_corpus, query)
            _legacy_runtime_label_log_036l4g79(
                "legacy_activate",
                legacy_source="sqlite_fail_materialized",
                legacy_reason="sqlite_executor_exception",
                legacy_df_kind=_legacy_runtime_df_kind_036l4g79(real_df),
                legacy_query_shape=_legacy_runtime_query_shape_036l4g79(query),
                query=query,
                selected_corpus=selected_corpus,
                original_df_type=type(df).__name__,
                materialized_df_type=type(real_df).__name__,
            )
            return _legacy_adapter_call_121(
                _legacy_find_lemma_context,
                query,
                real_df,
                selected_corpus,
                left_context_size,
                right_context_size,
                warnings_list,
                legacy_source="sqlite_fail_materialized",
                legacy_reason="sqlite_executor_exception",
                route_name="sqlite_fail_materialized",
                extra={"original_df_type": type(df).__name__, "materialized_df_type": type(real_df).__name__},
            )

    search_diag_log("BACKEND_USE legacy_direct corpus=%r query=%r df_type=%s", selected_corpus, query, type(df).__name__)
    _legacy_runtime_label_log_036l4g79(
        "legacy_activate",
        legacy_source="legacy_direct_dataframe",
        legacy_reason="non_lazy_dataframe_direct",
        legacy_df_kind=_legacy_runtime_df_kind_036l4g79(df),
        legacy_query_shape=_legacy_runtime_query_shape_036l4g79(query),
        query=query,
        selected_corpus=selected_corpus,
        df_type=type(df).__name__,
    )
    return _legacy_adapter_call_121(
        _legacy_find_lemma_context,
        query,
        df,
        selected_corpus,
        left_context_size,
        right_context_size,
        warnings_list,
        legacy_source="legacy_direct_dataframe",
        legacy_reason="non_lazy_dataframe_direct",
        route_name="legacy_direct_dataframe",
        extra={"df_type": type(df).__name__},
    )

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
    global current_page
    current_page = 0
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
            theme = THEMES[motyw.get()]
            _listbox_panel = _StatisticsListboxPanel(
                ctk_module=ctk,
                math_module=math,
                parent_frame=parent_frame,
                sorted_lemma_freq=sorted_lemma_freq,
                vars_dict=vars_dict,
                merge_dict=merge_dict,
                update_plot_callback=update_plot_callback,
                theme=theme,
                items_per_page=items_per_page,
            )
            return _listbox_panel.container, _listbox_panel.set_data

        container_listbox, set_data_listbox = build_listbox_ui_local(
            checkboxes_frame, state.s_lemma_total_freq, lemma_vars, merge_entry_vars, update_plot
        )
        container_listbox.pack(fill="both", expand=True)

        def toggle_listboxes(*args):
            mode = wykres_sort_mode.get()
            _statistics_listbox_data = _select_statistics_listbox_data(
                mode=mode,
                s_lemma_global_tfidf=state.s_lemma_global_tfidf,
                monthly_zscore_for_use=state.monthly_zscore_for_use,
                s_lemma_global_pmw=state.s_lemma_global_pmw,
                s_lemma_total_freq=state.s_lemma_total_freq,
                unique_lemmas=state.unique_lemmas,
            )
            set_data_listbox(_statistics_listbox_data)

        # Podmiana eventów na nowy lokalny toggle_listboxes
        for trace_id in wykres_sort_mode.trace_info():
            wykres_sort_mode.trace_remove(*trace_id[0:2])
        wykres_sort_mode.trace_add("write", toggle_listboxes)
        toggle_listboxes()

        # ==========================================
        # --- NATYCHMIASTOWE PRZYWRACANIE KOLOKACJI ---
        # ==========================================
        paginator_colloc["data"] = list(getattr(state, "colloc_data", []))
        paginator_colloc["current_page"][0] = 0
        update_table(paginator_colloc)

        # ==========================================
        # --- NATYCHMIASTOWE PRZYWRACANIE PROFILU ---
        # ==========================================
        global current_profile_dict, current_profile_target_lemma
        current_profile_dict = dict(getattr(state, "current_profile_dict", {}))
        current_profile_target_lemma = getattr(state, "profile_target_lemma", None)
        profile_data = list(getattr(state, "profile_data", []))
        profile_rel_options = list(getattr(state, "profile_rel_options", []))
        profile_selected_rel = getattr(state, "profile_selected_rel", "Brak wyników")

        if current_profile_dict and profile_data:
            profile_rel_menu_btn.configure(state="normal")

            # Odtwarzamy logikę wybierania relacji po cofnięciu w historii
            display_to_key = {opt: opt.rsplit(" (", 1)[0] for opt in profile_rel_options}

            def on_rel_select_history(selected_display_name):
                profile_rel_var.set(selected_display_name)

                # LOGIKA 1: Widok z lotu ptaka w Historii
                if selected_display_name == "★ Podsumowanie profilu":
                    pagination_profile_frame.pack_forget()
                    profile_table.pack_forget()
                    profile_dashboard_frame.pack(fill="both", expand=True)
                    render_profile_dashboard(on_rel_select_history)
                    return

                # LOGIKA 2: Standardowa tabela w Historii
                profile_dashboard_frame.pack_forget()
                pagination_profile_frame.pack(fill="x", pady=(0, 5))
                profile_table.pack(fill="both", expand=True)

                actual_key = display_to_key.get(selected_display_name)
                if not actual_key: return

                rows = current_profile_dict[actual_key]
                table_rows = []
                for i, row_obj in enumerate(rows):
                    display_colloc = row_obj.collocate
                    if getattr(row_obj, "collocate_upos", ""):
                        display_colloc = f"{display_colloc} [{row_obj.collocate_upos}]"
                    table_rows.append([
                        i + 1, display_colloc, row_obj.cooc_freq, row_obj.doc_freq,
                        row_obj.global_freq, row_obj.ll_score, row_obj.mi_score,
                        row_obj.t_score, row_obj.log_dice
                    ])
                paginator_profile["data"] = table_rows
                paginator_profile["current_page"][0] = 0
                update_table(paginator_profile)
                profile_rel_var.set(selected_display_name)

            # Odbudowanie drzewa nawigacyjnego
            build_profile_tree_menu(profile_rel_options, display_to_key, on_rel_select_history)

            profile_rel_var.set(profile_selected_rel)

            paginator_profile["data"] = list(profile_data)
            paginator_profile["current_page"][0] = 0
            update_table(paginator_profile)
        else:
            profile_rel_menu_btn.configure(state="disabled")
            profile_rel_var.set("Brak wyników")
            paginator_profile["data"] = []
            update_table(paginator_profile)

        # Odbudowanie wykresów z uwzględnieniem danych ze zbuforowanego stanu!
        force_recalculate_plot()
    else:
        # Wyczyść listboxy jeśli brak dat w wybranym korpusie
        for child in checkboxes_frame.winfo_children():
            child.destroy()


# --- HISTORIA NAWIGACJI (ZAKŁADKI + WYNIKI) ---
nav_history = []
nav_index = -1
is_navigating = False  # Blokada chroniąca przed zapętleniem podczas cofania


def push_nav_state(*args):
    """Zapisuje obecny stan aplikacji do historii nawigacji."""
    global nav_history, nav_index, is_navigating

    # Jeśli właśnie trwają zautomatyzowane zmiany (bo kliknęliśmy Wstecz), nic nie zapisuj
    if is_navigating:
        return

    # Nie zapisuj pustych stanów (przed pierwszym wyszukiwaniem)
    if not current_state or not current_state.query:
        return

        # Budujemy "migawkę" obecnego stanu GUI
    state = {
        "search_state": current_state,
        "main_tab": tabview.get(),
        "sub_tab": selected_table.get() if 'selected_table' in globals() else ""
    }

    # Nie duplikuj, jeśli użytkownik np. kliknął dwa razy w tę samą zakładkę
    if nav_history and nav_history[nav_index] == state:
        return

    # Jeśli użytkownik cofnął się, a potem kliknął coś nowego -> ucinamy przyszłość (jak w przeglądarce)
    if nav_index < len(nav_history) - 1:
        nav_history = nav_history[:nav_index + 1]

    nav_history.append(state)
    nav_index += 1

    # Ogranicznik pamięci, żeby historia nie rosła w nieskończoność (opcjonalnie)
    if len(nav_history) > 50:
        nav_history.pop(0)
        nav_index -= 1

    update_nav_buttons()


def go_back():
    global nav_index
    if nav_index > 0:
        nav_index -= 1
        restore_nav_state(nav_history[nav_index])


def go_forward():
    global nav_index
    if nav_index < len(nav_history) - 1:
        nav_index += 1
        restore_nav_state(nav_history[nav_index])


def restore_nav_state(state):
    """Fizycznie zmienia widoki i ładuje dane na podstawie zapisanej 'migawki'."""
    global is_navigating
    is_navigating = True  # ZAMYKAMY nasłuchiwanie na zmiany!

    try:
        # 1. Przywróć wyniki wyszukiwania (jeśli dotyczyły innego zapytania)
        if current_state != state["search_state"]:
            restore_from_history(state["search_state"])

        # 2. Przełącz główną zakładkę (Wyniki / Statystyki / Trendy)
        if tabview.get() != state["main_tab"]:
            tabview.set(state["main_tab"])

        # 3. Przełącz pod-zakładkę (Tylko jeśli jesteśmy w "Statystyki")
        if state["main_tab"] == "Statystyki" and state["sub_tab"]:
            if selected_table.get() != state["sub_tab"]:
                selected_table.set(state["sub_tab"])
                show_table(state["sub_tab"])
    finally:
        is_navigating = False  # OTWIERAMY nasłuchiwanie ponownie
        update_nav_buttons()


def update_nav_buttons():
    """Włącza/Wyłącza przyciski w zależności od miejsca w historii."""
    if 'btn_nav_back' not in globals(): return

    btn_nav_back.configure(state="normal" if nav_index > 0 else "disabled")
    btn_nav_forward.configure(state="normal" if nav_index < len(nav_history) - 1 else "disabled")

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
    _technical_warning_markers = ('fallback', 'wolniejszego trybu', 'trybu fallback', 'SQLite', 'CQL')
    warnings_list = [w for w in (warnings_list or []) if not any(m.lower() in str(w).lower() for m in _technical_warning_markers)]
    if not warnings_list:
        return
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

        # stare SRL
        "srl", "srl_role", "srl_pred", "srl_pred_lemma",

        # nowe SRL predicate-centred
        "srl_is_pred",

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
                            (
                                    "coref", "head", "dependent", "window_base", "window_orth",
                                    "srl_arg", "srl_tmp", "srl_loc", "srl_mnr", "srl_dir",
                                    "srl_adv", "srl_mod", "srl_neg", "srl_cau", "srl_prp", "srl_dis", "srl_ext"
                            )):
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


# --- ZMIENNE GLOBALNE DLA FILTRU WSD ---
current_wsd_lemma = None
unfiltered_wsd_results = None  # Tu będziemy trzymać kopię wyników przed filtrowaniem


def filter_by_selected_sense(choice):
    """Odfiltrowuje tablicę wyników pozostawiając tylko wybraną ramę."""
    global full_results_sorted, current_page, unfiltered_wsd_results

    # Zabezpieczenie oryginalnych wyników przed pierwszym filtrowaniem
    if unfiltered_wsd_results is None:
        unfiltered_wsd_results = list(full_results_sorted)

    if choice == "Wszystkie ramy":
        # Powrót do pełnych wyników w ułamku sekundy
        full_results_sorted = list(unfiltered_wsd_results)
        unfiltered_wsd_results = None
        current_page = 0
        label_results_count.configure(text=f"Znaleziono: {len(full_results_sorted)}")
        display_page(global_query, global_selected_corpus)
        return

    # Wyciągamy ID ramy z tekstu wyboru, np.:
    # "Rama semantyczna 1: UE, unia..." -> 1
    # "Rama kontekstowa 2: mówić, powiedzieć..." -> 2
    # Zostawiamy też fallback kompatybilności dla starych etykiet typu "Sens 1: ..."
    try:
        frame_id = None

        if choice.startswith("Rama semantyczna"):
            frame_id = int(choice.split("Rama semantyczna", 1)[1].split(":", 1)[0].strip())
        elif choice.startswith("Rama kontekstowa"):
            frame_id = int(choice.split("Rama kontekstowa", 1)[1].split(":", 1)[0].strip())
        elif choice.startswith("Profil"):
            frame_id = int(choice.split("Profil", 1)[1].split(":", 1)[0].strip())
        elif choice.startswith("Sens"):
            frame_id = int(choice.split("Sens", 1)[1].split(":", 1)[0].strip())
        else:
            return
    except ValueError:
        return

    loading_win = ctk.CTkToplevel(app)
    loading_win.title("Filtrowanie ram")
    loading_win.geometry("360x120")
    loading_win.attributes("-topmost", True)
    x = app.winfo_x() + (app.winfo_width() // 2) - 180
    y = app.winfo_y() + (app.winfo_height() // 2) - 60
    loading_win.geometry(f"+{x}+{y}")
    ctk.CTkLabel(
        loading_win,
        text=f"Filtrowanie {len(unfiltered_wsd_results)} wyników...\nTo może chwilę potrwać.",
        font=("Verdana", 12)
    ).pack(expand=True)
    loading_win.update()

    try:
        filtered = []
        df = dataframes[global_selected_corpus]

        # Zawsze filtrujemy z "pełnej" puli zapytania, żeby móc przeskakiwać między ramami
        for res in unfiltered_wsd_results:
            r_idx = res[11]
            match_start = res[12]
            match_end = res[13] if len(res) > 13 else match_start

            row_data = df.loc[r_idx]
            tokens = row_data.tokens
            lemmas = row_data.lemmas
            sentence_ids = row_data.sentence_ids

            # Szukamy, pod którym indeksem w dopasowanym fragmencie ukrywa się nasz wyraz
            target_idx = match_start
            for i in range(match_start, match_end + 1):
                if lemmas[i].lower() == current_wsd_lemma.lower():
                    target_idx = i
                    break

            sent_id = sentence_ids[target_idx]
            sent_start = target_idx
            while sent_start > 0 and sentence_ids[sent_start - 1] == sent_id:
                sent_start -= 1
            sent_end = target_idx
            while sent_end < len(sentence_ids) and sentence_ids[sent_end] == sent_id:
                sent_end += 1

            sentence_tokens = [
                {"lemma": lemmas[i], "form": tokens[i]}
                for i in range(sent_start, sent_end)
            ]
            local_target_idx = target_idx - sent_start

            # Właściwa weryfikacja: silnik nadal zwraca ID ramy przez stare pole/ścieżkę sense_id
            sid = semantic_engine.disambiguate_instance(
                sentence_tokens,
                local_target_idx,
                current_wsd_lemma
            )
            if sid == frame_id:
                filtered.append(res)

        full_results_sorted = filtered
        current_page = 0
        label_results_count.configure(
            text=f"Rama {frame_id}: {len(filtered)} z {len(unfiltered_wsd_results)}"
        )
        display_page(global_query, global_selected_corpus)

    finally:
        loading_win.destroy()


def sort_search_results_in_place(results, choice):
    if _is_searchcursor_like(results):
        return
    """
    Sortuje listę wyników konkordancji in-place.

    Indeksy wyniku:
    x[3] = matched_text / orth
    x[4] = matched_lemmas / base
    x[6] = title
    x[7] = author
    x[9] = left_context
    x[10] = right_context
    """
    if not results:
        return

    import string
    from collections import Counter

    def first_real_token(text):
        if not text:
            return ""
        for tok in str(text).split():
            cleaned = tok.strip(string.punctuation).lower()
            if cleaned:
                return cleaned
        return ""

    def last_real_token(text):
        if not text:
            return ""
        for tok in reversed(str(text).split()):
            cleaned = tok.strip(string.punctuation).lower()
            if cleaned:
                return cleaned
        return ""

    if choice == "Data publikacji":
        results.sort(key=lambda x: str(x[0]) if x[0] else "")

    elif choice == "Tytuł":
        results.sort(key=lambda x: str(x[6]) if x[6] else "")

    elif choice == "Autor":
        results.sort(key=lambda x: str(x[7]) if x[7] else "")

    elif choice == "Alfabetycznie":
        results.sort(key=lambda x: str(x[3]).lower() if x[3] else "")

    elif choice == "Prawy kontekst":
        results.sort(key=lambda x: first_real_token(x[10]))

    elif choice == "Lewy kontekst":
        results.sort(key=lambda x: last_real_token(x[9]))

    elif choice == "Frekwencja base":
        base_counter = Counter(str(x[4]) for x in results)

        results.sort(
            key=lambda x: (
                -base_counter[str(x[4])],      # najczęstsze base najpierw
                str(x[4]).lower(),             # potem alfabetycznie po base
                str(x[3]).lower()              # potem po orth
            )
        )

    elif choice == "Frekwencja orth":
        orth_counter = Counter(str(x[3]) for x in results)

        results.sort(
            key=lambda x: (
                -orth_counter[str(x[3])],      # najczęstsze orth najpierw
                str(x[3]).lower(),             # potem alfabetycznie po orth
                str(x[4]).lower()              # potem po base
            )
        )



# Try to prepare a metadata-sorted first-page preview without full materialization.

def _searchcursor_hit_doc_pos(hit):
    try:
        if isinstance(hit, dict):
            doc_id = hit.get("doc_id", hit.get("doc", hit.get("row_id", hit.get("row"))))
            pos = hit.get("position", hit.get("pos", hit.get("token_idx", hit.get("i", 0))))
            return int(doc_id), int(pos or 0)
        if isinstance(hit, (tuple, list)) and hit:
            doc_id = hit[0]
            pos = hit[1] if len(hit) > 1 else 0
            return int(doc_id), int(pos or 0)
        doc_id = getattr(hit, "doc_id", getattr(hit, "doc", getattr(hit, "row_id", None)))
        pos = getattr(hit, "position", getattr(hit, "pos", getattr(hit, "token_idx", 0)))
        return int(doc_id), int(pos or 0)
    except Exception:
        return None, 0


def _search_cursor_index_connection_or_path_035e(cursor):
    try:
        index = getattr(cursor, "index", None)
    except Exception:
        index = None
    if index is not None:
        for con_attr in ("con", "conn", "connection", "db"):
            try:
                con = getattr(index, con_attr, None)
                if con is not None and hasattr(con, "execute"):
                    return con, None
            except Exception:
                pass
        for path_attr in ("index_path", "search_path", "path", "db_path", "database_path", "filename"):
            try:
                value = getattr(index, path_attr, None)
                if value:
                    return None, str(value)
            except Exception:
                pass
    for path_attr in ("search_path", "index_path"):
        try:
            value = getattr(cursor, path_attr, None)
            if value:
                return None, str(value)
        except Exception:
            pass
    try:
        corpus_path = getattr(cursor, "corpus_path", None)
        if corpus_path:
            return None, str(Path(corpus_path).with_suffix(".search"))
    except Exception:
        pass
    return None, None


def _load_doc_metadata_sort_map_035e(cursor, sort_option, doc_ids):
    column_by_sort = {"Data publikacji": "date", "Autor": "author", "Tytuł": "title"}
    col = column_by_sort.get(str(sort_option or ""))
    if not col or not doc_ids:
        return None
    con, path = _search_cursor_index_connection_or_path_035e(cursor)
    close_after = False
    try:
        if con is None:
            if not path:
                return None
            con = sqlite3.connect(str(path))
            close_after = True
        ids = sorted(set(int(x) for x in doc_ids if x is not None))
        out = {}
        for start in range(0, len(ids), 800):
            chunk = ids[start:start + 800]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT doc_id, {col} FROM doc_stats WHERE doc_id IN ({placeholders})"
            for doc_id, value in con.execute(query, chunk).fetchall():
                out[int(doc_id)] = "" if value is None else str(value)
        return out
    except Exception as e:
        try:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG search.metadata_sort.fail] sort=%r reason=%r", sort_option, e, exc_info=True)
        except Exception:
            pass
        return None
    finally:
        if close_after:
            try:
                con.close()
            except Exception:
                pass


def _try_prepare_metadata_sorted_searchcursor_preview(cursor, sort_option):
    supported = {"Data publikacji", "Autor", "Tytuł"}
    sort_option = str(sort_option or "").strip()
    if sort_option not in supported:
        return False
    hits = getattr(cursor, "_hits", None)
    if not isinstance(hits, list) or not hits:
        return False
    doc_pos = [_searchcursor_hit_doc_pos(h) for h in hits]
    doc_ids = [dp[0] for dp in doc_pos if dp and dp[0] is not None]
    meta_map = _load_doc_metadata_sort_map_035e(cursor, sort_option, doc_ids)
    if not meta_map:
        return False

    def key_for_hit(hit):
        doc_id, pos = _searchcursor_hit_doc_pos(hit)
        value = meta_map.get(doc_id, "")
        if sort_option in ("Autor", "Tytuł"):
            value = value.lower()
        return (value, doc_id if doc_id is not None else -1, pos or 0)

    try:
        t0 = time.perf_counter()
        hits.sort(key=key_for_hit)
        try:
            cache = getattr(cursor, "_result_cache", None)
            if hasattr(cache, "clear"):
                cache.clear()
        except Exception:
            pass
        return True
    except Exception as e:
        try:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG search.metadata_sort.fail] sort=%r reason=%r", sort_option, e, exc_info=True)
        except Exception:
            pass
        return False




# If metadata preview is unavailable, try an alphabetical first-page preview for simple base/orth queries.

def _decode_sqlite_json_blob_035f(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple, dict)):
            return value
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytes):
            import json as _json_035f
            import zlib as _zlib_035f
            raw = bytes(value)
            try:
                return _json_035f.loads(_zlib_035f.decompress(raw).decode("utf-8"))
            except Exception:
                pass
            try:
                return _json_035f.loads(raw.decode("utf-8"))
            except Exception:
                return default
        if isinstance(value, str):
            import json as _json_035f
            try:
                return _json_035f.loads(value)
            except Exception:
                return default
    except Exception:
        return default
    return default


def _as_list_035f(value):
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
    return []


def _extract_simple_alpha_query_attr_value(query):
    """Return (attr, value) only for exact simple [base="..."]/[orth="..."] queries."""
    try:
        import html as _html_035f
        import re as _re_035f
        q = _html_035f.unescape(str(query or "").strip())
        m = _re_035f.match(r'^\s*\[\s*(base|orth)\s*=\s*(["\'])(.*?)\2\s*\]\s*$', q, _re_035f.IGNORECASE)
        if not m:
            return None, None
        attr = m.group(1).lower()
        value = m.group(3)
        if not value or any(ch in value for ch in ["|", "*", "?", "(", ")", "/", "[", "]"]):
            return None, None
        if any(ch.isspace() for ch in value):
            return None, None
        return attr, value
    except Exception:
        return None, None


def _load_doc_tokens_for_searchcursor_hits(cursor, hits):
    try:
        doc_ids = []
        for hit in hits:
            doc_id, _pos = _searchcursor_hit_doc_pos(hit)
            if doc_id is not None:
                doc_ids.append(int(doc_id))
        ids = sorted(set(doc_ids))
        if not ids:
            return None
        con, path = _search_cursor_index_connection_or_path_035e(cursor)
        close_after = False
        if con is None:
            if not path:
                return None
            con = sqlite3.connect(str(path))
            close_after = True
        try:
            out = {}
            for start in range(0, len(ids), 800):
                chunk = ids[start:start + 800]
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT doc_id, tokens FROM docs WHERE doc_id IN ({placeholders})"
                for doc_id, tokens_raw in con.execute(query, chunk).fetchall():
                    out[int(doc_id)] = _as_list_035f(_decode_sqlite_json_blob_035f(tokens_raw, []))
            return out
        finally:
            if close_after:
                try:
                    con.close()
                except Exception:
                    pass
    except Exception as e:
        return None


def _try_prepare_alpha_sorted_searchcursor_preview(cursor, sort_option, query):
    try:
        if str(sort_option or "").strip() != "Alfabetycznie":
            return False
        attr, value = _extract_simple_alpha_query_attr_value(query)
        if attr not in ("base", "orth"):
            return False
        hits = getattr(cursor, "_hits", None)
        if not isinstance(hits, list) or not hits:
            return False
        tokens_by_doc = _load_doc_tokens_for_searchcursor_hits(cursor, hits)
        if not tokens_by_doc:
            return False
        missing = 0
        def key_for_hit(hit):
            nonlocal missing
            doc_id, pos = _searchcursor_hit_doc_pos(hit)
            toks = tokens_by_doc.get(doc_id, [])
            token = ""
            try:
                ipos = int(pos or 0)
                if 0 <= ipos < len(toks):
                    token = toks[ipos]
                else:
                    missing += 1
            except Exception:
                missing += 1
                ipos = 0
            return (str(token).lower(), doc_id if doc_id is not None else -1, ipos)
        t0 = time.perf_counter()
        hits.sort(key=key_for_hit)
        try:
            cache = getattr(cursor, "_result_cache", None)
            if hasattr(cache, "clear"):
                cache.clear()
        except Exception:
            pass
        return missing == 0
    except Exception as e:
        try:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG search.alpha_sort.fail] query=%r reason=%r", query, e, exc_info=True)
        except Exception:
            pass
        return False

def resort_results(choice):
    global full_results_sorted, current_page, global_query, global_selected_corpus

    if not full_results_sorted:
        return

    sort_search_results_in_place(full_results_sorted, choice)

    current_page = 0
    display_page(global_query, global_selected_corpus)

# Funkcja obsługująca wyszukiwanie


# KORPUSUJ_MIGRATION_036L4G34C_CAPTURE_SEARCH_REQUEST
def _search_request_capture_enabled_036l4g34c():
    try:
        cfg = globals().get("config", {}) or {}
        val = cfg.get("search_request_capture_enabled", True)
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "tak", "on"}
        return bool(val)
    except Exception:
        return True


def _safe_widget_get_036l4g34c(name, default=None):
    try:
        obj = globals().get(name)
        if obj is None:
            return default
        if hasattr(obj, "get"):
            try:
                if name == "entry_query" or obj.__class__.__name__.lower().endswith("textbox"):
                    return obj.get("1.0", "end").strip()
            except TypeError:
                pass
            return obj.get()
    except Exception:
        pass
    return default


def _safe_int_036l4g34c(value, default=10):
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _log_search_request_boundary_event(status, **data):
    try:
        import logging as _logging_036l4g34c
        if korpusuj_verbose_diagnostics_enabled_145c1():
            pass
    except Exception:
        pass


def _build_search_request_from_boundary_036l4g34c(runtime_locals=None):
    runtime_locals = runtime_locals or {}
    try:
        from korpusuj.search.search_service import SearchRequest
    except Exception as exc:
        _log_search_request_boundary_event("import_failed", reason=repr(exc))
        return None

    query = runtime_locals.get("query")
    if query is None:
        query = _safe_widget_get_036l4g34c("entry_query", "")
    query = str(query or "").strip()

    corpus_name = runtime_locals.get("selected_corpus") or runtime_locals.get("corpus_name")
    if corpus_name is None:
        corpus_name = _safe_widget_get_036l4g34c("corpus_var", "")
    corpus_name = str(corpus_name or "").strip()

    left_context = runtime_locals.get("left_context_size") or runtime_locals.get("left_context")
    if left_context is None:
        left_context = _safe_widget_get_036l4g34c("entry_left_context", 10)
    left_context = _safe_int_036l4g34c(left_context, 10)

    right_context = runtime_locals.get("right_context_size") or runtime_locals.get("right_context")
    if right_context is None:
        right_context = _safe_widget_get_036l4g34c("entry_right_context", 10)
    right_context = _safe_int_036l4g34c(right_context, 10)

    sort_option = runtime_locals.get("sort_option")
    if sort_option is None:
        sort_option = _safe_widget_get_036l4g34c("sort_option_var", None)
    if sort_option is not None:
        sort_option = str(sort_option).strip()

    date_from = runtime_locals.get("date_from") or runtime_locals.get("date_start")
    if date_from is None:
        date_from = _safe_widget_get_036l4g34c("date_from_var", None) or _safe_widget_get_036l4g34c("date_start_var", None)
    date_to = runtime_locals.get("date_to") or runtime_locals.get("date_end")
    if date_to is None:
        date_to = _safe_widget_get_036l4g34c("date_to_var", None) or _safe_widget_get_036l4g34c("date_end_var", None)
    selected_sense = runtime_locals.get("selected_sense")
    if selected_sense is None:
        selected_sense = globals().get("current_wsd_lemma", None)

    kwargs = {
        "query": query,
        "corpus_name": corpus_name,
        "left_context": left_context,
        "right_context": right_context,
        "sort_option": sort_option,
        "date_from": date_from,
        "date_to": date_to,
        "selected_sense": selected_sense,
    }
    try:
        import dataclasses as _dataclasses_036l4g34c
        if _dataclasses_036l4g34c.is_dataclass(SearchRequest):
            allowed = {f.name for f in _dataclasses_036l4g34c.fields(SearchRequest)}
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        pass
    try:
        return SearchRequest(**kwargs)
    except Exception:
        for keys in (("query", "corpus_name", "left_context", "right_context", "sort_option"), ("query", "corpus_name", "left_context", "right_context"), ("query", "corpus_name")):
            try:
                return SearchRequest(**{k: kwargs[k] for k in keys if k in kwargs})
            except Exception:
                pass
        return dict(kwargs)


def _capture_search_request_at_boundary(runtime_locals=None):
    try:
        if not _search_request_capture_enabled_036l4g34c():
            return None
        req = _build_search_request_from_boundary_036l4g34c(runtime_locals or {})
        if req is None:
            _log_search_request_boundary_event("missing")
            return None
        gs = globals().get("gui_state")
        if not isinstance(gs, dict):
            gs = {}
            globals()["gui_state"] = gs
        gs["search_request"] = req
        try:
            state = globals().get("current_state", None)
            if state is not None:
                setattr(state, "search_request", req)
        except Exception:
            pass
        _log_search_request_boundary_event("captured", request_type=type(req).__name__)
        return req
    except Exception as exc:
        _log_search_request_boundary_event("capture_failed", reason=repr(exc))
        return None
# END KORPUSUJ_MIGRATION_036L4G34C_CAPTURE_SEARCH_REQUEST


# KORPUSUJ_MIGRATION_036L4G35A_QUERY_FROM_SEARCH_REQUEST




# END KORPUSUJ_MIGRATION_036L4G35A_QUERY_FROM_SEARCH_REQUEST


# KORPUSUJ_MIGRATION_036L4G35B_CORE_FIELDS_FROM_SEARCH_REQUEST
def _search_request_core_from_request_enabled():
    try:
        cfg = globals().get("config", {}) or {}
        val = cfg.get("search_thread_use_request_core", True)
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "tak", "on"}
        return bool(val)
    except Exception:
        return True


def _get_current_gui_search_request():
    try:
        gs = globals().get("gui_state", None)
        if isinstance(gs, dict):
            req = gs.get("search_request")
            if req is not None:
                return req
    except Exception:
        pass
    try:
        state = globals().get("current_state", None)
        req = getattr(state, "search_request", None) if state is not None else None
        if req is not None:
            return req
    except Exception:
        pass
    return None

def _get_search_request_field(req, *names):
    try:
        if req is None:
            return None
        for name in names:
            if isinstance(req, dict):
                value = req.get(name)
            else:
                value = getattr(req, name, None)
            if value is not None:
                if isinstance(value, str):
                    value = value.strip()
                    return value if value else None
                return value
    except Exception:
        pass
    return None

def _coerce_search_request_int(value, fallback):
    try:
        if value is None:
            return int(fallback)
        return int(str(value).strip())
    except Exception:
        try:
            return int(fallback)
        except Exception:
            return 10

def _is_search_request_corpus_known(corpus_name):
    try:
        if not corpus_name:
            return False
        dfs = globals().get("dataframes", None)
        if isinstance(dfs, dict) and corpus_name in dfs:
            return True
        files = globals().get("files", None)
        if isinstance(files, dict) and corpus_name in files:
            return True
        opts = globals().get("corpus_options", None)
        if opts is not None and corpus_name in opts:
            return True
    except Exception:
        pass
    return False

def _resolve_search_core_fields_from_request_or_fallback(runtime_locals=None, ui_state=None):
    runtime_locals = runtime_locals or {}
    ui_state = ui_state if isinstance(ui_state, dict) else {}
    fallback_query = str(runtime_locals.get("query", ui_state.get("query", "")) or "").strip()
    fallback_corpus = str(runtime_locals.get("selected_corpus", ui_state.get("selected_corpus", ui_state.get("corpus_name", ""))) or "").strip()
    fallback_left = runtime_locals.get("left_context_size", ui_state.get("left_context", 10))
    fallback_right = runtime_locals.get("right_context_size", ui_state.get("right_context", 10))
    fallback_sort = runtime_locals.get("sort_option", ui_state.get("sort_option", None))
    fallback_left_int = _coerce_search_request_int(fallback_left, 10)
    fallback_right_int = _coerce_search_request_int(fallback_right, 10)
    fallback_result = (fallback_query, fallback_corpus, fallback_left_int, fallback_right_int, fallback_sort)
    try:
        if not _search_request_core_from_request_enabled():
            _log_search_request_boundary_event("disabled", query=fallback_query, corpus_name=fallback_corpus)
            return fallback_result
        req = _get_current_gui_search_request()
        if req is None:
            _log_search_request_boundary_event("fallback", reason="missing_request", query=fallback_query, corpus_name=fallback_corpus)
            return fallback_result
        query = _get_search_request_field(req, "query") or fallback_query
        corpus = _get_search_request_field(req, "corpus_name", "selected_corpus") or fallback_corpus
        if not _is_search_request_corpus_known(corpus):
            _log_search_request_boundary_event("fallback", reason="unknown_request_corpus", request_corpus=corpus, fallback_corpus=fallback_corpus)
            corpus = fallback_corpus
        left_context = _coerce_search_request_int(_get_search_request_field(req, "left_context"), fallback_left_int)
        right_context = _coerce_search_request_int(_get_search_request_field(req, "right_context"), fallback_right_int)
        sort_option = _get_search_request_field(req, "sort_option")
        if sort_option is None:
            sort_option = fallback_sort
        mismatches = []
        if query != fallback_query:
            mismatches.append("query")
        if corpus != fallback_corpus:
            mismatches.append("corpus_name")
        if left_context != fallback_left_int:
            mismatches.append("left_context")
        if right_context != fallback_right_int:
            mismatches.append("right_context")
        if str(sort_option or "") != str(fallback_sort or ""):
            mismatches.append("sort_option")
        if mismatches:
            _log_search_request_boundary_event("using_request_mismatch", fields=mismatches, query=query, corpus_name=corpus)
        else:
            _log_search_request_boundary_event("using_request", query=query, corpus_name=corpus)
        return query, corpus, left_context, right_context, sort_option
    except Exception as exc:
        _log_search_request_boundary_event("fallback", reason=repr(exc), query=fallback_query, corpus_name=fallback_corpus)
        return fallback_result
# END KORPUSUJ_MIGRATION_036L4G35B_CORE_FIELDS_FROM_SEARCH_REQUEST


# KORPUSUJ_MIGRATION_036L4G36A_CAPTURE_BACKEND_CONTEXT
def _search_backend_context_capture_enabled():
    try:
        cfg = globals().get("config", {}) or {}
        val = cfg.get("search_backend_context_capture_enabled", True)
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "tak", "on"}
        return bool(val)
    except Exception:
        return True

def _log_search_backend_context_event(status, **data):
    try:
        import logging as _logging_036l4g36a
        if korpusuj_verbose_diagnostics_enabled_145c1():
            pass
    except Exception:
        pass

def _search_backend_context_existing_request_036l4g36a():
    try:
        gs = globals().get("gui_state", None)
        if isinstance(gs, dict):
            req = gs.get("search_request")
            if req is not None:
                return req
    except Exception:
        pass
    try:
        state = globals().get("current_state", None)
        req = getattr(state, "search_request", None) if state is not None else None
        if req is not None:
            return req
    except Exception:
        pass
    return None

def _search_backend_context_safe_path_036l4g36a(value):
    try:
        if value is None:
            return None
        return str(value)
    except Exception:
        return None

def _search_backend_context_file_exists_036l4g36a(path):
    try:
        if not path:
            return False
        from pathlib import Path as _Path_036l4g36a
        return _Path_036l4g36a(str(path)).exists()
    except Exception:
        return False

def _search_backend_context_sidecar_036l4g36a(parquet_path, suffix):
    try:
        if not parquet_path:
            return None
        from pathlib import Path as _Path_036l4g36a
        return str(_Path_036l4g36a(str(parquet_path)).with_suffix(suffix))
    except Exception:
        return None

def _search_backend_context_len_or_none_036l4g36a(value):
    try:
        return len(value)
    except Exception:
        return None

def _search_backend_context_metadata_columns_036l4g36a(df, limit=200):
    try:
        excluded = globals().get("LINGUISTIC_COLUMNS_EXCLUDED_FROM_METADATA", set()) or set()
        cols = []
        if hasattr(df, "columns"):
            cols = list(getattr(df, "columns"))
        elif hasattr(df, "metadata_columns"):
            cols = list(getattr(df, "metadata_columns"))
        out = []
        for c in cols:
            sc = str(c)
            if sc not in excluded:
                out.append(sc)
        return out[:limit]
    except Exception:
        return []

def _search_backend_context_indexed_attrs_036l4g36a():
    try:
        fn = globals().get("get_search_indexed_attrs")
        if callable(fn):
            return tuple(fn())
    except Exception:
        pass
    try:
        return tuple(globals().get("DEFAULT_INDEXED_ATTRS", ("base", "orth", "pos", "upos", "deprel", "ner")))
    except Exception:
        return ("base", "orth")

def _make_search_backend_context_payload(runtime_locals=None):
    runtime_locals = runtime_locals or {}
    corpus_name = runtime_locals.get("selected_corpus") or runtime_locals.get("corpus_name")
    if corpus_name is None:
        try:
            req = _search_backend_context_existing_request_036l4g36a()
            if isinstance(req, dict):
                corpus_name = req.get("corpus_name") or req.get("selected_corpus")
            else:
                corpus_name = getattr(req, "corpus_name", None) or getattr(req, "selected_corpus", None)
        except Exception:
            pass
    corpus_name = str(corpus_name or "").strip()
    df = runtime_locals.get("df")
    search_df = runtime_locals.get("search_df")
    try:
        parquet_path = None
        files = globals().get("files", None)
        if isinstance(files, dict) and corpus_name in files:
            parquet_path = files.get(corpus_name)
        if parquet_path is None:
            parquet_path = getattr(df, "parquet_path", None) or getattr(search_df, "parquet_path", None)
        parquet_path = _search_backend_context_safe_path_036l4g36a(parquet_path)
        search_path = getattr(search_df, "search_path", None) or getattr(df, "search_path", None)
        if search_path is None:
            search_path = _search_backend_context_sidecar_036l4g36a(parquet_path, ".search")
        search_path = _search_backend_context_safe_path_036l4g36a(search_path)
        dep_cache_path = _search_backend_context_sidecar_036l4g36a(parquet_path, ".dep_cache")
        lazy_cls = globals().get("LazyCorpus", None)
        df_is_lazy = bool(lazy_cls is not None and df is not None and isinstance(df, lazy_cls))
        search_df_is_lazy = bool(lazy_cls is not None and search_df is not None and isinstance(search_df, lazy_cls))
        metadata_columns = _search_backend_context_metadata_columns_036l4g36a(df)
        payload = {
            "corpus_name": corpus_name,
            "parquet_path": parquet_path,
            "search_path": search_path,
            "dep_cache_path": dep_cache_path,
            "has_parquet": _search_backend_context_file_exists_036l4g36a(parquet_path),
            "has_search_index": _search_backend_context_file_exists_036l4g36a(search_path),
            "has_dep_cache": _search_backend_context_file_exists_036l4g36a(dep_cache_path),
            "df_type": type(df).__name__ if df is not None else None,
            "search_df_type": type(search_df).__name__ if search_df is not None else None,
            "df_is_lazy": df_is_lazy,
            "search_df_is_lazy": search_df_is_lazy,
            "stats_rows": _search_backend_context_len_or_none_036l4g36a(df),
            "search_rows": _search_backend_context_len_or_none_036l4g36a(search_df),
            "indexed_attrs": _search_backend_context_indexed_attrs_036l4g36a(),
            "metadata_columns": metadata_columns,
            "metadata_column_count": len(metadata_columns),
            "config_snapshot": {
                "index_profile": (globals().get("config", {}) or {}).get("index_profile"),
                "regex_sqlite_enabled": (globals().get("config", {}) or {}).get("regex_sqlite_enabled"),
                "regex_sqlite_broad_policy": (globals().get("config", {}) or {}).get("regex_sqlite_broad_policy"),
                "dependency_cache_ram_mode": (globals().get("config", {}) or {}).get("dependency_cache_ram_mode"),
            },
        }
        return payload
    except Exception as exc:
        _log_search_backend_context_event("payload_failed", reason=repr(exc), corpus_name=corpus_name)
        return {"corpus_name": corpus_name}

def _build_search_backend_context_object(payload):
    try:
        from korpusuj.search.search_service import SearchBackendContext
    except Exception:
        return dict(payload)
    try:
        import dataclasses as _dataclasses_036l4g36a
        if _dataclasses_036l4g36a.is_dataclass(SearchBackendContext):
            allowed = {f.name for f in _dataclasses_036l4g36a.fields(SearchBackendContext)}
            kwargs = {k: v for k, v in payload.items() if k in allowed}
            try:
                return SearchBackendContext(**kwargs)
            except TypeError:
                # If the scaffold has only a few fields, try minimal known subsets.
                for keys in (("corpus_name", "parquet_path", "search_path"), ("corpus_name",), ("find_lemma_context_adapter",)):
                    subset = {k: payload.get(k) for k in keys if k in allowed and k in payload}
                    try:
                        return SearchBackendContext(**subset)
                    except Exception:
                        pass
    except Exception:
        pass
    return dict(payload)

def _capture_search_backend_context(runtime_locals=None):
    try:
        if not _search_backend_context_capture_enabled():
            return None
        payload = _make_search_backend_context_payload(runtime_locals or {})
        ctx = _build_search_backend_context_object(payload)
        gs = globals().get("gui_state")
        if not isinstance(gs, dict):
            gs = {}
            globals()["gui_state"] = gs
        gs["search_backend_context"] = ctx
        gs["search_backend_context_snapshot_036l4g36a"] = payload
        try:
            state = globals().get("current_state", None)
            if state is not None:
                setattr(state, "search_backend_context", ctx)
                setattr(state, "search_backend_context_snapshot_036l4g36a", payload)
        except Exception:
            pass
        _log_search_backend_context_event(
            "captured",
            context_type=type(ctx).__name__,
            corpus_name=payload.get("corpus_name"),
            has_search_index=payload.get("has_search_index"),
            has_dep_cache=payload.get("has_dep_cache"),
            search_df_type=payload.get("search_df_type"),
            metadata_column_count=payload.get("metadata_column_count"),
        )
        return ctx
    except Exception as exc:
        _log_search_backend_context_event("capture_failed", reason=repr(exc))
        return None


def _prepare_and_find_search_backend_results(
    *,
    query,
    selected_corpus,
    left_context_size,
    right_context_size,
    sort_option=None,
    ui_state=None,
    search_token=None,
    t_start=None,
):
    """Prepare the shared search backend, execute the query and return the payload consumed by the GUI continuation.
    
    The helper stops before GUI paging, result presentation, statistics and widget scheduling.
    """

    validate_query_for_ui(query)
    selected_corpus = str(selected_corpus or "").strip()
    if not selected_corpus or selected_corpus not in dataframes:
        raise QueryValidationError(
            "Brak aktywnego korpusu. Najpierw wczytaj i wybierz korpus."
        )
    df = dataframes[selected_corpus]
    # Keep the full DataFrame for statistics/panels/fallbacks.
    # For search execution, prefer LazyCorpus/SQLite .search where available.
    search_df = _make_lazy_corpus_for_search(selected_corpus, df)
    # Capture backend context for diagnostics and downstream GUI state.
    _capture_search_backend_context(locals())
    # Retired request/context validation remains detached; smoke scanners cover this boundary.
    try:
        if korpusuj_diagnostics_enabled_145c1():
            logging.info(
                "[DIAG search.route] corpus=%r stats_df_type=%s search_df_type=%s",
                selected_corpus, type(df).__name__, type(search_df).__name__
            )
    except Exception:
        pass

    warnings_list = []

    # Local complete state for one search run.
    local_state = SearchState()
    local_state.query = query
    local_state.corpus = selected_corpus
    t_parsed = time.perf_counter()
    # Timing: cost of find_lemma_context/backend before count/materialization.
    t_find_start_035d = time.perf_counter()

    results = find_lemma_context(
        query,
        search_df,
        selected_corpus,
        left_context_size,
        right_context_size,
        warnings_list=warnings_list
    )
    t_find_done_035d = time.perf_counter()

    return {
        "query": query,
        "selected_corpus": selected_corpus,
        "left_context_size": left_context_size,
        "right_context_size": right_context_size,
        "sort_option": sort_option,
        "ui_state": ui_state,
        "df": df,
        "search_df": search_df,
        "warnings_list": warnings_list,
        "local_state": local_state,
        "results": results,
        "t_parsed": t_parsed,
        "t_find_start_035d": t_find_start_035d,
        "t_find_done_035d": t_find_done_035d,
    }

# END KORPUSUJ_MIGRATION_036L4G39D_BACKEND_PREPARE_AND_FIND


# KORPUSUJ_MIGRATION_036L4G39F_SEARCHCURSOR_COUNT_HITS

def _count_searchcursor_hits_with_fast_estimate(results, *, search_token=None):
    """Count final SearchCursor hits, using an exact estimate only when the cursor guarantees exactness."""
    try:
        from korpusuj.search.result_materialization import count_final_searchcursor_hits

        payload = count_final_searchcursor_hits(
            results,
            search_token=search_token,
            logger=logging,
            perf_counter=time.perf_counter,
        )
        data = dict(payload or {})
        total_hits = int(data.get("total_hits", data.get("exact_hits", 0)) or 0)
        strategy = str(data.get("strategy") or "")
        source = str(data.get("source") or "final_count")

        # Preserve legacy GUI timing keys expected by existing diagnostic/log code.
        # The final helper exposes t_count_start/t_count_done for both exact-estimate
        # and exact-materialization paths; map them onto the historical 035d keys.
        t_start = data.get("t_count_start")
        t_done = data.get("t_count_done")
        if t_start is None:
            t_start = time.perf_counter()
        if t_done is None:
            t_done = t_start

        out = dict(data)
        out["total_hits"] = total_hits
        out["exact_hits"] = int(data.get("exact_hits", total_hits) or total_hits)
        out.setdefault("t_count_exact_start_035d", t_start)
        out.setdefault("t_count_exact_done_035d", t_done)
        out.setdefault("t_count_fast_start_035d", t_start)
        out.setdefault("t_count_fast_done_035d", t_done)
        out.setdefault("count_source_107", "final_count_helper_154")
        out.setdefault("total_hits_source", source)
        if strategy:
            out.setdefault("total_hits_counting_strategy", strategy)
        out.setdefault("gui_count_wrapper_154", True)
        return out
    except Exception as _final_count_exc_154:
        # Conservative fallback: preserve the pre-154 exact-count helper behavior.
        try:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG gui.count_wrapper_154] status=fallback reason=%r token=%r",
                    repr(_final_count_exc_154),
                    search_token,
                    exc_info=True,
                )
        except Exception:
            pass

        from korpusuj.search.result_materialization import count_searchcursor_hits_036l4g48e

        return count_searchcursor_hits_036l4g48e(
            results,
            search_token=search_token,
            logger=logging,
            perf_counter=time.perf_counter,
        )

# END KORPUSUJ_MIGRATION_036L4G39F_SEARCHCURSOR_COUNT_HITS


# KORPUSUJ_MIGRATION_036L4G39G_SEARCHCURSOR_MATERIALIZE_CANCEL_CHECK

def _materialize_searchcursor_results_with_cancel_check(results, *, cancel_check=None, search_token=None):
    """Materialize SearchCursor rows while honoring cancellation and search-token checks."""
    from korpusuj.search.result_materialization import materialize_searchcursor_results_036l4g48e

    return materialize_searchcursor_results_036l4g48e(
        results,
        cancel_check=cancel_check,
        search_token=search_token,
        logger=logging,
        perf_counter=time.perf_counter,
    )

# END KORPUSUJ_MIGRATION_036L4G39G_SEARCHCURSOR_MATERIALIZE_CANCEL_CHECK

def _gui_headless_find_lemma_context_adapter_036l4g51f2(
    query,
    corpus_obj,
    corpus_name,
    left_context_size,
    right_context_size,
    warnings_list=None,
):
    # Adapter for the active GUI-headless boundary.
    # 036L4G51F removed the old shadow-only adapter name together with
    # the default-off 40D shadow execution scaffold. The active boundary
    # still needs a GUI-free adapter delegating to the existing backend.
    return find_lemma_context(
        query,
        corpus_obj,
        corpus_name,
        left_context_size,
        right_context_size,
        warnings_list=warnings_list,
    )

# END KORPUSUJ_MIGRATION_036L4G51F2_GUI_HEADLESS_ACTIVE_ADAPTER

def _postprocess_headless_shadow_results(
    *,
    shadow_results,
    sort_option=None,
    search_token=None,
    cancel_check=None,
):
    """Postprocess true headless shadow results before parity comparison.

    This helper intentionally works only on shadow results. It must not mutate
    normal GUI results, full_results_sorted, local_state, widgets, display_page,
    app.after scheduling, or statistics.
    """

    was_cursor = False
    count_payload = None
    materialize_payload = None
    cancelled = False

    try:
        was_cursor = _is_searchcursor_like(shadow_results)
    except Exception:
        was_cursor = False

    if was_cursor:
        try:
            count_payload = _count_searchcursor_hits_with_fast_estimate(
                shadow_results,
                search_token=search_token,
            )
        except Exception as exc:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG search.shadow_exec.fail] status=count_failed data=%r",
                    {"token": search_token, "error": repr(exc)},
                    exc_info=True,
                )
            count_payload = None

        materialize_payload = _materialize_searchcursor_results_with_cancel_check(
            shadow_results,
            cancel_check=cancel_check,
            search_token=search_token,
        )
        if materialize_payload.get("cancelled"):
            cancelled = True
            shadow_list = []
        else:
            shadow_list = list(materialize_payload.get("results") or [])
    else:
        try:
            shadow_list = list(shadow_results or [])
        except Exception:
            shadow_list = []

    if not cancelled:
        try:
            sort_search_results_in_place(shadow_list, sort_option)
        except Exception as exc:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG search.shadow_exec.fail] status=sort_failed data=%r",
                    {"token": search_token, "sort_option": sort_option, "error": repr(exc)},
                    exc_info=True,
                )

    return {
        "results": shadow_list,
        "cancelled": cancelled,
        "was_cursor": was_cursor,
        "count_payload": count_payload,
        "materialize_payload": materialize_payload,
        "shadow_materialized_len": len(shadow_list),
        "shadow_sample_order": "post_sort_shadow_order",
        "postprocess_stage": "postprocess_036L4G40E",
    }

# Headless shadow postprocess boundary ends here.



# END KORPUSUJ_MIGRATION_036L4G40D_TRUE_HEADLESS_SHADOW_EXECUTION


# KORPUSUJ_MIGRATION_036L4G41_GUI_RUN_SEARCH_HEADLESS_BOUNDARY

def _gui_run_search_headless_enabled_036l4g41():
    """Return True only when GUI should use run_search_headless as backend boundary."""
    try:
        cfg = globals().get("config", {}) or {}
        val = cfg.get("gui_search_boundary_enabled", True) if hasattr(cfg, "get") else True
    except Exception:
        val = True
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "tak", "yes", "on"}
    return bool(val)


def _gui_run_search_headless_bool_config_036l4g41(key, default=False):
    try:
        cfg = globals().get("config", {}) or {}
        val = cfg.get(key, default) if hasattr(cfg, "get") else default
    except Exception:
        val = default
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "tak", "yes", "on"}
    return bool(val)


def _gui_run_search_headless_int_config_036l4g41(key, default=0, *, minimum=0, maximum=1000000):
    try:
        cfg = globals().get("config", {}) or {}
        val = cfg.get(key, default) if hasattr(cfg, "get") else default
        val = int(val)
    except Exception:
        val = int(default)
    return max(int(minimum), min(int(val), int(maximum)))



# KORPUSUJ_MIGRATION_036L4G43_BOUNDARY_BROAD_REGEX_GUARD_METADATA_AWARE

def _gui_headless_strip_metadata_segments_036l4g43(query):
    try:
        q = str(query or "")
        out = []
        i = 0
        n = len(q)
        while i < n:
            if q[i] != "<":
                out.append(q[i]); i += 1; continue
            start = i; i += 1; quote = None; esc = False
            while i < n:
                c = q[i]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif quote:
                    if c == quote: quote = None
                elif c in ("'", '"'):
                    quote = c
                elif c == ">":
                    i += 1; break
                i += 1
            out.append(" " * max(1, i - start))
        return "".join(out)
    except Exception:
        return str(query or "")


def _gui_headless_extract_token_segments_036l4g43(query):
    try:
        q = str(query or "")
        segments = []
        i = 0
        n = len(q)
        while i < n:
            if q[i] != "[":
                i += 1; continue
            start = i; i += 1; quote = None; esc = False; depth = 1
            while i < n:
                c = q[i]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif quote:
                    if c == quote: quote = None
                elif c in ("'", '"'):
                    quote = c
                elif c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth <= 0:
                        i += 1; segments.append(q[start:i]); break
                i += 1
            else:
                segments.append(q[start:])
        return segments
    except Exception:
        return []


def _gui_headless_query_has_possible_broad_token_regex_036l4g43(query):
    try:
        masked = _gui_headless_strip_metadata_segments_036l4g43(query)
        return any(".*" in seg for seg in _gui_headless_extract_token_segments_036l4g43(masked))
    except Exception:
        try:
            return ".*" in str(query or "")
        except Exception:
            return False

# END KORPUSUJ_MIGRATION_036L4G43_BOUNDARY_BROAD_REGEX_GUARD_METADATA_AWARE


def _log_gui_headless_boundary_fallback(
    payload,
    *,
    query,
    selected_corpus,
    search_token=None,
    sort_option=None,
):
    # Log every fallback from the GUI headless boundary to the prepare/find path.
    try:
        data = dict(payload or {})
        reason = data.get("reason") or ("boundary_disabled" if payload is None else "unknown")
        status = data.get("status") or ("disabled" if payload is None else "unknown")
        try:
            max_raw_hits = _gui_run_search_headless_int_config_036l4g41(
                "gui_search_boundary_max_raw_hits", 0, minimum=0, maximum=1000000
            )
        except Exception:
            max_raw_hits = None
        try:
            allow_broad_regex = _gui_run_search_headless_bool_config_036l4g41(
                "gui_search_boundary_allow_broad_regex", True
            )
        except Exception:
            allow_broad_regex = None
        try:
            fallback_on_error = _gui_run_search_headless_bool_config_036l4g41(
                "gui_search_boundary_fallback_on_error", True
            )
        except Exception:
            fallback_on_error = None
        logging.warning(
            "[DIAG search.backend] fallback_to_prepare_path data=%r",
            {
                "status": status,
                "reason": reason,
                "token": search_token,
                "query": query,
                "corpus": selected_corpus,
                "sort_option": sort_option,
                "fallback_enabled": data.get("fallback_enabled", fallback_on_error),
                "boundary_enabled": _gui_run_search_headless_enabled_036l4g41(),
                "allow_broad_regex": allow_broad_regex,
                "max_raw_hits": max_raw_hits,
                "error": data.get("error"),
                "raw_total_hits": data.get("raw_total_hits"),
            },
        )
    except Exception:
        try:
            logging.exception("[DIAG search.backend] fallback_to_prepare_path logging failed")
        except Exception:
            pass

# KORPUSUJ_PATCH_173B5_GUI_COREF_NATIVE_SEARCHCURSOR_ROUTE
# Coreference queries use the existing native GUI SearchCursor continuation.
def _gui_query_uses_coref_173b5(query):
    try:
        text = str(query or "")
    except Exception:
        return False
    return re.search(
        r"(?i)\bcoref(?:\s*\(\s*[hpm]\s*\))?\s*(?:!=|=)",
        text,
    ) is not None


def _try_run_gui_search_via_headless_service(
    *,
    query,
    selected_corpus,
    left_context_size,
    right_context_size,
    sort_option=None,
    search_token=None,
    warnings_list=None,
):
    """Try using run_search_headless as the GUI backend boundary.

    Returns a payload with used=True when the boundary was used successfully.
    Returns None when disabled. Returns used=False for skip/failure so caller can
    fall back to the existing 39D/F/G path.

    This helper must not touch GUI widgets, app.after, display_page,
    full_results_sorted, statistics, or current_state aliases.
    """
    # Log a concise normal search request for the GUI headless boundary.
    try:
        logging.info(
            "Search request: request_id=%s query=%r corpus=%r sort=%r",
            search_token,
            query,
            selected_corpus,
            sort_option,
        )
    except Exception:
        pass

    if not _gui_run_search_headless_enabled_036l4g41():
        return None

    # KORPUSUJ_PATCH_173B5_GUI_COREF_NATIVE_SEARCHCURSOR_ROUTE
    # Skip only the generic materialized GUI service boundary. The caller then
    # continues through _prepare_and_find_search_backend_results, which still
    # prefers LazyCorpus/.search SQLite and returns the native SearchCursor.
    if _gui_query_uses_coref_173b5(query):
        return {
            "used": False,
            "status": "skipped",
            "reason": "coref_uses_native_gui_searchcursor_path_173b5",
            "fallback_enabled": True,
        }

    fallback_on_error = _gui_run_search_headless_bool_config_036l4g41(
        "gui_search_boundary_fallback_on_error", True
    )
    allow_broad_regex = _gui_run_search_headless_bool_config_036l4g41(
        "gui_search_boundary_allow_broad_regex", True
    )
    max_raw_hits = _gui_run_search_headless_int_config_036l4g41(
        "gui_search_boundary_max_raw_hits", 0, minimum=0, maximum=1000000
    )

    try:
        if (not allow_broad_regex) and _gui_headless_query_has_possible_broad_token_regex_036l4g43(query):
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG search.backend] status=skipped data=%r",
                    {
                        "reason": "possible_broad_regex",
                        "guard_scope": "token_segments_036L4G43",
                        "token": search_token,
                        "query": query,
                        "corpus": selected_corpus,
                        "fallback_enabled": fallback_on_error,
                    },
                )
            return {"used": False, "status": "skipped", "reason": "possible_broad_regex", "fallback_enabled": fallback_on_error}
    except Exception:
        pass

    try:
        from korpusuj.search.search_service import SearchBackendContext, SearchRequest, run_search_service
    except Exception as exc:
        if korpusuj_diagnostics_enabled_145c1():
            logging.info(
                "[DIAG search.backend] status=failed data=%r",
                {"reason": "import_failed", "error": repr(exc), "token": search_token, "query": query, "corpus": selected_corpus},
                exc_info=True,
            )
        return {"used": False, "status": "failed", "reason": "import_failed", "fallback_enabled": fallback_on_error}

    try:
        validate_query_for_ui(query)
        selected_corpus = str(selected_corpus or "").strip()
        if not selected_corpus or selected_corpus not in dataframes:
            return {
                "used": False,
                "status": "skipped",
                "reason": "missing_or_unknown_corpus",
                "fallback_enabled": fallback_on_error,
            }
        df = dataframes[selected_corpus]
        search_df = _make_lazy_corpus_for_search(selected_corpus, df)

        req = SearchRequest(
            query=str(query or ""),
            corpus_name=str(selected_corpus or ""),
            left_context=int(left_context_size),
            right_context=int(right_context_size),
            sort_option=sort_option,
            limit=None,
            offset=0,
            options={"source": "engine.gui_run_search_headless_boundary_036l4g41"},
        )
        ctx = SearchBackendContext(
            dataframes={selected_corpus: search_df},
            corpora={selected_corpus: search_df},
            corpus_name=selected_corpus,
            df_type=type(df).__name__ if df is not None else None,
            search_df_type=type(search_df).__name__ if search_df is not None else None,
            find_lemma_context_adapter=_gui_headless_find_lemma_context_adapter_036l4g51f2,
        )

        bundle = run_search_service(req, ctx)
        raw_total = getattr(bundle, "total_hits", None)
        if raw_total is not None and max_raw_hits > 0:
            try:
                if int(raw_total) > int(max_raw_hits):
                    if korpusuj_diagnostics_enabled_145c1():
                        logging.info(
                            "[DIAG search.backend] status=skipped data=%r",
                            {
                                "reason": "raw_total_hits_exceeds_guard",
                                "token": search_token,
                                "query": query,
                                "corpus": selected_corpus,
                                "raw_total_hits": raw_total,
                                "max_raw_hits": max_raw_hits,
                                "fallback_enabled": fallback_on_error,
                            },
                        )
                    return {
                        "used": False,
                        "status": "skipped",
                        "reason": "raw_total_hits_exceeds_guard",
                        "fallback_enabled": fallback_on_error,
                    }
            except Exception:
                pass

        post = _postprocess_headless_shadow_results(
            shadow_results=bundle.results,
            sort_option=sort_option,
            search_token=search_token,
            cancel_check=None,
        )
        if post.get("cancelled"):
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG search.backend] status=skipped data=%r",
                    {"reason": "postprocess_cancelled", "token": search_token, "query": query, "corpus": selected_corpus, "fallback_enabled": fallback_on_error},
                )
            return {"used": False, "status": "skipped", "reason": "postprocess_cancelled", "fallback_enabled": fallback_on_error}

        results = list(post.get("results") or [])
        if korpusuj_diagnostics_enabled_145c1():
            logging.info(
                "[DIAG search.backend] status=used data=%r",
                {
                    "token": search_token,
                    "query": query,
                    "corpus": selected_corpus,
                    "sort_option": sort_option,
                    "results_len": len(results),
                    "raw_total_hits": raw_total,
                    "shadow_was_cursor": post.get("was_cursor"),
                    "postprocess_stage": post.get("postprocess_stage"),
                    "sample_order": post.get("shadow_sample_order"),
                    "fallback_enabled": fallback_on_error,
                },
            )
        return {
            "used": True,
            "status": "used",
            "results": results,
            "df": df,
            "search_df": search_df,
            "warnings_list": warnings_list if warnings_list is not None else [],
            "bundle": bundle,
            "postprocess": post,
        }
    except Exception as exc:
        if korpusuj_diagnostics_enabled_145c1():
            logging.info(
                "[DIAG search.backend] status=failed data=%r",
                {"reason": "exception", "error": repr(exc), "token": search_token, "query": query, "corpus": selected_corpus, "fallback_enabled": fallback_on_error},
                exc_info=True,
            )
        return {"used": False, "status": "failed", "reason": "exception", "fallback_enabled": fallback_on_error}

# END KORPUSUJ_MIGRATION_036L4G41_GUI_RUN_SEARCH_HEADLESS_BOUNDARY


# KORPUSUJ_MIGRATION_036L4G51N_RESULT_ALIAS_PUBLICATION_HELPERS

def _publish_current_search_state_aliases(
    local_state,
    *,
    include_identity=True,
    include_results=False,
    results_override=None,
    reset_page=False,
    set_status=True,
):
    # Publish legacy GUI/global aliases for the current search state.
    # This stays in engine.py for now because it writes GUI/global compatibility state.
    if include_identity:
        globals()['current_state'] = local_state
        globals()['global_query'] = local_state.query
        globals()['global_selected_corpus'] = local_state.corpus
    if include_results:
        results_value = results_override if results_override is not None else list(local_state.results)
        globals()['full_results_sorted'] = results_value
    if reset_page:
        globals()['current_page'] = 0
    if set_status:
        globals()['search_status'] = 0


def _reset_empty_search_results_aliases_036l4g51n():
    # Reset legacy aliases for an empty-result search path.
    globals()['full_results_sorted'] = []
    globals()['search_status'] = 0
    globals()['current_page'] = 0

# END KORPUSUJ_MIGRATION_036L4G51N_RESULT_ALIAS_PUBLICATION_HELPERS

# KORPUSUJ_MIGRATION_036L4G52C_STATISTICS_ALIAS_PUBLICATION_HELPER

def _publish_statistics_state_aliases_036l4g52c(local_state):
    # Publish final statistics aliases from local_state to legacy globals.
    # Kept in engine.py for now because this writes GUI/global compatibility state.
    globals()['monthly_lemma_freq'] = dict(local_state.monthly_lemma_freq)
    globals()['monthly_freq_for_use'] = dict(local_state.monthly_freq_for_use)
    globals()['monthly_tfidf_for_use'] = dict(local_state.monthly_tfidf_for_use)
    globals()['monthly_zscore_for_use'] = dict(local_state.monthly_zscore_for_use)

# END KORPUSUJ_MIGRATION_036L4G52C_STATISTICS_ALIAS_PUBLICATION_HELPER

# KORPUSUJ_MIGRATION_036L4G52F_LOCAL_STATE_STATISTICS_STAGING_HELPER

def _stage_statistics_payload_on_local_state_036l4g52f(local_state, statistics_payload_context):
    # Stage computed statistics payload values onto local_state.
    # Kept in engine.py for now because this is GUI/runtime orchestration state.
    statistics_payload = statistics_payload_context['statistics']
    local_state.statistics = statistics_payload
    local_state.monthly_freq_for_use = statistics_payload_context['monthly_freq_for_use']
    local_state.monthly_tfidf_for_use = statistics_payload_context['monthly_tfidf_for_use']
    local_state.monthly_zscore_for_use = statistics_payload_context['monthly_zscore_for_use']
    local_state.has_dates = statistics_payload.has_dates
    local_state.fq_data = statistics_payload_context['fq_data']
    local_state.fq_data_token = statistics_payload_context['fq_data_token']
    local_state.fq_data_month = statistics_payload_context['fq_data_month']
    local_state.s_lemma_total_freq = statistics_payload.s_lemma_total_freq
    local_state.s_lemma_global_pmw = statistics_payload.s_lemma_global_pmw
    local_state.s_lemma_global_tfidf = statistics_payload.s_lemma_global_tfidf
    local_state.unique_lemmas = statistics_payload_context['unique_lemmas']
    local_state.true_monthly_totals = statistics_payload_context['true_monthly_totals']
    local_state.lemma_df_cache = statistics_payload_context['lemma_df_cache']

# END KORPUSUJ_MIGRATION_036L4G52F_LOCAL_STATE_STATISTICS_STAGING_HELPER

# KORPUSUJ_MIGRATION_036L4G52J_FREQUENCY_TABLE_OUTPUT_PUBLICATION_HELPER

def _publish_frequency_table_outputs_036l4g52j(global_frequency_tables, monthly_frequency_tables):
    # Publish complete frequency-table outputs to legacy/global GUI compatibility variables.
    # Kept in engine.py for now because this writes runtime GUI/global state.
    global fq_data_token, fq_data, s_lemma_total_freq, s_lemma_global_pmw, s_lemma_global_tfidf
    global monthly_freq_for_use, monthly_tfidf_for_use, monthly_zscore_for_use, fq_data_month
    fq_data_token = global_frequency_tables.fq_data_token
    fq_data = global_frequency_tables.fq_data
    s_lemma_total_freq = global_frequency_tables.s_lemma_total_freq
    s_lemma_global_pmw = global_frequency_tables.s_lemma_global_pmw
    s_lemma_global_tfidf = global_frequency_tables.s_lemma_global_tfidf
    monthly_freq_for_use = monthly_frequency_tables.monthly_freq_for_use
    monthly_tfidf_for_use = monthly_frequency_tables.monthly_tfidf_for_use
    monthly_zscore_for_use = monthly_frequency_tables.monthly_zscore_for_use
    fq_data_month = monthly_frequency_tables.fq_data_month
    return (
        fq_data_token,
        fq_data,
        s_lemma_total_freq,
        s_lemma_global_pmw,
        s_lemma_global_tfidf,
        monthly_freq_for_use,
        monthly_tfidf_for_use,
        monthly_zscore_for_use,
        fq_data_month,
    )

# END KORPUSUJ_MIGRATION_036L4G52J_FREQUENCY_TABLE_OUTPUT_PUBLICATION_HELPER

# KORPUSUJ_MIGRATION_036L4G52N_STATISTICS_FREQUENCY_INPUT_CONTEXT_HELPER

def _prepare_statistics_frequency_input_context(
    statistics_worker_contract,
    inverted_indexes,
    df,
    monthly_lemma_freq,
    true_monthly_totals,
):
    # Prepare frequency/statistics inputs while preserving mutable mapping identity.
    # This helper intentionally returns a context dict; search_thread must rebind
    # downstream local/closure names from that context immediately after the call.
    frequency_inputs = collect_search_frequency_inputs(statistics_worker_contract["results"])
    unique_matched_tokens = frequency_inputs.unique_matched_tokens
    unique_lemmas = frequency_inputs.unique_lemmas
    monthly_lemma_freq.clear()
    monthly_lemma_freq.update(frequency_inputs.monthly_lemma_freq)
    exact_orth_df = frequency_inputs.exact_orth_df
    exact_lemma_df = frequency_inputs.exact_lemma_df
    raw_monthly_counts = inverted_indexes[statistics_worker_contract["corpus"]].get("monthly_token_counts", {}) or {}
    monthly_counts_flattened, true_monthly_totals_source = normalize_monthly_token_counts_for_search(raw_monthly_counts)
    true_monthly_totals.clear()
    true_monthly_totals.update(true_monthly_totals_source)
    total_token_count = sum(true_monthly_totals.values())
    total_docs = _safe_len_or_zero(df)
    return {
        "frequency_inputs": frequency_inputs,
        "unique_matched_tokens": unique_matched_tokens,
        "unique_lemmas": unique_lemmas,
        "exact_orth_df": exact_orth_df,
        "exact_lemma_df": exact_lemma_df,
        "raw_monthly_counts": raw_monthly_counts,
        "monthly_counts_flattened": monthly_counts_flattened,
        "true_monthly_totals_source": true_monthly_totals_source,
        "total_token_count": total_token_count,
        "total_docs": total_docs,
    }

# END KORPUSUJ_MIGRATION_036L4G52N_STATISTICS_FREQUENCY_INPUT_CONTEXT_HELPER

# KORPUSUJ_MIGRATION_036L4G52R_STATISTICS_PAYLOAD_BUILDERS

def _build_search_statistics_payload(
    true_monthly_totals,
    monthly_freq_for_use,
    monthly_tfidf_for_use,
    monthly_zscore_for_use,
    fq_data,
    fq_data_token,
    fq_data_month,
    s_lemma_total_freq,
    s_lemma_global_pmw,
    s_lemma_global_tfidf,
):
    # Build the final SearchStatistics payload from explicit inputs.
    # Keep this helper GUI-free and scheduling-free.
    return SearchStatistics(
        true_monthly_totals=dict(true_monthly_totals),
        monthly_freq_for_use=dict(monthly_freq_for_use),
        monthly_tfidf_for_use=dict(monthly_tfidf_for_use),
        monthly_zscore_for_use=dict(monthly_zscore_for_use),
        fq_data=list(fq_data),
        fq_data_token=list(fq_data_token),
        fq_data_month=list(fq_data_month),
        s_lemma_total_freq=list(s_lemma_total_freq),
        s_lemma_global_pmw=list(s_lemma_global_pmw),
        s_lemma_global_tfidf=list(s_lemma_global_tfidf),
        has_dates=True,
    )


def _build_statistics_payload_context(
    statistics,
    unique_lemmas,
    lemma_df_cache,
    monthly_lemma_freq,
):
    # Bundle statistics and non-GUI auxiliary data for later worker handoff.
    # Keep this helper GUI-free and scheduling-free.
    return {
        "statistics": statistics,
        "unique_lemmas": unique_lemmas,
        "lemma_df_cache": lemma_df_cache,
        "monthly_lemma_freq": dict(monthly_lemma_freq),
        "true_monthly_totals": dict(statistics.true_monthly_totals),
        "monthly_freq_for_use": dict(statistics.monthly_freq_for_use),
        "monthly_tfidf_for_use": dict(statistics.monthly_tfidf_for_use),
        "monthly_zscore_for_use": dict(statistics.monthly_zscore_for_use),
        "fq_data": list(statistics.fq_data),
        "fq_data_token": list(statistics.fq_data_token),
        "fq_data_month": list(statistics.fq_data_month),
    }

# END KORPUSUJ_MIGRATION_036L4G52R_STATISTICS_PAYLOAD_BUILDERS

# KORPUSUJ_MIGRATION_036L4G52U_STATISTICS_PLOT_GENERATION_HELPER

def _generate_statistics_plot_image(
    monthly_freq_for_use,
    plot_style,
    output_path=str(writable_temp_root() / "temp_plot.png"),
    get_plot_stack_func=None,
    np_module=None,
    os_module=None,
):
    # Generate the statistics plot image file without touching GUI widgets.
    # update_plot_images() remains the GUI consumer of output_path.
    if get_plot_stack_func is None:
        get_plot_stack_func = get_plot_stack
    if np_module is None:
        np_module = np
    if os_module is None:
        os_module = os

    plot_stack = get_plot_stack_func()
    Figure = plot_stack["Figure"]
    FigureCanvasAgg = plot_stack["FigureCanvasAgg"]
    yearly_grouped = {}
    for key, data_ in monthly_freq_for_use.items():
        year, month = key.split('-')
        if year == '0000' or month == '0':
            continue
        yearly_grouped.setdefault(year, {})
        for lemma, val in data_.items():
            yearly_grouped[year][lemma] = yearly_grouped[year].get(lemma, 0) + val
    keys = sorted(yearly_grouped.keys(), key=int)
    x_labels = keys
    x = np_module.arange(len(keys))

    fig = Figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot(111)

    if plot_style == "ciemny":
        fig.patch.set_facecolor('#2C2F33')
        ax.set_facecolor('#2C2F33')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.tick_params(colors='black')

    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.2)
    ax.set_xlabel('Rok')
    ax.set_ylabel('Frekwencja')

    max_labels = 24
    n_labels = len(x_labels)
    step = int(np_module.ceil(n_labels / max_labels)) if n_labels > max_labels else 1
    labeled_idx = set([0, n_labels - 1] + list(range(0, n_labels, step)))
    labels = [lbl if i in labeled_idx else "" for i, lbl in enumerate(x_labels)]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='right')
    if len(x) > 0:
        ax.set_xlim(x[0] - 1, x[-1] + 1)
    for tick, label in zip(ax.xaxis.get_major_ticks(), labels):
        size = 3 if label == "" else 7
        tick.tick1line.set_markersize(size)
        tick.tick2line.set_markersize(size)

    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    fig.savefig(str(output_path), bbox_inches='tight')
    return {
        "generated": True,
        "output_path": output_path,
        "years_count": len(keys),
        "series_count": len({lemma for year_data in yearly_grouped.values() for lemma in year_data}),
    }

# END KORPUSUJ_MIGRATION_036L4G52U_STATISTICS_PLOT_GENERATION_HELPER


def _select_statistics_listbox_data(
    mode,
    s_lemma_global_tfidf,
    monthly_zscore_for_use,
    s_lemma_global_pmw,
    s_lemma_total_freq,
    unique_lemmas,
):
    """Select the statistics dataset associated with the requested list-box mode."""
    def _get_max_scores_036l4g57c(freq_dict):
        scores = {}
        for lemma in unique_lemmas:
            max_val = max((freq_dict[m].get(lemma, 0) for m in freq_dict), default=0)
            scores[lemma] = max_val
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if mode == 'TF-IDF':
        return s_lemma_global_tfidf
    if mode == 'Z-score':
        return _get_max_scores_036l4g57c(monthly_zscore_for_use)
    if mode == 'Częstość względna':
        return s_lemma_global_pmw
    return s_lemma_total_freq


class _StatisticsListboxPanel:
    """Manage the statistics list-box widget and its current dataset."""

    def __init__(
        self,
        *,
        ctk_module,
        math_module,
        parent_frame,
        sorted_lemma_freq,
        vars_dict,
        merge_dict,
        update_plot_callback,
        theme,
        items_per_page=100,
    ):
        self.ctk = ctk_module
        self.math = math_module
        self.parent_frame = parent_frame
        self.vars_dict = vars_dict
        self.merge_dict = merge_dict
        self.update_plot_callback = update_plot_callback
        self.theme = theme
        self.items_per_page = items_per_page
        self.update_job = {'after_id': None}
        self.current_page_idx = {'idx': 0}
        self.state = {'data': sorted_lemma_freq}
        self.container = None
        self.listbox_frame = None
        self.rename_entry = None
        self._build(sorted_lemma_freq)

    def _build(self, sorted_lemma_freq):
        self.vars_dict.clear()
        self.merge_dict.clear()

        for lemma, _ in sorted_lemma_freq:
            self.vars_dict[lemma] = self.ctk.BooleanVar(value=False)
            self.merge_dict[lemma] = self.ctk.StringVar(value=lemma)

        theme = self.theme
        ctk = self.ctk

        self.container = ctk.CTkFrame(self.parent_frame, fg_color=theme['frame_fg'], corner_radius=15)
        self.rename_entry = ctk.CTkEntry(
            self.container,
            placeholder_text='Nowa nazwa dla zaznaczonych',
            font=('JetBrains Mono', 12),
            fg_color=theme['subframe_fg'],
            corner_radius=8,
            height=35,
        )
        self.rename_entry.pack(fill='x', padx=10, pady=(5, 5))
        rename_btn = ctk.CTkButton(
            self.container,
            text='Grupuj/Zmień nazwę',
            font=('Verdana', 12, 'bold'),
            fg_color=theme['button_fg'],
            hover_color=theme['button_hover'],
            text_color=theme['button_text'],
            corner_radius=8,
            height=35,
        )
        rename_btn.pack(fill='x', padx=10, pady=(5, 10))

        nav_frame = ctk.CTkFrame(self.container, fg_color=theme['subframe_fg'], corner_radius=12)
        nav_frame.pack(fill='x', padx=10, pady=(0, 10))
        nav_frame.grid_columnconfigure(0, weight=0)
        nav_frame.grid_columnconfigure(1, weight=1)
        nav_frame.grid_columnconfigure(2, weight=0)

        prev_btn = ctk.CTkButton(
            nav_frame, text='<', width=40, height=35,
            fg_color=theme['button_fg'], hover_color=theme['button_hover'],
            text_color=theme['button_text'], corner_radius=8,
        )
        prev_btn.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.lbl_page = ctk.CTkLabel(
            nav_frame, text='1 / 1', font=('Verdana', 12, 'bold'),
            text_color=theme['label_text'],
        )
        self.lbl_page.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        next_btn = ctk.CTkButton(
            nav_frame, text='>', width=40, height=35,
            fg_color=theme['button_fg'], hover_color=theme['button_hover'],
            text_color=theme['button_text'], corner_radius=8,
        )
        next_btn.grid(row=0, column=2, sticky='e', padx=5, pady=5)

        self.listbox_frame = ctk.CTkScrollableFrame(
            self.container, fg_color=theme['subframe_fg'], corner_radius=8, height=300,
        )
        self.listbox_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        rename_btn.configure(command=self.rename_selected)
        prev_btn.configure(command=lambda: self.show_page(self.current_page_idx['idx'] - 1) if self.current_page_idx['idx'] > 0 else None)
        next_btn.configure(command=lambda: self.show_page(self.current_page_idx['idx'] + 1) if self.current_page_idx['idx'] < max(1, self.math.ceil(len(self.state['data']) / self.items_per_page)) - 1 else None)

        self.show_page(0)

    def delayed_update(self):
        if self.update_job['after_id']:
            self.container.after_cancel(self.update_job['after_id'])
        self.update_job['after_id'] = self.container.after(300, self.update_plot_callback)

    def show_page(self, page_idx):
        try:
            if not self.listbox_frame.winfo_exists():
                return
        except Exception:
            return
        for widget in self.listbox_frame.winfo_children():
            widget.destroy()

        current_data = self.state['data']
        total_pages = max(1, self.math.ceil(len(current_data) / self.items_per_page))
        if page_idx >= total_pages:
            page_idx = max(0, total_pages - 1)
        if page_idx < 0:
            page_idx = 0

        start = page_idx * self.items_per_page
        end = min(start + self.items_per_page, len(current_data))
        theme = self.theme
        for lemma, score in current_data[start:end]:
            display_name = self.merge_dict[lemma].get()
            score_str = f'{score:.2f}' if isinstance(score, float) else str(score)
            cb = self.ctk.CTkCheckBox(
                self.listbox_frame,
                text=f'{display_name} ({score_str})',
                variable=self.vars_dict[lemma],
                command=self.delayed_update,
                font=('Verdana', 12),
                fg_color=theme['button_fg'],
                hover_color=theme['button_hover'],
                text_color=theme['label_text'],
            )
            cb.pack(anchor='w', pady=4, padx=5)

        self.lbl_page.configure(text=f'{page_idx + 1} / {total_pages}')
        self.current_page_idx['idx'] = page_idx

    def rename_selected(self):
        new_text = self.rename_entry.get().strip()
        if not new_text:
            return
        renamed_any = False
        for lemma, _ in self.state['data']:
            if self.vars_dict[lemma].get():
                self.merge_dict[lemma].set(new_text)
                renamed_any = True
        if renamed_any:
            self.show_page(self.current_page_idx['idx'])
            self.delayed_update()

    def set_data(self, new_sorted_data):
        self.state['data'] = new_sorted_data
        self.show_page(0)
def _schedule_no_results_branch_completion_036l4g56b(
    app_obj,
    last_search_warnings,
    show_search_warnings_func,
    update_no_results_func,
):
    """Schedule completion of the GUI branch that reports an empty search result."""
    app_obj.after(0, lambda: show_search_warnings_func(last_search_warnings))
    app_obj.after(0, update_no_results_func)
    return {
        'scheduled': True,
        'callbacks': ['show_search_warnings', 'update_no_results'],
    }
def _schedule_search_thread_error_display_036l4g55d(
    app_obj,
    label_results_count_widget,
    error_msg,
    show_search_error_func,
):
    """Schedule display of an error raised by the background search worker."""
    label_results_count_widget.after(0, lambda: label_results_count_widget.configure(text=''))
    app_obj.after(0, lambda msg=error_msg: show_search_error_func(msg))
    return {
        'scheduled': True,
        'callbacks': ['clear_label_results_count', 'show_search_error'],
    }
def _schedule_statistics_worker_completion(
    app_obj,
    local_state,
    last_search_warnings,
    show_search_warnings_func,
    add_to_history_func,
    push_nav_state_func,
    complete_statistics_gui_update_func,
):
    """Schedule publication of statistics state after the worker finishes."""
    app_obj.after(0, lambda: show_search_warnings_func(last_search_warnings))
    app_obj.after(0, lambda: add_to_history_func(local_state))
    app_obj.after(100, push_nav_state_func)
    app_obj.after(0, complete_statistics_gui_update_func)
    return {
        'scheduled': True,
        'callbacks': [
            'show_search_warnings',
            'add_to_history',
            'push_nav_state',
            'complete_statistics_gui_update',
        ],
    }


def _schedule_statistics_worker_error(
    app_obj,
    error_msg,
    show_search_error_func,
):
    """Schedule statistics worker error display on the GUI thread."""
    app_obj.after(0, lambda msg=error_msg: show_search_error_func(msg))
    return {
        'scheduled': True,
        'callbacks': ['show_search_error'],
    }
def _publish_statistics_state_aliases_with_current_state_036l4g53b(
    local_state,
    state_lock_obj=None,
    globals_dict=None,
    publish_statistics_aliases_func=None,
):
    """Publish the current statistics state through the GUI compatibility aliases."""
    if globals_dict is None:
        globals_dict = globals()
    if publish_statistics_aliases_func is None:
        publish_statistics_aliases_func = _publish_statistics_state_aliases_036l4g52c

    def _publish():
        globals_dict['current_state'] = local_state
        publish_statistics_aliases_func(local_state)

    if state_lock_obj is None:
        _publish()
    else:
        with state_lock_obj:
            _publish()

    return {
        'published': True,
        'current_state_set': True,
        'statistics_aliases_published': True,
    }
def _update_statistics_panels(
    df,
    ui_state,
    fq_data_token,
    fq_data,
    fq_data_month,
    paginator_token,
    paginator_fq,
    paginator_month,
    frekw_dane_tabela_orth,
    frekw_dane_tabela,
    frekw_dane_tabela_month,
    update_table_func=None,
    update_plot_images_func=None,
):
    # Update statistics/frequency GUI panels after statistics are ready.
    # This helper is GUI-adjacent by design and must not move to korpusuj.search.*.
    # It intentionally does not touch concordance, listboxes, state aliases, or scheduling.
    if update_table_func is None:
        update_table_func = update_table
    if update_plot_images_func is None:
        update_plot_images_func = update_plot_images

    if "Data publikacji" not in df.columns:
        return {
            "updated": False,
            "plotted": False,
            "fq_data_token_count": 0,
            "fq_data_count": 0,
            "fq_data_month_count": 0,
        }

    paginator_token["data"] = fq_data_token
    update_table_func(paginator_token)
    frekw_dane_tabela_orth.set_data(fq_data_token[:15])

    paginator_fq["data"] = fq_data
    update_table_func(paginator_fq)
    frekw_dane_tabela.set_data(fq_data[:15])

    paginator_month["data"] = fq_data_month
    update_table_func(paginator_month)
    frekw_dane_tabela_month.set_data(fq_data_month[:15])

    plotted = False
    if ui_state["is_plotting"] == 'Tak':
        update_plot_images_func()
        plotted = True

    return {
        "updated": True,
        "plotted": plotted,
        "fq_data_token_count": len(fq_data_token),
        "fq_data_count": len(fq_data),
        "fq_data_month_count": len(fq_data_month),
    }

# END KORPUSUJ_MIGRATION_036L4G52Y_UPDATE_STATISTICS_PANELS_HELPER

def search():
    theme = THEMES[motyw.get()]
    global search_status, precalculated_bins
    global search_in_progress, active_search_token, last_search_error, last_search_warnings

    selected_corpus = str(corpus_var.get() or "").strip()
    if not selected_corpus or selected_corpus not in dataframes:
        messagebox.showwarning(
            "Brak aktywnego korpusu",
            "Najpierw wczytaj i wybierz korpus, a następnie uruchom wyszukiwanie.",
        )
        return

    with search_guard:
        if search_in_progress:
            active_search_token += 1
            search_in_progress = False
            try:
                globals()['search_status'] = 0
                label_results_count.configure(text="Przerwano wyszukiwanie / liczenie statystyk.")
                button_search.configure(text="" if globals().get("s_img") else "Szukaj", image=globals().get("s_img"), state="normal")
            except Exception:
                pass
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
        "selected_corpus": selected_corpus,
        "left_context": left_ctx_val,
        "right_context": right_ctx_val,
        "sort_option": sort_option_var.get(),
        "is_plotting": plotting.get(),
        "plot_style": styl_wykresow.get(),
        "theme_name": motyw.get(),
    }

    # Clear the checkboxes and result text widget.
    for frame in [checkboxes_frame]:
        for child in frame.winfo_children():
            child.destroy()
    checkboxes_frame.update_idletasks()

    try:
        button_search.configure(text="■", image=None, state="normal", text_color="white")
    except Exception:
        button_search.configure(state="normal")

    if 'paginator_colloc' in globals():
        paginator_colloc["data"] = []
        paginator_colloc["current_page"][0] = 0
        update_table(paginator_colloc)

    if 'paginator_profile' in globals():
        paginator_profile["data"] = []
        paginator_profile["current_page"][0] = 0
        update_table(paginator_profile)
        # Zmiana: podmieniamy na nowy przycisk i blokujemy go
        profile_rel_menu_btn.configure(state="disabled")
        profile_rel_var.set("Brak danych")

        profile_node_menu.configure(values=["Token 1"])
        profile_node_var.set("Token 1")
        current_profile_dict.clear()
        global current_profile_target_lemma
        current_profile_target_lemma = ""

    def begin_search_ui():
        """Natychmiast czyści stare wyniki i pokazuje komunikat startu wyszukiwania."""
        try:
            globals()['full_results_sorted'] = []
            globals()['current_page'] = 0
            globals()['global_query'] = gui_state.get("query", "")
            globals()['global_selected_corpus'] = gui_state.get("selected_corpus", "")
            globals()['search_status'] = 1
            label_results_count.configure(text=f"Rozpoczynam wyszukiwanie w korpusie: {globals()['global_selected_corpus']}...")
            display_page(globals()['global_query'], globals()['global_selected_corpus'])
            try:
                app.update_idletasks()
            except Exception:
                pass
        except Exception as e:
            logging.warning("[APP search.ui.fail] reason=%r", e, exc_info=True)

    begin_search_ui()

    # Wstrzykujemy stan GUI jako drugi argument do funkcji
    def search_thread(search_token, ui_state):

        try:
            # The old pre-boundary SearchRequest log was removed because it emitted incomplete values.
            # Boundary logging now happens after request capture and headless-service payload resolution.
            t_start = time.perf_counter()
            # Wyciągamy wartości BEZPIECZNIE ze słownika, zamiast z GUI!
            query = ui_state["query"]
            theme = THEMES.get(ui_state.get("theme_name", "jasny"), THEMES["jasny"])
            # Capture a SearchRequest snapshot for the search boundary.
            _capture_search_request_at_boundary(locals())
            # Resolve core search fields from SearchRequest or fall back to UI/local values.
            query, selected_corpus, left_context_size, right_context_size, sort_option = _resolve_search_core_fields_from_request_or_fallback(locals(), ui_state)
            # Legacy backend path follows if the service-backed headless path is not used.
            # Try the service-backed headless path before falling back to the legacy backend path.
            headless_payload = _try_run_gui_search_via_headless_service(
                query=query,
                selected_corpus=selected_corpus,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                sort_option=sort_option,
                search_token=search_token,
                warnings_list=locals().get("warnings_list", []),
            )
            # Initialize timing locals for the headless-service path; backend path overwrites them below.
            t_parsed = t_start
            t_find_start_035d = t_start
            t_find_done_035d = time.perf_counter()
            if headless_payload and headless_payload.get("used"):
                results = headless_payload["results"]
                df = headless_payload["df"]
                search_df = headless_payload["search_df"]
                warnings_list = headless_payload.get("warnings_list", locals().get("warnings_list", []))
                # Prepare local state expected by the downstream GUI result pipeline.
                # The legacy backend path normally prepares local_state.
                # When the headless service path is used, the legacy backend block is skipped,
                # so the downstream GUI result pipeline still needs local_state.
                try:
                    local_state = ui_state
                    local_state.query = query
                    local_state.corpus = selected_corpus
                    local_state.left_context_size = left_context_size
                    local_state.right_context_size = right_context_size
                    local_state.sort_option = sort_option
                except Exception:
                    from types import SimpleNamespace
                    try:
                        state_dict = dict(getattr(ui_state, "__dict__", {}) or {})
                    except Exception:
                        state_dict = {}
                    state_dict.update({
                        "query": query,
                        "corpus": selected_corpus,
                        "left_context_size": left_context_size,
                        "right_context_size": right_context_size,
                        "sort_option": sort_option,
                    })
                    local_state = SimpleNamespace(**state_dict)
            else:
                _log_gui_headless_boundary_fallback(
                    headless_payload,
                    query=query,
                    selected_corpus=selected_corpus,
                    search_token=search_token,
                    sort_option=sort_option,
                )
                backend_payload = _prepare_and_find_search_backend_results(
                    query=query,
                    selected_corpus=selected_corpus,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    sort_option=sort_option,
                    ui_state=ui_state,
                    search_token=search_token,
                    t_start=t_start,
                )
                query = backend_payload["query"]
                selected_corpus = backend_payload["selected_corpus"]
                left_context_size = backend_payload["left_context_size"]
                right_context_size = backend_payload["right_context_size"]
                sort_option = backend_payload["sort_option"]
                df = backend_payload["df"]
                search_df = backend_payload["search_df"]
                warnings_list = backend_payload["warnings_list"]
                local_state = backend_payload["local_state"]
                results = backend_payload["results"]
                t_parsed = backend_payload["t_parsed"]
                t_find_start_035d = backend_payload["t_find_start_035d"]
                t_find_done_035d = backend_payload["t_find_done_035d"]

            if _is_searchcursor_like(results):
                # Count SearchCursor hits before materialization, using fast exact estimates when available.
                count_payload = _count_searchcursor_hits_with_fast_estimate(
                    results,
                    search_token=search_token,
                )
                exact_hits_035d = count_payload["exact_hits"]
                total_hits = count_payload["total_hits"]
                t_count_exact_start_035d = count_payload["t_count_exact_start_035d"]
                t_count_exact_done_035d = count_payload["t_count_exact_done_035d"]
                t_count_fast_start_035d = count_payload["t_count_fast_start_035d"]
                t_count_fast_done_035d = count_payload["t_count_fast_done_035d"]

                local_state.results = results
                # Try to prepare a metadata-sorted first-page preview without full materialization.
                metadata_sorted_first_page_ready = _try_prepare_metadata_sorted_searchcursor_preview(
                    results, sort_option
                )
                # If metadata preview is unavailable, try an alphabetical first-page preview for simple base/orth queries.
                alpha_sorted_first_page_ready = False
                if not metadata_sorted_first_page_ready:
                    alpha_sorted_first_page_ready = _try_prepare_alpha_sorted_searchcursor_preview(
                        results, sort_option, query
                    )
                # Publish only identity/status first; do not expose SearchCursor as the stable final list here.
                # SearchCursor can provide a fast first-page preview, but its temporary order may differ
                # from the final materialized + sort_search_results_in_place(...) order.
                # Therefore this first publish uses include_results=False. If the cursor was prepared
                # for a metadata/alpha first-page preview below, that preview is published explicitly
                # via results_override=results and then replaced by the final materialized result list later.
                with state_lock:
                    _publish_current_search_state_aliases(local_state, include_identity=True, include_results=False, reset_page=False, set_status=True)

                def show_cursor_materialization_status():
                    if search_token != active_search_token:
                        return
                    label_results_count.configure(
                        text=f"Znaleziono trafień: {total_hits:,} (przygotowuję listę wyników...)".replace(',', ' ')
                    )

                def show_searchcursor_warnings_if_current():
                    if search_token != active_search_token:
                        return
                    show_search_warnings(warnings_list)

                app.after(0, show_searchcursor_warnings_if_current)
                app.after(0, show_cursor_materialization_status)

                # Publish the prepared SearchCursor as a temporary first-page preview, not as the final list.
                if metadata_sorted_first_page_ready or alpha_sorted_first_page_ready:
                    with state_lock:
                        _publish_current_search_state_aliases(local_state, include_identity=False, include_results=True, results_override=results, reset_page=True, set_status=True)

                    def show_metadata_sorted_first_page_035e():
                        if search_token != active_search_token:
                            return
                        try:
                            label_results_count.configure(text=f"Znaleziono trafień: {total_hits:,} (ładuję pełną listę i statystyki...)".replace(',', ' '))
                        except Exception as exc:
                            if korpusuj_diagnostics_enabled_145c1():
                                logging.info(
                                    "[DIAG search.preview_gui.fail] stage=label_count data=%r",
                                    {"token": search_token, "error": repr(exc)},
                                    exc_info=True,
                                )
                        try:
                            display_page(local_state.query, local_state.corpus)
                        except Exception as exc:
                            if korpusuj_diagnostics_enabled_145c1():
                                logging.info(
                                    "[DIAG search.preview_gui.fail] stage=display_page data=%r",
                                    {
                                        "token": search_token,
                                        "query": getattr(local_state, "query", None),
                                        "corpus": getattr(local_state, "corpus", None),
                                        "error": repr(exc),
                                    },
                                    exc_info=True,
                                )
                    app.after(0, show_metadata_sorted_first_page_035e)
                else:
                    pass

                # Materialize SearchCursor results with cancellation support before final sorting/statistics.
                materialize_payload = _materialize_searchcursor_results_with_cancel_check(
                    results,
                    cancel_check=lambda: search_token != active_search_token,
                    search_token=search_token,
                )
                if materialize_payload.get("cancelled"):
                    return
                results = materialize_payload["results"]
                t_materialize_start_035d = materialize_payload["t_materialize_start_035d"]
                t_materialize_done_035d = materialize_payload["t_materialize_done_035d"]

            t_matched = time.perf_counter()

            # Jeśli to już nie jest aktualne wyszukiwanie, niczego nie nadpisuj.
            if search_token != active_search_token:
                logging.info("Discarding stale search results [request_id=%s]", search_token)
                return

            global last_search_warnings
            last_search_warnings = warnings_list

            if not results:
                logging.info("No results found [request_id=%s]", search_token)
            else:
                # PATCH_130D_NUMBER_RESULTS_WITH_QUERY: normal completion log with query context.
                try:
                    logging.info(
                        'Number of results: %s request_id=%s query=%r corpus=%r sort=%r',
                        len(results) if hasattr(results, "__len__") else locals().get("raw_total"),
                        locals().get("search_token"),
                        locals().get("query"),
                        locals().get("selected_corpus"),
                        locals().get("sort_option"),
                    )
                except Exception:
                    logging.info("Number of results: %s", len(results) if hasattr(results, "__len__") else "unknown")

            # ======================================================================================
            # --- ZMIANA NR 2: MATEMATYKA ZOSTUJE W TLE, DO GUI PRZEKAZUJEMY TYLKO GOTOWE WYNIKI ---
            # ======================================================================================

            global monthly_freq_for_use, monthly_tfidf_for_use, monthly_zscore_for_use
            global fq_data, lemma_df_cache

            monthly_tfidf_for_use = {}
            monthly_zscore_for_use = {}
            fq_data = []
            fq_data_token = []
            fq_data_month = []



            # Keep using the boundary-resolved sort_option from SearchRequest/UI fallback.
    

            # Prepare the stable result list before the first full GUI display.
            # --- SORTOWANIE W MIEJSCU (WYKONUJE SIĘ W TLE) ---
            t_result_prepare_start = time.perf_counter()
            t_sort_start = time.perf_counter()
            sort_search_results_in_place(results, sort_option)
            t_sort_done = time.perf_counter()

            # Przypisujemy posortowaną (lub nie) listę do zmiennej używanej dalej
            results_sorted = results

            t_sorted = time.perf_counter()

            if results_sorted:
                # wyniki najpierw w lokalnym stanie:
                t_state_alias_start = time.perf_counter()
                local_state.results = results_sorted

                # atomowa podmiana stanu i TYLKO wtedy aktualizacja GUI
                with state_lock:
                    _publish_current_search_state_aliases(local_state, include_identity=True, include_results=True, reset_page=False, set_status=True)
                    # Kompatybilność z istniejącym GUI (czytającym ze "starych" globali):

                t_state_alias_done = time.perf_counter()


                # ==========================================================
                # --- NOWOŚĆ: BŁYSKAWICZNE WYŚWIETLENIE PIERWSZEJ STRONY ---
                liczba_trafien = len(results_sorted)

                def show_first_results():
                    global current_page
                    t_first_gui_start = time.perf_counter()
                    current_page = 0

                    label_results_count.configure(
                        text=f"Znaleziono trafień: {liczba_trafien:,} (Ładowanie statystyk...)".replace(',', ' ')
                    )

                    t_display_page_start = time.perf_counter()
                    display_page(local_state.query, local_state.corpus)
                    t_display_page_done = time.perf_counter()

                    # Compute and apply the available publication-date range without blocking the first render.
                    def compute_date_range_from_results_for_entries(results_for_dates):
                        """Compute min/max publication date for date filter fields without touching GUI.
                        
                        KORPUSUJ_MIGRATION_PATCH_113_DATE_RANGE_UNIQUE_DATE_FASTPATH
                        
                        Minimal fast path: keep current semantics by using res[0] as the
                        date source, but parse each distinct raw date string only once.
                        """
                        raw_dates_113 = set()
                        for res in results_for_dates:
                            try:
                                raw_date_113 = res[0]
                            except Exception:
                                continue
                            if raw_date_113 is None:
                                continue
                            if not isinstance(raw_date_113, str):
                                try:
                                    raw_date_113 = str(raw_date_113)
                                except Exception:
                                    continue
                            raw_date_113 = raw_date_113.strip()
                            if raw_date_113:
                                raw_dates_113.add(raw_date_113)
                        if not raw_dates_113:
                            return None, None
                        dates = []
                        for raw_date_113 in raw_dates_113:
                            d = parse_date_safe(raw_date_113)
                            if d:
                                dates.append(d)
                        if not dates:
                            return None, None
                        return min(dates), max(dates)

                    def apply_date_range_to_entries(min_date, max_date):
                        """Apply computed date range to GUI entries. Must run in Tk thread."""
                        if not min_date or not max_date:
                            return

                        current_state = date_start_entry.cget("state")
                        date_start_entry.configure(state="normal")
                        date_end_entry.configure(state="normal")

                        date_start_entry.delete(0, "end")
                        date_start_entry.insert(0, min_date.strftime("%d-%m-%Y"))
                        date_end_entry.delete(0, "end")
                        date_end_entry.insert(0, max_date.strftime("%d-%m-%Y"))

                        date_start_entry.configure(state=current_state)
                        date_end_entry.configure(state=current_state)

                    def start_date_range_worker():

                        if search_token != active_search_token:
                            return

                        def date_range_worker():
                            if search_token != active_search_token:
                                return

                            t_auto_dates_worker_start = time.perf_counter()
                            min_date, max_date = compute_date_range_from_results_for_entries(results_sorted)
                            t_auto_dates_worker_done = time.perf_counter()

                            if korpusuj_verbose_diagnostics_enabled_145c1():
                                logging.info(
                                    "[DIAG perf.gui.result_prep] stage='date_worker' token=%s | auto_fill_dates_worker=%.4fs",
                                    search_token,
                                    t_auto_dates_worker_done - t_auto_dates_worker_start,
                                )

                            if search_token != active_search_token:
                                logging.info("Pomijam zakres dat dla anulowanego wyszukiwania [request_id=%s]",
                                             search_token)
                                return

                            def apply_date_range_if_current():
                                if search_token != active_search_token:
                                    return
                                apply_date_range_to_entries(min_date, max_date)

                            app.after(0, apply_date_range_if_current)

                        threading.Thread(target=date_range_worker, daemon=True).start()

                    # Dajemy Tk realną szansę zakończyć render pierwszej strony przed startem parsowania dat.
                    app.after(500, start_date_range_worker)

                    if korpusuj_verbose_diagnostics_enabled_145c1():
                        logging.info(
                            "[DIAG perf.gui.result_prep] stage='first_display' token=%s | gui_queue=%.4fs | display_page=%.4fs | first_gui_total=%.4fs",
                            search_token,
                            t_first_gui_start - t_first_page_scheduled,
                            t_display_page_done - t_display_page_start,
                            t_display_page_done - t_first_gui_start,
                        )


                if korpusuj_verbose_diagnostics_enabled_145c1():
                    logging.info(
                        "[DIAG perf.gui.result_prep] stage='prepare' token=%s | sort=%.4fs | state_alias=%.4fs | before_first_gui_schedule=%.4fs | results=%s",
                        search_token,
                        t_sort_done - t_sort_start,
                        t_state_alias_done - t_state_alias_start,
                        time.perf_counter() - t_result_prepare_start,
                        len(results_sorted),
                    )

                # Zlecamy odświeżenie GUI do głównego wątku NATYCHMIAST
                t_first_page_scheduled = time.perf_counter()
                app.after(0, show_first_results)
                # ==========================================================

                # --- AGREGACJA STATYSTYK (WYKONUJE SIĘ W TLE) ---
                if "Data publikacji" in df.columns:


                    # Build the GUI-free statistics input contract for background preparation.
                    def build_statistics_worker_contract():
                        """Build a GUI-free input contract for the statistics computation path.

                        This is still consumed synchronously. A later migration can pass this
                        contract to a dedicated statistics worker thread.
                        """
                        return {
                            "results": list(results_sorted),
                            "corpus": local_state.corpus,
                            "has_publication_dates": "Data publikacji" in df.columns,
                            "plot_enabled": ui_state["is_plotting"] == 'Tak',
                            "plot_style": ui_state["plot_style"],
                        }

                    statistics_worker_contract = build_statistics_worker_contract()

                    _statistics_frequency_input_context = _prepare_statistics_frequency_input_context(
                        statistics_worker_contract,
                        inverted_indexes,
                        df,
                        monthly_lemma_freq,
                        true_monthly_totals,
                    )
                    _frequency_inputs = _statistics_frequency_input_context['frequency_inputs']
                    unique_matched_tokens = _statistics_frequency_input_context['unique_matched_tokens']
                    unique_lemmas = _statistics_frequency_input_context['unique_lemmas']
                    exact_orth_df = _statistics_frequency_input_context['exact_orth_df']
                    exact_lemma_df = _statistics_frequency_input_context['exact_lemma_df']
                    raw_monthly_counts = _statistics_frequency_input_context['raw_monthly_counts']
                    _monthly_counts_flattened = _statistics_frequency_input_context['monthly_counts_flattened']
                    _true_monthly_totals = _statistics_frequency_input_context['true_monthly_totals_source']
                    total_token_count = _statistics_frequency_input_context['total_token_count']
                    total_docs = _statistics_frequency_input_context['total_docs']

                    def _is_phrase_key(key):
                        """True dla dopasowań wielotokenowych, np. 'ministerstwo edukacja narodowy'."""
                        return isinstance(key, str) and len(key.split()) > 1

                    def _df_for_matched_key(key, attr, exact_df_map):
                        """
                        DF/Rozproszenie dla tabel wyników.

                        Dla fraz wielowyrazowych indeks odwrócony attr -> token/lemma NIE ma klucza
                        'ministerstwo edukacja narodowy', więc poprzedni kod wpadał w fallback = 1.
                        Dlatego dla fraz bierzemy DF z faktycznych trafień (unikalne row_idx).
                        Dla pojedynczych tokenów/lematów zachowujemy dotychczasowy globalny DF z indeksu.
                        """
                        if _is_phrase_key(key):
                            return max(1, len(exact_df_map.get(key, set())))

                        global_docs_set = inverted_indexes[global_selected_corpus].get(attr, {}).get(key, set())
                        return len(global_docs_set) if global_docs_set else max(1, len(exact_df_map.get(key, set())))

                    # DF dla tabeli base: frazy liczymy po faktycznych trafieniach, pojedyncze lematy globalnie.
                    lemma_df_cache = {
                        lemma: _df_for_matched_key(lemma, "base", exact_lemma_df)
                        for lemma in unique_lemmas
                    }


                    _global_frequency_tables = build_global_frequency_tables(
                        unique_matched_tokens=unique_matched_tokens,
                        monthly_lemma_freq=monthly_lemma_freq,
                        exact_orth_df=exact_orth_df,
                        exact_lemma_df=exact_lemma_df,
                        total_token_count=total_token_count,
                        total_docs=total_docs,
                        df_for_matched_key=_df_for_matched_key,
                    )


                    local_state.monthly_lemma_freq = dict(monthly_lemma_freq)
                    _monthly_frequency_tables = build_monthly_frequency_tables(
                        monthly_lemma_freq=monthly_lemma_freq,
                        unique_lemmas=unique_lemmas,
                        true_monthly_totals=true_monthly_totals,
                        total_docs=total_docs,
                        exact_lemma_df=exact_lemma_df,
                        df_for_matched_key=_df_for_matched_key,
                        calc_z_score_func=calc_z_score,
                    )
                    (
                        fq_data_token,
                        fq_data,
                        s_lemma_total_freq,
                        s_lemma_global_pmw,
                        s_lemma_global_tfidf,
                        monthly_freq_for_use,
                        monthly_tfidf_for_use,
                        monthly_zscore_for_use,
                        fq_data_month,
                    ) = _publish_frequency_table_outputs_036l4g52j(
                        _global_frequency_tables,
                        _monthly_frequency_tables,
                    )

                    if statistics_worker_contract["plot_enabled"]:
                        # Wykres również generowany w tle - dzięki API obiektowemu Matplotlib
                        _generate_statistics_plot_image(
                            monthly_freq_for_use=monthly_freq_for_use,
                            plot_style=statistics_worker_contract["plot_style"],
                            output_path=str(writable_temp_root() / "temp_plot.png"),
                        )

                # ZAPIS DO LOGA I KONSOLI:
                t_stats = time.perf_counter()  # <--- CZAS PO STATYSTYKACH

                # ZAPIS DO LOGA I KONSOLI:
                profiling_msg = (
                    f"⏱ [PROFILING] Token: {search_token} | "
                    f"Walidacja: {t_parsed - t_start:.4f}s | "
                    f"Skanowanie (find_lemma): {t_matched - t_parsed:.4f}s | "
                    f"Sortowanie: {t_sorted - t_matched:.4f}s | "
                    f"Statystyki+Wykres: {t_stats - t_sorted:.4f}s || "
                    f"CAŁOŚĆ: {t_stats - t_start:.4f}s"
                )
                if search_diag_enabled():
                    logging.info(profiling_msg)
                # =========================================================================
                # --- AKTUALIZACJA GUI (WYKONUJE SIĘ BEZPIECZNIE W GŁÓWNYM WĄTKU) ---
                # =========================================================================
                # Prepare final statistics payload/state without touching Tk widgets directly.
                def run_statistics_worker_sync(statistics_worker_contract):
                    """Prepare final statistics payload/state before applying GUI updates.
                    
                    This helper is still called synchronously. It must stay free of
                    GUI rendering, GUI scheduling, table updates, and concordance rendering.
                    """
                    search_statistics = _build_search_statistics_payload(
                        true_monthly_totals=true_monthly_totals,
                        monthly_freq_for_use=monthly_freq_for_use,
                        monthly_tfidf_for_use=monthly_tfidf_for_use,
                        monthly_zscore_for_use=monthly_zscore_for_use,
                        fq_data=fq_data,
                        fq_data_token=fq_data_token,
                        fq_data_month=fq_data_month,
                        s_lemma_total_freq=s_lemma_total_freq,
                        s_lemma_global_pmw=s_lemma_global_pmw,
                        s_lemma_global_tfidf=s_lemma_global_tfidf,
                    )
                    statistics_payload_context = _build_statistics_payload_context(
                        statistics=search_statistics,
                        unique_lemmas=unique_lemmas,
                        lemma_df_cache=lemma_df_cache,
                        monthly_lemma_freq=monthly_lemma_freq,
                    )
                    _stage_statistics_payload_on_local_state_036l4g52f(local_state, statistics_payload_context)
                    return statistics_payload_context

                # Schedule statistics preparation outside the main search flow.
                # Statystyki są od teraz przygotowywane przez osobny worker statystyk.
                # Zmienna pozostaje tylko jako czytelny znacznik granicy; wynik aplikuje worker.
                statistics_payload_context = None

                # Apply completed statistics data to GUI widgets in the Tk thread.
                def complete_statistics_gui_update():
                    global search_status
                    search_status = 0
                    liczba = len(local_state.results)
                    label_results_count.configure(
                        text=f"Znaleziono trafień: {liczba:,} (statystyki gotowe)".replace(',', ' ')
                    )
                    # Keep the already-rendered concordance stable when statistics finish.
                    # Nie przebudowujemy tabeli konkordancji po zakończeniu statystyk.
                    # Pierwsza strona została już pokazana wcześniej przez show_first_results(),
                    # a ten etap powinien aktualizować tylko panele statystyczne.

                    # Update only statistics/frequency panels after statistics are ready.
                    def update_statistics_panels():
                        """Update statistics/frequency panels after statistics are ready.
                        
                        This intentionally does not touch the concordance page.
                        Concordance is owned by the earlier stable-result display path.
                        """
                        _statistics_panels_update_036l4g52y = _update_statistics_panels(
                            df=df,
                            ui_state=ui_state,
                            fq_data_token=fq_data_token,
                            fq_data=fq_data,
                            fq_data_month=fq_data_month,
                            paginator_token=paginator_token,
                            paginator_fq=paginator_fq,
                            paginator_month=paginator_month,
                            frekw_dane_tabela_orth=frekw_dane_tabela_orth,
                            frekw_dane_tabela=frekw_dane_tabela,
                            frekw_dane_tabela_month=frekw_dane_tabela_month,
                            update_table_func=update_table,
                            update_plot_images_func=update_plot_images,
                        )
                        return _statistics_panels_update_036l4g52y

                    update_statistics_panels()

                    if "Data publikacji" in df.columns:
                        for child in checkboxes_frame.winfo_children():
                            child.destroy()

                        lemma_vars.clear()
                        merge_entry_vars.clear()

                        def build_listbox_ui(parent_frame, sorted_lemma_freq, vars_dict, merge_dict,
                                             update_plot_callback, items_per_page=100):
                            _listbox_panel = _StatisticsListboxPanel(
                                ctk_module=ctk,
                                math_module=math,
                                parent_frame=parent_frame,
                                sorted_lemma_freq=sorted_lemma_freq,
                                vars_dict=vars_dict,
                                merge_dict=merge_dict,
                                update_plot_callback=update_plot_callback,
                                theme=theme,
                                items_per_page=items_per_page,
                            )
                            return (
                                _listbox_panel.container,
                                _listbox_panel.listbox_frame,
                                _listbox_panel.rename_entry,
                                _listbox_panel.set_data,
                            )

                        container_listbox, listbox_frame, rename_entry, set_data_listbox = build_listbox_ui(
                            checkboxes_frame, s_lemma_total_freq, lemma_vars, merge_entry_vars, update_plot
                        )
                        container_listbox.pack(fill="both", expand=True)

                        def toggle_listboxes(*args):
                            mode = wykres_sort_mode.get()
                            _statistics_listbox_data = _select_statistics_listbox_data(
                                mode=mode,
                                s_lemma_global_tfidf=s_lemma_global_tfidf,
                                monthly_zscore_for_use=monthly_zscore_for_use,
                                s_lemma_global_pmw=s_lemma_global_pmw,
                                s_lemma_total_freq=s_lemma_total_freq,
                                unique_lemmas=unique_lemmas,
                            )
                            set_data_listbox(_statistics_listbox_data)

                        for trace_id in wykres_sort_mode.trace_info():
                            wykres_sort_mode.trace_remove(*trace_id[0:2])
                        wykres_sort_mode.trace_add("write", toggle_listboxes)
                        toggle_listboxes()

                        # Zapisz statystyki do jednego obiektu stanu.
                        # Publish statistics aliases after local_state has the completed statistics payload.
                        # Po zakończeniu statystyk aktualizujemy tylko stan/statystyczne aliasy.
                        # Aliasów konkordancji (`full_results_sorted`, `global_query`,
                        # `global_selected_corpus`) nie dotykamy tutaj, bo należą do
                        # wcześniejszej ścieżki stabilnej listy wyników.
                        def update_statistics_state_aliases():
                            return _publish_statistics_state_aliases_with_current_state_036l4g53b(
                                local_state=local_state,
                                state_lock_obj=state_lock,
                                globals_dict=globals(),
                                publish_statistics_aliases_func=_publish_statistics_state_aliases_036l4g52c,
                            )

                        update_statistics_state_aliases()

                # Run statistics payload preparation in a background worker and schedule GUI completion.
                def statistics_worker_async():
                    """Run statistics payload/state preparation outside the search worker flow.

                    This worker must not touch Tk widgets directly. It may only schedule GUI
                    work through app.after(...) after checking the active search token.
                    """
                    try:
                        if search_token != active_search_token:
                            logging.info("Pomijam start statystyk dla anulowanego wyszukiwania [request_id=%s]", search_token)
                            return

                        run_statistics_worker_sync(statistics_worker_contract)

                        if search_token != active_search_token:
                            logging.info("Pomijam wynik statystyk dla anulowanego wyszukiwania [request_id=%s]", search_token)
                            return

                        _schedule_statistics_worker_completion(
                            app_obj=app,
                            local_state=local_state,
                            last_search_warnings=last_search_warnings,
                            show_search_warnings_func=show_search_warnings,
                            add_to_history_func=add_to_history,
                            push_nav_state_func=push_nav_state,
                            complete_statistics_gui_update_func=complete_statistics_gui_update,
                        )
                    except Exception as e:
                        logging.exception("Statistics worker failed [request_id=%s]", search_token)
                        error_msg = f"Błąd obliczania statystyk: {e}"
                        _schedule_statistics_worker_error(
                            app_obj=app,
                            error_msg=error_msg,
                            show_search_error_func=show_search_error,
                        )

                threading.Thread(target=statistics_worker_async, daemon=True).start()
                return


            else:
                # Brak wyników
                def update_no_results():
                    global search_status, full_results_sorted, current_page  # <--- DODANO current_page
                    _reset_empty_search_results_aliases_036l4g51n()
                    label_results_count.configure(text="Znaleziono trafień: 0")
                    display_page(query, selected_corpus)

                _schedule_no_results_branch_completion_036l4g56b(
                    app_obj=app,
                    last_search_warnings=last_search_warnings,
                    show_search_warnings_func=show_search_warnings,
                    update_no_results_func=update_no_results,
                )



        except (QueryValidationError, QueryParseError) as e:

            logging.warning("Validation or Parse error in search thread [request_id=%s]: %s", search_token, e)

            if search_token == active_search_token:
                # 1. Zapisujemy treść błędu do bezpiecznej zmiennej tekstowej
                error_msg = str(e)
                _schedule_search_thread_error_display_036l4g55d(
                    app_obj=app,
                    label_results_count_widget=label_results_count,
                    error_msg=error_msg,
                    show_search_error_func=show_search_error,
                )

        except Exception as e:

            logging.exception("Error in search thread [request_id=%s]", search_token)

            if search_token == active_search_token:
                # To samo tutaj - zapisujemy sformatowany tekst przed wrzuceniem do lambdy
                error_msg = f"Nie udało się wykonać wyszukiwania.\nSzczegóły: {e}"
                _schedule_search_thread_error_display_036l4g55d(
                    app_obj=app,
                    label_results_count_widget=label_results_count,
                    error_msg=error_msg,
                    show_search_error_func=show_search_error,
                )

        finally:

            with search_guard:
                global search_in_progress
                if search_token == active_search_token:
                    search_in_progress = False
            if search_token == active_search_token:
                app.after(0, lambda: button_search.configure(text="" if globals().get("s_img") else "Szukaj", image=globals().get("s_img"), state="normal"))

    search_diag_log("SEARCH_CALL_THREAD thread=%s", threading.current_thread().name)

    def start_search_worker():
        try:
            thread = threading.Thread(target=search_thread, args=(local_token, gui_state), daemon=True)
            thread.start()
            search_diag_log("SEARCH_THREAD_STARTED")
        except Exception as e:
            logging.error("Nie udało się uruchomić wątku wyszukiwania: %s", e, exc_info=True)
            with search_guard:
                global search_in_progress
                if local_token == active_search_token:
                    search_in_progress = False
            try:
                button_search.configure(text="" if globals().get("s_img") else "Szukaj", image=globals().get("s_img"), state="normal")
            except Exception:
                pass
            try:
                show_search_error(f"Nie udało się rozpocząć wyszukiwania.\nSzczegóły: {e}")
            except Exception:
                pass

    try:
        app.after(30, start_search_worker)
    except Exception:
        start_search_worker()


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
    """Fit the saved trend chart inside the current chart viewport."""
    try:
        target_img = writable_temp_root() / "temp_plot.png"
        target_label = frekw_wykresy

        if not target_img.exists():
            logging.info("Plot image not found: %s", target_img)
            return

        target_label.update_idletasks()
        viewport_width = int(target_label.winfo_width())
        viewport_height = int(target_label.winfo_height())

        # Hidden or not-yet-realized tabs may temporarily report tiny sizes.
        if viewport_width < 50 or viewport_height < 50:
            return

        margin = 10
        available_width = max(1, viewport_width - 2 * margin)
        available_height = max(1, viewport_height - 2 * margin)

        with Image.open(target_img) as source_image:
            image = source_image.convert("RGBA")

        if image.width < 1 or image.height < 1:
            return

        # Tk/CustomTkinter already exposes widget dimensions in the coordinate
        # system used by the layout. Do not apply another DPI multiplier.
        scale = min(
            available_width / image.width,
            available_height / image.height,
            1.0,
        )
        target_width = max(1, int(round(image.width * scale)))
        target_height = max(1, int(round(image.height * scale)))

        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.LANCZOS)

        chart_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(target_width, target_height),
        )
        target_label.configure(image=chart_image, text="")
        target_label.image = chart_image
        target_label._image_ref = image

    except Exception as exc:
        logging.info("Error loading plot image: %s", exc)


def update_plot():
    global full_results_sorted, true_monthly_totals, lemma_vars, merge_entry_vars, lemma_df_cache, global_selected_corpus
    global precalculated_bins, precalculated_bin_totals, precalculated_lemma_counts
    global min_tokens_threshold

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

        writable_temp_root().mkdir(parents=True, exist_ok=True)
        canvas = FigureCanvasAgg(fig)
        fig.savefig(writable_temp_root() / "temp_plot.png", bbox_inches="tight")
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

    # --- DYNAMICZNY PRÓG (AUTO) Z MEDIANĄ I DOLNYM ZABEZPIECZENIEM ---
    valid_totals = [t for t in precalculated_bin_totals if t > 0]

    if min_tokens_threshold == 0 and valid_totals:
        # Mediana jest odporniejsza na skrajne wartości niż średnia
        median_bin_size = np.median(valid_totals)
        # Ustawiamy próg na 10% mediany, ale nie mniej niż 50 tokenów (ochrona mikro-korpusów)
        dynamic_threshold = max(50, median_bin_size * 0.1)
    else:
        dynamic_threshold = min_tokens_threshold


    # --------------------------------------

    # GŁÓWNA PĘTLA RYSOWANIA WYKRESÓW
    for idx, (g_name, raw_values) in enumerate(plot_data_raw.items()):
        pmw_values = []
        tfidf_values = []
        raw_filtered_values = []  # <--- NOWOŚĆ: Lista na bezpieczne surowe dane

        total_idf = sum(math.log10(total_docs / (lemma_df_cache.get(l, 1) or 1)) for l in groups[g_name])
        avg_idf = total_idf / len(groups[g_name]) if groups[g_name] else 0

        for i, v in enumerate(raw_values):
            total_in_bin = precalculated_bin_totals[i]

            # Używamy globalnej zmiennej z ustawień (min_tokens_threshold) zamiast sztywnej wartości
            if total_in_bin >= dynamic_threshold:
                pmw = 0 if not total_in_bin else v / (total_in_bin / 1e6)
                tf = (0 if not total_in_bin else v / total_in_bin)
                tfidf = tf * avg_idf * 100000

                pmw_values.append(pmw)
                tfidf_values.append(tfidf)
                raw_filtered_values.append(v)  # <--- Zostawiamy surową wartość
            else:
                pmw_values.append(np.nan)
                tfidf_values.append(np.nan)
                raw_filtered_values.append(np.nan)  # <--- PRZERYWAMY WYKRES RÓWNIEŻ TUTAJ

        if mode == "Częstość względna":
            final_vals = pmw_values

        elif mode == "TF-IDF":
            final_vals = tfidf_values

        elif mode == "Z-score":
            valid_vals = np.array(pmw_values, dtype=float)
            valid_count = np.sum(~np.isnan(valid_vals))

            if valid_count >= 2:
                mean_v = np.nanmean(valid_vals)
                std_v = np.nanstd(valid_vals)

                if std_v > 0:
                    final_vals = [
                        (v - mean_v) / std_v if not np.isnan(v) else np.nan
                        for v in valid_vals
                    ]
                else:
                    final_vals = [np.nan if np.isnan(v) else 0.0 for v in valid_vals]
            else:
                final_vals = [np.nan] * len(valid_vals)

        else:
            # Surowa liczba wystąpień - teraz uwzględnia bezpieczne przerwy!
            final_vals = raw_filtered_values

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

    if scale_mode_var.get() == "Ręczne":
        try:
            y_limit_str = entry_y_limit.get().strip()
            if y_limit_str.replace('.', '', 1).isdigit():
                y_limit_val = float(y_limit_str)

                # Zabezpieczenie przed wpisaniem zera lub minusa (poza Z-score)
                if y_limit_val > 0 or mode == "Z-score":
                    if mode in ["Częstość względna", "Liczba wystąpień", "TF-IDF"]:
                        ax.set_ylim(bottom=0, top=y_limit_val)
                    else:
                        ax.set_ylim(top=y_limit_val)
        except ValueError:
            pass  # Jeśli błąd parsowania, zostaje domyślne Auto z Matplotlib

    # Przeniesione idealne układanie legendy z kopii 4
    ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.32), frameon=False, labelcolor=text_color)

    fig.tight_layout(rect=[0, 0, 1, 0.85])

    # Bezpieczny zapis z użyciem FigureCanvasAgg
    writable_temp_root().mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    fig.savefig(writable_temp_root() / "temp_plot.png", bbox_inches="tight")
    update_plot_images()

def on_resize(event=None):
    """Debounce viewport-only chart image fitting."""
    global debounce_timer
    if debounce_timer:
        try:
            app.after_cancel(debounce_timer)
        except Exception:
            pass
    debounce_timer = app.after(100, update_plot_images)


def handle_tab_change():
    """Preserve tab navigation state and refresh a newly visible chart."""
    push_nav_state()
    if tabview.get() == "Trendy":
        app.after_idle(update_plot_images)
        app.after(100, update_plot_images)


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
            shutil.copy(str(writable_temp_root() / "temp_plot.png"), file_path)
            print(f"Plot saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać wykresu.\nSzczegóły: {e}")



# KORPUSUJ_PATCH_137R_GUI_SAFE_ROW_RESOLVER_FOR_HIGHLIGHTS_AND_DEPENDENCY_GRAPH
def _safe_gui_row_from_df_137r(df_obj, row_idx, *, purpose="gui_row", logger=None):
    """Resolve GUI row data without assuming row_idx is a pandas .loc label."""
    try:
        if row_idx is None or df_obj is None:
            return None
        try:
            idx = getattr(df_obj, "index", None)
            if idx is not None:
                try:
                    if row_idx in idx:
                        return df_obj.loc[row_idx]
                except Exception:
                    pass
        except Exception:
            pass
        try:
            row_pos = int(row_idx)
            if row_pos >= 0:
                try:
                    n = len(df_obj)
                except Exception:
                    n = None
                if n is not None and row_pos < int(n):
                    iloc = getattr(df_obj, "iloc", None)
                    if iloc is not None:
                        try:
                            return iloc[row_pos]
                        except Exception:
                            pass
        except Exception:
            pass
        for method_name in ("get_doc", "get_document", "load_doc", "load_document", "get_row"):
            try:
                method = getattr(df_obj, method_name, None)
                if callable(method):
                    try:
                        value = method(row_idx)
                    except TypeError:
                        continue
                    if value is not None:
                        return value
            except Exception:
                pass
        try:
            log = logger
            if log is None:
                import logging as log
            log.warning("[GUI_SAFE_ROW_RESOLVER_137R] purpose=%r row_idx=%r df_type=%s len=%r not_found", purpose, row_idx, type(df_obj).__name__, len(df_obj) if hasattr(df_obj, "__len__") else None)
        except Exception:
            pass
        return None
    except Exception as exc:
        try:
            log = logger
            if log is None:
                import logging as log
            log.warning("[GUI_SAFE_ROW_RESOLVER_137R] purpose=%r row_idx=%r failed=%r", purpose, row_idx, exc, exc_info=True)
        except Exception:
            pass
        return None
# END KORPUSUJ_PATCH_137R_GUI_SAFE_ROW_RESOLVER_FOR_HIGHLIGHTS_AND_DEPENDENCY_GRAPH


# KORPUSUJ_PATCH_137U_GUI_HIGHLIGHT_RUNTIME_STATE_DIAGNOSTICS
def _gui_highlight_runtime_diag_137u(event, **payload):
    """Write compact runtime diagnostics for GUI NER/coref/dependency highlight state."""
    try:
        import json as _json_137u
        import logging as _logging_137u

        def _safe(v, limit=500):
            try:
                if v is None or isinstance(v, (int, float, bool)):
                    return v
                if isinstance(v, str):
                    return v[:limit]
                if isinstance(v, (list, tuple)):
                    return [_safe(x, limit=120) for x in list(v)[:20]]
                if isinstance(v, dict):
                    return {str(k): _safe(val, limit=120) for k, val in list(v.items())[:20]}
                return repr(v)[:limit]
            except Exception:
                return None

        row_idx = payload.get("row_idx")
        df_obj = payload.get("df_obj")
        row_data = payload.get("row_data")
        info = {"event": event}

        for key, val in payload.items():
            if key not in {"df_obj", "row_data"}:
                info[key] = _safe(val)

        try:
            info["df_type"] = type(df_obj).__name__ if df_obj is not None else None
            info["df_len"] = len(df_obj) if df_obj is not None and hasattr(df_obj, "__len__") else None
            idx = getattr(df_obj, "index", None) if df_obj is not None else None
            info["df_index_type"] = type(idx).__name__ if idx is not None else None
            if idx is not None:
                try:
                    info["row_idx_in_df_index"] = row_idx in idx
                except Exception as exc:
                    info["row_idx_in_df_index_error"] = repr(exc)
                try:
                    info["df_index_sample"] = [repr(x) for x in list(idx[:8])]
                except Exception:
                    pass
            try:
                row_pos = int(row_idx)
                info["row_idx_int"] = row_pos
                info["row_idx_iloc_candidate"] = (info.get("df_len") is not None and 0 <= row_pos < int(info.get("df_len")))
            except Exception:
                info["row_idx_int"] = None
                info["row_idx_iloc_candidate"] = False
        except Exception as exc:
            info["df_probe_error"] = repr(exc)

        try:
            info["row_data_type"] = type(row_data).__name__ if row_data is not None else None
            for attr in ("start_ids", "end_ids", "ners", "corefs", "sentence_ids", "word_ids", "head_ids", "tokens", "lemmas", "orths", "upos", "upostags", "deprels"):
                if row_data is not None and hasattr(row_data, attr):
                    value = getattr(row_data, attr)
                    try:
                        info[f"row_{attr}_len"] = len(value)
                    except Exception:
                        info[f"row_{attr}_len"] = None
                    try:
                        info[f"row_{attr}_sample"] = _safe(list(value)[:8])
                    except Exception:
                        info[f"row_{attr}_sample"] = _safe(value)
        except Exception as exc:
            info["row_data_probe_error"] = repr(exc)

        try:
            if "text_full" in globals():
                tf = globals().get("text_full")
                body_mark = globals().get("current_body_start_mark")
                if tf is not None and body_mark:
                    try:
                        info["text_body_sample"] = tf.get(body_mark, f"{body_mark} + 300c")[:300]
                    except Exception as exc:
                        info["text_body_sample_error"] = repr(exc)
        except Exception:
            pass

        # 137y: runtime diagnostics from 137u were useful during debugging but are now muted.
        return
    except Exception:
        pass
# END KORPUSUJ_PATCH_137U_GUI_HIGHLIGHT_RUNTIME_STATE_DIAGNOSTICS


# KORPUSUJ_PATCH_137V_GUI_HIGHLIGHT_RESOLVE_ROW_BY_DISPLAY_METADATA
current_display_publication_date_137v = None
current_display_title_137v = None
current_display_author_137v = None
current_display_text_sample_137v = None
_gui_display_row_cache_137v = {}

def _norm_gui_meta_137v(value):
    try:
        if value is None:
            return ""
        return str(value).strip()
    except Exception:
        return ""

def _row_value(row_data, names):
    for name in names:
        try:
            if hasattr(row_data, "get"):
                value = row_data.get(name, None)
                if value is not None:
                    return value
        except Exception:
            pass
        try:
            if hasattr(row_data, name):
                value = getattr(row_data, name)
                if value is not None:
                    return value
        except Exception:
            pass
    return None

def _row_len_attr_137v(row_data, attr):
    try:
        value = getattr(row_data, attr)
        return len(value)
    except Exception:
        return None

def _row_start_idx_in_bounds_137v(row_data, start_idx, attr="start_ids"):
    try:
        if row_data is None or start_idx is None:
            return False
        length = _row_len_attr_137v(row_data, attr)
        if length is None:
            return False
        pos = int(start_idx)
        return 0 <= pos < int(length)
    except Exception:
        return False

def _row_matches_current_display_metadata_137v(row_data):
    try:
        title_cur = _norm_gui_meta_137v(globals().get("current_display_title_137v"))
        date_cur = _norm_gui_meta_137v(globals().get("current_display_publication_date_137v"))
        author_cur = _norm_gui_meta_137v(globals().get("current_display_author_137v"))
        if not title_cur and not date_cur and not author_cur:
            return True
        row_title = _norm_gui_meta_137v(_row_value(row_data, ("Tytuł", "title", "Title")))
        row_date = _norm_gui_meta_137v(_row_value(row_data, ("Data publikacji", "publication_date", "date", "Date")))
        row_author = _norm_gui_meta_137v(_row_value(row_data, ("Autor", "author", "Author")))
        if title_cur and row_title and row_title != title_cur:
            return False
        if date_cur and row_date and row_date != date_cur:
            return False
        # Author is often empty/noisy; enforce only if both sides are non-empty.
        if author_cur and row_author and row_author != author_cur:
            return False
        # If title was available, title match is sufficient; otherwise date/author soft match.
        if title_cur and row_title == title_cur:
            return True
        if date_cur and row_date == date_cur:
            return True
        if author_cur and row_author == author_cur:
            return True
        return not (row_title or row_date or row_author)
    except Exception:
        return False

def _row_for_position(df_obj, pos):
    try:
        iloc = getattr(df_obj, "iloc", None)
        if iloc is not None:
            return iloc[int(pos)]
    except Exception:
        pass
    try:
        loc = getattr(df_obj, "loc", None)
        if loc is not None:
            return loc[int(pos)]
    except Exception:
        pass
    for method_name in ("get_doc", "get_document", "load_doc", "load_document", "get_row"):
        try:
            method = getattr(df_obj, method_name, None)
            if callable(method):
                value = method(int(pos))
                if value is not None:
                    return value
        except Exception:
            pass
    return None

def _resolve_gui_row_by_display_metadata_137v(df_obj, row_idx, start_idx, *, purpose="gui_row"):
    """Resolve row_data for displayed full text, validating metadata and start index.

    137r avoided KeyError, but could select df.iloc[row_idx] for a result id that was
    not the displayed document. This resolver first tries the 137r candidate, then
    validates it against displayed title/date/author and start_idx bounds. If it does
    not match, it scans the corpus for a row matching displayed metadata and bounds.
    """
    try:
        cache = globals().setdefault("_gui_display_row_cache_137v", {})
        key = (
            _norm_gui_meta_137v(globals().get("global_selected_corpus")),
            _norm_gui_meta_137v(globals().get("current_display_title_137v")),
            _norm_gui_meta_137v(globals().get("current_display_publication_date_137v")),
            _norm_gui_meta_137v(globals().get("current_display_author_137v")),
            int(start_idx) if start_idx is not None else None,
        )
        if key in cache:
            cached_pos = cache.get(key)
            cached = _row_for_position(df_obj, cached_pos)
            if cached is not None and _row_start_idx_in_bounds_137v(cached, start_idx) and _row_matches_current_display_metadata_137v(cached):
                try:
                    _gui_highlight_runtime_diag_137u(f"{purpose}.resolved_137v_cache", df_obj=df_obj, row_idx=cached_pos, start_idx=start_idx, row_data=cached)
                except Exception:
                    pass
                return cached

        candidate = None
        try:
            candidate = _safe_gui_row_from_df_137r(df_obj, row_idx, purpose=purpose)
        except Exception:
            candidate = None
        if candidate is not None and _row_start_idx_in_bounds_137v(candidate, start_idx) and _row_matches_current_display_metadata_137v(candidate):
            try:
                cache[key] = int(row_idx)
            except Exception:
                pass
            try:
                _gui_highlight_runtime_diag_137u(f"{purpose}.resolved_137v_direct", df_obj=df_obj, row_idx=row_idx, start_idx=start_idx, row_data=candidate)
            except Exception:
                pass
            return candidate

        # Scan for metadata match. Suitable for the current corpus size (~thousands of docs).
        try:
            n = len(df_obj)
        except Exception:
            n = 0
        max_scan = min(int(n or 0), 20000)
        for pos in range(max_scan):
            row = _row_for_position(df_obj, pos)
            if row is None:
                continue
            if not _row_start_idx_in_bounds_137v(row, start_idx):
                continue
            if not _row_matches_current_display_metadata_137v(row):
                continue
            cache[key] = pos
            try:
                _gui_highlight_runtime_diag_137u(f"{purpose}.resolved_137v_metadata_scan", df_obj=df_obj, row_idx=pos, requested_row_idx=row_idx, start_idx=start_idx, row_data=row)
            except Exception:
                pass
            return row

        try:
            import logging as _logging_137v
            _logging_137v.warning(
                "[APP gui.display_row.resolve] purpose=%r row_idx=%r start_idx=%r title=%r date=%r author=%r unresolved",
                purpose,
                row_idx,
                start_idx,
                globals().get("current_display_title_137v"),
                globals().get("current_display_publication_date_137v"),
                globals().get("current_display_author_137v"),
            )
        except Exception:
            pass
        return None
    except Exception as exc:
        try:
            import logging as _logging_137v
            _logging_137v.warning("[APP gui.display_row.resolve] purpose=%r failed=%r", purpose, exc, exc_info=True)
        except Exception:
            pass
        return None
# END KORPUSUJ_PATCH_137V_GUI_HIGHLIGHT_RESOLVE_ROW_BY_DISPLAY_METADATA


# KORPUSUJ_PATCH_137W_GUI_DISPLAY_ROW_RESOLVER_STRICT_METADATA_AND_NO_BAD_CACHE
def _row_matches_current_display_metadata_137w(row_data):
    """Strict metadata match for displayed document -> row_data.

    137v allowed empty row metadata to count as a match, which could bind the
    displayed text to an unrelated df.iloc/doc candidate. 137w requires a real
    title match whenever the displayed title is known; date/author are additional
    checks, not substitutes for a missing title when title is available.
    """
    try:
        title_cur = _norm_gui_meta_137v(globals().get("current_display_title_137v"))
        date_cur = _norm_gui_meta_137v(globals().get("current_display_publication_date_137v"))
        author_cur = _norm_gui_meta_137v(globals().get("current_display_author_137v"))

        row_title = _norm_gui_meta_137v(_row_value(row_data, ("Tytuł", "title", "Title")))
        row_date = _norm_gui_meta_137v(_row_value(row_data, ("Data publikacji", "publication_date", "date", "Date")))
        row_author = _norm_gui_meta_137v(_row_value(row_data, ("Autor", "author", "Author")))

        # If GUI knows displayed title, row must expose and match the same title.
        if title_cur:
            if not row_title or row_title != title_cur:
                return False
        elif date_cur:
            if not row_date or row_date != date_cur:
                return False
        else:
            # No reliable metadata available: do not guess.
            return False

        if date_cur and row_date and row_date != date_cur:
            return False
        if author_cur and row_author and row_author != author_cur:
            return False
        return True
    except Exception:
        return False

def _resolve_gui_row_by_display_metadata_137w(df_obj, row_idx, start_idx, *, purpose="gui_row"):
    """Strict resolver: no unsafe cache, no metadata-free df.iloc fallback."""
    try:
        candidate = None
        try:
            candidate = _safe_gui_row_from_df_137r(df_obj, row_idx, purpose=purpose)
        except Exception:
            candidate = None
        if candidate is not None and _row_start_idx_in_bounds_137v(candidate, start_idx) and _row_matches_current_display_metadata_137w(candidate):
            try:
                _gui_highlight_runtime_diag_137u(f"{purpose}.resolved_137w_direct", df_obj=df_obj, row_idx=row_idx, start_idx=start_idx, row_data=candidate)
            except Exception:
                pass
            return candidate

        try:
            n = len(df_obj)
        except Exception:
            n = 0
        max_scan = min(int(n or 0), 20000)
        for pos in range(max_scan):
            row = _row_for_position(df_obj, pos)
            if row is None:
                continue
            if not _row_start_idx_in_bounds_137v(row, start_idx):
                continue
            if not _row_matches_current_display_metadata_137w(row):
                continue
            try:
                _gui_highlight_runtime_diag_137u(f"{purpose}.resolved_137w_metadata_scan", df_obj=df_obj, row_idx=pos, requested_row_idx=row_idx, start_idx=start_idx, row_data=row)
            except Exception:
                pass
            return row

        try:
            import logging as _logging_137w
            _logging_137w.warning(
                "[APP gui.display_row.resolve.strict] purpose=%r row_idx=%r start_idx=%r title=%r date=%r author=%r unresolved_strict",
                purpose,
                row_idx,
                start_idx,
                globals().get("current_display_title_137v"),
                globals().get("current_display_publication_date_137v"),
                globals().get("current_display_author_137v"),
            )
        except Exception:
            pass
        return None
    except Exception as exc:
        try:
            import logging as _logging_137w
            _logging_137w.warning("[APP gui.display_row.resolve.strict] purpose=%r failed=%r", purpose, exc, exc_info=True)
        except Exception:
            pass
        return None
# END KORPUSUJ_PATCH_137W_GUI_DISPLAY_ROW_RESOLVER_STRICT_METADATA_AND_NO_BAD_CACHE


# KORPUSUJ_PATCH_137Y_GUI_COREF_HIGHLIGHT_SUPPRESS_OVERBROAD_CLUSTERS
COREF_SUPPRESS_DOC_RATIO_137Y = 0.80

def _coref_cluster_id_137y(label):
    try:
        s = str(label).strip()
        import re as _re_137y
        m = _re_137y.search(r"(?:Head|Part)[-_]?(\d+)$", s, _re_137y.I)
        if m:
            return m.group(1)
        m = _re_137y.search(r"(\d+)$", s)
        if m:
            return m.group(1)
        return s
    except Exception:
        return None

def _coref_labels_for_token_137y(value):
    """Normalize one token's coref cell to deduplicated non-empty labels."""
    out = []
    try:
        if value is None:
            return []
        if isinstance(value, str):
            seq = [value]
        else:
            try:
                seq = list(value)
            except Exception:
                seq = [value]
        seen = set()
        for raw in seq:
            try:
                label = str(raw).strip()
            except Exception:
                continue
            if not label or label in {"0", "O", "_", "None", "nan", "NaN", "[]"}:
                continue
            # Preserve one representative per exact label per token; fixes Part-1 duplicates.
            if label in seen:
                continue
            seen.add(label)
            out.append(label)
    except Exception:
        return []
    return out

def _suppressed_coref_clusters_for_row_137y(corefs, *, threshold=None):
    """Return set of cluster IDs too broad to be useful for GUI highlighting."""
    try:
        if threshold is None:
            threshold = globals().get("COREF_SUPPRESS_DOC_RATIO_137Y", 0.80)
        values = list(corefs) if corefs is not None else []
        total = len(values)
        if total <= 0:
            return set()
        counts = {}
        for token_value in values:
            labels = _coref_labels_for_token_137y(token_value)
            cids = set()
            for lab in labels:
                cid = _coref_cluster_id_137y(lab)
                if cid:
                    cids.add(cid)
            for cid in cids:
                counts[cid] = counts.get(cid, 0) + 1
        suppressed = set()
        for cid, count in counts.items():
            ratio = float(count) / float(total) if total else 0.0
            if ratio >= float(threshold):
                suppressed.add(cid)
                try:
                    import logging as _logging_137y
                    _logging_137y.warning(
                        "[APP gui.coref.suppress] cluster=%r ratio=%.4f count=%s total=%s threshold=%.2f skipped",
                        cid, ratio, count, total, float(threshold)
                    )
                except Exception:
                    pass
        return suppressed
    except Exception:
        return set()
# END KORPUSUJ_PATCH_137Y_GUI_COREF_HIGHLIGHT_SUPPRESS_OVERBROAD_CLUSTERS

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
    _gui_highlight_runtime_diag_137u(
        "show_dependency_graph.before_resolve",
        df_obj=df,
        row_idx=current_graph_row_idx,
        start_idx=current_graph_start_idx,
        corpus=global_selected_corpus,
    )
    row_data = _resolve_gui_row_by_display_metadata_137w(df, current_graph_row_idx, current_graph_start_idx, purpose="show_dependency_graph")
    if row_data is None:
        _gui_highlight_runtime_diag_137u(
            "show_dependency_graph.row_missing",
            df_obj=df,
            row_idx=current_graph_row_idx,
            start_idx=current_graph_start_idx,
            corpus=global_selected_corpus,
        )
        try:
            messagebox.showwarning("Brak danych wiersza", "Nie udało się pobrać danych dla wybranego wyniku.")
        except Exception:
            pass
        return
    _gui_highlight_runtime_diag_137u(
        "show_dependency_graph.after_resolve",
        df_obj=df,
        row_idx=current_graph_row_idx,
        start_idx=current_graph_start_idx,
        row_data=row_data,
        corpus=global_selected_corpus,
    )

    sentence_ids = row_data.sentence_ids
    try:
        _graph_start_idx_137v = int(current_graph_start_idx) if current_graph_start_idx is not None else None
        if _graph_start_idx_137v is None or _graph_start_idx_137v < 0 or _graph_start_idx_137v >= len(sentence_ids):
            try:
                _gui_highlight_runtime_diag_137u(
                    "show_dependency_graph.start_idx_oob_137v",
                    df_obj=df,
                    row_idx=current_graph_row_idx,
                    start_idx=current_graph_start_idx,
                    row_data=row_data,
                    sentence_ids_len=len(sentence_ids),
                    title=current_display_title_137v,
                )
            except Exception:
                pass
            try:
                messagebox.showwarning("Brak danych wiersza", "Indeks tokenu jest poza zakresem wybranego dokumentu.")
            except Exception:
                pass
            return
    except Exception:
        return
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
    writable_temp_root().mkdir(parents=True, exist_ok=True)
    temp_graph_path = str(writable_temp_root() / "temp_graph.png")

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

    # Mouse wheel binding (zabezpieczone przed martwym widgetem)
    def _on_mousewheel(event):
        if canvas_tk.winfo_exists():
            canvas_tk.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shiftmouse(event):
        if canvas_tk.winfo_exists():
            canvas_tk.xview_scroll(int(-1 * (event.delta / 120)), "units")

    # Używamy bind na konkretnym oknie, a nie bind_all globalnie
    graph_win.bind("<MouseWheel>", _on_mousewheel)
    graph_win.bind("<Shift-MouseWheel>", _on_shiftmouse)


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
    _gui_highlight_runtime_diag_137u(
        "update_highlights.before_resolve",
        df_obj=df,
        row_idx=current_display_row_idx,
        start_idx=current_display_start_idx,
        body_start_mark=current_body_start_mark,
        show_ner_active=show_ner_active,
        show_coref_active=show_coref_active,
        corpus=global_selected_corpus,
    )
    row_data = _resolve_gui_row_by_display_metadata_137w(df, current_display_row_idx, current_display_start_idx, purpose="update_highlights")
    if row_data is None:
        _gui_highlight_runtime_diag_137u(
            "update_highlights.row_missing",
            df_obj=df,
            row_idx=current_display_row_idx,
            start_idx=current_display_start_idx,
            body_start_mark=current_body_start_mark,
            corpus=global_selected_corpus,
        )
        return
    _gui_highlight_runtime_diag_137u(
        "update_highlights.after_resolve",
        df_obj=df,
        row_idx=current_display_row_idx,
        start_idx=current_display_start_idx,
        row_data=row_data,
        body_start_mark=current_body_start_mark,
        corpus=global_selected_corpus,
    )
    start_ids = row_data.start_ids
    try:
        _start_idx_137v = int(current_display_start_idx) if current_display_start_idx is not None else None
        if _start_idx_137v is None or _start_idx_137v < 0 or _start_idx_137v >= len(start_ids):
            try:
                _gui_highlight_runtime_diag_137u(
                    "update_highlights.start_idx_oob_137v",
                    df_obj=df,
                    row_idx=current_display_row_idx,
                    start_idx=current_display_start_idx,
                    row_data=row_data,
                    start_ids_len=len(start_ids),
                    title=current_display_title_137v,
                )
            except Exception:
                pass
            return
    except Exception:
        return
    end_ids = row_data.end_ids
    ners = row_data.ners
    corefs = getattr(row_data, "corefs", None)
    suppressed_coref_clusters_137y = _suppressed_coref_clusters_for_row_137y(corefs) if corefs is not None else set()

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
            c_tags = _coref_labels_for_token_137y(corefs[k])

            for c_tag in c_tags:
                if c_tag not in ("0", "O", "_", None):
                    cluster_id = _coref_cluster_id_137y(c_tag)
                    if not cluster_id or cluster_id in suppressed_coref_clusters_137y:
                        continue
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
    current_display_publication_date_137v = publication_date
    current_display_title_137v = title
    current_display_author_137v = author
    try:
        current_display_text_sample_137v = (full_text[0] if full_text else "")[:500]
    except Exception:
        current_display_text_sample_137v = None
    # 137w: also force module-global storage; nested/global patch ordering can be fragile.
    try:
        globals()["current_display_publication_date_137v"] = publication_date
        globals()["current_display_title_137v"] = title
        globals()["current_display_author_137v"] = author
        globals()["current_display_text_sample_137v"] = current_display_text_sample_137v
    except Exception:
        pass

    _gui_highlight_runtime_diag_137u(
        "display_full_text.state_set",
        row_idx=row_idx,
        start_idx=start_idx,
        publication_date=publication_date,
        title=title,
        author=author,
        result_sample=result,
        full_text_sample=full_text,
    )

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
    _gui_highlight_runtime_diag_137u(
        "display_full_text.body_start_mark",
        row_idx=current_display_row_idx,
        start_idx=current_display_start_idx,
        body_start_mark=current_body_start_mark,
    )
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
    keywords = [
        "orth=", "orth!=",
        "window_base=", "window_base!=",
        "window_orth=", "window_orth!=",
        "base=", "base!=",
        "pos=", "pos!=",
        "upos=", "upos!=",
        "ner=", "ner!=",
        "head=", "head!=",
        "coref=", "coref!=",
        "dependent=", "dependent!=",
        "deprel=", "deprel!=",
        "number=", "number!=",
        "gender=", "gender!=",
        "degree=", "degree!=",
        "case=", "case!=",
        "person=", "person!=",
        "accentability=", "accentability!=",
        "post-prepositionality=", "post-prepositionality!=",
        "accommodability=", "accommodability!=",
        "aspect=", "aspect!=",
        "vocalicity=", "vocalicity!=",
        "agglutination=", "agglutination!=",
        "negation=", "negation!=",
        "||",
        "data>",
        "data<",
        "data=",
        "data!=",
        "data<=",
        "data>=",
        "autor=", "autor!=",
        "metadane:",
        "tytuł=", "tytuł!=",
        "children.group=",
        "frequency_base",
        "frequency_orth",
        "top=", "min=", "max=",
        "<s>"
    ]

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


# Drzewiasta struktura depreli (Pełna specyfikacja Polish UD)
DEPREL_TREE_DICT = {
    "Wszystkie": [],
    "root - głowa drzewa": [],

    "nsubj - podmiot nominalny": [
        "nsubj:pass - podmiot nominalny (strona bierna)"
    ],
    "csubj - podmiot zdaniowy": [
        "csubj:pass - podmiot zdaniowy (strona bierna)"
    ],

    "obj - argument syntetyczny (Acc / Gen)": [],
    "iobj - argument syntetyczny (Dat / Ins)": [],

    "ccomp - argument zdaniowy": [
        "ccomp:obj - argument zdaniowy czasownika",
        "ccomp:cleft - zdanie podrzędne zależne od zaimka 'to'"
    ],
    "xcomp - argument zdaniowy / bezokolicznikowy": [
        "xcomp:pred - argument orzecznikowy (dla czasowników innych niż cop)",
        "xcomp:obj - argument bezokolicznikowy (dopełnienie)",
        "xcomp:subj - argument bezokolicznikowy (podmiotowy)",
        "xcomp:cleft - argument bezokolicznikowy zależny od zaimka 'to'"
    ],

    "obl - modyfikator analityczny (okolicznik/dopełnienie)": [
        "obl:arg - argument przyimkowy czasownika",
        "obl:agent - sprawca w stronie biernej",
        "obl:cmpr - fraza porównawcza",
        "obl:orphan - argument z elipsą rzeczownika"
    ],
    "advmod - modyfikator przysłówkowy": [
        "advmod:arg - argument przysłówkowy czasownika",
        "advmod:emph - partykuła wzmacniająca / intensyfikator",
        "advmod:neg - partykuła przecząca"
    ],
    "advcl - modyfikator zdaniowy (zdanie okolicznikowe)": [
        "advcl:relcl - zdanie względne określające inne zdanie",
        "advcl:cmpr - zdanie okolicznikowe porównawcze"
    ],

    "amod - modyfikator przymiotnikowy": [
        "amod:flat - człon przymiotnikowy nazwy własnej"
    ],
    "nmod - modyfikator rzeczowny / przyimkowy": [
        "nmod:arg - argument rzeczowny",
        "nmod:poss - modyfikator dzierżawczy (np. zaimki)",
        "nmod:flat - nominalny człon nazwy własnej",
        "nmod:pred - wyrażenie orzecznikowe zależne od imiesłowu (bycia)"
    ],
    "nummod - modyfikator liczebnikowy": [
        "nummod:gov - liczebnik rządzący przypadkiem rzeczownika",
        "nummod:flat - liczebnikowy człon nazwy własnej"
    ],
    "det - określnik": [
        "det:nummod - zaimki ilościowe uzgadniające przypadek",
        "det:numgov - zaimki ilościowe rządzące przypadkiem"
    ],
    "acl - zdanie przydawkowe": [
        "acl:relcl - zdanie przydawkowe względne"
    ],

    "aux - czasownik posiłkowy": [
        "aux:pass - czasownik posiłkowy (strona bierna)",
        "aux:cnd - czasownik posiłkowy (tryb przypuszczający)",
        "aux:imp - czasownik posiłkowy (tryb rozkazujący)",
        "aux:clitic - aglutynacyjny formant ruchomy (np. -śmy)"
    ],
    "cop - łącznik": [
        "cop:locat - łącznik w funkcji lokatywnej"
    ],
    "case - wskaźnik przypadka / przyimek": [],
    "mark - wskaźnik zespolenia (spójnik podrzędny)": [],

    "cc - spójnik współrzędny": [
        "cc:preconj - spójnik wprowadzający (np. 'zarówno')"
    ],
    "conj - połączenie współrzędne / szereg": [],

    "expl - zaimek zwrotny / egzpletywny": [
        "expl:pv - właściwy zaimek zwrotny 'się'",
        "expl:impers - bezosobowe użycie 'się'"
    ],
    "discourse - element dyskursu": [
        "discourse:intj - wykrzyknik",
        "discourse:emo - emotikon / emoji"
    ],
    "parataxis - parataksa / wtrącenie": [
        "parataxis:insert - wtrącenie / komentarz",
        "parataxis:obj - mowa niezależna"
    ],
    "flat - struktura płaska": [
        "flat:foreign - słowo obcojęzyczne"
    ]
}


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


# ==========================================
# QUERY BUILDER UI
# ==========================================
from korpusuj.ui.query_builder import QueryBuilderWindow


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








# --- Klasa do rozwijanych paneli opcji (Akordeon) ---
settings_cards = []


# ==========================================
# UI CARDS
# ==========================================
from korpusuj.ui.cards import SettingsCard



def export_subcorpus_by_metadata():
    global dataframes, corpus_options

    if not dataframes:
        messagebox.showinfo("Brak danych", "Najpierw załaduj korpus bazowy.")
        return

    # Okienko konfiguracji
    win = ctk.CTkToplevel(app)
    win.title("Utwórz podkorpus z metadanych")
    win.geometry("450x450")
    win.transient(app)
    win.grab_set()

    theme = THEMES[motyw.get()]
    win.configure(fg_color=theme["app_bg"])

    frame = ctk.CTkFrame(win, fg_color=theme["subframe_fg"], corner_radius=12)
    frame.pack(fill="both", expand=True, padx=15, pady=15)

    ctk.CTkLabel(frame, text="Korpus bazowy:", font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(
        pady=(10, 0))
    combo_corpus = ctk.CTkOptionMenu(frame, values=corpus_options, fg_color=theme["button_fg"],
                                     text_color=theme["button_text"])
    combo_corpus.pack(pady=(0, 10))

    ctk.CTkLabel(frame, text="Data od (np. 2024-01-01):", text_color=theme["label_text"]).pack()
    entry_dstart = ctk.CTkEntry(frame, fg_color=theme["frame_fg"])
    entry_dstart.pack(pady=(0, 10))

    ctk.CTkLabel(frame, text="Data do (np. 2024-12-31):", text_color=theme["label_text"]).pack()
    entry_dend = ctk.CTkEntry(frame, fg_color=theme["frame_fg"])
    entry_dend.pack(pady=(0, 10))

    ctk.CTkLabel(frame, text="Autor (zawiera):", text_color=theme["label_text"]).pack()
    entry_author = ctk.CTkEntry(frame, fg_color=theme["frame_fg"])
    entry_author.pack(pady=(0, 10))

    ctk.CTkLabel(frame, text="Tytuł (zawiera):", text_color=theme["label_text"]).pack()
    entry_title = ctk.CTkEntry(frame, fg_color=theme["frame_fg"])
    entry_title.pack(pady=(0, 15))

    def on_generate():
        from korpusuj.export.subcorpus import filter_dataframe_by_metadata, export_dataframe_to_subcorpus_parquet

        base_corp = combo_corpus.get()
        d_start = entry_dstart.get().strip()
        d_end = entry_dend.get().strip()
        author = entry_author.get().strip()
        title = entry_title.get().strip()

        df = dataframes[base_corp]
        sub_df = filter_dataframe_by_metadata(
            df,
            date_from=d_start,
            date_to=d_end,
            author=author,
            title=title,
        )

        if sub_df.empty:
            messagebox.showwarning("Brak wyników", "Żadne teksty nie spełniają podanych kryteriów.")
            return

        corpus_dir = BASE_DIR_CORP if 'BASE_DIR_CORP' in globals() else os.path.expanduser("~")
        file_path = filedialog.asksaveasfilename(
            title="Zapisz podkorpus jako",
            defaultextension=".parquet",
            filetypes=[("Pliki Parquet", "*.parquet")],
            initialdir=corpus_dir
        )

        if not file_path:
            return

        win.destroy()

        # Ekran ładowania (jak w poprzedniej funkcji)
        loading_win = ctk.CTkToplevel(app)
        loading_win.title("Tworzenie podkorpusu")
        loading_win.geometry(f"350x120+{app.winfo_x() + 100}+{app.winfo_y() + 100}")
        loading_win.transient(app)
        loading_win.grab_set()
        ctk.CTkLabel(loading_win, text=f"Przeliczanie {len(sub_df)} tekstów...\nProszę czekać.",
                     font=("Verdana", 12)).pack(expand=True)
        loading_win.update()

        def worker():
            try:
                export_dataframe_to_subcorpus_parquet(sub_df, file_path)

                def update_ui():
                    loading_win.destroy()
                    messagebox.showinfo("Sukces", f"Zapisano podkorpus z {len(sub_df)} dokumentami.")

                app.after(0, update_ui)

            except Exception as e:
                logging.exception("Błąd tworzenia podkorpusu")
                app.after(0, lambda: loading_win.destroy())
                app.after(0,
                          lambda msg=str(e): messagebox.showerror("Błąd", f"Nie udało się utworzyć podkorpusu.\n{msg}"))

        threading.Thread(target=worker, daemon=True).start()

    ctk.CTkButton(frame, text="Generuj", font=("Verdana", 12, "bold"), fg_color=theme["button_fg"],
                  text_color=theme["button_text"], hover_color=theme["button_hover"], command=on_generate).pack(pady=10)

def export_to_subcorpus():
    global full_results_sorted, dataframes, global_selected_corpus, corpus_options, files, inverted_indexes

    if not full_results_sorted:
        messagebox.showinfo("Brak wyników", "Najpierw wyszukaj frazę, aby utworzyć podkorpus na bazie tych wyników.")
        return

    # Pobierz domyślny folder z korpusami
    corpus_dir = BASE_DIR_CORP if 'BASE_DIR_CORP' in globals() else os.path.expanduser("~")

    file_path = filedialog.asksaveasfilename(
        title="Zapisz podkorpus jako",
        defaultextension=".parquet",
        filetypes=[("Pliki Parquet", "*.parquet")],
        initialdir=corpus_dir
    )

    if not file_path:
        return

    # Tworzenie ekranu ładowania
    loading_win = ctk.CTkToplevel(app)
    loading_win.title("Tworzenie podkorpusu")

    # Wyśrodkowanie okienka
    app.update_idletasks()
    x = app.winfo_x() + (app.winfo_width() // 2) - 175
    y = app.winfo_y() + (app.winfo_height() // 2) - 60
    loading_win.geometry(f"350x120+{x}+{y}")
    loading_win.transient(app)
    loading_win.grab_set()

    ctk.CTkLabel(loading_win, text="Generowanie pliku Parquet...\nPrzeliczanie metadanych, proszę czekać.",
                 font=("Verdana", 12)).pack(expand=True)
    loading_win.update()

    def worker():
        try:
            from korpusuj.export.subcorpus import select_rows_from_search_results, export_dataframe_to_subcorpus_parquet

            df = dataframes[global_selected_corpus]
            sub_df = select_rows_from_search_results(df, full_results_sorted)

            if sub_df.empty:
                raise ValueError("Brak poprawnych wierszy wyników do eksportu podkorpusu.")

            export_dataframe_to_subcorpus_parquet(sub_df, file_path)

            def update_ui():
                loading_win.destroy()
                corpus_name = os.path.basename(file_path).replace(".parquet", "")
                messagebox.showinfo("Sukces", f"Zapisano podkorpus:\n{corpus_name}")

            app.after(0, update_ui)

        except Exception as e:
            logging.exception("Błąd tworzenia podkorpusu")
            app.after(0, lambda: loading_win.destroy())
            app.after(0, lambda msg=str(e): messagebox.showerror("Błąd", f"Nie udało się utworzyć podkorpusu.\n{msg}"))

    threading.Thread(target=worker, daemon=True).start()


def export_data():
    try:
        from korpusuj.export.excel import (
            build_search_results_export_df,
            build_table_export_df,
            build_profile_export_df,
            write_excel_workbook,
            write_csv_export,
            LEMMA_FREQUENCY_HEADERS,
            TOKEN_FREQUENCY_HEADERS,
            MONTH_FREQUENCY_HEADERS,
            COLLOCATION_HEADERS,
        )

        all_columns = [
            "Data publikacji", "context", "full_text_with_markers",
            "Rezultat", "matched_lemmas",
            "month_key", "Tytuł", "Autor", "additional_metadata",
            "Lewy kontekst", "Prawy kontekst", "row_index", "start_idx", "end_idx"
        ]

        df_export_slice = build_search_results_export_df(
            _resolve_lazy_fulltext_rows_for_export_111(full_results_sorted),
            all_columns=all_columns,
        )

        # Use safe default directory
        initial_dir = BASE_DIR if 'BASE_DIR' in globals() else os.path.expanduser("~")

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            initialdir=initial_dir
        )

        if not file_path:
            return  # user cancelled

        # Ensure folder exists. dirname can be empty if user gives a bare filename.
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if file_path.lower().endswith(".xlsx"):
            sheets = [("Wyniki wyszukiwania", df_export_slice)]

            if 'data' in paginator_fq and paginator_fq['data']:
                sheets.append((
                    "Częstość lematów",
                    build_table_export_df(paginator_fq['data'], LEMMA_FREQUENCY_HEADERS),
                ))

            if 'data' in paginator_token and paginator_token['data']:
                sheets.append((
                    "Częstość tokenów",
                    build_table_export_df(paginator_token['data'], TOKEN_FREQUENCY_HEADERS),
                ))

            if 'data' in paginator_month and paginator_month['data']:
                sheets.append((
                    "Częstość w czasie",
                    build_table_export_df(paginator_month['data'], MONTH_FREQUENCY_HEADERS),
                ))

            if 'paginator_colloc' in globals() and 'data' in paginator_colloc and paginator_colloc['data']:
                sheets.append((
                    "Kolokacje",
                    build_table_export_df(paginator_colloc['data'], COLLOCATION_HEADERS),
                ))

            if 'current_profile_dict' in globals() and current_profile_dict:
                sheets.append((
                    "Profil kolokacyjny",
                    build_profile_export_df(current_profile_dict),
                ))

            write_excel_workbook(file_path, sheets)

        else:
            # fallback CSV export: only the main search results table, like legacy export_data().
            write_csv_export(file_path, df_export_slice)

    except Exception as e:
        messagebox.showerror("Błąd eksportu",
                             f"Nie udało się zapisać pliku. Upewnij się, że plik nie jest otwarty w innym programie.\n\nSzczegóły: {e}")


def show_loading_screen():
    loading_win = ctk.CTkToplevel(app)
    loading_win.title("Ładowanie korpusu...")
    width, height = 640, 220
    app.update_idletasks()
    app_width = app.winfo_width(); app_height = app.winfo_height()
    app_x = app.winfo_x(); app_y = app.winfo_y()
    x = app_x + (app_width // 2) - (width // 2)
    y = app_y + (app_height // 2) - (height // 2)
    loading_win.geometry(f"{width}x{height}+{x}+{y}")
    loading_win.transient(app); loading_win.lift(); loading_win.attributes("-topmost", True)
    loading_win.protocol("WM_DELETE_WINDOW", lambda: None)
    frame = ctk.CTkFrame(loading_win, fg_color="transparent")
    frame.pack(fill="both", expand=True, padx=26, pady=22)
    ctk.CTkLabel(frame, text="Przygotowanie korpusu", font=("Verdana", 15, "bold")).pack(anchor="w", pady=(0, 10))
    loading_label = ctk.CTkLabel(frame, text="Proszę czekać, trwa przygotowywanie danych...", justify="left", anchor="w", wraplength=580, font=("Verdana", 12))
    loading_label.pack(fill="x", pady=(0, 14))
    progress_bar = ctk.CTkProgressBar(frame, mode="indeterminate", height=10)
    progress_bar.pack(fill="x", pady=(0, 12)); progress_bar.start()
    ctk.CTkLabel(frame, text="Proszę czekać. Przy większych korpusach przygotowanie danych może potrwać dłużej.", text_color="gray", justify="left", anchor="w", wraplength=580, font=("Verdana", 10)).pack(fill="x")
    loading_win.update_idletasks()
    return loading_win, loading_label, progress_bar

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
    global config
    data = {
        'font_family': font_family.get(),
        'fontsize': fontsize,
        'styl_wykresow': styl_wykresow.get(),
        'motyw': motyw.get(),
        'plotting': plotting.get(),
        'kontekst': kontekst,
        'min_tokens_threshold': min_tokens_threshold,
        'dependency_cache_ram_mode': _dependency_label_to_mode(dependency_ram_usage_var.get())
    }
    config.update(data)
    if data.get('dependency_cache_ram_mode') == 'none':
        _clear_dependency_ram_cache_for_corpus(None)
    _write_config_atomic(config)


# Settings window
def settings_window():
    global settings_popup, fontsize, font_family, plotting, kontekst, min_tokens_threshold, dependency_ram_usage_var
    theme = THEMES[motyw.get()]

    # Callbacks
    def restore_defaults():
        global settings_popup, fontsize, font_family, kontekst, plotting, min_tokens_threshold, dependency_ram_usage_var
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
        threshold_entry.delete(0, 'end')
        threshold_entry.insert(0, str(DEFAULT_SETTINGS['min_tokens_threshold']))
        min_tokens_threshold = DEFAULT_SETTINGS['min_tokens_threshold']
        dependency_ram_usage_var.set(_dependency_mode_to_label(DEFAULT_SETTINGS.get('dependency_cache_ram_mode', DEFAULT_DEPENDENCY_RAM_MODE)))
        apply_theme()
        save_config()
        settings_popup.destroy()
        settings_popup = None

    def on_save():
        global settings_popup, fontsize, font_family, kontekst, min_tokens_threshold, dependency_ram_usage_var
        try:
            fontsize = int(fontsize_entry.get())
        except ValueError:
            fontsize = DEFAULT_SETTINGS['fontsize']
        try:
            kontekst = int(kontekst_entry.get())
        except ValueError:
            kontekst = DEFAULT_SETTINGS['kontekst']

        try:
            min_tokens_threshold = int(threshold_entry.get())
        except ValueError:
            min_tokens_threshold = DEFAULT_SETTINGS['min_tokens_threshold']
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
    settings_popup.geometry('420x820')
    settings_popup.grab_set()
    settings_popup.configure(fg_color=theme["app_bg"])  # use theme

    # Frame for all settings
    settings_frame = ctk.CTkScrollableFrame(settings_popup, fg_color=theme["subframe_fg"], corner_radius=15)
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

    # RAM usage for dependency cache
    ctk.CTkLabel(settings_frame, text='Tryb zużycia RAM:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    ctk.CTkComboBox(settings_frame, values=['Oszczędny', 'Maksymalna wydajność'], variable=dependency_ram_usage_var,
                     fg_color=theme["button_fg"], dropdown_fg_color=theme["dropdown_fg"],
                     dropdown_hover_color=theme["dropdown_hover"], text_color=theme["button_text"],
                     font=("Verdana", 12), height=entry_height).pack(pady=5)

    # Context
    ctk.CTkLabel(settings_frame, text='Liczba tokenów w rozszerzonym kontekście:', font=("Verdana", 12, "bold"), text_color=theme["label_text"]).pack(pady=(10, 5))
    kontekst_entry = ctk.CTkEntry(settings_frame, width=150, height=entry_height, font=("Verdana", 12),
                                  fg_color=theme["frame_fg"], corner_radius=8)
    kontekst_entry.insert(0, str(kontekst))
    kontekst_entry.pack(pady=5)

    ctk.CTkLabel(settings_frame, text='Minimalny próg tokenów (koszyk wykresu):', font=("Verdana", 12, "bold"),
                 text_color=theme["label_text"]).pack(pady=(10, 5))
    threshold_entry = ctk.CTkEntry(settings_frame, width=150, height=entry_height, font=("Verdana", 12),
                                   fg_color=theme["frame_fg"], corner_radius=8)
    threshold_entry.insert(0, str(min_tokens_threshold))
    threshold_entry.pack(pady=5)

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

            file_path = flashcards_root() / f"{file_name}.txt"
            with file_path.open('a', encoding='utf-8') as file:
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
    folder_path = flashcards_root()
    return [path.stem for path in folder_path.iterdir() if path.is_file() and path.suffix == ".txt"]

webview_thread = None  # Global variable to track the thread


webview_process = None

def open_webview_window(file_name: str):
    import os
    import sys
    import subprocess
    from pathlib import Path

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    # Dla plików pomocy / instrukcji:
    # przekazujemy nazwę lub ścieżkę tak, żeby blok --run-webview sam ją poprawnie rozwiązał.
    safe_target = str(file_name).strip()

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--run-webview", safe_target]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), "--run-webview", safe_target]

    logging.info(f"open_webview_window -> {safe_target}")
    subprocess.Popen(cmd, creationflags=creationflags)



webview_thread = None

fiszki_process = None

def fiszki_load_file_content(value):
    """Uruchamia load_file_content. Na macOS jako osobny proces, na Windows w wątku."""
    global webview_thread, fiszki_process

    if sys.platform == "darwin":
        # MACOS: Odpalamy proces używając naszego nowego routera
        if fiszki_process is not None and fiszki_process.poll() is None:
            print("Fiszki już działają.")
            return

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--run-fiszki", str(value)]
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "--run-fiszki", str(value)]

        try:
            fiszki_process = subprocess.Popen(cmd)
        except Exception as e:
            logging.error(f"Nie udało się uruchomić fiszek na macOS: {e}")
    else:
        # WINDOWS / LINUX: Tutaj zostaw swój dotychczasowy kod oparty o threading.Thread
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
            _set_loading_status(loading_label, f"Przygotowanie indeksu {name} ({i}/{len(files)})...")

            cfg = globals().get("config", {}) or {}
            indexed_attrs = get_search_indexed_attrs()
            index_profile_label = cfg.get("index_profile", os.environ.get("KORPUSUJ_INDEX_PROFILE", "full"))
            try:
                batch_docs = int(cfg.get("index_batch_docs", 5000) or 5000)
            except Exception:
                batch_docs = 5000

            progress_callback = (
                lambda msg, n=name: _set_loading_status(loading_label, f"{n}: {msg}")
            ) if loading_label else None

            _set_loading_status(
                loading_label,
                f"Przygotowywanie indeksu wyszukiwania {name}...\n"
                f"Profil: {index_profile_label}; atrybuty: {', '.join(indexed_attrs)}"
            )

            bundle = prepare_loaded_corpus_bundle(
                name,
                path,
                indexed_attrs=indexed_attrs,
                batch_docs=batch_docs,
                progress_callback=progress_callback,
            )
            dataframes[name] = bundle.dataframe
            inverted_indexes[name] = bundle.inverted_index

            print(f"{name}: gotowy indeks SQLite {bundle.search_path} ({bundle.total_docs} dokumentów)")

            # Dependency warmup zostaje w engine.py jako orkiestracja UI/cache/threading.
            if _cfg_bool("dependency_cache_warmup", True):
                warmup_build_maps = _cfg_bool("dependency_cache_warmup_build_maps", True)
                warmup_materialize = _cfg_bool("dependency_cache_warmup_materialize", False)
                _set_loading_status(
                    loading_label,
                    f"Przygotowywanie cache zależności {name}..."
                )
                dep_progress = (
                    lambda msg, n=name: _set_loading_status(loading_label, f"{n}: {msg}")
                ) if loading_label else None
                warm_dependency_cache_for_corpus(
                    name,
                    build_maps=warmup_build_maps,
                    materialize=warmup_materialize,
                    progress_callback=dep_progress,
                )
        except Exception as e:
            logging.error("Błąd przygotowania korpusu %s: %s", name, e, exc_info=True)
            try:
                messagebox.showerror(
                    "Błąd ładowania korpusu",
                    f"Nie udało się przygotować korpusu {name}.\n\nSzczegóły: {e}",
                )
            except Exception:
                pass



# --- KORPUSUJ_FIX_024_SHOW_CORPUS_INFO_LAZYTERMINDEX ---






# --- END KORPUSUJ_FIX_024_SHOW_CORPUS_INFO_LAZYTERMINDEX ---


# --- KORPUSUJ_FIX_025B_SHOW_CORPUS_INFO_MONTHLY_COUNTS_LINEPATCH ---

# --- END KORPUSUJ_FIX_025B_SHOW_CORPUS_INFO_MONTHLY_COUNTS_LINEPATCH ---
def show_corpus_info():
    global dataframes, inverted_indexes, corpus_var

    selected = corpus_var.get()
    if not selected or selected not in dataframes:
        messagebox.showinfo("Brak danych", "Najpierw załaduj lub wybierz korpus.")
        return

    df = dataframes[selected]
    inv_idx = inverted_indexes[selected]

    # --- ZBIERANIE STATYSTYK ---
    info_model = build_corpus_info_model(df, inv_idx)
    total_docs = info_model.total_docs
    total_tokens = info_model.total_tokens
    unique_lemmas = info_model.unique_lemmas
    unique_orths = info_model.unique_orths
    date_range = info_model.date_range
    monthly_stats_str = info_model.monthly_stats_str
    meta_str = info_model.meta_str

    # --- TWORZENIE OKIENKA UI ---
    info_win = ctk.CTkToplevel(app)
    info_win.title(f"Informacje o korpusie: {selected}")
    info_win.geometry("500x550")
    info_win.transient(app)
    info_win.grab_set()

    theme = THEMES[motyw.get()]
    info_win.configure(fg_color=theme["app_bg"])

    frame = ctk.CTkFrame(info_win, fg_color=theme["subframe_fg"], corner_radius=12)
    frame.pack(fill="both", expand=True, padx=15, pady=15)

    info_text = (
        f"ROZMIAR KORPUSU:\n"
        f"  • Liczba tekstów: {total_docs:,}\n"
        f"  • Całkowita liczba tokenów: {total_tokens:,}\n"
        f"  • Unikalne lematy: {unique_lemmas:,}\n"
        f"  • Unikalne formy ortograficzne: {unique_orths:,}\n\n"
        f"ZAKRES CZASOWY:\n"
        f"  • {date_range}\n\n"
        f"DOSTĘPNE METADANE:\n  • {meta_str}\n\n"
        f"{monthly_stats_str}"
    ).replace(',', ' ')

    textbox = ctk.CTkTextbox(
        frame,
        font=("Verdana", 13),
        text_color=theme["label_text"],
        fg_color="transparent",
        wrap="word"
    )
    textbox.insert("1.0", info_text)
    textbox.configure(state="disabled")
    textbox.pack(padx=10, pady=10, fill="both", expand=True)

    btn_close = ctk.CTkButton(
        frame, text="Zamknij", command=info_win.destroy,
        font=("Verdana", 12, "bold"), fg_color=theme["button_fg"],
        text_color=theme["button_text"], hover_color=theme["button_hover"]
    )
    btn_close.pack(pady=10)


def load_corpora():
    global files, corpus_options, corpus_var, dataframes
    corpus_dir = os.path.join(BASE_DIR_CORP)
    file_paths = filedialog.askopenfilenames(
        title="Wybierz plik(i) korpusu",
        initialdir=corpus_dir if os.path.exists(corpus_dir) else os.path.expanduser("~"),
        filetypes=[("Parquet files", "*.parquet")]
    )
    if not file_paths:
        print("Nie wybrano plików.")
        return
    loading_screen, loading_label, loading_progress = show_loading_screen()
    corpus_options = [os.path.basename(f).replace(".parquet", "") for f in file_paths]
    corpus_var.set(corpus_options[0])
    files = {name: path for name, path in zip(corpus_options, file_paths)}
    def worker():
        try:
            load_data_on_startup(loading_label=loading_label)
            def finish_ok():
                try:
                    option_corpus.configure(values=corpus_options, variable=corpus_var)
                    load_semantic_neighbors()
                    _set_loading_status(loading_label, "Korpusy załadowane. Zamykam okno...")
                finally:
                    try: loading_progress.stop()
                    except Exception: pass
                    loading_screen.destroy()
                    print("Korpusy załadowane.")
            app.after(0, finish_ok)
        except Exception as e:
            logging.error("Błąd ładowania korpusów w tle: %s", e, exc_info=True)
            def finish_error(err=e):
                try: loading_progress.stop()
                except Exception: pass
                try: loading_screen.destroy()
                except Exception: pass
                messagebox.showerror("Błąd ładowania korpusów", f"Nie udało się załadować korpusów.\n\nSzczegóły: {err}")
            app.after(0, finish_error)
    threading.Thread(target=worker, daemon=True).start()

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
        "text_colors_colloc": ["#FFFFFF", "#65A46F", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"],
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
        "text_colors_colloc": ["black", "#000DFF", "black", "black", "black", "black", "black", "black", "black"],
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
    if 'profile_frame' in globals(): profile_frame.configure(fg_color=theme["frame_fg"])
    tabview.configure(fg_color=theme["frame_fg"])

    # Subframes (Widoczne, zaokrąglone kafelki z zawartością)
    for frame in [
        pagination_frame, entry_button_frame, pagination_lemma_frame, pagination_orth_frame,
        pagination_month_frame, pagination_colloc_frame,
        pagination_profile_frame
    ]:
        frame.configure(fg_color=theme["subframe_fg"], border_color=theme["subframe_fg"])

    # Nowe kontenery opcji bocznych stają się tłem
    plot_options_frame.configure(fg_color=theme["frame_fg"])
    colloc_options_frame.configure(fg_color=theme["frame_fg"])
    profile_options_frame.configure(fg_color=theme["frame_fg"])

    # Zaktualizuj wszystkie nowo stworzone Karty ustawień na Wykresach
    for card in settings_cards:
        card.update_theme(theme)

    # Kontenery strukturalne (MUSZĄ być przezroczyste, by było widać między nimi tło okna)
    for frame in [left_pane, right_pane, right_subframe, buttons_action_frame]:
        frame.configure(fg_color="transparent")

    # --- Zmiana motywu dynamicznych kontrolek (Obejście dla anonimowych etykiet i menu) ---
    def update_frame_children(parent_frame):
        for child in parent_frame.winfo_children():
            if isinstance(child, ctk.CTkLabel):
                if child.cget("text") == "❓":
                    continue
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

    update_frame_children(colloc_options_frame)
    update_frame_children(date_settings_frame)

    # --- Buttons ---
    for button in [
        button_search, settings_button, button_first, button_prev, button_next, button_last,
        button_first_lemma, button_prev_lemma, button_next_lemma, button_last_lemma,
        button_first_orth, button_prev_orth, button_next_orth, button_last_orth,
        button_first_month, button_prev_month, button_next_month, button_last_month,
        button_save_plot, save_selection_button,
        button_first_colloc, button_prev_colloc, button_next_colloc, button_last_colloc, btn_calc_colloc,
        btn_refresh_plot,
        button_first_profile, button_prev_profile, button_next_profile, button_last_profile, btn_calc_profile,
        btn_nav_back, btn_nav_forward
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

    profile_table.set_header_font(font_tuple)
    profile_table.set_font(font_tuple)
    profile_table.set_text_colors(theme["text_colors_colloc"])
    profile_table.set_row_colors(*theme["row_colors"])
    profile_table.set_selected_row_color(theme["selected_row"])
    profile_table.set_canvas_background(theme["canvas_bg"])

    # Syntax highlighting
    highlight_color = theme["highlight"]
    highlight_keyword = theme["highlight_keyword"]
    question_marks_color = theme["question"]
    keywords_color = theme["keywords"]
    punctuation_color = theme["punctuation"]
    text_inside_quotation_color = theme["quotation"]

    register_text_widget(text_full)


def show_table(choice):
    """Show exactly one Statistics view without covering the selector row."""
    statistics_frames = {
        "Formy podstawowe (base)": lemma_frame,
        "Formy ortograficzne (orth)": orth_frame,
        "Częstość w czasie": month_frame,
        "Kolokacje": colloc_frame,
        "Profil kolokacyjny": profile_frame,
    }

    for frame in statistics_frames.values():
        frame.grid_remove()

    selected_frame = statistics_frames.get(choice)
    if selected_frame is not None:
        selected_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=10,
            pady=(0, 10),
        )
        selected_frame.tkraise()

    if choice == "Profil kolokacyjny" and full_results_sorted:
        result = full_results_sorted[0]
        start_index = result[12]
        end_index = result[13]
        match_length = end_index - start_index
        lemmas = str(result[4]).split()

        options = []
        for index in range(match_length):
            hint = lemmas[index] if index < len(lemmas) else "?"
            options.append(f"Token {index + 1} ({hint})")

        if not options:
            options = ["Token 1"]

        profile_node_menu.configure(values=options)
        if match_length > 1:
            profile_node_menu.configure(
                fg_color="#D9A04F",
                button_color="#D9A04F",
            )
        else:
            profile_node_menu.configure(
                fg_color="#4B6CB7",
                button_color="#4B6CB7",
            )

        if profile_node_var.get() not in options:
            profile_node_var.set(options[0])

    push_nav_state()

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

    # Read GUI options on the main thread; computation itself is delegated to
    # korpusuj.search.collocations in the worker.
    mode = colloc_mode_var.get()
    upos_filter = upos_var.get()
    pos_filter = pos_var.get()
    form_mode = colloc_form_var.get()
    ignore_case = colloc_ignore_case_var.get()
    use_sentence_bound = sentence_boundary_var.get()
    sort_mode = colloc_sort_var.get()
    active_feat_filters = {
        feat: var.get().split(" ")[0]
        for feat, var in dynamic_feat_vars.items()
        if var.get() != "Wszystkie"
    }

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

    from korpusuj.search.collocations import (
        CollocationOptions,
        compute_collocations,
        collocation_table_to_legacy_rows,
    )

    options = CollocationOptions(
        mode=mode,
        upos_filter=upos_filter,
        pos_filter=pos_filter,
        form_mode=form_mode,
        ignore_case=bool(ignore_case),
        use_sentence_bound=bool(use_sentence_bound),
        sort_mode=sort_mode,
        active_feat_filters=dict(active_feat_filters),
        min_freq=min_freq,
        min_range=min_range,
        l_span=l_span,
        r_span=r_span,
        syn_dir=syn_dir,
        deprel_filter=deprel_filter,
    )

    btn_calc_colloc.configure(state="disabled", text="Liczenie...")

    def worker():
        try:
            df = dataframes[global_selected_corpus]
            inv_idx_data = inverted_indexes[global_selected_corpus]
            table = compute_collocations(
                full_results_sorted,
                df,
                inv_idx_data,
                options,
                feat_mapping=FEAT_MAPPING,
            )
            colloc_stats = collocation_table_to_legacy_rows(table)

            def update_ui():
                paginator_colloc["data"] = colloc_stats
                paginator_colloc["current_page"][0] = 0
                update_table(paginator_colloc)
                btn_calc_colloc.configure(state="normal", text="Oblicz")

                with state_lock:
                    current_state.colloc_data = list(colloc_stats)

            app.after(0, update_ui)

        except Exception as e:
            logging.exception("Błąd kolokacji")
            error_msg = str(e)

            def on_error(msg=error_msg):
                btn_calc_colloc.configure(state="normal", text="Oblicz")
                messagebox.showerror("Błąd kolokacji", f"Nie udało się obliczyć kolokacji.\nSzczegóły: {msg}")

            app.after(0, on_error)

    threading.Thread(target=worker, daemon=True).start()


current_profile_target_lemma = ""  # Deklaracja na poziomie modułu


def search_from_table_profile(selected_word):
    """Ekskluzywna funkcja wyszukująca dla Profilu Składniowego, używająca odwróconych drzew zależności."""
    if not selected_word or not selected_word.strip(): return

    global current_profile_target_lemma
    if not current_profile_target_lemma:
        messagebox.showinfo("Błąd", "Brak danych o badanym lemacie. Wygeneruj Profil ponownie.")
        return

    active_rel_str = profile_rel_var.get()
    import re
    # 1. Odcinamy liczbę na końcu, np. "Okoliczniki przyimkowe 'z' (15)" -> "Okoliczniki przyimkowe 'z'"
    rel_name_with_marker = re.sub(r'\s*\(\d+\)$', '', active_rel_str).strip()

    from korpusuj.semantic.word_profile import PROFILE_GRAMMARS
    rule = None
    marker_val = ""

    # 2. Inteligentne dopasowanie nazwy relacji do reguły z uwzględnieniem szablonów (templates)
    for pos, categories in PROFILE_GRAMMARS.items():
        for cat_name, cat_rule in categories.items():
            template = cat_rule.get("relation_name_template")
            if template:
                # np. "Porównanie '{marker}'"
                regex_pattern = re.escape(template).replace(r"\{marker\}", r"(.*?)")
                m = re.match(f"^{regex_pattern}$", rel_name_with_marker)
                if m:
                    extracted_marker = m.group(1)
                    allowed_markers = cat_rule.get("capture_child_lemma_allow")

                    # ROZWIĄZANIE BŁĘDU: Upewniamy się, że marker pasuje do tej konkretnej reguły.
                    # Zapobiega to "kradzieży" markera 'niż' przez regułę dedykowaną dla 'od'.
                    if allowed_markers and extracted_marker not in allowed_markers:
                        continue

                    rule = cat_rule
                    marker_val = extracted_marker
                    break
            elif cat_rule.get("cascade_case"):
                prefix = f"{cat_name} '"
                if rel_name_with_marker.startswith(prefix) and rel_name_with_marker.endswith("'"):
                    rule = cat_rule
                    marker_val = rel_name_with_marker[len(prefix):-1]
                    break
            else:
                if rel_name_with_marker == cat_name:
                    rule = cat_rule
                    break
        if rule:
            break

    # --- NOWOŚĆ: Szukamy czystego lematu bezpośrednio w danych profilu ---
    main_colloc = ""

    # Przeszukujemy słownik, aby sparować kliknięty tekst z oryginalnym obiektem
    for rel_key, rows in current_profile_dict.items():
        for row_obj in rows:
            # Odtwarzamy tekst dokładnie w takiej formie, w jakiej wyświetla się w tabeli
            test_str = row_obj.display_collocate
            if getattr(row_obj, "collocate_upos", ""):
                test_str += f" [{row_obj.collocate_upos}]"

            # Wersja dla widoku zbiorczego "★ POKAŻ WSZYSTKIE" (z tagiem relacji na końcu)
            rel_match = re.search(r'\(([^)]+)\)', row_obj.relation)
            test_str_with_rel = f"{test_str} [{rel_match.group(1)}]" if rel_match else test_str

            if selected_word == test_str or selected_word == test_str_with_rel:
                # ZNALEZIONO! Bierzemy IDEALNIE CZYSTY lemat schowany pod spodem
                main_colloc = row_obj.collocate
                break
        if main_colloc:
            break

    # Fallback awaryjny (gdyby z jakiegoś powodu nie znalazło w słowniku)
    if not main_colloc:
        main_colloc_full = selected_word.split(" [")[0].strip()
        # Bierzemy OSTATNIE słowo, aby z "z wschód" wziąć "wschód", a nie "z"
        main_colloc = main_colloc_full.split()[-1]
        # ------------------------------------------------------------------------

    ignore_case = profile_ignore_case_var.get()

    def format_val(val):
        if ignore_case:
            w_lower = val.lower()
            w_upper = val.capitalize()
            return f"{w_lower}|{w_upper}" if w_lower != w_upper else w_lower
        return val

    q_target = format_val(current_profile_target_lemma)
    q_colloc = format_val(main_colloc)

    if not rule:
        # Ostateczny Fallback - liniowe szukanie, jeśli gramatyka jest nierozpoznana
        new_query = f'[base="{q_target}"] [*][0,5] [base="{q_colloc}"] || [base="{q_colloc}"] [*][0,5] [base="{q_target}"]'
    else:
        target_is = rule["target_is"]
        deprels = rule["deprels"]
        deprel_str = "|".join(deprels)
        req_case = rule.get("req_case", "")
        req_upos = rule.get("req_upos", "")

        # Główne warunki dla kolokatu
        main_conds = [f'base="{q_colloc}"']

        if req_case:
            # Tłumaczymy przypadek z formatu Universal Dependencies (z Profilu)
            # na format używany w tagsecie NKJP (wyszukiwarka)
            case_to_nkjp = {
                "Nom": "nom",
                "Gen": "gen",
                "Dat": "dat",
                "Acc": "acc",
                "Ins": "inst|ins",  # NKJP używa "inst", zabezpieczamy też "ins"
                "Loc": "loc",
                "Voc": "voc"
            }
            search_case = case_to_nkjp.get(req_case, req_case.lower())
            main_conds.append(f'case="{search_case}"')

        if req_upos:
            main_conds.append(f'upos="{req_upos}"')

        req_upos_in = rule.get("req_upos_in", [])
        if req_upos_in:
            upos_str = "|".join(req_upos_in)
            main_conds.append(f'upos="{upos_str}"')

        # Wymagania orzecznika (być) i jego polaryzacji (nie)
        if rule.get("requires_copula"):
            main_conds.append(f'dependent={{base="być|to" & deprel="cop"}}')
            polarity = rule.get("copula_polarity", "positive")
            if polarity == "negative":
                main_conds.append(f'dependent={{base="nie"}}')
            elif polarity == "positive":
                main_conds.append(f'dependent!={{base="nie"}}')

        # Marker wydobyty z nazwy (np. przyimek 'z', 'od', spójnik 'jak', 'niż')
        if marker_val:
            q_marker = format_val(marker_val)
            # Pobieramy poprawne relacje, domyślnie 'case'
            allowed_deps = rule.get("capture_child_lemma_from_deprels", ["case"])
            dep_str = "|".join(allowed_deps)
            main_conds.append(f'dependent={{base="{q_marker}" & deprel="{dep_str}"}}')

        # Uzupełnienie o inne wykluczenia i wymogi
        if "req_lemma" in rule:
            r_lem = "|".join([format_val(x) for x in rule["req_lemma"]])
            main_conds.append(f'base="{r_lem}"')

        if "exclude_lemma" in rule:
            e_lem = "|".join([format_val(x) for x in rule["exclude_lemma"]])
            main_conds.append(f'base!="{e_lem}"')

        if "requires_child_lemma" in rule:
            rc_lem = "|".join([format_val(x) for x in rule["requires_child_lemma"]])
            main_conds.append(f'dependent={{base="{rc_lem}"}}')

        if "requires_child_deprel" in rule:
            rc_dep = "|".join(rule["requires_child_deprel"])
            main_conds.append(f'dependent={{deprel="{rc_dep}"}}')

        if "exclude_child_lemma" in rule:
            ec_lem = "|".join([format_val(x) for x in rule["exclude_child_lemma"]])
            main_conds.append(f'dependent!={{base="{ec_lem}"}}')

        if "exclude_child_deprel" in rule:
            ec_dep = "|".join(rule["exclude_child_deprel"])
            main_conds.append(f'dependent!={{deprel="{ec_dep}"}}')

        # --- NOWOŚĆ: Tłumaczenie reguł nadrzędnika (head) z poprzednich kroków na zapytania ---
        if "req_head_upos" in rule:
            h_upos = "|".join(rule["req_head_upos"])
            main_conds.append(f'head={{upos="{h_upos}"}}')

        if rule.get("req_head_feature") == "Degree=Cmp":
            # Mapowanie stopnia wyższego z UD na wewnętrzne tagi NKJP, których używa Twoja wyszukiwarka
            main_conds.append(f'head={{degree="com|sup"}}')

        if "exclude_shared_head_child_deprel" in rule:
            bad_deps = "|".join(rule["exclude_shared_head_child_deprel"])
            main_conds.append(f'head={{dependent!={{deprel="{bad_deps}"}}}}')
        # --------------------------------------------------------------------------------------

        # Budowa zapytań docelowych na podstawie archtektury powiązań "target_is"
        if target_is == "head":
            main_conds.append(f'deprel="{deprel_str}"')
            main_conds.append(f'head={{base="{q_target}"}}')
            new_query = f"[{' & '.join(main_conds)}]"

        elif target_is == "child":
            main_conds.append(f'dependent={{base="{q_target}" & deprel="{deprel_str}"}}')
            new_query = f"[{' & '.join(main_conds)}]"

        elif target_is == "symmetric":
            conds1 = list(main_conds)
            conds1.append(f'deprel="{deprel_str}"')
            conds1.append(f'head={{base="{q_target}"}}')
            q1 = f"[{' & '.join(conds1)}]"

            conds2 = list(main_conds)
            conds2.append(f'dependent={{base="{q_target}" & deprel="{deprel_str}"}}')
            q2 = f"[{' & '.join(conds2)}]"

            new_query = f"{q1} || {q2}"

        elif target_is == "sibling":
            main_conds.append(f'deprel="{deprel_str}"')
            target_deps = rule.get("target_deprels", [])
            target_dep_str = "|".join(target_deps) if target_deps else ""

            if target_dep_str:
                main_conds.append(f'head={{dependent={{base="{q_target}" & deprel="{target_dep_str}"}}}}')
            else:
                main_conds.append(f'head={{dependent={{base="{q_target}"}}}}')

            new_query = f"[{' & '.join(main_conds)}]"

    search_diag_log("PROFILE_STRUCT_QUERY query=%r", new_query)
    tabview.set("Wyniki wyszukiwania")
    entry_query.delete("1.0", ctk.END)
    entry_query.insert("1.0", new_query)
    search()

def search_from_table(selected_word):
    if not selected_word or not selected_word.strip():
        return
    t_prep_start = time.perf_counter()

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
        # Tworzymy dokładne dopasowania oddzielone "|", co parser potraktuje jako "exact",
        # a nie "regex", ratując naszą optymalizację kotwicy!
        w_lower = selected_word.lower()
        w_upper = selected_word.capitalize()
        # Zabezpieczenie, żeby nie robić "powinien|powinien" jeśli słowo nie ma liter
        if w_lower != w_upper:
            query_val = f"{w_lower}|{w_upper}"
        else:
            query_val = w_lower
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
        t_prep_end = time.perf_counter()

        # Logujemy czas przygotowania
        prep_time = t_prep_end - t_prep_start
        search_diag_log("COLLOC_QUERY_BUILD time=%.6fs query=%r", prep_time, new_query)
        # Przełączenie zakładki, aktualizacja pola tekstowego i wymuszenie wyszukiwania
        tabview.set("Wyniki wyszukiwania")
        entry_query.delete("1.0", ctk.END)
        entry_query.insert("1.0", new_query)
        search()

def show_wsd_dialog():
    """Otwiera okno wyboru ram semantycznych/dyskursywnych dla aktualnego zapytania."""
    global current_wsd_lemma, unfiltered_wsd_results

    if not full_results_sorted:
        messagebox.showinfo("Brak wyników", "Najpierw wykonaj wyszukiwanie, aby móc analizować ramy.")
        return

    if semantic_engine.vectors is None:
        messagebox.showwarning(
            "Brak danych",
            "Sieć semantyczna nie jest załadowana lub nie zawiera wektorów (analiza ram niedostępna)."
        )
        return

    import re
    bases = re.findall(r'\[base="([^"]+)"\]', global_query)
    lemma = bases[-1] if bases else global_query.strip().split()[-1] if global_query.strip() else None

    if not lemma:
        messagebox.showwarning("Błąd", "Nie udało się określić słowa do analizy ram.")
        return

    senses = semantic_engine.get_or_create_senses(lemma)
    if not senses:
        messagebox.showinfo("Ramy", f"Słowo '{lemma}' nie ma wyodrębnionych ram w tym korpusie.")
        return

    current_wsd_lemma = lemma

    # -------------------------
    # Helper do czyszczenia etykiety
    # -------------------------
    def clean_frame_label(sense: dict) -> str:
        label = (sense.get("label") or "").strip()
        anchors = sense.get("anchors", []) or []
        members = sense.get("members", []) or []

        # Usuń ewentualne prefixy z dawnych wersji inducera
        prefixes = [
            "rama semantyczna:",
            "Rama semantyczna:",
            "Rama kontekstowa:",
            "Rama kontekstowa:",
            "profil wokół:",
            "Profil wokół:",
            "rama użycia:",
            "Rama użycia:",
        ]

        clean = label
        for p in prefixes:
            if clean.startswith(p):
                clean = clean[len(p):].strip()
                break

        if clean:
            return clean

        preview = ", ".join((anchors or members)[:5])
        if len(anchors or members) > 5:
            preview += ", ..."
        return preview if preview else "nieokreślona"

    # Tworzenie okienka dialogowego
    wsd_win = ctk.CTkToplevel(app)
    wsd_win.title(f"Ramy semantyczne: {lemma}")
    wsd_win.geometry("540x340")
    wsd_win.attributes("-topmost", True)
    wsd_win.configure(fg_color=THEMES[motyw.get()]["app_bg"])

    ctk.CTkLabel(
        wsd_win,
        text=f"Wybierz ramę dla słowa: {lemma}",
        font=("Verdana", 13, "bold")
    ).pack(pady=15)

    dropdown_values = ["Wszystkie ramy"]
    for s in senses:
        frame_id = s.get("frame_id", s.get("sense_id", "?"))
        frame_type = s.get("frame_type", s.get("profile_type", "semantic"))
        clean_preview = clean_frame_label(s)

        if frame_type == "contextual":
            dropdown_values.append(f"Rama kontekstowa {frame_id}: {clean_preview}")
        else:
            dropdown_values.append(f"Rama semantyczna {frame_id}: {clean_preview}")

    selection_var = ctk.StringVar(value="Wszystkie ramy")

    def on_apply():
        choice = selection_var.get()
        wsd_win.destroy()
        filter_by_selected_sense(choice)

    combo = ctk.CTkOptionMenu(
        wsd_win,
        variable=selection_var,
        values=dropdown_values,
        width=440,
        height=35
    )
    combo.pack(pady=20)

    btn_apply = ctk.CTkButton(
        wsd_win,
        text="Filtruj wyniki",
        command=on_apply,
        fg_color="#4E8752",
        hover_color="#57965C"
    )
    btn_apply.pack(pady=20)


def open_topic_modeling():
    # 1. Pobieramy nazwę korpusu bezpośrednio z aktualnego wyboru w UI
    current_corpus_name = corpus_var.get()
    current_corpus_path = files.get(current_corpus_name)

    if not current_corpus_path:
        messagebox.showinfo("Brak korpusu", "Najpierw załaduj i wybierz korpus z menu po lewej stronie.")
        return

    parquet_path = str(Path(current_corpus_path).resolve())

    if not os.path.exists(parquet_path):
        messagebox.showerror("Błąd", f"Nie znaleziono pliku korpusu w lokalizacji:\n{parquet_path}")
        return

    html_path = parquet_path.replace(".parquet", "_raport_tematyczny.html")

    if os.path.exists(html_path):
        ans = messagebox.askyesnocancel(
            "Raport istnieje",
            "Znaleziono gotowy raport tematyczny dla tego korpusu.\n\n"
            "Czy chcesz wygenerować nowy (wymaga ponownych obliczeń i nadpisze stary)?\n\n"
            "Tak - Generuj nowy od zera\n"
            "Nie - Otwórz istniejący raport"
        )
        if ans is None:
            return
        if not ans:
            launch_webview(html_path)
            return

    # --- OKIENKO KONFIGURACJI ---
    setup_win = ctk.CTkToplevel(app)
    setup_win.title("Ustawienia Modelowania")
    setup_win.geometry("450x500")  # POWIĘKSZONE OKNO na nowe opcje
    setup_win.attributes("-topmost", True)

    x = app.winfo_x() + (app.winfo_width() // 2) - 225
    y = app.winfo_y() + (app.winfo_height() // 2) - 225
    setup_win.geometry(f"+{x}+{y}")

    ctk.CTkLabel(setup_win, text="Wybierz liczbę tematów do wygenerowania:", font=("Verdana", 12, "bold")).pack(
        pady=(20, 5))

    mode_var = ctk.StringVar(value="Domyślnie (Brak limitu)")

    def on_mode_change(*args):
        if mode_var.get() == "Ręczna liczba":
            entry_topics.configure(state="normal")
        else:
            entry_topics.configure(state="disabled")

    mode_var.trace_add("write", on_mode_change)

    rb_default = ctk.CTkRadioButton(setup_win, text="Domyślnie (Brak limitu)", variable=mode_var,
                                    value="Domyślnie (Brak limitu)")
    rb_default.pack(anchor="w", padx=40, pady=5)

    rb_auto = ctk.CTkRadioButton(setup_win, text="Auto (Automatyczna redukcja - 'auto')", variable=mode_var,
                                 value="Auto")
    rb_auto.pack(anchor="w", padx=40, pady=5)

    rb_manual = ctk.CTkRadioButton(setup_win, text="Ręczna liczba", variable=mode_var, value="Ręczna liczba")
    rb_manual.pack(anchor="w", padx=40, pady=5)

    entry_topics = ctk.CTkEntry(setup_win, placeholder_text="np. 20", state="disabled")
    entry_topics.pack(fill="x", padx=60, pady=5)

    # --- ZAAWANSOWANE OPCJE ---
    ctk.CTkLabel(setup_win, text="Opcje zaawansowane:", font=("Verdana", 12, "bold")).pack(pady=(15, 5))

    use_lemmas_var = ctk.BooleanVar(value=True)  # Zaznaczone domyślnie
    cb_lemmas = ctk.CTkCheckBox(setup_win, text="Pracuj na formach zlematyzowanych (base)", variable=use_lemmas_var)
    cb_lemmas.pack(anchor="w", padx=40, pady=5)

    use_stopwords_var = ctk.BooleanVar(value=True)
    cb_stopwords = ctk.CTkCheckBox(setup_win, text="Filtruj polskie stop-words (zalecane)",
                                   variable=use_stopwords_var)
    cb_stopwords.pack(anchor="w", padx=40, pady=5)

    # --- NOWE: SUWAK MMR (Różnorodność) ---
    ctk.CTkLabel(setup_win, text="Różnorodność słów (usuwanie synonimów):").pack(anchor="w", padx=40, pady=(10, 0))

    diversity_var = ctk.DoubleVar(value=0.2)  # Domyślnie 0.2

    div_frame = ctk.CTkFrame(setup_win, fg_color="transparent")
    div_frame.pack(fill="x", padx=40, pady=5)

    lbl_div_val = ctk.CTkLabel(div_frame, text="0.20", width=40)
    lbl_div_val.pack(side="right")

    def on_slider_move(val):
        lbl_div_val.configure(text=f"{val:.2f}")

    slider_div = ctk.CTkSlider(div_frame, from_=0.0, to=1.0, variable=diversity_var, command=on_slider_move)
    slider_div.pack(side="left", fill="x", expand=True, padx=(0, 10))

    ctk.CTkLabel(setup_win, text="0.0 = wyłączone | 1.0 = maks. różnorodność", font=("Verdana", 9),
                 text_color="gray").pack(anchor="w", padx=40)

    def start_process():
        mode = mode_var.get()
        nr_topics_val = None
        if mode == "Auto":
            nr_topics_val = "auto"
        elif mode == "Ręczna liczba":
            try:
                nr_topics_val = int(entry_topics.get())
                if nr_topics_val < 2:
                    raise ValueError
            except Exception:
                messagebox.showerror("Błąd", "Podaj prawidłową liczbę całkowitą (większą od 1) dla tematów.")
                return

        use_stopwords = use_stopwords_var.get()
        diversity_val = round(diversity_var.get(), 2)

        # --- NOWE: Pobierz decyzję o lematach z checkboxa ---
        use_lemmas = use_lemmas_var.get()

        setup_win.destroy()
        # --- ZMIANA: Dodano czwarty parametr use_lemmas ---
        _run_modeling_process(nr_topics_val, use_stopwords, diversity_val, use_lemmas)

    ctk.CTkButton(setup_win, text="Rozpocznij analizę", command=start_process).pack(pady=20)

    def _run_modeling_process(nr_topics_val, use_stopwords, diversity_val, use_lemmas):
        # 3. Tworzymy okienko ładowania
        loading_win = ctk.CTkToplevel(app)
        loading_win.title("Modelowanie Tematyczne")
        loading_win.geometry("1100x450")
        loading_win.attributes("-topmost", True)
        loading_win.configure(fg_color=THEMES[motyw.get()]["app_bg"])
        loading_win.grab_set()

        x = app.winfo_x() + (app.winfo_width() // 2) - 300
        y = app.winfo_y() + (app.winfo_height() // 2) - 225
        loading_win.geometry(f"+{x}+{y}")

        lbl_status = ctk.CTkLabel(loading_win, text="Przygotowywanie modelu BERTopic...", font=("Verdana", 14, "bold"))
        lbl_status.pack(pady=(20, 10))

        progress = ctk.CTkProgressBar(loading_win, mode="indeterminate", width=400)
        progress.pack(pady=5)
        progress.start()

        # Pole tekstowe na logi z terminala
        log_box = ctk.CTkTextbox(loading_win, width=550, height=250, font=("Consolas", 11), state="disabled")
        log_box.pack(pady=(15, 10), padx=20, fill="both", expand=True)

        # KORPUSUJ_PATCH_177B7_BERTOPIC_BUFFERED_PROGRESS_HEARTBEAT
        # Stream writes happen in the training thread. Tk widget updates happen
        # only in the GUI-thread timer below; no app.after() call is made per
        # tqdm fragment.
        training_progress_state = {
            "started_at": time.monotonic(),
            "stage": "Inicjalizacja modelowania",
            "last_stream_activity": time.monotonic(),
            "stream_fragments": 0,
            "stream_chars": 0,
            "gui_flushes": 0,
            "finished": False,
            "failed": False,
        }
        training_progress_lock = threading.Lock()

        class BufferedTextRedirector:
            MAX_PENDING_CHARS = 250_000
            MAX_LOG_LINES = 2_000

            def __init__(self, widget, state, state_lock):
                self.widget = widget
                self.state = state
                self.state_lock = state_lock
                self._parts = []
                self._pending_chars = 0
                self._current_terminal_line = ""
                self._closed = False

            def write(self, text):
                if self._closed or text is None:
                    return 0
                value = str(text)
                if not value:
                    return 0
                now = time.monotonic()
                with self.state_lock:
                    self.state["last_stream_activity"] = now
                    self.state["stream_fragments"] += 1
                    self.state["stream_chars"] += len(value)
                    self._parts.append(value)
                    self._pending_chars += len(value)
                    if self._pending_chars > self.MAX_PENDING_CHARS:
                        overflow = self._pending_chars - self.MAX_PENDING_CHARS
                        while self._parts and overflow > 0:
                            removed = self._parts.pop(0)
                            overflow -= len(removed)
                            self._pending_chars -= len(removed)
                return len(value)

            def flush(self):
                # File-like API compatibility. GUI flushing is timer-driven.
                return None

            def close(self):
                self._closed = True

            def _drain_parts(self):
                with self.state_lock:
                    if not self._parts:
                        return ""
                    value = "".join(self._parts)
                    self._parts.clear()
                    self._pending_chars = 0
                return value

            def _normalize_terminal_updates(self, value):
                # tqdm writes '\r' to replace one terminal line. Preserve
                # ordinary newline records but publish only the newest version
                # of a carriage-return line.
                completed = []
                current = self._current_terminal_line
                for char in value:
                    if char == "\r":
                        current = ""
                    elif char == "\n":
                        completed.append(current)
                        current = ""
                    else:
                        current += char
                self._current_terminal_line = current
                out = ""
                if completed:
                    out = "\n".join(completed) + "\n"
                return out, current

            def flush_to_widget(self):
                value = self._drain_parts()
                if not value:
                    return False
                completed, current = self._normalize_terminal_updates(value)
                self.widget.configure(state="normal")
                if completed:
                    self.widget.insert("end", completed)
                # Keep exactly one replaceable line for tqdm-style progress.
                try:
                    self.widget.index("progress_line_start")
                except Exception:
                    self.widget.mark_set("progress_line_start", "end")
                self.widget.delete("progress_line_start", "end")
                self.widget.mark_set("progress_line_start", "end")
                if current:
                    self.widget.insert("end", current)
                try:
                    line_count = int(self.widget.index("end-1c").split(".")[0])
                    if line_count > self.MAX_LOG_LINES:
                        remove_to = line_count - self.MAX_LOG_LINES + 1
                        self.widget.delete("1.0", f"{remove_to}.0")
                except Exception:
                    pass
                self.widget.see("end")
                self.widget.configure(state="disabled")
                with self.state_lock:
                    self.state["gui_flushes"] += 1
                return True

            def snapshot(self):
                with self.state_lock:
                    data = dict(self.state)
                    data["pending_fragments"] = len(self._parts)
                    data["pending_chars"] = self._pending_chars
                return data

        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        progress_redirector = BufferedTextRedirector(
            log_box,
            training_progress_state,
            training_progress_lock,
        )
        sys.stdout = progress_redirector
        sys.stderr = progress_redirector

        def _set_training_progress_stage(stage, *, finished=False, failed=False):
            with training_progress_lock:
                training_progress_state["stage"] = str(stage)
                if finished:
                    training_progress_state["finished"] = True
                if failed:
                    training_progress_state["failed"] = True

        progress_monitor_after_id = {"value": None}
        progress_window_closed = {"value": False}

        def _progress_widgets_alive_177b10():
            if progress_window_closed["value"]:
                return False
            try:
                return bool(loading_win.winfo_exists()) and bool(log_box.winfo_exists())
            except Exception:
                return False

        def restore_stdout(event=None):
            # <Destroy> also propagates from child widgets. Restore only when
            # the toplevel itself is being destroyed.
            try:
                if event is not None and getattr(event, "widget", None) is not loading_win:
                    return
            except Exception:
                pass
            progress_redirector.close()
            if sys.stdout is progress_redirector:
                sys.stdout = old_stdout
            if sys.stderr is progress_redirector:
                sys.stderr = old_stderr

        def _close_progress_window_177b10():
            if progress_window_closed["value"]:
                return
            progress_window_closed["value"] = True
            after_id = progress_monitor_after_id.get("value")
            progress_monitor_after_id["value"] = None
            if after_id is not None:
                try:
                    app.after_cancel(after_id)
                except Exception:
                    pass
            restore_stdout()
            try:
                if loading_win.winfo_exists():
                    loading_win.destroy()
            except Exception:
                pass

        loading_win.bind("<Destroy>", restore_stdout)

        def worker():
            try:
                _set_training_progress_stage("Inicjalizacja TopicEngine")
                print("Inicjalizacja TopicEngine...")
                engine = TopicEngine(parquet_path)
                app.after(0, lambda: lbl_status.configure(text="Wczytywanie i filtrowanie tekstów..."))

                _set_training_progress_stage("Wczytywanie i chunking dokumentów")
                if not engine.load_data(use_chunking=True, max_words_per_chunk=250, use_lemmas=use_lemmas):
                    raise Exception("Plik korpusu nie posiada kolumny 'Treść' lub jest pusty.")

                app.after(0, lambda: lbl_status.configure(text="Trenowanie modelu (może to potrwać)..."))
                print("Rozpoczęto trenowanie modelu BERTopic...")

                # --- NOWE: Przekazujemy nr_topics i wymuszamy nadpisanie ---
                _set_training_progress_stage("Trening BERTopic / tworzenie embeddingów")
                if not engine.train_model(nr_topics=nr_topics_val, force_retrain=True, use_stopwords=use_stopwords, diversity=diversity_val):
                    raise Exception("Błąd podczas treningu modelu.")

                freq_df = engine.model.get_topic_freq()
                valid_topics = freq_df[freq_df['Topic'] != -1]

                if valid_topics.empty:
                    print("OSTRZEŻENIE: Zbyt mało danych. Model sklasyfikował wszystko jako szum (-1).")
                    logging.warning("Zbyt mało danych. Model sklasyfikował wszystko jako szum (-1).")
                    app.after(0, _close_progress_window_177b10)
                    app.after(0, lambda: messagebox.showwarning("Brak tematów",
                                                                "Zbyt mało danych. Model nie odnalazł powiązań między dokumentami (tylko szum)."))
                    return

                _set_training_progress_stage("Generowanie opcjonalnych wizualizacji")
                print("Generowanie wizualizacji...")

                def _optional_topic_visualization(label, factory):
                    try:
                        return factory()
                    except Exception as vis_err:
                        print(f"OSTRZEŻENIE: Pominięto opcjonalną wizualizację '{label}': {vis_err}")
                        logging.warning(
                            "Pominięto opcjonalną wizualizację BERTopic %r; model i raport pozostają dostępne: %s",
                            label,
                            vis_err,
                            exc_info=True,
                        )
                        return None

                fig_map = _optional_topic_visualization(
                    "mapa tematów",
                    engine.visualize_topic_map,
                )
                tot = _optional_topic_visualization(
                    "tematy w czasie",
                    engine.calculate_topics_over_time,
                )
                fig_time = (
                    _optional_topic_visualization(
                        "wykres tematów w czasie",
                        lambda: engine.visualize_dynamic_topics(tot, top_n_topics=15),
                    )
                    if tot is not None
                    else None
                )
                fig_words = _optional_topic_visualization(
                    "ranking słów",
                    lambda: engine.visualize_word_scores(top_n_topics=15),
                )

                _set_training_progress_stage("Budowanie raportu HTML")
                print("Budowanie raportu HTML...")
                html_content = """
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <meta charset="utf-8">
                                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                                <style>
                                    body { font-family: 'Verdana', sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }
                                    .chart-container { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; padding: 20px; }
                                    h2 { color: #4B6CB7; text-align: center; border-bottom: 2px solid #4B6CB7; padding-bottom: 10px; margin-bottom: 20px; }
                                    table { width: 100%; border-collapse: collapse; text-align: left; }
                                    th, td { padding: 12px; border-bottom: 1px solid #ddd; }
                                    th { background-color: #4B6CB7; color: white; }
                                </style>
                            </head>
                            <body>
                            """

                if fig_time is not None:
                    html_content += "<div class='chart-container'><h2>Ewolucja tematów w czasie</h2>"
                    html_content += fig_time.to_html(full_html=False, include_plotlyjs='cdn')
                    html_content += "</div>"

                if fig_map is not None:
                    html_content += "<div class='chart-container'><h2>Mapa tematów (Intertopic Distance)</h2>"
                    html_content += fig_map.to_html(full_html=False, include_plotlyjs='cdn')
                    html_content += "</div>"

                if fig_words is not None:
                    html_content += "<div class='chart-container'><h2>Ranking słów kluczowych (c-TF-IDF Scores)</h2>"
                    html_content += fig_words.to_html(full_html=False, include_plotlyjs='cdn')
                    html_content += "</div>"

                try:
                    topic_info = engine.get_topic_info()
                    if topic_info is not None:
                        html_content += "<div class='chart-container'><h2>Słowa kluczowe zidentyfikowanych tematów</h2>"
                        html_content += "<table>"
                        html_content += "<tr><th>ID Tematu</th><th>Liczba tekstów</th><th>Najważniejsze słowa (Współczynnik c-TF-IDF)</th></tr>"

                        for _, row in topic_info.head(30).iterrows():
                            topic_id = row['Topic']
                            count = row['Count']
                            words_with_scores = engine.model.get_topic(topic_id)

                            if words_with_scores:
                                formatted_words = ", ".join(
                                    [f"{w} (<b>{s:.4f}</b>)" for w, s in words_with_scores[:10]])
                            else:
                                formatted_words = "Brak danych"

                            if topic_id == -1:
                                bg_color = "#f9ecec"
                                topic_name = "-1 (Szum / Niesklasyfikowane)"
                            else:
                                bg_color = "#ffffff"
                                topic_name = str(topic_id)

                            html_content += f"<tr style='background-color: {bg_color};'>"
                            html_content += f"<td style='font-weight: bold;'>{topic_name}</td>"
                            html_content += f"<td>{count}</td>"
                            html_content += f"<td>{formatted_words}</td>"
                            html_content += "</tr>"

                        html_content += "</table></div>"
                except Exception as ex:
                    print(f"Nie udało się wygenerować tabeli tematów: {ex}")
                    logging.info(f"Nie udało się wygenerować tabeli tematów: {ex}")

                html_content += "</body></html>"

                try:
                    print("Zapisywanie na dysku...")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                        f.flush()
                        os.fsync(f.fileno())

                    if os.path.exists(html_path):
                        logging.info(f"SUKCES: Plik HTML został utworzony: {html_path}")
                        print("Zakończono sukcesem. Uruchamianie WebView...")
                    else:
                        raise Exception("System zgłosił sukces, ale plik nie pojawił się na dysku.")

                except Exception as write_err:
                    print(f"Błąd zapisu: {write_err}")
                    logging.error(f"Błąd zapisu pliku HTML: {write_err}")
                    app.after(0, _close_progress_window_177b10)
                    raise Exception(f"Błąd zapisu raportu. Sprawdź czy masz uprawnienia do folderu.\n{write_err}")

                app.after(0, _close_progress_window_177b10)
                _set_training_progress_stage("Modelowanie zakończone", finished=True)
                launch_webview(html_path)

            except Exception as e:
                _set_training_progress_stage("Błąd modelowania", finished=True, failed=True)
                err_msg = str(e)
                print(f"BŁĄD KRYTYCZNY: {err_msg}")
                logging.exception(f"Błąd podczas generowania modelu BERTopic: {err_msg}")
                app.after(0, _close_progress_window_177b10)
                app.after(0, lambda msg=err_msg: messagebox.showerror("Błąd BERTopic", f"Szczegóły:\n{msg}"))

        def _format_elapsed_177b7(seconds):
            total = max(0, int(seconds))
            hours, remainder = divmod(total, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        def _schedule_training_progress_monitor_177b10():
            if not _progress_widgets_alive_177b10():
                return
            progress_monitor_after_id["value"] = app.after(
                250,
                _training_progress_monitor_tick_177b7,
            )

        def _training_progress_monitor_tick_177b7():
            progress_monitor_after_id["value"] = None
            if not _progress_widgets_alive_177b10():
                return
            try:
                progress_redirector.flush_to_widget()
                snapshot = progress_redirector.snapshot()
                now = time.monotonic()
                elapsed = now - snapshot["started_at"]
                silent_for = now - snapshot["last_stream_activity"]
                alive = bool(training_thread.is_alive())
                if snapshot.get("failed"):
                    worker_state = "błąd"
                elif snapshot.get("finished"):
                    worker_state = "zakończony"
                elif alive:
                    worker_state = "aktywny"
                else:
                    worker_state = "zatrzymany"
                status = (
                    f"{snapshot['stage']} | czas {_format_elapsed_177b7(elapsed)} | "
                    f"worker: {worker_state} | ostatni komunikat: {int(silent_for)} s | "
                    f"bufor: {snapshot['pending_fragments']} frag. / {snapshot['pending_chars']} zn."
                )
                lbl_status.configure(text=status)
                if silent_for >= 120 and alive and not snapshot.get("finished"):
                    loading_win.title("Modelowanie Tematyczne — długi krok bez komunikatów")
                else:
                    loading_win.title("Modelowanie Tematyczne")
                if alive or snapshot["pending_fragments"] or not snapshot.get("finished"):
                    _schedule_training_progress_monitor_177b10()
                elif _progress_widgets_alive_177b10():
                    progress_redirector.flush_to_widget()
            except Exception as monitor_err:
                # A destroyed widget is a normal terminal condition. Report and
                # reschedule only while the complete progress UI still exists.
                if _progress_widgets_alive_177b10():
                    try:
                        old_stderr.write(f"[BERTopic progress monitor] {monitor_err}\n")
                        old_stderr.flush()
                    except Exception:
                        pass
                    _schedule_training_progress_monitor_177b10()

        training_thread = threading.Thread(target=worker, daemon=True, name="korpusuj-bertopic-training")
        training_thread.start()
        _schedule_training_progress_monitor_177b10()


# Tworzenie interfejsu GUI
notify_status("Inicjalizacja silnika graficznego...")
app = ctk.CTk()
import tkinter as tk
tk._default_root = app
app.withdraw()

menu = Menu(app)

file_menu = menu.menu_bar(text="Plik", tearoff=0)
file_menu.add_command(label="Nowy projekt", command=load_corpora)
file_menu.add_command(label="Informacje o korpusie", command=show_corpus_info)
file_menu.add_command(label="Eksportuj wyniki", command=export_data)
file_menu.add_separator()
file_menu.add_command(label="Utwórz korpus", command=lambda: get_creator_module().main(app))
file_menu.add_command(label="Utwórz podkorpus z wyników", command=export_to_subcorpus)
file_menu.add_command(label="Utwórz podkorpus po metadanych", command=export_subcorpus_by_metadata)
file_menu.add_separator()
file_menu.add_command(label="Zamknij", command=lambda: exit())
file_menu = menu.menu_bar(text="Edytuj", tearoff=0)
file_menu.add_command(label="Cofnij", command=lambda: undo())
file_menu.add_command(label="Ponów", command=lambda: redo())
history_menu = menu.menu_bar(text="Historia", tearoff=0)
update_history_menu()

tools_menu = menu.menu_bar(text="Narzędzia", tearoff=0)
tools_menu.add_command(label="Sieć semantyczna", command=smart_show_semantic_network)
tools_menu.add_command(label="Filtrowanie wyników według ram", command=show_wsd_dialog)
tools_menu.add_command(label="Modelowanie tematyczne (BERTopic)", command=open_topic_modeling)

file_menu = menu.menu_bar(text="Ustawienia", tearoff=0)
file_menu.add_command(label="Preferencje", command=settings_window)
file_menu = menu.menu_bar(text="Pomoc", tearoff=0)
# Przekazujemy konkretne nazwy plików do funkcji
file_menu.add_command(label="Instrukcja użytkownika",
                      command=lambda: open_webview_window("temp/Instrukcja_uzytkownika.html"))
file_menu.add_command(label="Przewodnik po języku zapytań",
                      command=lambda: open_webview_window("temp/Przewodnik_po_jezyku_zapytan.html"))

app.title("Korpusuj")
icon_path = os.path.join(BASE_DIR, "favicon.ico")
try:
    app.iconbitmap(icon_path)
except Exception as e:
    logging.info(f"Ostrzeżenie: Nie udało się załadować ikony: {e}")


# Global vars
font_family = ctk.StringVar(value=config['font_family'])
fontsize = config['fontsize']
styl_wykresow = ctk.StringVar(value=config['styl_wykresow'])
motyw = ctk.StringVar(value=config['motyw'])
plotting = ctk.StringVar(value=config.get('plotting', DEFAULT_SETTINGS['plotting']))
dependency_ram_usage_var = ctk.StringVar(value=_dependency_mode_to_label(config.get('dependency_cache_ram_mode', DEFAULT_SETTINGS.get('dependency_cache_ram_mode', DEFAULT_DEPENDENCY_RAM_MODE))))
kontekst = config.get('kontekst', DEFAULT_SETTINGS['kontekst'])
min_tokens_threshold = config.get('min_tokens_threshold', DEFAULT_SETTINGS['min_tokens_threshold'])
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

nav_buttons_frame = ctk.CTkFrame(top_frame_container, fg_color="transparent")
nav_buttons_frame.grid(row=1, rowspan=2, column=0, padx=5)

btn_nav_back = ctk.CTkButton(
    nav_buttons_frame, text="<", width=35, height=35,
    font=("Verdana", 14, "bold"), fg_color="#4B6CB7", hover_color="#5B7CD9",
    state="disabled", command=go_back
)
btn_nav_back.pack(side="left", padx=2)

btn_nav_forward = ctk.CTkButton(
    nav_buttons_frame, text=">", width=35, height=35,
    font=("Verdana", 14, "bold"), fg_color="#4B6CB7", hover_color="#5B7CD9",
    state="disabled", command=go_forward
)
btn_nav_forward.pack(side="left", padx=2)

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
corpus_var.trace_add("write", lambda *args: load_semantic_neighbors())
option_corpus.grid(row=2, column=1, padx=1, pady=1, sticky="w")

# Query widget (keep background)
entry_query = ctk.CTkTextbox(
    top_frame_container, height=100, font=("JetBrains Mono Bold", 14),
    exportselection=False, corner_radius=12, fg_color="#1F2328"
)
entry_query.grid(row=0, rowspan=4, column=2, padx=15,  pady=(5,5), sticky="ew")
CQLAutocomplete(entry_query, feat_mapping=FEAT_MAPPING)

def open_query_builder():
    current_theme = THEMES[motyw.get()]
    QueryBuilderWindow(
        app,
        entry_query,
        current_theme,
        ner_prefixes=NER_PREFIXES,
        ner_types=NER_TYPES,
    )

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
    values=["Alfabetycznie", "Lewy kontekst", "Prawy kontekst", "Autor", "Tytuł", "Data publikacji", "Frekwencja base","Frekwencja orth"],
    variable=sort_option_var,
    command=resort_results,
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
    command=handle_tab_change,
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
kursor_separatora = "resizeleftright" if sys.platform == "darwin" else "size_we"
paned_window = tk.PanedWindow(result_frame, orient="horizontal", bg="#2C2F33", bd=0, sashwidth=8, sashcursor=kursor_separatora, opaqueresize=False)
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
    values=["Formy podstawowe (base)", "Formy ortograficzne (orth)", "Częstość w czasie", "Kolokacje", "Profil kolokacyjny"],
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

lemma_frame.rowconfigure(0, weight=0)
lemma_frame.rowconfigure(1, weight=1)
lemma_frame.columnconfigure(0, weight=1)


orth_frame = ctk.CTkFrame(
    tab_wyniki_frekw,
    fg_color="#2C2F33",
    corner_radius=15
)

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

orth_frame.rowconfigure(0, weight=0)
orth_frame.rowconfigure(1, weight=1)
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

month_frame.rowconfigure(0, weight=0)
month_frame.rowconfigure(1, weight=1)
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

# Lewy panel na opcje (przewijany)
colloc_options_frame = ctk.CTkScrollableFrame(colloc_frame, fg_color="transparent", corner_radius=0, width=280)
colloc_options_frame.pack(pady=(5, 10), padx=(10, 5), side="left", fill="y")

# Prawy panel na tabelę i paginację
colloc_data_frame = ctk.CTkFrame(colloc_frame, fg_color="transparent")
colloc_data_frame.pack(pady=10, padx=(0, 10), side="left", fill="both", expand=True)

# --- Zmienne sterujące ---
colloc_sort_var = ctk.StringVar(master=app, value="Log-Dice")
colloc_mode_var = ctk.StringVar(master=app, value="Liniowe")
syn_dir_var = ctk.StringVar(master=app, value="Podrzędnik")
syn_deprel_var = ctk.StringVar(master=app, value="Wszystkie")
upos_var = ctk.StringVar(master=app, value="Wszystkie")
pos_var = ctk.StringVar(master=app, value="Wszystkie")
colloc_form_var = ctk.StringVar(master=app, value="Lemat (base)")
sentence_boundary_var = ctk.BooleanVar(master=app, value=True)
colloc_ignore_case_var = ctk.BooleanVar(master=app, value=False)

font_ui = ("Verdana", 11, 'bold')
fg_opt = "#4B6CB7"

# ==========================================
# KARTA 1: Metoda wyszukiwania
# ==========================================
card_method = SettingsCard(colloc_options_frame, "Metoda wyszukiwania", expanded=True, theme=THEMES[motyw.get()], registry=settings_cards)
method_frame = card_method.content

ctk.CTkLabel(method_frame, text="Typ kontekstu:", font=font_ui).pack(anchor="w", pady=(5, 2))
ctk.CTkOptionMenu(method_frame, variable=colloc_mode_var, values=["Liniowe", "Składniowe"],
                  command=lambda e: toggle_colloc_mode(), fg_color=fg_opt, button_color=fg_opt).pack(fill="x", pady=(0, 10))

dynamic_method_frame = ctk.CTkFrame(method_frame, fg_color="transparent")
dynamic_method_frame.pack(fill="x", pady=(0, 5))

# Tryb Liniowy
frame_linear = ctk.CTkFrame(dynamic_method_frame, fg_color="transparent")
ctk.CTkLabel(frame_linear, text="L-span:", font=font_ui).grid(row=0, column=0, sticky="w", padx=(0, 5))
entry_l_span = ctk.CTkEntry(frame_linear, width=45, height=28, corner_radius=8)
entry_l_span.insert(0, "5")
entry_l_span.grid(row=0, column=1, sticky="w", padx=(0, 15))

ctk.CTkLabel(frame_linear, text="R-span:", font=font_ui).grid(row=0, column=2, sticky="w", padx=(0, 5))
entry_r_span = ctk.CTkEntry(frame_linear, width=45, height=28, corner_radius=8)
entry_r_span.insert(0, "5")
entry_r_span.grid(row=0, column=3, sticky="w")

# Tryb Składniowy
frame_syntactic = ctk.CTkFrame(dynamic_method_frame, fg_color="transparent")
ctk.CTkLabel(frame_syntactic, text="Kierunek:", font=font_ui).pack(anchor="w", pady=(0, 2))
ctk.CTkOptionMenu(frame_syntactic, variable=syn_dir_var, values=["Podrzędnik", "Nadrzędnik", "Oba"],
                  fg_color=fg_opt, button_color=fg_opt).pack(fill="x", pady=(0, 10))
ctk.CTkLabel(frame_syntactic, text="Relacja (deprel):", font=font_ui).pack(anchor="w", pady=(0, 2))

# 1. Zastępujemy CTkOptionMenu przyciskiem udającym rozwijaną listę
syn_deprel_btn = ctk.CTkButton(
    frame_syntactic,
    text="Wszystkie ▼",
    font=font_ui,
    fg_color=fg_opt,
    hover_color="#5B7CD9",
    text_color="white",
    anchor="w"
)
syn_deprel_btn.pack(fill="x")


# 2. Logika dynamicznej zmiany napisu na przycisku
def _update_syn_deprel_btn_text(*args):
    val = syn_deprel_var.get()
    # Jeśli nazwa jest bardzo długa, ucinamy żeby nie rozpychała lewego panelu
    disp = val if len(val) < 25 else val[:22] + "..."
    syn_deprel_btn.configure(text=f"{disp} ▼")


syn_deprel_var.trace_add("write", _update_syn_deprel_btn_text)
_update_syn_deprel_btn_text()  # Ustawienie poprawnego napisu na start

# 3. Tworzenie kaskadowego menu (wczytuje aktualne kolory motywu)
current_theme = THEMES[motyw.get()]
deprel_menu = tk.Menu(syn_deprel_btn, tearoff=0,
                      bg=current_theme["dropdown_fg"],
                      fg=current_theme["button_text"],
                      activebackground=current_theme["dropdown_hover"],
                      activeforeground=current_theme["button_text"],
                      font=("Verdana", 11))

# 4. Generowanie opcji ze słownika DEPREL_TREE_DICT
for main_cat, sub_cats in DEPREL_TREE_DICT.items():
    if not sub_cats:
        # Kategoria bez podkategorii
        deprel_menu.add_command(label=main_cat, command=lambda c=main_cat: syn_deprel_var.set(c))
    else:
        # Kategoria z podkategoriami (tworzymy sub-menu)
        sub_menu = tk.Menu(deprel_menu, tearoff=0,
                           bg=current_theme["dropdown_fg"],
                           fg=current_theme["button_text"],
                           activebackground=current_theme["dropdown_hover"],
                           activeforeground=current_theme["button_text"],
                           font=("Verdana", 11))

        # Opcja dla samej kategorii głównej (gwiazdka tylko dla warstwy wizualnej)
        sub_menu.add_command(label=f"★ {main_cat} (zbiorcze)", command=lambda c=main_cat: syn_deprel_var.set(c))
        sub_menu.add_separator()

        for sub_cat in sub_cats:
            sub_menu.add_command(label=sub_cat, command=lambda c=sub_cat: syn_deprel_var.set(c))

        deprel_menu.add_cascade(label=main_cat, menu=sub_menu)


# 5. Funkcja wywołująca menu pod przyciskiem (na kliknięcie)
def show_deprel_menu(event=None):
    if syn_deprel_btn.cget("state") != "disabled":
        x = syn_deprel_btn.winfo_rootx()
        y = syn_deprel_btn.winfo_rooty() + syn_deprel_btn.winfo_height()
        deprel_menu.tk_popup(x, y)


syn_deprel_btn.configure(command=show_deprel_menu)
def toggle_colloc_mode(*args):
    if colloc_mode_var.get() == "Liniowe":
        frame_linear.pack(fill="x", expand=True)
        frame_syntactic.pack_forget()
    else:
        frame_linear.pack_forget()
        frame_syntactic.pack(fill="x", expand=True)

toggle_colloc_mode()

chk_sentence_bound = ctk.CTkCheckBox(method_frame, text="Ogranicz do zdań", variable=sentence_boundary_var, font=font_ui, fg_color="#4E8752", hover_color="#57965C")
chk_sentence_bound.pack(anchor="w", pady=(10, 5))

# ==========================================
# KARTA 2: Filtry lingwistyczne
# ==========================================
card_filters = SettingsCard(colloc_options_frame, "Filtry lingwistyczne", expanded=False, theme=THEMES[motyw.get()], registry=settings_cards)
filters_frame = card_filters.content

ctk.CTkLabel(filters_frame, text="Forma kolokatu:", font=font_ui).pack(anchor="w", pady=(5, 2))
ctk.CTkOptionMenu(filters_frame, variable=colloc_form_var, values=["Lemat (base)", "Token (orth)"],
                  fg_color=fg_opt, button_color=fg_opt).pack(fill="x", pady=(0, 10))

ctk.CTkLabel(filters_frame, text="Część mowy (UPOS):", font=font_ui).pack(anchor="w", pady=(0, 2))
ctk.CTkOptionMenu(filters_frame, variable=upos_var, values=all_upos,
                  fg_color=fg_opt, button_color=fg_opt).pack(fill="x", pady=(0, 10))

ctk.CTkLabel(filters_frame, text="Część mowy NKJP (POS):", font=font_ui).pack(anchor="w", pady=(0, 2))

# 1. Przypisujemy dropdown do zmiennej, aby odnieść się do niego przy pakowaniu ramki
pos_menu = ctk.CTkOptionMenu(filters_frame, variable=pos_var, values=all_pos,
                             fg_color=fg_opt, button_color=fg_opt)
pos_menu.pack(fill="x", pady=(0, 10))

dynamic_feat_vars = {}
dynamic_features_frame = ctk.CTkFrame(filters_frame, fg_color="transparent")

# 2. Checkbox wrzucamy na stałe na dół karty, już pod ramkę z filtrami
chk_ignore_case = ctk.CTkCheckBox(filters_frame, text="Ignoruj wielkość liter", variable=colloc_ignore_case_var,
                                  font=font_ui, fg_color="#4E8752", hover_color="#57965C")
chk_ignore_case.pack(anchor="w", pady=(5, 5))


def update_dynamic_features(selected_val):
    for widget in dynamic_features_frame.winfo_children():
        widget.destroy()
    dynamic_feat_vars.clear()

    clean_pos = selected_val.split(" ")[0]
    if clean_pos in FEAT_MAPPING and clean_pos != "Wszystkie":

        # Pojawia się nowa cecha -> Wrzucamy ramkę z powrotem dokładnie pod "pos_menu"
        dynamic_features_frame.pack(fill="x", after=pos_menu)

        for feat in FEAT_MAPPING[clean_pos].keys():
            lbl_text = {"number": "Liczba", "case": "Przypadek", "gender": "Rodzaj",
                        "degree": "Stopień", "person": "Osoba", "aspect": "Aspekt",
                        "negation": "Zanegowanie", "accentability": "Akcentowość",
                        "post-prepositionality": "Poprzyimkowość", "accommodability": "Akomodacyjność",
                        "vocalicity": "Wokaliczność", "agglutination": "Aglutynacyjność",
                        "fullstoppedness": "Kropkowalność"}.get(feat, feat)

            ctk.CTkLabel(dynamic_features_frame, text=f"{lbl_text}:", font=font_ui).pack(anchor="w", pady=(0, 2))
            var = ctk.StringVar(value="Wszystkie")
            dynamic_feat_vars[feat] = var
            options = ["Wszystkie"] + MORPH_DICTS.get(feat, [])
            ctk.CTkOptionMenu(dynamic_features_frame, variable=var, values=options,
                              fg_color=fg_opt, button_color=fg_opt).pack(fill="x", pady=(0, 10))
    else:
        # Brak cech -> Całkowicie zwijamy ramkę z interfejsu, żeby nie było dziury
        dynamic_features_frame.pack_forget()


# 3. Podpinamy odświeżanie do menu (dopiero po zdefiniowaniu funkcji)
pos_menu.configure(command=update_dynamic_features)

# 4. Uruchomienie na start (ukryje ramkę od razu, bo domyślnie wybrane jest "Wszystkie")
update_dynamic_features(pos_var.get())


# ==========================================
# KARTA 3: Parametry statystyczne
# ==========================================
card_stats = SettingsCard(colloc_options_frame, "Parametry statystyczne", expanded=False, theme=THEMES[motyw.get()], registry=settings_cards)
stats_frame = card_stats.content

ctk.CTkLabel(stats_frame, text="Sortowanie:", font=font_ui).pack(anchor="w", pady=(5, 2))
ctk.CTkOptionMenu(stats_frame, variable=colloc_sort_var, values=["Log-Dice", "MI Score", "T-score", "Log-Likelihood"],
                  fg_color=fg_opt, button_color=fg_opt).pack(fill="x", pady=(0, 10))

freq_range_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
freq_range_frame.pack(fill="x", pady=(0, 5))

ctk.CTkLabel(freq_range_frame, text="Min f:", font=font_ui).grid(row=0, column=0, sticky="w", padx=(0, 5))
entry_min_freq = ctk.CTkEntry(freq_range_frame, width=45, height=28, corner_radius=8)
entry_min_freq.insert(0, "1")
entry_min_freq.grid(row=0, column=1, sticky="w", padx=(0, 5))

ctk.CTkLabel(freq_range_frame, text="Min r:", font=font_ui).grid(row=0, column=2, sticky="w", padx=(10, 5))
entry_min_range = ctk.CTkEntry(freq_range_frame, width=45, height=28, corner_radius=8)
entry_min_range.insert(0, "1")
entry_min_range.grid(row=0, column=3, sticky="w", padx=(0, 5))

# --- Ikonka pomocy dla Min f i Min r ---
colloc_help_icon = ctk.CTkLabel(freq_range_frame, text="❓", font=("Verdana", 14), text_color="#4B6CB7", cursor="hand2")
colloc_help_icon.grid(row=0, column=4, sticky="w", padx=(5, 0))

colloc_help_text = (
    "PARAMETRY FILTROWANIA:\n"
    "• Min f (Minimalna frekwencja): Ile razy dana para słów musi wystąpić\n"
    "  obok siebie w całym korpusie, aby algorytm w ogóle wziął ją pod uwagę.\n"
    "  (Pomaga odrzucić np. jednorazowe literówki lub przypadkowe zbitki).\n\n"
    "• Min r (Minimalny zasięg): W ilu RÓŻNYCH tekstach (dokumentach)\n"
    "  musi wystąpić kolokacja, aby została uznana za istotną statystycznie.\n"
    "  (Zapobiega faworyzowaniu specyficznych zwrotów użytych wielokrotnie\n"
    "  tylko w jednym konkretnym tekście / przez jednego autora)."
)

ToolTip(colloc_help_icon, colloc_help_text)


# --- Przycisk OBLICZ ---
btn_calc_colloc = ctk.CTkButton(colloc_options_frame, text="Oblicz", command=lambda: calculate_collocs(), corner_radius=8,
                                fg_color="#4E8752", hover_color="#57965C", font=("Verdana", 14, 'bold'), height=40)
btn_calc_colloc.pack(fill="x", pady=(20, 10), padx=5)


# ==========================================
# SEKCJA PRAWA: Tabela i Paginacja
# ==========================================
colloc_headers = ["Nr", "Kolokat", "Współwystąpienia", "Frekw. kolokatu", "Log-Likelihood", "MI Score", "T-score", "Log-Dice"]
colloc_widths = [50, 150, 100, 100, 120, 100, 100, 100]
colloc_justify = ["center", "center", "center", "center", "center", "center", "center", "center"]
colloc_data = []

# Ramka paginacji trafia teraz do colloc_data_frame
pagination_colloc_frame = ctk.CTkFrame(colloc_data_frame, fg_color="#1F2328", corner_radius=12)
pagination_colloc_frame.pack(fill="x", pady=(0, 5))

for col in range(5):
    pagination_colloc_frame.columnconfigure(col, weight=1)

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

# Główna tabela trafia do colloc_data_frame i wypełnia przestrzeń
colloc_table = table.CustomTable(
    colloc_data_frame, colloc_headers, colloc_data, colloc_widths, colloc_justify, 15,
    fulltext_data=[],
    search_callback=search_from_table
)
colloc_table.pack(fill="both", expand=True)

paginator_colloc = {
    "data": colloc_data,
    "current_page": [0],
    "table": colloc_table,
    "label": page_label_colloc,
    "items_per_page": 15
}

colloc_table.sort_callback = lambda col, asc: global_sort_callback(paginator_colloc, col, asc)

# ==========================================
# --- Ramka dla Profil kolokacyjny ---
# ==========================================
profile_frame = ctk.CTkFrame(tab_wyniki_frekw, fg_color="#2C2F33", corner_radius=15)

# Lewy panel na opcje (przewijany)
profile_options_frame = ctk.CTkScrollableFrame(profile_frame, fg_color="transparent", corner_radius=0, width=280)
profile_options_frame.pack(pady=(5, 10), padx=(10, 5), side="left", fill="y")

# Prawy panel na tabelę
profile_data_frame = ctk.CTkFrame(profile_frame, fg_color="transparent")
profile_data_frame.pack(pady=10, padx=(0, 10), side="left", fill="both", expand=True)

# Karta opcji
card_profile = SettingsCard(profile_options_frame, "Opcje profilu", expanded=True, theme=THEMES[motyw.get()], registry=settings_cards)
profile_settings = card_profile.content

ctk.CTkLabel(profile_settings, text="Minimalna frekwencja (Min f):", font=("Verdana", 11, 'bold')).pack(anchor="w", pady=(0, 2))
entry_profile_minf = ctk.CTkEntry(profile_settings, width=150, height=28, corner_radius=8)
entry_profile_minf.insert(0, "1")
entry_profile_minf.pack(fill="x", pady=(0, 10))

# --- NOWOŚĆ: Ignoruj wielkość liter ---
profile_ignore_case_var = ctk.BooleanVar(master=app, value=False)
chk_profile_ignore_case = ctk.CTkCheckBox(
    profile_settings, text="Ignoruj wielkość liter", variable=profile_ignore_case_var,
    font=("Verdana", 11, "bold"), fg_color="#4E8752", hover_color="#57965C"
)
chk_profile_ignore_case.pack(anchor="w", pady=(0, 10))

profile_mwe_var = ctk.BooleanVar(master=app, value=True)
chk_profile_mwe = ctk.CTkCheckBox(
    profile_settings, text="Wyciągaj całe frazy (MWE)", variable=profile_mwe_var,
    font=("Verdana", 11, "bold"), fg_color="#4E8752", hover_color="#57965C"
)
chk_profile_mwe.pack(anchor="w", pady=(0, 10))

ctk.CTkLabel(profile_settings, text="Słowo centralne (węzeł):", font=("Verdana", 11, 'bold')).pack(anchor="w", pady=(0, 2))
profile_node_var = ctk.StringVar(value="Token 1")
profile_node_menu = ctk.CTkOptionMenu(
    profile_settings,
    variable=profile_node_var,
    values=["Token 1"],
    font=("Verdana", 11, "bold"),
    fg_color="#4B6CB7", button_color="#4B6CB7",
    dropdown_fg_color="#4B6CB7", dropdown_hover_color="#5B7CD9", text_color="white"
)
profile_node_menu.pack(fill="x", pady=(0, 10))

btn_calc_profile = ctk.CTkButton(profile_options_frame, text="Generuj", corner_radius=8,
                                fg_color="#4E8752", hover_color="#57965C", font=("Verdana", 14, 'bold'), height=40)
btn_calc_profile.pack(fill="x", pady=(10, 10), padx=5)

# Karta na wybór relacji (Dropdown)
card_profile_rels = SettingsCard(profile_options_frame, "Wybór relacji", expanded=True, theme=THEMES[motyw.get()], registry=settings_cards)
profile_rels_frame = card_profile_rels.content

ctk.CTkLabel(profile_rels_frame, text="Kategoria składniowa:", font=("Verdana", 11, 'bold')).pack(anchor="w", pady=(5, 2))

profile_rel_var = ctk.StringVar(value="Brak danych")

# ZAMIENIAMY CTkOptionMenu NA CTkButton imitujący dropdown!
profile_rel_menu_btn = ctk.CTkButton(
    profile_rels_frame,
    text="Brak danych ▼",
    font=("Verdana", 11, "bold"),
    fg_color="#4B6CB7",
    hover_color="#5B7CD9",
    text_color="white",
    anchor="w",
    state="disabled"
)
profile_rel_menu_btn.pack(fill="x", pady=(0, 10))

# Dynamiczna zmiana napisu na przycisku po wybraniu opcji
def _update_profile_btn_text(*args):
    val = profile_rel_var.get()
    # Skracamy tekst, żeby nie wypychał panelu na boki
    disp = val if len(val) < 25 else val[:22] + "..."
    profile_rel_menu_btn.configure(text=f"{disp} ▼")

profile_rel_var.trace_add("write", _update_profile_btn_text)

# Pomocnicza funkcja generująca drzewo (kaskadowe menu) dla Profilu Składniowego
def build_profile_tree_menu(options_list, display_to_key_map, on_select_callback):
    current_theme = THEMES[motyw.get()]
    tree_menu = tk.Menu(profile_rel_menu_btn, tearoff=0,
                        bg=current_theme["dropdown_fg"],
                        fg=current_theme["button_text"],
                        activebackground=current_theme["dropdown_hover"],
                        activeforeground=current_theme["button_text"],
                        font=("Verdana", 11))
    tree_menu.add_command(
        label="★ Podsumowanie profilu",
        font=("Verdana", 11, "bold"),
        command=lambda: on_select_callback("★ Podsumowanie profilu")
    )
    tree_menu.add_separator()

    # --- WEWNĘTRZNA FUNKCJA AGREGUJĄCA ---
    def on_group_select(group_name, items_list):
        """Łączy dane z wielu relacji w jedną tabelę z dodatkowym tagiem relacji."""
        import re
        all_merged_rows = []
        for opt in items_list:
            actual_key = display_to_key_map.get(opt)
            if actual_key in current_profile_dict:
                all_merged_rows.extend(current_profile_dict[actual_key])

        # Sortowanie po sile związku (Log-Dice)
        all_merged_rows.sort(key=lambda r: (r.log_dice, r.cooc_freq), reverse=True)

        table_rows = []
        for i, row_obj in enumerate(all_merged_rows):
            display_colloc = row_obj.collocate

            # 1. Dodaj UPOS (np. [NOUN])
            if row_obj.collocate_upos:
                display_colloc = f"{display_colloc} [{row_obj.collocate_upos}]"

            # 2. DODAJ TYP RELACJI (np. [obj])
            # Wyciągamy tekst z nawiasu w nazwie relacji (np. "Dopełnienie (obj)" -> "obj")
            rel_match = re.search(r'\(([^)]+)\)', row_obj.relation)
            if rel_match:
                rel_tag = rel_match.group(1)
                display_colloc = f"{display_colloc} [{rel_tag}]"

            table_rows.append([
                i + 1, display_colloc, row_obj.cooc_freq, row_obj.doc_freq,
                row_obj.global_freq, row_obj.ll_score, row_obj.mi_score,
                row_obj.t_score, row_obj.log_dice
            ])

        paginator_profile["data"] = table_rows
        paginator_profile["current_page"][0] = 0
        update_table(paginator_profile)
        profile_rel_var.set(f"★ {group_name} (zbiorcze)")

    # Grupowanie opcji (logika get_group zostaje bez zmian)
    def get_group(name):
        n = name.lower()

        # 1. Węzły nadrzędne MUSZĄ być pierwsze!
        # Zabezpiecza to "Czynności, których jest podmiotem/dopełnieniem" przed wpadnięciem do grupy 1 lub 2
        if any(x in n for x in ["modyfikowane", "czynności, których"]):
            return "7. Węzły nadrzędne (Co określa?)"

        # 2. Zwrotność
        if "się" in n:
            return "8. Zwrotność (się)"

        # 3. Konstrukcje złożone i nazwy
        if any(x in n for x in ["wielowyrazowe", "złożenia", "człon", "flat", "fixed", "compound", "apozycj"]):
            return "6. Konstrukcje złożone i nazwy"

        # 4. Porównania
        if any(x in n for x in ["porównan", "punkt odniesienia"]):
            return "4. Porównania"

        # 5. Związki zdaniowe i szeregi (poprawiono "paratax" na "paratak")
        if any(x in n for x in ["zdaniow", "dołączenia", "paratak", "szereg", "współrzędne", "przydawkow"]):
            return "5. Związki zdaniowe i szeregi"

        # 6. Podmioty (teraz w 100% bezpieczne)
        if "podmiot" in n:
            return "1. Podmioty"

        # 7. Argumenty
        if any(x in n for x in ["argument", "dopełnien", "orzecznik"]):
            return "2. Argumenty (frazy wymagane)"

        # 8. Modyfikatory (dodano "operator", "agens", "połączenia z przyimkiem", zaimki skrócono do "zaim")
        if any(x in n for x in ["modyfikator", "okolicznik", "określnik", "przysłówek", "zaim", "przyimkow",
                                "intensyfikator", "operator", "agens"]):
            return "3. Modyfikatory (frazy niewymagane)"

        return "9. Pozostałe"

    grouped_options = {}
    for opt in options_list:
        actual_name = display_to_key_map.get(opt, opt.rsplit(" (", 1)[0])
        group = get_group(actual_name)
        grouped_options.setdefault(group, []).append(opt)

    for group_name in sorted(grouped_options.keys()):
        items = grouped_options[group_name]
        sub_menu = tk.Menu(tree_menu, tearoff=0,
                           bg=current_theme["dropdown_fg"], fg=current_theme["button_text"],
                           activebackground=current_theme["dropdown_hover"],
                           activeforeground=current_theme["button_text"],
                           font=("Verdana", 11))

        # --- NOWA OPCJA: Pokaż wszystkie z tej kategorii ---
        sub_menu.add_command(
            label=f"★ POKAŻ WSZYSTKIE ({len(items)})",
            font=("Verdana", 11, "bold"),
            command=lambda gn=group_name, it=items: on_group_select(gn, it)
        )
        sub_menu.add_separator()

        for opt in sorted(items):
            sub_menu.add_command(label=opt, command=lambda o=opt: on_select_callback(o))

        tree_menu.add_cascade(label=f"{group_name}", menu=sub_menu)

    def show_tree_menu(event=None):
        if profile_rel_menu_btn.cget("state") != "disabled":
            x = profile_rel_menu_btn.winfo_rootx()
            y = profile_rel_menu_btn.winfo_rooty() + profile_rel_menu_btn.winfo_height()
            tree_menu.tk_popup(x, y)

    profile_rel_menu_btn.configure(command=show_tree_menu)

# Zmienna globalna dla UI Profil kolokacyjny
current_profile_dict = {}

# Tabela dla Profil kolokacyjny
profile_headers = ["Nr", "Kolokat", "Współwyst.", "Zasięg (Dok.)", "Freq. Glob.", "Log-Likelihood", "MI Score", "T-score", "Log-Dice"]
profile_widths = [40, 150, 90, 100, 90, 110, 80, 80, 80]
profile_justify = ["center"] * 9
profile_data = []

pagination_profile_frame = ctk.CTkFrame(profile_data_frame, fg_color="#1F2328", corner_radius=12)
pagination_profile_frame.pack(fill="x", pady=(0, 5))
for col in range(5): pagination_profile_frame.columnconfigure(col, weight=1)

button_first_profile = ctk.CTkButton(pagination_profile_frame, text="|<", command=lambda: first_p(paginator_profile), **button_kwargs_small)
button_first_profile.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
button_prev_profile = ctk.CTkButton(pagination_profile_frame, text="<", command=lambda: prev_p(paginator_profile), **button_kwargs_small)
button_prev_profile.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
page_label_profile = ctk.CTkLabel(pagination_profile_frame, text="1/1", **label_kwargs_small)
page_label_profile.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
button_next_profile = ctk.CTkButton(pagination_profile_frame, text=">", command=lambda: next_p(paginator_profile), **button_kwargs_small)
button_next_profile.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
button_last_profile = ctk.CTkButton(pagination_profile_frame, text=">|", command=lambda: last_p(paginator_profile), **button_kwargs_small)
button_last_profile.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

profile_table = table.CustomTable(
    profile_data_frame, profile_headers, profile_data, profile_widths, profile_justify, 15,
    fulltext_data=[], search_callback=lambda w: search_from_table_profile(w)
)
profile_table.pack(fill="both", expand=True)

paginator_profile = {
    "data": profile_data,
    "current_page": [0],
    "table": profile_table,
    "label": page_label_profile,
    "items_per_page": 15
}
profile_table.sort_callback = lambda col, asc: global_sort_callback(paginator_profile, col, asc)

# =======================================================
# --- NOWOŚĆ: DASHBOARD DLA PROFILU (WORD SKETCH) ---
profile_dashboard_frame = ctk.CTkScrollableFrame(profile_data_frame, fg_color="transparent")


# (Nie pakujemy na starcie - będzie pokazane zamiennie z tabelą)

def render_profile_dashboard(on_select_callback):
    for widget in profile_dashboard_frame.winfo_children():
        widget.destroy()

    theme = THEMES[motyw.get()]
    row, col = 0, 0
    max_cols = 3  # Liczba kolumn kafelków

    profile_dashboard_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")

    # Sortujemy kategorie malejąco po liczbie unikalnych kolokatów
    sorted_relations = sorted(current_profile_dict.items(), key=lambda x: len(x[1]), reverse=True)

    for relation_key, rows in sorted_relations:
        if not rows: continue

        # Odtwarzamy oryginalną nazwę z menu (z liczbą) by przycisk wiedział, co kliknąć
        display_name = f"{relation_key} ({len(rows)})"

        card = ctk.CTkFrame(profile_dashboard_frame, corner_radius=8, fg_color=theme["subframe_fg"], border_width=1,
                            border_color="#3E3F42")
        card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

        lbl_title = ctk.CTkLabel(card, text=relation_key, font=("Verdana", 11, "bold"),
                                 text_color=theme.get("button_fg", "#4B6CB7"), wraplength=200)
        lbl_title.pack(pady=(10, 5), padx=10)

        lbl_subtitle = ctk.CTkLabel(card, text="Top 5 (wg Log-Dice)", font=("Verdana", 9, "italic"),
                                    text_color="gray50")
        lbl_subtitle.pack(pady=(0, 5))

        list_frame = ctk.CTkFrame(card, fg_color="transparent")
        list_frame.pack(fill="both", expand=True, padx=10)

        # Wyświetlamy Top 5 posortowane już przez Log-Dice
        for i, item in enumerate(rows[:5]):
            item_row = ctk.CTkFrame(list_frame, fg_color="transparent")
            item_row.pack(fill="x", pady=2)

            colloc_str = item.display_collocate
            if item.collocate_upos: colloc_str += f" [{item.collocate_upos}]"
            # Ucięcie za długich wyrazów
            if len(colloc_str) > 20: colloc_str = colloc_str[:17] + "..."

            ctk.CTkLabel(item_row, text=f"{i + 1}. {colloc_str}", font=("Verdana", 11)).pack(side="left")
            ctk.CTkLabel(item_row, text=f"{item.log_dice:.1f} LD", font=("Verdana", 10, "bold"), text_color=theme.get("label_text", "white")).pack(side="right")

        # Przycisk "Pełna lista" wywołujący funkcję przeskoku
        btn_details = ctk.CTkButton(
            card, text="Pełna lista ➔", font=("Verdana", 11, "bold"), height=24,
            fg_color="transparent", border_width=1, border_color=theme["button_fg"], text_color=theme["label_text"],
            hover_color=theme["button_hover"],
            command=lambda dn=display_name: on_select_callback(dn)
        )
        btn_details.pack(pady=(10, 10))

        col += 1
        if col >= max_cols:
            col = 0
            row += 1


# =======================================================


def calculate_word_profile():
    if not full_results_sorted:
        messagebox.showinfo("Brak", "Najpierw wyszukaj frazę.")
        return

    # Odczytanie przesunięcia słowa centralnego (offsetu)
    node_selection = profile_node_var.get()

    match = re.search(r'Token (\d+)', node_selection)
    node_offset = (int(match.group(1)) - 1) if match else 0

    target_lemmas_count = Counter()
    for res in full_results_sorted:
        lemmas = str(res[4]).split()
        if node_offset < len(lemmas):
            target_lemmas_count[lemmas[node_offset]] += 1

    if not target_lemmas_count:
        messagebox.showerror("Błąd", "Nie udało się ustalić lematu dla tego przesunięcia.")
        return

    target_lemma = target_lemmas_count.most_common(1)[0][0]

    # Zapisz w pamięci globalnej do klikania tabeli
    global current_profile_target_lemma
    current_profile_target_lemma = target_lemma

    try:
        min_f = int(entry_profile_minf.get() or "2")
    except ValueError:
        min_f = 2

    # --- POBIERAMY STAN CHECKBOXA ---
    ignore_case_val = profile_ignore_case_var.get()

    btn_calc_profile.configure(state="disabled", text="Generowanie...")

    def worker():
        try:
            df = dataframes[global_selected_corpus]
            inv_idx = inverted_indexes[global_selected_corpus]
            token_freq_dict_raw = inv_idx['base_tf']
            total_tokens_val = inv_idx.get('total_tokens', 1)

            # --- AGREGACJA FREKWENCJI GLOBALNEJ W ZALEŻNOŚCI OD WIELKOŚCI LITER ---
            if ignore_case_val:
                token_freq_dict = {}
                for k, v in token_freq_dict_raw.items():
                    kl = str(k).lower()
                    token_freq_dict[kl] = token_freq_dict.get(kl, 0) + v
            else:
                token_freq_dict = token_freq_dict_raw

            # Profil kolokacyjny oczekuje w res[12] indeksu tokenu w dokumencie.
            # Nie wolno przesuwać res[12] o node_offset, bo wtedy target_idx
            # wychodzi poza zakres list tokenowych/upostags.
            adjusted_results = list(full_results_sorted)

            # Wywołanie funkcji z przekazaniem flagi ignore_case
            mwe_val = profile_mwe_var.get()

            # KORPUSUJ_MIGRATION_036I_WORD_PROFILE_PROVIDER_TIMINGS
            profile_df = df
            profile_provider = None
            _profile_provider_init_t0 = time.perf_counter()
            try:
                from korpusuj.semantic.profile_provider import build_profile_provider_for_results
                profile_provider = build_profile_provider_for_results(df, adjusted_results)
                if profile_provider is not None:
                    profile_df = profile_provider
            except Exception as _profile_provider_error:
                logging.warning("[APP semantic.profile.fallback] init_failed=%r", _profile_provider_error)
                profile_provider = None
                profile_df = df
            _profile_provider_init_elapsed = time.perf_counter() - _profile_provider_init_t0

            _word_profile_compute_t0 = time.perf_counter()
            try:
                profile_dict = compute_word_profile(
                    results=adjusted_results,
                    df=profile_df,
                    token_freq_dict=token_freq_dict,
                    target_lemma=target_lemma,
                    total_tokens=total_tokens_val,
                    min_freq=min_f,
                    ignore_case=ignore_case_val,
                    expand_mwe=mwe_val
                )
            except Exception as _profile_provider_compute_error:
                if profile_provider is not None:
                    logging.warning("[APP semantic.profile.fallback] compute_failed=%r", _profile_provider_compute_error)
                    try:
                        profile_provider.close()
                    except Exception:
                        pass
                    profile_provider = None
                    profile_df = df
                    _fallback_compute_t0 = time.perf_counter()
                    profile_dict = compute_word_profile(
                        results=adjusted_results,
                        df=df,
                        token_freq_dict=token_freq_dict,
                        target_lemma=target_lemma,
                        total_tokens=total_tokens_val,
                        min_freq=min_f,
                        ignore_case=ignore_case_val,
                        expand_mwe=mwe_val
                    )
                    korpusuj_verbose_diagnostics_enabled_145c1() and logging.info("[DIAG perf.semantic.profile] fallback_compute=%.6fs", time.perf_counter() - _fallback_compute_t0)
                else:
                    raise
            finally:
                _word_profile_compute_elapsed = time.perf_counter() - _word_profile_compute_t0
                if profile_provider is not None:
                    try:
                        _diag = profile_provider.diagnostics()
                    except Exception:
                        _diag = {}
                    if korpusuj_verbose_diagnostics_enabled_145c1():
                        logging.info("[DIAG perf.semantic.profile] event='timing' init=%.6fs compute=%.6fs stats=%r timings=%r", _profile_provider_init_elapsed, _word_profile_compute_elapsed, _diag.get("stats"), _diag.get("timings"))
                    try:
                        profile_provider.close()
                    except Exception:
                        pass
                else:
                    if korpusuj_verbose_diagnostics_enabled_145c1():
                        logging.info("[DIAG perf.semantic.profile] event='provider_disabled' init=%.6fs compute=%.6fs", _profile_provider_init_elapsed, _word_profile_compute_elapsed)
            # END KORPUSUJ_MIGRATION_036I_WORD_PROFILE_PROVIDER_TIMINGS

            def update_ui():
                global current_profile_dict
                current_profile_dict = profile_dict

                if not profile_dict:
                    profile_rel_menu_btn.configure(state="disabled")
                    profile_rel_var.set("Brak wyników")
                    paginator_profile["data"] = []
                    update_table(paginator_profile)
                    btn_calc_profile.configure(state="normal", text="Generuj")
                    with state_lock:
                        current_state.current_profile_dict = {}
                        current_state.profile_data = []
                        current_state.profile_rel_options = ["Brak wyników"]
                        current_state.profile_selected_rel = "Brak wyników"
                    return

                options = []
                display_to_key = {}
                for rel_name in sorted(profile_dict.keys()):
                    rows = profile_dict[rel_name]
                    display_name = f"{rel_name} ({len(rows)})"
                    options.append(display_name)
                    display_to_key[display_name] = rel_name

                profile_rel_menu_btn.configure(state="normal")

                def on_rel_select(selected_display_name):
                    profile_rel_var.set(selected_display_name)

                    # LOGIKA 1: Widok z lotu ptaka
                    if selected_display_name == "★ Podsumowanie profilu":
                        pagination_profile_frame.pack_forget()
                        profile_table.pack_forget()
                        profile_dashboard_frame.pack(fill="both", expand=True)
                        render_profile_dashboard(on_rel_select)
                        return

                    # LOGIKA 2: Standardowa tabela dla wybranej relacji
                    profile_dashboard_frame.pack_forget()
                    pagination_profile_frame.pack(fill="x", pady=(0, 5))
                    profile_table.pack(fill="both", expand=True)

                    actual_key = display_to_key.get(selected_display_name)
                    if not actual_key: return

                    rows = current_profile_dict[actual_key]
                    table_rows = []
                    for i, row_obj in enumerate(rows):
                        display_colloc = row_obj.collocate
                        if row_obj.collocate_upos:
                            display_colloc = f"{display_colloc} [{row_obj.collocate_upos}]"

                        table_rows.append([
                            i + 1, display_colloc, row_obj.cooc_freq, row_obj.doc_freq,
                            row_obj.global_freq, row_obj.ll_score, row_obj.mi_score,
                            row_obj.t_score, row_obj.log_dice
                        ])

                    paginator_profile["data"] = table_rows
                    paginator_profile["current_page"][0] = 0
                    update_table(paginator_profile)

                    with state_lock:
                        current_state.current_profile_dict = dict(current_profile_dict)
                        current_state.profile_target_lemma = current_profile_target_lemma
                        current_state.profile_data = list(table_rows)
                        current_state.profile_rel_options = list(options)
                        current_state.profile_selected_rel = selected_display_name

                # Generowanie i przypinanie rozwijanego DRZEWA do przycisku
                build_profile_tree_menu(options, display_to_key, on_rel_select)

                # Ustawiamy domyślnie widok Word Sketch!
                first_option = "★ Podsumowanie profilu"
                profile_rel_var.set(first_option)
                on_rel_select(first_option)

                btn_calc_profile.configure(state="normal", text="Generuj")

            app.after(0, update_ui)


        except Exception as e:
            logging.exception("Błąd profilu")
            error_msg = str(e)  # <--- Zapisujemy błąd do trwałego stringa

            def on_error(msg=error_msg):  # <--- Przekazujemy go w bezpieczny sposób
                btn_calc_profile.configure(state="normal", text="Generuj")
                messagebox.showerror("Błąd profilu", f"Wystąpił błąd:\n{msg}")

            app.after(0, on_error)

    threading.Thread(target=worker, daemon=True).start()

btn_calc_profile.configure(command=calculate_word_profile)



# ------------------------------
# Plots
# ------------------------------
# Główny kontener opcji na lewo od wykresu:
plot_options_frame = ctk.CTkScrollableFrame(tab_wyniki_wykresy, fg_color="transparent", corner_radius=0, width=280)
plot_options_frame.pack(pady=10, padx=(10, 5), side="left", fill="y")

# Karta 1: Typ i zapis wykresu
card_type = SettingsCard(plot_options_frame, "Typ i zapis wykresu", expanded=True, theme=THEMES[motyw.get()], registry=settings_cards)
saveplot_button_frame = card_type.content

# --- RAMKA Z TYTUŁEM I CHMURKĄ (TOOLTIP) ---
type_label_frame = ctk.CTkFrame(saveplot_button_frame, fg_color="transparent")
type_label_frame.pack(pady=(5, 0), padx=5, fill="x")

plot_type_label = ctk.CTkLabel(type_label_frame, text="Wybierz typ wykresu:", font=("Verdana", 13, 'bold'))
plot_type_label.pack(side="left", padx=(5, 5))

plot_help_icon = ctk.CTkLabel(type_label_frame, text="❓", font=("Verdana", 14), text_color="#4B6CB7", cursor="hand2")
plot_help_icon.pack(side="left")

help_text = (
    "TYPY WYKRESÓW:\n"
    "• Liczba wystąpień: Surowa liczba trafień w danym okresie.\n"
    "• Częstość względna: Liczba trafień znormalizowana na 1 000 000 słów.\n"
    "• TF-IDF: Miara specyficzności słowa w danym okresie.\n"
    "• Z-score: Miara dynamiki zmian względem średniej dla danego słowa.\n\n"
    "Aplikacja może nie wyświetlać punktów na wykresie w okresach,\n"
    "w których brakuje danych lub ich liczba jest zbyt mała,\n"
    "aby wynik był statystycznie wiarygodny.\n"
    "W trybie Auto próg ten jest wyznaczany automatycznie.\n"
    "Można go zmienić lub wyłączyć w zakładce Opcje."
)

ToolTip(plot_help_icon, help_text)

wykres_sort_mode = ctk.StringVar(value="Liczba wystąpień")

plot_type_menu = ctk.CTkOptionMenu(
    saveplot_button_frame,
    variable=wykres_sort_mode,
    values=["Liczba wystąpień", "Częstość względna", "TF-IDF", "Z-score"],
    font=("Verdana", 12, 'bold'),
    fg_color="#4B6CB7", dropdown_fg_color="#4B6CB7", dropdown_hover_color="#5B7CD9",
    command=lambda _: force_recalculate_plot()
)
plot_type_menu.pack(pady=5, padx=5, fill="x")

# Karta 2: Opcje Dat i Czasu
card_date = SettingsCard(plot_options_frame, "Filtrowanie czasowe", expanded=False, theme=THEMES[motyw.get()], registry=settings_cards)
date_settings_frame = card_date.content

custom_date_var = ctk.BooleanVar(value=False)

def toggle_custom_dates():
    state = "normal" if custom_date_var.get() else "disabled"
    date_start_entry.configure(state=state)
    date_end_entry.configure(state=state)
    force_recalculate_plot()

dates_header_frame = ctk.CTkFrame(date_settings_frame, fg_color="transparent")
dates_header_frame.pack(pady=(10, 2), fill="x", padx=10)

chk_custom_dates = ctk.CTkCheckBox(dates_header_frame, text="Niestandardowy zakres dat",
                                   variable=custom_date_var, command=toggle_custom_dates,
                                   font=("Verdana", 11, "bold"))
chk_custom_dates.pack(side="left")

date_help_icon = ctk.CTkLabel(dates_header_frame, text="❓", font=("Verdana", 14), text_color="#4B6CB7", cursor="hand2")
date_help_icon.pack(side="left", padx=5)

date_help_text = (
    "Przy obliczaniu miar znormalizowanych (PMW, TF-IDF), system opiera się na\n"
    "sumarycznej objętości tekstów zliczonej w skali miesięcy. Jeśli zostanie zdefiniowany\n"
    "interwał mniejszy niż miesiąc (np. dni) lub ramy czasowe przecinające miesiąc\n"
    "w połowie, aplikacja stosuje podział proporcjonalny (np. dla 10 dni marca przyjmie\n"
    "do obliczeń ok. 32% całkowitej liczby słów z tego miesiąca).\n\n"
    "Należy pamiętać, że w takich przypadkach wykres prezentuje uśrednione\n"
    "przybliżenie statystyczne, a nie rzeczywistą, punktową frekwencję z każdego dnia."
)

ToolTip(date_help_icon, date_help_text)

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

# Karta 3: Skalowanie
card_scale = SettingsCard(plot_options_frame, "Skalowanie osi Y", expanded=False, theme=THEMES[motyw.get()], registry=settings_cards)
scale_frame = card_scale.content

ctk.CTkLabel(scale_frame, text="Skalowanie:", font=("Verdana", 11, "bold")).pack(side="left", padx=10, pady=5)

scale_mode_var = ctk.StringVar(value="Auto")

def on_scale_mode_change(value):
    if value == "Ręczne":
        entry_y_limit.pack(side="left", padx=(5, 10), pady=5)
    else:
        entry_y_limit.pack_forget()
        entry_y_limit.delete(0, 'end')
        force_recalculate_plot()

scale_mode_btn = ctk.CTkSegmentedButton(
    scale_frame,
    values=["Auto", "Ręczne"],
    variable=scale_mode_var,
    command=on_scale_mode_change,
    font=("Verdana", 11, "bold"),
    fg_color="#2C2F33",
    selected_color="#4B6CB7",
    selected_hover_color="#5B7CD9"
)
scale_mode_btn.pack(side="left", padx=(0, 5), pady=5)

entry_y_limit = ctk.CTkEntry(scale_frame, placeholder_text="Górny limit...", width=100, height=28)

# Karta 4: Wybór elementów na wykresie
# expand_card=True mówi systemowi, że ta karta może rosnąć (ponieważ ma wewnętrzny scroll listboxów z lematami)
card_checkboxes = SettingsCard(plot_options_frame, "Zaznaczone elementy", expanded=True, expand_card=True, theme=THEMES[motyw.get()], registry=settings_cards)
checkboxes_frame = card_checkboxes.content

# Przycisk "Zapisz wykres" pakowany do Pierwszej Karty
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
# --- Pomocnicze funkcje dla głównego pola wyszukiwania ---
def on_enter_query(event):
    on_enter(event)   # Uruchamia Twoje wyszukiwanie
    return "break"    # Blokuje wstawienie nowej linii

def insert_newline(event):
    # Ręcznie wstawia nową linię w miejscu kursora
    event.widget.insert("insert", "\n")
    return "break"    # Blokuje inne domyślne akcje
# ---------------------------------------------------------

# Przypisanie Enter do pola wpisywania lematu i całej aplikacji
entry_query.bind("<Return>", on_enter_query)           # Enter = Szukaj
entry_query.bind("<Shift-Return>", insert_newline)     # Shift+Enter = Nowa linia

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
# Installed before the __main__ guard so normal GUI script execution sees it
# before main()/app.mainloop() starts.
def _materialize_searchcursor_rows_via_cursor(results, *, cancel_check=None, search_token=None, chunk_size=1000):
    t_materialize_start_035d = time.perf_counter()
    rows = []
    cancelled = False

    def _cancelled():
        try:
            return bool(cancel_check and cancel_check())
        except Exception:
            return False

    try:
        total = len(results)
    except Exception:
        try:
            total = int(results.count_hits(exact=True))
        except Exception:
            total = 0

    try:
        chunk_size = int(chunk_size or 1000)
    except Exception:
        chunk_size = 1000
    chunk_size = max(1, chunk_size)

    try:
        get_range = getattr(results, "get_range", None)
        if callable(get_range):
            start = 0
            while start < total:
                if _cancelled():
                    cancelled = True
                    break
                stop = min(total, start + chunk_size)
                part = get_range(start, stop)
                if part:
                    rows.extend(part)
                start = stop
        else:
            for i in range(total):
                if _cancelled():
                    cancelled = True
                    break
                rows.append(results[i])
    except Exception:
        try:
            from korpusuj.search.result_materialization import materialize_searchcursor_results_036l4g48e
            payload = materialize_searchcursor_results_036l4g48e(
                results,
                cancel_check=cancel_check,
                search_token=search_token,
                logger=logging,
                perf_counter=time.perf_counter,
            )
            try:
                payload["materializer"] = "fallback_legacy_materializer_after_cursor_materialization_error"
            except Exception:
                pass
            return payload
        except Exception:
            raise

    t_materialize_done_035d = time.perf_counter()
    return {
        "results": rows,
        "cancelled": cancelled,
        "search_token": search_token,
        "t_materialize_start_035d": t_materialize_start_035d,
        "t_materialize_done_035d": t_materialize_done_035d,
        "materializer": "cursor_get_range_text_offsets_policy",
        "materialized_rows": len(rows),
        "materialized_total_requested": total,
    }


def _install_materialize_searchcursor_table_context_text_offsets():
    current = globals().get("_materialize_searchcursor_results_with_cancel_check")
    if getattr(current, "_table_context_text_offsets_wrapped", False):
        return

    def _materialize_searchcursor_results_with_cancel_check_cursor_rows(results, *, cancel_check=None, search_token=None):
        return _materialize_searchcursor_rows_via_cursor(
            results,
            cancel_check=cancel_check,
            search_token=search_token,
        )

    _materialize_searchcursor_results_with_cancel_check_cursor_rows._table_context_text_offsets_wrapped = True
    _materialize_searchcursor_results_with_cancel_check_cursor_rows._table_context_text_offsets_original = current
    globals()["_materialize_searchcursor_results_with_cancel_check"] = _materialize_searchcursor_results_with_cancel_check_cursor_rows

try:
    _install_materialize_searchcursor_table_context_text_offsets()
except Exception:
    pass

if __name__ == "__main__":
    main()
