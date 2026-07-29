from korpusuj.runtime_paths import models_root, writable_temp_root
import os
import sys
import logging
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import re
import json
import pandas as pd
from collections import Counter
import customtkinter as ctk
import tkinter.filedialog as fd
import threading
from docx import Document
import time
from tkinter import messagebox
import sys
import zipfile
import tempfile
import requests
import shutil
from korpusuj.runtime_paths import configure_ml_cache_environment as _configure_ml_cache_environment_182n
_configure_ml_cache_environment_182n()

import spacy
import spacy.cli
import pyarrow as pa
import pyarrow.parquet as pq
import gc
import torch
from datetime import datetime
from pathlib import Path
# --- IMPORTY WARUNKOWE ---
try:
    import stanza
except ImportError:
    stanza = None
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import typing
    import torch.utils.data.dataset

    if not hasattr(torch.utils.data.dataset, 'T_co'):
        torch.utils.data.dataset.T_co = typing.TypeVar('T_co', covariant=True)
    import herference
    #logging.info("SUKCES: Herference zaimportowane pomyślnie na górze pliku!")
except Exception as e:
    herference = None
    messagebox.showerror("Błąd importu Herference", f"Nie udało się zaimportować biblioteki herference:\n\n{e}")

#sys.stdout.reconfigure(encoding='utf-8', errors='replace')

nlp_stanza = None
nlp_spacy = None

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)


def get_application_root():
    """Return the writable application root for external runtime assets."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


APPLICATION_ROOT = get_application_root()
MODELS_DIR = models_root()

selected_files = {}
file_buttons = []


FILES_PER_PAGE = 100
file_page_index = 0

pagination_label = None
pagination_prev_button = None
pagination_next_button = None


# --- HELPER: FORMATOWANIE ROZMIARU (KB/MB) ---
def format_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = 0
    p = 1024
    s = size_bytes
    while s >= p and i < len(size_name) - 1:
        s /= p
        i += 1
    return f"{s:.2f} {size_name[i]}"


# --- UPDATED COLUMN MAPPER CLASS ---
class ColumnMapper(ctk.CTkToplevel):
    def __init__(self, parent, filename, columns, is_metadata=False):
        super().__init__(parent)
        self.title(f"Mapowanie: {filename}")
        self.result = None
        self.columns_options = ["<Pomiń>"] + list(columns)
        self.attributes("-topmost", True)

        # UI COLORS & FONTS (Matching Main Window)
        THEME_COLOR = "#4B6CB7"
        HOVER_COLOR = "#5B7CD9"
        FONT_BOLD = ("Verdana", 12, "bold")
        FONT_NORMAL = ("Verdana", 12)

        header_text = f"Skonfiguruj kolumny dla metadanych:\n{filename}" if is_metadata else f"Skonfiguruj kolumny dla pliku:\n{filename}"

        ctk.CTkLabel(self, text=header_text, font=("Verdana", 14, "bold")).pack(pady=10, padx=20)

        ctk.CTkLabel(self, text="Wskaż odpowiedniki kolumn.\n'Nazwa pliku' jest wymagana.",
                     font=("Verdana", 11)).pack(pady=(0, 15))

        self.vars = {}

        # Define fields based on whether this is a metadata file or a content file
        # Format: (Field Name, Is Required)
        self.fields_config = [
            ("Nazwa pliku", True),
            ("Tytuł", False),
            ("Data publikacji", False),
            ("Autor", False)
        ]

        # Only add "Treść" if it is NOT a metadata file
        if not is_metadata:
            self.fields_config.insert(2, ("Treść", False))

        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True, padx=10, pady=5)

        for field, is_required in self.fields_config:
            row = ctk.CTkFrame(self.container, fg_color="transparent")
            row.pack(fill="x", padx=5, pady=5)

            label_text = field + (" *" if is_required else "")
            lbl = ctk.CTkLabel(row, text=label_text, width=130, anchor="w",
                               font=FONT_BOLD if is_required else FONT_NORMAL)
            lbl.pack(side="left", padx=5)

            guessed = self.guess_column(field, columns)
            var = ctk.StringVar(value=guessed)
            self.vars[field] = var

            dropdown = ctk.CTkOptionMenu(
                row,
                values=self.columns_options,
                variable=var,
                font=FONT_NORMAL,
                fg_color=THEME_COLOR,
                button_color=THEME_COLOR,
                button_hover_color=HOVER_COLOR,
                dropdown_fg_color=THEME_COLOR,
                dropdown_hover_color=HOVER_COLOR
            )
            dropdown.pack(side="right", expand=True, fill="x", padx=5)

        save_btn = ctk.CTkButton(
            self,
            text="Zatwierdź",
            command=self.on_confirm,
            fg_color=THEME_COLOR,
            hover_color=HOVER_COLOR,
            font=FONT_BOLD,
            height=35
        )
        save_btn.pack(pady=20)

        self.update_idletasks()
        # Adjust height based on number of fields
        width = 450
        height = 350 if is_metadata else 400
        x = int(parent.winfo_x() + (parent.winfo_width() / 2) - (width / 2))
        y = int(parent.winfo_y() + (parent.winfo_height() / 2) - (height / 2))
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def guess_column(self, field, cols):
        field_lower = field.lower()
        mapping = {
            "nazwa pliku": ["filename", "nazwa pliku", "plik", "id", "file", "name"],
            "tytuł": ["title", "tytuł", "headline", "nazwa", "header"],
            "treść": ["content", "text", "body", "treść", "tekst", "artykuł", "opis"],
            "data publikacji": ["date", "data", "published", "created", "czas"],
            "autor": ["author", "autor", "twórca", "by"]
        }
        candidates = mapping.get(field_lower, [])
        cols_lower = [str(c).lower() for c in cols]

        for cand in candidates:
            if cand in cols_lower:
                idx = cols_lower.index(cand)
                return self.columns_options[idx + 1]
        return "<Pomiń>"

    def on_confirm(self):
        filename_col = self.vars["Nazwa pliku"].get()
        if filename_col == "<Pomiń>":
            messagebox.showwarning("Błąd mapowania", "Pole 'Nazwa pliku' jest wymagane!\nWybierz odpowiednią kolumnę.")
            return

        self.result = {k: v.get() for k, v in self.vars.items()}
        self.destroy()


# --- CHUNKING HELPERS: LOSSLESS + MULTI-STYLE RECORD-AWARE ---
# Extracted to a GUI-free shared module; imported here to preserve the existing
# creator.py runtime/API names used by the current CustomTkinter creator UI.
from korpusuj.corpus.creator_chunking import (
    BULLET_RECORD_START_RE,
    FIELD_BREAK_RE,
    IMPLIED_RECORD_BREAK_RE,
    LETTER_RECORD_START_RE,
    NUMERIC_RECORD_START_RE,
    ROMAN_RECORD_START_RE,
    SOFT_BREAK_RE,
    STRONG_SENT_END_RE,
    chunk_structured_records,
    chunk_text_safe,
    detect_record_style,
    get_record_start_regex,
    has_multiline_bullet_layout,
    soft_cut_preserve,
    split_structured_segments,
)


# --- GUI-free near-pure creator IO helpers (track 172f2) ---
from korpusuj.corpus.creator_io import (
    calculate_real_total_size,
    process_xlsx,
)


def unpack_archive(file_path, status_label):
    temp_dir = tempfile.mkdtemp(prefix="archive_extract_", dir=str(writable_temp_root()))
    extracted_files = []
    try:
        if file_path.lower().endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        extracted_files.append(os.path.join(root, file))
        status_label.configure(text=f"Rozpakowano archiwum: {os.path.basename(file_path)}")
        return extracted_files
    except Exception as e:
        status_label.configure(text=f"Błąd rozpakowywania: {e}")
        return []


def process_pdf(file_path, status_label, app):
    if fitz is None:
        return ""
    text = ""
    try:
        pdf_doc = fitz.open(file_path)
        reader = None
        for page in pdf_doc:
            text_page = page.get_text("text")
            if text_page.strip():
                text += text_page
            else:
                if easyocr is not None:
                    if reader is None:
                        easyocr_dir = MODELS_DIR / "easyocr"
                        easyocr_dir.mkdir(parents=True, exist_ok=True)
                        reader = easyocr.Reader(
                            ['pl'],
                            gpu=bool(torch.cuda.is_available()),
                            model_storage_directory=str(easyocr_dir),
                        )
                    status_label.configure(text=f"OCR: {os.path.basename(file_path)} str {page.number + 1}")
                    app.update_idletasks()
                    pix = page.get_pixmap()
                    result = reader.readtext(pix.tobytes("png"), detail=0)
                    text += " ".join(result) + " "
    except Exception as e:
        logging.warning(f"Błąd PDF: {e}")
        pass
    return text.replace('-\n', '').replace('\n', ' ')


def update_status(label, text, app):
    app.after(0, lambda: label.configure(text=text))
    app.update_idletasks()


# --- Stateful NLP initializer dependencies (track 172g2a repair) ---
from korpusuj.corpus.creator_core import ProgressReporter
from korpusuj.corpus.creator_core import CreatorRunOptions
from korpusuj.corpus.creator_orchestration import run_creator_job
from korpusuj.corpus.creator_gui_adapter import GuiProgressReporter
from korpusuj.corpus.creator_nlp import (
    CreatorModelState,
    initialize_spacy as _initialize_spacy_stateful,
    initialize_stanza as _initialize_stanza_stateful,
    process_single_text as _process_single_text_stateful,
    process_single_text_spacy as _process_single_text_spacy_stateful,
)
# --- Legacy GUI compatibility wrappers over stateful NLP initializers (track 172g2) ---
_creator_model_state = CreatorModelState()


def _creator_gui_reporter(status_label, app):
    return GuiProgressReporter(
        app=app,
        status_label=status_label,
        show_warning=getattr(messagebox, "showwarning", None),
        show_error=getattr(messagebox, "showerror", None),
    )


def _sync_legacy_model_globals_from_state():
    global nlp_stanza, nlp_spacy
    nlp_stanza = _creator_model_state.nlp_stanza
    nlp_spacy = _creator_model_state.nlp_spacy


def initialize_stanza(status_label, app):
    result = _initialize_stanza_stateful(
        _creator_model_state,
        _creator_gui_reporter(status_label, app),
        stanza_module=stanza,
        models_dir=str(MODELS_DIR),
    )
    _sync_legacy_model_globals_from_state()
    return result


def initialize_spacy(status_label, app):
    result = _initialize_spacy_stateful(
        _creator_model_state,
        _creator_gui_reporter(status_label, app),
        spacy_module=spacy,
        herference_module=herference,
        requests_module=requests,
        models_dir=str(MODELS_DIR),
    )
    _sync_legacy_model_globals_from_state()
    return result


# --- INITIALIZATION ---

# KORPUSUJ_PATCH_138C_FIX_CREATOR_COREF_SPAN_MAPPING_AND_LABEL_DEDUP
# END KORPUSUJ_PATCH_138C_FIX_CREATOR_COREF_SPAN_MAPPING_AND_LABEL_DEDUP

# --- NLP PROCESSING ---
# --- HELPER DLA SPACY (ODTWARZANIE PEŁNYCH TAGÓW NKJP Z CECH MORFOLOGICZNYCH UD) ---
# --- Legacy GUI wrappers over stateful text processors (track 172g4) ---
def process_single_text(text, filename, status_label, progress_bar, app):
    reporter = GuiProgressReporter(
        app=app,
        status_label=status_label,
        progress_bar_current=progress_bar,
        show_warning=getattr(messagebox, "showwarning", None),
        show_error=getattr(messagebox, "showerror", None),
    )
    return _process_single_text_stateful(
        text, filename, _creator_model_state, reporter
    )


def process_single_text_spacy(text, filename, status_label, progress_bar, app):
    reporter = GuiProgressReporter(
        app=app,
        status_label=status_label,
        progress_bar_current=progress_bar,
        show_warning=getattr(messagebox, "showwarning", None),
        show_error=getattr(messagebox, "showerror", None),
    )
    return _process_single_text_spacy_stateful(
        text, filename, _creator_model_state, reporter
    )


def process_file_global(file_path, status_label, progress_bar, app, model_name, excel_mappings=None,
                        processed_set=None):
    ext = os.path.splitext(file_path)[1].lower()
    if processed_set is None: processed_set = set()

    try:
        current_file_size = os.path.getsize(file_path)
    except OSError:
        current_file_size = 0

    if ext == ".zip":
        status_label.configure(text=f"Rozpakowuję archiwum: {os.path.basename(file_path)}")
        extracted_files = unpack_archive(file_path, status_label)

        for inner_file in extracted_files:
            yield from process_file_global(inner_file, status_label, progress_bar, app, model_name, excel_mappings,
                                           processed_set)
            try:
                os.remove(inner_file)
            except:
                pass
        try:
            shutil.rmtree(os.path.dirname(extracted_files[0]))
        except:
            pass

    elif ext in [".txt", ".docx", ".pdf", ".xlsx"]:
        # --- EXCEL ---
        if ext == ".xlsx":
            mapping = excel_mappings.get(file_path) if excel_mappings else None
            rows = process_xlsx(file_path, mapping=mapping)
            total_rows = len(rows)
            bytes_per_row = current_file_size / total_rows if total_rows > 0 else 0

            for it in rows:
                virt_fname = it.get("filename", os.path.basename(file_path))
                title = it.get("Tytuł", "")

                v_lower = str(virt_fname).strip().lower()
                t_lower = str(title).strip().lower()

                # WZNAWIANIE: Jeśli wirtualny plik z Excela jest już zrobiony (szukamy po Nazwie lub po Tytule z Excela)
                if v_lower in processed_set or (t_lower and t_lower in processed_set):
                    yield {"skipped": True, "bytes_consumed": bytes_per_row, "filename": virt_fname}
                    continue

                text = it["Treść"]

                if model_name == "Stanza":
                    tokens = process_single_text(text, virt_fname, status_label, progress_bar, app)
                else:
                    tokens = process_single_text_spacy(text, virt_fname, status_label, progress_bar, app)

                if tokens:
                    yield {
                        "filename": virt_fname,
                        "Treść": text,
                        "tokens_detail": tokens,
                        "meta_override": it,
                        "bytes_consumed": bytes_per_row
                    }


        # --- PLIKI TEKSTOWE / PDF / DOCX ---
        else:
            file_base = os.path.basename(file_path)

            # WZNAWIANIE: Jeśli plik fizyczny jest już zrobiony
            if str(file_base).strip().lower() in processed_set:
                yield {"skipped": True, "bytes_consumed": current_file_size, "filename": file_base}
                return

            text = ""

            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            elif ext == ".docx":
                try:
                    text = "\n".join(p.text for p in Document(file_path).paragraphs)
                except:
                    pass
            elif ext == ".pdf":
                text = process_pdf(file_path, status_label, app)

            if text.strip():
                if model_name == "Stanza":
                    tokens = process_single_text(text, file_base, status_label, progress_bar, app)
                else:
                    tokens = process_single_text_spacy(text, file_base, status_label, progress_bar, app)

                if tokens:
                    yield {
                        "filename": file_base,
                        "Treść": text,
                        "tokens_detail": tokens,
                        "bytes_consumed": current_file_size
                    }


def _schedule_creator_completion(
        app, completion_callback, success, output_file=None, error_message=None):
    """Schedule one GUI-thread completion owner for a worker run."""
    if completion_callback is None:
        return
    app.after(
        0,
        lambda: completion_callback(
            success=success,
            output_file=output_file,
            error_message=error_message,
        ),
    )


# --- UPDATED WORKER FUNCTION ---
import glob


def process_files_thread_target(status_label, progress_bar_current, progress_bar_total, lbl_size_info, app,
                                output_parquet_file, metadata_path, model_name,
                                excel_mappings, enable_ner=True, enable_coreference=True,
                                resume_mode=False, completion_callback=None):
    """Thin GUI adapter over the shared creator orchestrator."""
    paths = [path for path, var in selected_files.items() if var.get() == 1]
    reporter = GuiProgressReporter(
        app=app, status_label=status_label,
        progress_bar_current=progress_bar_current, progress_bar_total=progress_bar_total,
        size_label=lbl_size_info,
        show_warning=getattr(messagebox, "showwarning", None),
        show_error=getattr(messagebox, "showerror", None),
    )
    options = CreatorRunOptions(
        input_files=paths, output_parquet_file=output_parquet_file,
        metadata_path=metadata_path, model_name=model_name,
        excel_mappings=excel_mappings or {}, resume_mode=bool(resume_mode),
        enable_ner=bool(enable_ner),
        enable_coreference=bool(enable_coreference),
    )
    result = run_creator_job(options, reporter, model_state=CreatorModelState(), models_dir=MODELS_DIR)
    _schedule_creator_completion(app, completion_callback, result.success,
                                 output_file=result.output_file, error_message=result.error_message)


def get_file_page_count():
    if not selected_files:
        return 1
    return max(1, (len(selected_files) + FILES_PER_PAGE - 1) // FILES_PER_PAGE)


def update_file_selection_status(status_label=None):
    """
    Aktualizuje etykietę paginacji i status wyboru plików.
    Nie tworzy nowych checkboxów — tylko odświeża teksty/liczniki.
    """
    total = len(selected_files)
    selected_count = sum(1 for var in selected_files.values() if var.get() == 1)
    page_count = get_file_page_count()

    if pagination_label is not None:
        pagination_label.configure(
            text=f"Strona {file_page_index + 1}/{page_count} | "
                 f"Zaznaczono {selected_count}/{total}"
        )

    if pagination_prev_button is not None:
        pagination_prev_button.configure(
            state="normal" if file_page_index > 0 else "disabled"
        )

    if pagination_next_button is not None:
        pagination_next_button.configure(
            state="normal" if file_page_index < page_count - 1 else "disabled"
        )

    if status_label is not None and total > 0:
        status_label.configure(
            text=f"Zaznacz pliki do przetworzenia. Wybrano {selected_count} z {total}."
        )
def reset_scrollable_frame_position(frame):
    """
    Resetuje pozycję przewinięcia CTkScrollableFrame do góry.
    Używa wewnętrznego canvasa CustomTkinter, dlatego jest opakowane
    defensywnie w try/except.
    """
    def _reset():
        try:
            frame._parent_canvas.yview_moveto(0)
        except Exception:
            try:
                frame._canvas.yview_moveto(0)
            except Exception:
                pass

    try:
        frame.after_idle(_reset)
    except Exception:
        _reset()


def render_file_page(frame, status_label=None):
    """
    Renderuje tylko jedną stronę checkboxów.
    Stan zaznaczenia jest przechowywany w selected_files[path] jako IntVar,
    więc przechodzenie między stronami nie gubi checkboxów.
    """
    global file_buttons, file_page_index

    # Najpierw wyzeruj scroll, żeby nie przenosić pozycji z poprzedniej strony
    reset_scrollable_frame_position(frame)

    # Usuń tylko widoczne checkboxy z poprzedniej strony
    for widget in frame.winfo_children():
        widget.destroy()

    file_buttons.clear()

    paths = list(selected_files.keys())
    page_count = get_file_page_count()

    if file_page_index < 0:
        file_page_index = 0
    if file_page_index >= page_count:
        file_page_index = page_count - 1

    start = file_page_index * FILES_PER_PAGE
    end = start + FILES_PER_PAGE
    page_paths = paths[start:end]

    for file_path in page_paths:
        var = selected_files[file_path]

        btn = ctk.CTkCheckBox(
            frame,
            text=os.path.basename(file_path),
            variable=var,
            command=lambda: update_file_selection_status(status_label)
        )
        btn.pack(anchor="w", padx=20, pady=6)
        file_buttons.append(btn)

    # Wymuś przeliczenie layoutu po utworzeniu nowych checkboxów
    try:
        frame.update_idletasks()
    except Exception:
        pass

    # I jeszcze raz wyzeruj scroll po przeliczeniu geometrii.
    # To jest ważne zwłaszcza dla ostatniej strony z kilkoma elementami.
    reset_scrollable_frame_position(frame)

    update_file_selection_status(status_label)


def change_file_page(delta, frame, status_label=None):
    """
    Przechodzi do poprzedniej/następnej strony listy plików.
    """
    global file_page_index

    page_count = get_file_page_count()
    file_page_index = max(0, min(file_page_index + delta, page_count - 1))

    render_file_page(frame, status_label)

# --- UI ---
def select_files(frame, progress_bar_current, progress_bar_total, lbl_size_info, status_label, app):
    global selected_files, file_buttons, file_page_index

    file_paths = fd.askopenfilenames(
        title="Wybierz pliki",
        initialdir=BASE_DIR,
        filetypes=[("All files", "*.*"),
                   ("Text files", "*.txt"),
                   ("Word Documents", "*.docx"),
                   ("PDF files", "*.pdf"),
                   ("Excel files", "*.xlsx"),
                   ("Archives", "*.zip")],
        parent=app
    )

    if not file_paths:
        status_label.configure(text="Nie wybrano żadnego pliku.")
        return

    added_count = 0

    # Tworzymy tylko stan zaznaczeń, bez tworzenia tysięcy widgetów naraz
    for file_path in file_paths:
        if file_path not in selected_files:
            selected_files[file_path] = ctk.IntVar(value=1)
            added_count += 1

    # Po dodaniu nowych plików wracamy na pierwszą stronę
    if added_count > 0:
        file_page_index = 0

    # Renderujemy tylko bieżącą stronę checkboxów
    render_file_page(frame, status_label)

    # Hide bars initially
    progress_bar_current.grid_remove()
    progress_bar_total.grid_remove()
    lbl_size_info.grid_remove()

    total = len(selected_files)
    selected_count = sum(1 for var in selected_files.values() if var.get() == 1)

    status_label.configure(
        text=f"Dodano {added_count} nowych plików. Wybrano {selected_count} z {total}."
    )


def main(parent_window=None):
    global model, selected_files, file_buttons, file_page_index
    global pagination_label, pagination_prev_button, pagination_next_button


    selected_files.clear()
    file_buttons.clear()
    file_page_index = 0

    pagination_label = None
    pagination_prev_button = None
    pagination_next_button = None

    if parent_window:
        app = ctk.CTkToplevel(parent_window)
        app.transient(parent_window)  # Trzyma okno kreatora nad głównym oknem
        app.grab_set()                # Blokuje klikanie w główne okno
    else:
        app = ctk.CTk()

    def center_window(app, width=1000, height=600):
        screen_width = app.winfo_screenwidth()
        screen_height = app.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        app.geometry(f"{width}x{height}+{x}+{y}")

    center_window(app, 1000, 600)
    app.title("Kreator korpusów")

    main_frame = ctk.CTkFrame(app)
    main_frame.pack(pady=5, fill="both", side="left")
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)

    # Grid config
    for i in range(12): main_frame.grid_rowconfigure(i, weight=0)

    model = ctk.StringVar(value="Stanza")
    option_model = ctk.CTkOptionMenu(
        main_frame,
        values=["Stanza", "spaCy"],
        variable=model,
        font=("Verdana", 12, 'bold'),
        fg_color="#4B6CB7",
        dropdown_fg_color="#4B6CB7",
        width=120, height=35, corner_radius=8
    )
    option_model.grid(row=0, column=0, columnspan=2, pady=10)

    enable_ner_var = ctk.BooleanVar(value=True)
    enable_coreference_var = ctk.BooleanVar(value=True)
    annotation_options_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    annotation_options_frame.grid(row=1, column=0, columnspan=2, pady=(0, 8))
    enable_ner_checkbox = ctk.CTkCheckBox(
        annotation_options_frame,
        text="Rozpoznawanie nazw własnych (NER)",
        variable=enable_ner_var,
    )
    enable_ner_checkbox.pack(side="left", padx=8)
    enable_coreference_checkbox = ctk.CTkCheckBox(
        annotation_options_frame,
        text="Koreferencja",
        variable=enable_coreference_var,
    )
    enable_coreference_checkbox.pack(side="left", padx=8)

    select_button = ctk.CTkButton(
        main_frame,
        text="Wybierz pliki",
        command=lambda: select_files(checkbox_frame, progress_bar_current, progress_bar_total, lbl_size_info,
                                     status_label, app),
        font=("Verdana", 12, 'bold'),
        corner_radius=8, height=35,
        fg_color='#4B6CB7', hover_color="#5B7CD9",
    )
    select_button.grid(row=2, column=0, columnspan=2, pady=10)

    # --- PASKI POSTĘPU ---

    # 1. Total Label
    lbl_total = ctk.CTkLabel(main_frame, text="Postęp całkowity:", font=("Verdana", 10))
    lbl_total.grid(row=4, column=0, sticky="w", padx=20)

    # NOWY ELEMENT: Etykieta rozmiaru (np. 15 MB / 100 MB)
    CREATOR_STATUS_WIDTH = 520
    CREATOR_STATUS_WRAPLENGTH = 500
    lbl_size_info = ctk.CTkLabel(
        main_frame,
        text="",
        width=260,
        wraplength=250,
        anchor="e",
        justify="right",
        font=("Verdana", 10, "bold"),
        text_color="#555555",
    )
    lbl_size_info.grid(row=4, column=1, sticky="e", padx=20)

    # 2. Total Bar
    progress_bar_total = ctk.CTkProgressBar(main_frame, progress_color="#32CD32")
    progress_bar_total.set(0)
    progress_bar_total.grid(row=5, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")

    # 3. Current Label
    lbl_current = ctk.CTkLabel(main_frame, text="Bieżący plik:", font=("Verdana", 10))
    lbl_current.grid(row=6, column=0, columnspan=2, sticky="w", padx=20)

    # 4. Current Bar
    progress_bar_current = ctk.CTkProgressBar(main_frame)
    progress_bar_current.set(0)
    progress_bar_current.grid(row=7, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")

    status_label = ctk.CTkLabel(
        main_frame,
        text="Gotowy",
        width=CREATOR_STATUS_WIDTH,
        wraplength=CREATOR_STATUS_WRAPLENGTH,
        anchor="w",
        justify="left",
        font=("Verdana", 12, 'bold'),
    )
    status_label.grid(
        row=8, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
    )

    # Ukrycie paska rozmiaru na start
    lbl_size_info.grid_remove()

    def reset_creator_form():
        """Reset per-run GUI state while preserving loaded NLP models."""
        global selected_files, file_buttons, file_page_index

        selected_files.clear()
        file_buttons.clear()
        file_page_index = 0
        render_file_page(checkbox_frame, status_label)

        progress_bar_current.set(0)
        progress_bar_total.set(0)
        progress_bar_current.grid_remove()
        progress_bar_total.grid_remove()
        lbl_size_info.configure(text="")
        lbl_size_info.grid_remove()

        try:
            switch_var.set("on")
        except Exception:
            pass
        enable_ner_var.set(True)
        enable_coreference_var.set(True)

        process_button.configure(state="normal")
        select_button.configure(state="normal")
        status_label.configure(
            text="Gotowy — wybierz pliki do utworzenia kolejnego korpusu."
        )

    def finalize_creator_run(success, output_file=None, error_message=None):
        """Own all completion dialogs and restore controls on the GUI thread."""
        if success:
            output_name = os.path.basename(output_file) if output_file else ""
            if output_name:
                status_label.configure(text=f"Gotowe: {output_name}")
            messagebox.showinfo(
                "Sukces",
                "Zakończono przetwarzanie i scalanie plików.",
                parent=app,
            )
            reset_creator_form()
            return

        process_button.configure(state="normal")
        select_button.configure(state="normal")
        status_label.configure(text="Przetwarzanie nie zostało zakończone.")
        messagebox.showerror(
            "Błąd",
            error_message or "Wystąpił błąd podczas przetwarzania.",
            parent=app,
        )

    # --- UPDATED START FUNCTION ---
    def start_processing():
        selected_paths = [path for path, var in selected_files.items() if var.get() == 1]
        if not selected_paths:
            status_label.configure(text="Nie wybrano pliku.")
            return

        excel_mappings = {}

        # 1. Map columns for SOURCE files (Regular Excel files)
        for path in selected_paths:
            if path.lower().endswith(".xlsx") and "metadane.xlsx" not in os.path.basename(path).lower():
                try:
                    df_headers = pd.read_excel(path, nrows=0)
                    cols = df_headers.columns.tolist()

                    # is_metadata=False -> Shows "Treść" field
                    mapper = ColumnMapper(app, os.path.basename(path), cols, is_metadata=False)

                    if mapper.result:
                        excel_mappings[path] = mapper.result
                    else:
                        status_label.configure(text=f"Anulowano: {os.path.basename(path)}")
                        return
                except Exception as e:
                    messagebox.showerror("Błąd", f"Excel error: {e}")
                    return

        # 2. Ask for METADATA file
        metadata_path = None
        if messagebox.askquestion("Metadane", "Czy dodać osobny plik z metadanymi (np. metadane.xlsx)?") == 'yes':
            metadata_path = fd.askopenfilename(parent=app, filetypes=[("Excel", "*.xlsx")])
            if not metadata_path: return

            # --- TRIGGER MAPPER FOR METADATA ---
            try:
                df_meta_headers = pd.read_excel(metadata_path, nrows=0)
                meta_cols = df_meta_headers.columns.tolist()

                # is_metadata=True -> Hides "Treść" field
                meta_mapper = ColumnMapper(app, f"METADANE: {os.path.basename(metadata_path)}", meta_cols,
                                           is_metadata=True)

                if meta_mapper.result:
                    excel_mappings[metadata_path] = meta_mapper.result
                else:
                    status_label.configure(text="Anulowano mapowanie metadanych.")
                    return
            except Exception as e:
                messagebox.showerror("Błąd Metadanych", f"Nie można odczytać pliku metadanych: {e}")
                return
            # -----------------------------------

        output_file = fd.asksaveasfilename(parent=app, defaultextension=".parquet",
                                           filetypes=[("Parquet", "*.parquet")])
        if not output_file: return

        # --- SPRAWDZANIE CZY MOŻNA WZNOWIĆ ---
        import glob
        resume_mode = False
        existing_parts = glob.glob(f"{output_file}.part_*")
        # NOWE: Sprawdzamy też, czy istnieje już główny plik .parquet
        if existing_parts or os.path.exists(output_file):
            ans = messagebox.askyesno("Punkt kontrolny",
                                      "Znaleziono pliki z poprzedniej sesji (tymczasowe lub główny plik).\nCzy chcesz wczytać ich zawartość i pominąć już zrobione teksty?")
            if ans:
                resume_mode = True
        # -------------------------------------

        enable_ner = bool(enable_ner_var.get())
        enable_coreference = bool(enable_coreference_var.get())

        process_button.configure(state="disabled")
        select_button.configure(state="disabled")

        progress_bar_total.grid()
        progress_bar_current.grid()
        lbl_size_info.grid()

        threading.Thread(
            target=process_files_thread_target,
            args=(
                status_label, progress_bar_current, progress_bar_total, lbl_size_info, app, output_file, metadata_path,
                model.get(),
                excel_mappings,
                enable_ner,
                enable_coreference,
                resume_mode,
                finalize_creator_run),
            daemon=True
        ).start()

    process_button = ctk.CTkButton(main_frame, text="Przetwórz pliki", command=start_processing,
                                   font=("Verdana", 12, 'bold'),
                                   corner_radius=8, height=35,
                                   fg_color='#4B6CB7', hover_color="#5B7CD9")
    process_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Prawy panel: osobno kontrolki paginacji, osobno scrollowana lista checkboxów
    right_panel = ctk.CTkFrame(app)
    right_panel.pack(pady=6, fill="both", expand=True, side="right")

    pagination_frame = ctk.CTkFrame(right_panel)
    pagination_frame.pack(fill="x", padx=6, pady=(6, 2))

    checkbox_frame = ctk.CTkScrollableFrame(right_panel)
    checkbox_frame.pack(pady=6, fill="both", expand=True)

    switch_var = ctk.StringVar(value="on")

    def toggle_all():
        val = 1 if switch_var.get() == "on" else 0

        # Ważne: zmieniamy stan wszystkich plików, nie tylko bieżącej strony
        for var in selected_files.values():
            var.set(val)

        update_file_selection_status(status_label)

    toggle_button = ctk.CTkSwitch(
        pagination_frame,
        text="Zaznacz wszystko",
        command=toggle_all,
        variable=switch_var,
        onvalue="on",
        offvalue="off"
    )
    toggle_button.grid(row=0, column=0, padx=8, pady=8, sticky="w")

    pagination_prev_button = ctk.CTkButton(
        pagination_frame,
        text="◀",
        width=40,
        command=lambda: change_file_page(-1, checkbox_frame, status_label)
    )
    pagination_prev_button.grid(row=0, column=1, padx=4, pady=8)

    pagination_label = ctk.CTkLabel(
        pagination_frame,
        text="Strona 1/1 | Zaznaczono 0/0"
    )
    pagination_label.grid(row=0, column=2, padx=8, pady=8)

    pagination_next_button = ctk.CTkButton(
        pagination_frame,
        text="▶",
        width=40,
        command=lambda: change_file_page(1, checkbox_frame, status_label)
    )
    pagination_next_button.grid(row=0, column=3, padx=4, pady=8)

    # Stan początkowy przycisków paginacji
    update_file_selection_status(status_label)

    if not parent_window:
        app.mainloop()


if __name__ == "__main__":
    main()