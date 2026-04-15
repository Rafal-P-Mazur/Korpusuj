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
import spacy
import spacy.cli
import pyarrow as pa
import pyarrow.parquet as pq
import gc
import torch
from datetime import datetime

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
    logging.info("SUKCES: Herference zaimportowane pomyślnie na górze pliku!")
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

selected_files = {}
file_buttons = []


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

# NUMERIC:
# Pozostaje "inline-capable", bo w części materiałów (jak 00898.txt)
# rekordy typu "1. Austria ... 2. Belgia ..." występują w jednym długim bloku,
# a niekoniecznie od nowej linii.
NUMERIC_RECORD_START_RE = re.compile(
    r'(?:^|(?<=\s))(\d{1,3}\.\s*[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# LETTERED:
# Tylko początek linii, żeby nie łapać przypadkowych "a)" w środku zdania.
LETTER_RECORD_START_RE = re.compile(
    r'(?m)^(?:\s{0,3})([a-zA-Z][\)\.]\s+[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# BULLETS:
# Tylko początek linii.
# Nie wymagamy Wielkiej Litery po markerze, bo w polskich wyliczeniach
# po dwukropku elementy listy często zaczynają się od małej litery.
BULLET_RECORD_START_RE = re.compile(
    r'(?m)^(?:\s{0,3})([-*•])(?=\s+\S)'
)

# ROMAN:
# Tylko początek linii.
ROMAN_RECORD_START_RE = re.compile(
    r'(?m)^(?:\s{0,3})([IVXLCDM]{1,8}[\)\.]\s+[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# Silny koniec zdania
STRONG_SENT_END_RE = re.compile(
    r'(?<!\bul)(?<!\bUl)(?<!\bnp)(?<!\bNp)(?<!\btzw)(?<!\bdr)(?<!\bks)(?<!prof)(?<!mgr)(?<!m\.in)(?<!\bal)(?<!\bAl)(?<!\blok)(?<!\bpl)(?<!\bPl)[.!?][\'")\]]*(?:\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])|$)'
)

# Separator pól dla rekordów półtabelarycznych
FIELD_BREAK_RE = re.compile(
    r'(?:\s[-–—]\s|;\s+|:\s+)'
)

# NOWE: Ukryta granica listy (np. "prawnik Izabela Aleksandra")
# Szuka: mała litera -> spacja -> (Wielka litera + małe) -> spacja -> (Wielka litera)
IMPLIED_RECORD_BREAK_RE = re.compile(
    r'(?<=[a-ząćęłńóśźż])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# Słabszy fallback po interpunkcji
SOFT_BREAK_RE = re.compile(
    r'[.!?,;:][\'")\]]*\s+'
)


def has_multiline_bullet_layout(text: str, min_items: int = 4) -> bool:
    """
    Dodatkowa heurystyka dla bulletów:
    sprawdza, czy tekst faktycznie zawiera wieloliniowy układ listy,
    a nie pojedyncze myślniki / gwiazdki przypadkowo występujące w tekście.
    """
    bullet_lines = 0
    for line in text.splitlines():
        if BULLET_RECORD_START_RE.match(line):
            bullet_lines += 1
            if bullet_lines >= min_items:
                return True
    return False


def detect_record_style(text: str) -> str | None:
    """
    Wykrywa styl listy / rekordów.
    Zwraca jedną z wartości:
      - "numeric"
      - "letter"
      - "bullet"
      - "roman"
      - None
    """
    text_len = max(len(text), 1)

    numeric_count = len(list(NUMERIC_RECORD_START_RE.finditer(text)))
    letter_count = len(list(LETTER_RECORD_START_RE.finditer(text)))
    bullet_count = len(list(BULLET_RECORD_START_RE.finditer(text)))
    roman_count = len(list(ROMAN_RECORD_START_RE.finditer(text)))

    numeric_density = numeric_count / text_len * 10000.0
    letter_density = letter_count / text_len * 10000.0
    bullet_density = bullet_count / text_len * 10000.0
    roman_density = roman_count / text_len * 10000.0

    # Najwyższy priorytet: numeric, bo jest najbardziej precyzyjny
    # i pokrywa przypadki "1. Polska ... 2. Czechy ..." także inline.
    if numeric_count >= 5 and numeric_density >= 4.0:
        return "numeric"

    # Litery: ostrożniej
    if letter_count >= 5 and letter_density >= 4.5:
        return "letter"

    # Rzymskie: zwykle rzadsze, ale dość charakterystyczne
    if roman_count >= 4 and roman_density >= 3.5:
        return "roman"

    # Bullets: najbardziej wieloznaczne, więc oprócz count/density
    # wymagamy też rzeczywistego układu wieloliniowej listy.
    if (
        bullet_count >= 4
        and bullet_density >= 4.0
        and has_multiline_bullet_layout(text, min_items=4)
    ):
        return "bullet"

    return None


def get_record_start_regex(style: str):
    """
    Zwraca regex startu rekordu dla zadanego stylu.
    """
    if style == "numeric":
        return NUMERIC_RECORD_START_RE
    if style == "letter":
        return LETTER_RECORD_START_RE
    if style == "bullet":
        return BULLET_RECORD_START_RE
    if style == "roman":
        return ROMAN_RECORD_START_RE
    return None


def split_structured_segments(text: str, style: str) -> tuple[str, list[str]]:
    """
    Dzieli tekst na:
      - preambułę (wszystko przed pierwszym rekordem),
      - listę rekordów.

    LOSSLESS:
      - niczego nie trimuje,
      - niczego nie normalizuje,
      - nie podmienia newline na spacje.
    """
    rx = get_record_start_regex(style)
    if rx is None:
        return text, []

    matches = list(rx.finditer(text))
    if not matches:
        return text, []

    starts = [m.start(1) for m in matches]
    preamble = text[:starts[0]]
    records = []

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        records.append(text[start:end])

    return preamble, records


def soft_cut_preserve(
    block: str,
    limit: int,
    backtrack_window: int = 800,
) -> list[str]:
    """
    Miękkie cięcie BLOKU bez zmiany treści (LOSSLESS).

    Priorytet cięcia:
      1) silny koniec zdania,
      2) separator pola,
      3) słabsza interpunkcja,
      4) ostatnia spacja,
      5) hard cut.
    """
    out: list[str] = []
    start = 0
    L = len(block)

    while start < L:
        remaining = L - start
        if remaining <= limit:
            out.append(block[start:])
            break

        end = start + limit
        search_start = max(start, end - backtrack_window)
        window = block[search_start:end]

        cut = -1
        last_match = None

        # 1) Silny koniec zdania
        for m in STRONG_SENT_END_RE.finditer(window):
            last_match = m
        if last_match:
            cut = search_start + last_match.end()
        else:
            # 2) Separator pola
            last_match = None
            for m in FIELD_BREAK_RE.finditer(window):
                last_match = m
            if last_match:
                cut = search_start + last_match.end()
            else:
                # 2.5) Ukryta granica listy (bezinterpunkcyjne przejście na Imię i Nazwisko)
                last_match = None
                for m in IMPLIED_RECORD_BREAK_RE.finditer(window):
                    last_match = m
                if last_match:
                    # Tniemy na spacji (start dopasowania), żeby nie ucinać Imienia
                    cut = search_start + last_match.start()
                else:
                    # 3) Słabsza interpunkcja
                    last_match = None
                    for m in SOFT_BREAK_RE.finditer(window):
                        last_match = m
                    if last_match:
                        cut = search_start + last_match.end()
                    else:
                        # 4) Ostatnia spacja
                        last_space = window.rfind(" ")
                        if last_space != -1:
                            cut = search_start + last_space
                        else:
                            # 5) Twarde cięcie
                            cut = end

        if cut <= start:
            cut = end

        out.append(block[start:cut])
        start = cut

    return out


def chunk_structured_records(
    text: str,
    chunk_size: int,
    style: str = "numeric",
    structured_chunk_size: int = 4000,
    max_records_per_chunk: int = 6,
    backtrack_window: int = 800,
    max_dotless_chars: int = 800,       # <-- NOWE: Przekazany próg tasiemca
    min_piece_in_danger: int = 1200     # <-- NOWE: Przekazany bezpieczny limit
) -> list[str]:
    """
    Dedykowane, LOSSLESS cięcie dla tekstów rekordowych / wyliczeniowych.
    Teraz z dynamicznym progiem bezpieczeństwa dla "tasiemców".
    """
    effective_limit = min(chunk_size, structured_chunk_size)
    preamble, records = split_structured_segments(text, style=style)

    out: list[str] = []

    # --- NOWE: Dynamiczny estymator limitu dla konkretnego bloku ---
    def get_dynamic_limit(block: str) -> int:
        last_end = 0
        for m in STRONG_SENT_END_RE.finditer(block):
            if m.start() - last_end > max_dotless_chars:
                return min(effective_limit, min_piece_in_danger)
            last_end = m.end()
        # Sprawdzamy resztówkę po ostatniej kropce
        if len(block) - last_end > max_dotless_chars:
            return min(effective_limit, min_piece_in_danger)
        return effective_limit
    # ---------------------------------------------------------------

    # 1) Preambuła
    if preamble:
        limit = get_dynamic_limit(preamble)
        if len(preamble) <= limit:
            out.append(preamble)
        else:
            out.extend(
                soft_cut_preserve(
                    preamble,
                    limit=limit,
                    backtrack_window=backtrack_window,
                )
            )

    # 2) Rekordy
    current_parts: list[str] = []
    current_len = 0
    current_records = 0

    def flush():
        nonlocal current_parts, current_len, current_records
        if current_parts:
            out.append("".join(current_parts))
            current_parts = []
            current_len = 0
            current_records = 0

    for rec in records:
        limit = get_dynamic_limit(rec) # Badamy konkretny rekord

        # Rekord większy niż jego dynamiczny limit -> tniemy miękko
        if len(rec) > limit:
            flush()
            out.extend(
                soft_cut_preserve(
                    rec,
                    limit=limit,
                    backtrack_window=backtrack_window,
                )
            )
            continue

        # Ograniczenie po rozmiarze i liczbie rekordów (uwzględnia dynamiczny limit!)
        if current_parts and (
            current_len + len(rec) > limit
            or current_records >= max_records_per_chunk
        ):
            flush()

        current_parts.append(rec)
        current_len += len(rec)
        current_records += 1

    flush()
    return out


def chunk_text_safe(
    text: str,
    chunk_size: int = 15000,
    max_dotless_chars: int = 800,
    backtrack_window: int = 600,
    min_piece_in_danger: int = 1200,
    structured_chunk_size: int = 4000,
    max_records_per_chunk: int = 6,
) -> list[str]:
    """
    Dzieli tekst na NLP-przyjazne fragmenty.
    """
    # 1) ROUTING: tekst rekordowy / wyliczeniowy
    style = detect_record_style(text)
    if style is not None:
        return chunk_structured_records(
            text,
            chunk_size=chunk_size,
            style=style,
            structured_chunk_size=structured_chunk_size,
            max_records_per_chunk=max_records_per_chunk,
            backtrack_window=max(backtrack_window, 800),
            max_dotless_chars=max_dotless_chars,       # <-- PRZEKAZUJEMY
            min_piece_in_danger=min_piece_in_danger    # <-- PRZEKAZUJEMY
        )

    # 2) ZWYKŁY TEKST
    ANY_PUNCT_GAP_RE = re.compile(r'[.!?,\-;—–:][\'")\]]*\s+')
    PARA_END_RE = re.compile(
        r'(?<!\b[a-ząćęłńóśźż])(?<!\b[a-ząćęłńóśźż]{2})(?<!\b[a-ząćęłńóśźż]{3})[.!?][\'")\]]*\s*$'
    )

    chunks: list[str] = []
    paragraphs = text.split("\n")
    current_chunk_parts: list[str] = []
    current_len = 0
    dotless = 0

    def flush():
        nonlocal current_chunk_parts, current_len, dotless
        if current_chunk_parts:
            chunks.append("".join(current_chunk_parts))
            current_chunk_parts = []
            current_len = 0
            dotless = 0

    def soft_cut(block: str, limit: int) -> list[str]:
        """
        LOSSLESS cięcie zwykłego tekstu.
        """
        out = []
        start = 0
        L = len(block)

        while start < L:
            remaining = L - start
            if remaining <= limit:
                out.append(block[start:])
                break

            end = start + limit
            search_start = max(start, end - backtrack_window)
            window = block[search_start:end]

            m = None
            cut = -1

            # Krok 1: silny koniec zdania
            for mm in STRONG_SENT_END_RE.finditer(window):
                m = mm

            if m:
                cut = search_start + m.end()
            else:
                # Krok 2: słabsza interpunkcja
                for mm in ANY_PUNCT_GAP_RE.finditer(window):
                    m = mm
                if m:
                    cut = search_start + m.end()
                else:
                    # Krok 3: ostatnia spacja
                    last_space = window.rfind(" ")
                    if last_space != -1:
                        cut = search_start + last_space
                    else:
                        # Krok 4: twarde cięcie
                        fallback_space = block[start:end].rfind(" ")
                        cut = start + fallback_space if fallback_space != -1 else end

            if cut <= start:
                cut = end

            out.append(block[start:cut])
            start = cut

        return out

    for i, para in enumerate(paragraphs):
        # split("\n") usuwa separator, więc dokładamy go tylko tam,
        # gdzie istniał w oryginale.
        para_full = para + ("\n" if i < len(paragraphs) - 1 else "")
        para_len = len(para_full)

        # Wykrywanie tasiemca na podstawie silnych końców zdań
        dangerous = False
        last_end = 0

        for m in STRONG_SENT_END_RE.finditer(para_full):
            if m.start() - last_end > max_dotless_chars:
                dangerous = True
                break
            last_end = m.end()

        if not dangerous and (para_len - last_end) > max_dotless_chars:
            dangerous = True

        # Niebezpieczny albo za duży akapit -> miękkie cięcie
        if dangerous or para_len > chunk_size:
            flush()
            target_limit = chunk_size if not dangerous else min(chunk_size, min_piece_in_danger)
            for piece in soft_cut(para_full, limit=target_limit):
                chunks.append(piece)
            continue

        # Normalne sklejanie akapitów
        if not PARA_END_RE.search(para_full.strip()):
            dotless += para_len
        else:
            dotless = 0

        if dotless > max_dotless_chars and current_len > 0:
            flush()
            current_chunk_parts.append(para_full)
            current_len = para_len
            dotless = 0 if PARA_END_RE.search(para_full.strip()) else para_len
            continue

        if current_len + para_len > chunk_size:
            flush()

        current_chunk_parts.append(para_full)
        current_len += para_len

    flush()
    return chunks






def unpack_archive(file_path, status_label):
    temp_dir = tempfile.mkdtemp(prefix="archive_extract_")
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


def calculate_real_total_size(file_paths):
    total_size = 0
    for path in file_paths:
        try:
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as z:
                    for info in z.infolist():
                        if not info.is_dir():
                            total_size += info.file_size
            else:
                total_size += os.path.getsize(path)
        except Exception as e:
            logging.warning(f"Błąd obliczania rozmiaru dla {path}: {e}")
            total_size += os.path.getsize(path)
    return total_size


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
                        reader = easyocr.Reader(['pl'], gpu=True)
                    status_label.configure(text=f"OCR: {os.path.basename(file_path)} str {page.number + 1}")
                    app.update_idletasks()
                    pix = page.get_pixmap()
                    result = reader.readtext(pix.tobytes("png"), detail=0)
                    text += " ".join(result) + " "
    except Exception as e:
        logging.warning(f"Błąd PDF: {e}")
        pass
    return text.replace('-\n', '').replace('\n', ' ')


def process_xlsx(file_path, mapping=None):
    try:
        df = pd.read_excel(file_path)
        data = []

        # Domyślne nazwy kolumn (gdyby mapping nie był podany, choć teraz wymuszamy)
        col_filename = "Nazwa pliku"
        col_title = "Tytuł"
        col_content = "Treść"
        col_date = "Data publikacji"
        col_author = "Autor"

        if mapping:
            col_filename = mapping.get("Nazwa pliku", col_filename)
            col_title = mapping.get("Tytuł", col_title)
            col_content = mapping.get("Treść", col_content)
            col_date = mapping.get("Data publikacji", col_date)
            col_author = mapping.get("Autor", col_author)

        def get_val(row, col_name):
            if col_name == "<Pomiń>" or col_name not in df.columns:
                return ""
            val = row[col_name]
            return str(val).strip() if pd.notna(val) else ""

        for _, row in df.iterrows():
            # Nazwa pliku jest kluczowa dla identyfikacji
            virt_filename = get_val(row, col_filename)
            if not virt_filename:
                # Fallback: jeśli w wierszu brakuje ID, generujemy placeholder lub pomijamy
                # Tu opcjonalnie: virt_filename = f"row_{_}.txt"
                continue

            title = get_val(row, col_title)
            content = get_val(row, col_content)

            if not content: continue
            if title and not content.startswith(title):
                content = f"{title}\n\n{content}".strip()

            data.append({
                "filename": virt_filename,  # Kluczowa zmiana: nazwa z kolumny
                "Tytuł": title,
                "Treść": content,
                "Data publikacji": get_val(row, col_date),
                "Autor": get_val(row, col_author)
            })
        return data
    except Exception as e:
        logging.warning(f"Błąd Excel {file_path}: {e}")
        return []


def update_status(label, text, app):
    app.after(0, lambda: label.configure(text=text))
    app.update_idletasks()


# --- INITIALIZATION ---
def initialize_stanza(status_label, app):
    global nlp_stanza
    if stanza is None:
        messagebox.showerror("Błąd", "Biblioteka Stanza nie jest zainstalowana.")
        return False

    # Tworzymy ścieżkę: obok pliku exe -> folder "models" -> folder "stanza"
    stanza_dir = os.path.join(BASE_DIR, "models", "stanza")
    os.makedirs(stanza_dir, exist_ok=True)

    # Stanza tworzy wewnątrz folder z kodem języka, np. "pl"
    model_path = os.path.join(stanza_dir, "pl")

    if not os.path.exists(model_path):
        try:
            status_label.configure(text="Proszę czekać - pobieram model Stanza (ok. 500 MB)...")
            app.update_idletasks()
            # Wymuszamy pobranie do naszego lokalnego folderu (zmiana na model_dir)
            stanza.download("pl", model_dir=stanza_dir)
        except Exception as e:
            messagebox.showerror("Błąd modelu Stanza", f"Nie udało się pobrać modelu: {e}")
            return False

    status_label.configure(text="Ładuję model Stanza z folderu 'models' - proszę czekać.")
    app.update_idletasks()

    try:
        # Wymuszamy wczytanie z naszego lokalnego folderu (zmiana na model_dir)
        nlp_stanza = stanza.Pipeline("pl", model_dir=stanza_dir, processors="tokenize,pos,lemma,ner,depparse,coref",
                                     use_gpu=True, n_process=1)
        status_label.configure(text="Model Stanza załadowany pomyślnie")
        return True
    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się załadować Stanza: {e}")
        return False


def initialize_spacy(status_label, app):
    global nlp_spacy

    model_name = "pl_core_news_lg"
    model_version = "3.8.0"

    # Zmieniona ścieżka główna
    spacy_dir = os.path.join(BASE_DIR, "models", "spacy")

    try:
        status_label.configure(text="Sprawdzam model SpaCy w folderze 'models'...")
        app.update_idletasks()

        # TRIK: Dodajemy nasz folder do ścieżek systemowych Pythona, żeby widział go jak paczkę zainstalowaną przez PIP!
        if spacy_dir not in sys.path:
            sys.path.insert(0, spacy_dir)

        # Sprawdzamy czy istnieje plik __init__.py wewnątrz rozpakowanego folderu
        if not os.path.exists(os.path.join(spacy_dir, model_name, "__init__.py")):
            status_label.configure(text=f"Pobieram model SpaCy ({model_name}). To może potrwać (ok. 500MB)...")
            app.update_idletasks()

            url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-{model_version}/{model_name}-{model_version}-py3-none-any.whl"

            os.makedirs(spacy_dir, exist_ok=True)
            whl_path = os.path.join(spacy_dir, "temp_model.whl")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(whl_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            status_label.configure(text="Rozpakowuję i instaluję model SpaCy w folderze 'models'...")
            app.update_idletasks()

            # Zamiast wycinać pliki, wypakowujemy CAŁE archiwum zachowując strukturę paczki Pythona!
            with zipfile.ZipFile(whl_path, 'r') as zip_ref:
                zip_ref.extractall(spacy_dir)

            os.remove(whl_path)
            status_label.configure(text="Model SpaCy zainstalowany pomyślnie.")
            app.update_idletasks()

        status_label.configure(text="Ładuję model SpaCy (proszę czekać)...")
        app.update_idletasks()

        # Teraz możemy bezpiecznie załadować po NAZWIE (a nie ścieżce), bo dodaliśmy folder do sys.path!
        nlp_spacy = spacy.load(model_name)

        if herference is not None:
            try:
                status_label.configure(text="Podpinam model herference (koreferencje)...")
                app.update_idletasks()
                nlp_spacy.add_pipe("herference")
            except Exception as e:
                messagebox.showerror("Błąd SpaCy",
                                     f"SpaCy załadowano, ale podpinanie herference zakończyło się błędem:\n\n{e}")

        status_label.configure(text="Model SpaCy załadowany pomyślnie z folderu lokalnego.")
        return nlp_spacy

    except Exception as e:
        messagebox.showerror("Błąd modelu SpaCy", f"Wystąpił błąd podczas pobierania lub ładowania modelu:\n{e}")
        return None


# --- NLP PROCESSING ---
def process_single_text(text, filename, status_label, progress_bar, app):
    if not text.strip(): return None
    chunks = chunk_text_safe(text, chunk_size=15000)
    all_processed_tokens = []
    global_sent_id_offset = 0
    global_char_offset = 0
    total_chunks = len(chunks)
    progress_bar.set(0)

    # --- NOWE: GLOBALNY LICZNIK DLA CAŁEGO PLIKU (Rozwiązuje problem kolizji chunków) ---
    global_cluster_id_counter = 1

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            global_char_offset += len(chunk)
            continue
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            update_status(status_label, f"Przetwarzam: {filename} (Część {i + 1}/{total_chunks})", app)
            doc = nlp_stanza(chunk)
        except Exception as e:
            logging.warning(f"Błąd Stanza (część {i + 1}): {e}")
            global_char_offset += len(chunk)
            continue

        if not doc.sentences:
            global_char_offset += len(chunk)
            continue

        # --- BUDOWANIE MAPY KOTWIC (HYBRYDA: ROLA + ID) DLA STANZA ---
        coref_anchors = {}
        try:
            # Stanza trzyma gotowe klastry w doc.coref: lista CorefChain (doc-level)
            if hasattr(doc, "coref") and doc.coref:
                for chain in doc.coref:
                    c_id_str = str(global_cluster_id_counter)
                    global_cluster_id_counter += 1

                    for mention in getattr(chain, "mentions", []):
                        s_idx = getattr(mention, "sentence", getattr(mention, "sent_id", None))
                        if s_idx is None:
                            continue

                        # Zabezpieczenie na zero-anaphora / nietypowe indeksy
                        if not isinstance(getattr(mention, "start_word", None), int) or not isinstance(
                                getattr(mention, "end_word", None), int):
                            continue

                        sentence_obj = doc.sentences[s_idx]
                        span_words = sentence_obj.words[mention.start_word:mention.end_word]
                        if not span_words:
                            continue

                        # --- Heurystyka wyznaczania Head ---
                        mention_ids = {w.id for w in span_words if isinstance(w.id, int)}
                        anchor = None
                        for w in span_words:
                            if not isinstance(w.id, int):
                                continue
                            if w.head not in mention_ids:
                                anchor = w
                                break
                        if anchor is None:
                            # fallback: ostatni "normalny" word, jeśli istnieje
                            anchor = next((w for w in reversed(span_words) if isinstance(w.id, int)), None)
                            if anchor is None:
                                continue

                        # --- Role Head/Part ---
                        for w in span_words:
                            if not isinstance(w.id, int):  # pomijamy puste węzły
                                continue
                            role = "Head" if w.id == anchor.id else "Part"
                            coref_anchors.setdefault((s_idx, w.id), []).append(f"{role}-{c_id_str}")

        except Exception as e:
            logging.warning(f"Błąd mapowania koreferencji Stanza: {e}")
        # -----------------------------------------------------------------



        chunk_char_pos = 0
        for sent_idx, sentence in enumerate(doc.sentences, start=1):
            real_sent_id = sent_idx + global_sent_id_offset
            # Current file progress
            current_progress = (i / total_chunks) + ((sent_idx / len(doc.sentences)) / total_chunks)
            if sent_idx % 10 == 0:
                progress_bar.set(current_progress)
                app.update_idletasks()

            word_to_ner = {word.id: token.ner for token in sentence.tokens for word in token.words}
            for word in sentence.words:
                start_idx_local = chunk.find(word.text, chunk_char_pos)
                if start_idx_local == -1: start_idx_local = chunk_char_pos
                end_idx_local = start_idx_local + len(word.text) - 1
                chunk_char_pos = end_idx_local + 1

                start_idx_global = start_idx_local + global_char_offset
                end_idx_global = end_idx_local + global_char_offset

                # Szybki odczyt gotowej etykiety
                sent_idx_stanza = sent_idx - 1
                coref_val = coref_anchors.get((sent_idx_stanza, word.id), [])

                all_processed_tokens.append({
                    "token": word.text,
                    "lemma": word.lemma,
                    "sentenceID": real_sent_id,
                    "wordID": word.id,
                    "headID": word.head,
                    "deprel": word.deprel,
                    "postag": word.xpos,
                    "start": start_idx_global,
                    "end": end_idx_global,
                    "ner": word_to_ner.get(word.id, "0"),
                    "upos": word.upos,
                    "coref": coref_val
                })
        global_sent_id_offset += len(doc.sentences)
        global_char_offset += len(chunk)
        del doc
    return all_processed_tokens


# --- HELPER DLA SPACY (ODTWARZANIE PEŁNYCH TAGÓW NKJP Z CECH MORFOLOGICZNYCH UD) ---
def reconstruct_nkjp_tag(tag_str, morph_obj):
    base = tag_str.lower()

    def m(key, mapping):
        vals = morph_obj.get(key)
        val = vals[0] if vals else ""
        # Jeśli brak cechy lub nie ma jej w mapowaniu, dajemy "_"
        return mapping.get(val, "_")

    num = m("Number", {"Sing": "sg", "Plur": "pl", "Ptan": "ptan"})
    case = m("Case",
             {"Nom": "nom", "Gen": "gen", "Dat": "dat", "Acc": "acc", "Ins": "inst", "Loc": "loc", "Voc": "voc"})

    gen_vals = morph_obj.get("Gender")
    gen_val = gen_vals[0] if gen_vals else ""
    anim_vals = morph_obj.get("Animacy")
    anim_val = anim_vals[0] if anim_vals else ""

    if gen_val == "Masc":
        if anim_val == "Hum":
            gen = "m1"
        elif anim_val == "Nhum":
            gen = "m2"
        elif anim_val == "Inan":
            gen = "m3"
        else:
            gen = "m1"
    elif gen_val == "Fem":
        gen = "f"
    elif gen_val == "Neut":
        gen = "n"
    else:
        gen = ""

    person = m("Person", {"1": "pri", "2": "sec", "3": "ter"})
    aspect = m("Aspect", {"Imp": "imperf", "Perf": "perf"})
    degree = m("Degree", {"Pos": "pos", "Cmp": "com", "Sup": "sup"})

    # Składanie łańcucha z zachowaniem ścisłej, pozycyjnej kolejności NKJP
    parts = [base]
    if base in ("subst", "depr"):
        parts.extend([num, case, gen])
    elif base in ("adj", "adja", "adjp"):
        parts.extend([num, case, gen, degree])
    elif base in ("ppron12", "ppron3"):
        parts.extend([num, case, gen, person, "", ""])
    elif base == "num":
        parts.extend([num, case, gen, ""])
    elif base in ("fin", "bedzie", "impt"):
        parts.extend([num, person, aspect])
    elif base in ("praet", "winien"):
        parts.extend([num, gen, aspect, ""])
    elif base in ("inf", "pcon", "pant", "imps"):
        parts.extend([aspect])
    elif base in ("ger", "pact", "ppas"):
        parts.extend([num, case, gen, aspect, ""])
    elif base == "adv":
        parts.extend([degree])

    if len(parts) == 1:
        return base
    return ":".join(parts)


def process_single_text_spacy(text, filename, status_label, progress_bar, app):
    if not text.strip(): return None
    nlp_spacy.max_length = 2000000
    chunks = chunk_text_safe(text, chunk_size=15000)
    all_processed_tokens = []
    global_sent_id_offset = 0
    global_char_offset = 0
    total_chunks = len(chunks)
    progress_bar.set(0)

    # --- NOWE: GLOBALNY LICZNIK DLA CAŁEGO PLIKU ---
    global_cluster_id_counter = 1

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            global_char_offset += len(chunk)
            continue
        try:
            update_status(status_label, f"Przetwarzam: {filename} (Część {i + 1}/{total_chunks})", app)
            doc = nlp_spacy(chunk)
        except Exception:
            global_char_offset += len(chunk)
            continue

        sentences = list(doc.sents)
        if not sentences:
            global_char_offset += len(chunk)
            continue

        # --- DODANE: MAPA KOTWIC (HERFERENCE: ROLA + ID) ---
        coref_anchors = {}
        if hasattr(doc._, "coref") and doc._.coref:
            try:
                text_obj = doc._.coref  # api.Text
                for cluster in text_obj.clusters:
                    if not getattr(cluster, "mentions", None):
                        continue

                    c_id_str = str(global_cluster_id_counter)
                    global_cluster_id_counter += 1

                    for mention in cluster.mentions:
                        span = getattr(mention, "span", None)

                        # Fallback: jeśli align nie ustawił span, spróbuj indices
                        if span is None:
                            idx = getattr(mention, "indices", None)
                            if idx and len(idx) == 2:
                                start, end = idx  # end w api.Mention jest inkluzywny
                                # defensywnie: end może być < start, sprawdzamy też granice dokumentu
                                if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(doc):
                                    span = doc[start:end + 1]

                        if not span:
                            continue

                        # --- Head/Part heurystyka ---
                        mention_tokens = set(span)
                        anchor = None
                        for token in span:
                            # Szukamy korzenia (słowa, którego nadrzędnik jest poza wzmianką)
                            if token.head not in mention_tokens or token.head == token:
                                anchor = token
                                break

                        if anchor is None:
                            anchor = span[-1]

                        for token in span:
                            role = "Head" if token.i == anchor.i else "Part"
                            coref_anchors.setdefault(token.i, []).append(f"{role}-{c_id_str}")

            except Exception as e:
                logging.warning(f"Błąd mapowania koreferencji (herference): {e}")
        # ----------------------------------------


        for sent_idx, sentence in enumerate(sentences, start=1):
            real_sent_id = sent_idx + global_sent_id_offset
            current_progress = (i / total_chunks) + ((sent_idx / len(sentences)) / total_chunks)
            if sent_idx % 20 == 0:
                progress_bar.set(current_progress)
                app.update_idletasks()

            for token in sentence:
                start_idx_global = token.idx + global_char_offset
                end_idx_global = start_idx_global + len(token.text) - 1

                # Pobieranie kotwicy. Jeśli brak, wstawiamy "O"
                coref_val = coref_anchors.get(token.i, [])

                # --- ODTWARZAMY PEŁNY TAG NKJP ---
                full_nkjp_tag = reconstruct_nkjp_tag(token.tag_, token.morph)

                all_processed_tokens.append({
                    "token": token.text,
                    "lemma": token.lemma_,
                    "sentenceID": real_sent_id,
                    "wordID": token.i + 1,
                    "headID": token.head.i + 1 if token.head != token else 0,
                    "deprel": token.dep_,
                    "postag": full_nkjp_tag,  # <--- TUTAJ UŻYWAMY NASZEGO HELPERA!
                    "start": start_idx_global,
                    "end": end_idx_global,
                    "ner": token.ent_type_ if token.ent_type_ else "O",
                    "upos": token.pos_,
                    "coref": coref_val
                })
        global_sent_id_offset += len(sentences)
        global_char_offset += len(chunk)
        del doc
        gc.collect()
    return all_processed_tokens


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


# --- UPDATED WORKER FUNCTION ---
import glob


def process_files_thread_target(status_label, progress_bar_current, progress_bar_total, lbl_size_info, app,
                                output_parquet_file,
                                metadata_path, model_name,
                                excel_mappings, resume_mode=False):
    global selected_files, nlp_stanza, nlp_spacy

    selected_paths = [path for path, var in selected_files.items() if var.get() == 1]
    if not selected_paths:
        status_label.configure(text="Nie wybrano pliku do przetworzenia.")
        return

    # --- TOTAL SIZE CALCULATION ---
    status_label.configure(text="Obliczam całkowity rozmiar zadań...")
    app.update_idletasks()

    try:
        total_size_bytes = calculate_real_total_size(selected_paths)
    except Exception as e:
        logging.warning(f"Błąd obliczania rozmiaru: {e}")
        total_size_bytes = 1

    if total_size_bytes == 0: total_size_bytes = 1
    total_size_str = format_size(total_size_bytes)

    processed_size_bytes = 0.0
    progress_bar_total.set(0)
    lbl_size_info.configure(text=f"0 B / {total_size_str}")

    # --- 1. METADANE LOADING (UPDATED WITH MAPPING) ---
    metadata_dict = {}
    extra_meta_columns = []

    if metadata_path:
        try:
            status_label.configure(text="Wczytuję metadane...")
            app.update_idletasks()
            df_meta = pd.read_excel(metadata_path)

            if excel_mappings and metadata_path in excel_mappings:
                meta_map = excel_mappings[metadata_path]
                rename_dict = {}
                for std_col, user_col in meta_map.items():
                    if user_col != "<Pomiń>":
                        rename_dict[user_col] = std_col
                df_meta.rename(columns=rename_dict, inplace=True)

            if "Nazwa pliku" in df_meta.columns:
                extra_meta_columns = [col for col in df_meta.columns if col != "Nazwa pliku"]
            else:
                logging.warning("Brak kolumny 'Nazwa pliku' w metadanych po mapowaniu.")

            pl_months = {
                'stycznia': '01', 'lutego': '02', 'marca': '03', 'kwietnia': '04',
                'maja': '05', 'czerwca': '06', 'lipca': '07', 'sierpnia': '08',
                'września': '09', 'października': '10', 'listopada': '11', 'grudnia': '12',
                'styczeń': '01', 'luty': '02', 'marzec': '03', 'kwiecień': '04',
                'maj': '05', 'czerwiec': '06', 'lipiec': '07', 'sierpień': '08',
                'wrzesień': '09', 'październik': '10', 'listopad': '11', 'grudzień': '12'
            }

            if "Nazwa pliku" in df_meta.columns:
                for _, row in df_meta.iterrows():
                    fn = str(row["Nazwa pliku"]).strip()
                    metadata_dict[fn] = {}
                    metadata_dict[fn.lower()] = metadata_dict[fn]

                    for col in df_meta.columns:
                        if col == "Nazwa pliku": continue
                        val = row[col]

                        if col == "Data publikacji" and pd.notna(val):
                            if isinstance(val, (pd.Timestamp, datetime)):
                                try:
                                    metadata_dict[fn][col] = val.strftime('%Y-%m-%d')
                                except:
                                    metadata_dict[fn][col] = str(val)
                            else:
                                val_str = str(val).strip()
                                match = re.search(r'(\d{1,2})\s+([a-ząćęłńóśźż]+)\s+(\d{4})', val_str.lower())
                                if match:
                                    d, m_txt, y = match.groups()
                                    if m_txt in pl_months:
                                        metadata_dict[fn][col] = f"{y}-{pl_months[m_txt]}-{d.zfill(2)}"
                                    else:
                                        metadata_dict[fn][col] = val_str
                                else:
                                    metadata_dict[fn][col] = val_str
                        else:
                            metadata_dict[fn][col] = str(val).strip() if pd.notna(val) else ""
            status_label.configure(text="Metadane wczytane.")
        except Exception as e:
            logging.warning(f"Błąd metadanych: {e}")

    # 2. Init Model
    if model_name == "Stanza":
        if not initialize_stanza(status_label, app): return
    else:
        if not initialize_spacy(status_label, app): return

    progress_bar_current.set(0)
    app.update_idletasks()

    global_base_tf = Counter()
    global_orth_tf = Counter()
    global_total_tokens = 0
    global_token_counts = {}

    # 3. WZNAWIANIE Z CHECKPOINTÓW
    # 3. WZNAWIANIE Z CHECKPOINTÓW
    BATCH_SIZE = 20  # Zmniejszony bufor: częstsze zapisy, mniejsze ryzyko utraty po przerwaniu
    batch_data = []
    temp_files_created = []
    batch_counter = 0
    processed_set = set()

    if resume_mode:
        # --- RATOWANIE GŁÓWNEGO PLIKU PARQUET ---
        # Jeśli główny plik istnieje (bo np. wczorajsze scalanie przerwało w połowie),
        # zmieniamy mu nazwę na plik tymczasowy. Dzięki temu program wczyta z niego
        # to, co wczoraj zrobił i scali to wszystko na nowo na samym końcu.
        if os.path.exists(output_parquet_file):
            try:
                recovered_name = f"{output_parquet_file}.part_000_recovered"
                os.rename(output_parquet_file, recovered_name)
                logging.warning(f"Odzyskano główny plik jako: {recovered_name}")
            except Exception as e:
                logging.warning(f"Nie udało się zmienić nazwy głównego pliku: {e}")
        # ----------------------------------------

        existing_parts = glob.glob(f"{output_parquet_file}.part_*")
        existing_parts.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        if existing_parts:
            status_label.configure(text="Odtwarzanie danych z poprzedniej sesji...")
            app.update_idletasks()

            for p_file in existing_parts:
                try:
                    df_part = pd.read_parquet(p_file)

                    # Odtwarzamy nazwy plików - ujednolicone do małych liter bez spacji
                    if "Oryginalna_nazwa_pliku" in df_part.columns:
                        processed_set.update(
                            [str(x).strip().lower() for x in df_part["Oryginalna_nazwa_pliku"].tolist()])
                    elif "Tytuł" in df_part.columns:
                        processed_set.update([str(x).strip().lower() for x in df_part["Tytuł"].tolist()])

                    # Odtwarzamy statystyki liczników
                    for _, row in df_part.iterrows():
                        toks = row.get("tokens", [])
                        lems = row.get("lemmas", [])
                        if hasattr(toks, "tolist"): toks = toks.tolist()
                        if hasattr(lems, "tolist"): lems = lems.tolist()

                        global_orth_tf.update(toks)
                        global_base_tf.update(lems)
                        global_total_tokens += len(toks)

                        p_date = str(row.get("Data publikacji", "0000-00-00")).strip()
                        parts = p_date.split('-')
                        y = parts[0] if len(parts) > 0 else "0000"
                        m = parts[1] if len(parts) > 1 else "00"
                        if y not in global_token_counts: global_token_counts[y] = {}
                        if m not in global_token_counts[y]: global_token_counts[y][m] = 0
                        global_token_counts[y][m] += len(toks)

                    temp_files_created.append(p_file)
                except Exception as e:
                    logging.warning(f"Błąd odtwarzania punktu kontrolnego {p_file}: {e}")

            # --- MOST DLA STARYCH CHECKPOINTÓW Z METADANYMI ---
            for fname, meta in metadata_dict.items():
                meta_title = str(meta.get("Tytuł", "")).strip().lower()
                if meta_title and meta_title in processed_set:
                    processed_set.add(str(fname).strip().lower())
            # ------------------------------------------------

            # BEZPIECZNY LICZNIK PLIKÓW (chroni przed nadpisaniem, gdy brakuje plików od 0)
            max_num = -1
            for p_file in temp_files_created:
                try:
                    num = int(p_file.split('_')[-1])
                    if num > max_num:
                        max_num = num
                except:
                    pass
            batch_counter = max_num + 1

            status_label.configure(text=f"Wznowiono: wczytano {len(processed_set)} gotowych tekstów.")
            app.update_idletasks()



    start_time = time.time()
    actual_processing_bytes = 0

    total_files_count = len(selected_paths)

    text_columns_to_force = ["Oryginalna_nazwa_pliku", "Tytuł", "Treść", "Data publikacji",
                             "Autor"] + extra_meta_columns

    try:
        for idx, file_path in enumerate(selected_paths):
            filename = os.path.basename(file_path)

            if metadata_path and os.path.abspath(file_path) == os.path.abspath(metadata_path):
                processed_size_bytes += os.path.getsize(file_path)
                continue
            if filename.lower() == "metadane.xlsx":
                processed_size_bytes += os.path.getsize(file_path)
                continue

            status_label.configure(text=f"Plik {idx + 1}/{total_files_count}: {filename}")
            app.update_idletasks()

            # --- TUTAJ PRZEKAZUJEMY processed_set ---
            for item in process_file_global(file_path, status_label, progress_bar_current, app, model_name,
                                            excel_mappings, processed_set):

                consumed = item.get("bytes_consumed", 0)
                processed_size_bytes += consumed

                # Liczymy bajty tylko dla faktycznie robionych plików (do ETA)
                if not item.get("skipped"):
                    actual_processing_bytes += consumed

                if total_size_bytes > 0:
                    prog = processed_size_bytes / total_size_bytes
                    if prog > 1.0: prog = 1.0
                    progress_bar_total.set(prog)

                    curr_str = format_size(processed_size_bytes)

                    # --- OBLICZANIE ETA ---
                    elapsed = time.time() - start_time
                    eta_str = "--:--"
                    if elapsed > 5 and actual_processing_bytes > 0:
                        speed = actual_processing_bytes / elapsed
                        remaining_bytes = total_size_bytes - processed_size_bytes
                        if remaining_bytes > 0 and speed > 0:
                            eta_secs = remaining_bytes / speed
                            m, s = divmod(int(eta_secs), 60)
                            h, m = divmod(m, 60)
                            if h > 0:
                                eta_str = f"{h}h {m}m"
                            else:
                                eta_str = f"{m}m {s}s"
                        else:
                            eta_str = "0s"

                    lbl_size_info.configure(text=f"{curr_str} / {total_size_str} | ETA: {eta_str}")
                    app.update_idletasks()
                    # ----------------------

                if item.get("skipped"):
                    skipped_name = item.get("filename", "nieznany plik")
                    status_label.configure(text=f"Pomijam gotowy: {skipped_name}")
                    app.update_idletasks()
                    continue

                processed_tokens = item.get("tokens_detail", [])
                text = item.get("Treść", "")
                fname_processed = item.get("filename", filename)
                meta_override = item.get("meta_override", {})

                entry = {
                    "Oryginalna_nazwa_pliku": fname_processed,  # <--- TA KOLUMNA RATUJE WZNAWIANIE
                    "Tytuł": fname_processed,
                    "Treść": text,
                    "Data publikacji": "0000-00-00",
                    "Autor": "#"
                }

                for col in extra_meta_columns:
                    entry[col] = ""

                matched_meta = None
                if fname_processed in metadata_dict:
                    matched_meta = metadata_dict[fname_processed]
                elif os.path.splitext(fname_processed)[0] in metadata_dict:
                    matched_meta = metadata_dict[os.path.splitext(fname_processed)[0]]
                elif fname_processed.lower() in metadata_dict:
                    matched_meta = metadata_dict[fname_processed.lower()]

                if matched_meta:
                    entry.update(matched_meta)

                if meta_override:
                    if meta_override.get("Tytuł"): entry["Tytuł"] = meta_override["Tytuł"]
                    if meta_override.get("Data publikacji"): entry["Data publikacji"] = meta_override["Data publikacji"]
                    if meta_override.get("Autor"): entry["Autor"] = meta_override["Autor"]


                tokens_list = [t["token"] for t in processed_tokens]
                lemmas_list = [t["lemma"] for t in processed_tokens]

                global_orth_tf.update(tokens_list)
                global_base_tf.update(lemmas_list)
                global_total_tokens += len(tokens_list)

                entry["tokens"] = tokens_list
                entry["lemmas"] = lemmas_list
                entry["postags"] = [t["postag"].split(":")[0] if t["postag"] else "" for t in processed_tokens]
                entry["full_postags"] = [t["postag"] for t in processed_tokens]
                entry["deprels"] = [t["deprel"] for t in processed_tokens]
                entry["word_ids"] = [t["wordID"] for t in processed_tokens]
                entry["sentence_ids"] = [t["sentenceID"] for t in processed_tokens]
                entry["head_ids"] = [t["headID"] for t in processed_tokens]
                entry["start_ids"] = [t["start"] for t in processed_tokens]
                entry["end_ids"] = [t["end"] for t in processed_tokens]
                entry["ners"] = [t["ner"] for t in processed_tokens]
                entry["upostags"] = [t["upos"] for t in processed_tokens]
                entry["corefs"] = [t.get("coref", []) for t in processed_tokens]

                try:
                    p_date = str(entry.get("Data publikacji", "0000-00-00")).strip()
                    parts = p_date.split('-')
                    y = parts[0] if len(parts) > 0 else "0000"
                    m = parts[1] if len(parts) > 1 else "00"
                    if y not in global_token_counts: global_token_counts[y] = {}
                    if m not in global_token_counts[y]: global_token_counts[y][m] = 0
                    global_token_counts[y][m] += len(entry["tokens"])
                except:
                    pass

                batch_data.append(entry)

                if len(batch_data) >= BATCH_SIZE:
                    part_file = f"{output_parquet_file}.part_{batch_counter}"
                    status_label.configure(text=f"Zapisuję bufor ({len(batch_data)} wpisów)...")
                    app.update_idletasks()

                    df_batch = pd.DataFrame(batch_data)
                    for col in text_columns_to_force:
                        if col in df_batch.columns:
                            df_batch[col] = df_batch[col].fillna("").astype(str)

                    df_batch.to_parquet(part_file, engine='pyarrow', compression='snappy')
                    temp_files_created.append(part_file)
                    batch_counter += 1
                    batch_data = []
                    del df_batch
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

        if batch_data:
            part_file = f"{output_parquet_file}.part_{batch_counter}"
            df_batch = pd.DataFrame(batch_data)
            for col in text_columns_to_force:
                if col in df_batch.columns:
                    df_batch[col] = df_batch[col].fillna("").astype(str)
            df_batch.to_parquet(part_file, engine='pyarrow', compression='snappy')
            temp_files_created.append(part_file)
            del df_batch
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    except Exception as e:
        messagebox.showerror("Błąd", str(e))
        return

    # 4. Merging
    status_label.configure(text="Scalanie plików...")
    progress_bar_current.set(0)
    progress_bar_total.set(1.0)
    lbl_size_info.configure(text=f"{total_size_str} / {total_size_str}")
    app.update_idletasks()

    metadata_export = {
        "base_tf": dict(global_base_tf),
        "orth_tf": dict(global_orth_tf),
        "total_tokens": global_total_tokens,
        "monthly_token_counts": global_token_counts
    }
    meta_json_bytes = json.dumps(metadata_export, ensure_ascii=False).encode('utf-8')

    final_writer = None
    reference_columns = None

    try:
        total_parts = len(temp_files_created)
        for i, part_file in enumerate(temp_files_created):
            progress_bar_current.set((i + 1) / total_parts)

            logging.warning(f"Scalam część {i + 1} z {total_parts}: {part_file}")

            status_label.configure(text=f"Scalanie plików... (paczka {i + 1}/{total_parts})")
            app.update_idletasks()

            df_part = pd.read_parquet(part_file)

            # --- NAPRAWA SCHEMATÓW I KOLEJNOŚCI ---
            # Najpierw dodajemy brakujące podstawowe kolumny tekstowe (dla pewności)
            for col in text_columns_to_force:
                if col not in df_part.columns:
                    df_part[col] = ""

            # Jeśli to pierwsza paczka, zapamiętujemy jej układ kolumn jako WZÓR
            if reference_columns is None:
                reference_columns = df_part.columns.tolist()
            else:
                # Dla każdej kolejnej paczki upewniamy się, że ma wszystkie kolumny ze wzoru...
                for col in reference_columns:
                    if col not in df_part.columns:
                        df_part[col] = ""


                df_part = df_part[reference_columns]
            # --------------------------------------------------

            table = pa.Table.from_pandas(df_part)

            if final_writer is None:
                existing_meta = table.schema.metadata or {}
                merged_meta = {**existing_meta, b"korpus_meta": meta_json_bytes}
                table = table.replace_schema_metadata(merged_meta)

                final_writer = pq.ParquetWriter(output_parquet_file, table.schema, compression='snappy')
            else:
                table = table.cast(final_writer.schema)

            final_writer.write_table(table)
            del df_part
            del table
            gc.collect()

            # Próbujemy usunąć plik po udanym scaleniu
            try:
                os.remove(part_file)
            except:
                pass

        status_label.configure(text=f"Gotowe! Plik: {os.path.basename(output_parquet_file)}")
        progress_bar_current.set(1.0)
        app.update_idletasks()
        app.after(0, lambda: messagebox.showinfo("Sukces", "Zakończono przetwarzanie i scalanie plików!"))

    except Exception as e:
        # Tego brakowało! Teraz jeśli coś wybuchnie, zobaczysz dlaczego.
        error_msg = f"Wystąpił błąd krytyczny podczas scalania plików:\n{str(e)}"
        logging.warning(error_msg)
        app.after(0, lambda: messagebox.showerror("Błąd scalania", error_msg))
        status_label.configure(text="Błąd scalania!")

    finally:
        if final_writer:
            final_writer.close()

    status_label.configure(text=f"Gotowe! Plik: {os.path.basename(output_parquet_file)}")
    progress_bar_current.set(1.0)
    messagebox.showinfo("Sukces", "Zakończono przetwarzanie.")




# --- UI ---
def select_files(frame, progress_bar_current, progress_bar_total, lbl_size_info, status_label, app):
    global selected_files, file_buttons
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

    for file_path in file_paths:
        if file_path not in selected_files:
            var = ctk.IntVar(value=1)
            selected_files[file_path] = var
            btn = ctk.CTkCheckBox(frame, text=os.path.basename(file_path), variable=var)
            btn.pack(anchor="w", padx=20, pady=10)
            file_buttons.append(btn)

    # Hide bars initially
    progress_bar_current.grid_remove()
    progress_bar_total.grid_remove()
    lbl_size_info.grid_remove()
    status_label.configure(text="Zaznacz pliki, które mają zostać przetworzone.")


def main(parent_window=None):
    global model, selected_files, file_buttons
    selected_files.clear()
    file_buttons.clear()
    if parent_window:
        app = ctk.CTkToplevel(parent_window)
        app.transient(parent_window)  # Trzyma okno kreatora nad głównym oknem
        app.grab_set()                # Blokuje klikanie w główne okno
    else:
        app = ctk.CTk()

    def center_window(app, width=800, height=600):
        screen_width = app.winfo_screenwidth()
        screen_height = app.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        app.geometry(f"{width}x{height}+{x}+{y}")

    center_window(app, 800, 600)
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

    select_button = ctk.CTkButton(
        main_frame,
        text="Wybierz pliki",
        command=lambda: select_files(checkbox_frame, progress_bar_current, progress_bar_total, lbl_size_info,
                                     status_label, app),
        font=("Verdana", 12, 'bold'),
        corner_radius=8, height=35,
        fg_color='#4B6CB7', hover_color="#5B7CD9",
    )
    select_button.grid(row=1, column=0, columnspan=2, pady=10)

    # --- PASKI POSTĘPU ---

    # 1. Total Label
    lbl_total = ctk.CTkLabel(main_frame, text="Postęp całkowity:", font=("Verdana", 10))
    lbl_total.grid(row=4, column=0, sticky="w", padx=20)

    # NOWY ELEMENT: Etykieta rozmiaru (np. 15 MB / 100 MB)
    lbl_size_info = ctk.CTkLabel(main_frame, text="", font=("Verdana", 10, "bold"), text_color="#555555")
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

    status_label = ctk.CTkLabel(main_frame, text="Gotowy", font=("Verdana", 12, 'bold'))
    status_label.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

    # Ukrycie paska rozmiaru na start
    lbl_size_info.grid_remove()

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

        process_button.configure(state="disabled")

        progress_bar_total.grid()
        progress_bar_current.grid()
        lbl_size_info.grid()

        threading.Thread(
            target=process_files_thread_target,
            args=(
                status_label, progress_bar_current, progress_bar_total, lbl_size_info, app, output_file, metadata_path,
                model.get(),
                excel_mappings, resume_mode),  # <--- PRZEKAZUJEMY resume_mode
            daemon=True
        ).start()

    process_button = ctk.CTkButton(main_frame, text="Przetwórz pliki", command=start_processing,
                                   font=("Verdana", 12, 'bold'),
                                   corner_radius=8, height=35,
                                   fg_color='#4B6CB7', hover_color="#5B7CD9")
    process_button.grid(row=2, column=0, columnspan=2, pady=10)

    checkbox_frame = ctk.CTkScrollableFrame(app)
    checkbox_frame.pack(pady=6, fill="both", expand=True, side="right")

    switch_var = ctk.StringVar(value="on")

    def toggle_all():
        val = 1 if switch_var.get() == "on" else 0
        for var in selected_files.values(): var.set(val)

    toggle_button = ctk.CTkSwitch(checkbox_frame, text="Zaznacz wszystko", command=toggle_all,
                                  variable=switch_var, onvalue="on", offvalue="off")
    toggle_button.pack(padx=20, pady=20)

    if not parent_window:
        app.mainloop()


if __name__ == "__main__":
    main()