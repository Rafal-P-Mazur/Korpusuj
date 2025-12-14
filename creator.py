import os
import re
import json
import pandas as pd
import customtkinter as ctk
import tkinter.filedialog as fd
import threading
from docx import Document
from PIL import Image
import openpyxl
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

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
    i = int(os.path.abspath(os.path.curdir).find('some_impossible_str'))  # dummy
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





# --- FUNKCJE POMOCNICZE ---
def chunk_text_safe(text, chunk_size=50000):
    chunks = []
    paragraphs = text.split('\n')
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_full = paragraph + '\n'
        para_len = len(paragraph_full)

        if current_length + para_len <= chunk_size:
            current_chunk.append(paragraph_full)
            current_length += para_len
        elif para_len <= chunk_size:
            if current_chunk:
                chunks.append("".join(current_chunk))
            current_chunk = [paragraph_full]
            current_length = para_len
        else:
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_length = 0

            start = 0
            text_len = len(paragraph_full)
            while start < text_len:
                if text_len - start <= chunk_size:
                    chunks.append(paragraph_full[start:])
                    break

                end = start + chunk_size
                window_size = 500
                search_start = max(start, end - window_size)
                window = paragraph_full[search_start:end]

                safe_breaks = [m.start() for m in re.finditer(r'(?<=[.!?])\s', window)]

                if safe_breaks:
                    cut_offset = safe_breaks[-1]
                    real_end = search_start + cut_offset
                else:
                    last_space = window.rfind(' ')
                    if last_space != -1:
                        real_end = search_start + last_space
                    else:
                        real_end = end

                chunks.append(paragraph_full[start:real_end])
                start = real_end

    if current_chunk:
        chunks.append("".join(current_chunk))
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
            print(f"Błąd obliczania rozmiaru dla {path}: {e}")
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
        print(f"Błąd PDF: {e}")
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
        print(f"Błąd Excel {file_path}: {e}")
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
    model_dir = os.path.expanduser("~/stanza_resources/pl")
    if not os.path.exists(model_dir):
        try:
            status_label.configure(text="Proszę czekać - pobieram model Stanza")
            app.update_idletasks()
            stanza.download("pl")
        except Exception as e:
            messagebox.showerror("Błąd modelu Stanza", f"Nie udało się pobrać modelu: {e}")
            return False
    status_label.configure(text="Ładuję model Stanza - proszę czekać.")
    app.update_idletasks()
    try:
        nlp_stanza = stanza.Pipeline("pl", processors="tokenize,pos,lemma,ner,depparse", use_gpu=True, n_process=1)
        status_label.configure(text="Model Stanza załadowany")
        return True
    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się załadować Stanza: {e}")
        return False


def initialize_spacy(status_label, app):
    global nlp_spacy
    model_name = "pl_core_news_lg"
    try:
        status_label.configure(text="Sprawdzam model SpaCy...")
        app.update_idletasks()
        if not spacy.util.is_package(model_name):
            status_label.configure(text=f"Pobieram model '{model_name}' (może to potrwać)...")
            app.update_idletasks()
            try:
                spacy.cli.download(model_name)
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się automatycznie pobrać modelu SpaCy.\nSzczegóły: {e}")
                return False
        status_label.configure(text="Ładuję model SpaCy...")
        app.update_idletasks()
        nlp_spacy = spacy.load(model_name)
        status_label.configure(text=f"Model SpaCy '{model_name}' załadowany")
        return nlp_spacy
    except Exception as e:
        messagebox.showerror("Błąd modelu SpaCy", f"Nie udało się załadować modelu:\n{e}")
        return None


# --- NLP PROCESSING ---
def process_single_text(text, filename, status_label, progress_bar, app):
    if not text.strip(): return None
    chunks = chunk_text_safe(text, chunk_size=50000)
    all_processed_tokens = []
    global_sent_id_offset = 0
    global_char_offset = 0
    total_chunks = len(chunks)
    progress_bar.set(0)

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
            print(f"Błąd Stanza (część {i + 1}): {e}")
            global_char_offset += len(chunk)
            continue

        if not doc.sentences:
            global_char_offset += len(chunk)
            continue

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
                    "upos": word.upos
                })
        global_sent_id_offset += len(doc.sentences)
        global_char_offset += len(chunk)
        del doc
    return all_processed_tokens


def process_single_text_spacy(text, filename, status_label, progress_bar, app):
    if not text.strip(): return None
    nlp_spacy.max_length = 2000000
    chunks = chunk_text_safe(text, chunk_size=100000)
    all_processed_tokens = []
    global_sent_id_offset = 0
    global_char_offset = 0
    total_chunks = len(chunks)
    progress_bar.set(0)

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

        for sent_idx, sentence in enumerate(sentences, start=1):
            real_sent_id = sent_idx + global_sent_id_offset
            current_progress = (i / total_chunks) + ((sent_idx / len(sentences)) / total_chunks)
            if sent_idx % 20 == 0:
                progress_bar.set(current_progress)
                app.update_idletasks()

            for token in sentence:
                start_idx_global = token.idx + global_char_offset
                end_idx_global = start_idx_global + len(token.text) - 1
                all_processed_tokens.append({
                    "token": token.text,
                    "lemma": token.lemma_,
                    "sentenceID": real_sent_id,
                    "wordID": token.i + 1,
                    "headID": token.head.i + 1 if token.head != token else 0,
                    "deprel": token.dep_,
                    "postag": token.tag_,
                    "start": start_idx_global,
                    "end": end_idx_global,
                    "ner": token.ent_type_ if token.ent_type_ else "O",
                    "upos": token.pos_
                })
        global_sent_id_offset += len(sentences)
        global_char_offset += len(chunk)
        del doc
        gc.collect()
    return all_processed_tokens


def process_file_global(file_path, status_label, progress_bar, app, model_name, excel_mappings=None):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        current_file_size = os.path.getsize(file_path)
    except OSError:
        current_file_size = 0

    if ext == ".zip":
        status_label.configure(text=f"Rozpakowuję archiwum: {os.path.basename(file_path)}")
        extracted_files = unpack_archive(file_path, status_label)

        for inner_file in extracted_files:
            yield from process_file_global(inner_file, status_label, progress_bar, app, model_name, excel_mappings)
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
            # Tutaj dostajemy listę słowników z kluczem 'filename' wyciągniętym z kolumny
            rows = process_xlsx(file_path, mapping=mapping)
            total_rows = len(rows)
            bytes_per_row = current_file_size / total_rows if total_rows > 0 else 0

            for it in rows:
                text = it["Treść"]
                # Używamy filename z Excela (wirtualna nazwa pliku)
                virt_fname = it.get("filename", os.path.basename(file_path))

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
                    tokens = process_single_text(text, os.path.basename(file_path), status_label, progress_bar, app)
                else:
                    tokens = process_single_text_spacy(text, os.path.basename(file_path), status_label, progress_bar,
                                                       app)

                if tokens:
                    yield {
                        "filename": os.path.basename(file_path),
                        "Treść": text,
                        "tokens_detail": tokens,
                        "bytes_consumed": current_file_size
                    }


# --- UPDATED WORKER FUNCTION ---
def process_files_thread_target(status_label, progress_bar_current, progress_bar_total, lbl_size_info, app,
                                output_parquet_file,
                                metadata_path, model_name,
                                excel_mappings):
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
        print(f"Błąd obliczania rozmiaru: {e}")
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

            # --- APPLY MAPPING TO METADATA ---
            # If the user mapped columns for this metadata file, rename them in the DF
            if excel_mappings and metadata_path in excel_mappings:
                meta_map = excel_mappings[metadata_path]
                # Invert map: We need { "Original Column Name": "Standard Name" }
                # The mapper returns { "Standard Name": "Original Column Name" }
                rename_dict = {}
                for std_col, user_col in meta_map.items():
                    if user_col != "<Pomiń>":
                        rename_dict[user_col] = std_col

                df_meta.rename(columns=rename_dict, inplace=True)
            # ---------------------------------

            if "Nazwa pliku" in df_meta.columns:
                extra_meta_columns = [col for col in df_meta.columns if col != "Nazwa pliku"]
            else:
                print("Warning: Metadane file missing 'Nazwa pliku' column (after mapping).")

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
            print(f"Błąd metadanych: {e}")

    # 2. Init Model
    if model_name == "Stanza":
        if not initialize_stanza(status_label, app): return
    else:
        if not initialize_spacy(status_label, app): return

    progress_bar_current.set(0)
    app.update_idletasks()

    # 3. Processing
    BATCH_SIZE = 100
    batch_data = []
    temp_files_created = []
    global_token_counts = {}
    batch_counter = 0
    total_files_count = len(selected_paths)
    text_columns_to_force = ["Tytuł", "Treść", "Data publikacji", "Autor"] + extra_meta_columns

    try:
        for idx, file_path in enumerate(selected_paths):
            filename = os.path.basename(file_path)

            # Skip metadata file if it was selected in the main list
            if metadata_path and os.path.abspath(file_path) == os.path.abspath(metadata_path):
                processed_size_bytes += os.path.getsize(file_path)
                continue
            if filename.lower() == "metadane.xlsx":
                processed_size_bytes += os.path.getsize(file_path)
                continue

            status_label.configure(text=f"Plik {idx + 1}/{total_files_count}: {filename}")
            app.update_idletasks()

            for item in process_file_global(file_path, status_label, progress_bar_current, app, model_name,
                                            excel_mappings):

                # --- PROGRESS UPDATE ---
                consumed = item.get("bytes_consumed", 0)
                processed_size_bytes += consumed

                if total_size_bytes > 0:
                    prog = processed_size_bytes / total_size_bytes
                    if prog > 1.0: prog = 1.0
                    progress_bar_total.set(prog)

                    curr_str = format_size(processed_size_bytes)
                    lbl_size_info.configure(text=f"{curr_str} / {total_size_str}")
                    app.update_idletasks()
                # -----------------------

                processed_tokens = item.get("tokens_detail", [])
                text = item.get("Treść", "")
                fname_processed = item.get("filename", filename)
                meta_override = item.get("meta_override", {})

                entry = {
                    "Tytuł": fname_processed,
                    "Treść": text,
                    "Data publikacji": "0000-00-00",
                    "Autor": "#"
                }

                # Dynamic meta fields
                for col in extra_meta_columns:
                    entry[col] = ""

                # Meta Matching
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

                entry["tokens"] = [t["token"] for t in processed_tokens]
                entry["lemmas"] = [t["lemma"] for t in processed_tokens]
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

        # Flush remaining
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

    token_counts_json = json.dumps(global_token_counts, ensure_ascii=False)
    final_writer = None

    try:
        total_parts = len(temp_files_created)
        for i, part_file in enumerate(temp_files_created):
            progress_bar_current.set((i + 1) / total_parts)
            app.update_idletasks()

            df_part = pd.read_parquet(part_file)
            df_part["token_counts"] = token_counts_json

            table = pa.Table.from_pandas(df_part)

            if final_writer is None:
                final_writer = pq.ParquetWriter(output_parquet_file, table.schema, compression='snappy')
            final_writer.write_table(table)

            del df_part
            del table
            gc.collect()
            try:
                os.remove(part_file)
            except:
                pass
    finally:
        if final_writer: final_writer.close()

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


def main():
    global model
    app = ctk.CTk()
    app.attributes("-topmost", True)

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
            metadata_path = fd.askopenfilename(filetypes=[("Excel", "*.xlsx")])
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

        output_file = fd.asksaveasfilename(defaultextension=".parquet", filetypes=[("Parquet", "*.parquet")])
        if not output_file: return

        process_button.configure(state="disabled")

        progress_bar_total.grid()
        progress_bar_current.grid()
        lbl_size_info.grid()

        threading.Thread(
            target=process_files_thread_target,
            args=(
            status_label, progress_bar_current, progress_bar_total, lbl_size_info, app, output_file, metadata_path,
            model.get(),
            excel_mappings),
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

    app.mainloop()


if __name__ == "__main__":
    main()