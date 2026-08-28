# -*- coding: utf-8 -*-
"""GUI-independent orchestration for corpus creation jobs.

This module coordinates creator options, progress reporting, model setup,
input processing, resume handling and publication of the canonical Parquet
corpus without importing GUI components.
"""
from __future__ import annotations
from korpusuj.runtime_paths import models_root, writable_temp_root
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from collections import Counter
from datetime import datetime
import gc, glob, json, logging, os, re, shutil, sys, tempfile, time, zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from korpusuj.runtime_paths import configure_ml_cache_environment as _configure_ml_cache_environment_182n
_configure_ml_cache_environment_182n()

import requests, spacy, torch
from docx import Document
try:
    import stanza
except ImportError:
    stanza = None
try:
    import fitz
except ImportError:
    fitz = None
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    import typing, torch.utils.data.dataset
    if not hasattr(torch.utils.data.dataset, "T_co"):
        torch.utils.data.dataset.T_co = typing.TypeVar("T_co", covariant=True)
    import herference
except Exception:
    herference = None
from korpusuj.corpus.creator_core import CreatorRunOptions, NullProgressReporter, ProgressReporter
from korpusuj.corpus.lemma_corrections import (
    LemmaCorrectionsError,
    apply_lemma_corrections,
    disabled_lemma_corrections,
    lemma_corrections_identity,
    lemma_corrections_metadata,
    load_lemma_corrections,
)
from korpusuj.corpus.creator_io import calculate_real_total_size, process_xlsx
from korpusuj.corpus.creator_nlp import (
    CreatorModelState, initialize_spacy as _initialize_spacy,
    initialize_stanza as _initialize_stanza,
    process_single_text as _process_single_text,
    process_single_text_spacy as _process_single_text_spacy,
)

@dataclass(slots=True)
class CreatorRunResult:
    """Describe the published corpus and counters produced by a creator job."""
    success: bool
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

class _Selected:
    def get(self): return 1
class _Status:
    def __init__(self, reporter): self.reporter = reporter
    def configure(self, *, text="", **_kw): self.reporter.status(str(text))
class _Progress:
    def __init__(self, setter): self.setter = setter
    def set(self, value): self.setter(float(value))
class _Size:
    def __init__(self, reporter): self.reporter = reporter
    def configure(self, *, text="", **_kw): self.reporter.size_info(str(text))
class _App:
    def __init__(self, reporter): self.reporter = reporter
    def update_idletasks(self): self.reporter.tick()
    def after(self, _delay, callback): callback()

_model_state = CreatorModelState()
_reporter = NullProgressReporter()
_models_dir = ""
_active_input_selection = {}
nlp_stanza = None
nlp_spacy = None
_enable_ner = True
_enable_coreference = True

def _sync_models():
    global nlp_stanza, nlp_spacy
    nlp_stanza, nlp_spacy = _model_state.nlp_stanza, _model_state.nlp_spacy

def initialize_stanza(_label, _app):
    result = _initialize_stanza(
        _model_state, _reporter, stanza_module=stanza, models_dir=_models_dir,
        enable_ner=_enable_ner, enable_coreference=_enable_coreference,
    )
    _sync_models(); return bool(result)

def initialize_spacy(_label, _app):
    result = _initialize_spacy(
        _model_state, _reporter, spacy_module=spacy, herference_module=herference,
        requests_module=requests, models_dir=_models_dir, enable_ner=_enable_ner,
        enable_coreference=_enable_coreference,
    )
    _sync_models(); return bool(result)

# Technical Unicode formatting characters observed in web-derived Polish text.
# This explicit list is intentionally narrower than the whole Unicode Cf category.
_NLP_UNSAFE_FORMAT_TRANSLATION = str.maketrans({
    "\u00ad": None,
    "\u200b": None,
    "\u200c": None,
    "\u200d": None,
    "\u200e": None,
    "\u2060": None,
    "\u2063": None,
    "\u2066": None,
})
_UNICODE_NORMALIZATION_STATS = {"documents": 0, "characters": 0}


def _normalize_creator_text_for_nlp(text):
    value = str(text or "")
    normalized = value.translate(_NLP_UNSAFE_FORMAT_TRANSLATION)
    removed = len(value) - len(normalized)
    if removed:
        _UNICODE_NORMALIZATION_STATS["documents"] += 1
        _UNICODE_NORMALIZATION_STATS["characters"] += removed
    return normalized


def _reset_unicode_normalization_stats():
    _UNICODE_NORMALIZATION_STATS["documents"] = 0
    _UNICODE_NORMALIZATION_STATS["characters"] = 0


def _log_unicode_normalization_summary():
    documents = _UNICODE_NORMALIZATION_STATS["documents"]
    characters = _UNICODE_NORMALIZATION_STATS["characters"]
    if documents:
        logging.warning(
            "CREATOR_UNICODE_NORMALIZATION | documents=%d | removed_characters=%d",
            documents,
            characters,
        )


def _apply_optional_annotation_contract(processed_tokens):
    if processed_tokens is None:
        return None
    if not _enable_ner:
        for token in processed_tokens:
            token["ner"] = "O"
    if not _enable_coreference:
        for token in processed_tokens:
            token["coref"] = []
        mentions = getattr(processed_tokens, "coref_mentions", None)
        if mentions is not None:
            mentions.clear()
    return processed_tokens

def process_single_text(text, filename, *_legacy):
    return _apply_optional_annotation_contract(
        _process_single_text(text, filename, _model_state, _reporter)
    )

def process_single_text_spacy(text, filename, *_legacy):
    return _apply_optional_annotation_contract(
        _process_single_text_spacy(text, filename, _model_state, _reporter)
    )

def _schedule_creator_completion(_app, callback, success, output_file=None, error_message=None):
    if callback: callback(success=success, output_file=output_file, error_message=error_message)

def _option(options, name, default=None): return getattr(options, name, default)


_active_lemma_corrections = disabled_lemma_corrections()


def _lemma_corrections_metadata():
    return lemma_corrections_metadata(_active_lemma_corrections)


def _read_declared_lemma_corrections(parquet_file):
    try:
        metadata = pq.read_metadata(parquet_file).metadata or {}
        raw_meta = metadata.get(b"korpus_meta")
        if not raw_meta:
            return None
        parsed = json.loads(raw_meta.decode("utf-8"))
        value = parsed.get("lemma_corrections")
        return value if isinstance(value, dict) else None
    except Exception as exc:
        logging.warning("Nie można odczytać lemma_corrections z %s: %s", parquet_file, exc)
        return None


def _validate_resume_lemma_corrections(parquet_file):
    declared = _read_declared_lemma_corrections(parquet_file)
    current = lemma_corrections_identity(_active_lemma_corrections)
    if declared is None:
        if current.get("enabled"):
            raise ValueError(
                "Niezgodna konfiguracja lemma_corrections dla resume: "
                f"plik={parquet_file!r} nie deklaruje korekt, a bieżący przebieg je włącza."
            )
        return
    declared_identity = {key: declared.get(key) for key in current}
    if declared_identity != current:
        raise ValueError(
            "Niezgodna konfiguracja lemma_corrections dla resume: "
            f"plik={parquet_file!r}, zapisane={declared_identity!r}, bieżące={current!r}."
        )


def _annotation_layers_metadata():
    return {
        "ner": bool(_enable_ner),
        "coreference": bool(_enable_coreference),
    }


def _read_declared_annotation_layers(parquet_file):
    try:
        metadata = pq.read_metadata(parquet_file).metadata or {}
        raw_meta = metadata.get(b"korpus_meta")
        if not raw_meta:
            return None
        parsed = json.loads(raw_meta.decode("utf-8"))
        layers = parsed.get("annotation_layers")
        if not isinstance(layers, dict):
            return None
        if "ner" not in layers or "coreference" not in layers:
            return None
        return {
            "ner": bool(layers["ner"]),
            "coreference": bool(layers["coreference"]),
        }
    except Exception as exc:
        logging.warning(
            "Nie można odczytać annotation_layers z %s: %s; "
            "traktuję jako unknown/undeclared.", parquet_file, exc,
        )
        return None


def _validate_resume_annotation_layers(parquet_file):
    declared = _read_declared_annotation_layers(parquet_file)
    current = _annotation_layers_metadata()
    if declared is None:
        logging.warning(
            "Resume: %s nie deklaruje annotation_layers; stan pozostaje "
            "unknown/undeclared i nie jest interpretowany jako disabled.",
            parquet_file,
        )
        return
    if declared != current:
        raise ValueError(
            "Niezgodna konfiguracja annotation_layers dla resume: "
            f"plik={parquet_file!r}, zapisane={declared!r}, bieżące={current!r}. "
            "Nie można scalać części utworzonych z różnymi warstwami anotacji."
        )


def _write_creator_part(dataframe, part_file):
    table = pa.Table.from_pandas(dataframe)
    existing_meta = table.schema.metadata or {}
    part_meta = {
        "annotation_layers": _annotation_layers_metadata(),
        "lemma_corrections": _lemma_corrections_metadata(),
    }
    merged_meta = {
        **existing_meta,
        b"korpus_meta": json.dumps(part_meta, ensure_ascii=False).encode("utf-8"),
    }
    table = table.replace_schema_metadata(merged_meta)
    pq.write_table(table, part_file, compression="snappy")


def _application_root():
    if getattr(sys, "frozen", False): return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]

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
                        easyocr_dir = Path(_models_dir) / "easyocr"
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

                text = _normalize_creator_text_for_nlp(it["Treść"])

                if model_name == "Stanza":
                    tokens = process_single_text(text, virt_fname, status_label, progress_bar, app)
                else:
                    tokens = process_single_text_spacy(text, virt_fname, status_label, progress_bar, app)

                if tokens is None:
                    raise RuntimeError(
                        "CREATOR_NLP_DOCUMENT_FAILURE | "
                        f"document={virt_fname!r} | NLP returned no annotation"
                    )

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
                text = _normalize_creator_text_for_nlp(text)
                if model_name == "Stanza":
                    tokens = process_single_text(text, file_base, status_label, progress_bar, app)
                else:
                    tokens = process_single_text_spacy(text, file_base, status_label, progress_bar, app)

                if tokens is None:
                    raise RuntimeError(
                        "CREATOR_NLP_DOCUMENT_FAILURE | "
                        f"document={file_base!r} | NLP returned no annotation"
                    )

                if tokens:
                    yield {
                        "filename": file_base,
                        "Treść": text,
                        "tokens_detail": tokens,
                        "bytes_consumed": current_file_size
                    }

def _run_creator_job_impl(status_label, progress_bar_current, progress_bar_total, lbl_size_info, app,
                                output_parquet_file,
                                metadata_path, model_name,
                                excel_mappings, resume_mode=False, completion_callback=None):
    global _active_input_selection, nlp_stanza, nlp_spacy

    selected_paths = [path for path, var in _active_input_selection.items() if var.get() == 1]
    if not selected_paths:
        status_label.configure(text="Nie wybrano pliku do przetworzenia.")
        _schedule_creator_completion(
            app, completion_callback, False,
            error_message="Nie wybrano pliku do przetworzenia.",
        )
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
        if not initialize_stanza(status_label, app):
            _schedule_creator_completion(
                app, completion_callback, False,
                error_message="Nie udało się zainicjalizować modelu Stanza.",
            )
            return
    else:
        if not initialize_spacy(status_label, app):
            _schedule_creator_completion(
                app, completion_callback, False,
                error_message="Nie udało się zainicjalizować modelu SpaCy.",
            )
            return

    progress_bar_current.set(0)
    app.update_idletasks()

    global_base_tf = Counter()
    global_orth_tf = Counter()
    global_total_tokens = 0
    global_token_counts = {}

    # 3. WZNAWIANIE Z CHECKPOINTÓW
    BATCH_SIZE = 20  # Zmniejszony bufor: częstsze zapisy, mniejsze ryzyko utraty po przerwaniu
    _reset_unicode_normalization_stats()
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
                    _validate_resume_annotation_layers(p_file)
                    _validate_resume_lemma_corrections(p_file)
                except ValueError as exc:
                    error_message = str(exc)
                    logging.warning(error_message)
                    _schedule_creator_completion(
                        app, completion_callback, False,
                        error_message=error_message,
                    )
                    return

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


                apply_lemma_corrections(processed_tokens, _active_lemma_corrections)
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
                # --- COREF_MENTIONS_DOCUMENT_COLUMN_174J1 ---
                entry["coref_mentions"] = list(getattr(processed_tokens, "coref_mentions", []) or [])
                # --- END COREF_MENTIONS_DOCUMENT_COLUMN_174J1 ---

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

                    _write_creator_part(df_batch, part_file)
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
            _write_creator_part(df_batch, part_file)
            temp_files_created.append(part_file)
            del df_batch
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    except Exception as e:
        error_message = str(e)
        logging.exception("Błąd przetwarzania plików")
        _schedule_creator_completion(
            app, completion_callback, False, error_message=error_message
        )
        return

    # 4. Merging
    _log_unicode_normalization_summary()
    status_label.configure(text="Scalanie plików...")
    progress_bar_current.set(0)
    progress_bar_total.set(1.0)
    lbl_size_info.configure(text=f"{total_size_str} / {total_size_str}")
    app.update_idletasks()

    metadata_export = {
        "base_tf": dict(global_base_tf),
        "orth_tf": dict(global_orth_tf),
        "total_tokens": global_total_tokens,
        "monthly_token_counts": global_token_counts,
        "annotation_layers": _annotation_layers_metadata(),
        "lemma_corrections": _lemma_corrections_metadata(),
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

        progress_bar_current.set(1.0)
        app.update_idletasks()

    except Exception as e:
        # Tego brakowało! Teraz jeśli coś wybuchnie, zobaczysz dlaczego.
        error_msg = f"Wystąpił błąd krytyczny podczas scalania plików:\n{str(e)}"
        logging.warning(error_msg)
        status_label.configure(text="Błąd scalania!")
        _schedule_creator_completion(
            app, completion_callback, False, error_message=error_msg
        )
        return

    finally:
        if final_writer:
            final_writer.close()

    status_label.configure(
        text=f"Gotowe: {os.path.basename(output_parquet_file)}"
    )
    progress_bar_current.set(1.0)
    _schedule_creator_completion(
        app,
        completion_callback,
        True,
        output_file=output_parquet_file,
    )


def run_creator_job(options, reporter=None, *, model_state=None, models_dir=None, cancel_requested=None):
    """Run the existing creator workflow without Tkinter dependencies."""
    global _model_state, _reporter, _models_dir, _active_input_selection
    global _enable_ner, _enable_coreference, _active_lemma_corrections
    _reporter = reporter or NullProgressReporter()
    _model_state = model_state or CreatorModelState()
    _models_dir = str(models_root(models_dir or _option(options, "models_dir", None)))
    _enable_ner = bool(_option(options, "enable_ner", True))
    _enable_coreference = bool(_option(options, "enable_coreference", True))
    try:
        _active_lemma_corrections = load_lemma_corrections(
            _option(options, "lemma_corrections_path", None)
        )
    except LemmaCorrectionsError as exc:
        return CreatorRunResult(False, error_message=str(exc))
    files = list(_option(options, "input_files", []) or [])
    if not files: return CreatorRunResult(False, error_message="Nie podano plików wejściowych.")
    if cancel_requested and cancel_requested(): return CreatorRunResult(False, error_message="Anulowano przed rozpoczęciem.")

    backend = str(_option(options, "model_name", "stanza") or "stanza").strip().lower()
    if backend == "stanza":
        runtime_model_name = "Stanza"
    elif backend == "spacy":
        runtime_model_name = "spaCy"
    else:
        return CreatorRunResult(
            False,
            error_message=f"Nieobsługiwany backend NLP: {backend!r}",
        )
    _active_input_selection = {str(path): _Selected() for path in files}
    payload = {}
    def completed(**values): payload.update(values)
    _run_creator_job_impl(
        _Status(_reporter), _Progress(_reporter.current), _Progress(_reporter.total),
        _Size(_reporter), _App(_reporter), str(_option(options, "output_parquet_file", "")),
        _option(options, "metadata_path", None), runtime_model_name,
        dict(_option(options, "excel_mappings", {}) or {}), bool(_option(options, "resume_mode", False)), completed,
    )
    return CreatorRunResult(bool(payload.get("success")), payload.get("output_file"), payload.get("error_message"))

__all__ = ["CreatorRunResult", "run_creator_job"]
