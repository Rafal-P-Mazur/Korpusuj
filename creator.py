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


sys.stdout.reconfigure(encoding='utf-8', errors='replace')

nlp_stanza = None

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)

# Safe initial directory for file dialog
initial_dir = BASE_DIR

# Download and load the Polish Stanza model (only needs to be done once)
def initialize_stanza(status_label, app):
    import stanza
    global nlp_stanza
    model_dir = os.path.expanduser("~/stanza_resources/pl")

    if not os.path.exists(model_dir):
        # Model not downloaded
        try:
            status_label.configure(text="Model Stanza nie znaleziony, próbuję pobrać...")
            app.update_idletasks()
            stanza.download("pl")  # This will fail if no internet
        except Exception as e:
            messagebox.showerror(
                "Błąd modelu Stanza",
                "Model języka polskiego dla Stanza nie jest zainstalowany i nie można go pobrać. "
                "Proszę pobrać go na komputerze z dostępem do internetu i skopiować katalog:\n"
                "~/.cache/stanza_resources/pl\n"
                "lub ręcznie uruchomić:\nstanza.download('pl')"
            )
            status_label.configure(text="Brak modelu Stanza - przetwarzanie zatrzymane.")
            return False  # Initialization failed

    # Model exists, load pipeline
    status_label.configure(text="Ładuję model Stanza - proszę czekać.")
    nlp_stanza = stanza.Pipeline("pl", processors="tokenize,pos,lemma,ner,depparse", use_gpu=True, n_process=5)
    status_label.configure(text="Model Stanza załadowany")
    return True


def unpack_archive(file_path, status_label):
    """Unpack ZIP  into a temp dir and return list of extracted file paths."""
    temp_dir = tempfile.mkdtemp(prefix="archive_extract_")
    extracted_files = []

    try:
        if file_path.lower().endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                extracted_files = [os.path.join(temp_dir, f) for f in zip_ref.namelist()]

        status_label.configure(text=f"Rozpakowano archiwum: {os.path.basename(file_path)}")
        return extracted_files

    except Exception as e:
        status_label.configure(text=f"Błąd rozpakowywania: {e}")
        return []

def update_status(label, text, app):
    app.after(0, lambda: label.configure(text=text))

# Global variables
selected_files = {}  # Dictionary to store file paths and their check status
file_buttons = []  # List to store radio buttons

def update_status(label, text, app):
    app.after(0, lambda: label.configure(text=text))
    app.update_idletasks()


def process_single_text(text, filename, status_label, progress_bar, app):
    """Apply Stanza sentence splitting and token tagging to a text."""
    if not text.strip():
        update_status(status_label, f"Nie znaleziono tekstu w pliku {filename}!", app)
        return None

    update_status(status_label, f"Dzielę tekst na zdania: {filename}", app)
    print(f"Dzielę tekst na zdania: {filename}")

    doc = nlp_stanza(text)
    total_sentences = len(doc.sentences)

    if total_sentences == 0:
        update_status(status_label, f"Nie znaleziono zdań w pliku {filename}!", app)
        print(f"Nie znaleziono zdań w pliku {filename}!")
        return None

    processed_tokens = []
    update_status(status_label, f"Rozpoczynam tagowanie tekstu: {filename}", app)
    progress_bar.set(0)

    char_pos = 0
    for sent_id, sentence in enumerate(doc.sentences, start=1):
        word_to_ner = {word.id: token.ner for token in sentence.tokens for word in token.words}
        for word in sentence.words:
            start_idx = text.find(word.text, char_pos)
            if start_idx == -1:
                continue
            end_idx = start_idx + len(word.text) - 1
            char_pos = end_idx + 1
            ner_tag = word_to_ner.get(word.id, "0")
            processed_tokens.append({
                "token": word.text,
                "lemma": word.lemma,
                "sentenceID": sent_id,
                "wordID": word.id,
                "headID": word.head,
                "head": sentence.words[word.head - 1].text if word.head > 0 else "root",
                "deprel": word.deprel,
                "postag": word.xpos,
                "start": start_idx,
                "end": end_idx,
                "ner": ner_tag,
                "upos": word.upos
            })

    return processed_tokens


def process_file(file_path, status_label, progress_bar, app):
    """Process a single file or archive and return enriched data list."""
    ext = os.path.splitext(file_path)[1].lower()
    all_data = []

    if ext in [".txt", ".docx", ".pdf", ".xlsx"]:
        # Read file content
        if ext in [".txt", ".docx", ".pdf"]:
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif ext == ".docx":
                doc = Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)
            else:  # PDF
                text = process_pdf(file_path, status_label, app)

            processed_tokens = process_single_text(text, os.path.basename(file_path),
                                                   status_label, progress_bar, app)
            if processed_tokens:
                item = {"filename": os.path.basename(file_path),
                        "Treść": text,
                        "tokens": [t["token"] for t in processed_tokens],
                        "lemmas": [t["lemma"] for t in processed_tokens],
                        "postags": [t["postag"].split(":")[0] if t["postag"] else "" for t in processed_tokens],
                        "full_postags": [t["postag"] for t in processed_tokens],
                        "deprels": [t["deprel"] for t in processed_tokens],
                        "word_ids": [t["wordID"] for t in processed_tokens],
                        "sentence_ids": [t["sentenceID"] for t in processed_tokens],
                        "head_ids": [t["headID"] for t in processed_tokens],
                        "start_ids": [t["start"] for t in processed_tokens],
                        "end_ids": [t["end"] for t in processed_tokens],
                        "ners": [t["ner"] for t in processed_tokens],
                        "upostags": [t["upos"] for t in processed_tokens]}
                all_data.append(item)

        elif ext == ".xlsx":
            items = process_xlsx(file_path)
            for it in items:
                it["filename"] = os.path.basename(file_path)
            all_data.extend(items)

    elif ext in [".zip"]:
        extracted_files = unpack_archive(file_path, status_label)
        update_status(status_label, f"Rozpakowano archiwum: {os.path.basename(file_path)}", app)

        for inner_file in extracted_files:
            all_data.extend(process_file(inner_file, status_label, progress_bar, app))

    else:
        print(f"Nieobsługiwany format pliku: {file_path}")

    return all_data

def process_xlsx(file_path):
    """
    Reads data from an XLSX file, extracting 'Tytuł', 'Treść', 'Data publikacji', and 'Autor'.
    """
    try:
        df = pd.read_excel(file_path)


        columns = df.columns

        data = []
        for _, row in df.iterrows():  # Don't skip the first row
            title = str(row["Tytuł"]).strip() if "Tytuł" in columns else ""
            content = str(row["Treść"]).strip() if "Treść" in columns else ""
            date = str(row["Data publikacji"]).strip() if "Data publikacji" in columns else ""
            author = str(row["Autor"]).strip() if "Autor" in columns else ""
            rok = str(row["rok"]).strip() if "rok" in columns else ""
            miesiąc = str(row["miesiąc"]).strip() if "miesiąc" in columns else ""
            dzień = str(row["dzień"]).strip() if "dzień" in columns else ""

            if "Lead" in df.columns:
                lead = str(row.get("Lead", "")).strip()
                content = f"{lead}\n\n{content}".strip() if lead else content

            if "Tytuł" in df.columns:
                title = str(row.get("Tytuł", "")).strip()
                content = f"{title}\n\n{content}".strip() if title else content # Prepend title if it exists

            data.append({
                "Tytuł": title,
                "Treść": content,
                "Data publikacji": date,
                "Autor": author,
                "rok":rok,
                "miesiąc":miesiąc,
                "dzień":dzień
            })
        return data
    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku XLSX: {e}")
        return []

def fix_hyphenation(text):
    # Remove hyphen at the end of a line (along with the newline and any surrounding whitespace)
    return re.sub(r'-\s*\n\s*', '', text)
    # Remove any newline characters
    return text.replace('\n', ' ')


def process_pdf(file_path, status_label, app):
    import fitz
    import easyocr


    """
    Extract text from a PDF. If it's an image-based PDF, OCR is applied using easyocr.
    The GUI status label is updated during OCR processing.
    """
    pdf_doc = fitz.open(file_path)  # Open the PDF
    text = ""

    # Initialize EasyOCR reader with Polish language
    reader = easyocr.Reader(['pl'])  # List of language codes, 'pl' for Polish

    # Create a temporary directory to store images
    temp_dir = "./temp_images"
    os.makedirs(temp_dir, exist_ok=True)  # Create the temp directory if it doesn't exist

    # Extract text using PyMuPDF (fitz)
    for page in pdf_doc:
        page_number = page.number + 1  # Page numbers start from 0 in PyMuPDF
        text_page = page.get_text("text")

        if text_page:
            text += text_page
        else:
            # Update GUI status
            status_label.configure(text=f"Rozpoznaję tekst za pomocą OCR, strona: {page_number}/{len(pdf_doc)}...")
            app.update_idletasks()  # Force UI update

            # Convert page to an image
            img = page.get_pixmap()
            img_path = os.path.join(temp_dir, f"page_{page_number}.png")
            img.save(img_path)

            # Use easyocr to extract text
            ocr_text = reader.readtext(img_path, detail=0)
            text += " ".join(ocr_text)

    # Fix hyphenation issues in the extracted text
    text = fix_hyphenation(text)

    return text


# Compute token counts per year and month for the whole corpus
def compute_token_counts(data_list):
    token_counts = {}
    for item in data_list:
        # Count tokens in the 'tags' field by splitting on spaces
        tokens = item.get("tokens", [])
        count = len(tokens)

        # Determine year and month
        year = item.get("rok") or None
        month = item.get("miesiąc") or None
        # Fallback to Data publikacji if rok/miesiąc missing
        if not year or not month:
            date_str = item.get("Data publikacji", "")
            parts = date_str.split("-")
            if len(parts) >= 2:
                year, month = parts[0], parts[1]

        if not year or not month:
            continue  # Skip if date info missing

        token_counts.setdefault(year, {}).setdefault(month, 0)
        token_counts[year][month] += count
    return token_counts


def select_files(frame, progress_bar, status_label, app):
    global selected_files, file_buttons
    file_paths = fd.askopenfilenames(
        title="Wybierz pliki",
        initialdir=BASE_DIR,
        filetypes=[("All files", "*.*"),
                   ("Text files", "*.txt"),
                   ("Word Documents", "*.docx"),
                   ("PDF files", "*.pdf"),
                   ("Archives", "*.zip")],
        parent=app

    )

    if not file_paths:
        status_label.configure(text="Nie wybrano żadnego pliku.")
        return

        # Add new files to the existing selected_files and create new checkboxes for them
    for file_path in file_paths:
        if file_path not in selected_files:
            var = ctk.IntVar(value=1)  # Default to hecked
            selected_files[file_path] = var
            btn = ctk.CTkCheckBox(frame, text=os.path.basename(file_path), variable=var)
            btn.pack(anchor="w", padx=20, pady=10)
            file_buttons.append(btn)

    progress_bar.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
    progress_bar.grid_remove()
    status_label.configure(text="Zaznacz pliki, które mają zostać przetworzone.")

def main():


    # Set up customtkinter appearance and theme
    app = ctk.CTk()
    app.attributes("-topmost", True)
    def center_window(app, width=800, height=400):
        screen_width = app.winfo_screenwidth()
        screen_height = app.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        app.geometry(f"{width}x{height}+{x}+{y}")
    center_window(app, 800, 400)
    app.title("Kreator korpusów")
    #icon_path = os.path.join(BASE_DIR, "favicon.ico")
    #app.iconbitmap(icon_path)

    main_frame = ctk.CTkFrame(app)
    main_frame.pack(pady=5, fill="both", side="left")
    # Configure grid layout (2 columns for better alignment)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)

    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_rowconfigure(1, weight=1)
    main_frame.grid_rowconfigure(2, weight=1)
    main_frame.grid_rowconfigure(3, weight=1)

    # Create buttons
    select_button = ctk.CTkButton(
        main_frame,
        text="Wybierz pliki",
        command=lambda: select_files(checkbox_frame, progress_bar, status_label, app),
        font=("Verdana", 12, 'bold'),
        corner_radius=8,
        height=35,

    )
    select_button.grid(row=0, column=0, columnspan=2, pady=10)

    process_button = ctk.CTkButton(main_frame, text="Przetwórz pliki", command=lambda: start_processing(),
                                   font=("Verdana", 12, 'bold'),
                                   corner_radius=8,
                                   height=35,
                                   )
    process_button.grid(row=1, column=0, columnspan=2, pady=10)

    # **Progress Bar (Row Reserved, Initially Hidden)**
    progress_bar = ctk.CTkProgressBar(main_frame)
    progress_bar.set(0)
    progress_bar.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
    progress_bar.grid_remove()  # Hides the progress bar but keeps the row reserved

    # **Status Label**
    status_label = ctk.CTkLabel(main_frame, text="Wybierz pliki do przetworzenia", font=("Verdana", 12, 'bold'))
    status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    # **Frame for Checkboxes**
    checkbox_frame = ctk.CTkScrollableFrame(app)
    checkbox_frame.pack(pady=5, fill="both", expand=True, side="right")

    switch_var = ctk.StringVar(value="on")

    def process_file(file_path, status_label, progress_bar, app):
        """Process a single file or archive and return enriched data list."""
        ext = os.path.splitext(file_path)[1].lower()
        all_data = []

        if ext in [".txt", ".docx", ".pdf", ".xlsx"]:
            # Read file content
            if ext in [".txt", ".docx", ".pdf"]:
                if ext == ".txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                elif ext == ".docx":
                    doc = Document(file_path)
                    text = "\n".join(p.text for p in doc.paragraphs)
                else:  # PDF
                    text = process_pdf(file_path, status_label, app)

                # Tokenize and tag once
                processed_tokens = process_single_text(
                    text, os.path.basename(file_path), status_label, progress_bar, app
                )
                if processed_tokens:
                    item = {
                        "filename": os.path.basename(file_path),
                        "Treść": text,
                        "tokens": [t["token"] for t in processed_tokens],
                        "lemmas": [t["lemma"] for t in processed_tokens],
                        "postags": [t["postag"].split(":")[0] if t["postag"] else "" for t in processed_tokens],
                        "full_postags": [t["postag"] for t in processed_tokens],
                        "deprels": [t["deprel"] for t in processed_tokens],
                        "word_ids": [t["wordID"] for t in processed_tokens],
                        "sentence_ids": [t["sentenceID"] for t in processed_tokens],
                        "head_ids": [t["headID"] for t in processed_tokens],
                        "start_ids": [t["start"] for t in processed_tokens],
                        "end_ids": [t["end"] for t in processed_tokens],
                        "ners": [t["ner"] for t in processed_tokens],
                        "upostags": [t["upos"] for t in processed_tokens],

                        # Store full token info for later reuse
                        "tokens_detail": processed_tokens
                    }
                    all_data.append(item)

            elif ext == ".xlsx":
                items = process_xlsx(file_path)
                for it in items:
                    it["filename"] = os.path.basename(file_path)
                all_data.extend(items)

        elif ext in [".zip"]:
            extracted_files = unpack_archive(file_path, status_label)
            update_status(status_label, f"Rozpakowano archiwum: {os.path.basename(file_path)}", app)

            for inner_file in extracted_files:
                all_data.extend(process_file(inner_file, status_label, progress_bar, app))

        else:
            print(f"Nieobsługiwany format pliku: {file_path}")

        return all_data

    def process_files(status_label, progress_bar, app):
        global selected_files, nlp_stanza

        selected_paths = [path for path, var in selected_files.items() if var.get() == 1]

        if not selected_paths:
            status_label.configure(text="Nie wybrano pliku do przetworzenia.")
            return

        metadata_dict = {}
        use_metadata = messagebox.askquestion("Metadane", "Czy chcesz dodać metadane (plik metadane.xlsx)?",
                                              icon="question", parent=app)

        if use_metadata == 'yes':
            status_label.configure(text="Wybierz plik metadanych (metadane.xlsx)...")
            app.update_idletasks()

            metadata_path = fd.askopenfilename(
                title="Wybierz plik metadanych (metadane.xlsx)",
                initialdir=BASE_DIR,
                filetypes=[("Excel files", "*.xlsx")],
                parent=app
            )

            if metadata_path:
                try:
                    df_meta = pd.read_excel(metadata_path)
                    if "Nazwa pliku" in df_meta.columns:
                        for _, row in df_meta.iterrows():
                            filename = str(row["Nazwa pliku"]).strip()
                            metadata_dict[filename] = {
                                col: (
                                    str(row[col]).strip() if pd.notna(row[col]) and str(row[col]).strip()
                                    else "0000-00-00" if col == "Data publikacji"
                                    else "#"
                                )
                                for col in df_meta.columns if col != "Nazwa pliku"
                            }
                except Exception as e:
                    print(f"Błąd wczytywania metadanych: {e}")
            else:
                status_label.configure(text="Nie wybrano pliku metadanych. Przetwarzanie anulowane.")
                return
        else:
            status_label.configure(text="Tworzę korpus bez metadanych")

        initialize_stanza(status_label, app)

        progress_bar.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        progress_bar.set(0)
        app.update_idletasks()

        data = []
        for file_path in selected_paths:
            filename = os.path.basename(file_path)

            # Skip metadane.xlsx itself
            if filename.lower() == "metadane.xlsx":
                continue

            status_label.configure(text=f"Przetwarzam plik {file_path}")
            print(f"Przetwarzam plik {file_path}")

            file_data_list = process_file(file_path, status_label, progress_bar, app)

            for item in file_data_list:
                text = item.get("Treść", "")
                filename = item.get("filename", os.path.basename(file_path))

                if not text.strip():
                    continue

                # REUSE tokens from process_file (avoid re-running Stanza)
                processed_tokens = item.get("tokens_detail", [])

                # apply metadata
                if filename in metadata_dict:
                    meta = metadata_dict[filename]
                    entry = {
                        "Tytuł": meta.get("Tytuł", filename),
                        "Treść": text,
                    }
                    for key, value in meta.items():
                        if key != "Tytuł":
                            entry[key] = value
                else:
                    entry = {
                        "Tytuł": filename,
                        "Treść": text,
                        "Data publikacji": "0000-00-00",
                        "Autor": "#"
                    }

                # enrich with parsed tokens
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

                data.append(entry)

        token_counts = compute_token_counts(data)
        combined_output = data + [{"token_counts": token_counts}]

        # Save Parquet (same as before)
        try:
            output_parquet_file = fd.asksaveasfilename(
                defaultextension=".parquet",
                filetypes=[("Parquet files", "*.parquet")],
                initialdir=BASE_DIR,
                title="Zapisz plik Parquet z korpusem jako...",
                parent=app
            )

            if output_parquet_file:
                token_counts_json = json.dumps(token_counts, ensure_ascii=False)
                df = pd.DataFrame([item for item in combined_output if "tokens" in item])
                df["token_counts"] = token_counts_json

                os.makedirs(os.path.dirname(output_parquet_file), exist_ok=True)
                df.to_parquet(output_parquet_file, engine="pyarrow", index=False, compression="snappy")
                status_label.configure(text=f"Parquet zapisany:\n {output_parquet_file}")
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")
            print("Parquet save error:", e)

        time.sleep(0.01)

    def toggle_all():
        """Toggle check/uncheck all checkboxes."""
        if switch_var.get()=="on":
            for var in selected_files.values():
                var.set(1)
        elif switch_var.get()=="off":
            for var in selected_files.values():
                var.set(0)

    def start_processing():
        # Disable button at the start
        process_button.configure(state="disabled")

        # Start the thread
        threading.Thread(target=process_files_thread, daemon=True).start()

    def process_files_thread():
        try:
            process_files(status_label, progress_bar, app)
        finally:
            # Re-enable button after processing is done
            process_button.configure(state="normal")

    # Create a "Check/Uncheck All" button and place it in the UI
    toggle_button = ctk.CTkSwitch(checkbox_frame, text="Odznacz/Zaznacz wszystkie pliki", font=("Verdana", 12, 'bold'), state=True, command=toggle_all, variable=switch_var, onvalue="on", offvalue="off")
    toggle_button.pack(padx=20, pady=20)  # Use pack for the toggle button
    app.mainloop()

# Make sure nothing runs on import
if __name__ == "__main__":
    main()