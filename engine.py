import os
import sys
from pathlib import Path
import subprocess
import logging
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
        import semantic_trainer

        exit_code = semantic_trainer.main()
        sys.exit(exit_code)
    except Exception as e:

        logging.info(f"Krytyczny błąd w procesie podrzędnym trainera: {e}")
        sys.exit(1)

if "--run-semantic-report" in sys.argv:
    try:
        sys.argv.remove("--run-semantic-report")
        import semantic_reports_analytical_v7_1 as semantic_report
        exit_code = semantic_report.main()
        sys.exit(exit_code)
    except Exception as e:
        logging.info(f"Błąd procesu raportu semantycznego: {e}")
        sys.exit(1)

if "--run-fiszki" in sys.argv:
    try:
        val = sys.argv[sys.argv.index("--run-fiszki") + 1]
        import fiszki_tkinter

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
import time
from word_profile import compute_word_profile, flatten_word_profile
from sense_inducer import SenseInducer

def notify_status(msg):
    # Sprawdzamy, czy launcher jest uruchomiony i ma funkcję update_status
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'update_status'):
        sys.modules['__main__'].update_status(msg)

notify_status("Wczytywanie bibliotek systemowych...")


# ==========================================
# MODUŁ SEMANTYCZNY
# ==========================================
class SemanticEngine:
    """Klasa zarządzająca logiką, ładowaniem i pamięcią sieci semantycznej."""

    def __init__(self):
        self.df_neighbors = None
        self.index = None
        self.knn_set = None

        # Nowe zmienne dla WSD
        self.vectors = None
        self.senses_cache = {}

        # NOWE: Cache dla kontekstu grafowego, żeby nie zamrozić UI
        self.graph_sense_cache = {}

        self.hubness_index = {}  # Dodane: cache na preobliczoną hubowość

    def network_exists(self, current_corpus_path):
        """Sprawdza, czy na dysku istnieją pliki wygenerowanej sieci semantycznej dla danego korpusu."""
        if not current_corpus_path:
            return False
        base_path = str(Path(current_corpus_path).with_suffix(""))
        return any(os.path.exists(p) for p in [
            f"{base_path}.wektor",
            f"{base_path}.semantic.fasttext.neighbors.parquet",
            f"{base_path}.semantic.neighbors.parquet"
        ])

    def open_training_setup(self, parent_app, current_corpus_name, current_corpus_path, theme, on_success_callback):
        """Otwiera okno UI z parametrami przed uruchomieniem budowania sieci semantycznej."""
        if not current_corpus_path:
            messagebox.showwarning("Brak danych", "Najpierw wybierz korpus z menu po lewej stronie!")
            return

        setup_win = ctk.CTkToplevel(parent_app)
        setup_win.title("Konfiguracja sieci semantycznej")
        setup_win.geometry("450x450")
        setup_win.configure(fg_color=theme["app_bg"])
        setup_win.attributes("-topmost", True)

        ctk.CTkLabel(setup_win, text=f"Ustawienia sieci: {current_corpus_name}",
                     font=("Verdana", 14, "bold")).pack(pady=(20, 15))

        frame = ctk.CTkFrame(setup_win, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=30)

        def add_param(label_text, default_val, is_dropdown=False, options=None):
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=label_text, width=180, anchor="w").pack(side="left")
            if is_dropdown:
                var = ctk.StringVar(value=default_val)
                w = ctk.CTkOptionMenu(row, variable=var, values=options)
            else:
                var = ctk.StringVar(value=str(default_val))
                w = ctk.CTkEntry(row, textvariable=var)
            w.pack(side="right", fill="x", expand=True, padx=(10, 0))
            return var

        # Parametry
        algo_var = add_param("Algorytm (--algo):", "fasttext", True, ["fasttext", "word2vec"])
        min_count_var = add_param("Min. wystąpień (--min-count):", "10")
        epochs_var = add_param("Epoki (--epochs):", "20")
        window_var = add_param("Rozmiar okna (--window):", "15")
        vocab_var = add_param("Słownik (--neighbors...):", "10000")
        precomp_var = add_param("Zapisz top N (--precompute...):", "200")

        def on_start():
            params = {
                "algo": algo_var.get(),
                "min_count": min_count_var.get(),
                "epochs": epochs_var.get(),
                "window": window_var.get(),
                "vocab": vocab_var.get(),
                "precomp": precomp_var.get()
            }
            setup_win.destroy()
            self._run_training_process(parent_app, current_corpus_name, current_corpus_path, theme, on_success_callback,
                                       params)

        ctk.CTkButton(setup_win, text="Rozpocznij budowanie", font=("Verdana", 12, "bold"),
                      height=40, command=on_start).pack(pady=20, padx=30, fill="x")

    def _run_training_process(self, parent_app, current_corpus_name, current_corpus_path, theme, on_success_callback,
                              params):
        """Właściwy proces budowania sieci (uruchamiany po zatwierdzeniu konfiguracji)."""
        win = ctk.CTkToplevel(parent_app)
        win.title("Budowanie sieci semantycznej")
        win.geometry("600x450")
        win.configure(fg_color=theme["app_bg"])
        win.attributes("-topmost", True)

        ctk.CTkLabel(win, text=f"Budowanie sieci dla: {current_corpus_name}",
                     font=("Verdana", 14, "bold")).pack(pady=(15, 5))

        progress = ctk.CTkProgressBar(win, mode="indeterminate", height=10)
        progress.pack(fill="x", padx=20, pady=10)
        progress.start()

        log_box = ctk.CTkTextbox(win, wrap="word", font=("Consolas", 11),
                                 fg_color="#1E1E1E", text_color="#00FF00")
        log_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        corpus_path_safe = str(Path(current_corpus_path).resolve())

        def run_training_in_background():
            if getattr(sys, "frozen", False):
                # Wersja skompilowana (.exe) - wywołujemy bezpośrednio plik binarny z flagą
                cmd = [sys.executable, "--run-semantic-trainer"]
            else:
                # Wersja skryptowa (.py) - wywołujemy interpreter, potem ścieżkę skryptu, potem flagę
                cmd = [sys.executable, os.path.abspath(__file__), "--run-semantic-trainer"]
            cmd.extend([
                "--parquet", corpus_path_safe,
                "--algo", params["algo"],
                "--min-count", str(params["min_count"]),  # Zawsze bezpieczniej wymusić str()
                "--epochs", str(params["epochs"]),
                "--window", str(params["window"]),
                "--neighbors-for-top-vocab", str(params["vocab"]),
                "--precompute-neighbors", str(params["precomp"]),
                "--no-lower",
                "--no-full-model",
                "--allowed-upos", "NOUN", "PROPN", "ADJ", "VERB"
            ])
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace', bufsize=1, creationflags=creationflags
                )  # ^^^ DODANO errors='replace' ^^^

                for line in process.stdout:
                    win.after(0, lambda l=line: (log_box.insert("end", l), log_box.see("end")))

                process.wait()

                if process.returncode == 0:
                    win.after(0, lambda: log_box.insert("end", "\nBudowanie zakończone i spakowane pomyślnie!\n"))
                    win.after(0, on_success_callback)
                else:
                    win.after(0, lambda: log_box.insert("end", f"\nWystąpił błąd (kod: {process.returncode})\n"))
                    logging.error(f"Błąd procesu semantic_trainer.py. Kod: {process.returncode}")
            except Exception as e:
                win.after(0, lambda: log_box.insert("end", f"\nKrytyczny błąd uruchamiania: {e}\n"))
                logging.error(f"Krytyczny błąd podczas uruchamiania treningu semantycznego: {e}", exc_info=True)
            finally:
                win.after(0, progress.stop)

        threading.Thread(target=run_training_in_background, daemon=True).start()

    def build_semantic_report(
            self,
            parent_app,
            current_corpus_name,
            current_corpus_path,
            lemma,
            theme,
            open_report_callback,
            params=None,
    ):
        if not current_corpus_path:
            messagebox.showwarning("Brak danych", "Najpierw wybierz korpus z menu po lewej stronie!")
            return

        if not lemma or not str(lemma).strip():
            messagebox.showwarning("Brak lemy", "Najpierw wybierz lub wpisz słowo centralne do raportu.")
            return

        params = params or {}
        report_top_k = str(params.get("report_top_k", params.get("top_k", 0)))
        min_similarity = str(params.get("min_similarity", 0.30))

        safe_lemma = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(lemma).strip())
        corpus_base = Path(current_corpus_path).with_suffix("")
        report_dir = corpus_base.parent / f"{corpus_base.name}.semantic_reports" / safe_lemma

        win = ctk.CTkToplevel(parent_app)
        win.title("Generowanie raportu semantycznego")
        win.geometry("700x500")
        win.configure(fg_color=theme["app_bg"])
        win.attributes("-topmost", True)

        ctk.CTkLabel(
            win,
            text=f"Raport semantyczny: {lemma}",
            font=("Verdana", 14, "bold")
        ).pack(pady=(15, 5))

        progress = ctk.CTkProgressBar(win, mode="indeterminate", height=10)
        progress.pack(fill="x", padx=20, pady=10)
        progress.start()

        log_box = ctk.CTkTextbox(
            win,
            wrap="word",
            font=("Consolas", 11),
            fg_color="#1E1E1E",
            text_color="#00FF00"
        )
        log_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        corpus_path_safe = str(Path(current_corpus_path).resolve())
        report_dir_safe = str(report_dir.resolve())

        def run_report_in_background():
            if getattr(sys, "frozen", False):
                cmd = [sys.executable, "--run-semantic-report"]
            else:
                cmd = [sys.executable, os.path.abspath(__file__), "--run-semantic-report"]

            report_top_k = str(params.get("report_top_k", params.get("top_k", 0)))
            min_similarity = str(params.get("min_similarity", 0.30))

            cmd.extend([
                "--artifacts", corpus_path_safe,
                "--lemma", str(lemma).strip(),
                "--output-dir", report_dir_safe,

                "--top-k-neighbors", report_top_k,
                "--min-similarity", min_similarity,

                "--top-core", "12",
                "--top-distinctive", "12",
                "--top-interpretive", "12",

                "--table-size", "40",
                "--tail-size", "20",
                "--orphan-size", "50",

                "--globality-threshold", "0.40",
                "--frame-edge-threshold", "0.42",
                "--bridge-similarity-threshold", "0.45",
                "--frame-assignment-min-similarity", "0.10",
                "--core-quantile", "0.60",
                "--max-plot-words", "120",
            ])


            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags
                )

                for line in process.stdout:
                    win.after(0, lambda l=line: (log_box.insert("end", l), log_box.see("end")))

                process.wait()

                if process.returncode == 0:
                    html_path = str((report_dir / "report.html").resolve())
                    win.after(0, lambda: log_box.insert("end", "\nRaport wygenerowany pomyślnie.\n"))
                    win.after(0, lambda: open_report_callback(html_path))
                else:
                    win.after(0,
                              lambda: log_box.insert("end", f"\nWystąpił błąd raportu (kod: {process.returncode})\n"))
                    logging.error(f"Błąd procesu semantic_reports_analytical_v7_1.py. Kod: {process.returncode}")

            except Exception as e:
                win.after(0, lambda: log_box.insert("end", f"\nKrytyczny błąd uruchamiania: {e}\n"))
                logging.error(f"Krytyczny błąd podczas generowania raportu semantycznego: {e}", exc_info=True)
            finally:
                win.after(0, progress.stop)

        threading.Thread(target=run_report_in_background, daemon=True).start()


    def load_neighbors(self, current_corpus_path):
        import numpy as np
        if not current_corpus_path:
            self.df_neighbors = None
            self.index = None
            self.knn_set = None
            self.vectors = None
            self.senses_cache = {}
            self.graph_sense_cache = {}
            self.hubness_index = {}  # <--- POPRAWKA 5: Reset przy braku korpusu
            return

        import zipfile
        corpus_path_obj = Path(current_corpus_path)
        base_path = str(corpus_path_obj.with_suffix(""))
        wektor_path = f"{base_path}.wektor"

        loaded_df = None
        loaded_vectors_df = None

        if os.path.exists(wektor_path):
            try:
                with zipfile.ZipFile(wektor_path, 'r') as zf:
                    parquet_files = [f for f in zf.namelist() if f.endswith(".neighbors.parquet")]
                    vector_files = [f for f in zf.namelist() if f.endswith(".vectors.parquet")]

                    if parquet_files:
                        with zf.open(parquet_files[0]) as f:
                            loaded_df = pd.read_parquet(f)
                    if vector_files:
                        with zf.open(vector_files[0]) as f:
                            loaded_vectors_df = pd.read_parquet(f)

                    notify_status(f"Dane sieci semantycznej załadowane: {os.path.basename(wektor_path)}")
            except Exception as e:
                logging.error(f"Błąd odczytu archiwum .wektor sieci semantycznej: {e}", exc_info=True)
        else:
            # Stare formaty offline
            for suffix in [".semantic.fasttext.neighbors.parquet", ".semantic.neighbors.parquet"]:
                p = f"{base_path}{suffix}"
                if os.path.exists(p):
                    try:
                        loaded_df = pd.read_parquet(p)
                        notify_status(f"Dane sieci semantycznej załadowane (stary format): {os.path.basename(p)}")
                        break
                    except Exception as e:
                        logging.error(f"Błąd odczytu pliku parquet sieci semantycznej {p}: {e}", exc_info=True)

            v_path = f"{base_path}.semantic.vectors.parquet"
            if os.path.exists(v_path):
                try:
                    loaded_vectors_df = pd.read_parquet(v_path)
                except Exception as e:
                    logging.error(f"Błąd odczytu wektorów: {e}")

        self.df_neighbors = loaded_df

        # Inicjalizacja słownika wektorów z DataFrame
        if loaded_vectors_df is not None:
            self.vectors = {row['lemma']: np.array(row['vector']) for _, row in loaded_vectors_df.iterrows()}
        else:
            self.vectors = None
            logging.warning("Sieć semantyczna załadowana, ale brakuje wektorów (WSD nie będzie działać).")

        self.senses_cache = {}
        self.graph_sense_cache = {}

        if loaded_df is not None:
            self.index = {}
            self.knn_set = {}
            has_freq = 'neighbor_freq' in loaded_df.columns
            loaded_df = loaded_df.sort_values(by=['lemma', 'similarity'], ascending=[True, False])
            MUTUAL_M = 50

            for lemma, group in loaded_df.groupby('lemma'):
                neighbors = group['neighbor'].tolist()
                scores = group['similarity'].tolist()
                freqs = group['neighbor_freq'].tolist() if has_freq else [0] * len(neighbors)

                self.index[lemma] = list(zip(neighbors, scores, freqs))
                self.knn_set[lemma] = set(neighbors[:MUTUAL_M])
        else:
            self.index = None
            self.knn_set = None



        # Resetujemy przy każdym ładowaniu nowego modelu/sąsiadów
        self.hubness_index = {}

        if not self.index:
            return

        # ========================================================
        # NOWA LOGIKA HUBNOŚCI: Globalne "In-Degree"
        # ========================================================
        # Prawdziwy hub to słowo, które bardzo często pojawia się w
        # listach sąsiadów INNYCH słów. Liczymy globalną frekwencję.
        hub_counts = {}
        for lemma, neighbors in self.index.items():
            for n_word, n_score, _ in neighbors:
                # Bierzemy pod uwagę tylko w miarę silne relacje (>0.40)
                if float(n_score) >= 0.40:
                    hub_counts[n_word] = hub_counts.get(n_word, 0) + 1

        counts = list(hub_counts.values())
        if not counts:
            return

        # Wyznaczamy dynamiczne statystyki populacji
        # (dzięki temu algorytm zadziała i dla małych, i dla gigantycznych korpusów)
        import numpy as np
        p50 = float(np.percentile(counts, 50))  # Mediana
        max_count = float(max(counts)) if counts else 1.0  # Absolutny król hubów

        all_words = set(self.index.keys()).union(set(hub_counts.keys()))

        for word in all_words:
            c = hub_counts.get(word, 0)

            if c <= p50:
                self.hubness_index[word] = 0.0
            else:
                # Skala logarytmiczna: hiper-huby dostają ~1.0, huby domenowe wyraźnie mniej
                numerator = math.log((c - p50) + 1)
                denominator = math.log((max_count - p50) + 1) if max_count > p50 else 1.0

                self.hubness_index[word] = min(1.0, numerator / denominator)

    def get_max_available_neighbors(self):
        """Zwraca maksymalną liczbę sąsiadów dostępną w wczytanym indeksie (ze słownika)."""
        if self.index:
            # Pobieramy pierwszy z brzegu wpis i sprawdzamy długość listy jego sąsiadów
            first_key = next(iter(self.index))
            return len(self.index[first_key])
        return 0

    def get_neighbors(self, word, top_n=25):
        """Pobiera sąsiadów z uwzględnieniem limitu top_n."""
        if self.index is None:
            return word, []

        search_word = word.strip()
        if search_word in self.index:
            return search_word, self.index[search_word][:top_n]
        elif search_word.lower() in self.index:
            return search_word.lower(), self.index[search_word.lower()][:top_n]
        elif search_word.capitalize() in self.index:
            return search_word.capitalize(), self.index[search_word.capitalize()][:top_n]
        return search_word, []

    def is_mutual_knn(self, u: str, v: str) -> bool:
        # [BEZ ZMIAN]
        if self.knn_set is None:
            return False
        return (v in self.knn_set.get(u, set())) and (u in self.knn_set.get(v, set()))

    @staticmethod
    def dynamic_bridge_threshold(freq_u: int, freq_v: int, base: float = 0.55) -> float:
        # [BEZ ZMIAN]
        import math
        fu, fv = max(0, int(freq_u or 0)), max(0, int(freq_v or 0))
        if fu == 0 and fv == 0: return base
        hub = max(fu, fv)
        boost = 0.06 * max(0.0, math.log10(hub / 2000)) if hub > 0 else 0.0
        return max(0.55, min(0.78, base + boost))

    # ==========================================
    # NOWE METODY DO OBSŁUGI SENSÓW (WSD)
    # ==========================================

    def get_or_create_senses(self, lemma):
        """Pobiera wygenerowane sensy z cache lub liczy je na żądanie."""
        if not self.vectors or not self.index:
            return []

        # Używamy nowej metody do normalizacji klucza
        actual_lemma = self._resolve_key(lemma, self.index)

        # Sprawdzamy czy znormalizowane słowo jest też w wektorach
        if not actual_lemma or actual_lemma not in self.vectors:
            return []

        # Jeśli już wcześniej policzyliśmy klastry dla tego słowa
        if actual_lemma in self.senses_cache:
            return self.senses_cache[actual_lemma]

        # Liczymy klastry i zapisujemy do cache
        from sense_inducer import SenseInducer
        debug_semantic_frames = False
        senses = SenseInducer.induce(
            actual_lemma,
            self.vectors,
            self.index,
            debug=debug_semantic_frames
        )
        self.senses_cache[actual_lemma] = senses

        return senses

    def get_cached_senses(self, lemma):
        """
        Zwraca sensy tylko wtedy, gdy są już w cache.
        NIE uruchamia indukcji sensów.
        """
        if not self.vectors or not self.index:
            return []

        actual_lemma = self._resolve_key(lemma, self.index)

        if not actual_lemma or actual_lemma not in self.vectors:
            return []

        return self.senses_cache.get(actual_lemma, [])



    def disambiguate_instance(self, sentence_tokens, target_idx, lemma):
        """Zwraca ID sensu dla podanego słowa w zdaniu. (Oczekuje tokenów w formie słowników np. {'lemma': '...'})"""
        senses = self.get_or_create_senses(lemma)
        if not senses:
            return None

        # Zbuduj wektor kontekstu omijając badane słowo
        ctx = []
        for i, tok in enumerate(sentence_tokens):
            if i == target_idx:
                continue
            tok_lemma = tok.get("lemma", "").lower()
            if tok_lemma in self.vectors:
                ctx.append(self.vectors[tok_lemma])

        if not ctx:
            return None

        ctx_vec = np.mean(ctx, axis=0)

        best_sid = None
        best_score = -1

        for s in senses:
            score = np.dot(ctx_vec, s["vector"]) / (np.linalg.norm(ctx_vec) * np.linalg.norm(s["vector"]) + 1e-9)
            if score > best_score:
                best_sid = s["sense_id"]
                best_score = score

        return best_sid

    # ==========================================
    # GRAPH-CONDITIONED EXPANSION (GRAPH-WSD)
    # ==========================================

    def _resolve_key(self, lemma, target_dict):
        if not target_dict or not lemma:
            return None
        search_word = lemma.strip()
        for candidate in [search_word, search_word.lower(), search_word.capitalize()]:
            if candidate in target_dict:
                return candidate
        return None

    def get_representation_vector(self, lemma, sense_id=None):
        actual_lemma = self._resolve_key(lemma, self.vectors)
        if not actual_lemma:
            return None

        if sense_id is None:
            return self.vectors[actual_lemma]

        senses = self.get_or_create_senses(actual_lemma)
        for s in senses:
            if s["sense_id"] == sense_id:
                return s["vector"]
        return self.vectors[actual_lemma]

    def build_graph_context_vector(self, root_lemma, parent_lemma, root_sense_id=None, parent_sense_id=None,
                                   local_neighbor_lemmas=None, alpha=0.45, beta=0.40, gamma=0.10, delta=0.05):
        vecs = []
        v_root = self.get_representation_vector(root_lemma, root_sense_id)
        v_parent = self.get_representation_vector(parent_lemma, parent_sense_id)
        v_parent_base = self.get_representation_vector(parent_lemma, None)

        if v_root is not None: vecs.append((alpha, v_root))
        if v_parent is not None: vecs.append((beta, v_parent))
        if v_parent_base is not None: vecs.append((delta, v_parent_base))

        if gamma > 0 and local_neighbor_lemmas and self.vectors:
            local_vecs = []
            for w in local_neighbor_lemmas:
                norm_w = self._resolve_key(w, self.vectors)
                if norm_w: local_vecs.append(self.vectors[norm_w])

            if local_vecs:
                v_local = np.mean(local_vecs, axis=0)
                vecs.append((gamma, v_local))

        if not vecs: return None
        ctx = sum(weight * vec for weight, vec in vecs)
        norm = np.linalg.norm(ctx) + 1e-9
        return ctx / norm

    def _cos(self, u, v):
        if u is None or v is None: return -1.0
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    def choose_graph_sense(
            self,
            child_lemma,
            root_lemma,
            parent_lemma,
            root_sense_id=None,
            parent_sense_id=None,
            local_neighbor_lemmas=None,
            allow_induce=True
    ):
        actual_child = self._resolve_key(child_lemma, self.vectors)
        actual_root = self._resolve_key(root_lemma, self.vectors) if root_lemma else None
        actual_parent = self._resolve_key(parent_lemma, self.vectors) if parent_lemma else None

        if not actual_child:
            return None, None, -1.0

        # allow_induce w kluczu cache, żeby nie mieszać:
        # - fallbacku bez indukcji
        # - pełnego wyniku po indukcji
        cache_key = (
            actual_child,
            actual_root,
            actual_parent,
            root_sense_id,
            parent_sense_id,
            bool(allow_induce)
        )

        if cache_key in self.graph_sense_cache:
            return self.graph_sense_cache[cache_key]

        ctx_vec = self.build_graph_context_vector(
            root_lemma,
            parent_lemma,
            root_sense_id,
            parent_sense_id,
            local_neighbor_lemmas
        )
        v_child_base = self.vectors.get(actual_child)

        if ctx_vec is None:
            res = (None, v_child_base, -1.0)
            self.graph_sense_cache[cache_key] = res
            return res

        # KLUCZOWA ZMIANA:
        # allow_induce=False -> tylko cache, bez odpalania SenseInducer
        if allow_induce:
            senses = self.get_or_create_senses(actual_child)
        else:
            senses = self.get_cached_senses(actual_child)

        if not senses:
            score = self._cos(ctx_vec, v_child_base) if v_child_base is not None else -1.0
            res = (None, v_child_base, score)
            self.graph_sense_cache[cache_key] = res
            return res

        best_sid, best_vec, best_score = None, None, -float("inf")
        for s in senses:
            sc = self._cos(ctx_vec, s["vector"])
            if sc > best_score:
                best_sid, best_vec, best_score = s["sense_id"], s["vector"], sc

        res = (best_sid, best_vec, best_score)
        self.graph_sense_cache[cache_key] = res
        return res


    def get_or_create_frames(self, lemma):
        return self.get_or_create_senses(lemma)

    def choose_graph_frame(
            self,
            child_lemma,
            root_lemma,
            parent_lemma,
            root_sense_id=None,
            parent_sense_id=None,
            local_neighbor_lemmas=None
    ):
        return self.choose_graph_sense(
            child_lemma,
            root_lemma,
            parent_lemma,
            root_sense_id=root_sense_id,
            parent_sense_id=parent_sense_id,
            local_neighbor_lemmas=local_neighbor_lemmas
        )

    def get_halo_candidates(self, center_lemma, top_n=150, min_sim=0.35):
        """Pobiera kandydatów do tła semantycznego (Halo) bez naruszania struktury grafu Core."""
        if not self.index:
            return []

        matched_center = self._resolve_key(center_lemma, self.index)
        if not matched_center:
            return []

        raw = self.index.get(matched_center, [])
        candidates = []
        for u, base_sim, _ in raw[:top_n]:
            sim = float(base_sim)
            if sim >= min_sim:
                candidates.append((u, sim))

        return candidates

    def get_contextual_neighbors(self, center_lemma, top_n=25, root_lemma=None, parent_lemma=None, root_sense_id=None,
                                 parent_sense_id=None, local_neighbor_lemmas=None, base_weight=0.45, parent_weight=0.30,
                                 root_weight=0.20, local_weight=0.00, domain_lambda=0.20):
        matched_center = self._resolve_key(center_lemma, self.index)
        if not matched_center: return center_lemma, []

        raw = self.index[matched_center]
        root_vec = self.get_representation_vector(root_lemma, root_sense_id) if root_lemma else None
        parent_vec = self.get_representation_vector(parent_lemma, parent_sense_id) if parent_lemma else None
        local_vec = None

        if local_weight > 0 and local_neighbor_lemmas and self.vectors:
            local_vecs = [self.vectors[self._resolve_key(w, self.vectors)] for w in local_neighbor_lemmas if
                          self._resolve_key(w, self.vectors)]
            if local_vecs: local_vec = np.mean(local_vecs, axis=0)

        out = []
        # Przekazujemy lokalnych sąsiadów TYLKO gdy ich waga w eksperymencie jest > 0
        effective_local = local_neighbor_lemmas if local_weight > 0 else None

        for u, base_sim, freq in raw:
            actual_u = self._resolve_key(u, self.vectors)
            if not actual_u: continue

            child_sid, child_vec, sense_score = self.choose_graph_sense(
                actual_u,
                root_lemma or matched_center,
                parent_lemma or matched_center,
                root_sense_id,
                parent_sense_id,
                effective_local,
                allow_induce=False
            )
            s_parent = self._cos(parent_vec, child_vec) if parent_vec is not None else 0.0
            s_root = self._cos(root_vec, child_vec) if root_vec is not None else 0.0
            s_local = self._cos(local_vec, child_vec) if local_vec is not None else 0.0

            # 1. Baza do karania - to Twoje dotychczasowe obliczenia
            contextual_score = (
                    base_weight * float(base_sim)
                    + parent_weight * s_parent
                    + root_weight * s_root
                    + local_weight * s_local
            )

            # 2. Pobranie hubności dla słowa 'u' z indeksu
            actual_candidate = self._resolve_key(u, self.hubness_index) or u
            hubness_penalty = self.hubness_index.get(actual_candidate, 0.0)

            # 3. Nałożenie kary na ostateczny wynik (z ujemnym score na selektywnych listach)
            final_score = contextual_score - (domain_lambda * hubness_penalty)

            out.append({
                "lemma": u,
                "base_similarity": float(base_sim),
                "contextual_score": float(contextual_score),
                "score": float(final_score),  # Ukarany score do rankingu
                "graph_weight": float(contextual_score),  # Prawdziwe podobieństwo do krawędzi
                "freq": int(freq),
                "sense_id": child_sid,
                "sense_score": float(sense_score)
            })

        out.sort(key=lambda x: x["score"], reverse=True)
        # ZMIANA: Zwracamy wszystkie wyliczone i posortowane węzły (limit np. do 150)
        return matched_center, out[:150]

class SemanticNetworkViewer:
    """Klasa renderująca i zarządzająca oknem grafu sieci semantycznej."""

    def __init__(self, parent_app, engine, theme, insert_query_callback):
        self.app = parent_app
        self.engine = engine
        self.theme = theme
        self.on_insert_query = insert_query_callback

        stack = get_plot_stack()
        self.nx = stack["nx"]
        self.FigureCanvasTkAgg = stack["FigureCanvasTkAgg"]

        # --- STAN TOPOLOGICZNY ---
        self.G = self.nx.Graph()
        self.pos = {}
        self.current_center = None

        # --- STAN HALO ---
        self.halo_nodes = {}

        # --- ZMIENNE STANU GRAFU DLA KONTEKSTU ---
        self.current_root = None
        self.node_parent = {}
        self.node_sense_id = {}
        self.node_root = {}
        self.expanded_centers_history = []

        self.draw_static_bridges = False
        self.draw_contextual_bridges = True

        # --- WSD ---
        self.selected_sense_id = None
        self.selected_members = set()
        self.current_senses = []
        self.last_neighbors = []

        self.win = ctk.CTkToplevel(self.app)
        self.win.title("Sieć semantyczna")
        self.win.geometry("1400x750")
        self.win.configure(fg_color=self.theme["app_bg"])
        self.win.attributes("-topmost", True)


        self.domain_lambda_var = tk.DoubleVar(value=0.20)

        self.layout_seed_var = ctk.StringVar(value="")

        # Inicjalizacja UI (w tym self.ax i self.canvas)
        self._build_ui(stack)

        # TOOLTIP: Inicjalizacja po zbudowaniu self.ax
        self._init_tooltip()

        # Podpięcie zdarzeń interakcji (Hover + Click)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def _init_tooltip(self):
        """Pomocnik bezpiecznie odtwarzający tooltip po wyczyszczeniu osi."""
        if hasattr(self, 'annot'):
            try:
                self.annot.remove()
            except Exception:
                pass

        self.annot = self.ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0", ec="black", lw=1, alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
        )
        self.annot.set_visible(False)
        self.annot.set_zorder(10)

    def _get_stable_angle(self, *parts):
        """Generuje stabilny kąt (w radianach) używając kryptograficznego hasha."""
        import hashlib
        import math
        key = "::".join(parts).encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).hexdigest()
        return math.radians(int(digest, 16) % 360)

    def _build_ui(self, stack):
        self.graph_container = ctk.CTkFrame(self.win, fg_color="white", corner_radius=12)
        self.graph_container.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.side_panel = ctk.CTkFrame(self.win, fg_color="transparent", width=500)
        self.side_panel.pack(side="right", fill="y", padx=(0, 20), pady=20)
        self.side_panel.pack_propagate(False)

        self.search_frame = ctk.CTkFrame(self.side_panel, fg_color="transparent")
        self.search_frame.pack(fill="x", pady=(0, 10))

        self.entry_word = ctk.CTkEntry(self.search_frame, placeholder_text="Słowo centralne...", font=("Verdana", 14),
                                       height=35)
        self.entry_word.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_word.bind("<Return>", self.execute_search)

        self.btn_go = ctk.CTkButton(self.search_frame, text="Eksploruj", width=90, height=35,
                                    font=("Verdana", 12, "bold"),
                                    command=self.execute_search)
        self.btn_go.pack(side="right")

        self.append_mode_var = ctk.BooleanVar(value=True)
        self.append_checkbox = ctk.CTkCheckBox(
            self.side_panel, text="Rozwijaj obecną gałąź",
            variable=self.append_mode_var, font=("Verdana", 11)
        )
        self.append_checkbox.pack(fill="x", pady=(0, 10))

        self.mode_var = ctk.StringVar(value="Eksploracja")
        self.mode_selector = ctk.CTkSegmentedButton(
            self.side_panel, values=["Eksploracja", "Kręgosłup (MST)", "Klastry"],
            variable=self.mode_var, command=lambda _: self.render_graph()
        )
        self.mode_selector.pack(fill="x", pady=(0, 10))

        # --- WSD controls (overlay + sort listy) ---
        self.wsd_var = ctk.StringVar(value="Wszystkie ramy")

        self.wsd_label = ctk.CTkLabel(
            self.side_panel, text="Profil użycia:", font=("Verdana", 12, "bold")
        )

        self.wsd_label.pack(fill="x", pady=(0, 4))

        self.wsd_menu = ctk.CTkOptionMenu(
            self.side_panel,
            variable=self.wsd_var,
            values=["Wszystkie ramy"],
            command=self.on_wsd_select,
            state="disabled"
        )
        self.wsd_menu.pack(fill="x", pady=(0, 10))

        self.btn_reset = ctk.CTkButton(self.side_panel, text="Wyczyść sieć", fg_color="#D9534F",
                                       command=self.reset_graph)
        self.btn_reset.pack(fill="x", pady=(0, 10))

        self.neighbors_limit_var = ctk.IntVar(value=25)
        self.btn_settings = ctk.CTkButton(self.side_panel, text="⚙ Ustawienia grafu",
                                          command=self.open_settings,
                                          fg_color="#6c757d", hover_color="#5a6268")
        self.btn_settings.pack(fill="x", pady=(0, 10))

        self.btn_report = ctk.CTkButton(
            self.side_panel,
            text="Raport semantyczny",
            command=self.generate_semantic_report,
            fg_color="#2E8B57",
            hover_color="#256F46"
        )
        self.btn_report.pack(fill="x", pady=(0, 10))

        self.results_frame = ctk.CTkScrollableFrame(
            self.side_panel,
            fg_color=self.theme["subframe_fg"],
            corner_radius=12,
            width=480
        )
        self.results_frame.pack(fill="both", expand=True)

        self.fig = stack["Figure"](figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = self.FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_container)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        self.fig.canvas.mpl_connect('scroll_event', self.zoom_on_scroll)

    def zoom_on_scroll(self, event):
        if event.xdata is None or event.ydata is None: return
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([event.xdata - new_width * (1 - relx), event.xdata + new_width * relx])
        self.ax.set_ylim([event.ydata - new_height * (1 - rely), event.ydata + new_height * rely])
        self.canvas.draw_idle()

    def reset_graph(self):
        self.G.clear()
        self.halo_nodes.clear() # Usunięto if hasattr(...)
        self.pos = {}
        self.current_center = None
        self.current_root = None
        self.ax.clear()
        self.canvas.draw()
        for widget in self.results_frame.winfo_children(): widget.destroy()

        self.node_parent.clear()
        self.node_sense_id.clear()
        self.node_root.clear()
        self.last_neighbors = []
        self.current_senses = []
        self.selected_sense_id = None
        self.selected_members = set()

        if hasattr(self, 'expanded_centers_history'):
            self.expanded_centers_history.clear()
        self._init_tooltip()

    def open_settings(self):
        # 1. ZABEZPIECZENIE: Sprawdzamy, czy okno już istnieje
        if hasattr(self, 'settings_win') and self.settings_win is not None and self.settings_win.winfo_exists():
            self.settings_win.lift()  # Wyciągnij na wierzch
            self.settings_win.focus()  # Zwróć na nie uwagę klawiatury/myszki
            return

        max_avail = self.engine.get_max_available_neighbors()
        if max_avail == 0:
            max_avail = 50

        # Przypisujemy okno do zmiennej instancji (self.settings_win)
        self.settings_win = ctk.CTkToplevel(self.win)
        self.settings_win.title("Ustawienia Grafu")
        self.settings_win.geometry("350x450")  # <--- POWIĘKSZONE OKNO

        # 2. NAPRAWA CHOWANIA SIĘ POD SPÓD
        self.settings_win.transient(self.win)  # Zawsze utrzymuj nad oknem grafu
        self.settings_win.grab_set()  # Blokuje klikanie w graf, dopóki to okno jest otwarte

        self.settings_win.configure(fg_color=self.theme["app_bg"])

        # Pozycjonowanie na środku okna grafu
        x = self.win.winfo_x() + (self.win.winfo_width() // 2) - 175
        y = self.win.winfo_y() + (self.win.winfo_height() // 2) - 200  # <--- ZMIENIONE WYRÓWNANIE DO ŚRODKA
        self.settings_win.geometry(f"+{x}+{y}")

        ctk.CTkLabel(self.settings_win, text=f"Liczba wyświetlanych sąsiadów\n(Max w tej sieci: {max_avail})",
                     font=("Verdana", 12)).pack(pady=10)

        slider = ctk.CTkSlider(
            self.settings_win,
            from_=5,
            to=max_avail,
            number_of_steps=max_avail - 5,
            variable=self.neighbors_limit_var
        )
        slider.pack(pady=10, padx=20)

        val_label = ctk.CTkLabel(self.settings_win, textvariable=self.neighbors_limit_var, font=("Verdana", 12, "bold"))
        val_label.pack()

        # --- NOWA SEKCJA: Preferencja domenowa ---
        domain_frame = ctk.CTkFrame(self.settings_win, fg_color="transparent")
        domain_frame.pack(fill="x", padx=10, pady=(15, 5))


        title_label = ctk.CTkLabel(domain_frame, text="Preferuj słownictwo domenowe", font=("Verdana", 12, "bold"))
        title_label.pack(pady=(0, 5))

        lambda_val_label = ctk.CTkLabel(domain_frame, text="", font=("Verdana", 11))
        lambda_val_label.pack(pady=(0, 5))

        def update_lambda_label(val):
            val = float(val)
            if val < 0.1:
                desc = "Wyłączone (standard)"
            elif val <= 0.3:
                desc = "Lekka preferencja (domyślnie)"
            elif val <= 0.6:
                desc = "Wyraźnie domenowo"
            else:
                desc = "Mocno selektywne"
            lambda_val_label.configure(text=f"Wartość: {val:.2f} — {desc}")

        # TWORZYMY SUWAK TYLKO RAZ:
        lambda_scale = ctk.CTkSlider(
            domain_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            variable=self.domain_lambda_var,
            command=update_lambda_label
        )
        lambda_scale.pack(fill="x", padx=15, pady=5)

        # Inicjalizacja tekstu - wywołujemy ręcznie po stworzeniu widgetów
        update_lambda_label(self.domain_lambda_var.get())

        tooltip_label = ctk.CTkLabel(
            domain_frame,
            text="Zmniejsza wagę słów generycznych (hubów),\nwydobywając słownictwo specyficzne.",
            text_color="gray",
            font=("Verdana", 10)
        )
        tooltip_label.pack(pady=(0, 5))


        def apply_and_close():
            # 1. Zapisujemy historię eksploracji, żeby zachować strukturę drzewa i gałęzi
            history_to_redraw = list(getattr(self, 'expanded_centers_history', []))
            saved_center = getattr(self, 'current_center', None)

            # 2. Bezpiecznie zamykamy okno ustawień
            if hasattr(self, 'settings_win') and self.settings_win is not None:
                self.settings_win.destroy()
                self.settings_win = None

            if history_to_redraw:
                # 3. Czyścimy "brudny" graf
                self.reset_graph()

                # 4. Odtwarzamy krok po kroku. Ponieważ nasza matematyczna "podłoga" działa
                # teraz perfekcyjnie, śmieciowe słowa po prostu nie przetrwają tego odtworzenia!
                for step in history_to_redraw:
                    self.explore_node(step["word"], parent=step.get("parent"))

                    # Przywrócenie ramy WSD, jeśli była wybrana
                    if step.get("sense_id") is not None:
                        self.node_sense_id[step["word"]] = step["sense_id"]

                # Aktualizujemy pasek wyszukiwania do ostatniego aktywnego węzła
                if self.current_center:
                    self.entry_word.delete(0, "end")
                    self.entry_word.insert(0, self.current_center)

            elif saved_center:
                # Fallback, jeśli nie było historii
                self.reset_graph()
                self.entry_word.delete(0, "end")
                self.entry_word.insert(0, saved_center)
                self.explore_node(saved_center, parent=None)

        # 3. Zabezpieczenie zamknięcia okna "iksem" (X) w rogu
        def on_close():
            self.settings_win.destroy()
            self.settings_win = None

        self.settings_win.protocol("WM_DELETE_WINDOW", on_close)

        seed_frame = ctk.CTkFrame(self.settings_win, fg_color="transparent")
        seed_frame.pack(fill="x", padx=10, pady=(5, 5))

        ctk.CTkLabel(seed_frame, text="Ziarno losowości (Seed)", font=("Verdana", 12, "bold")).pack(pady=(0, 5))

        seed_entry = ctk.CTkEntry(
            seed_frame,
            textvariable=self.layout_seed_var,
            placeholder_text="Zostaw puste dla losowości",
            justify="center"
        )
        seed_entry.pack(fill="x", padx=15)

        ctk.CTkLabel(
            seed_frame,
            text="Wpisz liczbę całkowitą, aby zamrozić układ grafu.",
            text_color="gray",
            font=("Verdana", 10)
        ).pack(pady=(0, 5))

        ctk.CTkButton(self.settings_win, text="Zastosuj", command=apply_and_close,
                      fg_color=self.theme["button_fg"], hover_color=self.theme["button_hover"]).pack(pady=10)

    def generate_semantic_report(self):
        lemma = (self.current_center or self.entry_word.get().strip())
        if not lemma:
            messagebox.showwarning("Brak lemy", "Najpierw wybierz lub wpisz słowo centralne.")
            return

        current_corpus_name = corpus_var.get()
        current_corpus_path = files.get(current_corpus_name)

        self.engine.build_semantic_report(
            parent_app=self.app,
            current_corpus_name=current_corpus_name,
            current_corpus_path=current_corpus_path,
            lemma=lemma,
            theme=self.theme,
            open_report_callback=open_webview_window,
            params={
                "report_top_k": 0,
                "hops": 2,
                "top_k": self.neighbors_limit_var.get(),
                "min_similarity": 0.45,
            }
        )

    def hit_test_core(self, event, pixel_threshold=25):
        """Zwraca nazwę głównego węzła (Core), jeśli w niego kliknięto/najechano."""
        if not self.G.nodes or event.x is None or event.y is None:
            return None

        import numpy as np
        click_px = np.array([event.x, event.y])

        closest_word = None
        min_dist_px = float('inf')

        for word in self.G.nodes():
            if word not in self.pos:
                continue

            # Transformacja współrzędnych danych na piksele ekranu
            node_px = self.ax.transData.transform(self.pos[word])

            dist_px = np.linalg.norm(click_px - node_px)
            if dist_px < min_dist_px and dist_px < pixel_threshold:
                min_dist_px = dist_px
                closest_word = word

        return closest_word

    def render_graph(self):
        import math
        self.ax.clear()

        # --- ZMIANA 1: Sprawdzamy czy mamy cokolwiek do rysowania (Core lub Halo) ---
        has_core = len(self.G.nodes()) > 0
        has_halo = bool(getattr(self, 'halo_nodes', None))

        if not has_core and not has_halo:
            self._init_tooltip()  # <--- DODANO TO!
            self.canvas.draw()
            return

        mode = self.mode_var.get()
        node_sizes, labels, node_colors = [], {}, []

        # Cała Twoja obecna logika Core (wykona się tylko jeśli self.G nie jest puste)
        if has_core:

            for n in self.G.nodes():
                n_type = self.G.nodes[n].get('type')
                freq = self.G.nodes[n].get('freq', 1)
                wdeg = sum(self.G[n][nbr].get('weight', 0) for nbr in self.G.neighbors(n))

                # Używamy pierwiastka dla lepszego odzwierciedlenia różnic
                # 300 to rozmiar bazowy, 5 to siła rośnięcia węzła.
                base_size = 300 + (math.sqrt(max(freq, 1)) * 5)

                # Zabezpieczenie, żeby węzeł nie zajął przypadkiem całego ekranu dla skrajnych słów (opcjonalne)
                base_size = min(base_size, 2000)

                if n == getattr(self, 'current_root', None):
                    final_size = max(base_size * 1.25, 1400)
                elif n == getattr(self, 'current_center', None):
                    if self.G.nodes[n].get('terminal'):
                        final_size = base_size * 1.05
                    else:
                        final_size = base_size * 1.15
                elif n_type == 'center':
                    if self.G.nodes[n].get('terminal'):
                        final_size = base_size * 0.95
                    else:
                        final_size = base_size * 1.05
                else:
                    final_size = base_size

                node_sizes.append(final_size)

                if n == getattr(self, 'current_root', None) or n == getattr(self, 'current_center',
                                                                            None) or n_type == 'center' or wdeg > 1.2 or len(
                        self.G.nodes()) < 50:
                    labels[n] = n

            if mode == "Klastry":
                from networkx.algorithms import community
                try:
                    comms = community.greedy_modularity_communities(self.G, weight='weight')
                    palette = ['#FF595E', '#1982C4', '#8AC926', '#FFCA3A', '#6A4C93', '#F15BB5', '#00BBF9', '#00F5D4']
                    for n in self.G.nodes():
                        for i, comm in enumerate(comms):
                            if n in comm:
                                node_colors.append(palette[i % len(palette)])
                                break
                        else:
                            node_colors.append('#CCCCCC')
                except Exception:
                    node_colors = ['#1982C4'] * len(self.G.nodes())
            else:
                for n in self.G.nodes():
                    if n == getattr(self, 'current_root', None):
                        node_colors.append('#FFCA3A')  # Złoty Rdzeń Absolutny
                    elif self.G.nodes[n].get('terminal'):
                        node_colors.append('#9E9E9E')  # Zgaszony szary dla liści
                    elif n == getattr(self, 'current_center', None):
                        node_colors.append('#FF2E63')  # Czerwone aktywne centrum
                    elif self.G.nodes[n].get('type') == 'center':
                        node_colors.append('#08D9D6')  # Morskie historyczne centra
                    else:
                        node_colors.append('#EAEAEA')  # Jasnoszary dla sąsiadów

            # --- WSD overlay z OCHRONĄ ROOTA ---
            members = getattr(self, 'selected_members', set()) or set()
            if members:
                accent = "#9A5BB6"
                dim = "lightgray"
                new_colors = []
                for n, current_color in zip(self.G.nodes(), node_colors):
                    if n == getattr(self, 'current_root', None):
                        new_colors.append('#FFCA3A')  # Ochrona: Root zawsze zostaje złoty!
                    else:
                        new_colors.append(accent if n in members else dim)
                node_colors = new_colors

            # --- DYNAMICZNE OBRYSY (Stroke) DLA CZYTELNOŚCI ---
            edge_colors_list = []
            line_widths_list = []
            for n in self.G.nodes():
                if n == getattr(self, 'current_root', None):
                    edge_colors_list.append('#2B2D42')  # Ciemnogranatowy, gruby obrys dla roota
                    line_widths_list.append(2.5)
                elif self.G.nodes[n].get('terminal'):
                    edge_colors_list.append('#707070')  # Ciemniejszy szary obrys dla ślepych zaułków
                    line_widths_list.append(1.5)
                else:
                    edge_colors_list.append('white')  # Czysty, biały obrys dla reszty (jak dotychczas)
                    line_widths_list.append(1.0)

            edges_to_draw = self.G.edges(data=True)
            if mode == "Kręgosłup (MST)":
                T = self.nx.maximum_spanning_tree(self.G, weight='weight')
                edges_to_draw = T.edges(data=True)


            # --- CIĄGŁE SKALOWANIE LINII ZAMIAST KUBEŁKÓW ---
            # --- CIĄGŁE SKALOWANIE LINII ZAMIAST KUBEŁKÓW ---
            # --- CIĄGŁE SKALOWANIE LINII Z DYNAMICZNĄ NORMALIZACJĄ ---
            edges_to_draw_list = list(edges_to_draw)
            if edges_to_draw_list:
                line_widths = []
                alphas = []

                center = getattr(self, 'current_center', None)
                root = getattr(self, 'current_root', None)

                # 1. Znajdujemy absolutne maksimum i minimum TYLKO dla głównych krawędzi
                main_weights = [d.get('weight', 0.0) for u, v, d in edges_to_draw_list
                                if (u == center or v == center or u == root or v == root)]

                if main_weights:
                    max_w = max(main_weights)
                    min_w = min(main_weights)
                    diff = max_w - min_w
                    if diff < 0.05: diff = 1.0  # Zabezpieczenie przed dzieleniem przez zero (graf z 1 sąsiadem)
                else:
                    max_w, min_w, diff = 1.0, 0.0, 1.0

                # 2. Rysujemy linie z rozciągnięciem kontrastu
                for u, v, d in edges_to_draw_list:
                    w = d.get('weight', 0.0)
                    is_main_edge = (u == center or v == center or u == root or v == root)

                    if is_main_edge:
                        # GŁÓWNE KRAWĘDZIE: Przeliczamy wagę na skalę od 0.0 (najsłabsza) do 1.0 (najsilniejsza)
                        norm_w = max(0.0, min(1.0, (w - min_w) / diff))

                        # Grubości skalujemy od 0.5 px (najsłabsza) do 5.5 px (lider!)
                        line_widths.append(0.5 + (norm_w ** 2) * 5.0)

                        # Przezroczystość od 20% do 90%
                        alphas.append(0.20 + (norm_w * 0.70))
                    else:
                        # MOSTY KONTEKSTOWE: Pozostają wyciszone w tle
                        line_widths.append((w ** 3) * 1.5)
                        alphas.append(max(0.05, min(0.30, w ** 2)))

                # 3. W Matplotlib musimy narysować krawędzie pętlą dla indywidualnego 'alpha'
                for (u, v, d), width, alpha in zip(edges_to_draw_list, line_widths, alphas):
                    self.nx.draw_networkx_edges(
                        self.G, self.pos,
                        edgelist=[(u, v)],
                        ax=self.ax,
                        width=width,
                        alpha=alpha,
                        edge_color='#8A9AAB'
                    )


            # Rysujemy węzły z dodaniem obrysów!
            node_collection = self.nx.draw_networkx_nodes(
                self.G, self.pos, ax=self.ax,
                node_size=node_sizes, node_color=node_colors,
                edgecolors=edge_colors_list, linewidths=line_widths_list
            )
            if node_collection is not None:
                node_collection.set_zorder(3)

            self.nx.draw_networkx_labels(self.G, self.pos, labels=labels, ax=self.ax, font_size=9,
                                         font_color='#1A202C', font_weight='bold',
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

        # --- ZMIANA 2: Węzły HALO jako chmura punktów na samym dole ---
        self.halo_scatter = None  # <--- DODANE ZEROWANIE NA SAMYM POCZĄTKU
        if has_halo:
            halo_positions = [d['pos'] for d in self.halo_nodes.values() if 'pos' in d]
            if halo_positions:
                hx, hy = zip(*halo_positions)
                # Zapisujemy referencję do scatter, przyda się do hit-testingu i zorder=1 rysuje je pod grafem
                self.halo_scatter = self.ax.scatter(
                    hx, hy,
                    s=40, c='gray', alpha=0.4, edgecolors='none', zorder=1
                )
        else:
            self.halo_scatter = None


        # --- ZMIANA 3: Odtworzenie Tooltipa ---
        self._init_tooltip()
        self.ax.margins(0.15)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas.draw()

    def hit_test_halo(self, event, pixel_threshold=10):
        """Zwraca słowo Halo, jeśli kliknięto/najechano blisko niego, licząc w pikselach."""
        if not self.halo_nodes or event.x is None or event.y is None:
            return None

        click_px = np.array([event.x, event.y])  # event.x i event.y to pozycje w PIKSELACH

        closest_word = None
        min_dist_px = float('inf')

        for word, data in self.halo_nodes.items():
            if 'pos' not in data: continue

            # Transformacja współrzędnych danych (layoutu) na piksele ekranu
            node_px = self.ax.transData.transform(data['pos'])

            dist_px = np.linalg.norm(click_px - node_px)
            if dist_px < min_dist_px and dist_px < pixel_threshold:
                min_dist_px = dist_px
                closest_word = word

        return closest_word


    def on_hover(self, event):
        if event.inaxes != self.ax: return

        # 1. Najpierw sprawdzamy Core
        hovered_core = self.hit_test_core(event, pixel_threshold=20)
        if hovered_core:
            pos = self.pos[hovered_core]
            self.annot.xy = pos
            self.annot.set_text(hovered_core)
            self.annot.set_visible(True)
            self.canvas.draw_idle()
            return

        # 2. Potem sprawdzamy Halo
        hovered_halo = self.hit_test_halo(event, pixel_threshold=10)
        if hovered_halo:
            pos = self.halo_nodes[hovered_halo]['pos']
            self.annot.xy = pos
            self.annot.set_text(hovered_halo)
            self.annot.set_visible(True)
            self.canvas.draw_idle()
        else:
            # Ukrycie etykiety, jeśli kursor jest w pustym miejscu
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax: return

        # --- NOWE 1: Sprawdzamy najpierw główne węzły (Core) ---
        clicked_core = self.hit_test_core(event, pixel_threshold=25)
        if clicked_core:
            #print(f"Aktywowanie istniejącego węzła: {clicked_core}")

            # Aktualizujemy pasek wyszukiwania w GUI
            self.entry_word.delete(0, 'end')
            self.entry_word.insert(0, clicked_core)

            # Symulujemy wciśnięcie przycisku/Entera (uwzględnia tryb dołączania)
            self.execute_search()
            return  # Przerywamy, żeby nie sprawdzać tła

        # --- NOWE 2: Sprawdzamy węzły tła (Halo) tylko jeśli nie kliknięto w Core ---
        clicked_halo = self.hit_test_halo(event, pixel_threshold=10)
        if clicked_halo:
            #print(f"Awansowanie węzła tła: {clicked_halo}")

            anchor = self.halo_nodes[clicked_halo].get('anchor')
            if clicked_halo in self.halo_nodes:
                del self.halo_nodes[clicked_halo]

            self.explore_node(clicked_halo, parent=anchor)

    def _format_sense_label(self, sense: dict) -> str:
        sid = sense.get("frame_id", sense.get("sense_id", "?"))
        label = (sense.get("label") or "").strip()
        anchors = sense.get("anchors", []) or []
        members = sense.get("members", []) or []
        frame_type = sense.get("frame_type", sense.get("profile_type", "semantic"))

        if frame_type == "contextual":
            prefix = "Rama kontekstowa"
        else:
            prefix = "Rama semantyczna"

        preview_terms = (anchors or members)[:4]
        preview = ", ".join(preview_terms)
        if len(anchors or members) > 4:
            preview += ", ..."

        if label:
            raw_tokens = {t.strip() for t in label.split(",") if t.strip()}
            anchor_tokens = {t.strip() for t in anchors[:3] if isinstance(t, str) and t.strip()}
            overlap = len(raw_tokens & anchor_tokens)
            bad_prefix = label.lower().startswith(("rama", "profil", "sense"))

            if not bad_prefix and (not anchor_tokens or overlap > 0):
                return f"{prefix} {sid}: {label}"

        return f"{prefix} {sid}: {preview}"


    def execute_search(self, event=None):
        word = self.entry_word.get().strip()
        if not word: return

        # Znormalizowane sprawdzanie, czy to nie jest to samo słowo
        current_norm = self.engine._resolve_key(getattr(self, 'current_center', None), self.engine.index)
        word_norm = self.engine._resolve_key(word, self.engine.index)

        if current_norm and word_norm and current_norm == word_norm:
            return

        if self.append_mode_var.get() and getattr(self, 'current_center', None):
            self.explore_node(word, parent=self.current_center)
        else:
            self.explore_node(word, parent=None)

    def explore_node(self, word, parent=None):
        if not word: return
        if parent is None:
            self.current_root = word



        root_lemma = self.current_root or word
        root_sense_id = self.node_sense_id.get(root_lemma)
        parent_sense_id = self.node_sense_id.get(parent) if parent else None

        local_neighbors = [n for n in self.G.neighbors(parent)] if parent and self.G.has_node(parent) else []

        # Pobieramy szerszą listę uwzględniającą karę (lambda)
        # Pobieramy szerszą listę uwzględniającą karę (lambda)
        matched_word, all_res = self.engine.get_contextual_neighbors(
            center_lemma=word, top_n=150,
            root_lemma=root_lemma, parent_lemma=parent or word,
            root_sense_id=root_sense_id, parent_sense_id=parent_sense_id, local_neighbor_lemmas=local_neighbors,
            domain_lambda=self.domain_lambda_var.get()
        )

        for widget in self.results_frame.winfo_children(): widget.destroy()

        self.current_center = matched_word

        # --- ZMIANA: Prawidłowy, rozciągliwy podział na Core oraz Halo ---
        limit = self.neighbors_limit_var.get()

        if all_res:
            best_score = all_res[0]["score"]
            lambda_val = float(self.domain_lambda_var.get())

            # Zoptymalizowany margines bezpieczeństwa
            # Używamy 0.45 zamiast 0.60, żeby podłoga wpadła idealnie
            # w "przepaść" wygenerowaną przez algorytm.
            margin = 0.35 + (0.45 * lambda_val)
            raw_floor = best_score - margin

            # Twarde dno: Podłoga odcięcia nigdy nie powinna być niższa niż -0.15.
            # Jeśli słowo po karze spada poniżej -0.15, to jest w 100% zepsutym hubem.
            score_floor = max(0.05, raw_floor)

            filtered_core = [x for x in all_res if x["score"] >= score_floor]
            core_res = filtered_core[:limit]

            core_lemmas = {x["lemma"] for x in core_res}
            halo_res = [x for x in all_res if x["lemma"] not in core_lemmas]

            # --- DEBUG LOG ---
            print(f"\n=== LAMBDA = {lambda_val:.2f} | center = {matched_word} ===")
            print(f"Lider: {best_score:.3f} | Margines: {margin:.3f} | PODŁOGA: {score_floor:.3f}")
            for item in all_res:
                marker = "✅ (CORE)" if item["score"] >= score_floor and item["lemma"] in core_lemmas else "❌ (HALO)"
                print(f"{item['lemma']:18s} score={item['score']:+.3f} {marker}")
        else:
            core_res = []
            halo_res = []

        self.last_neighbors = core_res  # W panelu bocznym pokazujemy tylko Core

        if parent:
            center_sid, _, _ = self.engine.choose_graph_sense(self.current_center, root_lemma, parent, root_sense_id,
                                                              parent_sense_id)
            self.node_sense_id[self.current_center] = center_sid
        else:
            self.node_sense_id.setdefault(self.current_center, None)

        if self.G.has_node(self.current_center):
            self.G.nodes[self.current_center]['type'] = 'center'

        # Zapisz historię eksploracji węzłów ręcznie klikniętych (tzw. "Pinned")
        step_record = {
            "word": matched_word,
            "parent": parent,
            "root": root_lemma,
            "sense_id": self.node_sense_id.get(matched_word)
        }

        self.expanded_centers_history = [s for s in self.expanded_centers_history if
                                         not (s.get("word") == matched_word and s.get("parent") == parent)]
        self.expanded_centers_history.append(step_record)

        if parent:
            self.node_parent[self.current_center] = parent
            self.node_root[self.current_center] = root_lemma

        # Pobieranie WSD (bez zmian)
        self.current_senses = self.engine.get_or_create_senses(self.current_center)
        if self.current_senses:
            values = ["Wszystkie ramy"] + [self._format_sense_label(s) for s in self.current_senses]
            self.wsd_menu.configure(values=values, state="normal")
            self.wsd_var.set("Wszystkie ramy")
        else:
            self.wsd_menu.configure(values=["Wszystkie ramy"], state="disabled")
            self.wsd_var.set("Wszystkie ramy")

        self.selected_sense_id = None
        self.selected_members = set()

        if not core_res:
            ctk.CTkLabel(self.results_frame, text=f"Ślepy zaułek (liść).\nBrak własnych powiązań dla: {matched_word}",
                         text_color="gray").pack(pady=20)

            self._add_terminal_core_node(self.current_center, parent=parent)
            self._update_core_layout()
            self._cleanup_halo()
            self._update_halo_positions()
            self.render_graph()
            return

            # --- ZMIANA: Przekazujemy również halo_res do aktualizacji grafu ---
        self.update_graph_data(self.current_center, core_res, halo_res, parent)
        self.render_graph()
        self._render_neighbors_list()

    def _add_contextual_bridges(self, neighbors_data, sim_threshold=0.62, max_bridges_per_node=2):
        """Łączy nowo dodanych sąsiadów w lokalną siatkę bazując na aktualnym sensie/wektorze."""
        reps = {}
        # <--- POPRAWKA 3: Budowa mostów bez kary za hubowość
        eligible_neighbors = [
            item for item in neighbors_data
            if item.get("contextual_score", item.get("base_similarity", 0.0)) >= 0.35
        ]

        # 1. Pobierzemy faktyczne wektory (reprezentacje) używane w tym widoku
        for item in eligible_neighbors:  # ZMIANA: pętla iteruje teraz po przefiltrowanej liście
            lemma = item["lemma"]
            sid = item.get("sense_id")
            vec = self.engine.get_representation_vector(lemma, sid)
            if vec is not None:
                reps[lemma] = vec

        bridge_counts = {lemma: 0 for lemma in reps}
        lemmas = list(reps.keys())

        # 2. Pętla porównująca każdego sąsiada z każdym innym sąsiadem
        for i in range(len(lemmas)):
            for j in range(i + 1, len(lemmas)):
                u, v = lemmas[i], lemmas[j]

                # Zabezpieczenie przed "makaronem" (zbyt gęstą siecią)
                if bridge_counts[u] >= max_bridges_per_node or bridge_counts[v] >= max_bridges_per_node:
                    continue

                # Liczymy rzeczywiste podobieństwo węzłów w locie
                sim = self.engine._cos(reps[u], reps[v])

                if sim >= sim_threshold:
                    if self.G.has_edge(u, v):
                        self.G[u][v]["weight"] = max(self.G[u][v].get("weight", 0), sim)
                    else:
                        self.G.add_edge(u, v, weight=sim)

                    bridge_counts[u] += 1
                    bridge_counts[v] += 1

    def _prune_center_neighbors(self, center_word, desired_core_words):
        """
        Usuwa z core tych sąsiadów centrum, którzy nie należą już do nowego top N (desired_core).
        Zdegradowane słowa wrzuca do tła (halo), o ile nie są zablokowanymi centrami (pinned).
        """
        if not self.G.has_node(center_word):
            return

        desired = set(desired_core_words)
        pinned_centers = {step["word"] for step in getattr(self, 'expanded_centers_history', [])}

        # Iterujemy po aktualnych sąsiadach w grafie
        for nbr in list(self.G.neighbors(center_word)):
            # Centrum i historyczne centra zostają nienaruszone
            if nbr == center_word or nbr in desired or nbr in pinned_centers:
                continue

            # Nie degradujemy węzłów, które same są centrami
            if self.G.nodes[nbr].get("type") == "center":
                continue

            # Pobieramy dotychczasową siłę połączenia, by zachować estetykę tła
            sim = self.G[center_word][nbr].get("weight", 0.35)

            # Downgrade do halo
            self.halo_nodes[nbr] = {
                "anchor": center_word,
                "sim": max(0.35, float(sim))
            }

            # Odpinamy krawędź od centrum
            if self.G.has_edge(center_word, nbr):
                self.G.remove_edge(center_word, nbr)

            # Jeśli węzeł został sam (sierota) -> usuń go całkowicie ze struktury
            if self.G.has_node(nbr) and self.G.degree(nbr) == 0:
                self.G.remove_node(nbr)
                self.pos.pop(nbr, None)
                self.node_sense_id.pop(nbr, None)
                self.node_parent.pop(nbr, None)
                self.node_root.pop(nbr, None)

    # ZMIANA: Dodany argument halo_data
    def update_graph_data(self, center_word, core_data, halo_data, parent=None):
        """Główny orkiestrator aktualizacji grafu rozbity na czytelne kroki."""

        # 1. Pobierz listę słów, które TERAZ mają prawo być w Core
        desired_core_words = [item["lemma"] for item in core_data]

        # 2. NAJPIERW usuń stare sąsiedztwo, które już nie mieści się w core przy obecnej lambdzie
        self._prune_center_neighbors(center_word, desired_core_words)

        # 3. DOPIERO POTEM dodaj nowe krawędzie i węzły Core
        self._update_core_topology(center_word, core_data, parent)
        self._update_core_layout()

        # 4. Zaktualizuj tło korzystając z posortowanej listy po nałożeniu kary Lambda!
        self._update_halo_candidates_from_data(center_word, halo_data)

        self._cleanup_halo()  # Usuwa węzły, które ewentualnie awansowały z Halo do Core
        self._update_halo_positions()

    def _update_halo_candidates_from_data(self, center_word, halo_data):
        """Popula tło semantyczne kandydatami wyliczonymi zgodnie z aktualnym rygorem (lambda)."""
        for item in halo_data:
            n_word = item["lemma"]
            # Estetyczna siła grawitacji tła nadal korzysta z obiektywnego podobieństwa
            sim = item.get("base_similarity", 0.35)

            if sim >= 0.35:
                if n_word not in self.halo_nodes or sim > self.halo_nodes[n_word].get('sim', 0):
                    self.halo_nodes[n_word] = {'anchor': center_word, 'sim': sim}

    def _update_core_topology(self, center_word, neighbors_data, parent=None):
        """Zarządza dodawaniem węzłów i krawędzi (Core)."""

        def add_or_update_edge(u, v, w):
            if self.G.has_edge(u, v):
                self.G[u][v]['weight'] = max(self.G[u][v].get('weight', 0), w)
            else:
                self.G.add_edge(u, v, weight=w)

        # --- POPRAWKA 2: Wymuszenie statusu centrum ---

        if not self.G.has_node(center_word):
            self.G.add_node(center_word, type='center', freq=1, terminal=False)
        else:
            self.G.nodes[center_word]['type'] = 'center'
            self.G.nodes[center_word]['terminal'] = False

        # Gwarancja połączenia dla rzadkich słów
        if parent and self.G.has_node(parent) and parent != center_word:
            p_vec = self.engine.get_representation_vector(parent)
            c_vec = self.engine.get_representation_vector(center_word)
            sim = self.engine._cos(p_vec, c_vec) if p_vec is not None and c_vec is not None else 0.5
            add_or_update_edge(center_word, parent, max(0.3, sim))

        # Dodawanie sąsiadów
        for item in neighbors_data:
            n_word = item["lemma"]

            # --- ZMIANA: Pobieramy wagę dla krawędzi (bez kary za hubowość) ---
            # Jeśli graph_weight nie istnieje (dla bezpieczeństwa wstecznego), używamy score
            edge_weight = item.get("graph_weight", item["score"])

            n_freq = item["freq"]
            sense_id = item["sense_id"]

            if not self.G.has_node(n_word):
                self.G.add_node(n_word, type='neighbor', freq=n_freq, sense_id=sense_id, parent=center_word,
                                root=self.current_root)
            else:
                self.G.nodes[n_word]["sense_id"] = sense_id

            self.node_sense_id[n_word] = sense_id

            # --- ZMIANA: Używamy edge_weight zamiast ukaranego score ---
            add_or_update_edge(center_word, n_word, edge_weight)

        # --- PRZYWRÓCONY KOD MOZSTÓW Z POPRZEDNIEJ WERSJI ---
        if getattr(self, 'draw_contextual_bridges', True):
            self._add_contextual_bridges(neighbors_data)
        elif getattr(self, 'draw_static_bridges', False):
            min_bridge_sim = 0.55
            for item in neighbors_data:
                n_word = item["lemma"]
                for nn_word, nn_score, nn_freq in self.engine.index.get(n_word, [])[:10]:
                    if nn_score < min_bridge_sim: break
                    if self.G.has_node(nn_word) and nn_word != n_word:
                        if not self.engine.is_mutual_knn(n_word, nn_word): continue
                        thr = self.engine.dynamic_bridge_threshold(
                            self.G.nodes[n_word].get('freq', 0), self.G.nodes[nn_word].get('freq', 0),
                            base=min_bridge_sim
                        )
                        if nn_score >= thr:
                            add_or_update_edge(n_word, nn_word, nn_score)

    def _add_terminal_core_node(self, center_word, parent=None):
        """Dodaje węzeł do grafu jawnie jako ślepy zaułek (terminal node)."""

        def add_or_update_edge(u, v, w):
            if self.G.has_edge(u, v):
                self.G[u][v]['weight'] = max(self.G[u][v].get('weight', 0), w)
            else:
                self.G.add_edge(u, v, weight=w)

        # 1. Dodajemy węzeł ze specjalną flagą terminal=True
        if not self.G.has_node(center_word):
            self.G.add_node(center_word, type='center', freq=1, terminal=True)
        else:
            self.G.nodes[center_word]['type'] = 'center'
            self.G.nodes[center_word]['terminal'] = True

        # 2. Gwarancja połączenia z rodzicem (żeby nie latał w próżni)
        if parent and self.G.has_node(parent) and parent != center_word:
            p_vec = self.engine.get_representation_vector(parent)
            c_vec = self.engine.get_representation_vector(center_word)
            sim = self.engine._cos(p_vec, c_vec) if p_vec is not None and c_vec is not None else 0.5
            add_or_update_edge(center_word, parent, max(0.3, sim))

    def _update_core_layout(self):
        """Przelicza fizykę ułożenia głównych węzłów."""
        import math
        num_nodes = len(self.G.nodes())

        # Zwiększamy 'k', żeby graf miał więcej miejsca na odepchnięcie słabych słów
        dynamic_k = min(1.5, max(0.3, 3.0 / math.sqrt(num_nodes) if num_nodes > 0 else 0.5))


        # Manipulacja sprężynami dla fizyki układu
        for u, v, d in self.G.edges(data=True):
            w = d.get('weight', 0.5)
            # Potęga 4 sprawi, że słabsze słowa zredukują się do ułamków, a silne zostaną mocne
            d['physics_weight'] = w ** 4

        raw_seed = self.layout_seed_var.get().strip()
        try:
            current_seed = int(raw_seed) if raw_seed else None
        except ValueError:
            current_seed = None  # Bezpieczny fallback, gdyby ktoś wpisał litery

        # Wywołanie algorytmu z użyciem nowej wagi i większej liczby iteracji
        self.pos = self.nx.spring_layout(
            self.G,
            pos=self.pos if self.pos else None,
            k=dynamic_k,
            iterations=50,  # Więcej iteracji, żeby węzły zdążyły odlecieć
            weight='physics_weight',  # <--- KLUCZOWE: Mówimy algorytmowi, by użył zmanipulowanej wagi
            seed = current_seed  # <--- Podpięcie zmiennej
        )

    def _update_halo_candidates(self, center_word):
        """Pobiera nowych kandydatów do tła korzystając z czystego API z engine'u."""
        candidates = self.engine.get_halo_candidates(center_word, top_n=150, min_sim=0.35)

        for n_word, sim in candidates:
            if not self.G.has_node(n_word):
                # Usunięto zbędny hasattr, bo halo_nodes jest gwarantowane w __init__
                if n_word not in self.halo_nodes or sim > self.halo_nodes[n_word].get('sim', 0):
                    self.halo_nodes[n_word] = {'anchor': center_word, 'sim': sim}

    def _update_halo_positions(self):
        """Układa kropki tła za pomocą barycentrum grawitacyjnego i stabilnego hashowania."""
        import math
        core_vectors = {}
        for n in self.G.nodes():
            vec = self.engine.get_representation_vector(n, self.node_sense_id.get(n))
            if vec is not None:
                core_vectors[n] = vec

        for word, data in list(self.halo_nodes.items()):
            w_vec = self.engine.get_representation_vector(word)
            anchor = data.get('anchor')

            if w_vec is None or not core_vectors or anchor not in self.pos:
                ax, ay = self.pos.get(anchor, (0, 0))
                # Tutaj też powiększamy dystans awaryjny
                distance = 2.0 + (1.0 - data.get('sim', 0.5)) * 3.0
                angle = self._get_stable_angle(anchor, word)
                data['pos'] = (ax + distance * math.cos(angle), ay + distance * math.sin(angle))
                continue

            sum_x, sum_y, sum_weights = 0.0, 0.0, 0.0

            for core_node, c_vec in core_vectors.items():
                if core_node in self.pos:
                    sim = self.engine._cos(w_vec, c_vec)
                    if sim > 0.1:
                        weight = sim ** 3
                        sum_x += self.pos[core_node][0] * weight
                        sum_y += self.pos[core_node][1] * weight
                        sum_weights += weight

            if sum_weights > 0:
                base_x = sum_x / sum_weights
                base_y = sum_y / sum_weights

                jitter_angle = self._get_stable_angle(word, "jitter")
                max_sim_to_anchor = data.get('sim', 0.5)

                # --- NOWOŚĆ: WYPYCHANIE TŁA (HALO) POZA GRAF ---
                # 1.5 to "twarda tarcza" (minimalna odległość wypchnięcia poza rdzeń)
                # 3.0 to współczynnik rozpraszania chmury (im słabsze słowo, tym dalej leci)
                distance_push = 1.5 + (1.0 - max_sim_to_anchor) * 3.0

                data['pos'] = (base_x + distance_push * math.cos(jitter_angle),
                               base_y + distance_push * math.sin(jitter_angle))
            else:
                del self.halo_nodes[word]

    def _cleanup_halo(self):
        """Gwarantuje, że węzeł nigdy nie występuje jednocześnie w grafie i w tle."""
        keys_to_delete = [w for w in self.halo_nodes if self.G.has_node(w)]
        for w in keys_to_delete:
            del self.halo_nodes[w]

    def _render_neighbors_list(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        res = self.last_neighbors or []
        if not res:
            return

        members = self.selected_members or set()

        def sort_key(item):
            n_word = item["lemma"]
            in_sense = (n_word in members) if members else False
            return (1 if in_sense else 0, float(item["score"]), int(item["freq"]))

        res_sorted = sorted(res, key=sort_key, reverse=True)

        for item in res_sorted:
            n_word = item["lemma"]
            n_freq = item["freq"]
            n_score = item.get("score", 0.0)
            n_base_sim = item.get("base_similarity", 0.0)

            has_network = (
                    (n_word in self.engine.index)
                    or (n_word.lower() in self.engine.index)
                    or (n_word.capitalize() in self.engine.index)
            )
            btn_state = "normal" if has_network else "disabled"

            t_color = "gray50" if (members and n_word not in members) else self.theme["label_text"]
            cmd = (lambda w=n_word, p=self.current_center: self.explore_node(w, parent=p)) if has_network else None

            row = ctk.CTkFrame(self.results_frame, fg_color="transparent")
            row.pack(fill="x", pady=2, padx=2)

            # Kolumny: lemma | score | sim | freq | +
            row.grid_columnconfigure(0, weight=1, minsize=140)
            row.grid_columnconfigure(1, weight=0)
            row.grid_columnconfigure(2, weight=0)
            row.grid_columnconfigure(3, weight=0)
            row.grid_columnconfigure(4, weight=0)

            # 1. Lemma
            ctk.CTkButton(
                row,
                text=n_word,
                anchor="w",
                fg_color="transparent",
                text_color=t_color,
                state=btn_state,
                command=cmd,
                height=28
            ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

            # 2. Score (krótszy napis, żeby się mieścił)
            ctk.CTkLabel(
                row,
                text=f"sc {n_score:.2f}",
                text_color="#1982C4",
                font=("Verdana", 9, "bold"),
                width=52
            ).grid(row=0, column=1, padx=2)

            # 3. Base similarity
            ctk.CTkLabel(
                row,
                text=f"sim {n_base_sim:.2f}",
                text_color="#8A9AAB",
                font=("Verdana", 8),
                width=50
            ).grid(row=0, column=2, padx=2)

            # 4. Frekwencja
            ctk.CTkLabel(
                row,
                text=f"f {n_freq:,}".replace(",", " "),
                text_color="gray60",
                font=("Verdana", 8),
                width=48
            ).grid(row=0, column=3, padx=2)

            # 5. Plus
            ctk.CTkButton(
                row,
                text="+",
                width=26,
                height=24,
                command=lambda w=n_word: self.on_insert_query(w)
            ).grid(row=0, column=4, padx=(4, 0))


    def on_wsd_select(self, choice: str):
        if choice == "Wszystkie ramy":
            self.selected_sense_id = None
            self.selected_members = set()
            self.node_sense_id[self.current_center] = None
        else:
            sid = None
            try:
                # Obsługa nowych etykiet:
                # "Rama semantyczna 0: ..."
                # "Rama kontekstowa 1: ..."
                if choice.startswith("Rama semantyczna"):
                    sid = int(choice.split("Rama semantyczna", 1)[1].split(":", 1)[0].strip())
                elif choice.startswith("Rama kontekstowa"):
                    sid = int(choice.split("Rama kontekstowa", 1)[1].split(":", 1)[0].strip())
                else:
                    # fallback kompatybilności ze starymi etykietami
                    clean_choice = (
                        choice
                        .replace("Sens", "Rama")
                        .replace("Profil", "Rama")
                    )
                    sid = int(clean_choice.split("Rama", 1)[1].split(":", 1)[0].strip())
            except Exception as e:
                import logging
                logging.warning(f"Nie udało się sparsować wyboru ramy '{choice}': {e}")
                sid = None

            self.selected_sense_id = sid
            self.node_sense_id[self.current_center] = sid

            if sid is not None and 0 <= sid < len(self.current_senses):
                self.selected_members = set(self.current_senses[sid].get("members", []) or [])
            else:
                self.selected_members = set()

        for step in reversed(self.expanded_centers_history):
            if step.get("word") == self.current_center and step.get("parent") == self.node_parent.get(
                    self.current_center):
                step["sense_id"] = self.selected_sense_id
                break

        self.render_graph()
        self._render_neighbors_list()

class TopicEngine:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.model = None
        self.topics = None
        self.probs = None
        self.docs = []
        self.timestamps = []

        # Wybór modelu (Sentence Transformer).
        self.embedding_model_name = "sdadas/st-polish-paraphrase-from-mpnet"

    def load_data(self):
        """Wczytuje teksty i daty z pliku parquet wygenerowanego przez creator.py"""
        logging.info(f"Wczytywanie danych z {self.parquet_path}...")
        try:
            df = pd.read_parquet(self.parquet_path)

            # Wymagane kolumny: Treść i Data publikacji
            if "Treść" not in df.columns:
                raise ValueError("Brak kolumny 'Treść' w pliku parquet.")

            # Odsiewamy puste teksty
            df = df.dropna(subset=['Treść'])
            df = df[df['Treść'].str.strip() != ""]

            self.docs = df['Treść'].tolist()

            # Pobieranie dat, jeśli istnieją
            if "Data publikacji" in df.columns:
                self.timestamps = df['Data publikacji'].tolist()
            else:
                self.timestamps = ["0000-00-00"] * len(self.docs)

            logging.info(f"Wczytano {len(self.docs)} dokumentów do analizy tematycznej.")
            return True
        except Exception as e:
            logging.error(f"Błąd ładowania danych: {e}")
            return False

    def train_model(self, nr_topics=None, force_retrain=False, use_stopwords=True, diversity=0.2):
        """Trenuje model BERTopic lub ładuje gotowy z dysku."""
        import os
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        if not self.docs:
            return False

        model_save_path = self.parquet_path.replace(".parquet", ".bertopic")
        models_dir = os.path.join(BASE_DIR_CORP, "models", "sentence_transformers")
        os.makedirs(models_dir, exist_ok=True)
        sentence_model = SentenceTransformer(self.embedding_model_name, cache_folder=models_dir)

        if not force_retrain and os.path.exists(model_save_path):
            logging.info(f"Znaleziono gotowy model tematyczny: {model_save_path}. Wczytywanie...")
            self.model = BERTopic.load(model_save_path, embedding_model=sentence_model)
            return True

        # Słownik konfiguracji startowej
        bertopic_config = {
            "embedding_model": sentence_model,
            "language": "polish",
            "calculate_probabilities": False,
            "nr_topics": nr_topics,
            "verbose": True
        }

        # --- Warunkowe podłączenie MMR (Diversity) ---
        if diversity > 0.0:
            from bertopic.representation import MaximalMarginalRelevance
            representation_model = MaximalMarginalRelevance(diversity=diversity)
            bertopic_config["representation_model"] = representation_model
            logging.info(f"Włączono algorytm MMR wymuszający różnorodność słów (diversity={diversity})")



        # --- Warunkowe podłączenie stoplisty ---
        if use_stopwords:
            polish_stopwords = [
                "a", "aby", "ach", "acz", "aczkolwiek", "aj", "albo", "ale", "alez", "ależ", "ani", "az", "aż",
                "bardziej", "bardzo", "beda", "bedzie", "bez", "deda", "będą", "bede", "będę", "będzie", "bo",
                "bowiem", "by", "byc", "być", "byl", "byla", "byli", "bylo", "byly", "był", "była", "było",
                "były", "bynajmniej", "cala", "cali", "caly", "cała", "cały", "ci", "cie", "ciebie", "cię", "co",
                "cokolwiek", "cos", "coś", "czasami", "czasem", "czemu", "czy", "czyli", "daleko", "dla",
                "dlaczego", "dlatego", "do", "dobrze", "dokad", "dokąd", "dosc", "dość", "duzo", "dużo", "dwa",
                "dwaj", "dwie", "dwoje", "dzis", "dzisiaj", "dziś", "gdy", "gdyby", "gdyz", "gdyż", "gdzie",
                "gdziekolwiek", "gdzies", "gdzieś", "go", "i", "ich", "ile", "im", "inna", "inne", "inny",
                "innych", "iz", "iż", "ja", "jak", "jakas", "jakaś", "jakby", "jaki", "jakichs", "jakichś",
                "jakie", "jakis", "jakiś", "jakiz", "jakiż", "jakkolwiek", "jako", "jakos", "jakoś", "ją", "je",
                "jeden", "jedna", "jednak", "jednakze", "jednakże", "jedno", "jego", "jej", "jemu", "jesli",
                "jest", "jestem", "jeszcze", "jeśli", "jezeli", "jeżeli", "juz", "już", "kazdy", "każdy", "kiedy",
                "kilka", "kims", "kimś", "kto", "ktokolwiek", "ktora", "ktore", "ktorego", "ktorej", "ktory",
                "ktorych", "ktorym", "ktorzy", "ktos", "ktoś", "która", "które", "którego", "której", "który",
                "których", "którym", "którzy", "ku", "lat", "lecz", "lub", "ma", "mają", "mało", "mam", "mi",
                "miedzy", "między", "mimo", "mna", "mną", "mnie", "moga", "mogą", "moi", "moim", "moj", "moja",
                "moje", "moze", "mozliwe", "mozna", "może", "możliwe", "można", "mój", "mu", "musi", "my", "na",
                "nad", "nam", "nami", "nas", "nasi", "nasz", "nasza", "nasze", "naszego", "naszych", "natomiast",
                "natychmiast", "nawet", "nia", "nią", "nic", "nich", "nie", "niech", "niego", "niej", "niemu",
                "nigdy", "nim", "nimi", "niz", "niż", "no", "o", "obok", "od", "około", "on", "ona", "one",
                "oni", "ono", "oraz", "oto", "owszem", "pan", "pana", "pani", "po", "pod", "podczas", "pomimo",
                "ponad", "poniewaz", "ponieważ", "powinien", "powinna", "powinni", "powinno", "poza", "prawie",
                "przeciez", "przecież", "przed", "przede", "przedtem", "przez", "przy", "roku", "rowniez",
                "również", "sam", "sama", "są", "sie", "się", "skad", "skąd", "soba", "sobą", "sobie", "sposob",
                "sposób", "swoje", "ta", "tak", "taka", "taki", "takie", "takze", "także", "tam", "te", "tego",
                "tej", "ten", "teraz", "też", "to", "toba", "tobą", "tobie", "totez", "toteż", "tobą", "trzeba",
                "tu", "tutaj", "twoi", "twoim", "twoj", "twoja", "twoje", "twój", "twym", "ty", "tych", "tylko",
                "tym", "u", "w", "wam", "wami", "was", "wasz", "wasza", "wasze", "we", "według", "wiele", "wielu",
                "więc", "więcej", "wlasnie", "właśnie", "wszyscy", "wszystkich", "wszystkie", "wszystkim",
                "wszystko", "wtedy", "wy", "z", "za", "zaden", "zadna", "zadne", "zadnych", "zapewne", "zawsze",
                "ze", "zeby", "znowu", "zł", "znow", "znowu", "znów", "zostal", "został", "żaden", "żadna",
                "żadne", "żadnych", "że", "żeby"
            ]
            vectorizer_model = CountVectorizer(stop_words=polish_stopwords)
            bertopic_config["vectorizer_model"] = vectorizer_model
            logging.info("Dołączono własną listę stop-words do wektoryzatora.")

        logging.info(f"Rozpoczynam trening od zera z parametrem nr_topics={nr_topics}...")

        self.model = BERTopic(**bertopic_config)
        self.topics, self.probs = self.model.fit_transform(self.docs)

        logging.info(f"Trening zakończony. Zapisuję model do: {model_save_path}")
        self.model.save(model_save_path)

        return True

    def get_topic_info(self):
        """Zwraca DataFrame z informacjami o tematach (ID, Liczba tekstów, Słowa kluczowe)."""
        if self.model:
            return self.model.get_topic_info()
        return None

    def calculate_topics_over_time(self):
        """Generuje trendy bez ryzyka asynchronizacji danych po wczytaniu z cache."""
        if not self.model:
            return None

        # Przekazujemy absolutnie wszystkie dokumenty i daty, bez wycinania 0000-00-00.
        # Puste daty pojawią się po prostu jako pierwszy punkt na lewo od wykresu.
        try:
            topics_over_time = self.model.topics_over_time(
                self.docs,
                self.timestamps,
                nr_bins=20
            )
            return topics_over_time
        except Exception as e:
            logging.info(f"Błąd podczas obliczania trendów w czasie: {e}")
            return None

    def visualize_dynamic_topics(self, topics_over_time, top_n_topics=15):
        """Zwraca interaktywny wykres Plotly obrazujący trendy w czasie."""
        if self.model and topics_over_time is not None:
            return self.model.visualize_topics_over_time(topics_over_time, top_n_topics=top_n_topics)
        return None

    def visualize_topic_map(self):
        """Zwraca interaktywną mapę (UMAP) pokazującą jak tematy leżą względem siebie."""
        if self.model:
            return self.model.visualize_topics()
        return None

    def visualize_word_scores(self, top_n_topics=10):
        """Generuje wykres słupkowy Word scores (c-TF-IDF) dla top tematów."""
        if self.model:
            # Pokazuje najważniejsze słowa i ich wagi dla wybranych tematów
            return self.model.visualize_barchart(top_n_topics=top_n_topics, n_words=10)
        return None




# ==========================================
# KOMPATYBILNE WRAPPERY DLA RESZTY PROGRAMU
# ==========================================
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

    SemanticNetworkViewer(app, semantic_engine, theme, insert_to_query)

# ==========================================
# LAZY LOADERY (Wczytywanie na żądanie)
# ==========================================
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
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        # Potrzebujemy obu backendów - jednego do interfejsu (Tezaurus), drugiego do plików (Wykresy)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.cm as cm
        import networkx as nx

        _plot_stack = {
            "plt": plt,
            "Figure": Figure,
            "FigureCanvasTkAgg": FigureCanvasTkAgg,
            "FigureCanvasAgg": FigureCanvasAgg, # Odzyskany silnik!
            "cm": cm,
            "nx": nx
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
    colloc_data: list = field(default_factory=list)
    current_profile_dict: dict = field(default_factory=dict)
    profile_target_lemma: str = ""
    profile_data: list = field(default_factory=list)
    profile_rel_options: list = field(default_factory=list)
    profile_selected_rel: str = ""

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
    'kontekst': 250,
    'min_tokens_threshold': 0
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
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tw = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        # Tworzymy pływające okienko bez ramek
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        # Rysujemy chmurkę (zawsze w czytelnym, ciemnym motywie z ramką)
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#1F2328", foreground="#FFFFFF",
                         relief='solid', borderwidth=1,
                         font=("Verdana", 10))
        label.pack(ipadx=10, ipady=10)

    def leave(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None

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
    t_find_start = time.perf_counter()
    if warnings_list is None:
        warnings_list = []

    global search_status

    # Wymuszenie odświeżenia UI (pokazanie ekranu ładowania)
    text_result.after(0, lambda: display_page(query, selected_corpus))

    # Pobranie opcji frekwencyjnych z zapytania
    freq_opts = parse_frequency_attributes(query, "frequency_orth")
    freq_base_opts = parse_frequency_attributes(query, "frequency_base")

    # NAPRAWA: Usunięcie tagów frekwencyjnych z zapytania po ich wczytaniu
    query = re.sub(r'<frequency_orth\s+[^>]+>', '', query, flags=re.IGNORECASE).strip()
    query = re.sub(r'<frequency_base\s+[^>]+>', '', query, flags=re.IGNORECASE).strip()

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

    # --- PREFILTER: liczony raz per grupa, a maska metadanych raz globalnie ---
    group_jobs = []
    all_valid_row_ids = set()

    for group_tuple in parsed_query_groups:
        # Używamy prefiltru tylko dla tej konkretnej grupy
        group_row_ids = get_prefiltered_rows([group_tuple], selected_corpus, df.index)
        group_jobs.append((*group_tuple, group_row_ids))
        if group_row_ids:
            all_valid_row_ids.update(group_row_ids)

    if not all_valid_row_ids:
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

    # ✅ Apply all metadata filters once
    filtered_df_base = filtered_df_base[mask]

    # 3) Słowniki metadanych budujemy też tylko raz, na odfiltrowanym koszyku
    dates_dict = filtered_df_base[
        "Data publikacji"].to_dict() if "Data publikacji" in filtered_df_base.columns else {}
    titles_dict = filtered_df_base["Tytuł"].to_dict() if "Tytuł" in filtered_df_base.columns else {}
    authors_dict = filtered_df_base["Autor"].to_dict() if "Autor" in filtered_df_base.columns else {}

    exclude_cols = {
        "Data publikacji", "Tytuł", "Autor", "tags", "Treść", "token_counts",
        "tokens", "lemmas", "deprels", "postags", "full_postags",
        "word_ids", "sentence_ids", "head_ids", "start_ids", "end_ids", "ners", "upostags",
        "corefs"
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
            original_row_index = row.Index

            # --- 1. SZYBKIE LISTY PYTHONOWE ---
            tokens = row.tokens.tolist() if hasattr(row.tokens, "tolist") else row.tokens
            lemmas = row.lemmas.tolist() if hasattr(row.lemmas, "tolist") else row.lemmas
            deprels = row.deprels.tolist() if hasattr(row.deprels, "tolist") else row.deprels
            postags = row.postags.tolist() if hasattr(row.postags, "tolist") else row.postags

            upostags = getattr(row, "upostags", None)
            if upostags is not None: upostags = upostags.tolist() if hasattr(upostags, "tolist") else upostags

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
                if _deps_cache is None:
                    _deps_cache = build_dependency_maps(sentence_ids, word_ids, head_ids)
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
                        p_idx_map, _ = get_deps()
                        parent = p_idx_map[token_idx]

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
                        _, c_lookup_map = get_deps()
                        children = c_lookup_map[token_idx]
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
            # --- NOWOŚĆ: Wyciąganie "Kotwicy" do szybkiego przeskakiwania ---
            # Sprawdzamy, czy pierwszy segment zapytania wymaga konkretnego słowa (orth) lub lematu (base)
            anchor_type = None
            anchor_values = set()

            if token_query_conditions and len(token_query_conditions) > 0:
                first_cond_group = token_query_conditions[0]
                first_conds = first_cond_group if isinstance(first_cond_group, list) else [first_cond_group]

                # Szukamy, czy jest wymóg dokładnego dopasowania tekstowego
                for cond in first_conds:
                    if cond and len(cond) >= 5:
                        key, values, operator, is_nested, match_type = cond
                        if operator == "=" and match_type == "exact" and not is_nested:
                            if key in ("orth", "base"):
                                anchor_type = key
                                # Pobieramy wartości, upewniając się, że to stringi i na małe litery (jeśli ignorujesz wielkość)
                                anchor_values = set(v for v in values if isinstance(v, str))
                                break  # Znalazłem kotwicę, kończymy szukanie

            # Tworzymy szybką mapę pozycji dla tego rzędu, jeśli mamy kotwicę
            anchor_indices = []
            if anchor_type == "orth":
                # Używamy enumerate na liście tokens - to działa z prędkością C, a nie czystego Pythona
                anchor_indices = [idx for idx, t in enumerate(tokens) if t in anchor_values]
            elif anchor_type == "base":
                anchor_indices = [idx for idx, l in enumerate(lemmas) if l in anchor_values]

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
            full_left = tresc[max(0, start_ids[max(0, i - kontekst)]): start_ids[i]] if i > 0 else ""
            full_left = full_left[:-len(left_context)] if left_context else full_left

            full_right_limit = start_ids[min(len(start_ids) - 1, end_idx - 1 + kontekst)]
            full_right = tresc[end_ids[end_idx - 1] + 1: full_right_limit]
            full_right = full_right[len(right_context):] if right_context else full_right

            full_text_with_markers = [full_left, matched_text_actual, full_right]

            final_results.append((
                pub_date, context, full_text_with_markers,
                matched_text_actual, matched_lemmas,
                m_key, title, author, add_meta,
                left_context, right_context, row_idx, i, end_idx
            ))

        t_find_end = time.perf_counter()
        print(
            f"   -> [Wewnątrz find_lemma] Prefiltr indeksu: {t_prefilter - t_find_start:.4f}s | Pętla po tokenach: {t_find_end - t_prefilter:.4f}s")
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

        # ==========================================
        # --- NATYCHMIASTOWE PRZYWRACANIE KOLOKACJI ---
        # ==========================================
        paginator_colloc["data"] = list(state.colloc_data)
        paginator_colloc["current_page"][0] = 0
        update_table(paginator_colloc)

        # ==========================================
        # --- NATYCHMIASTOWE PRZYWRACANIE PROFILU ---
        # ==========================================
        global current_profile_dict, current_profile_target_lemma
        current_profile_dict = dict(state.current_profile_dict)
        current_profile_target_lemma = state.profile_target_lemma

        if current_profile_dict and state.profile_data:
            profile_rel_menu_btn.configure(state="normal")

            # Odtwarzamy logikę wybierania relacji po cofnięciu w historii
            display_to_key = {opt: opt.rsplit(" (", 1)[0] for opt in state.profile_rel_options}

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
            build_profile_tree_menu(state.profile_rel_options, display_to_key, on_rel_select_history)

            profile_rel_var.set(state.profile_selected_rel)

            paginator_profile["data"] = list(state.profile_data)
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


def resort_results(choice):
    global full_results_sorted, current_page, global_query, global_selected_corpus

    # Jeśli nie ma wyników do sortowania, nic nie rób
    if not full_results_sorted:
        return

    # Pomocnicze funkcje do sortowania po kontekście
    import string
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

    # Błyskawiczne sortowanie w miejscu (in-place)
    if choice == "Data publikacji":
        full_results_sorted.sort(key=lambda x: str(x[0]) if x[0] else "")
    elif choice == "Tytuł":
        full_results_sorted.sort(key=lambda x: str(x[6]) if x[6] else "")
    elif choice == "Autor":
        full_results_sorted.sort(key=lambda x: str(x[7]) if x[7] else "")
    elif choice == "Alfabetycznie":
        full_results_sorted.sort(key=lambda x: str(x[3]) if x[3] else "")
    elif choice == "Prawy kontekst":
        full_results_sorted.sort(key=lambda x: first_real_token(x[10]))
    elif choice == "Lewy kontekst":
        full_results_sorted.sort(key=lambda x: last_real_token(x[9]))

    # Resetujemy na pierwszą stronę i natychmiast odświeżamy tabelę
    current_page = 0
    display_page(global_query, global_selected_corpus)

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

    # Wstrzykujemy stan GUI jako drugi argument do funkcji
    def search_thread(search_token, ui_state):

        try:
            logging.info("Search started in thread: %s [token=%s]", threading.current_thread().name, search_token)
            t_start = time.perf_counter()
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
            t_parsed = time.perf_counter()
            # Przekazujemy selected_corpus zgodnie z nową definicją z Kroku 2!
            results = find_lemma_context(
                query,
                df,
                selected_corpus,
                left_context_size,
                right_context_size,
                warnings_list=warnings_list
            )
            t_matched = time.perf_counter()

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


            # --- SORTOWANIE W MIEJSCU (WYKONUJE SIĘ W TLE) ---
            if sort_option == "Data publikacji":
                results.sort(key=lambda x: str(x[0]) if x[0] else "")
            elif sort_option == "Tytuł":
                results.sort(key=lambda x: str(x[6]) if x[6] else "")
            elif sort_option == "Autor":
                results.sort(key=lambda x: str(x[7]) if x[7] else "")
            elif sort_option == "Alfabetycznie":
                results.sort(key=lambda x: str(x[3]) if x[3] else "")
            elif sort_option == "Prawy kontekst":
                results.sort(key=lambda x: first_real_token(x[10]))
            elif sort_option == "Lewy kontekst":
                results.sort(key=lambda x: last_real_token(x[9]))

            # Przypisujemy posortowaną (lub nie) listę do zmiennej używanej dalej
            results_sorted = results

            t_sorted = time.perf_counter()

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

                # ==========================================================
                # --- NOWOŚĆ: BŁYSKAWICZNE WYŚWIETLENIE PIERWSZEJ STRONY ---
                liczba_trafien = len(results_sorted)

                def show_first_results():
                    global current_page
                    current_page = 0
                    # Pokazujemy liczbę trafień od razu, dając znać, że statystyki jeszcze się liczą
                    label_results_count.configure(
                        text=f"Znaleziono trafień: {liczba_trafien:,} (Ładowanie statystyk...)".replace(',', ' '))
                    display_page(local_state.query, local_state.corpus)

                # Zlecamy odświeżenie GUI do głównego wątku NATYCHMIAST
                app.after(0, show_first_results)
                # ==========================================================

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
                        lemma: len(inverted_indexes[global_selected_corpus]["base"].get(lemma, set())) or 1
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
                            zscore = monthly_zscore_for_use[month_key].get(lemma) or 0.0
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
                t_stats = time.perf_counter()  # <--- CZAS PO STATYSTYKACH

                # ZAPIS DO LOGA I KONSOLI:
                t_stats = time.perf_counter()  # <--- CZAS PO STATYSTYKACH (wstaw tuż przed def update_gui(): )

                # ZAPIS DO LOGA I KONSOLI:
                profiling_msg = (
                    f"⏱ [PROFILING] Token: {search_token} | "
                    f"Walidacja: {t_parsed - t_start:.4f}s | "
                    f"Skanowanie (find_lemma): {t_matched - t_parsed:.4f}s | "
                    f"Sortowanie: {t_sorted - t_matched:.4f}s | "
                    f"Statystyki+Wykres: {t_stats - t_sorted:.4f}s || "
                    f"CAŁOŚĆ: {t_stats - t_start:.4f}s"
                )
                logging.info(profiling_msg)
                print(profiling_msg)
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

                app.after(100, push_nav_state)

                # Delegowanie pracy z UI do głównego wątku
                app.after(0, update_gui)


            else:
                # Brak wyników
                def update_no_results():
                    global search_status, full_results_sorted, current_page  # <--- DODANO current_page
                    full_results_sorted = []
                    search_status = 0
                    current_page = 0
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
        logging.info(f"Error loading image: {e}")


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
                pmw = v / (total_in_bin / 1e6)
                tf = v / total_in_bin
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


# --- Klasa do rozwijanych paneli opcji (Akordeon) ---
settings_cards = []


class SettingsCard(ctk.CTkFrame):
    def __init__(self, parent, title, expanded=False, expand_card=False):
        theme = THEMES[motyw.get()]
        super().__init__(parent, fg_color=theme["subframe_fg"], corner_radius=8, border_width=1, border_color="#3E3F42")

        self.expand_card = expand_card

        # anchor="nw" (North-West) gwarantuje dociśnięcie do lewej strony
        if self.expand_card:
            self.pack(fill="both", expand=True, pady=(0, 8), padx=0, anchor="nw")
        else:
            self.pack(fill="x", pady=(0, 8), padx=0, anchor="nw")

        self.title_text = title
        self.is_expanded = expanded

        self.btn_header = ctk.CTkButton(
            self,
            text=f"  {'▼' if expanded else '▶'}  {title}",  # Dodane spacje dla ładnego wcięcia
            command=self.toggle,
            fg_color="transparent",
            hover_color=theme.get("button_hover", "#404040"),
            anchor="w",
            font=("Verdana", 12, "bold"),
            text_color=theme["label_text"],
            height=32,
            corner_radius=8
        )
        self.btn_header.pack(fill="x", padx=2, pady=2)

        self.content = ctk.CTkFrame(self, fg_color="transparent")

        if self.is_expanded:
            self.pack_content()

        settings_cards.append(self)

    def pack_content(self):
        if self.expand_card:
            self.content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        else:
            self.content.pack(fill="x", padx=10, pady=(0, 10))

    def toggle(self):
        self.is_expanded = not self.is_expanded

        if self.is_expanded:
            self.btn_header.configure(text=f"  ▼  {self.title_text}")
            self.pack_content()
        else:
            self.btn_header.configure(text=f"  ▶  {self.title_text}")
            self.content.pack_forget()

    def update_theme(self, theme):
        self.configure(fg_color=theme["subframe_fg"])
        self.btn_header.configure(text_color=theme["label_text"], hover_color=theme.get("button_hover", "#404040"))

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
        base_corp = combo_corpus.get()
        d_start = entry_dstart.get().strip()
        d_end = entry_dend.get().strip()
        author = entry_author.get().strip()
        title = entry_title.get().strip()

        df = dataframes[base_corp]
        mask = pd.Series(True, index=df.index)

        # Błyskawiczne filtrowanie maską booleanową
        if "Data publikacji" in df.columns:
            if d_start: mask &= df["Data publikacji"].astype(str) >= d_start
            if d_end: mask &= df["Data publikacji"].astype(str) <= d_end
        if "Autor" in df.columns and author:
            mask &= df["Autor"].astype(str).str.contains(author, case=False, na=False)
        if "Tytuł" in df.columns and title:
            mask &= df["Tytuł"].astype(str).str.contains(title, case=False, na=False)

        sub_df = df[mask].copy()

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
                import pyarrow as pa
                import pyarrow.parquet as pq



                base_tf = Counter()
                orth_tf = Counter()
                total_tokens = 0
                monthly_token_counts = {}

                for row in sub_df.itertuples():
                    tokens = row.tokens.tolist() if hasattr(row.tokens, "tolist") else row.tokens
                    lemmas = row.lemmas.tolist() if hasattr(row.lemmas, "tolist") else row.lemmas

                    orth_tf.update(tokens)
                    base_tf.update(lemmas)
                    total_tokens += len(tokens)

                    if "Data publikacji" in sub_df.columns:
                        pub_date = str(getattr(row, "_4", getattr(row, "Data publikacji", "0000-00-00"))).strip()
                        parts = pub_date.split('-')
                        y = parts[0] if len(parts) > 0 else "0000"
                        m = parts[1] if len(parts) > 1 else "00"

                        monthly_token_counts.setdefault(y, {}).setdefault(m, 0)
                        monthly_token_counts[y][m] += len(tokens)

                metadata_export = {
                    "base_tf": dict(base_tf),
                    "orth_tf": dict(orth_tf),
                    "total_tokens": total_tokens,
                    "monthly_token_counts": monthly_token_counts
                }
                meta_json_bytes = json.dumps(metadata_export, ensure_ascii=False).encode('utf-8')

                table_pa = pa.Table.from_pandas(sub_df)
                existing_meta = table_pa.schema.metadata or {}
                merged_meta = {**existing_meta, b"korpus_meta": meta_json_bytes}
                table_pa = table_pa.replace_schema_metadata(merged_meta)

                pq.write_table(table_pa, file_path, compression='snappy')

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
            # 1. Pobranie unikalnych indeksów wierszy z wyników wyszukiwania (indeks 11 w krotce wyników)
            unique_row_indices = list(set([res[11] for res in full_results_sorted]))

            # 2. Wycięcie podkorpusu z oryginalnego DataFrame
            df = dataframes[global_selected_corpus]
            sub_df = df.loc[unique_row_indices].copy()

            # 3. Inicjalizacja nowych liczników dla metadanych
            import pyarrow as pa
            import pyarrow.parquet as pq

            base_tf = Counter()
            orth_tf = Counter()
            total_tokens = 0
            monthly_token_counts = {}

            # 4. Przeliczanie statystyk frekwencyjnych na nowo
            for row in sub_df.itertuples():
                tokens = row.tokens.tolist() if hasattr(row.tokens, "tolist") else row.tokens
                lemmas = row.lemmas.tolist() if hasattr(row.lemmas, "tolist") else row.lemmas

                orth_tf.update(tokens)
                base_tf.update(lemmas)
                total_tokens += len(tokens)

                # Zbieranie rozkładu w czasie (jeśli istnieje)
                if "Data publikacji" in sub_df.columns:
                    pub_date = str(getattr(row, "_4", getattr(row, "Data publikacji", "0000-00-00"))).strip()
                    parts = pub_date.split('-')
                    y = parts[0] if len(parts) > 0 else "0000"
                    m = parts[1] if len(parts) > 1 else "00"

                    if y not in monthly_token_counts:
                        monthly_token_counts[y] = {}
                    if m not in monthly_token_counts[y]:
                        monthly_token_counts[y][m] = 0
                    monthly_token_counts[y][m] += len(tokens)

            # 5. Przygotowanie metadanych JSON
            metadata_export = {
                "base_tf": dict(base_tf),
                "orth_tf": dict(orth_tf),
                "total_tokens": total_tokens,
                "monthly_token_counts": monthly_token_counts
            }
            meta_json_bytes = json.dumps(metadata_export, ensure_ascii=False).encode('utf-8')

            # 6. Zapisywanie pliku Parquet z dołączonymi metadanymi
            table_pa = pa.Table.from_pandas(sub_df)
            existing_meta = table_pa.schema.metadata or {}
            merged_meta = {**existing_meta, b"korpus_meta": meta_json_bytes}
            table_pa = table_pa.replace_schema_metadata(merged_meta)

            pq.write_table(table_pa, file_path, compression='snappy')

            def update_ui():
                loading_win.destroy()
                corpus_name = os.path.basename(file_path).replace(".parquet", "")
                messagebox.showinfo("Sukces", f"Zapisano podkorpus:\n{corpus_name}")

            app.after(0, update_ui)


        except Exception as e:
            logging.exception("Błąd tworzenia podkorpusu")
            app.after(0, lambda: loading_win.destroy())
            app.after(0, lambda msg=str(e): messagebox.showerror("Błąd", f"Nie udało się utworzyć podkorpusu.\n{msg}"))

    # Uruchomienie przeliczania i zapisu w tle
    threading.Thread(target=worker, daemon=True).start()


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

        # --- NAPRAWA: Zbroja przeciwko błędom Excela ---
        def clean_for_excel(df):
            df_cleaned = df.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', regex=True)
            for col in df_cleaned.select_dtypes(include=['object']).columns:
                df_cleaned[col] = df_cleaned[col].apply(
                    lambda x: f"'{x}" if isinstance(x, str) and str(x).startswith(('=', '-', '+', '@')) else x
                )
            return df_cleaned

        # Czyszczenie głównej tabeli
        df_export_slice = clean_for_excel(df_export_slice)

        # Export to Excel with multiple sheets
        if file_path.lower().endswith(".xlsx"):

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Sheet 1: main export
                df_export_slice.to_excel(writer, sheet_name="Wyniki wyszukiwania", index=False)

                # Sheet 2: paginator_fq['data']
                if 'data' in paginator_fq and paginator_fq['data']:
                    data_rows = paginator_fq['data']
                    headers = ["Nr", "Forma podstawowa (base)", "Liczba wystąpień", "Częstość względna",
                               "Rozproszenie (DF)", "Ogólne TF-IDF"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    clean_for_excel(df_data).to_excel(writer, sheet_name="Częstość lematów", index=False)

                # Sheet 3: paginator_token['data']
                if 'data' in paginator_token and paginator_token['data']:
                    data_rows = paginator_token['data']
                    headers = ["Nr", "Forma tekstowa (orth)", "Liczba wystąpień", "Częstość względna",
                               "Rozproszenie (DF)", "Ogólne TF-IDF"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    clean_for_excel(df_data).to_excel(writer, sheet_name="Częstość tokenów", index=False)

                # Sheet 4: paginator_month['data']
                if 'data' in paginator_month and paginator_month['data']:
                    data_rows = paginator_month['data']
                    headers = ["Rok", "Miesiąc", "Forma podstawowa", "Liczba wystąpień", "Częstość względna", "TF-IDF",
                               "Z-score"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    clean_for_excel(df_data).to_excel(writer, sheet_name="Częstość w czasie", index=False)

                # Sheet 5: Kolokacje
                if 'paginator_colloc' in globals() and 'data' in paginator_colloc and paginator_colloc['data']:
                    data_rows = paginator_colloc['data']
                    headers = ["Nr", "Kolokat", "f(nc)", "f(c)", "Log-Likelihood", "MI Score", "T-score",
                               "Log-Dice"]
                    df_data = pd.DataFrame(data_rows, columns=headers)
                    clean_for_excel(df_data).to_excel(writer, sheet_name="Kolokacje", index=False)

                # --- NOWOŚĆ: Sheet 6: Profil Kolokacyjny (zrzut CAŁOŚCI z pamięci) ---
                if 'current_profile_dict' in globals() and current_profile_dict:
                    all_merged_rows = []
                    # Wyciągnięcie absolutnie wszystkich kolokatów ze wszystkich relacji
                    for rel_name, rows in current_profile_dict.items():
                        all_merged_rows.extend(rows)

                    # Posortowanie globalne (najpierw najlepszy Log-Dice)
                    all_merged_rows.sort(key=lambda r: (r.log_dice, r.cooc_freq), reverse=True)

                    table_rows = []
                    for i, row_obj in enumerate(all_merged_rows):
                        display_colloc = row_obj.collocate
                        if getattr(row_obj, "collocate_upos", ""):
                            display_colloc = f"{display_colloc} [{row_obj.collocate_upos}]"

                        # Pakujemy wszystko do prostej listy, gotowej dla Excela
                        table_rows.append([
                            i + 1, display_colloc, row_obj.relation, row_obj.cooc_freq, row_obj.doc_freq,
                            row_obj.global_freq, row_obj.ll_score, row_obj.mi_score,
                            row_obj.t_score, row_obj.log_dice
                        ])

                    headers = ["Nr", "Kolokat", "Relacja składniowa", "Współwyst.", "Zasięg (Dok.)", "Freq. Glob.",
                               "Log-Likelihood", "MI Score", "T-score", "Log-Dice"]
                    df_data = pd.DataFrame(table_rows, columns=headers)
                    clean_for_excel(df_data).to_excel(writer, sheet_name="Profil kolokacyjny", index=False)

        else:
            # fallback CSV export
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
        'kontekst': kontekst,
        'min_tokens_threshold': min_tokens_threshold

    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Settings window
def settings_window():
    global settings_popup, fontsize, font_family, plotting, kontekst, min_tokens_threshold
    theme = THEMES[motyw.get()]

    # Callbacks
    def restore_defaults():
        global settings_popup, fontsize, font_family, kontekst, plotting, min_tokens_threshold
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
        apply_theme()
        save_config()
        settings_popup.destroy()
        settings_popup = None

    def on_save():
        global settings_popup, fontsize, font_family, kontekst, min_tokens_threshold
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
    settings_popup.geometry('420x750')
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


def show_corpus_info():
    global dataframes, inverted_indexes, corpus_var

    selected = corpus_var.get()
    if not selected or selected not in dataframes:
        messagebox.showinfo("Brak danych", "Najpierw załaduj lub wybierz korpus.")
        return

    df = dataframes[selected]
    inv_idx = inverted_indexes[selected]

    # --- ZBIERANIE STATYSTYK ---
    total_docs = len(df)
    total_tokens = inv_idx.get("total_tokens", 0)
    unique_lemmas = len(inv_idx.get("base", {}))
    unique_orths = len(inv_idx.get("orth", {}))

    # --- ZAKRES CZASOWY I TOKENY NA MIESIĄC ---
    date_range = "Brak danych o dacie"
    monthly_counts = inv_idx.get("monthly_token_counts", {})
    monthly_stats_str = ""

    if monthly_counts:
        dates = []
        monthly_stats_list = []  # Pomocnicza lista do zbudowania ładnego stringa

        # Sortujemy lata
        for y in sorted(monthly_counts.keys(), key=int):
            # Sortujemy miesiące w obrębie roku
            for m in sorted(monthly_counts[y].keys(), key=int):
                dates.append((int(y), int(m)))
                count = monthly_counts[y][m]
                monthly_stats_list.append(f"  • {int(m):02d}.{y}: {count:,} tokenów")

        if dates:
            min_d = min(dates)
            max_d = max(dates)
            date_range = f"{min_d[1]:02d}.{min_d[0]} - {max_d[1]:02d}.{max_d[0]}"

        if monthly_stats_list:
            monthly_stats_str = "LICZBA TOKENÓW NA MIESIĄC:\n" + "\n".join(monthly_stats_list)

    # --- DOSTĘPNE METADANE ---
    exclude_cols = {
        "Oryginalna_nazwa_pliku", "Treść", "token_counts", "tokens", "lemmas",
        "deprels", "postags", "full_postags", "word_ids", "sentence_ids",
        "head_ids", "start_ids", "end_ids", "ners", "upostags", "corefs"
    }
    meta_cols = [c for c in df.columns if c not in exclude_cols]
    meta_str = "\n  • ".join(meta_cols) if meta_cols else "  Brak dodatkowych metadanych"

    # --- TWORZENIE OKIENKA UI ---
    info_win = ctk.CTkToplevel(app)
    info_win.title(f"Informacje o korpusie: {selected}")
    info_win.geometry("500x550")  # Lekko powiększyłem okno
    info_win.transient(app)
    info_win.grab_set()

    theme = THEMES[motyw.get()]
    info_win.configure(fg_color=theme["app_bg"])

    frame = ctk.CTkFrame(info_win, fg_color=theme["subframe_fg"], corner_radius=12)
    frame.pack(fill="both", expand=True, padx=15, pady=15)

    # Budujemy cały tekst i podmieniamy przecinki z {count:,} na spacje
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

    # --- ZMIANA: Zamiast Label używamy Textboxa, żeby był suwak (scrollbar) ---
    textbox = ctk.CTkTextbox(
        frame,
        font=("Verdana", 13),
        text_color=theme["label_text"],
        fg_color="transparent",
        wrap="word"
    )
    textbox.insert("1.0", info_text)
    textbox.configure(state="disabled")  # Blokujemy edycję
    textbox.pack(padx=10, pady=10, fill="both", expand=True)

    btn_close = ctk.CTkButton(
        frame, text="Zamknij", command=info_win.destroy,
        font=("Verdana", 12, "bold"), fg_color=theme["button_fg"],
        hover_color=theme["button_hover"], text_color=theme["button_text"]
    )
    btn_close.pack(pady=10)

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

    load_semantic_neighbors()

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
    lemma_frame.grid_remove()
    orth_frame.grid_remove()
    month_frame.grid_remove()
    colloc_frame.grid_remove()
    if 'profile_frame' in globals():
        profile_frame.grid_remove()

    if choice == "Formy podstawowe (base)":
        lemma_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Formy ortograficzne (orth)":
        orth_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Częstość w czasie":
        month_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Kolokacje":
        colloc_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    elif choice == "Profil kolokacyjny":
        profile_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # --- DODANA LOGIKA: Dynamiczne opcje węzła ---
        if full_results_sorted:
            # Bierzemy pierwsze z brzegu trafienie, by sprawdzić jego długość
            res = full_results_sorted[0]
            start_i = res[12]
            end_i = res[13]
            match_len = end_i - start_i

            # Pobieramy podgląd lematów (np. "wielki zamek")
            lemmas = str(res[4]).split()

            options = []
            for i in range(match_len):
                hint = lemmas[i] if i < len(lemmas) else "?"
                options.append(f"Token {i + 1} ({hint})")

            if not options: options = ["Token 1"]

            # Wrzucamy opcje do menu. Jeśli są >1, podświetlamy na pomarańczowo, żeby zwrócić uwagę!
            profile_node_menu.configure(values=options)
            if match_len > 1:
                profile_node_menu.configure(fg_color="#D9A04F", button_color="#D9A04F")
            else:
                profile_node_menu.configure(fg_color="#4B6CB7", button_color="#4B6CB7")

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

    # 1. Pobieranie ustawień GUI (musi zostać w głównym wątku!)
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

    # Zablokuj przycisk, żeby ktoś nie kliknął 5 razy podczas liczenia
    btn_calc_colloc.configure(state="disabled", text="Liczenie...")

    def worker():
        try:

            def check_match(idx, row_upostags, row_postags, row_full_postags):
                # 1. Najpierw sprawdzamy ogólne POS/UPOS
                u_match = (upos_filter == "Wszystkie") if row_upostags is None else (
                            upos_filter == "Wszystkie" or row_upostags[idx] == upos_filter)
                p_match = (pos_filter == "Wszystkie" or row_postags[idx] == pos_filter)

                if not (u_match and p_match):
                    return False

                # 2. Następnie weryfikujemy dokładne cechy z full_postags, jeśli filtry są włączone
                if active_feat_filters and row_full_postags is not None:
                    full_tag = str(row_full_postags[idx])
                    tag_parts = full_tag.split(":")
                    tag_pos = tag_parts[0] if tag_parts else ""
                    tag_feats = tag_parts[1:] if len(tag_parts) > 1 else []

                    mapping = FEAT_MAPPING.get(tag_pos, {})
                    for feat, req_val in active_feat_filters.items():
                        if feat in mapping:
                            f_idx = mapping[feat]
                            # Sprawdzamy, czy cecha istnieje i pasuje
                            if f_idx < len(tag_feats) and tag_feats[f_idx] == req_val:
                                continue
                        return False  # Jeśli nie pasuje, odrzuć kolokat
                return True

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

                        is_punct = (upostags[i] == "PUNCT") if upostags is not None else (
                                    tokens[i] in string.punctuation)
                        total_actual_slots += 1

                        if not is_punct:
                            # Zastępujemy u_match i p_match nową funkcją weryfikującą wszystkie cechy
                            full_postags = getattr(row_data, "full_postags", None)
                            if check_match(i, upostags, postags, full_postags):
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

                                    if deprel_filter == "Wszystkie" or deprels[i] == deprel_filter:  # lub deprels[j]
                                        is_punct = (upostags[j] == "PUNCT") if upostags is not None else (
                                                tokens[j] in string.punctuation)
                                        if not is_punct:
                                            total_actual_slots += 1

                                            # Używamy nowej funkcji do weryfikacji:
                                            if check_match(j, upostags, postags,
                                                           getattr(row_data, "full_postags", None)):
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
            total_tokens = inv_idx_data.get('total_tokens', 1)
            if total_tokens == 0:
                total_tokens = 1

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

    from word_profile import PROFILE_GRAMMARS
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

    print(f"⏱ [PROFIL KOLOKACYJNY] Odpalam zapytanie strukturalne: {new_query}")
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
        print(f"⏱ [KOLOKACJE] Zbudowanie zapytania zajęło: {prep_time:.6f}s")
        print(f"   -> Odpalam zapytanie: {new_query}")
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
    setup_win.geometry("450x450")  # POWIĘKSZONE OKNO na nowe opcje
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
        diversity_val = round(diversity_var.get(), 2)  # Pobieramy wartość z suwaka

        setup_win.destroy()
        # Przekazujemy parametr dalej
        _run_modeling_process(nr_topics_val, use_stopwords, diversity_val)

    ctk.CTkButton(setup_win, text="Rozpocznij analizę", command=start_process).pack(pady=20)

    def _run_modeling_process(nr_topics_val, use_stopwords, diversity_val):
        # 3. Tworzymy okienko ładowania
        loading_win = ctk.CTkToplevel(app)
        loading_win.title("Modelowanie Tematyczne")
        loading_win.geometry("600x450")
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

        class TextRedirector:
            def __init__(self, widget):
                self.widget = widget

            def write(self, text):
                app.after(0, self._append_text, text)

            def _append_text(self, text):
                self.widget.configure(state="normal")
                self.widget.insert("end", text)
                self.widget.see("end")
                self.widget.configure(state="disabled")

            def flush(self):
                pass

        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TextRedirector(log_box)
        sys.stderr = TextRedirector(log_box)

        def restore_stdout(event=None):
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        loading_win.bind("<Destroy>", restore_stdout)

        def worker():
            try:
                print("Inicjalizacja TopicEngine...")
                engine = TopicEngine(parquet_path)

                app.after(0, lambda: lbl_status.configure(text="Wczytywanie i filtrowanie tekstów..."))
                if not engine.load_data():
                    raise Exception("Plik korpusu nie posiada kolumny 'Treść' lub jest pusty.")

                app.after(0, lambda: lbl_status.configure(text="Trenowanie modelu (może to potrwać)..."))
                print("Rozpoczęto trenowanie modelu BERTopic...")

                # --- NOWE: Przekazujemy nr_topics i wymuszamy nadpisanie ---
                if not engine.train_model(nr_topics=nr_topics_val, force_retrain=True, use_stopwords=use_stopwords, diversity=diversity_val):
                    raise Exception("Błąd podczas treningu modelu.")

                freq_df = engine.model.get_topic_freq()
                valid_topics = freq_df[freq_df['Topic'] != -1]

                if valid_topics.empty:
                    print("OSTRZEŻENIE: Zbyt mało danych. Model sklasyfikował wszystko jako szum (-1).")
                    logging.warning("Zbyt mało danych. Model sklasyfikował wszystko jako szum (-1).")
                    app.after(0, loading_win.destroy)
                    app.after(0, lambda: messagebox.showwarning("Brak tematów",
                                                                "Zbyt mało danych. Model nie odnalazł powiązań między dokumentami (tylko szum)."))
                    return

                print("Generowanie wizualizacji...")
                fig_map = engine.visualize_topic_map()
                tot = engine.calculate_topics_over_time()
                fig_time = engine.visualize_dynamic_topics(tot, top_n_topics=15) if tot is not None else None
                fig_words = engine.visualize_word_scores(top_n_topics=15)

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
                    app.after(0, loading_win.destroy)
                    raise Exception(f"Błąd zapisu raportu. Sprawdź czy masz uprawnienia do folderu.\n{write_err}")

                app.after(0, loading_win.destroy)
                launch_webview(html_path)

            except Exception as e:
                err_msg = str(e)
                print(f"BŁĄD KRYTYCZNY: {err_msg}")
                logging.exception(f"Błąd podczas generowania modelu BERTopic: {err_msg}")
                app.after(0, loading_win.destroy)
                app.after(0, lambda msg=err_msg: messagebox.showerror("Błąd BERTopic", f"Szczegóły:\n{msg}"))

        import threading
        threading.Thread(target=worker, daemon=True).start()


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
    command=resort_results,       # <--- DODANE
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
    command=lambda: push_nav_state(),
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
card_method = SettingsCard(colloc_options_frame, "Metoda wyszukiwania", expanded=True)
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
card_filters = SettingsCard(colloc_options_frame, "Filtry lingwistyczne", expanded=False)
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
card_stats = SettingsCard(colloc_options_frame, "Parametry statystyczne", expanded=False)
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
card_profile = SettingsCard(profile_options_frame, "Opcje profilu", expanded=True)
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
card_profile_rels = SettingsCard(profile_options_frame, "Wybór relacji", expanded=True)
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

            adjusted_results = []
            for res in full_results_sorted:
                res_list = list(res)
                res_list[12] = res_list[12] + node_offset
                adjusted_results.append(tuple(res_list))

            # Wywołanie funkcji z przekazaniem flagi ignore_case
            mwe_val = profile_mwe_var.get()
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
card_type = SettingsCard(plot_options_frame, "Typ i zapis wykresu", expanded=True)
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
card_date = SettingsCard(plot_options_frame, "Filtrowanie czasowe", expanded=False)
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
card_scale = SettingsCard(plot_options_frame, "Skalowanie osi Y", expanded=False)
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
card_checkboxes = SettingsCard(plot_options_frame, "Zaznaczone elementy", expanded=True, expand_card=True)
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


if __name__ == "__main__":
    main()