"""Semantic analysis services for artifact loading, neighbor indexes, hubness, frame induction, graph expansion and report generation."""
# D20_FASTTEXT_SUBWORD_CONTROLS
# D21_FASTTEXT_DEFAULT_5_6

import os
import sys
import math
import zipfile
import logging
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox


try:
    from korpusuj.search.diagnostics import korpusuj_diagnostics_enabled_145c1
except Exception:
    def korpusuj_diagnostics_enabled_145c1(config_obj=None):
        return False

def notify_status(msg):
    """Forward a semantic-processing status message to the optional application status hook."""
    try:
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and hasattr(main_mod, "update_status"):
            main_mod.update_status(msg)
    except Exception:
        pass


def _default_launcher_script() -> str:
    """Resolve the application launcher script used by semantic subprocesses.
    
    In source mode the launcher is resolved from the project root; frozen execution uses the packaged application location.
    """
    if getattr(sys, "frozen", False):
        return sys.executable

    try:
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / "engine.py")
    except Exception:
        return str(Path.cwd() / "engine.py")


class SemanticEngine:
    # D19_RANKING_CONTRACT
    # D18_NO_WSD
    # D16_MUTUAL_5NN_EXPLORATION
    """Klasa zarządzająca logiką, ładowaniem i pamięcią sieci semantycznej."""

    def __init__(self, launcher_script=None):
        self.df_neighbors = None
        self.index = None
        self.knn_set = None

        self.vectors = None

        self.hubness_index = {}  # Dodane: cache na preobliczoną hubowość\n        self.launcher_script = launcher_script or _default_launcher_script()

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
        setup_win.geometry("500x560")
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
        min_n_var = add_param("FastText min_n:", "5")
        max_n_var = add_param("FastText max_n:", "6")
        ctk.CTkLabel(
            frame,
            text="min_n i max_n sterują długością znakowych n-gramów tylko w FastText. Word2Vec ignoruje te pola.",
            wraplength=430,
            justify="left",
            text_color="gray",
            font=("Verdana", 9),
        ).pack(fill="x", pady=(0, 5))
        min_count_var = add_param("Min. wystąpień (--min-count):", "10")
        epochs_var = add_param("Epoki (--epochs):", "20")
        window_var = add_param("Rozmiar okna (--window):", "15")
        vocab_var = add_param("Słownik (--neighbors...):", "10000")
        precomp_var = add_param("Zapisz top N (--precompute...):", "200")

        def on_start():
            try:
                min_n = int(min_n_var.get())
                max_n = int(max_n_var.get())
                if min_n < 0 or max_n < 0:
                    raise ValueError("wartości nie mogą być ujemne")
                if min_n > max_n:
                    raise ValueError("min_n musi być mniejsze lub równe max_n")
            except ValueError as exc:
                messagebox.showerror("Niepoprawne subwordy FastText", str(exc), parent=setup_win)
                return

            params = {
                "algo": algo_var.get(),
                "fasttext_min_n": min_n,
                "fasttext_max_n": max_n,
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
                cmd = [sys.executable, getattr(self, "launcher_script", _default_launcher_script()), "--run-semantic-trainer"]
            cmd.extend([
                "--parquet", corpus_path_safe,
                "--algo", params["algo"],
                "--fasttext-min-n", str(params["fasttext_min_n"]),
                "--fasttext-max-n", str(params["fasttext_max_n"]),
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
                korpusuj_diagnostics_enabled_145c1() and logging.info("[DIAG semantic.report] returncode=%s", process.returncode)
                try:
                    win.after(0, lambda rc=process.returncode: (
                        log_box.insert("end", f"\n[DIAG] Proces raportu zakończony kodem: {rc}\n"),
                        log_box.see("end")
                    ))
                except Exception:
                    pass

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

        safe_lemma = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(lemma).strip())
        corpus_base = Path(current_corpus_path).with_suffix("")
        report_dir = corpus_base.parent / f"{corpus_base.name}.semantic_reports" / safe_lemma

        # SEMANTIC_REPORT_DIAG_007B:
        # Tworzymy katalog raportu przed uruchomieniem subprocessu.
        # Jeśli katalog nie powstaje, problem jest jeszcze przed tą metodą.
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            korpusuj_diagnostics_enabled_145c1() and logging.info("[DIAG semantic.report] ensured report_dir=%s", report_dir)
        except Exception as e:
            korpusuj_diagnostics_enabled_145c1() and logging.error("[DIAG semantic.report] cannot create report_dir=%s reason=%r", report_dir, e, exc_info=True)
            try:
                messagebox.showerror("Błąd raportu", f"Nie udało się utworzyć katalogu raportu:\n{report_dir}\n\n{e}")
            except Exception:
                pass
            return

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

        # SEMANTIC_REPORT_DIAG_007B: widoczny start diagnostyki w oknie logów.
        try:
            log_box.insert("end", "[DIAG] Przygotowuję raport semantyczny...\n")
            log_box.insert("end", f"[DIAG] Lemma: {str(lemma).strip()}\n")
            log_box.insert("end", f"[DIAG] Korpus: {corpus_path_safe}\n")
            log_box.insert("end", f"[DIAG] Katalog raportu: {report_dir_safe}\n")
            log_box.insert("end", f"[DIAG] Launcher script: {getattr(self, 'launcher_script', None)}\n\n")
            log_box.see("end")
            win.update_idletasks()
        except Exception:
            pass

        korpusuj_diagnostics_enabled_145c1() and logging.info(
            "[DIAG semantic.report] start lemma=%r corpus=%r output_dir=%r launcher=%r",
            str(lemma).strip(), corpus_path_safe, report_dir_safe, getattr(self, 'launcher_script', None)
        )

        def run_report_in_background():
            if getattr(sys, "frozen", False):
                cmd = [sys.executable, "--run-semantic-report"]
            else:
                cmd = [sys.executable, getattr(self, "launcher_script", _default_launcher_script()), "--run-semantic-report"]


            cmd.extend([
                "--artifacts", corpus_path_safe,
                "--lemma", str(lemma).strip(),
                "--output-dir", report_dir_safe,

                # D14d: zakres raportu pozostaje pełny; nie przekazujemy historycznych limitów prezentacyjnych.
            ])

            # SEMANTIC_REPORT_DIAG_007B: log komendy przed Popen.
            try:
                cmd_preview = " ".join(str(x) for x in cmd)
                korpusuj_diagnostics_enabled_145c1() and logging.info("[DIAG semantic.report] cmd=%s", cmd_preview)
                korpusuj_diagnostics_enabled_145c1() and logging.info(
                    "[DIAG semantic.report] launcher_exists=%s cwd=%s",
                    os.path.exists(str(cmd[1])) if len(cmd) > 1 else None,
                    os.getcwd(),
                )
                win.after(0, lambda c=cmd_preview: (
                    log_box.insert("end", "[DIAG] Uruchamiam proces raportu...\n"),
                    log_box.insert("end", f"[DIAG] CMD: {c}\n\n"),
                    log_box.see("end")
                ))
            except Exception as e:
                korpusuj_diagnostics_enabled_145c1() and logging.warning("[DIAG semantic.report] cannot log cmd: %r", e, exc_info=True)


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
                              lambda: (
                                  log_box.insert("end", f"\nWystąpił błąd raportu (kod: {process.returncode})\n"),
                                  log_box.insert("end", "Sprawdź korpusuj.log oraz powyższą komendę CMD.\n"),
                                  log_box.see("end")
                              ))
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


        if loaded_df is not None:
            self.index = {}
            self.knn_set = {}
            has_freq = 'neighbor_freq' in loaded_df.columns
            loaded_df = loaded_df.sort_values(by=['lemma', 'similarity'], ascending=[True, False])
            MUTUAL_M = 5  # D16: mutual 5-NN

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
            for n_word, _n_score, _ in neighbors:
                # D16: wszystkie zapisane listy, bez progu podobieństwa.
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

    # ==========================================
    # NOWE METODY DO OBSŁUGI SENSÓW (WSD)
    # ==========================================






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

    def get_representation_vector(self, lemma):
        """Zwraca podstawowy wektor lematu."""
        actual = self._resolve_key(lemma, self.vectors)
        return self.vectors.get(actual) if actual and self.vectors else None


    def _cos(self, u, v):
        if u is None or v is None: return -1.0
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))






    def get_contextual_neighbors(self, center_lemma, top_n=25, root_lemma=None, parent_lemma=None,
                                 base_weight=0.50, parent_weight=0.30,
                                 root_weight=0.20, domain_lambda=0.20):
        matched_center = self._resolve_key(center_lemma, self.index)
        if not matched_center: return center_lemma, []

        raw = self.index[matched_center]
        root_vec = self.get_representation_vector(root_lemma) if root_lemma else None
        matched_parent = self._resolve_key(parent_lemma, self.index) if parent_lemma else None
        parent_vec = self.get_representation_vector(matched_parent) if matched_parent else None
        initial_context = matched_parent is None
        ranking_context = "initial" if initial_context else "path"

        out = []

        for u, base_sim, freq in raw:
            actual_u = self._resolve_key(u, self.vectors)
            if not actual_u:
                continue

            # Bezpośredni powrót A -> B -> A nie dodaje nowej informacji do grafu.
            candidate_index_key = self._resolve_key(u, self.index)
            if matched_parent is not None and candidate_index_key == matched_parent:
                continue

            child_sid = None
            child_vec = self.get_representation_vector(actual_u)
            sense_score = 0.0
            s_parent = self._cos(parent_vec, child_vec) if parent_vec is not None else 0.0
            s_root = self._cos(root_vec, child_vec) if root_vec is not None else 0.0

            # Pierwszy krok wykorzystuje tylko podobieństwo do rozwijanego lematu.
            # Kolejne kroki zachowują odrębną informację rodzica i korzenia.
            if initial_context:
                contextual_score = float(base_sim)
            else:
                contextual_score = (
                    base_weight * float(base_sim)
                    + parent_weight * s_parent
                    + root_weight * s_root
                )

            # 2. Pobranie hubności dla słowa 'u' z indeksu
            actual_candidate = self._resolve_key(u, self.hubness_index) or u
            hubness_penalty = self.hubness_index.get(actual_candidate, 0.0)

            # 3. Nałożenie kary na ostateczny wynik (z ujemnym score na selektywnych listach)
            final_score = contextual_score - (domain_lambda * hubness_penalty)

            generality_penalty = float(domain_lambda * hubness_penalty)
            out.append({
                "lemma": u,
                "base_similarity": float(base_sim),
                "parent_similarity": float(s_parent),
                "root_similarity": float(s_root),
                "contextual_similarity": float(contextual_score),
                "ranking_context": ranking_context,
                "generality": float(hubness_penalty),
                "generality_penalty": generality_penalty,
                "selection_score": float(final_score),
                "semantic_edge_weight": float(contextual_score),
                "contextual_score": float(contextual_score),
                "score": float(final_score),
                "graph_weight": float(contextual_score),
                "freq": int(freq),
                "sense_id": child_sid,
                "sense_score": float(sense_score)
            })

        out.sort(key=lambda x: x["score"], reverse=True)
        # ZMIANA: Zwracamy wszystkie wyliczone i posortowane węzły (limit np. do 150)
        return matched_center, out[:150]
