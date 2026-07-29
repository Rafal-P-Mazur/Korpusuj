"""Widgets and data helpers for constructing CQL queries in the desktop interface.

Themes, feature dictionaries, dependency labels and NER labels are supplied explicitly by the application.
"""

import logging
import re
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk


NER_PREFIXES = []
NER_TYPES = []


def configure_query_builder_specs(ner_prefixes=None, ner_types=None):
    """
    Configure module-local specs used by ConditionRow.

    The source of truth remains in engine.py for now. QueryBuilderWindow
    calls this during initialization.
    """
    global NER_PREFIXES, NER_TYPES

    if ner_prefixes is not None:
        NER_PREFIXES = list(ner_prefixes)

    if ner_types is not None:
        NER_TYPES = list(ner_types)


class RegexHelperWindow(ctk.CTkToplevel):
    def __init__(self, parent, target_entry, theme, ner_prefixes=None, ner_types=None):
        configure_query_builder_specs(
            ner_prefixes=ner_prefixes,
            ner_types=ner_types,
        )
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
            "Okno ortograficzne (window_orth)": "window_orth",
            # "SRL: rola aktualnego elementu (srl_role)": "srl_role",
            # "SRL: predykat ramy (srl_pred)": "srl_pred",
            #
            # "SRL: ARG0 zawiera lemat (srl_arg0)": "srl_arg0",
            # "SRL: ARG1 zawiera lemat (srl_arg1)": "srl_arg1",
            # "SRL: ARG2 zawiera lemat (srl_arg2)": "srl_arg2",
            #
            # "SRL: ARG0 tekst (srl_arg0_text)": "srl_arg0_text",
            # "SRL: ARG1 tekst (srl_arg1_text)": "srl_arg1_text",
            #
            # "SRL: ARG0 głowa lemat (srl_arg0_head_base)": "srl_arg0_head_base",
            # "SRL: ARG1 głowa lemat (srl_arg1_head_base)": "srl_arg1_head_base",
            #
            # "SRL: czas TMP (srl_tmp)": "srl_tmp",
            # "SRL: miejsce LOC (srl_loc)": "srl_loc",
            # "SRL: sposób MNR (srl_mnr)": "srl_mnr",
            # "SRL: przyczyna CAU (srl_cau)": "srl_cau",
            # "SRL: cel PRP (srl_prp)": "srl_prp",
            # "SRL: zakres EXT (srl_ext)": "srl_ext",

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
    def __init__(self, parent, target_textbox, theme, ner_prefixes=None, ner_types=None):
        configure_query_builder_specs(
            ner_prefixes=ner_prefixes,
            ner_types=ner_types,
        )
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
