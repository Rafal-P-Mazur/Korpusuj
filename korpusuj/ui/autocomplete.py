"""CQL autocomplete widgets and suggestion logic for the search interface.

Shared feature mappings are supplied explicitly to the autocomplete component.
"""

import logging
import re
import tkinter as tk

try:
    import customtkinter as ctk
except Exception:
    ctk = None


class CQLAutocomplete:
    def __init__(self, textbox, feat_mapping=None):
        self.textbox = textbox
        self.feat_mapping = feat_mapping or {}
        self.popup = None
        self.listbox = None
        self.current_prefix = ""
        self.current_mode = None

        # Linia, przy której aktualnie pokazano popup.
        # Jeśli użytkownik kliknie inną linię textboxa, traktujemy to jako odkliknięcie.
        self.popup_anchor_line = None

        # Flaga używana po kliknięciu w inną linię:
        # globalny ButtonPress chowa popup, a ButtonRelease textboxa nie powinien go od razu wznowić.
        self._suppress_next_textbox_click = False

        # --- ZBIORY DANYCH WG TWOICH KEYWORDS ---
        self.attributes = [
            "orth=", "orth!=", "base=", "base!=", "pos=", "pos!=", "upos=", "upos!=",
            "ner=", "ner!=", "head=", "head!=", "coref=", "coref!=", "dependent=", "dependent!=",
            "deprel=", "deprel!=", "number=", "number!=", "window_base=", "window_base!=",
            "window_orth=", "window_orth!=", "gender=", "gender!=", "degree=", "degree!=",
            "case=", "case!=", "person=", "person!=", "accentability=", "accentability!=",
            "post-prepositionality=", "post-prepositionality!=", "accommodability=", "accommodability!=",
            "aspect=", "aspect!=", "vocalicity=", "vocalicity!=", "agglutination=", "agglutination!=",
            "negation=", "negation!=", "children.group=", "||",
            #"srl_role=", "srl_role!="
        ]

        # --- GENEROWANIE ATRYBUTÓW SRL (srl_arg0=, srl_arg0_head= itd.) ---
        # srl_bases = ["pred", "arg0", "arg1", "arg2", "tmp", "loc", "mnr", "cau", "prp", "ext"]
        # for base in srl_bases:
        #     self.attributes.extend([
        #         f"srl_{base}=", f"srl_{base}!=",
        #         f"srl_{base}_head=", f"srl_{base}_head!="
        #     ])

        self.global_tags = [
            "data>", "data<", "data=", "data!=", "data<=", "data>=", "autor=", "autor!=",
            "metadane:", "tytuł=", "tytuł!=", "frequency_base", "frequency_orth",
            "top=", "min=", "max=", "s>"
        ]

        self.pos_tags = [
            "subst", "depr", "adj", "adja", "adjp", "adjc", "conj", "ppron12",
            "ppron3", "siebie", "num", "numcol", "fin", "bedzie", "aglt", "praet",
            "impt", "imps", "inf", "pcon", "pant", "ger", "pact", "ppas", "winien",
            "adv", "prep", "comp", "qub", "interj", "brev", "burk", "interp", "xxx", "ign"
        ]

        self.upos_tags = [
            "NOUN", "PROPN", "ADJ", "VERB", "ADV", "PRON", "DET", "ADP",
            "NUM", "CCONJ", "SCONJ", "PART", "INTJ", "PUNCT", "SYM", "X"
        ]

        # --- SŁOWNIKI SRL ROLI SEMANTYCZNYCH Z TŁUMACZENIAMI ---
        self.srl_role_tags = [
            "pred - predykat (orzeczenie)",
            "arg0 - argument 0 (agens)",
            "arg1 - argument 1 (patiens)",
            "arg2 - argument 2 (beneficjent / cel / instrument)",
            "tmp - określnik czasu (temporal)",
            "loc - określnik miejsca (locative)",
            "mnr - określnik sposobu (manner)",
            "cau - określnik przyczyny (causal)",
            "prp - określnik celu (purpose)",
            "ext - określnik miary/stopnia (extent)"
        ]

        self.deprel_tree_dict = {
            "Wszystkie": [],
            "root - głowa drzewa": [],
            "nsubj - podmiot nominalny": ["nsubj:pass - podmiot nominalny (strona bierna)"],
            "csubj - podmiot zdaniowy": ["csubj:pass - podmiot zdaniowy (strona bierna)"],
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

        self.deprel_tags = []
        for key, values in self.deprel_tree_dict.items():
            if key == "Wszystkie":
                continue
            self.deprel_tags.append(key)
            self.deprel_tags.extend(values)

        self.morph_dicts = {
            "case": [
                "nom (mianownik)", "gen (dopełniacz)", "dat (celownik)",
                "acc (biernik)", "inst (narzędnik)", "loc (miejscownik)", "voc (wołacz)"
            ],
            "number": [
                "sg (pojedyncza)", "pl (mnoga)"
            ],
            "gender": [
                "m1 (męskoosobowy)", "m2 (męskozwierzęcy)", "m3 (męskorzeczowy)",
                "f (żeński)", "n (nijaki)"
            ],
            "degree": [
                "pos (równy)", "com (wyższy)", "sup (najwyższy)"
            ],
            "person": [
                "pri (pierwsza)", "sec (druga)", "ter (trzecia)"
            ],
            "aspect": [
                "imperf (niedokonany)", "perf (dokonany)"
            ],
            "negation": [
                "aff (niezanegowana - pisanie, czytanego)",
                "neg (zanegowana - niepisanie, nieczytanego)"
            ],
            "accentability": [
                "akc (akcentowana - jego, niego, tobie)",
                "nakc (nieakcentowana - go, -ń, ci)"
            ],
            "post-prepositionality": [
                "praep (poprzyimkowa - niego, -ń)",
                "npraep (niepoprzyimkowa - jego, go)"
            ],
            "accommodability": [
                "congr (uzgadniająca - dwaj, pięcioma)",
                "rec (rządząca - dwóch, dwu, pięciorgiem)"
            ],
            "vocalicity": [
                "wok (wokaliczna - -em)",
                "nwok (niewokaliczna - -m)"
            ],
            "agglutination": [
                "agl (aglutynacyjna - niósł)",
                "nagl (nieaglutynacyjna - niosł-)"
            ],
            "fullstoppedness": [
                "pun (z następującą kropką - tzn)",
                "npun (bez kropki - wg)"
            ]
        }

        tb = self.textbox._textbox

        tb.bind("<KeyRelease>", self.handle_keypress)
        tb.bind("<FocusOut>", self._on_focus_out)
        tb.bind("<ButtonRelease-1>", self._on_textbox_click, add="+")
        tb.bind("<Up>", self.navigate_up)
        tb.bind("<Down>", self.navigate_down)
        tb.bind("<Return>", self.insert_selection)

        # KLUCZOWE:
        # bind_all łapie kliknięcia także w inne ramki, panele, przyciski itd.
        # Bez tego FocusOut w CustomTkinterze bywa niewystarczający.
        tb.bind_all("<ButtonPress-1>", self._on_global_click, add="+")

    # =========================================================
    # OBSŁUGA ODKLIKNIĘCIA / FOKUSU
    # =========================================================

    def _is_widget_or_child(self, widget, parent):
        """Sprawdza, czy widget jest parentem albo potomkiem parenta."""
        try:
            while widget is not None:
                if widget == parent:
                    return True
                widget = getattr(widget, "master", None)
        except Exception:
            return False
        return False

    def _click_is_inside_popup(self, widget):
        if not self.popup:
            return False
        return self._is_widget_or_child(widget, self.popup)

    def _click_is_inside_textbox(self, widget):
        try:
            return widget == self.textbox._textbox
        except Exception:
            return False

    def _event_line_in_textbox(self, event):
        """Zwraca numer linii klikniętej w textboxie na podstawie współrzędnych eventu."""
        try:
            index = self.textbox._textbox.index(f"@{event.x},{event.y}")
            return int(index.split(".")[0])
        except Exception:
            return None

    def _current_insert_line(self):
        try:
            return int(self.textbox._textbox.index(tk.INSERT).split(".")[0])
        except Exception:
            return None

    def _on_global_click(self, event=None):
        """
        Globalny handler kliknięć.

        - Klik poza textboxem i poza popupem => zamyka popup.
        - Klik w popup => nic nie robi, żeby listbox mógł wstawić wybór.
        - Klik w textbox, ale w inną linię niż linia popupu => zamyka popup
          i blokuje natychmiastowe wznowienie w ButtonRelease.
        """
        if event is None:
            return

        widget = getattr(event, "widget", None)

        # Klik w popup/listbox — nie zamykamy, bo użytkownik może wybierać sugestię.
        if self._click_is_inside_popup(widget):
            return

        # Klik poza właściwym tk.Text — zamknij.
        if not self._click_is_inside_textbox(widget):
            self.hide_popup()
            return

        # Klik w textbox.
        # Jeśli popup jest widoczny i kliknięto inną linię, to jest "odkliknięcie".
        if self.popup and self.popup_anchor_line is not None:
            clicked_line = self._event_line_in_textbox(event)
            if clicked_line is not None and clicked_line != self.popup_anchor_line:
                self._suppress_next_textbox_click = True
                self.hide_popup()
                return

    def _on_textbox_click(self, event=None):
        """
        ButtonRelease w textboxie.

        Po kliknięciu w tę samą linię próbujemy wznowić autocomplete.
        Po kliknięciu w inną linię nie wznawiamy, bo ButtonPress już uznał to za odkliknięcie.
        """
        if self._suppress_next_textbox_click:
            self._suppress_next_textbox_click = False
            return

        try:
            self.textbox._textbox.after(1, self.handle_keypress)
        except Exception:
            self.handle_keypress()

    def _on_focus_out(self, event=None):
        """
        FocusOut zostawiamy jako dodatkową ochronę, ale z opóźnieniem.
        Dzięki temu klik w listbox nie niszczy popupu przed insert_selection.
        """
        try:
            self.textbox._textbox.after(80, self._hide_popup_if_focus_really_left)
        except Exception:
            self.hide_popup()

    def _hide_popup_if_focus_really_left(self):
        try:
            focus = self.textbox._textbox.focus_get()

            if focus == self.textbox._textbox:
                return

            if self.popup and self._is_widget_or_child(focus, self.popup):
                return

        except Exception:
            pass

        self.hide_popup()

    # =========================================================
    # LOGIKA AUTOUZUPEŁNIANIA
    # =========================================================

    def _check_value_mode(self, text, markers, tags, mode_name):
        for marker in markers:
            idx = text.rfind(marker)
            if idx != -1:
                if text.find('"', idx + len(marker)) == -1:
                    prefix = text[idx + len(marker):]
                    matches = [t for t in tags if t.startswith(prefix)]
                    return matches, prefix, mode_name
        return [], "", None

    def handle_keypress(self, event=None):
        if event and getattr(event, 'keysym', '') in (
            "Up", "Down", "Left", "Right", "Return", "Escape", "Shift_L", "Control_L"
        ):
            if event.keysym == "Escape":
                self.hide_popup()
            return

        cursor_index = self.textbox._textbox.index(tk.INSERT)
        text_before = self.textbox._textbox.get(f"{cursor_index} linestart", cursor_index)

        matches = []
        self.current_prefix = ""
        self.current_mode = None

        # 1. Wartości z MORPH_DICTS
        for attr, tags in self.morph_dicts.items():
            markers = [f'{attr}="', f'{attr}!="']
            matches, self.current_prefix, self.current_mode = self._check_value_mode(
                text_before, markers, tags, "morph"
            )
            if matches:
                break

        # 2. Wartości DEPREL
        if not matches:
            matches, self.current_prefix, self.current_mode = self._check_value_mode(
                text_before,
                ['deprel="', 'deprel!="'],
                self.deprel_tags,
                "deprel"
            )

        # # 3. Wartości SRL_ROLE
        # if not matches:
        #     matches, self.current_prefix, self.current_mode = self._check_value_mode(
        #         text_before,
        #         ['srl_role="', 'srl_role!="'],
        #         self.srl_role_tags,
        #         "srl_role"
        #     )

        # 4. Wartości UPOS
        if not matches:
            matches, self.current_prefix, self.current_mode = self._check_value_mode(
                text_before,
                ['upos="', 'upos!="'],
                self.upos_tags,
                "upos"
            )

        # 5. Wartości POS
        if not matches:
            matches, self.current_prefix, self.current_mode = self._check_value_mode(
                text_before,
                ['pos="', 'pos!="'],
                self.pos_tags,
                "pos"
            )

        # 6. ATRYBUTY
        if not matches:
            last_bracket = max(text_before.rfind('['), text_before.rfind('&'), text_before.rfind('{'))
            if last_bracket != -1:
                word = text_before[last_bracket + 1:].strip()
                if '"' not in word:
                    self.current_prefix = word
                    matches = [a for a in self.attributes if a.startswith(word)]
                    self.current_mode = "attr"

        # 7. TAGI GLOBALNE / METADANE
        if not matches and '<' in text_before:
            last_angle = text_before.rfind('<')
            word = text_before[last_angle + 1:].strip()
            if '"' not in word and '>' not in word:
                self.current_prefix = word
                matches = [g for g in self.global_tags if g.startswith(word)]
                self.current_mode = "global"

        if matches:
            self.show_popup(matches)
        else:
            self.hide_popup()

    def show_popup(self, matches):
        if not self.popup:
            self.popup = tk.Toplevel(self.textbox)
            self.popup.wm_overrideredirect(True)
            self.popup.attributes("-topmost", True)

            self.listbox = tk.Listbox(
                self.popup,
                font=("Verdana", 10),
                bg="#1F2328",
                fg="white",
                selectbackground="#4B6CB7",
                highlightthickness=1,
                bd=0,
                exportselection=False
            )
            self.listbox.pack(fill="both", expand=True)
            self.listbox.bind("<ButtonRelease-1>", self.insert_selection)

        self.listbox.delete(0, tk.END)
        for match in matches:
            self.listbox.insert(tk.END, match)
        self.listbox.selection_set(0)

        # zapamiętaj linię, do której należy popup
        self.popup_anchor_line = self._current_insert_line()

        try:
            inner_text = self.textbox._textbox
            bbox = inner_text.bbox(tk.INSERT)

            if bbox:
                x, y, width, height = bbox
                x_root = inner_text.winfo_rootx() + x + 2
                y_root = inner_text.winfo_rooty() + y + height + 5

                popup_h = min(len(matches) * 20 + 5, 150)
                screen_height = self.popup.winfo_screenheight()

                if y_root + popup_h > screen_height:
                    y_root = inner_text.winfo_rooty() + y - popup_h - 5

                self.popup.geometry(f"450x{popup_h}+{x_root}+{y_root}")
        except Exception:
            pass

    def hide_popup(self):
        if self.popup:
            try:
                self.popup.destroy()
            except Exception:
                pass
            self.popup = None
            self.listbox = None
            self.popup_anchor_line = None

    def navigate_up(self, event):
        if self.popup and self.popup.winfo_ismapped() and self.listbox:
            idx = self.listbox.curselection()
            if idx:
                self.listbox.selection_clear(idx[0])
                new_idx = max(0, idx[0] - 1)
                self.listbox.selection_set(new_idx)
                self.listbox.see(new_idx)
            return "break"

    def navigate_down(self, event):
        if self.popup and self.popup.winfo_ismapped() and self.listbox:
            idx = self.listbox.curselection()
            if idx:
                self.listbox.selection_clear(idx[0])
                new_idx = min(self.listbox.size() - 1, idx[0] + 1)
                self.listbox.selection_set(new_idx)
                self.listbox.see(new_idx)
            return "break"

    def insert_selection(self, event):
        if self.popup and self.popup.winfo_ismapped() and self.listbox:
            idx = self.listbox.curselection()
            if idx:
                word = self.listbox.get(idx[0])
                cursor_index = self.textbox._textbox.index(tk.INSERT)

                prefix_len = len(self.current_prefix)
                if prefix_len > 0:
                    start_delete = f"{cursor_index} - {prefix_len} chars"
                    self.textbox._textbox.delete(start_delete, cursor_index)

                # Wycinanie opisu dla wartości typu:
                # "nom (mianownik)" -> "nom"
                # "nsubj - podmiot nominalny" -> "nsubj"
                if self.current_mode in ("morph", "deprel", "srl_role"):
                    word = word.split(" ")[0]

                text_after = self.textbox._textbox.get(tk.INSERT, tk.END)
                should_trigger_next = False

                if word.endswith('=') or word.endswith('!=') or word in ("data>", "data<", "data<=", "data>="):
                    self.textbox._textbox.insert(tk.INSERT, word)

                    if not text_after.startswith('"'):
                        self.textbox._textbox.insert(tk.INSERT, '"')

                    should_trigger_next = True

                else:
                    self.textbox._textbox.insert(tk.INSERT, word)

                    if self.current_mode in ("pos", "upos", "deprel", "morph", "srl_role"):
                        chars_to_add = ""

                        if not text_after.startswith('"'):
                            chars_to_add += '"'

                        next_close = text_after.find(']')
                        next_open = text_after.find('[')

                        is_bracket_closed = next_close != -1 and (next_open == -1 or next_close < next_open)

                        if not is_bracket_closed:
                            chars_to_add += ']'

                        if chars_to_add:
                            self.textbox._textbox.insert(tk.INSERT, chars_to_add)

                if 'highlight_entry' in globals():
                    highlight_entry()

            self.hide_popup()

            if should_trigger_next:
                self.textbox._textbox.after(10, self.handle_keypress)

            return "break"
