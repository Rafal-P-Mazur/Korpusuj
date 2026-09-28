"""CustomTkinter and Matplotlib interface for semantic-network exploration.

Application-specific corpus selection, resource maps and report opening are provided through callbacks.
"""

import logging
import math
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

from korpusuj.ui.plots import get_plot_stack


class SemanticNetworkViewer:
    # D19_RANKING_CONTRACT
    # D18c_NO_SELECTED_MEMBERS
    # D18_NO_WSD
    # D17_EXPLORATION_UI
    # D16_MUTUAL_5NN_EXPLORATION
    """Klasa renderująca i zarządzająca oknem grafu sieci semantycznej."""

    def __init__(self, parent_app, engine, theme, insert_query_callback, current_corpus_name_provider=None, current_corpus_path_provider=None, open_report_callback=None):
        self.app = parent_app
        self.engine = engine
        self.theme = theme
        self.on_insert_query = insert_query_callback
        self.current_corpus_name_provider = current_corpus_name_provider
        self.current_corpus_path_provider = current_corpus_path_provider
        self.open_report_callback = open_report_callback or (lambda path: None)

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


        # --- WSD ---
        self.last_neighbors = []

        self.win = ctk.CTkToplevel(self.app)
        self.win.title("Sieć semantyczna")
        self.win.geometry("1400x750")
        self.win.configure(fg_color=self.theme["app_bg"])
        self.win.attributes("-topmost", True)


        self.domain_lambda_var = tk.DoubleVar(value=0.20)
        self.show_mutual_knn_edges_var = tk.BooleanVar(value=True)

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
        self.mode_selector.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            self.side_panel,
            text="Rozmiar węzła zależy od frekwencji. Linie ciągłe powstają podczas rozwijania, dodatkowe połączenia wynikają z mutual 5-NN.",
            wraplength=470,
            justify="left",
            text_color="gray",
            font=("Verdana", 9)
        ).pack(fill="x", pady=(0, 10))


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
        self.settings_win.title("Ustawienia grafu")
        self.settings_win.geometry("430x570")  # <--- POWIĘKSZONE OKNO

        # 2. NAPRAWA CHOWANIA SIĘ POD SPÓD
        self.settings_win.transient(self.win)  # Zawsze utrzymuj nad oknem grafu
        self.settings_win.grab_set()  # Blokuje klikanie w graf, dopóki to okno jest otwarte

        self.settings_win.configure(fg_color=self.theme["app_bg"])

        # Pozycjonowanie na środku okna grafu
        x = self.win.winfo_x() + (self.win.winfo_width() // 2) - 175
        y = self.win.winfo_y() + (self.win.winfo_height() // 2) - 200  # <--- ZMIENIONE WYRÓWNANIE DO ŚRODKA
        self.settings_win.geometry(f"+{x}+{y}")

        ctk.CTkLabel(self.settings_win, text=f"Liczba lematów dodawanych po rozwinięciu węzła\n(Maksymalnie dostępnych: {max_avail})",
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


        title_label = ctk.CTkLabel(domain_frame, text="Ograniczanie lematów ogólnych", font=("Verdana", 12, "bold"))
        title_label.pack(pady=(0, 5))

        lambda_val_label = ctk.CTkLabel(domain_frame, text="", font=("Verdana", 11))
        lambda_val_label.pack(pady=(0, 5))

        def update_lambda_label(val):
            lambda_val_label.configure(text=f"Wartość: {float(val):.2f}")

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
            text="Obniża pozycję lematów występujących na listach sąsiedztwa wielu różnych jednostek. Wartość 0 wyłącza korektę.",
            text_color="gray",
            font=("Verdana", 10)
        )
        tooltip_label.pack(pady=(0, 5))

        ctk.CTkCheckBox(
            self.settings_win,
            text="Pokazuj dodatkowe krawędzie mutual 5-NN",
            variable=self.show_mutual_knn_edges_var,
            font=("Verdana", 11)
        ).pack(fill="x", padx=25, pady=(8, 2))
        ctk.CTkLabel(
            self.settings_win,
            text="Połączenie jest dodawane, gdy każdy lemat należy do pięciu najbliższych sąsiadów drugiego.",
            wraplength=380,
            justify="left",
            text_color="gray",
            font=("Verdana", 9)
        ).pack(fill="x", padx=25, pady=(0, 6))

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

        ctk.CTkLabel(seed_frame, text="Seed rozmieszczenia", font=("Verdana", 12, "bold")).pack(pady=(0, 5))

        seed_entry = ctk.CTkEntry(
            seed_frame,
            textvariable=self.layout_seed_var,
            placeholder_text="Zostaw puste dla losowości",
            justify="center"
        )
        seed_entry.pack(fill="x", padx=15)

        ctk.CTkLabel(
            seed_frame,
            text="Wpływa tylko na położenie węzłów. Nie zmienia wyboru lematów ani krawędzi.",
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

        current_corpus_name = (
            self.current_corpus_name_provider()
            if self.current_corpus_name_provider is not None
            else ""
        )
        current_corpus_path = (
            self.current_corpus_path_provider(current_corpus_name)
            if self.current_corpus_path_provider is not None
            else None
        )

        self.engine.build_semantic_report(
            parent_app=self.app,
            current_corpus_name=current_corpus_name,
            current_corpus_path=current_corpus_path,
            lemma=lemma,
            theme=self.theme,
            open_report_callback=self.open_report_callback,
            params={
                "hops": 2,
                "top_k": self.neighbors_limit_var.get(),
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

        # parent=None oznacza pierwszy krok. Silnik użyje wtedy wyłącznie
        # podobieństwa do aktualnie rozwijanego lematu.
        matched_word, all_res = self.engine.get_contextual_neighbors(
            center_lemma=word, top_n=150,
            root_lemma=root_lemma, parent_lemma=parent,
            domain_lambda=self.domain_lambda_var.get()
        )

        for widget in self.results_frame.winfo_children(): widget.destroy()

        self.current_center = matched_word

        # D16: pierwsze N według selection_score, halo to kolejne N.
        limit = max(0, int(self.neighbors_limit_var.get()))
        core_res = all_res[:limit]
        halo_res = all_res[limit:limit * 2]

        self.last_neighbors = core_res  # główne węzły


        if self.G.has_node(self.current_center):
            self.G.nodes[self.current_center]['type'] = 'center'

        # Zapisz historię eksploracji węzłów ręcznie klikniętych (tzw. "Pinned")
        step_record = {
            "word": matched_word,
            "parent": parent,
            "root": root_lemma
        }

        self.expanded_centers_history = [s for s in self.expanded_centers_history if
                                         not (s.get("word") == matched_word and s.get("parent") == parent)]
        self.expanded_centers_history.append(step_record)

        if parent:
            self.node_parent[self.current_center] = parent
            self.node_root[self.current_center] = root_lemma


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

    def _add_contextual_bridges(self, neighbors_data, max_cross_edges_per_node=2):
        """Dodaje najsilniejsze dodatkowe krawędzie mutual 5-NN bez progu kosinusowego."""
        visible = {item["lemma"]: item for item in neighbors_data if self.G.has_node(item["lemma"])}
        candidates = []
        lemmas = sorted(visible)
        for i, u in enumerate(lemmas):
            for v in lemmas[i + 1:]:
                if self.G.has_edge(u, v) or not self.engine.is_mutual_knn(u, v):
                    continue
                u_vec = self.engine.get_representation_vector(u)
                v_vec = self.engine.get_representation_vector(v)
                if u_vec is not None and v_vec is not None:
                    candidates.append((float(self.engine._cos(u_vec, v_vec)), u, v))
        counts = {lemma: 0 for lemma in lemmas}
        for sim, u, v in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
            if counts[u] >= max_cross_edges_per_node or counts[v] >= max_cross_edges_per_node:
                continue
            self.G.add_edge(u, v, weight=sim, semantic_edge_weight=sim, edge_type="mutual_5nn")
            counts[u] += 1
            counts[v] += 1

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
            sim = float(self.G[center_word][nbr].get("semantic_edge_weight", self.G[center_word][nbr].get("weight", 0.0)))

            # D16: zachowujemy rzeczywistą siłę relacji, bez sztucznego minimum.
            self.halo_nodes[nbr] = {
                "anchor": center_word,
                "sim": sim
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
            sim = float(item.get("base_similarity", 0.0))

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

            # Semantyczna waga krawędzi nie zawiera kary za ogólność.
            edge_weight = item.get(
                "semantic_edge_weight",
                item.get("graph_weight", item.get("selection_score", item["score"]))
            )

            n_freq = item["freq"]
            sense_id = item["sense_id"]

            if not self.G.has_node(n_word):
                self.G.add_node(n_word, type='neighbor', freq=n_freq, sense_id=sense_id, parent=center_word,
                                root=self.current_root)
            else:
                self.G.nodes[n_word]["sense_id"] = sense_id

            self.node_sense_id[n_word] = sense_id

            add_or_update_edge(center_word, n_word, edge_weight)
            edge_data = self.G[center_word][n_word]
            edge_data["semantic_edge_weight"] = float(edge_weight)
            edge_data["edge_type"] = "expansion"
            edge_data["selection_score"] = float(item.get("selection_score", item["score"]))
            edge_data["generality_penalty"] = float(item.get("generality_penalty", 0.0))

        # D16: dodatkowe krawędzie wyłącznie mutual 5-NN.
        if self.show_mutual_knn_edges_var.get():
            self._add_contextual_bridges(neighbors_data)

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
            w = d.get('semantic_edge_weight', d.get('weight', 0.5))
            # Osobna waga układu. Transformacja pozostaje bez zmian w D15.
            d['layout_weight'] = w ** 4

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
            weight='layout_weight',
            seed = current_seed  # <--- Podpięcie zmiennej
        )


    def _update_halo_positions(self):
        """Układa kropki tła za pomocą barycentrum grawitacyjnego i stabilnego hashowania."""
        import math
        core_vectors = {}
        for n in self.G.nodes():
            vec = self.engine.get_representation_vector(n)
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

        # Po usunięciu WSD lista jest porządkowana wyłącznie
        # według wyniku wyboru, frekwencji i lematu.
        def sort_key(item):
            return (
                float(item.get("selection_score", item.get("score", 0.0))),
                int(item.get("freq", 0)),
                str(item.get("lemma", "")),
            )

        res_sorted = sorted(res, key=sort_key, reverse=True)

        for item in res_sorted:
            n_word = item["lemma"]
            n_freq = item["freq"]
            n_score = item.get("selection_score", item.get("score", 0.0))
            n_base_sim = item.get("base_similarity", 0.0)
            n_generality = item.get("generality", 0.0)

            has_network = (
                    (n_word in self.engine.index)
                    or (n_word.lower() in self.engine.index)
                    or (n_word.capitalize() in self.engine.index)
            )
            btn_state = "normal" if has_network else "disabled"

            t_color = self.theme["label_text"]
            cmd = (lambda w=n_word, p=self.current_center: self.explore_node(w, parent=p)) if has_network else None

            row = ctk.CTkFrame(self.results_frame, fg_color="transparent")
            row.pack(fill="x", pady=2, padx=2)

            # Kolumny: lemat | wynik | podobieństwo | ogólność | frekwencja | +
            row.grid_columnconfigure(0, weight=1, minsize=140)
            row.grid_columnconfigure(1, weight=0)
            row.grid_columnconfigure(2, weight=0)
            row.grid_columnconfigure(3, weight=0)
            row.grid_columnconfigure(4, weight=0)
            row.grid_columnconfigure(5, weight=0)

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
                text=f"wyn {n_score:.2f}",
                text_color="#1982C4",
                font=("Verdana", 9, "bold"),
                width=52
            ).grid(row=0, column=1, padx=2)

            # 3. Base similarity
            ctk.CTkLabel(
                row,
                text=f"pod {n_base_sim:.2f}",
                text_color="#8A9AAB",
                font=("Verdana", 8),
                width=50
            ).grid(row=0, column=2, padx=2)

            # 4. Ogólność
            ctk.CTkLabel(
                row,
                text=f"og {n_generality:.2f}",
                text_color="#8A6D3B",
                font=("Verdana", 8),
                width=48
            ).grid(row=0, column=3, padx=2)

            # 5. Frekwencja
            ctk.CTkLabel(
                row,
                text=f"f {n_freq:,}".replace(",", " "),
                text_color="gray60",
                font=("Verdana", 8),
                width=48
            ).grid(row=0, column=4, padx=2)

            # 6. Plus
            ctk.CTkButton(
                row,
                text="+",
                width=26,
                height=24,
                command=lambda w=n_word: self.on_insert_query(w)
            ).grid(row=0, column=5, padx=(4, 0))


