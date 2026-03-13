import customtkinter as ctk
import tkinter as tk


class CustomTable(ctk.CTkFrame):
    def __init__(self, master, headers, data, min_column_widths, justify_list, rows_per_page, fulltext_data, search_callback=None, search_column_index=1, sort_callback=None, sortable=True, **kwargs):
        super().__init__(master, **kwargs)
        self.sortable = sortable
        self._initializing = True
        self._suppress_resize = False

        self.tk_scaling = float(self.winfo_toplevel().tk.call("tk", "scaling"))
        self.base_scaling = 1.33  # 100% scaling reference
        self.scaling_padding_factor = 200  # tweak this as needed to change margin

        self.headers = headers
        self.data = data
        self.fulltext_data = fulltext_data
        self.min_column_widths = min_column_widths
        self.justify_list = justify_list
        self.label_refs = {}
        self.selected_row = None
        self.additional_event = None
        self.resize_delay = None

        # --- Zmienne do sortowania ---
        self.sort_callback = sort_callback
        self.sort_col = None
        self.sort_asc = True

        self.header_bg_color = "#4B6CB7"
        self.header_text_color = "white"
        self.even_row_color = "#2b2b2b"
        self.odd_row_color = "#333333"
        self.text_colors = ["white"] * len(self.headers)
        self.font = ("Arial", 14)
        self.header_font = ("Arial", 15)

        # Ustawiamy siatkę dla głównego kontenera tabeli
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas = ctk.CTkCanvas(self, background='#1d1e1e', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Pasek pionowy
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Pasek poziomy
        self.h_scrollbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=self.scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.table_frame = ctk.CTkFrame(self.canvas)

        self.table_window = self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw",
                                                      width=self.canvas.winfo_width())

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.after_idle(self.populate_table)  # ← Defer initial layout

        self.table_frame.bind("<Enter>", self.on_mouse_enter)
        self.table_frame.bind("<Leave>", self.on_mouse_leave)

        self.table_frame.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # Przewijanie poziome (Shift + Scroll)
        self.table_frame.bind("<Shift-MouseWheel>", self.on_shift_mouse_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mouse_wheel)

        self.canvas.bind("<Enter>", self.on_canvas_enter)
        self.canvas.bind("<Leave>", self.on_canvas_leave)

        self._initializing = False
        self.table_frame.bind("<Configure>", self.on_frame_resize)

        self.selected_row_color = "#103858"  # Default selected row color

        self.text_anchors = ["center"] * len(headers)

        self.rows_per_page = rows_per_page

        # Right-click menu
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Kopiuj", command=self.copy_selected_row)

        # Opcja wyszukiwania
        self.search_callback = search_callback
        self.search_column_index = search_column_index
        if self.search_callback:
            self.context_menu.add_command(label="Wyszukaj kolokację", command=self.search_selected_row)

    def set_additional_event(self, event_func):
        self.additional_event = event_func

    def set_header_colors(self, bg_color, text_color):
        self.header_bg_color = bg_color
        self.header_text_color = text_color
        self.populate_table()

    def set_header_font(self, font):
        self.header_font = font
        if hasattr(self, "header_labels"):
            for header_label in self.header_labels:
                header_label.configure(font=self.header_font)

    def set_row_colors(self, even_color, odd_color):
        self.even_row_color = even_color
        self.odd_row_color = odd_color
        self.populate_table()

    def set_text_colors(self, text_colors):
        if len(text_colors) == len(self.headers):
            self.text_colors = text_colors
            self.populate_table()

    def set_font(self, font):
        self.font = font
        self.populate_table()

    def set_data(self, data):
        self.selected_row = None
        self.data = data
        self.populate_table()

    def set_rows_number(self, rows_per_page):
        self.rows_per_page = rows_per_page
        # Force rebuild next time populate_table runs
        if hasattr(self, "header_created"):
            del self.header_created
        self.populate_table()

    def set_selected_row_color(self, color):
        self.selected_row_color = color

    def set_fulltext_data(self, fulltext_data):
        self.fulltext_data = fulltext_data

    def set_canvas_background(self, color):
        """Set background for canvas, inner table frame, and style scrollbar."""
        self.canvas_bg_color = color
        self.canvas.configure(background=color)
        self.table_frame.configure(fg_color=color)
        self.scrollbar.configure(fg_color=color)
        if hasattr(self, 'h_scrollbar'):
            self.h_scrollbar.configure(fg_color=color)

    def on_row_click(self, row_index):
        if self.selected_row is not None:
            self.restore_row_color(self.selected_row)
        for col in range(len(self.headers)):
            self.label_refs[(row_index, col)].configure(fg_color=self.selected_row_color)
        self.selected_row = row_index
        if self.additional_event:
            self.additional_event(row_index)

    def restore_row_color(self, row_index):
        original_color = self.even_row_color if row_index % 2 == 0 else self.odd_row_color
        for col in range(len(self.headers)):
            if (row_index, col) in self.label_refs:
                self.label_refs[(row_index, col)].configure(fg_color=original_color)

    def set_text_anchor(self, anchors):
        if len(anchors) == len(self.headers):
            self.text_anchors = anchors
            self.populate_table()

    def show_context_menu(self, event, row_index):
        """Show right-click menu on a row."""
        self.on_row_click(row_index)
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def copy_selected_row(self):
        """Copy the selected row's text to clipboard."""
        if self.selected_row is not None and 1 <= self.selected_row <= len(self.data):
            row_data = self.data[self.selected_row - 1]  # -1 because rows start at 1 in grid
            text_to_copy = "\t".join(str(item) for item in row_data)
            self.clipboard_clear()
            self.clipboard_append(text_to_copy)
            self.update()  # ensures clipboard gets updated

    def search_selected_row(self):
        """Pobiera słowo z wybranej kolumny i przekazuje do callbacka wyszukiwania."""
        if self.selected_row is not None and 1 <= self.selected_row <= len(self.data):
            row_data = self.data[self.selected_row - 1]
            if len(row_data) > self.search_column_index:
                selected_word = str(row_data[self.search_column_index])
                self.search_callback(selected_word)

    # --- NOWE FUNKCJE OBSŁUGI SORTOWANIA ---
    def update_header_text(self):
        if hasattr(self, "header_labels"):
            for c, label in enumerate(self.header_labels):
                display_text = self.headers[c]
                if self.sort_col == c:
                    display_text += " ▲" if self.sort_asc else " ▼"
                label.configure(text=display_text)

    def on_header_click(self, col_index):
        if self.sort_col == col_index:
            self.sort_asc = not self.sort_asc
        else:
            self.sort_col = col_index
            self.sort_asc = True

        self.update_header_text()

        # Użycie zewnętrznego callbacka (np. do paginacji)
        if self.sort_callback:
            self.sort_callback(col_index, self.sort_asc)
        else:
            # Lokalny fallback - sortuje tylko to, co tabela ma aktualnie w self.data
            def sort_key(row):
                val = row[col_index] if col_index < len(row) else ""
                return val if val is not None else ""

            try:
                # Opcjonalne utrzymanie synchronizacji fulltext_data
                if self.fulltext_data and len(self.fulltext_data) == len(self.data):
                    combined = list(zip(self.data, self.fulltext_data))
                    combined.sort(key=lambda x: sort_key(x[0]), reverse=not self.sort_asc)
                    self.data, self.fulltext_data = zip(*combined)
                    self.data = list(self.data)
                    self.fulltext_data = list(self.fulltext_data)
                else:
                    self.data.sort(key=sort_key, reverse=not self.sort_asc)
            except TypeError:
                # W przypadku mieszanych typów danych (np. liczby i tekst), sortuj po zamianie na string
                if self.fulltext_data and len(self.fulltext_data) == len(self.data):
                    combined = list(zip(self.data, self.fulltext_data))
                    combined.sort(key=lambda x: str(sort_key(x[0])), reverse=not self.sort_asc)
                    self.data, self.fulltext_data = zip(*combined)
                    self.data = list(self.data)
                    self.fulltext_data = list(self.fulltext_data)
                else:
                    self.data.sort(key=lambda row: str(sort_key(row)), reverse=not self.sort_asc)

            self.populate_table()

    def populate_table(self):
        self._suppress_resize = True

        # --- 1. INITIAL WIDGET CREATION (Runs only once or on rows_per_page change) ---
        if not hasattr(self, "header_created"):
            for widget in self.table_frame.winfo_children():
                widget.destroy()

            # Create Headers ze wskaźnikami sortowania
            self.header_labels = []
            for c, text in enumerate(self.headers):
                display_text = text
                if getattr(self, "sort_col", None) == c:
                    display_text += " ▲" if self.sort_asc else " ▼"

                header_label = ctk.CTkLabel(
                    self.table_frame, text=display_text,
                    fg_color=self.header_bg_color,
                    text_color=self.header_text_color,
                    font=self.header_font, padx=5
                )
                header_label.grid(row=0, column=c, sticky="nsew")
                header_label.bind("<MouseWheel>", self.on_mouse_wheel)
                header_label.bind("<Shift-MouseWheel>", self.on_shift_mouse_wheel)

                # --- TUTAJ ZMIANA ---
                if self.sortable:
                    header_label.configure(cursor="hand2")
                    header_label.bind("<Button-1>", lambda event, col=c: self.on_header_click(col))
                # --------------------

                self.header_labels.append(header_label)

            # Configure column weights proportionally based on min_column_widths
            total_min_width = sum(self.min_column_widths)
            for c in range(len(self.headers)):
                col_weight = max(1, int((self.min_column_widths[c] / total_min_width) * 10))
                self.table_frame.grid_columnconfigure(c, weight=col_weight, minsize=self.min_column_widths[c])

            # Pre-create row widgets pool
            self.label_refs = {}
            for r in range(1, self.rows_per_page + 1):  # rows start at 1
                row_color = self.even_row_color if r % 2 == 0 else self.odd_row_color
                for c in range(len(self.headers)):
                    label = ctk.CTkLabel(
                        self.table_frame, text="", wraplength=self.min_column_widths[c],
                        anchor=self.text_anchors[c], justify=self.justify_list[c],
                        fg_color=row_color, text_color=self.text_colors[c],
                        font=self.font, pady=10, padx=5, cursor="hand2"  # <--- KURSOR ŁAPKI
                    )
                    label.grid(row=r, column=c, sticky="nsew")

                    # Bindings
                    label.bind("<Button-1>", lambda event, row_index=r: self.on_row_click(row_index))
                    label.bind("<Button-3>", lambda event, row_index=r: self.show_context_menu(event, row_index))
                    label.bind("<MouseWheel>", self.on_mouse_wheel)
                    label.bind("<Shift-MouseWheel>", self.on_shift_mouse_wheel)

                    self.label_refs[(r, c)] = label

            self.header_created = True

        # --- 2. SINGLE-PASS DATA UPDATE (Lightning fast for pagination) ---
        data_len = len(self.data)

        for r in range(1, self.rows_per_page + 1):
            row_color = self.even_row_color if r % 2 == 0 else self.odd_row_color

            # Check if this row has data for this page
            if r <= data_len:
                row_data = self.data[r - 1]
                for c in range(len(self.headers)):
                    text_val = row_data[c] if c < len(row_data) else ""
                    self.label_refs[(r, c)].configure(
                        text=text_val,
                        fg_color=row_color,
                        text_color=self.text_colors[c],
                        font=self.font
                    )
            else:
                # Clear empty rows
                for c in range(len(self.headers)):
                    self.label_refs[(r, c)].configure(
                        text="",
                        fg_color=row_color,
                        text_color=self.text_colors[c],
                        font=self.font
                    )

        # Force UI update and scrollbar recalculation
        self.table_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update_scrollbar_visibility()
        self.update_wraplength()
        self._suppress_resize = False
        self.after_idle(lambda: self.canvas.yview_moveto(0))

    def update_scrollbar_visibility(self):
        self.canvas.update_idletasks()

        if not self.data:
            self.scrollbar.grid_remove()
            self.h_scrollbar.grid_remove()
            self.canvas.configure(yscrollcommand="", xscrollcommand="")
            return

        bbox = self.canvas.bbox("all")
        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()

        # Zarządzanie pionowym paskiem (tylko grid!)
        if bbox and bbox[3] > canvas_height:
            self.scrollbar.grid()
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
        else:
            self.scrollbar.grid_remove()
            self.canvas.configure(yscrollcommand="")

        # Zarządzanie poziomym paskiem (tylko grid!)
        if bbox and bbox[2] > canvas_width:
            self.h_scrollbar.grid()
            self.canvas.configure(xscrollcommand=self.h_scrollbar.set)
        else:
            self.h_scrollbar.grid_remove()
            self.canvas.configure(xscrollcommand="")

    # Upewnij się, że pod spodem zaraz masz add_row, a nie stare pozostałości
    def add_row(self, *new_row):
        self.data.append(new_row)
        self.populate_table()

    def add_row(self, *new_row):
        self.data.append(new_row)
        self.populate_table()

    def update_table_size(self, event=None):
        import customtkinter as ctk
        scaling = ctk.ScalingTracker.get_widget_scaling(self)

        # Prawidłowo przeskalowana minimalna szerokość całej tabeli
        min_w = sum(int(w * scaling) for w in self.min_column_widths)

        new_width = max(min_w, self.canvas.winfo_width())
        self.canvas.itemconfig(self.table_window, width=new_width)

    def on_canvas_resize(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update_table_size()

    def on_mouse_enter(self, event):
        self.table_frame.focus_set()

    def on_mouse_leave(self, event):
        self.table_frame.focus_set()

    def on_mouse_wheel(self, event):
        bbox = self.canvas.bbox("all")
        if bbox:
            canvas_height = self.canvas.winfo_height()
            if bbox[3] > canvas_height:
                if event.num == 4 or event.delta > 0:
                    direction = -1
                elif event.num == 5 or event.delta < 0:
                    direction = 1
                else:
                    return
                self.canvas.yview_scroll(direction, "units")

    def on_shift_mouse_wheel(self, event):
        bbox = self.canvas.bbox("all")
        if bbox:
            canvas_width = self.canvas.winfo_width()
            if bbox[2] > canvas_width:
                if event.num == 4 or event.delta > 0:
                    direction = -1
                elif event.num == 5 or event.delta < 0:
                    direction = 1
                else:
                    return
                self.canvas.xview_scroll(direction, "units")

    def on_canvas_enter(self, event):
        self.canvas.focus_set()

    def on_canvas_leave(self, event):
        self.table_frame.focus_set()

    def update_wraplength(self):
        n_cols = len(self.headers)
        if n_cols == 0:
            return

        import customtkinter as ctk
        scaling = ctk.ScalingTracker.get_widget_scaling(self)

        # Skalujemy minimalne szerokości, żeby odpowiadały fizycznym pikselom na High DPI
        scaled_min_widths = [int(w * scaling) for w in self.min_column_widths]
        total_min_width = sum(scaled_min_widths)

        # Bierzemy pod uwagę Canvas, ale nie pozwalamy mu spaść poniżej sumy minimalnych szerokości
        total_width = max(total_min_width, self.canvas.winfo_width())

        # Fizyczny margines dla tekstu (np. 25 pikseli po bokach)
        padding = int(25 * scaling)

        active_rows = min(len(self.data), self.rows_per_page)

        for c in range(n_cols):
            proportion = scaled_min_widths[c] / total_min_width
            col_width = max(scaled_min_widths[c], int(total_width * proportion))

            # KLUCZOWE: Omijamy błąd podwójnego skalowania w CustomTkinter!
            # CTkLabel automatycznie mnoży wartość 'wraplength' przez współczynnik 'scaling'.
            # My mamy już gotową szerokość w pikselach (col_width), więc musimy ją PODZIELIĆ
            # przez scaling przed przekazaniem do widgetu.
            new_wrap = max(20, int((col_width - padding) / scaling))

            for r in range(1, active_rows + 1):
                if (r, c) in self.label_refs:
                    self.label_refs[(r, c)].configure(wraplength=new_wrap)

    def on_frame_resize(self, event):
        if self._initializing or self._suppress_resize:
            return

        if self.resize_delay:
            self.table_frame.after_cancel(self.resize_delay)

        self.resize_delay = self.table_frame.after(150, self.update_wraplength)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update_scrollbar_visibility()
