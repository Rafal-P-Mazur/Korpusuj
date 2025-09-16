import customtkinter as ctk
import tkinter as tk


class CustomTable(ctk.CTkFrame):
    def __init__(self, master, headers, data, min_column_widths, justify_list, rows_per_page, fulltext_data,  **kwargs):
        super().__init__(master, **kwargs)
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

        self.header_bg_color = "#4B6CB7"
        self.header_text_color = "white"
        self.even_row_color = "#2b2b2b"
        self.odd_row_color = "#333333"
        self.text_colors = ["white"] * len(self.headers)
        self.font = ("Arial", 14)
        self.header_font = ("Arial", 15)


        self.canvas = ctk.CTkCanvas(self, background='#1d1e1e', highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, side="left")

        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.pack(fill="y", side="right")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.table_frame = ctk.CTkFrame(self.canvas)
        self.table_window = self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw",
                                                      width=self.canvas.winfo_width())

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.after_idle(self.populate_table)  # ← Defer initial layout

        self.table_frame.bind("<Enter>", self.on_mouse_enter)
        self.table_frame.bind("<Leave>", self.on_mouse_leave)
        self.table_frame.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
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



    def set_additional_event(self, event_func):
        self.additional_event = event_func

    def set_header_colors(self, bg_color, text_color):
        self.header_bg_color = bg_color
        self.header_text_color = text_color
        self.populate_table()

    def set_header_font(self, font):
        self.header_font = font
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
        self.table_frame.configure(fg_color=color)  # match scrollable area

        # Style scrollbar to match the theme
        self.scrollbar.configure(
            fg_color=color,  # trough background
            #button_color="#7289DA",  # thumb color

        )

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
        # Ensure row gets selected before showing menu
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



    def populate_table(self):
        self._suppress_resize = True



        # Clear only first time (headers), not every time
        if not hasattr(self, "header_created"):
            for widget in self.table_frame.winfo_children():
                widget.destroy()

            # Headers
            for c, text in enumerate(self.headers):
                header_label = ctk.CTkLabel(
                    self.table_frame, text=text, fg_color=self.header_bg_color,
                    text_color=self.header_text_color, font=self.header_font, padx=5
                )
                header_label.grid(row=0, column=c, sticky="nsew")
                header_label.bind("<MouseWheel>", self.on_mouse_wheel)

            # Pre-create row widgets pool (empty text at first)
            self.label_refs = {}
            self.row_labels = []  # keep reference to row label sets
            for r in range(1, self.rows_per_page + 1):  # rows start at 1
                row_widgets = []
                row_color = self.even_row_color if r % 2 == 0 else self.odd_row_color
                for c in range(len(self.headers)):
                    label = ctk.CTkLabel(
                        self.table_frame, text="", wraplength=self.min_column_widths[c],
                        anchor=self.text_anchors[c], justify=self.justify_list[c],
                        fg_color=row_color, text_color=self.text_colors[c],
                        font=self.font, pady=10, padx=5
                    )
                    label.grid(row=r, column=c, sticky="nsew")
                    label.bind("<Button-1>", lambda event, row_index=r: self.on_row_click(row_index))
                    label.bind("<Button-3>", lambda event, row_index=r: self.show_context_menu(event, row_index))
                    label.bind("<MouseWheel>", self.on_mouse_wheel)
                    self.label_refs[(r, c)] = label
                    row_widgets.append(label)

                self.row_labels.append(row_widgets)

            self.header_labels = []
            for c, text in enumerate(self.headers):
                header_label = ctk.CTkLabel(
                    self.table_frame, text=text,
                    fg_color=self.header_bg_color,
                    text_color=self.header_text_color,
                    font=self.header_font, padx=5
                )
                header_label.grid(row=0, column=c, sticky="nsew")
                header_label.bind("<MouseWheel>", self.on_mouse_wheel)
                self.header_labels.append(header_label)

            # Configure columns
            for c in range(len(self.headers)):
                self.table_frame.grid_columnconfigure(c, weight=1)

            self.header_created = True



        # --- Update the row pool with new data (cap at rows_per_page) ---
        for r, row in enumerate(self.data[:self.rows_per_page], start=1):
            for c, text in enumerate(row):
                if (r, c) in self.label_refs:  # safety check
                    self.label_refs[(r, c)].configure(text=text)

        # Clear any extra rows (if less than rows_per_page)
        for r in range(len(self.data) + 1, self.rows_per_page + 1):
            for c in range(len(self.headers)):
                if (r, c) in self.label_refs:
                    self.label_refs[(r, c)].configure(text="")

        self.table_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update_scrollbar_visibility()
        self.update_wraplength()
        self._suppress_resize = False

        # --- Update row colors and text colors for all visible rows ---
        for r in range(1, self.rows_per_page + 1):
            row_color = self.even_row_color if r % 2 == 0 else self.odd_row_color
            for c in range(len(self.headers)):
                if (r, c) in self.label_refs:
                    self.label_refs[(r, c)].configure(
                        fg_color=row_color,
                        text_color=self.text_colors[c],
                        font=self.font
                    )
        self.after_idle(lambda: self.canvas.yview_moveto(0))



    def update_scrollbar_visibility(self):
        self.canvas.update_idletasks()

        if not self.data:  # ← skip scrollbar entirely if no rows
            self.scrollbar.pack_forget()
            self.canvas.configure(yscrollcommand="")
            return

        bbox = self.canvas.bbox("all")
        canvas_height = self.canvas.winfo_height()

        if bbox and bbox[3] > canvas_height:
            self.scrollbar.pack(fill="y", side="right")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
        else:
            self.scrollbar.pack_forget()
            self.canvas.configure(yscrollcommand="")

    def add_row(self, *new_row):
        self.data.append(new_row)
        self.populate_table()

    def update_table_size(self, event=None):
        self.canvas.itemconfig(self.table_window, width=self.canvas.winfo_width())

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
                direction = -1 if event.delta > 0 else 1
                self.canvas.yview_scroll(direction, "units")

    def on_canvas_enter(self, event):
        self.canvas.focus_set()

    def on_canvas_leave(self, event):
        self.table_frame.focus_set()

    def update_wraplength(self):
        n_cols = len(self.headers)
        if n_cols == 0:
            return

        total_width = max(1, self.table_frame.winfo_width())
        col_width = total_width // n_cols

        # compute padding once based on stored scaling
        padding = max(0, int((self.tk_scaling - self.base_scaling) * self.scaling_padding_factor))

        for c in range(n_cols):
            self.table_frame.grid_columnconfigure(c, weight=1)
            for r in range(1, self.rows_per_page + 1):
                if (r, c) in self.label_refs:
                    self.label_refs[(r, c)].configure(
                        wraplength=max(1, col_width - padding)
                    )

    def on_frame_resize(self, event):
        if self._initializing or self._suppress_resize:
            return

        if self.resize_delay:
            self.table_frame.after_cancel(self.resize_delay)
        self.resize_delay = self.table_frame.after(10, self.update_wraplength)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.update_scrollbar_visibility()
