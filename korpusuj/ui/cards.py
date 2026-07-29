"""Reusable settings-card widgets for collocations, profiles, plots and other option panels.

Theme data and the optional central card registry are supplied by callers.
"""

import customtkinter as ctk


DEFAULT_SETTINGS_CARD_THEME = {
    "frame_fg": "#2b2b2b",
    "subframe_fg": "#333333",
    "button_fg": "#4B6CB7",
    "button_hover": "#3A5795",
    "label_text": "white",
}


class SettingsCard(ctk.CTkFrame):
    def __init__(self, parent, title, expanded=False, expand_card=False, theme=None, registry=None):
        theme = theme or DEFAULT_SETTINGS_CARD_THEME
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

        if registry is not None:
            registry.append(self)

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
