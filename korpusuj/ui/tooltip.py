"""Tooltip widgets used by Tkinter and CustomTkinter controls."""

import tkinter as tk

try:
    import customtkinter as ctk
except Exception:
    ctk = None


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
