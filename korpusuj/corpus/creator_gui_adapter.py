# -*- coding: utf-8 -*-
"""GUI adapter for creator progress events.

This module is intentionally GUI-facing. The protocol itself remains in the
GUI-free ``korpusuj.corpus.creator_core`` module.
"""
from __future__ import annotations

from typing import Any, Callable

from korpusuj.corpus.creator_core import ProgressReporter


class GuiProgressReporter:
    """Map creator progress events to the existing Tk/CustomTkinter widgets.

    All widget arguments are optional, which permits gradual migration of the
    current creator call paths. Widget mutations are scheduled through
    ``app.after(0, ...)`` when possible. If scheduling is unavailable or fails,
    the mutation is applied directly.

    Warning/error dialogs are injected as callables so importing this module
    does not import tkinter or customtkinter. The current GUI can pass
    ``messagebox.showwarning`` and ``messagebox.showerror`` explicitly.
    """

    def __init__(
        self,
        *,
        app: Any = None,
        status_label: Any = None,
        progress_bar_current: Any = None,
        progress_bar_total: Any = None,
        size_label: Any = None,
        show_warning: Callable[[str, str], Any] | None = None,
        show_error: Callable[[str, str], Any] | None = None,
        warning_title: str = "Ostrzeżenie",
        error_title: str = "Błąd",
    ) -> None:
        self.app = app
        self.status_label = status_label
        self.progress_bar_current = progress_bar_current
        self.progress_bar_total = progress_bar_total
        self.size_label = size_label
        self.show_warning = show_warning
        self.show_error = show_error
        self.warning_title = warning_title
        self.error_title = error_title

    def _schedule(self, callback: Callable[[], Any]) -> None:
        app_after = getattr(self.app, "after", None)
        if callable(app_after):
            try:
                app_after(0, callback)
                return
            except Exception:
                pass
        try:
            callback()
        except Exception:
            pass

    @staticmethod
    def _configure_text(widget: Any, message: str) -> None:
        if widget is None:
            return
        configure = getattr(widget, "configure", None)
        if callable(configure):
            configure(text=str(message))

    @staticmethod
    def _set_value(widget: Any, value: float) -> None:
        if widget is None:
            return
        setter = getattr(widget, "set", None)
        if callable(setter):
            setter(float(value))

    def status(self, message: str) -> None:
        self._schedule(lambda: self._configure_text(self.status_label, message))

    def current(self, value: float) -> None:
        self._schedule(lambda: self._set_value(self.progress_bar_current, value))

    def total(self, value: float) -> None:
        self._schedule(lambda: self._set_value(self.progress_bar_total, value))

    def size_info(self, message: str) -> None:
        self._schedule(lambda: self._configure_text(self.size_label, message))

    def warning(self, message: str) -> None:
        if callable(self.show_warning):
            self._schedule(lambda: self.show_warning(self.warning_title, str(message)))
        else:
            self.status(message)

    def error(self, message: str, exc: Exception | None = None) -> None:
        rendered = str(message)
        if exc is not None:
            rendered = f"{rendered}: {exc}"
        if callable(self.show_error):
            self._schedule(lambda: self.show_error(self.error_title, rendered))
        else:
            self.status(rendered)

    def tick(self) -> None:
        update = getattr(self.app, "update_idletasks", None)
        if callable(update):
            try:
                update()
            except Exception:
                pass


__all__ = ["GuiProgressReporter"]
