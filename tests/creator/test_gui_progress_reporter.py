# -*- coding: utf-8 -*-
"""Lightweight tests for GuiProgressReporter (track 172g1)."""
from __future__ import annotations

from korpusuj.corpus.creator_core import ProgressReporter
from korpusuj.corpus.creator_gui_adapter import GuiProgressReporter


class DummyApp:
    def __init__(self):
        self.after_calls = []
        self.tick_count = 0

    def after(self, delay, callback):
        self.after_calls.append(delay)
        callback()

    def update_idletasks(self):
        self.tick_count += 1


class DummyLabel:
    def __init__(self):
        self.text = None

    def configure(self, **kwargs):
        self.text = kwargs.get("text")


class DummyProgress:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


def make_reporter(**overrides):
    app = overrides.pop("app", DummyApp())
    status = overrides.pop("status_label", DummyLabel())
    current = overrides.pop("progress_bar_current", DummyProgress())
    total = overrides.pop("progress_bar_total", DummyProgress())
    size = overrides.pop("size_label", DummyLabel())
    reporter = GuiProgressReporter(
        app=app,
        status_label=status,
        progress_bar_current=current,
        progress_bar_total=total,
        size_label=size,
        **overrides,
    )
    return reporter, app, status, current, total, size


def test_gui_progress_reporter_satisfies_runtime_protocol():
    reporter, *_ = make_reporter()
    assert isinstance(reporter, ProgressReporter)


def test_status_current_total_and_size_are_mapped():
    reporter, app, status, current, total, size = make_reporter()
    reporter.status("Przetwarzam")
    reporter.current(0.25)
    reporter.total(0.75)
    reporter.size_info("10 MB")

    assert status.text == "Przetwarzam"
    assert current.value == 0.25
    assert total.value == 0.75
    assert size.text == "10 MB"
    assert app.after_calls == [0, 0, 0, 0]


def test_warning_and_error_use_injected_dialog_callbacks():
    calls = []
    reporter, *_ = make_reporter(
        show_warning=lambda title, message: calls.append(("warning", title, message)),
        show_error=lambda title, message: calls.append(("error", title, message)),
    )
    reporter.warning("Uwaga")
    reporter.error("Niepowodzenie", ValueError("x"))

    assert calls == [
        ("warning", "Ostrzeżenie", "Uwaga"),
        ("error", "Błąd", "Niepowodzenie: x"),
    ]


def test_warning_and_error_fall_back_to_status_label():
    reporter, _, status, *_ = make_reporter()
    reporter.warning("Ostrzeżenie testowe")
    assert status.text == "Ostrzeżenie testowe"
    reporter.error("Błąd testowy")
    assert status.text == "Błąd testowy"


def test_tick_calls_update_idletasks():
    reporter, app, *_ = make_reporter()
    reporter.tick()
    assert app.tick_count == 1


def test_direct_fallback_without_app_after():
    status = DummyLabel()
    reporter = GuiProgressReporter(app=None, status_label=status)
    reporter.status("Bez scheduler-a")
    assert status.text == "Bez scheduler-a"


def test_missing_widgets_and_callbacks_are_safe():
    reporter = GuiProgressReporter()
    reporter.status("x")
    reporter.current(0.1)
    reporter.total(0.2)
    reporter.size_info("x")
    reporter.warning("x")
    reporter.error("x", RuntimeError("y"))
    reporter.tick()
