# -*- coding: utf-8 -*-
"""GUI-free protocol and run-option types for corpus creation.

This module is additive scaffolding. Existing creator execution paths are not
routed through these types yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProgressReporter(Protocol):
    """Receives creator status/progress events without depending on a GUI."""

    def status(self, message: str) -> None:
        ...

    def current(self, value: float) -> None:
        ...

    def total(self, value: float) -> None:
        ...

    def size_info(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...

    def error(self, message: str, exc: Exception | None = None) -> None:
        ...

    def tick(self) -> None:
        ...


@dataclass(slots=True)
class NullProgressReporter:
    """No-op reporter suitable for headless calls and functional tests."""

    def status(self, message: str) -> None:
        return None

    def current(self, value: float) -> None:
        return None

    def total(self, value: float) -> None:
        return None

    def size_info(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str, exc: Exception | None = None) -> None:
        return None

    def tick(self) -> None:
        return None


@dataclass(slots=True)
class CreatorRunOptions:
    """Explicit inputs for a future GUI-independent creator orchestration call."""

    input_files: list[str]
    output_parquet_file: str
    metadata_path: str | None = None
    model_name: str = "stanza"
    excel_mappings: dict[str, Any] | None = None
    resume_mode: bool = False
    processed_set: set[str] | None = None
    enable_ner: bool = True
    enable_coreference: bool = True
    lemma_corrections_path: str | None = None
    def __post_init__(self) -> None:
        self.input_files = [str(path) for path in self.input_files]
        self.output_parquet_file = str(self.output_parquet_file)
        if self.metadata_path is not None:
            self.metadata_path = str(self.metadata_path)
        if self.lemma_corrections_path is not None:
            self.lemma_corrections_path = str(self.lemma_corrections_path)
        self.model_name = str(self.model_name or "stanza")
        if self.processed_set is not None:
            self.processed_set = {str(path) for path in self.processed_set}


__all__ = [
    "CreatorRunOptions",
    "NullProgressReporter",
    "ProgressReporter",
]
