# -*- coding: utf-8 -*-
"""Konserwatywne czyszczenie lematów przechowywanych przez Korpusuj."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

INVISIBLE_FORMATTING = {
    "\u00ad",  # SOFT HYPHEN
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u200e",  # LEFT-TO-RIGHT MARK
    "\u2060",  # WORD JOINER
    "\u2063",  # INVISIBLE SEPARATOR
    "\u2066",  # LEFT-TO-RIGHT ISOLATE
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
}
_WHITESPACE = re.compile(r"\s+")


def sanitize_stored_lemma(value: Any) -> str:
    """NFC + jawne artefakty formatowania + białe znaki + trim.

    Pisownia, wielkość liter, apostrofy i łączniki pozostają bez zmian.
    Jeśli czyszczenie dałoby pusty napis, zachowywana jest wartość źródłowa.
    """
    original = str(value or "")
    text = unicodedata.normalize("NFC", original)
    text = "".join(ch for ch in text if ch not in INVISIBLE_FORMATTING)
    text = _WHITESPACE.sub(" ", text).strip()
    return text if text else original
