# -*- coding: utf-8 -*-
"""Pure text chunking helpers for corpus creation."""
from __future__ import annotations

import re

# --- CHUNKING HELPERS: LOSSLESS + MULTI-STYLE RECORD-AWARE ---

# NUMERIC:
# Pozostaje "inline-capable", bo w części materiałów (jak 00898.txt)
# rekordy typu "1. Austria ... 2. Belgia ..." występują w jednym długim bloku,
# a niekoniecznie od nowej linii.
NUMERIC_RECORD_START_RE = re.compile(
    r'(?:^|(?<=\s))(\d{1,3}\.\s*[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# LETTERED:
# Tylko początek linii, żeby nie łapać przypadkowych "a)" w środku zdania.
LETTER_RECORD_START_RE = re.compile(
    r'(?m)^(?:\s{0,3})([a-zA-Z][\)\.]\s+[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# BULLETS:
# Tylko początek linii.
# Nie wymagamy Wielkiej Litery po markerze, bo w polskich wyliczeniach
# po dwukropku elementy listy często zaczynają się od małej litery.
BULLET_RECORD_START_RE = re.compile(
    r'(?m)^(?:\s{0,3})([-*•])(?=\s+\S)'
)

# ROMAN:
# Tylko początek linii.
ROMAN_RECORD_START_RE = re.compile(
    r'(?m)^(?:\s{0,3})([IVXLCDM]{1,8}[\)\.]\s+[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# Silny koniec zdania
STRONG_SENT_END_RE = re.compile(
    r'(?<!\bul)(?<!\bUl)(?<!\bnp)(?<!\bNp)(?<!\btzw)(?<!\bdr)(?<!\bks)(?<!prof)(?<!mgr)(?<!m\.in)(?<!\bal)(?<!\bAl)(?<!\blok)(?<!\bpl)(?<!\bPl)[.!?][\'")\]]*(?:\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])|$)'
)

# Separator pól dla rekordów półtabelarycznych
FIELD_BREAK_RE = re.compile(
    r'(?:\s[-–—]\s|;\s+|:\s+)'
)

# NOWE: Ukryta granica listy (np. "prawnik Izabela Aleksandra")
# Szuka: mała litera -> spacja -> (Wielka litera + małe) -> spacja -> (Wielka litera)
IMPLIED_RECORD_BREAK_RE = re.compile(
    r'(?<=[a-ząćęłńóśźż])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ])'
)

# Słabszy fallback po interpunkcji
SOFT_BREAK_RE = re.compile(
    r'[.!?,;:][\'")\]]*\s+'
)


def has_multiline_bullet_layout(text: str, min_items: int = 4) -> bool:
    """
    Dodatkowa heurystyka dla bulletów:
    sprawdza, czy tekst faktycznie zawiera wieloliniowy układ listy,
    a nie pojedyncze myślniki / gwiazdki przypadkowo występujące w tekście.
    """
    bullet_lines = 0
    for line in text.splitlines():
        if BULLET_RECORD_START_RE.match(line):
            bullet_lines += 1
            if bullet_lines >= min_items:
                return True
    return False


def detect_record_style(text: str) -> str | None:
    """
    Wykrywa styl listy / rekordów.
    Zwraca jedną z wartości:
      - "numeric"
      - "letter"
      - "bullet"
      - "roman"
      - None
    """
    text_len = max(len(text), 1)

    numeric_count = len(list(NUMERIC_RECORD_START_RE.finditer(text)))
    letter_count = len(list(LETTER_RECORD_START_RE.finditer(text)))
    bullet_count = len(list(BULLET_RECORD_START_RE.finditer(text)))
    roman_count = len(list(ROMAN_RECORD_START_RE.finditer(text)))

    numeric_density = numeric_count / text_len * 10000.0
    letter_density = letter_count / text_len * 10000.0
    bullet_density = bullet_count / text_len * 10000.0
    roman_density = roman_count / text_len * 10000.0

    # Najwyższy priorytet: numeric, bo jest najbardziej precyzyjny
    # i pokrywa przypadki "1. Polska ... 2. Czechy ..." także inline.
    if numeric_count >= 5 and numeric_density >= 4.0:
        return "numeric"

    # Litery: ostrożniej
    if letter_count >= 5 and letter_density >= 4.5:
        return "letter"

    # Rzymskie: zwykle rzadsze, ale dość charakterystyczne
    if roman_count >= 4 and roman_density >= 3.5:
        return "roman"

    # Bullets: najbardziej wieloznaczne, więc oprócz count/density
    # wymagamy też rzeczywistego układu wieloliniowej listy.
    if (
        bullet_count >= 4
        and bullet_density >= 4.0
        and has_multiline_bullet_layout(text, min_items=4)
    ):
        return "bullet"

    return None


def get_record_start_regex(style: str):
    """
    Zwraca regex startu rekordu dla zadanego stylu.
    """
    if style == "numeric":
        return NUMERIC_RECORD_START_RE
    if style == "letter":
        return LETTER_RECORD_START_RE
    if style == "bullet":
        return BULLET_RECORD_START_RE
    if style == "roman":
        return ROMAN_RECORD_START_RE
    return None


def split_structured_segments(text: str, style: str) -> tuple[str, list[str]]:
    """
    Dzieli tekst na:
      - preambułę (wszystko przed pierwszym rekordem),
      - listę rekordów.

    LOSSLESS:
      - niczego nie trimuje,
      - niczego nie normalizuje,
      - nie podmienia newline na spacje.
    """
    rx = get_record_start_regex(style)
    if rx is None:
        return text, []

    matches = list(rx.finditer(text))
    if not matches:
        return text, []

    starts = [m.start(1) for m in matches]
    preamble = text[:starts[0]]
    records = []

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        records.append(text[start:end])

    return preamble, records


def soft_cut_preserve(
    block: str,
    limit: int,
    backtrack_window: int = 800,
) -> list[str]:
    """
    Miękkie cięcie BLOKU bez zmiany treści (LOSSLESS).

    Priorytet cięcia:
      1) silny koniec zdania,
      2) separator pola,
      3) słabsza interpunkcja,
      4) ostatnia spacja,
      5) hard cut.
    """
    out: list[str] = []
    start = 0
    L = len(block)

    while start < L:
        remaining = L - start
        if remaining <= limit:
            out.append(block[start:])
            break

        end = start + limit
        search_start = max(start, end - backtrack_window)
        window = block[search_start:end]

        cut = -1
        last_match = None

        # 1) Silny koniec zdania
        for m in STRONG_SENT_END_RE.finditer(window):
            last_match = m
        if last_match:
            cut = search_start + last_match.end()
        else:
            # 2) Separator pola
            last_match = None
            for m in FIELD_BREAK_RE.finditer(window):
                last_match = m
            if last_match:
                cut = search_start + last_match.end()
            else:
                # 2.5) Ukryta granica listy (bezinterpunkcyjne przejście na Imię i Nazwisko)
                last_match = None
                for m in IMPLIED_RECORD_BREAK_RE.finditer(window):
                    last_match = m
                if last_match:
                    # Tniemy na spacji (start dopasowania), żeby nie ucinać Imienia
                    cut = search_start + last_match.start()
                else:
                    # 3) Słabsza interpunkcja
                    last_match = None
                    for m in SOFT_BREAK_RE.finditer(window):
                        last_match = m
                    if last_match:
                        cut = search_start + last_match.end()
                    else:
                        # 4) Ostatnia spacja
                        last_space = window.rfind(" ")
                        if last_space != -1:
                            cut = search_start + last_space
                        else:
                            # 5) Twarde cięcie
                            cut = end

        if cut <= start:
            cut = end

        out.append(block[start:cut])
        start = cut

    return out


def chunk_structured_records(
    text: str,
    chunk_size: int,
    style: str = "numeric",
    structured_chunk_size: int = 4000,
    max_records_per_chunk: int = 6,
    backtrack_window: int = 800,
    max_dotless_chars: int = 800,       # <-- NOWE: Przekazany próg tasiemca
    min_piece_in_danger: int = 1200     # <-- NOWE: Przekazany bezpieczny limit
) -> list[str]:
    """
    Dedykowane, LOSSLESS cięcie dla tekstów rekordowych / wyliczeniowych.
    Teraz z dynamicznym progiem bezpieczeństwa dla "tasiemców".
    """
    effective_limit = min(chunk_size, structured_chunk_size)
    preamble, records = split_structured_segments(text, style=style)

    out: list[str] = []

    # --- NOWE: Dynamiczny estymator limitu dla konkretnego bloku ---
    def get_dynamic_limit(block: str) -> int:
        last_end = 0
        for m in STRONG_SENT_END_RE.finditer(block):
            if m.start() - last_end > max_dotless_chars:
                return min(effective_limit, min_piece_in_danger)
            last_end = m.end()
        # Sprawdzamy resztówkę po ostatniej kropce
        if len(block) - last_end > max_dotless_chars:
            return min(effective_limit, min_piece_in_danger)
        return effective_limit
    # ---------------------------------------------------------------

    # 1) Preambuła
    if preamble:
        limit = get_dynamic_limit(preamble)
        if len(preamble) <= limit:
            out.append(preamble)
        else:
            out.extend(
                soft_cut_preserve(
                    preamble,
                    limit=limit,
                    backtrack_window=backtrack_window,
                )
            )

    # 2) Rekordy
    current_parts: list[str] = []
    current_len = 0
    current_records = 0

    def flush():
        nonlocal current_parts, current_len, current_records
        if current_parts:
            out.append("".join(current_parts))
            current_parts = []
            current_len = 0
            current_records = 0

    for rec in records:
        limit = get_dynamic_limit(rec) # Badamy konkretny rekord

        # Rekord większy niż jego dynamiczny limit -> tniemy miękko
        if len(rec) > limit:
            flush()
            out.extend(
                soft_cut_preserve(
                    rec,
                    limit=limit,
                    backtrack_window=backtrack_window,
                )
            )
            continue

        # Ograniczenie po rozmiarze i liczbie rekordów (uwzględnia dynamiczny limit!)
        if current_parts and (
            current_len + len(rec) > limit
            or current_records >= max_records_per_chunk
        ):
            flush()

        current_parts.append(rec)
        current_len += len(rec)
        current_records += 1

    flush()
    return out


def chunk_text_safe(
    text: str,
    chunk_size: int = 15000,
    max_dotless_chars: int = 800,
    backtrack_window: int = 600,
    min_piece_in_danger: int = 1200,
    structured_chunk_size: int = 4000,
    max_records_per_chunk: int = 6,
) -> list[str]:
    """
    Dzieli tekst na NLP-przyjazne fragmenty.
    """
    # 1) ROUTING: tekst rekordowy / wyliczeniowy
    style = detect_record_style(text)
    if style is not None:
        return chunk_structured_records(
            text,
            chunk_size=chunk_size,
            style=style,
            structured_chunk_size=structured_chunk_size,
            max_records_per_chunk=max_records_per_chunk,
            backtrack_window=max(backtrack_window, 800),
            max_dotless_chars=max_dotless_chars,       # <-- PRZEKAZUJEMY
            min_piece_in_danger=min_piece_in_danger    # <-- PRZEKAZUJEMY
        )

    # 2) ZWYKŁY TEKST
    ANY_PUNCT_GAP_RE = re.compile(r'[.!?,\-;—–:][\'")\]]*\s+')
    PARA_END_RE = re.compile(
        r'(?<!\b[a-ząćęłńóśźż])(?<!\b[a-ząćęłńóśźż]{2})(?<!\b[a-ząćęłńóśźż]{3})[.!?][\'")\]]*\s*$'
    )

    chunks: list[str] = []
    paragraphs = text.split("\n")
    current_chunk_parts: list[str] = []
    current_len = 0
    dotless = 0

    def flush():
        nonlocal current_chunk_parts, current_len, dotless
        if current_chunk_parts:
            chunks.append("".join(current_chunk_parts))
            current_chunk_parts = []
            current_len = 0
            dotless = 0

    def soft_cut(block: str, limit: int) -> list[str]:
        """
        LOSSLESS cięcie zwykłego tekstu.
        """
        out = []
        start = 0
        L = len(block)

        while start < L:
            remaining = L - start
            if remaining <= limit:
                out.append(block[start:])
                break

            end = start + limit
            search_start = max(start, end - backtrack_window)
            window = block[search_start:end]

            m = None
            cut = -1

            # Krok 1: silny koniec zdania
            for mm in STRONG_SENT_END_RE.finditer(window):
                m = mm

            if m:
                cut = search_start + m.end()
            else:
                # Krok 2: słabsza interpunkcja
                for mm in ANY_PUNCT_GAP_RE.finditer(window):
                    m = mm
                if m:
                    cut = search_start + m.end()
                else:
                    # Krok 3: ostatnia spacja
                    last_space = window.rfind(" ")
                    if last_space != -1:
                        cut = search_start + last_space
                    else:
                        # Krok 4: twarde cięcie
                        fallback_space = block[start:end].rfind(" ")
                        cut = start + fallback_space if fallback_space != -1 else end

            if cut <= start:
                cut = end

            out.append(block[start:cut])
            start = cut

        return out

    for i, para in enumerate(paragraphs):
        # split("\n") usuwa separator, więc dokładamy go tylko tam,
        # gdzie istniał w oryginale.
        para_full = para + ("\n" if i < len(paragraphs) - 1 else "")
        para_len = len(para_full)

        # Wykrywanie tasiemca na podstawie silnych końców zdań
        dangerous = False
        last_end = 0

        for m in STRONG_SENT_END_RE.finditer(para_full):
            if m.start() - last_end > max_dotless_chars:
                dangerous = True
                break
            last_end = m.end()

        if not dangerous and (para_len - last_end) > max_dotless_chars:
            dangerous = True

        # Niebezpieczny albo za duży akapit -> miękkie cięcie
        if dangerous or para_len > chunk_size:
            flush()
            target_limit = chunk_size if not dangerous else min(chunk_size, min_piece_in_danger)
            for piece in soft_cut(para_full, limit=target_limit):
                chunks.append(piece)
            continue

        # Normalne sklejanie akapitów
        if not PARA_END_RE.search(para_full.strip()):
            dotless += para_len
        else:
            dotless = 0

        if dotless > max_dotless_chars and current_len > 0:
            flush()
            current_chunk_parts.append(para_full)
            current_len = para_len
            dotless = 0 if PARA_END_RE.search(para_full.strip()) else para_len
            continue

        if current_len + para_len > chunk_size:
            flush()

        current_chunk_parts.append(para_full)
        current_len += para_len

    flush()
    return chunks

__all__ = [
    'NUMERIC_RECORD_START_RE',
    'LETTER_RECORD_START_RE',
    'BULLET_RECORD_START_RE',
    'ROMAN_RECORD_START_RE',
    'STRONG_SENT_END_RE',
    'FIELD_BREAK_RE',
    'IMPLIED_RECORD_BREAK_RE',
    'SOFT_BREAK_RE',
    'has_multiline_bullet_layout',
    'detect_record_style',
    'get_record_start_regex',
    'split_structured_segments',
    'soft_cut_preserve',
    'chunk_structured_records',
    'chunk_text_safe',
]
