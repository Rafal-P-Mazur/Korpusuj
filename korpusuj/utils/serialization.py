# -*- coding: utf-8 -*-
from __future__ import annotations
import ast, json, zlib
from typing import Any
import pandas as pd

def _json_zlib_dumps(obj: Any, level: int = 1) -> bytes:
    return zlib.compress(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), level=level)
def _json_zlib_loads(blob: bytes | None, default=None):
    if blob is None: return default
    try: return json.loads(zlib.decompress(blob).decode("utf-8"))
    except Exception: return default
def _as_plain_list(value):
    if value is None: return []
    try:
        if hasattr(value, "tolist"): value=value.tolist()
    except Exception: pass
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    if isinstance(value, str):
        s=value.strip()
        if not s: return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed=ast.literal_eval(s)
                if isinstance(parsed,(list,tuple)): return list(parsed)
            except Exception: pass
        return [value]
    try:
        if pd.isna(value): return []
    except Exception: pass
    return [value]
def _safe_scalar(value):
    try:
        if hasattr(value,"tolist"): value=value.tolist()
    except Exception: pass
    if isinstance(value,(list,tuple,dict,set)): return value
    try:
        if pd.isna(value): return ""
    except Exception: pass
    return value
