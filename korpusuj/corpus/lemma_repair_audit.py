# -*- coding: utf-8 -*-
"""Adapter produkcyjny audytu D3 oparty na wznawialnym workspace SQLite."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .lemma_repair_models import LemmaRepairError, LemmaRepairOptions, LemmaRepairPaths
from . import lemma_repair_analysis as engine

def run_audit(paths: LemmaRepairPaths, options: LemmaRepairOptions, reporter: Any = None) -> dict:
    if not paths.parquet.is_file(): raise LemmaRepairError(f"Brak Parquetu: {paths.parquet}")
    if not paths.search.is_file(): raise LemmaRepairError(f"Brak indeksu .search: {paths.search}")
    if reporter: reporter.status("Sprawdzanie lematow w indeksie korpusu...")
    argv = ["--search", str(paths.search), "--parquet", str(paths.parquet),
            "--workspace", str(paths.workspace), "--report", str(paths.audit_export_md()),
            "--json", str(paths.audit_json()), "--commit-docs", str(options.commit_docs),
            "--min-source-count", str(options.min_source_count), "--min-source-docs", str(options.min_source_docs),
            "--max-source-forms", str(options.max_source_forms), "--examples-per-rule", str(options.examples_per_rule),
            "--context-tokens", str(options.context_tokens)]
    if options.resume and paths.workspace.exists(): argv.append("--resume")
    try: engine.main(argv)
    except SystemExit as exc:
        if int(exc.code or 0) != 0: raise LemmaRepairError(f"Audyt D3 zakonczyl sie kodem {exc.code}")
    import json
    payload = json.loads(paths.audit_json().read_text(encoding="utf-8"))
    return payload.get("summary", {})
