# -*- coding: utf-8 -*-
"""Postprocessing lematow wykonywany po utworzeniu Parquetu kreatora."""
from __future__ import annotations
import json, os, shutil
from pathlib import Path
from typing import Any
from korpusuj.corpus.lemma_repair_models import LemmaRepairOptions, LemmaRepairPaths
from korpusuj.corpus import lemma_repair_service
from korpusuj.dependency.lifecycle import build_index_artifacts_atomic

VALID_CREATOR_LEMMA_REPAIR_MODES={"off","common-core","common-core-plus"}

def _progress(reporter):
    def callback(*args, **kwargs):
        value=kwargs.get("progress")
        if value is None and args:
            value=args[-1] if isinstance(args[-1],(int,float)) else None
        if value is not None:
            try: reporter.total(float(value))
            except Exception: pass
        reporter.tick()
    return callback

def run_creator_lemma_repair(parquet_path: str, mode: str, reporter: Any) -> dict:
    """Build working index, repair canonical Parquet and rebuild final artifacts.

    The source Parquet is retained inside the project directory only until the
    repaired corpus and fresh derived artifacts have been published. On an
    error the source is restored. No system TEMP path is used.
    """
    if mode not in VALID_CREATOR_LEMMA_REPAIR_MODES:
        raise ValueError(f"Nieznany tryb korekty lematyzacji: {mode}")
    if mode=="off": return {"mode":"off","applied":False}
    parquet=Path(parquet_path).resolve(); search=parquet.with_suffix(".search")
    workdir=parquet.parent/(parquet.stem+".lemma_repair")
    repaired=parquet.with_name(parquet.name+".lemma_repair_stage")
    original=parquet.with_name(parquet.name+".lemma_repair_source_stage")
    if repaired.exists() or original.exists():
        raise RuntimeError("Istnieja pliki etapowe korekty lematyzacji; usun je lub dokoncz odzyskiwanie.")
    reporter.status("Budowanie indeksu roboczego do kontroli lematyzacji...")
    build_index_artifacts_atomic(str(parquet),str(search),progress_callback=_progress(reporter))
    paths=LemmaRepairPaths.build(parquet,search,workdir,repaired)
    options=LemmaRepairOptions(mode=mode,batch_size=128,resume=True)
    reporter.status("Analiza i weryfikacja lematyzacji przez SGJP...")
    result=lemma_repair_service.run(paths,options,reporter,apply_changes=True)
    apply_data=dict(result.data or {})
    if not repaired.is_file(): raise RuntimeError("Korekta nie utworzyla wynikowego Parquetu etapowego.")
    reporter.status("Publikowanie poprawionego korpusu i przebudowa indeksu...")
    os.replace(parquet,original); os.replace(repaired,parquet)
    try:
        build_index_artifacts_atomic(str(parquet),str(search),progress_callback=_progress(reporter))
    except Exception:
        failed=parquet.with_name(parquet.name+".lemma_repair_failed")
        if failed.exists(): failed.unlink()
        os.replace(parquet,failed); os.replace(original,parquet)
        build_index_artifacts_atomic(str(parquet),str(search),progress_callback=_progress(reporter))
        raise
    else:
        original.unlink(missing_ok=True)
        failed=parquet.with_name(parquet.name+".lemma_repair_failed")
        failed.unlink(missing_ok=True)
    summary={"mode":mode,"applied":True,"parquet":str(parquet),"search":str(search),
             "workdir":str(workdir),"apply":apply_data}
    (workdir/(parquet.stem+".creator_summary.json")).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    reporter.status("Korpus, korekta lematyzacji i indeks sa gotowe.")
    return summary
