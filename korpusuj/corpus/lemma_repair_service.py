# -*- coding: utf-8 -*-
"""Wspolna usluga korekty lematow uzywana przez CLI i przyszly adapter GUI."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any
from .lemma_repair_models import LemmaRepairOptions, LemmaRepairPaths, LemmaRepairResult
from .lemma_repair_audit import run_audit
from .lemma_repair_pipeline import prepare_policy
from .lemma_repair_apply import preview, apply

def _status(paths, stage, data):
    value={"schema_version":1,"updated_at":datetime.now(timezone.utc).isoformat(),"stage":stage,**data}
    paths.status_json().write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    return value

def audit(paths: LemmaRepairPaths, options: LemmaRepairOptions, reporter: Any=None):
    data=run_audit(paths,options,reporter); _status(paths,"audit_complete",data); return LemmaRepairResult(True,"audit_complete",str(paths.status_json()),data)

def prepare(paths: LemmaRepairPaths, options: LemmaRepairOptions, reporter: Any=None):
    data=prepare_policy(paths,options,reporter); _status(paths,"decisions_ready",data); return LemmaRepairResult(True,"decisions_ready",str(paths.status_json()),data)

def dry_run(paths: LemmaRepairPaths, options: LemmaRepairOptions, reporter: Any=None):
    data=preview(paths,options.batch_size,reporter); _status(paths,"preview_complete",data); return LemmaRepairResult(True,"preview_complete",str(paths.status_json()),data)

def apply_approved(paths: LemmaRepairPaths, options: LemmaRepairOptions, reporter: Any=None):
    data=apply(paths,options.batch_size,reporter); _status(paths,"apply_complete",data); return LemmaRepairResult(True,"apply_complete",str(paths.status_json()),data)

def run(paths: LemmaRepairPaths, options: LemmaRepairOptions, reporter: Any=None, apply_changes: bool=False):
    if not paths.audit_json().exists(): audit(paths,options,reporter)
    if not paths.decisions_auto().exists(): prepare(paths,options,reporter)
    dry_run(paths,options,reporter)
    return apply_approved(paths,options,reporter) if apply_changes else LemmaRepairResult(True,"preview_complete",str(paths.status_json()),json.loads(paths.preview_json().read_text(encoding="utf-8")))
