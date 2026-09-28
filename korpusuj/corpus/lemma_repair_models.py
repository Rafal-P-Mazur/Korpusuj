# -*- coding: utf-8 -*-
"""Kontrakty wspolnej uslugi korekty lematow D3."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_MODES = {"off", "common-core", "common-core-plus", "review"}

@dataclass(slots=True)
class LemmaRepairPaths:
    parquet: Path
    search: Path
    workspace: Path
    artifact_prefix: Path
    output: Path | None = None

    @classmethod
    def build(cls, parquet: str | Path, search: str | Path, workdir: str | Path | None = None, output: str | Path | None = None):
        p = Path(parquet).expanduser().resolve(); s = Path(search).expanduser().resolve()
        root = Path(workdir).expanduser().resolve() if workdir else p.parent / (p.stem + ".lemma_repair")
        root.mkdir(parents=True, exist_ok=True); prefix = root / p.stem
        return cls(p, s, prefix.with_suffix(".workspace.sqlite"), prefix, Path(output).expanduser().resolve() if output else None)

    def audit_json(self) -> Path: return self.artifact_prefix.with_suffix(".audit.json")
    def audit_export_md(self) -> Path: return self.artifact_prefix.with_suffix(".audit.md")
    def decisions_auto(self) -> Path: return self.artifact_prefix.with_suffix(".auto.json")
    def decisions_review(self) -> Path: return self.artifact_prefix.with_suffix(".review.json")
    def decisions_rejected(self) -> Path: return self.artifact_prefix.with_suffix(".rejected.json")
    def sample_export_md(self) -> Path: return self.artifact_prefix.with_suffix(".sample.md")
    def summary_export_md(self) -> Path: return self.artifact_prefix.with_suffix(".summary.md")
    def status_json(self) -> Path: return self.artifact_prefix.with_suffix(".status.json")
    def preview_json(self) -> Path: return self.artifact_prefix.with_suffix(".preview.json")
    def apply_json(self) -> Path: return self.artifact_prefix.with_suffix(".apply.json")

@dataclass(slots=True)
class LemmaRepairOptions:
    mode: str = "common-core"
    resume: bool = True
    batch_size: int = 128
    commit_docs: int = 100
    min_source_count: int = 2
    min_source_docs: int = 2
    max_source_forms: int = 2
    examples_per_rule: int = 4
    context_tokens: int = 12
    seed: int = 20260928
    export_diagnostics: bool = False
    def __post_init__(self):
        if self.mode not in VALID_MODES: raise ValueError(f"Nieznany tryb korekty lematow: {self.mode}")

@dataclass(slots=True)
class LemmaRepairResult:
    success: bool
    stage: str
    status_path: str
    data: dict[str, Any] = field(default_factory=dict)

class LemmaRepairError(RuntimeError): pass
