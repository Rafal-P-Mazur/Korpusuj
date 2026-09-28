# -*- coding: utf-8 -*-
"""Dry-run i atomowe zastosowanie decyzji D3 do Parquetu Korpusuj."""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any
import pyarrow.parquet as pq
from .lemma_repair_models import LemmaRepairError, LemmaRepairPaths
from . import _lemma_repair_apply_engine as engine

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024), b""): h.update(block)
    return h.hexdigest()

def _load(paths: LemmaRepairPaths):
    payload=json.loads(paths.decisions_auto().read_text(encoding="utf-8"))
    if payload.get("tool") != "prepare_d3_lemma_repair_decisions" or payload.get("bucket") != "auto":
        raise LemmaRepairError("Plik decyzji nie jest pula AUTO generatora D3.")
    if (payload.get("source") or {}).get("sha256") != _sha(paths.parquet):
        raise LemmaRepairError("SHA-256 Parquetu nie zgadza sie z plikiem decyzji.")
    rules=[r for r in payload.get("rules",[]) if r.get("status")=="accept"]
    keys={}
    for r in rules:
        key=(str(r["orth"]),str(r["lemma"]),str(r["upos"]).upper())
        if key in keys and keys[key]["replacement"] != r["replacement"]:
            raise LemmaRepairError(f"Konflikt celow dla reguly: {key}")
        keys[key]=r
    if not keys: raise LemmaRepairError("Brak zaakceptowanych regul.")
    return payload, keys

def preview(paths: LemmaRepairPaths, batch_size: int=128, reporter: Any=None) -> dict:
    payload, rules = _load(paths)
    if reporter: reporter.status("Sprawdzanie dopasowan korekt bez zapisu...")
    counts=Counter(); docs=0; tokens=0; affected=0
    pf=pq.ParquetFile(paths.parquet)
    try:
        for batch in pf.iter_batches(batch_size=max(1,batch_size), columns=["tokens","lemmas","upostags"]):
            data=batch.to_pydict()
            for toks,lems,ups in zip(data["tokens"],data["lemmas"],data["upostags"]):
                docs+=1; changed=False
                if not(len(toks)==len(lems)==len(ups)): raise LemmaRepairError(f"Nierownolegle tablice w dokumencie {docs-1}")
                for o,l,u in zip(toks,lems,ups):
                    tokens+=1; key=(str(o).strip(),str(l).strip(),str(u).strip().upper())
                    if key in rules:
                        counts["|".join((*key,str(rules[key]["replacement"])))] += 1; changed=True
                if changed: affected+=1
    finally: pf.close()
    expected={"|".join((r["orth"],r["lemma"],r["upos"],r["replacement"])):int(r.get("observed_count") or 0) for r in rules.values()}
    unmatched=sorted(k for k in expected if counts[k]==0)
    mismatched=sorted({k:{"expected":expected[k],"observed":counts[k]} for k in expected if expected[k] and expected[k]!=counts[k]}.items())
    result={"schema_version":1,"stage":"preview_complete","source":str(paths.parquet),"source_sha256":_sha(paths.parquet),
            "accepted_rules":len(rules),"matched_rules":sum(1 for k in expected if counts[k]>0),"unmatched_rules":unmatched,
            "count_mismatches":dict(mismatched),"matched_tokens":sum(counts.values()),"affected_documents":affected,
            "documents":docs,"tokens":tokens,"counts_by_rule":dict(sorted(counts.items()))}
    paths.preview_json().write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

def apply(paths: LemmaRepairPaths, batch_size: int=128, reporter: Any=None) -> dict:
    if paths.output is None: raise LemmaRepairError("Brak sciezki wynikowego Parquetu.")
    check=preview(paths,batch_size,reporter)
    if check["unmatched_rules"] or check["count_mismatches"]:
        raise LemmaRepairError("Dry-run wykazal reguly bez trafien lub rozbieznosci licznikow. Wyniku nie zapisano.")
    if reporter: reporter.status("Stosowanie bezpiecznych korekt lematow...")
    accepted,payload=engine.load_decisions(paths.decisions_auto(), paths.parquet)
    result=engine.rewrite_with_rules(paths.parquet,paths.output,paths.decisions_auto(),accepted,payload,batch_size=batch_size)
    paths.apply_json().write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result
