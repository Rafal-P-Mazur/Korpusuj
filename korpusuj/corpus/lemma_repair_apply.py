# -*- coding: utf-8 -*-
"""Dry-run i atomowe zastosowanie decyzji D3 do Parquetu Korpusuj."""
from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path
from typing import Any
import pyarrow.parquet as pq
from .lemma_repair_models import LemmaRepairError,LemmaRepairPaths
from . import lemma_repair_rewrite as engine
from .lemma_repair_rules import ner_broad, rule_context_matches
from korpusuj.utils.text_normalization import sanitize_stored_lemma

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def _key(r):return (str(r["orth"]),str(r["lemma"]),str(r["upos"]).upper(),str(r.get("morph_from") or ""),str(r.get("required_ner_broad") or "").upper())
def _id(r):return "|".join((*_key(r),str(r["replacement"]),str(r.get("morph_to") or "")))
def _load(paths):
 p=json.loads(paths.decisions_auto().read_text(encoding="utf-8"))
 if p.get("tool")!="prepare_d3_lemma_repair_decisions" or p.get("bucket")!="auto":raise LemmaRepairError("Plik decyzji nie jest pula AUTO generatora D3.")
 if (p.get("source") or {}).get("sha256")!=_sha(paths.parquet):raise LemmaRepairError("SHA-256 Parquetu nie zgadza sie z plikiem decyzji.")
 out={}
 for r in [x for x in p.get("rules",[]) if x.get("status")=="accept"]:
  k=_key(r);prev=out.get(k);target=(str(r["replacement"]),str(r.get("morph_to") or ""))
  if prev and (str(prev["replacement"]),str(prev.get("morph_to") or ""))!=target:raise LemmaRepairError(f"Konflikt celow dla reguly: {k}")
  out[k]=r
 if not out:raise LemmaRepairError("Brak zaakceptowanych regul.")
 return p,out
def _match(rules,o,l,u,m,n):
 broad=ner_broad(n)
 return rules.get((o,l,u,m,broad)) or rules.get((o,l,u,"",broad)) or rules.get((o,l,u,m,"")) or rules.get((o,l,u,"",""))
def preview(paths:LemmaRepairPaths,batch_size:int=128,reporter:Any=None):
 payload,rules=_load(paths)
 if reporter:reporter.status("Sprawdzanie dopasowan korekt bez zapisu...")
 counts=Counter();docs=tokens=affected=morph_changes=sanitized_lemmas=empty_after_sanitization=0;pf=pq.ParquetFile(paths.parquet);cols=set(pf.schema_arrow.names);mc="full_postags" if "full_postags" in cols else ("postags" if "postags" in cols else None);nc="ners" if "ners" in cols else ("ner" if "ner" in cols else None);read=["tokens","lemmas","upostags"]+([mc] if mc else [])+([nc] if nc else [])
 try:
  for batch in pf.iter_batches(batch_size=max(1,batch_size),columns=read):
   data=batch.to_pydict();mrs=data[mc] if mc else [None]*batch.num_rows;nrs=data[nc] if nc else [None]*batch.num_rows
   for ts,ls,us,ms,ns in zip(data["tokens"],data["lemmas"],data["upostags"],mrs,nrs):
    docs+=1;changed=False;ts=list(ts or []);ls=list(ls or []);us=list(us or []);ms=list(ms or [""]*len(ts));ns=list(ns or ["O"]*len(ts))
    if not(len(ts)==len(ls)==len(us)==len(ms)==len(ns)):raise LemmaRepairError(f"Nierownolegle tablice w dokumencie {docs-1}")
    for pos,(o,l,u,m,n) in enumerate(zip(ts,ls,us,ms,ns)):
     tokens+=1;original_lemma=str(l or "");sanitized=sanitize_stored_lemma(original_lemma)
     if sanitized!=original_lemma:sanitized_lemmas+=1;changed=True
     if not sanitized:empty_after_sanitization+=1
     r=_match(rules,str(o).strip(),original_lemma.strip(),str(u).strip().upper(),str(m or "").strip(),n)
     if r is None or not rule_context_matches(r,n,docs-1,pos):continue
     counts[_id(r)]+=1;changed=True
     if str(r.get("morph_to") or "") and str(r.get("morph_to"))!=str(m or "").strip():morph_changes+=1
    if changed:affected+=1
 finally:pf.close()
 expected={_id(r):int(r.get("observed_count") or 0) for r in rules.values()};unmatched=sorted(k for k in expected if counts[k]==0);mismatched={k:{"expected":expected[k],"observed":counts[k]} for k in expected if expected[k] and expected[k]!=counts[k]};result={"schema_version":2,"stage":"preview_complete","source":str(paths.parquet),"source_sha256":_sha(paths.parquet),"accepted_rules":len(rules),"matched_rules":sum(counts[k]>0 for k in expected),"unmatched_rules":unmatched,"count_mismatches":mismatched,"matched_tokens":sum(counts.values()),"sanitized_lemmas":sanitized_lemmas,"empty_after_sanitization":empty_after_sanitization,"morph_tag_changes":morph_changes,"affected_documents":affected,"documents":docs,"tokens":tokens,"counts_by_rule":dict(sorted(counts.items()))};paths.preview_json().write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");return result
def apply(paths:LemmaRepairPaths,batch_size:int=128,reporter:Any=None):
 if paths.output is None:raise LemmaRepairError("Brak sciezki wynikowego Parquetu.")
 check=preview(paths,batch_size,reporter)
 if check["unmatched_rules"] or check["count_mismatches"]:raise LemmaRepairError("Dry-run wykazal reguly bez trafien lub rozbieznosci licznikow. Wyniku nie zapisano.")
 if reporter:reporter.status("Stosowanie bezpiecznych korekt lematow i tagow...")
 accepted,payload=engine.load_decisions(paths.decisions_auto(),paths.parquet);result=engine.rewrite_with_rules(paths.parquet,paths.output,paths.decisions_auto(),accepted,payload,batch_size=batch_size);paths.apply_json().write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");return result
