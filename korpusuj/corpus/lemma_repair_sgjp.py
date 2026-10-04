# -*- coding: utf-8 -*-
"""SGJP-based lemma repair: direct evidence, contextual evidence and finalization.

The implementation is consolidated from the former direct and contextual
modules without changing function bodies or decision policy.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import lemma_repair_analysis as d3
from .lemma_repair_models import LemmaRepairError, LemmaRepairPaths

ALLOWED_UPOS={"NOUN","VERB","ADJ"}

def _clean(x): return str(x or "").strip()

def _key(r): return (_clean(r.get("orth")),_clean(r.get("lemma")),_clean(r.get("upos")).upper())

def _letters(value): return bool(value and all(ch.isalpha() for ch in value))

def _load_analysis(con, morfeusz, orth):
    row=con.execute("SELECT status,analyses_json FROM sgjp_cache WHERE orth=?",(orth,)).fetchone()
    if row:
        try:
            cached=json.loads(row[1])
            for analysis in cached:
                analysis["lemma"]=d3.normalize_lemma(analysis.get("lemma"))
            return row[0],cached
        except Exception:
            pass
    status,analyses=d3.analyse_form(morfeusz,orth)
    for analysis in analyses:
        analysis["lemma"]=d3.normalize_lemma(analysis.get("lemma"))
    con.execute("INSERT OR REPLACE INTO sgjp_cache VALUES(?,?,?)",(orth,status,json.dumps(analyses,ensure_ascii=False)))
    return status,analyses

SYNCRETIC_CASE_SETS={frozenset({"nom","acc"}),frozenset({"gen","acc"})}

def _safe_syncretic_noun(morph_counts,analyses,target):
    parsed=[d3.parse_tag(a.get("tag")) for a in analyses if _clean(a.get("lemma"))==target and _clean(a.get("upos")).upper()=="NOUN"]
    if not parsed:return False
    numbers={p.get("number","") for p in parsed};genders={p.get("gender","") for p in parsed};cases=set().union(*[{*str(p.get("case","")).split(".")} for p in parsed])
    if len(numbers)!=1 or len(genders)!=1 or frozenset(cases) not in SYNCRETIC_CASE_SETS:return False
    observed={d3.parse_tag(tag).get("case","") for tag in morph_counts}
    return bool(observed) and observed.issubset(cases)

def _direct_rule(orth,lemma,upos,count,docs,morph_counts,analyses):
    if upos not in ALLOWED_UPOS or count<2 or docs<2: return None
    if not (orth[0].islower() and lemma[0].islower()): return None
    if not (_letters(orth) and _letters(lemma)): return None
    # Najpierw kontrola globalna, niezalezna od UPOS modelu. Jezeli SGJP
    # potwierdza obecny lemat w dowolnej kategorii, nie poprawiamy go. Chroni
    # to np. "mam + miec + NOUN" przed zmiana na "mama" przy blednym UPOS.
    lexical_analyses = [
        a for a in analyses
        if _clean(a.get("lemma")) and _clean(a.get("upos")).upper() not in {"PUNCT", "SYM", "X"}
    ]
    all_lemmas = {_clean(a.get("lemma")) for a in lexical_analyses}
    if lemma in all_lemmas:
        return None

    same_upos=[a for a in lexical_analyses if _clean(a.get("upos")).upper()==upos]
    targets=sorted({_clean(a.get("lemma")) for a in same_upos if _clean(a.get("lemma"))})
    if len(targets)!=1:
        return None
    target=targets[0]

    # Forma homograficzna miedzy czesciami mowy nie moze byc automatycznie
    # rozstrzygana na podstawie UPOS modelu, bo ten sam UPOS moze byc bledny.
    # Do AUTO trafia tylko forma, dla ktorej wszystkie leksykalne analizy SGJP
    # prowadza do jednego lematu globalnie.
    competing_lemmas = all_lemmas - {target}
    if competing_lemmas:
        return None
    if not target or not target[0].islower() or not _letters(target): return None
    form_info=[{"orth":orth,"morph_counts":morph_counts,"analyses":analyses}]
    morph_status,details=d3.morph_evidence(form_info,target,upos)
    classification="SAFE_DIRECT_SGJP_REPAIR"
    reasons=["DIRECT_SGJP_UNIQUE_FULL_MORPH"]
    if morph_status!="FULL_MORPH_MATCH":
        if upos=="NOUN" and _safe_syncretic_noun(morph_counts,same_upos,target):
            classification="SAFE_DIRECT_SGJP_SYNCRETIC_CASE_REPAIR"
            reasons=["DIRECT_SGJP_UNIQUE_LEMMA_SYNCRETIC_CASE"]
            morph_status="SYNCRETIC_CASE_UNIQUE_LEMMA"
        else:
            return None
    elif upos=="ADJ":
        classification="SAFE_DIRECT_SGJP_ADJ_REPAIR"
        reasons=["DIRECT_SGJP_ADJ_FULL_MORPH"]
    compatible=sorted({_clean(a.get("lemma")) for a in same_upos if _clean(a.get("lemma"))==target})
    if compatible != [target]: return None
    return {
      "orth":orth,"lemma":lemma,"upos":upos,"replacement":target,
      "reason":"SGJP wskazuje jeden w pelni zgodny morfologicznie lemat dla obserwowanej formy.",
      "status":"accept","decision_bucket":"auto",
      "decision_reasons":reasons,
      "classification":classification,
      "source_classification":"DIRECT_TRIPLE_SCAN",
      "sgjp_status":"SGJP_UNIQUE_TARGET","morph_status":morph_status,
      "compatible_lemmas":[target],"stanza_morph_counts":morph_counts,
      "morph_comparisons":details,"observed_count":count,"source_count":count,
      "target_count":0,"source_document_count":docs,"target_document_count":0,
      "source_form_count":1,"target_form_count":0,"examples":[],
    }

def augment_direct_sgjp(paths: LemmaRepairPaths, reporter: Any=None) -> dict[str,int]:
    if not paths.workspace.is_file(): raise LemmaRepairError(f"Brak workspace audytu: {paths.workspace}")
    auto=json.loads(paths.decisions_auto().read_text(encoding="utf-8"))
    review=json.loads(paths.decisions_review().read_text(encoding="utf-8"))
    rejected=json.loads(paths.decisions_rejected().read_text(encoding="utf-8"))
    existing={_key(r):r for r in auto.get("rules",[])}
    blocked_targets=defaultdict(set)
    for r in auto.get("rules",[]): blocked_targets[_key(r)].add(_clean(r.get("replacement")))
    con=sqlite3.connect(paths.workspace)
    morfeusz,_version=d3.morfeusz_engine()
    promoted=[]; scanned=0
    try:
        rows=con.execute("""SELECT lemma,upos,orth,SUM(token_count),SUM(document_count)
                            FROM form_stats
                            WHERE upos IN ('NOUN','VERB','ADJ')
                            GROUP BY lemma,upos,orth
                            HAVING SUM(token_count)>=2 AND SUM(document_count)>=2
                            ORDER BY orth,lemma,upos""").fetchall()
        for lemma,upos,orth,count,docs in rows:
            lemma=_clean(lemma); upos=_clean(upos).upper(); orth=_clean(orth); scanned+=1
            if not all((lemma,upos,orth)): continue
            morph_counts={str(m):int(c) for m,c in con.execute(
                "SELECT morph,token_count FROM form_stats WHERE lemma=? AND upos=? AND orth=?",
                (lemma,upos,orth)).fetchall()}
            status,analyses=_load_analysis(con,morfeusz,orth)
            if status!="ok": continue
            rule=_direct_rule(orth,lemma,upos,int(count),int(docs),morph_counts,analyses)
            if not rule: continue
            key=_key(rule); target=_clean(rule.get("replacement"))
            if key in existing:
                if _clean(existing[key].get("replacement"))!=target:
                    continue
                continue
            existing[key]=rule; promoted.append(rule)
            if reporter and len(promoted)%100==0: reporter.status(f"Bezposrednia walidacja SGJP: dodano {len(promoted)} regul...")
        con.commit()
    finally: con.close()
    promoted_keys={_key(r) for r in promoted}
    auto["rules"]=sorted(existing.values(),key=lambda r:(-int(r.get("observed_count") or 0),r.get("upos",""),str(r.get("orth","")).casefold()))
    auto.setdefault("settings",{})["direct_sgjp"]={"enabled":True,"classification":"SAFE_DIRECT_SGJP_REPAIR","required_morph_status":"FULL_MORPH_MATCH","minimum_count":2,"minimum_documents":2,"allowed_upos":["NOUN","VERB","ADJ"]}
    auto["direct_sgjp_promoted_rules"]=len(promoted)
    review["rules"]=[r for r in review.get("rules",[]) if _key(r) not in promoted_keys]
    rejected["rules"]=[r for r in rejected.get("rules",[]) if _key(r) not in promoted_keys]
    for path,payload in ((paths.decisions_auto(),auto),(paths.decisions_review(),review),(paths.decisions_rejected(),rejected)):
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    direct_path=paths.artifact_prefix.with_suffix(".direct_sgjp.json")
    direct_path.write_text(json.dumps({"schema_version":1,"scanned_triples":scanned,"promoted_rules":len(promoted),"rules":promoted},ensure_ascii=False,indent=2),encoding="utf-8")
    return {"direct_sgjp_scanned_triples":scanned,"direct_sgjp_promoted_rules":len(promoted),"auto_rules":len(auto["rules"])}

PREPOSITION_CASES={
 "w":{"loc","acc"},"we":{"loc","acc"},"na":{"loc","acc"},
 "o":{"loc","acc"},"po":{"loc","acc"},"przy":{"loc"},
 "do":{"gen"},"od":{"gen"},"bez":{"gen"},"dla":{"gen"},
 "z":{"gen","inst"},"ze":{"gen","inst"},"u":{"gen"},
 "ku":{"dat"},"dzięki":{"dat"},"przeciw":{"dat"},
 "nad":{"inst","acc"},"pod":{"inst","acc"},"przed":{"inst"},"za":{"inst","acc"},
}

def clean(x):return str(x or "").strip()

def norm(x):return clean(x).casefold()

def as_list(x):
 if x is None:return []
 if hasattr(x,"tolist"):x=x.tolist()
 return list(x) if isinstance(x,(list,tuple)) else []

def value_set(x):return {p for p in clean(x).casefold().split(".") if p}

def case_of(tag):return clean(d3.parse_tag(tag).get("case"))

def technical(a):return clean(a.get("upos")).upper() in {"","X","PUNCT","SYM"}

def _position_id(doc,pos):return f"{doc}:{pos}"

def _rule_id(rule):return "|".join((rule["orth"],rule["lemma"],rule["upos"],clean(rule.get("morph_from")),rule["replacement"],clean(rule.get("morph_to"))))

def finalize_sgjp_replacements(paths:LemmaRepairPaths):
 payload=json.loads(paths.decisions_auto().read_text(encoding="utf-8"));kept=[];normalized=removed_noop=0
 for rule in payload.get("rules",[]):
  source=(clean(rule.get("decision_source"))+" "+clean(rule.get("classification"))).upper()
  if "SGJP" not in source:
   kept.append(rule);continue
  raw=clean(rule.get("replacement"));target=d3.normalize_lemma(raw)
  if raw!=target:rule["replacement_raw_sgjp"]=raw;rule["replacement"]=target;normalized+=1
  if not target:continue
  if norm(target)==norm(rule.get("lemma")) and not clean(rule.get("morph_to")):
   removed_noop+=1;continue
  kept.append(rule)
 payload["rules"]=kept;payload.setdefault("settings",{})["sgjp_lemma_normalization"]="split at first colon"
 payload["sgjp_targets_normalized"]=normalized;payload["sgjp_noop_rules_removed"]=removed_noop
 paths.decisions_auto().write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
 return {"sgjp_targets_normalized":normalized,"sgjp_noop_rules_removed":removed_noop,"auto_rules":len(kept)}

def augment_contextual_sgjp(paths:LemmaRepairPaths,reporter:Any=None):
 auto=json.loads(paths.decisions_auto().read_text(encoding="utf-8"));existing={_rule_id(r):r for r in auto.get("rules",[])}
 engine,_=d3.morfeusz_engine(); groups={}; diagnostics=Counter(); pf=pq.ParquetFile(paths.parquet);cols=set(pf.schema_arrow.names)
 morph_col="full_postags" if "full_postags" in cols else ("postags" if "postags" in cols else None)
 read=["tokens","lemmas","upostags"]+([morph_col] if morph_col else []);doc=0
 try:
  for batch in pf.iter_batches(batch_size=128,columns=read):
   data=batch.to_pydict();mrows=data[morph_col] if morph_col else [None]*batch.num_rows
   for tokens,lemmas,upos,morphs in zip(data["tokens"],data["lemmas"],data["upostags"],mrows):
    tokens,lemmas,upos=map(as_list,(tokens,lemmas,upos));morphs=as_list(morphs) if morph_col else [""]*len(tokens)
    if not(len(tokens)==len(lemmas)==len(upos)==len(morphs)):doc+=1;continue
    for pos,(orth,lemma,part,tag) in enumerate(zip(tokens,lemmas,upos,morphs)):
     orth,lemma,part,tag=clean(orth),clean(lemma),clean(part).upper(),clean(tag)
     if part!="NOUN" or pos==0 or not orth or not lemma:continue
     prep=norm(tokens[pos-1]); governed=PREPOSITION_CASES.get(prep)
     if not governed:continue
     observed_case=case_of(tag)
     allowed={observed_case} if observed_case in governed else set(governed)
     status,analyses=d3.analyse_form(engine,orth)
     if status!="ok":continue
     candidates=[]
     for a in analyses:
      if technical(a) or clean(a.get("upos")).upper()!="NOUN":continue
      parsed=d3.parse_tag(a.get("tag"));cases=value_set(parsed.get("case"))
      if cases.intersection(allowed):candidates.append(a)
     targets={d3.normalize_lemma(a.get("lemma")) for a in candidates if d3.normalize_lemma(a.get("lemma"))}
     tags={clean(a.get("tag")) for a in candidates if clean(a.get("tag"))}
     if len(targets)!=1 or len(tags)!=1:diagnostics["context_not_unique"]+=1;continue
     target=next(iter(targets));morph_to=next(iter(tags))
     if norm(target)==norm(lemma):continue
     key=(orth,lemma,part,tag,target,morph_to,prep)
     item=groups.setdefault(key,{"positions":[],"documents":set(),"examples":[]})
     item["positions"].append({"doc_id":doc,"token_index":pos,"previous_orth":clean(tokens[pos-1])});item["documents"].add(doc)
     if len(item["examples"])<6:item["examples"].append(" ".join(map(clean,tokens[max(0,pos-7):pos+8])))
    doc+=1
 finally:pf.close()
 promoted=[]
 for (orth,lemma,upos,mfrom,target,mto,prep),item in groups.items():
  if len(item["positions"])<2 or len(item["documents"])<2:diagnostics["below_threshold"]+=1;continue
  rule={"orth":orth,"lemma":lemma,"upos":upos,"replacement":target,"status":"accept","decision_bucket":"auto","classification":"SAFE_PREPOSITIONAL_SGJP_REPAIR","decision_source":"SGJP_CONTEXT","decision_reasons":["PREPOSITIONAL_CASE_FILTER","POSITIONAL_RULE"],"reason":"Kontekst przyimkowy pozostawia jeden lemat i jeden tag SGJP.","morph_from":mfrom,"morph_to":mto,"positions":item["positions"],"observed_count":len(item["positions"]),"source_document_count":len(item["documents"]),"examples":item["examples"]}
  rid=_rule_id(rule)
  if rid not in existing:existing[rid]=rule;promoted.append(rule)
 auto["rules"]=sorted(existing.values(),key=lambda r:(-int(r.get("observed_count") or 0),clean(r.get("orth")).casefold()))
 auto.setdefault("settings",{})["contextual_sgjp"]={"enabled":True,"position_specific":True,"minimum_count":2,"minimum_documents":2,"preposition_case_filter":True}
 auto["contextual_sgjp_promoted_rules"]=len(promoted);paths.decisions_auto().write_text(json.dumps(auto,ensure_ascii=False,indent=2),encoding="utf-8")
 report={"schema_version":1,"promoted_rules":len(promoted),"counters":dict(diagnostics),"rules":promoted};paths.artifact_prefix.with_suffix(".contextual_sgjp.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 return {"contextual_sgjp_promoted_rules":len(promoted),"auto_rules":len(auto["rules"])}
