# -*- coding: utf-8 -*-
"""Bezposrednia walidacja kazdej obserwowanej trojki orth+lemma+UPOS w SGJP.

Sciezka uzupelnia audyt paradygmatyczny. Nie wymaga, aby bledny lemat mial
najwyzej dwie formy ani aby poprawny lemat byl juz czestszy w korpusie.
"""
from __future__ import annotations
import json, sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any
from . import _lemma_repair_d3_engine as d3
from .lemma_repair_models import LemmaRepairError, LemmaRepairPaths

ALLOWED_UPOS={"NOUN","VERB"}

def _clean(x): return str(x or "").strip()
def _key(r): return (_clean(r.get("orth")),_clean(r.get("lemma")),_clean(r.get("upos")).upper())
def _letters(value): return bool(value and all(ch.isalpha() for ch in value))

def _load_analysis(con, morfeusz, orth):
    row=con.execute("SELECT status,analyses_json FROM sgjp_cache WHERE orth=?",(orth,)).fetchone()
    if row:
        try: return row[0],json.loads(row[1])
        except Exception: pass
    status,analyses=d3.analyse_form(morfeusz,orth)
    con.execute("INSERT OR REPLACE INTO sgjp_cache VALUES(?,?,?)",(orth,status,json.dumps(analyses,ensure_ascii=False)))
    return status,analyses

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
    if morph_status!="FULL_MORPH_MATCH": return None
    compatible=sorted({_clean(a.get("lemma")) for a in same_upos if _clean(a.get("lemma"))==target})
    if compatible != [target]: return None
    return {
      "orth":orth,"lemma":lemma,"upos":upos,"replacement":target,
      "reason":"SGJP wskazuje jeden w pelni zgodny morfologicznie lemat dla obserwowanej formy.",
      "status":"accept","decision_bucket":"auto",
      "decision_reasons":["DIRECT_SGJP_UNIQUE_FULL_MORPH"],
      "classification":"SAFE_DIRECT_SGJP_REPAIR",
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
                            WHERE upos IN ('NOUN','VERB')
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
    auto.setdefault("settings",{})["direct_sgjp"]={"enabled":True,"classification":"SAFE_DIRECT_SGJP_REPAIR","required_morph_status":"FULL_MORPH_MATCH","minimum_count":2,"minimum_documents":2,"allowed_upos":["NOUN","VERB"]}
    auto["direct_sgjp_promoted_rules"]=len(promoted)
    review["rules"]=[r for r in review.get("rules",[]) if _key(r) not in promoted_keys]
    rejected["rules"]=[r for r in rejected.get("rules",[]) if _key(r) not in promoted_keys]
    for path,payload in ((paths.decisions_auto(),auto),(paths.decisions_review(),review),(paths.decisions_rejected(),rejected)):
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    direct_path=paths.artifact_prefix.with_suffix(".direct_sgjp.json")
    direct_path.write_text(json.dumps({"schema_version":1,"scanned_triples":scanned,"promoted_rules":len(promoted),"rules":promoted},ensure_ascii=False,indent=2),encoding="utf-8")
    return {"direct_sgjp_scanned_triples":scanned,"direct_sgjp_promoted_rules":len(promoted),"auto_rules":len(auto["rules"])}
