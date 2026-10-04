# -*- coding: utf-8 -*-
"""Adapter produkcyjny korzystający bezpośrednio z zatwierdzonych klasyfikatorów eksperymentalnych."""
from __future__ import annotations
import json, sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import pyarrow.parquet as pq
from korpusuj.runtime_paths import resource_root
from .lemma_repair_models import LemmaRepairPaths
from .lemma_repair_rules import ner_broad, rule_context_matches
from . import proper_name_sgjp as sgjp_ref
from . import proper_name_prng as prng_ref

AUTO_SGJP={"SAFE_NER_SGJP_REPAIR","SAFE_NER_SGJP_LEMMA_AND_GENDER_REPAIR"}
EXTERNAL_BASE={"PROPER_NAME_CASE_CONFLICT","EXTERNAL_REGISTRY_CANDIDATE","NO_SGJP_ANALYSIS","NO_USEFUL_SGJP_ANALYSIS"}

def clean(x): return str(x or "").strip()

def _as_list(value):
    """Convert Arrow, NumPy, tuple or list values to a plain list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []
def key(rule): return (clean(rule.get("orth")),clean(rule.get("lemma")),clean(rule.get("upos")).upper(),clean(rule.get("morph_from")),clean(rule.get("required_ner_broad")).upper())

def prng_hits(con,form):
    rows=con.execute("SELECT prng_id,canonical_name,form,form_kind,object_type,name_status FROM name_forms WHERE form_key=? ORDER BY canonical_name,prng_id",(prng_ref.norm(form),)).fetchall()
    return [{"prng_id":r[0],"canonical_name":r[1],"name":r[2],"kind":r[3],"object_type":r[4],"name_status":r[5]} for r in rows]

def open_prng(path):
    if not path.is_file(): return None,"missing"
    try:
        con=sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro",uri=True);con.execute("PRAGMA query_only=ON")
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")};meta=dict(con.execute("SELECT key,value FROM metadata"))
        if not {"metadata","name_forms"}<=tables or meta.get("schema_version")!="1" or str(con.execute("PRAGMA integrity_check").fetchone()[0]).lower()!="ok":
            con.close();return None,"incompatible"
        return con,"ok"
    except Exception as exc:return None,f"error:{type(exc).__name__}"

def make_rule(row,source):
 repair=row.get("morph_repair") or {}
 broad=clean(row.get("ner_broad")).upper()
 if not broad or broad=="O":
  broad=ner_broad(row.get("ner_raw"))
 rule={"orth":row["orth"],"lemma":row["model_lemma"],"upos":row["upos"],"replacement":row["target"],"reason":"Klasyfikator nazw własnych zatwierdzony w eksperymencie.","status":"accept","decision_bucket":"auto","decision_reasons":list(row.get("decision_reasons") or []),"classification":row["category"],"decision_source":source,"morph_from":clean(repair.get("from")),"morph_to":clean(repair.get("to")),"observed_count":int(row["token_count"]),"source_document_count":int(row["document_count"]),"examples":list(row.get("examples") or [])[:6]}
 if broad and broad not in {"O","PROPN"}:
  rule["required_ner_broad"]=broad
 rule["source_span_kind"]=clean(row.get("span_kind"))
 return rule

def recount(paths,auto):
 rules=list(auto.get("rules",[]));exact={};generic={}
 for rule in rules:
  base=(clean(rule.get("orth")),clean(rule.get("lemma")),clean(rule.get("upos")).upper(),clean(rule.get("required_ner_broad")).upper())
  morph=clean(rule.get("morph_from"));(exact if morph else generic)[(*base,morph) if morph else base]=rule
 counts=Counter();docs=defaultdict(set);pf=pq.ParquetFile(paths.parquet);cols=set(pf.schema_arrow.names);mc="full_postags" if "full_postags" in cols else ("postags" if "postags" in cols else None);nc="ners" if "ners" in cols else ("ner" if "ner" in cols else None);read=["tokens","lemmas","upostags"]+([mc] if mc else [])+([nc] if nc else []);doc=0
 try:
  for batch in pf.iter_batches(batch_size=128,columns=read):
   data=batch.to_pydict();mrows=data[mc] if mc else [None]*batch.num_rows;nrows=data[nc] if nc else [None]*batch.num_rows
   for ts,ls,us,ms,ns in zip(data["tokens"],data["lemmas"],data["upostags"],mrows,nrows):
    ts,ls,us=map(_as_list,(ts,ls,us));ms=_as_list(ms) if mc else [""]*len(ts);ns=_as_list(ns) if nc else ["O"]*len(ts)
    if not(len(ts)==len(ls)==len(us)==len(ms)==len(ns)):doc+=1;continue
    for pos,(o,l,u,m,n) in enumerate(zip(ts,ls,us,ms,ns)):
     broad=ner_broad(n);base=(clean(o),clean(l),clean(u).upper(),broad);rule=exact.get((*base,clean(m))) or generic.get(base)
     if rule is None:
      base=(clean(o),clean(l),clean(u).upper(),"");rule=exact.get((*base,clean(m))) or generic.get(base)
     if rule is not None and rule_context_matches(rule,n,doc,pos):counts[id(rule)]+=1;docs[id(rule)].add(doc)
    doc+=1
 finally:pf.close()
 kept=[]
 for rule in rules:
  count=counts[id(rule)]
  if count:rule["observed_count"]=count;rule["source_document_count"]=len(docs[id(rule)]);kept.append(rule)
 auto["rules"]=kept;auto["recounted_after_proper_names"]=True;return auto



def _corpus_has_usable_ner(paths):
    """Kolumna istnieje i zawiera przynajmniej jedna rzeczywista etykiete encji."""
    pf=pq.ParquetFile(paths.parquet)
    try:
        if "ners" not in set(pf.schema_arrow.names):
            return False
        for batch in pf.iter_batches(batch_size=128,columns=["ners"]):
            for row in batch.to_pydict()["ners"]:
                for value in (row or []):
                    raw=clean(value).upper()
                    if raw not in {"","O","_"}:
                        return True
    finally:
        pf.close()
    return False


def _collect_propn_fallback(paths):
    """Izolowane PROPN bez sasiedniego PROPN. Brak PRNG i korekt rodzaju."""
    pf=pq.ParquetFile(paths.parquet)
    columns=set(pf.schema_arrow.names)
    required={"tokens","lemmas","upostags"}
    if not required<=columns:
        pf.close(); return {}
    morph="full_postags" if "full_postags" in columns else ("postags" if "postags" in columns else None)
    read=["tokens","lemmas","upostags"]+([morph] if morph else [])
    out={};doc_id=0
    try:
        for batch in pf.iter_batches(batch_size=128,columns=read):
            data=batch.to_pydict();mrows=data[morph] if morph else [None]*batch.num_rows
            for tokens,lemmas,upos,morphs in zip(data["tokens"],data["lemmas"],data["upostags"],mrows):
                tokens=list(tokens or []);lemmas=list(lemmas or []);upos=list(upos or []);morphs=list(morphs or [""]*len(tokens))
                if not(len(tokens)==len(lemmas)==len(upos)==len(morphs)):
                    doc_id+=1;continue
                seen=set()
                for i,(orth,lemma,pos,tag) in enumerate(zip(tokens,lemmas,upos,morphs)):
                    orth,lemma,pos,tag=clean(orth),clean(lemma),clean(pos).upper(),clean(tag)
                    if pos!="PROPN" or not orth or not lemma or not orth[:1].isupper() or not any(ch.isalpha() for ch in orth):continue
                    before=clean(upos[i-1]).upper() if i else ""
                    after=clean(upos[i+1]).upper() if i+1<len(upos) else ""
                    if before=="PROPN" or after=="PROPN":continue
                    key=(orth,lemma,pos,tag)
                    item=out.setdefault(key,{"orth":orth,"lemma":lemma,"upos":pos,"morph":tag,"count":0,"documents":set(),"examples":[]})
                    item["count"]+=1;item["documents"].add(doc_id)
                    if len(item["examples"])<6:item["examples"].append(" ".join(map(clean,tokens[max(0,i-8):i+9])))
                doc_id+=1
    finally:
        pf.close()
    return out


def _propn_fallback_rules(paths,engine):
    rules=[];observations=_collect_propn_fallback(paths)
    for item in observations.values():
        if item["count"]<2 or len(item["documents"])<2:continue
        obs=sgjp_ref.Observation(orth=item["orth"],lemma=item["lemma"],upos=item["upos"],morph=item["morph"],ner_raw="PROPN_FALLBACK",ner_broad="PROPN",span_kind="SINGLE_TOKEN_ENTITY",span_text=item["orth"],span_length=1,count=item["count"],documents=item["documents"],examples=item["examples"])
        status,analyses=sgjp_ref.analyses_for(engine,obs.orth);row=sgjp_ref.classify(obs,analyses,status,{})
        # Bez NER dopuszczamy tylko pelna zgodnosc SGJP. Bez PRNG i bez korekty rodzaju.
        if row.get("decision")=="AUTO_CANDIDATE" and row.get("category")=="SAFE_NER_SGJP_REPAIR" and row.get("target"):
            row=dict(row);row["category"]="SAFE_PROPN_SGJP_REPAIR";row["decision_reasons"]=list(row.get("decision_reasons") or [])+["UPOS_PROPN_FALLBACK","ISOLATED_PROPN"]
            rule=make_rule(row,"SGJP_PROPN_FALLBACK");rule["entity_evidence"]="PROPN_FALLBACK";rules.append(rule)
    return observations,rules



def _normalized_surface_inventory(paths):
    """Zbior kapitalizowanych form powierzchniowych w calym korpusie."""
    inventory=set()
    pf=pq.ParquetFile(paths.parquet)
    try:
        for batch in pf.iter_batches(batch_size=128,columns=["tokens"]):
            for row in batch.to_pydict()["tokens"]:
                for value in (row or []):
                    text=sgjp_ref.normalized_lemma(value)
                    if text and text[:1].isupper():
                        inventory.add(text.casefold())
    finally:
        pf.close()
    return inventory


def _sgjp_auto_guard(row,surface_inventory):
    """Zwraca (allow, reason) dla kandydata AUTO z SGJP."""
    category=clean(row.get("category"))
    broad=clean(row.get("ner_broad")).upper()
    model=sgjp_ref.normalized_lemma(row.get("model_lemma"))
    target=sgjp_ref.normalized_lemma(row.get("target"))
    if row.get("span_kind")!="SINGLE_TOKEN_ENTITY":
        return False,"MULTI_TOKEN_ENTITY_MEMBER"
    # Jezeli modelowy lemat istnieje w korpusie jako kapitalizowana forma
    # mianownikowa, a cel SGJP nie istnieje, chronimy m.in. Musk -> Muskie
    # oraz Fico -> Fica.
    if model and target and model.casefold() in surface_inventory and target.casefold() not in surface_inventory:
        return False,"ATTESTED_MODEL_LEMMA_CONFLICT"
    # Po rozszerzeniu klas korekty rodzaju sa AUTO tylko dla lokalizacji,
    # organizacji i obiektow. Osoby i nieznane klasy trafiaja do diagnostyki.
    if category=="SAFE_NER_SGJP_LEMMA_AND_GENDER_REPAIR" and broad not in {"LOC","ORG","FAC"}:
        return False,"GENDER_REPAIR_REQUIRES_LOC_ORG_FAC"
    return True,""


def _prng_auto_guard(row):
    """PRNG tylko dla samodzielnej encji LOC i niepospolitego lematu modelu."""
    if clean(row.get("ner_broad")).upper()!="LOC":
        return False,"PRNG_REQUIRES_LOC"
    if row.get("span_kind")!="SINGLE_TOKEN_ENTITY":
        return False,"PRNG_MULTI_TOKEN_MEMBER"
    model=sgjp_ref.normalized_lemma(row.get("model_lemma"))
    # Blokuje kraj -> Kraje, gora -> Gory, huta -> Huta. Zachowuje
    # kapitalizowane, choc bledne lematy typu Chersic -> Cherson.
    if model and model[:1].islower():
        return False,"PRNG_LOWERCASE_COMMON_LEMMA"
    return True,""



# Jawne wyjątki konwencji korpusowej. SGJP opisuje leksem Reuter, ale w
# kontekstach "agencja Reutera" nie rozstrzyga, czy kanonicznym lematem nazwy
# instytucji ma być Reuter czy Reuters. Takie przypadki wymagaja REVIEW.
_CONVENTIONAL_ORG_REVIEW_PREFIXES={"reuter"}
_ADJECTIVAL_KINDS={"adj","adja","adjp","adjc"}


def _target_supported_only_as_adjective(row):
    """Czy wybrany cel ma w SGJP wyłącznie analizy przymiotnikowe."""
    target_key=sgjp_ref.lemma_key(row.get("target"))
    if not target_key:
        return False
    kinds=set()
    for analysis in (row.get("analyses") or []):
        if analysis.get("normalized_key")!=target_key:
            continue
        kind=sgjp_ref.parse_tag(analysis.get("tag","")).get("kind","")
        if kind:
            kinds.add(kind)
    return bool(kinds) and kinds.issubset(_ADJECTIVAL_KINDS)


def _expanded_semantic_guard(row):
    """Zwraca (allow, reason) dla semantycznie ryzykownych nazw."""
    if row.get("span_kind")!="SINGLE_TOKEN_ENTITY":
        return True,""  # obsluzy to wczesniejszy guard techniczny
    observed_upos=clean(row.get("upos")).upper()
    if observed_upos in {"NOUN","PROPN"} and _target_supported_only_as_adjective(row):
        return False,"SUBSTANTIVIZED_PROPER_ADJECTIVE_REVIEW"
    orth_key=sgjp_ref.lemma_key(row.get("orth"))
    target_key=sgjp_ref.lemma_key(row.get("target"))
    if any(orth_key.startswith(prefix) or target_key.startswith(prefix) for prefix in _CONVENTIONAL_ORG_REVIEW_PREFIXES):
        return False,"CONVENTIONAL_ORG_LEMMA_REVIEW"
    return True,""

def augment_proper_name_repairs(paths:LemmaRepairPaths,reporter:Any=None):
    auto=json.loads(paths.decisions_auto().read_text(encoding="utf-8"));review=json.loads(paths.decisions_review().read_text(encoding="utf-8"));rejected=json.loads(paths.decisions_rejected().read_text(encoding="utf-8"));existing={key(r):r for r in auto.get("rules",[])}
    surface_inventory=_normalized_surface_inventory(paths)
    ner_available=_corpus_has_usable_ner(paths)
    engine=sgjp_ref.morfeusz_engine();prng_path=resource_root()/"temp"/"prng_world.sqlite";con,prng_status=open_prng(prng_path);promoted=[];rows=[];counts=Counter()
    if ner_available:
        observations,ner_counts,span_counts,documents,tokens=sgjp_ref.collect(paths.parquet,128,8,6)
    else:
        fallback_observations,fallback_rules=_propn_fallback_rules(paths,engine)
        observations={}
        for rule in fallback_rules:
            k=key(rule);previous=existing.get(k)
            if previous and clean(previous.get("replacement"))!=clean(rule.get("replacement")):counts["conflicts"]+=1;continue
            if previous:counts["already_present"]+=1;continue
            existing[k]=rule;promoted.append(rule);counts["SGJP_PROPN_FALLBACK"]+=1
        counts["propn_fallback_observations"]=len(fallback_observations)
    try:
        for index,obs in enumerate(observations.values(),1):
            status,analyses=sgjp_ref.analyses_for(engine,obs.orth);row=sgjp_ref.classify(obs,analyses,status,{});rows.append(row)
            if row["token_count"]<2 or row["document_count"]<2:counts["below_threshold"]+=1;continue
            rule=None
            if row["decision"]=="AUTO_CANDIDATE" and row["category"] in AUTO_SGJP and row.get("target"):
                allowed,guard_reason=_sgjp_auto_guard(row,surface_inventory)
                if allowed:
                    allowed,guard_reason=_expanded_semantic_guard(row)
                if allowed:
                    rule=make_rule(row,"SGJP_GENDER" if row["category"].endswith("GENDER_REPAIR") else "SGJP")
                else:
                    counts[f"guard_{guard_reason}"]+=1
            elif row.get("ner_broad")=="LOC" and con is not None:
                prng_allowed,prng_guard_reason=_prng_auto_guard(row)
                if not prng_allowed:
                    counts[f"guard_{prng_guard_reason}"]+=1
                    continue
                candidate={"orth":row["orth"],"model_lemma":row["model_lemma"],"target":row.get("target", ""),"morph":row["morph"],"ner_raw":row["ner_raw"],"span_text":row["span_text"],"category":row["category"],"token_count":row["token_count"],"document_count":row["document_count"],"examples":row.get("examples",[])}
                result=prng_ref.classify_candidate(candidate,prng_hits(con,row["orth"]))
                if result["category"]=="PRNG_SAFE_LEMMA_REPAIR" and result["decision"]=="AUTO_CANDIDATE" and result["changes_lemma"]:
                    prng_row=dict(row);prng_row.update(target=result["proposed_lemma"],category="PRNG_SAFE_LEMMA_REPAIR",decision_reasons=["PRNG_EXACT_CASE_FORM"]);rule=make_rule(prng_row,"PRNG")
                elif result["category"]=="PRNG_MODEL_LEMMA_CONFIRMED":counts["prng_confirmed"]+=1
            if rule:
                k=key(rule);previous=existing.get(k)
                if previous and clean(previous.get("replacement"))!=clean(rule.get("replacement")):counts["conflicts"]+=1;continue
                if previous:counts["already_present"]+=1;continue
                existing[k]=rule;promoted.append(rule);counts[rule["decision_source"]]+=1
            if reporter and index%250==0:reporter.status(f"Nazwy własne: zgodność z eksperymentami {index:,}/{len(observations):,}...")
    finally:
        if con is not None:con.close()
    promoted_keys={key(r) for r in promoted}
    auto["rules"]=sorted(existing.values(),key=lambda r:(-int(r.get("observed_count") or 0),clean(r.get("orth")).casefold(),clean(r.get("morph_from"))))
    auto=recount(paths,auto)
    auto.setdefault("settings",{})["proper_name_repairs"]={"implementation":"exact_experiment_adapters","sgjp_reference_version":sgjp_ref.VERSION,"prng_reference_version":prng_ref.VERSION,"prng_path":str(prng_path),"prng_status":prng_status,"minimum_count":2,"minimum_documents":2,"expanded_ner_guards":"single-token; attested model lemma protection; PRNG LOC + capitalized model lemma; gender AUTO only LOC/ORG/FAC; substantivized adjective review; conventional organization review","entity_evidence":"NER" if ner_available else "PROPN_FALLBACK","propn_fallback_policy":"isolated PROPN; SGJP full match only; no PRNG; no gender repair"};auto["proper_name_promoted_rules"]=len(promoted)
    review["rules"]=[r for r in review.get("rules",[]) if key(r) not in promoted_keys];rejected["rules"]=[r for r in rejected.get("rules",[]) if key(r) not in promoted_keys]
    for path,payload in ((paths.decisions_auto(),auto),(paths.decisions_review(),review),(paths.decisions_rejected(),rejected)):path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    diagnostic={"schema_version":2,"implementation":"exact_experiment_adapters","sgjp_reference_version":sgjp_ref.VERSION,"prng_reference_version":prng_ref.VERSION,"observations":len(observations),"promoted_rules":len(promoted),"counters":dict(counts),"prng_status":prng_status,"rules":promoted};paths.artifact_prefix.with_suffix(".proper_names.json").write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"proper_name_observations":len(observations),"proper_name_promoted_rules":len(promoted),"prng_available":con is not None,"prng_status":prng_status,"auto_rules":len(auto["rules"])}

def recount_final_auto_rules(paths):
 payload=json.loads(paths.decisions_auto().read_text(encoding="utf-8"));before=len(payload.get("rules",[]));payload=recount(paths,payload);after=len(payload.get("rules",[]));payload["final_auto_recounted"]=True;payload["final_auto_zero_hit_removed"]=before-after;paths.decisions_auto().write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");return {"final_auto_recounted":True,"final_auto_zero_hit_removed":before-after,"auto_rules":after}
