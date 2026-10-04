# -*- coding: utf-8 -*-
"""Skalowalny audyt D3 lematów Stanza: Korpusuj .search -> workspace SQLite -> SGJP.

Nie modyfikuje pliku .search ani Parquet. Workspace powstaje obok korpusu i
umożliwia wznowienie pracy. Narzędzie wybiera kandydatów głównie na podstawie
SGJP, a nie przez pełne porównanie fuzzy wszystkich lematów.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sqlite3, sys, time, zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VERSION = "3.0.0"
SGJP_TO_UPOS = {
 "subst":"NOUN","depr":"NOUN","ger":"NOUN","adj":"ADJ","adja":"ADJ","adjp":"ADJ","adjc":"ADJ",
 "adv":"ADV","fin":"VERB","bedzie":"AUX","aglt":"AUX","praet":"VERB","impt":"VERB","imps":"VERB",
 "inf":"VERB","pcon":"VERB","pant":"VERB","winien":"VERB","pred":"VERB","pact":"ADJ","ppas":"ADJ",
 "num":"NUM","numcomp":"NUM","ppron12":"PRON","ppron3":"PRON","siebie":"PRON","prep":"ADP",
 "conj":"CCONJ","comp":"SCONJ","qub":"PART","interj":"INTJ","interp":"PUNCT","brev":"X","burk":"X","xxx":"X","ign":"X"
}
DEFAULT_EXCLUDED = {"ADP","AUX","CCONJ","DET","INTJ","PART","PRON","PUNCT","SCONJ","SYM","X"}

class AuditError(RuntimeError): pass

def now(): return datetime.now(timezone.utc).isoformat()
def clean(x): return str(x or "").strip()
def sha256(path: Path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def decode(blob, default):
 if not blob: return default
 try: return json.loads(zlib.decompress(blob).decode("utf-8"))
 except Exception:
  try: return json.loads(bytes(blob).decode("utf-8"))
  except Exception: return default
def normalize_lemma(x):
 value=clean(x)
 # Morfeusz/SGJP dopisuje po pierwszym dwukropku identyfikator techniczny
 # paradygmatu lub wariantu, np. czas:S, Turek:Sm1~cy, rzad:Sm3~adu.
 # W warstwie lemmas przechowujemy zawsze haslo bazowe.
 return value.split(":",1)[0]
def tag_upos(tag): return SGJP_TO_UPOS.get(clean(tag).split(":",1)[0].casefold())

def connect_workspace(path: Path):
 con=sqlite3.connect(path)
 con.execute("PRAGMA journal_mode=WAL")
 con.execute("PRAGMA synchronous=NORMAL")
 con.execute("PRAGMA temp_store=MEMORY")
 con.executescript("""
 CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS form_stats(
  lemma TEXT NOT NULL, upos TEXT NOT NULL, orth TEXT NOT NULL, morph TEXT NOT NULL,
  token_count INTEGER NOT NULL, document_count INTEGER NOT NULL,
  PRIMARY KEY(lemma,upos,orth,morph));
 CREATE TABLE IF NOT EXISTS lemma_stats(
  lemma TEXT NOT NULL, upos TEXT NOT NULL, token_count INTEGER NOT NULL,
  document_count INTEGER NOT NULL, form_count INTEGER NOT NULL, morph_count INTEGER NOT NULL,
  PRIMARY KEY(lemma,upos));
 CREATE TABLE IF NOT EXISTS sgjp_cache(orth TEXT PRIMARY KEY,status TEXT NOT NULL,analyses_json TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS candidates(
  source_lemma TEXT NOT NULL,target_lemma TEXT NOT NULL,upos TEXT NOT NULL,
  source_count INTEGER NOT NULL,target_count INTEGER NOT NULL,source_docs INTEGER NOT NULL,target_docs INTEGER NOT NULL,
  source_forms INTEGER NOT NULL,target_forms INTEGER NOT NULL,sgjp_status TEXT NOT NULL,classification TEXT NOT NULL,
  evidence_json TEXT NOT NULL,rules_json TEXT NOT NULL,PRIMARY KEY(source_lemma,target_lemma,upos));
 CREATE TABLE IF NOT EXISTS examples(
  source_lemma TEXT NOT NULL,target_lemma TEXT NOT NULL,upos TEXT NOT NULL,orth TEXT NOT NULL,
  doc_id INTEGER NOT NULL,token_index INTEGER NOT NULL,context TEXT NOT NULL,document_label TEXT NOT NULL,
  PRIMARY KEY(source_lemma,target_lemma,upos,orth,doc_id,token_index));
 """)
 return con
def state_get(con,key,default=None):
 r=con.execute("SELECT value FROM state WHERE key=?",(key,)).fetchone(); return r[0] if r else default
def state_set(con,key,value):
 con.execute("INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(key,str(value))); con.commit()

def verify_search(path: Path):
 con=sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True)
 try:
  cols={r[1] for r in con.execute("PRAGMA table_info(docs)")}
  needed={"doc_id","metadata_json","tokens","lemmas","upostags"}
  if not needed<=cols: raise AuditError("Nieobsługiwany .search; brak kolumn: "+", ".join(sorted(needed-cols)))
  return cols, int(con.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
 finally: con.close()

def aggregate(search: Path, work, excluded:set[str], commit_docs:int):
 if state_get(work,"aggregation_complete")=="1": return
 last=int(state_get(work,"last_doc_id","-1")); cols,_=verify_search(search)
 morph_col="full_postags" if "full_postags" in cols else ("postags" if "postags" in cols else None)
 sql="SELECT doc_id,tokens,lemmas,upostags"+(f",{morph_col}" if morph_col else "")+" FROM docs WHERE doc_id>? ORDER BY doc_id"
 src=sqlite3.connect(f"file:{search.as_posix()}?mode=ro",uri=True); src.row_factory=sqlite3.Row
 started=time.time(); processed=0
 try:
  for row in src.execute(sql,(last,)):
   doc_id=int(row["doc_id"]); toks=decode(row["tokens"],[]); lems=decode(row["lemmas"],[]); ups=decode(row["upostags"],[])
   morph=decode(row[morph_col],[]) if morph_col else [""]*len(toks)
   if not(len(toks)==len(lems)==len(ups)==len(morph)):
    state_set(work,"malformed_docs",int(state_get(work,"malformed_docs","0"))+1); continue
   counts=Counter(); present=set()
   for o,l,u,m in zip(toks,lems,ups,morph):
    key=(clean(l),clean(u).upper(),clean(o),clean(m) or "<brak>")
    if not key[0] or not key[1] or not key[2] or key[1] in excluded: continue
    counts[key]+=1; present.add(key)
   work.executemany("""INSERT INTO form_stats VALUES(?,?,?,?,?,?)
    ON CONFLICT(lemma,upos,orth,morph) DO UPDATE SET token_count=token_count+excluded.token_count,
    document_count=document_count+excluded.document_count""",
    [(l,u,o,m,n,1 if (l,u,o,m) in present else 0) for (l,u,o,m),n in counts.items()])
   processed+=1; last=doc_id
   if processed%commit_docs==0:
    state_set(work,"last_doc_id",last)
    print(f"[aggregation] doc_id={last:,}; session_docs={processed:,}; elapsed={(time.time()-started)/60:.1f} min",file=sys.stderr,flush=True)
  state_set(work,"last_doc_id",last); state_set(work,"aggregation_complete",1)
 finally: src.close()

def summarize(work):
 if state_get(work,"summary_complete")=="1": return
 work.execute("DELETE FROM lemma_stats")
 work.execute("""INSERT INTO lemma_stats
 SELECT lemma,upos,SUM(token_count),SUM(document_count),COUNT(DISTINCT orth),COUNT(DISTINCT morph)
 FROM form_stats GROUP BY lemma,upos""")
 work.execute("CREATE INDEX IF NOT EXISTS idx_lemma_stats_shape ON lemma_stats(form_count,token_count,document_count)")
 work.execute("CREATE INDEX IF NOT EXISTS idx_form_stats_source ON form_stats(lemma,upos)")
 work.commit(); state_set(work,"summary_complete",1)

def morfeusz_engine():
 try: import morfeusz2
 except ImportError as e: raise AuditError("Brak morfeusz2. Uruchom: python -m pip install morfeusz2") from e
 return morfeusz2.Morfeusz(), clean(getattr(morfeusz2,"__version__","unknown"))
def analyse_form(engine,orth):
 out=[]; mismatch=False
 try: raw=list(engine.analyse(orth))
 except Exception as e: return "error",[{"error":f"{type(e).__name__}: {e}"}]
 for x in raw:
  if not isinstance(x,(list,tuple)) or len(x)<3 or not isinstance(x[2],(list,tuple)) or len(x[2])<3: continue
  try: mismatch |= int(x[0])!=0 or int(x[1])!=1
  except Exception: pass
  tag=clean(x[2][2]); out.append({"orth":clean(x[2][0]),"lemma":normalize_lemma(x[2][1]),"tag":tag,"upos":tag_upos(tag)})
 unique=[]; seen=set()
 for a in out:
  k=(a["orth"],a["lemma"],a["tag"],a["upos"])
  if k not in seen: seen.add(k); unique.append(a)
 return ("tokenization_mismatch" if mismatch else "ok") if unique else "no_analysis",unique

def analysis_prefix(analysis):
 return clean(analysis.get("tag")).split(":",1)[0].casefold()

TAG_LAYOUTS={
 "subst":("number","case","gender"), "depr":("number","case","gender"),
 "adj":("number","case","gender","degree"), "adja":(), "adjp":(), "adjc":(),
 "adv":("degree",), "fin":("number","person","aspect"),
 "praet":("number","gender","aspect"), "impt":("number","person","aspect"),
 "imps":("aspect",), "inf":("aspect",), "pcon":("aspect",), "pant":("aspect",),
 "ger":("number","case","gender","aspect","polarity"),
 "pact":("number","case","gender","aspect","polarity"),
 "ppas":("number","case","gender","aspect","polarity"),
 "num":("number","case","gender","accommodability"),
}

def parse_tag(tag):
 parts=[x.casefold() for x in clean(tag).split(":") if x]
 if not parts: return {"kind":"unknown","raw":clean(tag)}
 kind=parts[0]; out={"kind":kind,"raw":clean(tag)}
 for name,value in zip(TAG_LAYOUTS.get(kind,()),parts[1:]): out[name]=value
 return out

def _tag_value_set(value):
 # SGJP laczy rownowazne wartosci kropka, np. gen.acc albo nom.voc.
 # Zgodnosc zachodzi, gdy zbiory dopuszczalnych wartosci maja przeciecie.
 return {part for part in clean(value).casefold().split(".") if part}

def compare_tags(stanza_tag,sgjp_tag):
 a=parse_tag(stanza_tag); b=parse_tag(sgjp_tag)
 shared=sorted((set(a)&set(b))-{"kind","raw"})
 matches={}
 mismatches={}
 for key in shared:
  a_values=_tag_value_set(a[key]); b_values=_tag_value_set(b[key])
  if a_values and b_values and a_values.intersection(b_values):
   matches[key]={"stanza":a[key],"sgjp":b[key],"overlap":sorted(a_values.intersection(b_values))}
  else:
   mismatches[key]={"stanza":a[key],"sgjp":b[key]}
 return {"stanza":a,"sgjp":b,"shared":shared,"matches":matches,"mismatches":mismatches}

def morph_evidence(form_info,target_lemma,upos):
 """Porównaj każde obserwowane XPOS Stanza z analizami SGJP celu."""
 details=[]; any_conflict=False; all_have_comparable=True; all_strong=True
 for item in form_info:
  target_analyses=[a for a in item["analyses"] if a.get("lemma")==target_lemma and a.get("upos")==upos]
  for stanza_tag,count in item["morph_counts"].items():
   alternatives=[compare_tags(stanza_tag,a.get("tag","")) for a in target_analyses]
   compatible=[x for x in alternatives if not x["mismatches"]]
   if not alternatives:
    outcome="UNKNOWN_NO_SGJP_TARGET"; best=None
   elif compatible:
    best=max(compatible,key=lambda x:len(x["shared"])); n=len(best["shared"])
    outcome="MATCH" if n>=2 else "PARTIAL_MATCH"
   else:
    best=min(alternatives,key=lambda x:len(x["mismatches"])); outcome="MISMATCH"; any_conflict=True
   if not best or not best["shared"]: all_have_comparable=False
   if outcome!="MATCH": all_strong=False
   details.append({"orth":item["orth"],"count":count,"stanza_tag":stanza_tag,"outcome":outcome,"best_comparison":best})
 if any_conflict: status="MORPH_CONFLICT"
 elif all_have_comparable and all_strong: status="FULL_MORPH_MATCH"
 elif all_have_comparable: status="PARTIAL_MORPH_MATCH"
 else: status="MORPH_UNKNOWN"
 return status,details

def classify_candidate(source_lemma,target_lemma,upos,form_info,current_confirmed,target_all,only_target):
 if source_lemma.casefold()==target_lemma.casefold(): return "CASE_NORMALIZATION_ONLY",None,[]
 supporting=[a for item in form_info for a in item["analyses"] if a.get("lemma")==target_lemma and a.get("upos")==upos]
 prefixes={analysis_prefix(a) for a in supporting}
 if upos=="NOUN" and "ger" in prefixes: return "CONVENTION_GERUND",None,[]
 if upos=="ADJ" and "pact" in prefixes: return "CONVENTION_PARTICIPLE_ACTIVE",None,[]
 if upos=="ADJ" and "ppas" in prefixes: return "CONVENTION_PARTICIPLE_PASSIVE",None,[]
 compatible_union={lemma for item in form_info for lemma in item["compatible"]}
 if current_confirmed or len(compatible_union)>1 or not target_all:
  return "AMBIGUOUS_OR_HOMOGRAPHIC",None,[]
 morph_status,morph_details=morph_evidence(form_info,target_lemma,upos)
 if morph_status=="MORPH_CONFLICT": return "MORPH_CONFLICT",morph_status,morph_details
 if only_target and morph_status=="FULL_MORPH_MATCH": return "SAFE_FULL_MORPH_REPAIR",morph_status,morph_details
 if only_target and morph_status in {"PARTIAL_MORPH_MATCH","MORPH_UNKNOWN"}: return "SAFE_PARTIAL_MORPH_REPAIR",morph_status,morph_details
 return "REVIEW_OTHER",morph_status,morph_details

def build_candidates(work,min_count,min_docs,max_forms,max_per_source):
 engine,version=morfeusz_engine(); state_set(work,"morfeusz_version",version)
 work.execute("DELETE FROM candidates"); work.commit()
 sources=work.execute("SELECT * FROM lemma_stats WHERE token_count>=? AND document_count>=? AND form_count<=? ORDER BY token_count DESC",(min_count,min_docs,max_forms)).fetchall()
 for i,s in enumerate(sources,1):
  src_lemma,upos,src_count,src_docs,src_forms,_=s; votes=Counter()
  raw_rows=work.execute("SELECT orth,morph,token_count FROM form_stats WHERE lemma=? AND upos=?",(src_lemma,upos)).fetchall()
  grouped={}
  for orth,morph,n in raw_rows:
   item=grouped.setdefault(orth,{"orth":orth,"count":0,"morph_counts":Counter()})
   item["count"]+=int(n); item["morph_counts"][clean(morph)]+=int(n)
  form_info=[]
  for orth,item in grouped.items():
   cached=work.execute("SELECT status,analyses_json FROM sgjp_cache WHERE orth=?",(orth,)).fetchone()
   if cached: status,analyses=cached[0],json.loads(cached[1])
   else:
    status,analyses=analyse_form(engine,orth); work.execute("INSERT OR REPLACE INTO sgjp_cache VALUES(?,?,?)",(orth,status,json.dumps(analyses,ensure_ascii=False)))
   compatible=sorted({a.get("lemma") for a in analyses if a.get("upos")==upos and a.get("lemma")})
   for lemma in compatible:
    if lemma!=src_lemma: votes[lemma]+=item["count"]
   item.update({"status":status,"compatible":compatible,"analyses":analyses})
   form_info.append(item)
  accepted=0
  for target,votes_n in votes.most_common(max_per_source):
   t=work.execute("SELECT token_count,document_count,form_count FROM lemma_stats WHERE lemma=? AND upos=?",(target,upos)).fetchone()
   if not t: continue
   tc,td,tf=map(int,t)
   if tc<=src_count or tf<=src_forms: continue
   current=any(src_lemma in x["compatible"] for x in form_info); target_all=all(target in x["compatible"] for x in form_info); only_target=all(x["compatible"]==[target] for x in form_info)
   status="SGJP_UNIQUE_TARGET" if only_target and not current else ("SGJP_TARGET_AMONG_CANDIDATES" if target_all else "SGJP_PARTIAL_TARGET")
   classification,morph_status,morph_details=classify_candidate(src_lemma,target,upos,form_info,current,target_all,only_target)
   evidence=["SGJP wskazuje lemat docelowy dla form źródłowych.","Lemat docelowy istnieje w korpusie i ma bogatszy paradygmat."]
   if classification=="CASE_NORMALIZATION_ONLY": evidence.append("Różnica dotyczy wyłącznie wielkości liter.")
   elif classification=="CONVENTION_GERUND": evidence.append("SGJP reprezentuje rzeczownik odczasownikowy przez lemat czasownika (tag ger).")
   elif classification.startswith("CONVENTION_PARTICIPLE_"): evidence.append("SGJP reprezentuje imiesłów przez lemat czasownika (tag pact/ppas).")
   elif classification=="SAFE_FULL_MORPH_REPAIR": evidence.append("Jednoznaczny cel SGJP i pełna zgodność dostępnych cech fleksyjnych.")
   elif classification=="SAFE_PARTIAL_MORPH_REPAIR": evidence.append("Jednoznaczny cel SGJP, brak konfliktu fleksyjnego, ale porównanie cech jest częściowe.")
   elif classification=="MORPH_CONFLICT": evidence.append("Analiza SGJP celu jest sprzeczna z co najmniej jedną cechą full_postag Stanza.")
   rules=[{"orth":item["orth"],"lemma":src_lemma,"upos":upos,"replacement":target,"status":"review","observed_count":item["count"],"sgjp_status":item["status"],"compatible_lemmas":item["compatible"],"classification":classification,"stanza_morph_counts":dict(item["morph_counts"]),"morph_status":morph_status,"morph_comparisons":[d for d in morph_details if d["orth"]==item["orth"]]} for item in form_info if target in item["compatible"]]
   work.execute("INSERT OR REPLACE INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(src_lemma,target,upos,src_count,tc,src_docs,td,src_forms,tf,status,classification,json.dumps(evidence,ensure_ascii=False),json.dumps(rules,ensure_ascii=False)))
   accepted+=1
  if i%500==0:
   work.commit(); print(f"[sgjp] sources={i:,}/{len(sources):,}; candidates={work.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]:,}",file=sys.stderr,flush=True)
 work.commit(); state_set(work,"candidates_complete",1)

def collect_examples(search:Path,work,examples_per_rule:int,context:int):
 work.execute("DELETE FROM examples"); work.commit()
 wanted={}
 for s,t,u,rj in work.execute("SELECT source_lemma,target_lemma,upos,rules_json FROM candidates"):
  for r in json.loads(rj): wanted.setdefault((r["orth"],s,u),[]).append((s,t,u))
 if not wanted: return
 counts=Counter(); src=sqlite3.connect(f"file:{search.as_posix()}?mode=ro",uri=True); src.row_factory=sqlite3.Row
 try:
  for row in src.execute("SELECT doc_id,metadata_json,tokens,lemmas,upostags FROM docs ORDER BY doc_id"):
   toks=decode(row["tokens"],[]); lems=decode(row["lemmas"],[]); ups=decode(row["upostags"],[]); meta=decode(row["metadata_json"],{})
   if not(len(toks)==len(lems)==len(ups)): continue
   for idx,(o,l,u) in enumerate(zip(toks,lems,ups)):
    key=(clean(o),clean(l),clean(u).upper())
    for cand in wanted.get(key,[]):
     ck=(*cand,clean(o))
     if counts[ck]>=examples_per_rule: continue
     left=max(0,idx-context); right=min(len(toks),idx+context+1); ctx=" ".join(map(clean,toks[left:right])); label=clean(meta.get("Oryginalna_nazwa_pliku") or meta.get("Tytuł") or f"doc_id={row['doc_id']}")
     work.execute("INSERT OR IGNORE INTO examples VALUES(?,?,?,?,?,?,?,?)",(*cand,clean(o),int(row["doc_id"]),idx,ctx,label)); counts[ck]+=1
  work.commit(); state_set(work,"examples_complete",1)
 finally: src.close()

def report(work,search:Path,parquet:Path|None,md:Path,js:Path):
 rows=work.execute("SELECT * FROM candidates ORDER BY CASE classification WHEN 'SAFE_FULL_MORPH_REPAIR' THEN 0 WHEN 'SAFE_PARTIAL_MORPH_REPAIR' THEN 1 WHEN 'MORPH_CONFLICT' THEN 2 WHEN 'AMBIGUOUS_OR_HOMOGRAPHIC' THEN 3 WHEN 'REVIEW_OTHER' THEN 4 WHEN 'CASE_NORMALIZATION_ONLY' THEN 5 WHEN 'CONVENTION_GERUND' THEN 6 WHEN 'CONVENTION_PARTICIPLE_ACTIVE' THEN 7 WHEN 'CONVENTION_PARTICIPLE_PASSIVE' THEN 8 ELSE 9 END,source_count DESC").fetchall(); candidates=[]
 for r in rows:
  s,t,u,sc,tc,sd,td,sf,tf,status,classification,ev,rules=r
  examples=[{"orth":x[0],"doc_id":x[1],"token_index":x[2],"context":x[3],"document":x[4]} for x in work.execute("SELECT orth,doc_id,token_index,context,document_label FROM examples WHERE source_lemma=? AND target_lemma=? AND upos=?",(s,t,u))]
  candidates.append({"source_lemma":s,"target_lemma":t,"upos":u,"source_count":sc,"target_count":tc,"source_docs":sd,"target_docs":td,"source_forms":sf,"target_forms":tf,"sgjp_status":status,"classification":classification,"evidence":json.loads(ev),"rules":json.loads(rules),"examples":examples})
 payload={"schema_version":1,"tool":"experimental_stanza_lemma_repair_sqlite","version":VERSION,"generated_at":now(),"search":str(search),"search_sha256":sha256(search),"parquet":str(parquet) if parquet else None,"workspace":str(Path(work.execute('PRAGMA database_list').fetchone()[2])),"summary":{"documents_processed":int(state_get(work,'last_doc_id','-1'))+1,"form_rows":work.execute('SELECT COUNT(*) FROM form_stats').fetchone()[0],"lemma_rows":work.execute('SELECT COUNT(*) FROM lemma_stats').fetchone()[0],"sgjp_cached_forms":work.execute('SELECT COUNT(*) FROM sgjp_cache').fetchone()[0],"candidates":len(candidates),"classification_counts":dict(Counter(c["classification"] for c in candidates))},"candidates":candidates}
 js.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
 lines=["# Skalowalny audyt D3 lematyzacji Stanza z SGJP","",f"**Wygenerowano:** `{payload['generated_at']}`  ",f"**Źródło:** `{search}`  ",f"**Workspace:** `{payload['workspace']}`","","## Podsumowanie","",f"- Dokumenty: **{payload['summary']['documents_processed']:,}**",f"- Agregaty form: **{payload['summary']['form_rows']:,}**",f"- Lematy + UPOS: **{payload['summary']['lemma_rows']:,}**",f"- Formy sprawdzone w SGJP: **{payload['summary']['sgjp_cached_forms']:,}**",f"- Kandydaci: **{len(candidates):,}**"]+[f"- {k}: **{v:,}**" for k,v in sorted(payload["summary"]["classification_counts"].items())]+["","Do dalszego zatwierdzania służą przede wszystkim `SAFE_FULL_MORPH_REPAIR` i ostrożniej `SAFE_PARTIAL_MORPH_REPAIR`. Pozostałe klasy są diagnostyczne.","","Wszystkie reguły pozostają w stanie `review`. Audyt nie modyfikował `.search` ani Parquetu.",""]
 for i,c in enumerate(candidates,1):
  lines += [f"## {i}. `{c['source_lemma']}` → `{c['target_lemma']}` ({c['upos']})","",f"**Klasyfikacja:** `{c['classification']}`  ",f"**SGJP:** `{c['sgjp_status']}`  ",f"**Źródło:** {c['source_count']:,} wystąpień, {c['source_docs']:,} dokumentów, {c['source_forms']} form  ",f"**Cel:** {c['target_count']:,} wystąpień, {c['target_docs']:,} dokumentów, {c['target_forms']} form","","**Reguły:**"]
  for r in c['rules']: lines.append(f"- `{r['orth']}` + `{r['lemma']}` + `{r['upos']}` → `{r['replacement']}` ({r['observed_count']} wystąpień; morph: `{r.get('morph_status')}`; Stanza: `{r.get('stanza_morph_counts')}`)")
  if c['examples']:
   lines += ["","**Przykłady:**"]+[f"- `{x['document']}`: {x['context']}" for x in c['examples']]
  lines.append("")
 md.write_text("\n".join(lines)+"\n",encoding="utf-8")

def main(argv=None):
 p=argparse.ArgumentParser(description="Skalowalny audyt D3 lematów z Korpusuj .search i SGJP",allow_abbrev=False)
 p.add_argument("--search",required=True); p.add_argument("--parquet"); p.add_argument("--workspace",required=True); p.add_argument("--report",required=True); p.add_argument("--json",required=True)
 p.add_argument("--resume",action="store_true"); p.add_argument("--reset",action="store_true"); p.add_argument("--commit-docs",type=int,default=100); p.add_argument("--min-source-count",type=int,default=2); p.add_argument("--min-source-docs",type=int,default=2); p.add_argument("--max-source-forms",type=int,default=2); p.add_argument("--max-candidates-per-source",type=int,default=5); p.add_argument("--examples-per-rule",type=int,default=4); p.add_argument("--context-tokens",type=int,default=12); p.add_argument("--exclude-upos",default=",".join(sorted(DEFAULT_EXCLUDED)))
 a=p.parse_args(argv); search=Path(a.search).resolve(); workspace=Path(a.workspace).resolve(); md=Path(a.report).resolve(); js=Path(a.json).resolve(); parquet=Path(a.parquet).resolve() if a.parquet else None
 if not search.is_file(): raise AuditError(f"Brak .search: {search}")
 if a.reset and workspace.exists(): workspace.unlink()
 if workspace.exists() and not a.resume and not a.reset: raise AuditError("Workspace istnieje. Użyj --resume albo --reset.")
 workspace.parent.mkdir(parents=True,exist_ok=True); md.parent.mkdir(parents=True,exist_ok=True); js.parent.mkdir(parents=True,exist_ok=True)
 work=connect_workspace(workspace)
 try:
  cols={r[1] for r in work.execute("PRAGMA table_info(candidates)")}
  if state_get(work,"tool_version") not in (None,VERSION) or "classification" not in cols:
   raise AuditError("Workspace pochodzi ze starszej wersji. Użyj nowej ścieżki --workspace albo uruchom z --reset.")
  source_sig=sha256(search); old=state_get(work,"search_sha256")
  if old and old!=source_sig: raise AuditError("Workspace należy do innego pliku .search.")
  state_set(work,"search_sha256",source_sig); state_set(work,"search_path",search); state_set(work,"tool_version",VERSION)
  aggregate(search,work,{x.strip().upper() for x in a.exclude_upos.split(',') if x.strip()},a.commit_docs)
  summarize(work)
  if state_get(work,"candidates_complete")!="1": build_candidates(work,a.min_source_count,a.min_source_docs,a.max_source_forms,a.max_candidates_per_source)
  if state_get(work,"examples_complete")!="1": collect_examples(search,work,a.examples_per_rule,a.context_tokens)
  report(work,search,parquet,md,js)
  print(json.dumps({"success":True,"workspace":str(workspace),"report":str(md),"json":str(js),"candidates":work.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]},ensure_ascii=False,indent=2)); return 0
 finally: work.close()

if __name__=="__main__":
 try: raise SystemExit(main())
 except AuditError as e: print(f"ERROR: {e}",file=sys.stderr); raise SystemExit(2)
