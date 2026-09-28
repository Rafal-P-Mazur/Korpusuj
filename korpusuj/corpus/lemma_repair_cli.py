# -*- coding: utf-8 -*-
"""CLI produkcyjnej uslugi korekty lematow D3."""
from __future__ import annotations
import argparse, json, sys
from .creator_core import NullProgressReporter
from .lemma_repair_models import LemmaRepairError, LemmaRepairOptions, LemmaRepairPaths
from . import lemma_repair_service as service

def parser():
    p=argparse.ArgumentParser(prog="python -m korpusuj.corpus.lemma_repair_cli",allow_abbrev=False)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("audit","prepare","preview","apply","run","status"):
        q=sub.add_parser(name)
        q.add_argument("--parquet",required=True); q.add_argument("--search",required=True); q.add_argument("--workdir")
        if name in {"apply","run"}: q.add_argument("--output")
        if name=="run": q.add_argument("--apply-approved",action="store_true")
        if name not in {"status"}: q.add_argument("--mode",choices=("common-core","common-core-plus","review"),default="common-core")
        q.add_argument("--batch-size",type=int,default=128)
    return p

def main(argv=None):
    a=parser().parse_args(argv); paths=LemmaRepairPaths.build(a.parquet,a.search,a.workdir,getattr(a,"output",None))
    if a.command=="status":
        if not paths.status_json().is_file(): raise LemmaRepairError("Brak stanu korekty dla wskazanego korpusu.")
        print(paths.status_json().read_text(encoding="utf-8")); return 0
    options=LemmaRepairOptions(mode=a.mode,batch_size=a.batch_size); reporter=NullProgressReporter()
    fn={"audit":service.audit,"prepare":service.prepare,"preview":service.dry_run,"apply":service.apply_approved}.get(a.command)
    result=service.run(paths,options,reporter,getattr(a,"apply_approved",False)) if a.command=="run" else fn(paths,options,reporter)
    print(json.dumps({"success":result.success,"stage":result.stage,"status":result.status_path,"data":result.data},ensure_ascii=False,indent=2)); return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except LemmaRepairError as exc: print(f"ERROR: {exc}",file=sys.stderr); raise SystemExit(2)
