# -*- coding: utf-8 -*-
"""Public CLI for creating, rebuilding and inspecting the derived index artifact set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from korpusuj.dependency.lifecycle import build_index_artifacts_atomic, dependency_cache_path, inspect_index_artifacts
from korpusuj.index.sqlite_index import INDEX_PROFILES, search_sidecar_path

SCHEMA_VERSION = 2
STATUS_EXIT_CODES = {"fresh": 0, "missing": 3, "stale": 4, "incompatible": 5, "corrupt": 6}
ALLOWED_ATTRS = tuple(INDEX_PROFILES["full"])


class CliInputError(ValueError): pass


def _parser():
    parser=argparse.ArgumentParser(prog="python -m korpusuj.index.cli",description="Zarządzanie zestawem indeksów .search + .dep_cache.")
    subs=parser.add_subparsers(dest="command",required=True)
    for name in ("create","rebuild","status"):
        sub=subs.add_parser(name); sub.add_argument("parquet"); sub.add_argument("--index",dest="index_path")
        attrs=sub.add_mutually_exclusive_group(); attrs.add_argument("--profile",choices=tuple(INDEX_PROFILES),default=None); attrs.add_argument("--attrs")
        sub.add_argument("--format",choices=("json","text"),default="json"); sub.add_argument("--pretty",action="store_true")
        if name!="status": sub.add_argument("--force",action="store_true"); sub.add_argument("--progress",choices=("auto","off","on"),default="auto")
    return parser


def _paths(args):
    parquet=Path(args.parquet).expanduser()
    if parquet.suffix.lower()!=".parquet": raise CliInputError("Źródło musi mieć rozszerzenie .parquet")
    if not parquet.is_file(): raise CliInputError(f"Plik Parquet nie istnieje: {parquet}")
    index=Path(args.index_path).expanduser() if args.index_path else Path(search_sidecar_path(parquet))
    if index.suffix.lower()!=".search": raise CliInputError("Indeks musi mieć rozszerzenie .search")
    return parquet.resolve(),index.resolve(),Path(dependency_cache_path(parquet)).resolve()


def _attrs(args):
    if args.attrs is None: return tuple(INDEX_PROFILES[args.profile or "full"])
    values=[x.strip().lower() for x in args.attrs.split(",") if x.strip()]
    if not values: raise CliInputError("--attrs nie może być pustą listą")
    unknown=sorted(set(values)-set(ALLOWED_ATTRS))
    if unknown: raise CliInputError("Nieobsługiwane atrybuty: "+", ".join(unknown))
    if len(values)!=len(set(values)): raise CliInputError("--attrs nie może zawierać duplikatów")
    selected=set(values); return tuple(x for x in ALLOWED_ATTRS if x in selected)


def _progress(mode):
    enabled=mode=="on" or (mode=="auto" and sys.stderr.isatty())
    return (lambda message: print(message,file=sys.stderr,flush=True)) if enabled else None


def _artifact_public(item, path_key):
    return {"status":item.get("status"),"path":item.get(path_key),"reasons":list(item.get("reasons") or []),
            "integrity_check":item.get("integrity_check"),"meta":item.get("meta") or {},"row_count":item.get("row_count")}


def _payload(command,parquet,index,dep,ok,**extra):
    value={"schema_version":SCHEMA_VERSION,"command":command,"ok":bool(ok),"parquet_path":str(parquet) if parquet else None,
           "index_path":str(index) if index else None,"dep_cache_path":str(dep) if dep else None}; value.update(extra); return value


def _public(result):
    return {"status":result["status"],"artifacts":{"search":_artifact_public(result["search"],"index_path"),
            "dep_cache":_artifact_public(result["dep_cache"],"cache_path")}}


def _emit(payload,fmt,pretty):
    if fmt=="json": print(json.dumps(payload,ensure_ascii=False,indent=2 if pretty else None,separators=None if pretty else (",",":"))); return
    print(" | ".join(str(x) for x in (payload.get("command","index").upper(),payload.get("status","error").upper(),payload.get("action") or "",payload.get("index_path") or "") if x))
    if payload.get("message"): print(payload["message"])
    for name,item in (payload.get("details") or {}).get("artifacts",{}).items(): print(f"{name}: {item.get('status')} | {item.get('path')}")


def main(argv:Sequence[str]|None=None)->int:
    """Run the index lifecycle CLI and return its documented exit code."""
    args=_parser().parse_args(argv); parquet=index=dep=None
    try:
        parquet,index,dep=_paths(args); attrs=_attrs(args)
        before=inspect_index_artifacts(parquet,index,attrs,check_integrity=True)
        if args.command=="status": result=before; action=None
        elif args.command=="create":
            if (index.exists() or dep.exists()) and not args.force: raise CliInputError("Co najmniej jeden artefakt już istnieje; użyj --force")
            result=build_index_artifacts_atomic(parquet,index,attrs,progress_callback=_progress(args.progress)); action="replaced" if args.force else "created"
        else:
            if not index.exists() and not dep.exists(): raise CliInputError("Zestaw indeksów nie istnieje; użyj create")
            if before["status"]=="fresh" and not args.force: result=before; action="unchanged"
            else: result=build_index_artifacts_atomic(parquet,index,attrs,progress_callback=_progress(args.progress)); action="rebuilt"
        status=result["status"]; ok=status=="fresh"; payload=_payload(args.command,parquet,index,dep,ok,status=status,action=action,requested_attrs=list(attrs),details=_public(result))
        code=STATUS_EXIT_CODES.get(status,1) if args.command=="status" else (0 if ok else 1)
    except CliInputError as exc:
        payload=_payload(args.command,parquet,index,dep,False,status="error",action=None,error_type="input",message=str(exc)); code=2
    except Exception as exc:
        print(f"Błąd index CLI: {type(exc).__name__}: {exc}",file=sys.stderr)
        payload=_payload(args.command,parquet,index,dep,False,status="error",action=None,error_type="runtime",message=str(exc)); code=1
    _emit(payload,args.format,args.pretty); return code


if __name__=="__main__": raise SystemExit(main())
