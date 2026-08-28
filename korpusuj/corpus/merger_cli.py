"""CLI for the guarded Korpusuj corpus merger."""
from __future__ import annotations
import argparse, json, sys
from .merger import CorpusMergeError, merge_corpora

def parser():
    p = argparse.ArgumentParser(description="Scal gotowe, zgodne korpusy Korpusuj bez NLP.")
    p.add_argument("--input", action="append", required=True, dest="inputs")
    p.add_argument("--output", required=True)
    p.add_argument("--report")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--no-duplicate-check", action="store_true")
    p.add_argument("--allow-undeclared-annotation-layers", action="store_true", help="Zezwól, gdy wszystkie wejścia są historycznymi korpusami bez annotation_layers.")
    return p

def main(argv=None):
    p, a = parser(), parser().parse_args(argv)
    if len(a.inputs) < 2: p.error("--input musi wystąpić co najmniej dwa razy")
    if a.batch_size < 1: p.error("--batch-size musi być dodatnie")
    def progress(done, total): print(f"[merge] {done}/{total}", file=sys.stderr)
    try:
        result = merge_corpora(a.inputs, a.output, report_path=a.report, replace=a.replace,
                               batch_size=a.batch_size, check_duplicates=not a.no_duplicate_check,
                               allow_undeclared_annotation_layers=a.allow_undeclared_annotation_layers,
                               progress_callback=progress)
    except CorpusMergeError as exc:
        print(json.dumps({"success": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2
    except Exception as exc:
        print(json.dumps({"success": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0
if __name__ == "__main__": raise SystemExit(main())
