# -*- coding: utf-8 -*-
"""Focused source-contract tests for patch 173b5; engine.py is not imported."""
import ast
import re
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / "engine.py"
HELPER = "_gui_query_uses_coref_173b5"
BOUNDARY = "_try_run_gui_search_via_headless_service"
REASON = "coref_uses_native_gui_searchcursor_path_173b5"


def _tree_text():
    text = ENGINE.read_text(encoding="utf-8")
    return ast.parse(text, filename=str(ENGINE)), text


def _function(tree, name):
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, (name, len(found))
    return found[0]


def _detector():
    tree, _ = _tree_text()
    node = _function(tree, HELPER)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"re": re}
    exec(compile(module, str(ENGINE), "exec"), ns)
    return ns[HELPER]


def test_detector_accepts_supported_coref_variants():
    detect = _detector()
    assert detect('[coref="Polska"]')
    assert detect('[coref(H)="Polska"]')
    assert detect('[coref(P)="Polska"]')
    assert detect('[coref(M)="Polska"]')
    assert detect('[coref ( h ) != "Polska"]')


def test_detector_rejects_unrelated_queries():
    detect = _detector()
    assert not detect('[base="Polska"]')
    assert not detect('[orth="coreference"]')
    assert not detect('<tytuł="coref">')
    assert not detect('')
    assert not detect(None)


def test_boundary_skips_coref_before_service_call():
    tree, text = _tree_text()
    boundary = _function(tree, BOUNDARY)
    source = ast.get_source_segment(text, boundary)
    assert REASON in source
    assert f"{HELPER}(query)" in source
    assert source.index(f"{HELPER}(query)") < source.index("run_search_service(req, ctx)")
    assert '"used": False' in source
    assert '"status": "skipped"' in source


def test_native_sqlite_searchcursor_continuation_remains_present():
    _, text = _tree_text()
    assert "_prepare_and_find_search_backend_results(" in text
    assert "search_df = _make_lazy_corpus_for_search(selected_corpus, df)" in text
    assert "if _is_searchcursor_like(results):" in text
