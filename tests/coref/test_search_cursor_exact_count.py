# -*- coding: utf-8 -*-
"""Focused source-contract tests for patch 173b6."""
from __future__ import annotations

import ast
from pathlib import Path

CURSOR = Path(__file__).resolve().parents[2] / "korpusuj" / "search" / "cursor.py"
METHOD = "count_hits_estimate_is_exact"
MARKER = "KORPUSUJ_PATCH_173B6_COREF_EXACT_COUNT_CONTRACT"


def _source():
    return CURSOR.read_text(encoding="utf-8")


def _searchcursor_node():
    tree = ast.parse(_source(), filename=str(CURSOR))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SearchCursor"]
    assert len(classes) == 1
    return classes[0]


def _method_source():
    text = _source()
    cls = _searchcursor_node()
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == METHOD]
    assert len(methods) == 1
    return ast.get_source_segment(text, methods[0])


def test_coref_is_excluded_from_generic_indexed_exact_estimate():
    source = _method_source()
    assert MARKER in source
    assert "_is_coref_condition(cond)" in source
    assert "return False" in source


def test_coref_guard_precedes_generic_single_condition_exact_rule():
    source = _method_source()
    assert source.index("_is_coref_condition(cond)") < source.index("return (not self._plan_uses_dependency()")


def test_exact_count_still_exhausts_cursor_when_estimate_is_not_exact():
    text = _source()
    cls = _searchcursor_node()
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "count_hits"]
    assert len(methods) == 1
    source = ast.get_source_segment(text, methods[0])
    assert "if exact and self._count_cache is None" in source
    assert "self._ensure_all()" in source


def test_base_and_other_indexed_single_conditions_keep_existing_fast_rule():
    source = _method_source()
    assert "len(groups) == 1" in source
    assert 'len(groups[0].get("conds", [])) == 1' in source
    assert 'not self.plan.get("metadata_filters")' in source
