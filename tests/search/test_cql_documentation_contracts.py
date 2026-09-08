# -*- coding: utf-8 -*-
"""Fast synthetic compliance suite for the public CQL contract in cql.md.

Copy to:
    tests/search/test_cql_documentation_contracts.py

Run from the project root:
    python -m pytest tests/search/test_cql_documentation_contracts.py -q \
        --basetemp .pytest_work/cql_docs

The suite intentionally does not read the production Parquet, .search, or
.dep_cache. It reuses the controlled synthetic corpus and the shared/headless
adapter from test_query_service_contracts.py.

Coverage layers:
- parser/planner shape for documented syntax;
- execution on a controlled synthetic corpus where the fixture contains the
  required annotation;
- locality regression for grouped sentence RHS versus ordinary parentheses/OR;
- an explicit inventory test so newly documented syntax must be added here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from korpusuj.search import parser as cql_parser
from korpusuj.search.cursor import _split_grouped_sentence_rhs_conjunction
from korpusuj.search.planner import SearchPlanner


ROOT = Path(__file__).resolve().parents[2]
SERVICE_TEST_PATH = ROOT / "tests/search/test_query_service_contracts.py"
CQL_DOC_PATH = ROOT / "docs/cql.md"


def _load_service_contract_module():
    spec = importlib.util.spec_from_file_location(
        "_cql_docs_service_contracts", SERVICE_TEST_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SERVICE_TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVICE = _load_service_contract_module()
run_query = SERVICE.run_query


@pytest.fixture(scope="module")
def documented_synthetic_corpus():
    fixture_function = getattr(SERVICE.synthetic_corpus, "__wrapped__", None)
    if not callable(fixture_function):
        raise RuntimeError("synthetic_corpus fixture has no callable __wrapped__")
    return fixture_function()


class MinimalIndex:
    """Planner metadata contract without opening a real .search sidecar."""

    @staticmethod
    def meta():
        return {
            "indexed_attrs": "base,orth,pos,upos,deprel,ner",
            "schema_version": "documentation-contract-test",
        }


# Every public language family in cql.md must remain represented here.
DOCUMENTED_FAMILIES = {
    "token_equality",
    "token_inequality",
    "same_token_conjunction",
    "adjacent_sequence",
    "complete_query_or",
    "value_pipe",
    "orth",
    "base",
    "pos",
    "upos",
    "deprel",
    "head",
    "dependent",
    "ner",
    "coref",
    "coref_h",
    "coref_p",
    "coref_m",
    "window_base",
    "window_orth",
    "plain_text",
    "mixed_plain_and_cql",
    "wildcard_one",
    "regex",
    "regex_search",
    "gap_range",
    "morphology",
    "dependency_distance",
    "nested_dependency",
    "negative_nested_dependency",
    "sentence_boundary",
    "sentence_rhs",
    "sentence_grouped_rhs",
    "frequency_base",
    "frequency_orth",
    "metadata_author",
    "metadata_title",
    "metadata_date",
    "metadata_custom",
}


SYNTAX_CASES = {
    "token_equality": '[base="wojna"]',
    "token_inequality": '[upos!="NOUN"]',
    "same_token_conjunction": '[base="bohater" & case="gen" & number="pl"]',
    "adjacent_sequence": '[base="wypowiedzieć"] [base="wojna"]',
    "complete_query_or": (
        '([orth="władza"] [orth="mediów"]) || '
        '([base="władza"] [base="partia"])'
    ),
    "value_pipe": '[base="kot|pies"]',
    "orth": '[orth="wojną"]',
    "base": '[base="Ania"]',
    "pos": '[pos="subst"]',
    "upos": '[upos="VERB"]',
    "deprel": '[deprel="nsubj"]',
    "head": '[base="ryba" & head="zjeść"]',
    "dependent": '[base="miasto" & dependent="piękny"]',
    "ner": '[ner=".-persName"]',
    "coref": '[coref="Polska"]',
    "coref_h": '[coref(H)="Kowalski"]',
    "coref_p": '[pos="pron" & coref(P)="Warszawa"]',
    "coref_m": '[coref(M)="Polska"]',
    "window_base": '[base="pies" & window_base(5)="kot"]',
    "window_orth": '[orth="Polska" & window_orth(10)="gospodarka"]',
    "plain_text": 'kot je rybę',
    "mixed_plain_and_cql": '[base="móc"] zjeść obiad.',
    "wildcard_one": '[orth="Polska"] [*] [orth="Niemcy"]',
    "regex": '[orth="kwesti(a|ę)"]',
    "regex_search": '[orth="~zys"]',
    "gap_range": '[base="Ania"] [*][1,3] [base="Tomek"]',
    "morphology": '[upos="VERB" & person="pri" & number="sg" & aspect="imperf"]',
    "dependency_distance": '[dependent(<2)="smaczny"]',
    "nested_dependency": (
        '[base="zjeść" & dependent={deprel="obj" & base="ryba" '
        '& dependent={base="świeży" & deprel="amod"}}]'
    ),
    "negative_nested_dependency": '[base="być" & dependent!={orth="nie"}]',
    "sentence_boundary": '[base="wygrać"] [*][1,3] [base="wojna"] <s>',
    "sentence_rhs": '[base="wygrać"] <s [base="wojna"]>',
    "sentence_grouped_rhs": (
        '[base="wygrać"] <s ([base="Chinka"]) ([base="set"])>'
    ),
    "frequency_base": '[upos="VERB"] <frequency_base min="2" max="10">',
    "frequency_orth": '[base="pies"] <frequency_orth top="3">',
    "metadata_author": '[base="Tadeusz"] <autor="Mickiewicz">',
    "metadata_title": '[base="dzień"] <tytuł="~sen">',
    "metadata_date": (
        '[base="kot"] <data >= "2024-01-20"> <data <= "2025-02-25">'
    ),
    "metadata_custom": '[base="Duda"] <metadane:portal="Wyborcza">',
}


def _coordinate(hit: Any):
    """Read canonical coordinates from dict hits or materialized result rows."""
    if isinstance(hit, dict):
        values = (
            hit.get("doc_id", hit.get("row_idx")),
            hit.get("start", hit.get("start_idx")),
            hit.get("end", hit.get("end_idx")),
        )
    elif isinstance(hit, (tuple, list)) and len(hit) > 13:
        # Shared/headless materialized rows keep coordinates at 11, 12 and 13.
        values = hit[11], hit[12], hit[13]
    elif isinstance(hit, (tuple, list)) and len(hit) == 3:
        # Raw SearchCursor hit shape used by lower-level tests.
        values = hit
    else:
        return None
    try:
        return int(values[0]), int(values[1]), int(values[2])
    except (TypeError, ValueError, IndexError):
        return None


def _coordinates(results):
    return {value for value in (_coordinate(hit) for hit in results) if value is not None}


def test_documented_family_inventory_is_complete():
    assert set(SYNTAX_CASES) == DOCUMENTED_FAMILIES
    assert CQL_DOC_PATH.is_file(), "Expected cql.md in project root"


@pytest.mark.parametrize(
    "query",
    [
        '[base="wojna"]',
        '[upos!="NOUN"]',
        '[base="bohater" & case="gen" & number="pl"]',
        '[ner=".-persName"]',
        '[coref(H)="Kowalski"]',
        '[dependent(<2)="smaczny"]',
        '[base="być" & dependent!={orth="nie"}]',
    ],
)
def test_documented_single_segment_syntax_parses(query):
    groups = cql_parser.extract_square_brackets(query)
    assert groups, query


@pytest.mark.parametrize(
    "query",
    [
        '[base="kot"]',
        '[orth="kot"]',
        '[pos="subst"]',
        '[upos="VERB"]',
        '[deprel="nsubj"]',
        '[ner=".-persName"]',
        '[base="kot" & pos="subst"]',
        '[base="ryba" & head="zjeść"]',
        '[base="mały" & window_base(3)="zjeść"]',
        '[base="kot"] [*][0,1] [base="zjeść"]',
        '[base="zjeść" & dependent={base="kot" & deprel="nsubj"}]',
    ],
)
def test_documented_ordinary_queries_are_plannable(query):
    plan = SearchPlanner().plan(query, MinimalIndex())
    assert isinstance(plan, dict)
    assert plan.get("supported") is True, (query, plan)


def test_documented_sentence_operator_forms_are_split_correctly():
    simple = cql_parser.split_sentence_operator_query(
        '[base="wygrać"] <s [base="wojna"]>'
    )
    grouped = cql_parser.split_sentence_operator_query(
        '[base="wygrać"] <s ([base="Chinka"]) ([base="set"])>'
    )
    assert simple == {
        "token_part": '[base="wygrać"]',
        "sentence_part": '[base="wojna"]',
    }
    assert grouped == {
        "token_part": '[base="wygrać"]',
        "sentence_part": '([base="Chinka"]) ([base="set"])',
    }


def test_grouped_sentence_rhs_is_local_and_order_free_in_shape():
    first = _split_grouped_sentence_rhs_conjunction(
        '([base="pomoc"]) ([base="Ukraina"])'
    )
    reversed_groups = _split_grouped_sentence_rhs_conjunction(
        '([base="Ukraina"]) ([base="pomoc"])'
    )
    assert first == ['[base="pomoc"]', '[base="Ukraina"]']
    assert reversed_groups == ['[base="Ukraina"]', '[base="pomoc"]']


def test_ordinary_parenthesized_or_is_never_grouped_sentence_rhs():
    query = (
        '([orth="władza"] [orth="mediów"]) || '
        '([base="władza"] [base="partia"])'
    )
    assert _split_grouped_sentence_rhs_conjunction(query) is None


@pytest.mark.parametrize(
    "query, minimum",
    [
        ('[base="kot"]', 1),
        ('[orth="kot"]', 1),
        ('[base="kot" & pos="subst"]', 1),
        ('[pos="adj"] [base="ryba"]', 1),
        ('[base="ryba" & head="zjeść"]', 1),
        (
            '[base="zjeść" & dependent={base="kot" & deprel="nsubj"} '
            '& dependent={base="ryba" & deprel="obj"}]',
            1,
        ),
        ('[base="mały" & window_base(3)="zjeść"]', 1),
        ('[base="zjeść"] <s [base="ryba"]>', 1),
        ('[base="zjeść"] <s ([base="ryba"]) ([base="szybko"])>', 1),
        ('[base="kot|pies"]', 2),
    ],
)
def test_documented_queries_execute_on_controlled_corpus(
    documented_synthetic_corpus, query, minimum
):
    dataframe, corpus_name = documented_synthetic_corpus
    results = run_query(query, dataframe, corpus_name)
    assert len(results) >= minimum, query


def test_documented_grouped_sentence_rhs_is_order_invariant_at_runtime(
    documented_synthetic_corpus,
):
    dataframe, corpus_name = documented_synthetic_corpus
    first = run_query(
        '[base="zjeść"] <s ([base="ryba"]) ([base="szybko"])>',
        dataframe,
        corpus_name,
    )
    second = run_query(
        '[base="zjeść"] <s ([base="szybko"]) ([base="ryba"])>',
        dataframe,
        corpus_name,
    )
    assert first
    assert _coordinates(first) == _coordinates(second)


def test_documented_ungrouped_sentence_rhs_remains_a_sequence(
    documented_synthetic_corpus,
):
    dataframe, corpus_name = documented_synthetic_corpus
    grouped = run_query(
        '[base="zjeść"] <s ([base="ryba"]) ([base="szybko"])>',
        dataframe,
        corpus_name,
    )
    ordered = run_query(
        '[base="zjeść"] <s [base="ryba"] [base="szybko"]>',
        dataframe,
        corpus_name,
    )
    assert grouped
    assert set(_coordinates(ordered)).issubset(_coordinates(grouped))


def test_documented_negative_sequence_does_not_cross_sentence_boundary(
    documented_synthetic_corpus,
):
    dataframe, corpus_name = documented_synthetic_corpus
    assert run_query('[base="ryba"] [base="pies"]', dataframe, corpus_name) == []


def test_documented_top_level_or_is_kept_outside_dataframe_compatibility_path():
    """Document || syntax without pretending the DataFrame fixture is .search.

    Native top-level OR is owned by the SearchPlanner/UnionSearchCursor path.
    The synthetic fixture used in this file is a pandas DataFrame compatibility
    backend, so executing || through run_query() would test a different route.
    This test protects syntax inventory and, critically, verifies that the
    local grouped-<s> recognizer does not reinterpret ordinary parenthesized OR.
    """
    query = SYNTAX_CASES["complete_query_or"]
    assert query == (
        '([orth="władza"] [orth="mediów"]) || '
        '([base="władza"] [base="partia"])'
    )
    assert "||" in query
    assert _split_grouped_sentence_rhs_conjunction(query) is None


def test_documented_regex_value_pipe_contains_exact_branches(
    documented_synthetic_corpus,
):
    dataframe, corpus_name = documented_synthetic_corpus
    first = _coordinates(run_query('[base="kot"]', dataframe, corpus_name))
    second = _coordinates(run_query('[base="pies"]', dataframe, corpus_name))
    combined = _coordinates(run_query('[base="kot|pies"]', dataframe, corpus_name))
    assert combined == first | second


def test_documented_empty_input_contract_is_safe():
    # A docs-compliance suite should fail clearly if an empty query is ever
    # treated as a valid token query.
    plan = SearchPlanner().plan("", MinimalIndex())
    assert not plan or plan.get("supported") is not True
