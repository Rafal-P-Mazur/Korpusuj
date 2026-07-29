# -*- coding: utf-8 -*-
"""Exception types raised while parsing, validating and executing search queries."""
from __future__ import annotations


class QueryValidationError(Exception):
    pass


class SearchExecutionError(Exception):
    pass


class QueryParseError(Exception):
    pass
