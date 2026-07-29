# -*- coding: utf-8 -*-
"""Shared backend objects used to connect loaded corpora with indexed search execution."""
from __future__ import annotations

import logging

import pandas as pd


class LazyCorpus:
    def __init__(self, parquet_path, search_path, columns=None, total_docs=0, meta=None):
        self.parquet_path = str(parquet_path); self.search_path = str(search_path)
        self.columns = list(columns or []); self.total_docs = int(total_docs or 0); self.meta = meta or {}; self._df = None
    def __len__(self): return self.total_docs
    @property
    def index(self): return range(self.total_docs)
    def materialize(self):
        if self._df is None:
            logging.warning("Fallback do pandas.read_parquet dla złożonego CQL: %s", self.parquet_path)
            self._df = pd.read_parquet(self.parquet_path); self.columns = list(self._df.columns); self.total_docs = len(self._df)
        return self._df
    def __getattr__(self, name): return getattr(self.materialize(), name)
