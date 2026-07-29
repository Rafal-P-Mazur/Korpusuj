# -*- coding: utf-8 -*-
"""State models shared by search execution and result presentation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchState:
    query: str = ""
    corpus: str = ""
    results: list = field(default_factory=list)
    monthly_lemma_freq: dict = field(default_factory=dict)
    true_monthly_totals: dict = field(default_factory=dict)
    monthly_freq_for_use: dict = field(default_factory=dict)
    monthly_tfidf_for_use: dict = field(default_factory=dict)
    monthly_zscore_for_use: dict = field(default_factory=dict)
    lemma_df_cache: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    fq_data: list = field(default_factory=list)
    fq_data_token: list = field(default_factory=list)
    fq_data_month: list = field(default_factory=list)
    s_lemma_total_freq: list = field(default_factory=list)
    s_lemma_global_pmw: list = field(default_factory=list)
    s_lemma_global_tfidf: list = field(default_factory=list)
    unique_lemmas: set = field(default_factory=set)
    has_dates: bool = False
    colloc_data: list = field(default_factory=list)
    current_profile_dict: dict = field(default_factory=dict)
    profile_target_lemma: str = ""
    profile_data: list = field(default_factory=list)
    profile_rel_options: list = field(default_factory=list)
    profile_selected_rel: str = ""

