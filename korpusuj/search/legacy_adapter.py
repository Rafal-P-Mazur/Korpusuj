# -*- coding: utf-8 -*-
"""Controlled adapter for the maintained legacy search fallback.

The adapter records why fallback was selected and keeps compatibility calls
separate from the shared indexed search path.
"""
from __future__ import annotations
import os
from typing import Any, Callable


# KORPUSUJ_PATCH_145C1_SAFE_DIAGNOSTICS_IMPORT
try:
    from korpusuj.search.diagnostics import (
        korpusuj_diagnostics_enabled_145c1,
        korpusuj_verbose_diagnostics_enabled_145c1,
    )
except Exception:
    def korpusuj_diagnostics_enabled_145c1(config_obj=None):
        return False
    def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
        return False
# END KORPUSUJ_PATCH_145C1_SAFE_DIAGNOSTICS_IMPORT

LEGACY_ADAPTER_PATCH_MARKER_121 = "KORPUSUJ_MIGRATION_PATCH_121_LEGACY_ADAPTER_FACADE"
LEGACY_SOURCE_REGEX_ROUTE_DATAFRAME = "regex_route_dataframe"
LEGACY_SOURCE_SQLITE_FAIL_MATERIALIZED = "sqlite_fail_materialized"
LEGACY_SOURCE_DIRECT_DATAFRAME = "legacy_direct_dataframe"
LEGACY_REASON_REGEX_ROUTE = "query_requires_legacy_regex_backend+sqlite_route_disabled_or_unavailable"
LEGACY_REASON_SQLITE_EXCEPTION = "sqlite_executor_exception"
LEGACY_REASON_DIRECT_DATAFRAME = "non_lazy_dataframe_direct"
LEGACY_REASON_STRICT_SUPPRESSED = "strict_no_legacy_on_sqlite_exception"

def _bool_from_value_121(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "off", "disabled", "disable"}:
        return False
    if text in {"1", "true", "yes", "on", "enabled", "enable"}:
        return True
    return default

def legacy_fallback_on_sqlite_exception_enabled_121(config: Any = None) -> bool:
    """Return whether LazyCorpus SQLite exceptions may fall back to legacy.

    Default remains True. Tests/strict mode can set config or env:
    KORPUSUJ_LEGACY_FALLBACK_ON_SQLITE_EXCEPTION_121=0
    """
    env_value = os.environ.get("KORPUSUJ_LEGACY_FALLBACK_ON_SQLITE_EXCEPTION_121")
    if env_value is not None:
        return _bool_from_value_121(env_value, default=True)
    try:
        if isinstance(config, dict) and "legacy_fallback_on_sqlite_exception_121" in config:
            return _bool_from_value_121(config.get("legacy_fallback_on_sqlite_exception_121"), default=True)
    except Exception:
        pass
    return True

def legacy_route_payload_121(*, legacy_source: str, legacy_reason: str, query: Any = None,
                             selected_corpus: Any = None, df: Any = None,
                             route_name: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "legacy_source": legacy_source,
        "legacy_reason": legacy_reason,
        "route_name": route_name or legacy_source,
        "query": query,
        "selected_corpus": selected_corpus,
        "df_type": type(df).__name__ if df is not None else None,
    }
    if extra:
        payload.update(extra)
    return payload

def call_legacy_find_lemma_context_121(legacy_impl: Callable[..., Any], query: Any, df: Any,
                                       selected_corpus: Any, left_context_size: int = 10,
                                       right_context_size: int = 10,
                                       warnings_list: list[str] | None = None, *,
                                       legacy_source: str, legacy_reason: str,
                                       route_name: str | None = None, logger: Any = None,
                                       extra: dict[str, Any] | None = None) -> Any:
    """Log a stable adapter-boundary marker, then call legacy implementation."""
    payload = legacy_route_payload_121(
        legacy_source=legacy_source,
        legacy_reason=legacy_reason,
        query=query,
        selected_corpus=selected_corpus,
        df=df,
        route_name=route_name,
        extra=extra,
    )
    try:
        if logger is not None:
            if korpusuj_diagnostics_enabled_145c1():
                logger.info("[DIAG legacy.adapter] event=%r data=%r", "call_legacy", payload)
    except Exception:
        pass
    return legacy_impl(query, df, selected_corpus, left_context_size, right_context_size, warnings_list)

# KORPUSUJ_MIGRATION_PATCH_122_STRICT_SQLITE_EXCEPTION_AND_LEGACY_ROUTE_OBSERVABILITY_NO_BODY_SLICING
LEGACY_ADAPTER_PATCH_MARKER_122 = "KORPUSUJ_MIGRATION_PATCH_122_STRICT_SQLITE_EXCEPTION_AND_LEGACY_ROUTE_OBSERVABILITY_NO_BODY_SLICING"
LEGACY_STRICT_ENV_VAR_122 = "KORPUSUJ_LEGACY_FALLBACK_ON_SQLITE_EXCEPTION_121"

LEGACY_ROUTE_NAMES_122 = (
    LEGACY_SOURCE_REGEX_ROUTE_DATAFRAME,
    LEGACY_SOURCE_SQLITE_FAIL_MATERIALIZED,
    LEGACY_SOURCE_DIRECT_DATAFRAME,
)


def legacy_no_slice_policy_122() -> dict[str, Any]:
    """Return the policy that keeps the maintained fallback matcher intact."""
    return {
        "do_not_slice_legacy_function": True,
        "legacy_function": "_legacy_find_lemma_context",
        "policy": "keep_body_intact_behind_adapter_facade",
        "allowed_future_moves": [
            "strict-mode hardening",
            "route observability",
            "whole-function move only if dependency injection is proven safe",
        ],
        "disallowed_next_patches": [
            "extract nested legacy helpers one by one",
            "split matcher internals into partial modules",
            "partial slice of _legacy_find_lemma_context body",
        ],
    }


def legacy_strict_config_snapshot_122(config: Any = None) -> dict[str, Any]:
    """Return an observable snapshot of strict legacy fallback config."""
    env_value = os.environ.get(LEGACY_STRICT_ENV_VAR_122)
    cfg_present = False
    cfg_value = None
    try:
        if isinstance(config, dict) and "legacy_fallback_on_sqlite_exception_121" in config:
            cfg_present = True
            cfg_value = config.get("legacy_fallback_on_sqlite_exception_121")
    except Exception:
        cfg_present = False
        cfg_value = None
    enabled = legacy_fallback_on_sqlite_exception_enabled_121(config)
    return {
        "env_var": LEGACY_STRICT_ENV_VAR_122,
        "env_value": env_value,
        "config_key": "legacy_fallback_on_sqlite_exception_121",
        "config_present": cfg_present,
        "config_value": cfg_value,
        "fallback_enabled": bool(enabled),
        "strict_mode": not bool(enabled),
    }


def legacy_observability_payload_122(
    *,
    legacy_source: str,
    legacy_reason: str,
    query: Any = None,
    selected_corpus: Any = None,
    df: Any = None,
    route_name: str | None = None,
    event: str = "route_enter",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = legacy_route_payload_121(
        legacy_source=legacy_source,
        legacy_reason=legacy_reason,
        query=query,
        selected_corpus=selected_corpus,
        df=df,
        route_name=route_name,
        extra=extra,
    )
    payload.update({
        "event": event,
        "patch": "122",
        "no_slice_policy": True,
        "known_route": legacy_source in LEGACY_ROUTE_NAMES_122,
    })
    return payload


def log_legacy_route_observability_122(
    logger: Any,
    *,
    legacy_source: str,
    legacy_reason: str,
    query: Any = None,
    selected_corpus: Any = None,
    df: Any = None,
    route_name: str | None = None,
    event: str = "route_enter",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log structured observability data for entry to or exit from a legacy search route."""
    payload = legacy_observability_payload_122(
        legacy_source=legacy_source,
        legacy_reason=legacy_reason,
        query=query,
        selected_corpus=selected_corpus,
        df=df,
        route_name=route_name,
        event=event,
        extra=extra,
    )
    try:
        if logger is not None:
            if korpusuj_diagnostics_enabled_145c1():
                logger.info("[DIAG legacy.route] event=%r data=%r", event, payload)
    except Exception:
        pass
    return payload


def legacy_adapter_selftest_122(config: Any = None) -> dict[str, Any]:
    """Tiny non-invasive self-test hook for scan/diagnostics."""
    return {
        "patch_marker": LEGACY_ADAPTER_PATCH_MARKER_122,
        "no_slice_policy": legacy_no_slice_policy_122(),
        "strict_config": legacy_strict_config_snapshot_122(config),
        "route_names": list(LEGACY_ROUTE_NAMES_122),
        "has_call_boundary": callable(call_legacy_find_lemma_context_121),
    }

