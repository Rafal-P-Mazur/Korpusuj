# -*- coding: utf-8 -*-
"""Diagnostic logging and plan-summary helpers for the shared search engine."""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional


SEARCH_DIAG_ENV = "KORPUSUJ_SEARCH_DIAG"
_config_provider: Optional[Callable[[], dict]] = None


def configure_search_diagnostics(*, config_provider: Optional[Callable[[], dict]] = None) -> None:
    global _config_provider
    _config_provider = config_provider


def search_diag_enabled() -> bool:
    try:
        env = str(os.environ.get(SEARCH_DIAG_ENV, "")).strip().lower()
        if env in ("1", "true", "tak", "yes", "on"):
            return True
    except Exception:
        pass
    try:
        cfg = _config_provider() if _config_provider is not None else {}
        return bool((cfg or {}).get("search_diag", False))
    except Exception:
        return False


def search_diag_log(message, *args, **kwargs):
    """Safe diagnostic logger; must never interrupt search execution."""
    try:
        if not korpusuj_diagnostics_enabled_145c1():
            return
    except Exception:
        return
    if not search_diag_enabled():
        return
    try:
        logging.info("[DIAG search] " + message, *args, **kwargs)
    except Exception:
        pass


def summarize_search_plan_for_log(plan):
    try:
        return {
            "supported": plan.get("supported"),
            "reason": plan.get("reason"),
            "uses_dependency": plan.get("uses_dependency"),
            "groups_count": len(plan.get("token_groups") or []),
            "metadata_filters_count": len(plan.get("metadata_filters") or []),
        }
    except Exception:
        return {"supported": None}

# PATCH_130_MARKER: semantic verbose diagnostics helpers for diagnostics.py
def search_verbose_diagnostics_enabled(config=None) -> bool:
    # Return True when deep Korpusuj search diagnostics should be emitted.
    truthy = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
    try:
        for env_name in (
            "KORPUSUJ_VERBOSE_DIAGNOSTICS",
            "KORPUSUJ_SEARCH_VERBOSE",
            "KORPUSUJ_SEARCH_MIGRATION_DEBUG",
        ):
            if os.environ.get(env_name, "").strip() in truthy:
                return True
        if isinstance(config, dict):
            for key in (
                "verbose_diagnostics",
                "search_verbose",
                "search_diagnostics_verbose",
                "search_migration_debug",
            ):
                value = config.get(key)
                if value is True:
                    return True
                if isinstance(value, str) and value.strip() in truthy:
                    return True
    except Exception:
        return False
    return False


def search_verbose_diag_log(logger, marker: str, semantic_event: str, message: str, *args, config=None, **kwargs) -> None:
    # Emit deep diagnostic log only when verbose diagnostics are enabled.
    if not search_verbose_diagnostics_enabled(config=config):
        return
    try:
        prefix = f"[{marker}] event={semantic_event!r} " if marker else f"event={semantic_event!r} "
        logger.info(prefix + message, *args, **kwargs)
    except Exception:
        return

# KORPUSUJ_PATCH_145C1_SAFE_CANONICAL_DIAGNOSTICS_GATE
def _korpusuj_truthy_145c1(value):
    try:
        return str(value).strip().lower() in {"1", "true", "yes", "tak", "on", "debug", "verbose"}
    except Exception:
        return False


def korpusuj_diagnostics_enabled_145c1(config_obj=None):
    import os as _os_145c1
    for env_name in (
        "KORPUSUJ_VERBOSE_DIAGNOSTICS",
        "KORPUSUJ_SEARCH_VERBOSE",
        "KORPUSUJ_SEARCH_MIGRATION_DEBUG",
        "KORPUSUJ_137_DIAGNOSTIC_LOGS",
        "KORPUSUJ_VERBOSE_EXECUTION_DIAGNOSTICS",
        "KORPUSUJ_VERBOSE_LOGS",
        "KORPUSUJ_VERBOSE",
        "KORPUSUJ_DEBUG_LOGS",
    ):
        try:
            if _korpusuj_truthy_145c1(_os_145c1.environ.get(env_name, "")):
                return True
        except Exception:
            pass
    try:
        cfg = config_obj if isinstance(config_obj, dict) else globals().get("config", None)
        if isinstance(cfg, dict):
            for key in (
                "logging_diagnostics_logs", "logging_verbose", "verbose_diagnostics",
                "search_verbose", "search_diagnostics_verbose", "search_migration_debug",
            ):
                if cfg.get(key) is True or _korpusuj_truthy_145c1(cfg.get(key)):
                    return True
    except Exception:
        pass
    return False


def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
    return korpusuj_diagnostics_enabled_145c1(config_obj=config_obj)
# END KORPUSUJ_PATCH_145C1_SAFE_CANONICAL_DIAGNOSTICS_GATE

# KORPUSUJ_PATCH_145C2A_SPLIT_DIAGNOSTIC_AND_VERBOSE_GATES
# Category-aware logging gates. 145c1 function names are kept as compatibility wrappers.
def _korpusuj_truthy_145c2(value):
    try:
        return str(value).strip().lower() in {"1", "true", "yes", "tak", "on", "debug", "verbose"}
    except Exception:
        return False


def _korpusuj_config_value_145c2(config_obj, key):
    try:
        if isinstance(config_obj, dict) and key in config_obj:
            return config_obj.get(key)
        cfg = globals().get("config", None)
        if isinstance(cfg, dict):
            return cfg.get(key)
    except Exception:
        pass
    return None


def korpusuj_logging_diagnostics_enabled_145c2(config_obj=None):
    import os as _os_145c2
    for env_name in (
        "KORPUSUJ_VERBOSE_DIAGNOSTICS",
        "KORPUSUJ_SEARCH_MIGRATION_DEBUG",
        "KORPUSUJ_137_DIAGNOSTIC_LOGS",
        "KORPUSUJ_DEBUG_LOGS",
    ):
        try:
            if _korpusuj_truthy_145c2(_os_145c2.environ.get(env_name, "")):
                return True
        except Exception:
            pass
    for key in (
        "logging_diagnostics_logs",
        "verbose_diagnostics",
        "search_diagnostics_verbose",
        "search_migration_debug",
    ):
        value = _korpusuj_config_value_145c2(config_obj, key)
        if value is True or _korpusuj_truthy_145c2(value):
            return True
    return False


def korpusuj_logging_verbose_enabled_145c2(config_obj=None):
    import os as _os_145c2
    for env_name in (
        "KORPUSUJ_VERBOSE_EXECUTION_DIAGNOSTICS",
        "KORPUSUJ_VERBOSE_LOGS",
        "KORPUSUJ_VERBOSE",
        "KORPUSUJ_SEARCH_VERBOSE",
        "KORPUSUJ_SEARCH_MIGRATION_DEBUG",
        "KORPUSUJ_137_DIAGNOSTIC_LOGS",
        "KORPUSUJ_DEBUG_LOGS",
    ):
        try:
            if _korpusuj_truthy_145c2(_os_145c2.environ.get(env_name, "")):
                return True
        except Exception:
            pass
    for key in (
        "logging_verbose",
        "search_verbose",
        "search_migration_debug",
    ):
        value = _korpusuj_config_value_145c2(config_obj, key)
        if value is True or _korpusuj_truthy_145c2(value):
            return True
    return False


def korpusuj_diagnostics_enabled_145c1(config_obj=None):
    return korpusuj_logging_diagnostics_enabled_145c2(config_obj=config_obj)


def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
    return korpusuj_logging_verbose_enabled_145c2(config_obj=config_obj)
# END KORPUSUJ_PATCH_145C2A_SPLIT_DIAGNOSTIC_AND_VERBOSE_GATES

# KORPUSUJ_PATCH_145C2B_STRICT_VERBOSE_GATE
# Stricter split than 145c2a:
# - diagnostics gate = route/state/contract logs;
# - verbose gate = timing/materialization/prefetch/profile logs.
# Compatibility function names are kept, but semantics are now split.
def _korpusuj_truthy_145c2b(value):
    try:
        return str(value).strip().lower() in {"1", "true", "yes", "tak", "on", "debug", "verbose"}
    except Exception:
        return False


def _korpusuj_config_value_145c2b(config_obj, key):
    try:
        if isinstance(config_obj, dict) and key in config_obj:
            return config_obj.get(key)
        cfg = globals().get("config", None)
        if isinstance(cfg, dict):
            return cfg.get(key)
    except Exception:
        pass
    return None


def korpusuj_logging_diagnostics_enabled_145c2(config_obj=None):
    import os as _os_145c2b
    for env_name in (
        "KORPUSUJ_DIAGNOSTIC_LOGS",
        "KORPUSUJ_VERBOSE_DIAGNOSTICS",
        "KORPUSUJ_137_DIAGNOSTIC_LOGS",
    ):
        try:
            if _korpusuj_truthy_145c2b(_os_145c2b.environ.get(env_name, "")):
                return True
        except Exception:
            pass
    for key in (
        "logging_diagnostics_logs",
        "verbose_diagnostics",
        "search_diagnostics_verbose",
    ):
        value = _korpusuj_config_value_145c2b(config_obj, key)
        if value is True or _korpusuj_truthy_145c2b(value):
            return True
    return False


def korpusuj_logging_verbose_enabled_145c2(config_obj=None):
    import os as _os_145c2b
    for env_name in (
        "KORPUSUJ_VERBOSE_EXECUTION_DIAGNOSTICS",
        "KORPUSUJ_VERBOSE_LOGS",
        "KORPUSUJ_VERBOSE",
        "KORPUSUJ_SEARCH_VERBOSE",
        "KORPUSUJ_SEARCH_MIGRATION_DEBUG",
        "KORPUSUJ_DEBUG_LOGS",
    ):
        try:
            if _korpusuj_truthy_145c2b(_os_145c2b.environ.get(env_name, "")):
                return True
        except Exception:
            pass
    for key in (
        "logging_verbose",
        "search_verbose",
        "search_migration_debug",
    ):
        value = _korpusuj_config_value_145c2b(config_obj, key)
        if value is True or _korpusuj_truthy_145c2b(value):
            return True
    return False


def korpusuj_diagnostics_enabled_145c1(config_obj=None):
    return korpusuj_logging_diagnostics_enabled_145c2(config_obj=config_obj)


def korpusuj_verbose_diagnostics_enabled_145c1(config_obj=None):
    return korpusuj_logging_verbose_enabled_145c2(config_obj=config_obj)
# END KORPUSUJ_PATCH_145C2B_STRICT_VERBOSE_GATE

# KORPUSUJ_PATCH_145C3C2_CLEAN_VERBOSE_LOG_HELPER
def korpusuj_verbose_log_145c2(marker, semantic_event, message, *args, **kwargs):
    try:
        if not korpusuj_verbose_diagnostics_enabled_145c1():
            return
        import logging as _logging_145c2
        prefix = f"[{marker}] event={semantic_event!r} " if marker else f"event={semantic_event!r} "
        _logging_145c2.info(prefix + message, *args, **kwargs)
    except Exception:
        return
# END KORPUSUJ_PATCH_145C3C2_CLEAN_VERBOSE_LOG_HELPER

