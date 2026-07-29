# -*- coding: utf-8 -*-
"""Store the configurable runtime dependencies used by dependency-cache operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DependencyRuntimeState:
    dependency_maps_cache: Dict[Any, Any] = field(default_factory=dict)
    dependency_disk_caches: Dict[str, Any] = field(default_factory=dict)
    dependency_warmup_threads: Dict[str, Any] = field(default_factory=dict)
    dependency_warmup_stop_flags: Dict[str, Any] = field(default_factory=dict)
    dependency_warmup_lock: Any = None

    maps_cache_maxsize: int = 50000
    candidate_max_docs: int = 3000
    candidate_stream_batch_docs: int = 256
    candidate_ram_budget_mb: int = 512
    cache_preload_batch_size: int = 500

    default_ram_mode: str = "none"
    default_ram_usage_label: str = "Oszczędny"
    ram_usage_labels: Dict[str, str] = field(default_factory=dict)
    ram_mode_labels: Dict[str, str] = field(default_factory=dict)

    disk_cache_version: str = ""
    legacy_disk_cache_version: str = ""
    parent_magic: Any = b""


_runtime_state: Optional[DependencyRuntimeState] = None


def configure_dependency_runtime_state(
    *,
    dependency_maps_cache: Dict[Any, Any],
    dependency_disk_caches: Dict[str, Any],
    dependency_warmup_threads: Dict[str, Any],
    dependency_warmup_stop_flags: Dict[str, Any],
    dependency_warmup_lock: Any,
    maps_cache_maxsize: int,
    candidate_max_docs: int,
    candidate_stream_batch_docs: int,
    candidate_ram_budget_mb: int,
    cache_preload_batch_size: int,
    default_ram_mode: str,
    default_ram_usage_label: str,
    ram_usage_labels: Dict[str, str],
    ram_mode_labels: Dict[str, str],
    disk_cache_version: str = "",
    legacy_disk_cache_version: str = "",
    parent_magic: Any = b"",
) -> None:
    """Configure dependency runtime state from engine.py.

    In 4e.1 this mirrors existing globals and does not change runtime behavior.
    """
    global _runtime_state
    _runtime_state = DependencyRuntimeState(
        dependency_maps_cache=dependency_maps_cache,
        dependency_disk_caches=dependency_disk_caches,
        dependency_warmup_threads=dependency_warmup_threads,
        dependency_warmup_stop_flags=dependency_warmup_stop_flags,
        dependency_warmup_lock=dependency_warmup_lock,
        maps_cache_maxsize=int(maps_cache_maxsize),
        candidate_max_docs=int(candidate_max_docs),
        candidate_stream_batch_docs=int(candidate_stream_batch_docs),
        candidate_ram_budget_mb=int(candidate_ram_budget_mb),
        cache_preload_batch_size=int(cache_preload_batch_size),
        default_ram_mode=str(default_ram_mode),
        default_ram_usage_label=str(default_ram_usage_label),
        ram_usage_labels=dict(ram_usage_labels or {}),
        ram_mode_labels=dict(ram_mode_labels or {}),
        disk_cache_version=str(disk_cache_version or ""),
        legacy_disk_cache_version=str(legacy_disk_cache_version or ""),
        parent_magic=parent_magic,
    )


def get_dependency_runtime_state() -> DependencyRuntimeState:
    if _runtime_state is None:
        raise RuntimeError("Dependency runtime state not configured. Call configure_dependency_runtime_state(...) first.")
    return _runtime_state


def dependency_runtime_state_configured() -> bool:
    return _runtime_state is not None
