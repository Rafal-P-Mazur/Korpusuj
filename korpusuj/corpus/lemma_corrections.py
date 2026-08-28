# -*- coding: utf-8 -*-
"""Optional, externally configured lemma corrections for corpus creation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


class LemmaCorrectionsError(ValueError):
    """Invalid lemma-corrections configuration."""


@dataclass
class LemmaCorrectionsConfig:
    path: str | None = None
    name: str = ""
    schema_version: int = 1
    sha256: str | None = None
    rules: dict[tuple[str, str, str], tuple[str, str]] = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)

    @property
    def enabled(self) -> bool:
        return bool(self.path)


def disabled_lemma_corrections() -> LemmaCorrectionsConfig:
    return LemmaCorrectionsConfig()


def _require_text(rule: dict[str, Any], key: str, index: int) -> str:
    value = rule.get(key)
    if not isinstance(value, str) or not value:
        raise LemmaCorrectionsError(f"rules[{index}].{key} must be a non-empty string")
    return value


def load_lemma_corrections(path_value: str | None) -> LemmaCorrectionsConfig:
    if not path_value:
        return disabled_lemma_corrections()
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise LemmaCorrectionsError(f"Lemma corrections file does not exist: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:
        raise LemmaCorrectionsError(f"Invalid lemma corrections JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LemmaCorrectionsError("Lemma corrections root must be an object")
    if payload.get("schema_version") != 1:
        raise LemmaCorrectionsError("lemma corrections schema_version must equal 1")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise LemmaCorrectionsError("lemma corrections name must be a non-empty string")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise LemmaCorrectionsError("lemma corrections rules must be a non-empty array")

    rules: dict[tuple[str, str, str], tuple[str, str]] = {}
    for index, rule in enumerate(raw_rules):
        if not isinstance(rule, dict):
            raise LemmaCorrectionsError(f"rules[{index}] must be an object")
        orth = _require_text(rule, "orth", index)
        lemma = _require_text(rule, "lemma", index)
        upos = _require_text(rule, "upos", index)
        replacement = _require_text(rule, "replacement", index)
        reason = str(rule.get("reason") or "")
        key = (orth, lemma, upos)
        if key in rules:
            raise LemmaCorrectionsError(f"duplicate lemma correction key: {key!r}")
        rules[key] = (replacement, reason)

    return LemmaCorrectionsConfig(
        path=str(path), name=name.strip(), schema_version=1,
        sha256=hashlib.sha256(raw).hexdigest(), rules=rules,
    )


def apply_lemma_corrections(token_details: Any, config: LemmaCorrectionsConfig) -> None:
    if not config.enabled:
        return
    for token in token_details or ():
        if not isinstance(token, dict):
            continue
        key = (str(token.get("token", "")), str(token.get("lemma", "")), str(token.get("upos", "")))
        action = config.rules.get(key)
        if action is None:
            continue
        replacement, _reason = action
        token["lemma"] = replacement
        config.counts["|".join((*key, replacement))] += 1


def lemma_corrections_identity(config: LemmaCorrectionsConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "schema_version": config.schema_version if config.enabled else None,
        "name": config.name if config.enabled else None,
        "config_sha256": config.sha256 if config.enabled else None,
    }


def lemma_corrections_metadata(config: LemmaCorrectionsConfig) -> dict[str, Any]:
    out = lemma_corrections_identity(config)
    out["corrected_tokens"] = int(sum(config.counts.values()))
    out["counts_by_rule"] = dict(sorted(config.counts.items()))
    return out
