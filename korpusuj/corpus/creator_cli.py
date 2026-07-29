# -*- coding: utf-8 -*-
"""GUI-free command-line adapter for corpus creation.

Usage:
    python -m korpusuj.corpus.creator_cli --input PATH --output corpus.parquet

The CLI validates paths and optional XLSX metadata before invoking models. It
then delegates all processing, annotation-layer metadata, checkpointing, and
resume compatibility to the shared creator orchestration.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from korpusuj.corpus.creator_core import CreatorRunOptions
from korpusuj.corpus.creator_nlp import CreatorModelState
from korpusuj.corpus.creator_orchestration import run_creator_job

BASIC_METADATA_FIELDS = ("Nazwa pliku", "Tytuł", "Data publikacji", "Autor")
REQUIRED_METADATA_FIELD = "Nazwa pliku"
OPTIONAL_METADATA_FIELDS = ("Tytuł", "Data publikacji", "Autor")
SUPPORTED_INPUT_EXTENSIONS = {".docx", ".pdf", ".xlsx", ".zip"}


class CreatorCliConfigurationError(ValueError):
    """Invalid user input detected before creator/model initialization."""


class StderrProgressReporter:
    """Progress reporter that never contaminates machine-readable stdout."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)

    def _write(self, level: str, message: Any) -> None:
        if self.enabled and str(message or "").strip():
            print(f"[{level}] {message}", file=sys.stderr, flush=True)

    def status(self, message: str) -> None:
        self._write("status", message)

    def current(self, value: float) -> None:
        return None

    def total(self, value: float) -> None:
        return None

    def size_info(self, message: str) -> None:
        self._write("size", message)

    def warning(self, message: str) -> None:
        self._write("warning", message)

    def error(self, message: str, exc: Exception | None = None) -> None:
        suffix = f": {exc}" if exc is not None else ""
        self._write("error", f"{message}{suffix}")

    def tick(self) -> None:
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the public argument parser for the corpus creator CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m korpusuj.corpus.creator_cli",
        description="Create a Korpusuj Parquet corpus through the shared creator runtime.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input", action="append", required=True, metavar="PATH",
        help="Input file or directory; repeatable. Directories are expanded non-recursively.",
    )
    parser.add_argument(
        "--output", required=True, metavar="FILE.parquet",
        help="Final output Parquet file.",
    )
    parser.add_argument(
        "--model", choices=["stanza", "spacy"], default="stanza",
        help="NLP backend; default: stanza.",
    )
    parser.add_argument(
        "--metadata", default=None, metavar="FILE.xlsx",
        help="Optional XLSX metadata file.",
    )
    parser.add_argument(
        "--mapping", default=None, metavar="FILE.json",
        help="Optional UTF-8 JSON mapping: Korpusuj field -> actual XLSX column.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from compatible existing parts/output.")
    parser.add_argument(
        "--no-ner", action="store_false", dest="enable_ner", default=True,
        help="Disable named-entity recognition.",
    )
    parser.add_argument(
        "--no-coreference", action="store_false", dest="enable_coreference", default=True,
        help="Disable coreference annotation.",
    )
    parser.add_argument(
        "--format", choices=["json", "text"], default="json",
        help="Final status format written to stdout; default: json.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON status.")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages on stderr; final stdout status is unchanged.",
    )
    return parser


def _expand_input_paths(values: Sequence[str]) -> list[str]:
    files: dict[str, Path] = {}
    for raw in values:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise CreatorCliConfigurationError(f"Input path does not exist: {path}")
        candidates = [path] if path.is_file() else sorted(
            (item for item in path.iterdir() if item.is_file()),
            key=lambda item: (item.name.casefold(), str(item)),
        )
        for candidate in candidates:
            if candidate.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            files[str(resolved).casefold()] = resolved
    expanded = [str(files[key]) for key in sorted(files)]
    if not expanded:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise CreatorCliConfigurationError(
            f"No supported input files were found. Supported extensions: {allowed}"
        )
    return expanded


def _validate_output_path(value: str) -> Path:
    output = Path(value).expanduser().resolve()
    if output.suffix.lower() != ".parquet":
        raise CreatorCliConfigurationError("--output must end with .parquet")
    if output.exists() and output.is_dir():
        raise CreatorCliConfigurationError(f"Output path is a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _load_mapping_json(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise CreatorCliConfigurationError(f"Mapping file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CreatorCliConfigurationError(
            f"Invalid mapping JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise CreatorCliConfigurationError("Mapping JSON must be an object.")
    unknown = sorted(set(payload) - set(BASIC_METADATA_FIELDS))
    if unknown:
        raise CreatorCliConfigurationError(
            "Unknown Korpusuj metadata fields in mapping: " + ", ".join(unknown)
        )
    mapping: dict[str, str] = {}
    for target, source in payload.items():
        if not isinstance(source, str) or not source.strip():
            raise CreatorCliConfigurationError(
                f"Mapping value for {target!r} must be a non-empty column name."
            )
        mapping[str(target)] = source.strip()
    if REQUIRED_METADATA_FIELD not in mapping:
        raise CreatorCliConfigurationError(
            f"Mapping must define {REQUIRED_METADATA_FIELD!r} for metadata joining."
        )
    return mapping


def _read_metadata_columns(path: Path) -> list[str]:
    try:
        import pandas as pd
        frame = pd.read_excel(path, nrows=0, engine="openpyxl")
    except Exception as exc:
        raise CreatorCliConfigurationError(f"Could not read metadata XLSX headers: {exc}") from exc
    return [str(column) for column in frame.columns]


def _build_metadata_configuration(
    metadata_value: str | None,
    mapping_value: str | None,
) -> tuple[str | None, dict[str, dict[str, str]]]:
    if mapping_value and not metadata_value:
        raise CreatorCliConfigurationError("--mapping requires --metadata.")
    if not metadata_value:
        return None, {}

    metadata = Path(metadata_value).expanduser().resolve()
    if not metadata.is_file():
        raise CreatorCliConfigurationError(f"Metadata file does not exist: {metadata}")
    if metadata.suffix.lower() != ".xlsx":
        raise CreatorCliConfigurationError("--metadata must point to an .xlsx file.")

    columns = _read_metadata_columns(metadata)
    column_set = set(columns)
    if mapping_value:
        mapping = _load_mapping_json(Path(mapping_value).expanduser().resolve())
    else:
        mapping = {field: field for field in BASIC_METADATA_FIELDS if field in column_set}
        if REQUIRED_METADATA_FIELD not in mapping:
            raise CreatorCliConfigurationError(
                f"Metadata XLSX must contain {REQUIRED_METADATA_FIELD!r}, or use --mapping."
            )

    missing_sources = sorted({source for source in mapping.values() if source not in column_set})
    if missing_sources:
        raise CreatorCliConfigurationError(
            "Mapped XLSX columns do not exist: " + ", ".join(missing_sources)
        )
    return str(metadata), {str(metadata): mapping}


def _result_payload(result: Any, options: CreatorRunOptions) -> dict[str, Any]:
    if dataclasses.is_dataclass(result):
        payload = dataclasses.asdict(result)
    else:
        payload = {
            "success": bool(getattr(result, "success", False)),
            "output_file": getattr(result, "output_file", None),
            "error_message": getattr(result, "error_message", None),
            "warnings": list(getattr(result, "warnings", []) or []),
        }
    payload["model"] = options.model_name
    payload["enable_ner"] = options.enable_ner
    payload["enable_coreference"] = options.enable_coreference
    payload["resume"] = options.resume_mode
    payload["input_files"] = len(options.input_files)
    return payload


def _write_status(payload: dict[str, Any], *, output_format: str, pretty: bool) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None))
        return
    state = "SUCCESS" if payload.get("success") else "ERROR"
    print(f"{state}: {payload.get('output_file') or payload.get('error_message') or ''}")
    for warning in payload.get("warnings") or []:
        print(f"WARNING: {warning}")


def _configuration_error_payload(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "output_file": None,
        "error_message": str(message),
        "warnings": [],
        "error_type": "configuration",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the corpus creator CLI and return its process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        input_files = _expand_input_paths(args.input)
        output = _validate_output_path(args.output)
        metadata_path, excel_mappings = _build_metadata_configuration(
            args.metadata, args.mapping
        )
        options = CreatorRunOptions(
            input_files=input_files,
            output_parquet_file=str(output),
            metadata_path=metadata_path,
            model_name=args.model,
            excel_mappings=excel_mappings,
            resume_mode=bool(args.resume),
            enable_ner=bool(args.enable_ner),
            enable_coreference=bool(args.enable_coreference),
        )
    except CreatorCliConfigurationError as exc:
        payload = _configuration_error_payload(str(exc))
        _write_status(payload, output_format=args.format, pretty=bool(args.pretty))
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    reporter = StderrProgressReporter(enabled=not bool(args.quiet))
    try:
        result = run_creator_job(
            options,
            reporter,
            model_state=CreatorModelState(),
            models_dir=Path.cwd() / "models",
        )
    except Exception as exc:
        payload = {
            "success": False,
            "output_file": None,
            "error_message": str(exc) or type(exc).__name__,
            "warnings": [],
            "error_type": type(exc).__name__,
            "model": options.model_name,
            "enable_ner": options.enable_ner,
            "enable_coreference": options.enable_coreference,
            "resume": options.resume_mode,
            "input_files": len(options.input_files),
        }
        _write_status(payload, output_format=args.format, pretty=bool(args.pretty))
        print(f"Creator runtime error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = _result_payload(result, options)
    _write_status(payload, output_format=args.format, pretty=bool(args.pretty))
    return 0 if bool(payload.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
