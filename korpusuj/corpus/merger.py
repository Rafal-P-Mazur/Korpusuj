"""Guarded merger for compatible canonical Korpusuj Parquet corpora.

Operates only on ready Parquet rows. It never invokes the NLP creator.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

KORPUS_META_KEY = b"korpus_meta"
REQUIRED_COLUMNS = ("Oryginalna_nazwa_pliku", "Treść", "Data publikacji", "tokens", "lemmas",
                    "postags", "full_postags", "deprels", "word_ids", "sentence_ids", "head_ids",
                    "start_ids", "end_ids", "ners", "upostags", "corefs", "coref_mentions")
PARALLEL_COLUMNS = ("lemmas", "postags", "full_postags", "deprels", "word_ids", "sentence_ids",
                    "head_ids", "start_ids", "end_ids", "ners", "upostags", "corefs")

class CorpusMergeError(RuntimeError):
    pass

@dataclass(frozen=True)
class CorpusInputInfo:
    path: str
    rows: int
    row_groups: int
    bytes: int
    annotation_layers: dict[str, bool]

@dataclass
class MergeResult:
    success: bool
    output_path: str
    rows: int
    total_tokens: int
    inputs: list[CorpusInputInfo]
    warnings: list[str] = field(default_factory=list)
    report_path: str | None = None
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _meta(schema: pa.Schema, path: Path) -> dict[str, Any]:
    raw = (schema.metadata or {}).get(KORPUS_META_KEY)
    if raw is None:
        raise CorpusMergeError(f"Brak korpus_meta: {path}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise CorpusMergeError(f"Niepoprawne korpus_meta w {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusMergeError(f"korpus_meta nie jest obiektem: {path}")
    return value

def _layers(meta: Mapping[str, Any], path: Path) -> dict[str, bool] | None:
    value = meta.get("annotation_layers")
    if value is None:
        return None
    if not isinstance(value, Mapping) or not {"ner", "coreference"}.issubset(value):
        raise CorpusMergeError(f"Niepełne lub niepoprawne annotation_layers: {path}")
    return {"ner": bool(value["ner"]), "coreference": bool(value["coreference"])}

def _merge_type(a: pa.DataType, b: pa.DataType, name: str) -> pa.DataType:
    if a == b:
        return a
    if pa.types.is_null(a):
        return b
    if pa.types.is_null(b):
        return a
    if pa.types.is_list(a) and pa.types.is_list(b):
        return pa.list_(_merge_type(a.value_type, b.value_type, name))
    if pa.types.is_large_list(a) and pa.types.is_large_list(b):
        return pa.large_list(_merge_type(a.value_type, b.value_type, name))
    if pa.types.is_struct(a) and pa.types.is_struct(b):
        if [f.name for f in a] != [f.name for f in b]:
            raise CorpusMergeError(f"Niezgodna struktura {name}: {a} != {b}")
        return pa.struct([pa.field(x.name, _merge_type(x.type, y.type, name),
                                  nullable=x.nullable or y.nullable) for x, y in zip(a, b)])
    raise CorpusMergeError(f"Niezgodny typ {name}: {a} != {b}")

def _target_schema(schemas: Sequence[pa.Schema]) -> pa.Schema:
    names = schemas[0].names
    if any(schema.names != names for schema in schemas[1:]):
        raise CorpusMergeError("Niezgodne nazwy lub kolejność kolumn.")
    fields = []
    for i, name in enumerate(names):
        original = schemas[0].field(i)
        typ, nullable = original.type, original.nullable
        for schema in schemas[1:]:
            other = schema.field(i)
            typ = _merge_type(typ, other.type, name)
            nullable = nullable or other.nullable
        fields.append(pa.field(name, typ, nullable=nullable, metadata=original.metadata))
    return pa.schema(fields)

def _cast(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        column = table[field.name]
        if column.type != field.type:
            try:
                column = column.cast(field.type, safe=True)
            except Exception as exc:
                raise CorpusMergeError(f"Nie można ujednolicić {field.name}: {column.type} -> {field.type}: {exc}") from exc
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)

def _as_list(value: Any, name: str, path: Path, row: int) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CorpusMergeError(f"{name} nie jest listą: {path}, wiersz {row}")
    return value

def _month(value: Any) -> tuple[str, str] | None:
    text = str(value or "").strip()
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if m and 1 <= int(m.group(2)) <= 12:
        return m.group(1), str(int(m.group(2)))
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text)
    if m and 1 <= int(m.group(2)) <= 12:
        return m.group(3), str(int(m.group(2)))
    return None

def _validate_mentions(value: Any, count: int, path: Path, row: int) -> None:
    for mention in _as_list(value, "coref_mentions", path, row):
        if not isinstance(mention, Mapping):
            raise CorpusMergeError(f"Niepoprawny coref_mentions: {path}, wiersz {row}")
        missing = {"cluster_id", "mention_id", "start", "end", "head"} - set(mention)
        if missing:
            raise CorpusMergeError(f"coref_mentions bez pól {sorted(missing)}: {path}, wiersz {row}")
        start, end, head = int(mention["start"]), int(mention["end"]), int(mention["head"])
        if not (0 <= start < end <= count and start <= head < end):
            raise CorpusMergeError(f"Niepoprawne współrzędne coref_mentions: {path}, wiersz {row}")

def inspect_merge_inputs(values: Sequence[str | Path], *, allow_undeclared_annotation_layers: bool = False):
    if len(values) < 2:
        raise CorpusMergeError("Wymagane są co najmniej dwa wejścia.")
    paths = [Path(v).expanduser().resolve() for v in values]
    if len(set(paths)) != len(paths):
        raise CorpusMergeError("Ten sam input podano więcej niż raz.")
    if any(not p.is_file() for p in paths):
        raise CorpusMergeError("Brak wejścia: " + ", ".join(str(p) for p in paths if not p.is_file()))
    pfs = [pq.ParquetFile(p) for p in paths]
    schemas = [pf.schema_arrow for pf in pfs]
    for path, schema in zip(paths, schemas):
        missing = [name for name in REQUIRED_COLUMNS if name not in schema.names]
        if missing:
            raise CorpusMergeError(f"Brak wymaganych kolumn w {path}: {missing}")
    metas = [_meta(schema, path) for schema, path in zip(schemas, paths)]
    layers = [_layers(meta, path) for meta, path in zip(metas, paths)]
    declared = [item for item in layers if item is not None]
    if declared and len(declared) != len(layers):
        raise CorpusMergeError("Nie można mieszać wejść z zadeklarowanym i niezadeklarowanym annotation_layers.")
    if not declared:
        if not allow_undeclared_annotation_layers:
            raise CorpusMergeError("Wszystkie wejścia nie deklarują annotation_layers. Użyj --allow-undeclared-annotation-layers tylko dla świadomie zweryfikowanych korpusów historycznych.")
        effective_layers = {"ner": False, "coreference": False}
    else:
        if any(item != declared[0] for item in declared[1:]):
            raise CorpusMergeError(f"Niezgodne annotation_layers: {layers}")
        effective_layers = declared[0]
    schema = _target_schema(schemas)
    infos = [CorpusInputInfo(str(path), pf.metadata.num_rows, pf.metadata.num_row_groups,
                             path.stat().st_size, effective_layers)
             for path, pf in zip(paths, pfs)]
    return paths, pfs, schema, infos

def _report(result: MergeResult, path: Path) -> None:
    lines = ["# Korpusuj corpus merge report", "", "**Status:** `success`  ",
             f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
             f"**Output:** `{result.output_path}`", "", "## Inputs", ""]
    lines += [f"- `{item.path}`: {item.rows} documents, {item.row_groups} row groups; annotation_layers={item.annotation_layers}"
              for item in result.inputs]
    lines += ["", "## Result", "", f"- Documents: **{result.rows}**",
              f"- Tokens: **{result.total_tokens}**", "- NLP rerun: **No**",
              "- Inputs modified: **No**", "- Existing sidecars merged: **No**",
              "- New `.search` and `.dep_cache` required: **Yes**"]
    if result.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in result.warnings]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def merge_corpora(input_paths: Sequence[str | Path], output_path: str | Path, *,
                  report_path: str | Path | None = None, replace: bool = False,
                  batch_size: int = 128, check_duplicates: bool = True,
                  allow_undeclared_annotation_layers: bool = False,
                  progress_callback: Callable[[int, int], None] | None = None) -> MergeResult:
    paths, pfs, schema, infos = inspect_merge_inputs(
        input_paths, allow_undeclared_annotation_layers=allow_undeclared_annotation_layers
    )
    output = Path(output_path).expanduser().resolve()
    if output in paths:
        raise CorpusMergeError("Output nie może być inputem.")
    if output.exists() and not replace:
        raise CorpusMergeError(f"Output już istnieje: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = Path(report_path).expanduser().resolve() if report_path else output.with_suffix(".merge_report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    stage1 = output.with_name(output.name + ".merge_stage")
    stage2 = output.with_name(output.name + ".final_stage")
    for stage in (stage1, stage2):
        if stage.exists(): stage.unlink()
    expected = sum(i.rows for i in infos)
    total_tokens = 0
    base_tf, orth_tf = Counter(), Counter()
    monthly: defaultdict[str, Counter[str]] = defaultdict(Counter)
    names, hashes = {}, {}
    warnings, done = [], 0
    first_md = dict(pfs[0].schema_arrow.metadata or {})
    physical_schema = schema.with_metadata({k: v for k, v in first_md.items() if k != KORPUS_META_KEY})
    try:
        writer = pq.ParquetWriter(stage1, physical_schema, compression="snappy")
        try:
            for path, pf in zip(paths, pfs):
                row_base = 0
                for batch in pf.iter_batches(batch_size=batch_size):
                    table = _cast(pa.Table.from_batches([batch]), schema)
                    data = table.to_pydict()
                    for offset in range(table.num_rows):
                        row = row_base + offset
                        tokens = _as_list(data["tokens"][offset], "tokens", path, row)
                        for column in PARALLEL_COLUMNS:
                            if len(_as_list(data[column][offset], column, path, row)) != len(tokens):
                                raise CorpusMergeError(f"Niezgodna długość {column}/tokens: {path}, wiersz {row}")
                        _validate_mentions(data["coref_mentions"][offset], len(tokens), path, row)
                        total_tokens += len(tokens)
                        orth_tf.update(str(x) for x in tokens if x is not None)
                        base_tf.update(str(x) for x in data["lemmas"][offset] if x is not None)
                        key = _month(data["Data publikacji"][offset])
                        if key: monthly[key[0]][key[1]] += len(tokens)
                        elif data["Data publikacji"][offset] not in (None, ""):
                            warnings.append(f"Nie rozpoznano daty: {path}, wiersz {row}")
                        if check_duplicates:
                            name = str(data["Oryginalna_nazwa_pliku"][offset] or "").strip()
                            text = str(data["Treść"][offset] or "")
                            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                            if name and name in names:
                                raise CorpusMergeError(f"Kolizja Oryginalna_nazwa_pliku {name!r}: {names[name]} oraz {path}:{row}")
                            if text and digest in hashes:
                                raise CorpusMergeError(f"Duplikat treści SHA-256={digest}: {hashes[digest]} oraz {path}:{row}")
                            if name: names[name] = f"{path}:{row}"
                            if text: hashes[digest] = f"{path}:{row}"
                    writer.write_table(table.replace_schema_metadata(physical_schema.metadata))
                    row_base += table.num_rows
                    done += table.num_rows
                    if progress_callback: progress_callback(done, expected)
        finally:
            writer.close()
        merged_meta = {"base_tf": dict(sorted(base_tf.items())), "orth_tf": dict(sorted(orth_tf.items())),
                       "total_tokens": total_tokens,
                       "monthly_token_counts": {year: dict(sorted(c.items(), key=lambda x: int(x[0])))
                                                for year, c in sorted(monthly.items())},
                       "annotation_layers": infos[0].annotation_layers}
        md = dict(physical_schema.metadata or {})
        md[KORPUS_META_KEY] = json.dumps(merged_meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        final_schema = schema.with_metadata(md)
        source = pq.ParquetFile(stage1)
        writer2 = pq.ParquetWriter(stage2, final_schema, compression="snappy")
        try:
            for batch in source.iter_batches(batch_size=batch_size):
                writer2.write_table(pa.Table.from_batches([batch], schema=final_schema))
        finally:
            writer2.close()
            source.close()

        check = pq.ParquetFile(stage2)
        try:
            if check.metadata.num_rows != expected:
                raise CorpusMergeError(f"Błędna liczba wierszy: {check.metadata.num_rows} != {expected}")
            check_meta = _meta(check.schema_arrow, stage2)
            if check_meta.get("total_tokens") != total_tokens or check_meta.get("annotation_layers") != infos[0].annotation_layers:
                raise CorpusMergeError("Walidacja metadanych wyniku nie powiodła się.")
        finally:
            # PyArrow keeps this file open on Windows until explicitly closed.
            check.close()

        if replace and output.exists():
            output.unlink()
        os.replace(stage2, output)
        result = MergeResult(True, str(output), expected, total_tokens, infos, warnings, str(report))
        _report(result, report)
        return result
    finally:
        for stage in (stage1, stage2):
            try:
                if stage.exists(): stage.unlink()
            except OSError:
                pass
