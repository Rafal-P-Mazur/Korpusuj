# -*- coding: utf-8 -*-
"""Coordinate dependency-cache access, preloading, warm-up and runtime state for corpus operations."""
from __future__ import annotations

import logging
import time
import threading

from korpusuj.dependency.policy import DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE


# KORPUSUJ_PATCH_145C3A_LOGGING_GATES_IMPORT
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
# END KORPUSUJ_PATCH_145C3A_LOGGING_GATES_IMPORT

def _get_dependency_disk_cache_for_corpus_impl(corpus_name):
    try:
        corpus_path = files.get(corpus_name)
        if not corpus_path:
            df_obj = dataframes.get(corpus_name)
            corpus_path = getattr(df_obj, "parquet_path", None)
        if not corpus_path:
            return None

        expected_path = _dependency_cache_path_for_corpus_path(corpus_path)
        cache = dependency_disk_caches.get(corpus_name)
        if cache is not None and getattr(cache, "cache_path", None) == expected_path:
            return cache
        if cache is not None:
            try:
                cache.close()
            except Exception:
                pass
        cache = DependencyMapDiskCache(corpus_path)
        dependency_disk_caches[corpus_name] = cache
        return cache
    except Exception as e:
        if korpusuj_diagnostics_enabled_145c1():
            logging.info("[DIAG dependency.cache] corpus=%s reason=%r", corpus_name, e)
        return None

def __clear_dependency_ram_cache_for_corpus_impl(corpus_name=None):
    """3e: zwalnia RAM zajęty przez dependency_maps_cache.

    corpus_name=None czyści cały cache; inaczej usuwa tylko wpisy danego korpusu.
    Używane przede wszystkim po przełączeniu na tryb Oszczędny.
    """
    try:
        if corpus_name is None:
            removed = len(dependency_maps_cache)
            dependency_maps_cache.clear()
        else:
            keys_to_remove = [k for k in dependency_maps_cache.keys() if isinstance(k, tuple) and k and k[0] == corpus_name]
            removed = len(keys_to_remove)
            for k in keys_to_remove:
                dependency_maps_cache.pop(k, None)
        if removed:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] corpus=%s removed=%s", corpus_name or "*", removed)
        return removed
    except Exception:
        return 0

def __put_dependency_ram_cache_impl(cache_key, dep_maps):
    if _get_dependency_cache_ram_mode() == "none":
        # 3e: twarda gwarancja trybu Oszczędny — żadnych map zależności w RAM.
        try:
            corpus_name = cache_key[0] if isinstance(cache_key, tuple) and cache_key else None
            _clear_dependency_ram_cache_for_corpus(corpus_name)
        except Exception:
            pass
        return False
    dependency_maps_cache[cache_key] = dep_maps
    if len(dependency_maps_cache) > DEPENDENCY_MAPS_CACHE_MAXSIZE:
        try:
            dependency_maps_cache.pop(next(iter(dependency_maps_cache)))
        except Exception:
            pass
    return True

def _preload_dependency_maps_for_candidates_impl(corpus_name, doc_ids, diag=None, batch_size=DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE):
    """3m1: candidate preload ograniczony globalnym limitem i wołany małymi batchami."""
    if _get_dependency_cache_ram_mode() != "candidate":
        return 0
    disk_cache = get_dependency_disk_cache_for_corpus(corpus_name)
    if disk_cache is None:
        return 0
    try:
        unique_ids = sorted({int(x) for x in doc_ids})
    except Exception:
        unique_ids = []
    requested = len(unique_ids)
    try:
        cfg = globals().get("config", {}) or {}
        max_docs = max(1, int(cfg.get("dependency_candidate_max_docs", DEPENDENCY_CANDIDATE_MAX_DOCS) or DEPENDENCY_CANDIDATE_MAX_DOCS))
        budget_mb = int(cfg.get("dependency_candidate_ram_budget_mb", DEPENDENCY_CANDIDATE_RAM_BUDGET_MB) or DEPENDENCY_CANDIDATE_RAM_BUDGET_MB)
    except Exception:
        max_docs = DEPENDENCY_CANDIDATE_MAX_DOCS
        budget_mb = DEPENDENCY_CANDIDATE_RAM_BUDGET_MB
    current = _dependency_ram_cache_size_for_corpus(corpus_name)
    remaining_slots = max(0, max_docs - current)
    if remaining_slots <= 0:
        if diag is not None: diag["dep_maps_candidate_preload_skipped_full"] = requested
        return 0
    missing = [doc_id for doc_id in unique_ids if (corpus_name, doc_id) not in dependency_maps_cache]
    if not missing:
        if diag is not None:
            diag["dep_maps_candidate_preload_requested"] = requested
            diag["dep_maps_candidate_preload_loaded"] = 0
            diag["dep_maps_candidate_preload_already_cached"] = requested
        return 0
    selected = list(missing[:remaining_slots])
    budget_bytes = max(1, budget_mb) * 1024 * 1024
    try: payload_bytes = int(disk_cache.payload_bytes_for_doc_ids(selected) or 0)
    except Exception: payload_bytes = 0
    while selected and payload_bytes > budget_bytes:
        selected = selected[:max(1, int(len(selected) * 0.75))]
        try: payload_bytes = int(disk_cache.payload_bytes_for_doc_ids(selected) or 0)
        except Exception: payload_bytes = 0
        if len(selected) <= 1 and payload_bytes > budget_bytes:
            break
    if not selected:
        return 0
    t0 = time.perf_counter()
    maps = disk_cache.get_many(selected, batch_size=batch_size)
    loaded = 0
    for doc_id, dep_maps in maps.items():
        if _put_dependency_ram_cache((corpus_name, int(doc_id)), dep_maps): loaded += 1
    elapsed = time.perf_counter() - t0
    if diag is not None:
        diag["dep_maps_candidate_preload_requested"] = requested
        diag["dep_maps_candidate_preload_missing"] = len(missing)
        diag["dep_maps_candidate_preload_selected"] = len(selected)
        diag["dep_maps_candidate_preload_loaded"] = loaded
        diag["dep_maps_candidate_preload_payload_bytes"] = int(payload_bytes or 0)
        diag["time_dep_maps_candidate_preload"] = diag.get("time_dep_maps_candidate_preload", 0.0) + elapsed
    if korpusuj_diagnostics_enabled_145c1():
        logging.info("[DIAG dependency.cache] corpus=%s requested=%s selected=%s loaded=%s current_before=%s max_docs=%s ram_cache_size=%s time=%.6fs",
                     corpus_name, requested, len(selected), loaded, current, max_docs, _dependency_ram_cache_size_for_corpus(corpus_name), elapsed)
    return loaded

def _preload_all_dependency_maps_for_corpus_impl(corpus_name, disk_cache=None, diag=None):
    """Tryb Duże: wczytuje cały .dep_cache do RAM po ładowaniu korpusu."""
    if _get_dependency_cache_ram_mode() != "all":
        return 0
    if disk_cache is None:
        disk_cache = get_dependency_disk_cache_for_corpus(corpus_name)
    if disk_cache is None:
        return 0
    t0 = time.perf_counter()
    maps = disk_cache.get_all()
    loaded = 0
    for doc_id, dep_maps in maps.items():
        if _put_dependency_ram_cache((corpus_name, int(doc_id)), dep_maps):
            loaded += 1
    elapsed = time.perf_counter() - t0
    if diag is not None:
        diag["dep_maps_all_preload_loaded"] = loaded
        diag["time_dep_maps_all_preload"] = diag.get("time_dep_maps_all_preload", 0.0) + elapsed
    if korpusuj_diagnostics_enabled_145c1():
        logging.info(
            "[DIAG dependency.cache] corpus=%s loaded=%s disk_rows=%s ram_cache_size=%s time=%.6fs",
            corpus_name, loaded, disk_cache.row_count(), len(dependency_maps_cache), elapsed
        )
    return loaded

def __cache_dependency_maps_for_row_impl(corpus_name, row, diag=None, disk_cache=None, store_ram=False, commit=True):
    try:
        row_id = int(row.Index)
    except Exception:
        return False

    cache_key = (corpus_name, row_id)
    if _get_dependency_cache_ram_mode() != "none" and cache_key in dependency_maps_cache:
        if diag is not None:
            diag["ram_hits"] = diag.get("ram_hits", 0) + 1
        return False

    if disk_cache is None:
        disk_cache = get_dependency_disk_cache_for_corpus(corpus_name)

    if disk_cache is not None:
        cached_disk = disk_cache.get(row_id)
        if cached_disk is not None:
            if store_ram:
                _put_dependency_ram_cache(cache_key, cached_disk)
            if diag is not None:
                diag["disk_hits"] = diag.get("disk_hits", 0) + 1
            return False

    try:
        tokens = _as_list_for_warmup(getattr(row, "tokens", None))
        sentence_ids = _as_list_for_warmup(getattr(row, "sentence_ids", None))
        word_ids = _as_list_for_warmup(getattr(row, "word_ids", None))
        head_ids = _as_list_for_warmup(getattr(row, "head_ids", None))

        if not tokens or len(sentence_ids) != len(tokens) or len(word_ids) != len(tokens) or len(head_ids) != len(tokens):
            if diag is not None:
                diag["skipped_bad_lengths"] = diag.get("skipped_bad_lengths", 0) + 1
            return False

        dep_maps = build_dependency_maps(sentence_ids, word_ids, head_ids)

        if disk_cache is not None:
            disk_cache.put(row_id, dep_maps, commit=commit)
            if diag is not None:
                diag["disk_written"] = diag.get("disk_written", 0) + 1

        if store_ram:
            _put_dependency_ram_cache(cache_key, dep_maps)

        if diag is not None:
            diag["built"] = diag.get("built", 0) + 1
        return True
    except Exception:
        if diag is not None:
            diag["errors"] = diag.get("errors", 0) + 1
        return False

def __select_dependency_parquet_columns_impl(parquet_path):
    """Wybiera minimalne kolumny potrzebne do .dep_cache bez morfologii."""
    required_candidates = ("tokens", "sentence_ids", "sentence_id", "word_ids", "head_ids")
    try:
        pf = pq.ParquetFile(parquet_path)
        available = list(getattr(pf, "schema_arrow", pf.schema).names)
    except Exception:
        return None
    selected = []
    for col in required_candidates:
        if col in available and col not in selected:
            selected.append(col)
    if "tokens" not in selected or "word_ids" not in selected or "head_ids" not in selected:
        return None
    if "sentence_ids" not in selected and "sentence_id" not in selected:
        return None
    return selected

def _build_dependency_cache_from_parquet_batches_impl(corpus_name, parquet_path, disk_cache=None, batch_docs=5000,
                                                progress_callback=None, diag=None, stop_flag_getter=None):
    """Buduje .dep_cache partiami z Parquetu, bez LazyCorpus.materialize()."""
    parquet_path = str(parquet_path)
    batch_docs = max(1, int(batch_docs or 5000))
    if disk_cache is None:
        disk_cache = get_dependency_disk_cache_for_corpus(corpus_name)
    if disk_cache is None:
        return 0
    pf = pq.ParquetFile(parquet_path)
    read_columns = _select_dependency_parquet_columns(parquet_path)
    if read_columns is None:
        if korpusuj_diagnostics_enabled_145c1():
            logging.info("[DIAG dependency.cache] event='skip_batch_build' corpus=%s reason=missing_dependency_columns", corpus_name)
        _safe_dependency_progress(f"Pominięto budowę cache zależności dla {corpus_name}: brak wymaganych kolumn dependency.", progress_callback)
        return 0
    total_docs = int(getattr(pf.metadata, "num_rows", 0) or 0)
    disk_cache.mark_rebuild_started(total_docs=total_docs)
    built = 0
    seen = 0
    import types
    for record_batch in pf.iter_batches(batch_size=batch_docs, columns=read_columns):
        if stop_flag_getter is not None and stop_flag_getter():
            disk_cache.commit()
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] event='stopped' corpus=%s seen=%s built=%s", corpus_name, seen, built)
            return built
        df_part = record_batch.to_pandas()
        df_part.index = range(seen, seen + len(df_part))
        for row in df_part.itertuples(index=True):
            if "sentence_ids" not in read_columns and "sentence_id" in read_columns and not hasattr(row, "sentence_ids"):
                row = types.SimpleNamespace(
                    Index=int(row.Index), tokens=getattr(row, "tokens", None),
                    sentence_ids=getattr(row, "sentence_id", None), word_ids=getattr(row, "word_ids", None),
                    head_ids=getattr(row, "head_ids", None),
                )
            if _cache_dependency_maps_for_row(corpus_name, row, diag=diag, disk_cache=disk_cache, store_ram=False, commit=False):
                built += 1
            seen += 1
            if seen % 1000 == 0:
                disk_cache.commit()
                if total_docs:
                    pct = 100.0 * seen / max(1, total_docs)
                    _safe_dependency_progress(f"Budowanie cache map zależności: {seen:,} / {total_docs:,} ({pct:.1f}%)".replace(",", " "), progress_callback)
                else:
                    _safe_dependency_progress(f"Budowanie cache map zależności: {seen:,} dokumentów".replace(",", " "), progress_callback)
        del df_part
    disk_cache.commit()
    disk_cache.mark_complete(total_docs=total_docs or seen)
    _safe_dependency_progress(f"Cache map zależności gotowy: {corpus_name}", progress_callback)
    if korpusuj_diagnostics_enabled_145c1():
        logging.info("[DIAG dependency.cache] event='batch_done' corpus=%s seen=%s built=%s disk_rows=%s diag=%s", corpus_name, seen, built, disk_cache.row_count(), diag)
    return built

def _warm_dependency_cache_for_corpus_impl(corpus_name, build_maps=True, materialize=False, progress_callback=None):
    """Przygotowuje dependency cache 3k bez blokowania GUI i bez domyślnej materializacji."""
    t0 = time.perf_counter()
    diag = {"built": 0, "disk_written": 0, "disk_hits": 0, "ram_hits": 0, "skipped_bad_lengths": 0, "errors": 0}

    try:
        with dependency_warmup_lock:
            dependency_warmup_stop_flags[corpus_name] = False

        ram_mode = _get_dependency_cache_ram_mode()
        if ram_mode == "none":
            # 3e: Oszczędny nie materializuje LazyCorpus i nie trzyma dependency maps w RAM.
            materialize = False
            _clear_dependency_ram_cache_for_corpus(corpus_name)

        if korpusuj_diagnostics_enabled_145c1():
            logging.info("[DIAG dependency.cache] event='start' corpus=%s mode=%s build_maps=%s materialize=%s", corpus_name, ram_mode, build_maps, materialize)

        df_obj = dataframes.get(corpus_name)
        if df_obj is None:
            logging.info("[DIAG dependency.cache] corpus=%s reason=no_dataframe", corpus_name)
            return

        if isinstance(df_obj, LazyCorpus):
            total_docs_hint = int(getattr(df_obj, "total_docs", 0) or 0)
        else:
            try:
                total_docs_hint = len(df_obj)
            except Exception:
                total_docs_hint = 0

        disk_cache = get_dependency_disk_cache_for_corpus(corpus_name)
        disk_cache_ready = bool(
            disk_cache is not None and disk_cache.is_fresh_and_complete(total_docs_hint if total_docs_hint else None)
        )

        if disk_cache_ready:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG dependency.cache] event='disk_ready' corpus=%s path=%s rows=%s",
                    corpus_name, disk_cache.cache_path, disk_cache.row_count()
                )
            if ram_mode == "all":
                _safe_dependency_progress(f"Wczytywanie cache map zależności do pamięci RAM: {corpus_name}...", progress_callback)
                preload_all_dependency_maps_for_corpus(corpus_name, disk_cache=disk_cache, diag=diag)
            if not materialize:
                _safe_dependency_progress(f"Cache map zależności gotowy na dysku: {corpus_name} (tryb {ram_mode}).", progress_callback)
                if korpusuj_diagnostics_enabled_145c1():
                    logging.info(
                        "[DIAG dependency.cache] event='ready_no_materialize' corpus=%s mode=%s disk_rows=%s ram_cache_size=%s time=%.6fs",
                        corpus_name, ram_mode, disk_cache.row_count() if disk_cache is not None else None,
                        len(dependency_maps_cache), time.perf_counter() - t0
                    )
                return
            _safe_dependency_progress(f"Cache map zależności gotowy na dysku: {corpus_name}. Przygotowuję indeks pomocniczy...", progress_callback)

        # 3k: poza explicit materialize nie materializujemy LazyCorpus.
        if isinstance(df_obj, LazyCorpus):
            if materialize:
                t_mat0 = time.perf_counter()
                real_df = df_obj.materialize()
                t_mat1 = time.perf_counter()
                if korpusuj_diagnostics_enabled_145c1():
                    logging.info("[DIAG dependency.cache] event='materialized_explicit' corpus=%s rows=%s time=%.6fs", corpus_name, len(real_df), t_mat1 - t_mat0)
            else:
                if not build_maps:
                    if korpusuj_diagnostics_enabled_145c1():
                        logging.info("[DIAG dependency.cache] event='skip_materialize' corpus=%s reason=build_maps_false", corpus_name)
                    _safe_dependency_progress(f"Cache dependency będzie czytany z dysku na żądanie: {corpus_name}", progress_callback)
                    return
                batch_docs = int((globals().get("config", {}) or {}).get("index_batch_docs", 5000) or 5000)
                build_dependency_cache_from_parquet_batches(corpus_name, df_obj.parquet_path, disk_cache=disk_cache, batch_docs=batch_docs, progress_callback=progress_callback, diag=diag, stop_flag_getter=lambda: dependency_warmup_stop_flags.get(corpus_name))
                if _get_dependency_cache_ram_mode() == "all":
                    _safe_dependency_progress(f"Wczytywanie cache map zależności do pamięci RAM: {corpus_name}...", progress_callback)
                    preload_all_dependency_maps_for_corpus(corpus_name, disk_cache=disk_cache, diag=diag)
                if korpusuj_diagnostics_enabled_145c1():
                    logging.info("[DIAG dependency.cache] event='done_no_materialize' corpus=%s mode=%s diag=%s disk_rows=%s ram_cache_size=%s time=%.6fs", corpus_name, _get_dependency_cache_ram_mode(), diag, disk_cache.row_count() if disk_cache is not None else None, len(dependency_maps_cache), time.perf_counter() - t0)
                return
        else:
            real_df = df_obj

        total_docs = len(real_df)

        t_idx0 = time.perf_counter()
        try:
            ensure_legacy_inverted_index_for_corpus(corpus_name, real_df)
        except Exception as e:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] corpus=%s reason=%r", corpus_name, e)
        t_idx1 = time.perf_counter()
        if korpusuj_diagnostics_enabled_145c1():
            logging.info("[DIAG dependency.cache] corpus=%s time=%.6fs", corpus_name, t_idx1 - t_idx0)

        if disk_cache_ready:
            if _get_dependency_cache_ram_mode() == "all":
                _safe_dependency_progress(f"Wczytywanie cache map zależności do pamięci RAM: {corpus_name}...", progress_callback)
                preload_all_dependency_maps_for_corpus(corpus_name, disk_cache=disk_cache, diag=diag)
            _safe_dependency_progress(f"Cache map zależności gotowy: {corpus_name}", progress_callback)
            if korpusuj_diagnostics_enabled_145c1():
                logging.info(
                    "[DIAG dependency.cache] corpus=%s mode=%s disk_rows=%s ram_cache_size=%s time=%.6fs",
                    corpus_name, _get_dependency_cache_ram_mode(), disk_cache.row_count() if disk_cache is not None else None,
                    len(dependency_maps_cache), time.perf_counter() - t0
                )
            return

        if not build_maps:
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] corpus=%s", corpus_name)
            return

        if disk_cache is not None:
            disk_cache.mark_rebuild_started(total_docs=total_docs)

        _safe_dependency_progress(f"Budowanie cache map zależności: {corpus_name}...", progress_callback)

        for n, row in enumerate(real_df.itertuples(index=True), start=1):
            if dependency_warmup_stop_flags.get(corpus_name):
                if korpusuj_diagnostics_enabled_145c1():
                    logging.info("[DIAG dependency.cache] corpus=%s n=%s diag=%s", corpus_name, n, diag)
                if disk_cache is not None:
                    disk_cache.commit()
                return

            _cache_dependency_maps_for_row(
                corpus_name,
                row,
                diag=diag,
                disk_cache=disk_cache,
                store_ram=False,
                commit=False,
            )

            if n % 1000 == 0:
                if disk_cache is not None:
                    disk_cache.commit()
                if korpusuj_verbose_diagnostics_enabled_145c1():
                    logging.info(
                        "[DIAG perf.dependency.cache] event='progress' corpus=%s rows=%s/%s built=%s disk_hits=%s written=%s errors=%s elapsed=%.2fs",
                        corpus_name, n, total_docs, diag.get("built", 0), diag.get("disk_hits", 0),
                        diag.get("disk_written", 0), diag.get("errors", 0), time.perf_counter() - t0
                    )
                _safe_dependency_progress(f"Budowanie cache map zależności: {n:,} / {total_docs:,}".replace(",", " "), progress_callback)

        if disk_cache is not None:
            disk_cache.commit()
            disk_cache.mark_complete(total_docs=total_docs)

        if _get_dependency_cache_ram_mode() == "all":
            _safe_dependency_progress(f"Wczytywanie cache map zależności do pamięci RAM: {corpus_name}...", progress_callback)
            preload_all_dependency_maps_for_corpus(corpus_name, disk_cache=disk_cache, diag=diag)

        _safe_dependency_progress(f"Cache map zależności gotowy: {corpus_name}", progress_callback)
        if korpusuj_diagnostics_enabled_145c1():
            logging.info(
                "[DIAG dependency.cache] corpus=%s mode=%s diag=%s disk_rows=%s ram_cache_size=%s time=%.6fs",
                corpus_name, _get_dependency_cache_ram_mode(), diag, disk_cache.row_count() if disk_cache is not None else None,
                len(dependency_maps_cache), time.perf_counter() - t0
            )
    except Exception as e:
        logging.error("[APP dependency.cache.error] corpus=%s reason=%s", corpus_name, e, exc_info=True)

def _start_dependency_cache_warmup_impl(corpus_name, build_maps=None, materialize=None):
    if not _cfg_bool("dependency_cache_warmup", True):
        return
    if build_maps is None:
        build_maps = _cfg_bool("dependency_cache_warmup_build_maps", True)
    if materialize is None:
        materialize = _cfg_bool("dependency_cache_warmup_materialize", False)

    if _get_dependency_cache_ram_mode() == "none":
        materialize = False
        _clear_dependency_ram_cache_for_corpus(corpus_name)

    try:
        old_thread = dependency_warmup_threads.get(corpus_name)
        if old_thread is not None and old_thread.is_alive():
            if korpusuj_diagnostics_enabled_145c1():
                logging.info("[DIAG dependency.cache] corpus=%s reason=already_running", corpus_name)
            return
    except Exception:
        pass

    t = threading.Thread(
        target=warm_dependency_cache_for_corpus,
        kwargs={"corpus_name": corpus_name, "build_maps": bool(build_maps), "materialize": bool(materialize)},
        daemon=True,
        name=f"dep-cache-3k-{corpus_name}"
    )
    dependency_warmup_threads[corpus_name] = t
    t.start()

def get_dependency_disk_cache_for_corpus(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return _get_dependency_disk_cache_for_corpus_impl(*args, **kwargs)


def _clear_dependency_ram_cache_for_corpus(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return __clear_dependency_ram_cache_for_corpus_impl(*args, **kwargs)


def _put_dependency_ram_cache(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return __put_dependency_ram_cache_impl(*args, **kwargs)


def preload_dependency_maps_for_candidates(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return _preload_dependency_maps_for_candidates_impl(*args, **kwargs)


def preload_all_dependency_maps_for_corpus(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return _preload_all_dependency_maps_for_corpus_impl(*args, **kwargs)


def _cache_dependency_maps_for_row(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return __cache_dependency_maps_for_row_impl(*args, **kwargs)


def _select_dependency_parquet_columns(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return __select_dependency_parquet_columns_impl(*args, **kwargs)


def build_dependency_cache_from_parquet_batches(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return _build_dependency_cache_from_parquet_batches_impl(*args, **kwargs)


def warm_dependency_cache_for_corpus(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return _warm_dependency_cache_for_corpus_impl(*args, **kwargs)


def start_dependency_cache_warmup(engine_globals, *args, **kwargs):
    globals().update(engine_globals)
    return _start_dependency_cache_warmup_impl(*args, **kwargs)

