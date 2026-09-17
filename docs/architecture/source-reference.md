# Referencja modułów pakietu `korpusuj`

Ten dokument jest indeksem kodu źródłowego. Obejmuje wszystkie moduły Pythona z przekazanego pakietu. Dla każdego modułu podaje symbole publiczne oraz prywatne funkcje, które rozpoczynają ważny etap przetwarzania. Szczegółowe przebiegi znajdują się w dokumentach podsystemów.

## `corpus`

### `korpusuj/corpus/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/corpus/creator.py`

- `get_application_root` (wiersz 75): Return the writable application root for external runtime assets.
- `format_sizesize_bytes` (wiersz 98)
- `class ColumnMapper` (wiersz 112)
  - Metody publiczne: `guess_columnself, field, cols`, `on_confirmself`
- `unpack_archivefile_path, status_label` (wiersz 257)
- `process_pdffile_path, status_label, app` (wiersz 270)
- `update_statuslabel, text, app` (wiersz 302)
- `initialize_stanzastatus_label, app` (wiersz 338)
- `initialize_spacystatus_label, app` (wiersz 349)
- `process_single_texttext, filename, status_label, progress_bar, app` (wiersz 370)
- `process_single_text_spacytext, filename, status_label, progress_bar, app` (wiersz 383)
- `process_file_globalfile_path, status_label, progress_bar, app, model_name, excel_mappings=None, processed_set=None` (wiersz 396)
- `process_files_thread_targetstatus_label, progress_bar_current, progress_bar_total, lbl_size_info, app, output_parquet_file, metadata_path, model_name, excel_mappings, enable_ner=True, enable_coreference=True, resume_mode=False, completion_callback=None` (wiersz 515): Thin GUI adapter over the shared creator orchestrator.
- `get_file_page_count` (wiersz 540)
- `update_file_selection_statusstatus_label=None` (wiersz 546): Aktualizuje etykietę paginacji i status wyboru plików.
- `reset_scrollable_frame_positionframe` (wiersz 575): Resetuje pozycję przewinięcia CTkScrollableFrame do góry.
- `render_file_pageframe, status_label=None` (wiersz 596): Renderuje tylko jedną stronę checkboxów.
- `change_file_pagedelta, frame, status_label=None` (wiersz 650): Przechodzi do poprzedniej/następnej strony listy plików.
- `select_filesframe, progress_bar_current, progress_bar_total, lbl_size_info, status_label, app` (wiersz 662)
- `mainparent_window=None` (wiersz 709)

### `korpusuj/corpus/creator_chunking.py`

Pure text chunking helpers for corpus creation.

- `has_multiline_bullet_layouttext: str, min_items: int=4` (wiersz 59): Dodatkowa heurystyka dla bulletów: sprawdza, czy tekst faktycznie zawiera wieloliniowy układ listy, a nie pojedyncze myślniki / gwiazdki przypadkowo występujące w tekście.
- `detect_record_styletext: str` (wiersz 74): Wykrywa styl listy / rekordów.
- `get_record_start_regexstyle: str` (wiersz 121): Zwraca regex startu rekordu dla zadanego stylu.
- `split_structured_segmentstext: str, style: str` (wiersz 136): Dzieli tekst na: - preambułę (wszystko przed pierwszym rekordem), - listę rekordów.
- `soft_cut_preserveblock: str, limit: int, backtrack_window: int=800` (wiersz 166): Miękkie cięcie BLOKU bez zmiany treści (LOSSLESS).
- `chunk_structured_recordstext: str, chunk_size: int, style: str='numeric', structured_chunk_size: int=4000, max_records_per_chunk: int=6, backtrack_window: int=800, max_dotless_chars: int=800, min_piece_in_danger: int=1200` (wiersz 243): Dedykowane, LOSSLESS cięcie dla tekstów rekordowych / wyliczeniowych.
- `chunk_text_safetext: str, chunk_size: int=15000, max_dotless_chars: int=800, backtrack_window: int=600, min_piece_in_danger: int=1200, structured_chunk_size: int=4000, max_records_per_chunk: int=6` (wiersz 332): Dzieli tekst na NLP-przyjazne fragmenty.

### `korpusuj/corpus/creator_cli.py`

GUI-free command-line adapter for corpus creation.

- `class CreatorCliConfigurationError` (wiersz 30): Invalid user input detected before creator/model initialization.
- `class StderrProgressReporter` (wiersz 34): Progress reporter that never contaminates machine-readable stdout.
  - Metody publiczne: `statusself, message: str`, `currentself, value: float`, `totalself, value: float`, `size_infoself, message: str`, `warningself, message: str`, `errorself, message: str, exc: Exception | None=None`, `tickself`
- `build_arg_parser` (wiersz 67): Build the public argument parser for the corpus creator CLI.
- `mainargv: list[str] | None=None` (wiersz 264): Run the corpus creator CLI and return its process exit code.

### `korpusuj/corpus/creator_core.py`

GUI-free protocol and run-option types for corpus creation.

- `class ProgressReporter` (wiersz 14): Receives creator status/progress events without depending on a GUI.
  - Metody publiczne: `statusself, message: str`, `currentself, value: float`, `totalself, value: float`, `size_infoself, message: str`, `warningself, message: str`, `errorself, message: str, exc: Exception | None=None`, `tickself`
- `class NullProgressReporter` (wiersz 40): No-op reporter suitable for headless calls and functional tests.
  - Metody publiczne: `statusself, message: str`, `currentself, value: float`, `totalself, value: float`, `size_infoself, message: str`, `warningself, message: str`, `errorself, message: str, exc: Exception | None=None`, `tickself`
- `class CreatorRunOptions` (wiersz 66): Explicit inputs for a future GUI-independent creator orchestration call.
  - Pola: `input_files: list[str]`, `output_parquet_file: str`, `metadata_path: str | None = None`, `model_name: str = 'stanza'`, `excel_mappings: dict[str, Any] | None = None`, `resume_mode: bool = False`, `processed_set: set[str] | None = None`, `enable_ner: bool = True`, `enable_coreference: bool = True`, `lemma_corrections_path: str | None = None`

### `korpusuj/corpus/creator_gui_adapter.py`

GUI adapter for creator progress events.

- `class GuiProgressReporter` (wiersz 14): Map creator progress events to the existing Tk/CustomTkinter widgets.
  - Metody publiczne: `statusself, message: str`, `currentself, value: float`, `totalself, value: float`, `size_infoself, message: str`, `warningself, message: str`, `errorself, message: str, exc: Exception | None=None`, `tickself`

### `korpusuj/corpus/creator_io.py`

GUI-free near-pure IO helpers used by the corpus creator.

- `calculate_real_total_sizefile_paths` (wiersz 14)
- `class UnsafeZipEntryError` (wiersz 33): A ZIP member would escape or subvert the selected extraction root.
- `safe_extract_ziparchive_path, destination` (wiersz 88): Safely extract one ZIP after validating its complete manifest.
- `process_xlsxfile_path, mapping=None` (wiersz 115)

### `korpusuj/corpus/creator_nlp.py`

Lightweight creator NLP state types.

- `class CreatorModelState` (wiersz 21): Owns NLP/SRL session objects explicitly instead of relying on globals.
  - Pola: `nlp_stanza: Any = None`, `nlp_spacy: Any = None`
  - Metody publiczne: `clear_allself`
- `reconstruct_nkjp_tagtag_str, morph_obj` (wiersz 64)
- `initialize_stanzastate: CreatorModelState, reporter, *, stanza_module, models_dir: str, enable_ner: bool=True, enable_coreference: bool=True` (wiersz 127): Load Stanza into ``state`` without tkinter/messagebox dependencies.
- `initialize_spacystate: CreatorModelState, reporter, *, spacy_module, herference_module, requests_module, models_dir: str, enable_ner: bool=True, enable_coreference: bool=True` (wiersz 221): Load local SpaCy/herference resources into ``state`` without GUI calls.
- `process_single_texttext, filename, state: CreatorModelState, reporter` (wiersz 344)
- `process_single_text_spacytext, filename, state: CreatorModelState, reporter` (wiersz 501)

### `korpusuj/corpus/creator_orchestration.py`

GUI-independent orchestration for corpus creation jobs.

- `class CreatorRunResult` (wiersz 61): Describe the published corpus and counters produced by a creator job.
  - Pola: `success: bool`, `output_file: Optional[str] = None`, `error_message: Optional[str] = None`, `warnings: list[str] = field(default_factory=list)`
- `initialize_stanza_label, _app` (wiersz 97)
- `initialize_spacy_label, _app` (wiersz 104)
- `process_single_texttext, filename, *_legacy` (wiersz 167)
- `process_single_text_spacytext, filename, *_legacy` (wiersz 172)
- `format_sizesize_bytes` (wiersz 290)
- `unpack_archivefile_path, status_label` (wiersz 302)
- `process_pdffile_path, status_label, app` (wiersz 314)
- `process_file_globalfile_path, status_label, progress_bar, app, model_name, excel_mappings=None, processed_set=None` (wiersz 345)
- `run_creator_joboptions, reporter=None, *, model_state=None, models_dir=None, cancel_requested=None` (wiersz 954): Run the existing creator workflow without Tkinter dependencies.
- `_write_creator_partdataframe, part_file` (wiersz 271)
- `_run_creator_job_implstatus_label, progress_bar_current, progress_bar_total, lbl_size_info, app, output_parquet_file, metadata_path, model_name, excel_mappings, resume_mode=False, completion_callback=None` (wiersz 457)

### `korpusuj/corpus/info.py`

Corpus information helpers for Korpusuj.

- `class CorpusInfoModel` (wiersz 23)
  - Pola: `total_docs: int`, `total_tokens: int`, `unique_lemmas: int`, `unique_orths: int`, `date_range: str`, `monthly_stats_str: str`, `meta_cols: list[str]`, `meta_str: str`
- `safe_len_or_zeroobj` (wiersz 34)
- `safe_total_tokens_for_corpus_infoinv_idx, df=None` (wiersz 41)
- `safe_unique_values_from_df_for_corpus_infodf, column_name` (wiersz 74)
- `safe_lazy_term_index_count_for_corpus_infoterm_index, inv_idx=None, attr=None, df=None, df_column=None` (wiersz 91): Return unique term count for dict indexes and Korpusuj LazyTermIndex.
- `parse_year_month_for_corpus_infoyear_value, month_value=None` (wiersz 167)
- `normalize_monthly_counts_for_corpus_infomonthly_counts` (wiersz 200)
- `get_corpus_metadata_columnsdf, exclude_cols=None` (wiersz 230)
- `build_corpus_info_modeldf, inv_idx` (wiersz 249)

### `korpusuj/corpus/lemma_corrections.py`

Optional, externally configured lemma corrections for corpus creation.

- `class LemmaCorrectionsError` (wiersz 13): Invalid lemma-corrections configuration.
- `class LemmaCorrectionsConfig` (wiersz 18)
  - Pola: `path: str | None = None`, `name: str = ''`, `schema_version: int = 1`, `sha256: str | None = None`, `rules: dict[tuple[str, str, str], tuple[str, str]] = field(default_factory=dict)`, `counts: Counter = field(default_factory=Counter)`
  - Metody publiczne: `enabledself`
- `disabled_lemma_corrections` (wiersz 31)
- `load_lemma_correctionspath_value: str | None` (wiersz 42)
- `apply_lemma_correctionstoken_details: Any, config: LemmaCorrectionsConfig` (wiersz 84)
- `lemma_corrections_identityconfig: LemmaCorrectionsConfig` (wiersz 99)
- `lemma_corrections_metadataconfig: LemmaCorrectionsConfig` (wiersz 108)

### `korpusuj/corpus/loading.py`

Corpus loading helpers for Korpusuj.

- `class LoadedCorpusBundle` (wiersz 22): Hold the canonical corpus, derived index and metadata prepared for callers.
  - Pola: `name: str`, `parquet_path: str`, `search_path: str`, `columns: list[str]`, `total_docs: int`, `total_tokens: int`, `monthly_token_counts: dict`, `korpus_meta: dict`, `search_meta: dict`, `dataframe: object`, `inverted_index: dict`
- `search_sidecar_pathparquet_path` (wiersz 37): Return .search sidecar path for a Parquet corpus path.
- `read_korpus_meta_from_parquet_schemaparquet_path` (wiersz 42): Read Parquet columns and optional korpus_meta JSON metadata.
- `ensure_search_index_for_parquetparquet_path, search_path, indexed_attrs, batch_docs=5000, progress_callback=None, builder=None` (wiersz 55): Ensure .search sidecar is fresh and return SearchIndex metadata.
- `parse_monthly_counts_from_metasearch_meta: dict, korpus_meta: dict` (wiersz 94): Prefer monthly counts from .search metadata, fallback to Parquet korpus_meta.
- `build_lazy_corpus_bundlename, parquet_path, search_path, columns, total_docs, total_tokens, monthly_counts, korpus_meta=None, search_meta=None` (wiersz 114): Build LazyCorpus and inverted_index dict without mutating engine globals.
- `prepare_loaded_corpus_bundlename, parquet_path, indexed_attrs, batch_docs=5000, progress_callback=None` (wiersz 161): Prepare one corpus bundle from Parquet + .search sidecar.

### `korpusuj/corpus/merger.py`

Guarded merger for compatible canonical Korpusuj Parquet corpora.

- `class CorpusMergeError` (wiersz 27)
- `class CorpusInputInfo` (wiersz 31)
  - Pola: `path: str`, `rows: int`, `row_groups: int`, `bytes: int`, `annotation_layers: dict[str, bool]`
- `class MergeResult` (wiersz 39)
  - Pola: `success: bool`, `output_path: str`, `rows: int`, `total_tokens: int`, `inputs: list[CorpusInputInfo]`, `warnings: list[str] = field(default_factory=list)`, `report_path: str | None = None`
  - Metody publiczne: `to_dictself`
- `inspect_merge_inputsvalues: Sequence[str | Path], *, allow_undeclared_annotation_layers: bool=False` (wiersz 143)
- `merge_corporainput_paths: Sequence[str | Path], output_path: str | Path, *, report_path: str | Path | None=None, replace: bool=False, batch_size: int=128, check_duplicates: bool=True, allow_undeclared_annotation_layers: bool=False, progress_callback: Callable[[int, int], None] | None=None` (wiersz 190)

### `korpusuj/corpus/merger_cli.py`

CLI for the guarded Korpusuj corpus merger.

- `parser` (wiersz 6)
- `mainargv=None` (wiersz 17)

## `dependency`

### `korpusuj/dependency/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/dependency/cache.py`

Dependency-cache access helpers for corpus search and analysis.

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/dependency/disk_cache.py`

- `class DependencyMapDiskCache` (wiersz 42): Persistent dependency cache 3f: doc_id -> compact parent_idx int32 BLOB.
  - Metody publiczne: `closeself`, `metaself`, `set_metaself, values`, `row_countself`, `payload_bytes_for_doc_idsself, doc_ids`, `is_fresh_and_completeself, total_docs=None`, `mark_rebuild_startedself, total_docs=None`, `mark_completeself, total_docs=None`, `getself, doc_id`, `get_manyself, doc_ids, batch_size=DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE`, `get_allself, batch_size=DEPENDENCY_CACHE_PRELOAD_BATCH_SIZE`, `putself, doc_id, dep_maps, commit=True`, `commitself`

### `korpusuj/dependency/lifecycle.py`

Inspect, build and atomically publish dependency and search index artifacts.

- `dependency_cache_pathparquet_path: str | os.PathLike[str]` (wiersz 27)
- `inspect_dependency_cacheparquet_path: str | os.PathLike[str], cache_path: str | os.PathLike[str] | None=None, *, check_integrity: bool=False` (wiersz 46): Inspect dependency-cache identity, completeness and optional SQLite integrity.
- `build_dependency_cache_atomicparquet_path: str | os.PathLike[str], cache_path: str | os.PathLike[str] | None=None, *, batch_docs: int=5000, progress_callback=None, publish: bool=True` (wiersz 134): Build and validate a dependency cache before optionally publishing it atomically.
- `json_safevalue: Any` (wiersz 200)
- `inspect_index_artifactsparquet_path, search_path=None, indexed_attrs=None, *, check_integrity=True` (wiersz 205): Inspect the combined .search and .dep_cache artifact set for a Parquet corpus.
- `build_index_artifacts_atomicparquet_path, search_path=None, indexed_attrs=None, *, batch_docs=5000, progress_callback=None` (wiersz 244): Build, validate and publish the .search and .dep_cache artifacts as one set.
- `_publish_pairsearch_stage: str, search_target: str, dep_stage: str, dep_target: str` (wiersz 223)

### `korpusuj/dependency/maps.py`

- `class LazyChildrenLookup` (wiersz 6): 3f hotfix: parent-only children lookup z adaptacyjną materializacją.
- `build_dependency_mapssentence_ids, word_ids, head_ids` (wiersz 98)

### `korpusuj/dependency/policy.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/dependency/runtime.py`

Coordinate dependency-cache access, preloading, warm-up and runtime state for corpus operations.

- `configure_dependency_runtime_bindings_189p*, state, config_provider, corpus_path_provider, loaded_corpus_provider, ram_mode_provider, ram_cache_size_provider, progress_reporter, legacy_index_ensurer, diagnostics_enabled, verbose_diagnostics_enabled` (wiersz 40): Bind the narrow engine services required by dependency runtime once.
- `get_dependency_disk_cache_for_corpus*args, **kwargs` (wiersz 567)
- `preload_dependency_maps_for_candidates*args, **kwargs` (wiersz 582)
- `preload_all_dependency_maps_for_corpus*args, **kwargs` (wiersz 587)
- `build_dependency_cache_from_parquet_batches*args, **kwargs` (wiersz 602)
- `warm_dependency_cache_for_corpus*args, **kwargs` (wiersz 607)
- `start_dependency_cache_warmup*args, **kwargs` (wiersz 612)

### `korpusuj/dependency/runtime_state.py`

Store the configurable runtime dependencies used by dependency-cache operations.

- `class DependencyRuntimeState` (wiersz 10)
  - Pola: `dependency_maps_cache: Dict[Any, Any] = field(default_factory=dict)`, `dependency_disk_caches: Dict[str, Any] = field(default_factory=dict)`, `dependency_warmup_threads: Dict[str, Any] = field(default_factory=dict)`, `dependency_warmup_stop_flags: Dict[str, Any] = field(default_factory=dict)`, `dependency_warmup_lock: Any = None`, `maps_cache_maxsize: int = 50000`, `candidate_max_docs: int = 3000`, `candidate_stream_batch_docs: int = 256`, `candidate_ram_budget_mb: int = 512`, `cache_preload_batch_size: int = 500`, `default_ram_mode: str = 'none'`, `default_ram_usage_label: str = 'Oszczędny'`, `ram_usage_labels: Dict[str, str] = field(default_factory=dict)`, `ram_mode_labels: Dict[str, str] = field(default_factory=dict)`, `disk_cache_version: str = ''`, `legacy_disk_cache_version: str = ''`, `parent_magic: Any = b''`
- `configure_dependency_runtime_state*, dependency_maps_cache: Dict[Any, Any], dependency_disk_caches: Dict[str, Any], dependency_warmup_threads: Dict[str, Any], dependency_warmup_stop_flags: Dict[str, Any], dependency_warmup_lock: Any, maps_cache_maxsize: int, candidate_max_docs: int, candidate_stream_batch_docs: int, candidate_ram_budget_mb: int, cache_preload_batch_size: int, default_ram_mode: str, default_ram_usage_label: str, ram_usage_labels: Dict[str, str], ram_mode_labels: Dict[str, str], disk_cache_version: str='', legacy_disk_cache_version: str='', parent_magic: Any=b''` (wiersz 36): Configure dependency runtime state from engine.py.
- `get_dependency_runtime_state` (wiersz 82)
- `dependency_runtime_state_configured` (wiersz 88)

### `korpusuj/dependency/warmup.py`

Dependency-cache warm-up entry points used by application startup and corpus loading.

Brak publicznych klas lub funkcji na poziomie modułu.

## `export`

### `korpusuj/export/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/export/excel.py`

Build tabular data for Excel exports of search results, tables and collocational profiles.

- `clean_for_exceldf` (wiersz 9)
- `build_search_results_export_dfresults, all_columns=None, visible_columns=None` (wiersz 60)
- `build_table_export_dfdata_rows, headers` (wiersz 80)
- `build_profile_export_dfprofile_dict` (wiersz 92)
- `write_excel_workbookfile_path, sheets` (wiersz 132)
- `write_csv_exportfile_path, dataframe` (wiersz 141)

### `korpusuj/export/subcorpus.py`

Helpery eksportu podkorpusów Korpusuj.

- `select_rows_from_search_resultsdf: pd.DataFrame, results: Iterable[Any]` (wiersz 98): Wybiera dokumenty z DataFrame na podstawie wyników wyszukiwania.
- `filter_dataframe_by_metadatadf: pd.DataFrame, date_from: str | None=None, date_to: str | None=None, author: str | None=None, title: str | None=None` (wiersz 127): Filtruje DataFrame po podstawowych metadanych.
- `compute_subcorpus_metadatadf: pd.DataFrame` (wiersz 165): Przelicza metadane frekwencyjne dla podkorpusu.
- `write_subcorpus_parquetdf: pd.DataFrame, file_path: str | Path, metadata: dict[str, Any] | None=None` (wiersz 205): Zapisuje DataFrame do Parquet z metadanymi Korpusuj w schema.metadata[b"korpus_meta"].
- `export_dataframe_to_subcorpus_parquetdf: pd.DataFrame, file_path: str | Path` (wiersz 224): Wygodny helper: przelicz metadane i zapisz podkorpus.

## `index`

### `korpusuj/index/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/index/builder.py`

Build the derived SQLite search index from a canonical Parquet corpus.

- `class SearchIndexBuilder` (wiersz 14): Stabilny builder indeksu: profil atrybutów, bezpieczne ograniczenie kolumn, większe batche.
  - Metody publiczne: `is_freshparquet_path, index_path=None, indexed_attrs=None`, `build_from_parquetself, parquet_path, index_path=None, batch_docs=5000, indexed_attrs=None, progress_callback=None`

### `korpusuj/index/cli.py`

Public CLI for creating, rebuilding and inspecting the derived index artifact set.

- `class CliInputError` (wiersz 19)
- `mainargv: Sequence[str] | None=None` (wiersz 79): Run the index lifecycle CLI and return its documented exit code.

### `korpusuj/index/lru.py`

- `class LRUCache` (wiersz 4)
  - Metody publiczne: `getself, key, default=None`, `putself, key, value`

### `korpusuj/index/postings.py`

- `class PostingList` (wiersz 4)
  - Metody publiczne: `encodepostings_by_doc`, `decodeblob`

### `korpusuj/index/sqlite_index.py`

- `get_search_indexed_attrsprofile=None` (wiersz 40)
- `search_sidecar_pathparquet_path` (wiersz 47)
- `class SearchIndex` (wiersz 59)
  - Metody publiczne: `closeself`, `metaself`, `total_docsself`, `total_tokensself`, `get_term_infoself, attr, value`, `get_postingsself, attr, value`, `get_doc_ids_for_termself, attr, value`, `get_ner_labelsself, doc_id, token_count`, `get_docself, doc_id`, `get_full_postags_036l4f4self, doc_id`, `get_corefs_138i3self, doc_id`, `get_coref_mentionsself, doc_id`, `filter_docs_by_metadataself, filters`, `get_docs_manyself, doc_ids, chunk_size=800`, `get_docs_many_036l4g8self, doc_ids, chunk_size=800`, `get_docs_many_for_result_tableself, doc_ids, chunk_size=800`, `get_docs_manyself, doc_ids, chunk_size=800`
- `class RegexSQLiteTooBroadError` (wiersz 447)
- `class RegexSQLiteCompileError` (wiersz 450)
- `class LazyTermIndex` (wiersz 663)
  - Metody publiczne: `getself, value, default=None`

### `korpusuj/index/status.py`

Inspect search-index freshness, compatibility and source identity.

- `inspect_search_indexparquet_path, index_path=None, indexed_attrs=None, *, check_integrity=False` (wiersz 19): Inspect a derived .search sidecar without creating or modifying it.

## `search`

### `korpusuj/search/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/search/backend.py`

Shared backend objects used to connect loaded corpora with indexed search execution.

- `class DocumentRow` (wiersz 10): Mapping returned by SQLite with legacy attribute-style row access.
- `class LazyCorpus` (wiersz 19): SQLite-backed corpus facade with an explicit Parquet compatibility fallback.
  - Metody publiczne: `indexself`, `get_docself, doc_id`, `get_ner_labelsself, doc_id, token_count`, `get_docs_manyself, doc_ids, chunk_size=800`, `get_background_frequenciesself, attr, values, ignore_case=False, chunk_size=800`, `background_indexself`, `materializeself`, `closeself`

### `korpusuj/search/cli.py`

Command-line interface for corpus search, analytics and export.

- `build_arg_parser` (wiersz 3738): Build the public argument parser for search, analytics and export.
- `mainargv: list[str] | None=None` (wiersz 3895): Run the search CLI and return its process exit code.

### `korpusuj/search/collocations.py`

GUI-free collocation computation for Korpusuj search results.

- `class CollocateFilterGroup` (wiersz 20): One public CLI collocate-filter group.
  - Pola: `upos: str | None = None`, `pos: str | None = None`, `tag: str | None = None`
- `class CollocationOptions` (wiersz 32): Configure lexical or syntactic collocation computation over search matches.
  - Pola: `mode: str = 'Liniowe'`, `upos_filter: str = 'Wszystkie'`, `pos_filter: str = 'Wszystkie'`, `form_mode: str = 'Lemat (base)'`, `ignore_case: bool = False`, `use_sentence_bound: bool = False`, `sort_mode: str = 'Log-Likelihood'`, `active_feat_filters: dict[str, str] = field(default_factory=dict)`, `collocate_filter_groups: list[CollocateFilterGroup] = field(default_factory=list)`, `min_freq: int = 1`, `min_range: int = 1`, `l_span: int = 5`, `r_span: int = 5`, `syn_dir: str = 'Podrzędnik'`, `deprel_filter: str = 'Wszystkie'`
- `class CollocationRow` (wiersz 52)
  - Pola: `rank: int`, `colloc: str`, `fnc: int`, `fc: int`, `ll: float`, `mi: float`, `t: float`, `log_dice: float`
- `class CollocationTable` (wiersz 64)
  - Pola: `rows: list[CollocationRow]`, `total_actual_slots: int`, `total_results: int`, `options: CollocationOptions`
- `class CollocateOccurrence` (wiersz 73): Concrete token occurrence counted as a collocate of a source query match.
  - Pola: `source_row_idx: int`, `source_match_start_idx: int`, `source_match_end_idx: int`, `source_match_text: str | None`, `collocate_idx: int`, `collocate_end_idx: int`, `collocate: str`, `collocate_form: str`, `collocate_rank: int | None = None`, `mode: str = 'linear'`, `direction: str = 'unknown'`, `distance: int | None = None`, `deprel: str | None = None`
- `class CollocateOccurrenceTable` (wiersz 92): Concrete collocate occurrences selected from the same slots as aggregate collocations.
  - Pola: `rows: list[CollocateOccurrence]`, `options: CollocationOptions`, `selected_collocates: list[str]`, `source_total_results: int`
- `get_clean_collocword: Any` (wiersz 131): Return normalized collocate text or empty string for invalid/punctuation-only items.
- `collect_collocate_occurrencesresults: Iterable[Any], df: Any, options: CollocationOptions, selected_collocates: Any=None, feat_mapping: dict[str, dict[str, int]] | None=None` (wiersz 495): Collect concrete candidate slots actually counted as collocates.
- `compute_collocationsresults: Iterable[Any], df: Any, inverted_index: dict[str, Any], options: CollocationOptions, feat_mapping: dict[str, dict[str, int]] | None=None` (wiersz 591): Compute collocation table from concordance result rows.
- `collocation_table_to_legacy_rowstable: CollocationTable` (wiersz 753): Return GUI/export-compatible rows: [rank, colloc, fnc, fc, ll, mi, t, log_dice].

### `korpusuj/search/cursor.py`

Lazy search cursors for discovering, counting, paging and materializing query matches.

- `condition_parts_138i3cond` (wiersz 99)
- `iter_index_doc_ids_138i3cursor_obj` (wiersz 193)
- `make_lazy_fulltext_ref_111index, doc_id, start, end, left_context_size=10, right_context_size=10, full_context_size=250` (wiersz 946): Return a lightweight reference for extended/full context.
- `is_lazy_fulltext_ref_111value` (wiersz 987)
- `resolve_lazy_fulltext_ref_111full_text_or_ref, context=None` (wiersz 1004): Resolve a lazy fulltext ref to [full_left, matched, full_right].
- `resolve_result_row_fulltext_111row` (wiersz 1151): Return row with result[2] resolved if it is a lazy fulltext ref.
- `class SearchCursor` (wiersz 1239): Lazily discover, count, page and materialize final matches for a planned query.
  - Metody publiczne: `count_hits_estimateself`, `count_hits_estimate_is_exactself`, `count_hitsself, exact=False`, `get_pageself, page=0, page_size=100`, `get_rangeself, start, stop`
- `class UnionSearchCursor` (wiersz 3970): Merge top-level OR branches while preserving lazy paging and duplicate elimination.
  - Metody publiczne: `get_rangeself, start, stop`, `get_pageself, page=0, page_size=100`, `count_hits_estimateself`, `count_hits_estimate_is_exactself`, `count_hitsself, exact=False`
- `get_table_context_source_policy` (wiersz 4204): Return the documented table-context source policy for scanners/tests.
- `release_materialized_searchcursor_caches_189n2results` (wiersz 5391): Release request-only cursor structures after independent rows exist.

### `korpusuj/search/cursor_runtime.py`

Configure and expose the runtime services used by SearchCursor instances.

- `class SearchCursorRuntime` (wiersz 10)
  - Pola: `dependency_cache_corpus_name_from_path: Callable[[str], str]`, `get_dependency_cache_ram_mode: Callable[[], str]`, `dependency_ram_cache_size_for_corpus: Callable[[str], int]`, `put_dependency_ram_cache: Callable[[str, int, Any], None]`, `preload_dependency_maps_for_candidates: Callable[..., int]`, `dependency_maps_cache: Any`, `dependency_maps_cache_maxsize: int = 50000`, `candidate_max_docs: int = 20000`, `candidate_stream_batch_docs: int = 512`, `full_context_size: int = 250`
- `configure_search_cursor_runtime*, dependency_cache_corpus_name_from_path: Callable[[str], str], get_dependency_cache_ram_mode: Callable[[], str], dependency_ram_cache_size_for_corpus: Callable[[str], int], put_dependency_ram_cache: Callable[[str, int, Any], None], preload_dependency_maps_for_candidates: Callable[..., int], dependency_maps_cache: Any, dependency_maps_cache_maxsize: int=50000, candidate_max_docs: int=20000, candidate_stream_batch_docs: int=512, full_context_size: int=250` (wiersz 27)
- `get_search_cursor_runtime` (wiersz 55)
- `dependency_cache_corpus_name_from_pathpath: str` (wiersz 61)
- `get_dependency_cache_ram_mode` (wiersz 65)
- `dependency_ram_cache_size_for_corpuscorpus_name: str` (wiersz 69)
- `put_dependency_ram_cachecorpus_name: str, doc_id: int, dep_maps: Any` (wiersz 73)
- `preload_dependency_maps_for_candidate_docs*args, **kwargs` (wiersz 77)
- `get_dependency_maps_cache` (wiersz 81)
- `candidate_max_docs` (wiersz 85)
- `candidate_stream_batch_docs` (wiersz 89)
- `full_context_size` (wiersz 93)
- `configure_full_context_size_providerprovider: Optional[Callable[[], int]]` (wiersz 98): Inject a lazy provider for the extended/full context size.
- `get_full_context_size` (wiersz 107): Return current extended context size from settings/global state.

### `korpusuj/search/diagnostics.py`

Diagnostic logging and plan-summary helpers for the shared search engine.

- `configure_search_diagnostics*, config_provider: Optional[Callable[[], dict]]=None` (wiersz 14)
- `search_diag_enabled` (wiersz 19)
- `search_diag_logmessage, *args, **kwargs` (wiersz 33): Safe diagnostic logger; must never interrupt search execution.
- `summarize_search_plan_for_logplan` (wiersz 48)
- `search_verbose_diagnostics_enabledconfig=None` (wiersz 61)
- `search_verbose_diag_loglogger, marker: str, semantic_event: str, message: str, *args, config=None, **kwargs` (wiersz 89)
- `korpusuj_diagnostics_enabled_145c1config_obj=None` (wiersz 107)
- `korpusuj_verbose_diagnostics_enabled_145c1config_obj=None` (wiersz 138)
- `korpusuj_logging_diagnostics_enabled_145c2config_obj=None` (wiersz 163)
- `korpusuj_logging_verbose_enabled_145c2config_obj=None` (wiersz 188)
- `korpusuj_diagnostics_enabled_145c1config_obj=None` (wiersz 215)
- `korpusuj_verbose_diagnostics_enabled_145c1config_obj=None` (wiersz 219)
- `korpusuj_logging_diagnostics_enabled_145c2config_obj=None` (wiersz 247)
- `korpusuj_logging_verbose_enabled_145c2config_obj=None` (wiersz 270)
- `korpusuj_diagnostics_enabled_145c1config_obj=None` (wiersz 296)
- `korpusuj_verbose_diagnostics_enabled_145c1config_obj=None` (wiersz 300)
- `korpusuj_verbose_log_145c2marker, semantic_event, message, *args, **kwargs` (wiersz 305)

### `korpusuj/search/errors.py`

Exception types raised while parsing, validating and executing search queries.

- `class QueryValidationError` (wiersz 6)
- `class SearchExecutionError` (wiersz 10)
- `class QueryParseError` (wiersz 14)

### `korpusuj/search/executor.py`

Plan and execute corpus searches through the shared cursor and index interfaces.

- `configure_search_executor*, search_cursor_cls=None, search_index_cls=None` (wiersz 16)
- `class SearchExecutor` (wiersz 36): Execute planned queries against a configured SearchCursor implementation.
  - Metody publiczne: `executeself, query, left_context_size=10, right_context_size=10`
- `class CorpusSearchExecutor` (wiersz 83): Bind shared search execution to a loaded corpus and index.
  - Metody publiczne: `searchself, query, left_context_size=10, right_context_size=10`

### `korpusuj/search/headless.py`

GUI-independent request, result and backend contracts for corpus search.

- `class HeadlessSearchNotConfiguredError` (wiersz 22): Raised when headless search cannot run because no backend adapter was supplied.
- `class SearchMessage` (wiersz 27): A GUI-independent message returned by headless search.
  - Pola: `level: str`, `text: str`, `code: str | None = None`, `details: dict[str, Any] = field(default_factory=dict)`
- `class SearchRequest` (wiersz 37): GUI-independent search request.
  - Pola: `query: str`, `corpus_name: str`, `left_context: int = 10`, `right_context: int = 10`, `sort_option: str | None = None`, `date_from: str | None = None`, `date_to: str | None = None`, `selected_sense: str | None = None`, `limit: int | None = 100`, `offset: int = 0`, `options: dict[str, Any] = field(default_factory=dict)`
- `class SearchBackendContext` (wiersz 54): Bundle the corpus, index and adapter objects required by headless search.
  - Pola: `dataframes: Mapping[str, Any] = field(default_factory=dict)`, `corpora: Mapping[str, Any] | None = None`, `current_corpus_path: str | None = None`, `config: Mapping[str, Any] | None = None`, `corpus_name: str = ''`, `parquet_path: str | None = None`, `search_path: str | None = None`, `dep_cache_path: str | None = None`, `has_parquet: bool = False`, `has_search_index: bool = False`, `has_dep_cache: bool = False`, `df_type: str | None = None`, `search_df_type: str | None = None`, `df_is_lazy: bool = False`, `search_df_is_lazy: bool = False`, `stats_rows: int | None = None`, `search_rows: int | None = None`, `indexed_attrs: tuple[str, ...] = ()`, `metadata_columns: tuple[str, ...] = ()`, `metadata_column_count: int = 0`, `config_snapshot: dict = field(default_factory=dict)`, `find_lemma_context_adapter: object | None = None`
- `class SearchHit` (wiersz 80): JSON-serializable representation of a single concordance hit returned by headless search.
  - Pola: `doc_id: int | None = None`, `start: int | None = None`, `end: int | None = None`, `match_text: str = ''`, `left_context: str = ''`, `right_context: str = ''`, `extended_left: str = ''`, `extended_match: str = ''`, `extended_right: str = ''`, `metadata: dict[str, Any] = field(default_factory=dict)`, `raw: Any | None = None`
- `class SearchResultBundle` (wiersz 97): GUI-independent search result envelope.
  - Pola: `request: SearchRequest`, `results: Any = field(default_factory=list)`, `total_hits: int | None = None`, `warnings: list[str] = field(default_factory=list)`, `messages: list[SearchMessage] = field(default_factory=list)`, `statistics_payload: dict[str, Any] | None = None`, `metadata: dict[str, Any] = field(default_factory=dict)`, `limit: int | None = None`, `offset: int = 0`, `has_more: bool | None = None`
- `validate_search_requestrequest: SearchRequest` (wiersz 112): Validate the basic shape of a headless request.
- `normalize_search_result_to_hitrow: Any` (wiersz 182): Normalize a shared-search result row to the public SearchHit representation.
- `normalize_search_results_to_hitsresults: Any, *, limit: int | None=None, offset: int=0` (wiersz 265): Normalize an iterable of current result rows to a list of SearchHit.
- `run_search_headlessrequest: SearchRequest, context: SearchBackendContext` (wiersz 331): Run a GUI-independent search using an injected backend adapter.

### `korpusuj/search/headless_runner.py`

Build and run GUI-independent search contexts over canonical Parquet corpora.

- `configure_non_gui_search_cursor_runtime*, full_context_size: int=250, candidate_max_docs: int=3000, candidate_stream_batch_docs: int=256, dependency_maps_cache: Any=None` (wiersz 39): Configure SearchCursor runtime with safe non-GUI defaults.
- `build_lazy_corpus_for_headlessparquet_path: str | Path, *, search_path: str | Path | None=None, corpus_name: str | None=None, columns: list[str] | None=None, total_docs: int | None=None, meta: dict[str, Any] | None=None` (wiersz 117): Build a LazyCorpus for non-GUI headless execution.
- `build_corpus_search_executor_for_headlesslazy_corpus: LazyCorpus` (wiersz 145): Wire SearchExecutor dependencies and return a CorpusSearchExecutor.
- `class MaterializedSearchResults` (wiersz 190): List-compatible page of results that preserves the full hit count.
  - Metody publiczne: `count_hitsself`
- `build_non_gui_find_lemma_context_adapterexecutor: CorpusSearchExecutor, *, limit: int | None=None, normalize_hits: bool=False, normalize_fn: Callable[..., Any] | None=None` (wiersz 371): Build the request-aware non-GUI search adapter.
- `build_headless_context_from_parquetparquet_path: str | Path, *, corpus_name: str | None=None, search_path: str | Path | None=None, limit: int | None=None, normalize_hits: bool=False, normalize_fn: Callable[..., Any] | None=None, full_context_size: int=250, candidate_max_docs: int=3000, candidate_stream_batch_docs: int=256, dependency_maps_cache: Any=None, config: dict[str, Any] | None=None` (wiersz 458): Build a SearchBackendContext backed by a Parquet corpus and its .search sidecar.

### `korpusuj/search/legacy_adapter.py`

Controlled adapter for the maintained legacy search fallback.

- `legacy_fallback_on_sqlite_exception_enabled_121config: Any=None` (wiersz 46): Return whether LazyCorpus SQLite exceptions may fall back to legacy.
- `legacy_route_payload_121*, legacy_source: str, legacy_reason: str, query: Any=None, selected_corpus: Any=None, df: Any=None, route_name: str | None=None, extra: dict[str, Any] | None=None` (wiersz 62)
- `call_legacy_find_lemma_context_121legacy_impl: Callable[..., Any], query: Any, df: Any, selected_corpus: Any, left_context_size: int=10, right_context_size: int=10, warnings_list: list[str] | None=None, *, legacy_source: str, legacy_reason: str, route_name: str | None=None, logger: Any=None, extra: dict[str, Any] | None=None` (wiersz 77): Log a stable adapter-boundary marker, then call legacy implementation.
- `legacy_no_slice_policy_122` (wiersz 113): Return the policy that keeps the maintained fallback matcher intact.
- `legacy_strict_config_snapshot_122config: Any=None` (wiersz 132): Return an observable snapshot of strict legacy fallback config.
- `legacy_observability_payload_122*, legacy_source: str, legacy_reason: str, query: Any=None, selected_corpus: Any=None, df: Any=None, route_name: str | None=None, event: str='route_enter', extra: dict[str, Any] | None=None` (wiersz 156)
- `log_legacy_route_observability_122logger: Any, *, legacy_source: str, legacy_reason: str, query: Any=None, selected_corpus: Any=None, df: Any=None, route_name: str | None=None, event: str='route_enter', extra: dict[str, Any] | None=None` (wiersz 185): Log structured observability data for entry to or exit from a legacy search route.
- `legacy_adapter_selftest_122config: Any=None` (wiersz 217): Tiny non-invasive self-test hook for scan/diagnostics.

### `korpusuj/search/models.py`

State models shared by search execution and result presentation.

- `class SearchState` (wiersz 9)
  - Pola: `query: str = ''`, `corpus: str = ''`, `results: list = field(default_factory=list)`, `monthly_lemma_freq: dict = field(default_factory=dict)`, `true_monthly_totals: dict = field(default_factory=dict)`, `monthly_freq_for_use: dict = field(default_factory=dict)`, `monthly_tfidf_for_use: dict = field(default_factory=dict)`, `monthly_zscore_for_use: dict = field(default_factory=dict)`, `lemma_df_cache: dict = field(default_factory=dict)`, `warnings: list = field(default_factory=list)`, `fq_data: list = field(default_factory=list)`, `fq_data_token: list = field(default_factory=list)`, `fq_data_month: list = field(default_factory=list)`, `s_lemma_total_freq: list = field(default_factory=list)`, `s_lemma_global_pmw: list = field(default_factory=list)`, `s_lemma_global_tfidf: list = field(default_factory=list)`, `unique_lemmas: set = field(default_factory=set)`, `has_dates: bool = False`, `colloc_data: list = field(default_factory=list)`, `current_profile_dict: dict = field(default_factory=dict)`, `profile_target_lemma: str = ''`, `profile_data: list = field(default_factory=list)`, `profile_rel_options: list = field(default_factory=list)`, `profile_selected_rel: str = ''`

### `korpusuj/search/output_schema.py`

JSON-safe output schema helpers for Korpusuj search results.

- `search_message_to_jsonablemessage: SearchMessage | Mapping[str, Any] | Any` (wiersz 56): Convert a SearchMessage-like object to a JSON-safe dict.
- `search_request_to_jsonablerequest: SearchRequest | Mapping[str, Any] | Any` (wiersz 75): Convert a SearchRequest-like object to a JSON-safe dict.
- `search_hit_to_jsonablehit: SearchHit` (wiersz 120): Convert a normalized SearchHit to a JSON-safe dict.
- `search_result_to_jsonablerow: SearchHit | Any` (wiersz 137): Convert one SearchHit or raw legacy result row to a JSON-safe dict.
- `search_results_to_jsonableresults: Iterable[Any] | Any` (wiersz 143): Convert an iterable of result rows to JSON-safe result dictionaries.
- `search_bundle_to_jsonablebundle: SearchResultBundle | Any` (wiersz 154): Convert a SearchResultBundle-like object to public schema v1 dict.

### `korpusuj/search/parser.py`

Parse CQL into token, sentence and frequency conditions used by search planning.

- `class QueryParseError` (wiersz 10): Fallback parse error; overwritten by engine.QueryParseError at runtime.
- `split_top_levels, delimiter` (wiersz 14)
- `find_top_level_operators, op` (wiersz 38)
- `parse_single_conditions` (wiersz 52)
- `parse_conditionss` (wiersz 129)
- `extract_square_bracketss: str` (wiersz 157): Extracts top-level [ ...
- `parse_query_groupgroup` (wiersz 219)
- `parse_sentence_conditionss` (wiersz 245)
- `parse_frequency_attributesquery, attr='frequency'` (wiersz 267)
- `parse_frequency_attributequery` (wiersz 289)
- `parse_frequency_base_attributequery` (wiersz 296)
- `split_sentence_operator_queryquery` (wiersz 304): Split ``LHS <s RHS>`` into its two ordinary CQL query parts.

### `korpusuj/search/planner.py`

Translate parsed CQL conditions into executable plans for the SQLite search index.

- `normalize_plain_text_queryquery` (wiersz 145): Convert a fully naked query to adjacent orth token brackets.
- `class SearchPlanner` (wiersz 167): Translate parsed CQL into executable indexed-search plans.
  - Metody publiczne: `planself, query, index`
- `is_coref_doc_filter_attr_138l2attr` (wiersz 853)

### `korpusuj/search/result_materialization.py`

Shared helpers for exact hit counting and SearchCursor materialization.

- `count_searchcursor_hits_036l4g48eresults: Any, *, search_token: Any=None, logger: Any=None, perf_counter: Optional[Callable[[], float]]=None` (wiersz 38): Count every final SearchCursor hit exactly once.
- `count_final_searchcursor_hitsresults: Any, *, search_token: Any=None, logger: Any=None, perf_counter: Optional[Callable[[], float]]=None` (wiersz 70): Return a JSON-safe final hit count for SearchCursor-like results.
- `materialize_searchcursor_results_036l4g48eresults: Any, *, cancel_check: Optional[Callable[[], Any]]=None, search_token: Any=None, logger: Any=None, perf_counter: Optional[Callable[[], float]]=None` (wiersz 129): Materialize SearchCursor results in the dictionary shape consumed by GUI and export callers.

### `korpusuj/search/search_service.py`

Stable service-layer entry points for GUI-independent corpus search.

- `run_search_servicerequest: SearchRequest, context: SearchBackendContext` (wiersz 40): Run search through the current headless adapter contract.
- `build_search_service_context_from_parquet*args: Any, **kwargs: Any` (wiersz 48): Build a non-GUI SearchBackendContext for a Parquet corpus and its .search sidecar.

### `korpusuj/search/sentence_operator.py`

- `split_sentence_operator_queryquery: Any` (wiersz 6)
- `with_sentence_operator_metadataplan: Any, query: Any, parse_sentence_conditions` (wiersz 16)
- `as_listvalue: Any` (wiersz 25)
- `sentence_boundssentence_ids: Sequence[Any], token_index: int` (wiersz 36)
- `text_value_matchescandidate: Any, values: Sequence[Any], operator: str='=', match_type: str='exact'` (wiersz 44)
- `token_attrdoc: Dict[str, Any], key: str, token_index: int` (wiersz 60)
- `condition_partscondition: Any` (wiersz 66)
- `token_matches_condition_groupdoc: Dict[str, Any], token_index: int, group: Any` (wiersz 78)
- `match_pattern_in_rangedoc: Dict[str, Any], start_index: int, conditions: Sequence[Any], end_limit: int` (wiersz 87)
- `sentence_satisfies_conditionsdoc: Dict[str, Any], match_start: int, ordered: bool, conditions: Sequence[Any]` (wiersz 105)
- `hit_partshit: Any` (wiersz 112)

### `korpusuj/search/statistics.py`

Shared search-frequency and statistics computation for GUI and CLI callers.

- `class SearchStatistics` (wiersz 22): Container for search statistics computed after concordance hits.
  - Pola: `true_monthly_totals: dict[str, int] = field(default_factory=dict)`, `monthly_freq_for_use: dict[str, Any] = field(default_factory=dict)`, `monthly_tfidf_for_use: dict[str, Any] = field(default_factory=dict)`, `monthly_zscore_for_use: dict[str, Any] = field(default_factory=dict)`, `fq_data: list[Any] = field(default_factory=list)`, `fq_data_token: list[Any] = field(default_factory=list)`, `fq_data_month: list[Any] = field(default_factory=list)`, `s_lemma_total_freq: list[Any] = field(default_factory=list)`, `s_lemma_global_pmw: list[Any] = field(default_factory=list)`, `s_lemma_global_tfidf: list[Any] = field(default_factory=list)`, `s_lemma_monthly_trends: list[Any] = field(default_factory=list)`, `s_lemma_monthly_tfidf: list[Any] = field(default_factory=list)`, `s_lemma_monthly_zscore: list[Any] = field(default_factory=list)`, `has_dates: bool = False`
- `normalize_monthly_token_counts_for_searchraw_monthly_counts: Any` (wiersz 44): Normalize corpus monthly token counts used by search statistics.
- `class SearchFrequencyInputs` (wiersz 84): Intermediate frequency inputs derived from concordance hits.
  - Pola: `unique_matched_tokens: dict[Any, int] = field(default_factory=dict)`, `unique_lemmas: set[Any] = field(default_factory=set)`, `monthly_lemma_freq: dict[str, dict[Any, int]] = field(default_factory=dict)`, `exact_orth_df: dict[Any, set[Any]] = field(default_factory=dict)`, `exact_lemma_df: dict[Any, set[Any]] = field(default_factory=dict)`
- `collect_search_frequency_inputsresults_sorted: Any` (wiersz 94): Collect first-stage frequency inputs from sorted concordance results.
- `class GlobalFrequencyTables` (wiersz 211): Global frequency/statistics tables derived from frequency inputs.
  - Pola: `fq_data_token: list[Any] = field(default_factory=list)`, `fq_data: list[Any] = field(default_factory=list)`, `s_lemma_total_freq: list[Any] = field(default_factory=list)`, `s_lemma_global_pmw: list[Any] = field(default_factory=list)`, `s_lemma_global_tfidf: list[Any] = field(default_factory=list)`
- `build_global_frequency_tables*, unique_matched_tokens: dict[Any, int], monthly_lemma_freq: dict[str, dict[Any, int]], exact_orth_df: dict[Any, set[Any]], exact_lemma_df: dict[Any, set[Any]], total_token_count: int | float, total_docs: int | float, df_for_matched_key` (wiersz 221): Build global token/lemma frequency tables for search statistics.
- `class MonthlyFrequencyTables` (wiersz 289): Monthly frequency/statistics tables derived from search frequency inputs.
  - Pola: `monthly_freq_for_use: dict[str, dict[Any, float]] = field(default_factory=dict)`, `monthly_tfidf_for_use: dict[str, dict[Any, float]] = field(default_factory=dict)`, `monthly_zscore_for_use: dict[str, dict[Any, float]] = field(default_factory=dict)`, `fq_data_month: list[Any] = field(default_factory=list)`
- `build_monthly_frequency_tables*, monthly_lemma_freq: dict[str, dict[Any, int]], unique_lemmas: set[Any], true_monthly_totals: dict[str, int], total_docs: int | float, exact_lemma_df: dict[Any, set[Any]], df_for_matched_key, calc_z_score_func` (wiersz 298): Build monthly PMW/TF-IDF/z-score tables for search statistics.

## `semantic`

### `korpusuj/semantic/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/semantic/engine.py`

Semantic analysis services for artifact loading, neighbor indexes, hubness, frame induction, graph expansion and report generation.

- `notify_statusmsg` (wiersz 25): Forward a semantic-processing status message to the optional application status hook.
- `class SemanticEngine` (wiersz 50): Klasa zarządzająca logiką, ładowaniem i pamięcią sieci semantycznej.
  - Metody publiczne: `network_existsself, current_corpus_path`, `open_training_setupself, parent_app, current_corpus_name, current_corpus_path, theme, on_success_callback`, `build_semantic_reportself, parent_app, current_corpus_name, current_corpus_path, lemma, theme, open_report_callback, params=None`, `load_neighborsself, current_corpus_path`, `get_max_available_neighborsself`, `get_neighborsself, word, top_n=25`, `is_mutual_knnself, u: str, v: str`, `dynamic_bridge_thresholdfreq_u: int, freq_v: int, base: float=0.55`, `get_or_create_sensesself, lemma`, `get_cached_sensesself, lemma`, `disambiguate_instanceself, sentence_tokens, target_idx, lemma`, `get_representation_vectorself, lemma, sense_id=None`, `build_graph_context_vectorself, root_lemma, parent_lemma, root_sense_id=None, parent_sense_id=None, local_neighbor_lemmas=None, alpha=0.45, beta=0.4, gamma=0.1, delta=0.05`, `choose_graph_senseself, child_lemma, root_lemma, parent_lemma, root_sense_id=None, parent_sense_id=None, local_neighbor_lemmas=None, allow_induce=True`, `get_or_create_framesself, lemma`, `choose_graph_frameself, child_lemma, root_lemma, parent_lemma, root_sense_id=None, parent_sense_id=None, local_neighbor_lemmas=None`, `get_halo_candidatesself, center_lemma, top_n=150, min_sim=0.35`, `get_contextual_neighborsself, center_lemma, top_n=25, root_lemma=None, parent_lemma=None, root_sense_id=None, parent_sense_id=None, local_neighbor_lemmas=None, base_weight=0.45, parent_weight=0.3, root_weight=0.2, local_weight=0.0, domain_lambda=0.2`

### `korpusuj/semantic/profile_provider.py`

- `class ProfileRowData` (wiersz 79)
  - Pola: `lemmas: list`, `word_ids: list`, `head_ids: list`, `deprels: list`, `upostags: list`, `postags: list`, `full_postags: list`, `sentence_ids: list`, `feats: list`
- `class ProfileDocProvider` (wiersz 99)
  - Metody publiczne: `closeself`, `diagnosticsself`, `preloadself, doc_ids: Iterable[int]`, `get_rowself, doc_id: int`
- `profile_provider_statusdf` (wiersz 310)
- `can_use_profile_providerdf` (wiersz 337)
- `build_profile_provider_for_resultsdf, results: Iterable[Any]` (wiersz 342)

### `korpusuj/semantic/reports_analytical_v7_1.py`

- `configure_loggingverbose: bool=False` (wiersz 31)
- `class ArtifactBundle` (wiersz 41)
  - Pola: `label: str`, `source_path: str`, `df_neighbors: pd.DataFrame`, `vectors: Dict[str, np.ndarray]`, `metadata: Dict`, `index: Dict[str, List[Tuple[str, float, int]]]`
  - Metody publiczne: `resolve_keyself, lemma: str`, `neighbors_ofself, lemma: str, top_k: int=80, min_similarity: float=0.0`, `lemma_freqself, lemma: str`, `max_neighbors_forself, lemma: str`
- `load_artifact_bundlepath_like: str, label: Optional[str]=None` (wiersz 142)
- `cosine_similarityu: Optional[np.ndarray], v: Optional[np.ndarray]` (wiersz 205)
- `normalized_centroidvectors: List[np.ndarray]` (wiersz 215)
- `pairwise_mean_cosvectors: List[np.ndarray]` (wiersz 222)
- `percentilevalues: List[float], p: float` (wiersz 232)
- `minmax_scale_dictvalues: Dict[str, float]` (wiersz 236)
- `class ReportConfigV7_1` (wiersz 251)
  - Pola: `lemma: str`, `output_dir: str`, `top_k_neighbors: int = 0`, `min_similarity: float = 0.3`, `top_n_core_words: int = 15`, `top_n_distinctive_words: int = 15`, `top_n_interpretive_words: int = 15`, `members_table_size: int = 50`, `tail_table_size: int = 24`, `orphan_table_size: int = 60`, `use_sense_inducer: bool = True`, `export_csv: bool = True`, `hubness_similarity_threshold: float = 0.4`, `frame_edge_threshold: float = 0.42`, `bridge_similarity_threshold: float = 0.45`, `frame_assignment_min_similarity: float = 0.1`, `core_quantile: float = 0.6`, `max_plot_words: int = 120`, `local_neighbor_window: int = 80`
- `class AnalyticalSemanticReportBuilderV7_1` (wiersz 273)
  - Metody publiczne: `build_globality_indexself`, `get_globalityself, lemma: str`, `collect_semantic_fieldself, key: str`, `induce_framesself, key: str, candidate_words: List[str]`, `compute_word_metricsself, key: str, field_rows: List[Dict], frames: List[Dict], local_graph: nx.Graph`, `compute_frame_metricsself, key: str, frames: List[Dict], word_df: pd.DataFrame`, `compute_global_overviewself, key: str, field_rows: List[Dict], local_graph: nx.Graph, frame_df: pd.DataFrame`, `compute_projectionself, key: str, word_df: pd.DataFrame, frame_df: pd.DataFrame, frames: List[Dict]`, `compute_frame_similarityself, frames: List[Dict]`, `build_orphan_rowsself, word_df: pd.DataFrame`, `compute_diagnosticsself, key: str, field_rows: List[Dict], word_df: pd.DataFrame, frames: List[Dict], local_graph: nx.Graph`, `export_sidecarsself, payload: Dict, methodology: Dict, diagnostics: Dict`, `compute_reverse_fieldself, target_lemma: str`, `render_htmlself, payload: Dict`, `buildself`
- `build_arg_parser` (wiersz 1766)
- `mainargv: Optional[Sequence[str]]=None` (wiersz 1792)

### `korpusuj/semantic/sense_inducer.py`

- `class SenseInducer` (wiersz 8): Kompatybilna wstecz wersja inducera: - nadal zwraca sense_id, members, vector
  - Metody publiczne: `cosine_simu, v`, `chinese_whispersG, iters=20, seed=42`, `inducecls, lemma, vectors_dict, semantic_index, max_neighbors=None, sim_threshold=None, min_cluster_size=None, debug=False`

### `korpusuj/semantic/trainer.py`

- `configure_loggingverbose: bool=False` (wiersz 32)
- `parse_maybe_listvalue` (wiersz 57)
- `normalize_lemmalemma: str, lower: bool=True, strip_whitespace: bool=True` (wiersz 80)
- `is_punctuation_liketoken: str` (wiersz 91)
- `is_numeric_liketoken: str` (wiersz 102)
- `has_lettertoken: str` (wiersz 110)
- `clean_ner_piecetoken: str` (wiersz 115): Czyści pojedynczy segment encji przed sklejeniem: - usuwa białe znaki, - obcina śmieci interpunkcyjne z początku/końca, - normalizuje wielokrotne podkreślniki.
- `is_source_signature_liketoken: str` (wiersz 138): Wykrywa ciągi typu: - olnk/PAP/wPolityce.pl - red/wPolityce.pl/PAP/X/Fb/media - PAP/ans - URL-like / source-line / tag redakcyjny
- `is_valid_ner_piecetoken: str` (wiersz 173): Czy pojedynczy segment nadaje się do bycia częścią encji?
- `build_entity_lemmaparts: List[str]` (wiersz 191): Buduje bezpieczną postać MWE/encji.
- `class CorpusStats` (wiersz 225)
  - Pola: `documents_seen: int = 0`, `documents_used: int = 0`, `sentences_seen: int = 0`, `sentences_used: int = 0`, `tokens_seen: int = 0`, `tokens_used: int = 0`, `unique_lemmas: int = 0`, `filtered_out_punct: int = 0`, `filtered_out_numeric: int = 0`, `filtered_out_upos: int = 0`, `filtered_out_noise: int = 0`, `empty_sentences_after_filter: int = 0`
- `class TrainingConfig` (wiersz 242)
  - Pola: `parquet_path: str`, `output_dir: str`, `algo: str = 'word2vec'`, `vector_size: int = 150`, `window: int = 15`, `min_count: int = 5`, `workers: int = max(1, (os.cpu_count() or 2) - 1)`, `sg: int = 1`, `epochs: int = 10`, `negative: int = 10`, `sample: float = 1e-05`, `seed: int = 42`, `lower: bool = True`, `keep_punct: bool = False`, `keep_numeric: bool = False`, `allowed_upos: Optional[List[str]] = None`, `batch_size: int = 512`, `save_full_model: bool = True`, `save_text_vectors: bool = False`, `precompute_neighbors: int = 0`, `neighbors_for_top_vocab: int = 2000`
- `class ParquetSentenceIterator` (wiersz 269)
- `ensure_gensim_available` (wiersz 523)
- `train_embedding_modelsentences: Iterable[List[str]], config: TrainingConfig` (wiersz 531)
- `build_output_prefixparquet_path: Path, output_dir: Path, algo: str` (wiersz 574)
- `save_metadataprefix: Path, config: TrainingConfig, stats: CorpusStats, top_lemmas: Sequence` (wiersz 579)
- `save_neighborsprefix: Path, model, lemma_counter: Counter, n_neighbors: int, top_vocab: int` (wiersz 593)
- `save_lightweight_vectorsprefix: Path, model, lemma_counter: Counter, top_vocab: int, n_neighbors: int=0` (wiersz 622)
- `save_model_artifactsprefix: Path, model, config: TrainingConfig, iterator: ParquetSentenceIterator` (wiersz 680)
- `build_arg_parser` (wiersz 727)
- `train_from_parquetconfig: TrainingConfig` (wiersz 758)
- `mainargv: Optional[Sequence[str]]=None` (wiersz 809)

### `korpusuj/semantic/word_profile.py`

- `safe_llo: float, e: float` (wiersz 8): Bezpieczne log-likelihood bez ryzyka dzielenia przez zero.
- `class WordProfileHit` (wiersz 18)
  - Pola: `row_idx: int`, `token_idx: int`
- `class WordProfileRow` (wiersz 24)
  - Pola: `relation: str`, `collocate: str`, `cooc_freq: int`, `doc_freq: int`, `global_freq: int`, `log_dice: float`, `mi_score: float`, `t_score: float`, `ll_score: float`, `collocate_upos: str = ''`, `example_refs: List[Tuple[int, int, int]] = field(default_factory=list)`, `display_collocate: str = ''`
- `get_mwe_phrasehead_idx: int, word_ids: List[Any], children_by_head: Dict[int, List[int]], deprels: List[Any], lemmas: List[Any], ignore_case: bool=True` (wiersz 291): Zbiera i skleja wielowyrazowe jednostki (MWE) na podstawie drzewa zależności.
- `unpack_word_profile_hitres: Any` (wiersz 325)
- `default_case_extractorfull_postag: str` (wiersz 329)
- `find_sentence_boundssentence_ids: List[Any], token_idx: int` (wiersz 357)
- `find_preposition_for_token_fasttoken_word_id: int, children_by_head: Dict[int, List[int]], deprels: List[Any], lemmas: List[Any], ignore_case: bool=True` (wiersz 367)
- `normalize_lemmaval: Any, ignore_case: bool=True` (wiersz 379)
- `get_child_indices_for_word_idtoken_word_id: int, children_by_head: Dict[int, List[int]]` (wiersz 383)
- `child_lemmas_for_word_idtoken_word_id: int, children_by_head: Dict[int, List[int]], lemmas: List[Any], deprels: List[Any], ignore_case: bool=True, only_deprels: Optional[Iterable[str]]=None` (wiersz 389)
- `first_child_lemma_matchingtoken_word_id: int, children_by_head: Dict[int, List[int]], lemmas: List[Any], deprels: List[Any], ignore_case: bool=True, only_deprels: Optional[Iterable[str]]=None, allow_lemmas: Optional[Iterable[str]]=None` (wiersz 406)
- `candidate_word_id_for_rulerule: Dict[str, Any], candidate_idx: int, target_idx: int, word_ids: List[Any]` (wiersz 428)
- `check_extended_rule_filters*, rule: Dict[str, Any], candidate_idx: int, target_idx: int, word_ids: List[Any], head_ids: List[Any], children_by_head: Dict[int, List[int]], lemmas: List[Any], deprels: List[Any], upostags: List[Any], feats: List[Any], idx_by_word_id: Dict[Any, int], ignore_case: bool` (wiersz 437)
- `build_dynamic_relation_name*, relation_name: str, rule: Dict[str, Any], candidate_idx: int, target_idx: int, word_ids: List[Any], children_by_head: Dict[int, List[int]], lemmas: List[Any], deprels: List[Any], ignore_case: bool` (wiersz 566)
- `is_valid_collocatelemma: str, upos: str` (wiersz 610)
- `match_rule*, rule: Dict[str, Any], candidate_idx: int, target_idx: int, word_ids: List[Any], head_ids: List[Any], deprels: List[Any]` (wiersz 619)
- `compute_log_dicecooc_freq: int, target_global_freq: int, collocate_global_freq: int` (wiersz 646)
- `get_rule_setrule_id_key, values_list, ignore_case` (wiersz 654): Zwraca gotowy zbiór set() dla danej reguły, zapamiętując go przy pierwszym użyciu.
- `compute_word_profileresults: Iterable[Any], df, token_freq_dict: Dict[str, int], target_lemma: str, total_tokens: int, min_freq: int=2, max_rows_per_relation: Optional[int]=None, keep_examples: int=5, case_extractor: Callable[[str], str]=default_case_extractor, ignore_case: bool=True, expand_mwe: bool=False` (wiersz 666)
- `flatten_word_profileprofile: Dict[str, List[WordProfileRow]]` (wiersz 935)

## `topics`

### `korpusuj/topics/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/topics/engine.py`

BERTopic services for loading documents, chunking text, training or loading models, visualization and topics-over-time analysis.

- `class TopicEngine` (wiersz 27)
  - Metody publiczne: `load_dataself, use_chunking=True, max_words_per_chunk=250, use_lemmas=False`, `train_modelself, nr_topics=None, force_retrain=False, use_stopwords=True, diversity=0.2`, `get_topic_infoself`, `calculate_topics_over_timeself`, `visualize_dynamic_topicsself, topics_over_time, top_n_topics=15`, `visualize_topic_mapself`, `visualize_word_scoresself, top_n_topics=10`

## `ui`

### `korpusuj/ui/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/ui/autocomplete.py`

CQL autocomplete widgets and suggestion logic for the search interface.

- `class CQLAutocomplete` (wiersz 16)
  - Metody publiczne: `handle_keypressself, event=None`, `show_popupself, matches`, `hide_popupself`, `navigate_upself, event`, `navigate_downself, event`, `insert_selectionself, event`

### `korpusuj/ui/cards.py`

Reusable settings-card widgets for collocations, profiles, plots and other option panels.

- `class SettingsCard` (wiersz 18)
  - Metody publiczne: `pack_contentself`, `toggleself`, `update_themeself, theme`

### `korpusuj/ui/fiszki_tkinter.py`

- `save_contentfile_path, html_content, encoding` (wiersz 10): Saves the raw HTML content to preserve formatting.
- `extract_and_savewindow_like, file_path, encoding_used` (wiersz 20): Gets HTML from editor and saves it (parity with PySide2).
- `class EditorAPI` (wiersz 37): Bridge for JS ↔ Python calls.
  - Metody publiczne: `save_directself, html_content`, `trigger_saveself`
- `create_html_editorfile_path, file_content, choice, encoding_used` (wiersz 55): Creates the HTML editor in a pywebview window.
- `load_file_contentchoice` (wiersz 566): Loads file content and opens the HTML editor.

### `korpusuj/ui/main_window.py`

Main-window integration points for the desktop interface.

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/ui/plots.py`

Plotting utilities for the desktop interface.

- `get_plot_stack` (wiersz 9)

### `korpusuj/ui/query_builder.py`

Widgets and data helpers for constructing CQL queries in the desktop interface.

- `configure_query_builder_specsner_prefixes=None, ner_types=None` (wiersz 18): Configure module-local specs used by ConditionRow.
- `class RegexHelperWindow` (wiersz 34)
  - Metody publiczne: `insert_regexself, text`
- `class ConditionRow` (wiersz 97)
  - Metody publiczne: `setup_uiself`, `on_attr_changeself, selected_attr`, `open_regex_helperself, entry_widget`, `add_nested_ruleself`, `remove_nested_ruleself, row`, `get_query_stringself`
- `class GapBlock` (wiersz 390)
  - Metody publiczne: `get_query_stringself`
- `class MetaBlock` (wiersz 420)
  - Metody publiczne: `on_type_changeself, selected_type`, `get_query_stringself`
- `class QueryBuilderWindow` (wiersz 529)
  - Metody publiczne: `add_meta_blockself`, `add_token_blockself`, `add_gap_blockself`, `add_ruleself, card`, `remove_ruleself, card, row`, `remove_blockself, block`, `generate_and_insertself`

### `korpusuj/ui/results_view.py`

Result-view integration points for concordance presentation.

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/ui/semantic_network_viewer.py`

CustomTkinter and Matplotlib interface for semantic-network exploration.

- `class SemanticNetworkViewer` (wiersz 17): Klasa renderująca i zarządzająca oknem grafu sieci semantycznej.
  - Metody publiczne: `zoom_on_scrollself, event`, `reset_graphself`, `open_settingsself`, `generate_semantic_reportself`, `hit_test_coreself, event, pixel_threshold=25`, `render_graphself`, `hit_test_haloself, event, pixel_threshold=10`, `on_hoverself, event`, `on_clickself, event`, `execute_searchself, event=None`, `explore_nodeself, word, parent=None`, `update_graph_dataself, center_word, core_data, halo_data, parent=None`, `on_wsd_selectself, choice: str`

### `korpusuj/ui/settings_window.py`

Settings-window integration points for the desktop interface.

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/ui/tables.py`

- `class CustomTable` (wiersz 5)
  - Metody publiczne: `set_additional_eventself, event_func`, `set_header_colorsself, bg_color, text_color`, `set_header_fontself, font`, `set_row_colorsself, even_color, odd_color`, `set_text_colorsself, text_colors`, `set_fontself, font`, `set_dataself, data`, `set_rows_numberself, rows_per_page`, `set_selected_row_colorself, color`, `set_fulltext_dataself, fulltext_data`, `set_canvas_backgroundself, color`, `on_row_clickself, row_index`, `restore_row_colorself, row_index`, `set_text_anchorself, anchors`, `show_context_menuself, event, row_index`, `copy_selected_rowself`, `search_selected_rowself`, `update_header_textself`, `on_header_clickself, col_index`, `populate_tableself`, `update_scrollbar_visibilityself`, `add_rowself, *new_row`, `add_rowself, *new_row`, `update_table_sizeself, event=None`, `on_canvas_resizeself, event`, `on_mouse_enterself, event`, `on_mouse_leaveself, event`, `on_mouse_wheelself, event`, `on_shift_mouse_wheelself, event`, `on_canvas_enterself, event`, `on_canvas_leaveself, event`, `update_wraplengthself`, `on_frame_resizeself, event`

### `korpusuj/ui/tooltip.py`

Tooltip widgets used by Tkinter and CustomTkinter controls.

- `class ToolTip` (wiersz 11)
  - Metody publiczne: `enterself, event=None`, `leaveself, event=None`

## `utils`

### `korpusuj/utils/__init__.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/utils/serialization.py`

Brak publicznych klas lub funkcji na poziomie modułu.

### `korpusuj/utils/text.py`

Shared text-processing utilities used across the application.

Brak publicznych klas lub funkcji na poziomie modułu.
