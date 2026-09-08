from pathlib import Path

def test_complete_sqlite_documents_include_highlight_arrays():
    root=Path(__file__).resolve().parents[2]
    text=(root/"korpusuj/index/sqlite_index.py").read_text(encoding="utf-8")
    expected='("deprels", "postags", "upostags", "full_postags", "ners", "corefs", "coref_mentions")'
    assert text.count(expected) >= 2

def test_gui_highlights_tolerate_missing_optional_arrays():
    root=Path(__file__).resolve().parents[2]
    text=(root/"engine.py").read_text(encoding="utf-8")
    assert 'ners = getattr(row_data, "ners", []) or []' in text
    assert '["O"] * (token_count - len(ners))' in text
    assert '[None] * (token_count - len(corefs))' in text
