import sqlite3
from korpusuj.index.postings import PostingList
from korpusuj.index.sqlite_index import SearchIndex

def make_index(path):
    con=sqlite3.connect(path)
    con.execute("CREATE TABLE terms(attr TEXT, value TEXT, df INTEGER, cf INTEGER, postings BLOB)")
    con.execute("INSERT INTO terms VALUES (?,?,?,?,?)",("ner","B-persName",1,1,PostingList.encode({7:[1]})))
    con.execute("INSERT INTO terms VALUES (?,?,?,?,?)",("ner","I-persName",1,1,PostingList.encode({7:[2]})))
    con.commit(); con.close()
    index=SearchIndex.__new__(SearchIndex)
    index.con=sqlite3.connect(path); index.con.row_factory=sqlite3.Row
    return index

def test_on_demand_ner_labels_are_token_aligned_and_cached(tmp_path):
    index=make_index(tmp_path/"ner.search")
    try:
        assert index.get_ner_labels(7,4)==["O","B-persName","I-persName","O"]
        index.con.execute("DELETE FROM terms"); index.con.commit()
        assert index.get_ner_labels(7,4)==["O","B-persName","I-persName","O"]
    finally:
        index.con.close()

def test_ordinary_document_loaders_do_not_call_ner_reconstruction():
    import inspect
    assert "get_ner_labels(" not in inspect.getsource(SearchIndex.get_doc)
    assert "get_ner_labels(" not in inspect.getsource(SearchIndex.get_docs_many)
