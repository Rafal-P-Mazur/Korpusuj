import json, sqlite3, zlib
from korpusuj.search.cursor import make_lazy_fulltext_ref_111, resolve_lazy_fulltext_ref_111

def enc(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False).encode("utf-8"))

def test_locator_resolves_real_full_context(tmp_path):
    path=tmp_path/"sample.search"
    con=sqlite3.connect(path)
    con.execute("CREATE TABLE docs(doc_id INTEGER PRIMARY KEY, metadata_json BLOB, text BLOB, tokens BLOB, lemmas BLOB, start_ids BLOB, end_ids BLOB, sentence_ids BLOB, deprels BLOB, postags BLOB, upostags BLOB, full_postags BLOB, corefs BLOB, coref_mentions BLOB)")
    text="Ala ma bardzo świeżą rybę dzisiaj."
    tokens=["Ala","ma","bardzo","świeżą","rybę","dzisiaj","."]
    starts=[0,4,7,14,21,26,33]; ends=[2,5,12,19,24,31,33]
    values=(1,enc({}),zlib.compress(text.encode()),enc(tokens),enc(["Ala","mieć","bardzo","świeży","ryba","dzisiaj","."]),enc(starts),enc(ends),enc([0]*7),enc([]),enc([]),enc([]),enc([]),enc([]),enc([]))
    con.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values); con.commit(); con.close()
    class Index: index_path=str(path)
    context=["bardzo ","świeżą rybę"," dzisiaj"]
    ref=make_lazy_fulltext_ref_111(Index(),1,3,5,1,1,3)
    resolved=resolve_lazy_fulltext_ref_111(ref,context)
    assert resolved[1]=="świeżą rybę"
    assert resolved[0] or resolved[2]
    assert context==["bardzo ","świeżą rybę"," dzisiaj"]
