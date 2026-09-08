from __future__ import annotations
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from korpusuj.corpus.merger import CorpusMergeError, merge_corpora

def write(path, name, text, date, layers=None):
    layers = layers or {"ner": True, "coreference": False}
    names = ["Oryginalna_nazwa_pliku","Treść","Data publikacji","tokens","lemmas","postags","full_postags","deprels","word_ids","sentence_ids","head_ids","start_ids","end_ids","ners","upostags","corefs","coref_mentions"]
    types = [pa.string(),pa.string(),pa.string()] + [pa.list_(pa.string())]*5 + [pa.list_(pa.int64())]*5 + [pa.list_(pa.string())]*2 + [pa.list_(pa.list_(pa.null())),pa.list_(pa.null())]
    meta = {"total_tokens":999,"base_tf":{"wrong":999},"orth_tf":{"wrong":999},"monthly_token_counts":{},"annotation_layers":layers}
    schema = pa.schema([pa.field(n,t) for n,t in zip(names,types)], metadata={b"korpus_meta":json.dumps(meta).encode()})
    data = [[name],[text],[date],[["Ala","ma"]],[["Ala","mieć"]],[["s","f"]],[["s","f"]],[["nsubj","root"]],[[1,2]],[[0,0]],[[2,0]],[[0,4]],[[3,6]],[ ["O","O"] ],[["NOUN","VERB"]],[[[],[]]],[[]]]
    pq.write_table(pa.Table.from_arrays(data, schema=schema), path)

def test_merge(tmp_path):
    a,b,o=tmp_path/"a.parquet",tmp_path/"b.parquet",tmp_path/"o.parquet"
    write(a,"a.docx","left","2024-01-02"); write(b,"b.docx","right","03-02-2025")
    result=merge_corpora([a,b],o)
    assert result.rows==2 and result.total_tokens==4
    meta=json.loads(pq.ParquetFile(o).schema_arrow.metadata[b"korpus_meta"])
    assert meta["total_tokens"]==4 and meta["base_tf"]=={"Ala":2,"mieć":2}
    assert meta["monthly_token_counts"]=={"2024":{"1":2},"2025":{"2":2}}

def test_duplicate_blocked(tmp_path):
    a,b=tmp_path/"a.parquet",tmp_path/"b.parquet"
    write(a,"a.docx","same","2024-01-01"); write(b,"b.docx","same","2024-01-02")
    with pytest.raises(CorpusMergeError, match="Duplikat treści"): merge_corpora([a,b],tmp_path/"o.parquet")

def test_layers_blocked(tmp_path):
    a,b=tmp_path/"a.parquet",tmp_path/"b.parquet"
    write(a,"a.docx","a","2024-01-01",{"ner":True,"coreference":False})
    write(b,"b.docx","b","2024-01-02",{"ner":False,"coreference":False})
    with pytest.raises(CorpusMergeError, match="annotation_layers"): merge_corpora([a,b],tmp_path/"o.parquet")
