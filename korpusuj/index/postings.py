# -*- coding: utf-8 -*-
from __future__ import annotations
from korpusuj.utils.serialization import _json_zlib_dumps,_json_zlib_loads
class PostingList:
    @staticmethod
    def encode(postings_by_doc):
        payload=[[int(d),sorted({int(p) for p in ps})] for d,ps in sorted(postings_by_doc.items()) if ps]
        return _json_zlib_dumps(payload)
    @staticmethod
    def decode(blob):
        raw=_json_zlib_loads(blob,[]) or []
        return {int(d):[int(p) for p in ps] for d,ps in raw}
