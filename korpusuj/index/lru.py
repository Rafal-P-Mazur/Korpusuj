# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import OrderedDict
class LRUCache:
    def __init__(self,maxsize=256): self.maxsize=int(maxsize or 0); self.data=OrderedDict()
    def get(self,key,default=None):
        if key not in self.data: return default
        self.data.move_to_end(key); return self.data[key]
    def put(self,key,value):
        if self.maxsize<=0: return value
        if key in self.data: self.data.move_to_end(key)
        self.data[key]=value
        while len(self.data)>self.maxsize: self.data.popitem(last=False)
        return value
