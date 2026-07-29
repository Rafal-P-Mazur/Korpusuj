# -*- coding: utf-8 -*-
from __future__ import annotations
import array, struct, sys
from korpusuj.dependency.policy import DEPENDENCY_PARENT_MAGIC

class LazyChildrenLookup:
    """3f hotfix: parent-only children lookup z adaptacyjną materializacją.

    Proste dependent query korzystają z kilku lookupów i zostają lekkie.
    Złożone zapytania typu dependent={...} & dependent={...} mogą sprawdzać
    wiele tokenów w jednym dokumencie; po kilku lookupach budujemy pełny
    children_lookup raz dla dokumentu, żeby uniknąć kosztu O(tokens * checks).
    """
    __slots__ = ("parent_idx", "_children", "_lookup_count", "_materialize_after")

    def __init__(self, parent_idx, materialize_after=8):
        self.parent_idx = parent_idx
        self._children = {}
        self._lookup_count = 0
        self._materialize_after = int(materialize_after or 8)

    def __len__(self):
        return len(self.parent_idx)

    def _ensure_full(self):
        if not isinstance(self._children, list):
            children = [[] for _ in range(len(self.parent_idx))]
            for child_i, parent_i in enumerate(self.parent_idx):
                try:
                    p = int(parent_i)
                except Exception:
                    p = -1
                if 0 <= p < len(children):
                    children[p].append(child_i)
            self._children = children
        return self._children

    def __getitem__(self, idx):
        idx = int(idx)
        if isinstance(self._children, list):
            return self._children[idx]
        self._lookup_count += 1
        if self._lookup_count > self._materialize_after:
            return self._ensure_full()[idx]
        cached = self._children.get(idx)
        if cached is not None:
            return cached
        found = []
        for child_i, parent_i in enumerate(self.parent_idx):
            try:
                if int(parent_i) == idx:
                    found.append(child_i)
            except Exception:
                continue
        self._children[idx] = found
        return found

    def __iter__(self):
        return iter(self._ensure_full())

def _encode_parent_idx_int32(parent_idx):
    try:
        arr = array.array("i", (int(x) for x in parent_idx))
        if arr.itemsize != 4:
            data = struct.pack("<" + ("i" * len(arr)), *arr) if arr else b""
        else:
            if sys.byteorder != "little":
                arr.byteswap()
            data = arr.tobytes()
        return DEPENDENCY_PARENT_MAGIC + struct.pack("<I", len(arr)) + data
    except Exception:
        return None

def _decode_parent_idx_int32(payload):
    try:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            return None
        payload = bytes(payload)
        if not payload.startswith(DEPENDENCY_PARENT_MAGIC):
            return None
        off = len(DEPENDENCY_PARENT_MAGIC)
        if len(payload) < off + 4:
            return None
        n = struct.unpack("<I", payload[off:off + 4])[0]
        data = payload[off + 4:off + 4 + (n * 4)]
        if len(data) != n * 4:
            return None
        arr = array.array("i")
        arr.frombytes(data)
        if sys.byteorder != "little" and arr.itemsize == 4:
            arr.byteswap()
        if len(arr) != n:
            return None
        return arr
    except Exception:
        return None

def build_dependency_maps(sentence_ids, word_ids, head_ids):
    # Build efficient parent/child lookup tables.
    # Returns:
    #     parent_idx: list[int] -> parent index of each token (-1 if none)
    #     children_lookup: list[list[int]] -> children indices for each token

    num_tokens = len(word_ids)
    parent_idx = [-1] * num_tokens
    children_lookup = [[] for _ in range(num_tokens)]

    # Map (sentence, word_id) -> index
    parent_lookup = {(sentence_ids[i], word_ids[i]): i for i in range(num_tokens)}

    for i in range(num_tokens):
        key = (sentence_ids[i], head_ids[i])
        p = parent_lookup.get(key, -1)
        parent_idx[i] = p
        if p >= 0:
            children_lookup[p].append(i)

    return parent_idx, children_lookup
