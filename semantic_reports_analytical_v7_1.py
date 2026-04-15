from __future__ import annotations

import argparse
import io
import json
import logging
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from sense_inducer import SenseInducer
except Exception:  # pragma: no cover
    SenseInducer = None

LOGGER = logging.getLogger("semantic_reports")


# =========================================================
# IO i bundle artefaktów
# =========================================================

def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class ArtifactBundle:
    label: str
    source_path: str
    df_neighbors: pd.DataFrame
    vectors: Dict[str, np.ndarray]
    metadata: Dict
    index: Dict[str, List[Tuple[str, float, int]]]

    def resolve_key(self, lemma: str) -> Optional[str]:
        if not lemma:
            return None
        lemma = str(lemma).strip()
        for cand in (lemma, lemma.lower(), lemma.capitalize()):
            if cand in self.index or cand in self.vectors:
                return cand
        return None

    def neighbors_of(self, lemma: str, top_k: int = 80, min_similarity: float = 0.0) -> List[Tuple[str, float, int]]:
        key = self.resolve_key(lemma)
        if not key or key not in self.index:
            return []
        raw = self.index[key]
        if top_k <= 0:
            top_k = len(raw)
        out = []
        for n, sim, freq in raw[:top_k]:
            if float(sim) < min_similarity:
                continue
            out.append((str(n), float(sim), int(freq)))
        return out

    def lemma_freq(self, lemma: str) -> int:
        key = self.resolve_key(lemma)
        if not key or self.df_neighbors is None or self.df_neighbors.empty:
            return 0
        rows = self.df_neighbors[self.df_neighbors["lemma"] == key]
        if rows.empty:
            return 0
        try:
            return int(rows.iloc[0].get("lemma_freq", 0))
        except Exception:
            return 0

    def max_neighbors_for(self, lemma: str) -> int:
        key = self.resolve_key(lemma)
        if not key or key not in self.index:
            return 0
        return len(self.index[key])


def _build_index(df_neighbors: pd.DataFrame) -> Dict[str, List[Tuple[str, float, int]]]:
    idx: Dict[str, List[Tuple[str, float, int]]] = {}
    if df_neighbors is None or df_neighbors.empty:
        return idx
    df_neighbors = df_neighbors.sort_values(["lemma", "similarity"], ascending=[True, False]).copy()
    has_freq = "neighbor_freq" in df_neighbors.columns
    for lemma, group in df_neighbors.groupby("lemma"):
        idx[str(lemma)] = [
            (str(r["neighbor"]), float(r["similarity"]), int(r.get("neighbor_freq", 0) if has_freq else 0))
            for _, r in group.iterrows()
        ]
    return idx


def _infer_bundle_paths(base_path: Path) -> Dict[str, Optional[Path]]:
    base_no_suffix = base_path.with_suffix("") if base_path.suffix else base_path
    candidates = {
        "wektor": [
            base_path if base_path.suffix == ".wektor" else None,
            Path(str(base_no_suffix) + ".wektor"),
        ],
        "neighbors": [
            base_path if base_path.name.endswith(".neighbors.parquet") else None,
            Path(str(base_no_suffix) + ".semantic.fasttext.neighbors.parquet"),
            Path(str(base_no_suffix) + ".semantic.word2vec.neighbors.parquet"),
            Path(str(base_no_suffix) + ".semantic.neighbors.parquet"),
            Path(str(base_no_suffix) + ".neighbors.parquet"),
        ],
        "vectors": [
            base_path if base_path.name.endswith(".vectors.parquet") else None,
            Path(str(base_no_suffix) + ".semantic.fasttext.vectors.parquet"),
            Path(str(base_no_suffix) + ".semantic.word2vec.vectors.parquet"),
            Path(str(base_no_suffix) + ".semantic.vectors.parquet"),
            Path(str(base_no_suffix) + ".vectors.parquet"),
        ],
        "meta": [
            Path(str(base_no_suffix) + ".semantic.meta.json"),
            Path(str(base_no_suffix) + ".meta.json"),
            Path(str(base_no_suffix) + ".json"),
        ],
    }
    resolved: Dict[str, Optional[Path]] = {}
    for key, vals in candidates.items():
        resolved[key] = None
        for cand in vals:
            if cand is not None and cand.exists():
                resolved[key] = cand
                break
    return resolved


def load_artifact_bundle(path_like: str, label: Optional[str] = None) -> ArtifactBundle:
    base_path = Path(path_like)
    paths = _infer_bundle_paths(base_path)
    label = label or base_path.stem
    df_neighbors: Optional[pd.DataFrame] = None
    vectors_df: Optional[pd.DataFrame] = None
    metadata: Dict = {}

    if paths["wektor"] is not None:
        with zipfile.ZipFile(paths["wektor"], "r") as zf:
            neigh_name = next((n for n in zf.namelist() if n.endswith(".neighbors.parquet")), None)
            vect_name = next((n for n in zf.namelist() if n.endswith(".vectors.parquet")), None)
            meta_name = next((n for n in zf.namelist() if n.endswith(".json")), None)
            if neigh_name is None:
                raise FileNotFoundError("W archiwum .wektor nie znaleziono pliku .neighbors.parquet")
            with zf.open(neigh_name) as fh:
                df_neighbors = pd.read_parquet(io.BytesIO(fh.read()))
            if vect_name is not None:
                with zf.open(vect_name) as fh:
                    vectors_df = pd.read_parquet(io.BytesIO(fh.read()))
            if meta_name is not None:
                with zf.open(meta_name) as fh:
                    try:
                        metadata = json.loads(fh.read().decode("utf-8"))
                    except Exception:
                        metadata = {}
    else:
        if paths["neighbors"] is None:
            raise FileNotFoundError(f"Nie znaleziono artefaktów dla ścieżki: {path_like}")
        df_neighbors = pd.read_parquet(paths["neighbors"])
        if paths["vectors"] is not None:
            vectors_df = pd.read_parquet(paths["vectors"])
        if paths["meta"] is not None:
            try:
                metadata = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

    if df_neighbors is None or df_neighbors.empty:
        raise ValueError("Brak danych sąsiedztwa (.neighbors.parquet)")

    vectors: Dict[str, np.ndarray] = {}
    if vectors_df is not None and not vectors_df.empty:
        for _, row in vectors_df.iterrows():
            lemma = str(row["lemma"])
            vec = row["vector"]
            vectors[lemma] = vec.astype(np.float32) if isinstance(vec, np.ndarray) else np.asarray(vec,
                                                                                                   dtype=np.float32)

    return ArtifactBundle(
        label=label,
        source_path=str(base_path),
        df_neighbors=df_neighbors,
        vectors=vectors,
        metadata=metadata,
        index=_build_index(df_neighbors),
    )


# =========================================================
# Matematyka
# =========================================================

def cosine_similarity(u: Optional[np.ndarray], v: Optional[np.ndarray]) -> float:
    if u is None or v is None:
        return 0.0
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def normalized_centroid(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    centroid = np.mean(np.vstack(vectors), axis=0)
    return centroid / (float(np.linalg.norm(centroid)) + 1e-9)


def pairwise_mean_cos(vectors: List[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 1.0 if len(vectors) == 1 else 0.0
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_similarity(vectors[i], vectors[j]))
    return float(np.mean(sims)) if sims else 0.0


def percentile(values: List[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), p)) if values else 0.0


def minmax_scale_dict(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = list(values.values())
    mn, mx = float(min(vals)), float(max(vals))
    if mx - mn < 1e-9:
        return {k: 0.0 for k in values}
    return {k: float((v - mn) / (mx - mn)) for k, v in values.items()}


# =========================================================
# Konfiguracja
# =========================================================

@dataclass
class ReportConfigV7_1:
    lemma: str
    output_dir: str
    top_k_neighbors: int = 0
    min_similarity: float = 0.30
    top_n_core_words: int = 15
    top_n_distinctive_words: int = 15
    top_n_interpretive_words: int = 15
    members_table_size: int = 50
    tail_table_size: int = 24
    orphan_table_size: int = 60
    use_sense_inducer: bool = True
    export_csv: bool = True
    hubness_similarity_threshold: float = 0.40
    frame_edge_threshold: float = 0.42
    bridge_similarity_threshold: float = 0.45
    frame_assignment_min_similarity: float = 0.10
    core_quantile: float = 0.60
    max_plot_words: int = 120
    local_neighbor_window: int = 80


class AnalyticalSemanticReportBuilderV7_1:
    def __init__(self, bundle: ArtifactBundle, config: ReportConfigV7_1):
        self.bundle = bundle
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._globality_index: Optional[Dict[str, float]] = None

    # -----------------------------------------------------
    # Silnik analityczny
    # -----------------------------------------------------
    def build_globality_index(self) -> Dict[str, float]:
        if self._globality_index is not None:
            return self._globality_index
        counts: Dict[str, int] = {}
        for _, neighbors in self.bundle.index.items():
            for n_word, n_score, _ in neighbors:
                if float(n_score) >= self.config.hubness_similarity_threshold:
                    counts[n_word] = counts.get(n_word, 0) + 1
        if not counts:
            self._globality_index = {}
            return self._globality_index
        p50 = float(np.percentile(list(counts.values()), 50))
        max_count = float(max(counts.values())) if counts else 1.0
        out = {}
        for word in set(self.bundle.index.keys()).union(set(counts.keys())):
            c = counts.get(word, 0)
            if c <= p50:
                out[word] = 0.0
            else:
                num = math.log((c - p50) + 1)
                den = math.log((max_count - p50) + 1) if max_count > p50 else 1.0
                out[word] = float(min(1.0, num / den))
        self._globality_index = out
        return out

    def get_globality(self, lemma: str) -> float:
        key = self.bundle.resolve_key(lemma) or lemma
        return float(self.build_globality_index().get(key, 0.0))

    def collect_semantic_field(self, key: str) -> List[Dict]:
        top_k = self.config.top_k_neighbors if self.config.top_k_neighbors > 0 else self.bundle.max_neighbors_for(key)
        rows = []
        seen = set()
        for neighbor, sim, freq in self.bundle.neighbors_of(key, top_k=top_k,
                                                            min_similarity=self.config.min_similarity):
            nkey = self.bundle.resolve_key(neighbor)
            if not nkey or nkey == key or nkey in seen or nkey not in self.bundle.vectors:
                continue
            seen.add(nkey)
            rows.append({
                "lemma": nkey,
                "similarity_to_lemma": float(sim),
                "freq": int(freq),
                "globality": self.get_globality(nkey),
            })
        rows.sort(key=lambda x: (x["similarity_to_lemma"], x["freq"]), reverse=True)
        return rows

    def _build_local_graph(self, key: str, field_rows: List[Dict]) -> nx.Graph:
        words = [row["lemma"] for row in field_rows]
        local_set = set(words)
        G = nx.Graph()
        G.add_node(key, kind="root")
        for row in field_rows:
            G.add_node(row["lemma"], kind="word", freq=int(row["freq"]))
            G.add_edge(key, row["lemma"], weight=float(row["similarity_to_lemma"]), edge_type="root")
        for word in words:
            for n_word, sim, _ in self.bundle.neighbors_of(word, top_k=self.config.local_neighbor_window,
                                                           min_similarity=self.config.frame_edge_threshold):
                n_key = self.bundle.resolve_key(n_word)
                if not n_key or n_key == key or n_key not in local_set or n_key == word:
                    continue
                if G.has_edge(word, n_key):
                    G[word][n_key]["weight"] = max(float(G[word][n_key]["weight"]), float(sim))
                else:
                    G.add_edge(word, n_key, weight=float(sim), edge_type="local")
        return G

    def _fallback_frames(self, key: str, candidate_words: List[str]) -> List[Dict]:
        dummy_rows = [{"lemma": w, "similarity_to_lemma": 0.0, "freq": self.bundle.lemma_freq(w)} for w in
                      candidate_words]
        sub = self._build_local_graph(key, dummy_rows).subgraph(candidate_words).copy()
        weak_edges = [(u, v) for u, v, d in sub.edges(data=True) if
                      float(d.get("weight", 0.0)) < self.config.frame_edge_threshold]
        sub.remove_edges_from(weak_edges)
        if sub.number_of_nodes() == 0:
            return []
        if sub.number_of_edges() == 0:
            communities = [{w} for w in candidate_words]
        else:
            try:
                communities = list(nx.algorithms.community.greedy_modularity_communities(sub, weight="weight"))
            except Exception:
                communities = [set(comp) for comp in nx.connected_components(sub)]
        frames = []
        for i, comm in enumerate(communities, start=1):
            members = [w for w in sorted(comm) if w in self.bundle.vectors]
            if len(members) < 2:
                continue
            centroid = normalized_centroid([self.bundle.vectors[m] for m in members])
            if centroid is None:
                continue
            ranked = sorted(members, key=lambda w: cosine_similarity(self.bundle.vectors[w], centroid), reverse=True)
            frames.append({
                "id": str(i),
                "label": ", ".join(ranked[:3]),
                "type": "grafowa",
                "members": members,
                "centroid": centroid,
                "anchors": ranked[:4],
            })
        return frames

    def induce_frames(self, key: str, candidate_words: List[str]) -> List[Dict]:
        frames_raw = []
        if self.config.use_sense_inducer and SenseInducer is not None:
            try:
                frames_raw = SenseInducer.induce(key, self.bundle.vectors, self.bundle.index, debug=False) or []
            except Exception as exc:
                LOGGER.warning("SenseInducer nie powiódł się: %s", exc)
                frames_raw = []
        frames = []
        if frames_raw:
            for i, fr in enumerate(frames_raw, start=1):
                members = []
                seen = set()
                for m in fr.get("members", []) or []:
                    k = self.bundle.resolve_key(str(m))
                    if not k or k == key or k not in candidate_words or k in seen or k not in self.bundle.vectors:
                        continue
                    seen.add(k)
                    members.append(k)
                if len(members) < 2:
                    continue
                centroid = normalized_centroid([self.bundle.vectors[m] for m in members])
                if centroid is None:
                    continue
                anchors = [self.bundle.resolve_key(a) or a for a in (fr.get("anchors", []) or [])]
                anchors = [a for a in anchors if isinstance(a, str)] or members[:4]

                frame_type = str(fr.get("frame_type", fr.get("type", "semantic")))

                raw_label = str(fr.get("label") or "").strip()
                anchor_label = ", ".join(anchors[:3]) if anchors else f"Rama {i}"

                if not raw_label:
                    label = anchor_label
                else:
                    raw_tokens = {t.strip() for t in raw_label.split(",") if t.strip()}
                    anchor_tokens = {t.strip() for t in anchors[:3] if isinstance(t, str) and t.strip()}
                    overlap = len(raw_tokens & anchor_tokens)

                    bad_prefix = raw_label.lower().startswith(("rama", "profil", "sense"))

                    # Jeśli label w ogóle nie pokrywa się z anchorami, to go nie ufamy.
                    suspicious_label = (overlap == 0)

                    if bad_prefix or suspicious_label:
                        label = anchor_label
                    else:
                        label = raw_label

                frames.append({
                    "id": str(fr.get("frame_id", fr.get("sense_id", i))),
                    "label": label,
                    "type": frame_type,
                    "members": members,
                    "centroid": centroid,
                    "anchors": anchors[:4],
                })

        if not frames:
            frames = self._fallback_frames(key, candidate_words)
        assigned = set()
        for fr in frames:
            assigned.update(fr["members"])
        leftovers = [w for w in candidate_words if w not in assigned and w in self.bundle.vectors]
        for word in leftovers:
            wv = self.bundle.vectors[word]
            best_frame = None
            best_sim = -1.0
            for fr in frames:
                sim = cosine_similarity(wv, fr["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_frame = fr
            if best_frame is not None and best_sim >= self.config.frame_assignment_min_similarity:
                best_frame["members"].append(word)
        clean = []
        for i, fr in enumerate(frames, start=1):
            members = sorted(set([m for m in fr["members"] if m in self.bundle.vectors and m != key]))
            if len(members) < 2:
                continue
            centroid = normalized_centroid([self.bundle.vectors[m] for m in members])
            if centroid is None:
                continue
            ranked = sorted(members, key=lambda w: cosine_similarity(self.bundle.vectors[w], centroid), reverse=True)
            anchors = [a for a in fr.get("anchors", []) if isinstance(a, str)] or ranked[:4]
            label = str(fr.get("label") or ", ".join(anchors[:3]))
            clean.append({
                "id": str(fr.get("id", i)),
                "label": label,
                "type": str(fr.get("type", "semantyczna")),
                "members": members,
                "centroid": centroid,
                "anchors": anchors[:4],
            })
        return clean

    def compute_word_metrics(self, key: str, field_rows: List[Dict], frames: List[Dict],
                             local_graph: nx.Graph) -> pd.DataFrame:
        if not field_rows:
            return pd.DataFrame()
        field_words = [r["lemma"] for r in field_rows]
        field_centroid = normalized_centroid([self.bundle.vectors[w] for w in field_words if w in self.bundle.vectors])
        frame_by_word, frame_centroids = {}, {}
        for fr in frames:
            frame_centroids[str(fr["id"])] = fr["centroid"]
            for m in fr["members"]:
                frame_by_word[m] = str(fr["id"])
        degree_strength = {
            n: float(sum(float(local_graph[n][nbr].get("weight", 0.0)) for nbr in local_graph.neighbors(n)))
            for n in local_graph.nodes() if n != key
        }
        degree_strength_scaled = minmax_scale_dict(degree_strength)
        log_freq_scaled = minmax_scale_dict({r["lemma"]: math.log1p(max(0, int(r["freq"]))) for r in field_rows})
        sim_to_lemma_scaled = minmax_scale_dict({r["lemma"]: float(r["similarity_to_lemma"]) for r in field_rows})
        try:
            betweenness = nx.betweenness_centrality(local_graph, weight="weight")
        except Exception:
            betweenness = {}
        records = []
        for row in field_rows:
            word = row["lemma"]
            vec = self.bundle.vectors.get(word)
            frame_id = frame_by_word.get(word, "")

            # ---------------------------
            # MIARY RAMOWE (zostają)
            # ---------------------------
            frame_typicality = cosine_similarity(vec, frame_centroids[frame_id]) if frame_id in frame_centroids else (
                cosine_similarity(vec, field_centroid) if field_centroid is not None else 0.0
            )
            other_sims = [
                cosine_similarity(vec, centroid)
                for fid, centroid in frame_centroids.items()
                if fid != frame_id
            ]
            frame_distinctiveness = frame_typicality - (max(other_sims) if other_sims else 0.0)
            frame_salience = (
                    0.40 * frame_typicality
                    + 0.30 * frame_distinctiveness
                    + 0.15 * log_freq_scaled.get(word, 0.0)
                    + 0.10 * degree_strength_scaled.get(word, 0.0)
                    + 0.05 * sim_to_lemma_scaled.get(word, 0.0)
                    - 0.15 * float(row["globality"])
            )

            # ---------------------------
            # NOWE MIARY FIELD-LEVEL
            # ---------------------------
            field_typicality = float(cosine_similarity(vec, field_centroid)) if field_centroid is not None else 0.0

            # "Swoistość pola":
            # słowo jest tym bardziej swoiste dla pola,
            # im bardziej siedzi w centrum pola i im mniej jest globalnym hubem
            field_distinctiveness = field_typicality * (1.0 - float(row["globality"]))

            # "Nośność pola":
            # interpretacyjna nośność dla całego pola semantycznego,
            # a nie dla pojedynczej ramy
            field_salience = (
                    0.45 * field_typicality
                    + 0.20 * field_distinctiveness
                    + 0.15 * log_freq_scaled.get(word, 0.0)
                    + 0.10 * degree_strength_scaled.get(word, 0.0)
                    + 0.10 * sim_to_lemma_scaled.get(word, 0.0)
            )

            records.append({
                "lemma": word,
                "frame_id": frame_id,
                "similarity_to_lemma": float(row["similarity_to_lemma"]),
                "freq": int(row["freq"]),
                "globality": float(row["globality"]),
                "local_strength": float(degree_strength.get(word, 0.0)),
                "bridge_score": float(betweenness.get(word, 0.0)),

                # stare miary ramowe
                "typicality": float(frame_typicality),
                "distinctiveness": float(frame_distinctiveness),
                "salience": float(frame_salience),

                # nowe miary field-level
                "field_typicality": float(field_typicality),
                "field_distinctiveness": float(field_distinctiveness),
                "field_salience": float(field_salience),

                # alias diagnostyczny / kompatybilność
                "similarity_to_field_centroid": float(field_typicality),
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df

        core_flags = []
        for _, row in df.iterrows():
            if not row["frame_id"]:
                core_flags.append(False)
                continue
            vals = df[df["frame_id"] == row["frame_id"]]["typicality"].tolist()
            core_flags.append(float(row["typicality"]) >= percentile(vals, self.config.core_quantile * 100.0))

        df["is_core"] = core_flags
        df["is_periphery"] = ~df["is_core"]

        # UWAGA:
        # zostawiamy sortowanie po starych metrykach ramowych, żeby nie ruszać
        # logiki szczegółów ram i istniejących widoków.
        return df.sort_values(["salience", "typicality"], ascending=[False, False]).reset_index(drop=True)

    def compute_frame_metrics(self, key: str, frames: List[Dict], word_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        key_vec = self.bundle.vectors.get(key)
        frame_centroids = {str(fr["id"]): fr["centroid"] for fr in frames}
        for fr in frames:
            fid = str(fr["id"])
            members_df = word_df[word_df["frame_id"] == fid].copy()
            if members_df.empty:
                continue
            member_vectors = [self.bundle.vectors[m] for m in members_df["lemma"].tolist() if m in self.bundle.vectors]
            centroid = fr["centroid"]
            other_sims = [cosine_similarity(centroid, c) for oid, c in frame_centroids.items() if oid != fid]
            separation = 1.0 - (max(other_sims) if other_sims else 0.0)
            nearest_frame_id = None
            nearest_frame_sim = -1.0
            for oid, c in frame_centroids.items():
                if oid == fid:
                    continue
                s = cosine_similarity(centroid, c)
                if s > nearest_frame_sim:
                    nearest_frame_sim = s
                    nearest_frame_id = oid
            records.append({
                "frame_id": fid,
                "frame_label": fr["label"],
                "frame_type": fr.get("type", "semantyczna"),
                "size": int(len(members_df)),
                "core_size": int(members_df["is_core"].sum()),
                "periphery_size": int((~members_df["is_core"]).sum()),
                "coverage_share": float(len(members_df) / len(word_df)) if len(word_df) else 0.0,
                "cohesion_pairwise": float(pairwise_mean_cos(member_vectors)),
                "cohesion_centroid_mean": float(members_df["typicality"].mean()),
                "distinctiveness_mean": float(members_df["distinctiveness"].mean()),
                "salience_mean": float(members_df["salience"].mean()),
                "globality_mean": float(members_df["globality"].mean()),
                "similarity_centroid_to_lemma": float(
                    cosine_similarity(key_vec, centroid)) if key_vec is not None else 0.0,
                "separation_from_other_frames": float(separation),
                "frequency_sum": int(members_df["freq"].sum()),
                "frequency_mean": float(members_df["freq"].mean()),
                "anchors": fr.get("anchors", [])[:4],
                "nearest_frame_id": nearest_frame_id,
                "nearest_frame_similarity": nearest_frame_sim,
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(["coverage_share", "salience_mean", "cohesion_centroid_mean"],
                                ascending=[False, False, False]).reset_index(drop=True)
            df["frame_rank"] = range(1, len(df) + 1)
        return df

    def compute_global_overview(self, key: str, field_rows: List[Dict], local_graph: nx.Graph,
                                frame_df: pd.DataFrame) -> Dict:
        sims = [float(r["similarity_to_lemma"]) for r in field_rows]
        globalities = [float(r["globality"]) for r in field_rows]
        field_words = [r["lemma"] for r in field_rows]
        field_vectors = [self.bundle.vectors[w] for w in field_words if w in self.bundle.vectors]
        field_centroid = normalized_centroid(field_vectors)
        dispersion_vals = [
            1.0 - cosine_similarity(self.bundle.vectors[w], field_centroid)
            for w in field_words if field_centroid is not None and w in self.bundle.vectors
        ]
        return {
            "lemma": key,
            "lemma_freq": int(self.bundle.lemma_freq(key)),
            "available_neighbors_for_lemma": int(self.bundle.max_neighbors_for(key)),
            "selected_neighbors": int(len(field_rows)),
            "liczba_ram": int(len(frame_df)),
            "graph_nodes": int(local_graph.number_of_nodes()),
            "graph_edges": int(local_graph.number_of_edges()),
            "graph_density": float(nx.density(local_graph)) if local_graph.number_of_nodes() > 1 else 0.0,
            "field_similarity_mean": float(np.mean(sims)) if sims else 0.0,
            "field_similarity_median": float(np.median(sims)) if sims else 0.0,
            "field_similarity_p90": percentile(sims, 90.0) if sims else 0.0,
            "field_globality_mean": float(np.mean(globalities)) if globalities else 0.0,
            "field_dispersion_mean": float(np.mean(dispersion_vals)) if dispersion_vals else 0.0,
            "field_cohesion_pairwise": float(pairwise_mean_cos(field_vectors)) if field_vectors else 0.0,
            "frame_cohesion_weighted": float(np.average(frame_df["cohesion_centroid_mean"],
                                                        weights=frame_df["size"])) if not frame_df.empty else 0.0,
            "frame_separation_mean": float(
                frame_df["separation_from_other_frames"].mean()) if not frame_df.empty else 0.0,
            "neighbors_top_k": int(
                self.config.top_k_neighbors if self.config.top_k_neighbors > 0 else self.bundle.max_neighbors_for(key)),
            "min_similarity": float(self.config.min_similarity),
        }

    def compute_projection(self, key: str, word_df: pd.DataFrame, frame_df: pd.DataFrame, frames: List[Dict]) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        plot_word_df = word_df.copy().head(self.config.max_plot_words)
        vectors, labels = [], []
        if key in self.bundle.vectors:
            vectors.append(self.bundle.vectors[key])
            labels.append(("root", key))
        for _, row in plot_word_df.iterrows():
            vec = self.bundle.vectors.get(row["lemma"])
            if vec is None:
                continue
            vectors.append(vec)
            labels.append(("word", row["lemma"]))
        coords = PCA(n_components=2).fit_transform(np.vstack(vectors)) if len(vectors) >= 2 else np.zeros(
            (len(vectors), 2), dtype=np.float32)
        label_to_meta = plot_word_df.set_index("lemma").to_dict(orient="index")
        word_points = []
        for (kind, label), xy in zip(labels, coords):
            if kind == "root":
                word_points.append({
                    "kind": "root",
                    "lemma": label,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "frame_id": "",
                    "size_metric": 1.0,
                    "salience": 1.0,
                })
            else:
                meta = label_to_meta[label]
                word_points.append({
                    "kind": "word",
                    "lemma": label,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "frame_id": str(meta.get("frame_id", "")),
                    "size_metric": float(meta.get("salience", 0.0)),
                    "salience": float(meta.get("salience", 0.0)),
                    "freq": int(meta.get("freq", 0)),
                    "typicality": float(meta.get("typicality", 0.0)),
                    "distinctiveness": float(meta.get("distinctiveness", 0.0)),
                    "similarity_to_lemma": float(meta.get("similarity_to_lemma", 0.0)),
                })
        frame_vectors = [fr["centroid"] for fr in frames]
        frame_ids = [str(fr["id"]) for fr in frames]
        if len(frame_vectors) >= 2:
            coords_f = PCA(n_components=2).fit_transform(np.vstack(frame_vectors))
        elif len(frame_vectors) == 1:
            coords_f = np.zeros((1, 2), dtype=np.float32)
        else:
            coords_f = np.zeros((0, 2), dtype=np.float32)
        frame_meta = frame_df.set_index("frame_id").to_dict(orient="index") if not frame_df.empty else {}
        frame_points = []
        for fid, xy in zip(frame_ids, coords_f):
            meta = frame_meta.get(fid, {})
            frame_points.append({
                "frame_id": fid,
                "frame_label": str(meta.get("frame_label", fid)),
                "frame_rank": int(meta.get("frame_rank", 0)),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "size": int(meta.get("size", 0)),
                "cohesion": float(meta.get("cohesion_centroid_mean", 0.0)),
                "separation": float(meta.get("separation_from_other_frames", 0.0)),
            })
        return pd.DataFrame(word_points), pd.DataFrame(frame_points)

    def compute_frame_similarity(self, frames: List[Dict]) -> pd.DataFrame:
        rows = []
        for fr_a in frames:
            for fr_b in frames:
                rows.append({
                    "frame_id_a": str(fr_a["id"]),
                    "frame_label_a": fr_a["label"],
                    "frame_id_b": str(fr_b["id"]),
                    "frame_label_b": fr_b["label"],
                    "centroid_similarity": float(cosine_similarity(fr_a["centroid"], fr_b["centroid"])),
                })
        return pd.DataFrame(rows)

    def build_orphan_rows(self, word_df: pd.DataFrame) -> List[Dict]:
        if word_df.empty:
            return []
        orphans = word_df[word_df["frame_id"].fillna("") == ""].copy()
        if orphans.empty:
            return []
        return [
            {
                "word": r["lemma"],
                "freq": int(r["freq"]),
                "similarity_to_lemma": float(r["similarity_to_lemma"]),
                "globality": float(r["globality"]),
                "salience": float(r["salience"]),
            }
            for _, r in orphans.sort_values(["similarity_to_lemma", "salience"], ascending=[False, False]).head(
                self.config.orphan_table_size).iterrows()
        ]

    def compute_diagnostics(self, key: str, field_rows: List[Dict], word_df: pd.DataFrame, frames: List[Dict],
                            local_graph: nx.Graph) -> Dict:
        assigned_count = int((word_df["frame_id"].fillna("") != "").sum()) if not word_df.empty else 0
        orphan_count = int((word_df["frame_id"].fillna("") == "").sum()) if not word_df.empty else 0
        candidate_words = [r["lemma"] for r in field_rows]
        missing_vectors = [w for w in candidate_words if w not in self.bundle.vectors]
        return {
            "notes": [],
            "frames_total": len(frames),
            "candidate_neighbors_total": len(candidate_words),
            "assigned_neighbors_total": assigned_count,
            "orphans_total": orphan_count,
            "coverage_ratio": round((assigned_count / len(candidate_words)) if candidate_words else 0.0, 4),
            "root_has_vector": bool(key in self.bundle.vectors),
            "missing_vectors_total": len(missing_vectors),
            "graph_nodes": int(local_graph.number_of_nodes()),
            "graph_edges": int(local_graph.number_of_edges()),
        }

    # -----------------------------------------------------
    # Eksporty
    # -----------------------------------------------------
    def export_sidecars(self, payload: Dict, methodology: Dict, diagnostics: Dict) -> None:
        payload_json = {
            "lemma": payload.get("lemma"),
            "overview": payload.get("overview", {}),
            "methodology": methodology,
            "diagnostics": diagnostics,
            "word_df": payload["word_df"].to_dict(orient="records") if payload.get("word_df") is not None else [],
            "frame_df": payload["frame_df"].to_dict(orient="records") if payload.get("frame_df") is not None else [],
            "frame_similarity_df": payload["frame_similarity_df"].to_dict(orient="records") if payload.get(
                "frame_similarity_df") is not None else [],
            "words_coords_df": payload["words_coords_df"].to_dict(orient="records") if payload.get(
                "words_coords_df") is not None else [],
            "frames_coords_df": payload["frames_coords_df"].to_dict(orient="records") if payload.get(
                "frames_coords_df") is not None else [],
            "orphans_df": payload["orphans_df"].to_dict(orient="records") if payload.get(
                "orphans_df") is not None else [],
        }

        (self.output_dir / "report.payload.json").write_text(
            json.dumps(payload_json, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (self.output_dir / "metrics_lemma.json").write_text(
            json.dumps(payload["overview"], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (self.output_dir / "methodology.json").write_text(
            json.dumps(methodology, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (self.output_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        if self.config.export_csv:
            payload["word_df"].to_csv(self.output_dir / "semantic_field.csv", index=False, encoding="utf-8")
            payload["frame_df"].to_csv(self.output_dir / "frames.csv", index=False, encoding="utf-8")
            payload["word_df"].to_csv(self.output_dir / "frame_members.csv", index=False, encoding="utf-8")
            payload["frame_similarity_df"].to_csv(self.output_dir / "frame_similarity.csv", index=False,
                                                  encoding="utf-8")
            payload["words_coords_df"].to_csv(self.output_dir / "coordinates_words.csv", index=False, encoding="utf-8")
            payload["frames_coords_df"].to_csv(self.output_dir / "coordinates_frames.csv", index=False,
                                               encoding="utf-8")
            payload["edges_df"].to_csv(self.output_dir / "edges.csv", index=False, encoding="utf-8")
            if payload["orphans_df"] is not None and not payload["orphans_df"].empty:
                payload["orphans_df"].to_csv(self.output_dir / "periphery_orphans.csv", index=False, encoding="utf-8")

    def compute_reverse_field(self, target_lemma: str) -> pd.DataFrame:
        df = self.bundle.df_neighbors
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Kto ma target_lemma w swoim polu
        mask = (df["neighbor"] == target_lemma) & (df["similarity"] >= self.config.min_similarity)
        reverse_hits = df[mask].copy()

        if reverse_hits.empty:
            return pd.DataFrame()

        records = []
        v_target = self.bundle.vectors.get(target_lemma)
        if v_target is None:
            return pd.DataFrame()

        for _, row in reverse_hits.iterrows():
            x_lemma = str(row["lemma"])

            # Pobieramy sąsiadów X z indeksu (już tam są dzięki trainerowi)
            # To jest bardzo szybkie
            raw_neighbors = self.bundle.neighbors_of(
                x_lemma,
                top_k=self.config.top_k_neighbors,
                min_similarity=self.config.min_similarity
            )

            # Zbieramy wektory sąsiadów (z wyłączeniem samej lemy docelowej,
            # by nie zawyżać typowości własnym wektorem)
            x_vectors = [
                self.bundle.vectors[n]
                for n, _, _ in raw_neighbors
                if n in self.bundle.vectors and n != target_lemma
            ]

            if len(x_vectors) < 2:  # Potrzebujemy tła do porównania
                continue

            centroid_x = normalized_centroid(x_vectors)  # Używamy Twojej funkcji z v7_1
            if centroid_x is None:
                continue

            typicality = cosine_similarity(v_target, centroid_x)  #

            # Dynamiczny role_hint zamiast sztywnego 0.75?
            # Można to uzależnić od średniej typowości w polu X
            role_hint = "core" if typicality > 0.65 else "context"

            records.append({
                "parent_lemma": x_lemma,
                "parent_freq": int(row.get("lemma_freq", 0)),
                "similarity_to_parent": float(row["similarity"]),
                "typicality_in_field": float(typicality),
                "role_hint": role_hint,
            })

        res = pd.DataFrame(records)
        return res.sort_values("typicality_in_field", ascending=False) if not res.empty else res
    # -----------------------------------------------------
    # HTML / Plotly
    # -----------------------------------------------------
    def render_html(self, payload: Dict) -> str:
        template = dedent("""
        <!DOCTYPE html>
        <html lang="pl">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Analityczny raport semantyczny V7.2</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
          <style>
            :root {
              --bg: #f8fafc; --panel: #ffffff; --text: #0f172a; --muted: #64748b;
              --border: #e2e8f0; --accent: #2563eb; --radius: 14px;
            }
            * { box-sizing: border-box; }
            body { margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--text); }
            header { padding:18px 24px; background:#0f172a; color:#fff; display:flex; justify-content:space-between; align-items:center; gap:16px; position:sticky; top:0; z-index:10; }
            .meta { color:#cbd5e1; font-size:13px; }
            .btn { background:transparent; color:#fff; border:1px solid #475569; border-radius:8px; padding:8px 12px; cursor:pointer; font-weight:600; }
            .btn:hover { background:#1e293b; }
            .wrap { padding:20px; display:grid; gap:16px; }
            .cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:12px; }
            .card,.panel { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); box-shadow:0 1px 2px rgba(15,23,42,.05); }
            .card { padding:14px 16px; }
            .card .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
            .card .value { font-size:28px; font-weight:700; margin-top:8px; }
            .help { display:inline-flex; width:16px; height:16px; border-radius:999px; align-items:center; justify-content:center; background:#e2e8f0; color:#334155; font-size:11px; cursor:help; margin-left:6px; }
            .panel-head { padding:16px 18px; border-bottom:1px solid var(--border); }
            .panel-body { padding:14px 16px; }
            .grid-main { display:grid; grid-template-columns: 40% 60%; gap:16px; }
            .detail-grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:16px; }
            .two-col { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
            .plot-panel-body { display:flex; flex-direction:column; min-height:760px; }
            .plot-tab-panel { flex:1; min-height:620px; }
            .plot-tab-panel.active { display:flex; }
            .plot-container { width:100%; height:100%; min-height:620px; }
            #relations-heatmap { height:360px; }
            .chart { height:280px; }
            .table-wrap { max-height:420px; overflow:auto; border:1px solid var(--border); border-radius:10px; }
            table { width:100%; border-collapse:collapse; font-size:13px; }
            th, td { padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align:middle; }
            th { position:sticky; top:0; background:#f8fafc; z-index:2; font-size:12px; color:var(--muted); }
            .summary-table tbody tr { cursor:pointer; }
            .summary-table tbody tr:hover { background:#f8fbff; }
            .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; background:#dbeafe; color:#1d4ed8; }
            .section-note { font-size:12px; color:var(--muted); margin-top:4px; }
            .empty { min-height:240px; display:flex; align-items:center; justify-content:center; color:var(--muted); text-align:center; padding:24px; }
            .metrics { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px; margin-bottom:16px; }
            .metric { padding:12px; border:1px solid var(--border); border-radius:10px; background:#fcfdff; }
            .metric .k { font-size:12px; color:var(--muted); }
            .metric .v { font-size:20px; font-weight:700; margin-top:4px; }
            .metric .i { font-size:11px; color:#475569; margin-top:6px; }
            .tabs { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; position:relative; z-index:20; }
            .tab-btn { border:1px solid var(--border); background:#fff; border-radius:10px; padding:8px 12px; cursor:pointer; }
            .tab-btn.active { background:#eff6ff; border-color:#93c5fd; color:#1d4ed8; font-weight:600; }
            .tab-panel { display:none; }
            .tab-panel.active { display:block; }
            .info-box { padding:16px 18px; border:1px dashed var(--border); border-radius:12px; color:var(--muted); background:#fcfdff; }
            .modal { display:none; position:fixed; inset:0; background:rgba(15,23,42,.55); align-items:center; justify-content:center; padding:24px; z-index:50; }
            .modal-box { width:min(980px,100%); max-height:88vh; overflow:auto; background:#fff; border-radius:16px; padding:24px; }
            .close-x { float:right; font-size:26px; cursor:pointer; color:var(--muted); }
            .method-grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:16px; }
            .method-card { border:1px solid var(--border); border-radius:12px; padding:12px; background:#fcfdff; }
            .method-card h4 { margin:0 0 8px 0; font-size:14px; }
            .method-card p { margin:0; font-size:13px; color:#334155; line-height:1.5; }
            @media (max-width: 1200px) {
              .grid-main, .detail-grid, .two-col, .metrics, .method-grid { grid-template-columns:1fr; }
              .plot-panel-body { min-height:680px; }
              .plot-tab-panel { min-height:560px; }
              .plot-container { min-height:560px; }
            }
          </style>
        </head>
        <body>
          <header>
            <div>
              <div style="font-size: 21px; font-weight: 700;">Raport semantyczny V7.2</div>
              <div id="header-meta" class="meta"></div>
            </div>
            <button class="btn" id="method-btn">Metodologia i metryki</button>
          </header>

          <div class="wrap">
            <section class="cards" id="summary-cards"></section>
            <section class="cards" id="technical-cards"></section>

            <section class="panel">
              <div class="panel-head">
                <h2>Globalne rankingi słów</h2>
                <div class="section-note">
                Top 25 słów według miar liczonych względem centroidu całego pola semantycznego.
                </div>
              </div>
              <div class="panel-body">
                <div class="detail-grid">
                  <div class="table-wrap">
                    <table>
                      <thead><tr><th>Top Centralność pola</th><th>Rama</th><th>Wynik</th></tr></thead>
                      <tbody id="global-typ-tbody"></tbody>
                    </table>
                  </div>
                  <div class="table-wrap">
                    <table>
                      <thead><tr><th>Top Swoistość pola</th><th>Rama</th><th>Wynik</th></tr></thead>
                      <tbody id="global-dis-tbody"></tbody>
                    </table>
                  </div>
                  <div class="table-wrap">
                    <table>
                      <thead><tr><th>Top Nośność pola</th><th>Rama</th><th>Wynik</th></tr></thead>
                      <tbody id="global-sal-tbody"></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </section>

            <section class="grid-main">
              <div class="panel">
                <div class="panel-head">
                  <h2>Przestrzeń semantyczna (PCA)</h2>
                  <div class="section-note">Przełączaj zakładki, aby zobaczyć całe słownictwo w tle ram lub same centroidy ram.</div>
                </div>
                <div class="panel-body plot-panel-body">
                  <div class="tabs" data-tab-group="pca">
                    <button class="tab-btn active" data-tab-group="pca" data-target="pca-words">Mapa słów</button>
                    <button class="tab-btn" data-tab-group="pca" data-target="pca-frames">Mapa ram</button>
                  </div>
                  <div class="tab-panel plot-tab-panel active" id="pca-words" data-tab-group="pca">
                    <div id="words-map" class="plot-container"></div>
                  </div>
                  <div class="tab-panel plot-tab-panel" id="pca-frames" data-tab-group="pca">
                    <div id="frames-map" class="plot-container"></div>
                  </div>
                </div>
              </div>

              <div class="panel">
                <div class="panel-head">
                  <h2 id="detail-title">Rama semantyczna</h2>
                  <div id="detail-subtitle" class="section-note">Wybierz ramę z mapy lub tabeli poniżej.</div>
                </div>
                <div class="panel-body">
                  <div id="detail-empty" class="empty">Kliknij wybraną ramę, aby zobaczyć szczegóły.</div>
                  <div id="detail-content" style="display:none;">
                    <div class="metrics" id="frame-metrics"></div>
                    <div class="detail-grid">
                      <div class="panel"><div class="panel-head"><h3>Top Typowość</h3></div><div class="panel-body"><div id="core-chart" class="chart"></div></div></div>
                      <div class="panel"><div class="panel-head"><h3>Top Swoistość</h3></div><div class="panel-body"><div id="distinctive-chart" class="chart"></div></div></div>
                      <div class="panel"><div class="panel-head"><h3>Top Nośność</h3></div><div class="panel-body"><div id="interpretive-chart" class="chart"></div></div></div>
                    </div>

                    <div style="height:16px"></div>
                    <div class="tabs" data-tab-group="detail">
                      <button class="tab-btn active" data-tab-group="detail" data-target="detail-members">Tabela główna</button>
                      <button class="tab-btn" data-tab-group="detail" data-target="detail-tail">peryferie ramy</button>
                    </div>
                    <div class="tab-panel active" id="detail-members" data-tab-group="detail">
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Słowo</th>
                              <th>Freq</th>
                              <th>Typowość</th>
                              <th>Swoistość</th>
                              <th>Nośność</th>
                              <th>Siła lokalna</th>
                              <th>Ogólność</th>
                            </tr>
                          </thead>
                          <tbody id="members-tbody"></tbody>
                        </table>
                      </div>
                    </div>
                    <div class="tab-panel" id="detail-tail" data-tab-group="detail">
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Słowo</th>
                              <th>Typowość</th>
                              <th>Swoistość</th>
                              <th>Nośność</th>
                              <th>Komentarz diagnostyczny</th>
                            </tr>
                          </thead>
                          <tbody id="tail-tbody"></tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section class="panel">
              <div class="panel-head">
                <h2>Podsumowanie ram</h2>
              </div>
              <div class="panel-body">
                <div class="table-wrap">
                  <table class="summary-table">
                    <thead>
                      <tr>
                        <th>Rama (Rank)</th>
                        <th>Rozmiar</th>
                        <th>Rdzeń</th>
                        <th>peryferie</th>
                        <th title="Średnia wartość typowości wszystkich elementów należących do ramy. Stanowi wskaźnik wewnętrznej spójności i jednorodności semantycznej grupy.">Zwartość <span class="help">?</span></th>
                        <th title="Miara dystansu semantycznego między centroidem danej ramy a najbliższym sąsiadującym klastrem. Odzwierciedla stopień odrębności tematycznej.">Separacja <span class="help">?</span></th>
                        <th title="Średnia wartość nośności interpretacyjnej przypisanych elementów. Wskazuje na ogólny potencjał wyrazistości pojęciowej danej ramy.">Średnia nośność <span class="help">?</span></th>
                        <th>Najbliższa rama</th>
                      </tr>
                    </thead>
                    <tbody id="frames-summary-body"></tbody>
                  </table>
                </div>
              </div>
            </section>

            <section class="two-col">
              <div class="panel">
                <div class="panel-head">
                  <h2>Relacje między ramami</h2>
                </div>
                <div class="panel-body"><div id="relations-heatmap"></div></div>
              </div>
            
              <div class="panel" id="reverse-field-panel">
                <div class="panel-head">
                  <h2>Obecność w innych polach</h2>
                  <div class="section-note">
                    W jakich polach innych pojęć pojawia się słowo: <strong id="reverse-lemma-name"></strong>?
                  </div>
                </div>
                <div class="panel-body" id="reverse-field-body"></div>
              </div>
            
              <div class="panel" id="orphans-panel">
                <div class="panel-head">
                  <h2>peryferie pola semantycznego</h2>
                  <div class="section-note">Słowa z pola lemy, które nie zostały przypisane do żadnej ramy.</div>
                </div>
                <div class="panel-body" id="orphans-panel-body"></div>
              </div>
            </section>

          <div class="modal" id="modal">
            <div class="modal-box">
              <span class="close-x" id="close-modal">×</span>
              <h2>Metodologia i interpretacja miar</h2>
              <p style="line-height:1.6; color:#334155;">Raport wykorzystuje aparat grafowy (NetworkX) oraz algebrę liniową do wydobycia struktury semantycznej wokół lemy. Ramy są generowane i oceniane przy użyciu poniższych metryk analitycznych.</p>

              <div class="method-grid" id="method-grid">
                <div class="method-card"><h4>Nośność interpretacyjna ramy (Salience)</h4><p>Łączy w sobie typowość słowa dla ramy, jego swoistość, zlogarytmowaną frekwencję oraz siłę lokalną w grafie. Jest karana za wysoką ogólność (hubness). Słowa o wysokiej nośności najlepiej nadają się do nazwania i zinterpretowania ramy.</p></div>
                <div class="method-card"><h4>Typowość ramy (Typicality)</h4><p>Podobieństwo kosinusowe słowa do uśrednionego środka ramy (centroidu). Słowa z wysoką typowością leżą w samym rdzeniu przestrzennym danej grupy znaczeniowej.</p></div>
                <div class="method-card"><h4>Swoistość ramy (Distinctiveness)</h4><p>Różnica między typowością dla własnej ramy a podobieństwem do centroidu najbliższej innej ramy. Słowo o niskiej swoistości leży na pograniczu dwóch ram.</p></div>
                <div class="method-card"><h4>Centralność pola (Field typicality)</h4><p>Podobieństwo słowa do centroidu całego pola semantycznego danej lemy. Ta miara zasila globalny ranking field-level i wskazuje słowa najbardziej centralne dla całego pola, a nie tylko dla jednej ramy.</p></div>
                <div class="method-card"><h4>Swoistość pola (Field distinctiveness)</h4><p>Miara łącząca centralność w całym polu z niską ogólnością (globality). Premiuje słowa mocno reprezentujące pole badanej lemy, ale niebędące ogólnymi hubami.</p></div>
                <div class="method-card"><h4>Nośność pola (Field salience)</h4><p>Miara interpretacyjna dla całego pola semantycznego. Łączy centralność pola, swoistość pola, zlogarytmowaną frekwencję, siłę lokalną oraz podobieństwo do lemy centralnej. To ona zasila nowe globalne rankingi słów.</p></div>
              </div>

              <h3 style="margin-top:20px;">Parametry wykonania</h3>
              <pre id="method-pre" style="white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;padding:12px;border-radius:12px;"></pre>
              <h3 style="margin-top:20px;">Diagnostyka</h3>
              <pre id="diag-pre" style="white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;padding:12px;border-radius:12px;"></pre>
            </div>
          </div>

          <script>
            const DATA = __PAYLOAD_JSON__;
            let selectedFrameId = null;

            function fmt(x, digits = 3) {
              if (x === null || x === undefined || Number.isNaN(x)) return '—';
              return Number(x).toFixed(digits);
            }

            function tailComment(row) {
              if (row.globality > 0.6) return 'Słowo ogólne (hub), osłabia precyzję.';
              if (row.distinctiveness < 0.05) return 'Silne pogranicze z inną ramą.';
              if (row.typicality < 0.3) return 'Dalekie peryferie (niska typowość).';
              return 'Umiarkowane powiązanie z rdzeniem.';
            }

            const frameColors = ['#2563eb','#0f766e','#7c3aed','#dc2626','#ea580c','#0891b2', '#4d7c0f', '#be123c'];
            const getFrameColor = (rank) => rank ? frameColors[(rank - 1) % frameColors.length] || '#64748b' : '#cbd5e1';

            function buildCards() {
              const o = DATA.overview;
              
              // Główne karty podsumowujące
              const cards = [
                { 
                  label: 'Lema Centralna', value: o.lemma, 
                  tip: 'Główny wyraz będący osią analizy i punktem odniesienia, wokół którego zbudowano całą przestrzeń semantyczną.' 
                },
                { 
                  label: 'Liczba wydzielonych ram', value: o.liczba_ram, 
                  tip: 'Liczba zidentyfikowanych, odrębnych klastrów znaczeniowych. Wyższa liczba sugeruje silną wieloznaczność (polisemię) lemy lub jej występowanie w wielu bardzo różnych kontekstach.' 
                },
                { 
                  label: 'Wyselekcjonowani sąsiedzi', value: o.selected_neighbors, 
                  tip: 'Słowa włączone do ostatecznej analizy grafowej. Mniejsza liczba oznacza, że lema ma wysoce specyficzne otoczenie i niewiele słów zdołało przekroczyć wymagany próg podobieństwa.' 
                },
                { 
                  label: 'Średnie podobieństwo pola', value: fmt(o.field_similarity_mean), 
                  tip: 'Średnie podobieństwo kosinusowe sąsiadów do lemy centralnej. Wynik powyżej 0.6 oznacza bardzo silnie powiązane pole, a niższy niż 0.4 sugeruje luźniejsze skojarzenia.' 
                },
                { 
                  label: 'Średnia ogólność pola', value: fmt(o.field_globality_mean), 
                  tip: 'Średni poziom ogólności (hubness) słów w polu. Wysoki wynik (>0.5) oznacza obecność słów potocznych i pospolitych, podczas gdy niski wskazuje na pole wysoce specyficzne i niszowe.' 
                },
                { 
                  label: 'Średnia separacja ram', value: fmt(o.frame_separation_mean), 
                  tip: 'Średni dystans między centroidami wydzielonych ram. Wysoka separacja to wyraźne, nieprzenikające się znaczenia, a niska sygnalizuje płynne granice między kontekstami.' 
                }
              ];
              
              document.getElementById('summary-cards').innerHTML = cards.map(c => `
                <div class="card">
                  <div class="label" style="display:flex; align-items:center;">
                    ${c.label} ${c.tip ? `<span class="help" title="${c.tip}">?</span>` : ''}
                  </div>
                  <div class="value">${c.value}</div>
                </div>`).join('');
            
              // Karty techniczne
              const techCards = [
                { 
                  label: 'Frekwencja lemy', value: o.lemma_freq, 
                  tip: 'Całkowita liczba wystąpień lemy w zbadanym korpusie. Rzadkie lemy mogą generować mniej stabilne modele wektorowe, co wymaga ostrożniejszej interpretacji ram.' 
                },
                { 
                  label: 'Gęstość grafu', value: fmt(o.graph_density), 
                  tip: 'Stosunek istniejących krawędzi do maksymalnej ich możliwej liczby. Wysoka gęstość (>0.4) dowodzi, że słowa z pola silnie łączą się również ze sobą nawzajem, tworząc zwartą domenę tematyczną.' 
                },
                { 
                  label: 'Spójność pola (pairwise)', value: fmt(o.field_cohesion_pairwise), 
                  tip: 'Średnie podobieństwo kosinusowe między wszystkimi parami wektorów w przestrzeni. Wysoka spójność potwierdza, że zbiór jest silnie zogniskowany wokół wspólnego tematu.' 
                },
                { 
                  label: 'Ważona zwartość ram', value: fmt(o.frame_cohesion_weighted), 
                  tip: 'Średnia spójność wewnętrzna ram ważona ich rozmiarem. Wysoka wartość (>0.7) wskazuje na precyzyjne zgrupowanie i dużą jednorodność wyłonionych klastrów.' 
                },
                { 
                  label: 'Parametr Top-K', value: o.neighbors_top_k, 
                  tip: 'Zdefiniowany w konfiguracji analizy górny limit liczby pobieranych najbliższych sąsiadów.' 
                },
                { 
                  label: 'Minimalne podobieństwo', value: fmt(o.min_similarity, 2), 
                  tip: 'Próg podobieństwa wymagany do włączenia sąsiada w przestrzeń analizy. Wyższy próg generuje pole semantyczne o większej precyzji powiązań i węższym zakresie tematycznym.' 
                }
              ];
              
              document.getElementById('technical-cards').innerHTML = techCards.map(c => `
                <div class="card">
                  <div class="label" style="display:flex; align-items:center;">
                    ${c.label} ${c.tip ? `<span class="help" title="${c.tip}">?</span>` : ''}
                  </div>
                  <div class="value">${c.value}</div>
                </div>`).join('');
            }

            function renderGlobalRankings() {
              const words = [...DATA.word_df];
            
              const getRank = (fid) => {
                if (!fid) return '—';
                const fr = DATA.frame_df.find(f => f.frame_id === fid);
                return fr ? `R${fr.frame_rank}` : '—';
              };
            
              const getRankNum = (fid) => {
                if (!fid) return null;
                const fr = DATA.frame_df.find(f => f.frame_id === fid);
                return fr ? fr.frame_rank : null;
              };
            
              // NOWE: prawdziwe rankingi field-level
              const typWords = [...words]
                .sort((a, b) => (b.field_typicality ?? 0) - (a.field_typicality ?? 0))
                .slice(0, 25);
            
              const disWords = [...words]
                .sort((a, b) => (b.field_distinctiveness ?? 0) - (a.field_distinctiveness ?? 0))
                .slice(0, 25);
            
              const salWords = [...words]
                .sort((a, b) => (b.field_salience ?? 0) - (a.field_salience ?? 0))
                .slice(0, 25);
            
              const fillTable = (id, data, key) => {
                const tbody = document.getElementById(id);
                tbody.innerHTML = data.map(w => {
                  const rankNum = getRankNum(w.frame_id);
                  const bgColor = rankNum ? getFrameColor(rankNum) : '#cbd5e1';
                  const textColor = rankNum ? '#fff' : '#0f172a';
                  return `<tr>
                    <td><b>${w.lemma}</b></td>
                    <td><span class="badge" style="background:${bgColor}; color:${textColor}">${getRank(w.frame_id)}</span></td>
                    <td>${fmt(w[key])}</td>
                  </tr>`;
                }).join('');
              };
            
              fillTable('global-typ-tbody', typWords, 'field_typicality');
              fillTable('global-dis-tbody', disWords, 'field_distinctiveness');
              fillTable('global-sal-tbody', salWords, 'field_salience');
            }

            function renderMaps() {
              const framesCoords = DATA.frames_coords_df;
              const wordsCoords = DATA.words_coords_df;
              const mapLayout = {
                margin: { l:20, r:20, t:10, b:50 },
                xaxis: { showticklabels:false, showgrid:false, zeroline:false },
                yaxis: { showticklabels:false, showgrid:false, zeroline:false },
                paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', dragmode:'pan',
                hovermode: 'closest'
              };

              const traceFrames = {
                x: framesCoords.map(f => f.x),
                y: framesCoords.map(f => f.y),
                customdata: framesCoords.map(f => f.frame_id),
                text: framesCoords.map(f => `<b>${formatFrameDisplayName(f)}</b><br>Rank: ${f.frame_rank}<br>Liczba słów: ${f.size}<br>Zwartość: ${fmt(f.cohesion)}<br>Separacja: ${fmt(f.separation)}`),
                mode: 'markers', hoverinfo: 'text',
                marker: {
                  size: framesCoords.map(f => Math.max(18, Math.min(56, 12 + f.size * 1.6))),
                  color: framesCoords.map(f => getFrameColor(f.frame_rank)),
                  line: { color: 'white', width: 2 }, opacity: 0.95
                },
                name: 'Ramy'
              };
              Plotly.newPlot('frames-map', [traceFrames], mapLayout, {responsive:true, displaylogo:false});
              document.getElementById('frames-map').on('plotly_click', evt => {
                if (evt.points && evt.points[0] && evt.points[0].customdata) {
                  selectFrame(evt.points[0].customdata);
                  document.getElementById('detail-title').scrollIntoView({ behavior: 'smooth' });
                }
              });

              const wordsTraces = [];
              const rootCoords = wordsCoords.find(x => x.kind === 'root');
              const rankMap = {};
              DATA.frame_df.forEach(f => { rankMap[f.frame_id] = f.frame_rank; });
              const groupedWords = {};
              wordsCoords.filter(w => w.kind === 'word').forEach(w => {
                const rank = rankMap[w.frame_id] || 999;
                if (!groupedWords[rank]) groupedWords[rank] = { x: [], y: [], text: [], customdata: [], rank: rank, size: [] };
                groupedWords[rank].x.push(w.x);
                groupedWords[rank].y.push(w.y);
                groupedWords[rank].customdata.push(w.frame_id);
                groupedWords[rank].size.push(Math.max(8, Math.min(18, 8 + (w.size_metric || 0) * 10)));
                groupedWords[rank].text.push(`<b>${w.lemma}</b><br>Rama: ${rank === 999 ? 'Brak' : rank}<br>Nośność: ${fmt(w.salience)}<br>Typowość: ${fmt(w.typicality || 0)}<br>Swoistość: ${fmt(w.distinctiveness || 0)}<br>Freq: ${w.freq}`);
              });
              Object.values(groupedWords).sort((a,b) => a.rank - b.rank).forEach(group => {
                wordsTraces.push({
                  x: group.x, y: group.y, text: group.text, customdata: group.customdata,
                  mode: 'markers', hoverinfo: 'text',
                  marker: { size: group.size, color: group.rank === 999 ? '#94a3b8' : getFrameColor(group.rank), opacity: 0.78 },
                  name: group.rank === 999 ? 'peryferie' : `Rama ${group.rank}`
                });
              });
              if (rootCoords) {
                wordsTraces.push({
                  x: [rootCoords.x], y: [rootCoords.y],
                  text: [`<b>${rootCoords.lemma}</b><br>Lema centralna`],
                  hoverinfo: 'text', mode: 'markers',
                  marker: { symbol: 'star', size: 18, color: '#0f172a', line: { color: 'white', width: 2 } },
                  name: 'Lema'
                });
              }
              Plotly.newPlot('words-map', wordsTraces, mapLayout, {responsive:true, displaylogo:false});
              document.getElementById('words-map').on('plotly_click', evt => {
                if (evt.points && evt.points[0] && evt.points[0].customdata) {
                  selectFrame(evt.points[0].customdata);
                  document.getElementById('detail-title').scrollIntoView({ behavior: 'smooth' });
                }
              });
            }
            function formatFrameDisplayName(frame) {
              if (!frame) return '—';
            
              const label = String(frame.frame_label || '').trim();
              const type = String(frame.frame_type || 'semantic').toLowerCase();
            
              if (type === 'contextual') {
                return `Rama kontekstowa: ${label}`;
              }
            
              return `Rama semantyczna: ${label}`;
            }


            function renderFramesSummaryTable() {
              const tbody = document.getElementById('frames-summary-body');
              tbody.innerHTML = '';
              DATA.frame_df.forEach(f => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                  <td>
                      <span class="badge">R${f.frame_rank}</span>
                      <b>${formatFrameDisplayName(f)}</b>
                    </td>
                  <td>${f.size}</td>
                  <td>${f.core_size}</td>
                  <td>${f.periphery_size}</td>
                  <td>${fmt(f.cohesion_centroid_mean)}</td>
                  <td>${fmt(f.separation_from_other_frames)}</td>
                  <td>${fmt(f.salience_mean)}</td>
                  <td>${f.nearest_frame_id ? f.nearest_frame_id : '—'} (${fmt(f.nearest_frame_similarity)})</td>`;
                tr.addEventListener('click', () => {
                  selectFrame(f.frame_id);
                  document.getElementById('detail-title').scrollIntoView({ behavior: 'smooth' });
                });
                tbody.appendChild(tr);
              });
            }

            function renderRelations() {
              const ids = [...new Set(DATA.frame_similarity_df.map(x => x.frame_label_a))];
              if (ids.length === 0) return;
              const matrix = ids.map(id_a => ids.map(id_b => {
                const match = DATA.frame_similarity_df.find(x => x.frame_label_a === id_a && x.frame_label_b === id_b);
                return match ? match.centroid_similarity : 0;
              }));
              Plotly.newPlot('relations-heatmap', [{
                z: matrix, x: ids, y: ids, type: 'heatmap', colorscale: 'Blues', zmin: 0, zmax: 1,
              }], { margin: { l:140, r:20, t:10, b:120 }, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)' }, {responsive:true, displaylogo:false});
            }

            function renderBarChart(targetId, words, metricKey, color, title) {
              const labels = words.map(x => x.lemma).slice().reverse();
              const values = words.map(x => x[metricKey]).slice().reverse();
              Plotly.react(targetId, [{ x: values, y: labels, type: 'bar', orientation: 'h', marker: { color } }], {
                margin: { l: 100, r: 20, t: 20, b: 40 }, xaxis: { title },
                paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'
              }, {responsive:true, displaylogo:false});
            }

            function selectFrame(frameId) {
              if (!frameId) return;
              selectedFrameId = frameId;
              const frame = DATA.frame_df.find(f => f.frame_id === frameId);
              if (!frame) return;
              const words = DATA.word_df.filter(w => w.frame_id === frameId);
              document.getElementById('detail-empty').style.display = 'none';
              document.getElementById('detail-content').style.display = 'block';
              document.getElementById('detail-title').textContent = formatFrameDisplayName(frame);
              document.getElementById('detail-subtitle').textContent = `Rank: ${frame.frame_rank} · Anchory: ${(frame.anchors || []).join(', ')}`;
              const mHtml = [
                `<div class="metric"><div class="k">Rozmiar ramy</div><div class="v">${frame.size}</div><div class="i">Rdzeń: ${frame.core_size} | peryferie: ${frame.periphery_size}</div></div>`,
                `<div class="metric"><div class="k">Zwartość ramy</div><div class="v">${fmt(frame.cohesion_centroid_mean)}</div><div class="i">Średnia typowość wektora</div></div>`,
                `<div class="metric"><div class="k">Nośność ramy</div><div class="v">${fmt(frame.salience_mean)}</div><div class="i">Średnia waga dla ramy</div></div>`,
              ].join('');
              document.getElementById('frame-metrics').innerHTML = mHtml;
              const coreWords = [...words].sort((a,b) => b.typicality - a.typicality).slice(0, 10);
              const distWords = [...words].sort((a,b) => b.distinctiveness - a.distinctiveness).slice(0, 10);
              const salWords = [...words].sort((a,b) => b.salience - a.salience).slice(0, 10);
              const color = getFrameColor(frame.frame_rank);
              renderBarChart('core-chart', coreWords, 'typicality', color, 'Typowość');
              renderBarChart('distinctive-chart', distWords, 'distinctiveness', color, 'Swoistość');
              renderBarChart('interpretive-chart', salWords, 'salience', color, 'Nośność (Salience)');
              const tbody = document.getElementById('members-tbody');
              tbody.innerHTML = '';
              // Filtrujemy tylko Rdzeń i dodajemy ładny badge
              words.filter(w => !w.is_periphery).sort((a,b) => b.salience - a.salience).forEach(row => {
                const tr = document.createElement('tr');
                const badge = '<span style="font-size:10px; color:#059669; background:#d1fae5; padding:2px 6px; border-radius:4px; margin-left:6px;">Rdzeń</span>';
                tr.innerHTML = `
                  <td><b>${row.lemma}</b> ${badge}</td>
                  <td>${row.freq}</td>
                  <td>${fmt(row.typicality)}</td>
                  <td>${fmt(row.distinctiveness)}</td>
                  <td><b>${fmt(row.salience)}</b></td>
                  <td>${fmt(row.local_strength)}</td>
                  <td>${fmt(row.globality)}</td>`;
                tbody.appendChild(tr);
              });
              const tailBody = document.getElementById('tail-tbody');
              tailBody.innerHTML = '';
              const peripheryWords = words.filter(w => w.is_periphery).sort((a,b) => b.typicality - a.typicality);
              if (peripheryWords.length === 0) {
                tailBody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#64748b;padding:16px;">Rama jest bardzo spójna, wszystkie słowa weszły do rdzenia.</td></tr>';
              } else {
                peripheryWords.forEach(row => {
                  const tr = document.createElement('tr');
                  tr.innerHTML = `
                    <td><b>${row.lemma}</b></td>
                    <td>${fmt(row.typicality)}</td>
                    <td>${fmt(row.distinctiveness)}</td>
                    <td>${fmt(row.salience)}</td>
                    <td>${tailComment(row)}</td>`;
                  tailBody.appendChild(tr);
                });
              }
            }

            function renderOrphansPanel() {
              const panel = document.getElementById('orphans-panel');
              const panelBody = document.getElementById('orphans-panel-body');
              const orphans = DATA.orphans_df || [];
            
              // Jeśli nie ma orphanów, ukryj cały panel
              if (!orphans.length) {
                if (panel) panel.style.display = 'none';
                return;
              }
            
              // Jeśli są orphany, upewnij się, że panel jest widoczny
              if (panel) panel.style.display = '';
            
              panelBody.innerHTML = `
                <div class="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Słowo</th>
                        <th title="Częstość absolutna występowania słowa w analizowanym korpusie.">Freq <span class="help">?</span></th>
                        <th title="Miara bliskości wektora słowa względem centroidu ramy. Wyższa wartość wskazuje na silniejszą przynależność do rdzenia semantycznego ramy.">Typowość <span class="help">?</span></th>
                        <th title="Stopień unikalności słowa dla danej ramy (różnica między typowością a podobieństwem do najbliższej sąsiedniej ramy). Wyższa wartość oznacza mniejszą wieloznaczność.">Swoistość <span class="help">?</span></th>
                        <th title="Złożona wskaźnik uwzględniający m.in. typowość, swoistość i siłę lokalną. Identyfikuje słowa o najwyższym potencjale reprezentatywnym dla danej ramy.">Nośność <span class="help">?</span></th>
                        <th title="Suma wag krawędzi łączących dany węzeł (słowo) z pozostałymi elementami w wyodrębnionej podsieci grafu.">Siła lokalna <span class="help">?</span></th>
                        <th title="Znormalizowany wskaźnik (0-1) określający stopień wszechobecności słowa w globalnej przestrzeni korpusu. Wysokie wartości wskazują na słowa o wysokiej ogólności (tzw. węzły typu hub).">Ogólność <span class="help">?</span></th>
                      </tr>
                    </thead>
                    <tbody id="orphans-tbody"></tbody>
                  </table>
                </div>`;
            
              const tbody = document.getElementById('orphans-tbody');
              tbody.innerHTML = orphans.map(row => `
                <tr>
                  <td><b>${row.word}</b></td>
                  <td>${row.freq}</td>
                  <td>${fmt(row.similarity_to_lemma)}</td>
                  <td>${fmt(row.salience)}</td>
                  <td>${fmt(row.globality)}</td>
                </tr>`).join('');
            }

            function renderReverseField() {
              const container = document.getElementById('reverse-field-body');
              const data = DATA.reverse_field_df || [];
            
              // Ustawiamy nazwę lemy w nagłówku panelu
              const lemmaHeader = document.getElementById('reverse-lemma-name');
              if (lemmaHeader) lemmaHeader.textContent = DATA.overview.lemma;
            
              if (data.length === 0) {
                container.innerHTML = `
                  <div class="info-box">
                    Lema nie została znaleziona jako istotny element w polach innych pojęć.
                  </div>`;
                return;
              }
            
              container.innerHTML = `
                <div class="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Pojęcie nadrzędne</th>
                        <th title="Bliskość lemy względem centrum znaczeniowego danego pojęcia.">Typowość w polu</th>
                        <th>Podobieństwo</th>
                        <th>Rola</th>
                      </tr>
                    </thead>
                    <tbody id="reverse-tbody"></tbody>
                  </table>
                </div>`;
            
              const tbody = document.getElementById('reverse-tbody');
              tbody.innerHTML = data.map(row => {
                const isCore = row.role_hint === 'core';
                const badgeStyle = isCore 
                  ? 'background:#d1fae5; color:#065f46; border: 1px solid #a7f3d0;' 
                  : 'background:#f1f5f9; color:#475569; border: 1px solid #e2e8f0;';
                
                return `
                  <tr>
                    <td>
                      <b>${row.parent_lemma}</b> 
                      <span style="color:var(--muted); font-size:11px; margin-left:4px;">(freq: ${row.parent_freq})</span>
                    </td>
                    <td>
                      <div style="font-weight:700; color:var(--accent); font-size:14px;">${fmt(row.typicality_in_field)}</div>
                    </td>
                    <td>${fmt(row.similarity_to_parent)}</td>
                    <td>
                      <span class="badge" style="${badgeStyle} border-radius:4px; text-transform:uppercase; font-size:10px;">
                        ${row.role_hint}
                      </span>
                    </td>
                  </tr>`;
              }).join('');
            }

            function initTabsAndModals() {
              // Nasłuch na body - zadziała zawsze, niweluje błędy renderowania
              document.body.addEventListener('click', (e) => {
                const btn = e.target.closest('.tab-btn');
                if (!btn) return; // Jeśli kliknięto coś innego, ignoruj
            
                const target = btn.getAttribute('data-target');
                if (!target) return;
            
                // Pobieramy grupę zakładek, do której należy przycisk (np. "detail" albo "pca")
                const group = btn.getAttribute('data-tab-group');
                if (!group) return;
            
                // Kasujemy klasę active tylko dla elementów posiadających tę samą grupę!
                document.querySelectorAll(`.tab-btn[data-tab-group="${group}"]`).forEach(x => x.classList.remove('active'));
                document.querySelectorAll(`.tab-panel[data-tab-group="${group}"]`).forEach(p => p.classList.remove('active'));
            
                // Odpalamy klikniętą zakładkę
                btn.classList.add('active');
                const targetPanel = document.getElementById(target);
                if (targetPanel) {
                    targetPanel.classList.add('active');
                }
                setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
              });
            
              // Modale
              document.getElementById('method-btn').addEventListener('click', () => { document.getElementById('modal').style.display = 'flex'; });
              document.getElementById('close-modal').addEventListener('click', () => { document.getElementById('modal').style.display = 'none'; });
              document.getElementById('modal').addEventListener('click', e => { if (e.target.id === 'modal') document.getElementById('modal').style.display = 'none'; });
              if (document.getElementById('method-pre')) document.getElementById('method-pre').textContent = JSON.stringify(DATA.methodology, null, 2);
              if (document.getElementById('diag-pre')) document.getElementById('diag-pre').textContent = JSON.stringify(DATA.diagnostics, null, 2);
            }
            document.addEventListener('DOMContentLoaded', () => {
              document.getElementById('header-meta').textContent = `Lema: ${DATA.overview.lemma} · Ramy: ${DATA.overview.liczba_ram}`;
              buildCards();
              renderGlobalRankings();
              renderMaps();
              renderFramesSummaryTable();
              renderRelations();
              renderOrphansPanel();
              renderReverseField();
              initTabsAndModals();
              if (DATA.frame_df.length) selectFrame(DATA.frame_df[0].frame_id);
              setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
            });
          </script>
        </body>
        </html>
        """)

        payload_json = {
            "overview": payload["overview"],
            "word_df": payload["word_df"].to_dict(orient="records"),
            "frame_df": payload["frame_df"].to_dict(orient="records"),
            "frame_similarity_df": payload["frame_similarity_df"].to_dict(orient="records"),
            "words_coords_df": payload["words_coords_df"].to_dict(orient="records"),
            "frames_coords_df": payload["frames_coords_df"].to_dict(orient="records"),
            "orphans_df": payload["orphans_df"].to_dict(orient="records") if payload["orphans_df"] is not None else [],
            "methodology": payload["methodology"],
            "diagnostics": payload["diagnostics"],
            "reverse_field_df": payload["reverse_field_df"].to_dict(orient="records") if not payload["reverse_field_df"].empty else [],
        }
        return template.replace("__PAYLOAD_JSON__", json.dumps(payload_json, ensure_ascii=False))

    # -----------------------------------------------------
    # Build
    # -----------------------------------------------------
    def build(self) -> Dict[str, object]:
        key = self.bundle.resolve_key(self.config.lemma)
        if not key:
            raise KeyError(f"Nie znaleziono lemy w indeksie/wektorach: {self.config.lemma}")
        if key not in self.bundle.vectors:
            raise KeyError(f"Lema '{key}' nie ma wektora i nie może zostać użyta do raportu.")

        field_rows = self.collect_semantic_field(key)
        if len(field_rows) < 2:
            raise ValueError(f"Brak sąsiadów spełniających warunki dla lemy: {key}")
        candidate_words = [row["lemma"] for row in field_rows]
        local_graph = self._build_local_graph(key, field_rows)
        frames = self.induce_frames(key, candidate_words)
        if not frames:
            raise ValueError("Nie udało się wygenerować ram semantycznych dla wskazanej lemy.")

        word_df = self.compute_word_metrics(key, field_rows, frames, local_graph)
        frame_df = self.compute_frame_metrics(key, frames, word_df)
        frame_similarity_df = self.compute_frame_similarity(frames)
        words_coords_df, frames_coords_df = self.compute_projection(key, word_df, frame_df, frames)
        edges_df = pd.DataFrame([
            {"source": u, "target": v, "weight": float(d.get("weight", 0.0)), "edge_type": d.get("edge_type", "")}
            for u, v, d in local_graph.edges(data=True)
        ])
        overview = self.compute_global_overview(key, field_rows, local_graph, frame_df)
        diagnostics = self.compute_diagnostics(key, field_rows, word_df, frames, local_graph)
        orphans_rows = self.build_orphan_rows(word_df)
        orphans_df = pd.DataFrame(orphans_rows) if orphans_rows else pd.DataFrame()
        reverse_field_df = self.compute_reverse_field(key)

        methodology = {
            "source_path": self.bundle.source_path,
            "bundle_label": self.bundle.label,
            "lemma": key,
            "neighbors_top_k": int(
                self.config.top_k_neighbors if self.config.top_k_neighbors > 0 else self.bundle.max_neighbors_for(key)),
            "min_similarity": float(self.config.min_similarity),
            "frame_source": "SenseInducer" if (
                        self.config.use_sense_inducer and SenseInducer is not None) else "fallback_greedy_modularity",
            "use_sense_inducer": bool(self.config.use_sense_inducer and SenseInducer is not None),
            "frame_edge_threshold": float(self.config.frame_edge_threshold),
            "bridge_similarity_threshold": float(self.config.bridge_similarity_threshold),
            "hubness_similarity_threshold": float(self.config.hubness_similarity_threshold),
            "frame_assignment_min_similarity": float(self.config.frame_assignment_min_similarity),
            "core_quantile": float(self.config.core_quantile),
            "max_plot_words": int(self.config.max_plot_words),
            "local_neighbor_window": int(self.config.local_neighbor_window),
            "typicality": "cos(word, centroid_ramy)",
            "distinctiveness": "typicality - max cos(word, centroid_innej_ramy)",
            "salience": "0.40*typicality + 0.30*distinctiveness + 0.15*log(freq) + 0.10*local_strength + 0.05*sim_to_lemma - 0.15*globality",
            "projection_2d": "PCA na wektorach słów / centroidach ram",
        }

        payload = {
            "lemma": key,
            "overview": overview,
            "methodology": methodology,
            "diagnostics": diagnostics,
            "word_df": word_df,
            "frame_df": frame_df,
            "frame_similarity_df": frame_similarity_df,
            "words_coords_df": words_coords_df,
            "frames_coords_df": frames_coords_df,
            "edges_df": edges_df,
            "orphans_df": orphans_df,
            "reverse_field_df": reverse_field_df,
        }

        self.export_sidecars(payload, methodology, diagnostics)
        html_text = self.render_html(payload)
        report_path = self.output_dir / "report.html"
        report_path.write_text(html_text, encoding="utf-8")
        LOGGER.info("Raport V7.2 wygenerowany: %s", report_path)
        return {
            "lemma": key,
            "output_dir": str(self.output_dir),
            "report_path": str(report_path.resolve()),
            "diagnostics": diagnostics,
        }


# =========================================================
# CLI
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generuje hybrydowy raport semantyczny V7.2 (logika V4 + dopracowane UI).")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--lemma", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k-neighbors", type=int, default=0)
    p.add_argument("--min-similarity", type=float, default=0.30)
    p.add_argument("--top-core", type=int, default=15, dest="top_n_core_words")
    p.add_argument("--top-distinctive", type=int, default=15, dest="top_n_distinctive_words")
    p.add_argument("--top-interpretive", type=int, default=15, dest="top_n_interpretive_words")
    p.add_argument("--table-size", type=int, default=50, dest="members_table_size")
    p.add_argument("--tail-size", type=int, default=24, dest="tail_table_size")
    p.add_argument("--orphan-size", type=int, default=60, dest="orphan_table_size")
    p.add_argument("--globality-threshold", type=float, default=0.40, dest="hubness_similarity_threshold")
    p.add_argument("--frame-edge-threshold", type=float, default=0.42)
    p.add_argument("--bridge-similarity-threshold", type=float, default=0.45)
    p.add_argument("--frame-assignment-min-similarity", type=float, default=0.10)
    p.add_argument("--core-quantile", type=float, default=0.60)
    p.add_argument("--max-plot-words", type=int, default=120)
    p.add_argument("--local-neighbor-window", type=int, default=80)
    p.add_argument("--no-sense-inducer", action="store_true")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    bundle = load_artifact_bundle(args.artifacts)
    config = ReportConfigV7_1(
        lemma=args.lemma,
        output_dir=args.output_dir,
        top_k_neighbors=args.top_k_neighbors,
        min_similarity=args.min_similarity,
        top_n_core_words=args.top_n_core_words,
        top_n_distinctive_words=args.top_n_distinctive_words,
        top_n_interpretive_words=args.top_n_interpretive_words,
        members_table_size=args.members_table_size,
        tail_table_size=args.tail_table_size,
        orphan_table_size=args.orphan_table_size,
        use_sense_inducer=not args.no_sense_inducer,
        export_csv=not args.no_csv,
        hubness_similarity_threshold=args.hubness_similarity_threshold,
        frame_edge_threshold=args.frame_edge_threshold,
        bridge_similarity_threshold=args.bridge_similarity_threshold,
        frame_assignment_min_similarity=args.frame_assignment_min_similarity,
        core_quantile=args.core_quantile,
        max_plot_words=args.max_plot_words,
        local_neighbor_window=args.local_neighbor_window,
    )
    result = AnalyticalSemanticReportBuilderV7_1(bundle, config).build()
    LOGGER.info("Raport V7.2 wygenerowany: %s", result["report_path"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())