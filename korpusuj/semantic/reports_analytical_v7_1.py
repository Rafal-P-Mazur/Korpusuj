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
    from korpusuj.semantic.sense_inducer import SenseInducer
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
    use_sense_inducer: bool = True
    export_csv: bool = True
    # SEMANTIC_MUTUAL_KNN_D1: jawna konstrukcja grafu klasteryzacji.
    frame_graph_mode: str = "mutual_knn"
    frame_graph_knn_k: int = 5
    frame_graph_seed: int = 42
    frame_graph_iterations: int = 100


# KORPUSUJ_PATCH_D14E_FULL_ARTIFACT_FIELD_AND_REVERSE_FIELD_EXPLANATION
# KORPUSUJ_PATCH_D14D_FULL_PCA_AND_REMOVE_LEGACY_PRESENTATION_LIMITS
# KORPUSUJ_PATCH_D14C_COMPACT_RUNTIME_DISCLOSURE
# KORPUSUJ_PATCH_D14B_METHOD_CONTRACT_CLARIFICATION
# KORPUSUJ_PATCH_D14_RELATIONAL_MEASURES_AND_REPORT_POLISH
class AnalyticalSemanticReportBuilderV7_1:
    # SEMANTIC_METHOD_CONTRACT_V1: metadata-only contract; baseline calculations stay unchanged.
    ANALYSIS_METHOD_VERSION = "semantic_report_v7_2"
    # D13_REMOVE_THRESHOLDS_AND_HELPER_GRAPH: globality bez progu; usunięto pomocniczy graf progowy raportu.
    # D11_CHINESE_WHISPERS_CONVERGENCE: CW kończy się po zbieżności; limit iteracji jest zabezpieczeniem.
    # D9A_REMOVE_FRAME_SALIENCE: brak zagregowanej nośności ramowej; pozostają miary bezpośrednie.
    # D10_REMOVE_FIELD_SALIENCE_DISCLOSE_CONTRACT: usunięto nośność pola i ujawniono granice pola oraz filtr rozmiaru ram.
    CLUSTERING_GRAPH_NAME = "mutual_knn_clustering_graph"

    def build_method_contract(self, key: str) -> Dict[str, object]:
        """Return the effective, serializable method contract used by this report.

        This method records existing runtime behavior only. It does not select
        parameters, rebuild frames, or change any numerical result.
        """
        inducer_available = bool(self.config.use_sense_inducer and SenseInducer is not None)
        inducer = SenseInducer if inducer_available else None
        return {
            "schema_version": 1,
            "analysis_method_version": self.ANALYSIS_METHOD_VERSION,
            "lemma": key,
            "selection": {
                "selection_rule": "full_available_neighbor_list",
                "neighbor_artifact_capacity": int(self.bundle.max_neighbors_for(key)),
                "boundary_interpretation": "artifact_capacity_not_semantic_cutoff",
                "similarity_threshold": None,
                "technical_validation_only": True,
            },
            "candidate_pool_construction": {
                "builder": "semantic_field_selection",
                "population": "all_report_field_lemmas_with_vectors",
                "same_population_as_report_field": True,
                "similarity_threshold": None,
            },
            "clustering_graph": {
                "name": (self.CLUSTERING_GRAPH_NAME if self.config.frame_graph_mode == "mutual_knn" else "legacy_threshold_clustering_graph"),
                "purpose": "frame_clustering",
                "builder": ("mutual_knn" if self.config.frame_graph_mode == "mutual_knn" else "SenseInducer_legacy_threshold") if inducer_available else "fallback_greedy_modularity",
                "mode": self.config.frame_graph_mode,
                "similarity_threshold": (None if self.config.frame_graph_mode == "mutual_knn" else (float(inducer.DEFAULT_SIM_THRESHOLD) if inducer else None)),
                "knn_k": (int(self.config.frame_graph_knn_k) if self.config.frame_graph_mode == "mutual_knn" else None),
                "mutual_required": (True if self.config.frame_graph_mode == "mutual_knn" else None),
                "edge_weight_mode": "cosine",
                "minimum_cluster_size": (int(inducer.MIN_CLUSTER_SIZE) if inducer else 2),
                "small_cluster_policy": "clusters_below_minimum_are_not_presented_as_frames_but_are_reported_in_diagnostics",
            },
            "clustering": {
                "algorithm": "chinese_whispers" if inducer_available else "greedy_modularity",
                "implementation": "korpusuj.semantic.sense_inducer.SenseInducer.chinese_whispers" if inducer_available else "networkx.greedy_modularity_communities",
                "reference_seed": int(self.config.frame_graph_seed) if inducer_available else None,
                "maximum_iterations": int(self.config.frame_graph_iterations) if inducer_available else None,
                "stopping_rule": "full_iteration_without_label_changes" if inducer_available else None,
                "uses_edge_weights": True,
                "early_stopping": True if inducer_available else None,
            },
            "assignment": {
                "mode": "best_normalized_frame_centroid",
                "operation": "nonmember_relation_description",
                "similarity_threshold": None,
                "all_nonmember_lemmas_described": True,
                "centroid_recomputed_after_assignment": False,
                "relation_changes_frame_membership": False,
                "relation_changes_frame_centroid": False,
                "membership_sources_recorded": True,
            },
            "nonmember_relation": {
                "mode": "two_nearest_normalized_frame_centroids",
                "reference_frame": "nearest_frame_centroid",
                "competing_frame": "second_nearest_frame_centroid",
                "typicality": "similarity_to_nearest_frame_centroid",
                "distinctiveness": "nearest_similarity_minus_second_nearest_similarity",
                "changes_frame_membership": False,
                "changes_frame_centroid": False,
            },
            "description": {
                "globality_method": "threshold_free_in_degree_over_all_stored_neighbor_lists",
                "similarity_threshold": None,
            },
            "reported_measures": {
                "frame_level": ["typicality", "distinctiveness"],
                "field_level": ["field_typicality", "globality", "field_distinctiveness"],
                "descriptive": ["frequency", "similarity_to_lemma"],
                "aggregated_indices": False,
            },
            "export": {"csv": bool(self.config.export_csv)},
        }

    def __init__(self, bundle: ArtifactBundle, config: ReportConfigV7_1):
        self.bundle = bundle
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._globality_index: Optional[Dict[str, float]] = None
        # SEMANTIC_DIAGNOSTICS_D2: diagnostyka nie zmienia wynikow.
        # SEMANTIC_FRAME_RELATIONS_D4: centroid opisuje zwiazek, nie czlonkostwo.
        # SEMANTIC_PRESENTATION_D4B: spojne nazwy, statusy i eksporty.
        # SEMANTIC_UNIFIED_FRAME_TABLE_D4C: jedna tabela bez redundantnych zakladek.
        # SEMANTIC_FULL_FIELD_KNN5_D6: klasteryzacja calego pola, domyslne k=5.
        # SEMANTIC_HIDE_LOCAL_STRENGTH_D6B: kolumna ukryta w HTML; obliczenia i CSV bez zmian.
        # SEMANTIC_REMOVE_RESOLVED_PARAMETERS_D7: pelna lista sasiadow, relacje bez progu, bez core/periphery.
        # SEMANTIC_D7_HTML_REPAIR_V3: przywrocono kod JS usuniety przez zbyt szeroka kotwice.
        self._clustering_graph_diagnostics: Dict[str, object] = {}
        self._assignment_diagnostics: Dict[str, Dict[str, object]] = {}

    # -----------------------------------------------------
    # Silnik analityczny
    # -----------------------------------------------------
    def build_globality_index(self) -> Dict[str, float]:
        if self._globality_index is not None:
            return self._globality_index
        counts: Dict[str, int] = {}
        # D13: in-degree po pełnych listach zapisanych w artefakcie.
        # Podobieństwo nie jest ponownie odcinane arbitralnym progiem.
        for _, neighbors in self.bundle.index.items():
            for n_word, _n_score, _ in neighbors:
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
        top_k = self.bundle.max_neighbors_for(key)
        rows = []
        seen = set()
        for neighbor, sim, freq in self.bundle.neighbors_of(key, top_k=top_k, min_similarity=0.0):
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

    def _build_fallback_mutual_knn_graph(self, candidate_words: List[str]) -> nx.Graph:
        """Threshold-free mutual kNN graph used only by the fallback path."""
        words = sorted({word for word in candidate_words if word in self.bundle.vectors})
        graph = nx.Graph()
        graph.add_nodes_from(words)
        if len(words) < 2:
            return graph
        matrix = np.vstack([np.asarray(self.bundle.vectors[word], dtype=float) for word in words])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
        similarities = matrix @ matrix.T
        np.fill_diagonal(similarities, -np.inf)
        effective_k = min(max(1, int(self.config.frame_graph_knn_k)), len(words) - 1)
        selected = []
        for i in range(len(words)):
            indices = np.argpartition(-similarities[i], effective_k - 1)[:effective_k]
            selected.append(set(int(j) for j in indices))
        for i in range(len(words)):
            for j in selected[i]:
                if i < j and i in selected[j]:
                    graph.add_edge(words[i], words[j], weight=float(similarities[i, j]))
        return graph

    def _fallback_frames(self, key: str, candidate_words: List[str]) -> List[Dict]:
        sub = self._build_fallback_mutual_knn_graph(candidate_words)
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

    def _induce_mutual_knn_frames(self, key: str, candidate_words: List[str]) -> List[Dict]:
        """Buduje wazony mutual k-NN bez globalnego progu cosinusowego."""
        if SenseInducer is None:
            return []
        k = max(1, int(self.config.frame_graph_knn_k))
        # D6: kazdy lemat pola raportu ma taka sama mozliwosc wejscia do ramy.
        words = sorted({
            w for w in candidate_words
            if w != key and w in self.bundle.vectors
        })
        if len(words) < 2:
            return []
        matrix = np.vstack([np.asarray(self.bundle.vectors[w], dtype=float) for w in words])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
        similarities = matrix @ matrix.T
        np.fill_diagonal(similarities, -np.inf)
        effective_k = min(k, len(words) - 1)
        selected = []
        for i, word in enumerate(words):
            idx = np.argpartition(-similarities[i], effective_k - 1)[:effective_k]
            selected.append(set(int(j) for j in idx))
        graph = nx.Graph()
        graph.add_nodes_from(words)
        nodes_before_isolate_removal = len(words)
        for i in range(len(words)):
            for j in selected[i]:
                if i < j and i in selected[j]:
                    graph.add_edge(words[i], words[j], weight=float(similarities[i, j]))
        isolates = [node for node in graph.nodes() if graph.degree(node) == 0]
        graph.remove_nodes_from(isolates)
        self._clustering_graph_diagnostics = {
            "mode": "mutual_knn",
            "population_source": "semantic_field",
            "full_field_population": True,
            "pool_nodes": int(nodes_before_isolate_removal),
            "field_nodes_with_vectors": int(nodes_before_isolate_removal),
            "nodes_after_isolate_removal": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()), "isolates_removed": int(len(isolates)),
            "knn_k": int(k), "mutual_required": True, "edge_weight_mode": "cosine",
        }
        if graph.number_of_nodes() == 0:
            self._clustering_graph_diagnostics.update({
                "minimum_cluster_size": int(SenseInducer.MIN_CLUSTER_SIZE),
                "clusters_before_min_size_filter": 0,
                "retained_frames": 0,
                "discarded_small_clusters": 0,
                "discarded_small_cluster_members": [],
                "discarded_small_cluster_nodes": 0,
            })
            return []
        clusters, cw_diagnostics = SenseInducer.chinese_whispers(
            graph,
            iters=int(self.config.frame_graph_iterations),
            seed=int(self.config.frame_graph_seed),
            stop_on_convergence=True,
            return_diagnostics=True,
        )
        self._clustering_graph_diagnostics["chinese_whispers"] = cw_diagnostics
        frames = []
        minimum = int(SenseInducer.MIN_CLUSTER_SIZE)
        discarded_small_cluster_members = []
        for cluster in clusters:
            members = sorted(set(cluster))
            if len(members) < minimum:
                discarded_small_cluster_members.append(members)
                continue
            centroid = normalized_centroid([self.bundle.vectors[w] for w in members])
            ranked = sorted(members, key=lambda w: cosine_similarity(self.bundle.vectors[w], centroid), reverse=True)
            frames.append({
                "members": members, "anchors": ranked[:4],
                "label": ", ".join(ranked[:3]),
                "frame_type": "semantic",
                "graph_mode": "mutual_knn",
            })
        frames.sort(key=lambda fr: (-len(fr["members"]), fr["members"][0]))
        self._clustering_graph_diagnostics.update({
            "minimum_cluster_size": int(minimum),
            "clusters_before_min_size_filter": int(len(clusters)),
            "retained_frames": int(len(frames)),
            "retained_graph_members": int(sum(len(fr["members"]) for fr in frames)),
            "discarded_small_clusters": int(len(discarded_small_cluster_members)),
            "discarded_small_cluster_members": discarded_small_cluster_members,
            "discarded_small_cluster_nodes": int(sum(len(members) for members in discarded_small_cluster_members)),
        })
        return frames

    def induce_frames(self, key: str, candidate_words: List[str]) -> List[Dict]:
        frames_raw = []
        if self.config.frame_graph_mode not in {"mutual_knn", "legacy_threshold"}:
            raise ValueError(f"Nieznany frame_graph_mode: {self.config.frame_graph_mode}")
        if self.config.use_sense_inducer and SenseInducer is not None:
            try:
                if self.config.frame_graph_mode == "mutual_knn":
                    frames_raw = self._induce_mutual_knn_frames(key, candidate_words)
                else:
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
                member_set = set(members)
                anchors = [self.bundle.resolve_key(a) or a for a in (fr.get("anchors", []) or [])]
                anchors = [a for a in anchors if isinstance(a, str) and a in member_set]
                if not anchors:
                    anchors = sorted(members, key=lambda w: cosine_similarity(self.bundle.vectors[w], centroid), reverse=True)[:4]

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
                    "graph_members": list(members),
                    "centroid_assigned_members": [],
                    "centroid": centroid,
                    "anchors": anchors[:4],
                })

        if not frames:
            frames = self._fallback_frames(key, candidate_words)
        self._assignment_diagnostics = {}
        assigned = set()
        for fr in frames:
            fr.setdefault("graph_members", list(fr["members"]))
            fr.setdefault("centroid_assigned_members", [])
            assigned.update(fr["members"])
            for member in fr.get("graph_members", []):
                self._assignment_diagnostics[member] = {
                    "best_frame_id": str(fr.get("id", "")),
                    "best_similarity": float(cosine_similarity(self.bundle.vectors[member], fr["centroid"])),
                    "second_frame_id": None, "second_similarity": None,
                    "assignment_margin": None, "assignment_status": "graph_cluster",
                }
        leftovers = [w for w in candidate_words if w not in assigned and w in self.bundle.vectors]
        for word in leftovers:
            wv = self.bundle.vectors[word]
            ranked_frames = sorted(
                ((float(cosine_similarity(wv, fr["centroid"])), fr) for fr in frames),
                key=lambda item: (-item[0], str(item[1].get("id", ""))),
            )
            best_sim, best_frame = ranked_frames[0] if ranked_frames else (-1.0, None)
            second_sim, second_frame = ranked_frames[1] if len(ranked_frames) > 1 else (None, None)
            accepted = bool(best_frame is not None)
            self._assignment_diagnostics[word] = {
                "best_frame_id": str(best_frame.get("id", "")) if best_frame is not None else None,
                "best_similarity": float(best_sim) if best_frame is not None else None,
                "second_frame_id": str(second_frame.get("id", "")) if second_frame is not None else None,
                "second_similarity": float(second_sim) if second_sim is not None else None,
                "assignment_margin": float(best_sim-second_sim) if second_sim is not None else None,
                "assignment_status": "frame_relation" if accepted else "unassigned",
            }
            # D4: relacja centroidowa nie zmienia skladu ramy.
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
                "graph_members": sorted(set(fr.get("graph_members", []))),
                "centroid_assigned_members": sorted(set(fr.get("centroid_assigned_members", []))),
                "centroid": centroid,
                "anchors": anchors[:4],
            })
        return clean

    def compute_word_metrics(self, key: str, field_rows: List[Dict], frames: List[Dict]) -> pd.DataFrame:
        if not field_rows:
            return pd.DataFrame()
        field_words = [r["lemma"] for r in field_rows]
        field_centroid = normalized_centroid([self.bundle.vectors[w] for w in field_words if w in self.bundle.vectors])
        frame_by_word, frame_centroids = {}, {}
        assignment_source_by_word = {}
        for fr in frames:
            frame_centroids[str(fr["id"])] = fr["centroid"]
            graph_members = set(fr.get("graph_members", []))
            centroid_members = set(fr.get("centroid_assigned_members", []))
            for m in fr["members"]:
                frame_by_word[m] = str(fr["id"])
                if m in graph_members:
                    assignment_source_by_word[m] = "graph_cluster"
                elif m in centroid_members:
                    assignment_source_by_word[m] = "centroid_assignment"
                else:
                    assignment_source_by_word[m] = "unknown_baseline_source"
        records = []
        for row in field_rows:
            word = row["lemma"]
            vec = self.bundle.vectors.get(word)
            frame_id = frame_by_word.get(word, "")
            assignment_diag = self._assignment_diagnostics.get(word, {})
            assignment_status = assignment_diag.get(
                "assignment_status", assignment_source_by_word.get(word, "none")
            )
            # D14: członkostwo grafowe i rama odniesienia to dwa różne pojęcia.
            # Członek klastra jest mierzony względem własnej ramy grafowej,
            # a lemat spoza klastrów względem najbliższej ramy wskazanej przez centroid.
            metric_frame_id = frame_id
            if not metric_frame_id and assignment_status == "frame_relation":
                metric_frame_id = str(assignment_diag.get("best_frame_id") or "")

            # ---------------------------
            # MIARY RAMOWE
            # ---------------------------
            frame_typicality = (
                cosine_similarity(vec, frame_centroids[metric_frame_id])
                if metric_frame_id in frame_centroids
                else 0.0
            )
            other_sims = [
                cosine_similarity(vec, centroid)
                for fid, centroid in frame_centroids.items()
                if fid != metric_frame_id
            ]
            frame_distinctiveness = frame_typicality - (max(other_sims) if other_sims else 0.0)

            # ---------------------------
            # NOWE MIARY FIELD-LEVEL
            # ---------------------------
            field_typicality = float(cosine_similarity(vec, field_centroid)) if field_centroid is not None else 0.0

            # "Swoistość pola":
            # słowo jest tym bardziej swoiste dla pola,
            # im bardziej siedzi w centrum pola i im mniej jest globalnym hubem
            field_distinctiveness = field_typicality * (1.0 - float(row["globality"]))

            records.append({
                "lemma": word,
                "frame_id": frame_id,
                "assignment_source": assignment_status,
                "best_frame_id": self._assignment_diagnostics.get(word, {}).get("best_frame_id"),
                "best_frame_similarity": self._assignment_diagnostics.get(word, {}).get("best_similarity"),
                "second_frame_id": self._assignment_diagnostics.get(word, {}).get("second_frame_id"),
                "second_frame_similarity": self._assignment_diagnostics.get(word, {}).get("second_similarity"),
                "assignment_margin": self._assignment_diagnostics.get(word, {}).get("assignment_margin"),
                "similarity_to_lemma": float(row["similarity_to_lemma"]),
                "freq": int(row["freq"]),
                "globality": float(row["globality"]),

                # stare miary ramowe
                "typicality": float(frame_typicality),
                "distinctiveness": float(frame_distinctiveness),

                # nowe miary field-level
                "field_typicality": float(field_typicality),
                "field_distinctiveness": float(field_distinctiveness),

                # alias diagnostyczny / kompatybilność
                "similarity_to_field_centroid": float(field_typicality),
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df


        # D9a: bez zagregowanej nośności ramowej. Porządek techniczny tabeli
        # wykorzystuje bezpośrednie miary ramowe, bez ich ważonego łączenia.
        return df.sort_values(
            ["typicality", "distinctiveness", "freq"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def compute_frame_metrics(self, key: str, frames: List[Dict], word_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        key_vec = self.bundle.vectors.get(key)
        frame_centroids = {str(fr["id"]): fr["centroid"] for fr in frames}
        frame_labels = {str(fr["id"]): fr["label"] for fr in frames}
        for fr in frames:
            fid = str(fr["id"])
            members_df = word_df[word_df["frame_id"] == fid].copy()
            if members_df.empty:
                continue
            member_vectors = [self.bundle.vectors[m] for m in members_df["lemma"].tolist() if m in self.bundle.vectors]
            centroid = fr["centroid"]
            nearest_frame_id = None
            nearest_frame_similarity = None
            for other_frame_id, other_centroid in frame_centroids.items():
                if other_frame_id == fid:
                    continue
                similarity = float(cosine_similarity(centroid, other_centroid))
                if nearest_frame_similarity is None or similarity > nearest_frame_similarity:
                    nearest_frame_similarity = similarity
                    nearest_frame_id = other_frame_id
            separation = (
                1.0 - nearest_frame_similarity
                if nearest_frame_similarity is not None
                else None
            )
            records.append({
                "frame_id": fid,
                "frame_label": fr["label"],
                "frame_type": fr.get("type", "semantyczna"),
                "size": int(len(members_df)),
                "coverage_share": float(len(members_df) / len(word_df)) if len(word_df) else 0.0,
                "cohesion_pairwise": float(pairwise_mean_cos(member_vectors)),
                "cohesion_centroid_mean": float(members_df["typicality"].mean()),
                "distinctiveness_mean": float(members_df["distinctiveness"].mean()),
                "globality_mean": float(members_df["globality"].mean()),
                "similarity_centroid_to_lemma": float(
                    cosine_similarity(key_vec, centroid)) if key_vec is not None else 0.0,
                "separation_from_other_frames": float(separation) if separation is not None else None,
                "frequency_sum": int(members_df["freq"].sum()),
                "frequency_mean": float(members_df["freq"].mean()),
                "anchors": fr.get("anchors", [])[:4],
                "nearest_frame_id": nearest_frame_id,
                "nearest_frame_label": frame_labels.get(nearest_frame_id) if nearest_frame_id else None,
                "nearest_frame_similarity": nearest_frame_similarity,
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(
                ["coverage_share", "cohesion_centroid_mean", "distinctiveness_mean"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            df["frame_rank"] = range(1, len(df) + 1)
        return df

    def compute_global_overview(self, key: str, field_rows: List[Dict], frame_df: pd.DataFrame) -> Dict:
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
            "neighbors_top_k": int(self.bundle.max_neighbors_for(key)),
        }

    def compute_projection(self, key: str, word_df: pd.DataFrame, frame_df: pd.DataFrame, frames: List[Dict]) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        plot_word_df = word_df.copy()
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
                })
            else:
                meta = label_to_meta[label]
                word_points.append({
                    "kind": "word",
                    "lemma": label,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "frame_id": str(meta.get("frame_id", "")),
                    "size_metric": float(meta.get("typicality", 0.0)),
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

    def compute_diagnostics(self, key: str, field_rows: List[Dict], word_df: pd.DataFrame, frames: List[Dict]) -> Dict:
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
            "relation_df": payload["relation_df"].to_dict(orient="records") if payload.get("relation_df") is not None else [],
            "frame_similarity_df": payload["frame_similarity_df"].to_dict(orient="records") if payload.get(
                "frame_similarity_df") is not None else [],
            "words_coords_df": payload["words_coords_df"].to_dict(orient="records") if payload.get(
                "words_coords_df") is not None else [],
            "frames_coords_df": payload["frames_coords_df"].to_dict(orient="records") if payload.get(
                "frames_coords_df") is not None else [],
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
        (self.output_dir / "method_contract.json").write_text(
            json.dumps(methodology.get("method_contract", {}), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (self.output_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        if self.config.export_csv:
            export_word_df = payload["word_df"].copy()
            export_relation_df = payload["relation_df"].copy()
            export_word_df.to_csv(self.output_dir / "semantic_field.csv", index=False, encoding="utf-8")
            payload["frame_df"].to_csv(self.output_dir / "frames.csv", index=False, encoding="utf-8")
            export_relation_df.to_csv(self.output_dir / "frame_relations.csv", index=False, encoding="utf-8")
            export_word_df[export_word_df["assignment_source"] == "graph_cluster"].to_csv(self.output_dir / "frame_members.csv", index=False, encoding="utf-8")
            payload["frame_similarity_df"].to_csv(self.output_dir / "frame_similarity.csv", index=False,
                                                  encoding="utf-8")
            payload["words_coords_df"].to_csv(self.output_dir / "coordinates_words.csv", index=False, encoding="utf-8")
            payload["frames_coords_df"].to_csv(self.output_dir / "coordinates_frames.csv", index=False,
                                               encoding="utf-8")
            stale_edges = self.output_dir / "edges.csv"
            if stale_edges.exists():
                stale_edges.unlink()
            legacy_orphans = self.output_dir / "periphery_orphans.csv"
            if legacy_orphans.exists():
                legacy_orphans.unlink()

    def compute_reverse_field(self, target_lemma: str) -> pd.DataFrame:
        df = self.bundle.df_neighbors
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Kto ma target_lemma w swoim polu
        mask = (df["neighbor"] == target_lemma)
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
                top_k=self.bundle.max_neighbors_for(x_lemma),
                min_similarity=0.0
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
          <title>Raport semantyczny</title>
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
              <div style="font-size: 21px; font-weight: 700;">Raport semantyczny</div>
              <div id="header-meta" class="meta"></div>
            </div>
            <button class="btn" id="method-btn">Metodologia</button>
          </header>

          <div class="wrap">
            <section class="cards" id="summary-cards"></section>
            <section class="cards" id="technical-cards"></section>

            <section class="panel">
              <div class="panel-head">
                <h2>Globalne rankingi słów</h2>
                <div class="section-note">
                25 lematów o najwyższej centralności i specyficzności w całym analizowanym polu.
                </div>
              </div>
              <div class="panel-body">
                <div class="two-col">
                  <div class="table-wrap">
                    <table>
                      <thead><tr><th>Najwyższa centralność pola</th><th>Powiązane ramy</th><th>Wynik</th></tr></thead>
                      <tbody id="global-typ-tbody"></tbody>
                    </table>
                  </div>
                  <div class="table-wrap">
                    <table>
                      <thead><tr><th>Najwyższa specyficzność pola</th><th>Powiązane ramy</th><th>Wynik</th></tr></thead>
                      <tbody id="global-dis-tbody"></tbody>
                    </table>
                  </div>
                </div>
              </div>
            </section>

            <section class="grid-main">
              <div class="panel">
                <div class="panel-head">
                  <h2>Przestrzeń semantyczna (PCA)</h2>
                  <div class="section-note">Przełączaj zakładki, aby zobaczyć całe słownictwo w tle ram lub same centroidy ram. Położenie na mapie jest dwuwymiarową projekcją PCA i przybliża relacje obecne w pełnej przestrzeni wektorowej.</div>
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
                    <div class="two-col">
                      <div class="panel"><div class="panel-head"><h3>Najwyższa typowość</h3></div><div class="panel-body"><div id="core-chart" class="chart"></div></div></div>
                      <div class="panel"><div class="panel-head"><h3>Najwyższa swoistość</h3></div><div class="panel-body"><div id="distinctive-chart" class="chart"></div></div></div>
                    </div>

                    <div style="height:16px"></div>
                    <div class="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Lemat</th>
                            <th>Status</th>
                            <th>Powiązane ramy</th>
                            <th>Frekwencja</th>
                            <th>Typowość</th>
                            <th>Swoistość</th>
                            <th>Ogólność</th>
                          </tr>
                        </thead>
                        <tbody id="frame-lemmas-tbody"></tbody>
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
                        <th>Rama</th>
                        <th>Rozmiar</th>
                        <th title="Średnia wartość typowości wszystkich elementów należących do ramy. Stanowi wskaźnik wewnętrznej spójności i jednorodności semantycznej grupy.">Zwartość <span class="help">?</span></th>
                        <th title="Miara dystansu semantycznego między centroidem danej ramy a najbliższym sąsiadującym klastrem. Odzwierciedla stopień odrębności tematycznej.">Separacja <span class="help">?</span></th>
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
                  <div class="section-note">Macierz przedstawia podobieństwo między centroidami ram. Wyższa wartość oznacza bardziej zbliżone położenie dwóch ram w przestrzeni wektorowej.</div>
                </div>
                <div class="panel-body"><div id="relations-heatmap"></div></div>
              </div>
            
              <div class="panel" id="reverse-field-panel">
                <div class="panel-head">
                  <h2>Obecność w innych polach</h2>
                  <div class="section-note">
                    Pokazuje, w polach semantycznych jakich innych pojęć występuje badany lemat oraz jak centralne zajmuje w nich miejsce.
                  </div>
                </div>
                <div class="panel-body" id="reverse-field-body"></div>
              </div>
            </section>

          <div class="modal" id="modal">
            <div class="modal-box">
              <span class="close-x" id="close-modal">×</span>
              <h2>Metodologia i interpretacja miar</h2>
              <!-- D13F_METHODOLOGY_LAYOUT -->
              <p style="line-height:1.6; color:#334155;">Raport wykorzystuje graf wzajemnego sąsiedztwa oraz reprezentacje wektorowe do wydobycia lokalnych skupień podobieństwa wokół badanej lemy. Podobieństwo wektorowe może odzwierciedlać zarówno podobieństwo kontekstów użycia, jak i podobieństwo budowy wyrazów wynikające z reprezentacji subwordowych modelu FastText.</p>

              <div style="margin:14px 0 18px 0; line-height:1.65; color:#334155;">
                <p>Każdy lemat <code>w</code> jest reprezentowany przez wektor <code>v(w)</code>. Centroid ramy <code>c(r)</code> reprezentuje wspólne położenie lematów należących do ramy <code>r</code>, a centroid pola <code>c(F)</code> reprezentuje wspólne położenie wszystkich lematów analizowanego pola. Centroid zbioru <code>A</code> jest obliczany jako znormalizowana suma jego wektorów:</p>
                <p style="font-family:Consolas,monospace; font-size:1.04em; margin:10px 0;"><b>c(A) = Σ<sub>w∈A</sub> v(w) / ||Σ<sub>w∈A</sub> v(w)||</b></p>
              </div>

              <h3 style="margin:18px 0 10px 0;">Miary ramowe</h3>
              <div class="method-grid">
                <div class="method-card">
                  <h4>Typowość ramowa</h4>
                  <div style="font-size:.88em; color:#64748b; margin-bottom:8px;"><code>typicality</code></div>
                  <p>Określa stopień zgodności kierunku wektora lematu z kierunkiem centroidu ramy odniesienia. Dla członka klastra jest nią własna rama grafowa, a dla lematu spoza klastrów najbliższa rama centroidowa.</p>
                  <p style="font-family:Consolas,monospace; font-size:1.04em;"><b>T<sub>r</sub>(w) = [v(w) · c(r)] / [||v(w)|| · ||c(r)||]</b></p>
                  <p>Wyższa wartość oznacza bardziej centralne położenie lematu w ramie.</p>
                </div>
                <div class="method-card">
                  <h4>Swoistość ramowa</h4>
                  <div style="font-size:.88em; color:#64748b; margin-bottom:8px;"><code>distinctiveness</code></div>
                  <p>Określa, o ile podobieństwo lematu do ramy odniesienia przewyższa jego największe podobieństwo do innej ramy.</p>
                  <p style="font-family:Consolas,monospace; font-size:1.04em;"><b>D<sub>r</sub>(w) = T<sub>r</sub>(w) − max<sub>s≠r</sub> {[v(w) · c(s)] / [||v(w)|| · ||c(s)||]}</b></p>
                  <p>Wartość bliska zeru wskazuje położenie na pograniczu ram. Dla członka klastra wartość ujemna oznacza większe podobieństwo do centroidu innej ramy niż do centroidu własnej ramy. Dla lematu spoza klastrów rama odniesienia jest ramą najbliższą, dlatego swoistość jest nieujemna.</p>
                </div>
              </div>

              <h3 style="margin:18px 0 10px 0;">Miary całego pola</h3>
              <div class="method-grid" id="method-grid">
                <div class="method-card">
                  <h4>Centralność pola</h4>
                  <div style="font-size:.88em; color:#64748b; margin-bottom:8px;"><code>field_typicality</code></div>
                  <p>Określa stopień zgodności kierunku wektora lematu z kierunkiem centroidu całego pola.</p>
                  <p style="font-family:Consolas,monospace; font-size:1.04em;"><b>C<sub>F</sub>(w) = [v(w) · c(F)] / [||v(w)|| · ||c(F)||]</b></p>
                  <p>Wyższa wartość oznacza bardziej centralne położenie lematu w analizowanym polu.</p>
                </div>
                <div class="method-card">
                  <h4>Ogólność</h4>
                  <div style="font-size:.88em; color:#64748b; margin-bottom:8px;"><code>globality</code></div>
                  <p>Określa rozpowszechnienie lematu na listach sąsiedztwa innych lematów.</p>
                  <p style="font-family:Consolas,monospace; font-size:1.04em;"><b>G(w) = 0</b>, gdy <b>d<sub>in</sub>(w) ≤ p<sub>50</sub></b></p>
                  <p style="font-family:Consolas,monospace; font-size:1.04em;"><b>G(w) = log[d<sub>in</sub>(w) − p<sub>50</sub> + 1] / log[d<sub>max</sub> − p<sub>50</sub> + 1]</b>, gdy <b>d<sub>in</sub>(w) &gt; p<sub>50</sub></b></p>
                  <p><code>d<sub>in</sub>(w)</code> to liczba list sąsiedztwa zawierających lemat <code>w</code>; <code>p<sub>50</sub></code> to mediana tej liczby, a <code>d<sub>max</sub></code> jej wartość maksymalna. Wyższa wartość oznacza lemat występujący w sąsiedztwie większej liczby różnych jednostek.</p>
                </div>
                <div class="method-card">
                  <h4>Specyficzność pola</h4>
                  <div style="font-size:.88em; color:#64748b; margin-bottom:8px;"><code>field_distinctiveness</code></div>
                  <p>Określa centralność lematu w analizowanym polu po pomniejszeniu jej odpowiednio do ogólności lematu.</p>
                  <p style="font-family:Consolas,monospace; font-size:1.04em;"><b>S<sub>F</sub>(w) = C<sub>F</sub>(w) · [1 − G(w)]</b></p>
                  <p>Wyższa wartość oznacza lemat centralny dla badanego pola, który nie występuje często na listach sąsiedztwa wielu innych jednostek.</p>
                </div>
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
                const fr = DATA.frame_df.find(f => String(f.frame_id) === String(fid));
                return fr ? `R${fr.frame_rank}` : `R${fid}`;
              };
              const getRankNum = (fid) => {
                const fr = DATA.frame_df.find(f => String(f.frame_id) === String(fid));
                return fr ? Number(fr.frame_rank) : null;
              };

              const typWords = [...words]
                .sort((a, b) => (b.field_typicality ?? 0) - (a.field_typicality ?? 0))
                .slice(0, 25);

              const disWords = [...words]
                .sort((a, b) => (b.field_distinctiveness ?? 0) - (a.field_distinctiveness ?? 0))
                .slice(0, 25);

              const fillTable = (id, data, key) => {
                const tbody = document.getElementById(id);
                tbody.innerHTML = data.map(w => {
                  const createsFrame = w.assignment_source === 'graph_cluster';
                  let frameBadges = '';
                  if (createsFrame) {
                    const rankNum = getRankNum(w.frame_id);
                    const color = rankNum ? getFrameColor(rankNum) : '#64748b';
                    frameBadges = `<span class="badge" style="background:${color};color:#fff">${getRank(w.frame_id)}</span>`;
                  } else {
                    const ids = [w.best_frame_id, w.second_frame_id].filter(Boolean);
                    frameBadges = ids.map(fid => `<span class="badge" style="background:#e2e8f0;color:#334155;margin-right:4px">${getRank(fid)}</span>`).join('');
                  }
                  return `<tr>
                    <td><b>${w.lemma}</b></td>
                    <td>${frameBadges || '—'}</td>
                    <td>${fmt(w[key])}</td>
                  </tr>`;
                }).join('');
              };
              fillTable('global-typ-tbody', typWords, 'field_typicality');
              fillTable('global-dis-tbody', disWords, 'field_distinctiveness');
            }

            function renderMaps() {
              if (typeof Plotly === 'undefined') {
                const warning = '<div class="info-box">Nie udało się załadować biblioteki Plotly. Mapy i macierz relacji wymagają dostępu do skryptu Plotly, ale pozostałe tabele raportu nadal działają.</div>';
                const wordsMap = document.getElementById('words-map');
                const framesMap = document.getElementById('frames-map');
                if (wordsMap) wordsMap.innerHTML = warning;
                if (framesMap) framesMap.innerHTML = warning;
                return;
              }
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
                groupedWords[rank].text.push(`<b>${w.lemma}</b><br>Rama: ${rank === 999 ? 'Brak' : rank}<br>Typowość: ${fmt(w.typicality || 0)}<br>Swoistość: ${fmt(w.distinctiveness || 0)}<br>Freq: ${w.freq}`);
              });
              Object.values(groupedWords).sort((a,b) => a.rank - b.rank).forEach(group => {
                wordsTraces.push({
                  x: group.x, y: group.y, text: group.text, customdata: group.customdata,
                  mode: 'markers', hoverinfo: 'text',
                  marker: { size: group.size, color: group.rank === 999 ? '#94a3b8' : getFrameColor(group.rank), opacity: 0.78 },
                  name: group.rank === 999 ? 'Lematy związane z ramami' : `Rama ${group.rank}`
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
                  <td>${fmt(f.cohesion_centroid_mean)}</td>
                  <td>${f.separation_from_other_frames == null ? '—' : fmt(f.separation_from_other_frames)}</td>
                  <td>${f.nearest_frame_id
                    ? '<b>' + (f.nearest_frame_label || f.nearest_frame_id) + '</b> (' + fmt(f.nearest_frame_similarity) + ')'
                    : '—'}</td>`;
                tr.addEventListener('click', () => {
                  selectFrame(f.frame_id);
                  document.getElementById('detail-title').scrollIntoView({ behavior: 'smooth' });
                });
                tbody.appendChild(tr);
              });
            }

            function renderRelations() {
              if (typeof Plotly === 'undefined') {
                const target = document.getElementById('relations-heatmap');
                if (target) target.innerHTML = '<div class="info-box">Macierz relacji nie jest dostępna, ponieważ biblioteka Plotly nie została załadowana.</div>';
                return;
              }
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
              const frame = DATA.frame_df.find(f => String(f.frame_id) === String(frameId));
              if (!frame) return;
              const members = DATA.word_df.filter(w => String(w.frame_id) === String(frameId) && w.assignment_source === 'graph_cluster');
              const relations = (DATA.relation_df || []).filter(w => String(w.nearest_frame_id) === String(frameId));

              document.getElementById('detail-empty').style.display = 'none';
              document.getElementById('detail-content').style.display = 'block';
              document.getElementById('detail-title').textContent = formatFrameDisplayName(frame);
              document.getElementById('detail-subtitle').textContent = `Rank: ${frame.frame_rank} · Anchory: ${(frame.anchors || []).join(', ')}`;

              const mHtml = [
                `<div class="metric"><div class="k">Rozmiar ramy</div><div class="v">${frame.size}</div><div class="i">Liczba lematów należących do skupienia grafowego</div></div>`,
                `<div class="metric"><div class="k">Zwartość ramy</div><div class="v">${fmt(frame.cohesion_centroid_mean)}</div><div class="i">Średnie podobieństwo członków do centroidu</div></div>`,
              ].join('');
              document.getElementById('frame-metrics').innerHTML = mHtml;

              const coreWords = [...members].sort((a,b) => b.typicality - a.typicality).slice(0, 10);
              const distWords = [...members].sort((a,b) => b.distinctiveness - a.distinctiveness).slice(0, 10);
              const color = getFrameColor(frame.frame_rank);
              renderBarChart('core-chart', coreWords, 'typicality', color, 'Typowość');
              renderBarChart('distinctive-chart', distWords, 'distinctiveness', color, 'Swoistość');

              const combined = [];
              members.forEach(row => combined.push({
                ...row,
                table_status: 'Rama',
                similarity_to_current_frame: row.best_frame_similarity ?? row.typicality,
                related_frames: [],
              }));
              relations.forEach(row => combined.push({
                ...row,
                table_status: 'Związany z ramą',
                similarity_to_current_frame: row.nearest_frame_similarity,
                related_frames: [row.second_frame_id].filter(Boolean),
              }));
              combined.sort((a,b) => (b.similarity_to_current_frame ?? -Infinity) - (a.similarity_to_current_frame ?? -Infinity));

              const tbody = document.getElementById('frame-lemmas-tbody');
              tbody.innerHTML = combined.map(row => {
                const related = (row.related_frames || []).map(fid => {
                  const relatedFrame = DATA.frame_df.find(f => String(f.frame_id) === String(fid));
                  return relatedFrame ? `R${relatedFrame.frame_rank}` : `R${fid}`;
                }).join(', ') || '—';
                return `<tr>
                  <td><b>${row.lemma}</b></td>
                  <td><span class="badge" style="background:${row.table_status === 'Rama' ? '#dbeafe' : '#e2e8f0'};color:#334155">${row.table_status}</span></td>
                  <td>${related}</td>
                  <td>${row.freq ?? '—'}</td>
                  <td>${fmt(row.typicality)}</td>
                  <td>${fmt(row.distinctiveness)}</td>
                  <td>${fmt(row.globality)}</td>
                </tr>`;
              }).join('');
            }

            function renderReverseField() {
              const container = document.getElementById('reverse-field-body');
              const data = DATA.reverse_field_df || [];
            
            
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
              const boolPl = value => value ? 'tak' : 'nie';
              const methodology = DATA.methodology || {};
              const contract = methodology.method_contract || {};
              const selection = methodology.semantic_field_selection || {};
              const graph = contract.clustering_graph || {};
              const clustering = contract.clustering || {};
              const relation = contract.nonmember_relation || {};
              const reported = contract.reported_measures || {};
              const runtimeCw = (methodology.frame_construction_disclosure || {}).chinese_whispers || {};

              const methodSummary = {
                wersja_metody: contract.analysis_method_version || null,
                korpus: methodology.bundle_label || null,
                lemat: methodology.lemma || DATA.lemma || null,
                pole_semantyczne: {
                  dostepne_lematy: selection.available_neighbors ?? null,
                  wykorzystane_lematy: selection.used_neighbors ?? null,
                  zakres: 'pełna lista dostępna w artefakcie'
                },
                graf_klasteryzacji: {
                  metoda: 'mutual k-NN',
                  k: graph.knn_k ?? null,
                  wymagana_wzajemnosc: boolPl(Boolean(graph.mutual_required)),
                  waga_krawedzi: 'podobieństwo cosinusowe'
                },
                klasteryzacja: {
                  algorytm: clustering.algorithm === 'chinese_whispers' ? 'Chinese Whispers' : clustering.algorithm,
                  seed: clustering.reference_seed ?? null,
                  maksimum_iteracji: clustering.maximum_iterations ?? null,
                  wykonane_iteracje: runtimeCw.iterations_used ?? null,
                  osiagnieta_zbieznosc: boolPl(Boolean(runtimeCw.converged)),
                  warunek_zatrzymania: 'pełna iteracja bez zmiany etykiet',
                  minimalny_rozmiar_ramy: graph.minimum_cluster_size ?? null
                },
                lematy_spoza_klastrow: {
                  opis: relation.mode === 'two_nearest_normalized_frame_centroids'
                    ? 'dwie najbliższe ramy według znormalizowanych centroidów'
                    : relation.mode,
                  zmieniaja_sklad_ramy: boolPl(Boolean(relation.changes_frame_membership)),
                  zmieniaja_centroid_ramy: boolPl(Boolean(relation.changes_frame_centroid))
                },
                ogolnosc: 'obecność lematu na listach sąsiedztwa innych lematów',
                indeksy_zagregowane: boolPl(Boolean(reported.aggregated_indices))
              };

              const diagnostics = DATA.diagnostics || {};
              const dGraph = diagnostics.clustering_graph || {};
              const dCw = dGraph.chinese_whispers || {};
              const assignment = diagnostics.assignment || {};
              const diagnosticSummary = {
                pole: {
                  lematy: diagnostics.candidate_neighbors_total ?? null,
                  brakujace_wektory: diagnostics.missing_vectors_total ?? null,
                  lematy_na_mapie: (diagnostics.projection || {}).mapped_word_records ?? null
                },
                graf: {
                  wezly_przed_usunieciem_izolatow: dGraph.pool_nodes ?? null,
                  wezly_po_usunieciu_izolatow: dGraph.nodes_after_isolate_removal ?? null,
                  krawedzie: dGraph.edges ?? null,
                  izolaty: dGraph.isolates_removed ?? null
                },
                klasteryzacja: {
                  iteracje: dCw.iterations_used ?? null,
                  zbieznosc: boolPl(Boolean(dCw.converged)),
                  zmiany_etykiet: dCw.label_changes_by_iteration || [],
                  klastry_przed_filtrem: dGraph.clusters_before_min_size_filter ?? null,
                  zachowane_ramy: dGraph.retained_frames ?? diagnostics.frames_total ?? null,
                  odrzucone_male_klastry: dGraph.discarded_small_clusters ?? null,
                  lematy_w_malych_klastrach: dGraph.discarded_small_cluster_nodes ?? null
                },
                pokrycie: {
                  czlonkowie_ram: assignment.graph_cluster ?? diagnostics.frame_members_total ?? null,
                  relacje_do_ram: assignment.frame_relation ?? diagnostics.frame_related_lemmas_total ?? null,
                  nieopisane_lematy: assignment.unassigned ?? diagnostics.unrelated_lemmas_total ?? null,
                  pelne_pokrycie: boolPl(Number(diagnostics.described_candidates_ratio) === 1)
                }
              };

              if (document.getElementById('method-pre')) document.getElementById('method-pre').textContent = JSON.stringify(methodSummary, null, 2);
              if (document.getElementById('diag-pre')) document.getElementById('diag-pre').textContent = JSON.stringify(diagnosticSummary, null, 2);
            }
            document.addEventListener('DOMContentLoaded', () => {
              const runSection = (name, fn) => {
                try {
                  fn();
                } catch (error) {
                  console.error(`[Raport] Błąd sekcji ${name}:`, error);
                }
              };

              // Kontrolki zakładek i modali muszą być aktywne niezależnie od
              // powodzenia zewnętrznej biblioteki wykresów.
              runSection('zakładki i modale', initTabsAndModals);
              runSection('nagłówek', () => {
                const header = document.getElementById('header-meta');
                if (header) header.textContent = `Lema: ${DATA.overview.lemma} · Ramy: ${DATA.overview.liczba_ram}`;
              });
              runSection('karty', buildCards);
              runSection('globalne rankingi', renderGlobalRankings);
              runSection('podsumowanie ram', renderFramesSummaryTable);
              runSection('obecność w innych polach', renderReverseField);

              // Wybór pierwszej ramy nie zależy od mapy i powinien działać
              // również wtedy, gdy Plotly jest niedostępne.
              runSection('pierwsza rama', () => {
                if (DATA.frame_df.length) selectFrame(DATA.frame_df[0].frame_id);
              });

              runSection('mapy', renderMaps);
              runSection('relacje między ramami', renderRelations);
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
            "relation_df": payload["relation_df"].to_dict(orient="records"),
            "frame_similarity_df": payload["frame_similarity_df"].to_dict(orient="records"),
            "words_coords_df": payload["words_coords_df"].to_dict(orient="records"),
            "frames_coords_df": payload["frames_coords_df"].to_dict(orient="records"),
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
        frames = self.induce_frames(key, candidate_words)
        if not frames:
            raise ValueError("Nie udało się wygenerować ram semantycznych dla wskazanej lemy.")

        word_df = self.compute_word_metrics(key, field_rows, frames)
        relation_df = word_df[word_df["assignment_source"] == "frame_relation"].copy()
        if not relation_df.empty:
            relation_df = relation_df.rename(columns={"best_frame_id": "nearest_frame_id", "best_frame_similarity": "nearest_frame_similarity", "assignment_margin": "similarity_difference"})
        frame_df = self.compute_frame_metrics(key, frames, word_df)
        frame_similarity_df = self.compute_frame_similarity(frames)
        words_coords_df, frames_coords_df = self.compute_projection(key, word_df, frame_df, frames)
        mapped_word_records = int((words_coords_df["kind"] == "word").sum()) if not words_coords_df.empty else 0
        expected_word_records = int(len(word_df))
        if mapped_word_records != expected_word_records:
            raise RuntimeError(
                f"Mapa PCA jest niekompletna: {mapped_word_records} z {expected_word_records} lematów."
            )
        overview = self.compute_global_overview(key, field_rows, frame_df)
        diagnostics = self.compute_diagnostics(key, field_rows, word_df, frames)
        frame_members_total = int((word_df['assignment_source'] == 'graph_cluster').sum())
        frame_related_lemmas_total = int((word_df['assignment_source'] == 'frame_relation').sum())
        unrelated_lemmas_total = int((word_df['assignment_source'] == 'unassigned').sum())
        candidates_total = int(len(word_df))
        diagnostics['frame_members_total'] = frame_members_total
        diagnostics['frame_related_lemmas_total'] = frame_related_lemmas_total
        diagnostics['unrelated_lemmas_total'] = unrelated_lemmas_total
        diagnostics['frame_membership_ratio'] = round(frame_members_total / candidates_total, 4) if candidates_total else 0.0
        diagnostics['frame_relation_ratio'] = round(frame_related_lemmas_total / candidates_total, 4) if candidates_total else 0.0
        diagnostics['described_candidates_ratio'] = round((frame_members_total + frame_related_lemmas_total) / candidates_total, 4) if candidates_total else 0.0
        diagnostics["projection"] = {
            "mapped_word_records": mapped_word_records,
            "expected_word_records": expected_word_records,
            "complete": mapped_word_records == expected_word_records,
        }
        for legacy_key in ('assigned_neighbors_total', 'orphans_total', 'coverage_ratio'):
            diagnostics.pop(legacy_key, None)
        diagnostics["clustering_graph"] = dict(self._clustering_graph_diagnostics)
        diagnostics["assignment"] = {
            "graph_cluster": int(sum(v.get("assignment_status")=="graph_cluster" for v in self._assignment_diagnostics.values())),
            "frame_relation": int(sum(v.get("assignment_status")=="frame_relation" for v in self._assignment_diagnostics.values())),
            "unassigned": int(sum(v.get("assignment_status")=="unassigned" for v in self._assignment_diagnostics.values())),
            "with_second_frame": int(sum(v.get("second_frame_id") is not None for v in self._assignment_diagnostics.values())),
        }
        reverse_field_df = self.compute_reverse_field(key)

        method_contract = self.build_method_contract(key)
        used_similarities = [float(row["similarity_to_lemma"]) for row in field_rows]
        semantic_field_selection = {
            "source": "full_available_neighbor_list",
            "neighbor_artifact_capacity": int(self.bundle.max_neighbors_for(key)),
            "boundary_interpretation": "artifact_capacity_not_semantic_cutoff",
            "available_neighbors": int(self.bundle.max_neighbors_for(key)),
            "used_neighbors": int(len(field_rows)),
            "minimum_used_similarity": (min(used_similarities) if used_similarities else None),
            "maximum_used_similarity": (max(used_similarities) if used_similarities else None),
            "rejected_missing_or_duplicate_or_self": int(max(0, self.bundle.max_neighbors_for(key) - len(field_rows))),
            "similarity_threshold": None,
            "technical_validation_only": True,
        }
        diagnostics["semantic_field_selection"] = semantic_field_selection

        methodology = {
            "method_contract": method_contract,
            "source_path": self.bundle.source_path,
            "bundle_label": self.bundle.label,
            "lemma": key,
            "semantic_field_selection": semantic_field_selection,
            "frame_construction_disclosure": {
                "graph_mode": self.config.frame_graph_mode,
                "knn_k": int(self.config.frame_graph_knn_k),
                "mutuality_required": bool(self.config.frame_graph_mode == "mutual_knn"),
                "minimum_frame_size": int(SenseInducer.MIN_CLUSTER_SIZE) if SenseInducer is not None else 2,
                "small_clusters_reported_in_diagnostics": True,
                "chinese_whispers": dict(self._clustering_graph_diagnostics.get("chinese_whispers", {})),
            },
            "frame_source": "SenseInducer" if (
                        self.config.use_sense_inducer and SenseInducer is not None) else "fallback_greedy_modularity",
            "use_sense_inducer": bool(self.config.use_sense_inducer and SenseInducer is not None),
            "globality": {
                "method": "threshold_free_in_degree_over_all_stored_neighbor_lists",
                "similarity_threshold": None,
            },
            "typicality": "cos(word, centroid_ramy)",
            "distinctiveness": "typicality - max cos(word, centroid_innej_ramy)",
            "projection_2d": "PCA na wektorach słów / centroidach ram",
        }

        payload = {
            "lemma": key,
            "overview": overview,
            "methodology": methodology,
            "diagnostics": diagnostics,
            "word_df": word_df,
            "frame_df": frame_df,
            "relation_df": relation_df,
            "frame_similarity_df": frame_similarity_df,
            "words_coords_df": words_coords_df,
            "frames_coords_df": frames_coords_df,
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
    p.add_argument("--frame-graph-mode", choices=("mutual_knn", "legacy_threshold"), default="mutual_knn")
    p.add_argument("--frame-graph-knn-k", type=int, default=5)
    p.add_argument("--frame-graph-seed", type=int, default=42)
    p.add_argument(
        "--frame-graph-iterations",
        type=int,
        default=100,
        help="Maksymalna liczba iteracji Chinese Whispers; algorytm kończy się wcześniej po zbieżności.",
    )
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
        use_sense_inducer=not args.no_sense_inducer,
        export_csv=not args.no_csv,
        frame_graph_mode=args.frame_graph_mode,
        frame_graph_knn_k=args.frame_graph_knn_k,
        frame_graph_seed=args.frame_graph_seed,
        frame_graph_iterations=args.frame_graph_iterations,
    )
    result = AnalyticalSemanticReportBuilderV7_1(bundle, config).build()
    LOGGER.info("Raport V7.2 wygenerowany: %s", result["report_path"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())